"""EC2 control plane. Uses the installed AWS CLI and no Python AWS dependencies."""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import shlex
import subprocess
import tarfile
import tempfile
import time
import uuid
from pathlib import Path

from infra.aws.common import AWS, digest, restore_checkpoint, write_json
from infra.aws.template import template
from benchmarking.suite_plan import experiment_configs, load_plan, uses_s3, validate_plan

ROOT = Path(__file__).resolve().parents[1]
SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def memory_bytes(value):
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([kmgt])(?:i?b)?", str(value).lower())
    if not match:
        raise ValueError("unsupported container memory limit: " + str(value))
    return int(float(match[1]) * 1024 ** ("kmgt".index(match[2]) + 1))


def validate_resources(config, instance):
    if "groups" in config:
        validate_plan(config)
        for group in experiment_configs(config):
            validate_resources({**group, "suite": {**group["suite"], **config["suite"]}}, instance)
        return
    if instance["ProcessorInfo"]["SupportedArchitectures"] != ["arm64"]:
        raise ValueError("an ARM64 instance is required")
    cpus = config["suite"]["container_cpus"]
    if not 0 < cpus <= instance["VCpuInfo"]["DefaultVCpus"]:
        raise ValueError("configuration CPUs must be positive and fit the instance")
    memory = instance["MemoryInfo"]["SizeInMiB"] * 1024**2
    if memory_bytes(config["suite"]["container_memory"]) > memory - 2 * 1024**3:
        raise ValueError("configuration leaves less than 2 GiB for the EC2 host; choose a larger instance")
    if memory_bytes(config["benchmark"]["memory_limit"]) > memory_bytes(config["suite"]["container_memory"]):
        raise ValueError("DuckDB memory limit exceeds the container memory limit")
    storage = instance.get("InstanceStorageInfo", {}).get("TotalSizeInGB", 0) * 10**9
    needed = max((w.get("estimated_peak_gib", 1) for w in config["workloads"]), default=1) + config["suite"]["disk_reserve_gib"]
    if storage < needed * 1024**3:
        raise ValueError("estimated dataset/spill requirement exceeds local instance storage")


def bootstrap(spec, bundle_key, bundle_sha):
    """Small user data; heavyweight code is a checksum-pinned S3 bundle."""
    launch_key = "runs/" + spec["run_id"] + "/attempts/" + spec["attempt"] + "/launch.json"
    import hashlib
    launch_sha = hashlib.sha256(json.dumps(spec, indent=2, sort_keys=True).encode()).hexdigest()
    cli = spec["aws_cli"]
    q = shlex.quote
    return f'''#!/bin/bash
set -euo pipefail
exec > >(tee -a /var/log/join-pruning.log /dev/console) 2>&1
bootstrap_failed() {{
  code=$?
  echo "Bootstrap failed with status $code"
  if command -v aws >/dev/null; then
    aws --region {q(spec['region'])} --cli-connect-timeout 10 --cli-read-timeout 20 s3 cp /var/log/join-pruning.log {q('s3://' + spec['bucket'] + '/runs/' + spec['run_id'] + '/attempts/' + spec['attempt'] + '/bootstrap.log')} || true
  fi
  shutdown -h now
  exit "$code"
}}
trap bootstrap_failed ERR
mkdir -p /opt/join-pruning
# Arm the host timer before networking/package installation can fail.
remaining=$(( ({int(spec['stop_at'])} - $(date +%s)) / 60 ))
if [ "$remaining" -le 0 ]; then shutdown -h now; exit 1; fi
shutdown -h +"$remaining"
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends docker.io python3 curl unzip ca-certificates
curl --fail --location --retry 3 --output /opt/join-pruning/awscliv2.zip {q('https://awscli.amazonaws.com/awscli-exe-linux-aarch64-' + cli['version'] + '.zip')}
echo {q(cli['sha256'] + '  /opt/join-pruning/awscliv2.zip')} | sha256sum -c -
unzip -q /opt/join-pruning/awscliv2.zip -d /opt/join-pruning
/opt/join-pruning/aws/install --update
systemctl enable --now docker
aws --region {q(spec['region'])} s3api get-object --bucket {q(spec['bucket'])} --key {q(launch_key)} /opt/join-pruning/launch.json
echo {q(launch_sha + '  /opt/join-pruning/launch.json')} | sha256sum -c -
aws --region {q(spec['region'])} s3api get-object --bucket {q(spec['bucket'])} --key {q(bundle_key)} /opt/join-pruning/worker.tar.gz
echo {q(bundle_sha + '  /opt/join-pruning/worker.tar.gz')} | sha256sum -c -
tar -xzf /opt/join-pruning/worker.tar.gz -C /opt/join-pruning
cat >/etc/systemd/system/join-pruning.service <<'UNIT'
[Unit]
Description=Join pruning benchmark and durable result delivery
After=docker.service network-online.target
Requires=docker.service
[Service]
Type=exec
WorkingDirectory=/opt/join-pruning
ExecStart=/usr/bin/python3 /opt/join-pruning/host.py
StandardOutput=append:/var/log/join-pruning.log
StandardError=append:/var/log/join-pruning.log
TimeoutStopSec=30
KillMode=control-group
Restart=no
[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable --now join-pruning.service
'''


class Cloud:
    def __init__(self, args):
        self.args = args
        self.settings = json.loads(args.aws_config.read_text())
        for key in ("profile", "region"):
            if getattr(args, key, None):
                self.settings[key] = getattr(args, key)
        self.aws = AWS(self.settings["region"], self.settings["profile"])
        self.stack = self.settings["stack"]
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9-]{0,40}", self.stack):
            raise ValueError("invalid infrastructure stack name")
        self.local = ROOT / "out/aws" / self.stack
        self.local.mkdir(parents=True, exist_ok=True)
        if getattr(args, "run_id", None) and not SAFE.fullmatch(args.run_id):
            raise ValueError("invalid run ID")

    def outputs(self):
        stack = self.aws.command("cloudformation", "describe-stacks", "--stack-name", self.stack)["Stacks"][0]
        if stack["StackStatus"] not in ("CREATE_COMPLETE", "UPDATE_COMPLETE"):
            raise RuntimeError("shared stack is not ready: " + stack["StackStatus"])
        return {x["OutputKey"]: x["OutputValue"] for x in stack["Outputs"]}

    def instances(self, run=None, states=None):
        filters = [{"Name": "tag:BenchStack", "Values": [self.stack]}]
        if run:
            filters.append({"Name": "tag:BenchRun", "Values": [run]})
        if states:
            filters.append({"Name": "instance-state-name", "Values": states})
        result = self.aws.command("ec2", "describe-instances", payload={"Filters": filters})
        return [i for r in result["Reservations"] for i in r["Instances"]]

    def doctor(self):
        identity = self.aws.command("sts", "get-caller-identity")
        instance_type = self.args.instance_type or self.settings["instance_type"]
        instance = self.aws.command("ec2", "describe-instance-types", "--instance-types", instance_type)["InstanceTypes"][0]
        quota = self.aws.command("service-quotas", "get-service-quota", "--service-code", "ec2", "--quota-code", "L-1216C47A")["Quota"]["Value"]
        reservations = self.aws.command("ec2", "describe-instances", payload={"Filters": [
            {"Name": "instance-state-name", "Values": ["pending", "running"]} ]})["Reservations"]
        types = [i["InstanceType"] for r in reservations for i in r["Instances"] if i.get("InstanceLifecycle") != "spot"
                 and i["InstanceType"][0] in "acdhimrtz"]
        usage = 0
        if types:
            sizes = self.aws.command("ec2", "describe-instance-types", "--instance-types", *sorted(set(types)))["InstanceTypes"]
            cpus = {s["InstanceType"]: s["VCpuInfo"]["DefaultVCpus"] for s in sizes}
            usage = sum(cpus[t] for t in types)
        offerings = self.aws.command("ec2", "describe-instance-type-offerings", payload={
            "LocationType": "availability-zone", "Filters": [{"Name": "instance-type", "Values": [instance_type]}]})["InstanceTypeOfferings"]
        ami = self.aws.command("ssm", "get-parameter", "--name", self.settings["ami_parameter"])["Parameter"]["Value"]
        image = self.aws.command("ec2", "describe-images", "--image-ids", ami)["Images"][0]
        if image["Architecture"] != "arm64" or image["OwnerId"] != "099720109477":
            raise ValueError("AMI must be an ARM64 image published by Canonical")
        report = {"account": identity["Account"], "identity": identity["Arn"], "region": self.settings["region"],
                  "instance_type": instance_type, "vcpus": instance["VCpuInfo"]["DefaultVCpus"],
                  "memory_gib": instance["MemoryInfo"]["SizeInMiB"] / 1024,
                  "quota_vcpus": quota, "used_vcpus": usage, "zones": sorted(o["Location"] for o in offerings), "ami": ami,
                  "root_device": image["RootDeviceName"]}
        if self.args.config:
            validate_resources(load_plan(self.args.config), instance)
        report["quota_ok"] = quota - usage >= report["vcpus"]
        try:
            pricing = AWS("us-east-1", self.aws.profile, self.aws.credentials)
            products = pricing.command("pricing", "get-products", payload={"ServiceCode": "AmazonEC2", "Filters": [
                {"Type": "TERM_MATCH", "Field": k, "Value": v} for k, v in {
                    "instanceType": instance_type, "regionCode": self.settings["region"], "operatingSystem": "Linux",
                    "tenancy": "Shared", "preInstalledSw": "NA", "capacitystatus": "Used"}.items()]})["PriceList"]
            rates = [float(d["pricePerUnit"]["USD"]) for p in products
                     for t in json.loads(p)["terms"]["OnDemand"].values() for d in t["priceDimensions"].values()
                     if d["unit"] == "Hrs"]
            report["compute_usd_per_hour"] = min(rates)
            report["compute_usd_at_max_runtime"] = min(rates) * self.settings["max_runtime_seconds"] / 3600
            report["additional_costs"] = "EBS, public IPv4, S3, ECR, logs, watchdog and data transfer are additional; retained storage continues after stop."
        except Exception as exc:
            report["pricing_unavailable"] = str(exc)
        write_json(self.local / "preflight.json", report)
        print(json.dumps(report, indent=2), flush=True)
        return report

    def deploy(self):
        report = self.doctor()
        if not report["zones"]:
            raise RuntimeError("requested instance type is not offered in this region")
        path = self.local / "shared.json"
        write_json(path, template())
        self.aws.command("cloudformation", "validate-template", "--template-body", "file://" + str(path))
        command = ["aws", "--profile", self.settings["profile"], "--region", self.settings["region"],
                   "cloudformation", "deploy", "--stack-name", self.stack, "--template-file", str(path),
                   "--capabilities", "CAPABILITY_NAMED_IAM", "--parameter-overrides", "AvailabilityZone=" + report["zones"][0],
                   "--no-fail-on-empty-changeset", "--tags", "BenchStack=" + self.stack]
        subprocess.run(command, check=True)
        write_json(self.local / "outputs.json", self.outputs())
        print("Shared infrastructure ready. No EC2 instance has been launched.")

    def publish(self):
        try:
            inspected = subprocess.check_output(["docker", "image", "inspect", self.args.image],
                                                stderr=subprocess.PIPE, text=True)
        except FileNotFoundError:
            raise SystemExit("Error: Docker CLI not found. Install Docker and start its daemon before publishing.") from None
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or "").strip()
            if "no such image" in detail.lower():
                build = shlex.join(["./bench", "build", "--target", "paper", "--image", self.args.image])
                raise SystemExit(
                    f"Error: Docker image {self.args.image!r} is missing from the active Docker daemon.\n"
                    "Check `docker context show` and any DOCKER_HOST/DOCKER_CONTEXT overrides.\n"
                    f"Build the complete paper image in the same context with:\n  {build}\n"
                    "Then retry your publish command."
                ) from None
            raise SystemExit(f"Error: Docker could not inspect {self.args.image!r}:\n"
                             f"{detail or 'docker image inspect failed'}") from None
        info = json.loads(inspected)[0]
        if info["Architecture"] != "arm64" or info["Os"] != "linux":
            raise ValueError("publish requires a Linux ARM64 image")
        outputs = self.outputs()
        # Docker Desktop's containerd store reports the OCI index as .Id;
        # classic Linux Docker reports the image config digest. Obtain the
        # canonical config digest from the local export rather than conflating
        # either of those IDs with the registry's platform manifest digest.
        import hashlib
        with tempfile.TemporaryDirectory(dir=self.local) as directory:
            exported = Path(directory) / "image.tar"
            subprocess.run(["docker", "image", "save", "--output", str(exported), self.args.image], check=True)
            with tarfile.open(exported) as tar:
                entries = json.load(tar.extractfile("manifest.json"))
                configs = [tar.extractfile(entry["Config"]).read() for entry in entries]
                configs = [blob for blob in configs if json.loads(blob).get("architecture") == "arm64"]
                if len(configs) != 1:
                    raise ValueError("image export must contain exactly one ARM64 runtime config")
                config_id = "sha256:" + hashlib.sha256(configs[0]).hexdigest()
        tag = "arm64-" + config_id.split(":")[1][:32]
        target = outputs["Repository"] + ":" + tag
        auth = self.aws.command("ecr", "get-authorization-token")["authorizationData"][0]
        password = base64.b64decode(auth["authorizationToken"]).decode().split(":", 1)[1]
        subprocess.run(["docker", "login", "--username", "AWS", "--password-stdin", auth["proxyEndpoint"]], input=password, text=True, check=True)
        subprocess.run(["docker", "tag", self.args.image, target], check=True)
        existing = self.aws.command("ecr", "list-images", "--repository-name", self.stack)["imageIds"]
        if not any(i.get("imageTag") == tag for i in existing):
            # Publish a single platform manifest. Docker's attestation/index
            # push may otherwise try to replace the same immutable ECR tag.
            subprocess.run(["docker", "push", "--platform", "linux/arm64", target], check=True)
        detail = self.aws.command("ecr", "describe-images", "--repository-name", self.stack, "--image-ids", "imageTag=" + tag)["imageDetails"][0]
        remote = self.aws.command("ecr", "batch-get-image", "--repository-name", self.stack, "--image-ids", "imageTag=" + tag)
        manifest = json.loads(remote["images"][0]["imageManifest"])
        if manifest.get("config", {}).get("digest") != config_id:
            raise ValueError("immutable ECR tag does not contain the expected single-platform image")
        published = {"image": outputs["Repository"] + "@" + detail["imageDigest"], "image_id": config_id,
                     "local_image_id": info["Id"], "at": time.time()}
        write_json(self.local / "image.json", published)
        if not SAFE.fullmatch(self.args.name):
            raise ValueError("invalid published image name")
        self.aws.put_json(outputs["Bucket"], "runs/published-images/" + self.args.name + ".json", published)
        if self.args.name == "native":
            self.aws.put_json(outputs["Bucket"], "runs/published-image.json", published)
        print(json.dumps(published, indent=2))

    def bundle(self, bucket):
        path = self.local / "worker.tar.gz"
        with tarfile.open(path, "w:gz") as tar:
            for name in ("common.py", "host.py", "scratch.py", "checkpoint_resume.py"):
                tar.add(ROOT / "infra/aws" / name, arcname=name)
        checksum = digest(path)
        key = "bundles/" + checksum + ".tar.gz"
        self.aws.put(bucket, key, path)
        return key, checksum

    def launch(self, resume=False):
        run = self.args.run_id
        duration = self.args.max_runtime_seconds if self.args.max_runtime_seconds is not None else self.settings["max_runtime_seconds"]
        if duration < 600 or duration > self.settings["max_runtime_seconds"]:
            raise ValueError("runtime must be between 600 seconds and the configured maximum")
        if not SAFE.fullmatch(run):
            raise ValueError("run ID must be a safe name of at most 64 characters")
        outputs = self.outputs()
        bucket, prefix = outputs["Bucket"], "runs/" + run
        if self.instances(run, ["pending", "running", "stopping"]):
            raise RuntimeError("this run already has an active instance")
        restore = None
        if resume:
            original = self.aws.get_json(bucket, prefix + "/definition.json")
            config, published = original["config"], original["published"]
            try:
                restore = self.aws.get_json(bucket, prefix + "/latest.json")
            except RuntimeError as exc:
                if "NoSuchKey" not in str(exc):
                    raise
            if restore and restore.get("experiment_status") == "complete":
                raise ValueError("run is already complete; fetch its results")
        else:
            config = load_plan(self.args.config)
            if not SAFE.fullmatch(self.args.image_name):
                raise ValueError("invalid published image name")
            key = "runs/published-image.json" if self.args.image_name == "native" else "runs/published-images/" + self.args.image_name + ".json"
            published = self.aws.get_json(bucket, key)
        needs_s3 = uses_s3(config)
        if needs_s3 and not outputs.get("ScratchBucket"):
            raise ValueError("DuckLake S3 requires the updated stack; run ./bench ec2 deploy first")
        instance_type = self.args.instance_type or (original["instance_type"] if resume else self.settings["instance_type"])
        self.args.instance_type = instance_type
        report = self.doctor()
        if not report["quota_ok"]:
            raise RuntimeError("insufficient On-Demand vCPU quota")
        if outputs["AvailabilityZone"] not in report["zones"]:
            raise ValueError("stack subnet does not offer the requested instance type")
        details = self.aws.command("ec2", "describe-instance-types", "--instance-types", instance_type)["InstanceTypes"][0]
        validate_resources(config, details)
        if resume and instance_type != original["instance_type"]:
            raise ValueError("resume requires the original instance type for comparable results")
        # Verify the already-installed independent watchdog before allocating compute.
        rule = self.aws.command("events", "describe-rule", "--name", outputs["WatchdogRule"])
        if rule["State"] != "ENABLED":
            raise RuntimeError("shutdown watchdog is disabled")
        health = self.local / "watchdog-health.json"
        result = self.aws.command("lambda", "invoke", "--function-name", outputs["Watchdog"], health)
        if result.get("FunctionError"):
            raise RuntimeError("shutdown watchdog health check failed: " + health.read_text())
        attempt = uuid.uuid4().hex
        # A conditional lease makes simultaneous launches fail atomically.
        condition = {"if_none_match": "*"}
        if resume:
            try:
                previous, header = self.aws.get_json(bucket, prefix + "/lease.json", metadata=True)
            except RuntimeError as exc:
                if "NoSuchKey" not in str(exc):
                    raise
                previous = None  # Definition persisted, but reservation never did.
            if previous:
                known = [i for i in self.instances(run) if any(t["Key"] == "BenchAttempt" and t["Value"] == previous["attempt"]
                                                             for t in i.get("Tags", []))]
                if any(i["State"]["Name"] not in ("stopped", "terminated") for i in known):
                    raise RuntimeError("the previous lease still owns an active instance")
                if not known and time.time() < previous["at"] + 300:
                    raise RuntimeError("another launch is reserved; retry after its five-minute reservation expires")
                if previous.get("instance_id") and not known:
                    # Don't interpret eventual consistency or missing permissions as
                    # proof that a known allocation no longer exists.
                    try:
                        described = self.aws.command("ec2", "describe-instances", "--instance-ids", previous["instance_id"])
                    except RuntimeError as exc:
                        if "InvalidInstanceID.NotFound" not in str(exc) or time.time() < previous["at"] + 300:
                            raise
                    else:
                        if any(i["State"]["Name"] not in ("stopped", "terminated")
                               for r in described["Reservations"] for i in r["Instances"]):
                            raise RuntimeError("the previous lease still owns an active instance")
                condition = {"if_match": header["ETag"]}
        if not resume:
            self.aws.put_json(bucket, prefix + "/definition.json", {"config": config, "published": published,
                              "instance_type": instance_type, "ami": report["ami"], "root_device": report["root_device"]}, if_none_match="*")
        lease = {"attempt": attempt, "at": time.time(), "phase": "launching"}
        reservation = self.aws.put_json(bucket, prefix + "/lease.json", lease, **condition)
        bundle_key, bundle_sha = self.bundle(bucket)
        start = time.time()
        reserve = max(240, min(self.settings["finalization_seconds"], duration / 4))
        spec = {"region": self.settings["region"], "bucket": bucket, "run_id": run, "attempt": attempt,
                "scratch_bucket": outputs.get("ScratchBucket") if needs_s3 else None,
                "aws_cli": self.settings["aws_cli"],
                "config": config, **published, "restore": restore, "started_at": start,
                "stop_at": start + duration - 120, "measurement_deadline": start + duration - reserve,
                "maximum_lifetime_seconds": duration, "instance_type": instance_type,
                "ami": original["ami"] if resume else report["ami"], "worker_bundle_sha256": bundle_sha}
        self.aws.put_json(bucket, prefix + "/attempts/" + attempt + "/launch.json", spec)
        write_json(self.local / (run + ".json"), spec)
        tags = [{"Key": k, "Value": str(v)} for k, v in {"Name": self.stack + ":" + run,
                "BenchStack": self.stack, "BenchRun": run, "BenchAttempt": attempt, "BenchStopAt": int(spec["stop_at"])}.items()]
        payload = {"ImageId": spec["ami"], "InstanceType": instance_type, "MinCount": 1, "MaxCount": 1,
                   "ClientToken": attempt, "IamInstanceProfile": {"Arn": outputs["InstanceProfile"]},
                   "InstanceInitiatedShutdownBehavior": "stop", "MetadataOptions": {"HttpTokens": "required", "HttpPutResponseHopLimit": 2 if needs_s3 else 1},
                   "NetworkInterfaces": [{"DeviceIndex": 0, "SubnetId": outputs["Subnet"], "Groups": [outputs["SecurityGroup"]],
                                           "AssociatePublicIpAddress": True, "DeleteOnTermination": True}],
                   "BlockDeviceMappings": [{"DeviceName": name, "Ebs": {"VolumeSize": size, "VolumeType": "gp3",
                                             "Encrypted": True, "DeleteOnTermination": False}}
                                            for name, size in [(original["root_device"] if resume else report["root_device"], self.settings["root_volume_gib"]),
                                                               ("/dev/sdf", self.settings["results_volume_gib"])]],
                   "TagSpecifications": [{"ResourceType": kind, "Tags": tags} for kind in ("instance", "volume")],
                   "UserData": base64.b64encode(bootstrap(spec, bundle_key, bundle_sha).encode()).decode()}
        if len(base64.b64decode(payload["UserData"])) > 16384:
            raise ValueError("EC2 user data exceeds 16 KiB")
        # Dry-run verifies launch/PassRole permissions without allocating resources.
        try:
            self.aws.command("ec2", "run-instances", payload={**payload, "DryRun": True})
        except RuntimeError as exc:
            if "DryRunOperation" not in str(exc):
                raise
        owner = self.aws.get_json(bucket, prefix + "/lease.json")
        if owner["attempt"] != attempt or time.time() >= lease["at"] + 240:
            raise RuntimeError("launch reservation expired; resume the saved definition")
        launched = self.aws.command("ec2", "run-instances", payload=payload, timeout=45)["Instances"][0]
        spec["instance_id"] = launched["InstanceId"]
        write_json(self.local / (run + ".json"), spec)
        self.aws.put_json(bucket, prefix + "/lease.json", {**lease, "phase": "allocated", "instance_id": spec["instance_id"]},
                          if_match=reservation["ETag"])
        self.aws.put_json(bucket, prefix + "/attempts/" + attempt + "/instance.json", spec)
        print(json.dumps({"run_id": run, "instance_id": spec["instance_id"], "stop_at": spec["stop_at"], "type": instance_type}, indent=2))

    def status(self):
        outputs = self.outputs()
        instances = self.instances(self.args.run_id)
        report = {"instances": [{"id": i["InstanceId"], "state": i["State"]["Name"], "type": i["InstanceType"]} for i in instances]}
        for name in ("status", "latest"):
            try:
                value = self.aws.get_json(outputs["Bucket"], "runs/" + self.args.run_id + "/" + name + ".json")
                if name == "latest":
                    value["file_count"] = len(value.pop("files", {}))
                    value["checkpoint_parts"] = len(value.pop("parts", [None]))
                report[name] = value
            except RuntimeError as exc:
                report[name] = {"unavailable": str(exc)}
        print(json.dumps(report, indent=2))

    def fetch(self):
        outputs = self.outputs()
        marker = self.aws.get_json(outputs["Bucket"], "runs/" + self.args.run_id + "/latest.json")
        destination = self.args.results.resolve()
        target = destination / self.args.run_id
        if target.exists():
            raise FileExistsError(f"refusing to overwrite {target}; choose --results")
        destination.mkdir(parents=True, exist_ok=True)
        archive = destination / (self.args.run_id + ".tar.gz")
        with tempfile.TemporaryDirectory(dir=destination) as temp:
            restore_checkpoint(self.aws, outputs["Bucket"], marker, temp, self.args.run_id)
            Path(temp, self.args.run_id).rename(target)
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(target, arcname=self.args.run_id)
        archive.with_suffix(archive.suffix + ".sha256").write_text(digest(archive) + "  " + archive.name + "\n")
        write_json(destination / (self.args.run_id + ".delivery.json"), marker)
        print(f"Verified results: {target}")

    def stop(self):
        instances = self.instances(self.args.run_id, ["running", "pending"])
        if instances:
            self.aws.command("ec2", "stop-instances", "--instance-ids", *[i["InstanceId"] for i in instances])
        print("Stop requested. EBS artifacts are retained; active measurements may need retrying.")

    def cleanup(self):
        instances = self.instances(self.args.run_id, ["pending", "running", "stopping", "stopped"])
        if any(i["State"]["Name"] != "stopped" for i in instances):
            raise ValueError("cleanup requires stopped instances")
        # Default cleanup never deletes the only copy of undelivered data.
        if self.args.delete_volumes:
            bucket = self.outputs()["Bucket"]
            for instance in instances:
                tags = {t["Key"]: t["Value"] for t in instance["Tags"]}
                marker = self.aws.get_json(bucket, "runs/" + self.args.run_id + "/attempts/" + tags["BenchAttempt"] + "/delivered.json")
                if not marker.get("final"):
                    raise ValueError("no verified final delivery for this instance")
        for instance in instances:
            self.aws.command("ec2", "terminate-instances", "--instance-ids", instance["InstanceId"])
        # Volumes are retained unless explicitly requested; repeated cleanup can
        # find orphaned volumes through their run tags after termination.
        if self.args.delete_volumes:
            ids = [i["InstanceId"] for i in instances]
            if ids:
                self.aws.command("ec2", "wait", "instance-terminated", "--instance-ids", *ids, timeout=300)
            volumes = self.aws.command("ec2", "describe-volumes", payload={"Filters": [
                {"Name": "tag:BenchStack", "Values": [self.stack]}, {"Name": "tag:BenchRun", "Values": [self.args.run_id]}]})["Volumes"]
            for volume in volumes:
                tags = {t["Key"]: t["Value"] for t in volume["Tags"]}
                marker = self.aws.get_json(bucket, "runs/" + self.args.run_id + "/attempts/" + tags["BenchAttempt"] + "/delivered.json")
                if not marker.get("final"):
                    raise ValueError("no verified final delivery for this volume")
                if volume["State"] == "available":
                    self.aws.command("ec2", "delete-volume", "--volume-id", volume["VolumeId"])
        print("Run compute cleaned up. S3 results, ECR images and shared infrastructure retained.")


def add_parser(commands):
    parser = commands.add_parser("ec2", help="deploy and operate the Graviton suite")
    parser.add_argument("--aws-config", type=Path, default=ROOT / "configs/aws.json")
    parser.add_argument("--profile")
    parser.add_argument("--region")
    actions = parser.add_subparsers(dest="ec2_action", required=True)
    for name in ("doctor", "deploy", "publish", "launch", "resume", "status", "fetch", "stop", "cleanup"):
        action = actions.add_parser(name)
        action.set_defaults(func=run)
        if name in ("doctor", "deploy", "launch", "resume"):
            action.add_argument("--instance-type")
            action.add_argument("--config", type=Path, default=ROOT / "configs/paper.json" if name == "launch" else None)
        if name == "publish":
            action.add_argument("--image", default="join-pruning-suite:local")
            action.add_argument("--name", default="paper", help="published image alias (paper, native or extensions)")
        if name == "launch":
            action.add_argument("--image-name", default="paper")
        if name in ("launch", "resume", "status", "fetch", "stop", "cleanup"):
            action.add_argument("--run-id", required=True)
        if name in ("launch", "resume"):
            action.add_argument("--max-runtime-seconds", type=int)
        if name == "fetch":
            action.add_argument("--results", type=Path, default=ROOT / "out/ec2-results")
        if name == "cleanup":
            action.add_argument("--delete-volumes", action="store_true")


def run(args):
    cloud = Cloud(args)
    if args.ec2_action not in ("doctor", "deploy"):
        identity = cloud.aws.command("sts", "get-caller-identity")
        if identity["Arn"].endswith(":root"):
            print("Using configured root session: AWS does not permit root to assume the operator role. "
                  "Use an IAM identity for routine operation.", flush=True, file=sys.stderr)
        else:
            partition = identity["Arn"].split(":")[1]
            cloud.aws = cloud.aws.assume(f"arn:{partition}:iam::{identity['Account']}:role/{cloud.stack}-operator")
    if args.ec2_action == "resume":
        cloud.launch(resume=True)
    else:
        result = getattr(cloud, args.ec2_action)()
        if args.ec2_action == "doctor" and not result["quota_ok"]:
            raise SystemExit(2)
