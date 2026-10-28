"""Independent, minute-period EC2 stop watchdog; deployed before any run."""
import os
import time
import boto3


def handler(event, context):
    ec2 = boto3.client("ec2")
    stopped = []
    pages = ec2.get_paginator("describe_instances").paginate(Filters=[
        {"Name": "tag:BenchStack", "Values": [os.environ["STACK"]]},
        {"Name": "instance-state-name", "Values": ["running", "pending", "stopping"]},
    ])
    for page in pages:
        for reservation in page["Reservations"]:
            for instance in reservation["Instances"]:
                tags = {t["Key"]: t["Value"] for t in instance.get("Tags", [])}
                deadline = float(tags.get("BenchStopAt", "0"))
                if not deadline or time.time() < deadline:
                    continue
                if instance["State"]["Name"] == "pending":
                    continue  # Next invocation catches it when EC2 permits stopping.
                force = time.time() >= deadline + 120
                if instance["State"]["Name"] == "stopping" and not force:
                    continue
                ec2.stop_instances(InstanceIds=[instance["InstanceId"]], Force=force)
                stopped.append(instance["InstanceId"])
    return {"stop_requested": stopped}
