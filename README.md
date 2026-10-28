# Prefix Range Filter paper experiments

Requires Git, Python **3.12+**, Docker with Buildx (Linux ARM64 support), and
AWS CLI v2. Authenticate an AWS profile and set `profile` and `region` in
[`configs/aws.json`](configs/aws.json). Deployment needs CloudFormation/IAM
permissions; routine operations assume the created `join-pruning-operator` role.
The default paper host is `m9gd.2xlarge`; `doctor` checks availability, quota,
memory, local disk, and estimated compute cost in your region.

Clone your fork, then run these commands from its root:

```bash
git clone <your-fork-url> join-pruning-benchmark
cd join-pruning-benchmark
./bench build
# Optional local workflow check (downloads ~1.8 GB of JOB inputs):
./bench local all --config configs/smoke-paper.json --run-id smoke-01

./bench ec2 doctor --config configs/paper.json
./bench ec2 deploy
./bench ec2 publish
./bench ec2 launch --run-id paper-01
./bench ec2 status --run-id paper-01
```

The default image contains the native and DuckLake-enabled DuckDB builds, RPT+,
all microbenchmarks, generators, and plotting dependencies. Builds need internet
access and substantial disk space; `./bench build --jobs 4` changes compilation
parallelism. Datasets are downloaded with checksum validation or generated
automatically; no sibling checkout or prebuilt binary is needed.

[`configs/paper.json`](configs/paper.json) runs these groups sequentially in one
container, at the configured paper scales and repetition counts:

| Group | Experiments |
| --- | --- |
| `native` | JPB, string joins, JOB, skewed TPC-H, SSB, all six parameter sweeps, point/range microbenchmarks, JPB filter simulation |
| `ordered-ducklake` | Ordered dimensions through 1B rows, their filter simulations, and DuckLake over S3 |

Each EC2 attempt has a **24-hour** budget including startup and preparation.
Measurements stop at 23h45m to finalize results; shutdown starts at 23h58m.
Instances stop early when finished. If an attempt is incomplete, wait until it
stops, then resume with another 24-hour attempt:

```bash
./bench ec2 resume --run-id paper-01
```

Completed measurements are reused. Data lost from local NVMe are regenerated.
Keep the same run ID, configuration, and published image for resume. An optional
EC2 smoke run, including S3, uses `--config configs/smoke-paper-s3.json
--instance-type m9gd.xlarge --max-runtime-seconds 3600` with a separate run ID.

After `status` reports final delivery, fetch and verify the artifacts:

```bash
./bench ec2 fetch --run-id paper-01
```

Results and a checksummed archive appear under `out/ec2-results/`. Within
`paper-01/`, `summary.json` reports each group's completion; figures are in
`groups/<group>/paper/figures/`. Each group also contains raw timings, profiles,
microbenchmark CSVs, configs, logs, and provenance. Partial results can be
fetched too; fetch refuses to overwrite an existing directory (`--results`
selects another destination). To regenerate figures locally using the image:

```bash
./bench local analyze --run-id paper-01 --results out/ec2-results
```

After fetching, clean up each stopped run (including smoke runs):

```bash
./bench ec2 cleanup --run-id paper-01 --delete-volumes
# To interrupt an active run first:
# ./bench ec2 stop --run-id paper-01
```

Cleanup removes run instances and their delivered EBS volumes. S3 results, ECR
images, and shared infrastructure remain and can incur storage charges. When all
runs are cleaned up, delete the shared stack using your configured profile/region:

```bash
aws --profile join-pruning --region eu-central-1 cloudformation delete-stack --stack-name join-pruning
aws --profile join-pruning --region eu-central-1 cloudformation wait stack-delete-complete --stack-name join-pruning
```

The stack retains its S3 buckets and ECR repository. Archive any wanted results,
then empty/delete these retained resources in the AWS console when no longer
needed. An undelivered run's EBS volumes must be recovered before deletion.
