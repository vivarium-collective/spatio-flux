# Running the COMETS comparison on EC2 (GovCloud)

Step-by-step guide for running the spatio-flux vs cometspy comparison
report on the SMS API VPC's Ray cluster. The actual workload runs in-VPC;
your laptop just orchestrates and pulls results back.

This uses the **explicit ShardManager** path (`--mode large`) — the
optimized one that beats cometspy on a real cluster. For the API
showcase using `address: "ray:DynamicFBA"`, see the design notes in
`deploy/README.md` and `run_spatioflux_ray_protocol()` in
`spatio_flux/experiments/compare_comets.py`.

---

## Quick reference

```sh
AWS_PROFILE=<your-sso-profile> AWS_DEFAULT_REGION=us-gov-west-1 \
    ./scripts/run-comparison-on-ec2.sh \
        -s smsvpctest \
        -b <s3-working-bucket> \
        --mode large
```

When this finishes you'll find `out/comets_compare/report.html` in the
repo with two scaling plots and the new EC2-Ray column.

---

## One-time setup

### 1. AWS CLI + Session Manager plugin

Install the AWS CLI v2 and Session Manager plugin if you don't have
them. Same prerequisites as `sms-tunnel.sh`:

- AWS CLI v2 — <https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html>
- Session Manager plugin — <https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-install-plugin.html>

### 2. Authenticate to GovCloud

```sh
aws sso login --profile <your-sso-profile>
```

Verify:

```sh
AWS_PROFILE=<your-sso-profile> AWS_DEFAULT_REGION=us-gov-west-1 \
    aws sts get-caller-identity
```

### 3. Working S3 bucket

Pick or create a bucket the submit-node IAM role can read+write. The
script writes to `s3://<bucket>/spatio-flux/runs/<RUN_ID>/`. If you
don't already have one:

```sh
AWS_PROFILE=<profile> AWS_DEFAULT_REGION=us-gov-west-1 \
    aws s3 mb s3://spatio-flux-runs-<your-account-suffix>
```

Make a note of the bucket name — you'll pass it as `-b`.

### 4. IAM instance profile for Ray worker nodes

Create an IAM instance profile named `ray-spatio-flux-node` (or override
via `IAM_INSTANCE_PROFILE` env var) with:

- `s3:GetObject` on `s3://<your-bucket>/spatio-flux/*` so workers can
  pull the source tarball + COMETS jar at boot,
- `s3:PutObject` on the same prefix so the head node can sync results
  back.

Ray's autoscaler also passes this role to the EC2s it spawns. The
submit-node IAM role needs `iam:PassRole` for it.

### 5. COMETS jar locally

The orchestrator script ships your local COMETS install up to S3. Point
to it via `--comets-src` or the `COMETS_JAR_TARBALL` env var. Default:
`~/comets_install/comets_linux/comets_2.12.5`.

If you don't have it locally:

1. Download from <https://www.runcomets.org> (Google Drive link, manual).
2. Extract somewhere your laptop can reach.
3. Pass the path to the script.

---

## Running it

### The happy path

```sh
cd ~/code/spatio-flux

AWS_PROFILE=<your-sso-profile> AWS_DEFAULT_REGION=us-gov-west-1 \
    ./scripts/run-comparison-on-ec2.sh \
        -s smsvpctest \
        -b <s3-working-bucket> \
        --mode large
```

What you'll see:

```
→ region:      us-gov-west-1
→ stack:       smsvpctest
→ bucket:      s3://<bucket>
→ run id:      20260504T203015Z
→ mode:        large
→ submit node: i-0123456789abcdef0
→ bundling source ...
   source: 4.5M
→ bundling COMETS from /home/.../comets_2.12.5 ...
   comets: 130M
→ uploading to s3://<bucket>/spatio-flux/runs/20260504T203015Z/ ...
→ kicking off ec2-bootstrap.sh on submit node via SSM ...
   ssm command id: abc-123-...
→ waiting for command to finish (this may take a while) ...
.................................................. ← one dot per 30s
   ✓ Success
→ syncing results to ./out/comets_compare/ ...

✅ done. Open: ./out/comets_compare/report.html
```

Total wall time: **20–35 min** for `--mode large` on a fresh cluster
(first `ray up` takes 5–10 min; the actual sweep takes 10–15 min).
Subsequent runs reuse the cluster and finish in 10–15 min.

### Customizing the run

| flag | default | what it does |
|---|---|---|
| `-s, --stack` | `smsvpctest` | CloudFormation stack prefix (the `${STACK}-batch` stack supplies the submit-node EC2). |
| `-b, --bucket` | (required) | S3 bucket for artifacts + results. |
| `-c, --comets-src` | `~/comets_install/comets_linux/comets_2.12.5` | Local COMETS install dir to ship. |
| `-m, --mode` | `large` | `small` (plot A, all 4 local impls), `large` (plot B, cometspy + ray shards), or `both`. |
| `--n-shards N` | `os.cpu_count()` on head | Override the number of shard actors. The cluster yaml provisions 4× m5.4xlarge = 64 vCPU; setting `--n-shards 64` matches that. |
| `--solver NAME` | `highs_direct` for `--mode large` | LP backend. `glpk` (cobra default), `hybrid` (HiGHS via optlang — slower than glpk in this stack, see below), `scipy`, or `highs_direct` (bare highspy with warm-start, **fastest**). |
| `--skip-upload` | off | Don't re-upload source + COMETS to S3. Use after the first run if neither has changed. |
| `--skip-ray-up` | off | Don't `ray up`. Use when the cluster is already up from a previous run. |

### Common patterns

**Iterating on local code changes:**

```sh
# First run — fully fresh
./scripts/run-comparison-on-ec2.sh -s smsvpctest -b <bucket> --mode large

# Subsequent runs — reuse cluster
./scripts/run-comparison-on-ec2.sh -s smsvpctest -b <bucket> \
    --mode large --skip-ray-up
```

**Tighter benchmark (skip the largest grids):**

Edit `LARGE_GRIDS` in `spatio_flux/experiments/compare_comets.py`
(default `[8, 16, 32, 64, 128, 256]`) before running. Or pass through a
narrower override — see the script comments.

**Run only a single grid for fast iteration:**

```sh
# After ray-up, run from your laptop pointed at a one-off N
# (uses the local Composite, not EC2):
.venv/bin/python -m spatio_flux.experiments.compare_comets \
    --mode large --solver highs_direct
# Edit LARGE_GRIDS=[64] in compare_comets.py first.
```

---

## What's happening behind the scenes

```
   your laptop                   submit-node EC2          Ray cluster
   ───────────                   ────────────────         ───────────
   bundle src + COMETS jar
       │
       │ aws s3 cp
       ▼
   s3://bucket/spatio-flux/runs/<RUN_ID>/
       ▲
       │ aws s3 cp + ssm send-command
       ▼
   submit node runs ec2-bootstrap.sh
       │
       │   • aws s3 cp src + comets
       │   • build venv (uv pip install -e .)
       │   • discover VPC/subnet from instance metadata
       │   • envsubst deploy/ray-govcloud-cluster.yaml
       │   • ray up <rendered.yaml>
       │
       │   ray creates head + 4 worker EC2s in same VPC
       │                                       │
       │   ray cluster ready (5-10 min)        │
       │                                       ▼
       │                               head: ┌─────────┐
       │   python -m compare_comets    workers: 4× m5.4xlarge
       │     --mode large              │      └─────────┘
       │     --ray-address <head>      │
       │     --solver highs_direct     │
       │     ────────────────────►     ► ShardManager spawns
       │                                  64 actors across cluster
       │                                  │
       │                                  ► batched dFBA solves,
       │                                    diffusion on head,
       │                                    cometspy on head for
       │                                    comparison
       │                                  │
       │   results written to            │
       │   out/comets_compare/  ◄────────┘
       │
       │   aws s3 sync results to s3://...
       ▼
   results in s3://bucket/.../<RUN_ID>/results/
       │
       │ aws s3 sync (laptop pulls back)
       ▼
   ./out/comets_compare/report.html
```

---

## Output

Everything lands in `./out/comets_compare/`:

| file | what |
|---|---|
| `report.html` | Self-contained HTML report (all plots base64-embedded). Open in any browser. |
| `scaling.png` | Plot A: small-grid comparison of all 4 local implementations (only present if you ran `--mode small` or `both`). |
| `scaling_large.png` | Plot B: large-grid scaling, cometspy vs ray-shards-on-EC2. |
| `scaling.json` / `scaling_large.json` | Raw timing data. Re-running merges new grid sizes; existing ones get overwritten. |
| `compare_n<N>.gif` | Per-grid animations of biomass + substrates. |
| `snapshots_<field>_n<N>.png` | Per-implementation snapshot strips. |
| `biomass_trace_n<N>.png` | Biomass-vs-time trace at the seed cell. |

The intermediate run output also lives in S3 at
`s3://<bucket>/spatio-flux/runs/<RUN_ID>/results/` — useful for sharing
or regenerating reports without re-running the sim.

---

## Tearing down the cluster

The cluster idle-timeout is **15 min** (set in `deploy/ray-govcloud-cluster.yaml`),
so workers scale to zero after a quarter-hour idle. Head stays up
until you tear it down explicitly.

To kill everything:

```sh
# SSM into the submit node and run ray down
AWS_PROFILE=<profile> AWS_DEFAULT_REGION=us-gov-west-1 \
    aws ssm start-session --target <submit-instance-id>

# inside the submit node:
cd /tmp/spatio-flux-run/src
ray down /tmp/spatio-flux-run/cluster.yaml --yes
```

Or via SSM send-command in one shot:

```sh
aws --profile <profile> --region us-gov-west-1 ssm send-command \
    --instance-ids <submit-instance-id> \
    --document-name AWS-RunShellScript \
    --parameters 'commands=["ray down /tmp/spatio-flux-run/cluster.yaml --yes"]'
```

---

## Cost

| component | type | $/hr (govcloud on-demand, approx) |
|---|---|---:|
| head | m5.2xlarge | $0.45 |
| worker × 4 | m5.4xlarge | $0.90 each = $3.60 |
| **total** | | **~$4/hr** |

A typical full sweep + cluster lifetime is ~30 min, so ~$2 per run.
Leave the cluster up across iterations and you'll pay $4/hr until you
tear it down — workers do scale to zero after 15 min idle, so the steady
state when you're not running is just the head ($0.45/hr).

For long benchmarks, uncomment the spot-pricing block in the worker
config in `deploy/ray-govcloud-cluster.yaml` to drop ~70%, with the
caveat that workers can be preempted mid-run.

---

## Troubleshooting

### "could not resolve ${STACK}-batch SubmitNodeInstanceId"

Either the stack prefix is wrong, the stack isn't deployed, or your AWS
credentials don't grant `cloudformation:DescribeStacks` for it.

```sh
# Check what stacks you can see:
AWS_PROFILE=<profile> AWS_DEFAULT_REGION=us-gov-west-1 \
    aws cloudformation list-stacks --query 'StackSummaries[?StackStatus==`CREATE_COMPLETE`].StackName'

# Confirm the batch stack exists:
AWS_PROFILE=<profile> AWS_DEFAULT_REGION=us-gov-west-1 \
    aws cloudformation describe-stacks --stack-name <STACK>-batch
```

### SSM command stays "Pending" forever

The submit-node EC2 needs the SSM agent running and an IAM role with
`AmazonSSMManagedInstanceCore` attached. Verify:

```sh
AWS_PROFILE=<profile> AWS_DEFAULT_REGION=us-gov-west-1 \
    aws ssm describe-instance-information \
        --filters "Key=InstanceIds,Values=<submit-instance-id>"
```

If the instance doesn't appear, the agent isn't reachable. Restart it
or check the IAM role.

### `ray up` fails with "no AMI found"

The `AMI_ID` is resolved from SSM Parameter Store at bootstrap time.
GovCloud's Ubuntu AMI parameter path may differ from the commercial
default. Check:

```sh
AWS_PROFILE=<profile> AWS_DEFAULT_REGION=us-gov-west-1 \
    aws ssm get-parameter \
        --name '/aws/service/canonical/ubuntu/server/22.04/stable/current/amd64/hvm/ebs-gp2/ami-id'
```

If it's missing, supply your own AMI ID via `AMI_ID` env var in the
SSM command, or hard-code in the cluster yaml.

### `apt-get` / `pip install` fails on workers

The worker private subnets need NAT egress for PyPI + Ubuntu mirrors.
If the subnets are fully airgapped, you'll need to either:

- Add a NAT gateway to the worker subnets, or
- Pre-bake an AMI with all dependencies installed (then point
  `AMI_ID` at it), or
- Push to a private PyPI mirror in-VPC and patch `setup_commands` to
  use it.

### "Ray actors OOM at N=128+"

Default is 4× m5.4xlarge = 16 GB/worker. At N=128 (16k cells), the
shard actors hold sizeable cobra Models. If you see memory-pressure
worker kills, edit `deploy/ray-govcloud-cluster.yaml` to bump worker
type to `m5.8xlarge` (32 GB) or `r5.4xlarge` (128 GB).

### Solver "hybrid" runs slower than expected

Known: cobra-via-optlang's HiGHS adapter has ~50× wrapper overhead
that swamps HiGHS's speed advantage on small LPs. For the e. coli core
LP, GLPK at 0.5 ms is faster than `hybrid` at 2.8 ms. Use
`--solver highs_direct` (bare highspy with warm-start, ~46 µs/solve)
for the architecturally-best path. That's the default for `--mode large`.

### Results pull back empty

Check the SSM command's logs:

```sh
aws --profile <profile> --region us-gov-west-1 ssm get-command-invocation \
    --command-id <command-id> --instance-id <submit-instance-id> \
    --query 'StandardOutputContent' --output text | tail -100
```

The bootstrap script also writes `/tmp/ec2-bootstrap.log` on the submit
node and uploads it to `s3://<bucket>/.../ec2-bootstrap.log`.

---

## Files involved

- `scripts/run-comparison-on-ec2.sh` — laptop-side orchestrator.
- `scripts/ec2-bootstrap.sh` — runs on the submit node, drives `ray up` + experiment.
- `deploy/ray-govcloud-cluster.yaml` — Ray cluster yaml, envsubst-templated.
- `deploy/README.md` — broader deployment docs (commercial AWS variant + GovCloud).
- `spatio_flux/experiments/compare_comets.py` — the experiment.
