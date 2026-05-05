# Distributed deployment

## Ray on EC2 (multi-machine)

`ray-ec2-cluster.yaml` spins up a small Ray cluster (1 head + 2-4 workers
on m5.xlarge instances) and provisions spatio-flux on every node.

### Provision

```sh
pip install "ray[default]>=2.10" boto3
aws configure   # if you haven't already
ray up deploy/ray-ec2-cluster.yaml
```

First boot takes ~5-7 minutes (AMI pull, apt install, pip install). Subsequent
`ray up` runs are fast.

### Run a sweep

```sh
ray attach deploy/ray-ec2-cluster.yaml
# inside the head:
COMETS_HOME=/opt/comets python -m spatio_flux.experiments.compare_comets
```

The `RayProcess` actor pool will distribute work across the worker nodes
automatically — Ray's scheduler picks workers with free resources, and pool
actors get placed across the cluster, not pinned to the head.

### Tear down

```sh
ray down deploy/ray-ec2-cluster.yaml
```

### Cost

| component | type | $/hr (on-demand, us-east-1) |
|---|---|---:|
| head | m5.2xlarge | $0.384 |
| worker × 4 | m5.xlarge | $0.192 each = $0.768 |
| **total** | | **~$1.15/hr** |

Drop ~70% by uncommenting the spot-pricing block in the worker config —
acceptable for benchmark sweeps, risky for long jobs.

### Caveats

- The COMETS jar isn't scriptably downloadable; we ship it via `file_mounts`
  from your laptop. Adjust the source path in the YAML to your local copy.
- AMI ID is region-specific. The default is Ubuntu 22.04 in us-east-1.
  Look up the right ID for your region at
  <https://cloud-images.ubuntu.com/locator/ec2/>.
- `ray up` creates a security group + key pair on first invocation. Both
  are managed by Ray and reused across `up`/`down` cycles.

## GovCloud (us-gov-west-1) via SMS API VPC

`ray-govcloud-cluster.yaml` is the variant that runs *inside* the SMS API
VPC, with the cluster head and workers placed on private subnets. You
don't `ray up` from your laptop — the laptop can't reach private EC2s.
Instead, the orchestrator script SSMs into the existing submit-node EC2
(the same one `sms-tunnel.sh` hops through) and lets that drive
`ray up`.

### Workflow

```sh
AWS_PROFILE=<sso-profile> AWS_DEFAULT_REGION=us-gov-west-1 \
    ./scripts/run-comparison-on-ec2.sh \
        -s smsvpctest \
        -b <s3-bucket> \
        --mode large
```

What happens:

1. Laptop bundles the spatio-flux source + COMETS jar into tarballs and
   `aws s3 cp`s them to `s3://<bucket>/spatio-flux/runs/<RUN_ID>/`.
2. Laptop pushes `scripts/ec2-bootstrap.sh` to the same S3 prefix and
   `aws ssm send-command`s the submit node to run it.
3. On the submit node, `ec2-bootstrap.sh`:
   - pulls source + COMETS jar from S3,
   - installs the spatio-flux venv,
   - reads VPC/subnet from instance metadata,
   - resolves the Ubuntu AMI from SSM Parameter Store,
   - renders `ray-govcloud-cluster.yaml` with envsubst,
   - `ray up`s the cluster (idempotent — re-runs reuse it),
   - runs `compare_comets --mode=large --ray-address=<head>:6379`,
   - `aws s3 sync`s `out/comets_compare/` back to S3.
4. Laptop polls SSM until done, then `aws s3 sync`s the results back to
   `out/comets_compare/`.

### Required AWS prerequisites

- The submit-node EC2 IAM role needs:
  - `s3:GetObject` / `s3:PutObject` on the working bucket,
  - `ssm:GetParameter` for the Ubuntu AMI parameter,
  - permission to call the EC2 APIs `ray up` uses (RunInstances,
    DescribeInstances, …) within the SMS VPC, with PassRole rights on
    the `ray-spatio-flux-node` instance profile.
- An IAM instance profile named `ray-spatio-flux-node` with read access
  to the S3 working bucket (so worker nodes can pull artifacts at boot).
- The private subnet must have NAT egress so worker nodes can `apt
  install` and `pip install` from PyPI. Pure airgapped subnets need a
  pre-baked AMI / Docker image instead.

### Cost note

Default sizing is 1× m5.2xlarge head (8 vCPU) + 4× m5.4xlarge workers
(64 vCPU total). At GovCloud on-demand pricing this is ~$3-4/hour — fine
for the ~20-30 minute large-grid sweep, painful if left running.
`idle_timeout_minutes: 15` will scale workers to zero after a quarter-hour
idle; use `ray down` from the submit node to fully tear down.
