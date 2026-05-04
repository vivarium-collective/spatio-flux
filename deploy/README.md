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
