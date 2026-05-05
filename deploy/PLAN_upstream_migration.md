# Plan: move EC2 deployment scaffolding upstream

Drafted 2026-05-05 after a multi-hour debug session bringing up a Ray
cluster on the SMS API VPC in GovCloud. Saved here so the discoveries
don't have to be re-discovered.

## Context

We landed a working Ray-on-EC2 deployment for the spatio-flux comparison
report from inside `smsvpctest`. The path to working was longer than it
should be — most failures were one-line config issues whose only feedback
loop was "wait 5–14 minutes, read a 500-line log, fix one thing, retry."

The pieces that ended up working — once discovered — are reusable. The
spatio-flux-specific parts are thin. The right home for the rest is
upstream in `process-bigraph` so any future Ray workload gets this for
free.

## What's reusable vs what isn't

### Reusable (general, belongs upstream)

| piece | currently | role |
|---|---|---|
| `scripts/setup-iam-for-ray.sh` | spatio-flux/scripts/ | one-shot IAM bootstrap: cluster-mgmt policy on a submit-node role + worker role + instance profile. Parametrized by submit-role-name + bucket. |
| Cluster yaml's IMDS-based VPC discovery | encoded in `ec2-bootstrap.sh` | reads VPC/subnet/SG/IP from instance metadata via IMDSv2, avoids needing `ec2:DescribeInstances` |
| `scripts/teardown-cluster.sh` | spatio-flux/scripts/ | three-pass: graceful `ray down`, force-terminate orphans, delete stale SG |
| `scripts/preview-cluster-yaml.sh` | spatio-flux/scripts/ | local envsubst preview, no AWS calls — saves 5+ minutes per yaml-debug cycle |
| Cluster yaml template | `deploy/ray-govcloud-cluster.yaml` | the actual working configuration, with envsubst placeholders |
| Bootstrap script | `scripts/ec2-bootstrap.sh` | runs on submit node, drives `ray up` + experiment + teardown |
| Orchestrator | `scripts/run-comparison-on-ec2.sh` | laptop-side: bundle + S3 + SSM + poll + sync results |
| Discovered patterns | mostly comments in scripts | self-referencing-port-22 ingress, fail-fast IP-CIDR check, periodic log upload, bash-vs-envsubst variable expansion, IMDSv2 token dance |

### Spatio-flux-specific (stays here)

- `compare_comets.py` itself
- The COMETS jar shipping (general "ship a binary" pattern, but contents are domain-specific)
- The `--mode large` / `LARGE_GRIDS` semantics

## The three paths considered

| path | effort | benefit | downside |
|---|---|---|---|
| Document findings, scripts stay in spatio-flux | hours | next run on *this* VPC is fast | every other process-bigraph user re-discovers everything |
| Move to upstream `process-bigraph[ray-aws]` extra | 1–2 days | future workloads inherit the setup | upstream maintenance burden |
| Switch to Skypilot | 2–3 days | broader cloud coverage, fewer surprises | gives up our `ShardManager` integration; new abstraction to learn |

**Recommended**: middle path. Reserve Skypilot evaluation for when we
need a second cloud provider.

## Concrete migration steps

1. **Create `process_bigraph/deploy/ray_aws/`** in the upstream repo.
   - Move and generalize:
     - `cluster.yaml.j2` (Jinja or envsubst-templated cluster yaml — was deploy/ray-govcloud-cluster.yaml)
     - `bootstrap.sh` (was scripts/ec2-bootstrap.sh)
     - `orchestrator.sh` (was scripts/run-comparison-on-ec2.sh)
     - `teardown.sh` (was scripts/teardown-cluster.sh)
     - `setup_iam.sh` (was scripts/setup-iam-for-ray.sh)
     - `preview_yaml.sh` (was scripts/preview-cluster-yaml.sh)
   - Generalize:
     - SMS-specific values become required arguments (stack-prefix, bucket, IAM role name)
     - Cluster name becomes parameter, not hardcoded
     - Source-tarball path becomes parameter; the orchestrator no longer assumes spatio-flux layout

2. **Add the extra to `pyproject.toml`**:
   ```toml
   [project.optional-dependencies]
   ray-aws = ["ray[default]>=2.10", "boto3"]
   ```

3. **Single CLI entry point**: `python -m process_bigraph.deploy.ray_aws`
   with subcommands:
   - `setup-iam <role-name> <bucket>` → runs setup_iam.sh
   - `up [config.yaml]` → renders + ray up
   - `run <python-module> [args...]` → bundle src + run experiment via SSM
   - `down` → teardown
   - `preview` → render yaml locally

4. **Documentation** at `process-bigraph/docs/deploy_ray_aws.md`:
   - Quickstart for "I have a process-bigraph Composite, I want to run it on EC2 with N workers"
   - Reference for the IAM permissions
   - Troubleshooting matrix (every failure mode we hit, with fix)
   - Note about GovCloud-specific quirks (private VPC, ssmmessages perms, etc.)

5. **Spatio-flux side**: replace `scripts/run-comparison-on-ec2.sh` with
   a thin wrapper:
   ```sh
   python -m process_bigraph.deploy.ray_aws run \
       spatio_flux.experiments.compare_comets \
       --bucket "$S3_BUCKET" \
       --stack-prefix smsvpctest \
       --extra-files ~/comets_install/.../comets_2.12.5 \
       -- --mode large
   ```
   Keeps the COMETS-jar shipping (domain-specific) but offloads
   everything else.

## Specific findings worth documenting (do not lose these)

These cost us at least one full debug cycle each. Bake them into the
upstream docs and/or assertions:

- **GovCloud submit-node SSM access pattern** is different from "ssh
  from laptop"; orchestrator must drive `ray up` *from* the submit
  node, not from the laptop.
- **`SubnetIds` (plural) is honored, `SubnetId` (singular) is not**
  by the autoscaler in recent Ray versions. Singular silently falls
  through to auto-discovery.
- **`provider.security_group.IpPermissions: []` makes the SG have no
  rules**, *not* "let Ray pick defaults." Don't write empty IpPermissions.
- **`UserIdGroupPairs.GroupId` between an autoscaler-created SG and an
  external SG can fail** with `InvalidGroup.NotFound: two resources
  belong to different networks`. Use `IpRanges.CidrIp` instead.
- **IMDSv1 is rejected by 401** on modern AMIs; always use IMDSv2 token
  dance.
- **`envsubst <file` expands every `$VAR`**; pass a whitelist
  (`envsubst '$X $Y' <file`) so `$HOME`, `$PATH`, `$RAY_HEAD_IP` get
  preserved as literal shell variables for the runtime context.
- **`python -m ensurepip` is disabled on Debian/Ubuntu system python**;
  install uv via curl into `~/.local/bin` instead.
- **uv is installed globally, not into the venv**; drive uv from outside
  with `uv pip install --python /path/to/.venv/bin/python ...`.
- **Trailing `| tee` in an SSM command makes any failure look like
  Success** because `tee` exits 0. Either drop the tee or set
  `pipefail` and check `${PIPESTATUS[@]}`.
- **`aws ssm start-session` (interactive) and `aws ssm send-command`
  use different agent paths**; one can fail with `TargetNotConnected`
  while the other works. Always offer both fallback paths in the
  diagnostic helpers.
- **Submit node may be aarch64 even if workers are x86_64**; both
  archs need uv/highspy/cobra wheels.
- **Auto-teardown trap should run before SSM upload of the log** so a
  failed run still leaves the bootstrap log on S3.
- **Periodic background log upload** is essential for any debugging —
  a SIGKILL'd bootstrap leaves no trace if logs only get pushed at end.
- **Fail-fast IP/CIDR check after head launch** saves the 10-minute
  Ray autoscaler wait_ready timeout when SubnetIds didn't take effect.

## Open questions to resolve before opening the upstream PR

- Should the cluster-bring-up part live in `process_bigraph` or in a
  separate `bigraph-deploy` repo? `process_bigraph` is the natural home
  given the Ray protocol already lives there and `ray-aws` is just an
  opt-in extra. Lean: keep it in `process-bigraph` until a second cloud
  appears.
- Skypilot comparison: do a 2-hour spike before committing — get a tiny
  spatio-flux experiment running on Skypilot in our environment. If
  Skypilot handles GovCloud + private VPC + the bastion-via-SSM pattern
  cleanly, the calculus changes.
- ~~Container-first or VM-first?~~ **Resolved: pre-baked AMI is the
  next priority** after the report lands. Evidence: we hit 4+ distinct
  `setup_commands` failures across debug cycles (uv 404, ensurepip
  disabled, GPG cache corruption, slow apt), and each costs 5-15 min.
  A pre-baked AMI eliminates the whole class. Docker is a heavier
  alternative (ECR push, image-per-experiment, network gotchas) and
  isn't needed since one image fits every spatio-flux run. Plan:
  - Manually launch a worker-sized instance with the current AMI.
  - Run our setup_commands by hand, verify ray + cobra + cometspy work.
  - `aws ec2 create-image` to snapshot.
  - Register as `ami-spatio-flux-worker` in the working bucket's region.
  - Update cluster yaml: `ImageId: ami-spatio-flux-worker`, drop
    `setup_commands` entirely. Workers come up in 1-2 min instead of
    8-10. Cycle time on experiment-only changes: same as on the head
    today (just re-upload + re-run).
  - Refresh the AMI when deps change (every few months in practice).

## Pivot: Docker + ECR (executed 2026-05-05)

After hitting four distinct apt/pip failures inside the SMS VPC's
55 kB/s egress to Ubuntu mirrors, switched to a Docker image pre-built
on the laptop. New flow:

  1. `./scripts/build-and-push-image.sh` — laptop side: build image
     with all deps (apt + pip happen here, on fast internet), push to
     GovCloud ECR. Writes URI to `deploy/.spatio-flux-image`.
  2. `./scripts/run-comparison-on-ec2.sh ...` — uses the docker-mode
     cluster yaml. Workers' setup_commands now: only docker.io + awscli
     install + ECR login. Container has spatio-flux + ray + cometspy
     + COMETS jar baked in.

Files added/changed:
  - `deploy/Dockerfile` — single-stage ubuntu:22.04 + python3.11 + uv
    + spatio-flux source + COMETS jar.
  - `scripts/build-and-push-image.sh` — bundle context, build, ECR
    create-if-missing, login, tag, push, write URI to file.
  - `setup-iam-for-ray.sh` — added ECR perms (push for submit-node,
    pull for worker role).
  - `deploy/ray-govcloud-cluster.yaml` — `docker:` block, minimal
    setup_commands, ray start commands run inside container.

Open: still need at least the `docker.io + awscli` apt install on each
worker boot. If apt is unreliable for even those two packages, next
pivot is a tiny pre-baked AMI with just docker installed.

**Update (later same session):** swapped the worker AMI to ECS-optimized
Amazon Linux 2 (`/aws/service/ecs/optimized-ami/amazon-linux-2/recommended/image_id`).
That AMI ships with docker daemon + awscli pre-installed and running.
The cluster yaml's setup_commands shrank to a single line: ECR login.
Plus `initialization_commands: yum install -y rsync` for Ray's file-mount
step. **Zero apt at runtime.**

**Still-open: orchestration-on-arm64-submit-node.** The submit node is
aarch64, our worker image is amd64, so we can't reuse the image for
the orchestration calls (`ray up`, `ray job submit`) the submit node
needs to make. Currently we install ray + boto3 in a venv on the submit
node, which persists across runs. Future cleanup: build the worker
image multi-arch (`docker buildx --platform linux/amd64,linux/arm64`),
publish both tags, run the orchestration commands inside an arm64
container on the submit node. Zero local install. ~30 min of extra
build work.

## Reference: status as of this plan

- Spatio-flux compare_comets has working `run_spatioflux_ray_remote`
  (explicit ShardManager) and `run_spatioflux_ray_protocol` (declarative
  `address: "ray:DynamicFBA"` via the new upstream Ray protocol).
- The tick-phase Protocol API + `RayProtocol` + `_ShardFacade` are in
  `process-bigraph/protocols/ray.py`, with one passing test.
- The HiGHS direct-wrapper (`HiGHSFBASolver`) is in
  `spatio_flux/library/highs_solver.py`.
- The EC2 deployment is *almost* working — last known failure was the
  uv-not-in-venv path issue, fixed but not yet re-tested. The next
  cycle should hit `ray start` on the head, then worker setup, then
  experiment kickoff.
