# Obstacles encountered building Ray-on-EC2 for the COMETS comparison

A chronological record of every meaningful obstacle hit while trying to
get the spatio-flux COMETS comparison running on a Ray cluster launched
from inside the SMS API GovCloud VPC. Written 2026-05-05 mid-session,
before final resolution.

---

## Phase 1 — Architecture and design

### 1. Scope expansion from "extend the report"
The original ask was minor: drop 4x4, add 64x64, add an EC2 implementation.
This expanded into a much larger architectural conversation about
ShardManager design, RayProtocol, shadow processes, and pushing
distributed-transport code upstream to process-bigraph as `[extras]`.

### 2. ShardManager generality
Question of whether the manager should assume a grid (no — it should be
general). Required redesigning around explicit lifecycle (`shutdown_pools()`)
because accumulated Ray actor pools were causing OOM at N=32.

### 3. Shadow process vs. address-based protocol
Two different API shapes for distributed processes. Settled on shadow
process pattern as the more user-friendly entry point.

---

## Phase 2 — Local algorithmic work

### 4. HiGHS-via-optlang was 5–7× slower than GLPK
Wrapper overhead made the supposed-faster solver actually slower than
what we were replacing.

**Resolution**: wrote a bare `highspy` wrapper (`HiGHSFBASolver`) bypassing
optlang. End result: 0.046 ms warm vs GLPK's 0.55 ms — 12× faster than
optlang-HiGHS.

### 5. OOM at N=32 shards
Accumulated Ray actor pools across iterations weren't cleaned up.

**Resolution**: explicit `shutdown_pools()` + `ShardManager` lifecycle
encapsulation. 3.5× speedup at N=32 vs. legacy.

### 6. Per-cell projections needed typed schema dispatch
Projection updates from per-cell processes are int-keyed dict trees onto
Array fields. `apply_updates` had to resolve `combined_schema` against
`self.schema`, and `apply(Array)` had to handle nested-int-key dict updates.

### 7. `resolve_subclass` dropped list-valued schema fields
`Tuple._values` wasn't being recursed. Specificity loss inside
`Tuple`/list-of-Node was the signature. Caused pymunk velocity to
accumulate without proper resolution.

---

## Phase 3 — Deployment infrastructure attempts

### 8. SubnetId vs SubnetIds
Ray autoscaler uses the plural `SubnetIds` field. Initial yaml used the
singular form (which is what the AWS API doc shows). Took a while to
trace the silent failure.

### 9. SG cross-VPC reference issue
Initial `IpPermissions` used `UserIdGroupPairs.GroupId` to reference the
submit node's SG. Doesn't work cross-VPC.

**Resolution**: switched to `IpRanges.CidrIp` with the submit node's `/32`.

### 10. GovCloud restricted-egress NAT speed
First Ubuntu-based approach: apt mirrors hit at ~55 kB/s through the
restricted NAT. A 200MB package set was effectively unreachable.

GPG cache was also corrupted, causing apt to refuse package signatures
even when downloads succeeded.

**Resolution**: pivot away from apt entirely.

### 11. The "we're never apt-ing again" decision
After multiple debug cycles fighting apt under restricted egress, user
explicitly stated this approach wasn't viable.

**Resolution**: bake everything that needs network into a Docker image
on a fast-internet machine (laptop), pull from ECR (S3-backed, fast in-VPC).

Memory saved: `feedback_aws_restricted_egress_use_containers.md`.

### 12. `HOME` unbound in SSM
`aws ssm send-command --document-name AWS-RunShellScript` runs as root
with NO `HOME` or `USER` set. Any reference under `set -u` exits
immediately.

**Resolution**: `export HOME="${HOME:-/root}"` at the top of every
SSM-driven script.

Memory saved: `feedback_ssm_shell_env_gotchas.md`.

### 13. `| tee` masking SSM exit codes
Ending an SSM-driven script with `... | tee /tmp/log` makes `tee`'s exit
code (always 0) win, so genuine failures were reported as Success.

**Resolution**: never end SSM commands with `| tee`. Use internal
logging functions.

### 14. envsubst expanding `$HOME` on submit node
Cluster-yaml templates contain `${ECR_HOST}` etc., but envsubst with no
arg-list expands EVERYTHING including `$HOME` and `$PATH` from the
submit node's shell, producing nonsense.

**Resolution**: whitelist specific vars: `envsubst '$AMI_ID $VPC_ID ...'`.

### 15. Multi-line commands breaking on paste
User fed back that `\`-continuation commands were unreliable when copied
from chat to terminal.

**Resolution**: single-line commands joined with `&&`; values set via
`export VAR=...` once at top of a block.

Memory saved: `feedback_paste_ready_commands.md`.

### 16. Accumulated script cruft
After many iterations, `scripts/run-comparison-on-ec2.sh` and
`scripts/ec2-bootstrap.sh` had grown to ~700 lines combined with dead
code paths (`--bake-ami`, `--local-ray`).

**Resolution**: clean rewrite to ~350 lines total. Drop dead flags.

---

## Phase 4 — Cloud-init timing race against Ray autoscaler

### 17. UserData runs too late in cloud-init
sshd starts in cloud-init's `init` phase. UserData (shell script) runs
in `cloud-final`, much later. Ray's autoscaler succeeds at SSH (~17s
into boot) and immediately attempts file_mounts (rsync) at ~20s. UserData's
`yum install -y rsync` hasn't finished yet. Some runs win the race; most
don't.

There is no cloud-init hook that runs *before* sshd-with-network and
*after* package management is available. `bootcmd` is too early
(no network); `runcmd`/UserData are too late (sshd already up).

### 18. Decision to bake an AMI
After confirming the race is unfixable in UserData, decided to bake
rsync into a custom AMI on top of ECS-optimized AL2.

First bake (`ami-0d4974ccfbadc9a1f`, rsync only) succeeded at file_mounts.

### 19. But that bake failed at docker pull
`docker pull` at step 5/7 returned "no basic auth credentials". Same
race: UserData was still writing `~/.docker/config.json` when Ray ran
docker pull.

### 20. amazon-ecr-credential-helper not pre-installed
Despite ECR being an AWS service, `amazon-ecr-credential-helper` (which
provides `/usr/bin/docker-credential-ecr-login`) is NOT pre-installed
on ECS-optimized AL2. Only the docker daemon is.

**Resolution**: extend the bake to also `yum install
amazon-ecr-credential-helper` and write `~/.docker/config.json` with
`credsStore: ecr-login`.

### 21. Stale Ray config cache caused SG-not-found
After the first cluster failed and was torn down, the next run's
`ray up` tried to launch with the previous run's SG ID (cached). That
SG had been deleted. Got `InvalidGroup.NotFound` for several minutes
before timeout.

**Resolution**: pass `--no-config-cache` to `ray up` so the cache is
rebuilt every run.

---

## Phase 5 — The current rsync mystery (unresolved at time of writing)

After the rsync+credential-helper bake (`ami-0d48caab58c33de76`),
file_mounts started failing again — but in a new way:

```
rsync: connection unexpectedly closed (0 bytes received so far) [sender]
rsync error: error in rsync protocol data stream (code 12) at io.c(232)
[sender=3.4.0]
```

Receiver-side rsync exits before sending any protocol byte.

### 22. Theory: /home/ec2-user root-owned
`mkdir -p /home/ec2-user/.docker` running as root in the bake might have
created `/home/ec2-user` itself if absent, leaving it root-owned. sshd's
`StrictModes` would then misbehave.

**RULED OUT**: diagnostic shows `ec2-user:ec2-user 700` — perms are correct.

### 23. Theory: Shell init writes to stdout (poisons rsync protocol pipe)
`bash -c` for ec2-user might source `/etc/profile.d/*.sh` or `/etc/bashrc`
which print something to stdout. rsync sender would see that as protocol
data and bail.

**RULED OUT**: diagnostic added a stdout-poisoning check —
`stdout bytes: 0`. Shell init is clean.

### 24. Theory: cloud-init still running mid-rsync
First failure showed cloud-init in `modules-final` when rsync ran.

**RULED OUT**: subsequent failures have `cloud-init: done` for 5+ minutes
before rsync attempt — failure persists.

### 25. Theory: rsync 3.4.0 strict args vs 3.1.2 receiver
rsync 3.4.0 (sender on submit, Ubuntu) added `--secluded-args` by
default for CVE-2024-12747/12087/12088. The 3.1.2 receiver (head, AL2)
might not handle the new arg encoding, causing immediate exit.

**RULED OUT**: ran three reproductions in the diagnostic — default,
`RSYNC_OLD_ARGS=1` env var, `--old-args` flag. All three fail identically.

### 26. Theory: literal `~` in destination path
Ray uses `/tmp/.../~/file` as a destination. rsync 3.4.0's stricter
path validation might reject this.

**RULED OUT**: my diagnostic uses `/tmp/rsync-diag-default-$$` (no `~`)
and fails identically.

### 27. Theory: stale leftover state from `--no-reboot` snapshot
`ec2 create-image --no-reboot` captures filesystem state without a
clean shutdown. Could have leftover `/tmp` or `/var/run` artifacts
that confuse rsync.

**Untested** — would require a `reboot=true` bake to compare.

### 28. ECS agent constant container churn
Diagnostics show a Docker container TaskDelete event every 16–17 seconds
on the head. ECS-optimized AL2 ships with the ECS agent enabled by
default; it tries to register with an ECS cluster (which doesn't exist
for our use case) and runs auxiliary containers on a fixed cadence.
Each container creation/teardown can shuffle iptables/bridge state.

**Possibly contributing to** the rsync failure but not confirmed —
mkdir over the same SSH path works seconds before rsync fails.

---

## Phase 6 — Instrumentation buildout

These were process improvements, not problem resolutions:

### 29. Manual SSM probing was friction
User asked: "If you need the information can you put it into the log
instead of asking me to ssm-send into the node?"

**Resolution**: built `collect_diagnostics()` in ec2-bootstrap.sh that
runs in the cleanup trap on any non-zero exit. Dumps:
- EC2 console output of the head (kernel + cloud-init)
- SSM live probe (filesystem, daemons, sshd journal, docker journal)
- Filesystem ownership stats
- rsync stdout-poisoning runtime check
- `/etc/profile.d/` script enumeration
- Three rsync reproduction attempts (default / RSYNC_OLD_ARGS / --old-args)
- rsync `--server` direct invocation (latest addition, pending verification)

### 30. IAM auto-discovery for setup script
`setup-iam-for-ray.sh` originally required `-r <submit-role-name>`,
which the user had to look up manually each time.

**Resolution**: walks `<stack>-batch` CFN output → instance ID →
profile ARN → role name automatically.

### 31. run-comparison-on-ec2.sh failure output cleanup
On failure it was printing SSM stdout/stderr (truncated to 24KB) +
last 200 lines of bootstrap log — overlapping content.

**Resolution**: pull the full bootstrap log from S3 in one block, no
truncation. The diagnostics are appended to the log by the cleanup
trap, so a single failed run prints everything inline.

---

## Phase 7 — What we still don't know

As of this writing, **we don't have a confirmed root cause for the
rsync failure on the rsync+ECR-helper-baked AMI.** All three theories
that seemed plausible were disproven by direct diagnostic data.

The most recently added diagnostic (rsync `--server` invoked directly
via SSM, bypassing ssh entirely) hasn't been run yet. That's the
cleanest discriminator between "rsync is broken in this AMI environment"
vs. "the ssh exec path is broken." Result of that run determines:

- If it works: the issue is in the ssh exec path (PATH not set, PAM
  limit, or similar). Fix is targeted.
- If it fails identically: rsync itself is broken on this AMI. Likely
  cause then is something the bake corrupted (but `which rsync`,
  `rsync --version` both succeed). At that point, sensible move is to
  pivot rather than dig further.

---

## Patterns observed across the obstacles

1. **GovCloud restricted egress is brutal**. Anything that fetches at
   runtime is a cliff. Bake on fast internet, pull from in-VPC
   registries (S3, ECR). Memory saved on this.
2. **cloud-init has no early-enough hook for package install**. If you
   need software present before sshd is reachable, it has to be in the
   AMI. UserData can't get there in time.
3. **Ray's autoscaler has rigid expectations** — rsync over ssh, specific
   path patterns with literal `~`, mkdir then chown then rsync sequence,
   ssh exec without bash --login. Hard to deviate from.
4. **`--no-reboot` snapshots can carry weird state** — possibly relevant
   to current rsync failure (untested).
5. **Diagnostic instrumentation matters early**. Most of the time was
   spent in fast-feedback iteration once the bootstrap log
   self-collected diagnostics. Pre-instrumentation, every cycle was
   "rerun, paste big log, theorize" with high latency.

---

## Time/token cost estimate

Hard to count exactly, but conservatively:
- 4–5 hours of session time.
- Probably a similar amount of tooling/cycles burnt before that.
- Several baked AMIs (small, fast — not expensive)
- Multiple cluster bring-ups, all failed before head was usable
- One head was kept running across iterations via `--keep-cluster`.
  Cost: ~$0.45/hr, currently still running.

---

## Open question: when do we pivot?

Stated commitment from earlier in the session: "If this round fails too,
I'll stop and take stock — maybe we genuinely should pivot to Skypilot
for the run." That commitment has been deferred multiple times.

Reasonable pivots if rsync direct-invocation diagnostic doesn't surface
an obvious fix:

- **Skypilot** — different cluster manager, different file-transport
  approach. Sacrifices the "from-scratch Ray autoscaler" framing of
  the report. ~1 hour estimated to wire up.
- **Stock Amazon Linux 2023** as base — has rsync pre-installed, no ECS
  agent doing constant container churn, newer cloud-init. ~30 min to
  re-bake and try. Still in unknown territory if the failure isn't
  ECS-AL2-specific.
- **Skip Ray autoscaler entirely** — manually orchestrate worker EC2s
  via `aws ec2 run-instances` + a static head node. Lots of rewrite,
  but no rsync-over-ssh in the picture.
