#!/usr/bin/env python3
"""
ec2_cluster.py — provision a Ray cluster on EC2 via SSM, run an experiment,
tear down. Replaces Ray's autoscaler-based bring-up.

Why not Ray's autoscaler:
- Hardcodes rsync over ssh for file_mounts; rsync interop between
  rsync 3.4.0 (Ubuntu submit node) and 3.1.2 (AL2 head) was breaking
  with no diagnosable cause across many debug cycles.
- Caches cluster config under /tmp/ray-config-* and goes stale across
  teardown/redeploy (SG-not-found errors).
- Assumes a stock-Linux shell environment (sources ~/.bashrc, runs
  bash --login -c -i, etc.); breaks on minimal AMIs.

What we use instead:
- aws ec2 run-instances to launch head + workers (parallel)
- SSM agent on each instance (already running on ECS-AL2) for control
- docker login to ECR via aws ecr get-login-password
- docker run --network host so Ray's many ports work without SG fiddling
- ray start --head and ray start --address=<head>:6379 inside the
  container, same image as everything else
- python -m spatio_flux.experiments.compare_comets via docker exec on
  the head, then aws s3 sync results

Required env (set by ec2-bootstrap.sh from the SSM caller):
    S3_PREFIX        e.g. s3://bucket/spatio-flux/runs/<run-id>
    STACK_PREFIX     e.g. smsvpctest
    IMAGE_URI        ECR worker image URI
    MODE             small | large | both
Optional:
    N_WORKERS        default 4
    N_SHARDS         passed through to compare_comets
    SOLVER           passed through to compare_comets
    KEEP_CLUSTER     1 to skip terminate at end
    HEAD_INSTANCE_TYPE      default m5.2xlarge
    WORKER_INSTANCE_TYPE    default m5.4xlarge
"""

from __future__ import annotations

import os
import re
import sys
import time
import urllib.request

import boto3


# ---------------------------------------------------------------------------
# IMDSv2 — discover network setup from the submit node we're running on.
# ---------------------------------------------------------------------------

def imds_get(path: str) -> str:
    token = urllib.request.urlopen(urllib.request.Request(
        "http://169.254.169.254/latest/api/token",
        method="PUT",
        headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
    )).read().decode()
    return urllib.request.urlopen(urllib.request.Request(
        f"http://169.254.169.254/latest/{path}",
        headers={"X-aws-ec2-metadata-token": token},
    )).read().decode()


def discover_network() -> dict:
    region = imds_get("meta-data/placement/region")
    mac = imds_get("meta-data/mac")
    return {
        "region": region,
        "vpc_id": imds_get(f"meta-data/network/interfaces/macs/{mac}/vpc-id"),
        "subnet_id": imds_get(f"meta-data/network/interfaces/macs/{mac}/subnet-id"),
        "sg_id": imds_get(f"meta-data/network/interfaces/macs/{mac}/security-group-ids").split()[0],
        "submit_ip": imds_get("meta-data/local-ipv4"),
    }


# ---------------------------------------------------------------------------
# SSM helpers
# ---------------------------------------------------------------------------

def ssm_run(
    ssm,
    instance_ids: list[str],
    commands: list[str] | str,
    *,
    name: str,
    timeout_s: int = 300,
    poll_interval: float = 3.0,
) -> dict[str, dict]:
    """Send shell commands to instances, wait for all to finish, return invocations."""
    if isinstance(commands, str):
        commands = [commands]
    print(f"  [ssm:{name}] sending to {len(instance_ids)} instance(s)")
    cmd_id = ssm.send_command(
        InstanceIds=instance_ids,
        DocumentName="AWS-RunShellScript",
        Parameters={"commands": commands, "executionTimeout": [str(timeout_s)]},
        Comment=name[:100],
    )["Command"]["CommandId"]

    deadline = time.time() + timeout_s + 30
    pending = set(instance_ids)
    results: dict[str, dict] = {}
    while pending and time.time() < deadline:
        time.sleep(poll_interval)
        for iid in list(pending):
            try:
                inv = ssm.get_command_invocation(CommandId=cmd_id, InstanceId=iid)
            except ssm.exceptions.InvocationDoesNotExist:
                continue
            if inv["Status"] in ("Success", "Failed", "Cancelled", "TimedOut"):
                results[iid] = inv
                pending.discard(iid)
                marker = "✓" if inv["Status"] == "Success" else "✗"
                print(f"  [ssm:{name}] {marker} {iid}: {inv['Status']}")
    for iid in pending:
        results[iid] = {"Status": "TimedOut", "StandardErrorContent": "ssm poll timeout"}
        print(f"  [ssm:{name}] ✗ {iid}: TimedOut")

    failed = {iid: r for iid, r in results.items() if r.get("Status") != "Success"}
    if failed:
        for iid, r in failed.items():
            print(f"  [ssm:{name}]   stderr from {iid}:")
            print(f"    {r.get('StandardErrorContent', '')[:1500]}")
            print(f"  [ssm:{name}]   stdout from {iid}:")
            print(f"    {r.get('StandardOutputContent', '')[:1500]}")
        raise RuntimeError(f"SSM '{name}' failed on {len(failed)}/{len(instance_ids)} instance(s)")
    return results


def wait_ssm_agents(ssm, instance_ids: list[str], *, timeout_s: int = 240) -> None:
    """Block until SSM agent on every instance has reported in."""
    deadline = time.time() + timeout_s
    pending = set(instance_ids)
    while pending and time.time() < deadline:
        resp = ssm.describe_instance_information(
            Filters=[{"Key": "InstanceIds", "Values": list(pending)}],
        )
        ready = {i["InstanceId"] for i in resp.get("InstanceInformationList", [])
                 if i.get("PingStatus") == "Online"}
        pending -= ready
        if pending:
            time.sleep(5)
    if pending:
        raise RuntimeError(f"SSM agent not online after {timeout_s}s: {pending}")


# ---------------------------------------------------------------------------
# Cluster lifecycle
# ---------------------------------------------------------------------------

CLUSTER_TAG = "spatio-flux-cluster"


def launch_instances(
    ec2, *, ami_id: str, head_type: str, worker_type: str, n_workers: int,
    subnet_id: str, sg_id: str, iam_profile: str, cluster_id: str,
) -> tuple[str, list[str]]:
    """Launch head + N workers in two parallel run-instances calls."""
    base_kwargs = dict(
        ImageId=ami_id,
        SubnetId=subnet_id,
        SecurityGroupIds=[sg_id],
        IamInstanceProfile={"Name": iam_profile},
    )

    head_resp = ec2.run_instances(
        **base_kwargs,
        InstanceType=head_type,
        MinCount=1, MaxCount=1,
        BlockDeviceMappings=[{"DeviceName": "/dev/xvda",
                              "Ebs": {"VolumeSize": 80, "VolumeType": "gp3"}}],
        TagSpecifications=[{
            "ResourceType": "instance",
            "Tags": [
                {"Key": "Name", "Value": f"{cluster_id}-head"},
                {"Key": CLUSTER_TAG, "Value": cluster_id},
                {"Key": "spatio-flux-role", "Value": "head"},
            ],
        }],
    )
    head_id = head_resp["Instances"][0]["InstanceId"]

    worker_resp = ec2.run_instances(
        **base_kwargs,
        InstanceType=worker_type,
        MinCount=n_workers, MaxCount=n_workers,
        BlockDeviceMappings=[{"DeviceName": "/dev/xvda",
                              "Ebs": {"VolumeSize": 60, "VolumeType": "gp3"}}],
        TagSpecifications=[{
            "ResourceType": "instance",
            "Tags": [
                {"Key": "Name", "Value": f"{cluster_id}-worker"},
                {"Key": CLUSTER_TAG, "Value": cluster_id},
                {"Key": "spatio-flux-role", "Value": "worker"},
            ],
        }],
    )
    worker_ids = [i["InstanceId"] for i in worker_resp["Instances"]]
    return head_id, worker_ids


def wait_running(ec2, instance_ids: list[str]) -> None:
    waiter = ec2.get_waiter("instance_running")
    waiter.wait(InstanceIds=instance_ids,
                WaiterConfig={"Delay": 5, "MaxAttempts": 60})


def get_private_ips(ec2, instance_ids: list[str]) -> dict[str, str]:
    resp = ec2.describe_instances(InstanceIds=instance_ids)
    ips = {}
    for r in resp["Reservations"]:
        for i in r["Instances"]:
            ips[i["InstanceId"]] = i["PrivateIpAddress"]
    return ips


def find_existing_cluster(ec2, cluster_id: str) -> tuple[str | None, list[str]]:
    """Return (head_id, worker_ids) for a previously-launched cluster, or (None, [])."""
    resp = ec2.describe_instances(Filters=[
        {"Name": f"tag:{CLUSTER_TAG}", "Values": [cluster_id]},
        {"Name": "instance-state-name", "Values": ["pending", "running"]},
    ])
    head, workers = None, []
    for r in resp["Reservations"]:
        for inst in r["Instances"]:
            tags = {t["Key"]: t["Value"] for t in inst.get("Tags", [])}
            if tags.get("spatio-flux-role") == "head":
                head = inst["InstanceId"]
            elif tags.get("spatio-flux-role") == "worker":
                workers.append(inst["InstanceId"])
    return head, workers


# ---------------------------------------------------------------------------
# Docker + Ray bring-up
# ---------------------------------------------------------------------------

def ecr_login_cmd(image_uri: str, region: str) -> str:
    """Shell snippet that authenticates docker to ECR for the given image URI."""
    ecr_host = image_uri.split("/", 1)[0]
    return (f"aws ecr get-login-password --region {region} "
            f"| docker login --username AWS --password-stdin {ecr_host}")


def docker_pull_all(ssm, instance_ids: list[str], image_uri: str, region: str) -> None:
    ssm_run(ssm, instance_ids, [
        "set -e",
        ecr_login_cmd(image_uri, region),
        f"docker pull {image_uri}",
    ], name="docker-pull", timeout_s=600)


def start_ray_head(ssm, head_id: str, image_uri: str) -> None:
    ssm_run(ssm, [head_id], [
        "set -e",
        "docker rm -f spatio_flux_ray 2>/dev/null || true",
        # --network host: Ray uses many random ports for object manager,
        # gcs, raylet, dashboard agent etc. Putting the container on the
        # host network avoids needing to map each one and avoids docker
        # bridge NAT for inter-node Ray traffic.
        # --restart unless-stopped: survives daemon restart for long runs.
        # ray start --block keeps PID 1 alive so docker doesn't exit.
        f"docker run -d --name spatio_flux_ray --network host --restart unless-stopped "
        f"  --entrypoint ray {image_uri} "
        f"  start --head --port=6379 --dashboard-host=0.0.0.0 --block",
        "sleep 5",
        # Sanity: did ray actually come up?
        "docker logs --tail 30 spatio_flux_ray",
    ], name="start-head", timeout_s=180)


def start_ray_workers(ssm, worker_ids: list[str], image_uri: str, head_ip: str) -> None:
    ssm_run(ssm, worker_ids, [
        "set -e",
        "docker rm -f spatio_flux_ray 2>/dev/null || true",
        f"docker run -d --name spatio_flux_ray --network host --restart unless-stopped "
        f"  --entrypoint ray {image_uri} "
        f"  start --address={head_ip}:6379 --block",
        "sleep 5",
        "docker logs --tail 30 spatio_flux_ray",
    ], name="start-workers", timeout_s=180)


def wait_workers_registered(ssm, head_id: str, expected: int, *, timeout_s: int = 180) -> None:
    """Poll `ray status` on the head until N workers are connected."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            res = ssm_run(ssm, [head_id], [
                "docker exec spatio_flux_ray ray status 2>&1 | head -50",
            ], name="ray-status", timeout_s=30)
            out = res[head_id]["StandardOutputContent"]
            # `ray status` lists active nodes as ` node_<hex>`; counting
            # distinct hex IDs gives a reliable node count across Ray
            # versions. Hex IDs are typically 32+ chars.
            ids = set(re.findall(r"node_[a-f0-9]{20,}", out))
            print(f"  ray status nodes seen: {len(ids)} (need {expected + 1})")
            if len(ids) >= expected + 1:
                print(f"  ✓ all nodes registered")
                return
        except Exception as e:
            print(f"  ray status query failed: {e}")
        time.sleep(5)
    raise RuntimeError(f"workers didn't register within {timeout_s}s")


def run_experiment(
    ssm, head_id: str, *,
    mode: str, n_shards: str, solver: str,
    s3_prefix: str, region: str,
    timeout_s: int = 3600,
) -> None:
    extras = []
    if n_shards:
        extras += ["--n-shards", n_shards]
    if solver:
        extras += ["--solver", solver]
    extras_str = " ".join(extras)
    ssm_run(ssm, [head_id], [
        "set -e",
        # Run the experiment inside the head's running ray container so
        # ray.init() inside compare_comets connects to the local cluster.
        f"docker exec spatio_flux_ray bash -c '"
        f"  cd /app/spatio-flux && "
        f"  python -m spatio_flux.experiments.compare_comets --mode {mode} {extras_str} && "
        f"  aws --region {region} s3 sync out/comets_compare/ {s3_prefix}/results/'",
    ], name="experiment", timeout_s=timeout_s)


def collect_node_diag(ssm, instance_ids: list[str], label: str) -> None:
    """Best-effort dump of container state on each node — stays in the bootstrap log."""
    try:
        ssm_run(ssm, instance_ids, [
            "echo '=== docker ps ===' && docker ps -a",
            "echo '=== spatio_flux_ray logs (tail 100) ===' "
            "&& docker logs --tail 100 spatio_flux_ray 2>&1 || true",
            "echo '=== systemd docker ===' && systemctl is-active docker",
            "echo '=== disk ===' && df -h / /var/lib/docker 2>&1 | head -5",
        ], name=f"diag-{label}", timeout_s=60)
    except Exception as e:
        print(f"  diag collection failed: {e}")


def terminate(ec2, instance_ids: list[str]) -> None:
    if not instance_ids:
        return
    print(f"→ terminating {len(instance_ids)} instance(s): {instance_ids}")
    ec2.terminate_instances(InstanceIds=instance_ids)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    s3_prefix = os.environ["S3_PREFIX"]
    stack_prefix = os.environ["STACK_PREFIX"]
    image_uri = os.environ["IMAGE_URI"]
    mode = os.environ["MODE"]
    n_workers = int(os.environ.get("N_WORKERS") or "4")
    n_shards = os.environ.get("N_SHARDS") or ""
    solver = os.environ.get("SOLVER") or ""
    keep_cluster = bool(int(os.environ.get("KEEP_CLUSTER") or "0"))
    head_type = os.environ.get("HEAD_INSTANCE_TYPE") or "m5.2xlarge"
    worker_type = os.environ.get("WORKER_INSTANCE_TYPE") or "m5.4xlarge"
    iam_profile = os.environ.get("IAM_INSTANCE_PROFILE") or "ray-spatio-flux-node"
    cluster_id = os.environ.get("CLUSTER_ID") or f"sf-{stack_prefix}"

    net = discover_network()
    region = net["region"]

    print(f"→ region={region} subnet={net['subnet_id']} sg={net['sg_id']}")
    print(f"→ image={image_uri}")
    print(f"→ cluster_id={cluster_id}  workers={n_workers}")

    ec2 = boto3.client("ec2", region_name=region)
    ssm = boto3.client("ssm", region_name=region)

    # AMI: prefer baked, fall back to ECS-AL2. The ec2_cluster.py
    # control plane doesn't depend on rsync or amazon-ecr-credential-helper
    # — those were only needed to satisfy Ray's autoscaler — but if
    # someone already baked an AMI it works fine.
    ami_id = os.environ.get("BAKED_AMI_ID") or ""
    if not ami_id:
        ami_id = ssm.get_parameter(
            Name="/aws/service/ecs/optimized-ami/amazon-linux-2/recommended/image_id",
        )["Parameter"]["Value"]
    print(f"→ ami={ami_id}")

    # Idempotent: reuse an existing tagged cluster if one's already up.
    head_id, worker_ids = find_existing_cluster(ec2, cluster_id)
    launched_now = False
    if head_id and len(worker_ids) >= n_workers:
        print(f"→ reusing existing cluster: head={head_id} workers={worker_ids}")
    else:
        # Stale partial cluster — terminate before relaunching.
        if head_id or worker_ids:
            print(f"→ partial existing cluster found, terminating: "
                  f"head={head_id} workers={worker_ids}")
            terminate(ec2, ([head_id] if head_id else []) + worker_ids)
            time.sleep(5)
        print(f"→ launching {n_workers + 1} instances")
        head_id, worker_ids = launch_instances(
            ec2,
            ami_id=ami_id, head_type=head_type, worker_type=worker_type,
            n_workers=n_workers,
            subnet_id=net["subnet_id"], sg_id=net["sg_id"],
            iam_profile=iam_profile, cluster_id=cluster_id,
        )
        launched_now = True
        print(f"→ launched: head={head_id} workers={worker_ids}")

    all_ids = [head_id] + worker_ids
    rc = 1
    try:
        if launched_now:
            print("→ waiting for instances to be running")
            wait_running(ec2, all_ids)
            print("→ waiting for SSM agents (~60-90s on fresh instances)")
            wait_ssm_agents(ssm, all_ids)

        ips = get_private_ips(ec2, all_ids)
        head_ip = ips[head_id]
        print(f"→ head_ip={head_ip}")

        # Always (re)pull and (re)start the container — even on reused
        # clusters. The container can be missing if a previous run
        # crashed mid-bring-up or if docker --restart unless-stopped
        # didn't recover after an instance reboot. docker pull is fast
        # when cached; start_ray_* runs `docker rm -f` first so this is
        # safe and idempotent.
        print("→ docker pull on all nodes (idempotent)")
        docker_pull_all(ssm, all_ids, image_uri, region)
        print("→ starting ray head")
        start_ray_head(ssm, head_id, image_uri)
        print("→ starting ray workers")
        start_ray_workers(ssm, worker_ids, image_uri, head_ip)
        print("→ waiting for workers to register")
        wait_workers_registered(ssm, head_id, n_workers)

        print(f"→ running experiment mode={mode}")
        run_experiment(
            ssm, head_id,
            mode=mode, n_shards=n_shards, solver=solver,
            s3_prefix=s3_prefix, region=region,
        )

        print("✅ experiment complete")
        rc = 0
    except Exception:
        import traceback
        traceback.print_exc()
        print("→ collecting diagnostics from cluster nodes")
        collect_node_diag(ssm, all_ids, "failure")
        rc = 1
    finally:
        if keep_cluster:
            print(f"→ keeping cluster (KEEP_CLUSTER=1).")
            print(f"   tear down later: aws ec2 terminate-instances --region {region} "
                  f"--instance-ids {' '.join(all_ids)}")
        else:
            terminate(ec2, all_ids)

    return rc


if __name__ == "__main__":
    sys.exit(main())
