#!/usr/bin/env python3
"""ec2_cluster.py — provision a Ray cluster on EC2 via SSM, run the
spatio-flux comets-comparison experiment, tear down. Now a thin wrapper
around ``process_bigraph.protocols.clusters.ec2_ssm.EC2SSMRayCluster``;
the cluster lifecycle lives upstream so downstream repos all share it.

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
    BAKED_AMI_ID            override default ECS-AL2 image
    IAM_INSTANCE_PROFILE    default ray-spatio-flux-node
    CLUSTER_ID              default sf-<STACK_PREFIX>
"""
from __future__ import annotations

import os
import sys

# Try the canonical upstream import first. On the SMS submit node we
# install only boto3 (no full process_bigraph) — bootstrap vendors
# ec2_ssm.py next to this script in $WORK, so we fall back to a local
# import there. Both paths land at the same EC2SSMRayCluster class.
try:
    from process_bigraph.protocols.clusters.ec2_ssm import EC2SSMRayCluster
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ec2_ssm import EC2SSMRayCluster  # type: ignore


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
    baked_ami_id = os.environ.get("BAKED_AMI_ID") or None
    cluster_id = os.environ.get("CLUSTER_ID") or f"sf-{stack_prefix}"

    print(f"→ image={image_uri}")
    print(f"→ cluster_id={cluster_id}  workers={n_workers}")

    # The container name is "spatio_flux_ray" (NOT the upstream default
    # "process_bigraph_ray") because that's what existing ops scripts —
    # diagnose-current-run.sh, cancel-current-run.sh, etc — grep for. Keep
    # the historical name to avoid touching ops tooling.
    cluster = EC2SSMRayCluster(
        image_uri=image_uri,
        n_workers=n_workers,
        cluster_id=cluster_id,
        keep_cluster=keep_cluster,
        head_instance_type=head_type,
        worker_instance_type=worker_type,
        iam_profile=iam_profile,
        baked_ami_id=baked_ami_id,
        container_name="spatio_flux_ray",
    )

    try:
        with cluster:
            extras = []
            if n_shards:
                extras += ["--n-shards", n_shards]
            if solver:
                extras += ["--solver", solver]
            extras_str = " ".join(extras)
            # Multi-stage: run compare_comets capturing to log, capture
            # exit code, ALWAYS upload the log (so failures have
            # forensics in S3 too), upload results only on success,
            # exit with the experiment's rc. Caller-controlled redirect
            # because the upstream cluster.exec_blocking_on_head treats
            # log_path as streaming-only, not auto-redirect.
            log_path = "/tmp/experiment.log"
            command = (
                f"set +e; "
                f"cd /app/spatio-flux; "
                f": > {log_path}; "
                f"python -u -m spatio_flux.experiments.compare_comets "
                f"  --mode {mode} {extras_str} > {log_path} 2>&1; "
                f"rc=$?; "
                f'echo "=== experiment exit: $rc ===" >> {log_path}; '
                f"aws --region {cluster.region} s3 cp {log_path} "
                f"  {s3_prefix}/experiment.log; "
                f"if [ $rc -eq 0 ]; then "
                f"  aws --region {cluster.region} s3 sync out/comets_compare/ "
                f"  {s3_prefix}/results/; "
                f"fi; "
                f"exit $rc"
            )
            print(f"→ running experiment mode={mode}")
            cluster.exec_blocking_on_head(
                command,
                log_path=log_path,
                timeout_s=3600,
                poll_interval=30.0,
            )
        print("✅ experiment complete")
        return 0
    except Exception:
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
