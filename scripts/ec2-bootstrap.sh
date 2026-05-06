#!/usr/bin/env bash
#
# ec2-bootstrap.sh — runs on the SMS submit node, driven by the laptop
# orchestrator via SSM. Invokes the Python cluster orchestrator
# (ec2_cluster.py) which handles cluster lifecycle directly via SSM.
#
# Why this script is small:
# - All the cluster lifecycle logic lives in ec2_cluster.py (SSM-based,
#   replaces Ray's autoscaler).
# - This wrapper just sets up the environment, installs boto3 if needed,
#   sets up live log streaming to S3, and exec's the orchestrator.
#
# Required env (set by run-comparison-on-ec2.sh via SSM):
#   S3_PREFIX        s3://bucket/spatio-flux/runs/<RUN_ID>
#   STACK_PREFIX     CFN stack prefix (e.g. smsvpctest)
#   IMAGE_URI        ECR URI of the worker image
#   MODE             small | large | both
# Optional:
#   N_SHARDS, SOLVER, N_WORKERS, BAKED_AMI_ID, KEEP_CLUSTER
#   HEAD_INSTANCE_TYPE, WORKER_INSTANCE_TYPE

set -euo pipefail

# SSM AWS-RunShellScript runs as root with no HOME / USER. Set defaults
# so `set -u` doesn't trip on the first reference.
export HOME="${HOME:-/root}"
export USER="${USER:-root}"

LOG="/tmp/ec2-bootstrap.log"
LOG_UPLOADER_PID=""

# Truncate log on each run so the live-tail doesn't show stale history.
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

echo "=== ec2-bootstrap @ $(hostname) $(date -u +%FT%TZ) ==="
echo "S3_PREFIX=$S3_PREFIX  STACK=$STACK_PREFIX  MODE=$MODE"

# ----- log uploader (push to S3 every 20s for live tailing) ------------
start_log_uploader() {
    [[ -z "${S3_PREFIX:-}" ]] && return
    (
        while true; do
            sleep 20
            [[ -f "$LOG" ]] && \
                aws --region "$REGION" s3 cp "$LOG" \
                    "${S3_PREFIX}/ec2-bootstrap.log" >/dev/null 2>&1 || true
        done
    ) &
    LOG_UPLOADER_PID=$!
    disown 2>/dev/null || true
}

cleanup() {
    local exit_code=$?
    set +e
    [[ -n "$LOG_UPLOADER_PID" ]] && kill "$LOG_UPLOADER_PID" 2>/dev/null || true
    aws --region "${REGION:-us-gov-west-1}" s3 cp "$LOG" \
        "${S3_PREFIX}/ec2-bootstrap.log" >/dev/null 2>&1 || true
    return $exit_code
}
trap cleanup EXIT

# ----- region from IMDS (one curl, no python yet) ----------------------
TOKEN=$(curl -fsS -X PUT \
    -H 'X-aws-ec2-metadata-token-ttl-seconds: 60' \
    http://169.254.169.254/latest/api/token)
REGION=$(curl -fsS -H "X-aws-ec2-metadata-token: $TOKEN" \
    http://169.254.169.254/latest/meta-data/placement/region)
echo "→ region=$REGION"

start_log_uploader

# ----- install boto3 if needed (idempotent) ----------------------------
# We don't need ray on the submit node anymore — just boto3 to drive the
# cluster orchestrator. Reuse the existing venv if present.
SUBMIT_VENV="$HOME/spatio-flux-submit-venv"
if [[ ! -x "$SUBMIT_VENV/bin/python3" ]]; then
    echo "→ creating submit venv at $SUBMIT_VENV (one-time)"
    if ! command -v uv >/dev/null; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
    fi
    uv venv --python 3.11 "$SUBMIT_VENV"
fi
if ! "$SUBMIT_VENV/bin/python3" -c 'import boto3' 2>/dev/null; then
    echo "→ installing boto3 in submit venv"
    if ! command -v uv >/dev/null; then
        export PATH="$HOME/.local/bin:$PATH"
    fi
    uv pip install --python "$SUBMIT_VENV/bin/python3" boto3
fi

# ----- fetch the orchestrator from S3 ----------------------------------
WORK="/tmp/spatio-flux-run"
mkdir -p "$WORK"
aws --region "$REGION" s3 cp \
    "${S3_PREFIX}/ec2_cluster.py" \
    "$WORK/ec2_cluster.py"
echo "→ orchestrator: $WORK/ec2_cluster.py"

# ----- exec the orchestrator -------------------------------------------
# `-u` forces unbuffered stdout/stderr. Without it, python block-buffers
# when piped through tee, hiding all the orchestrator's progress prints
# until the process exits. With -u, every print() flushes immediately
# and shows up in the live S3 log on the next 20s upload tick.
exec "$SUBMIT_VENV/bin/python3" -u "$WORK/ec2_cluster.py"
