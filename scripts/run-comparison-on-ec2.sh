#!/usr/bin/env bash
#
# run-comparison-on-ec2.sh — drive the COMETS-vs-spatio-flux comparison
# from this laptop, executing the actual workload on a Ray cluster
# launched from inside the SMS API VPC.
#
# Mechanism (all Docker-based):
#   1. Read the worker image URI from deploy/.spatio-flux-image
#      (written by scripts/build-and-push-image.sh).
#   2. Upload the cluster yaml template + ec2-bootstrap.sh to S3.
#   3. SSM-run ec2-bootstrap.sh on the submit node — it ray-ups the
#      cluster, ray-job-submits the experiment inside the container,
#      and the experiment uploads its own results to S3.
#   4. aws s3 sync results back to ./out/comets_compare/.
#
# Required env (or pass as flags):
#   AWS_PROFILE              SSO profile
#   AWS_DEFAULT_REGION       us-gov-west-1
#   STACK_PREFIX             CFN stack prefix (-s, default smsvpctest)
#   S3_BUCKET                bucket the submit-node IAM role can read+write (-b)

set -euo pipefail

# Top-level wall timer — every exit path prints how long the whole
# orchestration took, from `./scripts/run-comparison-on-ec2.sh` invocation
# to whatever made it exit (success, failure, ctrl-C). Uses EXIT trap so
# it fires regardless of how we leave.
SCRIPT_T0=$(date +%s)
print_elapsed() {
    # `rc` is set by the surrounding trap line (so we can use this fn
    # in chained traps without clobbering each other's $?).
    local rc="${rc:-$?}"
    local elapsed=$(( $(date +%s) - SCRIPT_T0 ))
    local mins=$((elapsed / 60))
    local secs=$((elapsed % 60))
    printf '\n⏱  total elapsed: %dm %ds (rc=%d)\n' "$mins" "$secs" "$rc" >&2
    return "$rc"
}
# Initial trap that just prints elapsed if we exit before WORK_DIR is
# defined (e.g. arg-parse failure). Replaced later with the combined
# rm-then-print version once WORK_DIR exists.
trap 'rc=$?; print_elapsed' EXIT

AWS_PROFILE="${AWS_PROFILE:-}"
AWS_REGION="${AWS_DEFAULT_REGION:-${AWS_REGION:-us-gov-west-1}}"
STACK_PREFIX="${STACK_PREFIX:-smsvpctest}"
S3_BUCKET="${S3_BUCKET:-}"
MODE="large"
N_SHARDS=""
SOLVER=""
KEEP_CLUSTER=0

show_help() {
    cat <<EOF
run-comparison-on-ec2.sh — run the comparison on EC2, results back here.

Usage:
  ./scripts/run-comparison-on-ec2.sh -s <stack> -b <bucket> [OPTIONS]

Required:
  -s, --stack PREFIX     CFN stack prefix (default: ${STACK_PREFIX})
  -b, --bucket NAME      S3 working bucket

Options:
  -m, --mode {small|large|both}   default: ${MODE}
      --n-shards N                 shard count
      --solver NAME                cobra solver (glpk, hybrid, highs_direct)
      --keep-cluster               don't ray-down at the end (\$0.45/hr idle)
  -h, --help                       show this help

Pre-requisites (one-time):
  ./scripts/setup-iam-for-ray.sh   ← grants IAM perms
  ./scripts/build-and-push-image.sh ← builds + pushes the worker Docker image

Env fallbacks for AWS_PROFILE / AWS_DEFAULT_REGION / STACK_PREFIX / S3_BUCKET.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -s|--stack)       STACK_PREFIX="$2"; shift 2 ;;
        -b|--bucket)      S3_BUCKET="$2"; shift 2 ;;
        -m|--mode)        MODE="$2"; shift 2 ;;
        --n-shards)       N_SHARDS="$2"; shift 2 ;;
        --solver)         SOLVER="$2"; shift 2 ;;
        --keep-cluster)   KEEP_CLUSTER=1; shift ;;
        -h|--help)        show_help; exit 0 ;;
        *) echo "Unknown option: $1" >&2; show_help; exit 1 ;;
    esac
done

[[ -z "$AWS_PROFILE" ]] && { echo "AWS_PROFILE required" >&2; exit 1; }
[[ -z "$S3_BUCKET"   ]] && { echo "S3_BUCKET required (-b or env)" >&2; exit 1; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ------- Worker image URI from deploy/.spatio-flux-image ---------------
IMAGE_FILE="$REPO_ROOT/deploy/.spatio-flux-image"
[[ -f "$IMAGE_FILE" ]] || {
    echo "ERROR: $IMAGE_FILE missing." >&2
    echo "       Run ./scripts/build-and-push-image.sh first." >&2
    exit 1
}
IMAGE_URI="$(tr -d '[:space:]' < "$IMAGE_FILE")"

# ------- Optional pre-baked host AMI (rsync pre-installed) -------------
# scripts/bake-rsync-ami.sh writes deploy/.spatio-flux-base-ami when it
# finishes. If present, the bootstrap will use that AMI for cluster
# nodes instead of the stock ECS-optimized AL2 lookup. The bake adds
# rsync, which Ray's autoscaler step 2/7 (file_mounts) needs before
# the docker container is even pulled.
BAKED_AMI_FILE="$REPO_ROOT/deploy/.spatio-flux-base-ami"
BAKED_AMI_ID=""
if [[ -f "$BAKED_AMI_FILE" ]]; then
    BAKED_AMI_ID="$(tr -d '[:space:]' < "$BAKED_AMI_FILE")"
fi

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
S3_PREFIX="s3://${S3_BUCKET}/spatio-flux/runs/${RUN_ID}"

cat <<EOF
→ image:    $IMAGE_URI
→ host AMI: ${BAKED_AMI_ID:-<stock ECS-AL2 via SSM lookup>}
→ region:   $AWS_REGION
→ stack:    $STACK_PREFIX
→ bucket:   s3://$S3_BUCKET
→ run id:   $RUN_ID
→ mode:     $MODE
EOF

# ------- discover submit-node EC2 ID via CFN ---------------------------
SUBMIT_INSTANCE_ID="$(
    aws --profile "$AWS_PROFILE" --region "$AWS_REGION" \
        cloudformation describe-stacks --stack-name "${STACK_PREFIX}-batch" \
        --query "Stacks[0].Outputs[?OutputKey=='SubmitNodeInstanceId'].OutputValue" \
        --output text 2>/dev/null
)"
[[ -z "$SUBMIT_INSTANCE_ID" || "$SUBMIT_INSTANCE_ID" == "None" ]] && {
    echo "ERROR: could not resolve ${STACK_PREFIX}-batch SubmitNodeInstanceId" >&2
    exit 1
}
echo "→ submit:   $SUBMIT_INSTANCE_ID"

# ------- upload the files the submit node needs ------------------------
# ec2-bootstrap.sh: SSM entry point, sets up env + installs boto3.
# ec2_cluster.py:   spatio-flux orchestrator wrapper.
# ec2_ssm.py:       upstream EC2SSMRayCluster (vendored next to
#                   ec2_cluster.py; submit venv has no full
#                   process_bigraph install, restricted egress).
# No source tarball, no COMETS jar (those live inside the Docker image).
PB_CLUSTERS_DIR="$(cd "$REPO_ROOT/.." && pwd)/process-bigraph/process_bigraph/protocols/clusters"
if [[ ! -f "$PB_CLUSTERS_DIR/ec2_ssm.py" ]]; then
    echo "✗ expected upstream ec2_ssm.py at $PB_CLUSTERS_DIR/ec2_ssm.py — clone process-bigraph as a sibling repo first" >&2
    exit 1
fi
echo "→ uploading bootstrap + orchestrator + vendored ec2_ssm.py ..."
aws --profile "$AWS_PROFILE" --region "$AWS_REGION" \
    s3 cp "$REPO_ROOT/scripts/ec2-bootstrap.sh" "${S3_PREFIX}/ec2-bootstrap.sh" >/dev/null
aws --profile "$AWS_PROFILE" --region "$AWS_REGION" \
    s3 cp "$REPO_ROOT/scripts/ec2_cluster.py" "${S3_PREFIX}/ec2_cluster.py" >/dev/null
aws --profile "$AWS_PROFILE" --region "$AWS_REGION" \
    s3 cp "$PB_CLUSTERS_DIR/ec2_ssm.py" "${S3_PREFIX}/ec2_ssm.py" >/dev/null

# ------- send via SSM --------------------------------------------------
BOOTSTRAP_CMD="aws --region '${AWS_REGION}' s3 cp '${S3_PREFIX}/ec2-bootstrap.sh' /tmp/ec2-bootstrap.sh && chmod +x /tmp/ec2-bootstrap.sh && S3_PREFIX='${S3_PREFIX}' STACK_PREFIX='${STACK_PREFIX}' MODE='${MODE}' N_SHARDS='${N_SHARDS}' SOLVER='${SOLVER}' KEEP_CLUSTER='${KEEP_CLUSTER}' IMAGE_URI='${IMAGE_URI}' BAKED_AMI_ID='${BAKED_AMI_ID}' /tmp/ec2-bootstrap.sh"

WORK_DIR="$(mktemp -d)"
trap 'rc=$?; rm -rf "$WORK_DIR"; print_elapsed' EXIT
PARAMS_FILE="$WORK_DIR/ssm-params.json"
python3 -c "import json,sys; print(json.dumps({'commands':[sys.argv[1]]}))" \
    "$BOOTSTRAP_CMD" > "$PARAMS_FILE"

echo "→ kicking off ec2-bootstrap.sh via SSM ..."
COMMAND_ID="$(
    aws --profile "$AWS_PROFILE" --region "$AWS_REGION" \
        ssm send-command \
        --instance-ids "$SUBMIT_INSTANCE_ID" \
        --document-name AWS-RunShellScript \
        --comment "spatio-flux comparison run ${RUN_ID}" \
        --parameters "file://${PARAMS_FILE}" \
        --cloud-watch-output-config "CloudWatchOutputEnabled=true" \
        --query 'Command.CommandId' --output text
)"
echo "   ssm command id: $COMMAND_ID"

# ------- poll: stream the bootstrap log from S3 ------------------------
# The submit node uploads ec2-bootstrap.log to S3 every 20s. Mirror it
# locally and emit only the new lines on each poll so the user sees
# real-time progress (instance launches, docker pulls, ray status,
# experiment output) instead of just dots. Also detect log silence
# as a possible "stuck" indicator.
echo "→ following bootstrap log (refresh every 20s) ..."
LOCAL_LOG="$WORK_DIR/bootstrap.log"
: > "$LOCAL_LOG"
LAST_LINES=0
SILENT_TICKS=0
while :; do
    STATUS="$(aws --profile "$AWS_PROFILE" --region "$AWS_REGION" \
        ssm get-command-invocation \
        --command-id "$COMMAND_ID" \
        --instance-id "$SUBMIT_INSTANCE_ID" \
        --query 'Status' --output text 2>/dev/null || echo Pending)"

    # Pull the latest bootstrap log; show new lines since last poll.
    if aws --profile "$AWS_PROFILE" --region "$AWS_REGION" \
            s3 cp "${S3_PREFIX}/ec2-bootstrap.log" "${LOCAL_LOG}.new" \
            >/dev/null 2>&1; then
        NEW_LINES=$(wc -l < "${LOCAL_LOG}.new")
        if (( NEW_LINES > LAST_LINES )); then
            tail -n +$((LAST_LINES + 1)) "${LOCAL_LOG}.new" | sed 's/^/  │ /'
            LAST_LINES=$NEW_LINES
            SILENT_TICKS=0
        else
            SILENT_TICKS=$((SILENT_TICKS + 1))
            if (( SILENT_TICKS == 6 )); then  # 2 min
                echo "  ⚠ log silent for 2 min (could be a long step like docker pull or experiment runtime)"
            elif (( SILENT_TICKS == 30 )); then  # 10 min
                echo "  ⚠⚠ log silent for 10 min — likely stuck. SSM status: $STATUS"
            fi
        fi
        mv "${LOCAL_LOG}.new" "$LOCAL_LOG"
    fi

    case "$STATUS" in
        Success)
            echo
            echo "   ✓ Success"
            break ;;
        Failed|Cancelled|TimedOut)
            echo
            echo "   ✗ Status=$STATUS"
            echo
            echo "═════════════════════════════════════════════════════════════════════"
            echo "  experiment.log on S3 (full python output)"
            echo "═════════════════════════════════════════════════════════════════════"
            aws --profile "$AWS_PROFILE" --region "$AWS_REGION" \
                s3 cp "${S3_PREFIX}/experiment.log" - 2>/dev/null \
                || echo "(no experiment.log on S3 — experiment didn't get to upload)"
            echo
            echo "═════════════════════════════════════════════════════════════════════"
            echo "  bootstrap log was streamed above; full copies:"
            echo "    aws s3 cp ${S3_PREFIX}/ec2-bootstrap.log -"
            echo "    aws s3 cp ${S3_PREFIX}/experiment.log -"
            echo "═════════════════════════════════════════════════════════════════════"
            exit 1 ;;
    esac
    sleep 20
done

# ------- pull results back ---------------------------------------------
LOCAL_OUT="$REPO_ROOT/out/comets_compare"
mkdir -p "$LOCAL_OUT"
echo "→ syncing results to $LOCAL_OUT ..."
aws --profile "$AWS_PROFILE" --region "$AWS_REGION" \
    s3 sync "${S3_PREFIX}/results/" "$LOCAL_OUT/"

echo
echo "✅ done. Open: $LOCAL_OUT/report.html"
