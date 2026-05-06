#!/usr/bin/env bash
#
# rescue-results.sh — recover results from a hung run (e.g. when python
# wedged at exit and the wrapper bash never got to s3 sync).
#
# Steps performed (all read-only on the cluster except the SSM s3 sync,
# which only writes new objects to your bucket):
#   1. Discover the latest run_id under s3://<bucket>/spatio-flux/runs/
#      (or use --run-id).
#   2. Discover the head node from running cluster instances (handles
#      BOTH process-bigraph-* and spatio-flux-* tag schemes).
#   3. Issue an SSM command on the head to ``aws s3 sync`` from the
#      container's /app/spatio-flux/out/comets_compare/ to
#      s3://<bucket>/.../results/.
#   4. ``aws s3 sync`` from S3 to the laptop's ./out/comets_compare/.
#
# Cluster + python stay untouched; you can clean up afterward with
# nuke-all.sh.
#
# Required env: AWS_PROFILE
# Required args: -b/--bucket <bucket>
# Optional args:
#   --run-id <id>     skip auto-discovery, use this run_id
#   --stack <prefix>  default: smsvpctest
#   --region <r>      default: us-gov-west-1

set -uo pipefail

AWS_PROFILE="${AWS_PROFILE:-}"
AWS_REGION="${AWS_DEFAULT_REGION:-${AWS_REGION:-us-gov-west-1}}"
STACK_PREFIX="${STACK_PREFIX:-smsvpctest}"
BUCKET=""
RUN_ID=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -b|--bucket)  BUCKET="$2"; shift 2 ;;
        --run-id)     RUN_ID="$2"; shift 2 ;;
        --stack)      STACK_PREFIX="$2"; shift 2 ;;
        --region)     AWS_REGION="$2"; shift 2 ;;
        -h|--help)    sed -n '2,/^$/p' "$0" | sed 's/^# \?//'; exit 0 ;;
        *) echo "Unknown: $1" >&2; exit 1 ;;
    esac
done

[[ -z "$AWS_PROFILE" ]] && { echo "AWS_PROFILE required" >&2; exit 1; }
[[ -z "$BUCKET" ]]      && { echo "-b/--bucket required" >&2; exit 1; }

aws_call() { aws --profile "$AWS_PROFILE" --region "$AWS_REGION" "$@"; }

S3_BUCKET_URI="s3://${BUCKET}"
RUNS_PREFIX="${S3_BUCKET_URI}/spatio-flux/runs/"

# ---------------------------------------------------------------------
# 1. Discover run_id (or use --run-id)
# ---------------------------------------------------------------------
if [[ -z "$RUN_ID" ]]; then
    echo "═══ discovering latest run_id under $RUNS_PREFIX ═══"
    # ``aws s3 ls`` lists prefixes (PRE) lexicographically; for our
    # ISO-like run_ids (20260506T...) lexicographic == chronological.
    RUN_ID="$(aws_call s3 ls "$RUNS_PREFIX" 2>/dev/null \
        | awk '/^ *PRE / {print $2}' | sed 's:/$::' | sort | tail -1)"
    if [[ -z "$RUN_ID" ]]; then
        echo "  ✗ no runs under $RUNS_PREFIX" >&2
        exit 1
    fi
    echo "  ✓ run_id=$RUN_ID"
else
    echo "═══ using provided run_id=$RUN_ID ═══"
fi

S3_PREFIX="${RUNS_PREFIX}${RUN_ID}"

# ---------------------------------------------------------------------
# 2. Discover head node (both tag schemes)
# ---------------------------------------------------------------------
echo
echo "═══ locating head node ═══"
HEAD_ID="$(aws_call ec2 describe-instances \
    --filters "Name=tag:process-bigraph-cluster,Values=sf-${STACK_PREFIX}" \
              "Name=tag:process-bigraph-role,Values=head" \
              "Name=instance-state-name,Values=running" \
    --query 'Reservations[0].Instances[0].InstanceId' --output text 2>/dev/null)"
if [[ -z "$HEAD_ID" || "$HEAD_ID" == "None" ]]; then
    echo "  not under new tags — trying legacy spatio-flux-* tags"
    HEAD_ID="$(aws_call ec2 describe-instances \
        --filters "Name=tag:spatio-flux-cluster,Values=sf-${STACK_PREFIX}" \
                  "Name=tag:spatio-flux-role,Values=head" \
                  "Name=instance-state-name,Values=running" \
        --query 'Reservations[0].Instances[0].InstanceId' --output text 2>/dev/null)"
fi
if [[ -z "$HEAD_ID" || "$HEAD_ID" == "None" ]]; then
    echo "  ✗ no running head found for stack=$STACK_PREFIX" >&2
    exit 1
fi
echo "  ✓ head=$HEAD_ID"

# ---------------------------------------------------------------------
# 3. Trigger s3 sync from inside the head container
# ---------------------------------------------------------------------
echo
echo "═══ syncing /app/spatio-flux/out/comets_compare/ → ${S3_PREFIX}/results/ ═══"
INNER="aws --region $AWS_REGION s3 sync /app/spatio-flux/out/comets_compare/ ${S3_PREFIX}/results/"
PARAMS="$(mktemp)"
trap 'rm -f "$PARAMS"' EXIT
python3 -c "import json,sys; print(json.dumps({'commands':['docker exec spatio_flux_ray bash -c '+repr(sys.argv[1])]}))" \
    "$INNER" > "$PARAMS"

CMD_ID="$(aws_call ssm send-command \
    --instance-ids "$HEAD_ID" \
    --document-name AWS-RunShellScript \
    --parameters "file://$PARAMS" \
    --query 'Command.CommandId' --output text 2>/dev/null)"
if [[ -z "$CMD_ID" || "$CMD_ID" == "None" ]]; then
    echo "  ✗ failed to send SSM command" >&2
    exit 1
fi
echo "  ssm cmd_id=$CMD_ID — waiting for completion ..."
for _ in $(seq 1 60); do
    STATUS="$(aws_call ssm get-command-invocation \
        --command-id "$CMD_ID" --instance-id "$HEAD_ID" \
        --query 'Status' --output text 2>/dev/null || echo Pending)"
    [[ "$STATUS" == "Success" || "$STATUS" == "Failed" \
        || "$STATUS" == "TimedOut" || "$STATUS" == "Cancelled" ]] && break
    sleep 2
done
echo "  status=$STATUS"
aws_call ssm get-command-invocation \
    --command-id "$CMD_ID" --instance-id "$HEAD_ID" \
    --query 'StandardOutputContent' --output text 2>/dev/null \
    | sed 's/^/    /' | tail -20
if [[ "$STATUS" != "Success" ]]; then
    echo "  ✗ SSM sync failed; stderr:"
    aws_call ssm get-command-invocation \
        --command-id "$CMD_ID" --instance-id "$HEAD_ID" \
        --query 'StandardErrorContent' --output text 2>/dev/null \
        | sed 's/^/    /' | tail -20
    exit 1
fi

# ---------------------------------------------------------------------
# 4. Sync from S3 to the laptop
# ---------------------------------------------------------------------
LOCAL_OUT="$(cd "$(dirname "$0")/.." && pwd)/out/comets_compare"
mkdir -p "$LOCAL_OUT"
echo
echo "═══ syncing ${S3_PREFIX}/results/ → $LOCAL_OUT/ ═══"
aws_call s3 sync "${S3_PREFIX}/results/" "$LOCAL_OUT/" 2>&1 | sed 's/^/  /' | tail -30

echo
echo "═══ done ═══"
echo "Report likely at: $LOCAL_OUT/report.html"
ls -la "$LOCAL_OUT/" 2>/dev/null | sed 's/^/  /' | head -20
