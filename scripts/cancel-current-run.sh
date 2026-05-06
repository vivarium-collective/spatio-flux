#!/usr/bin/env bash
#
# cancel-current-run.sh — auto-discover and cancel the in-progress
# comparison run. Cancels any InProgress SSM command on the submit
# node AND kills the python compare_comets process inside the head's
# container so the experiment SSM command on the head doesn't keep
# running after the orchestrator dies.
#
# Cluster stays up (--keep-cluster semantics): re-run
# `./scripts/run-comparison-on-ec2.sh ...` to launch a new experiment.
#
# Usage:
#   AWS_PROFILE=<profile> ./scripts/cancel-current-run.sh [-s <stack>]

set -uo pipefail

AWS_PROFILE="${AWS_PROFILE:-}"
AWS_REGION="${AWS_DEFAULT_REGION:-${AWS_REGION:-us-gov-west-1}}"
STACK_PREFIX="${STACK_PREFIX:-smsvpctest}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -s|--stack) STACK_PREFIX="$2"; shift 2 ;;
        -h|--help)  sed -n '2,/^$/p' "$0" | sed 's/^# \?//'; exit 0 ;;
        *) echo "Unknown: $1" >&2; exit 1 ;;
    esac
done

[[ -z "$AWS_PROFILE" ]] && { echo "AWS_PROFILE required" >&2; exit 1; }

aws_call() { aws --profile "$AWS_PROFILE" --region "$AWS_REGION" "$@"; }

# ----- 1. cancel SSM commands on submit node -------------------------
SUBMIT_ID="$(aws_call cloudformation describe-stacks \
    --stack-name "${STACK_PREFIX}-batch" \
    --query "Stacks[0].Outputs[?OutputKey=='SubmitNodeInstanceId'].OutputValue" \
    --output text 2>/dev/null)"

echo "═══ submit node: $SUBMIT_ID ═══"
PENDING="$(aws_call ssm list-command-invocations \
    --instance-id "$SUBMIT_ID" \
    --max-items 10 \
    --query 'CommandInvocations[?Status==`InProgress` || Status==`Pending`].CommandId' \
    --output text 2>/dev/null)"
if [[ -n "$PENDING" && "$PENDING" != "None" ]]; then
    for CMD in $PENDING; do
        aws_call ssm cancel-command --command-id "$CMD" 2>/dev/null \
            && echo "  ✓ cancelled $CMD" \
            || echo "  ✗ cancel failed for $CMD"
    done
else
    echo "  (no in-progress commands on submit node)"
fi

# ----- 2. kill python on head ----------------------------------------
HEAD_ID="$(aws_call ec2 describe-instances \
    --filters "Name=tag:spatio-flux-cluster,Values=sf-${STACK_PREFIX}" \
              "Name=tag:spatio-flux-role,Values=head" \
              "Name=instance-state-name,Values=running" \
    --query 'Reservations[0].Instances[0].InstanceId' --output text 2>/dev/null)"

if [[ -n "$HEAD_ID" && "$HEAD_ID" != "None" ]]; then
    echo
    echo "═══ killing wedged python on head $HEAD_ID ═══"
    PARAMS="$(mktemp)"
    cat > "$PARAMS" <<'EOF'
{
  "commands": [
    "echo === before kill ===",
    "docker exec spatio_flux_ray bash -c 'pgrep -af compare_comets || echo none' 2>&1 | head -10",
    "echo === killing ===",
    "docker exec spatio_flux_ray bash -c 'pkill -9 -f compare_comets; sleep 1; echo done' 2>&1 || true",
    "echo === after kill ===",
    "docker exec spatio_flux_ray bash -c 'pgrep -af compare_comets || echo cleared' 2>&1 | head -5"
  ]
}
EOF
    KILL_CMD="$(aws_call ssm send-command \
        --instance-ids "$HEAD_ID" \
        --document-name AWS-RunShellScript \
        --parameters "file://$PARAMS" \
        --query 'Command.CommandId' --output text 2>/dev/null)"
    rm -f "$PARAMS"
    if [[ -n "$KILL_CMD" && "$KILL_CMD" != "None" ]]; then
        for _ in $(seq 1 15); do
            STATUS="$(aws_call ssm get-command-invocation \
                --command-id "$KILL_CMD" --instance-id "$HEAD_ID" \
                --query 'Status' --output text 2>/dev/null || echo Pending)"
            [[ "$STATUS" == "Success" || "$STATUS" == "Failed" ]] && break
            sleep 1
        done
        aws_call ssm get-command-invocation \
            --command-id "$KILL_CMD" --instance-id "$HEAD_ID" \
            --query 'StandardOutputContent' --output text 2>/dev/null \
            | sed 's/^/  /'
    fi
else
    echo
    echo "(no running head — cluster torn down or not yet launched)"
fi

echo
echo "═══ done ═══"
echo "Cluster (if any) is preserved. Re-run with smaller scale:"
echo "  ./scripts/run-comparison-on-ec2.sh -s ${STACK_PREFIX} -b <bucket> --mode small --keep-cluster"
