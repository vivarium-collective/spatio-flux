#!/usr/bin/env bash
#
# diagnose-current-run.sh — paste-no-substitutions diagnostic for the
# most recent comparison run. Auto-discovers run-id, SSM command, and
# instance IDs. Run this when run-comparison-on-ec2.sh seems hung.
#
# Usage:
#   AWS_PROFILE=<sso-profile> ./scripts/diagnose-current-run.sh [-s <stack>] [-b <bucket>]

# NB: deliberately NOT using `set -e` — every aws call below uses `|| true`
# or similar, so we always print every section even if one query fails.
set -uo pipefail

AWS_PROFILE="${AWS_PROFILE:-}"
AWS_REGION="${AWS_DEFAULT_REGION:-${AWS_REGION:-us-gov-west-1}}"
STACK_PREFIX="${STACK_PREFIX:-smsvpctest}"
S3_BUCKET="${S3_BUCKET:-smsvpctest-shared-sharedbucket60d199d6-abfvwv0day91}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -s|--stack)  STACK_PREFIX="$2"; shift 2 ;;
        -b|--bucket) S3_BUCKET="$2"; shift 2 ;;
        -h|--help)   sed -n '2,/^$/p' "$0" | sed 's/^# \?//'; exit 0 ;;
        *) echo "Unknown: $1" >&2; exit 1 ;;
    esac
done

[[ -z "$AWS_PROFILE" ]] && { echo "AWS_PROFILE required" >&2; exit 1; }
aws_call() { aws --profile "$AWS_PROFILE" --region "$AWS_REGION" "$@"; }

# ----- 1. find the submit node ----------------------------------------
SUBMIT_INSTANCE_ID="$(
    aws_call cloudformation describe-stacks --stack-name "${STACK_PREFIX}-batch" \
        --query "Stacks[0].Outputs[?OutputKey=='SubmitNodeInstanceId'].OutputValue" \
        --output text 2>/dev/null
)"
echo "═══ submit node: $SUBMIT_INSTANCE_ID ═══"

# ----- 2. find the most recent run-id from S3 -------------------------
RUN_ID="$(
    aws_call s3 ls "s3://${S3_BUCKET}/spatio-flux/runs/" \
        | awk '/^.*PRE / {print $2}' \
        | sed 's|/$||' \
        | sort | tail -1
)"
echo "═══ latest run id: $RUN_ID ═══"
S3_PREFIX="s3://${S3_BUCKET}/spatio-flux/runs/${RUN_ID}"

# ----- 3. last SSM command targeting the submit node -----------------
echo
echo "═══ recent SSM invocations on submit node (last 5) ═══"
aws_call ssm list-command-invocations \
    --instance-id "$SUBMIT_INSTANCE_ID" \
    --max-items 5 \
    --query 'CommandInvocations[*].[RequestedDateTime,Status,CommandId,Comment]' \
    --output text 2>/dev/null

LATEST_CMD="$(
    aws_call ssm list-command-invocations \
        --instance-id "$SUBMIT_INSTANCE_ID" \
        --max-items 1 \
        --query 'CommandInvocations[0].CommandId' \
        --output text 2>/dev/null
)"
if [[ -n "$LATEST_CMD" && "$LATEST_CMD" != "None" ]]; then
    echo
    echo "═══ latest command details: $LATEST_CMD ═══"
    STATUS="$(aws_call ssm get-command-invocation \
        --command-id "$LATEST_CMD" --instance-id "$SUBMIT_INSTANCE_ID" \
        --query 'Status' --output text 2>/dev/null || echo unknown)"
    DETAILS="$(aws_call ssm get-command-invocation \
        --command-id "$LATEST_CMD" --instance-id "$SUBMIT_INSTANCE_ID" \
        --query 'StatusDetails' --output text 2>/dev/null || echo unknown)"
    EXEC_START="$(aws_call ssm get-command-invocation \
        --command-id "$LATEST_CMD" --instance-id "$SUBMIT_INSTANCE_ID" \
        --query 'ExecutionStartDateTime' --output text 2>/dev/null || echo unknown)"
    echo "  status:        $STATUS"
    echo "  details:       $DETAILS"
    echo "  exec started:  $EXEC_START"
    echo "  --- stderr (last 30 lines) ---"
    aws_call ssm get-command-invocation \
        --command-id "$LATEST_CMD" --instance-id "$SUBMIT_INSTANCE_ID" \
        --query 'StandardErrorContent' --output text 2>/dev/null | tail -30 \
        || echo "  (no stderr yet)"
fi

# ----- 4. bootstrap log on S3 -----------------------------------------
echo
echo "═══ ec2-bootstrap.log (S3 last-modified + tail) ═══"
aws_call s3 ls "${S3_PREFIX}/ec2-bootstrap.log" 2>/dev/null \
    || echo "(no log on S3 yet)"
echo
aws_call s3 cp "${S3_PREFIX}/ec2-bootstrap.log" - 2>/dev/null | tail -50 \
    || echo "(could not fetch log)"

# ----- 5. cluster instance state -------------------------------------
echo
echo "═══ cluster instances (tag spatio-flux-cluster=sf-${STACK_PREFIX}) ═══"
aws_call ec2 describe-instances \
    --filters "Name=tag:spatio-flux-cluster,Values=sf-${STACK_PREFIX}" \
              "Name=instance-state-name,Values=pending,running,stopping,stopped" \
    --query 'Reservations[*].Instances[*].[InstanceId,State.Name,PrivateIpAddress,Tags[?Key==`spatio-flux-role`].Value|[0]]' \
    --output text 2>/dev/null \
    || echo "(no cluster instances)"

# ----- 6. experiment.log if present (S3, may be stale) --------------
echo
echo "═══ experiment.log on S3 (uploaded at experiment END only) ═══"
aws_call s3 ls "${S3_PREFIX}/experiment.log" 2>/dev/null \
    && aws_call s3 cp "${S3_PREFIX}/experiment.log" - 2>/dev/null | tail -30 \
    || echo "(no experiment.log on S3 yet — experiment hasn't finished)"

# ----- 7. LIVE experiment log on the head via SSM -------------------
# This is the canonical way to tell stalled vs running. If the file
# size or mtime keeps changing across consecutive runs of this script,
# the experiment is making progress. If it's frozen for 5+ min, stalled.
echo
echo "═══ LIVE experiment.log on head via SSM (definitive) ═══"
HEAD_ID="$(
    aws_call ec2 describe-instances \
        --filters "Name=tag:spatio-flux-cluster,Values=sf-${STACK_PREFIX}" \
                  "Name=tag:spatio-flux-role,Values=head" \
                  "Name=instance-state-name,Values=running" \
        --query 'Reservations[0].Instances[0].InstanceId' --output text 2>/dev/null
)"
if [[ -z "$HEAD_ID" || "$HEAD_ID" == "None" ]]; then
    echo "(no running head instance — cluster torn down)"
else
    # /tmp/experiment.log lives INSIDE the spatio_flux_ray container (the
    # experiment writes to its own /tmp, not the host's). Probe via
    # `docker exec`. Use a JSON params file to avoid the inline-shorthand
    # quoting escapes (\"-within-single-quote-within-AWS-CLI-shorthand-array).
    # Two separate SSM commands so each gets its own 24KB stdout buffer.
    # First probe: file state + experiment.log tail.
    # Second probe: deep stuck-state (ps, py-spy, ray list).
    run_probe() {
        local title="$1" params_file="$2"
        echo
        echo "─── $title ───"
        local cmd_id
        cmd_id="$(aws_call ssm send-command \
            --instance-ids "$HEAD_ID" \
            --document-name AWS-RunShellScript \
            --parameters "file://$params_file" \
            --query 'Command.CommandId' --output text 2>/dev/null || echo '')"
        rm -f "$params_file"
        [[ -z "$cmd_id" || "$cmd_id" == "None" ]] && { echo "(send failed)"; return; }
        for _ in $(seq 1 30); do
            local status
            status="$(aws_call ssm get-command-invocation \
                --command-id "$cmd_id" --instance-id "$HEAD_ID" \
                --query 'Status' --output text 2>/dev/null || echo Pending)"
            [[ "$status" == "Success" || "$status" == "Failed" ]] && break
            sleep 1
        done
        aws_call ssm get-command-invocation \
            --command-id "$cmd_id" --instance-id "$HEAD_ID" \
            --query 'StandardOutputContent' --output text 2>/dev/null \
            || echo "(probe failed)"
    }

    P0="$(mktemp)"
    cat > "$P0" <<'PROBE0'
{
  "commands": [
    "echo === image digest running on head ===",
    "docker inspect --format '{{index .RepoDigests 0}}' spatio_flux_ray 2>/dev/null || echo no-digest",
    "echo === in-container shard_manager.py freshness ===",
    "docker exec spatio_flux_ray python -c \"from spatio_flux.library.shard_manager import ShardManager; print('has_flush_pending:', hasattr(ShardManager, 'flush_pending')); print('has_enqueue:',       hasattr(ShardManager, 'enqueue')); print('has_collect:',       hasattr(ShardManager, 'collect'))\" 2>&1",
    "echo === in-container _ShardFacade has invoke override ===",
    "docker exec spatio_flux_ray python -c \"from spatio_flux.library.shard_manager import _ShardFacade; from process_bigraph import Process; print('overrides_invoke:', _ShardFacade.invoke is not Process.invoke)\" 2>&1",
    "echo === pip-installed spatio_flux version ===",
    "docker exec spatio_flux_ray bash -c 'cd /app/spatio-flux && (git rev-parse --short HEAD 2>/dev/null || echo no-git); stat -c %y spatio_flux/library/shard_manager.py' 2>&1"
  ]
}
PROBE0
    run_probe "image / code freshness check" "$P0"

    P1="$(mktemp)"
    cat > "$P1" <<'PROBE1'
{
  "commands": [
    "docker exec spatio_flux_ray bash -c 'echo SIZE_BYTES=$(stat -c%s /tmp/experiment.log 2>/dev/null || echo 0); echo MTIME=$(stat -c%y /tmp/experiment.log 2>/dev/null || echo never); echo LINES=$(wc -l < /tmp/experiment.log 2>/dev/null || echo 0)'",
    "echo === last 30 lines of experiment.log ===",
    "docker exec spatio_flux_ray tail -30 /tmp/experiment.log 2>/dev/null || echo NO_FILE"
  ]
}
PROBE1
    run_probe "experiment.log state + tail" "$P1"

    P2="$(mktemp)"
    cat > "$P2" <<'PROBE2'
{
  "commands": [
    "echo === python processes ===",
    "docker exec spatio_flux_ray bash -c 'ps -eo pid,pcpu,pmem,etime,stat,comm,args | grep -E \"python|ray\" | grep -v grep | head -20' 2>&1 || true",
    "echo === py-spy stack of compare_comets process ===",
    "docker exec spatio_flux_ray bash -c 'pip show py-spy >/dev/null 2>&1 || pip install py-spy --quiet 2>&1 | tail -3; PID=$(pgrep -f compare_comets | head -1); if [ -n \"$PID\" ]; then py-spy dump --pid $PID 2>&1 | head -60; else echo NO_COMPARE_COMETS_PROC; fi' 2>&1 || true",
    "echo === ray status (resource usage) ===",
    "docker exec spatio_flux_ray ray status 2>&1 | head -30 || true",
    "echo === ray list actors (alive/dead/restarting) ===",
    "docker exec spatio_flux_ray ray list actors --limit 8 2>&1 | head -30 || true"
  ]
}
PROBE2
    run_probe "deep state: processes + py-spy + ray actors" "$P2"
fi

echo
echo "═══ done ═══"
echo
echo "Tip: run this script again 60s from now. If SIZE_BYTES grows or"
echo "MTIME advances, the experiment is making progress. Frozen for"
echo "5+ min = stalled."
