#!/usr/bin/env bash
#
# nuke-all.sh — comprehensive audit + cancel of any in-flight spatio-flux
# / process-bigraph cluster work. Handles BOTH the legacy `spatio-flux-*`
# tag scheme AND the post-lift `process-bigraph-*` scheme.
#
# Default mode (no flags): READ-ONLY. Lists every running cluster instance,
# every in-progress SSM command, and every python compare_comets running
# inside any head container.
#
# --cancel-ssm           cancel ALL in-progress SSM commands account-wide
# --kill-python          pkill -9 compare_comets inside every head container
# --terminate-all        TERMINATE every running cluster instance (BOTH schemes)
# --terminate-legacy     TERMINATE only spatio-flux-* tagged instances (the OLD
#                        tag scheme, pre-Step-5-lift)
# --terminate-new        TERMINATE only process-bigraph-* tagged instances
# --terminate-cluster ID TERMINATE by cluster_id (matches BOTH schemes; use
#                        with --terminate-legacy or --terminate-new for
#                        scheme-specific surgery)
# --all                  equivalent to --cancel-ssm --kill-python
#                        --terminate-all
#
# Usage:
#   AWS_PROFILE=<profile> ./scripts/nuke-all.sh                          # audit
#   AWS_PROFILE=<profile> ./scripts/nuke-all.sh --cancel-ssm
#   AWS_PROFILE=<profile> ./scripts/nuke-all.sh --terminate-legacy       # kill yesterday's
#   AWS_PROFILE=<profile> ./scripts/nuke-all.sh --terminate-cluster sf-x # kill cluster_id=sf-x

set -uo pipefail

AWS_PROFILE="${AWS_PROFILE:-}"
AWS_REGION="${AWS_DEFAULT_REGION:-${AWS_REGION:-us-gov-west-1}}"

CANCEL_SSM=0
KILL_PYTHON=0
TERMINATE_ALL=0
TERMINATE_LEGACY=0
TERMINATE_NEW=0
TERMINATE_CLUSTER=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cancel-ssm)          CANCEL_SSM=1; shift ;;
        --kill-python)         KILL_PYTHON=1; shift ;;
        --terminate-all)       TERMINATE_ALL=1; shift ;;
        --terminate-legacy)    TERMINATE_LEGACY=1; shift ;;
        --terminate-new)       TERMINATE_NEW=1; shift ;;
        --terminate-cluster)   TERMINATE_CLUSTER="$2"; shift 2 ;;
        --all)                 CANCEL_SSM=1; KILL_PYTHON=1; TERMINATE_ALL=1; shift ;;
        -h|--help)             sed -n '2,/^$/p' "$0" | sed 's/^# \?//'; exit 0 ;;
        *) echo "Unknown: $1" >&2; exit 1 ;;
    esac
done

[[ -z "$AWS_PROFILE" ]] && { echo "AWS_PROFILE required" >&2; exit 1; }

aws_call() { aws --profile "$AWS_PROFILE" --region "$AWS_REGION" "$@"; }

# =====================================================================
# 1. Running cluster instances (BOTH tag schemes)
# =====================================================================
echo "═══ running cluster instances (any cluster-tag scheme) ═══"
# Get the raw Tags list (not pre-extracted via JMESPath) — JMESPath's
# `||` parsing across both tag-key schemes is brittle; pull the full
# Tags array and resolve role/cluster/name in Python instead.
INSTANCES_JSON="$(aws_call ec2 describe-instances \
    --filters "Name=tag-key,Values=process-bigraph-cluster,spatio-flux-cluster" \
              "Name=instance-state-name,Values=pending,running,stopping" \
    --query 'Reservations[].Instances[].{Id:InstanceId,Type:InstanceType,Launch:LaunchTime,State:State.Name,Tags:Tags}' \
    --output json 2>/dev/null)"

PARSED="$(echo "$INSTANCES_JSON" | python3 - <<'PY' 2>/dev/null
import json, sys
data = json.load(sys.stdin)
out = []
for inst in data:
    tags = {t['Key']: t['Value'] for t in (inst.get('Tags') or [])}
    # Detect tag scheme: legacy (spatio-flux-*) or new (process-bigraph-*).
    if 'process-bigraph-cluster' in tags:
        scheme = 'new'
        role = tags.get('process-bigraph-role') or '?'
        cluster = tags['process-bigraph-cluster']
    elif 'spatio-flux-cluster' in tags:
        scheme = 'legacy'
        role = tags.get('spatio-flux-role') or '?'
        cluster = tags['spatio-flux-cluster']
    else:
        scheme = '?'
        role = '?'
        cluster = '?'
    name = tags.get('Name') or '?'
    out.append({
        'Id': inst['Id'], 'Type': inst['Type'], 'Launch': inst['Launch'],
        'State': inst['State'], 'Role': role, 'Cluster': cluster,
        'Scheme': scheme, 'Name': name,
    })
print(json.dumps(out))
PY
)"

INSTANCE_IDS=$(echo "$PARSED" | python3 -c "import json,sys; print(' '.join(i['Id'] for i in json.load(sys.stdin)))" 2>/dev/null)
HEAD_IDS=$(echo "$PARSED" | python3 -c "import json,sys; print(' '.join(i['Id'] for i in json.load(sys.stdin) if i['Role'] == 'head'))" 2>/dev/null)

if [[ -z "$INSTANCE_IDS" ]]; then
    echo "  (none)"
else
    echo "$PARSED" | python3 -c "
import json, sys
for i in json.load(sys.stdin):
    print(f\"  {i['Id']:<22} {i['Type']:<13} {i['Scheme']:<6} {i['Role']:<8} {i['State']:<8} cluster={i['Cluster']:<18} launched {i['Launch']}\")"
    echo
    echo "  total: $(echo $INSTANCE_IDS | wc -w) instance(s)"
    echo "  heads: $(echo $HEAD_IDS | wc -w) ($HEAD_IDS)"
fi

# =====================================================================
# 2. In-progress SSM commands (account-wide)
# =====================================================================
echo
echo "═══ in-progress SSM commands (account-wide) ═══"
PENDING_CMDS_JSON="$(aws_call ssm list-commands --max-items 50 \
    --query 'Commands[?Status==`InProgress` || Status==`Pending`].{Id:CommandId,Status:Status,At:RequestedDateTime,Comment:Comment,Instances:InstanceIds}' \
    --output json 2>/dev/null)"
PENDING_CMDS="$(echo "$PENDING_CMDS_JSON" | python3 -c "import json,sys; print(' '.join(c['Id'] for c in json.load(sys.stdin)))" 2>/dev/null)"
if [[ -z "$PENDING_CMDS" ]]; then
    echo "  (none)"
else
    echo "$PENDING_CMDS_JSON" | python3 -c "
import json, sys
for c in json.load(sys.stdin):
    print(f\"  {c['Id']:<37} {c['Status']:<11} on {','.join(c['Instances'])} -- {c.get('Comment') or ''} ({c['At']})\")"
fi

# =====================================================================
# 3. python compare_comets in head containers
# =====================================================================
PYTHON_PROBE_DONE=0
if [[ -n "$HEAD_IDS" ]]; then
    echo
    echo "═══ python compare_comets in head containers ═══"
    for HEAD in $HEAD_IDS; do
        echo "  → $HEAD"
        CMD="$(aws_call ssm send-command \
            --instance-ids "$HEAD" \
            --document-name AWS-RunShellScript \
            --parameters 'commands=["docker exec spatio_flux_ray bash -c '\''pgrep -af compare_comets || echo none'\'' 2>&1 | head -10"]' \
            --query 'Command.CommandId' --output text 2>/dev/null)"
        if [[ -n "$CMD" && "$CMD" != "None" ]]; then
            for _ in $(seq 1 10); do
                STATUS="$(aws_call ssm get-command-invocation \
                    --command-id "$CMD" --instance-id "$HEAD" \
                    --query 'Status' --output text 2>/dev/null || echo Pending)"
                [[ "$STATUS" == "Success" || "$STATUS" == "Failed" ]] && break
                sleep 1
            done
            aws_call ssm get-command-invocation \
                --command-id "$CMD" --instance-id "$HEAD" \
                --query 'StandardOutputContent' --output text 2>/dev/null \
                | sed 's/^/    /'
        fi
    done
    PYTHON_PROBE_DONE=1
fi

# =====================================================================
# Action: --cancel-ssm
# =====================================================================
if [[ "$CANCEL_SSM" == 1 && -n "$PENDING_CMDS" ]]; then
    echo
    echo "═══ cancelling $(echo $PENDING_CMDS | wc -w) SSM command(s) ═══"
    for CMD in $PENDING_CMDS; do
        aws_call ssm cancel-command --command-id "$CMD" 2>/dev/null \
            && echo "  ✓ cancelled $CMD" \
            || echo "  ✗ cancel failed for $CMD"
    done
fi

# =====================================================================
# Action: --kill-python
# =====================================================================
if [[ "$KILL_PYTHON" == 1 && -n "$HEAD_IDS" ]]; then
    echo
    echo "═══ pkill -9 compare_comets on every head ═══"
    for HEAD in $HEAD_IDS; do
        echo "  → $HEAD"
        CMD="$(aws_call ssm send-command \
            --instance-ids "$HEAD" \
            --document-name AWS-RunShellScript \
            --parameters 'commands=["docker exec spatio_flux_ray bash -c '\''pkill -9 -f compare_comets; sleep 1; pgrep -af compare_comets || echo cleared'\'' 2>&1 | head -5"]' \
            --query 'Command.CommandId' --output text 2>/dev/null)"
        if [[ -n "$CMD" && "$CMD" != "None" ]]; then
            for _ in $(seq 1 10); do
                STATUS="$(aws_call ssm get-command-invocation \
                    --command-id "$CMD" --instance-id "$HEAD" \
                    --query 'Status' --output text 2>/dev/null || echo Pending)"
                [[ "$STATUS" == "Success" || "$STATUS" == "Failed" ]] && break
                sleep 1
            done
            aws_call ssm get-command-invocation \
                --command-id "$CMD" --instance-id "$HEAD" \
                --query 'StandardOutputContent' --output text 2>/dev/null \
                | sed 's/^/    /'
        fi
    done
fi

# =====================================================================
# Action: termination flags. Resolve which instances to terminate by
# combining the requested filters (scheme and/or cluster_id) against
# the parsed instance list. All flags can be combined; a flag that
# selects nothing is silently a no-op.
# =====================================================================
TARGET_IDS=""
TARGET_DESCRIPTION=""
if [[ "$TERMINATE_ALL" == 1 ]]; then
    TARGET_IDS="$INSTANCE_IDS"
    TARGET_DESCRIPTION="every running cluster instance"
elif [[ "$TERMINATE_LEGACY" == 1 || "$TERMINATE_NEW" == 1 || -n "$TERMINATE_CLUSTER" ]]; then
    TARGET_IDS=$(echo "$PARSED" | python3 - <<PY 2>/dev/null
import json, sys
data = json.load(sys.stdin)
want_legacy = $TERMINATE_LEGACY
want_new = $TERMINATE_NEW
target_cluster = "$TERMINATE_CLUSTER"
ids = []
for i in data:
    if target_cluster and i['Cluster'] != target_cluster:
        continue
    if want_legacy and i['Scheme'] != 'legacy':
        continue
    if want_new and i['Scheme'] != 'new':
        continue
    if not (want_legacy or want_new or target_cluster):
        continue
    ids.append(i['Id'])
print(' '.join(ids))
PY
)
    parts=()
    [[ "$TERMINATE_LEGACY" == 1 ]] && parts+=("scheme=legacy")
    [[ "$TERMINATE_NEW" == 1 ]] && parts+=("scheme=new")
    [[ -n "$TERMINATE_CLUSTER" ]] && parts+=("cluster=$TERMINATE_CLUSTER")
    TARGET_DESCRIPTION="$(IFS=,; echo "${parts[*]}")"
fi

if [[ -n "$TARGET_IDS" ]]; then
    echo
    echo "═══ TERMINATING $(echo $TARGET_IDS | wc -w) instance(s) [$TARGET_DESCRIPTION] ═══"
    aws_call ec2 terminate-instances --instance-ids $TARGET_IDS \
        --query 'TerminatingInstances[].[InstanceId,CurrentState.Name]' \
        --output text | sed 's/^/  /'
elif [[ "$TERMINATE_LEGACY" == 1 || "$TERMINATE_NEW" == 1 || -n "$TERMINATE_CLUSTER" ]]; then
    echo
    echo "═══ termination requested but no instances matched [$TARGET_DESCRIPTION] ═══"
fi

echo
echo "═══ done ═══"
if [[ "$CANCEL_SSM" == 0 && "$KILL_PYTHON" == 0 && "$TERMINATE_ALL" == 0 \
        && "$TERMINATE_LEGACY" == 0 && "$TERMINATE_NEW" == 0 \
        && -z "$TERMINATE_CLUSTER" ]]; then
    echo "Read-only audit. To act:"
    echo "  --cancel-ssm           cancel pending SSM commands"
    echo "  --kill-python          pkill compare_comets in head containers (cluster stays up)"
    echo "  --terminate-legacy     terminate spatio-flux-* tagged instances only"
    echo "  --terminate-new        terminate process-bigraph-* tagged instances only"
    echo "  --terminate-cluster ID terminate by cluster_id"
    echo "  --terminate-all        terminate everything"
    echo "  --all                  cancel-ssm + kill-python + terminate-all"
fi
