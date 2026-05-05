#!/usr/bin/env bash
#
# teardown-cluster.sh — emergency stop. Tears down anything left over
# from a spatio-flux Ray cluster: live EC2 instances tagged for it, and
# the cluster's security group. Three-pass:
#
#   1. Try a graceful ``ray down`` via SSM on the submit node — works
#      when ``ray up`` got far enough to leave a cluster.yaml on disk.
#   2. Force-terminate any EC2 instances still tagged with the cluster
#      name (catches orphans from a half-failed ray up).
#   3. Wait for terminations, then delete the cluster security group
#      so subsequent ``ray up`` calls re-create it with current rules.
#
# Use when:
#   - You used ``--keep-cluster`` and want to release the bill.
#   - A run failed in a way that left orphan EC2s + SG behind.
#   - You changed the cluster yaml (e.g. SG rules) and need a clean slate.
#
# Usage:
#   AWS_PROFILE=<sso-profile> AWS_DEFAULT_REGION=us-gov-west-1 \
#       ./scripts/teardown-cluster.sh -s smsvpctest [--yes]

set -euo pipefail

AWS_PROFILE="${AWS_PROFILE:-}"
AWS_REGION="${AWS_DEFAULT_REGION:-${AWS_REGION:-us-gov-west-1}}"
STACK_PREFIX="${STACK_PREFIX:-smsvpctest}"
ASSUME_YES=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        -s|--stack) STACK_PREFIX="$2"; shift 2 ;;
        --yes|-y)   ASSUME_YES=1; shift ;;
        -h|--help)
            sed -n '2,/^$/p' "$0" | sed 's/^# \?//'
            exit 0 ;;
        *) echo "Unknown: $1" >&2; exit 1 ;;
    esac
done

[[ -z "$AWS_PROFILE" ]] && { echo "AWS_PROFILE required" >&2; exit 1; }

CLUSTER_NAME="spatio-flux-ray-${STACK_PREFIX}"
# The cluster yaml's provider.security_group.GroupName is
# ``ray-spatio-flux-${STACK_PREFIX}`` (deploy/ray-govcloud-cluster.yaml).
# We also probe the cluster-name and a few near-misses in case an older
# yaml or different Ray version named it differently.
SG_NAME="ray-spatio-flux-${STACK_PREFIX}"

aws_call() {
    aws --profile "$AWS_PROFILE" --region "$AWS_REGION" "$@"
}

# -------- Step 0: enumerate what we're about to kill ---------------------
echo "═══ what's still running (cluster=$CLUSTER_NAME) ═══"
LIVE_INSTANCES="$(
    aws_call ec2 describe-instances \
        --filters \
            "Name=tag:ray-cluster-name,Values=${CLUSTER_NAME}" \
            "Name=instance-state-name,Values=pending,running,stopping,stopped" \
        --query 'Reservations[*].Instances[*].[InstanceId,State.Name,InstanceType,PrivateIpAddress]' \
        --output text 2>/dev/null
)"

if [[ -n "$LIVE_INSTANCES" ]]; then
    echo "$LIVE_INSTANCES"
else
    echo "  (no live instances)"
fi

# Multiple values in a single Name=... filter is the OR form. Multiple
# Name=... clauses would AND them together (and then the SG never
# matches). Ray's autoscaler may use either ``$name`` or ``ray-$name``
# as the GroupName depending on version, so we list both as values.
ALL_SGS="$(
    aws_call ec2 describe-security-groups \
        --filters "Name=group-name,Values=${SG_NAME},${CLUSTER_NAME},ray-${CLUSTER_NAME}" \
        --query 'SecurityGroups[*].[GroupId,GroupName,VpcId]' --output text 2>/dev/null
)"
if [[ -n "$ALL_SGS" ]]; then
    echo
    echo "security groups:"
    echo "$ALL_SGS"
fi

if [[ -z "$LIVE_INSTANCES" && -z "$ALL_SGS" ]]; then
    echo
    echo "Nothing to tear down. Done."
    exit 0
fi

# -------- Confirm --------------------------------------------------------
if [[ "$ASSUME_YES" -ne 1 ]]; then
    echo
    read -p "Terminate everything above? [y/N] " yn
    case "$yn" in
        [yY]*) ;;
        *) echo "Aborted."; exit 1 ;;
    esac
fi

# -------- Step 1: graceful ray down via SSM (best-effort) ---------------
echo
echo "═══ Step 1: graceful ray down via SSM ═══"
SUBMIT_INSTANCE_ID="$(
    aws_call cloudformation describe-stacks --stack-name "${STACK_PREFIX}-batch" \
        --query "Stacks[0].Outputs[?OutputKey=='SubmitNodeInstanceId'].OutputValue" \
        --output text 2>/dev/null
)"
if [[ -n "$SUBMIT_INSTANCE_ID" && "$SUBMIT_INSTANCE_ID" != "None" ]]; then
    echo "→ submit node: $SUBMIT_INSTANCE_ID"
    CMD='ls /tmp/spatio-flux-run/cluster.yaml >/dev/null 2>&1 && ray down --yes /tmp/spatio-flux-run/cluster.yaml || echo "(no cluster yaml on submit node — skipping ray down)"'
    COMMAND_ID="$(
        aws_call ssm send-command \
            --instance-ids "$SUBMIT_INSTANCE_ID" \
            --document-name AWS-RunShellScript \
            --parameters "commands=[\"$CMD\"]" \
            --query 'Command.CommandId' --output text 2>/dev/null || echo ''
    )"
    if [[ -n "$COMMAND_ID" ]]; then
        for _ in $(seq 1 40); do  # up to ~10 min
            STATUS="$(aws_call ssm get-command-invocation \
                --command-id "$COMMAND_ID" --instance-id "$SUBMIT_INSTANCE_ID" \
                --query 'Status' --output text 2>/dev/null || echo Pending)"
            case "$STATUS" in
                Success)             echo "  ✓ ray down complete"; break ;;
                Failed|Cancelled|TimedOut)
                    echo "  ⚠  ssm ray down failed ($STATUS) — continuing to step 2" ; break ;;
                *) printf '.'; sleep 15 ;;
            esac
        done
    else
        echo "  ⚠  could not send ssm command (submit node unreachable?) — continuing"
    fi
else
    echo "  ⚠  no submit node found — skipping ssm path"
fi

# -------- Step 2: force-terminate any orphans ---------------------------
echo
echo "═══ Step 2: terminate orphan instances ═══"
LIVE_INSTANCE_IDS="$(echo "$LIVE_INSTANCES" | awk '{print $1}' | tr '\n' ' ')"
if [[ -n "$LIVE_INSTANCE_IDS" ]]; then
    # Re-check — ray down may have already terminated them.
    STILL_ALIVE="$(
        aws_call ec2 describe-instances \
            --instance-ids $LIVE_INSTANCE_IDS \
            --query 'Reservations[*].Instances[?State.Name!=`terminated` && State.Name!=`shutting-down`].InstanceId' \
            --output text 2>/dev/null | tr '\n' ' '
    )"
    if [[ -n "$STILL_ALIVE" ]]; then
        echo "→ force-terminating: $STILL_ALIVE"
        aws_call ec2 terminate-instances --instance-ids $STILL_ALIVE >/dev/null
        echo "→ waiting for terminated state ..."
        aws_call ec2 wait instance-terminated --instance-ids $STILL_ALIVE
        echo "  ✓ all instances terminated"
    else
        echo "  ✓ already terminated by ray down"
    fi
else
    echo "  (no instances to terminate)"
fi

# -------- Step 3: delete the cluster SG ---------------------------------
echo
echo "═══ Step 3: delete cluster security group ═══"
delete_sg() {
    local sg_id="$1" sg_name="$2"
    if aws_call ec2 delete-security-group --group-id "$sg_id" 2>/dev/null; then
        echo "  ✓ deleted $sg_name ($sg_id)"
    else
        # Failure is usually "still has dependencies" — print why.
        ERR="$(aws_call ec2 delete-security-group --group-id "$sg_id" 2>&1 || true)"
        echo "  ⚠  could not delete $sg_name ($sg_id):"
        echo "     $(echo "$ERR" | tail -2)"
    fi
}
if [[ -n "$ALL_SGS" ]]; then
    while read -r sg_id sg_name; do
        [[ -z "$sg_id" ]] && continue
        delete_sg "$sg_id" "$sg_name"
    done <<< "$ALL_SGS"
else
    echo "  (no matching SG)"
fi

echo
echo "═══ Done ═══"
echo "Verify: aws ec2 describe-instances \\"
echo "    --filters 'Name=tag:ray-cluster-name,Values=$CLUSTER_NAME' \\"
echo "              'Name=instance-state-name,Values=pending,running,stopping,stopped' \\"
echo "    --query 'Reservations[*].Instances[*].[InstanceId,State.Name]'"
