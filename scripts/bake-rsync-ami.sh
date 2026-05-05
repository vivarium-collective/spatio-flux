#!/usr/bin/env bash
#
# bake-rsync-ami.sh — bake a tiny custom AMI on top of ECS-optimized AL2
# with rsync pre-installed. Solves the timing race where Ray's
# file_mounts step runs before cloud-init has a chance to yum-install
# rsync via UserData.
#
# Drives the bake from your laptop via SSM on the submit node:
#   1. submit-node launches an ECS-AL2 instance in the same subnet
#   2. SSM into it: yum install rsync
#   3. create-image, wait for available, terminate builder
#   4. write the new AMI ID to deploy/.spatio-flux-base-ami
#
# After this completes, the cluster yaml can pick up the AMI ID from
# the file (orchestrator already supports this) and bring-up no longer
# depends on UserData timing.
#
# Required env or flags:
#   AWS_PROFILE              SSO profile
#   AWS_DEFAULT_REGION       us-gov-west-1
#   STACK_PREFIX             CFN stack prefix (-s, default smsvpctest)

set -euo pipefail

AWS_PROFILE="${AWS_PROFILE:-}"
AWS_REGION="${AWS_DEFAULT_REGION:-${AWS_REGION:-us-gov-west-1}}"
STACK_PREFIX="${STACK_PREFIX:-smsvpctest}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -s|--stack) STACK_PREFIX="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,/^$/p' "$0" | sed 's/^# \?//'; exit 0 ;;
        *) echo "Unknown: $1" >&2; exit 1 ;;
    esac
done

[[ -z "$AWS_PROFILE" ]] && { echo "AWS_PROFILE required" >&2; exit 1; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

aws_call() { aws --profile "$AWS_PROFILE" --region "$AWS_REGION" "$@"; }

# Discover submit-node + its subnet/SG. We bake in the same subnet so
# the builder has the same networking as future cluster nodes will.
SUBMIT_INSTANCE_ID="$(
    aws_call cloudformation describe-stacks --stack-name "${STACK_PREFIX}-batch" \
        --query "Stacks[0].Outputs[?OutputKey=='SubmitNodeInstanceId'].OutputValue" \
        --output text 2>/dev/null
)"
[[ -z "$SUBMIT_INSTANCE_ID" || "$SUBMIT_INSTANCE_ID" == "None" ]] && {
    echo "ERROR: could not resolve ${STACK_PREFIX}-batch SubmitNodeInstanceId" >&2
    exit 1
}

read SUBNET_ID SG_ID < <(
    aws_call ec2 describe-instances --instance-ids "$SUBMIT_INSTANCE_ID" \
        --query 'Reservations[0].Instances[0].[SubnetId,SecurityGroups[0].GroupId]' \
        --output text
)
echo "→ submit node: $SUBMIT_INSTANCE_ID"
echo "→ subnet:      $SUBNET_ID"
echo "→ sg:          $SG_ID"

# Look up the ECS-optimized AL2 AMI as the base.
BASE_AMI="$(aws_call ssm get-parameter \
    --name '/aws/service/ecs/optimized-ami/amazon-linux-2/recommended/image_id' \
    --query 'Parameter.Value' --output text)"
echo "→ base AMI:    $BASE_AMI (ECS-optimized AL2)"

# Launch the builder.
BUILDER_ID="$(aws_call ec2 run-instances \
    --image-id "$BASE_AMI" \
    --instance-type t3.small \
    --subnet-id "$SUBNET_ID" \
    --security-group-ids "$SG_ID" \
    --iam-instance-profile Name=ray-spatio-flux-node \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=spatio-flux-rsync-ami-builder}]' \
    --query 'Instances[0].InstanceId' --output text)"
echo "→ builder:     $BUILDER_ID"
trap '
    echo "→ terminating $BUILDER_ID"
    aws_call ec2 terminate-instances --instance-ids "$BUILDER_ID" >/dev/null 2>&1 || true
    [[ -n "${WORK_DIR:-}" && -d "$WORK_DIR" ]] && rm -rf "$WORK_DIR"
' EXIT

aws_call ec2 wait instance-running --instance-ids "$BUILDER_ID"
echo "→ waiting for SSM agent ..."
sleep 30

# Bake two things into the AMI:
#   1. rsync — needed by Ray's file_mounts (step 2/7 of `ray up`)
#   2. /home/ec2-user/.docker/config.json with credsStore=ecr-login —
#      needed by Ray's docker pull (step 5/7); UserData wrote this
#      previously but lost the cloud-init race vs. Ray's first SSH.
#
# Both are pre-existing on the ECS-optimized AL2 base image except for
# rsync (yum install) and the docker config json (write_files). Doing
# them at bake-time eliminates all UserData timing dependencies — by
# the time Ray's autoscaler SSHes in, both are already on disk.
WORK_DIR="$(mktemp -d)"
SETUP_SCRIPT="$WORK_DIR/bake-setup.sh"
cat >"$SETUP_SCRIPT" <<'BAKE_EOF'
set -euo pipefail
# rsync — host-side, needed by Ray autoscaler step 2/7 (file_mounts).
# amazon-ecr-credential-helper — provides /usr/bin/docker-credential-ecr-login,
# the binary docker resolves when config.json says "credsStore": "ecr-login".
# Despite the name, it's NOT pre-installed on ECS-optimized AL2 (only the
# Docker daemon is). Lives in amzn2-core under this exact package name.
yum install -y rsync amazon-ecr-credential-helper

# Defensive: ensure /home/ec2-user exists and is owned by ec2-user
# BEFORE creating the .docker subdir. mkdir -p running as root would
# otherwise create the parent root-owned, breaking sshd StrictModes
# and rsync-over-ssh on subsequent ec2-user logins.
[[ -d /home/ec2-user ]] || mkdir -p /home/ec2-user
chown ec2-user:ec2-user /home/ec2-user
chmod 0755 /home/ec2-user

mkdir -p /home/ec2-user/.docker
cat > /home/ec2-user/.docker/config.json <<'JSON_EOF'
{
  "credsStore": "ecr-login"
}
JSON_EOF
chown -R ec2-user:ec2-user /home/ec2-user/.docker
chmod 0600 /home/ec2-user/.docker/config.json

echo "--- verify ---"
which rsync && rsync --version | head -1
which docker-credential-ecr-login
echo "--- ownership (everything below MUST be ec2-user) ---"
stat -c '%U:%G %a %n' /home/ec2-user
stat -c '%U:%G %a %n' /home/ec2-user/.docker
stat -c '%U:%G %a %n' /home/ec2-user/.docker/config.json
echo "--- daemons ---"
systemctl is-active sshd docker amazon-ssm-agent
echo "--- non-interactive shell stdout test (rsync poisoning check) ---"
# Run a no-op as ec2-user the same way ssh's exec-command path does.
# If anything prints to stdout here, it'll break rsync-over-ssh the
# same way: rsync sender sees that text instead of a protocol greeting.
sudo -u ec2-user bash -c 'true' >/tmp/stdout-test 2>&1
echo "  bytes printed: $(wc -c < /tmp/stdout-test)"
[[ -s /tmp/stdout-test ]] && echo "  >>> WARN: shell writes to stdout, will break rsync <<<" && cat /tmp/stdout-test
echo "--- /etc/profile.d/ ---"
ls /etc/profile.d/
echo "--- docker config ---"
cat /home/ec2-user/.docker/config.json
BAKE_EOF

# Wrap the script as a single-element JSON commands array. This dodges
# all the AWS-CLI shorthand quoting hell that arises from embedding a
# heredoc in --parameters.
PARAMS_FILE="$WORK_DIR/ssm-params.json"
python3 -c "
import json, sys
with open(sys.argv[1]) as f:
    print(json.dumps({'commands': [f.read()]}))
" "$SETUP_SCRIPT" > "$PARAMS_FILE"

CMD_ID="$(aws_call ssm send-command \
    --instance-ids "$BUILDER_ID" \
    --document-name AWS-RunShellScript \
    --parameters "file://$PARAMS_FILE" \
    --query 'Command.CommandId' --output text)"

echo "→ baking rsync + docker credsStore (ssm: $CMD_ID) ..."
for _ in $(seq 1 30); do
    STATUS="$(aws_call ssm get-command-invocation \
        --command-id "$CMD_ID" --instance-id "$BUILDER_ID" \
        --query 'Status' --output text 2>/dev/null || echo Pending)"
    case "$STATUS" in
        Success)
            echo "✓ bake setup ok"
            aws_call ssm get-command-invocation \
                --command-id "$CMD_ID" --instance-id "$BUILDER_ID" \
                --query 'StandardOutputContent' --output text | sed 's/^/    /'
            break ;;
        Failed|Cancelled|TimedOut)
            echo "✗ $STATUS"
            echo "--- stderr ---"
            aws_call ssm get-command-invocation \
                --command-id "$CMD_ID" --instance-id "$BUILDER_ID" \
                --query 'StandardErrorContent' --output text
            echo "--- stdout ---"
            aws_call ssm get-command-invocation \
                --command-id "$CMD_ID" --instance-id "$BUILDER_ID" \
                --query 'StandardOutputContent' --output text
            exit 1 ;;
        *) printf '.'; sleep 5 ;;
    esac
done
echo

# Snapshot.
AMI_NAME="spatio-flux-base-$(date -u +%Y%m%dT%H%M%SZ)"
NEW_AMI="$(aws_call ec2 create-image \
    --instance-id "$BUILDER_ID" \
    --name "$AMI_NAME" \
    --description "ECS-optimized AL2 + rsync, pre-baked for spatio-flux Ray cluster" \
    --no-reboot \
    --query 'ImageId' --output text)"
echo "→ creating $NEW_AMI ..."
aws_call ec2 wait image-available --image-ids "$NEW_AMI"
echo "✓ AMI $NEW_AMI available"

# Persist for the orchestrator.
echo "$NEW_AMI" > "$REPO_ROOT/deploy/.spatio-flux-base-ami"
echo
echo "═══ Done ═══"
echo "Base AMI: $NEW_AMI"
echo "Saved to deploy/.spatio-flux-base-ami"
echo "Re-run ./scripts/run-comparison-on-ec2.sh — orchestrator picks this up automatically."
