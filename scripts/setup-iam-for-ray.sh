#!/usr/bin/env bash
#
# setup-iam-for-ray.sh — provision the two IAM pieces ``ray up`` needs:
#
#   1. Inline policy ``RaySpatioFluxClusterMgmt`` on the submit-node role
#      (granted the EC2/IAM/SSM permissions Ray's autoscaler uses).
#   2. Worker instance profile ``ray-spatio-flux-node`` with EC2 trust +
#      S3 read on the working bucket + SSM-agent baseline.
#
# Idempotent — safe to re-run; existing roles/profiles are reused.
#
# Required env (or pass as flags):
#   AWS_PROFILE              SSO profile (admin perms required)
#   AWS_DEFAULT_REGION       us-gov-west-1
#   SUBMIT_ROLE_NAME         the submit-node role to expand
#   S3_BUCKET                bucket the workers should be able to read

set -euo pipefail

AWS_PROFILE="${AWS_PROFILE:-}"
AWS_REGION="${AWS_DEFAULT_REGION:-${AWS_REGION:-us-gov-west-1}}"
STACK_PREFIX="${STACK_PREFIX:-smsvpctest}"
SUBMIT_ROLE_NAME="${SUBMIT_ROLE_NAME:-}"
S3_BUCKET="${S3_BUCKET:-}"
WORKER_ROLE_NAME="ray-spatio-flux-node"
WORKER_PROFILE_NAME="ray-spatio-flux-node"

show_help() {
    cat <<EOF
setup-iam-for-ray.sh — provision IAM for the Ray cluster used by the
spatio-flux comparison.

Usage:
    AWS_PROFILE=<sso-profile> ./scripts/setup-iam-for-ray.sh -s <stack> -b <s3-bucket>

Required (one of):
    -s, --stack PREFIX        CFN stack prefix; auto-discovers the submit
                              role from <stack>-batch's SubmitNodeInstanceId
                              output (default: ${STACK_PREFIX})
    -r, --submit-role NAME    explicit submit-node role to expand
                              (skips auto-discovery)
    -b, --bucket NAME         the S3 bucket workers will read from

Options:
    --worker-role NAME        worker role + instance profile name
                              (default: ray-spatio-flux-node)
    -h, --help                show this help

Run as a user with permissions to:
    - iam:PutRolePolicy on the submit role
    - iam:CreateRole / CreateInstanceProfile / AttachRolePolicy
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -s|--stack)        STACK_PREFIX="$2"; shift 2 ;;
        -r|--submit-role)  SUBMIT_ROLE_NAME="$2"; shift 2 ;;
        -b|--bucket)       S3_BUCKET="$2"; shift 2 ;;
        --worker-role)     WORKER_ROLE_NAME="$2"; WORKER_PROFILE_NAME="$2"; shift 2 ;;
        -h|--help)         show_help; exit 0 ;;
        *) echo "Unknown option: $1" >&2; show_help; exit 1 ;;
    esac
done

[[ -z "$AWS_PROFILE" ]] && { echo "AWS_PROFILE required" >&2; exit 1; }
[[ -z "$S3_BUCKET"   ]] && { echo "S3 bucket required (-b)" >&2; exit 1; }

aws_call() {
    aws --profile "$AWS_PROFILE" --region "$AWS_REGION" "$@"
}
aws_iam() {
    aws_call iam "$@"
}

# Auto-discover the submit role from the stack if not explicitly given.
# Walks: <stack>-batch CFN output → submit instance ID → instance profile
# ARN → role name. Requires ec2:DescribeInstances + iam:GetInstanceProfile,
# both of which the user's admin SSO already has.
if [[ -z "$SUBMIT_ROLE_NAME" ]]; then
    echo "→ auto-discovering submit role from stack ${STACK_PREFIX}-batch ..."
    SUBMIT_INSTANCE_ID="$(aws_call cloudformation describe-stacks \
        --stack-name "${STACK_PREFIX}-batch" \
        --query "Stacks[0].Outputs[?OutputKey=='SubmitNodeInstanceId'].OutputValue" \
        --output text 2>/dev/null)"
    if [[ -z "$SUBMIT_INSTANCE_ID" || "$SUBMIT_INSTANCE_ID" == "None" ]]; then
        echo "  ✗ couldn't resolve SubmitNodeInstanceId from ${STACK_PREFIX}-batch" >&2
        echo "    pass -r <role-name> explicitly, or check STACK_PREFIX" >&2
        exit 1
    fi
    PROFILE_ARN="$(aws_call ec2 describe-instances \
        --instance-ids "$SUBMIT_INSTANCE_ID" \
        --query 'Reservations[0].Instances[0].IamInstanceProfile.Arn' \
        --output text)"
    PROFILE_NAME="${PROFILE_ARN##*/}"
    SUBMIT_ROLE_NAME="$(aws_iam get-instance-profile \
        --instance-profile-name "$PROFILE_NAME" \
        --query 'InstanceProfile.Roles[0].RoleName' --output text)"
    echo "  ✓ discovered: $SUBMIT_ROLE_NAME"
fi

# Discover the AWS partition (aws / aws-us-gov / aws-cn) so resource
# ARNs in the policy match the deployment.
PARTITION="$(
    aws --profile "$AWS_PROFILE" --region "$AWS_REGION" \
        sts get-caller-identity --query 'Arn' --output text \
    | awk -F: '{print $2}'
)"
ACCOUNT_ID="$(
    aws --profile "$AWS_PROFILE" --region "$AWS_REGION" \
        sts get-caller-identity --query 'Account' --output text
)"

echo "→ partition: $PARTITION"
echo "→ account:   $ACCOUNT_ID"
echo "→ region:    $AWS_REGION"
echo

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# =========================================================================
# 1) Expand the submit-node role with cluster-management permissions.
# =========================================================================
echo "═══ Step 1: cluster-mgmt policy on submit role ═══"
echo "→ role: $SUBMIT_ROLE_NAME"

cat >"$WORK/cluster-mgmt-policy.json" <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "EC2Describe",
      "Effect": "Allow",
      "Action": [
        "ec2:DescribeInstances",
        "ec2:DescribeInstanceStatus",
        "ec2:DescribeInstanceTypes",
        "ec2:DescribeIamInstanceProfileAssociations",
        "ec2:DescribeSubnets",
        "ec2:DescribeSecurityGroups",
        "ec2:DescribeAvailabilityZones",
        "ec2:DescribeImages",
        "ec2:DescribeKeyPairs",
        "ec2:DescribeTags",
        "ec2:DescribeRouteTables",
        "ec2:DescribeVpcs",
        "ec2:DescribeNetworkInterfaces",
        "ec2:DescribeNetworkAcls",
        "ec2:DescribePrefixLists",
        "ec2:DescribeAccountAttributes",
        "ec2:DescribeSpotInstanceRequests",
        "ec2:DescribeSpotPriceHistory"
      ],
      "Resource": "*"
    },
    {
      "Sid": "EC2Manage",
      "Effect": "Allow",
      "Action": [
        "ec2:RunInstances",
        "ec2:TerminateInstances",
        "ec2:StartInstances",
        "ec2:StopInstances",
        "ec2:RebootInstances",
        "ec2:CreateTags",
        "ec2:CreateKeyPair",
        "ec2:DeleteKeyPair",
        "ec2:CreateSecurityGroup",
        "ec2:AuthorizeSecurityGroupIngress",
        "ec2:RevokeSecurityGroupIngress",
        "ec2:DeleteSecurityGroup",
        "ec2:AssociateIamInstanceProfile",
        "ec2:ReplaceIamInstanceProfileAssociation",
        "ec2:DisassociateIamInstanceProfile",
        "ec2:RequestSpotInstances",
        "ec2:CancelSpotInstanceRequests"
      ],
      "Resource": "*"
    },
    {
      "Sid": "PassWorkerRole",
      "Effect": "Allow",
      "Action": "iam:PassRole",
      "Resource": "arn:${PARTITION}:iam::${ACCOUNT_ID}:role/${WORKER_ROLE_NAME}"
    },
    {
      "Sid": "ManageWorkerInstanceProfile",
      "Effect": "Allow",
      "Action": [
        "iam:GetInstanceProfile",
        "iam:GetRole",
        "iam:ListRolePolicies",
        "iam:ListAttachedRolePolicies"
      ],
      "Resource": [
        "arn:${PARTITION}:iam::${ACCOUNT_ID}:instance-profile/${WORKER_PROFILE_NAME}",
        "arn:${PARTITION}:iam::${ACCOUNT_ID}:role/${WORKER_ROLE_NAME}"
      ]
    },
    {
      "Sid": "SSMAMILookup",
      "Effect": "Allow",
      "Action": [
        "ssm:GetParameter",
        "ssm:GetParameters"
      ],
      "Resource": "arn:${PARTITION}:ssm:*::parameter/aws/service/canonical/*"
    },
    {
      "Sid": "DiagnosticsConsoleOutput",
      "Effect": "Allow",
      "Action": "ec2:GetConsoleOutput",
      "Resource": "*"
    },
    {
      "Sid": "DiagnosticsSSMSendCommand",
      "Effect": "Allow",
      "Action": "ssm:SendCommand",
      "Resource": [
        "arn:${PARTITION}:ssm:${AWS_REGION}::document/AWS-RunShellScript",
        "arn:${PARTITION}:ec2:${AWS_REGION}:${ACCOUNT_ID}:instance/*"
      ]
    },
    {
      "Sid": "DiagnosticsSSMGetCommandInvocation",
      "Effect": "Allow",
      "Action": "ssm:GetCommandInvocation",
      "Resource": "*"
    },
    {
      "Sid": "OrchestratorSSMDescribeInstances",
      "Effect": "Allow",
      "Action": "ssm:DescribeInstanceInformation",
      "Resource": "*"
    },
    {
      "Sid": "S3WorkingBucket",
      "Effect": "Allow",
      "Action": [
        "s3:ListBucket"
      ],
      "Resource": "arn:${PARTITION}:s3:::${S3_BUCKET}"
    },
    {
      "Sid": "S3WorkingBucketObjects",
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:PutObjectAcl",
        "s3:DeleteObject"
      ],
      "Resource": "arn:${PARTITION}:s3:::${S3_BUCKET}/spatio-flux/*"
    },
    {
      "Sid": "EC2InstanceConnect",
      "Effect": "Allow",
      "Action": [
        "ec2-instance-connect:SendSSHPublicKey"
      ],
      "Resource": "*"
    },
    {
      "Sid": "ECRPushAndManage",
      "Effect": "Allow",
      "Action": [
        "ecr:GetAuthorizationToken",
        "ecr:BatchCheckLayerAvailability",
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchGetImage",
        "ecr:DescribeRepositories",
        "ecr:CreateRepository",
        "ecr:InitiateLayerUpload",
        "ecr:UploadLayerPart",
        "ecr:CompleteLayerUpload",
        "ecr:PutImage"
      ],
      "Resource": "*"
    }
  ]
}
EOF

aws_iam put-role-policy \
    --role-name "$SUBMIT_ROLE_NAME" \
    --policy-name RaySpatioFluxClusterMgmt \
    --policy-document "file://$WORK/cluster-mgmt-policy.json"
echo "✓ inline policy RaySpatioFluxClusterMgmt attached"
echo

# =========================================================================
# 2) Create the worker role + instance profile.
# =========================================================================
echo "═══ Step 2: worker role + instance profile ═══"
echo "→ role/profile: $WORKER_ROLE_NAME"

cat >"$WORK/worker-trust.json" <<EOF
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Service": "ec2.amazonaws.com"},
    "Action": "sts:AssumeRole"
  }]
}
EOF

if aws_iam get-role --role-name "$WORKER_ROLE_NAME" >/dev/null 2>&1; then
    echo "  role $WORKER_ROLE_NAME already exists — reusing"
else
    aws_iam create-role \
        --role-name "$WORKER_ROLE_NAME" \
        --assume-role-policy-document "file://$WORK/worker-trust.json" \
        --description "Ray cluster worker role for spatio-flux comparison" \
        >/dev/null
    echo "✓ created role $WORKER_ROLE_NAME"
fi

# S3 read on the bucket so workers can pull artifacts at boot, plus
# ECR pull permissions so they can fetch the spatio-flux Docker image.
cat >"$WORK/worker-ecr.json" <<EOF
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": [
      "ecr:GetAuthorizationToken",
      "ecr:BatchCheckLayerAvailability",
      "ecr:GetDownloadUrlForLayer",
      "ecr:BatchGetImage"
    ],
    "Resource": "*"
  }]
}
EOF
aws_iam put-role-policy \
    --role-name "$WORKER_ROLE_NAME" \
    --policy-name ECRPull \
    --policy-document "file://$WORK/worker-ecr.json"
echo "✓ inline policy ECRPull attached"

cat >"$WORK/worker-s3.json" <<EOF
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": [
      "s3:GetObject",
      "s3:ListBucket"
    ],
    "Resource": [
      "arn:${PARTITION}:s3:::${S3_BUCKET}",
      "arn:${PARTITION}:s3:::${S3_BUCKET}/*"
    ]
  }, {
    "Sid": "WriteResultsBack",
    "Effect": "Allow",
    "Action": [
      "s3:PutObject",
      "s3:PutObjectAcl"
    ],
    "Resource": "arn:${PARTITION}:s3:::${S3_BUCKET}/spatio-flux/runs/*"
  }]
}
EOF

aws_iam put-role-policy \
    --role-name "$WORKER_ROLE_NAME" \
    --policy-name S3RW \
    --policy-document "file://$WORK/worker-s3.json"
echo "✓ inline policy S3RW attached"

# SSM-agent baseline lets you `aws ssm start-session` into worker nodes
# for live debugging. Not required for the experiment itself, but
# saves a lot of pain when something goes wrong on a worker.
aws_iam attach-role-policy \
    --role-name "$WORKER_ROLE_NAME" \
    --policy-arn "arn:${PARTITION}:iam::aws:policy/AmazonSSMManagedInstanceCore" \
    >/dev/null 2>&1 || true
echo "✓ AmazonSSMManagedInstanceCore attached"

if aws_iam get-instance-profile --instance-profile-name "$WORKER_PROFILE_NAME" \
        >/dev/null 2>&1; then
    echo "  instance profile $WORKER_PROFILE_NAME already exists — reusing"
else
    aws_iam create-instance-profile \
        --instance-profile-name "$WORKER_PROFILE_NAME" >/dev/null
    echo "✓ created instance profile $WORKER_PROFILE_NAME"
fi

# Add role to instance profile (idempotent — silently no-ops if attached).
if ! aws_iam get-instance-profile --instance-profile-name "$WORKER_PROFILE_NAME" \
        --query 'InstanceProfile.Roles[?RoleName==`'"$WORKER_ROLE_NAME"'`]' \
        --output text | grep -q "$WORKER_ROLE_NAME"; then
    aws_iam add-role-to-instance-profile \
        --instance-profile-name "$WORKER_PROFILE_NAME" \
        --role-name "$WORKER_ROLE_NAME"
    echo "✓ added role to instance profile"
else
    echo "  role already in instance profile"
fi

echo
echo "═══ Done ═══"
echo
echo "You can now drop --local-ray and run:"
echo "  ./scripts/run-comparison-on-ec2.sh -s smsvpctest -b $S3_BUCKET --mode large"
echo
echo "The next ray up will:"
echo "  - assume the now-empowered submit role"
echo "  - launch worker EC2s with the $WORKER_PROFILE_NAME instance profile"
echo "  - have the workers pull source + COMETS from your bucket"
echo
echo "─── For reference: permissions YOUR (laptop) SSO role needs ───"
echo "These should already be on your admin SSO role; verify if a step fails:"
cat <<EOF
    cloudformation:DescribeStacks         (find submit-node ID by stack output)
    s3:ListBucket    on $S3_BUCKET        (browse the runs prefix)
    s3:GetObject     on $S3_BUCKET/*      (download bootstrap log + results)
    s3:PutObject     on $S3_BUCKET/*      (upload source/COMETS/bootstrap)
    ssm:SendCommand                       (kick off ec2-bootstrap.sh)
    ssm:GetCommandInvocation              (poll status, fetch output on failure)
    ssm:StartSession                      (manual debug into the submit node)
    iam:PutRolePolicy                     (this script — cluster-mgmt policy)
    iam:CreateRole / GetRole              (this script — worker role)
    iam:CreateInstanceProfile             (this script — worker profile)
    iam:AttachRolePolicy                  (this script — SSM agent baseline)
    iam:AddRoleToInstanceProfile          (this script — wire worker role)
    sts:GetCallerIdentity                 (partition + account discovery)
EOF
