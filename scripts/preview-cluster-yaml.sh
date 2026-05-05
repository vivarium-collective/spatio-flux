#!/usr/bin/env bash
#
# preview-cluster-yaml.sh — render deploy/ray-govcloud-cluster.yaml
# locally to see exactly what Ray will see, without waiting on a real
# ray up cycle. Substitutes envsubst variables with the same values
# ec2-bootstrap.sh would (or with placeholders if you don't have AWS
# credentials handy).
#
# Use during cluster yaml debugging — hits in seconds, not 14 minutes.

set -euo pipefail

# Defaults — override via env or by editing inline.
export STACK_PREFIX="${STACK_PREFIX:-smsvpctest}"
export AWS_REGION_RESOLVED="${AWS_REGION_RESOLVED:-us-gov-west-1}"
export VPC_ID="${VPC_ID:-vpc-013f0c1012b271b06}"
export SUBNET_ID="${SUBNET_ID:-subnet-08621613bcb558caa}"
export SG_ID="${SG_ID:-sg-0da6ec85b177071fc}"
export SUBMIT_NODE_IP="${SUBMIT_NODE_IP:-10.99.44.121}"
export AMI_ID="${AMI_ID:-ami-03c8c0f857bdcabf5}"
export IAM_INSTANCE_PROFILE="${IAM_INSTANCE_PROFILE:-ray-spatio-flux-node}"
export IMAGE_URI="${IMAGE_URI:-123456789012.dkr.ecr.us-gov-west-1.amazonaws.com/spatio-flux-worker:latest}"
export ECR_HOST="${ECR_HOST:-${IMAGE_URI%%/*}}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEMPLATE="$REPO_ROOT/deploy/ray-govcloud-cluster.yaml"

echo "=== rendered $TEMPLATE ==="
# Match the envsubst call ec2-bootstrap.sh uses — only substitute these
# named vars, leave $HOME / $PATH / $RAY_HEAD_IP etc as literal shell
# vars for the head's bash to expand at run time.
envsubst '$AMI_ID $VPC_ID $SUBNET_ID $SG_ID $SUBMIT_NODE_IP $AWS_REGION_RESOLVED $S3_PREFIX $S3_BUCKET_NAME $STACK_PREFIX $IAM_INSTANCE_PROFILE $IMAGE_URI $ECR_HOST' \
    < "$TEMPLATE"
echo
echo "=== values used ==="
for v in STACK_PREFIX AWS_REGION_RESOLVED VPC_ID SUBNET_ID SG_ID \
         SUBMIT_NODE_IP AMI_ID IAM_INSTANCE_PROFILE; do
    printf '  %-22s = %s\n' "$v" "${!v}"
done
echo
echo "If anything looks wrong, edit the values at the top of this script"
echo "or export them and re-run. No AWS calls are made."
