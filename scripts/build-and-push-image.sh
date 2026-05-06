#!/usr/bin/env bash
#
# build-and-push-image.sh — laptop-side build of the spatio-flux worker
# image, push to GovCloud ECR. One-time per code/dependency change.
#
# What this does:
#   1. Stage the docker build context (spatio-flux source + COMETS jar)
#      in a temp dir.
#   2. docker build (against your laptop's fast internet — the apt + pip
#      slowness happens here, not in the VPC).
#   3. Create the ECR repo if it doesn't exist.
#   4. docker login to ECR.
#   5. docker tag + push.
#   6. Print the image URI to drop into the cluster yaml.
#
# Required env or flags:
#   AWS_PROFILE              SSO profile
#   AWS_DEFAULT_REGION       us-gov-west-1
#   COMETS_JAR_TARBALL       path to local COMETS install dir (default ~/comets_install/...)
#
# Optional:
#   IMAGE_NAME               default: spatio-flux-worker
#   IMAGE_TAG                default: latest
#
# Usage:
#   AWS_PROFILE=<sso-profile> AWS_DEFAULT_REGION=us-gov-west-1 \
#       ./scripts/build-and-push-image.sh

set -euo pipefail

AWS_PROFILE="${AWS_PROFILE:-}"
AWS_REGION="${AWS_DEFAULT_REGION:-${AWS_REGION:-us-gov-west-1}}"
COMETS_SRC="${COMETS_JAR_TARBALL:-${HOME}/comets_install/comets_linux/comets_2.12.5}"
IMAGE_NAME="${IMAGE_NAME:-spatio-flux-worker}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
PLATFORM="${PLATFORM:-linux/amd64}"   # workers are m5.4xlarge = x86_64

while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--comets-src) COMETS_SRC="$2"; shift 2 ;;
        --tag)           IMAGE_TAG="$2"; shift 2 ;;
        --name)          IMAGE_NAME="$2"; shift 2 ;;
        --platform)      PLATFORM="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,/^$/p' "$0" | sed 's/^# \?//'
            exit 0 ;;
        *) echo "Unknown: $1" >&2; exit 1 ;;
    esac
done

[[ -z "$AWS_PROFILE" ]] && { echo "AWS_PROFILE required" >&2; exit 1; }
[[ ! -d "$COMETS_SRC" ]] && { echo "COMETS source dir not found: $COMETS_SRC" >&2; exit 1; }
command -v docker >/dev/null || { echo "docker not on PATH" >&2; exit 1; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---------- 1. stage build context ----------------------------------------
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "→ staging build context in $WORK ..."
mkdir -p "$WORK/src" "$WORK/comets"
# Copy the source tree, excluding heavy / irrelevant bits. tar+untar is
# the simplest portable filter.
tar --exclude='.venv' --exclude='out' --exclude='out_*' --exclude='originalout' \
    --exclude='.git' --exclude='__pycache__' --exclude='*.egg-info' \
    --exclude='dist' --exclude='profile' --exclude='demo' \
    -C "$REPO_ROOT" -cf - . | tar -C "$WORK/src" -xf -
cp -R "$COMETS_SRC"/. "$WORK/comets/"
cp "$REPO_ROOT/deploy/Dockerfile" "$WORK/Dockerfile"

# Sibling repo: process-bigraph. We install from local source instead of
# PyPI when a sibling checkout exists, so framework changes (e.g.
# tick_lifecycle hooks) are picked up immediately. Default lookup is
# ../process-bigraph relative to spatio-flux. Override with
# PROCESS_BIGRAPH_SRC=<path> or set to "skip" to use PyPI.
# Always create the staging directory (even if empty) so the Dockerfile's
# COPY directive doesn't error on missing path.
PROCESS_BIGRAPH_SRC="${PROCESS_BIGRAPH_SRC:-${REPO_ROOT}/../process-bigraph}"
mkdir -p "$WORK/process-bigraph"
if [[ "$PROCESS_BIGRAPH_SRC" != "skip" && -d "$PROCESS_BIGRAPH_SRC" ]]; then
    tar --exclude='.venv' --exclude='.git' --exclude='__pycache__' \
        --exclude='*.egg-info' --exclude='dist' --exclude='build' \
        -C "$PROCESS_BIGRAPH_SRC" -cf - . | tar -C "$WORK/process-bigraph" -xf -
    echo "   process-bigraph: local source ($(du -sh "$WORK/process-bigraph" | cut -f1))"
else
    # Empty marker dir; Dockerfile checks for pyproject.toml presence.
    touch "$WORK/process-bigraph/.empty"
    echo "   process-bigraph: PyPI (no local source at $PROCESS_BIGRAPH_SRC)"
fi

echo "   src:    $(du -sh "$WORK/src" | cut -f1)"
echo "   comets: $(du -sh "$WORK/comets" | cut -f1)"

# ---------- 2. docker build -----------------------------------------------
LOCAL_TAG="$IMAGE_NAME:$IMAGE_TAG"
echo
echo "→ docker build $LOCAL_TAG (platform=$PLATFORM) ..."
docker build --platform "$PLATFORM" -t "$LOCAL_TAG" "$WORK"
echo "✓ image built: $LOCAL_TAG"

# ---------- 3. ECR repo create-if-missing ---------------------------------
ACCOUNT_ID="$(aws --profile "$AWS_PROFILE" --region "$AWS_REGION" \
    sts get-caller-identity --query Account --output text)"
PARTITION="$(aws --profile "$AWS_PROFILE" --region "$AWS_REGION" \
    sts get-caller-identity --query Arn --output text | awk -F: '{print $2}')"
ECR_HOST="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
ECR_URI="${ECR_HOST}/${IMAGE_NAME}:${IMAGE_TAG}"

echo
echo "→ ECR account=$ACCOUNT_ID region=$AWS_REGION repo=$IMAGE_NAME"
if ! aws --profile "$AWS_PROFILE" --region "$AWS_REGION" \
        ecr describe-repositories --repository-names "$IMAGE_NAME" >/dev/null 2>&1; then
    echo "   creating ECR repo $IMAGE_NAME ..."
    aws --profile "$AWS_PROFILE" --region "$AWS_REGION" \
        ecr create-repository --repository-name "$IMAGE_NAME" \
        --image-scanning-configuration scanOnPush=true \
        --image-tag-mutability MUTABLE >/dev/null
    echo "   ✓ created"
else
    echo "   (already exists)"
fi

# ---------- 4. login + 5. tag + push --------------------------------------
echo
echo "→ docker login to $ECR_HOST ..."
aws --profile "$AWS_PROFILE" --region "$AWS_REGION" \
    ecr get-login-password \
    | docker login --username AWS --password-stdin "$ECR_HOST"

echo "→ tag + push $ECR_URI ..."
docker tag "$LOCAL_TAG" "$ECR_URI"
docker push "$ECR_URI"

echo
echo "═══ Done ═══"
echo
echo "Image URI: $ECR_URI"
echo
echo "Recording in deploy/.spatio-flux-image (used by run-comparison-on-ec2.sh)"
echo "$ECR_URI" > "$REPO_ROOT/deploy/.spatio-flux-image"
echo
echo "Now run:"
echo "  ./scripts/run-comparison-on-ec2.sh -s smsvpctest -b <bucket> --mode large --keep-cluster"
echo
echo "The orchestrator will pick up the image URI from deploy/.spatio-flux-image."
