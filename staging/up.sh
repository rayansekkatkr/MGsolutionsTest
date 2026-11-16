#!/usr/bin/env bash
# Builds the staging environment and deploys exam/template.yaml into it, unchanged, on real GPU hosts.
# Usage: bash staging/up.sh        (about 20 minutes)   region: STAGING_REGION, default ap-northeast-2
source "$(dirname "$0")/lib.sh"

ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
REGISTRY="$ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com"
echo "== account $ACCOUNT, region $AWS_REGION"

# ---- GPU hosts the account can start here: 4 vCPUs each, minus G instances already running (the benchmark runner)
QUOTA="$(aws service-quotas get-service-quota --service-code ec2 --quota-code L-DB2E81BA --query 'Quota.Value' --output text)"
USED="$(aws ec2 describe-instances --filters "Name=instance-type,Values=g*" "Name=instance-state-name,Values=pending,running" \
  --query "length(Reservations[].Instances[?!contains(to_string(Tags), 'exam-rstg')][])" --output text)"
HOSTS=$(python3 -c "print(min(int('${STAGING_HOSTS:-2}'), int(float('$QUOTA')) // 4 - int('$USED')))")
echo "== GPU vCPU quota $QUOTA, GPU instances already running outside staging: $USED -> $HOSTS hosts"
if (( HOSTS < 2 )); then
  echo "Two GPU hosts are needed (one is terminated during the test). Stop the benchmark runner (bash gpu/down.sh)"
  echo "or use a region where the quota is free, e.g. STAGING_REGION=us-east-1 bash staging/up.sh"
  exit 1
fi
# Two zones that offer the instance type. An offering is not a guarantee of capacity at a given
# moment: STAGING_ZONES="us-east-1c us-east-1d" chooses other zones when a launch fails for lack of it.
FARGATE="$(aws service-quotas get-service-quota --service-code fargate --quota-code L-3032A538 --query 'Quota.Value' --output text 2>/dev/null || echo unknown)"
echo "== Fargate vCPU quota: $FARGATE (needed: $HOSTS for the edge proxies + 4 for the load client)"
ZONES="${STAGING_ZONES:-$(aws ec2 describe-instance-type-offerings --location-type availability-zone --filters Name=instance-type,Values=g6.xlarge \
  --query 'InstanceTypeOfferings[].Location' --output text | tr '\t' '\n' | sort | head -2 | tr '\n' ' ')}"
set -- $ZONES
if [[ $# -lt 2 ]]; then echo "Fewer than two zones offer g6.xlarge in $AWS_REGION: $ZONES"; exit 1; fi
echo "== zones: $1 $2"

# ---- 1. platform: VPC without internet, endpoints, roles, repositories, bucket, load client
if [[ "$(stack_status "$PLATFORM_STACK")" == DELETE_IN_PROGRESS ]]; then
  echo "== waiting for the previous platform stack to finish deleting"
  aws cloudformation wait stack-delete-complete --stack-name "$PLATFORM_STACK"
fi
echo "== [1/5] platform stack ($(utc) UTC, about 4 minutes)"
aws cloudformation deploy --stack-name "$PLATFORM_STACK" --template-file "$ROOT/staging/platform.yaml" \
  --capabilities CAPABILITY_IAM --no-fail-on-empty-changeset --parameter-overrides "AzA=$1" "AzB=$2"

# ---- 2. private copies of the images, registry to registry
echo "== [2/5] copying images into the account's private registry (the inference image is about 5 GB)"
CRANE="$(command -v crane || true)"
if [[ -z "$CRANE" ]]; then
  OS="$(uname -s)"; ARCH="$(uname -m)"; [[ "$ARCH" == aarch64 ]] && ARCH=arm64
  curl -fsSL "https://github.com/google/go-containerregistry/releases/latest/download/go-containerregistry_${OS}_${ARCH}.tar.gz" \
    | tar -xz -C "$OUT" crane
  CRANE="$OUT/crane"
fi
aws ecr get-login-password | "$CRANE" auth login "$REGISTRY" -u AWS --password-stdin >/dev/null
image_address() { python3 -c "import json;print(json.load(open('$ROOT/exam/images.json'))['$1'])"; }
INFERENCE_SOURCE="$(image_address Inference)"
HAPROXY_TAG="$(image_address Haproxy)"; HAPROXY_TAG="${HAPROXY_TAG##*:}"
HAPROXY_IMAGE="$REGISTRY/exam-staging/haproxy:$HAPROXY_TAG"
PYTHON_IMAGE="$REGISTRY/exam-staging/python:3.12-slim"
INFERENCE_IMAGE="$REGISTRY/exam-staging/inference:candidate"
# public.ecr.aws/docker/library is the official mirror of Docker Hub's official images
"$CRANE" copy --platform linux/amd64 "public.ecr.aws/docker/library/haproxy:$HAPROXY_TAG" "$HAPROXY_IMAGE" >/dev/null
"$CRANE" copy --platform linux/amd64 "public.ecr.aws/docker/library/python:3.12-slim" "$PYTHON_IMAGE" >/dev/null
echo "   inference image: $INFERENCE_SOURCE"
if ! "$CRANE" copy "$INFERENCE_SOURCE" "$INFERENCE_IMAGE" >/dev/null; then
  echo "Could not read $INFERENCE_SOURCE without logging in. The exam needs it public as well:"
  echo "GitHub > your profile > Packages > mgsolutionstest-inference > Package settings > Change visibility > Public"
  exit 1
fi
INFERENCE_IMAGE="$REGISTRY/exam-staging/inference@$("$CRANE" digest "$INFERENCE_IMAGE")"
echo "   copied as $INFERENCE_IMAGE"

# ---- 3. the load client's bundle: generator and photos
echo "== [3/5] load client bundle"
BUCKET="$(stack_output "$PLATFORM_STACK" BucketName)"
mkdir -p "$OUT/bundle/photos"
head -60 "$ROOT/tests/inference/images_gpu.txt" | while read -r url; do
  [ -s "$OUT/bundle/photos/$(basename "$url")" ] || curl -fsS --retry 3 -o "$OUT/bundle/photos/$(basename "$url")" "$url"
done
cp "$ROOT/bench/load_open.py" "$OUT/bundle/"
# COPYFILE_DISABLE: macOS tar would otherwise add a "._name" metadata entry next to every file
COPYFILE_DISABLE=1 tar -czf "$OUT/bundle.tar.gz" --exclude '._*' -C "$OUT/bundle" load_open.py photos
aws s3 cp --quiet "$OUT/bundle.tar.gz" "s3://$BUCKET/bundle.tar.gz"
echo "   $(ls "$OUT/bundle/photos" | wc -l | tr -d ' ') photos uploaded"

# ---- 4. parameters, as the exam passes them
echo "== [4/5] parameters"
AMI="$(aws ssm get-parameter --name /aws/service/ecs/optimized-ami/amazon-linux-2023/gpu/recommended/image_id --query Parameter.Value --output text)"
S3_PREFIX_LIST="$(aws ec2 describe-managed-prefix-lists --filters "Name=prefix-list-name,Values=com.amazonaws.$AWS_REGION.s3" --query 'PrefixLists[0].PrefixListId' --output text)"
python3 - "$OUT/params.json" <<PY
import json, sys
params = {
    "VpcId": "$(stack_output "$PLATFORM_STACK" VpcId)",
    "SubnetIds": "$(stack_output "$PLATFORM_STACK" SubnetIds)",
    "ClientSecurityGroupId": "$(stack_output "$PLATFORM_STACK" ClientSecurityGroupId)",
    "EndpointSecurityGroupId": "$(stack_output "$PLATFORM_STACK" EndpointSecurityGroupId)",
    "S3PrefixListId": "$S3_PREFIX_LIST",
    "RunId": "$RUN_ID",
    "TaskExecutionRoleArn": "$(stack_output "$PLATFORM_STACK" TaskExecutionRoleArn)",
    "AppRoleArn": "$(stack_output "$PLATFORM_STACK" AppRoleArn)",
    "InstanceProfileArn": "$(stack_output "$PLATFORM_STACK" InstanceProfileArn)",
    "EcsGpuAmiId": "$AMI",
    "ImageInference": "$INFERENCE_IMAGE",
    "ImageHaproxy": "$HAPROXY_IMAGE",
    "Hosts": "$HOSTS",
    "EdgeTasks": "$HOSTS",
}
json.dump([{"ParameterKey": k, "ParameterValue": v} for k, v in params.items()], open(sys.argv[1], "w"), indent=1)
PY

# ---- 5. the exam stack: the submitted file itself, with the exam's 20-minute creation limit
diagnose() {
  echo "---- stack events that failed"
  aws cloudformation describe-stack-events --stack-name "$APP_STACK" --output text \
    --query "StackEvents[?contains(ResourceStatus,'FAILED')].[Timestamp,LogicalResourceId,ResourceStatusReason]" | head -15 || true
  echo "---- latest ECS service events"
  aws ecs describe-services --cluster "$CLUSTER" --services "$SERVICE" --query 'services[0].events[:8].message' --output text 2>/dev/null | tr '\t' '\n' || true
  echo "---- stopped tasks"
  STOPPED="$(aws ecs list-tasks --cluster "$CLUSTER" --desired-status STOPPED --query 'taskArns[:3]' --output text 2>/dev/null || true)"
  if [[ -n "$STOPPED" && "$STOPPED" != None ]]; then
    # shellcheck disable=SC2086
    aws ecs describe-tasks --cluster "$CLUSTER" --tasks $STOPPED --output json \
      --query 'tasks[].{stopped:stoppedReason,containers:containers[].{name:name,reason:reason,exit:exitCode}}' || true
  fi
  echo "---- last lines of the inference containers"
  aws logs tail "/exam-run/r${RUN_ID}/inference" --since 30m --format short 2>/dev/null | tail -25 || true
}

STATUS="$(stack_status "$APP_STACK")"
if [[ "$STATUS" == CREATE_COMPLETE || "$STATUS" == UPDATE_COMPLETE ]]; then
  echo "== [5/5] exam stack already deployed ($STATUS). Run staging/down.sh first to redeploy from scratch."
else
  if [[ "$STATUS" != ABSENT ]]; then
    echo "== removing the exam stack left in state $STATUS"
    aws cloudformation delete-stack --stack-name "$APP_STACK"
    aws cloudformation wait stack-delete-complete --stack-name "$APP_STACK"
  fi
  echo "== [5/5] creating the exam stack ($(utc) UTC, limit 20 minutes as in the exam)"
  aws cloudformation create-stack --stack-name "$APP_STACK" --template-body "file://$ROOT/exam/template.yaml" \
    --parameters "file://$OUT/params.json" >/dev/null
  START=$SECONDS
  while true; do
    sleep 20
    STATUS="$(stack_status "$APP_STACK")"
    RUNNING="$(aws ecs describe-services --cluster "$CLUSTER" --services "$SERVICE" --query 'services[0].runningCount' --output text 2>/dev/null || echo -)"
    EVENT="$(aws ecs describe-services --cluster "$CLUSTER" --services "$SERVICE" --query 'services[0].events[0].message' --output text 2>/dev/null | cut -c1-110 || true)"
    printf '   +%4ds  %-20s tasks running: %s/%s  %s\n' "$((SECONDS - START))" "$STATUS" "$RUNNING" "$HOSTS" "$EVENT"
    case "$STATUS" in
      CREATE_COMPLETE) break ;;
      CREATE_IN_PROGRESS) ;;
      *) echo "== STACK CREATION FAILED ($STATUS)"; diagnose; exit 1 ;;
    esac
    if (( SECONDS - START > 1200 )); then
      echo "== STACK CREATION EXCEEDED 20 MINUTES (the exam would count this as a failed creation)"
      diagnose; exit 1
    fi
  done
  echo "== exam stack created in $((SECONDS - START)) s"
fi

echo "== endpoint: $(stack_output "$APP_STACK" EndpointUrl)"
echo "== next: bash staging/test.sh      and when finished: bash staging/down.sh   (same STAGING_REGION)"
