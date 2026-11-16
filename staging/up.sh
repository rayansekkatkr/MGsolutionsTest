#!/usr/bin/env bash
# Builds the staging environment and deploys the exam template into it.
# Usage: bash staging/up.sh        (about 10 to 15 minutes)
source "$(dirname "$0")/lib.sh"

ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
REGISTRY="$ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com"
echo "== account $ACCOUNT, region $AWS_REGION"

# ---- how many hosts the account's vCPU quota allows (t3.large = 2 vCPU each)
QUOTA="$(aws service-quotas get-service-quota --service-code ec2 --quota-code L-1216C47A --query 'Quota.Value' --output text 2>/dev/null || echo unknown)"
HOSTS="${STAGING_HOSTS:-3}"
if [[ "$QUOTA" =~ ^[0-9.]+$ ]]; then
  MAX=$(python3 -c "print(int(float('$QUOTA')) // 2)")
  if (( MAX < 2 )); then echo "On-demand vCPU quota is $QUOTA: at least 4 vCPUs are needed for two hosts. Stopping."; exit 1; fi
  if (( HOSTS > MAX )); then HOSTS=$MAX; fi
fi
echo "== on-demand vCPU quota: $QUOTA -> $HOSTS hosts"

# ---- 1. platform: VPC without internet, endpoints, roles, repositories, load client
echo "== [1/4] platform stack ($(utc) UTC, about 4 minutes)"
aws cloudformation deploy --stack-name "$PLATFORM_STACK" --template-file "$ROOT/staging/platform.yaml" \
  --capabilities CAPABILITY_IAM --no-fail-on-empty-changeset

# ---- 2. private copies of the two public images
echo "== [2/4] copying images into the account's private registry"
CRANE="$(command -v crane || true)"
if [[ -z "$CRANE" ]]; then
  OS="$(uname -s)"; ARCH="$(uname -m)"; [[ "$ARCH" == aarch64 ]] && ARCH=arm64
  curl -fsSL "https://github.com/google/go-containerregistry/releases/latest/download/go-containerregistry_${OS}_${ARCH}.tar.gz" \
    | tar -xz -C "$OUT" crane
  CRANE="$OUT/crane"
fi
aws ecr get-login-password | "$CRANE" auth login "$REGISTRY" -u AWS --password-stdin >/dev/null
HAPROXY_TAG="$(python3 -c "import json;print(json.load(open('$ROOT/exam/images.json'))['Haproxy'].rsplit(':',1)[1])")"
HAPROXY_IMAGE="$REGISTRY/exam-staging/haproxy:$HAPROXY_TAG"
PYTHON_IMAGE="$REGISTRY/exam-staging/python:3.12-slim"
# public.ecr.aws/docker/library is the official mirror of Docker Hub's official images
"$CRANE" copy --platform linux/amd64 "public.ecr.aws/docker/library/haproxy:$HAPROXY_TAG" "$HAPROXY_IMAGE" >/dev/null
"$CRANE" copy --platform linux/amd64 "public.ecr.aws/docker/library/python:3.12-slim" "$PYTHON_IMAGE" >/dev/null
echo "   haproxy:$HAPROXY_TAG digest in the account : $("$CRANE" digest "$HAPROXY_IMAGE")"
echo "   haproxy:$HAPROXY_TAG digest on Docker Hub  : $("$CRANE" digest --platform linux/amd64 "docker.io/library/haproxy:$HAPROXY_TAG" 2>/dev/null || echo 'not checked')"

# ---- 3. staging variant of the exam template and its parameters
echo "== [3/4] staging variant of exam/template.yaml"
python3 "$ROOT/staging/make_template.py" "$ROOT/exam/template.yaml" "$OUT/template.staging.yaml"
AMI="$(aws ssm get-parameter --name /aws/service/ecs/optimized-ami/amazon-linux-2023/recommended/image_id --query Parameter.Value --output text)"
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
    "InstanceProfileArn": "$(stack_output "$PLATFORM_STACK" InstanceProfileArn)",
    "ReferenceImageUri": "$PYTHON_IMAGE",
    "EcsGpuAmiId": "$AMI",
    "ImageHaproxy": "$HAPROXY_IMAGE",
    "Hosts": "$HOSTS",
}
json.dump([{"ParameterKey": k, "ParameterValue": v} for k, v in params.items()], open(sys.argv[1], "w"), indent=1)
PY

# ---- 4. the exam stack, with the exam's 20-minute creation limit
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
}

STATUS="$(stack_status "$APP_STACK")"
if [[ "$STATUS" == CREATE_COMPLETE || "$STATUS" == UPDATE_COMPLETE ]]; then
  echo "== [4/4] exam stack already deployed ($STATUS). Run staging/down.sh first to redeploy from scratch."
else
  if [[ "$STATUS" != ABSENT ]]; then
    echo "== removing the exam stack left in state $STATUS"
    aws cloudformation delete-stack --stack-name "$APP_STACK"
    aws cloudformation wait stack-delete-complete --stack-name "$APP_STACK"
  fi
  echo "== [4/4] creating the exam stack ($(utc) UTC, limit 20 minutes as in the exam)"
  aws cloudformation create-stack --stack-name "$APP_STACK" --template-body "file://$OUT/template.staging.yaml" \
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
echo "== next: bash staging/test.sh      and when finished: bash staging/down.sh"
