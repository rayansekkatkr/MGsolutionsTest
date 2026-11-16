#!/usr/bin/env bash
# Starts one GPU instance in the default VPC and registers it as a GitHub Actions runner (label gpu-l4).
# Usage: bash gpu/up.sh            needs the AWS CLI and either the gh CLI or RUNNER_TOKEN
# The instance is a g6.2xlarge when the quota allows it: same GPU as the exam's g6.xlarge, with
# four more vCPUs for the load generator, so that it does not take CPU time from the server under
# test. GPU_TYPE overrides the choice.
# Cost: that instance on demand while it is up. Stop it with: bash gpu/down.sh
set -euo pipefail
export AWS_REGION="${GPU_REGION:-ap-northeast-2}" AWS_DEFAULT_REGION="${GPU_REGION:-ap-northeast-2}" AWS_PAGER=""
REPO="${GPU_REPO:-rayansekkatkr/MGsolutionsTest}"
STACK=exam-gpu-runner
HERE="$(cd "$(dirname "$0")" && pwd)"

QUOTA="$(aws service-quotas get-service-quota --service-code ec2 --quota-code L-DB2E81BA --query Quota.Value --output text)"
echo "== region $AWS_REGION, GPU vCPU quota: $QUOTA"
if [[ "${QUOTA%.*}" -lt 4 ]]; then echo "The quota is not effective yet (4 vCPUs needed). Try again in a few minutes."; exit 1; fi
if [[ "${QUOTA%.*}" -ge 8 ]]; then TYPE="${GPU_TYPE:-g6.2xlarge}"; else TYPE="${GPU_TYPE:-g6.xlarge}"; fi

TOKEN="${RUNNER_TOKEN:-}"
if [[ -z "$TOKEN" ]]; then
  TOKEN="$(gh api -X POST "repos/$REPO/actions/runners/registration-token" --jq .token)"
fi

AMI="$(aws ssm get-parameter --name /aws/service/ecs/optimized-ami/amazon-linux-2023/gpu/recommended/image_id --query Parameter.Value --output text 2>/dev/null \
  || aws ssm get-parameter --name /aws/service/ecs/optimized-ami/amazon-linux-2/gpu/recommended/image_id --query Parameter.Value --output text)"
VPC="$(aws ec2 describe-vpcs --filters Name=is-default,Values=true --query 'Vpcs[0].VpcId' --output text)"
if [[ "$VPC" == None || -z "$VPC" ]]; then echo "No default VPC in $AWS_REGION. Create one with: aws ec2 create-default-vpc"; exit 1; fi
ZONES="$(aws ec2 describe-instance-type-offerings --location-type availability-zone \
  --filters "Name=instance-type,Values=$TYPE" --query 'InstanceTypeOfferings[].Location' --output text)"
echo "== AMI $AMI, zones offering $TYPE: $ZONES"

for ZONE in $ZONES; do
  SUBNET="$(aws ec2 describe-subnets --filters "Name=vpc-id,Values=$VPC" "Name=availability-zone,Values=$ZONE" \
    "Name=default-for-az,Values=true" --query 'Subnets[0].SubnetId' --output text)"
  [[ "$SUBNET" == None || -z "$SUBNET" ]] && continue
  echo "== trying $ZONE ($SUBNET)"
  if aws cloudformation deploy --stack-name "$STACK" --template-file "$HERE/runner.yaml" --capabilities CAPABILITY_IAM \
      --no-fail-on-empty-changeset --parameter-overrides "RunnerToken=$TOKEN" "RepoUrl=https://github.com/$REPO" \
      "AmiId=$AMI" "SubnetId=$SUBNET" "VpcId=$VPC" "MaxHours=${GPU_MAX_HOURS:-8}" "InstanceType=$TYPE"; then
    INSTANCE="$(aws cloudformation describe-stacks --stack-name "$STACK" --query "Stacks[0].Outputs[?OutputKey=='InstanceId'].OutputValue" --output text)"
    echo "== instance $INSTANCE ($TYPE) started in $ZONE. It installs the runner now (about 3 minutes)."
    echo "== it powers itself off after ${GPU_MAX_HOURS:-8} hours. To stop it earlier: bash gpu/down.sh"
    exit 0
  fi
  echo "== $ZONE failed:"
  aws cloudformation describe-stack-events --stack-name "$STACK" --output text \
    --query "StackEvents[?contains(ResourceStatus,'FAILED')].[LogicalResourceId,ResourceStatusReason]" | head -5 || true
  aws cloudformation delete-stack --stack-name "$STACK"
  aws cloudformation wait stack-delete-complete --stack-name "$STACK"
done
echo "== could not start a $TYPE in any zone of $AWS_REGION"
exit 1
