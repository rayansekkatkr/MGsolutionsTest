#!/usr/bin/env bash
# Deletes everything the staging scripts created.
# Usage: bash staging/down.sh        (about 10 minutes)   same STAGING_REGION as for up.sh
source "$(dirname "$0")/lib.sh"

delete_stack() {  # delete_stack STACK -> 0 when the stack is gone
  aws cloudformation delete-stack --stack-name "$1"
  aws cloudformation wait stack-delete-complete --stack-name "$1" 2>/dev/null || true
  [[ "$(stack_status "$1")" == ABSENT ]]
}

if [[ "$(stack_status "$APP_STACK")" != ABSENT ]]; then
  echo "== deleting $APP_STACK ($(utc) UTC)"
  if ! delete_stack "$APP_STACK"; then
    echo "   first attempt left these resources:"
    aws cloudformation describe-stack-events --stack-name "$APP_STACK" --output text \
      --query "StackEvents[?ResourceStatus=='DELETE_FAILED'].[LogicalResourceId,ResourceStatusReason]" | sort -u | head -10
    # A cluster cannot be deleted while container instances are still registered in it.
    for INSTANCE in $(aws ecs list-container-instances --cluster "$CLUSTER" --query 'containerInstanceArns' --output text 2>/dev/null || true); do
      aws ecs deregister-container-instance --cluster "$CLUSTER" --container-instance "$INSTANCE" --force >/dev/null || true
    done
    echo "   second attempt"
    delete_stack "$APP_STACK" || true
  fi
fi
echo "   $APP_STACK: $(stack_status "$APP_STACK")"

if [[ "$(stack_status "$PLATFORM_STACK")" != ABSENT ]]; then
  BUCKET="$(stack_output "$PLATFORM_STACK" BucketName)"
  if [[ -n "$BUCKET" && "$BUCKET" != None ]]; then aws s3 rm --quiet --recursive "s3://$BUCKET" || true; fi
  echo "== deleting $PLATFORM_STACK ($(utc) UTC)"
  delete_stack "$PLATFORM_STACK" || aws cloudformation describe-stack-events --stack-name "$PLATFORM_STACK" --output text \
    --query "StackEvents[?ResourceStatus=='DELETE_FAILED'].[LogicalResourceId,ResourceStatusReason]" | sort -u | head -10
fi
echo "   $PLATFORM_STACK: $(stack_status "$PLATFORM_STACK")"

# Container Insights creates this log group on its own; it is not part of either stack.
aws logs delete-log-group --log-group-name "/aws/ecs/containerinsights/$CLUSTER/performance" 2>/dev/null || true
LEFT="$(aws ec2 describe-instances --filters "Name=tag:aws:autoscaling:groupName,Values=$ASG" "Name=instance-state-name,Values=pending,running" \
  --query 'length(Reservations[].Instances[])' --output text)"
echo "== GPU hosts still running: $LEFT"
if [[ "$(stack_status "$APP_STACK")" == ABSENT && "$(stack_status "$PLATFORM_STACK")" == ABSENT ]]; then
  echo "== staging environment removed in $AWS_REGION"
else
  echo "== NOT fully removed. Send me the lines above; the hosts are what costs money."
  exit 1
fi
