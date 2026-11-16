#!/usr/bin/env bash
# Deletes everything the staging scripts created.
# Usage: bash staging/down.sh        (about 10 minutes)
source "$(dirname "$0")/lib.sh"

for STACK in "$APP_STACK" "$PLATFORM_STACK"; do
  if [[ "$(stack_status "$STACK")" != ABSENT ]]; then
    echo "== deleting $STACK ($(utc) UTC)"
    aws cloudformation delete-stack --stack-name "$STACK"
    aws cloudformation wait stack-delete-complete --stack-name "$STACK"
  fi
  echo "   $STACK: $(stack_status "$STACK")"
done

# Container Insights creates this log group on its own; it is not part of either stack.
aws logs delete-log-group --log-group-name "/aws/ecs/containerinsights/$CLUSTER/performance" 2>/dev/null || true
echo "== staging environment removed"
