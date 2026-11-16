#!/usr/bin/env bash
# Terminates the GPU runner and removes its registration.
# Usage: bash gpu/down.sh
set -euo pipefail
export AWS_REGION="${GPU_REGION:-ap-northeast-2}" AWS_DEFAULT_REGION="${GPU_REGION:-ap-northeast-2}" AWS_PAGER=""
REPO="${GPU_REPO:-rayansekkatkr/MGsolutionsTest}"
STACK=exam-gpu-runner

if aws cloudformation describe-stacks --stack-name "$STACK" >/dev/null 2>&1; then
  echo "== deleting $STACK"
  aws cloudformation delete-stack --stack-name "$STACK"
  aws cloudformation wait stack-delete-complete --stack-name "$STACK"
fi
echo "== stack: $(aws cloudformation describe-stacks --stack-name "$STACK" --query 'Stacks[0].StackStatus' --output text 2>/dev/null || echo ABSENT)"
if command -v gh >/dev/null; then
  for ID in $(gh api "repos/$REPO/actions/runners" --jq '.runners[] | select(.name=="gpu-l4") | .id' 2>/dev/null || true); do
    gh api -X DELETE "repos/$REPO/actions/runners/$ID" && echo "== runner registration $ID removed"
  done
fi
echo "== GPU runner removed"
