# Shared settings for the staging scripts. Source this file; do not run it.
set -euo pipefail
export AWS_REGION="${STAGING_REGION:-ap-northeast-2}"
export AWS_DEFAULT_REGION="$AWS_REGION"
export AWS_PAGER=""

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/staging/out"
mkdir -p "$OUT"

PLATFORM_STACK=exam-staging-platform
APP_STACK=exam-staging-app
RUN_ID=stg
CLUSTER="exam-r${RUN_ID}-cluster"
SERVICE="exam-r${RUN_ID}-inference"
ASG="exam-r${RUN_ID}-gpu"
TARGET_GROUP="exam-r${RUN_ID}-tg"

stack_output() {  # stack_output STACK KEY
  aws cloudformation describe-stacks --stack-name "$1" \
    --query "Stacks[0].Outputs[?OutputKey=='$2'].OutputValue" --output text
}

stack_status() {  # prints ABSENT when the stack does not exist
  aws cloudformation describe-stacks --stack-name "$1" --query 'Stacks[0].StackStatus' --output text 2>/dev/null || echo ABSENT
}

utc() { date -u +%H:%M:%S; }
