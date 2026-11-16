#!/usr/bin/env bash
# Rehearses the load test against the staging deployment: each stage's share for the hosts
# deployed, one host terminated without notice, overload, then the quiet period and the shrink.
# Usage: bash staging/test.sh        (about 30 minutes)
source "$(dirname "$0")/lib.sh"

URL="$(stack_output "$APP_STACK" EndpointUrl)"
SUBNETS="$(stack_output "$PLATFORM_STACK" SubnetIds)"
CLIENT_SG="$(stack_output "$PLATFORM_STACK" ClientSecurityGroupId)"
CLIENT_TASK="$(stack_output "$PLATFORM_STACK" ClientTaskArn)"
BUCKET="$(stack_output "$PLATFORM_STACK" BucketName)"
BUNDLE_URL="$(aws s3 presign "s3://$BUCKET/bundle.tar.gz" --expires-in 7200)"
TG_ARN="$(aws elbv2 describe-target-groups --names "$TARGET_GROUP" --query 'TargetGroups[0].TargetGroupArn' --output text)"
HOSTS="$(aws ecs describe-services --cluster "$CLUSTER" --services "$SERVICE" --query 'services[0].desiredCount' --output text)"

healthy() {
  aws elbv2 describe-target-health --target-group-arn "$TG_ARN" \
    --query "length(TargetHealthDescriptions[?TargetHealth.State=='healthy'])" --output text
}
rate() { python3 -c "print(round($1 * $HOSTS, 2))"; }   # one host's share times the hosts deployed

start_client() {  # start_client ARGS... -> sets TASK
  python3 - "$OUT/overrides.json" "$URL" "$BUNDLE_URL" "$*" <<'PY'
import json, sys
env = [{"name": n, "value": v} for n, v in zip(("URL", "BUNDLE_URL", "ARGS"), sys.argv[2:5])]
json.dump({"containerOverrides": [{"name": "client", "environment": env}]}, open(sys.argv[1], "w"))
PY
  TASK="$(aws ecs run-task --cluster exam-staging-client --launch-type FARGATE --task-definition "$CLIENT_TASK" \
    --network-configuration "awsvpcConfiguration={subnets=[$SUBNETS],securityGroups=[$CLIENT_SG],assignPublicIp=DISABLED}" \
    --overrides "file://$OUT/overrides.json" --query 'tasks[0].taskArn' --output text)"
  if ! aws ecs wait tasks-running --cluster exam-staging-client --tasks "$TASK"; then
    echo "load client did not start:"
    aws ecs describe-tasks --cluster exam-staging-client --tasks "$TASK" --query 'tasks[0].[stoppedReason,containers[0].reason]' --output text
    exit 1
  fi
}
client_output() {
  aws ecs wait tasks-stopped --cluster exam-staging-client --tasks "$TASK"
  aws logs get-log-events --log-group-name /exam-staging/client --log-stream-name "client/client/${TASK##*/}" \
    --start-from-head --query 'events[].message' --output text | tr '\t' '\n'
}
fleet() {  # one line: where the fleet and its alarms stand
  local desired running idle idle_logs
  desired="$(aws autoscaling describe-auto-scaling-groups --auto-scaling-group-names "$ASG" --query 'AutoScalingGroups[0].DesiredCapacity' --output text)"
  running="$(aws ec2 describe-instances --filters "Name=tag:aws:autoscaling:groupName,Values=$ASG" "Name=instance-state-name,Values=pending,running" \
    --query 'length(Reservations[].Instances[])' --output text)"
  idle="$(aws cloudwatch describe-alarms --alarm-names "exam-r${RUN_ID}-fleet-idle" --query 'MetricAlarms[0].StateValue' --output text)"
  idle_logs="$(aws cloudwatch describe-alarms --alarm-names "exam-r${RUN_ID}-fleet-idle-from-logs" --query 'MetricAlarms[0].StateValue' --output text)"
  echo "utc=$(utc) group desired=$desired, instances running=$running, healthy targets=$(healthy), alarms: fleet-idle=$idle from-logs=$idle_logs"
}

TEST_START="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
IDLE_HOSTS="$(aws autoscaling describe-auto-scaling-groups --auto-scaling-group-names "$ASG" --query 'AutoScalingGroups[0].MinSize' --output text)"
echo "== endpoint $URL, $HOSTS hosts, $IDLE_HOSTS when idle"
echo "== $(fleet)"

echo "== how the inference servers started (one line per host; tensorrt means the engines built for this GPU model loaded):"
aws logs filter-log-events --log-group-name "/exam-run/r${RUN_ID}/inference" --filter-pattern '"ready"' \
  --query 'events[].message' --output text | tr '\t' '\n' | sed 's/^/   /'

echo
echo "== A. stage 1 and 2 shares, every answer expected"
start_client --rate "$(rate 0.21)" --seconds 60 --min-correct-share 1
client_output
start_client --rate "$(rate 2.1)" --seconds 60 --min-correct-share 1 --max-p99-ms 300
client_output

# The host kill is off by default. It was measured in the first rehearsal (19 requests lost in the
# nine seconds the load balancer needed to stop routing to the host, none afterwards) and in the
# round 2 measurement itself, with the same load balancer settings. It leaves the fleet waiting
# for a replacement host, and a pending launch makes AWS refuse the shrink that phase E checks.
if [[ "${STAGING_KILL:-0}" == 1 ]]; then
  echo
  echo "== B. stage 2 share for 180 s with one host terminated without notice after 45 s"
  start_client --rate "$(rate 2.1)" --seconds 180 --windows
  sleep 45
  VICTIM="$(aws autoscaling describe-auto-scaling-groups --auto-scaling-group-names "$ASG" --query 'AutoScalingGroups[0].Instances[0].InstanceId' --output text)"
  KILLED_AT=$SECONDS
  echo "KILL utc=$(utc) instance=$VICTIM"
  aws ec2 terminate-instances --instance-ids "$VICTIM" --query 'TerminatingInstances[0].CurrentState.Name' --output text
  for _ in $(seq 1 12); do sleep 10; echo "   utc=$(utc) healthy targets: $(healthy)/$HOSTS"; done
  client_output

  echo
  echo "== waiting for the replacement host (up to 12 minutes)"
  while (( SECONDS - KILLED_AT < 720 )); do
    H="$(healthy)"
    if [[ "$H" == "$HOSTS" ]]; then echo "   fleet back to $H/$HOSTS healthy targets $((SECONDS - KILLED_AT)) s after the kill"; break; fi
    echo "   utc=$(utc) healthy targets: $H/$HOSTS"
    sleep 20
  done
  if [[ "$(healthy)" != "$HOSTS" ]]; then
    echo "   THE REPLACEMENT HOST DID NOT ARRIVE. Last scaling activity:"
    aws autoscaling describe-scaling-activities --auto-scaling-group-name "$ASG" --max-items 1 \
      --query 'Activities[0].[StatusCode,StatusMessage]' --output text | cut -c1-300 | sed 's/^/   /'
    echo "   The rest of the test runs on the hosts that are left."
  fi

fi

echo
echo "== C. stage 3 share for 90 s: every answer expected, within the fleet's capacity"
start_client --rate "$(rate 21)" --seconds 90 --procs 2 --min-correct-share 0.99 --max-p99-ms 400
client_output

echo
echo "== D. overload for 150 s (750 requests/s in all): answers at capacity, fast, no timeouts"
start_client --rate 750 --seconds 150 --procs 4 --min-correct-per-s "$((25 * $(healthy)))" --max-p99-ms 400
client_output
OVERLOAD_END=$SECONDS
echo "   overload ended at about utc=$(utc)"
echo "   what the inference servers measured meanwhile:"
aws logs filter-log-events --log-group-name "/exam-run/r${RUN_ID}/inference" --filter-pattern '"last 30 s"' \
  --start-time "$(( ($(date +%s) - 240) * 1000 ))" --query 'events[].message' --output text | tr '\t' '\n' | tail -10 | sed 's/^/   /'

echo
echo "== E. quiet period (stage 1 share for 330 s): the fleet should shrink to its idle size within 3 minutes"
start_client --rate "$(rate 0.21)" --seconds 330 --windows
SHRUNK=""
for _ in $(seq 1 24); do
  LINE="$(fleet)"
  echo "   +$((SECONDS - OVERLOAD_END)) s after the overload: $LINE"
  if [[ -z "$SHRUNK" && "$LINE" == *"instances running=$IDLE_HOSTS,"* && "$LINE" == *"desired=$IDLE_HOSTS,"* ]]; then SHRUNK="$((SECONDS - OVERLOAD_END))"; fi
  sleep 6
done
client_output
if [[ -n "$SHRUNK" ]]; then
  echo "   FLEET AT ITS IDLE SIZE about $SHRUNK s after the overload (the exam scores the cost from 180 s on)"
else
  echo "   THE FLEET DID NOT SHRINK to $IDLE_HOSTS host(s)"
fi
echo "   scaling activities:"
aws autoscaling describe-scaling-activities --auto-scaling-group-name "$ASG" --max-items 8 \
  --query 'Activities[].[StartTime,StatusCode,Description]' --output text | sed 's/^/   /'
echo "   what the tasks logged about the fleet:"
aws logs filter-log-events --log-group-name "/exam-run/r${RUN_ID}/inference" --filter-pattern '"fleet"' --max-items 400 \
  --query 'events[].message' --output text | tr '\t' '\n' | grep -v "^fleet-idle 1$" | tail -20 | sed 's/^/   /'


echo
echo "== what AWS recorded during the test"
python3 "$ROOT/staging/evidence.py" --region "$AWS_REGION" --run-id "$RUN_ID" --from "$TEST_START" --to "$(date -u +%Y-%m-%dT%H:%M:%SZ)"

echo
echo "== done. Remove everything with: bash staging/down.sh"
