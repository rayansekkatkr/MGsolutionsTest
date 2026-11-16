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

echo "== endpoint $URL, $HOSTS hosts"
echo "== $(fleet)"

echo
echo "== A. stage 1 and 2 shares, every answer expected"
start_client --rate "$(rate 0.21)" --seconds 60 --min-correct-share 1
client_output
start_client --rate "$(rate 2.1)" --seconds 60 --min-correct-share 1
client_output

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

echo
echo "== C. stage 3 share for 90 s: every answer expected, within the fleet's capacity"
start_client --rate "$(rate 21)" --seconds 90 --procs 2
client_output

echo
echo "== D. overload for 150 s (750 requests/s in all): answers at capacity, fast, no timeouts"
start_client --rate 750 --seconds 150 --procs 4
client_output
OVERLOAD_END=$SECONDS
echo "   overload ended at about utc=$(utc)"

echo
echo "== E. quiet period (stage 1 share for 330 s): the fleet should shrink to its idle size within 3 minutes"
start_client --rate "$(rate 0.21)" --seconds 330 --windows
for _ in $(seq 1 24); do echo "   +$((SECONDS - OVERLOAD_END)) s after the overload: $(fleet)"; sleep 6; done
client_output
echo "   alarm history:"
for ALARM in "exam-r${RUN_ID}-fleet-saturated" "exam-r${RUN_ID}-fleet-idle" "exam-r${RUN_ID}-fleet-idle-from-logs"; do
  aws cloudwatch describe-alarm-history --alarm-name "$ALARM" --history-item-type StateUpdate --max-items 6 \
    --query 'AlarmHistoryItems[].[Timestamp,HistorySummary]' --output text | sed "s/^/   $ALARM  /"
done
echo "   scaling activities:"
aws autoscaling describe-scaling-activities --auto-scaling-group-name "$ASG" --max-items 8 \
  --query 'Activities[].[StartTime,StatusCode,Description]' --output text | sed 's/^/   /'
echo "   what the tasks logged about the fleet:"
aws logs filter-log-events --log-group-name "/exam-run/r${RUN_ID}/inference" --filter-pattern '"fleet"' --max-items 400 \
  --query 'events[].message' --output text | tr '\t' '\n' | grep -v "^fleet-idle 1$" | tail -20 | sed 's/^/   /'

echo
echo "== done. Remove everything with: bash staging/down.sh"
