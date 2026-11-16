#!/usr/bin/env bash
# Rehearses the load test against the staging deployment.
# Usage: bash staging/test.sh        (about 8 minutes)
source "$(dirname "$0")/lib.sh"

URL="$(stack_output "$APP_STACK" EndpointUrl)"
SUBNETS="$(stack_output "$PLATFORM_STACK" SubnetIds)"
CLIENT_SG="$(stack_output "$PLATFORM_STACK" ClientSecurityGroupId)"
CLIENT_TASK="$(stack_output "$PLATFORM_STACK" ClientTaskArn)"
TG_ARN="$(aws elbv2 describe-target-groups --names "$TARGET_GROUP" --query 'TargetGroups[0].TargetGroupArn' --output text)"
HOSTS="$(aws ecs describe-services --cluster "$CLUSTER" --services "$SERVICE" --query 'services[0].desiredCount' --output text)"

healthy() {
  aws elbv2 describe-target-health --target-group-arn "$TG_ARN" \
    --query "length(TargetHealthDescriptions[?TargetHealth.State=='healthy'])" --output text
}

start_client() {  # start_client RATE DURATION -> sets TASK
  TASK="$(aws ecs run-task --cluster exam-staging-client --launch-type FARGATE --task-definition "$CLIENT_TASK" \
    --network-configuration "awsvpcConfiguration={subnets=[$SUBNETS],securityGroups=[$CLIENT_SG],assignPublicIp=DISABLED}" \
    --overrides "{\"containerOverrides\":[{\"name\":\"client\",\"environment\":[{\"name\":\"URL\",\"value\":\"$URL\"},{\"name\":\"RATE\",\"value\":\"$1\"},{\"name\":\"SECONDS\",\"value\":\"$2\"}]}]}" \
    --query 'tasks[0].taskArn' --output text)"
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

echo "== endpoint $URL"
echo "== healthy targets before the test: $(healthy)/$HOSTS"

echo
echo "== A. stage 2 rate (16.7 requests/s for 180 s) with one host terminated without notice"
start_client 16.7 180
sleep 45
VICTIM="$(aws autoscaling describe-auto-scaling-groups --auto-scaling-group-names "$ASG" --query 'AutoScalingGroups[0].Instances[0].InstanceId' --output text)"
KILLED_AT=$SECONDS
echo "KILL utc=$(utc) instance=$VICTIM"
aws ec2 terminate-instances --instance-ids "$VICTIM" --query 'TerminatingInstances[0].CurrentState.Name' --output text
for _ in $(seq 1 12); do sleep 10; echo "   utc=$(utc) healthy targets: $(healthy)/$HOSTS"; done
client_output

echo
echo "== waiting for the replacement host (up to 10 minutes)"
while (( SECONDS - KILLED_AT < 600 )); do
  H="$(healthy)"
  if [[ "$H" == "$HOSTS" ]]; then echo "   fleet back to $H/$HOSTS healthy targets $((SECONDS - KILLED_AT)) s after the kill"; break; fi
  echo "   utc=$(utc) healthy targets: $H/$HOSTS"
  sleep 20
done

echo
echo "== B. overload (150 requests/s for 60 s): the fleet must keep answering at capacity, without timeouts"
start_client 150 60
client_output

echo
echo "== healthy targets after the test: $(healthy)/$HOSTS"
echo "== done. Remove everything with: bash staging/down.sh"
