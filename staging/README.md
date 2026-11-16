# Staging rehearsal

Deploys the submitted template into my own AWS account before spending a measurement attempt.
The exam counts a failed stack creation as a used attempt, so the template is created for real
here first, then exercised with the scenario that decided round 1: stage 2 traffic with one
host terminated without notice.

## What it reproduces

| Exam environment | Staging equivalent |
| :--- | :--- |
| VPC without internet, AWS reachable through endpoints only | Same: private subnets, no internet gateway or NAT, six interface endpoints and the S3 gateway endpoint |
| Roles, security groups and subnets passed as parameters | Created by `platform.yaml` and passed the same way |
| Images copied into a private registry | Copied into ECR with `crane` |
| Virtual users inside the VPC | A Fargate task running an open-loop load generator with a 10 s timeout |
| 20-minute limit on stack creation | Enforced by `up.sh` |
| One EC2 host terminated during the test | `test.sh` terminates one host 45 s into the load |

## What differs

`make_template.py` derives the deployed template from `exam/template.yaml` with six mechanical
edits, each of which must match exactly once:

- `g6.xlarge` becomes `t3.large`, the GPU requirement and the agent's GPU flag are removed, and
  the task's CPU and memory are sized for the smaller host. My account has no GPU quota.
- The inference container runs a stand-in server (one thread per request, about 120 ms per
  request) because the reference image cannot be pulled outside the exam.

So this rehearsal validates the infrastructure and the admission proxy. It does not measure the
real model's speed, nor the load balancer at 6,000 requests per second.

## Running it

From AWS CloudShell or any shell with the AWS CLI, Python 3 and curl:

```bash
bash staging/up.sh      # 10 to 15 minutes: platform, images, exam stack
bash staging/test.sh    # about 8 minutes: host kill at stage 2 rate, then overload
bash staging/down.sh    # about 10 minutes: deletes everything
```

Region defaults to ap-northeast-2 (`STAGING_REGION` overrides it). Cost while it is up is about
0.6 USD per hour: three `t3.large` hosts, one load balancer and the interface endpoints.
