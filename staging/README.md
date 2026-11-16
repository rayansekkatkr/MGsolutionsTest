# Staging rehearsal

Deploys the submitted template into my own AWS account before spending a measurement attempt.
The exam counts a failed stack creation as a used attempt and the last attempt cannot be
repeated, so the template is created for real here first, on the exam's GPU instance type, and
taken through the events that decide the score: a host terminated without notice, overload, and
the quiet period after the peak.

## What it reproduces

| Exam environment | Staging equivalent |
| :--- | :--- |
| VPC without internet, AWS reachable through endpoints only | Same: private subnets, no internet gateway or NAT, interface endpoints and the S3 gateway endpoint |
| Roles, security groups and subnets passed as parameters | Created by `platform.yaml` and passed the same way |
| Images copied into a private registry | Copied registry to registry with `crane`, from the addresses in `exam/images.json` |
| `g6.xlarge` GPU hosts from the ECS GPU AMI | Same |
| Virtual users inside the VPC | A Fargate task running `bench/load_open.py`: open loop, real photos, 10 s timeout |
| 20-minute limit on stack creation | Enforced by `up.sh` |
| One EC2 host terminated during the test | `test.sh` terminates one host 45 s into a 20 requests/s load and expects every answer (`STAGING_KILL=0` skips it) |

The deployed template is `exam/template.yaml` itself, not a variant. Two parameters differ from
the exam: `Hosts` is 2, because my GPU quota is 8 vCPUs, and `EdgeTasks` is 2 to keep one edge
proxy per host. The load is scaled accordingly, so each host receives the same share as one of
the exam's eight. With two hosts a host kill removes half of the fleet instead of an eighth,
which makes the kill phase harder than the exam's.

## What it does not show

- The load balancers and the fleet at 6,000 requests per second. The overload phase sends 750
  requests per second in all, which is more than seven times what two hosts can answer. Each
  edge proxy then receives half of what it would at the exam's peak; `evidence.py` prints their
  CPU and memory per minute, to be read with that factor in mind.
- Whether the exam's VPC has a CloudWatch metrics endpoint. Staging has one. The idle signal also
  travels through the log group, and `test.sh` prints when each of the two alarms fired.

## Running it

From a shell with the AWS CLI, Python 3 and curl:

```bash
bash staging/up.sh      # about 20 minutes: platform, images, client bundle, exam stack
bash staging/test.sh    # about 30 minutes
bash staging/down.sh    # about 10 minutes: deletes everything
```

Region defaults to ap-northeast-2; `STAGING_REGION` overrides it and must be the same for the
three commands. Two GPU hosts must fit in the region's quota next to whatever GPU instance is
already running there. Cost while it is up is about 2.7 USD per hour: two `g6.xlarge`, two load
balancers, two edge proxies and the interface endpoints.
