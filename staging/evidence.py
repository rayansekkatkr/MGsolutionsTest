"""What AWS itself recorded during a rehearsal window: load balancer metrics per minute, and what
the fleet alarms did. CloudWatch keeps both after the stacks are deleted, so this also works on a
past run.

Usage: python3 staging/evidence.py --from 2026-11-16T08:38:00Z --to 2026-11-16T08:52:00Z [--region us-east-1] [--run-id stg]
"""
import argparse
import json
import subprocess

METRICS = [  # (column, metric, statistic)
    ("requests", "RequestCount", "Sum"),
    ("new conn", "NewConnectionCount", "Sum"),
    ("tgt 2xx", "HTTPCode_Target_2XX_Count", "Sum"),
    ("tgt 4xx", "HTTPCode_Target_4XX_Count", "Sum"),
    ("tgt 5xx", "HTTPCode_Target_5XX_Count", "Sum"),
    ("lb 502", "HTTPCode_ELB_502_Count", "Sum"),
    ("lb 503", "HTTPCode_ELB_503_Count", "Sum"),
    ("lb 504", "HTTPCode_ELB_504_Count", "Sum"),
    ("tgt conn err", "TargetConnectionErrorCount", "Sum"),
    ("rejected", "RejectedConnectionCount", "Sum"),
    ("t p50 ms", "TargetResponseTime", "p50"),
    ("t p99 ms", "TargetResponseTime", "p99"),
    ("t p99.9 ms", "TargetResponseTime", "p99.9"),
    ("t max ms", "TargetResponseTime", "Maximum"),
]


def aws(region, *args):
    out = subprocess.run(["aws", "--region", region, "--output", "json", *args], capture_output=True, text=True)
    if out.returncode:
        print("   aws", " ".join(args[:2]), "failed:", out.stderr.strip()[:200])
        return {}
    return json.loads(out.stdout or "{}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="start", required=True)
    parser.add_argument("--to", dest="end", required=True)
    parser.add_argument("--region", default="ap-northeast-2")
    parser.add_argument("--run-id", default="stg")
    args = parser.parse_args()

    listed = aws(args.region, "cloudwatch", "list-metrics", "--namespace", "AWS/ApplicationELB", "--metric-name", "RequestCount")
    balancers = sorted({d["Value"] for m in listed.get("Metrics", []) for d in m["Dimensions"]
                        if d["Name"] == "LoadBalancer" and d["Value"].startswith(f"app/exam-r{args.run_id}-alb/")})
    for balancer in balancers:
        queries = [{"Id": f"m{i}", "ReturnData": True,
                    "MetricStat": {"Metric": {"Namespace": "AWS/ApplicationELB", "MetricName": metric,
                                              "Dimensions": [{"Name": "LoadBalancer", "Value": balancer}]},
                                   "Period": 60, "Stat": stat}}
                   for i, (_, metric, stat) in enumerate(METRICS)]
        data = aws(args.region, "cloudwatch", "get-metric-data", "--start-time", args.start, "--end-time", args.end,
                   "--metric-data-queries", json.dumps(queries))
        table = {}
        for result in data.get("MetricDataResults", []):
            column = int(result["Id"][1:])
            for stamp, value in zip(result["Timestamps"], result["Values"]):
                table.setdefault(stamp[11:16], {})[column] = value
        if not table:
            continue
        print(f"== load balancer {balancer}, per minute (UTC)")
        print("   " + "time".ljust(6) + "".join(name.rjust(13) for name, _, _ in METRICS))
        for minute in sorted(table):
            cells = []
            for i, (name, _, _) in enumerate(METRICS):
                value = table[minute].get(i)
                if value is None:
                    cells.append("-")
                elif name.endswith("ms"):
                    cells.append(f"{1000 * value:.0f}")
                else:
                    cells.append(f"{value:.0f}")
            print("   " + minute.ljust(6) + "".join(cell.rjust(13) for cell in cells))

    print("== fleet alarms: state changes and actions")
    for alarm in ("fleet-saturated", "fleet-idle", "fleet-idle-from-logs", "fleet-wake"):
        history = aws(args.region, "cloudwatch", "describe-alarm-history", "--alarm-name", f"exam-r{args.run_id}-{alarm}",
                      "--start-date", args.start, "--end-date", args.end, "--max-items", "40")
        for item in sorted(history.get("AlarmHistoryItems", []), key=lambda item: item["Timestamp"]):
            if item["HistoryItemType"] in ("StateUpdate", "Action"):
                print(f"   {item['Timestamp'][11:19]}  {alarm:22} {item['HistoryItemType']:12} {item['HistorySummary'][:150]}")

    print("== fleet metrics published by the tasks, per minute (UTC)")
    namespace = f"exam-r{args.run_id}-fleet"
    queries = [{"Id": f"f{i}", "ReturnData": True,
                "MetricStat": {"Metric": {"Namespace": namespace, "MetricName": metric}, "Period": 60, "Stat": stat}}
               for i, (metric, stat) in enumerate([("FleetSaturated", "Average"), ("FleetIdle", "Average"),
                                                   ("FleetIdle", "SampleCount"), ("FleetIdleLines", "Sum")])]
    data = aws(args.region, "cloudwatch", "get-metric-data", "--start-time", args.start, "--end-time", args.end,
               "--metric-data-queries", json.dumps(queries))
    table = {}
    for result in data.get("MetricDataResults", []):
        for stamp, value in zip(result["Timestamps"], result["Values"]):
            table.setdefault(stamp[11:16], {})[result["Id"]] = value
    print("   time   saturated(avg)  idle(avg)  idle samples  idle log lines")
    for minute in sorted(table):
        row = table[minute]
        cell = lambda key, fmt: format(row[key], fmt) if key in row else "-"
        print(f"   {minute}  {cell('f0', '.2f'):>13}  {cell('f1', '.2f'):>9}  {cell('f2', '.0f'):>12}  {cell('f3', '.0f'):>14}")


if __name__ == "__main__":
    main()
