"""Open-loop load generator: fixed send rate, never waits for answers, 10 s client timeout.

Same shape as the exam's virtual users. Exits non-zero when an expectation is not met.
Usage: python load.py PORT RATE SECONDS [--all-ok] [--no-timeouts] [--max-p95-ms N] [--min-ok-per-s N]
"""
import argparse
import asyncio
import os
import sys
import time

p = argparse.ArgumentParser()
p.add_argument("port", type=int)
p.add_argument("rate", type=float)
p.add_argument("seconds", type=float)
p.add_argument("--all-ok", action="store_true")
p.add_argument("--no-timeouts", action="store_true")
p.add_argument("--max-p95-ms", type=float)
p.add_argument("--min-ok-per-s", type=float)
a = p.parse_args()

BODY = os.urandom(157_000)  # average photo size in the exam
results = []


async def one(i):
    t0 = time.perf_counter()
    rid = f"r{i}"

    async def go():
        reader, writer = await asyncio.open_connection("127.0.0.1", a.port)
        try:
            writer.write(f"POST /infer HTTP/1.1\r\nHost: x\r\nX-Request-Id: {rid}\r\n"
                         f"Content-Length: {len(BODY)}\r\nConnection: close\r\n\r\n".encode())
            writer.write(BODY)
            try:
                await writer.drain()
            except OSError:
                pass  # rejected before the body was read
            data = await reader.read()
        finally:
            writer.close()
        ok = data[:12] in (b"HTTP/1.1 200", b"HTTP/1.0 200") and rid.encode() in data
        return "ok" if ok else "rejected"

    try:
        status = await asyncio.wait_for(go(), 10)
    except asyncio.TimeoutError:
        status = "timeout"
    except OSError:
        status = "rejected"
    results.append((status, time.perf_counter() - t0))


async def main():
    t0 = time.perf_counter()
    n = int(a.rate * a.seconds)
    tasks = []
    for i in range(n):
        delay = t0 + i / a.rate - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        tasks.append(asyncio.create_task(one(i)))
    await asyncio.gather(*tasks)
    return n


n = asyncio.run(main())
ok = sorted(lat for status, lat in results if status == "ok")
count = {k: sum(1 for status, _ in results if status == k) for k in ("ok", "rejected", "timeout")}
pct = lambda q: ok[min(len(ok) - 1, int(q * len(ok)))] * 1000 if ok else float("nan")
ok_per_s = count["ok"] / a.seconds
print(f"rate={a.rate:>6.1f}/s sent={n:>5} ok={count['ok']:>5} ({100 * count['ok'] / n:5.1f}%, {ok_per_s:5.1f}/s) "
      f"rejected={count['rejected']:>5} timeout={count['timeout']:>5} p50={pct(.5):.0f}ms p95={pct(.95):.0f}ms p99={pct(.99):.0f}ms")

failed = []
if a.all_ok and count["ok"] != n:
    failed.append("expected every request to succeed")
if a.no_timeouts and count["timeout"]:
    failed.append("expected no timeouts")
if a.max_p95_ms is not None and not pct(.95) <= a.max_p95_ms:
    failed.append(f"expected p95 <= {a.max_p95_ms:.0f} ms")
if a.min_ok_per_s is not None and ok_per_s < a.min_ok_per_s:
    failed.append(f"expected at least {a.min_ok_per_s} successful answers per second")
for f in failed:
    print("FAILED:", f)
sys.exit(1 if failed else 0)
