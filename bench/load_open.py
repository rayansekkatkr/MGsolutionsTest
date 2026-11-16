"""Open-loop load with real photos: fixed send rate, never waits for answers, 10 s client timeout.

Same shape as the task's virtual users. Every 200 answer is checked: the request id must come back
and, when reference results are given, the joints must be within the task's tolerance of the
reference for that very photo (so a mix-up between requests would show as wrong answers).

Usage: python3 load_open.py --images DIR --url http://host:port --rate 100 --seconds 30 [--reference FILE.jsonl]
           [--windows] [--min-correct-share 0.99] [--min-correct-per-s 30] [--max-p99-ms 300] [--max-wrong-share 0.02]
Exits 1 when an expectation is not met, when a request times out, and when more answers are wrong
than allowed (none by default; fp16 results fall outside the tolerance for about 1 % of photos).
"""
import argparse
import asyncio
import json
import math
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

TOLERANCE_2D_PX = 3.5
TOLERANCE_3D_MM = 25.0


def mean_distance(a, b):
    return sum(math.dist(p, q) for p, q in zip(a, b)) / len(a)


def verdict(raw: bytes, request_id: str, reference):
    head, _, body = raw.partition(b"\r\n\r\n")
    status = head[9:12]
    if status != b"200":
        return "turned away" if status == b"503" else f"http {status.decode(errors='replace') or 'none'}"
    try:
        answer = json.loads(body)
        k2, k3 = answer["keypoints_2d"], answer["keypoints_3d"]
        if answer["request_id"] != request_id or len(k2) != 133 or len(k3) != 133:
            return "wrong"
        if reference is not None and (mean_distance(reference["keypoints_2d"], k2) > TOLERANCE_2D_PX
                                      or mean_distance(reference["keypoints_3d"], k3) * 1000 > TOLERANCE_3D_MM):
            return "wrong"
    except (ValueError, KeyError, TypeError):
        return "wrong"
    return "correct"


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--url", required=True)
    parser.add_argument("--rate", required=True, type=float)
    parser.add_argument("--seconds", required=True, type=float)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--windows", action="store_true", help="also print the share of correct answers per 10 s")
    parser.add_argument("--min-correct-share", type=float)
    parser.add_argument("--min-correct-per-s", type=float)
    parser.add_argument("--max-p99-ms", type=float)
    parser.add_argument("--max-wrong-share", type=float, default=0.0, help="of the answers given")
    args = parser.parse_args()

    target = urlparse(args.url)
    host, port = target.hostname, target.port or 80
    photos = [(p.name, p.read_bytes()) for p in sorted(args.images.glob("*.jpg"))]
    reference = {}
    if args.reference:
        reference = {row["file"]: row for row in map(json.loads, args.reference.open()) if "file" in row}
    results = []  # (send offset s, verdict, latency s)

    async def one(i: int, offset: float) -> None:
        name, body = photos[i % len(photos)]
        request_id = f"r{i}"
        started = time.perf_counter()

        async def exchange() -> bytes:
            reader, writer = await asyncio.open_connection(host, port)
            try:
                writer.write(f"POST /infer HTTP/1.1\r\nHost: {host}\r\nX-Request-Id: {request_id}\r\n"
                             f"Content-Type: image/jpeg\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode())
                writer.write(body)
                try:
                    await writer.drain()
                except OSError:
                    pass  # answered before the body was read
                return await reader.read()
            finally:
                writer.close()

        try:
            outcome = verdict(await asyncio.wait_for(exchange(), 10), request_id, reference.get(name))
        except asyncio.TimeoutError:
            outcome = "timeout"
        except OSError:
            outcome = "connection failed"
        results.append((offset, outcome, time.perf_counter() - started))

    started = time.perf_counter()
    total = int(args.rate * args.seconds)
    tasks = []
    for i in range(total):
        delay = started + i / args.rate - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        tasks.append(asyncio.create_task(one(i, time.perf_counter() - started)))
    sending_took = time.perf_counter() - started
    await asyncio.gather(*tasks)

    counts = {}
    for _, outcome, _ in results:
        counts[outcome] = counts.get(outcome, 0) + 1
    correct = sorted(latency for _, outcome, latency in results if outcome == "correct")
    q = lambda p: 1000 * correct[min(len(correct) - 1, int(p * len(correct)))] if correct else float("nan")
    others = "  ".join(f"{name}={n}" for name, n in sorted(counts.items()) if name != "correct")
    print(f"rate {args.rate:g}/s (sent at {total / sending_took:.1f}/s)  sent={total}  correct={len(correct)} "
          f"({100 * len(correct) / total:.1f}%, {len(correct) / args.seconds:.1f}/s)  "
          f"p50={q(.5):.0f}ms p95={q(.95):.0f}ms p99={q(.99):.0f}ms  {others}")
    if args.windows:
        for w in range(int(math.ceil(args.seconds / 10))):
            window = [outcome for offset, outcome, _ in results if 10 * w <= offset < 10 * (w + 1)]
            if window:
                print(f"  {10 * w:>4}-{10 * (w + 1):<4} s  {100 * window.count('correct') / len(window):6.2f}% correct of {len(window)}")


    failures = []
    wrong = counts.get("wrong", 0)
    if counts.get("timeout"):
        failures.append("timeouts")
    if wrong > args.max_wrong_share * (wrong + len(correct)):
        failures.append(f"{wrong} wrong answers")
    if args.min_correct_share is not None and len(correct) < args.min_correct_share * total:
        failures.append(f"correct share below {args.min_correct_share}")
    if args.min_correct_per_s is not None and len(correct) / args.seconds < args.min_correct_per_s:
        failures.append(f"fewer than {args.min_correct_per_s} correct answers per second")
    if args.max_p99_ms is not None and not q(.99) <= args.max_p99_ms:
        failures.append(f"p99 above {args.max_p99_ms} ms")
    for failure in failures:
        print("FAILED:", failure)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    asyncio.run(main())
