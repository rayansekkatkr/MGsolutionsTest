"""Closed-loop load: N workers each send one photo after another for a fixed time.

Measures what the server can sustain end to end (HTTP, decoding, batching, both models).
Usage: python3 load_closed.py --images DIR --url http://host:8080 --workers 32 --seconds 30
"""
import argparse
import itertools
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--url", required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--seconds", type=float, default=30)
    args = parser.parse_args()

    bodies = [p.read_bytes() for p in sorted(args.images.glob("*.jpg"))]
    latencies, errors, lock = [], [0], threading.Lock()
    stop_at = time.monotonic() + args.seconds

    def worker(offset: int) -> None:
        for i in itertools.count(offset):
            if time.monotonic() >= stop_at:
                return
            request = urllib.request.Request(args.url + "/infer", data=bodies[i % len(bodies)], method="POST",
                                             headers={"X-Request-Id": f"w{offset}-{i}"})
            started = time.monotonic()
            try:
                with urllib.request.urlopen(request, timeout=15) as response:
                    response.read()
                with lock:
                    latencies.append(time.monotonic() - started)
            except (urllib.error.URLError, OSError):
                with lock:
                    errors[0] += 1

    threads = [threading.Thread(target=worker, args=(n * 7,)) for n in range(args.workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    latencies.sort()
    q = lambda p: 1000 * latencies[min(len(latencies) - 1, int(p * len(latencies)))] if latencies else float("nan")
    print(f"workers={args.workers:>3} ok={len(latencies):>5} errors={errors[0]:>4} {len(latencies) / args.seconds:6.1f} req/s "
          f"p50={q(.5):.0f}ms p95={q(.95):.0f}ms p99={q(.99):.0f}ms")


if __name__ == "__main__":
    main()
