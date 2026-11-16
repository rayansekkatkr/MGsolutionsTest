"""Sends every photo of a folder to the server, several at a time so that requests get batched.

Usage: python client.py --images DIR --url http://host:8080 --out FILE.jsonl [--concurrency 8]
"""
import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--url", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()

    def send(path: Path) -> dict:
        request = urllib.request.Request(args.url + "/infer", data=path.read_bytes(), method="POST",
                                         headers={"X-Request-Id": path.name, "Content-Type": "image/jpeg"})
        started = time.monotonic()
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            answer = json.loads(response.read())
        assert answer["request_id"] == path.name, "request_id was not echoed"
        return {"file": path.name, "seconds": round(time.monotonic() - started, 3),
                "keypoints_2d": answer["keypoints_2d"], "keypoints_3d": answer["keypoints_3d"]}

    paths = sorted(args.images.glob("*.jpg"))
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        rows = list(pool.map(send, paths))
    elapsed = time.monotonic() - started
    args.out.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows))
    print(f"server: {len(rows)} photos in {elapsed:.1f} s ({len(rows) / elapsed:.2f} per second, concurrency {args.concurrency})")


if __name__ == "__main__":
    main()
