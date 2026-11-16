"""Where a batch's time goes: in the two networks on the GPU, or on the CPU around them.

Usage (inside the image with engines): python phase_probe.py --images DIR [--trt-dir /models/trt] [--batch 3] [--limit 150]
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import torch

sys.path.insert(0, "/app")
import engine as engine_module  # noqa: E402
from engine import Engine  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--trt-dir", default="/models/trt")
    parser.add_argument("--batch", type=int, default=3)
    parser.add_argument("--limit", type=int, default=150)
    args = parser.parse_args()

    engine = Engine(precision="fp16", trt_dir=args.trt_dir)
    spent = {"gpu": 0.0}
    original = engine_module.TrtNet.__call__

    def timed(self, x):
        torch.cuda.synchronize()
        start = time.perf_counter()
        out = original(self, x)
        torch.cuda.synchronize()
        spent["gpu"] += time.perf_counter() - start
        return out

    engine_module.TrtNet.__call__ = timed
    photos = [cv2.imread(str(p)) for p in sorted(args.images.glob("*.jpg"))[:args.limit]]
    batches = [photos[i:i + args.batch] for i in range(0, len(photos), args.batch)]
    for batch in batches[:5]:
        engine.infer(batch)
    spent["gpu"] = 0.0
    cpu_start, start = time.process_time(), time.perf_counter()
    for batch in batches:
        engine.infer(batch)
    torch.cuda.synchronize()
    wall, cpu, n = time.perf_counter() - start, time.process_time() - cpu_start, len(batches)
    print(f"threads: torch {torch.get_num_threads()}, cv2 {cv2.getNumThreads()} | per batch of {args.batch}: "
          f"{wall / n * 1000:.1f} ms, of which the two networks on the GPU {spent['gpu'] / n * 1000:.1f} ms "
          f"and everything else {(wall - spent['gpu']) / n * 1000:.1f} ms | CPU time used per batch {cpu / n * 1000:.1f} ms")


if __name__ == "__main__":
    main()
