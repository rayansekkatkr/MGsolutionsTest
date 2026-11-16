"""Throughput of the batched engine on a GPU, per precision and batch size.

Runs inside the inference image. Writes one result file per configuration so that accuracy can be
compared with the reference afterwards.
Usage: python gpu_bench.py --images DIR --out DIR [--batches 1,2,4,8,16] [--precisions fp32,fp16]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, "/app")
from engine import Engine  # noqa: E402


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run(engine: Engine, images: list, batch: int):
    detect_s = estimate_s = 0.0
    results = []
    for start in range(0, len(images), batch):
        chunk = images[start:start + batch]
        t0 = time.perf_counter()
        boxes = engine.detect(chunk)
        sync()
        t1 = time.perf_counter()
        results += engine.estimate(chunk, boxes)
        sync()
        t2 = time.perf_counter()
        detect_s += t1 - t0
        estimate_s += t2 - t1
    return results, detect_s, estimate_s


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--batches", default="1,2,4,8,16")
    parser.add_argument("--precisions", default="fp32,fp16")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    paths = sorted(args.images.glob("*.jpg"))
    images = [cv2.imread(str(p), cv2.IMREAD_COLOR) for p in paths]
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (CPU)"
    print(f"photos: {len(images)}  torch {torch.__version__}  GPU: {gpu}", flush=True)
    print(f"{'precision':9} {'batch':>5} {'photos/s':>9} {'detect ms':>10} {'pose ms':>8}", flush=True)

    for precision in args.precisions.split(","):
        engine = Engine(precision=precision)
        for batch in (int(b) for b in args.batches.split(",")):
            run(engine, images[:2 * batch], batch)  # warm-up for this batch size
            results, detect_s, estimate_s = run(engine, images, batch)
            total = detect_s + estimate_s
            print(f"{precision:9} {batch:>5} {len(images) / total:>9.1f} {1000 * detect_s / len(images):>10.1f} "
                  f"{1000 * estimate_s / len(images):>8.1f}", flush=True)
            with (args.out / f"{precision}_b{batch}.jsonl").open("w") as out:
                for path, (keypoints_2d, keypoints_3d) in zip(paths, results):
                    out.write(json.dumps({"file": path.name, "keypoints_2d": np.round(keypoints_2d, 2).tolist(),
                                          "keypoints_3d": np.round(keypoints_3d, 5).tolist()}, separators=(",", ":")) + "\n")
        del engine
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
