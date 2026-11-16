"""Where do the TensorRT configurations lose accuracy, and does a fallback recover it?

1. Person boxes of each detector variant against the exact ones (PyTorch fp32).
2. For each variant, how well the margin between the best candidate and its best rival predicts
   that the box differs: this is what a fallback to the exact detector would be triggered by.
3. End-to-end pass rate (task tolerance, against the reference results) for each pose variant
   with exact boxes, with the variant's boxes, and with the fallback at several margins.

Usage: python box_study.py --images DIR --reference FILE.jsonl --trt-dir DIR
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, "/app")
from engine import Engine, MIN_PERSON_SCORE  # noqa: E402

BATCH = 4


def boxes_of(engine, images):
    boxes, details = [], []
    for start in range(0, len(images), BATCH):
        boxes += engine.detect(images[start:start + BATCH], details)
    return boxes, details


def poses_of(engine, images, boxes):
    out = []
    for start in range(0, len(images), BATCH):
        out += engine.estimate(images[start:start + BATCH], boxes[start:start + BATCH])
    return out


def pass_rate(results, reference):
    ok = 0
    for (k2, k3), row in zip(results, reference):
        e2 = np.linalg.norm(k2 - np.asarray(row["keypoints_2d"]), axis=1).mean()
        e3 = 1000 * np.linalg.norm(k3 - np.asarray(row["keypoints_3d"]), axis=1).mean()
        ok += e2 <= 3.5 and e3 <= 25.0
    return 100 * ok / len(results)


def release(engine):
    del engine
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--trt-dir", required=True)
    args = parser.parse_args()

    paths = sorted(args.images.glob("*.jpg"))
    images = [cv2.imread(str(p), cv2.IMREAD_COLOR) for p in paths]
    by_file = {row["file"]: row for row in map(json.loads, args.reference.open())}
    reference = [by_file[p.name] for p in paths]
    margins = (0.02, 0.05, 0.1, 0.2)

    exact = Engine(precision="fp32")
    exact_boxes, exact_details = boxes_of(exact, images)
    print(f"exact pipeline (PyTorch fp32, direct box): {pass_rate(poses_of(exact, images, exact_boxes), reference):.1f}% within tolerance")
    release(exact)

    detectors = {}
    for name, kwargs in [("PyTorch fp16", dict(precision="fp16")),
                         ("TensorRT fp16, head fp32", dict(precision="fp16", trt_dir=args.trt_dir, trt_det="det_fp16_mixed", trt_pose="-")),
                         ("TensorRT fp16", dict(precision="fp16", trt_dir=args.trt_dir, trt_det="det_fp16", trt_pose="-")),
                         ("TensorRT fp32", dict(precision="fp16", trt_dir=args.trt_dir, trt_det="det_fp32", trt_pose="-"))]:
        engine = Engine(**kwargs)
        detectors[name] = boxes_of(engine, images)
        release(engine)

    print("\n--- 1. boxes against the exact ones: photos whose box differs by more than N pixels on any side")
    print(f"{'detector':28} {'>1 px':>7} {'>3 px':>7} {'>10 px':>7} {'>30 px':>7}   median diff px")
    differs = {}
    for name, (boxes, _) in detectors.items():
        diff = np.array([np.abs(a - b).max() for a, b in zip(boxes, exact_boxes)])
        differs[name] = diff
        print(f"{name:28} {(diff > 1).sum():>7} {(diff > 3).sum():>7} {(diff > 10).sum():>7} {(diff > 30).sum():>7}   {np.median(diff):.3f}")

    print("\n--- 2. fallback trigger: best score within MARGIN of a rival box's score, or of the 0.3 threshold")
    print(f"{'detector':28} {'margin':>7} {'photos flagged':>15} {'boxes off by >3 px not flagged':>32}")
    flagged = {}
    for name, (_, details) in detectors.items():
        best = np.array([d[0] for d in details])
        rival = np.array([d[1] for d in details])
        for margin in margins:
            flag = (best - rival < margin) | (np.abs(best - MIN_PERSON_SCORE) < margin)
            flagged[name, margin] = flag
            missed = ((differs[name] > 3) & ~flag).sum()
            print(f"{name:28} {margin:>7} {flag.sum():>10} ({100 * flag.mean():4.1f}%) {missed:>32}")

    print("\n--- 3. end to end, share of photos within tolerance")
    header = f"{'pose network':16} {'detector':28} {'exact boxes':>12} {'own boxes':>10}" + "".join(f"{'fallback ' + str(m):>15}" for m in margins)
    print(header)
    for pose_name, kwargs in [("TensorRT fp16", dict(precision="fp16", trt_dir=args.trt_dir, trt_det="-", trt_pose="pose_fp16")),
                              ("TensorRT fp32", dict(precision="fp16", trt_dir=args.trt_dir, trt_det="-", trt_pose="pose_fp32")),
                              ("PyTorch fp16", dict(precision="fp16"))]:
        engine = Engine(**kwargs)
        with_exact = pass_rate(poses_of(engine, images, exact_boxes), reference)
        for name, (boxes, _) in detectors.items():
            cells = [f"{pass_rate(poses_of(engine, images, boxes), reference):.1f}"]
            for margin in margins:
                mixed = [e if f else b for e, b, f in zip(exact_boxes, boxes, flagged[name, margin])]
                cells.append(f"{pass_rate(poses_of(engine, images, mixed), reference):.1f}")
            print(f"{pose_name:16} {name:28} {with_exact:>12.1f} {cells[0]:>10}" + "".join(f"{c:>15}" for c in cells[1:]), flush=True)
        release(engine)


if __name__ == "__main__":
    with torch.no_grad():
        main()
