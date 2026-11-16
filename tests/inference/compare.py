"""Compares server results with reference results using the task's own rule.

A photo is correct when the mean distance over the 133 joints is at most 3.5 px in 2D and 25 mm in 3D.
Usage: python compare.py REFERENCE.jsonl CANDIDATE.jsonl [--min-pass 1.0] [--max-2d PX] [--max-3d MM]
"""
import argparse
import json
import math
import sys

TOLERANCE_2D_PX = 3.5
TOLERANCE_3D_MM = 25.0


def load(path: str) -> dict:
    return {row["file"]: row for row in map(json.loads, open(path))}


def mean_distance(a: list, b: list) -> float:
    return sum(math.dist(p, q) for p, q in zip(a, b)) / len(a)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("candidate")
    parser.add_argument("--min-pass", type=float, default=1.0, help="required share of photos within tolerance")
    parser.add_argument("--max-2d", type=float, help="fail if any photo's 2D error exceeds this many pixels")
    parser.add_argument("--max-3d", type=float, help="fail if any photo's 3D error exceeds this many millimetres")
    args = parser.parse_args()

    reference, candidate = load(args.reference), load(args.candidate)
    missing = sorted(set(reference) - set(candidate))
    errors = []
    for name, ref in reference.items():
        if name not in candidate:
            continue
        cand = candidate[name]
        assert len(cand["keypoints_2d"]) == 133 and len(cand["keypoints_3d"]) == 133, f"{name}: not 133 joints"
        errors.append((name, mean_distance(ref["keypoints_2d"], cand["keypoints_2d"]),
                       mean_distance(ref["keypoints_3d"], cand["keypoints_3d"]) * 1000))
    passed = sum(1 for _, e2, e3 in errors if e2 <= TOLERANCE_2D_PX and e3 <= TOLERANCE_3D_MM)
    worst_2d = max(errors, key=lambda e: e[1])
    worst_3d = max(errors, key=lambda e: e[2])
    share = passed / len(reference)
    print(f"photos: {len(reference)}  answered: {len(errors)}  within tolerance: {passed} ({100 * share:.1f}%)")
    print(f"2D error px: mean {sum(e[1] for e in errors) / len(errors):.4f}  max {worst_2d[1]:.4f} ({worst_2d[0]})")
    print(f"3D error mm: mean {sum(e[2] for e in errors) / len(errors):.4f}  max {worst_3d[2]:.4f} ({worst_3d[0]})")

    failures = []
    if missing:
        failures.append(f"{len(missing)} photos without an answer: {missing[:5]}")
    if share < args.min_pass:
        failures.append(f"share within tolerance {share:.3f} is below {args.min_pass}")
    if args.max_2d is not None and worst_2d[1] > args.max_2d:
        failures.append(f"2D error {worst_2d[1]:.4f} px exceeds {args.max_2d}")
    if args.max_3d is not None and worst_3d[2] > args.max_3d:
        failures.append(f"3D error {worst_3d[2]:.4f} mm exceeds {args.max_3d}")
    for failure in failures:
        print("FAILED:", failure)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
