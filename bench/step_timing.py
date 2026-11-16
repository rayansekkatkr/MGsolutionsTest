"""Where the time goes, step by step, on the GPU. Also checks a direct way to get the person box.

The official post-processing decodes and filters thousands of candidate boxes for 80 classes and
runs NMS, then the pipeline keeps one box: the highest-scoring person. Class-wise NMS never removes
the highest-scoring box of a class, so that box can be read directly from the head's outputs.
This script measures both paths and compares the boxes they produce.

Usage (inside the image): python step_timing.py --images DIR [--precision fp16] [--batch 8] [--limit 200]
"""
import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, "/app")
from engine import MIN_PERSON_SCORE, PERSON_LABEL, Engine  # noqa: E402

T = defaultdict(float)


class timed:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.t = time.perf_counter()

    def __exit__(self, *a):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        T[self.name] += time.perf_counter() - self.t


def official_box(pred, image):
    p = pred.pred_instances.cpu().numpy()
    keep = (p.labels == PERSON_LABEL) & (p.scores >= MIN_PERSON_SCORE)
    if keep.any():
        return p.bboxes[int(np.argmax(np.where(keep, p.scores, -1)))].astype(np.float32)
    return np.array([0, 0, image.shape[1], image.shape[0]], dtype=np.float32)


def direct_boxes(head, cls_scores, bbox_preds, metas, images):
    """Highest-scoring person box per photo, straight from the head outputs (no NMS)."""
    from mmdet.structures.bbox import scale_boxes
    sizes = [c.shape[-2:] for c in cls_scores]
    priors = torch.cat(head.prior_generator.grid_priors(sizes, dtype=cls_scores[0].dtype,
                                                        device=cls_scores[0].device))
    person = torch.cat([c[:, PERSON_LABEL].flatten(1) for c in cls_scores], dim=1).sigmoid()   # [B, anchors]
    boxes_raw = torch.cat([b.permute(0, 2, 3, 1).reshape(b.shape[0], -1, 4) for b in bbox_preds], dim=1)
    score, index = person.max(dim=1)
    out = []
    for i, (meta, image) in enumerate(zip(metas, images)):
        if float(score[i]) < MIN_PERSON_SCORE:
            out.append(np.array([0, 0, image.shape[1], image.shape[0]], dtype=np.float32))
            continue
        j = int(index[i])
        box = head.bbox_coder.decode(priors[j:j + 1], boxes_raw[i, j:j + 1], max_shape=meta["img_shape"])
        box = scale_boxes(box, [1 / s for s in meta["scale_factor"]])
        out.append(box[0].float().cpu().numpy().astype(np.float32))
    return out


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--precision", default="fp16")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()

    images = [cv2.imread(str(p), cv2.IMREAD_COLOR) for p in sorted(args.images.glob("*.jpg"))[:args.limit]]
    engine = Engine(precision=args.precision)
    det, head = engine.det, engine.det.bbox_head
    from mmengine.dataset import pseudo_collate

    worst = 0.0
    different = 0
    for round_ in ("warm-up", "measure"):
        T.clear()
        subset = images[:2 * args.batch] if round_ == "warm-up" else images
        for start in range(0, len(subset), args.batch):
            chunk = subset[start:start + args.batch]
            with timed("det 1 pipeline (resize, pad) CPU"):
                prepared = [engine.det_pipeline(dict(img=im, img_id=0)) for im in chunk]
                batch = dict(inputs=[p["inputs"] for p in prepared], data_samples=[p["data_samples"] for p in prepared])
            with timed("det 2 to GPU + normalise"):
                data = det.data_preprocessor(batch, False)
            with timed("det 3 backbone + neck"):
                feats = det.extract_feat(data["inputs"])
            with timed("det 4 head"):
                cls_scores, bbox_preds = head(feats)
            metas = [s.metainfo for s in data["data_samples"]]
            with timed("det 5a official post-processing (NMS)"):
                results = head.predict_by_feat(cls_scores, bbox_preds, batch_img_metas=metas, rescale=True)
                from mmdet.structures import DetDataSample
                official = []
                for r, im in zip(results, chunk):
                    sample = DetDataSample()
                    sample.pred_instances = r
                    official.append(official_box(sample, im))
            with timed("det 5b direct person box (no NMS)"):
                direct = direct_boxes(head, cls_scores, bbox_preds, metas, chunk)
            for a, b in zip(official, direct):
                diff = float(np.abs(a - b).max())
                worst = max(worst, diff)
                different += diff > 0.01
            with timed("pose 1 pipeline (affine crop) CPU"):
                samples = []
                for im, box in zip(chunk, official):
                    s = dict(img=im, bbox=box[None], bbox_score=np.ones(1, dtype=np.float32))
                    s.update(engine.pose.dataset_meta)
                    samples.append(engine.pose_pipeline(s))
                collated = pseudo_collate(samples)
            with timed("pose 2 forward + decode"):
                if engine.fp16:
                    with torch.autocast("cuda", dtype=torch.float16):
                        engine.pose.test_step(collated)
                else:
                    engine.pose.test_step(collated)

    n = len(images)
    print(f"\n[{args.precision}, batch {args.batch}, {n} photos] milliseconds per photo")
    for name in sorted(T):
        print(f"  {name:42} {1000 * T[name] / n:7.2f}")
    kept = sum(v for k, v in T.items() if "5a" not in k)
    print(f"  {'total with the direct box':42} {1000 * kept / n:7.2f}  -> {n / kept:.1f} photos/s on one thread")
    print(f"  direct box vs official box: max difference {worst:.4f} px, photos differing by more than 0.01 px: {different}")


if __name__ == "__main__":
    main()
