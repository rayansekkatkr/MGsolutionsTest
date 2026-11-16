"""Reference results with the official per-photo APIs, as the task specification describes them.

  inference_detector  -> highest-scoring person box (whole photo if none reaches 0.3)
  inference_topdown   -> RTMW3D-x on that box, no flip test
  fp32, one photo at a time, cv2.imread

Runs inside the inference image. Usage: python reference_run.py --images DIR --out FILE.jsonl
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from engine import DET_CONFIG, DET_WEIGHTS, MIN_PERSON_SCORE, PERSON_LABEL, POSE_CONFIG, POSE_WEIGHTS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    import mmdet
    from mmdet.apis import inference_detector, init_detector
    from mmpose.apis import inference_topdown, init_model
    from mmpose.utils import adapt_mmdet_pipeline
    import rtmpose3d  # noqa: F401

    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    det = init_detector(str(Path(mmdet.__file__).parent / ".mim/configs/rtmdet" / DET_CONFIG), DET_WEIGHTS, device=device)
    det.cfg = adapt_mmdet_pipeline(det.cfg)
    pose = init_model(str(POSE_CONFIG), POSE_WEIGHTS, device=device)

    with args.out.open("w") as out:
        for path in sorted(args.images.glob("*.jpg")):
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            pred = inference_detector(det, image).pred_instances.cpu().numpy()
            keep = (pred.labels == PERSON_LABEL) & (pred.scores >= MIN_PERSON_SCORE)
            if keep.any():
                box = pred.bboxes[int(np.argmax(np.where(keep, pred.scores, -1)))].astype(np.float32)
            else:
                box = np.array([0, 0, image.shape[1], image.shape[0]], dtype=np.float32)
            instance = inference_topdown(pose, image, bboxes=box[None], bbox_format="xyxy")[0].pred_instances
            keypoints_3d = np.asarray(instance.keypoints, dtype=np.float64).reshape(-1, 133, 3)[0]
            keypoints_2d = np.asarray(instance.transformed_keypoints, dtype=np.float64).reshape(-1, 133, 2)[0]
            out.write(json.dumps({"file": path.name, "bbox": [round(float(v), 2) for v in box],
                                  "keypoints_2d": np.round(keypoints_2d, 2).tolist(),
                                  "keypoints_3d": np.round(keypoints_3d, 5).tolist()}, separators=(",", ":")) + "\n")
            print("reference:", path.name, flush=True)


if __name__ == "__main__":
    main()
