"""Writes test photos of about the size of the task's photos, and the stand-in engine's answers for them.

Usage: python3 make_photos.py OUT_DIR
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "stand_in"))
from engine import answer  # noqa: E402

out = Path(sys.argv[1])
(out / "photos").mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(1)
with (out / "reference.jsonl").open("w") as reference:
    for n in range(40):
        coarse = rng.integers(0, 255, (60, 80, 3), dtype=np.uint8)
        image = cv2.resize(coarse, (640, 480), interpolation=cv2.INTER_CUBIC)
        image = np.clip(image.astype(int) + 3 * n - 60 + rng.integers(-12, 12, image.shape), 0, 255).astype(np.uint8)
        path = out / "photos" / f"{n:04d}.jpg"
        cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        keypoints_2d, keypoints_3d = answer(cv2.imread(str(path), cv2.IMREAD_COLOR))
        reference.write(json.dumps({"file": path.name, "keypoints_2d": np.round(keypoints_2d, 2).tolist(),
                                    "keypoints_3d": np.round(keypoints_3d, 5).tolist()}) + "\n")
sizes = [p.stat().st_size for p in (out / "photos").glob("*.jpg")]
print(f"{len(sizes)} photos, {sum(sizes) // len(sizes) // 1000} KB on average")
