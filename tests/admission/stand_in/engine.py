"""Stand-in for inference/engine.py: same interface, waits instead of computing.

The timing follows the real engine on an L4 at fp16 (about 49, 59, 76 ms for 1, 2, 3 photos).
The answer depends on the photo, so that an answer returned to the wrong request is detected.
"""
import time

import numpy as np


def answer(image):
    level = float(image[::16, ::16].mean())
    keypoints_2d = np.arange(266, dtype=np.float64).reshape(133, 2) + level
    keypoints_3d = (np.arange(399, dtype=np.float64).reshape(133, 3) + level) / 1000
    return keypoints_2d, keypoints_3d


class Engine:
    trt_det = trt_pose = None

    def __init__(self, precision="fp32", **options):
        self.device, self.fp16 = "stand-in", precision == "fp16"

    def infer(self, images):
        time.sleep(0.035 + 0.0135 * len(images))
        return [answer(image) for image in images]
