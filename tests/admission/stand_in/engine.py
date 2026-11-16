"""Stand-in for inference/engine.py: same interface, waits instead of computing.

The timing follows the real engine on an L4 with its TensorRT engines: about 22, 40 and 57 ms for
1, 2 and 3 photos, of which roughly a third is CPU work that two workers can overlap and the rest
GPU work that they cannot. The answer depends on the photo, so that an answer returned to the
wrong request is detected.
"""
import threading
import time

import numpy as np


def answer(image):
    level = float(image[::16, ::16].mean())
    keypoints_2d = np.arange(266, dtype=np.float64).reshape(133, 2) + level
    keypoints_3d = (np.arange(399, dtype=np.float64).reshape(133, 3) + level) / 1000
    return keypoints_2d, keypoints_3d


class Engine:
    trt_det = trt_pose = None
    lanes = 3

    def __init__(self, precision="fp32", **options):
        self.device, self.fp16 = "stand-in", precision == "fp16"
        self._gpu = threading.Lock()

    def infer(self, images, lane=0):
        time.sleep(0.007 * len(images))                      # pipelines, on the CPU
        with self._gpu:
            time.sleep(0.0045 + 0.0105 * len(images))        # forward passes, one batch at a time
        return [answer(image) for image in images]
