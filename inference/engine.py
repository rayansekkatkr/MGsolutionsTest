"""Batched inference: RTMDet-x picks one person box per photo, RTMW3D-x estimates 133 joints in it.

The models, configs, test pipelines and weights are the official MMDetection and MMPose ones.
Two things differ from calling `inference_detector` and `inference_topdown` once per photo:
several photos go through each model in one forward pass, and the person box is read directly
from the detector's outputs instead of going through NMS (see `Engine.detect`).
"""
from pathlib import Path

import numpy as np
import torch

MMPOSE = Path("/opt/mmpose")
DET_CONFIG = "rtmdet_x_8xb32-300e_coco.py"
DET_WEIGHTS = "/models/rtmdet_x_8xb32-300e_coco_20220715_230555-cc79b9ae.pth"
POSE_CONFIG = MMPOSE / "projects/rtmpose3d/configs/rtmw3d-x_8xb32_cocktail14-384x288.py"
POSE_WEIGHTS = "/models/rtmw3d-x_8xb64_cocktail14-384x288-b0a0eab7_20240626.pth"
PERSON_LABEL = 0
MIN_PERSON_SCORE = 0.3
JOINTS = 133


class Engine:
    def __init__(self, precision: str = "fp32", device: str = ""):
        import mmdet
        from mmcv.transforms import Compose as DetCompose
        from mmdet.apis import init_detector
        from mmdet.utils import get_test_pipeline_cfg
        from mmengine.dataset import Compose as PoseCompose
        from mmpose.apis import init_model
        from mmpose.utils import adapt_mmdet_pipeline
        import rtmpose3d  # noqa: F401  registers the RTMW3D modules

        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.fp16 = precision == "fp16" and self.device.startswith("cuda")

        # Same numerical settings as the reference run.
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        det_config = Path(mmdet.__file__).parent / ".mim/configs/rtmdet" / DET_CONFIG
        self.det = init_detector(str(det_config), DET_WEIGHTS, device=self.device)
        self.det.cfg = adapt_mmdet_pipeline(self.det.cfg)
        det_pipeline = get_test_pipeline_cfg(self.det.cfg.copy())
        det_pipeline[0].type = "mmdet.LoadImageFromNDArray"
        self.det_pipeline = DetCompose(det_pipeline)

        self.pose = init_model(str(POSE_CONFIG), POSE_WEIGHTS, device=self.device)
        self.pose_pipeline = PoseCompose(self.pose.cfg.test_dataloader.dataset.pipeline)

        self._priors = {}
        if self.fp16:
            self._half_precision_features()

    def _half_precision_features(self) -> None:
        """fp16 for the detector's backbone and neck only; box decoding and NMS stay in fp32."""
        extract = self.det.extract_feat

        def extract_feat(x):
            with torch.autocast("cuda", dtype=torch.float16):
                features = extract(x)
            return tuple(f.float() for f in features)

        self.det.extract_feat = extract_feat

    @torch.no_grad()
    def detect(self, images: list) -> list:
        """One person box [x1, y1, x2, y2] per photo: the highest-scoring person, or the whole photo.

        The official post-processing decodes every candidate box of the 80 classes and runs NMS
        (about 40 ms per photo on an L4), and the pipeline then keeps a single box: the person
        with the highest score. Class-wise NMS never removes the highest-scoring box of a class,
        so that box is read directly from the head's outputs, with the head's own decoder and
        rescaling. On 200 photos the result is identical to the official one.
        """
        from mmdet.structures.bbox import scale_boxes

        prepared = [self.det_pipeline(dict(img=image, img_id=0)) for image in images]
        batch = dict(inputs=[p["inputs"] for p in prepared], data_samples=[p["data_samples"] for p in prepared])
        data = self.det.data_preprocessor(batch, False)
        head = self.det.bbox_head
        cls_scores, bbox_preds = head(self.det.extract_feat(data["inputs"]))

        sizes = tuple(tuple(c.shape[-2:]) for c in cls_scores)
        if sizes not in self._priors:
            self._priors[sizes] = torch.cat(head.prior_generator.grid_priors(
                list(sizes), dtype=cls_scores[0].dtype, device=cls_scores[0].device))
        priors = self._priors[sizes]
        person = torch.cat([c[:, PERSON_LABEL].flatten(1) for c in cls_scores], dim=1).sigmoid()  # [photos, anchors]
        distances = torch.cat([b.permute(0, 2, 3, 1).reshape(b.shape[0], -1, 4) for b in bbox_preds], dim=1)
        best_score, best_anchor = person.max(dim=1)
        best_score, best_anchor = best_score.tolist(), best_anchor.tolist()

        boxes = []
        for i, (sample, image) in enumerate(zip(data["data_samples"], images)):
            if best_score[i] < MIN_PERSON_SCORE:
                height, width = image.shape[:2]
                boxes.append(np.array([0, 0, width, height], dtype=np.float32))
                continue
            j = best_anchor[i]
            box = head.bbox_coder.decode(priors[j:j + 1], distances[i, j:j + 1], max_shape=sample.img_shape)
            box = scale_boxes(box, [1 / s for s in sample.scale_factor])
            boxes.append(box[0].float().cpu().numpy().astype(np.float32))
        return boxes

    @torch.no_grad()
    def estimate(self, images: list, boxes: list) -> list:
        """(keypoints_2d [133, 2] in photo pixels, keypoints_3d [133, 3] in metres) per photo."""
        from mmengine.dataset import pseudo_collate

        samples = []
        for image, box in zip(images, boxes):
            sample = dict(img=image, bbox=box[None], bbox_score=np.ones(1, dtype=np.float32))
            sample.update(self.pose.dataset_meta)
            samples.append(self.pose_pipeline(sample))
        if self.fp16:
            with torch.autocast("cuda", dtype=torch.float16):
                results = self.pose.test_step(pseudo_collate(samples))
        else:
            results = self.pose.test_step(pseudo_collate(samples))
        out = []
        for result in results:
            pred = result.pred_instances
            keypoints_3d = np.asarray(pred.keypoints, dtype=np.float64).reshape(-1, JOINTS, 3)[0]
            keypoints_2d = np.asarray(pred.transformed_keypoints, dtype=np.float64).reshape(-1, JOINTS, 2)[0]
            out.append((keypoints_2d, keypoints_3d))
        return out

    def infer(self, images: list) -> list:
        return self.estimate(images, self.detect(images))
