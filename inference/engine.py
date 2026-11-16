"""Batched inference: RTMDet-x picks one person box per photo, RTMW3D-x estimates 133 joints in it.

The models, configs, test pipelines and weights are the official MMDetection and MMPose ones.
Two things differ from calling `inference_detector` and `inference_topdown` once per photo:
several photos go through each model in one forward pass, and the person box is read directly
from the detector's outputs instead of going through NMS (see `Engine.detect`).
"""
import threading
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


_lane = threading.local()  # which lane the current thread works in (set by Engine.infer)


class TrtNet:
    """A TensorRT engine called with one CUDA tensor; returns the named outputs as CUDA tensors.

    Several threads may call it at the same time, each in its own lane: a lane has its own
    execution context, on its own optimisation profile of the engine.
    """

    def __init__(self, path: Path, outputs: list):
        import tensorrt as trt

        self._runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
        self._engine = self._runtime.deserialize_cuda_engine(path.read_bytes())
        if self._engine is None:
            raise RuntimeError(f"{path.name} could not be loaded by this TensorRT version or GPU")
        self.lanes = self._engine.num_optimization_profiles
        self._contexts = {}
        self._lock = threading.Lock()
        self._outputs = outputs

    def _context(self):
        lane = getattr(_lane, "index", 0)
        context = self._contexts.get(lane)
        if context is None:
            with self._lock:
                context = self._engine.create_execution_context()
                if lane:
                    context.set_optimization_profile_async(lane, torch.cuda.current_stream().cuda_stream)
                self._contexts[lane] = context
        return context

    def __call__(self, x: torch.Tensor) -> list:
        context = self._context()
        x = x.float().contiguous()
        context.set_input_shape("input", tuple(x.shape))
        context.set_tensor_address("input", x.data_ptr())
        results = []
        for name in self._outputs:
            out = torch.empty(tuple(context.get_tensor_shape(name)), dtype=torch.float32, device=x.device)
            context.set_tensor_address(name, out.data_ptr())
            results.append(out)
        # Same CUDA stream as this thread's PyTorch work: whatever reads the outputs next is queued behind this.
        if not context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("TensorRT execution failed")
        return results


class Engine:
    def __init__(self, precision: str = "fp32", device: str = "", head_fp16: bool = False,
                 channels_last: bool = False, cudnn_benchmark: bool = False,
                 trt_dir: str = "", trt_det: str = "det", trt_pose: str = "pose"):
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

        self.head_fp16 = head_fp16 and self.fp16
        self.channels_last = channels_last and self.device.startswith("cuda")

        # Same numerical settings as the reference run, unless a faster option is asked for.
        torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
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
        if self.channels_last:
            self.det.to(memory_format=torch.channels_last)
            self.pose.to(memory_format=torch.channels_last)
        self._streams, self._lock = {}, threading.Lock()
        self.trt_det = self.trt_pose = None
        if trt_dir and self.device.startswith("cuda"):
            self._use_tensorrt(Path(trt_dir), trt_det, trt_pose)
        if self.fp16 and self.trt_det is None:
            self._half_precision_features()

    def _use_tensorrt(self, directory: Path, det_name: str, pose_name: str) -> None:
        """Replaces the forward pass of a network by its TensorRT engine when DIR/NAME.engine loads.

        Everything around the forward pass is unchanged: the pipelines, the person box read-out
        and the pose head's own decoding. An engine is only used after it has reproduced the
        PyTorch network's outputs on a check input. A network whose engine is missing, does not
        load (other GPU model, other TensorRT version) or fails the check keeps running in PyTorch.
        """
        @torch.no_grad()
        def checked(path: Path, outputs: list, shape: tuple, reference) -> TrtNet:
            net = TrtNet(path, outputs)
            # A smooth image-like input (not noise) so that activations are in their usual range.
            rows = torch.linspace(-2, 2, shape[1], device=self.device)[:, None]
            columns = torch.linspace(-2, 2, shape[2], device=self.device)[None, :]
            planes = [torch.sin(3 * columns + rows), torch.cos(2 * rows - columns), rows * columns / 2]
            x = torch.stack([torch.stack(planes), torch.stack(planes[::-1])])  # [2, 3, height, width]
            expected = reference(x)
            # Typical (median) difference relative to the typical magnitude, per output. Not the
            # largest one: the box maps go through an exponential, and where there is nothing to
            # detect their values are huge and meaningless. A wrong or damaged engine is off by
            # about 1 on this measure, reduced precision by about 0.001.
            worst = max(float((a - b).abs().median() / b.abs().median().clamp(min=1e-6)) for a, b in zip(net(x), expected))
            print(f"{path.name}: typical relative difference from PyTorch on the check input {worst:.4f}", flush=True)
            if not worst < 0.2:
                raise RuntimeError(f"{path.name} does not reproduce the PyTorch outputs")
            return net

        try:
            if det_name != "-":
                def det_reference(x):
                    cls_scores, bbox_preds = self.det.bbox_head(self.det.extract_feat(x))
                    return [*cls_scores, *bbox_preds]
                self.trt_det = checked(directory / f"{det_name}.engine", ["cls0", "cls1", "cls2", "box0", "box1", "box2"],
                                       (3, 640, 640), det_reference)
        except Exception as error:
            print(f"detector stays in PyTorch: {error!r}", flush=True)
        try:
            if pose_name != "-":
                extract_feat, forward = self.pose.extract_feat, self.pose.head.forward
                net = checked(directory / f"{pose_name}.engine", ["out0", "out1", "out2"], (3, 384, 288),
                              lambda x: list(forward(extract_feat(x))))
                self.pose.extract_feat = lambda inputs: inputs       # the engine starts from the image
                self.pose.head.forward = lambda inputs: tuple(net(inputs))
                self.trt_pose = net
        except Exception as error:
            print(f"pose network stays in PyTorch: {error!r}", flush=True)

    def _half_precision_features(self) -> None:
        """fp16 for the detector's backbone and neck only; box decoding and NMS stay in fp32."""
        extract = self.det.extract_feat

        def extract_feat(x):
            with torch.autocast("cuda", dtype=torch.float16):
                features = extract(x)
            return features if self.head_fp16 else tuple(f.float() for f in features)

        self.det.extract_feat = extract_feat

    @torch.no_grad()
    def detect(self, images: list, details: list = None) -> list:
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
        inputs = data["inputs"]
        if self.channels_last:
            inputs = inputs.contiguous(memory_format=torch.channels_last)
        if self.trt_det is not None:
            outputs = self.trt_det(inputs)
            cls_scores, bbox_preds = outputs[:3], outputs[3:]
        elif self.head_fp16:
            features = self.det.extract_feat(inputs)
            with torch.autocast("cuda", dtype=torch.float16):
                cls_scores, bbox_preds = head(features)
            cls_scores = [c.float() for c in cls_scores]
            bbox_preds = [b.float() for b in bbox_preds]
        else:
            cls_scores, bbox_preds = head(self.det.extract_feat(inputs))

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
            if details is not None:
                details.append((best_score[i], self._rival_score(head, priors, distances[i], person[i], best_anchor[i], sample)))
            if best_score[i] < MIN_PERSON_SCORE:
                height, width = image.shape[:2]
                boxes.append(np.array([0, 0, width, height], dtype=np.float32))
                continue
            j = best_anchor[i]
            box = head.bbox_coder.decode(priors[j:j + 1], distances[i, j:j + 1], max_shape=sample.img_shape)
            box = scale_boxes(box, [1 / s for s in sample.scale_factor])
            boxes.append(box[0].float().cpu().numpy().astype(np.float32))
        return boxes

    @staticmethod
    def _rival_score(head, priors, distances, person, best_anchor, sample) -> float:
        """Highest person score among candidates whose box is clearly another box than the best one.

        Neighbouring anchors propose nearly the same box for the same person; they are not rivals.
        A rival is a candidate that overlaps the best box by less than 70 % (IoU).
        """
        from mmdet.structures.bbox import bbox_overlaps

        candidates = head.bbox_coder.decode(priors, distances, max_shape=sample.img_shape).float()
        overlap = bbox_overlaps(candidates[best_anchor:best_anchor + 1], candidates)[0]
        return float(person.masked_fill(overlap >= 0.7, 0.0).max())

    @torch.no_grad()
    def estimate(self, images: list, boxes: list) -> list:
        """(keypoints_2d [133, 2] in photo pixels, keypoints_3d [133, 3] in metres) per photo."""
        from mmengine.dataset import pseudo_collate

        samples = []
        for image, box in zip(images, boxes):
            sample = dict(img=image, bbox=box[None], bbox_score=np.ones(1, dtype=np.float32))
            sample.update(self.pose.dataset_meta)
            samples.append(self.pose_pipeline(sample))
        if self.fp16 and self.trt_pose is None:
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

    @property
    def lanes(self) -> int:
        """How many threads may call infer() at the same time, each with its own lane number."""
        return min([net.lanes for net in (self.trt_det, self.trt_pose) if net is not None] or [8])

    def infer(self, images: list, lane: int = 0) -> list:
        """Runs both models on a batch. Threads working in parallel must each use a different lane:
        a lane has its own CUDA stream and its own TensorRT execution contexts, so that one
        thread's CPU work (pipelines, decoding the outputs) overlaps the other's GPU work."""
        _lane.index = lane
        if lane and self.device.startswith("cuda"):
            with self._lock:
                stream = self._streams.setdefault(lane, torch.cuda.Stream())
            with torch.cuda.stream(stream):
                return self.estimate(images, self.detect(images))
        return self.estimate(images, self.detect(images))
