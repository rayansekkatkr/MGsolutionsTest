"""Builds TensorRT engines for the two networks. Run once on the GPU model that will serve them.

Each engine covers a forward pass only (normalised image in, raw head outputs out), for batches
of 1 to 4. Everything around it stays in engine.py.

Precision is chosen per part of the detector, because the task's 3D tolerance makes the person
box the sensitive quantity: a box off by a fraction of a pixel changes the estimated depth of
every joint. TF32 is always switched off: TensorRT enables it by default on recent GPUs, and it
silently gives "fp32" layers about the precision of fp16.

Usage: python trt_build.py --out DIR [--variants det_bn16,pose_16] [--reuse-onnx]
"""
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from engine import Engine  # noqa: E402

# name: (network, parts of the network kept in fp32; None = everything in fp32)
VARIANTS = {
    "det_32": ("det", None),
    "det_b16": ("det", ("/neck/", "/head/")),   # backbone in fp16
    "det_bn16": ("det", ("/head/",)),            # backbone and neck in fp16
    "det_16": ("det", ()),                       # everything in fp16
    "pose_32": ("pose", None),
    "pose_16": ("pose", ()),
}
SHAPES = {"det": (3, 640, 640), "pose": (3, 384, 288)}
OUTPUTS = {"det": ["cls0", "cls1", "cls2", "box0", "box1", "box2"], "pose": ["out0", "out1", "out2"]}
MAX_BATCH = 4


class DetNet(torch.nn.Module):
    """Detector, normalised image to the head's raw outputs: 3 class maps, then 3 box maps."""

    def __init__(self, det):
        super().__init__()
        self.backbone, self.neck, self.head = det.backbone, det.neck, det.bbox_head

    def forward(self, x):
        cls_scores, bbox_preds = self.head(self.neck(self.backbone(x)))
        return (*cls_scores, *bbox_preds)


class PoseNet(torch.nn.Module):
    """Pose network, normalised crop to the head's raw outputs (x, y, z distributions)."""

    def __init__(self, pose):
        super().__init__()
        self.backbone, self.neck, self.head = pose.backbone, pose.neck, pose.head

    def forward(self, x):
        return tuple(self.head(self.neck(self.backbone(x))))


def export(net, network: str, path: Path) -> None:
    dummy = torch.zeros((1, *SHAPES[network]), device="cuda")
    names = OUTPUTS[network]
    torch.onnx.export(net, dummy, str(path), opset_version=17, input_names=["input"], output_names=names,
                      dynamic_axes={name: {0: "batch"} for name in ["input"] + names}, do_constant_folding=True)
    print(f"exported {path.name}: {path.stat().st_size / 1e6:.0f} MB", flush=True)


def build(onnx_path: Path, engine_path: Path, network: str, fp32_parts) -> None:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    definition = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(definition, logger)
    if not parser.parse_from_file(str(onnx_path)):
        raise RuntimeError("; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 6 << 30)
    config.clear_flag(trt.BuilderFlag.TF32)
    kept = 0
    if fp32_parts is not None:
        config.set_flag(trt.BuilderFlag.FP16)
        if fp32_parts:
            config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
            for i in range(definition.num_layers):
                layer = definition.get_layer(i)
                if layer.type in (trt.LayerType.SHAPE, trt.LayerType.CONSTANT):
                    continue
                if any(part in layer.name for part in fp32_parts):
                    layer.precision = trt.float32
                    for o in range(layer.num_outputs):
                        if layer.get_output(o).dtype == trt.float32:
                            layer.set_output_type(o, trt.float32)
                    kept += 1
    profile = builder.create_optimization_profile()
    shape = SHAPES[network]
    profile.set_shape("input", (1, *shape), (3, *shape), (MAX_BATCH, *shape))
    config.add_optimization_profile(profile)
    started = time.time()
    blob = builder.build_serialized_network(definition, config)
    if blob is None:
        raise RuntimeError("engine build failed")
    engine_path.write_bytes(bytes(blob))
    print(f"built {engine_path.name}: {engine_path.stat().st_size / 1e6:.0f} MB in {time.time() - started:.0f} s "
          f"({definition.num_layers} layers, {kept} pinned to fp32)", flush=True)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--variants", default="det_bn16,pose_16")
    parser.add_argument("--reuse-onnx", action="store_true", help="skip the export when the .onnx file is already there")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    wanted = args.variants.split(",")

    engine = None
    for network in sorted({VARIANTS[name][0] for name in wanted}):
        path = args.out / f"{network}.onnx"
        if args.reuse_onnx and path.exists():
            continue
        engine = engine or Engine(precision="fp32")
        export((DetNet(engine.det) if network == "det" else PoseNet(engine.pose)).eval(), network, path)
    del engine
    torch.cuda.empty_cache()
    for name in wanted:
        network, fp32_parts = VARIANTS[name]
        build(args.out / f"{network}.onnx", args.out / f"{name}.engine", network, fp32_parts)


if __name__ == "__main__":
    main()
