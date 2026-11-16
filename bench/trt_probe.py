"""First look at TensorRT for the two networks: export, build, speed, and difference from PyTorch.

Run inside the inference image, as root, with tensorrt and onnx installed. Writes into --work.
Usage: python trt_probe.py --work DIR [--show-sources]
"""
import argparse
import inspect
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/app")
from engine import Engine  # noqa: E402


class DetNet(torch.nn.Module):
    """Detector from normalised image to the head's raw outputs: 3 class maps, then 3 box maps."""

    def __init__(self, det):
        super().__init__()
        self.backbone, self.neck, self.head = det.backbone, det.neck, det.bbox_head

    def forward(self, x):
        cls_scores, bbox_preds = self.head(self.neck(self.backbone(x)))
        return (*cls_scores, *bbox_preds)


class PoseNet(torch.nn.Module):
    """Pose network from normalised crop to the head's raw outputs."""

    def __init__(self, pose):
        super().__init__()
        self.backbone, self.head = pose.backbone, pose.head
        self.neck = pose.neck if pose.with_neck else None

    def forward(self, x):
        feats = self.backbone(x)
        if self.neck is not None:
            feats = self.neck(feats)
        out = self.head(feats)
        return tuple(out) if isinstance(out, (tuple, list)) else out


def export(net, shape, path, outputs):
    dummy = torch.zeros(shape, device="cuda")
    with torch.no_grad():
        torch.onnx.export(net, dummy, str(path), opset_version=17, input_names=["input"], output_names=outputs,
                          dynamic_axes={name: {0: "batch"} for name in ["input"] + outputs}, do_constant_folding=True)
    print(f"exported {path.name}: {path.stat().st_size / 1e6:.0f} MB", flush=True)


def build(onnx_path, engine_path, shape, fp16, fp32_layers=(), max_batch=4):
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        for i in range(parser.num_errors):
            print("parser:", parser.get_error(i))
        raise RuntimeError("ONNX parse failed")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 6 << 30)
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    kept = 0
    if fp32_layers:
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            if any(key in layer.name for key in fp32_layers) and layer.type not in (trt.LayerType.SHAPE, trt.LayerType.CONSTANT):
                try:
                    layer.precision = trt.float32
                    for o in range(layer.num_outputs):
                        if layer.get_output(o).dtype == trt.float32:
                            layer.set_output_type(o, trt.float32)
                    kept += 1
                except Exception:
                    pass
    profile = builder.create_optimization_profile()
    profile.set_shape("input", (1, *shape[1:]), (min(3, max_batch), *shape[1:]), (max_batch, *shape[1:]))
    config.add_optimization_profile(profile)
    started = time.time()
    blob = builder.build_serialized_network(network, config)
    if blob is None:
        raise RuntimeError("engine build failed")
    engine_path.write_bytes(bytes(blob))
    print(f"built {engine_path.name}: {engine_path.stat().st_size / 1e6:.0f} MB in {time.time() - started:.0f} s "
          f"({network.num_layers} layers, {kept} kept in fp32)", flush=True)


class Runner:
    def __init__(self, engine_path):
        import tensorrt as trt
        self.trt = trt
        self.engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(engine_path.read_bytes())
        self.context = self.engine.create_execution_context()
        self.names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]

    def __call__(self, x):
        x = x.contiguous()
        self.context.set_input_shape("input", tuple(x.shape))
        self.context.set_tensor_address("input", x.data_ptr())
        outputs = []
        for name in self.names[1:]:
            out = torch.empty(tuple(self.context.get_tensor_shape(name)), dtype=torch.float32, device="cuda")
            self.context.set_tensor_address(name, out.data_ptr())
            outputs.append(out)
        self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        return outputs


def timeit(fn, x, rounds=30):
    for _ in range(5):
        fn(x)
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(rounds):
        fn(x)
    torch.cuda.synchronize()
    return 1000 * (time.perf_counter() - started) / rounds


def compare(name, reference, candidate):
    worst = max(float((a.float() - b.float()).abs().max()) for a, b in zip(reference, candidate))
    print(f"    {name}: largest absolute difference from PyTorch fp32 = {worst:.4f}", flush=True)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--show-sources", action="store_true")
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)

    engine = Engine(precision="fp32")
    det, pose = engine.det, engine.pose
    if args.show_sources:
        import mmpose.models.pose_estimators.topdown as topdown
        for label, obj in [("pose head forward", type(pose.head).forward), ("pose head predict", type(pose.head).predict),
                           ("pose head decode", getattr(type(pose.head), "decode", None)),
                           ("pose estimator class", type(pose)), ("topdown predict", topdown.TopdownPoseEstimator.predict),
                           ("codec decode", type(pose.head.decoder).decode),
                           ("det head forward", type(det.bbox_head).forward)]:
            print(f"===== {label}: {getattr(obj, '__qualname__', obj)}")
            try:
                print(inspect.getsource(obj))
            except Exception as error:
                print("  (no source)", error)
        print("===== pose data preprocessor:", pose.data_preprocessor)
        print("===== det data preprocessor:", det.data_preprocessor)
        print("===== pose test_cfg:", pose.test_cfg, "| head test_cfg:", getattr(pose.head, "test_cfg", None))
        print("===== pose codec:", pose.cfg.codec)

    import tensorrt as trt
    print(f"tensorrt {trt.__version__}, torch {torch.__version__}, GPU {torch.cuda.get_device_name(0)}", flush=True)

    rng = torch.Generator(device="cuda").manual_seed(0)
    plans = [("det", DetNet(det).eval(), (1, 3, 640, 640), ["cls0", "cls1", "cls2", "box0", "box1", "box2"], ("head",)),
             ("pose", PoseNet(pose).eval(), (1, 3, 384, 288), None, ())]
    for name, net, shape, outputs, fp32_layers in plans:
        print(f"--- {name}", flush=True)
        try:
            sample = torch.randn((2, *shape[1:]), device="cuda", generator=rng)
            reference = net(sample)
            reference = list(reference) if isinstance(reference, (tuple, list)) else [reference]
            print("  outputs:", [tuple(r.shape) for r in reference])
            outputs = outputs or [f"out{i}" for i in range(len(reference))]
            onnx_path = args.work / f"{name}.onnx"
            export(net, shape, onnx_path, outputs)
            variants = [("fp16", True, ())]
            if fp32_layers:
                variants.append(("fp16 with the head in fp32", True, fp32_layers))
            variants.append(("fp32", False, ()))
            for label, fp16, keep in variants:
                path = args.work / f"{name}_{label.split()[0]}{'_mixed' if keep else ''}.engine"
                try:
                    build(onnx_path, path, shape, fp16, keep)
                    runner = Runner(path)
                    compare(f"TensorRT {label}", reference, runner(sample))
                    for batch in (1, 2, 3, 4):
                        x = torch.randn((batch, *shape[1:]), device="cuda", generator=rng)
                        print(f"    batch {batch}: TensorRT {label} {timeit(runner, x):6.1f} ms", flush=True)
                except Exception as error:
                    print(f"  TensorRT {label} failed: {error!r}", flush=True)
            for batch in (1, 2, 3, 4):
                x = torch.randn((batch, *shape[1:]), device="cuda", generator=rng)
                eager = timeit(net, x)
                with torch.autocast("cuda", dtype=torch.float16):
                    half = timeit(net, x)
                print(f"    batch {batch}: PyTorch fp32 {eager:6.1f} ms, PyTorch autocast fp16 {half:6.1f} ms", flush=True)
        except Exception as error:
            import traceback
            traceback.print_exc()
            print(f"  {name} failed: {error!r}", flush=True)


if __name__ == "__main__":
    main()
