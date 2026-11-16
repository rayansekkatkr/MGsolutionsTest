#!/usr/bin/env bash
# The experiment currently being run on the GPU runner. Edited for each measurement.
# This one: the TensorRT engines built by the previous job, inside the real engine code, on the
# 1,000 practice photos: throughput per batch size and accuracy against the reference.
set -euo pipefail
cd "$(dirname "$0")/.."
IMAGE=pose-inference:gpu
WORK="$HOME/exam-gpu"   # outside the checkout, which is wiped at the start of every run
REF="$WORK/practice_reference.jsonl"
docker build -q -t "$IMAGE" inference >/dev/null
ls "$WORK/trt"/*.engine
docker run --rm --gpus all --user 0 -v "$WORK:/work" -v "$PWD/bench:/bench:ro" \
  -e PYTHONPATH=/app:/opt/mmpose/projects/rtmpose3d "$IMAGE" bash -c '
    pip install -q --root-user-action=ignore tensorrt==8.6.1.post1 2>&1 | tail -3
    rm -rf /work/trt_out && mkdir -p /work/trt_out
    python /bench/gpu_bench.py --images /work/practice --out /work/trt_out --trt-dir /work/trt --batches 1,2,3,4 \
      --precisions "fp16,trt:det_fp16_mixed:pose_fp16,trt:det_fp16:pose_fp16,trt:det_fp16_mixed:pose_fp32,trt:-:pose_fp16,trt:det_fp32:pose_fp32" 2>&1 \
      | grep -v -i "warn\|meshgrid\|Loads checkpoint"
    chown -R '"$(id -u):$(id -g)"' /work/trt_out'
for f in "$WORK"/trt_out/*_b1.jsonl "$WORK"/trt_out/*_b3.jsonl; do
  printf '%-40s ' "[$(basename "$f" .jsonl)]"
  python3 tests/inference/compare.py "$REF" "$f" --min-pass 0 | head -1
done
echo "JOB DONE"
