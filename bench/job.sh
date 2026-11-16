#!/usr/bin/env bash
# The experiment currently being run on the GPU runner. Edited for each measurement.
# This one: cheap speed options (fp16 head, channels-last memory format, cuDNN autotuning) and
# several batch threads sharing the GPU. Each is checked for accuracy against the reference.
set -euo pipefail
cd "$(dirname "$0")/.."
IMAGE=pose-inference:gpu
# Kept outside the checkout: actions/checkout wipes untracked files at the start of every run.
WORK="$HOME/exam-gpu"
bash tests/inference/fetch_images.sh tests/inference/images_gpu.txt "$WORK/images"
docker build -q -t "$IMAGE" inference >/dev/null
trap 'docker rm -f server >/dev/null 2>&1 || true' EXIT
RUN=(docker run --rm --gpus all --user "$(id -u):$(id -g)" -v "$WORK:/work" -v "$PWD/tests/inference:/tests:ro"
     -v "$PWD/bench:/bench:ro" -e PYTHONPATH=/app:/opt/mmpose/projects/rtmpose3d "$IMAGE")
[ -s "$WORK/reference.jsonl" ] || "${RUN[@]}" python /tests/reference_run.py --images /work/images --out /work/reference.jsonl | tail -1

echo "--- engine options"
rm -rf "$WORK/options" && mkdir -p "$WORK/options"
"${RUN[@]}" python /bench/gpu_bench.py --images /work/images --out /work/options --batches 4,8 \
  --precisions "fp16,fp16+head_fp16,fp16+channels_last,fp16+cudnn_benchmark,fp16+head_fp16+channels_last+cudnn_benchmark" 2>&1 \
  | grep -v -i "warn\|meshgrid\|Loads checkpoint"
for f in "$WORK"/options/*_b4.jsonl; do
  printf '%-52s ' "[$(basename "$f" _b4.jsonl)]"
  python3 tests/inference/compare.py "$WORK/reference.jsonl" "$f" --min-pass 0 | head -1
done

echo "--- HTTP server: batch threads sharing the GPU (fp16, max batch 4)"
for workers in 1 2 3; do
  docker rm -f server >/dev/null 2>&1 || true
  docker run -d --name server --gpus all --read-only --user 10001 --cap-drop ALL --tmpfs /tmp:rw,nosuid,nodev,size=2048m \
    --shm-size 1024m -p 8080:8080 -e PRECISION=fp16 -e MAX_BATCH=4 -e WORKERS="$workers" -e MAX_PENDING=256 "$IMAGE" >/dev/null
  for i in $(seq 1 60); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8080/health || true)" = 200 ] && break
    sleep 2
  done
  python3 bench/load_closed.py --images "$WORK/images" --url http://127.0.0.1:8080 --workers $(( 8 * workers )) --seconds 25 &
  LOAD=$!
  sleep 15
  echo "[batch threads=$workers] mid-run: GPU $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader), host CPU idle $(top -bn1 | awk '/Cpu\(s\)/ {print $8}')%"
  wait "$LOAD"
done
echo "JOB DONE"
