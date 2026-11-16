#!/usr/bin/env bash
# GPU measurements, run on the self-hosted L4 runner:
#   1. reference results with the official per-photo APIs (fp32)
#   2. engine throughput per precision and batch size, and accuracy of each against the reference
#   3. end-to-end throughput of the HTTP server under closed-loop load
set -euo pipefail
cd "$(dirname "$0")/../.."
IMAGE="${IMAGE:-pose-inference:gpu}"
# Kept outside the checkout: actions/checkout wipes untracked files at the start of every run.
WORK="$HOME/exam-gpu"
mkdir -p "$WORK/images" "$WORK/bench"
trap 'docker rm -f server >/dev/null 2>&1 || true' EXIT

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo "vCPUs: $(nproc)"

echo "--- 1. build"
docker build -q -t "$IMAGE" inference

echo "--- 2. photos"
bash tests/inference/fetch_images.sh tests/inference/images_gpu.txt "$WORK/images"

RUN=(docker run --rm --gpus all --user "$(id -u):$(id -g)" -v "$WORK:/work" -v "$PWD/tests/inference:/tests:ro"
     -v "$PWD/bench:/bench:ro" -e PYTHONPATH=/app:/opt/mmpose/projects/rtmpose3d "$IMAGE")

echo "--- 3. reference: official APIs, fp32, one photo at a time"
START=$SECONDS
"${RUN[@]}" python /tests/reference_run.py --images /work/images --out /work/reference.jsonl | tail -1
echo "reference run took $((SECONDS - START)) s including model loading"

echo "--- 4. engine throughput"
"${RUN[@]}" python /bench/gpu_bench.py --images /work/images --out /work/bench 2>&1 | grep -v -i "warn\|meshgrid\|Loads checkpoint"

echo "--- 5. accuracy of each configuration against the reference (tolerance 3.5 px, 25 mm)"
for f in "$WORK"/bench/*.jsonl; do
  echo "[$(basename "$f" .jsonl)]"
  python3 tests/inference/compare.py "$WORK/reference.jsonl" "$f" --min-pass 0 | sed 's/^/  /'
done

echo "--- 6. HTTP server under closed-loop load"
for config in "fp16 4" "fp16 8" "fp16 16"; do
  set -- $config
  docker rm -f server >/dev/null 2>&1 || true
  docker run -d --name server --gpus all --read-only --user 10001 --cap-drop ALL --tmpfs /tmp:rw,nosuid,nodev,size=2048m \
    --shm-size 1024m -p 8080:8080 -e PRECISION="$1" -e MAX_BATCH="$2" -e MAX_PENDING=256 "$IMAGE" >/dev/null
  for i in $(seq 1 60); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8080/health || true)" = 200 ] && break
    sleep 2
  done
  echo "[precision=$1 max_batch=$2] ready after about $((i * 2)) s"
  python3 bench/load_closed.py --images "$WORK/images" --url http://127.0.0.1:8080 --workers $(( $2 * 2 )) --seconds 30 &
  LOAD=$!
  sleep 18
  echo "  mid-run: GPU $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader), host CPU idle $(top -bn1 | awk '/Cpu\(s\)/ {print $8}')%"
  wait "$LOAD"
done
echo "GPU MEASUREMENTS DONE"
