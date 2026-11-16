#!/usr/bin/env bash
# The experiment currently being run on the GPU runner. Edited for each measurement.
# This one, on the task's 1,000 practice photos:
#   1. reference results (official APIs, fp32, one photo at a time) and a per-photo summary of them
#   2. engine throughput and accuracy at fp16 for small batches
#   3. HTTP server: throughput and latency for each (max batch, requests in the system) pair
set -euo pipefail
cd "$(dirname "$0")/.."
IMAGE=pose-inference:gpu
# Kept outside the checkout: actions/checkout wipes untracked files at the start of every run.
WORK="$HOME/exam-gpu"
mkdir -p "$WORK/practice"
xargs -P 8 -I{} sh -c '[ -s "$1/$(basename "$2")" ] || curl -fsS --retry 3 -o "$1/$(basename "$2")" "$2"' _ "$WORK/practice" {} \
  < tests/inference/images_practice.txt
echo "$(find "$WORK/practice" -name '*.jpg' | wc -l) practice photos"
docker build -q -t "$IMAGE" inference >/dev/null
trap 'docker rm -f server >/dev/null 2>&1 || true' EXIT
RUN=(docker run --rm --gpus all --user "$(id -u):$(id -g)" -v "$WORK:/work" -v "$PWD/tests/inference:/tests:ro"
     -v "$PWD/bench:/bench:ro" -e PYTHONPATH=/app:/opt/mmpose/projects/rtmpose3d "$IMAGE")

echo "--- 1. reference on the practice photos"
REF="$WORK/practice_reference.jsonl"
[ "$(wc -l < "$REF" 2>/dev/null || echo 0)" = 1000 ] || "${RUN[@]}" python /tests/reference_run.py --images /work/practice --out /work/practice_reference.jsonl | tail -1
echo "SUMMARY BEGIN"
python3 - "$REF" <<'PY'
import json, sys
for row in map(json.loads, open(sys.argv[1])):
    k2, k3 = row["keypoints_2d"], row["keypoints_3d"]
    mean = lambda points, axis: sum(p[axis] for p in points) / len(points)
    print(row["file"], *(f"{mean(k2, a):.3f}" for a in range(2)), *(f"{mean(k3, a):.5f}" for a in range(3)))
PY
echo "SUMMARY END"

echo "--- 2. engine at fp16, small batches"
rm -rf "$WORK/practice_out" && mkdir -p "$WORK/practice_out"
"${RUN[@]}" python /bench/gpu_bench.py --images /work/practice --out /work/practice_out --batches 1,2,3,4 --precisions fp16 2>&1 \
  | grep -v -i "warn\|meshgrid\|Loads checkpoint"
for f in "$WORK"/practice_out/*.jsonl; do
  echo "[$(basename "$f" .jsonl)]"
  python3 tests/inference/compare.py "$REF" "$f" --min-pass 0 --list-failures | sed 's/^/  /'
done

echo "--- 3. HTTP server at fp16: max batch x requests in the system"
serve() {  # serve MAX_BATCH WORKERS
  docker rm -f server >/dev/null 2>&1 || true
  docker run -d --name server --gpus all --read-only --user 10001 --cap-drop ALL --tmpfs /tmp:rw,nosuid,nodev,size=2048m \
    --shm-size 1024m -p 8080:8080 -e PRECISION=fp16 -e MAX_BATCH="$1" -e WORKERS="$2" -e MAX_PENDING=256 "$IMAGE" >/dev/null
  for i in $(seq 1 60); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8080/health || true)" = 200 ] && break
    sleep 2
  done
}
load() {  # load IN_SYSTEM
  python3 bench/load_closed.py --images "$WORK/practice" --url http://127.0.0.1:8080 --workers "$1" --seconds 15
}
for config in "1 1:1 2 3" "2 1:2 3 4" "3 1:3 4 6" "4 1:4 5 6 8" "1 2:2 3 4" "2 2:4 6"; do
  settings="${config%%:*}"; set -- $settings
  serve "$1" "$2"
  echo "[max batch $1, batch threads $2]"
  for n in ${config##*:}; do load "$n"; done
done
echo "JOB DONE"
