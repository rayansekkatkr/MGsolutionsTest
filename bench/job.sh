#!/usr/bin/env bash
# The experiment currently being run on the GPU runner. Edited for each measurement.
# This one: batch workers in parallel. One worker leaves the GPU idle while it prepares the next
# batch on the CPU; with two, one prepares while the other's batch is on the GPU.
#   1. data: practice photos and reference results (a new runner starts empty)
#   2. engines with three lanes (reused if the image build already made them), and their accuracy
#   3. the server behind the proxy under overload: workers x batch size x requests allowed to wait
# The load generator shares the host's 4 vCPUs with the server, so the figures are on the low side.
set -euo pipefail
cd "$(dirname "$0")/.."
BASE=pose-inference:base
WORK="$HOME/exam-gpu"   # outside the checkout, which is wiped at the start of every run
REF="$WORK/practice_reference.jsonl"
mkdir -p "$WORK/practice" "$WORK/trt3" "$WORK/render"
trap 'docker rm -f server proxy >/dev/null 2>&1 || true' EXIT

echo "--- 1. data"
xargs -P 8 -I{} sh -c '[ -s "$1/$(basename "$2")" ] || curl -fsS --retry 3 -o "$1/$(basename "$2")" "$2"' _ "$WORK/practice" {} \
  < tests/inference/images_practice.txt
echo "$(find "$WORK/practice" -name '*.jpg' | wc -l) practice photos"
docker build -q -t "$BASE" inference >/dev/null
RUN=(docker run --rm --gpus all -v "$WORK:/work" -v "$PWD/tests/inference:/tests:ro" -v "$PWD/bench:/bench:ro" -v "$PWD/inference:/src:ro"
     -e PYTHONPATH=/app:/opt/mmpose/projects/rtmpose3d)
[ "$(wc -l < "$REF" 2>/dev/null || echo 0)" = 1000 ] || "${RUN[@]}" --user "$(id -u):$(id -g)" "$BASE" \
  python /tests/reference_run.py --images /work/practice --out /work/practice_reference.jsonl | tail -1

echo "--- 2. engines with three lanes"
"${RUN[@]}" --user 0 "$BASE" bash -c '
  pip install -q --root-user-action=ignore onnx==1.16.2 tensorrt==8.6.1.post1 2>&1 | tail -3
  { [ -s /work/trt3/det_bu16.engine ] && [ -s /work/trt3/pose_16.engine ]; } || python /src/trt_build.py --out /work/trt3 --lanes 3 --variants det_bu16,pose_16 2>&1 | grep -v -i "warn\|meshgrid\|Loads checkpoint"
  rm -rf /work/trt3_out && mkdir -p /work/trt3_out
  python /bench/gpu_bench.py --images /work/practice --out /work/trt3_out --trt-dir /work/trt3 --batches 1,3 \
    --precisions "trt:det_bu16:pose_16" 2>&1 | grep -v -i "warn\|meshgrid\|Loads checkpoint"
  chown -R '"$(id -u):$(id -g)"' /work/trt3 /work/trt3_out'
for f in "$WORK"/trt3_out/*_b3.jsonl; do
  printf '%-36s ' "[$(basename "$f" .jsonl)]"
  python3 tests/inference/compare.py "$REF" "$f" --min-pass 0 | head -1
done

echo "--- 3. server behind the proxy, overload at 300 requests/s of which the proxy lets 150 through"
docker run --rm --user "$(id -u):$(id -g)" -v "$PWD:/repo:ro" -v "$WORK/render:/out" "$BASE" \
  python /repo/tests/admission/render.py /repo/exam/template.yaml /repo/exam/images.json /out
# 150/s is what one host would receive at most from the edge proxies if a host turns out to serve
# 75 to 85 photos/s; the template's own default is for the 52 photos/s measured with one worker.
sed 's/be_sess_rate(infer) ge [0-9]*/be_sess_rate(infer) ge 150/' "$WORK/render/haproxy.cfg" > "$WORK/render/haproxy_150.cfg"
# On a runner with 8 vCPUs the server gets exactly what a g6.xlarge has (two cores, four threads)
# and the load generator runs on the other cores, so it no longer takes CPU time from the server.
# On a 4-vCPU runner they share, and the figures are on the low side.
if [ "$(nproc)" -ge 8 ]; then
  echo "thread siblings of cpu0: $(cat /sys/devices/system/cpu/cpu0/topology/thread_siblings_list)"
  HOST4="0,1,4,5"; LOAD4="2,3,6,7"; HOST6="0,1,2,4,5,6"; LOAD6="3,7"
else
  HOST4="0-3"; LOAD4="0-3"; HOST6=""; LOAD6=""
fi
CONTEXT="$(mktemp -d)"
cp "$WORK/trt3/det_bu16.engine" "$CONTEXT/det.engine"; cp "$WORK/trt3/pose_16.engine" "$CONTEXT/pose.engine"
docker build -q -t pose-inference:lanes --build-arg "BASE=$BASE" -f inference/Dockerfile.engines "$CONTEXT" >/dev/null
rm -rf "$CONTEXT"

measure() {   # measure SERVER_CPUS LOADER_CPUS WORKERS BATCH WAITING DECODE_THREADS
  local cpus="$1" loader="$2" workers="$3" batch="$4" waiting="$5" threads="$6"
  docker rm -f server proxy >/dev/null 2>&1 || true
  docker run -d --name proxy --network host --init --read-only --user "$(cat "$WORK/render/user.txt")" --cap-drop ALL \
    --cpuset-cpus "$cpus" --tmpfs /tmp:rw,nosuid,nodev,size=16m --ulimit nofile=65536:65536 \
    -e HAPROXY_CFG="$(cat "$WORK/render/haproxy_150.cfg")" \
    --entrypoint sh "$(cat "$WORK/render/image.txt")" -c "$(cat "$WORK/render/command.txt")" >/dev/null
  docker run -d --name server --gpus all --read-only --user 10001 --cap-drop ALL --tmpfs /tmp:rw,nosuid,nodev,size=2048m \
    --cpuset-cpus "$cpus" --shm-size 1024m -p 127.0.0.1:8080:8080 -e PRECISION=fp16 -e WORKERS="$workers" -e MAX_BATCH="$batch" \
    -e MAX_WAITING="$waiting" -e DECODE_THREADS="$threads" pose-inference:lanes >/dev/null
  for i in $(seq 1 90); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8081/health || true)" = 200 ] && break
    sleep 2
  done
  taskset -c "$loader" python3 bench/load_open.py --images "$WORK/practice" --reference "$REF" --url http://127.0.0.1:8081 \
    --rate 300 --seconds 30 --procs 2 --max-wrong-share 1 > "$WORK/load.txt" &
  sleep 12
  # GPU use averaged over ten seconds in the middle of the load; CPU use of the server container alone.
  GPU="$(timeout 10 nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -lms 500 | awk '{s+=$1; n++} END {printf "%.0f", s/n}')"
  CPU="$(docker stats --no-stream --format '{{.CPUPerc}}' server)"
  wait || true
  echo "[cpus=$cpus workers=$workers batch=$batch waiting=$waiting] $(head -1 "$WORK/load.txt" | sed -E 's/^rate [^)]*\) +sent=[0-9]+ +//')  GPU ${GPU}%, server CPU $CPU"
  docker logs server 2>&1 | grep -E "stays in PyTorch|last 30 s" | tail -1
}

# workers:batch:waiting. The first line is the configuration rehearsed on AWS (51.7 photos/s per host there).
echo "--- 3a. the server on a g6.xlarge's CPUs"
for config in 1:3:4 1:2:3 2:1:2 2:2:3 2:3:4 2:2:5 2:3:7 3:1:2 3:2:3 3:3:4 3:2:5 3:3:7; do
  IFS=: read -r workers batch waiting <<< "$config"
  measure "$HOST4" "$LOAD4" "$workers" "$batch" "$waiting" 3
done
if [ -n "$HOST6" ]; then
  echo "--- 3b. the same server with half as many CPUs again: is the CPU what holds the GPU back?"
  for config in 1:3:4 2:3:4 3:2:3 3:3:4 3:3:7; do
    IFS=: read -r workers batch waiting <<< "$config"
    measure "$HOST6" "$LOAD6" "$workers" "$batch" "$waiting" 5
  done
fi
echo "JOB DONE"
