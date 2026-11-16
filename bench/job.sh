#!/usr/bin/env bash
# The experiment currently being run on the GPU runner. Edited for each measurement.
# This one: why the GPU is only 60% busy. One process cannot do its Python work on two CPUs at
# once, and idle threads of the numeric libraries spin. Measured here:
#   1. data (a new runner starts empty) and the image with three-lane engines
#   2. how a batch's time splits between the GPU and the CPU around it
#   3. the server under overload: library threads quietened, batch workers, and several server
#      processes sharing the GPU behind the proxy
set -euo pipefail
cd "$(dirname "$0")/.."
BASE=pose-inference:base
WORK="$HOME/exam-gpu"   # outside the checkout, which is wiped at the start of every run
REF="$WORK/practice_reference.jsonl"
mkdir -p "$WORK/practice" "$WORK/trt3" "$WORK/render"
trap 'docker rm -f server1 server2 server3 proxy >/dev/null 2>&1 || true' EXIT

echo "--- 1. data"
xargs -P 8 -I{} sh -c '[ -s "$1/$(basename "$2")" ] || curl -fsS --retry 3 -o "$1/$(basename "$2")" "$2"' _ "$WORK/practice" {} \
  < tests/inference/images_practice.txt
echo "$(find "$WORK/practice" -name '*.jpg' | wc -l) practice photos"
docker build -q -t "$BASE" inference >/dev/null
RUN=(docker run --rm --gpus all -v "$WORK:/work" -v "$PWD/tests/inference:/tests:ro" -v "$PWD/bench:/bench:ro" -v "$PWD/inference:/src:ro"
     -e PYTHONPATH=/app:/opt/mmpose/projects/rtmpose3d)
[ "$(wc -l < "$REF" 2>/dev/null || echo 0)" = 1000 ] || "${RUN[@]}" --user "$(id -u):$(id -g)" "$BASE" \
  python /tests/reference_run.py --images /work/practice --out /work/practice_reference.jsonl | tail -1

echo "--- 2. image with three-lane engines, and where a batch's time goes"
"${RUN[@]}" --user 0 "$BASE" bash -c '
  { [ -s /work/trt3/det_bu16.engine ] && [ -s /work/trt3/pose_16.engine ]; } || {
    pip install -q --root-user-action=ignore onnx==1.16.2 tensorrt==8.6.1.post1 2>&1 | tail -3
    python /src/trt_build.py --out /work/trt3 --lanes 3 --variants det_bu16,pose_16 2>&1 | grep -v -i "warn\|meshgrid\|Loads checkpoint"
    chown -R '"$(id -u):$(id -g)"' /work/trt3; }'
CONTEXT="$(mktemp -d)"
cp "$WORK/trt3/det_bu16.engine" "$CONTEXT/det.engine"; cp "$WORK/trt3/pose_16.engine" "$CONTEXT/pose.engine"
docker build -q -t pose-inference:lanes --build-arg "BASE=$BASE" -f inference/Dockerfile.engines "$CONTEXT" >/dev/null
rm -rf "$CONTEXT"
# The server gets what a g6.xlarge has (two cores, four threads); the load generator runs on the other cores.
if [ "$(nproc)" -ge 8 ]; then HOST4="0,1,4,5"; LOAD4="2,3,6,7"; else HOST4="0-3"; LOAD4="0-3"; fi
QUIET=(-e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e NUMEXPR_NUM_THREADS=1 -e KMP_BLOCKTIME=0
       -e OMP_WAIT_POLICY=passive -e OPENCV_FOR_THREADS_NUM=1)
probe() { docker run --rm --gpus all --cpuset-cpus "$HOST4" -v "$WORK/practice:/photos:ro" -v "$PWD/bench:/bench:ro" "$@" \
            pose-inference:lanes python /bench/phase_probe.py --images /photos 2>&1 | grep -v -i "warn\|meshgrid\|Loads checkpoint\|relative difference" | tail -4 || true; }
echo "library threads as they are:"; probe
echo "library threads quietened:";  probe "${QUIET[@]}"

echo "--- 3. the server behind the proxy, overload at 300 requests/s of which the proxy lets 150 through"
docker run --rm --user "$(id -u):$(id -g)" -v "$PWD:/repo:ro" -v "$WORK/render:/out" "$BASE" \
  python /repo/tests/admission/render.py /repo/exam/template.yaml /repo/exam/images.json /out
sed 's/be_sess_rate(infer) ge [0-9]*/be_sess_rate(infer) ge 150/' "$WORK/render/haproxy.cfg" > "$WORK/render/haproxy_150.cfg"

measure() {   # measure PROCESSES WORKERS BATCH WAITING quiet|plain [threads]
  local processes="$1" workers="$2" batch="$3" waiting="$4" mode="$5" show="${6:-}"
  local extra=(); [ "$mode" = quiet ] && extra=("${QUIET[@]}")
  docker rm -f server1 server2 server3 proxy >/dev/null 2>&1 || true
  local servers="" names=()
  for i in $(seq 1 "$processes"); do
    local port=$((8070 + 10 * i))   # 8080, 8090, 8100
    servers="$servers    server model$i 127.0.0.1:$port maxconn 32\n"
    names+=("server$i")
    docker run -d --name "server$i" --gpus all --read-only --user 10001 --cap-drop ALL --tmpfs /tmp:rw,nosuid,nodev,size=2048m \
      --cpuset-cpus "$HOST4" --shm-size 1024m -p "127.0.0.1:$port:8080" -e PRECISION=fp16 -e WORKERS="$workers" -e MAX_BATCH="$batch" \
      -e MAX_WAITING="$waiting" "${extra[@]}" pose-inference:lanes >/dev/null
  done
  sed "s/^    server model 127.0.0.1:8080 maxconn 32$/${servers%\\n}/" "$WORK/render/haproxy_150.cfg" > "$WORK/render/haproxy_n.cfg"
  docker run -d --name proxy --network host --init --read-only --user "$(cat "$WORK/render/user.txt")" --cap-drop ALL \
    --cpuset-cpus "$HOST4" --tmpfs /tmp:rw,nosuid,nodev,size=16m --ulimit nofile=65536:65536 \
    -e HAPROXY_CFG="$(cat "$WORK/render/haproxy_n.cfg")" \
    --entrypoint sh "$(cat "$WORK/render/image.txt")" -c "$(cat "$WORK/render/command.txt")" >/dev/null
  for i in $(seq 1 "$processes"); do
    for _ in $(seq 1 90); do
      [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$((8070 + 10 * i))/health" || true)" = 200 ] && break
      sleep 2
    done
  done
  taskset -c "$LOAD4" python3 bench/load_open.py --images "$WORK/practice" --reference "$REF" --url http://127.0.0.1:8081 \
    --rate 300 --seconds 30 --procs 2 --max-wrong-share 1 > "$WORK/load.txt" &
  sleep 12
  GPU="$( (timeout 10 nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -lms 500 || true) | awk '{s+=$1; n++} END {if (n) printf "%.0f", s/n; else printf "?"}')"
  STATS="$(docker stats --no-stream --format '{{.CPUPerc}} {{.MemUsage}}' "${names[@]}" | awk '{c+=$1; m=m" "$2} END {printf "CPU %.0f%%, memory%s", c, m}' || echo '?')"
  GMEM="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader || echo '?')"
  if [ -n "$show" ]; then
    echo "   busiest threads of the server:"
    top -H -b -n 1 -p "$(docker inspect -f '{{.State.Pid}}' server1)" | sed -n '8,15p' | awk '{printf "     %s%% %s\n", $9, $12}' || true
  fi
  wait || true
  echo "[processes=$processes workers=$workers batch=$batch waiting=$waiting $mode] $(head -1 "$WORK/load.txt" | sed -E 's/^rate [^)]*\) +sent=[0-9]+ +//')  GPU ${GPU}%, $STATS, GPU memory $GMEM"
  for name in "${names[@]}"; do docker logs "$name" 2>&1 | grep -E "stays in PyTorch|last 30 s" | tail -1; done
}

measure 1 1 3 4 plain threads
measure 1 1 3 4 quiet threads
measure 1 2 3 4 quiet
measure 1 2 2 3 quiet
measure 1 3 2 3 quiet
measure 2 1 3 4 quiet
measure 2 1 2 3 quiet
measure 2 1 3 4 plain
measure 3 1 2 3 quiet
measure 3 1 3 4 quiet
measure 2 2 2 3 quiet
echo "JOB DONE"
