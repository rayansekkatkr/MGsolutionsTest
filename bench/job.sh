#!/usr/bin/env bash
# The experiment currently being run on the GPU runner. Edited for each measurement.
# This one: the task as it will run (inference server at fp16 behind the proxy configured by
# exam/template.yaml) under open-loop load with the 1,000 practice photos, every answer checked
# against the reference results. The load generator shares the host's 4 vCPUs with the server,
# so the figures are on the low side.
set -euo pipefail
cd "$(dirname "$0")/.."
IMAGE=pose-inference:gpu
WORK="$HOME/exam-gpu"   # outside the checkout, which is wiped at the start of every run
REF="$WORK/practice_reference.jsonl"
[ "$(wc -l < "$REF")" = 1000 ] && [ "$(find "$WORK/practice" -name '*.jpg' | wc -l)" = 1000 ]
docker build -q -t "$IMAGE" inference >/dev/null
trap 'docker rm -f server proxy >/dev/null 2>&1 || true' EXIT

mkdir -p "$WORK/render"
docker run --rm --user "$(id -u):$(id -g)" -v "$PWD:/repo:ro" -v "$WORK/render:/out" "$IMAGE" \
  python /repo/tests/admission/render.py /repo/exam/template.yaml /repo/exam/images.json /out
docker rm -f proxy >/dev/null 2>&1 || true
docker run -d --name proxy --network host --init --read-only --user "$(cat "$WORK/render/user.txt")" --cap-drop ALL \
  --tmpfs /tmp:rw,nosuid,nodev,size=16m --ulimit nofile=65536:65536 -e HAPROXY_CFG="$(cat "$WORK/render/haproxy.cfg")" \
  --entrypoint sh "$(cat "$WORK/render/image.txt")" -c "$(cat "$WORK/render/command.txt")" >/dev/null

serve() {  # serve MAX_BATCH
  docker rm -f server >/dev/null 2>&1 || true
  docker run -d --name server --gpus all --read-only --user 10001 --cap-drop ALL --tmpfs /tmp:rw,nosuid,nodev,size=2048m \
    --shm-size 1024m -p 127.0.0.1:8080:8080 -e PRECISION=fp16 -e MAX_BATCH="$1" "$IMAGE" >/dev/null
  for i in $(seq 1 90); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8081/health || true)" = 200 ] && break
    sleep 2
  done
  echo "[max batch $1] healthy through the proxy after about $((i * 2)) s"
}
load() { python3 bench/load_open.py --images "$WORK/practice" --reference "$REF" --url http://127.0.0.1:8081 "$@" || true; }

echo "--- 1. template defaults: accuracy through the whole path, then each stage's share for one host"
serve "$(cat "$WORK/render/max_batch.txt")"
load --rate 12 --seconds 84 --max-wrong-share 0.02
load --rate 2.1 --seconds 20
load --rate 21 --seconds 30
load --rate 208 --seconds 40 &
sleep 25
echo "  mid-run: GPU $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader), host CPU idle $(top -bn1 | awk '/Cpu\(s\)/ {print $8}')%"
wait
load --rate 750 --seconds 40
echo "  quiet for 25 s, then the fleet signal:"
sleep 25
docker logs server 2>&1 | grep -E "ready|fleet|last 30 s" | tail -12

echo "--- 2. other batch sizes at one host's share of stage 4"
for batch in 2 4; do
  serve "$batch"
  load --rate 21 --seconds 20
  load --rate 208 --seconds 30
done
echo "JOB DONE"
