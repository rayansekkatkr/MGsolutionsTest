#!/usr/bin/env bash
# Admission control and request recovery, end to end, without a GPU:
#   - the two proxy containers exactly as the ECS task definitions describe them (same image,
#     user, read-only root filesystem, dropped capabilities, command and configuration)
#   - the real inference/server.py and inference/admission.py, with a stand-in engine that takes
#     as long as the real one
# First one GPU task under each load stage's per-host share, then the whole chain (edge proxy,
# stand-in for the fleet ALB, two GPU tasks) with one task killed under load. Every answer is checked.
set -euo pipefail
cd "$(dirname "$0")"
OUT="$PWD/out"
LOAD="python3 ../../bench/load_open.py --images $OUT/photos --reference $OUT/reference.jsonl --url"

echo "--- 1. unit checks of the pen and of the fleet signal"
python3 test_admission.py

python3 render.py ../../exam/template.yaml ../../exam/images.json "$OUT"
python3 make_photos.py "$OUT"
IMAGE="$(cat "$OUT/image.txt")"

cleanup() {
  echo "--- server log"; tail -15 "$OUT/server.log" || true
  echo "--- proxy container logs"; docker logs proxy 2>&1 | tail -10 || true
  echo "--- edge container logs"; docker logs edge 2>&1 | tail -10 || true
  docker rm -f proxy edge fleet >/dev/null 2>&1 || true
  kill "${SERVER:-0}" "${SERVER_B:-0}" 2>/dev/null || true
}
trap cleanup EXIT

# The server with the stand-in engine in place of engine.py, settings as in the task definition.
rm -rf "$OUT/app" && mkdir -p "$OUT/app"
cp ../../inference/server.py ../../inference/admission.py stand_in/engine.py "$OUT/app/"
MAX_BATCH="$(cat "$OUT/max_batch.txt")" MAX_WAITING="$(cat "$OUT/max_waiting.txt")" PRECISION=fp16 python3 "$OUT/app/server.py" > "$OUT/server.log" 2>&1 &
SERVER=$!

docker pull -q "$IMAGE"
docker run --rm --entrypoint haproxy "$IMAGE" -v | head -1
docker run -d --name proxy --network host --init \
  --read-only --user "$(cat "$OUT/user.txt")" --cap-drop ALL \
  --tmpfs /tmp:rw,nosuid,nodev,size=16m --ulimit nofile=65536:65536 \
  -e HAPROXY_CFG="$(cat "$OUT/haproxy.cfg")" \
  --entrypoint sh "$IMAGE" -c "$(cat "$OUT/command.txt")" >/dev/null

echo "--- 2. the container starts and /health passes through the proxy"
for i in $(seq 1 20); do
  code="$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8081/health || true)"
  [ "$code" = 200 ] && break
  sleep 0.5
done
[ "$code" = 200 ] || { echo "FAILED: /health through the proxy returned $code"; exit 1; }
[ "$(docker inspect -f '{{.State.Running}}' proxy)" = true ] || { echo "FAILED: proxy container is not running"; exit 1; }
echo "ok"

echo "--- 3. one host's share of stages 1 to 3: every request answered"
$LOAD http://127.0.0.1:8081 --rate 0.21 --seconds 20 --min-correct-share 1
$LOAD http://127.0.0.1:8081 --rate 2.1 --seconds 20 --min-correct-share 1 --max-p99-ms 200
$LOAD http://127.0.0.1:8081 --rate 21 --seconds 30 --min-correct-share 0.995 --max-p99-ms 300

echo "--- 4. one host's share of stages 4 and 5: answers at capacity, and fast"
$LOAD http://127.0.0.1:8081 --rate 208 --seconds 20 --min-correct-per-s 55 --max-p99-ms 200
$LOAD http://127.0.0.1:8081 --rate 750 --seconds 20 --procs 2 --min-correct-per-s 55 --max-p99-ms 200

echo "--- 5. back to a light load right after the overload: every request answered again"
$LOAD http://127.0.0.1:8081 --rate 2.1 --seconds 10 --min-correct-share 1 --max-p99-ms 200

echo "--- 6. the proxy and the server are still healthy"
code="$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8081/health)"
[ "$code" = 200 ] || { echo "FAILED: /health after load returned $code"; exit 1; }
[ "$(docker inspect -f '{{.State.Running}}' proxy)" = true ] || { echo "FAILED: proxy container stopped"; exit 1; }
echo "ok"

echo "--- 7. the whole chain: edge proxy -> fleet ALB stand-in -> two GPU tasks"
# Second task: a server of its own on another port (the stand-in ALB calls it directly).
PORT=18080 MAX_BATCH="$(cat "$OUT/max_batch.txt")" MAX_WAITING="$(cat "$OUT/max_waiting.txt")" PRECISION=fp16 \
  python3 "$OUT/app/server.py" > "$OUT/server_b.log" 2>&1 &
SERVER_B=$!
docker run -d --name fleet --network host --user 99 -v "$PWD/fleet_alb.cfg:/fleet_alb.cfg:ro" \
  --entrypoint haproxy "$IMAGE" -db -f /fleet_alb.cfg >/dev/null
# The edge proxy as its Fargate task definition runs it: no writable directory at all.
docker run -d --name edge --network host --init --read-only --user "$(cat "$OUT/edge_user.txt")" --cap-drop ALL \
  --ulimit nofile=65536:65536 -e HAPROXY_CFG="$(sed 's/FLEET_ALB/127.0.0.1:8091/' "$OUT/edge.cfg")" \
  --entrypoint sh "$IMAGE" -c "$(cat "$OUT/edge_command.txt")" >/dev/null
for i in $(seq 1 20); do
  code="$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8082/health || true)"
  [ "$code" = 200 ] && break
  sleep 0.5
done
[ "$code" = 200 ] || { echo "FAILED: /health on the edge proxy returned $code"; exit 1; }
sleep 2
$LOAD http://127.0.0.1:8082 --rate 30 --seconds 15 --min-correct-share 1 --max-p99-ms 200

echo "--- 8. overload on the edge proxy: it lets its rate through and refuses the rest at once"
$LOAD http://127.0.0.1:8082 --rate 750 --seconds 20 --procs 2 --min-correct-per-s 85 --max-p99-ms 250

echo "--- 9. one of the two tasks is killed under load: no request may be lost"
$LOAD http://127.0.0.1:8082 --rate 30 --seconds 40 --windows --min-correct-share 1 &
LOADER=$!
sleep 12
kill -9 "$SERVER_B"
echo "killed the second task"
wait "$LOADER"
echo "ALL CHECKS PASSED"
