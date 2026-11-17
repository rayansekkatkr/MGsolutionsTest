#!/usr/bin/env bash
# Admission control, end to end, without a GPU:
#   - the proxy container exactly as the ECS task definition describes it (same image, user,
#     read-only root filesystem, dropped capabilities, tmpfs, command and configuration)
#   - the real inference/server.py and inference/admission.py behind it, with a stand-in engine
#     that takes as long as the real one
# and each load stage's per-host share applied to them, every answer checked.
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
  docker rm -f proxy >/dev/null 2>&1 || true
  [ -n "${SERVER:-}" ] && kill "$SERVER" 2>/dev/null || true
}
trap cleanup EXIT

# The server with the stand-in engine in place of engine.py, settings as in the task definition.
rm -rf "$OUT/app" && mkdir -p "$OUT/app"
cp ../../inference/server.py ../../inference/admission.py stand_in/engine.py "$OUT/app/"
MAX_BATCH="$(cat "$OUT/max_batch.txt")" MAX_WAITING="$(cat "$OUT/max_waiting.txt")" WORKERS="$(cat "$OUT/workers.txt")" DRAIN_S="$(cat "$OUT/drain_s.txt")" PRECISION=fp16 python3 "$OUT/app/server.py" > "$OUT/server.log" 2>&1 &
SERVER=$!

docker pull -q "$IMAGE"
docker run --rm --entrypoint haproxy "$IMAGE" -v | head -1
docker run -d --name proxy --network host --init \
  --read-only --user "$(cat "$OUT/user.txt")" --cap-drop ALL \
  --tmpfs /tmp:rw,nosuid,nodev,size=16m --ulimit nofile=65536:65536 \
  -e HAPROXY_CFG="$(cat "$OUT/haproxy.cfg")" -e DRAIN_S="$(cat "$OUT/drain_s.txt")" \
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
$LOAD http://127.0.0.1:8081 --rate 208 --seconds 20 --min-correct-per-s 40 --max-p99-ms 200
$LOAD http://127.0.0.1:8081 --rate 750 --seconds 20 --procs 2 --min-correct-per-s 40 --max-p99-ms 200

echo "--- 5. back to a light load right after the overload: every request answered again"
$LOAD http://127.0.0.1:8081 --rate 2.1 --seconds 10 --min-correct-share 1 --max-p99-ms 200

echo "--- 6. the proxy and the server are still healthy"
code="$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8081/health)"
[ "$code" = 200 ] || { echo "FAILED: /health after load returned $code"; exit 1; }
[ "$(docker inspect -f '{{.State.Running}}' proxy)" = true ] || { echo "FAILED: proxy container stopped"; exit 1; }
echo "ok"

echo "--- 7. told to stop: the health check fails at once, requests are still answered meanwhile"
# What a host or a task being stopped does: SIGTERM to both containers. The load balancer needs up
# to 10 s to take the task out; the requests it still routes here in that time must be answered.
$LOAD http://127.0.0.1:8081 --rate 20 --seconds 14 --windows --min-correct-share 0.99 > "$OUT/stop_load.txt" &
LOADER=$!
sleep 3
docker kill -s TERM proxy >/dev/null
kill -TERM "$SERVER"
STOP_AT=$SECONDS
sleep 1
code="$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8081/health)"
[ "$code" = 503 ] || { echo "FAILED: /health returned $code one second after the stop signal, expected 503"; exit 1; }
# A host shutting down may also stop the proxy process directly: the container must start it again.
docker exec proxy sh -c 'kill -TERM $(pidof haproxy)'
wait "$LOADER" || { cat "$OUT/stop_load.txt"; echo "FAILED: requests were lost while the task was stopping"; exit 1; }
head -1 "$OUT/stop_load.txt"
docker logs proxy 2>&1 | grep "stopping" || { echo "FAILED: the proxy did not report that it is stopping"; exit 1; }
for _ in $(seq 1 30); do
  if ! kill -0 "$SERVER" 2>/dev/null && [ "$(docker inspect -f '{{.State.Running}}' proxy)" = false ]; then break; fi
  sleep 1
done
kill -0 "$SERVER" 2>/dev/null && { echo "FAILED: the server did not stop by itself"; exit 1; }
[ "$(docker inspect -f '{{.State.Running}}' proxy)" = false ] || { echo "FAILED: the proxy did not stop by itself"; exit 1; }
echo "both stopped by themselves $((SECONDS - STOP_AT)) s after the signal"
echo "ALL CHECKS PASSED"
