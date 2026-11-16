#!/usr/bin/env bash
# Runs the proxy container exactly as the ECS task definition describes it (same image, user,
# read-only root filesystem, dropped capabilities, tmpfs, command and configuration) in front of
# a stand-in for the reference inference server, then applies each load stage's per-host share.
set -euo pipefail
cd "$(dirname "$0")"

python3 render.py ../../exam/template.yaml ../../exam/images.json out
IMAGE="$(cat out/image.txt)"

cleanup() {
  echo "--- proxy container logs"; docker logs proxy 2>&1 | tail -20 || true
  docker rm -f proxy >/dev/null 2>&1 || true
  kill "${STUB:-0}" 2>/dev/null || true
}
trap cleanup EXIT

docker pull -q "$IMAGE"
docker run --rm --entrypoint haproxy "$IMAGE" -v | head -1

# Stand-in model: one thread per request, no limit, 30 ms outside and 90 ms inside a lock.
python3 stub_server.py 8080 30 90 &
STUB=$!

docker run -d --name proxy --network host --init \
  --read-only --user "$(cat out/user.txt)" --cap-drop ALL \
  --tmpfs /tmp:rw,nosuid,nodev,size=16m --ulimit nofile=65536:65536 \
  -e HAPROXY_CFG="$(cat out/haproxy.cfg)" \
  --entrypoint sh "$IMAGE" -c "$(cat out/command.txt)" >/dev/null

echo "--- 1. the container starts and /health passes through the proxy"
for i in $(seq 1 20); do
  code="$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8081/health || true)"
  [ "$code" = 200 ] && break
  sleep 0.5
done
[ "$code" = 200 ] || { echo "FAILED: /health through the proxy returned $code"; exit 1; }
[ "$(docker inspect -f '{{.State.Running}}' proxy)" = true ] || { echo "FAILED: proxy container is not running"; exit 1; }
echo "ok"

echo "--- 2. baseline: the stand-in server alone at one host's share of stage 3 (collapse expected)"
python3 load.py 8080 21 20 || true
sleep 3

echo "--- 3. behind the proxy: stage 2, 3 and 5 shares per host"
python3 load.py 8081 2.1 15 --all-ok --max-p95-ms 1000
python3 load.py 8081 21 20 --no-timeouts --max-p95-ms 1500 --min-ok-per-s 7
python3 load.py 8081 750 15 --no-timeouts --max-p95-ms 1500 --min-ok-per-s 7

echo "--- 4. the proxy is still healthy after overload"
code="$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8081/health)"
[ "$code" = 200 ] || { echo "FAILED: /health after load returned $code"; exit 1; }
[ "$(docker inspect -f '{{.State.Running}}' proxy)" = true ] || { echo "FAILED: proxy container stopped"; exit 1; }
echo "ok"
echo "ALL CHECKS PASSED"
