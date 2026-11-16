#!/usr/bin/env bash
# Builds the inference image and checks, on CPU in fp32, that the batched server returns the same
# joints as the official per-photo APIs. It also starts the container the way the task definition
# does: non-root user, read-only root filesystem, no capabilities, only /tmp writable.
# Speed and fp16 need a GPU and are measured separately.
set -euo pipefail
cd "$(dirname "$0")/../.."
IMAGE="${IMAGE:-pose-inference:test}"
WORK="${WORK:-$PWD/tests/inference/out}"
mkdir -p "$WORK/images"

cleanup() {
  echo "--- server container logs"; docker logs server 2>&1 | grep -v "Loads checkpoint" | tail -25 || true
  docker rm -f server >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "--- 1. build"
docker build -q -t "$IMAGE" inference
docker image inspect "$IMAGE" --format 'image size: {{.Size}} bytes'

echo "--- 2. test photos (public COCO 2017 images)"
while read -r url; do
  [ -s "$WORK/images/$(basename "$url")" ] || curl -fsS --retry 3 -o "$WORK/images/$(basename "$url")" "$url"
done < tests/inference/images.txt
echo "$(ls "$WORK/images" | wc -l) photos"

echo "--- 3. reference: official APIs, one photo at a time"
docker run --rm --user "$(id -u):$(id -g)" -v "$WORK:/work" -v "$PWD/tests/inference:/tests:ro" \
  -e PYTHONPATH=/app:/opt/mmpose/projects/rtmpose3d "$IMAGE" \
  python /tests/reference_run.py --images /work/images --out /work/reference.jsonl | tail -3

echo "--- 4. server, started as the task definition starts it"
# MAX_AGE_S is raised because a CPU needs seconds per photo; in service the 8 s default applies.
docker run -d --name server --read-only --user 10001 --cap-drop ALL --tmpfs /tmp:rw,nosuid,nodev,size=2048m \
  --shm-size 1024m -p 8080:8080 -e MAX_BATCH=4 -e MAX_AGE_S=600 "$IMAGE" >/dev/null
for i in $(seq 1 120); do
  code="$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8080/health || true)"
  [ "$code" = 200 ] && break
  [ "$(docker inspect -f '{{.State.Running}}' server)" = true ] || { echo "FAILED: server container exited"; exit 1; }
  sleep 5
done
[ "$code" = 200 ] || { echo "FAILED: /health never returned 200 (last: $code)"; exit 1; }
echo "healthy after about $((i * 5)) s"

echo "--- 5. the same photos through the server, several at a time"
python3 tests/inference/client.py --images "$WORK/images" --url http://127.0.0.1:8080 --out "$WORK/server.jsonl" --concurrency 8

echo "--- 6. comparison (fp32 against fp32: the results must be practically identical)"
python3 tests/inference/compare.py "$WORK/reference.jsonl" "$WORK/server.jsonl" --min-pass 1.0 --max-2d 0.05 --max-3d 0.5

echo "--- 7. a body that is not an image gets a 400, and the server keeps working"
code="$(curl -s -o /dev/null -w '%{http_code}' -X POST -H 'X-Request-Id: bad' --data-binary 'not a jpeg' http://127.0.0.1:8080/infer)"
[ "$code" = 400 ] || { echo "FAILED: expected 400, got $code"; exit 1; }
[ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8080/health)" = 200 ] || { echo "FAILED: unhealthy after a bad request"; exit 1; }
echo "ok"
echo "ALL CHECKS PASSED"
