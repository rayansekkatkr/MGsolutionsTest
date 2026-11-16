#!/usr/bin/env bash
# Builds, checks and publishes the inference image with TensorRT engines. Runs on the GPU model the
# engines are for (the self-hosted L4 runner), because an engine is specific to the GPU it is built on.
#
#   1. base image from inference/Dockerfile (the same build the CPU workflow tests)
#   2. engines built with trt_build.py inside that image
#   3. final image = base + TensorRT + engines (Dockerfile.engines)
#   4. checks on the final image, through the proxy configured by exam/template.yaml:
#      the 1,000 practice photos against the reference results, then load
#   5. push, only if the share of answers within tolerance is high enough
#
# Usage: bash inference/build_engine_image.sh DET_VARIANT POSE_VARIANT [MIN_CORRECT_SHARE]   (names from trt_build.py)
#        REBUILD=1 forces new engines; otherwise engines already built on this machine are reused,
#        so that the engines published are the ones that were measured.
set -euo pipefail
cd "$(dirname "$0")/.."
DET="${1:?detector variant}"; POSE="${2:?pose variant}"
MIN_CORRECT="${3:-0.985}"
WORK="$HOME/exam-gpu"   # outside the checkout, which is wiped at the start of every run
REF="$WORK/practice_reference.jsonl"
BASE=pose-inference:base
FINAL=pose-inference:engines
REMOTE="ghcr.io/${GITHUB_REPOSITORY_OWNER:-rayansekkatkr}/mgsolutionstest-inference"
REMOTE="$(echo "$REMOTE" | tr '[:upper:]' '[:lower:]')"
TAG="$(git rev-parse --short=12 HEAD)-l4-${DET#det_}"

echo "--- 1. base image"
docker build -q -t "$BASE" inference >/dev/null

echo "--- 2. engines: $DET, $POSE"
mkdir -p "$WORK/trt"
if [[ "${REBUILD:-0}" == 1 || ! -s "$WORK/trt/$DET.engine" || ! -s "$WORK/trt/$POSE.engine" ]]; then
  docker run --rm --gpus all --user 0 -v "$WORK/trt:/work" -v "$PWD/inference:/src:ro" \
    -e PYTHONPATH=/app:/opt/mmpose/projects/rtmpose3d "$BASE" bash -c '
      pip install -q --root-user-action=ignore onnx==1.16.2 tensorrt==8.6.1.post1 2>&1 | tail -3
      python /src/trt_build.py --out /work --variants '"$DET,$POSE"' 2>&1 | grep -v -i "warn\|meshgrid\|Loads checkpoint"
      chown -R '"$(id -u):$(id -g)"' /work'
fi
ls -la "$WORK/trt/$DET.engine" "$WORK/trt/$POSE.engine"
sha256sum "$WORK/trt/$DET.engine" "$WORK/trt/$POSE.engine"

echo "--- 3. final image"
CONTEXT="$(mktemp -d)"
trap 'rm -rf "$CONTEXT"; docker rm -f server proxy >/dev/null 2>&1 || true' EXIT
cp "$WORK/trt/$DET.engine" "$CONTEXT/det.engine"
cp "$WORK/trt/$POSE.engine" "$CONTEXT/pose.engine"
docker build -q -t "$FINAL" --build-arg "BASE=$BASE" -f inference/Dockerfile.engines "$CONTEXT" >/dev/null
docker image inspect "$FINAL" --format 'size: {{.Size}} bytes'

echo "--- 4. checks on the final image"
mkdir -p "$WORK/render"
docker run --rm --user "$(id -u):$(id -g)" -v "$PWD:/repo:ro" -v "$WORK/render:/out" "$BASE" \
  python /repo/tests/admission/render.py /repo/exam/template.yaml /repo/exam/images.json /out
docker rm -f server proxy >/dev/null 2>&1 || true
docker run -d --name proxy --network host --init --read-only --user "$(cat "$WORK/render/user.txt")" --cap-drop ALL \
  --tmpfs /tmp:rw,nosuid,nodev,size=16m --ulimit nofile=65536:65536 -e HAPROXY_CFG="$(cat "$WORK/render/haproxy.cfg")" \
  --entrypoint sh "$(cat "$WORK/render/image.txt")" -c "$(cat "$WORK/render/command.txt")" >/dev/null
# Same constraints as the task definition: non-root, read-only root filesystem, no capabilities.
docker run -d --name server --gpus all --read-only --user 10001 --cap-drop ALL --tmpfs /tmp:rw,nosuid,nodev,size=2048m \
  --shm-size 1024m -p 127.0.0.1:8080:8080 -e PRECISION=fp16 -e MAX_BATCH="$(cat "$WORK/render/max_batch.txt")" "$FINAL" >/dev/null
for i in $(seq 1 90); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8081/health || true)" = 200 ] && break
  sleep 2
done
docker logs server 2>&1 | grep -E "engine|ready|PyTorch" || true
docker logs server 2>&1 | grep -q "detector=tensorrt pose=tensorrt" || { echo "FAILED: the server is not running both networks in TensorRT"; exit 1; }
load() { python3 bench/load_open.py --images "$WORK/practice" --reference "$REF" --url http://127.0.0.1:8081 "$@"; }
echo "accuracy on the practice photos, through the whole path:"
load --rate 12 --seconds 84 --max-wrong-share "$(python3 -c "print(1 - $MIN_CORRECT)")" --min-correct-share "$MIN_CORRECT"
echo "one host's share of stage 3, then of stages 4 and 5 (the load generator shares this host's CPUs):"
load --rate 21 --seconds 40 --max-wrong-share 0.03 || true
load --rate 208 --seconds 40 --procs 2 --max-wrong-share 0.03 || true
load --rate 750 --seconds 40 --procs 3 --max-wrong-share 0.03 || true
docker logs server 2>&1 | grep "last 30 s" | tail -4

echo "--- 5. publish"
if [[ -n "${GITHUB_TOKEN:-}" ]]; then
  echo "$GITHUB_TOKEN" | docker login ghcr.io -u "${GITHUB_ACTOR:-x}" --password-stdin >/dev/null
  docker tag "$FINAL" "$REMOTE:$TAG"
  docker push -q "$REMOTE:$TAG"
  PUBLISHED="$(docker image inspect "$REMOTE:$TAG" --format '{{index .RepoDigests 0}}')"
  echo "published: $PUBLISHED"
  docker logout ghcr.io >/dev/null
  if docker manifest inspect "$PUBLISHED" >/dev/null 2>&1; then echo "readable without logging in: yes"; else echo "readable without logging in: NO"; fi
else
  echo "no GITHUB_TOKEN: image built and checked, not pushed"
fi
echo "ENGINE IMAGE DONE"
