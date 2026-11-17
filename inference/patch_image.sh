#!/usr/bin/env bash
# Publishes the image named in inference/patch_base.txt with the current server code on top.
# Everything else in the image, the TensorRT engines first of all, stays byte for byte what was
# measured: /app/server.py and /app/admission.py are replaced by one more layer, written
# straight to the registry. No GPU and no download of the image are needed.
# Usage: GITHUB_TOKEN=... bash inference/patch_image.sh      needs crane
set -euo pipefail
cd "$(dirname "$0")/.."

BASE="$(tr -d '[:space:]' < inference/patch_base.txt)"
REMOTE="ghcr.io/${GITHUB_REPOSITORY_OWNER:-rayansekkatkr}/mgsolutionstest-inference"
TAG="$REMOTE:$(git rev-parse --short=12 HEAD)-server"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

mkdir -p "$WORK/app"
cp inference/server.py inference/admission.py "$WORK/app/"
chmod 644 "$WORK"/app/*.py
tar --owner=0 --group=0 --numeric-owner --mtime='2026-01-01 00:00:00' -C "$WORK" -cf "$WORK/layer.tar" app/server.py app/admission.py

echo "$GITHUB_TOKEN" | crane auth login ghcr.io -u "${GITHUB_ACTOR:-x}" --password-stdin
echo "--- base: $BASE"
crane append --base "$BASE" --new_layer "$WORK/layer.tar" --new_tag "$TAG"
DIGEST="$(crane digest "$TAG")"

# The result must be the base plus exactly one layer, and start the same way.
layers() { crane manifest "$1" | python3 -c 'import json, sys; print(" ".join(l["digest"] for l in json.load(sys.stdin)["layers"]))'; }
start() { crane config "$1" | python3 -c 'import json, sys; c = json.load(sys.stdin)["config"]; print(json.dumps({k: c.get(k) for k in ("Cmd", "Entrypoint", "Env", "User", "WorkingDir")}, sort_keys=True))'; }
BASE_LAYERS="$(layers "$BASE")"; NEW_LAYERS="$(layers "$TAG")"
[ "${NEW_LAYERS% *}" = "$BASE_LAYERS" ] || { echo "FAILED: the layers of the base are not all there, in order"; exit 1; }
[ "$(start "$BASE")" = "$(start "$TAG")" ] || { echo "FAILED: the image no longer starts the same way"; exit 1; }
echo "layers: $(echo "$BASE_LAYERS" | wc -w) of the base, unchanged, plus 1"
crane blob "$REMOTE@${NEW_LAYERS##* }" | tar -xzO app/server.py | cmp -s - inference/server.py \
  || { echo "FAILED: the added layer does not hold this commit's server.py"; exit 1; }
echo "the added layer holds this commit's server.py"
echo "PUBLISHED $REMOTE@$DIGEST"
