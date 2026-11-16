#!/usr/bin/env bash
# Downloads the photos listed in a file (one URL per line) into a folder, skipping those already there.
# Usage: bash fetch_images.sh LIST DIR
set -euo pipefail
mkdir -p "$2"
while read -r url; do
  [ -s "$2/$(basename "$url")" ] || curl -fsS --retry 3 -o "$2/$(basename "$url")" "$url"
done < "$1"
echo "$(find "$2" -name '*.jpg' | wc -l) photos in $2"
