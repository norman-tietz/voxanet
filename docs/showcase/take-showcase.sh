#!/usr/bin/env bash
# Re-shoots the README screenshot (resources/planet.png): applies the throwaway capture hooks
# (showcase.patch), builds, poses the camera, captures at a fixed game time, then removes the hooks
# again. See docs/showcase/README.md for what the arrangement is and how to adjust it.
#
# usage: docs/showcase/take-showcase.sh [out.png]   (default: resources/planet.png)
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

SEED=25          # galaxy: Yellow star, Earth-like #1 with volcanic #2 nearby
SHOT_T=1856.5    # game clock (s) of the shot: volcanic #2 just behind #1's limb
POSE="2.6 86 25" # <distance in planet radii> <sun angle from view, deg> <volcano angle from view, deg>
OUT="${1:-resources/planet.png}"
RAW="$(mktemp -d)/showcase_raw.png"
LOG="$(dirname "$RAW")/run.log"

for f in src/main.rs src/renderer.rs src/screenshot.rs; do
    git diff --quiet -- "$f" || { echo "$f has uncommitted changes; commit or stash them first" >&2; exit 1; }
done
pgrep -x voxanet >/dev/null && { echo "voxanet is already running; stop it first (pkill -x voxanet)" >&2; exit 1; }

git apply docs/showcase/showcase.patch
trap 'pkill -x voxanet || true; git apply -R docs/showcase/showcase.patch' EXIT

cargo build --release
# start the clock 60 s before the shot, leaving time for the bake and for streaming to settle
VOXANET_CLOCK=$(python3 -c "print($SHOT_T - 60)") VOXANET_SHOWCASE="$POSE" \
    ./target/release/voxanet --seed "$SEED" > "$LOG" 2>&1 &

# the hook prints "DBG showcase t=<game time> ..." every 2 s; capture once the shot time is reached
until grep -o "DBG showcase t=[0-9.]*" "$LOG" 2>/dev/null | tail -1 \
        | awk -F= -v T="$SHOT_T" '{ exit !($2 + 0 >= T - 0.1) }'; do
    sleep 0.2
    pgrep -x voxanet >/dev/null || { echo "voxanet exited, see $LOG" >&2; exit 1; }
done
grep "DBG showcase" "$LOG" | tail -1
echo "$RAW" > /tmp/voxanet_screenshot.trigger.tmp && mv /tmp/voxanet_screenshot.trigger.tmp /tmp/voxanet_screenshot.trigger
until grep -q "screenshot saved to $RAW" "$LOG"; do sleep 0.5; done

# the engine writes uncompressed RGBA (~33 MB at 4K); store it as a compressed RGB PNG
python3 -c "from PIL import Image; Image.open('$RAW').convert('RGB').save('$OUT', optimize=True)"
echo "saved $OUT"
