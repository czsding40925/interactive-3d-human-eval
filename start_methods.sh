#!/bin/bash
# start_methods.sh — i-NGP vs Nexels cross-method interactive eval pipeline.
#
# Brings up:
#   1. mediamtx                        (RTSP push + WHEP webrtc out)
#   2. interactive_renderer.py         (pyngp i-NGP, pushes rtsp://localhost:8554/ingp,  WS internal :8767)
#   3. nexels_renderer.py              (Nexels,      pushes rtsp://localhost:8554/nexel, WS internal :8768)
#   4. ws_mux.py                       (single browser-facing WS on :8765, demuxes by 'method' field)
#   5. serve.py                        (static + /submit, :8080)
#
# The mux exists because only port 8765 is open in the L4's AWS SG; both
# renderers' WSes ride on 8767/8768 internally and the mux fans messages
# in/out by `method: 'ingp'|'nexel'`.
#
# Browser opens viewer/cross_method.html.

set -e

REPO=/home/ubuntu/repos
STREAM=$REPO/interactive-3d-human-eval

# Pick the size tier: SIZE=small (default) or SIZE=medium.
# Only one tier can run at a time (single L4, single SG-open port set).
SIZE="${SIZE:-small}"
case "$SIZE" in
    small|medium) ;;
    *) echo "ERROR: SIZE must be 'small' or 'medium' (got '$SIZE')"; exit 1 ;;
esac

INGP_RUNS=$REPO/neural_fields/instant-ngp/experiments/runs/$SIZE
NEX_RUNS=$REPO/my_impl/CS348K_Project/runs/$SIZE

WIDTH=1440
HEIGHT=810
FPS=30

MEDIAMTX_BIN="${HOME}/mediamtx/mediamtx"
MEDIAMTX_CFG="$STREAM/mediamtx.yml"

INGP_PY=/home/ubuntu/repos/neural_fields/instant-ngp/build/pyngp.cpython-310-x86_64-linux-gnu.so
INGP_PYTHON=python3                              # whatever python imports pyngp built against
NEX_PYTHON=/opt/conda/envs/nexels/bin/python

# 7 mip-NeRF-360 scenes — both i-NGP small snapshots and Nexels small runs exist.
SCENES=(bicycle bonsai counter garden kitchen room stump)

# Build label=path lists.
INGP_ARGS=()
NEX_ARGS=()
for s in "${SCENES[@]}"; do
    INGP_ARGS+=("${s}=${INGP_RUNS}/${s}/snap.ingp")
    NEX_ARGS+=("${s}=${NEX_RUNS}/${s}")
done

cleanup() {
    echo ""
    echo "==> Shutting down..."
    kill $SERVE_PID  2>/dev/null || true
    kill $MUX_PID    2>/dev/null || true
    kill $NEX_PID    2>/dev/null || true
    kill $INGP_PID   2>/dev/null || true
    kill $MTX_PID    2>/dev/null || true
    echo "==> Done."
}
trap cleanup EXIT INT TERM

# Resolve current public IP (L4 rotates it on every stop/start) and patch
# mediamtx.yml's webrtcAdditionalHosts so the WHEP signal hands the browser
# a live ICE candidate. Without this, the page loads but video stays black.
PUBLIC_IP=$(curl -s -m 3 http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null \
            || curl -s -m 5 https://ifconfig.me 2>/dev/null \
            || echo "")
if [ -z "$PUBLIC_IP" ]; then
    echo "WARN: could not resolve public IP; mediamtx.yml not updated."
else
    echo "==> Patching mediamtx.yml webrtcAdditionalHosts → ['$PUBLIC_IP']"
    sed -i "s|^webrtcAdditionalHosts:.*|webrtcAdditionalHosts: ['${PUBLIC_IP}']|" "$MEDIAMTX_CFG"
fi

echo "==> [1/5] Starting mediamtx..."
"$MEDIAMTX_BIN" "$MEDIAMTX_CFG" > "$STREAM/mediamtx.log" 2>&1 &
MTX_PID=$!
sleep 1

echo "==> [2/5] Starting i-NGP renderer (pushes /ingp, internal WS :8767)..."
$INGP_PYTHON "$STREAM/interactive_renderer.py" \
    --snapshots "${INGP_ARGS[@]}" \
    --default bicycle \
    --width $WIDTH --height $HEIGHT --fps $FPS \
    --rtsp rtsp://localhost:8554/ingp \
    --ws-port 8767 \
    > "$STREAM/ingp_renderer.log" 2>&1 &
INGP_PID=$!

echo "==> [3/5] Starting Nexels renderer (pushes /nexel, internal WS :8768)..."
$NEX_PYTHON "$STREAM/nexels_renderer.py" \
    --scenes "${NEX_ARGS[@]}" \
    --default bicycle \
    --width $WIDTH --height $HEIGHT --fps $FPS \
    --rtsp rtsp://localhost:8554/nexel \
    --ws-port 8768 \
    > "$STREAM/nexels_renderer.log" 2>&1 &
NEX_PID=$!

echo "    Waiting 6s for both renderers to start streaming..."
sleep 6

echo "==> [4/5] Starting WS mux on :8765 (fans by 'method' field to 8767/8768)..."
python3 "$STREAM/ws_mux.py" --port 8765 \
    --ingp-ws  ws://localhost:8767 \
    --nexel-ws ws://localhost:8768 \
    > "$STREAM/ws_mux.log" 2>&1 &
MUX_PID=$!
sleep 1

echo "==> [5/5] Starting static + /submit server on :8080..."
python3 "$STREAM/serve.py" 8080 &
SERVE_PID=$!

VIEWER_URL="http://${PUBLIC_IP:-THIS_HOST}:8080/viewer/cross_method_side_by_side.html?size=${SIZE}"

echo ""
echo "========================================"
echo " Cross-method (i-NGP vs Nexels) eval is live!  [size=$SIZE]"
# OSC 8 hyperlink — clickable in iTerm2, VS Code, Windows Terminal, etc.
printf "  Viewer:        \e]8;;%s\e\\%s\e]8;;\e\\\n" "$VIEWER_URL" "$VIEWER_URL"
echo "  Browser WS:    ws://${PUBLIC_IP}:8765   (mux → ingp 8767 / nexel 8768)"
echo "  Logs:          $STREAM/{mediamtx,ingp_renderer,nexels_renderer,ws_mux}.log"
echo "========================================"
echo " Press Ctrl+C to stop everything."

wait $INGP_PID
