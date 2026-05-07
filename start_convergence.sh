#!/bin/bash
# start_convergence.sh — interactive 3DGS convergence eval pipeline.
#
# Brings up:
#   1. mediamtx                (RTSP push + WHEP webrtc out)
#   2. gsplat_renderer.py      (3DGS, pushes rtsp://localhost:8554/3dgs, WS internal :8769)
#   3. ws_mux.py               (single browser-facing WS on :8765, routes 'method:3dgs')
#   4. serve.py                (static + /submit, :8080)
#
# Browser opens viewer/convergence_interactive.html.

set -e

REPO=/home/ubuntu/repos
STREAM=$REPO/interactive-3d-human-eval
RUNS=$STREAM/runs_3dgs

SCENES=(truck train)
TANDT=$REPO/data/tandt_db/tandt

# Both scenes are loaded at once (8 ckpts on GPU). DEFAULT picks the
# scene:label the renderer starts on; client `swap` messages drive everything
# afterwards.
DEFAULT="${DEFAULT:-truck:30k}"

WIDTH=1920
HEIGHT=1080
FPS=60

MEDIAMTX_BIN="${HOME}/mediamtx/mediamtx"
MEDIAMTX_CFG="$STREAM/mediamtx.yml"

GSPLAT_PYTHON=/opt/conda/envs/gsplat/bin/python

cleanup() {
    echo ""
    echo "==> Shutting down..."
    kill $SERVE_PID  2>/dev/null || true
    kill $MUX_PID    2>/dev/null || true
    kill $GSP_PID    2>/dev/null || true
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

echo "==> [1/4] Starting mediamtx..."
"$MEDIAMTX_BIN" "$MEDIAMTX_CFG" > "$STREAM/mediamtx.log" 2>&1 &
MTX_PID=$!
sleep 1

SCENE_DATA_ARGS=()
CKPT_ARGS=()
for s in "${SCENES[@]}"; do
    SCENE_DATA_ARGS+=("${s}=${TANDT}/${s}")
    CKPT_ARGS+=("${s}:1k=${RUNS}/${s}/ckpts/ckpt_999_rank0.pt")
    CKPT_ARGS+=("${s}:3k=${RUNS}/${s}/ckpts/ckpt_2999_rank0.pt")
    CKPT_ARGS+=("${s}:7k=${RUNS}/${s}/ckpts/ckpt_6999_rank0.pt")
    CKPT_ARGS+=("${s}:30k=${RUNS}/${s}/ckpts/ckpt_29999_rank0.pt")
done

echo "==> [2/4] Starting gsplat renderer (scenes=${SCENES[*]}, $((${#SCENES[@]} * 4)) ckpts, /3dgs, WS :8769)..."
$GSPLAT_PYTHON "$STREAM/gsplat_renderer.py" \
    --scene-data "${SCENE_DATA_ARGS[@]}" \
    --ckpts      "${CKPT_ARGS[@]}" \
    --default "$DEFAULT" \
    --width $WIDTH --height $HEIGHT --fps $FPS \
    --rtsp rtsp://localhost:8554/3dgs \
    --ws-port 8769 \
    > "$STREAM/gsplat_renderer.log" 2>&1 &
GSP_PID=$!

echo "    Waiting 12s for renderer to load all ckpts and start streaming..."
sleep 12

echo "==> [3/4] Starting WS mux on :8765 (routes 'method:3dgs' → :8769)..."
python3 "$STREAM/ws_mux.py" --port 8765 \
    --3dgs-ws ws://localhost:8769 \
    --default-method 3dgs \
    > "$STREAM/ws_mux.log" 2>&1 &
MUX_PID=$!
sleep 1

echo "==> [4/4] Starting static + /submit server on :8080..."
python3 "$STREAM/serve.py" 8080 &
SERVE_PID=$!

VIEWER_URL="http://${PUBLIC_IP:-THIS_HOST}:8080/viewer/convergence_interactive.html"

echo ""
echo "========================================"
echo " Convergence (3DGS, scenes=${SCENES[*]}, default=$DEFAULT) eval is live!"
printf "  Viewer:        \e]8;;%s\e\\%s\e]8;;\e\\\n" "$VIEWER_URL" "$VIEWER_URL"
echo "  Browser WS:    ws://${PUBLIC_IP}:8765   (mux → 3dgs :8769)"
echo "  Logs:          $STREAM/{mediamtx,gsplat_renderer,ws_mux}.log"
echo "========================================"
echo " Press Ctrl+C to stop everything."

wait $GSP_PID
