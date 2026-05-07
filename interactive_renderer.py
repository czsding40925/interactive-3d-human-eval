#!/usr/bin/env python3
"""
interactive_renderer.py — pyngp-driven interactive renderer for the human eval.

Bypasses Xvfb / GL entirely:
  - Loads an i-NGP snapshot via pyngp.
  - Renders frames directly to CPU RGBA buffers.
  - Pipes raw frames to a child ffmpeg process that pushes RTSP to mediamtx.
  - Listens on a WebSocket for camera input (mouse drag / scroll) and snapshot
    swap commands.

Usage:
  python3 interactive_renderer.py \
    --snapshots large=/path/to/large/snap.ingp medium=/path/to/medium/snap.ingp \
    --width 640 --height 360 --fps 30 \
    --rtsp rtsp://localhost:8554/stream \
    --ws-port 8765
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

INGP_ROOT = Path("/home/ubuntu/repos/neural_fields/instant-ngp")
sys.path.append(str(INGP_ROOT / "build"))
import pyngp as ngp  # noqa: E402

import websockets  # noqa: E402


# ── State (shared between WS thread and render loop) ──────────────────────────
class State:
    def __init__(self, snapshots: dict[str, str], default_key: str):
        self.snapshots = snapshots                 # {label: path}
        self.current_key = default_key
        self.want_swap_to: str | None = None       # set by WS handler
        # When True, the render loop sleeps without rendering or pushing
        # frames. Toggled via WS {type:'pause'|'resume'}; used by the
        # combined viewer to free GPU during the convergence phase.
        self.paused: bool = False
        # Spherical orbit camera, populated once snapshot loads
        self.look = np.zeros(3, dtype=np.float32)
        self.yaw = 0.0
        self.pitch = 0.0
        self.radius = 1.0
        self.fov = 50.0
        self.scale = 1.0
        self.lock = threading.Lock()


def cartesian_from_spherical(yaw, pitch, radius):
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    return np.array([radius * cp * cy, radius * sp, radius * cp * sy], dtype=np.float32)


def look_at_matrix(pos, look, world_up=np.array([0.0, 1.0, 0.0], dtype=np.float32)):
    """Build i-NGP NeRF camera mat4x3 with col0=right, col1=down, col2=forward, col3=pos."""
    forward = look - pos
    n = np.linalg.norm(forward)
    if n < 1e-8:
        forward = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    else:
        forward = forward / n
    right = np.cross(forward, world_up)
    rn = np.linalg.norm(right)
    if rn < 1e-6:
        # forward nearly parallel to world_up — pick any orthogonal right
        right = np.cross(forward, np.array([1.0, 0.0, 0.0]))
        rn = np.linalg.norm(right)
    right = right / rn
    down = np.cross(forward, right)  # already unit length
    R = np.column_stack([right, down, forward]).astype(np.float32)
    m = np.zeros((3, 4), dtype=np.float32)
    m[:, :3] = R
    m[:, 3] = pos.astype(np.float32)
    return m


def init_camera_from_snapshot(testbed, state: State):
    testbed.set_camera_to_training_view(0)
    state.look = np.array(testbed.look_at, dtype=np.float32)
    cam_pos = np.array(testbed.camera_matrix, dtype=np.float32)[:, 3]
    off = cam_pos - state.look
    state.radius = float(np.linalg.norm(off))
    state.yaw = float(math.atan2(off[2], off[0]))
    state.pitch = float(math.asin(max(-1.0, min(1.0, off[1] / max(state.radius, 1e-6)))))
    state.fov = float(testbed.fov)
    state.scale = float(testbed.scale)


# ── WebSocket input handler ───────────────────────────────────────────────────
async def handle_ws(ws, state: State):
    last = {"nx": None, "ny": None, "buttons": 0}
    yaw_sens = 4.0   # radians per full-screen drag
    pitch_sens = 2.0
    zoom_per_tick = 1.05  # 5% per scroll click
    pitch_clamp = math.radians(85.0)

    async for raw in ws:
        try:
            d = json.loads(raw)
        except json.JSONDecodeError:
            continue
        t = d.get("type")
        with state.lock:
            if t == "mousedown":
                last["nx"] = d.get("nx", 0.5)
                last["ny"] = d.get("ny", 0.5)
                last["buttons"] = 1
            elif t == "mouseup":
                last["buttons"] = 0
            elif t == "mousemove":
                if not last["buttons"] or last["nx"] is None:
                    last["nx"] = d.get("nx", 0.5)
                    last["ny"] = d.get("ny", 0.5)
                    continue
                dx = d.get("nx", 0.5) - last["nx"]
                dy = d.get("ny", 0.5) - last["ny"]
                state.yaw   -= dx * yaw_sens
                state.pitch = max(-pitch_clamp, min(pitch_clamp, state.pitch + dy * pitch_sens))
                last["nx"] = d.get("nx", 0.5)
                last["ny"] = d.get("ny", 0.5)
            elif t == "scroll":
                delta = d.get("delta", 0)
                # +delta = scroll down = zoom out
                state.radius *= (zoom_per_tick ** delta)
                state.radius = max(0.1, min(20.0, state.radius))
            elif t == "swap":
                target = d.get("to")
                reset_view = bool(d.get("reset_view", False))
                if target in state.snapshots and target != state.current_key:
                    state.want_swap_to = (target, reset_view)
                elif target in state.snapshots and reset_view:
                    # Same snapshot but camera reset requested (e.g. start of pair).
                    state.want_swap_to = (target, True)
            elif t == "pause":
                state.paused = True
            elif t == "resume":
                state.paused = False


def run_ws_server(state: State, port: int):
    async def main():
        async def handler(ws):
            try:
                await handle_ws(ws, state)
            except websockets.exceptions.ConnectionClosed:
                pass
        async with websockets.serve(handler, "0.0.0.0", port):
            print(f"[ws] listening on :{port}", flush=True)
            await asyncio.Future()
    asyncio.run(main())


# ── Render loop ───────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshots", nargs="+", required=True,
                    help="label=path pairs, e.g. large=/p/to/large.ingp medium=/p/to/medium.ingp")
    ap.add_argument("--default", default=None, help="which snapshot to start with")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--rtsp", default="rtsp://localhost:8554/stream")
    ap.add_argument("--ws-port", type=int, default=8765)
    args = ap.parse_args()

    snapshots = {}
    for kv in args.snapshots:
        k, _, v = kv.partition("=")
        if not v:
            sys.exit(f"--snapshots entries must be label=path, got: {kv}")
        snapshots[k] = v
    default_key = args.default or next(iter(snapshots))
    state = State(snapshots, default_key)

    print(f"[ngp] loading {default_key}: {snapshots[default_key]}", flush=True)
    testbed = ngp.Testbed(ngp.TestbedMode.Nerf)
    testbed.load_snapshot(snapshots[default_key])
    init_camera_from_snapshot(testbed, state)
    print(f"[ngp] camera: look={state.look}, r={state.radius:.3f}, yaw={state.yaw:.2f}, pitch={state.pitch:.2f}", flush=True)

    # Start WebSocket thread
    threading.Thread(target=run_ws_server, args=(state, args.ws_port), daemon=True).start()

    # ffmpeg subprocess: rawvideo (rgba float? no — we'll send uint8 RGBA) → H264 → RTSP
    ffmpeg_cmd = [
        "/usr/bin/ffmpeg", "-loglevel", "warning",
        "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{args.width}x{args.height}",
        "-r", str(args.fps), "-use_wallclock_as_timestamps", "1",
        "-i", "pipe:0",
        "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
        "-pix_fmt", "yuv420p", "-g", str(args.fps),
        "-f", "rtsp", args.rtsp,
    ]
    print(f"[ffmpeg] {' '.join(ffmpeg_cmd)}", flush=True)
    ff = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    W, H = args.width, args.height
    frame_t = 1.0 / args.fps

    try:
        while True:
            t_frame_start = time.time()

            # Snapshot swap?
            with state.lock:
                paused = state.paused
                swap_to = state.want_swap_to
                state.want_swap_to = None
            if paused:
                time.sleep(frame_t)
                continue
            if swap_to:
                target, reset_view = swap_to
                print(f"[ngp] swapping to {target} (reset_view={reset_view})", flush=True)
                if target != state.current_key:
                    testbed.load_snapshot(snapshots[target])
                with state.lock:
                    state.current_key = target
                if reset_view:
                    # Cross-scene swap: re-init camera from the new snapshot's training view 0.
                    init_camera_from_snapshot(testbed, state)
                    print(f"[ngp] camera reset: look={state.look}, r={state.radius:.3f}", flush=True)
                else:
                    # Same scene, different model: keep orbit, refresh fov/scale.
                    with state.lock:
                        state.fov = float(testbed.fov)
                        state.scale = float(testbed.scale)

            # Compose camera matrix from current spherical state
            with state.lock:
                pos = state.look + cartesian_from_spherical(state.yaw, state.pitch, state.radius)
                m = look_at_matrix(pos, state.look)
                fov = state.fov
                scale = state.scale
            testbed.camera_matrix = m
            testbed.fov = fov
            testbed.scale = scale

            img = testbed.render(W, H, spp=1, linear=False)  # (H, W, 4) float32 in [0,1]
            img_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            try:
                ff.stdin.write(img_u8.tobytes())
            except BrokenPipeError:
                print("[ffmpeg] pipe closed, exiting render loop", flush=True)
                break

            # Pace to target fps (no sleeping if we're already too slow)
            sleep_for = frame_t - (time.time() - t_frame_start)
            if sleep_for > 0:
                time.sleep(sleep_for)
    finally:
        try:
            ff.stdin.close()
        except Exception:
            pass
        ff.wait(timeout=3)


if __name__ == "__main__":
    main()
