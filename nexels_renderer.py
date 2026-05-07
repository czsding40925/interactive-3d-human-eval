#!/usr/bin/env python3
"""
nexels_renderer.py — Nexels analogue of interactive_renderer.py.

Loads a NexelModel directly from a saved checkpoint (point_cloud.ply +
tensor.pth + cameras.json), without needing the source COLMAP data, and
renders frames headlessly via the diff-nexel-rasterization CUDA backend.
Frames go straight to ffmpeg → RTSP → mediamtx; mouse / scene-swap input
arrives over WebSocket.

Usage:
  /opt/conda/envs/nexels/bin/python nexels_renderer.py \
    --scenes bicycle=/path/to/runs/small/bicycle ... \
    --width 1200 --height 800 --fps 30 \
    --rtsp rtsp://localhost:8554/nexel --ws-port 8766
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import subprocess
import sys
import threading
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

NEXELS_ROOT = Path("/home/ubuntu/repos/GS/nexels")
sys.path.insert(0, str(NEXELS_ROOT))

# Pull in NexelModel + the renderer entry-point.
from scene.nexel_model import NexelModel  # noqa: E402
from nexel_renderer import render as nexel_render  # noqa: E402
from utils.graphics_utils import (  # noqa: E402
    getProjectionMatrix,
    getIntrinsicsMatrix,
    focal2fov,
)
from scene.cameras import MiniCam  # noqa: E402

import websockets  # noqa: E402


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_cfg_args(model_path: Path) -> Namespace:
    """Read the cfg_args file (a Namespace repr) saved at training time."""
    with open(model_path / "cfg_args") as f:
        ns = eval(f.read())  # the file IS a Namespace literal
    return ns


def latest_iteration(model_path: Path) -> int:
    pc_dir = model_path / "point_cloud"
    iters = [
        int(d.name.split("_")[-1])
        for d in pc_dir.iterdir() if d.name.startswith("iteration_")
    ]
    return max(iters)


def load_nexel_model(model_path: Path) -> NexelModel:
    """Construct + load a NexelModel from a runs/<size>/<scene>/ directory."""
    cfg = load_cfg_args(model_path)
    nexel_settings = Namespace(
        log_hash_table_size=cfg.log_hash_table_size,
        texture_limit=cfg.texture_limit,
        num_levels=cfg.num_levels,
        minres=cfg.minres,
        maxres=cfg.maxres,
        mlp_hidden_dim=cfg.mlp_hidden_dim,
        mlp_output_dim=cfg.mlp_output_dim,
        mlp_input_bias=cfg.mlp_input_bias,
        num_layers=cfg.num_layers,
        cap_max_init=cfg.cap_max_init,
        cap_max_final=cfg.cap_max_final,
    )
    nexels = NexelModel(cfg.sh_degree, cfg.texture_sh_degree, nexel_settings)
    it = latest_iteration(model_path)
    nexels.load_ply(str(model_path / "point_cloud" / f"iteration_{it}" / "point_cloud.ply"))
    nexels.load_tensor(str(model_path / "tensor" / f"iteration_{it}" / "tensor.pth"))
    nexels.active_app_degree = nexels.max_sh_degree
    nexels.active_texture_app_degree = nexels.max_texture_app_degree
    return nexels


def load_cameras_json(model_path: Path) -> list[dict]:
    with open(model_path / "cameras.json") as f:
        return json.load(f)


# ── Pipeline (orientation/projection helpers) ─────────────────────────────────
def normalize(v):
    n = np.linalg.norm(v)
    return v / max(n, 1e-9)


def build_world_frame(world_up):
    """Two horizontal basis vectors (e1, e2) plus world_up, used to express
    the orbit's yaw/pitch in a frame aligned to the scene's vertical axis."""
    world_up = normalize(world_up)
    ref = np.array([0.0, 0.0, 1.0]) if abs(world_up[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = normalize(np.cross(world_up, ref))
    e2 = np.cross(world_up, e1)
    return e1, e2, world_up


def pos_to_spherical(pos, look_at, e1, e2, up):
    off = pos - look_at
    r = float(np.linalg.norm(off))
    h = float(np.dot(off, up))
    pitch = float(math.asin(max(-1.0, min(1.0, h / max(r, 1e-9)))))
    horiz = off - h * up
    yaw = float(math.atan2(np.dot(horiz, e2), np.dot(horiz, e1))) if np.linalg.norm(horiz) > 1e-6 else 0.0
    return yaw, pitch, r


def spherical_to_pos(yaw, pitch, r, look_at, e1, e2, up):
    horiz = math.cos(pitch) * (math.cos(yaw) * e1 + math.sin(yaw) * e2)
    return look_at + r * (horiz + math.sin(pitch) * up)


def look_at_c2w(pos, look_at, world_up):
    """OpenCV camera frame (x=right, y=down, z=forward)."""
    forward = normalize(look_at - pos)
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        right = np.cross(forward, np.array([1.0, 0.0, 0.0]))
    right = normalize(right)
    down = np.cross(forward, right)
    R = np.stack([right, down, forward], axis=1)
    C2W = np.eye(4)
    C2W[:3, :3] = R
    C2W[:3, 3] = pos
    return C2W


def make_minicam(C2W, fovx, fovy, W, H, znear=0.01, zfar=100.0):
    W2C = np.linalg.inv(C2W).astype(np.float32)
    world_view_transform = torch.tensor(W2C).transpose(0, 1).cuda()
    projection_matrix = getProjectionMatrix(znear, zfar, fovx, fovy).transpose(0, 1).cuda()
    full_proj_transform = (
        world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))
    ).squeeze(0)
    return MiniCam(W, H, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform)


# ── Per-scene metadata derived from cameras.json ──────────────────────────────
class SceneMeta:
    def __init__(self, model_path: Path, render_w: int, render_h: int):
        cams = load_cameras_json(model_path)
        positions = np.array([c["position"] for c in cams], dtype=np.float64)
        rotations = np.array([c["rotation"] for c in cams], dtype=np.float64)
        # cameras.json rotation is c2w; camera's local Y axis (=down in OpenCV)
        # in world coords is rot @ [0,1,0] = rot[:, 1]. World up ≈ -mean(down).
        cam_down = rotations[:, :, 1]
        world_up = -cam_down.mean(axis=0)
        self.world_up = normalize(world_up)
        self.e1, self.e2, _ = build_world_frame(self.world_up)
        self.look_at = positions.mean(axis=0)

        # Use cam 0 as the canonical starting view.
        c0 = cams[0]
        pos0 = np.array(c0["position"], dtype=np.float64)
        self.fovx = focal2fov(c0["fx"], c0["width"])
        self.fovy = focal2fov(c0["fy"], c0["height"])
        # match aspect by tweaking fovx if requested resolution differs.
        train_aspect = c0["width"] / c0["height"]
        render_aspect = render_w / render_h
        if abs(train_aspect - render_aspect) > 1e-3:
            # keep fovy, recompute fovx to match render aspect.
            self.fovx = 2.0 * math.atan(math.tan(self.fovy / 2.0) * render_aspect)
        self.yaw, self.pitch, self.radius = pos_to_spherical(
            pos0, self.look_at, self.e1, self.e2, self.world_up
        )


# ── Shared state ──────────────────────────────────────────────────────────────
class State:
    def __init__(self, scenes: dict[str, str], default_key: str):
        self.scenes = scenes        # {label: model_path}
        self.current_key = default_key
        self.want_swap_to: tuple[str, bool] | None = None
        # When True, render loop sleeps without rendering / pushing frames.
        # Toggled via WS {type:'pause'|'resume'}; used by combined viewer.
        self.paused: bool = False
        # Spherical orbit camera (populated once a scene is active)
        self.look_at = np.zeros(3)
        self.yaw = 0.0
        self.pitch = 0.0
        self.radius = 1.0
        self.fovx = math.radians(50.0)
        self.fovy = math.radians(35.0)
        self.world_up = np.array([0.0, 0.0, 1.0])
        self.e1 = np.array([1.0, 0.0, 0.0])
        self.e2 = np.array([0.0, 1.0, 0.0])
        self.lock = threading.Lock()


def adopt_scene(state: State, meta: SceneMeta):
    with state.lock:
        state.look_at = meta.look_at
        state.yaw = meta.yaw
        state.pitch = meta.pitch
        state.radius = meta.radius
        state.fovx = meta.fovx
        state.fovy = meta.fovy
        state.world_up = meta.world_up
        state.e1 = meta.e1
        state.e2 = meta.e2


# ── WebSocket input handler (mirrors interactive_renderer.py) ─────────────────
async def handle_ws(ws, state: State):
    last = {"nx": None, "ny": None, "buttons": 0}
    yaw_sens = 4.0
    pitch_sens = 2.0
    zoom_per_tick = 1.05
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
                state.yaw -= dx * yaw_sens
                state.pitch = max(-pitch_clamp, min(pitch_clamp, state.pitch + dy * pitch_sens))
                last["nx"] = d.get("nx", 0.5)
                last["ny"] = d.get("ny", 0.5)
            elif t == "scroll":
                delta = d.get("delta", 0)
                state.radius *= (zoom_per_tick ** delta)
                state.radius = max(0.05, min(50.0, state.radius))
            elif t == "swap":
                target = d.get("to")
                reset_view = bool(d.get("reset_view", False))
                if target in state.scenes:
                    state.want_swap_to = (target, reset_view)
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
def render_frame(state: State, model: NexelModel, pipe_ns, bg, W, H) -> np.ndarray:
    with state.lock:
        pos = spherical_to_pos(state.yaw, state.pitch, state.radius,
                               state.look_at, state.e1, state.e2, state.world_up)
        look = state.look_at
        wu = state.world_up
        fovx = state.fovx
        fovy = state.fovy

    C2W = look_at_c2w(pos, look, wu)
    cam = make_minicam(C2W, fovx, fovy, W, H)

    with torch.no_grad():
        out = nexel_render(0, False, cam, model, pipe_ns, bg, 1.0,
                           compute_geometric=False, use_vis_features=False,
                           override_settings=-1)
    img = out["render"]                 # (3, H, W)
    alpha = out["alpha"].permute((2, 0, 1))
    rgb = img + (1.0 - alpha) * bg[:, None, None]
    rgb = rgb.clamp(0.0, 1.0)
    rgb_u8 = (rgb.permute(1, 2, 0) * 255.0).to(torch.uint8).contiguous().cpu().numpy()  # (H, W, 3)
    rgba = np.concatenate([rgb_u8, np.full((H, W, 1), 255, np.uint8)], axis=-1)
    return rgba


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", required=True,
                    help="label=path entries, e.g. bicycle=/p/runs/small/bicycle")
    ap.add_argument("--default", default=None)
    ap.add_argument("--width", type=int, default=1200)
    ap.add_argument("--height", type=int, default=800)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--rtsp", default="rtsp://localhost:8554/nexel")
    ap.add_argument("--ws-port", type=int, default=8766)
    args = ap.parse_args()

    scenes: dict[str, str] = {}
    for kv in args.scenes:
        k, _, v = kv.partition("=")
        if not v:
            sys.exit(f"--scenes entries must be label=path, got: {kv}")
        scenes[k] = v
    default_key = args.default or next(iter(scenes))
    state = State(scenes, default_key)

    W, H = args.width, args.height

    # Pipeline / background — match render.py's defaults.
    pipe_ns = Namespace(
        convert_SHs_python=False, compute_cov3D_python=False, debug=False,
        grid_threshold_factor=1.0, no_texture_clamp=False,
        no_texture_antialiasing=False, bool_settings=0, override_settings=-1,
    )
    bg = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")

    print(f"[nexels] loading default scene '{default_key}' from {scenes[default_key]}", flush=True)
    model = load_nexel_model(Path(scenes[default_key]))
    meta = SceneMeta(Path(scenes[default_key]), W, H)
    adopt_scene(state, meta)
    print(f"[nexels] orbit init: look_at={state.look_at}, r={state.radius:.3f}", flush=True)

    # Warm-up render so first user-visible frame isn't a 5s-stutter.
    _ = render_frame(state, model, pipe_ns, bg, W, H)

    threading.Thread(target=run_ws_server, args=(state, args.ws_port), daemon=True).start()

    ffmpeg_cmd = [
        "/usr/bin/ffmpeg", "-loglevel", "warning",
        "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}",
        "-r", str(args.fps), "-use_wallclock_as_timestamps", "1",
        "-i", "pipe:0",
        "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
        "-pix_fmt", "yuv420p", "-g", str(args.fps),
        "-f", "rtsp", args.rtsp,
    ]
    print(f"[ffmpeg] {' '.join(ffmpeg_cmd)}", flush=True)
    ff = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    frame_t = 1.0 / args.fps
    try:
        while True:
            t0 = time.time()
            with state.lock:
                paused = state.paused
                swap_to = state.want_swap_to
                state.want_swap_to = None
            if paused:
                time.sleep(frame_t)
                continue
            if swap_to:
                target, reset_view = swap_to
                if target != state.current_key:
                    print(f"[nexels] swapping to {target}", flush=True)
                    # Drop previous; load new (single-cache to keep VRAM stable).
                    del model
                    torch.cuda.empty_cache()
                    model = load_nexel_model(Path(scenes[target]))
                    meta = SceneMeta(Path(scenes[target]), W, H)
                    state.current_key = target
                    adopt_scene(state, meta)
                elif reset_view:
                    meta = SceneMeta(Path(scenes[target]), W, H)
                    adopt_scene(state, meta)

            frame = render_frame(state, model, pipe_ns, bg, W, H)
            try:
                ff.stdin.write(frame.tobytes())
            except BrokenPipeError:
                print("[ffmpeg] pipe closed", flush=True)
                break
            sleep_for = frame_t - (time.time() - t0)
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
