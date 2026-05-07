#!/usr/bin/env python3
"""
gsplat_renderer.py — interactive 3DGS renderer for the convergence study.

Loads N gsplat checkpoints of the same scene at different training iterations
(1k / 3k / 7k / 30k for the convergence demo). All ckpts are kept resident on
GPU so 'swap' messages between iters are instant — the smoothness slider can
then read the steady-state FPS difference between ckpts as a real signal.

Usage:
  /opt/conda/envs/gsplat/bin/python gsplat_renderer.py \
    --scene-data /home/ubuntu/repos/data/tandt_db/tandt/truck \
    --ckpts 1k=/.../runs_3dgs/truck/ckpts/ckpt_999_rank0.pt \
            3k=/.../ckpt_2999_rank0.pt \
            7k=/.../ckpt_6999_rank0.pt \
            30k=/.../ckpt_29999_rank0.pt \
    --width 1200 --height 675 --fps 30 \
    --rtsp rtsp://localhost:8554/3dgs --ws-port 8769

Mux convention: messages tagged `method: '3dgs'` are routed here by ws_mux.py.
Swap targets are the labels passed to --ckpts (e.g. '1k', '30k').
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
import torch

# gsplat lives in the conda env; rasterization is the only entry we need.
from gsplat.rendering import rasterization  # noqa: E402

# Borrow gsplat's COLMAP parser to derive scene-aligned camera frame & K matrix.
sys.path.insert(0, "/home/ubuntu/repos/GS/gsplat/examples")
from datasets.colmap import Parser  # noqa: E402

import websockets  # noqa: E402


# ── Helpers ───────────────────────────────────────────────────────────────────
def normalize(v):
    n = np.linalg.norm(v)
    return v / max(n, 1e-9)


def build_world_frame(world_up):
    """Two horizontal basis vectors (e1, e2) plus world_up — used to express
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
    """OpenCV camera frame (x=right, y=down, z=forward) — gsplat expects this."""
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


# ── Scene geometry derived from COLMAP cameras ───────────────────────────────
class SceneMeta:
    """Scene-frame basis + initial orbit pose, computed from COLMAP cameras
    (after gsplat's similarity normalisation, so it lives in the same space
    as the trained splats)."""

    def __init__(self, scene_data: Path, render_w: int, render_h: int):
        # Mirror the parser config used at training time.
        parser = Parser(str(scene_data), factor=1, normalize=True, test_every=8)
        c2ws = parser.camtoworlds.astype(np.float64)  # (N, 4, 4)
        positions = c2ws[:, :3, 3]
        # cam down (in world) ≈ rot @ [0, 1, 0] = rot[:, 1] for OpenCV-frame cams
        cam_down = c2ws[:, :3, 1]
        world_up = -cam_down.mean(axis=0)
        self.world_up = normalize(world_up)
        self.e1, self.e2, _ = build_world_frame(self.world_up)
        self.look_at = positions.mean(axis=0)

        # Use first training cam as canonical view.
        c0 = c2ws[0]
        pos0 = c0[:3, 3]
        self.yaw0, self.pitch0, self.radius0 = pos_to_spherical(
            pos0, self.look_at, self.e1, self.e2, self.world_up
        )

        # K from training; rescale to render resolution.
        K_train = list(parser.Ks_dict.values())[0].astype(np.float64)
        w_train, h_train = list(parser.imsize_dict.values())[0]
        # Match render aspect by scaling per-axis (small distortion if aspects
        # differ; pick render res close to native to minimise this).
        sx = render_w / w_train
        sy = render_h / h_train
        self.K = np.array([
            [K_train[0, 0] * sx, 0.0,                K_train[0, 2] * sx],
            [0.0,                K_train[1, 1] * sy, K_train[1, 2] * sy],
            [0.0,                0.0,                1.0],
        ], dtype=np.float64)


# ── Ckpt management ──────────────────────────────────────────────────────────
class Splats:
    """One ckpt's GPU-resident parameters, ready for rasterization()."""

    def __init__(self, ckpt_path: Path, device: str = "cuda"):
        sd = torch.load(str(ckpt_path), map_location=device, weights_only=True)
        self.step = int(sd.get("step", -1))
        s = sd["splats"]
        # Raw tensors — apply activations at render time (matches simple_trainer).
        self.means     = s["means"]      # (N, 3)
        self.quats     = s["quats"]      # (N, 4)
        self.scales    = s["scales"]     # (N, 3) log-scale
        self.opacities = s["opacities"]  # (N,) logit
        # SH coeffs concatenated for rasterization()
        self.colors    = torch.cat([s["sh0"], s["shN"]], dim=1)  # (N, 16, 3)
        self.n_splats  = self.means.shape[0]
        self.sh_degree = 3

    def rasterize(self, c2w: torch.Tensor, K: torch.Tensor, W: int, H: int):
        viewmats = torch.linalg.inv(c2w)[None]  # (1, 4, 4)
        Ks = K[None]
        renders, alphas, _ = rasterization(
            means=self.means,
            quats=self.quats,
            scales=torch.exp(self.scales),
            opacities=torch.sigmoid(self.opacities),
            colors=self.colors,
            viewmats=viewmats,
            Ks=Ks,
            width=W,
            height=H,
            sh_degree=self.sh_degree,
            packed=False,
            rasterize_mode="classic",
            render_mode="RGB",
        )
        return renders[0]  # (H, W, 3)


# ── Shared state ──────────────────────────────────────────────────────────────
class State:
    """Multi-scene state. Ckpts keyed by 'scene:label' (e.g. 'truck:30k');
    metas keyed by scene ('truck'). When a swap crosses scene boundaries,
    we also rotate the orbit basis + K matrix to match the new scene."""

    def __init__(self,
                 ckpts: dict[str, Splats],
                 metas: dict[str, SceneMeta],
                 K_ts: dict[str, torch.Tensor],
                 default_key: str):
        self.ckpts = ckpts
        self.metas = metas
        self.K_ts  = K_ts
        self.current_key = default_key
        self.want_swap_to: tuple[str, bool] | None = None
        # When True, the render loop sleeps without rendering or pushing
        # frames. Used by the combined viewer to free GPU during the
        # cross-method phase. Toggled via WS {type:'pause'|'resume'}.
        self.paused: bool = False
        # Spherical orbit camera, init to default scene's canonical view.
        m = metas[default_key.split(":")[0]]
        self.look_at = m.look_at.copy()
        self.yaw     = m.yaw0
        self.pitch   = m.pitch0
        self.radius  = m.radius0
        self.world_up = m.world_up
        self.e1 = m.e1
        self.e2 = m.e2
        self.lock = threading.Lock()

    def current_scene(self) -> str:
        return self.current_key.split(":")[0]


def reset_view(state: State):
    """Snap camera to the canonical view of the *current* scene (call after
    cross-scene swap or whenever the browser asks)."""
    with state.lock:
        m = state.metas[state.current_scene()]
        state.look_at = m.look_at.copy()
        state.yaw     = m.yaw0
        state.pitch   = m.pitch0
        state.radius  = m.radius0
        state.world_up = m.world_up
        state.e1 = m.e1
        state.e2 = m.e2


# ── WebSocket input handler (mirrors nexels_renderer.py) ─────────────────────
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
                rv = bool(d.get("reset_view", False))
                if target in state.ckpts:
                    state.want_swap_to = (target, rv)
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
def render_frame(state: State, W: int, H: int) -> np.ndarray:
    with state.lock:
        pos = spherical_to_pos(state.yaw, state.pitch, state.radius,
                               state.look_at, state.e1, state.e2, state.world_up)
        look = state.look_at
        wu = state.world_up
        key = state.current_key
        K_t = state.K_ts[state.current_scene()]

    C2W = look_at_c2w(pos, look, wu)
    c2w_t = torch.from_numpy(C2W).float().cuda()
    splats = state.ckpts[key]

    with torch.no_grad():
        rgb = splats.rasterize(c2w_t, K_t, W, H)  # (H, W, 3) float32 in [0, 1]
    rgb = rgb.clamp(0.0, 1.0)
    rgb_u8 = (rgb * 255.0).to(torch.uint8).contiguous().cpu().numpy()
    rgba = np.concatenate([rgb_u8, np.full((H, W, 1), 255, np.uint8)], axis=-1)
    return rgba


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-data", nargs="+", required=True,
                    help="scene=path entries, e.g. truck=/p/to/truck train=/p/to/train")
    ap.add_argument("--ckpts", nargs="+", required=True,
                    help="scene:label=path entries, e.g. truck:30k=/p/ckpt_29999_rank0.pt")
    ap.add_argument("--default", default=None,
                    help="which scene:label to start on (defaults to first --ckpts entry)")
    ap.add_argument("--width", type=int, default=1200)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--rtsp", default="rtsp://localhost:8554/3dgs")
    ap.add_argument("--ws-port", type=int, default=8769)
    args = ap.parse_args()

    scene_data: dict[str, str] = {}
    for kv in args.scene_data:
        k, _, v = kv.partition("=")
        if not v:
            sys.exit(f"--scene-data entries must be scene=path, got: {kv}")
        scene_data[k] = v

    ckpt_paths: dict[str, str] = {}
    for kv in args.ckpts:
        k, _, v = kv.partition("=")
        if not v or ":" not in k:
            sys.exit(f"--ckpts entries must be scene:label=path, got: {kv}")
        scene = k.split(":")[0]
        if scene not in scene_data:
            sys.exit(f"--ckpts scene '{scene}' has no matching --scene-data entry")
        ckpt_paths[k] = v
    default_key = args.default or next(iter(ckpt_paths))

    W, H = args.width, args.height

    print(f"[gsplat] building scene metas for {list(scene_data)}", flush=True)
    metas: dict[str, SceneMeta] = {}
    K_ts:  dict[str, torch.Tensor] = {}
    for scene, path in scene_data.items():
        m = SceneMeta(Path(path), W, H)
        metas[scene] = m
        K_ts[scene]  = torch.from_numpy(m.K).float().cuda()
        print(f"[gsplat]   {scene}: look_at={m.look_at}, r0={m.radius0:.3f}", flush=True)

    print(f"[gsplat] loading {len(ckpt_paths)} ckpts onto GPU…", flush=True)
    ckpts: dict[str, Splats] = {}
    for key, path in ckpt_paths.items():
        t0 = time.time()
        ckpts[key] = Splats(Path(path))
        print(f"[gsplat]   {key:>10}: {ckpts[key].n_splats:>7} splats "
              f"(step {ckpts[key].step}, {time.time()-t0:.2f}s)", flush=True)

    state = State(ckpts, metas, K_ts, default_key)

    # Warm-up render (first kernel launch is always slow; don't show it to users).
    _ = render_frame(state, W, H)

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
    fps_window = []
    last_log = time.time()
    try:
        while True:
            t0 = time.time()
            with state.lock:
                paused = state.paused
                swap_to = state.want_swap_to
                state.want_swap_to = None
            if paused:
                # Don't render, don't write to ffmpeg — just keep the loop
                # alive at frame_t cadence so resume reacts quickly.
                time.sleep(frame_t)
                continue
            if swap_to:
                target, rv = swap_to
                old_scene = state.current_scene()
                new_scene = target.split(":")[0]
                if target != state.current_key:
                    print(f"[gsplat] swap → {target} ({state.ckpts[target].n_splats} splats)",
                          flush=True)
                    state.current_key = target  # all ckpts on GPU; flip pointer.
                # Cross-scene swap forces a view reset (old basis isn't valid in new scene).
                if rv or new_scene != old_scene:
                    reset_view(state)

            frame = render_frame(state, W, H)
            try:
                ff.stdin.write(frame.tobytes())
            except BrokenPipeError:
                print("[ffmpeg] pipe closed", flush=True)
                break

            dt = time.time() - t0
            fps_window.append(dt)
            if len(fps_window) > 60:
                fps_window.pop(0)
            if time.time() - last_log > 5.0:
                avg = sum(fps_window) / len(fps_window)
                print(f"[gsplat] {state.current_key:>10}  render={avg*1000:.1f}ms/frame  "
                      f"({1.0/max(avg,1e-9):.1f} render-fps, capped at {args.fps})", flush=True)
                last_log = time.time()

            sleep_for = frame_t - dt
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
