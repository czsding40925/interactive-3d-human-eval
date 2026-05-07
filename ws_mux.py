#!/usr/bin/env python3
"""
ws_mux.py — single-port WebSocket multiplexer for the cross-method viewer.

The browser opens ONE WS to this server (port 8765, the only WS port the
AWS SG has open). Each message carries an optional `method` field; we strip
it and forward to the corresponding renderer's internal WS. Default route
is "ingp" so the legacy viewer/interactive.html (which doesn't tag
messages) keeps working.

Routes are independently optional — pass any subset of
--ingp-ws / --nexel-ws / --3dgs-ws and the mux will only connect to the
ones you specify. Messages tagged with a missing route are dropped.

  browser  ──ws──▶  ws_mux (8765)  ──▶  i-NGP renderer  (8767)
                              ├───▶  Nexels renderer (8768)
                              ╰───▶  3DGS renderer   (8769)

Usage (cross-method study):
  python3 ws_mux.py --port 8765 \
    --ingp-ws ws://localhost:8767 \
    --nexel-ws ws://localhost:8768

Usage (convergence-only study):
  python3 ws_mux.py --port 8765 --3dgs-ws ws://localhost:8769
"""
import argparse
import asyncio
import json

import websockets


async def forward_to(target_ws, msg: dict):
    try:
        await target_ws.send(json.dumps(msg))
    except websockets.exceptions.ConnectionClosed:
        pass


async def open_with_retry(uri: str, name: str, max_tries: int = 30):
    for i in range(max_tries):
        try:
            return await websockets.connect(uri, ping_interval=20, ping_timeout=20)
        except (ConnectionRefusedError, OSError):
            if i == 0:
                print(f"[mux] {name} not yet up at {uri}, retrying…", flush=True)
            await asyncio.sleep(1.0)
    print(f"[mux] giving up on {name} after {max_tries}s", flush=True)
    return None


async def make_handler(routes: dict[str, str], default_method: str):
    """`routes` maps method name → ws:// URI. Methods absent from the map are
    silently dropped client-side."""
    async def handler(client_ws):
        peer = client_ws.remote_address
        print(f"[mux] client connected from {peer}", flush=True)
        # Open one upstream socket per configured route.
        upstream: dict[str, object] = {}
        for name, uri in routes.items():
            ws = await open_with_retry(uri, name)
            if ws is not None:
                upstream[name] = ws
        try:
            async for raw in client_ws:
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                method = msg.pop("method", default_method)
                target = upstream.get(method)
                if target is None:
                    continue
                await forward_to(target, msg)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            for ws in upstream.values():
                try: await ws.close()
                except Exception: pass
            print(f"[mux] client {peer} disconnected", flush=True)
    return handler


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--ingp-ws",  default=None, help="ws:// URI for ingp renderer (omit to disable route)")
    ap.add_argument("--nexel-ws", default=None, help="ws:// URI for nexel renderer (omit to disable route)")
    ap.add_argument("--3dgs-ws",  dest="threedgs_ws", default=None,
                    help="ws:// URI for 3dgs renderer (omit to disable route)")
    ap.add_argument("--default-method", default="ingp",
                    help="route to use for messages with no `method` field (legacy viewer)")
    args = ap.parse_args()

    routes: dict[str, str] = {}
    if args.ingp_ws:     routes["ingp"]  = args.ingp_ws
    if args.nexel_ws:    routes["nexel"] = args.nexel_ws
    if args.threedgs_ws: routes["3dgs"]  = args.threedgs_ws
    if not routes:
        raise SystemExit("[mux] no routes configured — pass at least one of --ingp-ws / --nexel-ws / --3dgs-ws")

    handler = await make_handler(routes, args.default_method)
    async with websockets.serve(handler, "0.0.0.0", args.port):
        routes_str = "  ".join(f"{k}→{v}" for k, v in routes.items())
        print(f"[mux] listening on :{args.port}  ({routes_str})  default={args.default_method}",
              flush=True)
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
