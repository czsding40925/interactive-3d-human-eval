# Interactive Streaming Harness for Human Evaluation of Real-Time 3D Renderers

> Static-image metrics like PSNR, SSIM, and LPIPS evaluate Novel View Synthesis
> on fixed, held-out camera poses — but real users *drag the camera around*,
> probe scenes interactively, and form opinions conditioned on the views they
> chose to inspect. This repo is a streaming harness that lets researchers
> collect that opinion at scale.

![Live rating demo](assets/viewer_demo.gif)

📄 **[Full report (PDF)](report/main.pdf)** — extended methodology, statistical
analyses, limitations, and appendices.

🎓 *CS348K Spring 2026 final project · Stanford University · Connor Ding,
Ishita Gupta*

---

## Motivation

The dominant evaluation paradigm for novel view synthesis — PSNR / SSIM /
LPIPS computed on a held-out test camera — measures something different
from what an end user actually experiences. Real users **drag the camera
around interactively**, look where they want to look, and form an
opinion conditioned on *which views* they chose to inspect. Numbers
computed on a fixed set of static frames cannot see popping, floater
drift, or smoothness artifacts that only show up under free navigation.

The intended end-state of this project: a graphics researcher with a
new NVS method can plug their renderer into this framework, recruit
participants, and run a perception study against a baseline or in
isolation. The current repository ships a prototype of that vision,
together with two case studies exercising it.

We do **not** aim to replace numerical metrics — they remain cheap,
useful, and the right tool for in-the-loop training. We aim to provide
the qualitative, interactive signal those metrics cannot.

## What we built

![Streaming pipeline](assets/architecture.png)

The harness is renderer-agnostic: any process that consumes camera-pose
deltas on a WebSocket and pushes H.264 over RTSP can plug in. The plug-in
seam is the contract — register a route in `ws_mux.py` and a path in
`mediamtx.yml` and the same browser viewer works.

Per stimulus (or per pair, in comparison mode) we collect:

- **Foreground quality** — 0–100 slider (subject of the scene)
- **Background quality** — 0–100 slider (surrounds, where NVS methods tend to struggle)
- **Rendering smoothness** — 0–100 slider (only measurable under free navigation)
- **Comparison-mode extras:** overall preference (7-point scale, −3 strongly-A → +3 strongly-B) and aspect attribution (multi-select)
- **Free-text comments** on the scene
- **Mouse trajectory log** over the viewing window

Operational details — ports, launchers, mediamtx configuration — live in
`CLAUDE.md` (local-only) and the launcher scripts themselves.

## Two case studies

We exercise the framework with two case studies, each chosen to ask a
question that automated metrics cannot answer on their own.

### Case 1 — i-NGP vs Nexels at matched storage (~14 MB / scene)

Two NVS methods (a pure-neural representation, [Instant-NGP], and a
hybrid Gaussian-and-neural-network method, [Nexels]) rendered
side-by-side in independently-orbitable viewports. Six mip-NeRF-360
scenes (`bicycle / counter / garden / kitchen / room / stump`) plus
one practice pair (`bonsai`). Per pair, participants rate
foreground / background / smoothness for each side, pick an overall
A/B preference, and indicate which aspects shaped that choice.

```bash
./start_methods.sh
# Open viewer/cross_method_side_by_side.html
```

### Case 2 — Perceptual convergence of 3DGS

A single method (3D Gaussian Splatting) trained for varying numbers
of iterations (1k / 3k / 7k / 30k) on two Tanks-and-Temples scenes
(`train`, `truck`). All eight checkpoints are GPU-resident so swapping
between them is a constant-time pointer update. A per-scene reference
slideshow plays once before each scene's rating block.

```bash
./start_convergence.sh
# Open viewer/convergence_interactive.html
```

---

## Results

### Recruitment

| Study | n (participants) |
|---|---:|
| Case 1 (comparison) | **18** |
| Case 2 (convergence) | **12** |
| Post-session satisfaction survey | **9** |

### Case 1 results — humans and offline metrics disagree

Nexels wins on every offline metric on every scene, but the human A/B
vote is near-even in aggregate (44 / 49 / 12) and **`kitchen` reverses
entirely** to 14 / 2 / 2 i-NGP — a sign-test *p* = 0.004 reversal of
the offline ranking that every numerical metric supports.

![Per-scene preference with strength gradient](assets/fig_preference_per_scene.png)

| Scene | PSNR↑ (i-NGP / Nex) | LPIPS↓ (i-NGP / Nex) | FPS (i-NGP / Nex) | Votes (i-NGP / Tie / Nex) |
|---|---:|---:|---:|---:|
| bicycle | 21.9 / **22.2** | 0.59 / **0.42** | **62** / 30 | 2 / 3 / **13** |
| counter | 25.3 / **26.4** | 0.27 / **0.22** | **62** / 30 | 6 / 1 / **10** |
| garden | 23.1 / **23.5** | 0.46 / **0.32** | **75** / 34 | 7 / 4 / 7 |
| **kitchen** | 27.1 / **27.7** | 0.21 / **0.17** | **96** / 32 | **14** / 2 / 2 |
| room | 28.5 / **29.2** | 0.23 / **0.21** | **70** / 30 | **10** / 1 / 6 |
| stump | 21.7 / **23.9** | 0.57 / **0.40** | 27 / **33** | 5 / 1 / **11** |

(`bonsai` excluded — practice pair. Bold marks the per-scene winner on each column.)

### Case 1 results — why they disagreed

Looking inside each scene, Nexels has the edge on **detail** (foreground
and background) — but i-NGP has the edge on **motion**, winning the
smoothness slider on five of six scenes. The kitchen reversal is the
extreme case: the smoothness gap (28 points) outweighs Nexels' edge on
the other two aspects.

![Per-scene mean ratings by aspect](assets/fig_per_scene_ratings.png)

The asymmetry shows up cleanly in *which aspects participants cited* as
the reason for their A/B choice (Appendix B Table 5 in the report).
Nexels-choosing citers mention background 38 times across all scenes
but smoothness only 12 — a 3× gap. i-NGP-choosing citers cite all three
aspects nearly equally (24 / 21 / 22). The headline: **Nexels voters
were voting on what they saw; i-NGP voters were voting on what they saw
*and* how it moved.** This is the FPS-matters finding stated without a
regression.

### Case 2 results — `truck` smoothness regresses at 30k

`train` improves monotonically on all three aspects from 1k → 30k.
`truck` is different: foreground saturates between 7k and 30k, and
**smoothness drops 23 points** (82.9 → 60.1) as the model grows from
2.5M splats at 7k to 3.8M at 30k. Offline LPIPS keeps improving
(0.143 → 0.095), but the participant feels the frame-time tax.

![Convergence ratings vs offline LPIPS](assets/fig_convergence.png)

In compute-budgeting terms: for `truck`, training 3DGS 4× longer past
7k buys a 34% LPIPS improvement participants do not perceive, while
costing enough rendering speed that smoothness ratings drop measurably.
Offline metrics on still held-out images would not have seen this.

### Participant satisfaction (n=9)

![Post-session participant survey](assets/fig_participant_feedback.png)

Participants reported **overall positive feedback** — *"really impressed
by the different spaces it could capture"* · *"really liked this one
— clear how much better things were improving over time"* · *"VERY CLEAR
INSTRUCTIONS!"* — with **common suggestions** including a reset-view
button, less-sensitive zoom, consistent camera-drag direction, and the
ability to revise earlier ratings.

The strongest dimensions are overall experience (mean 4.4 / 5) and
instruction clarity (4.4 / 5). The weakest is viewer ease of use
(mean 3.6 / 5), where one participant rated it 2 — the same participant
who reported the highest mental fatigue.

## Try it / reproduce

The system runs on a single GPU instance (we used AWS L4, 24 GB VRAM).
Open ports `8080 · 8765 · 8889 · 8189` (RTSP `8554` is internal-only).

```bash
# Install dependencies
./install.sh

# Run the comparison case study (i-NGP vs Nexels)
./start_methods.sh                  # default SIZE=small
SIZE=medium ./start_methods.sh      # larger model tier

# Run the convergence case study (3DGS at 4 iter counts)
./start_convergence.sh
DEFAULT=train:1k ./start_convergence.sh   # start at a different ckpt
```

Open the viewer in a browser:
- Comparison: `http://<GPU_IP>:8080/viewer/cross_method_side_by_side.html`
- Convergence: `http://<GPU_IP>:8080/viewer/convergence_interactive.html`

Replace `<GPU_IP>` with whatever `curl ifconfig.me` reports on the GPU
host. The launchers self-heal the public-IP advertisement in
`mediamtx.yml`, so first-time launch works without manual config edits.

To add a new renderer, see the renderer contract in
[`report/main.pdf`](report/main.pdf) §2.2 — at a high level, expose a
WebSocket port for pose deltas, push H.264 over RTSP, and register one
line in `ws_mux.py` and one in `mediamtx.yml`.

---

## Repo layout

```
interactive-3d-human-eval/
  README.md                        this file
  install.sh

  start_methods.sh                 launcher: comparison case study
  start_convergence.sh             launcher: convergence case study

  interactive_renderer.py          i-NGP renderer (pyngp, headless)
  nexels_renderer.py               Nexels renderer (diff-nexel-rasterization)
  gsplat_renderer.py               3DGS renderer (gsplat.rendering.rasterization)
  ws_mux.py                        single-port WS multiplexer
  serve.py                         static + POST /submit
  mediamtx.yml                     mediamtx config

  viewer/
    cross_method_side_by_side.html comparison case study
    convergence_interactive.html   convergence case study

  assets/                          architecture diagram, result figures, demo GIF
```

## References & links

- 📄 **Full report (PDF):** [`report/main.pdf`](report/main.pdf)
- 🖼 **Result figures:** [`assets/`](assets/)
- The renderer wrappers build on
  [`pyngp`](https://github.com/NVlabs/instant-ngp) (Instant-NGP),
  `diff-nexel-rasterization` (Nexels),
  and [`gsplat`](https://github.com/nerfstudio-project/gsplat) (3D
  Gaussian Splatting).
- Datasets: [mip-NeRF-360] and [Tanks and Temples].

[Instant-NGP]: https://nvlabs.github.io/instant-ngp/
[Nexels]: https://github.com/example/nexels
[mip-NeRF-360]: https://jonbarron.info/mipnerf360/
[Tanks and Temples]: https://www.tanksandtemples.org/
