# Streaming — human-eval pipeline for 3D rendering

A web-based, interactive human-eval pipeline for comparing 3D rendering
methods under live novel-view synthesis. Two self-contained case studies
ship with the repo, each with its own backend launcher and viewer page.

![Live rating demo](assets/viewer_demo.gif)

The harness is renderer-agnostic: any process that consumes camera-pose
deltas on a WebSocket and pushes H.264 over RTSP can plug in.

![Streaming pipeline](assets/architecture.png)

## Motivation

The dominant evaluation paradigm for novel view synthesis — PSNR / SSIM /
LPIPS computed on a held-out test camera — measures something different
from what an end user actually experiences. Real users **drag the camera
around interactively**, look where they want to look, and form an
opinion that is conditioned on *which views* they chose to inspect. The
pipeline lets us collect that opinion at scale.

The intended end-state: a graphics researcher who has just developed a
new 3D reconstruction / NVS model can plug their renderer into this
framework, recruit participants, and run a perception study against a
baseline or in isolation. The current repository is a prototype of that
vision.

Note that the goal is **not** to replace numerical metrics like PSNR and
MS-SSIM — it is to provide a qualitative, interactive signal that those
metrics cannot. We focus on indoor and outdoor **scenes** rather than
synthetic objects.

## What the framework collects

Per stimulus (or per pair, in comparison mode):

- **Foreground quality** — 0–100 slider (the object of interest).
- **Background quality** — 0–100 slider.
- **Rendering smoothness** — 0–100 slider.
- **Comparison-mode extras**
  - Overall preference (7-point scale, –3 strongly-A → +3 strongly-B).
  - Aspect attribution (multi-select: foreground / background / smoothness).
- **Free-text comments** on the scene / session.

A planned post-session participant survey (Google Form, in preparation)
will add direct self-report on perceived smoothness, UI clarity, and
fatigue at session end.

## Case studies

We deploy the framework into two studies, each chosen to exercise a
different mode (comparison vs. single-method) and to address a question
that automated metrics cannot answer on their own. Per-study operational
details (launchers, ports, ckpts) live in `CLAUDE.md`.

### Case study 1 — i-NGP vs Nexels under matched budget

**Question.** Given a feed-forward neural representation (Instant-NGP)
and a neurally-textured-surfel representation (Nexels) trained to
comparable storage budgets, *which one do humans actually prefer when
they get to drive the camera*, and *which axis* (foreground, background,
smoothness) drives that preference?

Launcher: `./start_methods.sh`. Viewer:
`viewer/cross_method_side_by_side.html`.

Six mip-NeRF-360 scenes (`bicycle / counter / garden / kitchen / room /
stump`) are shown one pair at a time, with `bonsai` as a fixed practice
pair. Pair order shuffles per participant; A/B side coin-flips per pair;
method-name reveal is blinded except in test mode. Both renderers stream
concurrently into independently-draggable viewports.

The study is "successful" when (a) pooled overall-preference is
sign-consistent across participants, (b) aspect-attribution histograms
localise that gap to a specific axis, and (c) the smoothness slider
isn't pinned at one end — i.e. the rating channels are actually carrying
the signal.

### Case study 2 — perceptual convergence of 3DGS

**Question.** 3DGS trains for 30k iterations by default, but quality
plateaus much earlier on many scenes. *At what iteration does additional
training stop yielding perceptually meaningful gains?* And, secondarily,
*how well do reference-image metrics (PSNR / SSIM / LPIPS) predict that
plateau?* Agreement ⇒ cheap offline metrics are fine in this regime;
disagreement ⇒ that's a story worth telling.

Launcher: `./start_convergence.sh`. Viewer:
`viewer/convergence_interactive.html`.

Two scenes (`truck / train` from Tanks & Temples), each rendered from
four 3DGS checkpoints (1k / 3k / 7k / 30k iters) — eight stimuli total.
Scene order is randomised per participant; iters are shown ascending
within each scene so the rater feels the convergence trajectory. A
per-scene reference video (training photos as a slow slideshow) plays
once before that scene's block.

Read-out signals: (a) within-scene monotonicity in iter count, (b) a
per-scene plateau iter where rating is statistically indistinguishable
from 30k, and (c) Spearman rank correlation between each automated
metric and the pooled human rating across the eight stimuli.

---

## Progress — Checkpoint 2 (2026-05-22)

This section is the checkpoint snapshot of intermediate results. The
tables and figures below are the data the framework has collected so
far; the broader analysis lives in [`report/`](report/).

### Systems status

| Item | Status |
|---|---|
| Streaming harness (mediamtx / ws_mux / serve.py) | shipped, stable |
| i-NGP renderer wrapper | shipped |
| Nexels renderer wrapper | shipped |
| 3DGS renderer wrapper + `swap_state` | shipped (8 ckpts GPU-resident) |
| Comparison viewer (dual viewport) | shipped |
| Convergence viewer (single viewport + reference slideshow) | shipped |
| Submission persistence (full-snapshot JSON) | shipped |
| End-to-end latency budget (render → display) | **pending measurement** |
| Post-session participant satisfaction survey | **pending deployment** |

Both case studies have run end-to-end without renderer stalls, dropped
WebRTC streams, or mid-session abandonment. The comparison study
sustains the 30 fps target at 1440×810 with both renderers active; the
convergence study sustains the 60 fps target at 1920×1080. Checkpoint
swaps in the convergence study introduce no measurable visual hitch.

### Recruitment

| Study | n (current) | n (target) |
|---|---|---|
| Comparison (i-NGP vs Nexels) | 9 | ~15 |
| Convergence (3DGS at 4 iters) | 7 | ~15 |

### Intermediate results — comparison study (n = 9)

Per-scene offline metrics, FPS, storage footprint, and the mean human
ratings collected to date. `fg / bg / sm` are mean 0–100 slider scores
for foreground / background / rendering smoothness; `Votes` is the
count of participants who preferred that method on that scene. `bonsai`
is the practice scene and is excluded from the mean. LPIPS uses the
AlexNet backbone across all methods.

| Scene | Method | PSNR↑ | SSIM↑ | LPIPS↓ | FPS | MB | fg↑ | bg↑ | sm↑ | Votes |
|---|---|---|---|---|---|---|---|---|---|---|
| bicycle | i-NGP  | 21.93 | 0.462 | 0.592 | **62.0** | 14.5 | 74.3 | 42.6 | 74.0 | 1 |
|         | Nexels | **22.21** | **0.539** | **0.424** | 29.5 | **14.4** | **83.7** | **68.7** | **79.6** | **7** |
| counter | i-NGP  | 25.31 | 0.757 | 0.272 | **61.5** | **13.5** | 68.3 | 52.1 | **55.2** | 2 |
|         | Nexels | **26.42** | **0.833** | **0.216** | 30.0 | 14.4 | **71.1** | **62.1** | 51.8 | **6** |
| garden  | i-NGP  | 23.13 | 0.514 | 0.456 | **74.9** | **12.0** | 71.3 | 66.3 | **49.3** | 3 |
|         | Nexels | **23.46** | **0.636** | **0.322** | 34.0 | 14.4 | **79.9** | **74.9** | 47.1 | 3 |
| kitchen | i-NGP  | 27.12 | 0.767 | 0.214 | **96.2** | **12.6** | **81.1** | **63.2** | **63.7** | **8** |
|         | Nexels | **27.68** | **0.861** | **0.167** | 31.9 | 14.4 | 52.0 | 46.7 | 34.7 | 0 |
| room    | i-NGP  | 28.47 | 0.844 | 0.225 | **69.8** | 15.0 | 48.0 | 55.6 | 56.1 | **5** |
|         | Nexels | **29.15** | **0.877** | **0.207** | 29.8 | **14.4** | **57.7** | **56.7** | **57.6** | 3 |
| stump   | i-NGP  | 21.68 | 0.473 | 0.574 | 27.0 | 16.0 | 59.8 | **52.7** | **62.0** | 2 |
|         | Nexels | **23.88** | **0.604** | **0.398** | **32.9** | **14.4** | **73.2** | 47.3 | 56.3 | **7** |
| **mean (6)** | i-NGP | 24.61 | 0.636 | 0.389 | **65.2** | 13.9 | 67.1 | 55.4 | **60.1** | 21 |
|              | Nexels | **25.47** | **0.725** | **0.289** | 31.4 | 14.4 | **69.6** | **59.4** | 54.5 | **26** |

Aggregate A/B preference 21 / 26 / 7 (i-NGP / Nexels / tie), close to
even. The per-scene picture (figure below) is the more interesting view
— scene-by-scene preferences are heterogeneous and `kitchen` reverses
entirely to i-NGP despite Nexels winning every offline metric on that
scene.

![Per-scene A/B preference](assets/fig_preference_per_scene.png)

### Intermediate results — convergence study (n = 7)

Per-stimulus offline metrics and mean human ratings (3DGS at four
checkpoint counts on Tanks-and-Temples `train` and `truck`). LPIPS is
AlexNet (gsplat default). FPS is `ellipse_time` at the gsplat eval
resolution and is an upper bound on the interactive frame rate at
1920×1080.

| Scene | Iters | PSNR↑ | SSIM↑ | LPIPS↓ | FPS | # Splats | fg↑ | bg↑ | sm↑ |
|---|---|---|---|---|---|---|---|---|---|
| train | 1k  | 17.06 | 0.584 | 0.550 | 433.7 | 0.21M | 45.0 | 27.9 | 79.4 |
|       | 3k  | 18.75 | 0.650 | 0.413 | 254.1 | 0.49M | 61.1 | 37.4 | 76.6 |
|       | 7k  | 19.73 | 0.721 | 0.298 | 130.8 | 1.05M | 79.1 | 49.4 | 89.9 |
|       | 30k | 21.44 | 0.812 | 0.146 |  77.4 | 1.81M | 88.4 | 66.0 | 91.7 |
| truck | 1k  | 20.94 | 0.725 | 0.326 | 437.5 | 0.21M | 35.9 | 40.6 | 76.4 |
|       | 3k  | 22.65 | 0.800 | 0.207 | 144.0 | 1.14M | 56.1 | 56.0 | 84.9 |
|       | 7k  | 23.87 | 0.845 | 0.143 |  67.6 | 2.51M | 75.1 | 65.1 | 86.9 |
|       | 30k | 25.04 | 0.871 | 0.095 |  45.5 | 3.79M | 77.7 | 63.3 | 81.3 |

The `train` scene shows monotone improvement on all three aspects from
1k → 30k. The `truck` scene saturates: foreground goes 75.1 → 77.7 from
7k → 30k while smoothness drops from 86.9 → 81.3 over the same
interval, mirroring the FPS drop from 68 to 46 as the splat count grows
from 2.5M to 3.8M.

![Convergence ratings vs offline LPIPS](assets/fig_convergence.png)

### Against the framework's own success criteria

Stated success criteria (from the case-study descriptions above) vs.
what the current data shows:

**Comparison study.**

- *Pooled overall-preference sign-consistent across participants* —
  Partially. The aggregate is near-tie (21/26/7) but per-scene
  preferences are sign-consistent within scene (e.g. `kitchen` 8-0-1
  i-NGP; `bicycle` 7-1-1 Nexels). Sign-consistency at the aggregate
  level fails; sign-consistency at the per-scene level holds.
- *Aspect-attribution histograms localise the gap* — Yes. Background
  citations push toward Nexels; smoothness citations push toward i-NGP.
- *Smoothness slider is not pinned* — Yes. Smoothness ratings span
  34.7 (kitchen / Nexels) to 79.6 (bicycle / Nexels).

**Convergence study.**

- *Within-scene monotonicity in iter count* — Yes on `train` across all
  aspects; no on `truck` (smoothness drops 7k → 30k).
- *Per-scene plateau iter* — Identifiable on `truck` (foreground
  saturates by 7k). On `train` no plateau is visible by 30k.
- *Spearman rank correlation between automated metrics and pooled
  rating* — Computed but parked in the report appendix as an example
  analysis rather than a headline claim; we are positioning the
  framework as a data-gathering platform rather than an analysis
  package.

### What's still open

- **Recruitment.** Push n to ~15 per study before final submission.
- **Participant satisfaction.** Deploy the post-session Google Form
  survey so participant-side success has a direct measurement, not just
  the indirect indicators (completion rate, no abandons) we have now.
- **End-to-end latency.** Measure and publish a render → encode →
  network → decode → display latency budget; defend the
  "imperceptible latency" claim with numbers.
- **Inter-rater agreement.** Compute Kendall's W on A/B preference and
  ICC(2,k) on per-aspect slider scores once n ≥ 15.
- **Cross-cutting analysis (planned extension).** Mouse-trajectory
  logging is not yet implemented; once it is, the
  static-vs-interactive replay analysis (would the rating have come
  out the same if the participant only saw the held-out test camera?)
  becomes possible. Out of scope for this checkpoint.

---

## Repo layout

```
interactive-3d-human-eval/
  README.md                        this file
  install.sh

  start_methods.sh                 launcher: comparison study
  start_convergence.sh             launcher: convergence study

  interactive_renderer.py          i-NGP renderer (pyngp, headless)
  nexels_renderer.py               Nexels renderer (diff-nexel-rasterization)
  gsplat_renderer.py               3DGS renderer (gsplat.rendering.rasterization)
  ws_mux.py                        single-port WS multiplexer
  serve.py                         static + POST /submit
  mediamtx.yml                     mediamtx config

  viewer/
    cross_method_side_by_side.html comparison study (active)
    convergence_interactive.html   convergence study (active)

  assets/                          architecture diagram, result figures, demo GIF
```
