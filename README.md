# Streaming — human-eval pipeline for 3D rendering

A web-based, interactive human-eval pipeline for comparing 3D rendering
methods under live novel-view synthesis. Two self-contained case studies
ship with the repo, each with its own backend launcher and viewer page.

![Live rating demo](assets/viewer_demo.gif)

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
- **Smoothness** — binary yes/no (was the renderer smooth enough?).
- **Comparison-mode extras**
  - Overall preference (7-point scale, –3 strongly-A → +3 strongly-B).
  - Aspect attribution (multi-select: foreground / background / smoothness).
- **Free-text comments** on the scene.
- **Mouse trajectory log** over the entire viewing window
  (`{t, type, slot?, nx, ny, button?, delta?}` events).
- *(WIP)* audio recording of think-aloud commentary.

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
localise that gap to a specific axis, and (c) the smoothness binary
isn't pinned at yes/yes or no/no — i.e. the slider channels are
actually carrying the signal.

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

### Cross-cutting — does interactivity change the answer?

A second use of the data from both studies above. The mouse-trajectory
logs let us replay each participant's actual viewing path, sample the
views they spent the most time on, render those exact views as static
frames, and ask: *would the rating have come out the same had they only
seen the dataset's held-out test camera?* High Spearman correlation
between interactive and static ratings ⇒ static-image protocols are
good enough; low correlation ⇒ the interactive setup is uncovering
something static evaluation misses, and we can attribute the gap to
specific view directions where the methods diverge.

This feeds the broader CS348K project on **fixed-storage-budget
representation choice for NVS** — the human-eval signal collected here
is the ground truth that representation-vs-representation and
metric-vs-metric arguments get checked against.

<!-- ## Quick start

```bash
cd ~/repos/interactive-3d-human-eval
./start_methods.sh                  # Study 1, default SIZE=small
SIZE=medium ./start_methods.sh      # Study 1, larger model tier

./start_convergence.sh              # Study 2
DEFAULT=train:1k ./start_convergence.sh
```

The launchers self-heal `mediamtx.yml`'s public-IP advertisement on
each run, so first-time launch works without manual config edits.
Launcher logs go to `*.log` in the working directory.

Open in a browser: `http://<L4_IP>:8080/viewer/<study>.html`.

## Repo layout

```
interactive-3d-human-eval/
  README.md                        this file
  install.sh

  start_methods.sh                 launcher: comparison study
  start_convergence.sh             launcher: convergence study

  interactive_renderer.py          i-NGP renderer (pyngp, headless)
  nexels_renderer.py               Nexels renderer
  gsplat_renderer.py               3DGS renderer
  ws_mux.py                        single-port WS multiplexer
  serve.py                         static + POST /submit
  mediamtx.yml                     mediamtx config

  viewer/
    cross_method_side_by_side.html comparison study
    convergence_interactive.html   convergence study

  refs/                            reference photos / per-scene slideshows
  responses/                       participant snapshots (per-study buckets)
  runs_3dgs/                       3DGS ckpts + ply + stats + trajectories
  archive/                         legacy launchers / pre-mux viewers / mp4s
```
 -->
