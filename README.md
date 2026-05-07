# Streaming — human-eval pipeline for 3D rendering

A web-based, interactive human-eval pipeline for comparing 3D rendering
methods (Instant-NGP, Nexels, 3DGS) under live novel-view synthesis. Two
self-contained studies live here, each with its own backend launcher and
viewer page.

## Research questions and goals

This project exists because the dominant evaluation paradigm for novel
view synthesis — PSNR / SSIM / LPIPS computed on a held-out test camera
— measures something different from what an end user actually
experiences. Real users **drag the camera around interactively**, look
where they want to look, and form an opinion that is conditioned on
*which views* they chose to inspect. The pipeline lets us collect that
opinion at scale.

It targets three open questions:

**Q1 — Method comparison under matched compute / storage.**
Given a feed-forward neural representation (Instant-NGP) and a
neurally-textured-surfel representation (Nexels) trained to comparable
budgets, *which one do humans actually prefer when they get to drive the
camera*, and **which axis** (foreground sharpness, background sharpness,
or rendering smoothness) drives that preference?

**Q2 — Perceptual convergence of 3DGS.**
3DGS trains for 30k iterations by default, but quality plateaus much
earlier in many scenes. *At what iteration does additional training stop
yielding perceptually meaningful gains?* And, secondarily, *how well do
reference-image metrics (PSNR / SSIM / LPIPS) predict the human-perceived
plateau?* If the two agree, we can keep using fast offline metrics; if
they don't, that's an argument for human-in-the-loop evaluation.

**Q3 — Does interactive evaluation change the answer?**
Static-image protocols force the rater to look where the dataset's test
camera looks. Interactive protocols let the rater hunt for the *worst*
view (or the view that matters to them). The mouse-trajectory logs
collected by both studies let us compare interactive ratings against
what a static, fixed-camera protocol would have measured on the same
stimulus.

The work feeds the broader CS348K project on **fixed-storage-budget
representation choice for novel view synthesis** — the human-eval signal
collected here is the ground truth that representation-vs-representation
and metric-vs-metric arguments get checked against.

## Experiments and success criteria

### Study 1 — Comparison (i-NGP vs Nexels)

Launcher: `./start_methods.sh`. Viewer:
`viewer/cross_method_side_by_side.html`.

**Stimuli.** Six mip-NeRF-360 scenes (`bicycle`, `counter`, `garden`,
`kitchen`, `room`, `stump`) shown one pair at a time, with `bonsai` as a
fixed practice pair. Pair order shuffled per participant; A/B side
coin-flipped per pair; method-name reveal blinded except in test mode.
Both renderers stream concurrently into independently-draggable
viewports.

**Per-pair measurements.**

- Foreground sharpness, 0–100 slider, per method.
- Background sharpness, 0–100 slider, per method.
- Smoothness, binary yes/no (was the renderer smooth enough), per method.
- Overall preference, 7-point scale (–3 strongly-A to +3 strongly-B).
- Aspect attribution (multi-select: foreground / background / smoothness)
  — skipped when overall = "no preference".
- Optional free-text comment.
- Mouse trajectory log over the entire viewing window
  (`{t, type, slot, nx, ny, button?, delta?}` events; `t` is ms since
  the pair went live, `type` ∈ `{down, up, move, scroll}`).

**How we know it worked.**

1. **Sign-consistent preference exists.** After ~15–25 participants,
   pooled overall-preference distribution differs from zero at
   p < 0.05 across scenes — i.e. there is a method that humans
   reliably prefer when they drive the camera.
2. **Attribution localises the gap.** The aspect-attribution
   multi-select identifies which axis (FG / BG / smoothness) is doing
   the work. A claim like "Nexels wins because of background quality,
   not foreground" needs the attribution histogram to peak on
   "background" *and* the matching slider gap (Nexels − i-NGP) to be
   strictly larger on background than on foreground.
3. **Smoothness binary doesn't dominate.** If "is it smooth enough" is
   yes/yes for almost every pair, the methods are perceptually
   comparable on smoothness and the foreground/background sliders carry
   the signal. If it's no/no for either method, that method is failing
   at the system level (not the representation level) and we exclude it
   from the perceptual comparison and fix it first.

### Study 2 — Convergence (3DGS at 1k / 3k / 7k / 30k)

Launcher: `./start_convergence.sh`. Viewer:
`viewer/convergence_interactive.html`.

**Stimuli.** Two scenes (`truck`, `train` from Tanks & Temples), each
rendered from four 3DGS checkpoints (1k / 3k / 7k / 30k iters) — eight
stimuli total. Scene order randomised per participant; iters shown
ascending within each scene so the rater feels the convergence trajectory.
A per-scene reference video (training photos played as a slow slideshow)
appears once before that scene's block.

**Per-stim measurements.**

- Foreground sharpness, 0–100 slider.
- Background sharpness, 0–100 slider.
- Smoothness, binary yes/no.
- Dwell time on the stimulus.
- Mouse trajectory log (same schema as Study 1, `slot` field absent).
- Pre-computed reference metrics (PSNR / SSIM / LPIPS) shown only on
  the post-study debrief, never during rating.

**How we know it worked.**

1. **Monotonicity.** Within each scene, mean foreground and background
   ratings are non-decreasing in iter count (1k ≤ 3k ≤ 7k ≤ 30k). A
   participant who inverts that order is either confused or rating noise
   — we expect this to hold per-scene at the population level.
2. **Plateau iter is identified.** For each scene, find the smallest k
   such that rating(k) is statistically indistinguishable from
   rating(30k). If, say, k=7k for `truck`, that's the perceptual
   plateau — and a usable claim about "how long should you train 3DGS
   on this scene".
3. **Metric-to-human agreement, scored.** Spearman rank correlation
   between each automated metric (PSNR / SSIM / LPIPS) and the human
   foreground+background mean, computed across all eight stimuli.
   ρ ≥ 0.9 on a metric ⇒ that metric is a usable cheap proxy for human
   judgment in this regime. ρ < 0.7 ⇒ the metric and the humans
   disagree, and we have a story to tell about why.

### Study 3 — Interactive vs static (post-hoc, no separate launcher)

The mouse-trajectory logs from Studies 1 and 2 let us replay each
participant's actual viewing path. We can sample the views they spent
the most time on, render those exact views as static frames from each
method/checkpoint, and ask: would the rating have come out the same if
they had only seen the static held-out test camera?

**How we know it worked.** Spearman rank correlation between
interactive ratings (live) and static ratings (computed from the
held-out test view of the same stimulus) — high correlation means
static-image protocols are good enough; low correlation means the
interactive setup is uncovering something static evaluation misses, and
the gap can be attributed to specific view directions where the
methods diverge.

## Quick start

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

