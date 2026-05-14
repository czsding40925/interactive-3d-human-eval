"""
analyze.py — summary stats + plots for the two human-eval studies.

Studies:
  cross_method:   ingp vs nexel, 6 mip-NeRF-360 scenes per participant
  convergence:    3DGS at iters {1k, 3k, 7k, 30k} on truck + train

Reads responses/{cross_method,convergence}/*.json, writes:
  - tables to stdout (one section per study)
  - PNGs to analysis/{cross_method,convergence}/

Schema caveats handled:
  - ori-sxs-20260511 used smoothness = 'yes'/'no' (binary).
    Mapped 'yes'->100, 'no'->0 for compatibility; flagged on stdout.
  - cross_method overall_preference sign is A-vs-B (depends on order_flipped).
    Re-signed via overall_method so positive = nexel-preferred.

Run:
  python3 analyze.py
"""

from __future__ import annotations

import glob
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RESP = ROOT / "responses"
OUT = ROOT / "analysis"
(OUT / "cross_method").mkdir(parents=True, exist_ok=True)
(OUT / "convergence").mkdir(parents=True, exist_ok=True)

METHOD_COLORS = {"ingp": "#1f77b4", "nexel": "#ff7f0e"}
METRICS = ["foreground", "background", "smoothness"]
ITERS = [1000, 3000, 7000, 30000]


# -------------------------------------------------------------------- loading

def _coerce_smoothness(v):
    """ori-sxs legacy: 'yes'/'no' -> 100/0; numeric passes through."""
    if isinstance(v, str):
        return {"yes": 100.0, "no": 0.0}.get(v.lower(), np.nan)
    return float(v) if v is not None else np.nan


def load_cross_method() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (trials_df, mouse_summary_df)."""
    rows, mouse_rows = [], []
    legacy_flagged = False
    for f in sorted(glob.glob(str(RESP / "cross_method" / "*.json"))):
        d = json.load(open(f))
        if d.get("is_test"):
            continue
        pid = d["participant_id"]
        for r in d["results"]:
            rbm = r["ratings_by_method"]
            # smoothness can be str in legacy file
            sm_ingp_raw = rbm["smoothness"]["ingp"]
            sm_nexel_raw = rbm["smoothness"]["nexel"]
            is_legacy = isinstance(sm_ingp_raw, str) or isinstance(sm_nexel_raw, str)
            if is_legacy:
                legacy_flagged = True

            # Sign convention: positive => nexel preferred
            raw_pref = r["overall_preference"]
            om = r["overall_method"]
            if om == "nexel":
                signed_pref = abs(raw_pref)
            elif om == "ingp":
                signed_pref = -abs(raw_pref)
            else:
                signed_pref = 0  # tie

            rows.append({
                "participant": pid,
                "pair": r["pair"],
                "scene": r["scene"],
                "order_flipped": r["order_flipped"],
                "fg_ingp": rbm["foreground"]["ingp"],
                "fg_nexel": rbm["foreground"]["nexel"],
                "bg_ingp": rbm["background"]["ingp"],
                "bg_nexel": rbm["background"]["nexel"],
                "sm_ingp": _coerce_smoothness(sm_ingp_raw),
                "sm_nexel": _coerce_smoothness(sm_nexel_raw),
                "sm_legacy_binary": is_legacy,
                "pref_raw": raw_pref,
                "pref_signed_nexel": signed_pref,
                "overall_method": om,
                "aspects": tuple(r["aspects_influencing_overall"]),
                "comment": r.get("comment", ""),
            })

            # Mouse summary (per trial)
            events = r.get("mouse_events") or []
            if events:
                a_time, b_time = _slot_dwell(events)
                duration_s = (events[-1]["t"] - events[0]["t"]) / 1000.0
                mouse_rows.append({
                    "participant": pid,
                    "pair": r["pair"],
                    "scene": r["scene"],
                    "n_events": len(events),
                    "n_moves": sum(1 for e in events if e["type"] == "move"),
                    "n_drags": sum(1 for e in events if e["type"] == "down"),
                    "duration_s": duration_s,
                    "time_on_a_s": a_time / 1000.0,
                    "time_on_b_s": b_time / 1000.0,
                })

    if legacy_flagged:
        print("  [note] legacy ori-sxs smoothness 'yes'/'no' mapped to 100/0.\n"
              "         smoothness numbers from that file are NOT comparable on scale.\n")
    return pd.DataFrame(rows), pd.DataFrame(mouse_rows)


def _slot_dwell(events: list[dict]) -> tuple[float, float]:
    """Approximate time spent over slot A vs slot B from mouse-move slot tags."""
    a_ms = b_ms = 0.0
    last_t = events[0]["t"]
    last_slot = events[0].get("slot")
    for e in events[1:]:
        dt = e["t"] - last_t
        if last_slot == "A":
            a_ms += dt
        elif last_slot == "B":
            b_ms += dt
        if e.get("slot") is not None:
            last_slot = e["slot"]
        last_t = e["t"]
    return a_ms, b_ms


def load_convergence() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, mouse_rows = [], []
    for f in sorted(glob.glob(str(RESP / "convergence" / "*.json"))):
        d = json.load(open(f))
        if d.get("is_test"):
            continue
        pid = d["participant_id"]
        for r in d["phase2_results"]:
            rows.append({
                "participant": pid,
                "stim_index": r["stim_index"],
                "scene": r["scene"],
                "iter": r["iter"],
                "foreground": r["foreground"],
                "background": r["background"],
                "smoothness": r["smoothness"],
                "dwell_s": r["dwell_ms"] / 1000.0,
            })
            events = r.get("mouse_events") or []
            if events:
                duration_s = (events[-1]["t"] - events[0]["t"]) / 1000.0
                mouse_rows.append({
                    "participant": pid,
                    "scene": r["scene"],
                    "iter": r["iter"],
                    "n_events": len(events),
                    "n_moves": sum(1 for e in events if e["type"] == "move"),
                    "n_drags": sum(1 for e in events if e["type"] == "down"),
                    "duration_s": duration_s,
                    "events_per_sec": len(events) / duration_s if duration_s > 0 else np.nan,
                })
    return pd.DataFrame(rows), pd.DataFrame(mouse_rows)


# --------------------------------------------------------- cross_method plots

def plot_paired_sliders(df: pd.DataFrame, out_dir: Path) -> None:
    """Paired strip+line plot of ingp vs nexel slider scores per scene."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    pairs = [("foreground", "fg_ingp", "fg_nexel"),
             ("background", "bg_ingp", "bg_nexel"),
             ("smoothness", "sm_ingp", "sm_nexel")]
    for ax, (label, ci, cn) in zip(axes, pairs):
        scenes = sorted(df["scene"].unique())
        for _, row in df.iterrows():
            x_jit = scenes.index(row["scene"]) + np.random.uniform(-0.08, 0.08)
            marker = "x" if row["sm_legacy_binary"] and label == "smoothness" else "o"
            ax.plot([x_jit - 0.15, x_jit + 0.15], [row[ci], row[cn]], "-",
                    color="gray", alpha=0.4, linewidth=0.8)
            ax.plot(x_jit - 0.15, row[ci], marker, color=METHOD_COLORS["ingp"], markersize=6)
            ax.plot(x_jit + 0.15, row[cn], marker, color=METHOD_COLORS["nexel"], markersize=6)
        # Per-scene means
        scene_means = df.groupby("scene")[[ci, cn]].mean()
        for i, s in enumerate(scenes):
            ax.plot(i - 0.30, scene_means.loc[s, ci], "_",
                    color=METHOD_COLORS["ingp"], markersize=20, markeredgewidth=2.5)
            ax.plot(i + 0.30, scene_means.loc[s, cn], "_",
                    color=METHOD_COLORS["nexel"], markersize=20, markeredgewidth=2.5)
        ax.set_xticks(range(len(scenes)))
        ax.set_xticklabels(scenes, rotation=20)
        ax.set_title(label)
        ax.set_ylim(-5, 105)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("rating (0–100)")
    handles = [plt.Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=METHOD_COLORS[m], label=m, markersize=8)
               for m in ("ingp", "nexel")]
    handles.append(plt.Line2D([0], [0], marker="x", color="gray", linestyle="",
                              label="legacy y/n→100/0", markersize=8))
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Cross-method ratings: ingp vs nexel (paired by trial, mean = thick tick)",
                 y=1.08, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "sliders_by_method.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_preference_distribution(df: pd.DataFrame, out_dir: Path) -> None:
    """Signed preference (positive = nexel) per scene, one dot per participant."""
    fig, ax = plt.subplots(figsize=(8, 4.2))
    scenes = sorted(df["scene"].unique())
    for _, row in df.iterrows():
        x_jit = scenes.index(row["scene"]) + np.random.uniform(-0.12, 0.12)
        v = row["pref_signed_nexel"]
        color = METHOD_COLORS["nexel"] if v > 0 else (
            METHOD_COLORS["ingp"] if v < 0 else "gray")
        ax.plot(x_jit, v, "o", color=color, markersize=8, alpha=0.85)
        ax.annotate(row["participant"].split("-")[0], (x_jit, v),
                    fontsize=7, alpha=0.6, xytext=(4, -2), textcoords="offset points")
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xticks(range(len(scenes)))
    ax.set_xticklabels(scenes, rotation=20)
    ax.set_ylim(-3.5, 3.5)
    ax.set_yticks(range(-3, 4))
    ax.set_ylabel("preference  (positive = nexel)")
    ax.set_title("Overall preference per (scene, participant)")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "preference_distribution.png", dpi=140)
    plt.close(fig)


def plot_aspect_attribution(df: pd.DataFrame, out_dir: Path) -> None:
    """How often each aspect is checked when explaining the overall pick."""
    counts = Counter()
    for tup in df["aspects"]:
        for a in tup:
            counts[a] += 1
    n_with_pref = (df["pref_signed_nexel"] != 0).sum()
    labels = ["foreground", "background", "smoothness"]
    vals = [counts.get(x, 0) for x in labels]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    bars = ax.bar(labels, vals, color=["#4c72b0", "#55a868", "#c44e52"])
    for b, v in zip(bars, vals):
        ax.annotate(f"{v}/{n_with_pref}", (b.get_x() + b.get_width() / 2, v),
                    ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("# trials checked")
    ax.set_title(f"Aspects driving overall preference  (n={n_with_pref} non-tie trials)")
    ax.set_ylim(0, max(vals + [1]) * 1.25)
    fig.tight_layout()
    fig.savefig(out_dir / "aspect_attribution.png", dpi=140)
    plt.close(fig)


def plot_rating_diff_vs_preference(df: pd.DataFrame, out_dir: Path) -> None:
    """Does (nexel - ingp) rating diff correlate with signed preference?"""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    pairs = [("foreground", "fg_nexel", "fg_ingp"),
             ("background", "bg_nexel", "bg_ingp"),
             ("smoothness", "sm_nexel", "sm_ingp")]
    for ax, (label, cn, ci) in zip(axes, pairs):
        diff = df[cn] - df[ci]
        ax.scatter(diff, df["pref_signed_nexel"], alpha=0.7,
                   s=55, edgecolor="black", linewidth=0.4)
        # Linear fit
        mask = diff.notna() & df["pref_signed_nexel"].notna()
        if mask.sum() >= 2:
            m, b = np.polyfit(diff[mask], df["pref_signed_nexel"][mask], 1)
            xs = np.linspace(diff[mask].min(), diff[mask].max(), 50)
            ax.plot(xs, m * xs + b, "--", color="gray", linewidth=1)
            r = np.corrcoef(diff[mask], df["pref_signed_nexel"][mask])[0, 1]
            ax.set_title(f"{label}   (r = {r:+.2f})")
        else:
            ax.set_title(label)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_xlabel("nexel − ingp")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("preference (signed, +nexel)")
    fig.suptitle("Slider differences vs overall preference", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "rating_diff_vs_preference.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def _elo_pass(df: pd.DataFrame, k: float, score_mode: str,
              order: np.ndarray | None = None) -> tuple[dict, list[tuple]]:
    """Single Elo pass over trials. Returns final ratings + step trajectory.

    score_mode:
      'binary'  -> positive pref = nexel wins (1), negative = ingp wins (0), tie = 0.5
      'graded'  -> map −3…+3 → 0…1  (uses Likert magnitude)
    """
    R = {"ingp": 1500.0, "nexel": 1500.0}
    traj = [(0, R["ingp"], R["nexel"])]
    idxs = df.index.to_numpy() if order is None else order
    for step, idx in enumerate(idxs, 1):
        pref = df.at[idx, "pref_signed_nexel"]
        if score_mode == "binary":
            s_nexel = 1.0 if pref > 0 else (0.0 if pref < 0 else 0.5)
        else:  # graded
            s_nexel = (float(pref) + 3.0) / 6.0
        s_ingp = 1.0 - s_nexel
        e_nexel = 1.0 / (1.0 + 10 ** ((R["ingp"] - R["nexel"]) / 400.0))
        e_ingp = 1.0 - e_nexel
        R["nexel"] += k * (s_nexel - e_nexel)
        R["ingp"] += k * (s_ingp - e_ingp)
        traj.append((step, R["ingp"], R["nexel"]))
    return R, traj


def elo_with_bootstrap(df: pd.DataFrame, n_boot: int = 2000, k: float = 16.0,
                       score_mode: str = "graded", seed: int = 0) -> dict:
    """Bootstrap trial order to get a CI on final Elo ratings."""
    rng = np.random.default_rng(seed)
    finals = {"ingp": [], "nexel": []}
    base = df.index.to_numpy()
    for _ in range(n_boot):
        order = rng.permutation(base)
        R, _ = _elo_pass(df, k=k, score_mode=score_mode, order=order)
        finals["ingp"].append(R["ingp"])
        finals["nexel"].append(R["nexel"])
    out = {}
    for m in ("ingp", "nexel"):
        arr = np.array(finals[m])
        out[m] = {"mean": float(arr.mean()),
                  "lo": float(np.percentile(arr, 2.5)),
                  "hi": float(np.percentile(arr, 97.5))}
    return out


def plot_elo(df: pd.DataFrame, out_dir: Path) -> None:
    """Chronological-order trajectory (one line / method) + bootstrap CI at end."""
    chrono = df.sort_values(["participant", "pair"]).reset_index(drop=True)
    _, traj_graded = _elo_pass(chrono, k=16.0, score_mode="graded")
    _, traj_binary = _elo_pass(chrono, k=16.0, score_mode="binary")
    boot_graded = elo_with_bootstrap(chrono, n_boot=2000, k=16.0, score_mode="graded")
    boot_binary = elo_with_bootstrap(chrono, n_boot=2000, k=16.0, score_mode="binary")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, traj, boot, title in [
        (axes[0], traj_graded, boot_graded, "graded score (Likert −3…+3 → 0…1)"),
        (axes[1], traj_binary, boot_binary, "binary score (win / loss / draw)"),
    ]:
        steps = [t[0] for t in traj]
        r_ingp = [t[1] for t in traj]
        r_nexel = [t[2] for t in traj]
        ax.plot(steps, r_ingp, "-", color=METHOD_COLORS["ingp"], label="ingp", linewidth=1.8)
        ax.plot(steps, r_nexel, "-", color=METHOD_COLORS["nexel"], label="nexel", linewidth=1.8)
        # Bootstrap CI bands at the right edge
        x_end = steps[-1] + 0.5
        for m in ("ingp", "nexel"):
            b = boot[m]
            ax.errorbar(x_end, b["mean"],
                        yerr=[[b["mean"] - b["lo"]], [b["hi"] - b["mean"]]],
                        fmt="o", color=METHOD_COLORS[m], capsize=4, markersize=7)
            ax.annotate(f"{b['mean']:.0f}\n[{b['lo']:.0f}, {b['hi']:.0f}]",
                        (x_end + 0.5, b["mean"]),
                        fontsize=8, va="center", color=METHOD_COLORS[m])
        ax.axhline(1500, color="black", linewidth=0.5, linestyle=":")
        ax.set_xlabel("trial #  (chronological)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, steps[-1] + 4)
        ax.legend(loc="lower left", fontsize=9)
    axes[0].set_ylabel("Elo rating  (start = 1500, k = 16)")
    fig.suptitle("Cross-method Elo  (markers = bootstrap mean across trial-order shuffles, 95% CI)",
                 y=1.02, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "elo.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_cross_method_mouse(mouse_df: pd.DataFrame, out_dir: Path) -> None:
    if mouse_df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Total events per trial, grouped by participant
    ax = axes[0]
    participants = sorted(mouse_df["participant"].unique())
    for i, p in enumerate(participants):
        sub = mouse_df[mouse_df["participant"] == p].sort_values("pair")
        ax.plot(sub["pair"], sub["n_events"], "o-", label=p.split("-")[0])
    ax.set_xlabel("pair #")
    ax.set_ylabel("# mouse events")
    ax.set_title("Mouse activity per trial")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Time spent on A vs B per trial
    ax = axes[1]
    width = 0.35
    x = np.arange(len(mouse_df))
    ax.bar(x - width / 2, mouse_df["time_on_a_s"], width, label="time on A", color="#4c72b0")
    ax.bar(x + width / 2, mouse_df["time_on_b_s"], width, label="time on B", color="#c44e52")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r.participant.split('-')[0]}\n{r.scene}"
                        for r in mouse_df.itertuples()], fontsize=7, rotation=0)
    ax.set_ylabel("seconds hovering viewport")
    ax.set_title("Time spent on A vs B (mouse-over)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "mouse_summary.png", dpi=140)
    plt.close(fig)


# ---------------------------------------------------------- convergence plots

def plot_rating_vs_iter(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, m in zip(axes, METRICS):
        for (pid, scene), sub in df.groupby(["participant", "scene"]):
            sub = sub.sort_values("iter")
            label = f"{pid.split('-')[0]} / {scene}"
            ax.plot(sub["iter"], sub[m], "o-", label=label, alpha=0.85)
        ax.set_xscale("log")
        ax.set_xticks(ITERS)
        ax.set_xticklabels([f"{i // 1000}k" for i in ITERS])
        ax.set_xlabel("training iters")
        ax.set_title(m)
        ax.set_ylim(-5, 105)
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("rating (0–100)")
    axes[-1].legend(fontsize=8, loc="lower right", framealpha=0.9)
    fig.suptitle("Convergence: ratings vs 3DGS training iters", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "rating_vs_iter.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_mean_rating_vs_iter(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    colors = {"foreground": "#4c72b0", "background": "#55a868", "smoothness": "#c44e52"}
    for m in METRICS:
        agg = df.groupby("iter")[m].agg(["mean", "std", "count"]).reset_index()
        ax.errorbar(agg["iter"], agg["mean"], yerr=agg["std"],
                    marker="o", capsize=3, label=m, color=colors[m])
    ax.set_xscale("log")
    ax.set_xticks(ITERS)
    ax.set_xticklabels([f"{i // 1000}k" for i in ITERS])
    ax.set_xlabel("training iters")
    ax.set_ylabel("mean rating (± std)")
    ax.set_title("Mean rating across all (participant, scene) vs iters")
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "mean_rating_vs_iter.png", dpi=140)
    plt.close(fig)


def plot_dwell_vs_iter(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for (pid, scene), sub in df.groupby(["participant", "scene"]):
        sub = sub.sort_values("iter")
        ax.plot(sub["iter"], sub["dwell_s"], "o-",
                label=f"{pid.split('-')[0]} / {scene}", alpha=0.85)
    ax.set_xscale("log")
    ax.set_xticks(ITERS)
    ax.set_xticklabels([f"{i // 1000}k" for i in ITERS])
    ax.set_xlabel("training iters")
    ax.set_ylabel("dwell time (s)")
    ax.set_title("Time spent before rating each ckpt")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "dwell_vs_iter.png", dpi=140)
    plt.close(fig)


def plot_convergence_mouse(mouse_df: pd.DataFrame, out_dir: Path) -> None:
    if mouse_df.empty:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for (pid, scene), sub in mouse_df.groupby(["participant", "scene"]):
        sub = sub.sort_values("iter")
        ax.plot(sub["iter"], sub["events_per_sec"], "o-",
                label=f"{pid.split('-')[0]} / {scene}", alpha=0.85)
    ax.set_xscale("log")
    ax.set_xticks(ITERS)
    ax.set_xticklabels([f"{i // 1000}k" for i in ITERS])
    ax.set_xlabel("training iters")
    ax.set_ylabel("mouse events / second  (interaction rate)")
    ax.set_title("Camera-interaction rate vs iters")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "mouse_activity_vs_iter.png", dpi=140)
    plt.close(fig)


# ------------------------------------------------------------- text summaries

def print_cross_method_summary(df: pd.DataFrame, mouse_df: pd.DataFrame) -> None:
    print("=" * 72)
    print("CROSS-METHOD STUDY  (ingp vs nexel)")
    print("=" * 72)
    print(f"participants: {df['participant'].nunique()}   trials: {len(df)}")
    print(f"scenes:       {sorted(df['scene'].unique())}")
    print(f"order_flipped balance:  flipped={df['order_flipped'].sum()}/"
          f"{len(df)}  (~half is healthy)\n")

    print("--- per-method slider means (all participants pooled) ---")
    rows = []
    for label, ci, cn in [("foreground", "fg_ingp", "fg_nexel"),
                          ("background", "bg_ingp", "bg_nexel"),
                          ("smoothness", "sm_ingp", "sm_nexel")]:
        rows.append({
            "metric": label,
            "ingp_mean": df[ci].mean(),
            "nexel_mean": df[cn].mean(),
            "Δ(nexel−ingp)": df[cn].mean() - df[ci].mean(),
            "n_trials_nexel_higher": int((df[cn] > df[ci]).sum()),
        })
    print(pd.DataFrame(rows).to_string(index=False,
          float_format=lambda v: f"{v:6.2f}"))
    print()

    print("--- overall preference (signed: +nexel, −ingp) ---")
    pref = df["pref_signed_nexel"]
    counts = pref.value_counts().sort_index()
    print(counts.to_string())
    print(f"  mean signed preference: {pref.mean():+.2f}")
    print(f"  prefers nexel: {(pref > 0).sum()}   ingp: {(pref < 0).sum()}   "
          f"tie: {(pref == 0).sum()}\n")

    print("--- aspects influencing preference ---")
    counts = Counter()
    for tup in df["aspects"]:
        for a in tup:
            counts[a] += 1
    n_nontie = int((pref != 0).sum())
    for a in ["foreground", "background", "smoothness"]:
        c = counts.get(a, 0)
        print(f"  {a:11s}: {c}/{n_nontie}  ({100 * c / max(n_nontie, 1):.0f}%)")
    print()

    print("--- comments ---")
    for _, r in df.iterrows():
        if r["comment"].strip():
            who = r["participant"].split("-")[0]
            print(f"  [{who} / {r['scene']:8s}] {r['comment']}")
    print()

    if not mouse_df.empty:
        print("--- mouse activity ---")
        agg = mouse_df.agg({
            "n_events": "mean", "duration_s": "mean",
            "time_on_a_s": "mean", "time_on_b_s": "mean",
        })
        print(f"  events/trial (mean):   {agg['n_events']:.0f}")
        print(f"  duration/trial (mean): {agg['duration_s']:.1f} s")
        print(f"  time on A vs B (mean): {agg['time_on_a_s']:.1f}s vs {agg['time_on_b_s']:.1f}s")
        print()

    print("--- Elo  (start = 1500, k = 16, 2000-shuffle bootstrap of trial order) ---")
    print("  caveat: with 2 methods, Elo is monotone in mean preference — included for")
    print("          a slide-deck-friendly summary number, not new information.")
    for mode in ("graded", "binary"):
        boot = elo_with_bootstrap(df, n_boot=2000, k=16.0, score_mode=mode)
        gap = boot["nexel"]["mean"] - boot["ingp"]["mean"]
        # Expected score for the higher-rated side, given the gap.
        winner = "nexel" if gap >= 0 else "ingp"
        exp_winner_score = 1.0 / (1.0 + 10 ** (-abs(gap) / 400.0))
        print(f"  [{mode:6s}]  ingp: {boot['ingp']['mean']:7.1f}  "
              f"[{boot['ingp']['lo']:.0f}, {boot['ingp']['hi']:.0f}]"
              f"    nexel: {boot['nexel']['mean']:7.1f}  "
              f"[{boot['nexel']['lo']:.0f}, {boot['nexel']['hi']:.0f}]"
              f"    gap = {gap:+.1f}  (~{exp_winner_score:.0%} expected for {winner})")
    print()


def print_convergence_summary(df: pd.DataFrame, mouse_df: pd.DataFrame) -> None:
    print("=" * 72)
    print("CONVERGENCE STUDY  (3DGS iters)")
    print("=" * 72)
    print(f"participants: {df['participant'].nunique()}   trials: {len(df)}")
    print(f"scenes:       {sorted(df['scene'].unique())}\n")

    print("--- per-iter mean ratings (pooled across participants & scenes) ---")
    agg = df.groupby("iter")[METRICS].mean().reset_index()
    agg["iter"] = agg["iter"].map(lambda i: f"{i // 1000}k")
    print(agg.to_string(index=False, float_format=lambda v: f"{v:6.2f}"))
    print()

    print("--- monotonicity per (participant, scene, metric): 1k → 30k ---")
    rows = []
    for (pid, scene), sub in df.groupby(["participant", "scene"]):
        sub = sub.sort_values("iter")
        for m in METRICS:
            vals = sub[m].tolist()
            mono = "↑" if all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1)) else \
                   ("↓" if all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1)) else
                    "mixed")
            rows.append({
                "participant": pid.split("-")[0], "scene": scene, "metric": m,
                "1k→30k Δ": vals[-1] - vals[0], "trend": mono,
            })
    print(pd.DataFrame(rows).to_string(index=False))
    print()

    print("--- dwell time per iter (s, mean across participants×scenes) ---")
    print(df.groupby("iter")["dwell_s"].mean().to_string())
    print()

    if not mouse_df.empty:
        print("--- mouse interaction rate per iter (events/s) ---")
        print(mouse_df.groupby("iter")["events_per_sec"].mean()
              .to_string(float_format=lambda v: f"{v:.1f}"))
        print()


# ----------------------------------------------------------------------- main

def main() -> None:
    print("Loading cross_method...")
    cm_df, cm_mouse = load_cross_method()
    print("Loading convergence...")
    cv_df, cv_mouse = load_convergence()

    if not cm_df.empty:
        print_cross_method_summary(cm_df, cm_mouse)
        out = OUT / "cross_method"
        plot_paired_sliders(cm_df, out)
        plot_preference_distribution(cm_df, out)
        plot_aspect_attribution(cm_df, out)
        plot_rating_diff_vs_preference(cm_df, out)
        plot_elo(cm_df, out)
        plot_cross_method_mouse(cm_mouse, out)
        print(f"cross_method plots -> {out}/")
    else:
        print("(no cross_method data)\n")

    if not cv_df.empty:
        print_convergence_summary(cv_df, cv_mouse)
        out = OUT / "convergence"
        plot_rating_vs_iter(cv_df, out)
        plot_mean_rating_vs_iter(cv_df, out)
        plot_dwell_vs_iter(cv_df, out)
        plot_convergence_mouse(cv_mouse, out)
        print(f"convergence plots -> {out}/")
    else:
        print("(no convergence data)")


if __name__ == "__main__":
    main()
