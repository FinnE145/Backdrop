#!/usr/bin/env python3
"""
aesthetic_label_test.py

Three-phase script for calibrating the aesthetic threshold:
  1. Interactively label unlabelled images in local/aesthetic-library/
  2. Score all labelled images (decode → embed → aesthetic)
  3. Print stats, save JSON, save plot PNG
"""

import json
import subprocess
import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import pipeline.decode
import pipeline.embed
import pipeline.aesthetic

LIBRARY_DIR = Path("local/aesthetic-library")
IMAGE_EXTS = {".heic", ".jpg", ".jpeg", ".png"}

MODEL_NAME = "ViT-L-14-quickgelu"
CLIP_CHECKPOINT = "models/clip-vit-large-patch14.pt"
AESTHETIC_CHECKPOINT = "models/sac+logos+ava1-l14-linearMSE.pth"

COLOR_A = "#0d8c7d"  # dark teal — aesthetic
COLOR_N = "#e07b45"  # rust orange — not aesthetic


# ---------------------------------------------------------------------------
# Phase 1: Labelling
# ---------------------------------------------------------------------------

def _is_labelled(path: Path) -> bool:
    return path.stem.endswith("-A") or path.stem.endswith("-N")


def _rename_with_label(path: Path, label: str) -> Path:
    new_path = path.parent / (path.stem + f"-{label}" + path.suffix)
    path.rename(new_path)
    return new_path


def labelling_pass(library_dir: Path) -> list[tuple[Path, str]]:
    all_paths = sorted(
        p for p in library_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTS
    )
    already = [
        (p, "A" if p.stem.endswith("-A") else "N")
        for p in all_paths if _is_labelled(p)
    ]
    unlabelled = [p for p in all_paths if not _is_labelled(p)]

    print(f"\nFound {len(all_paths)} images ({len(already)} labelled, {len(unlabelled)} to review)")

    newly = []
    for path in unlabelled:
        subprocess.Popen(["open", str(path.resolve())])
        while True:
            raw = input(f"  {path.name}  [a=aesthetic / n=not / s=skip / q=quit]: ").strip().lower()
            if raw in ("a", "n", "s", "q"):
                break
            print("  Enter a, n, s, or q.")

        if raw == "q":
            print("Stopping labelling pass early.")
            break
        if raw == "s":
            print("  skipped")
            continue

        label = raw.upper()
        new_path = _rename_with_label(path, label)
        newly.append((new_path, label))
        print(f"  → {new_path.name}")

    return already + newly


# ---------------------------------------------------------------------------
# Phase 2: Scoring
# ---------------------------------------------------------------------------

def scoring_pass(labelled: list[tuple[Path, str]]) -> list[dict]:
    print(f"\nScoring {len(labelled)} images…")
    records = []
    for path, label in labelled:
        print(f"  {path.name}...", end=" ", flush=True)
        image, _, _ = pipeline.decode.decode_image(str(path))
        embedding = pipeline.embed.embed_image(
            image, model_name=MODEL_NAME, checkpoint_path=CLIP_CHECKPOINT
        )
        score = pipeline.aesthetic.score_image(embedding, AESTHETIC_CHECKPOINT)
        print(f"{score:.3f}")
        records.append({"name": path.name, "score": score, "label": label})
    return records


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def _group_stats(scores: list[float]) -> dict | None:
    if not scores:
        return None
    a = np.asarray(scores)
    return {
        "count": len(scores),
        "min": float(a.min()),
        "max": float(a.max()),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "std": float(a.std()),
    }


def _compute_cutoff(records: list[dict]) -> tuple[float | None, int, list[str]]:
    """
    Find the threshold minimising (A-below + N-above) misclassifications.
    Candidates are midpoints between consecutive distinct scores.
    When multiple candidates tie, use their midpoint.
    """
    a_sc = sorted(r["score"] for r in records if r["label"] == "A")
    n_sc = sorted(r["score"] for r in records if r["label"] == "N")
    if not a_sc or not n_sc:
        return None, 0, []

    all_sc = sorted(set(a_sc + n_sc))
    candidates = [all_sc[0] - 0.5]
    for i in range(len(all_sc) - 1):
        candidates.append((all_sc[i] + all_sc[i + 1]) / 2)
    candidates.append(all_sc[-1] + 0.5)

    best_err, best_ts = None, []
    for t in candidates:
        e = sum(1 for s in a_sc if s < t) + sum(1 for s in n_sc if s >= t)
        if best_err is None or e < best_err:
            best_err, best_ts = e, [t]
        elif e == best_err:
            best_ts.append(t)

    cutoff = (best_ts[0] + best_ts[-1]) / 2
    wrong = [
        r["name"] for r in records
        if (r["label"] == "A" and r["score"] < cutoff)
        or (r["label"] == "N" and r["score"] >= cutoff)
    ]
    return cutoff, best_err, wrong


def _overlap(sa: dict | None, sn: dict | None) -> list[float] | None:
    if sa and sn and sn["max"] >= sa["min"]:
        return [sa["min"], sn["max"]]
    return None


# ---------------------------------------------------------------------------
# Phase 3a: Console stats
# ---------------------------------------------------------------------------

def print_stats(
    records: list[dict],
    cutoff: float | None,
    cutoff_errors: int,
    misclassified: list[str],
) -> None:
    a_sc = [r["score"] for r in records if r["label"] == "A"]
    n_sc = [r["score"] for r in records if r["label"] == "N"]
    sa, sn = _group_stats(a_sc), _group_stats(n_sc)

    print("\n=== Stats ===")
    for lbl, s in [("A (aesthetic)", sa), ("N (not aesthetic)", sn)]:
        if s:
            print(f"\n  {lbl}  (n={s['count']})")
            print(
                f"    min={s['min']:.3f}  max={s['max']:.3f}  "
                f"mean={s['mean']:.3f}  median={s['median']:.3f}  std={s['std']:.3f}"
            )

    if sa and sn:
        gap = sa["mean"] - sn["mean"]
        print(f"\n  Means gap:  {gap:+.3f}")
        ov = _overlap(sa, sn)
        if ov:
            print(f"  Overlap:    [{ov[0]:.3f}, {ov[1]:.3f}]")
        else:
            print("  Overlap:    none")

    if cutoff is not None:
        print(f"\n  Recommended cutoff:  {cutoff:.3f}")
        print(f"  Misclassifications:  {cutoff_errors}", end="")
        print(f"  ({', '.join(misclassified)})" if misclassified else "")

    print("\n  Notable images:")
    for lbl in ("A", "N"):
        grp = sorted(
            [(r["score"], r["name"]) for r in records if r["label"] == lbl],
            reverse=True,
        )
        if not grp:
            continue
        top3 = grp[:3]
        bot3 = list(reversed(grp[-3:]))
        print(f"    {lbl} top 3: " + ", ".join(f"{n} ({s:.3f})" for s, n in top3))
        print(f"    {lbl} bot 3: " + ", ".join(f"{n} ({s:.3f})" for s, n in bot3))

    if misclassified:
        print(f"\n  Wrong-side:  {', '.join(misclassified)}")


# ---------------------------------------------------------------------------
# Phase 3b: JSON output
# ---------------------------------------------------------------------------

def write_json(
    records: list[dict],
    cutoff: float | None,
    cutoff_errors: int,
    misclassified: list[str],
    output_path: str,
) -> None:
    a_sc = [r["score"] for r in records if r["label"] == "A"]
    n_sc = [r["score"] for r in records if r["label"] == "N"]
    sa, sn = _group_stats(a_sc), _group_stats(n_sc)

    data = {
        "records": records,
        "stats": {
            "A": sa,
            "N": sn,
            "means_gap": (sa["mean"] - sn["mean"]) if sa and sn else None,
            "overlap": _overlap(sa, sn),
            "cutoff": cutoff,
            "cutoff_misclassification_count": cutoff_errors,
            "cutoff_misclassified_names": misclassified,
        },
        "meta": {
            "image_count": len(records),
            "datetime": datetime.datetime.now().isoformat(),
            "model": {
                "clip_model": MODEL_NAME,
                "clip_checkpoint": CLIP_CHECKPOINT,
                "aesthetic_checkpoint": AESTHETIC_CHECKPOINT,
            },
        },
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"JSON  → {output_path}")


# ---------------------------------------------------------------------------
# Phase 3c: Plot
# ---------------------------------------------------------------------------

def make_plot(
    records: list[dict],
    cutoff: float | None = None,
    png_path: str = "local/aesthetic_scores.png",
) -> None:
    computed_cutoff, _, _ = _compute_cutoff(records)
    if cutoff is None:
        cutoff = computed_cutoff

    # misclassifications relative to whichever cutoff we're displaying
    if cutoff is not None:
        wrong_names = [
            r["name"] for r in records
            if (r["label"] == "A" and r["score"] < cutoff)
            or (r["label"] == "N" and r["score"] >= cutoff)
        ]
    else:
        wrong_names = []
    wrong_set = set(wrong_names)

    a_recs = [r for r in records if r["label"] == "A"]
    n_recs = [r for r in records if r["label"] == "N"]
    a_sc = np.asarray([r["score"] for r in a_recs]) if a_recs else np.array([])
    n_sc = np.asarray([r["score"] for r in n_recs]) if n_recs else np.array([])

    sa = _group_stats(a_sc.tolist())
    sn = _group_stats(n_sc.tolist())
    ov = _overlap(sa, sn)
    has_overlap = ov is not None

    BIN_W = 0.15
    ALPHA = 0.55
    JITTER = 0.18
    rng = np.random.default_rng(42)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True,
        gridspec_kw={"height_ratios": [1, 2]},
    )
    fig.patch.set_facecolor("#f8f8f8")
    for ax in (ax_top, ax_bot):
        ax.set_facecolor("#f8f8f8")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Decorations shared between panels
    for ax in (ax_top, ax_bot):
        if has_overlap:
            ax.axvspan(ov[0], ov[1], color="#cccccc", alpha=0.4, zorder=1)
        if cutoff is not None:
            ax.axvline(cutoff, color="#444444", linestyle="--", linewidth=1.2, zorder=4)

    # ---- Top panel: strip/scatter ----
    ROW = {"A": 1.0, "N": 0.0}
    for group_label, recs, color in [("A", a_recs, COLOR_A), ("N", n_recs, COLOR_N)]:
        if not recs:
            continue
        scores = np.asarray([r["score"] for r in recs])
        yc = ROW[group_label]
        ys = yc + rng.uniform(-JITTER, JITTER, len(scores))
        ax_top.scatter(scores, ys, color=color, alpha=0.85, s=45, zorder=3)

        m = float(scores.mean())
        ax_top.axvline(m, color=color, linewidth=1.4, alpha=0.75, zorder=2)
        ax_top.text(
            m, yc + JITTER + 0.08, f"{m:.2f}",
            color=color, ha="center", va="bottom", fontsize=8, fontweight="bold",
        )

    # Annotations: each group's min + max + wrong-side images (deduplicated by name)
    to_ann: dict[str, tuple[float, float, str]] = {}
    for group_label, recs in [("A", a_recs), ("N", n_recs)]:
        if not recs:
            continue
        scores_map = {r["name"]: r["score"] for r in recs}
        min_s = min(scores_map.values())
        max_s = max(scores_map.values())
        for r in recs:
            if r["score"] in (min_s, max_s) or r["name"] in wrong_set:
                to_ann[r["name"]] = (r["score"], ROW[group_label], group_label)

    for name, (sx, yc, gl) in to_ann.items():
        y_off = yc + (0.44 if gl == "A" else -0.44)
        short = Path(name).stem[-12:]
        ax_top.annotate(
            short,
            xy=(sx, yc),
            xytext=(sx, y_off),
            fontsize=6.5,
            ha="center",
            va="bottom" if gl == "A" else "top",
            arrowprops=dict(arrowstyle="-", color="#999999", lw=0.8),
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#cccccc", lw=0.6),
            zorder=5,
        )

    ax_top.set_yticks([0, 1])
    ax_top.set_yticklabels(["NOT aesthetic (-N)", "aesthetic (-A)"], fontsize=9)
    ax_top.set_ylim(-0.75, 1.8)
    ax_top.set_title("LAION Aesthetic Score — Labelled Sample", fontsize=12, pad=8)

    handles = []
    if len(a_sc):
        handles.append(Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_A,
                               markersize=7, label=f"A (n={len(a_sc)})"))
    if len(n_sc):
        handles.append(Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_N,
                               markersize=7, label=f"N (n={len(n_sc)})"))
    if has_overlap:
        handles.append(Patch(facecolor="#cccccc", alpha=0.55, label="overlap zone"))
    if cutoff is not None:
        handles.append(Line2D([0], [0], color="#444444", linestyle="--", linewidth=1.2,
                               label=f"cut ≈ {cutoff:.2f}"))
    ax_top.legend(handles=handles, fontsize=8, loc="upper left")

    # ---- Bottom panel: histogram ----
    all_flat = list(a_sc) + list(n_sc)
    if all_flat:
        lo = min(all_flat) - BIN_W
        hi = max(all_flat) + BIN_W
        bins = np.arange(lo, hi + BIN_W, BIN_W)
    else:
        bins = 10

    if len(a_sc):
        ax_bot.hist(a_sc, bins=bins, color=COLOR_A, alpha=ALPHA, label=f"A (n={len(a_sc)})")
    if len(n_sc):
        ax_bot.hist(n_sc, bins=bins, color=COLOR_N, alpha=ALPHA, label=f"N (n={len(n_sc)})")

    ax_bot.set_xlabel("LAION Aesthetic Score", fontsize=10)
    ax_bot.set_ylabel("Count", fontsize=10)
    ax_bot.legend(fontsize=9)

    # Stats textbox
    lines = []
    if sa:
        lines.append(f"A   min={sa['min']:.2f}  max={sa['max']:.2f}  mean={sa['mean']:.2f}  med={sa['median']:.2f}")
    if sn:
        lines.append(f"N   min={sn['min']:.2f}  max={sn['max']:.2f}  mean={sn['mean']:.2f}  med={sn['median']:.2f}")
    if sa and sn:
        lines.append(f"gap  {sa['mean'] - sn['mean']:+.3f}")
    lines.append(f"ovlp  [{ov[0]:.2f}, {ov[1]:.2f}]" if has_overlap else "ovlp  none")
    if cutoff is not None:
        lines.append(f"cut  {cutoff:.2f}   {len(wrong_names)} miscls")
        for nm in wrong_names:
            lines.append(f"  ✗ {Path(nm).stem}")

    ax_bot.text(
        0.98, 0.97, "\n".join(lines),
        transform=ax_bot.transAxes,
        fontsize=7.5, va="top", ha="right",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", fc="white", ec="#cccccc", alpha=0.92),
    )

    plt.tight_layout()
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot  → {png_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(
    library_dir: Path = LIBRARY_DIR,
    json_path: str = "local/aesthetic_results.json",
    png_path: str = "local/aesthetic_scores.png",
    cutoff_override: float | None = None,
) -> None:
    labelled = labelling_pass(Path(library_dir))
    if not labelled:
        print("No labelled images found. Exiting.")
        return

    records = scoring_pass(labelled)
    if not records:
        print("No scored records. Exiting.")
        return

    cutoff, cutoff_errors, misclassified = _compute_cutoff(records)
    print_stats(records, cutoff, cutoff_errors, misclassified)
    write_json(records, cutoff, cutoff_errors, misclassified, json_path)
    make_plot(records, cutoff=cutoff_override, png_path=png_path)


if __name__ == "__main__":
    main()
