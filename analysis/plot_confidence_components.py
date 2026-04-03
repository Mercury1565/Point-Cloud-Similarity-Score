#!/usr/bin/env python3
"""
Plot label components for weight tuning in notebooks or from CLI.

This script computes per-frame-pair:
  - F1 (ID consistency)
  - mIoU (geometric consistency)
  - confidence_current (as produced by ConfidenceScorer)
  - confidence_tuned (harmonic confidence with user-provided weights)

It then writes CSV summaries and plots:
  1) F1 vs confidence_current, mIoU vs confidence_current
  2) F1 vs confidence_tuned,   mIoU vs confidence_tuned
  3) F1 vs mIoU colored by confidence_current
  4) F1 vs mIoU colored by confidence_tuned
  5) confidence_current vs confidence_tuned

Example:
    python analysis/plot_confidence_components.py \
        --inputs unified_nuscenes_mini.json unified_nuscenes_full.json \
        --output-dir output/weight_tuning \
        --w-f1 0.7 --w-miou 0.3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Tuple
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure imports work when this file is executed as: python analysis/<script>.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from confidence_scorer.scorer import ConfidenceScorer


def _normalize_weights(w_f1: float, w_miou: float) -> Tuple[float, float]:
    if w_f1 < 0.0 or w_miou < 0.0:
        raise ValueError("w_f1 and w_miou must be >= 0")
    s = w_f1 + w_miou
    if s <= 0.0:
        raise ValueError("w_f1 + w_miou must be > 0")
    return w_f1 / s, w_miou / s


def harmonic_confidence(
    f1: np.ndarray, miou: np.ndarray, w_f1: float, w_miou: float, epsilon: float = 1e-6
) -> np.ndarray:
    w_f1, w_miou = _normalize_weights(w_f1, w_miou)
    f1_safe = f1 + epsilon
    miou_safe = miou + epsilon
    conf = 1.0 / ((w_f1 / f1_safe) + (w_miou / miou_safe))
    return np.clip(conf, 0.0, 1.0)


def build_score_dataframe(unified_json_paths: Iterable[Path]) -> pd.DataFrame:
    scorer = ConfidenceScorer()
    rows: List[dict] = []

    for path in unified_json_paths:
        with path.open("r") as f:
            scenes = json.load(f)

        dataset_label = path.stem.replace("unified_", "")

        for scene in scenes:
            scene_id = str(scene.get("scene_id", "unknown"))
            frames = scene.get("frame_list", [])

            for t in range(1, len(frames)):
                frame_t = frames[t]
                frame_prev = frames[t - 1]
                objs_t = frame_t.get("object_list", [])
                objs_prev = frame_prev.get("object_list", [])

                score = scorer.calculate_score(objs_t, objs_prev)

                rows.append(
                    {
                        "source_file": path.name,
                        "dataset": dataset_label,
                        "scene_id": scene_id,
                        "frame_pair_idx": t,
                        "f1": float(score["f1"]),
                        "miou": float(score["miou"]),
                        "confidence_current": float(score["confidence_score"]),
                        "obj_count_t": len(objs_t),
                        "obj_count_prev": len(objs_prev),
                        "both_empty": (len(objs_t) == 0 and len(objs_prev) == 0),
                    }
                )

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No frame-pair rows were generated from the provided unified JSON files.")
    return df


def _pair_plot(df: pd.DataFrame, conf_col: str, title: str, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    axes[0].scatter(df["f1"], df[conf_col], s=9, alpha=0.22)
    axes[0].set_xlabel("F1 score")
    axes[0].set_ylabel("Confidence")
    axes[0].set_title("F1 vs Confidence")
    axes[0].set_xlim(-0.02, 1.02)
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].grid(alpha=0.25)
    r_f1 = df["f1"].corr(df[conf_col])
    axes[0].text(0.03, 0.96, f"Pearson r = {r_f1:.3f}", transform=axes[0].transAxes, va="top")

    axes[1].scatter(df["miou"], df[conf_col], s=9, alpha=0.22)
    axes[1].set_xlabel("mIoU")
    axes[1].set_ylabel("Confidence")
    axes[1].set_title("mIoU vs Confidence")
    axes[1].set_xlim(-0.02, 1.02)
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].grid(alpha=0.25)
    r_miou = df["miou"].corr(df[conf_col])
    axes[1].text(0.03, 0.96, f"Pearson r = {r_miou:.3f}", transform=axes[1].transAxes, va="top")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _joint_plot(df: pd.DataFrame, conf_col: str, title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.2))

    sc = ax.scatter(
        df["f1"],
        df["miou"],
        c=df[conf_col],
        cmap="viridis",
        s=11,
        alpha=0.35,
        vmin=0.0,
        vmax=1.0,
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Confidence")

    both_empty = df["both_empty"].values
    if both_empty.any():
        ax.scatter(
            df.loc[both_empty, "f1"],
            df.loc[both_empty, "miou"],
            s=26,
            facecolors="none",
            edgecolors="red",
            linewidths=1.0,
            label="Both frames empty",
        )
        ax.legend(loc="lower right")

    ax.set_xlabel("F1 score")
    ax.set_ylabel("mIoU")
    ax.set_title(title)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _overlay_plot(df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    ax.scatter(df["confidence_current"], df["confidence_tuned"], s=9, alpha=0.22)
    ax.plot([0.0, 1.0], [0.0, 1.0], "k--", linewidth=1.0)
    ax.set_xlabel("Current confidence")
    ax.set_ylabel("Tuned confidence")
    ax.set_title("Current vs Tuned Confidence")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25)
    r_val = df["confidence_current"].corr(df["confidence_tuned"])
    ax.text(0.03, 0.96, f"Pearson r = {r_val:.3f}", transform=ax.transAxes, va="top")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_plots(df: pd.DataFrame, output_dir: Path, w_f1: float, w_miou: float, epsilon: float) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    df = df.copy()
    df["confidence_tuned"] = harmonic_confidence(
        df["f1"].values, df["miou"].values, w_f1=w_f1, w_miou=w_miou, epsilon=epsilon
    )

    df.to_csv(output_dir / "pairwise_scores.csv", index=False)

    scene_summary = (
        df.groupby(["dataset", "scene_id"], as_index=False)
        .agg(
            pair_count=("frame_pair_idx", "count"),
            mean_f1=("f1", "mean"),
            mean_miou=("miou", "mean"),
            mean_conf_current=("confidence_current", "mean"),
            mean_conf_tuned=("confidence_tuned", "mean"),
            both_empty_pairs=("both_empty", "sum"),
        )
        .sort_values(["dataset", "mean_conf_current"], ascending=[True, False])
    )
    scene_summary.to_csv(output_dir / "scene_summary.csv", index=False)

    _pair_plot(
        df,
        "confidence_current",
        "F1/mIoU vs Current Confidence (all scenes)",
        output_dir / "f1_miou_vs_current_confidence.png",
    )
    _pair_plot(
        df,
        "confidence_tuned",
        f"F1/mIoU vs Tuned Confidence (w_f1={w_f1:.3f}, w_miou={w_miou:.3f})",
        output_dir / "f1_miou_vs_tuned_confidence.png",
    )
    _joint_plot(
        df,
        "confidence_current",
        "All scenes: F1 vs mIoU (color = current confidence)",
        output_dir / "all_scenes_current_confidence.png",
    )
    _joint_plot(
        df,
        "confidence_tuned",
        f"All scenes: F1 vs mIoU (color = tuned confidence, w_f1={w_f1:.3f}, w_miou={w_miou:.3f})",
        output_dir / "all_scenes_tuned_confidence.png",
    )
    _overlay_plot(df, output_dir / "current_vs_tuned_confidence.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot F1/mIoU/confidence relationships from unified JSON files."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="One or more unified_*.json files (e.g., unified_nuscenes_mini.json).",
    )
    parser.add_argument(
        "--output-dir",
        default="output/weight_tuning",
        help="Directory for plots and CSV summaries.",
    )
    parser.add_argument(
        "--w-f1",
        type=float,
        default=0.5,
        help="Weight for F1 in tuned harmonic confidence.",
    )
    parser.add_argument(
        "--w-miou",
        type=float,
        default=0.5,
        help="Weight for mIoU in tuned harmonic confidence.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-6,
        help="Numerical stability epsilon for harmonic confidence.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = [Path(p) for p in args.inputs]
    for path in input_paths:
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

    w_f1, w_miou = _normalize_weights(args.w_f1, args.w_miou)

    df = build_score_dataframe(input_paths)
    make_plots(
        df=df,
        output_dir=Path(args.output_dir),
        w_f1=w_f1,
        w_miou=w_miou,
        epsilon=args.epsilon,
    )

    print(f"Rows analyzed: {len(df)}")
    print(f"Unique scenes: {df['scene_id'].nunique()}")
    print(f"Outputs saved to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
