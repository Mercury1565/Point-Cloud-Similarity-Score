#!/usr/bin/env python3
"""
Plot predicted-score distributions from online_model inference JSON logs.

By default, it scans output/*/*_inference.json and generates:
  - One dataset-level histogram per dataset
  - One scene-level histogram per scene (for each dataset)
  - CSV summaries (all rows + per-scene stats)

Example:
    python3 analysis/plot_predicted_score_distributions.py \
        --inference-root output \
        --output-dir output/predicted_score_distributions
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd


def _safe_name(text: str, max_len: int = 120) -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._")
    if not clean:
        clean = "scene"
    return clean[:max_len]


def _find_inference_jsons(root: Path) -> List[Path]:
    return sorted(root.glob("*/*_inference.json"))


def _rows_from_inference_json(path: Path) -> List[Dict]:
    with path.open("r") as f:
        payload = json.load(f)

    dataset = str(payload.get("dataset", path.parent.name))
    rows: List[Dict] = []

    for scene_id, entries in payload.get("scenes", {}).items():
        for entry in entries:
            rows.append(
                {
                    "dataset": dataset,
                    "scene_id": str(scene_id),
                    "frame_idx": int(entry.get("frame_idx", -1)),
                    "predicted_confidence": float(entry.get("predicted_confidence", 0.0)),
                    "predicted_std": float(entry.get("predicted_std", 0.0)),
                    "actual_confidence": float(entry.get("actual_confidence", 0.0)),
                    "decision": str(entry.get("decision", "")),
                    "is_audit_frame": bool(entry.get("is_audit_frame", False)),
                    "source_json": str(path),
                }
            )
    return rows


def _plot_hist(series: pd.Series, title: str, output_path: Path, bins: int) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(series, bins=bins, range=(0.0, 1.0), alpha=0.85)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Predicted confidence")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _summarize_scene(g: pd.DataFrame) -> Dict:
    pred = g["predicted_confidence"]
    return {
        "dataset": g["dataset"].iloc[0],
        "scene_id": g["scene_id"].iloc[0],
        "num_frames": int(len(g)),
        "mean_pred": float(pred.mean()),
        "std_pred": float(pred.std(ddof=0)),
        "min_pred": float(pred.min()),
        "p10_pred": float(pred.quantile(0.10)),
        "p50_pred": float(pred.quantile(0.50)),
        "p90_pred": float(pred.quantile(0.90)),
        "max_pred": float(pred.max()),
        "reuse_rate_pct": float((g["decision"] == "REUSE").mean() * 100.0),
        "audit_rate_pct": float(g["is_audit_frame"].mean() * 100.0),
    }


def build_and_plot(inference_root: Path, output_dir: Path, bins: int) -> None:
    json_paths = _find_inference_jsons(inference_root)
    if not json_paths:
        raise FileNotFoundError(f"No *_inference.json files found under: {inference_root}")

    rows: List[Dict] = []
    for path in json_paths:
        rows.extend(_rows_from_inference_json(path))

    if not rows:
        raise ValueError("No frame-level rows found in inference JSON files.")

    df = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "predicted_scores_all.csv", index=False)

    scene_stats: List[Dict] = []

    # Dataset-level plots
    for dataset, g_ds in df.groupby("dataset", sort=True):
        ds_dir = output_dir / dataset
        ds_dir.mkdir(parents=True, exist_ok=True)

        _plot_hist(
            g_ds["predicted_confidence"],
            f"{dataset}: predicted confidence distribution (all scenes)",
            ds_dir / f"{dataset}_predicted_score_distribution.png",
            bins=bins,
        )

        # Scene-level plots
        scenes_dir = ds_dir / "scenes"
        scenes_dir.mkdir(parents=True, exist_ok=True)

        for scene_id, g_scene in g_ds.groupby("scene_id", sort=True):
            scene_stats.append(_summarize_scene(g_scene))

            scene_file = _safe_name(scene_id) + "_predicted_score_distribution.png"
            _plot_hist(
                g_scene["predicted_confidence"],
                f"{dataset} | {scene_id}\nPredicted confidence distribution",
                scenes_dir / scene_file,
                bins=bins,
            )

    scene_df = pd.DataFrame(scene_stats).sort_values(["dataset", "mean_pred"], ascending=[True, False])
    scene_df.to_csv(output_dir / "predicted_score_summary_by_scene.csv", index=False)

    print(f"Loaded JSON files: {len(json_paths)}")
    print(f"Total frame rows  : {len(df)}")
    print(f"Datasets          : {df['dataset'].nunique()}")
    print(f"Scenes            : {df['scene_id'].nunique()}")
    print(f"Saved outputs to  : {output_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot predicted score distributions for each dataset and each scene."
    )
    parser.add_argument(
        "--inference-root",
        type=str,
        default="output",
        help="Root folder containing dataset subfolders with *_inference.json files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/predicted_score_distributions",
        help="Where plots and summary CSVs will be written.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=40,
        help="Histogram bin count.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_and_plot(
        inference_root=Path(args.inference_root),
        output_dir=Path(args.output_dir),
        bins=args.bins,
    )


if __name__ == "__main__":
    main()
