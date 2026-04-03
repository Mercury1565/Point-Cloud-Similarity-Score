import argparse
import os
import glob
from online_model.simulation import run_simulation

KNOWN_DATASETS = ["nuscenes", "nuscenes_full", "kitti", "waymo"]


def _print_metrics(label: str, metrics) -> None:
    total = metrics.total_frames
    print(f"\n{'=' * 60}")
    print(f"  RESULTS  —  {label.upper()}")
    print(f"{'=' * 60}")
    print(f"Total Frames Processed : {total}")
    print(f"REUSE Decisions        : {metrics.reuse_count} ({metrics.reuse_count / total * 100:.1f}%)")
    print(f"FULL_DETECTION         : {metrics.full_detection_count} ({metrics.full_detection_count / total * 100:.1f}%)")
    print(f"Latency Saved          : {metrics.cumulative_latency_saved_ms:.1f} ms")
    print(f"Safety Violations      : {metrics.safety_violations}")
    print(f"Audit Frames           : {len(metrics.audit_frames)}")
    if metrics.mean_absolute_errors:
        avg_mae = sum(metrics.mean_absolute_errors) / len(metrics.mean_absolute_errors)
        print(f"Avg Prediction Error   : {avg_mae:.4f}")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="Run the Online Bayesian Perception Engine — train on a seed fraction, "
                    "then stream inference + online updates on the rest.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=KNOWN_DATASETS + ["all"],
        default="all",
        help="Dataset to simulate. 'all' runs a separate model for each dataset "
             "whose CSV exists in --csv-dir.",
    )
    parser.add_argument(
        "--csv-dir",
        type=str,
        default="data/csv",
        help="Directory containing *_training_data.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Root output directory. Per-dataset sub-dirs are created automatically.",
    )
    parser.add_argument(
        "--model-state-dir",
        type=str,
        default=None,
        help="If set, the Bayesian posterior for each dataset is saved here as "
             "{dataset}.npz for later reloading.",
    )
    parser.add_argument(
        "--seed-fraction",
        type=float,
        default=0.15,
        help="Fraction of each dataset used to cold-start (train) the model. "
             "Must be in (0, 1). "
             "The remaining (1 - seed-fraction) is streamed for inference.",
    )
    parser.add_argument(
        "--split-mode",
        type=str,
        choices=["frame", "scene"],
        default="frame",
        help="How to split seed vs streaming data. "
             "'frame' uses an exact frame-count split; "
             "'scene' keeps whole-scene boundaries.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.85,
        help="Lower-confidence-bound threshold for REUSE decisions.",
    )
    parser.add_argument(
        "--audit-interval",
        type=int,
        default=5,
        help="Every N-th streamed frame forces a full detection and a Bayesian "
             "posterior update.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Prior precision for Bayesian LR — higher = stronger regularisation.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=25.0,
        help="Noise precision for Bayesian LR — 1 / noise_variance.",
    )
    parser.add_argument(
        "--uncertainty-weight",
        type=float,
        default=1.0,
        help="k in REUSE rule: REUSE iff (mean - k*std) > threshold. "
             "Higher = more conservative.",
    )

    args = parser.parse_args()

    shared_kwargs = dict(
        csv_dir=args.csv_dir,
        visuals_output_dir=args.output_dir,
        confidence_threshold=args.confidence_threshold,
        audit_interval=args.audit_interval,
        seed_fraction=args.seed_fraction,
        alpha=args.alpha,
        beta=args.beta,
        uncertainty_weight=args.uncertainty_weight,
        model_state_dir=args.model_state_dir,
        split_mode=args.split_mode,
    )

    print("Configuration:")
    print(f"  Dataset(s)            : {args.dataset}")
    print(f"  CSV directory         : {args.csv_dir}")
    print(f"  Output directory      : {args.output_dir}")
    print(f"  Seed fraction         : {args.seed_fraction} "
          f"({args.seed_fraction*100:.0f}% train / {(1-args.seed_fraction)*100:.0f}% stream)")
    print(f"  Split mode            : {args.split_mode}")
    print(f"  Confidence threshold  : {args.confidence_threshold}")
    print(f"  Audit interval        : {args.audit_interval}")
    print(f"  Alpha (prior prec.)   : {args.alpha}")
    print(f"  Beta (noise prec.)    : {args.beta}")
    print(f"  Uncertainty weight    : {args.uncertainty_weight}")
    if args.model_state_dir:
        print(f"  Model state dir       : {args.model_state_dir}")
    print()

    if args.dataset == "all":
        available = [
            ds for ds in KNOWN_DATASETS
            if (
                os.path.exists(os.path.join(args.csv_dir, f"{ds}_training_data.csv"))
                or glob.glob(os.path.join(args.csv_dir, f"*{ds}*.csv"))
            )
        ]
        if not available:
            print(f"No dataset CSVs found in '{args.csv_dir}'. Nothing to run.")
            return

        print(f"Found datasets: {', '.join(available)}\n")
        for ds in available:
            try:
                metrics = run_simulation(dataset=ds, **shared_kwargs)
                _print_metrics(ds, metrics)
            except Exception as exc:
                print(f"[{ds}] ERROR: {exc}")
    else:
        metrics = run_simulation(dataset=args.dataset, **shared_kwargs)
        _print_metrics(args.dataset, metrics)


if __name__ == "__main__":
    main()
