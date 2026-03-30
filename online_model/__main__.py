import argparse
import os
from online_model.simulation import run_simulation

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(
        description="Run the Online Perception Engine simulation on CSV data"
    )

    # Add optional argument: --output-dir (default: "output")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory where output plots will be saved (default: output)"
    )
    
    # Add optional argument: --confidence-threshold (default: 0.85)
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.85,
        help="Predicted confidence above which REUSE is chosen (default: 0.85)"
    )
    
    # Add optional argument: --audit-interval (default: 5)
    parser.add_argument(
        "--audit-interval",
        type=int,
        default=5,
        help="Number of frames between model updates (default: 5)"
    )
    
    # Add optional argument: --seed-batch-size (default: 50)
    parser.add_argument(
        "--seed-batch-size",
        type=int,
        default=50,
        help="Initial number of frames for model warm-up (default: 50)"
    )
    
    # Parse arguments
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Call run_simulation() with provided arguments
    print(f"Configuration:")
    print(f"  - Confidence Threshold: {args.confidence_threshold}")
    print(f"  - Audit Interval: {args.audit_interval}")
    print(f"  - Seed Batch Size: {args.seed_batch_size}")
    print()
    
    metrics = run_simulation(
        csv_dir="data/csv",
        visuals_output_dir=args.output_dir,
        confidence_threshold=args.confidence_threshold,
        audit_interval=args.audit_interval,
        seed_batch_size=args.seed_batch_size
    )
    
    # Print metrics summary to console
    print("=" * 60)
    print("SIMULATION RESULTS")
    print("=" * 60)
    print(f"Total Frames Processed: {metrics.total_frames}")
    print(f"REUSE Decisions: {metrics.reuse_count} ({metrics.reuse_count / metrics.total_frames * 100:.1f}%)")
    print(f"FULL_DETECTION Decisions: {metrics.full_detection_count} ({metrics.full_detection_count / metrics.total_frames * 100:.1f}%)")
    print(f"Cumulative Latency Saved: {metrics.cumulative_latency_saved_ms:.2f} ms")
    print(f"Safety Violations: {metrics.safety_violations}")
    print(f"Audit Frames: {len(metrics.audit_frames)}")
    
    if metrics.mean_absolute_errors:
        avg_mae = sum(metrics.mean_absolute_errors) / len(metrics.mean_absolute_errors)
        print(f"Average Prediction Error (MAE): {avg_mae:.4f}")
    
    print("=" * 60)
    print()
    
if __name__ == "__main__":
    main()
