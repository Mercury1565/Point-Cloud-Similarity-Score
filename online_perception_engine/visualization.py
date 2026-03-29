"""Visualization functions for online perception engine metrics."""

import matplotlib.pyplot as plt
from online_perception_engine.engine import OnlinePerceptionEngine


def generate_learning_curve(engine: OnlinePerceptionEngine, output_path: str) -> None:
    """Generate learning curve visualization showing model adaptation over time.
    
    Creates a matplotlib figure plotting prediction errors (MAE) at audit frames
    to visualize how the model's prediction accuracy changes as it learns from
    streaming data.
    
    Args:
        engine: OnlinePerceptionEngine instance with processed decisions
        output_path: File path where the figure should be saved (e.g., "learning_curve.png")
    """
    # Extract audit frame indices and prediction errors from decisions
    audit_frames = [d.frame_idx for d in engine.decisions if d.is_audit_frame]
    errors = [d.prediction_error for d in engine.decisions if d.is_audit_frame]
    
    # Create matplotlib figure with specified size
    plt.figure(figsize=(10, 6))
    
    # Plot frame index on x-axis and MAE on y-axis with markers
    plt.plot(audit_frames, errors, marker='o')
    
    # Add labels
    plt.xlabel("Frame Index")
    plt.ylabel("Prediction Error (MAE)")
    
    # Add title
    plt.title("Learning Curve: Model Adaptation Over Time")
    
    # Add grid
    plt.grid(True)
    
    # Save figure to output_path
    plt.savefig(output_path)
    
    # Close the figure to free memory
    plt.close()


def generate_confidence_comparison(engine: OnlinePerceptionEngine, output_path: str) -> None:
    """Generate confidence comparison visualization showing predicted vs actual confidence.
    
    Creates a matplotlib figure plotting predicted and actual confidence values over time,
    along with horizontal lines for confidence_threshold and safety_threshold to visualize
    decision boundaries and safety margins.
    
    Args:
        engine: OnlinePerceptionEngine instance with processed decisions
        output_path: File path where the figure should be saved (e.g., "confidence_comparison.png")
    """
    # Extract frame indices, predicted confidence, and actual confidence from decisions
    frame_indices = [d.frame_idx for d in engine.decisions]
    predicted = [d.predicted_confidence for d in engine.decisions]
    actual = [d.actual_confidence for d in engine.decisions]
    
    # Create matplotlib figure with specified size
    plt.figure(figsize=(12, 6))
    
    # Plot predicted confidence line
    plt.plot(frame_indices, predicted, label="Predicted", alpha=0.7)
    
    # Plot actual confidence line
    plt.plot(frame_indices, actual, label="Actual", alpha=0.7)
    
    # Add horizontal line for confidence_threshold
    plt.axhline(y=engine.confidence_threshold, color='r', linestyle='--', label="Decision Threshold")
    
    # Add horizontal line for safety_threshold
    plt.axhline(y=engine.safety_threshold, color='orange', linestyle='--', label="Safety Threshold")
    
    # Add labels
    plt.xlabel("Frame Index")
    plt.ylabel("Confidence")
    
    # Add title
    plt.title("Predicted vs Actual Confidence Over Time")
    
    # Add legend
    plt.legend()
    
    # Add grid
    plt.grid(True)
    
    # Save figure to output_path
    plt.savefig(output_path)
    
    # Close the figure to free memory
    plt.close()
