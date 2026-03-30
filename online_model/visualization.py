import matplotlib.pyplot as plt
from online_model.engine import OnlinePerceptionEngine
import matplotlib.pyplot as plt
import numpy as np


def generate_learning_curve(engine: OnlinePerceptionEngine, output_path: str) -> None:
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

def generate_decision_histogram(engine: OnlinePerceptionEngine, output_path: str):
    decisions = engine.decisions
    preds = [d.predicted_confidence for d in decisions]
    actuals = [d.actual_confidence for d in decisions]
    
    full_det = []
    safe_reuse = []
    violations = []

    for p, a in zip(preds, actuals):
        if p <= engine.confidence_threshold:
            full_det.append(p)
        elif a < engine.safety_threshold:
            violations.append(p)
        else:
            safe_reuse.append(p)

    plt.figure(figsize=(10, 6))
    bins = np.linspace(0, 1, 20)
    
    plt.hist([full_det, safe_reuse, violations], bins=bins, stacked=True, 
             color=['#bdc3c7', '#2ecc71', '#e74c3c'], 
             label=['Full Detection', 'Safe Reuse', 'Safety Violation'],
             edgecolor='black', alpha=0.8)

    plt.axvline(engine.confidence_threshold, color='red', linestyle='--', label='Decision Threshold')
    plt.xlabel("Predicted Confidence")
    plt.ylabel("Frequency")
    plt.title("Decision Distribution & Safety Reliability")
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(output_path)
    plt.close()