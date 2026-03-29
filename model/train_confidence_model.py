import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import joblib
from typing import TypedDict, Tuple
import os

class TrainingData(TypedDict):
    """Structure for training data split."""
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


class EvaluationMetrics(TypedDict):
    """Model evaluation results."""
    mae: float
    r2: float
    skip_percentage: float
    skip_accuracy: float


class ROIAnalysis(TypedDict):
    """Computational ROI analysis results."""
    threshold: float
    total_frames: int
    skip_count: int
    skip_percentage: float
    correct_skips: int
    skip_accuracy: float

# Constants for model configuration and data splitting
DEFAULT_TEST_SIZE = 0.2  # 20% of data reserved for testing
DEFAULT_RANDOM_STATE = 42  # Seed for reproducible random splits
DEFAULT_CONFIDENCE_THRESHOLD = 0.85  # Threshold for skip decisions (85% confidence)
DEFAULT_N_ESTIMATORS = 100  # Number of trees in random forest
DEFAULT_MAX_DEPTH = 10  # Maximum depth of each tree
DEFAULT_MODEL_FILEPATH = "confidence_rf_model.pkl"


def load_and_split_data(
    filepath: str, 
    test_size: float = DEFAULT_TEST_SIZE, 
    random_state: int = DEFAULT_RANDOM_STATE
) -> TrainingData:
    """
    Load training data from CSV and split into training and testing sets.
    
    Parameters:
        filepath: Path to the CSV file containing training data
        test_size: Proportion of data to use for testing (default: 0.2)
        random_state: Random seed for reproducible splits (default: 42)
    
    Returns:
        TrainingData dictionary containing X_train, X_test, y_train, y_test
    """
    # Validate test_size is in valid range (0.0, 1.0)
    if not (0.0 < test_size < 1.0):
        raise ValueError(
            f"test_size must be in range (0.0, 1.0), got {test_size}"
        )
    
    # Validate random_state is a non-negative integer
    if not isinstance(random_state, int) or random_state < 0:
        raise ValueError(
            f"random_state must be a non-negative integer, got {random_state}"
        )
    
    # Check if file exists before attempting to load
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Training data file not found: {filepath}")
    
    # Load CSV data into pandas DataFrame
    df = pd.read_csv(filepath)
    
    # Identify target column
    target_col = 'target_confidence'
    
    # Validate minimum column count (at least one feature + target)
    if len(df.columns) < 2:
        raise ValueError(
            f"CSV must contain at least one feature column and a target column. Found columns: {df.columns.tolist()}"
        )
    
    # Separate features (X) from target variable (y)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Split data into training and testing sets with stratified sampling
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Return structured dictionary with all data splits
    return TrainingData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test
    )


def train_model(
    X_train: pd.DataFrame, 
    y_train: pd.Series
) -> RandomForestRegressor:
    """
    Train a RandomForestRegressor model on the provided training data.
    
    Parameters:
        X_train: DataFrame containing training features
        y_train: Series containing training target values (confidence scores)
    
    Returns:
        Fitted RandomForestRegressor model
    """
    # Validate training data is non-empty
    if X_train.empty:
        raise ValueError("X_train is empty. Cannot train model with no data.")
    
    if y_train.empty:
        raise ValueError("y_train is empty. Cannot train model with no target values.")
    
    # Ensure feature and target arrays have matching lengths
    if len(X_train) != len(y_train):
        raise ValueError(
            f"X_train and y_train must have the same length. "
            f"Got X_train: {len(X_train)}, y_train: {len(y_train)}"
        )
    
    # Check for NaN (missing) values in features
    if X_train.isnull().any().any():
        nan_count = X_train.isnull().sum().sum()
        raise ValueError(
            f"X_train contains {nan_count} NaN values. Please clean the data before training."
        )
    
    # Check for NaN (missing) values in target
    if y_train.isnull().any():
        nan_count = y_train.isnull().sum()
        raise ValueError(
            f"y_train contains {nan_count} NaN values. Please clean the data before training."
        )
    
    # Check for infinite values in features (can cause numerical instability)
    if np.isinf(X_train.values).any():
        inf_count = np.isinf(X_train.values).sum()
        raise ValueError(
            f"X_train contains {inf_count} infinite values. Please clean the data before training."
        )
    
    # Check for infinite values in target
    if np.isinf(y_train.values).any():
        inf_count = np.isinf(y_train.values).sum()
        raise ValueError(
            f"y_train contains {inf_count} infinite values. Please clean the data before training."
        )
    
    # Initialize RandomForestRegressor with optimized hyperparameters
    # n_estimators: number of decision trees in the forest
    # max_depth: maximum depth of each tree (prevents overfitting)
    # random_state: ensures reproducible results
    # n_jobs: number of parallel jobs (-1 uses all available CPU cores)
    model = RandomForestRegressor(
        n_estimators=DEFAULT_N_ESTIMATORS,
        max_depth=DEFAULT_MAX_DEPTH,
        random_state=DEFAULT_RANDOM_STATE,
        n_jobs=-1
    )
    
    # Train the model on the training data
    model.fit(X_train, y_train)
    
    return model


def evaluate_model(
    model: RandomForestRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> EvaluationMetrics:
    """
    Evaluate the trained model on test data and calculate performance metrics.
    
    Parameters:
        model: Fitted RandomForestRegressor model
        X_test: DataFrame containing test features
        y_test: Series containing test target values (actual confidence scores)
    
    Returns:
        EvaluationMetrics dictionary containing mae, r2, skip_percentage, skip_accuracy
    """
    # Generate predictions for test set
    y_pred = model.predict(X_test)
    
    # Calculate Mean Absolute Error (average prediction error magnitude)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Calculate R² score (proportion of variance explained by model)
    # R² = 1.0 indicates perfect predictions, 0.0 indicates no predictive power
    r2 = r2_score(y_test, y_pred)
    
    # Calculate computational ROI metrics (skip percentage and accuracy)
    roi = calculate_roi(y_test, y_pred, threshold=DEFAULT_CONFIDENCE_THRESHOLD)
    
    # Return comprehensive evaluation metrics
    return EvaluationMetrics(
        mae=mae,
        r2=r2,
        skip_percentage=roi['skip_percentage'],
        skip_accuracy=roi['skip_accuracy']
    )


def plot_predicted_vs_actual(
    y_test: pd.Series,
    y_pred: np.ndarray,
    visuals_output_dir: str
) -> None:
    """
    Generate scatter plot comparing predicted and actual confidence values.
    
    Parameters:
        y_test: Series containing actual confidence scores
        y_pred: Array containing predicted confidence scores
    """
    # Create figure with specified dimensions
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot predictions vs actuals with transparency for overlapping points
    ax.scatter(y_test, y_pred, alpha=0.5)
    
    # Add diagonal reference line (y=x) showing perfect prediction
    # Points on this line indicate exact matches between predicted and actual
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    # Label axes with clear descriptions
    ax.set_xlabel('Actual Confidence')
    ax.set_ylabel('Predicted Confidence')
    
    # Add descriptive title
    ax.set_title('Predicted vs Actual Confidence Scores')
    ax.legend()
    
    # Save figure with high resolution
    predicted_vs_actual_path = os.path.join(visuals_output_dir, "predicted_vs_actual.png")
    plt.savefig(predicted_vs_actual_path, dpi=300, bbox_inches='tight')
    
    # Close figure to free memory resources
    plt.close(fig)


def plot_feature_importance(
    model: RandomForestRegressor,
    feature_names: list,
    visuals_output_dir: str
) -> None:
    """
    Generate bar chart showing feature importance scores.
    
    Parameters:
        model: Fitted RandomForestRegressor model
        feature_names: List of feature names corresponding to model features
    """
    # Extract feature importance scores from trained model
    importances = model.feature_importances_
    
    # Create DataFrame for easier sorting and visualization
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    
    # Sort by importance in ascending order for horizontal bar chart
    importance_df = importance_df.sort_values('importance', ascending=True)
    
    # Create horizontal bar chart for better readability of feature names
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(importance_df['feature'], importance_df['importance'])
    
    # Label axes clearly
    ax.set_xlabel('Importance')
    ax.set_ylabel('Features')
    
    # Add descriptive title
    ax.set_title('Feature Importance for Confidence Prediction')
    
    # Save figure with high resolution
    feature_importance_path = os.path.join(visuals_output_dir, "feature_importance.png")
    plt.savefig(feature_importance_path, dpi=300, bbox_inches='tight')
    
    # Close figure to free memory resources
    plt.close(fig)


def calculate_roi(
    y_test: pd.Series,
    y_pred: np.ndarray,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
) -> ROIAnalysis:
    """
    Calculate computational ROI metrics based on confidence threshold.
    
    Parameters:
        y_test: Series containing actual confidence scores
        y_pred: Array containing predicted confidence scores
        threshold: Confidence threshold for skip decisions (default: 0.85)
    
    Returns:
        ROIAnalysis dictionary with threshold, total_frames, skip_count, 
        skip_percentage, correct_skips, skip_accuracy
    """
    # Validate threshold is in valid confidence range
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(
            f"threshold must be in range [0.0, 1.0], got {threshold}"
        )
    
    # Count total number of test frames
    total_frames = len(y_test)
    
    # Identify frames where model predicts high confidence (skip candidates)
    skip_mask = y_pred > threshold
    
    # Count how many frames would be skipped based on predictions
    skip_count = int(np.sum(skip_mask))
    
    # Calculate percentage of frames that would be skipped
    # Higher percentage = more computational savings
    skip_percentage = (skip_count / total_frames) * 100 if total_frames > 0 else 0.0
    
    # Identify correct skip decisions (both predicted and actual > threshold)
    # These are frames where skipping would be safe and accurate
    correct_skip_mask = (y_pred > threshold) & (y_test.values > threshold)
    
    # Count correct skip decisions
    correct_skips = int(np.sum(correct_skip_mask))
    
    # Calculate accuracy of skip decisions
    # High accuracy means model reliably identifies safe-to-skip frames
    skip_accuracy = (correct_skips / skip_count) * 100 if skip_count > 0 else 0.0
    
    # Return comprehensive ROI analysis
    return ROIAnalysis(
        threshold=threshold,
        total_frames=total_frames,
        skip_count=skip_count,
        skip_percentage=skip_percentage,
        correct_skips=correct_skips,
        skip_accuracy=skip_accuracy
    )


def save_model(
    model: RandomForestRegressor,
    filepath: str = DEFAULT_MODEL_FILEPATH
) -> None:
    """
    Save the trained model to disk using joblib serialization.
    
    Parameters:
        model: Fitted RandomForestRegressor model to save
        filepath: Path where the model should be saved (default: "confidence_rf_model.pkl")
    """
    # Serialize and save model to disk
    # joblib is optimized for large numpy arrays (better than pickle for ML models)
    try:
        joblib.dump(model, filepath)
    except Exception as e:
        # Raise IOError with context if save operation fails
        raise IOError(f"Failed to save model to {filepath}: {str(e)}")


def main() -> None:
    """
    Main pipeline orchestration function that executes the complete ML training workflow.
    
    This function:
    1. Loads and splits training data
    2. Trains a RandomForestRegressor model
    3. Evaluates model performance
    4. Generates visualizations
    5. Saves the trained model
    """
    # Load training data and split into train/test sets
    data = load_and_split_data(
        "training_data.csv",
        test_size=DEFAULT_TEST_SIZE,
        random_state=DEFAULT_RANDOM_STATE
    )
    
    # Train RandomForest model on training data
    model = train_model(data['X_train'], data['y_train'])
    
    # Generate predictions on test set for evaluation
    y_pred = model.predict(data['X_test'])
    
    # Evaluate model performance and calculate metrics
    metrics = evaluate_model(model, data['X_test'], data['y_test'])
    
    # Print evaluation metrics to console
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"R2 Score: {metrics['r2']:.4f}")
    
    # Calculate and print computational ROI metrics
    roi = calculate_roi(
        data['y_test'],
        y_pred,
        threshold=DEFAULT_CONFIDENCE_THRESHOLD
    )
    print(f"Skip Percentage: {roi['skip_percentage']:.2f}%")
    print(f"Skip Accuracy: {roi['skip_accuracy']:.2f}%")

    visuals_output_dir = "outputs"
    os.makedirs(visuals_output_dir, exist_ok=True)
    
    # Generate visualization: predicted vs actual scatter plot
    plot_predicted_vs_actual(data['y_test'], y_pred, visuals_output_dir)
    
    # Generate visualization: feature importance bar chart
    plot_feature_importance(model, data['X_train'].columns.tolist(), visuals_output_dir)
    
    # Save trained model to disk for deployment
    save_model(model, DEFAULT_MODEL_FILEPATH)
    
    # Confirm successful model save
    print(f"Model saved as {DEFAULT_MODEL_FILEPATH}")


if __name__ == "__main__":
    main()
