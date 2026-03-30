from typing import List, Optional
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

from online_model.types import FrameDecision, PerformanceMetrics


class OnlinePerceptionEngine:
    def __init__(
        self,
        confidence_threshold: float = 0.85,
        audit_interval: int = 5,
        seed_batch_size: int = 50,
        reuse_latency_ms: float = 0.5,
        full_detection_latency_ms: float = 80.0,
        safety_threshold: float = 0.70
    ):
        self.confidence_threshold = confidence_threshold
        self.audit_interval = audit_interval
        self.seed_batch_size = seed_batch_size
        self.reuse_latency_ms = reuse_latency_ms
        self.full_detection_latency_ms = full_detection_latency_ms
        self.safety_threshold = safety_threshold
        self.model: Optional[SGDRegressor] = None
        self.scaler: StandardScaler = StandardScaler()
        self.decisions: List[FrameDecision] = []
        self.metrics: Optional[PerformanceMetrics] = None
    
    def initialize_model(self, X_seed: np.ndarray, y_seed: np.ndarray) -> None:
        """Initialize SGDRegressor with seed batch for warm-up."""

        # Validate X_seed shape is 2D
        if X_seed.ndim != 2:
            raise ValueError(f"X_seed must be 2D array, got {X_seed.ndim}D")
        
        # Validate y_seed shape is 1D
        if y_seed.ndim != 1:
            raise ValueError(f"y_seed must be 1D array, got {y_seed.ndim}D")
        
        # Validate lengths match
        if X_seed.shape[0] != y_seed.shape[0]:
            raise ValueError(
                f"X_seed and y_seed lengths must match, got {X_seed.shape[0]} and {y_seed.shape[0]}"
            )
        
        # Validate at least 1 sample
        if X_seed.shape[0] < 1:
            raise ValueError(f"Seed batch must contain at least 1 sample, got {X_seed.shape[0]}")
        
        # Validate no NaN values in X_seed
        if np.isnan(X_seed).any():
            raise ValueError("X_seed contains NaN values")
        
        # Validate no infinite values in X_seed
        if np.isinf(X_seed).any():
            raise ValueError("X_seed contains infinite values")
        
        # Validate no NaN values in y_seed
        if np.isnan(y_seed).any():
            raise ValueError("y_seed contains NaN values")
        
        # Validate no infinite values in y_seed
        if np.isinf(y_seed).any():
            raise ValueError("y_seed contains infinite values")
        
        # Fit the scaler on seed batch for online scaling
        self.scaler.partial_fit(X_seed)
        
        # Transform the seed batch using the scaler
        X_seed_scaled = self.scaler.transform(X_seed)
        
        # Create SGDRegressor instance with stabilized parameters
        self.model = SGDRegressor(
            learning_rate='constant',
            eta0=0.05,
            # penalty='l2'
            # average=True,
            penalty=None,
            average=False
        )
        
        # Warm up the model with scaled seed batch
        self.model.partial_fit(X_seed_scaled, y_seed)


    def make_decision(self, predicted_confidence: float) -> str:
        """Determine whether to REUSE or run FULL_DETECTION based on predicted confidence. """

        if predicted_confidence > self.confidence_threshold:
            return "REUSE"
        else:
            return "FULL_DETECTION"

    def audit_and_update(
        self, 
        X_current: np.ndarray, 
        y_actual: float, 
        predicted_confidence: float
    ) -> float:
        """
        Calculate prediction error and update model weights incrementally.
        """

        # Calculate prediction error
        prediction_error = abs(y_actual - predicted_confidence)
        
        # Reshape X_current to 2D array (1, n_features)
        X_reshaped = X_current.reshape(1, -1)
        
        # Update scaler with new data for incremental scaling
        self.scaler.partial_fit(X_reshaped)
        
        # Transform X_current using the scaler
        X_scaled = self.scaler.transform(X_reshaped)
        
        # Update model weights using partial_fit with scaled data
        self.model.partial_fit(X_scaled, [y_actual])
        
        return prediction_error

    def process_frame(
        self, 
        frame_idx: int, 
        X_current: np.ndarray, 
        y_actual: float
    ) -> FrameDecision:
        """
        Process a single streaming frame: predict, decide, optionally audit and update.
        """
        # Validate model is initialized
        if self.model is None:
            raise ValueError("Model is not initialized. Call initialize_model() first.")
        
        # Validate X_current dimensionality matches expected feature count
        if X_current.ndim != 1:
            raise ValueError(f"X_current must be 1D array, got {X_current.ndim}D")
        
        # Validate y_actual is in range [0.0, 1.0]
        if not (0.0 <= y_actual <= 1.0):
            raise ValueError(f"y_actual must be in range [0.0, 1.0], got {y_actual}")
        
        # Reshape X_current to 2D and transform using scaler
        X_reshaped = X_current.reshape(1, -1)
        X_scaled = self.scaler.transform(X_reshaped)
        
        # Predict using scaled features
        predicted_confidence = self.model.predict(X_scaled)[0]
        
        # Clip predicted_confidence to range [0.0, 1.0] after prediction but before decision
        predicted_confidence = np.clip(predicted_confidence, 0.0, 1.0)
        
        # Call make_decision() to get decision type
        decision_type = self.make_decision(predicted_confidence)
        
        # Determine if frame is audit frame
        is_audit = (frame_idx % self.audit_interval == 0)
        
        # If is_audit, call audit_and_update() to get prediction_error
        if is_audit:
            prediction_error = self.audit_and_update(X_current, y_actual, predicted_confidence)
        else:
            prediction_error = 0.0
        
        # Create FrameDecision object with all fields
        decision = FrameDecision(
            frame_idx=frame_idx,
            predicted_confidence=predicted_confidence,
            actual_confidence=y_actual,
            decision=decision_type,
            prediction_error=prediction_error,
            is_audit_frame=is_audit
        )
        
        # Append decision to self.decisions list
        self.decisions.append(decision)
        
        # Return FrameDecision object
        return decision

    def calculate_metrics(self) -> PerformanceMetrics:
        """
        Aggregate performance metrics from all processed frames.
        """
        # Initialize counters
        total_frames = len(self.decisions)
        reuse_count = 0
        full_detection_count = 0
        safety_violations = 0
        
        # Initialize lists
        mae_list = []
        audit_frames = []
        
        # Iterate through self.decisions list
        for decision in self.decisions:
            # Count decision type
            if decision.decision == "REUSE":
                reuse_count += 1
                
                # Check for safety violations (REUSE with actual_confidence < safety_threshold)
                if decision.actual_confidence < self.safety_threshold:
                    safety_violations += 1
            else:
                full_detection_count += 1
            
            # Collect audit frame errors
            if decision.is_audit_frame:
                mae_list.append(decision.prediction_error)
                audit_frames.append(decision.frame_idx)
        
        # Calculate cumulative_latency_saved_ms
        cumulative_latency_saved_ms = reuse_count * (
            self.full_detection_latency_ms - self.reuse_latency_ms
        )
        
        # Create and return PerformanceMetrics object
        return PerformanceMetrics(
            total_frames=total_frames,
            reuse_count=reuse_count,
            full_detection_count=full_detection_count,
            cumulative_latency_saved_ms=cumulative_latency_saved_ms,
            mean_absolute_errors=mae_list,
            safety_violations=safety_violations,
            audit_frames=audit_frames
        )

