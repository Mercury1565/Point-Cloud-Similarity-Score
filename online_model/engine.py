from typing import List, Optional, Tuple
import numpy as np
from sklearn.preprocessing import StandardScaler

from online_model.types import FrameDecision, PerformanceMetrics


class BayesianLinearRegression:
    """
    Online Bayesian Linear Regression with conjugate Normal-Normal updates.

    Model:   y = w^T phi(x) + eps,   eps ~ N(0, beta^{-1})
    Prior:   w ~ N(0, alpha^{-1} I)
    phi(x) = [x; 1]  (feature vector augmented with bias term)

    Sequential updates use the Woodbury rank-1 identity, giving O(d^2) cost
    per observation instead of O(d^3) matrix inversion.

    Predictive distribution for a new input x*:
        p(y* | x*, data) = N(phi*^T m_n,  beta^{-1} + phi*^T S_n phi*)

    The predictive std captures both aleatoric noise (beta^{-1}) and
    epistemic uncertainty (phi*^T S_n phi*), shrinking as more data arrives.
    """

    def __init__(self, n_features: int, alpha: float = 1.0, beta: float = 25.0):
        """
        Args:
            n_features: Dimensionality of input x (before bias augmentation).
            alpha:      Prior precision — higher values pull weights toward 0
                        (stronger regularisation).
            beta:       Noise precision = 1 / noise_variance.  beta=25 implies
                        noise std ≈ 0.20, reasonable for confidence scores in [0,1].
        """
        self.d = n_features + 1  # augmented dimension (features + bias)
        self.alpha = alpha
        self.beta = beta

        # Posterior parameters (updated in-place)
        self.m = np.zeros(self.d)                   # posterior mean
        self.S = (1.0 / alpha) * np.eye(self.d)     # posterior covariance

    def _phi(self, x: np.ndarray) -> np.ndarray:
        """Augment a flat feature vector with a bias term."""
        return np.append(x, 1.0)

    def update(self, x: np.ndarray, y: float) -> None:
        """
        Rank-1 Bayesian update for a single observation (x, y).

        Woodbury covariance update:
            S_new = S - (beta * S phi phi^T S) / (1 + beta * phi^T S phi)

        Equivalent mean update (Kalman form):
            m_new = m + K * (y - phi^T m)
            K     = beta * S phi / (1 + beta * phi^T S phi)
        """
        phi = self._phi(x)                               # (d,)
        S_phi = self.S @ phi                             # (d,)
        denom = 1.0 + self.beta * (phi @ S_phi)          # scalar
        K = (self.beta / denom) * S_phi                  # (d,) — Kalman gain

        self.m = self.m + K * (y - phi @ self.m)
        self.S = self.S - np.outer(K, S_phi)

    def batch_update(self, X: np.ndarray, y: np.ndarray) -> None:
        """Sequential update over a batch — used for seed initialisation."""
        for xi, yi in zip(X, y):
            self.update(xi, float(yi))

    def predict(self, x: np.ndarray) -> Tuple[float, float]:
        """
        Return the predictive mean and std for input x.

        Returns:
            mean: E[y* | x*]   = phi*^T m_n
            std:  sqrt(Var[y*]) = sqrt(beta^{-1} + phi*^T S_n phi*)
        """
        phi = self._phi(x)
        mean = float(phi @ self.m)
        var = (1.0 / self.beta) + float(phi @ self.S @ phi)
        std = float(np.sqrt(max(var, 0.0)))
        return mean, std


class OnlinePerceptionEngine:
    def __init__(
        self,
        confidence_threshold: float = 0.85,
        audit_interval: int = 5,
        seed_batch_size: int = 50,
        reuse_latency_ms: float = 0.5,
        full_detection_latency_ms: float = 80.0,
        safety_threshold: float = 0.70,
        alpha: float = 1.0,
        beta: float = 25.0,
        uncertainty_weight: float = 1.0,
    ):
        """
        Args:
            confidence_threshold: Minimum lower-confidence-bound to choose REUSE.
            audit_interval:       Every N-th frame forces FULL_DETECTION for a
                                  ground-truth label used to update the posterior.
            seed_batch_size:      Number of labelled frames used for cold-start.
            reuse_latency_ms:     Latency cost of a REUSE decision.
            full_detection_latency_ms: Latency cost of a FULL_DETECTION decision.
            safety_threshold:     Actual confidence below which a REUSE decision
                                  counts as a safety violation.
            alpha:                Prior precision for BayesianLinearRegression.
            beta:                 Noise precision for BayesianLinearRegression.
            uncertainty_weight:   k in decision rule: REUSE iff (mean - k*std) >
                                  confidence_threshold.  Higher k is more
                                  conservative (fewer REUSE decisions).
        """
        self.confidence_threshold = confidence_threshold
        self.audit_interval = audit_interval
        self.seed_batch_size = seed_batch_size
        self.reuse_latency_ms = reuse_latency_ms
        self.full_detection_latency_ms = full_detection_latency_ms
        self.safety_threshold = safety_threshold
        self.alpha = alpha
        self.beta = beta
        self.uncertainty_weight = uncertainty_weight

        self.model: Optional[BayesianLinearRegression] = None
        # Scaler is fit once on the seed batch and then frozen.  Updating it
        # mid-stream would shift the feature space and invalidate the prior.
        self.scaler: StandardScaler = StandardScaler()
        self.decisions: List[FrameDecision] = []
        self.metrics: Optional[PerformanceMetrics] = None

    def initialize_model(self, X_seed: np.ndarray, y_seed: np.ndarray) -> None:
        """Fit the scaler and initialise the Bayesian posterior from the seed batch."""

        if X_seed.ndim != 2:
            raise ValueError(f"X_seed must be 2D array, got {X_seed.ndim}D")
        if y_seed.ndim != 1:
            raise ValueError(f"y_seed must be 1D array, got {y_seed.ndim}D")
        if X_seed.shape[0] != y_seed.shape[0]:
            raise ValueError(
                f"X_seed and y_seed lengths must match, "
                f"got {X_seed.shape[0]} and {y_seed.shape[0]}"
            )
        if X_seed.shape[0] < 1:
            raise ValueError(
                f"Seed batch must contain at least 1 sample, got {X_seed.shape[0]}"
            )
        if np.isnan(X_seed).any():
            raise ValueError("X_seed contains NaN values")
        if np.isinf(X_seed).any():
            raise ValueError("X_seed contains infinite values")
        if np.isnan(y_seed).any():
            raise ValueError("y_seed contains NaN values")
        if np.isinf(y_seed).any():
            raise ValueError("y_seed contains infinite values")

        # Fit and freeze the scaler on the seed distribution.
        self.scaler.fit(X_seed)
        X_seed_scaled = self.scaler.transform(X_seed)

        # Build the Bayesian model and condition it on the seed batch.
        n_features = X_seed.shape[1]
        self.model = BayesianLinearRegression(
            n_features=n_features,
            alpha=self.alpha,
            beta=self.beta,
        )
        self.model.batch_update(X_seed_scaled, y_seed)

    def make_decision(self, predicted_mean: float, predicted_std: float) -> str:
        """
        REUSE iff the lower confidence bound exceeds the threshold.

            lower_bound = mean - uncertainty_weight * std

        This makes the decision conservative when the model is uncertain:
        a high-variance prediction cannot trigger REUSE even if its mean
        is above the threshold.
        """
        lower_bound = predicted_mean - self.uncertainty_weight * predicted_std
        if lower_bound > self.confidence_threshold:
            return "REUSE"
        return "FULL_DETECTION"

    def audit_and_update(
        self,
        X_current: np.ndarray,
        y_actual: float,
        predicted_confidence: float,
    ) -> float:
        """
        Compute prediction error and perform a rank-1 Bayesian posterior update.
        The scaler is intentionally NOT updated here — see __init__ docstring.
        """
        prediction_error = abs(y_actual - predicted_confidence)

        X_scaled = self.scaler.transform(X_current.reshape(1, -1))[0]
        self.model.update(X_scaled, y_actual)

        return prediction_error

    def process_frame(
        self,
        frame_idx: int,
        X_current: np.ndarray,
        y_actual: float,
        scene_id: str = "",
    ) -> FrameDecision:
        """
        Process a single streaming frame: predict (with uncertainty), decide,
        and — on audit frames — update the posterior with the ground-truth label.
        """
        if self.model is None:
            raise ValueError("Model is not initialized. Call initialize_model() first.")
        if X_current.ndim != 1:
            raise ValueError(f"X_current must be 1D array, got {X_current.ndim}D")
        if not (0.0 <= y_actual <= 1.0):
            raise ValueError(f"y_actual must be in range [0.0, 1.0], got {y_actual}")

        X_scaled = self.scaler.transform(X_current.reshape(1, -1))[0]

        predicted_mean, predicted_std = self.model.predict(X_scaled)
        predicted_confidence = float(np.clip(predicted_mean, 0.0, 1.0))

        decision_type = self.make_decision(predicted_mean, predicted_std)

        is_audit = (frame_idx % self.audit_interval == 0)
        if is_audit:
            prediction_error = self.audit_and_update(X_current, y_actual, predicted_confidence)
        else:
            prediction_error = 0.0

        decision = FrameDecision(
            frame_idx=frame_idx,
            predicted_confidence=predicted_confidence,
            predicted_std=predicted_std,
            actual_confidence=y_actual,
            decision=decision_type,
            prediction_error=prediction_error,
            is_audit_frame=is_audit,
            scene_id=scene_id,
        )
        self.decisions.append(decision)
        return decision

    def save_state(self, path: str) -> None:
        """
        Persist the Bayesian posterior and frozen scaler statistics to disk.

        The saved file can be reloaded with load_state() to resume online
        learning without restarting from the seed batch.

        Args:
            path: File path (without extension — numpy adds .npz automatically).
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call initialize_model() first.")

        np.savez(
            path,
            # Bayesian posterior
            m=self.model.m,
            S=self.model.S,
            alpha=np.array(self.model.alpha),
            beta=np.array(self.model.beta),
            d=np.array(self.model.d),
            # Frozen scaler
            scaler_mean=self.scaler.mean_,
            scaler_scale=self.scaler.scale_,
            scaler_var=self.scaler.var_,
            scaler_n_features=np.array(self.scaler.n_features_in_),
            # Engine hyperparameters
            confidence_threshold=np.array(self.confidence_threshold),
            safety_threshold=np.array(self.safety_threshold),
            uncertainty_weight=np.array(self.uncertainty_weight),
        )

    def load_state(self, path: str) -> None:
        """
        Restore a previously saved Bayesian posterior and scaler from disk.

        After loading, the engine is ready to process new frames without
        re-running initialize_model().

        Args:
            path: File path written by save_state() (with or without .npz extension).
        """
        npz_path = path if path.endswith(".npz") else path + ".npz"
        data = np.load(npz_path)

        n_features = int(data["d"]) - 1  # d includes the bias term

        self.model = BayesianLinearRegression(
            n_features=n_features,
            alpha=float(data["alpha"]),
            beta=float(data["beta"]),
        )
        self.model.m = data["m"].copy()
        self.model.S = data["S"].copy()

        # Restore scaler in a fitted state by setting its learned attributes directly.
        self.scaler.mean_ = data["scaler_mean"].copy()
        self.scaler.scale_ = data["scaler_scale"].copy()
        self.scaler.var_ = data["scaler_var"].copy()
        self.scaler.n_features_in_ = int(data["scaler_n_features"])
        self.scaler.n_samples_seen_ = np.array(1)

    def calculate_metrics(self) -> PerformanceMetrics:
        """Aggregate performance metrics across all processed frames."""
        total_frames = len(self.decisions)
        reuse_count = 0
        full_detection_count = 0
        safety_violations = 0
        mae_list: List[float] = []
        audit_frames: List[int] = []

        for decision in self.decisions:
            if decision.decision == "REUSE":
                reuse_count += 1
                if decision.actual_confidence < self.safety_threshold:
                    safety_violations += 1
            else:
                full_detection_count += 1

            if decision.is_audit_frame:
                mae_list.append(decision.prediction_error)
                audit_frames.append(decision.frame_idx)

        cumulative_latency_saved_ms = reuse_count * (
            self.full_detection_latency_ms - self.reuse_latency_ms
        )

        return PerformanceMetrics(
            total_frames=total_frames,
            reuse_count=reuse_count,
            full_detection_count=full_detection_count,
            cumulative_latency_saved_ms=cumulative_latency_saved_ms,
            mean_absolute_errors=mae_list,
            safety_violations=safety_violations,
            audit_frames=audit_frames,
        )
