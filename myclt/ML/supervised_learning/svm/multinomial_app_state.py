"""
Application state for Multiclass SVM workflow (One-vs-Rest).

Manages data, model, hyperparameters, and metrics throughout the session.
Follows the same pattern as logistic_regression/multinomial_app_state.py.

Supports:
    - Multiclass SVM Classification (OneVsRestSVM with LinearSVM or KernelSVM)
"""

from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np

from .data import Dataset, Prepareddata
from .core import LinearSVM, KernelSVM, OneVsRestSVM
from .preprocessing import train_test_split, standardize_fit, standardize_apply
from myclt.ML.base_models import universal_rebuild_split, universal_print_status


@dataclass
class MultinomialAppState:
    """
    Complete state for multiclass SVM session.

    Tracks:
        - Raw dataset and prepared features/target
        - Train/test split
        - Scaling parameters
        - Model hyperparameters (base estimator type, C, kernel, gamma, etc.)
        - Trained OneVsRestSVM model and evaluation metrics
        - Class names for display
    """

    # Data
    dataset: Optional[Dataset] = None
    prepareddata: Optional[Prepareddata] = None

    # Train/test split parameters
    test_size: float = 0.2
    seed: int = 42
    use_scaling: bool = True

    # === Model hyperparameters ===
    base_estimator_type: str = "linear"  # 'linear' or 'kernel'
    C: float = 1.0
    learning_rate: float = 0.001
    epochs: int = 1000
    batch_size: int = 0  # 0 = full batch

    # Kernel parameters (if base_estimator_type == 'kernel')
    kernel: str = "rbf"  # 'linear', 'rbf', 'poly', 'sigmoid'
    gamma: float = 1.0
    degree: int = 3
    coef0: float = 1.0

    # Split data
    X_train: Optional[np.ndarray] = None
    X_test: Optional[np.ndarray] = None
    y_train: Optional[np.ndarray] = None
    y_test: Optional[np.ndarray] = None

    train_idx: Optional[np.ndarray] = None
    test_idx: Optional[np.ndarray] = None

    # Scaler parameters
    scaler_mean: Optional[np.ndarray] = None
    scaler_std: Optional[np.ndarray] = None

    # Trained model
    model: Optional[OneVsRestSVM] = None

    # Evaluation metrics
    metrics: dict = field(default_factory=dict)

    # Class names for display (e.g., ["Setosa", "Versicolor", "Virginica"])
    class_names: Optional[List[str]] = None

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}

        # Validate model type
        if self.base_estimator_type not in ('linear', 'kernel'):
            raise ValueError(
                f"base_estimator_type must be 'linear' or 'kernel', "
                f"got '{self.base_estimator_type}'"
            )

        # Validate hyperparameters
        if self.C <= 0:
            raise ValueError(f"C must be > 0, got {self.C}")

        if self.gamma <= 0:
            raise ValueError(f"gamma must be > 0, got {self.gamma}")

        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")

        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")

        if self.batch_size < 0:
            raise ValueError(f"batch_size must be >= 0, got {self.batch_size}")

        if not 0 < self.test_size < 1:
            raise ValueError(
                f"test_size must be between 0 and 1, "
                f"got {self.test_size}"
            )

    def reset_metrics(self) -> None:
        """Clear all evaluation metrics."""
        self.metrics.clear()


def print_status(s: MultinomialAppState) -> None:
    """
    Display comprehensive status of the current session.

    Args:
        s: Current MultinomialAppState
    """
    def format_metrics(state):
        if not state.metrics:
            return "none"
        return " | ".join(
            f"{k}={v:.4f}"
            for k, v in state.metrics.items()
        )

    def format_regularization(state):
        parts = [f"C={state.C}"]
        if state.base_estimator_type == "kernel":
            parts.append(f"kernel={state.kernel}")
            if state.kernel == "rbf":
                parts.append(f"γ={state.gamma}")
        else:
            parts.append("linear")
        return ", ".join(parts)

    extra = ""
    if s.model is not None and s.model.is_trained:
        extra = f" | classes={s.model.n_classes}"
        if s.model.estimators:
            total_sv = sum(
                getattr(est, 'n_support_vectors', 0)
                for est in s.model.estimators
            )
            extra += f" | SV={total_sv}"

    universal_print_status(s, f"Multiclass SVM{extra}", format_metrics, format_regularization)


def rebuild_split(s: MultinomialAppState) -> None:
    """
    Build train/test split from prepared data with optional scaling.

    Args:
        s: MultinomialAppState to rebuild

    Raises:
        RuntimeError: If features/target not selected yet
    """
    universal_rebuild_split(s, train_test_split, standardize_fit, standardize_apply)
