"""
Application state for Binary SVM workflow (Classification + Regression).

Manages data, model, hyperparameters, and metrics throughout the session.
Follows the same pattern as logistic_regression/app_state.py.

Supports both:
    - Binary SVM Classification (LinearSVM, KernelSVM)
    - Linear SVM Regression (LinearSVR, KernelSVR)
"""

from dataclasses import dataclass, field
from typing import Optional, List, Union
import numpy as np

from .data import Dataset, Prepareddata
from .core import LinearSVM, KernelSVM, LinearSVR, KernelSVR
from .preprocessing import train_test_split, standardize_fit, standardize_apply
from myclt.ML.base_models import universal_rebuild_split, universal_print_status


@dataclass
class AppState:
    """
    Complete state for binary SVM session.

    Tracks:
        - Raw dataset and prepared features/target
        - Train/test split
        - Scaling parameters
        - Model hyperparameters (C, kernel, gamma, degree, epsilon, etc.)
        - Trained model and evaluation metrics
    """

    # Data
    dataset: Optional[Dataset] = None
    prepareddata: Optional[Prepareddata] = None

    # Train/test split parameters
    test_size: float = 0.2
    seed: int = 42
    use_scaling: bool = True

    # === Model hyperparameters ===
    
    # Common
    # === Model hyperparameters ===

    # Common
    model_type: str = "linear_svm"  # Which algorithm to use
    C: float = 1.0                  # Regularization (inverse of λ), must be > 0
    learning_rate: float = 0.001
    epochs: int = 1000

    # LinearSVM specific
    batch_size: int = 0  # 0 = full batch

    # KernelSVM specific
    kernel: str = "rbf"  # 'linear', 'rbf', 'poly', 'sigmoid'
    gamma: float = 1.0   # Kernel coefficient (for rbf, poly, sigmoid), must be > 0
    degree: int = 3       # Polynomial degree (only for poly kernel)
    coef0: float = 1.0    # Independent term in poly/sigmoid kernel (only for poly/sigmoid)

    # SVR specific
    epsilon: float = 0.1  # ε-insensitive tube width (for regression), must be >= 0

    # Split data (after features/target chosen)
    X_train: Optional[np.ndarray] = None
    X_test: Optional[np.ndarray] = None
    y_train: Optional[np.ndarray] = None
    y_test: Optional[np.ndarray] = None

    # Indices of original prepared X used to build splits
    train_idx: Optional[np.ndarray] = None
    test_idx: Optional[np.ndarray] = None

    # Scaler parameters (fitted on training data)
    scaler_mean: Optional[np.ndarray] = None
    scaled_std: Optional[np.ndarray] = None

    # Trained model (one of LinearSVM, KernelSVM, LinearSVR, KernelSVR, OneVsRestSVM)
    model: Optional[Union[LinearSVM, KernelSVM, LinearSVR, KernelSVR]] = None

    # Evaluation metrics
    metrics: dict = field(default_factory=dict)

    # Mode: 'classifier' or 'regressor'
    mode: str = "classifier"

    # Valid model types for validation
    VALID_MODEL_TYPES = {"linear_svm", "kernel_svm", "linear_svr", "kernel_svr"}

    def __post_init__(self):
        # Validate model_type
        if self.model_type not in self.VALID_MODEL_TYPES:
            raise ValueError(
                f"Invalid model_type '{self.model_type}'. "
                f"Must be one of: {sorted(self.VALID_MODEL_TYPES)}"
            )

        # Validate hyperparameters
        if self.C <= 0:
            raise ValueError(f"C must be > 0, got {self.C}")
        if self.gamma <= 0:
            raise ValueError(f"gamma must be > 0, got {self.gamma}")
        if self.epsilon < 0:
            raise ValueError(f"epsilon must be >= 0, got {self.epsilon}")
        if not 0 < self.test_size < 1:
            raise ValueError(f"test_size must be between 0 and 1, got {self.test_size}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")

    def reset_metrics(self) -> None:
        """Clear all evaluation metrics."""
        self.metrics.clear()


def print_status(s: AppState) -> None:
    """
    Display comprehensive status of the current session.

    Args:
        s: Current AppState
    """
    def format_metrics(state):
        if not state.metrics:
            return "none"
        return " | ".join(
            f"{k}={v:.4f}"
            for k, v in state.metrics.items()
        )

    def format_regularization(state):
        kernel_params = []

        if state.model_type.startswith("kernel_"):
            kernel_params.append(f"kernel={state.kernel}")
            if state.kernel in ("rbf", "poly", "sigmoid"):
                kernel_params.append(f"γ={state.gamma}")
            if state.kernel == "poly":
                kernel_params.append(f"degree={state.degree}")
                kernel_params.append(f"coef0={state.coef0}")
            elif state.kernel == "sigmoid":
                kernel_params.append(f"coef0={state.coef0}")

        if state.model_type in ("linear_svr", "kernel_svr"):
            return ", ".join([f"C={state.C}", *kernel_params, f"ε={state.epsilon}"])
        else:
            return ", ".join([f"C={state.C}", *(kernel_params or ["linear"])])

    mode_str = "SVM Classification" if s.mode == "classifier" else "SVR Regression"
    extra = ""
    if s.model is not None and s.model.is_trained:
        if hasattr(s.model, 'n_support_vectors'):
            extra = f" | SV={s.model.n_support_vectors}"
        if hasattr(s.model, 'n_classes') and s.model.n_classes > 2:
            extra += f" | classes={s.model.n_classes}"

    universal_print_status(s, f"{mode_str}{extra}", format_metrics, format_regularization)


def rebuild_split(s: AppState) -> None:
    """
    Build train/test split from prepared data with optional scaling.
    Resets model and metrics since data pipeline changed.

    Args:
        s: AppState to rebuild

    Raises:
        RuntimeError: If features/target not selected yet
    """
    universal_rebuild_split(s, train_test_split, standardize_fit, standardize_apply)
