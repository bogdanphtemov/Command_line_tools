"""
Application state management for KNN CLI/UI.

Manages the complete state of KNN algorithm execution including:
- Dataset configuration and preprocessing settings
- Model parameters (k, distance metric, weighted averaging)
- Training data splits and scaling
- Trained model and evaluation metrics

Example:
    >>> state = AppState(model_type='regression', k=7, weighted=True)
    >>> state.k
    7
    
    >>> state.reset_model()  # Clear trained model while keeping parameters
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from .data import Dataset, Prepareddata
from .classifier import KNNClassifier
from .regressor import KNNRegressor
from .preprocessing import train_test_split, standardize_fit, standardize_apply
from ...base_models import universal_rebuild_split, universal_print_status
from .distance_metrics import SUPPORTED_METRICS


@dataclass
class AppState:
    """
    Application state for KNN algorithm.
    
    Note:
        Fields are mutable for flexible CLI/UI usage. After direct field mutation,
        call validate() to ensure consistency. Weighted averaging is currently
        only supported for regression (KNNClassifier doesn't support it yet).
    """
    
    dataset: Optional[Dataset] = None
    prepareddata: Optional[Prepareddata] = None
    test_size: float = 0.2
    seed: int = 42
    use_scaling: bool = True
    
    # KNN specific parameters
    k: int = 5
    distance_metric: str = 'euclidean'
    model_type: str = 'classification'  # 'classification' or 'regression'
    weighted: bool = False  # For regression only: use weighted averaging
    
    # split data (after features/target chosen)
    X_train: Optional[np.ndarray] = None
    X_test: Optional[np.ndarray] = None
    y_train: Optional[np.ndarray] = None
    y_test: Optional[np.ndarray] = None
    # indices of original prepared X used to build splits (filled by rebuild_split)
    train_idx: Optional[np.ndarray] = None
    test_idx: Optional[np.ndarray] = None

    # scaler params (train-fitted)
    scaler_mean: Optional[np.ndarray] = None
    scaler_std: Optional[np.ndarray] = None

    model: KNNClassifier | KNNRegressor | None = None
    
    # evaluation metrics
    metrics: dict[str, float] = field(default_factory=dict)
    
    def validate(self) -> None:
        """
        Validate all parameters. 
        
        Call after direct field mutation to ensure state consistency.
        
        Raises:
            ValueError: If any parameter constraint is violated
        """
        # Validate model_type
        if self.model_type not in ('classification', 'regression'):
            raise ValueError(
                f"Invalid model_type: {self.model_type}. "
                f"Must be 'classification' or 'regression'."
            )
        
        # Validate k parameter
        if not isinstance(self.k, int) or self.k < 1:
            raise ValueError(f"k must be a positive integer, got {self.k}")
        
        # Validate test_size parameter
        if not (0.0 < self.test_size < 1.0):
            raise ValueError(f"test_size must be in (0, 1), got {self.test_size}")
        
        # Validate seed parameter
        if not isinstance(self.seed, (int, np.integer)) or self.seed < 0:
            raise ValueError(f"seed must be a non-negative integer, got {self.seed}")
        
        # Validate use_scaling parameter (must be boolean)
        if not isinstance(self.use_scaling, bool):
            raise ValueError(f"use_scaling parameter must be boolean, got {type(self.use_scaling)}")
        
        # Validate distance_metric (use SUPPORTED_METRICS as single source of truth)
        if self.distance_metric not in SUPPORTED_METRICS:
            raise ValueError(
                f"Invalid distance_metric: {self.distance_metric}. "
                f"Must be one of: {', '.join(sorted(SUPPORTED_METRICS))}."
            )
        
        # Validate weighted parameter (must be boolean)
        if not isinstance(self.weighted, bool):
            raise ValueError(f"weighted parameter must be boolean, got {type(self.weighted)}")
        
        # Validate weighted is only used with regression (temporary limitation)
        if self.weighted and self.model_type != 'regression':
            raise ValueError(
                "weighted=True is currently only supported for regression models. "
                "KNNClassifier does not yet support weighted voting."
            )
    
    def __post_init__(self):
        """Validate parameters after initialization."""
        self.validate()
    
    def reset_metrics(self) -> None:
        """Clear all evaluation metrics."""
        self.metrics.clear()
    
    def reset_model(self) -> None:
        """
        Reset trained model and splits while keeping parameters.
        
        Note: Also clears scaler_mean/scaler_std. Use reset_fit() if you only need
        to clear the model and metrics while preserving the current scaling.
        """
        self.model = None
        self.X_train = self.X_test = None
        self.y_train = self.y_test = None
        self.train_idx = self.test_idx = None
        self.scaler_mean = self.scaler_std = None
        self.metrics.clear()
    
    def reset_fit(self) -> None:
        """
        Reset only model and metrics, preserving splits and scaling.
        
        Use this when changing hyperparameters (k, distance_metric) to avoid
        recomputing the data split and scaling.
        """
        self.model = None
        self.metrics.clear()


def print_status(s: AppState) -> None:
    """
    Display comprehensive status of the current session.
    
    Shows what data is loaded, features selected, model state, and metrics.
    Uses universal printer from base_models.
    """
    def format_metrics(state):
        if not state.metrics:
            return "none"
        
        def _fmt(v):
            """Safe formatting for metric values."""
            if v is None:
                return "n/a"
            if isinstance(v, bool):
                return str(v)
            if isinstance(v, (int, float, np.floating)):
                return f"{v:.4f}"
            return str(v)
        
        return " | ".join(
            f"{k}={_fmt(v)}"
            for k, v in state.metrics.items()
        )
    
    def format_model_info(state):
        # Format KNN model parameters
        parts = [f"KNN-{state.model_type.title()}", f"k={state.k}", f"metric={state.distance_metric}"]
        if state.model_type == 'regression':
            parts.append("weighted" if state.weighted else "unweighted")
        return " (" + ", ".join(parts) + ")"
    
    universal_print_status(s, f"KNN {s.model_type.title()}", format_metrics, format_model_info)


def rebuild_split(s: AppState) -> None:
    """
    Build train/test split from supervised data, apply scaling if enabled.
    Uses universal split builder from base_models.
    
    Args:
        s: AppState instance with prepared data and split parameters
        
    Raises:
        RuntimeError: If prepareddata is None (features/target not selected)
        
    Sets:
        X_train, X_test, y_train, y_test: Split data arrays
        train_idx, test_idx: Original indices for tracking
        scaler_mean, scaler_std: Standardization parameters if scaling enabled
        model: Reset to None (data pipeline changed)
        metrics: Cleared via reset_metrics()
        
    Note:
        Calls universal_rebuild_split() which validates prepareddata is not None
        and raises RuntimeError with message "!No features/target selection yet!"
    """
    universal_rebuild_split(s, train_test_split, standardize_fit, standardize_apply)