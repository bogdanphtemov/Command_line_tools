"""
Application state for Multinomial Logistic Regression workflow.

Manages data, model, hyperparameters, and metrics throughout the session.
Mirrors AppState from app_state.py but adapted for multiclass classification.
"""

from dataclasses import dataclass
from typing import Optional, List
import numpy as np

from .data import Dataset, Prepareddata
from .core import MultinomialLogisticRegression
from .preprocessing import train_test_split, standardize_fit, standardize_apply
from ...base_models import universal_rebuild_split, universal_print_status


@dataclass
class MultinomialAppState:
    """
    Complete state for multinomial logistic regression session.
    
    Tracks:
        - Raw dataset and prepared features/target
        - Train/test split
        - Scaling parameters
        - Model hyperparameters
        - Trained model and evaluation metrics
        - Class names for display
    """
    
    # Data
    dataset: Optional[Dataset] = None
    prepareddata: Optional[Prepareddata] = None
    
    # Train/test split parameters
    test_size: float = 0.2
    seed: int = 42
    use_scaling: bool = True
    
    # Model hyperparameters
    learning_rate: float = 0.01
    epochs: int = 1000
    lambda_l2: float = 0.0
    
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
    
    # Trained model
    model: Optional[MultinomialLogisticRegression] = None
    
    # Evaluation metrics
    metrics: dict[str, float] = None
    
    # Class names for display (e.g., ["Setosa", "Versicolor", "Virginica"])
    class_names: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}
    
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
        return "OFF" if state.lambda_l2 == 0.0 else f"L2(λ={state.lambda_l2})"
    
    # Add class count to status
    extra = ""
    if s.model is not None and s.model.is_trained:
        extra = f" | classes={s.model.n_classes}"
    
    universal_print_status(s, f"Multinomial Logistic Regression{extra}", format_metrics, format_regularization)


def rebuild_split(s: MultinomialAppState) -> None:
    """
    Build train/test split from prepared data with optional scaling.
    
    Resets model and metrics since data pipeline changed.
    
    Args:
        s: MultinomialAppState to rebuild
    
    Raises:
        RuntimeError: If features/target not selected yet
    """
    universal_rebuild_split(s, train_test_split, standardize_fit, standardize_apply)
