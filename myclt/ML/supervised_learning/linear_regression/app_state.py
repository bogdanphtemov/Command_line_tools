from dataclasses import dataclass
from typing import Optional 
import numpy as np

from .data import Dataset , Prepareddata
from .core import LinearRegressionGD
from .preprocessing import train_test_split , standardize_fit , standardize_apply
from ...base_models import universal_rebuild_split, universal_print_status

@dataclass
class AppState:
    dataset: Optional[Dataset] = None
    prepareddata: Optional[Prepareddata] = None
    test_size: float = 0.2
    seed: int = 42
    use_scaling: bool = True
    learning_rate: float = 0.05
    epochs: int = 2000
    
    # Regularization parameters
    use_l1: bool = False
    use_l2: bool = False
    lambda_l1: float = 0.01
    lambda_l2: float = 0.01
    
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
    scaled_std: Optional[np.ndarray] = None

    model: Optional[LinearRegressionGD] = None
    
    # evaluation metrics
    metrics: dict[str, float] = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}
    
    def reset_metrics(self) -> None:
        """Clear all evaluation metrics."""
        self.metrics.clear()

# this function determines which values ​​we have already defined and displays this information to the user
def print_status(s: AppState) -> None:
    """
    Display comprehensive status of the current session.
    
    Shows what data is loaded, features selected, model state, and metrics.
    Uses universal printer from base_models.
    """
    def format_metrics(state):
        if not state.metrics:
            return "none"
        return " | ".join(
            f"{k}={v:.4f}"
            for k, v in state.metrics.items()
        )
    
    def format_regularization(state):
        if not state.use_l1 and not state.use_l2:
            return "OFF"
        parts = []
        if state.use_l1:
            parts.append(f"L1(λ={state.lambda_l1})")
        if state.use_l2:
            parts.append(f"L2(λ={state.lambda_l2})")
        return " + ".join(parts)
    
    universal_print_status(s, "Linear Regression", format_metrics, format_regularization)

def rebuild_split(s: AppState) -> None:
    """
    Build train/test split from supervised data, apply scaling if enabled.
    Uses universal split builder from base_models.
    """
    universal_rebuild_split(s, train_test_split, standardize_fit, standardize_apply)
