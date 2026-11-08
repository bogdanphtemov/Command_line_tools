from dataclasses import dataclass, field
from typing import Optional 
import numpy as np

from .data import Dataset , Prepareddata
from .core import LinearRegressionGD
from .preprocessing import train_test_split , standardize_fit , standardize_apply
from ...base_models import universal_rebuild_split, universal_print_status, TEST_SIZE_MIN, TEST_SIZE_MAX

@dataclass(eq=False)
class AppState:
    dataset: Optional[Dataset] = None
    prepareddata: Optional[Prepareddata] = None
    test_size: float = 0.2
    seed: int = 42
    use_scaling: bool = True
    learning_rate: float = 0.05
    learning_rate_auto: bool = True
    epochs: int = 10000
    early_stopping: bool = True
    
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
    scaler_std: Optional[np.ndarray] = None

    model: Optional[LinearRegressionGD] = None
    
    # evaluation metrics
    metrics: dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        # Normalize learning_rate_auto to bool type
        self.learning_rate_auto = bool(self.learning_rate_auto)
        
        # Validate state on initialization
        self.validate()
    
    def reset_metrics(self) -> None:
        """Clear all evaluation metrics."""
        self.metrics.clear()
    
    def validate(self) -> None:
        """
        Validate all parameters.
        
        Raises:
            ValueError: If any parameter constraint is violated
        """
        if not (TEST_SIZE_MIN <= self.test_size <= TEST_SIZE_MAX):
            raise ValueError(
                f"test_size must be in [{TEST_SIZE_MIN}, {TEST_SIZE_MAX}], got {self.test_size}"
            )
        
        if isinstance(self.seed, bool) or not (
            isinstance(self.seed, (int, np.integer)) and self.seed >= 0
        ):
            raise ValueError(f"seed must be a non-negative integer, got {self.seed}")
        
        if not isinstance(self.learning_rate_auto, bool):
            raise ValueError(f"learning_rate_auto must be boolean, got {type(self.learning_rate_auto).__name__}")
        
        if not self.learning_rate_auto and self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")
        
        if self.lambda_l1 < 0:
            raise ValueError(f"lambda_l1 must be >= 0, got {self.lambda_l1}")
        
        if self.lambda_l2 < 0:
            raise ValueError(f"lambda_l2 must be >= 0, got {self.lambda_l2}")

# this function determines which values ​​we have already defined and displays this information to the user
def print_status(s: AppState) -> None:
    """
    Display comprehensive status of the current session.
    
    Shows what data is loaded, features selected, model state, and metrics.
    Uses universal printer from base_models.
    """
    def format_metrics(state):
        if not state.metrics:
            return "NONE"

        def _fmt(v):
            try:
                fv = float(v)
            except (TypeError, ValueError):
                return str(v)
            return f"{fv:.4g}" if np.isfinite(fv) else "—"

        return " | ".join(
            f"{k}={_fmt(v)}"
            for k, v in state.metrics.items()
        )
    
    def format_regularization(state):
        parts = []
        if state.use_l1 and state.lambda_l1 > 0:
            parts.append(f"L1(λ={state.lambda_l1})")
        if state.use_l2 and state.lambda_l2 > 0:
            parts.append(f"L2(λ={state.lambda_l2})")
        return " + ".join(parts) if parts else "OFF"
    
    universal_print_status(s, "Linear Regression", format_metrics, format_regularization)

def rebuild_split(s: AppState) -> None:
    """
    Build train/test split from supervised data, apply scaling if enabled.
    Uses universal split builder from base_models.
    """
    s.validate()
    universal_rebuild_split(s, train_test_split, standardize_fit, standardize_apply)
