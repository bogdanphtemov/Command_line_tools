import numpy as np
from typing import Dict , Any, Callable, Optional
from abc import ABC , abstractmethod

class BaseModel(ABC):
    """
    Base class for all ML models.
    
    REQUIRED for all implementations:
      - model_type: str class attribute (e.g., "linear_regression", "logistic_regression")
      - is_trained: bool property (returns True if model has learned parameters)
      - get_params(): Dict[str, Any] - returns all model parameters for saving
      - set_params(params): None - restores model from saved parameters
    
    This interface enables universal session storage/loading for ANY algorithm.
    """
    
    # Every model must define its type
    model_type: str = None
    
    # Every model must know if it's been trained
    @property
    @abstractmethod
    def is_trained(self) -> bool:
        """
        Returns True if model has been trained and is ready for predictions.
        Check this before saving/predicting.
        """
        raise NotImplementedError("Subclass must implement is_trained property")
    
    @abstractmethod
    def get_params(self) -> Dict[str , Any]:
        """
        Get all model parameters for saving.
        
        Must be JSON-serializable (use .tolist() for numpy arrays).
        Include:
          - Learned parameters (weights, biases, etc.)
          - Hyperparameters (learning_rate, epochs, regularization, etc.)
          - Any other state needed to restore the model
        
        Returns:
            Dict with all parameters needed for reconstruction
        """
        raise NotImplementedError("Subclass must implement get_params()")
    
    @abstractmethod
    def set_params(self , params: Dict[str , Any]) -> None:
        """
        Set model parameters from loaded data.
        
        Restores the model to a previously saved state.
        Must handle None values gracefully (for untrained models).
        
        Args:
            params: Dict from get_params() with parameters to restore
        """
        raise NotImplementedError("Subclass must implement set_params()")


class SupervisedModel(ABC):
    """
    Base class for supervised learning models (with labeled X, y).
    
    For: Linear Regression, Logistic Regression, KNN, Decision Trees, etc.
    
    REQUIRED for all implementations:
      - fit(X, y): None - train the model
      - predict(X): np.ndarray - make predictions
    """
    
    @abstractmethod
    def fit(self , X: np.ndarray , y: np.ndarray) -> None:
        """
        Train the model on labeled data.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
        """
        raise NotImplementedError("Subclass must implement fit()")
    
    @abstractmethod
    def predict(self , X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Predictions (n_samples,) or (n_samples, n_outputs)
        """
        raise NotImplementedError("Subclass must implement predict()")


# ============================================================================
# Universal utilities for AppState management (shared by all models)
# ============================================================================

def universal_rebuild_split(state: Any, train_test_split_fn: Callable, 
                            standardize_fit_fn: Callable, 
                            standardize_apply_fn: Callable) -> None:
    """
    Universal train/test split builder for any supervised learning model.
    
    Handles:
      - Splitting prepared data into train/test
      - Optional feature scaling (standardization)
      - Resetting model and metrics when data pipeline changes
    
    Args:
        state: AppState-like object with:
          - prepareddata: Prepareddata object or None
          - test_size, seed: Split parameters
          - use_scaling: Boolean flag for scaling
          - X_train, X_test, y_train, y_test: Will be set
          - scaler_mean, scaled_std: Will be set or None
          - model: Will be reset to None
          - reset_metrics(): Method to reset all metrics
        train_test_split_fn: Function to split data (X, y, test_size, seed) -> (X_train, X_test, y_train, y_test, train_idx, test_idx)
        standardize_fit_fn: Function to fit scaler (X_train) -> (X_scaled, mean, std)
        standardize_apply_fn: Function to apply scaler (X_test, mean, std) -> X_scaled
    
    Raises:
        RuntimeError: If features/target not selected (prepareddata is None)
    """
    if state.prepareddata is None:
        raise RuntimeError("!No features/target selection yet!")
    
    X, y = state.prepareddata.X, state.prepareddata.Y
    
    # Split data
    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split_fn(
        X, y, test_size=state.test_size, seed=state.seed
    )
    
    # Apply scaling if enabled
    if state.use_scaling:
        X_train_scaled, mean, std = standardize_fit_fn(X_train)
        X_test_scaled = standardize_apply_fn(X_test, mean, std)
        state.X_train, state.X_test = X_train_scaled, X_test_scaled
        state.scaler_mean, state.scaled_std = mean, std
    else:
        state.X_train, state.X_test = X_train, X_test
        state.scaler_mean, state.scaled_std = None, None
    
    state.y_train, state.y_test = y_train, y_test
    state.train_idx, state.test_idx = train_idx, test_idx
    
    # Reset model and metrics (data pipeline changed)
    state.model = None
    state.reset_metrics()


def universal_print_status(state: Any, model_name: str, 
                          metrics_format_fn: Callable[[Any], str],
                          regularization_format_fn: Callable[[Any], str]) -> None:
    """
    Universal status printer for any supervised learning model.
    
    Displays dataset, features, split, scaling, model, regularization, and metrics.
    
    Args:
        state: AppState-like object with dataset, prepareddata, model, etc.
        model_name: Name to display (e.g., "Linear Regression", "Logistic Regression")
        metrics_format_fn: Function(state) -> str to format model-specific metrics
        regularization_format_fn: Function(state) -> str to format regularization info
    
    Example:
        def lr_metrics(s):
            if not s.metrics:
                return "none"
            return " | ".join(
                f"{k}={v:.4f}"
                for k, v in s.metrics.items()
            )
        
        def lr_reg(s):
            if not s.use_l1 and not s.use_l2:
                return "OFF"
            parts = []
            if s.use_l1: parts.append(f"L1(λ={s.lambda_l1})")
            if s.use_l2: parts.append(f"L2(λ={s.lambda_l2})")
            return " + ".join(parts)
        
        universal_print_status(state, "Linear Regression", lr_metrics, lr_reg)
    """
    ds = "none" if state.dataset is None else f"loaded ({state.dataset.data.shape[0]} rows, {state.dataset.data.shape[1]} cols)"
    sup = "none" if state.prepareddata is None else f"{len(state.prepareddata.feature_names)} features → target '{state.prepareddata.target_name}'"
    trained = "no" if state.model is None else "yes"
    metrics = metrics_format_fn(state)
    reg_status = regularization_format_fn(state)
    
    print(f"\nDataset:      {ds}")
    print(f"Selection:    {sup}")
    print(f"Split:        test_size={state.test_size}; seed={state.seed}")
    print(f"Scaling:      {'ON' if state.use_scaling else 'OFF'}")
    print(f"Model:        trained={trained} (lr={state.learning_rate}, epochs={state.epochs})")
    print(f"Regularization: {reg_status}")
    print(f"Metrics:      {metrics}")
    print("=" * 80)


