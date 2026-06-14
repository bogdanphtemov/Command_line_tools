import numpy as np
from typing import Dict , Any, Callable, Optional, Tuple
from abc import ABC , abstractmethod


# ============================================================================
# Universal preprocessing functions (shared by all supervised models)
# ============================================================================

def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Random split with fixed seed for reproducibility.
    This is a standard practice in machine learning 
    that helps detect overfitting of the model.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target vector (n_samples,)
        test_size: Proportion of data for test set (0.05-0.5)
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, train_idx, test_idx)
    """
    if not(0.05 <= test_size <= 0.5):
        raise ValueError("!test_size should be between 0.05 and 0.5 for this tool!")
    
    # we take the number of examples
    n = X.shape[0]
    
    # creating a random number generator with a fixed code for reproducibility of results
    rng = np.random.RandomState(seed)
    
    # randomize indices
    idx = np.arange(n)
    rng.shuffle(idx)

    test_n = int(round(n * test_size))
    test_idx = idx[:test_n]

    train_idx = idx[test_n:]

    X_train = X[train_idx]
    y_train = y[train_idx]

    X_test = X[test_idx]
    y_test = y[test_idx]

    # Return indices as well so callers can persist/reconstruct splits reliably
    return X_train, X_test, y_train, y_test, train_idx, test_idx


def standardize_fit(X_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit standardization on TRAIN only:
    mean, std per feature.
    Returns scaled X_train + mean + std.
    
    Args:
        X_train: Training feature matrix (n_train_samples, n_features)
    
    Returns:
        Tuple of (X_scaled, mean, std_safe) where std_safe has 1.0 for zero-std features
    """
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)

    # division by zero protection
    std_safe = np.where(std == 0.0, 1.0, std)
    X_scaled = (X_train - mean) / std_safe

    return X_scaled, mean, std_safe


def standardize_apply(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    Apply fitted standardization to new data.
    
    Args:
        X: Feature matrix to scale (n_samples, n_features)
        mean: Mean from training data (n_features,)
        std: Standard deviation from training data (n_features,)
    
    Returns:
        Scaled feature matrix (n_samples, n_features)
    """
    return (X - mean) / std


# ============================================================================
# Base classes for ML models
# ============================================================================

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


