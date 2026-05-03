import numpy as np
from typing import Dict , Any
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


