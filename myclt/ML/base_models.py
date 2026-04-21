import numpy as np
from typing import Dict , Any
from abc import ABC , abstractmethod

class BaseModel(ABC):
    """
    Base class for all ML models
    Requires implementation of parameter-saving methods
    """
    model_type: str = None

    @property
    def is_trained(self) -> bool:
        raise NotImplemented
    
    @abstractmethod
    def get_params(self) -> Dict[str , Any]:
        """
        Get model parameters for saving
        Each model knows which parameters are important to her
        """
        pass
    
    @abstractmethod
    def set_params(self , params: Dict[str , Any]) -> None:
        """
        Set model parameters from the loaded data
        """
        pass

class SupervisedModel():
    """
    Base class for supervised models (with X, y)
    For: Linear Regression, Logistic Regression, Decision Trees, etc
    """
    
    # train the model
    @abstractmethod
    def fit(self , X: np.ndarray , y: np.ndarray) -> None:
        pass
    
    # Make a prediction
    @abstractmethod
    def predict(self , X: np.ndarray) -> np.ndarray:
        pass
    
    


