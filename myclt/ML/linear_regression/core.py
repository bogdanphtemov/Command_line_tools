import numpy as np 
from typing import Optional , List , Dict , Any

from ML.base_models import SupervisedModel

class LinearRegressionGD:
    """
    y_hat = X @ w + b
    Batch Gradient Descent minimizing MSE.
    """
    model_type = "linear_regression"
    # Initialize the class (model) constructor
    def __init__(self , learning_rate: float = 0.05 , epochs: int = 2000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w: Optional[np.ndarray] = None
        self.b: float = 0.0
        self.loss_history: List[float] = []

    # check if the model is trained
    @property
    def is_trained(self) -> bool:
        return self.w is not None

    # model training method
    def fit(self , X: np.ndarray , y: np.ndarray) -> None:
        # Initializing initial weights
        n_samples , n_features = X.shape
        self.w = np.zeros(n_features , dtype=float)
        self.b = 0.0 
        self.loss_history = []
        
        # model training cycle
        for epoch in range(1 , self.epochs + 1):
            # prediction and error detection
            y_pred = X @ self.w + self.b
            errors = y_pred - y
            loss = float(np.mean(errors ** 2))
            
            # determination of new weights
            self.loss_history.append(loss)
            dw = (2.0 / n_samples) * (X.T @ errors)
            db = (2.0 / n_samples) * float(np.sum(errors))

            # weight change
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db 
    # method of making predictions         
    def predict(self , X: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("!Model is not trained yet!")
        
        return X @ self.w + self.b
    
    def get_params(self) -> Dict[str , Any]:
        """
        Get parameters to save
        Includes weights, biases, and hyperparameters
        """
        return {
            "w":self.w.tolist() if self.w is not None else None,
            "b": float(self.b),
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "loss_history": self.loss_history,
        }
    
    # set parameters from downloaded data
    def set_params(self , params: Dict[str , Any]) -> None:
        if params["w"] is not None:
            self.w = np.array(params["w"] , dtype=float)
            self.b = params["b"]
            self.learning_rate = params["learning_rate"]
            self.epochs = params["epochs"]
            self.loss_history = params.get("loss_history" , [])