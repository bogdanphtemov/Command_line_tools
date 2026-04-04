import numpy as np 
from typing import Optional , List

class LinearRegressionGD:
    """
    y_hat = X @ w + b
    Batch Gradient Descent minimizing MSE.
    """
    # Initialize the class (model) constructor
    def __init__(self , learning_rate: float = 0.05 , epochs: int = 2000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w: Optional[np.ndarray] = None
        self.b: float = 0.0
        self.loss_history: List[float] = []
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