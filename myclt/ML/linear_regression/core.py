import numpy as np 
from typing import Optional , List , Dict , Any

from ML.base_models import SupervisedModel

class LinearRegressionGD:
    """
    y_hat = X @ w + b
    Batch Gradient Descent minimizing MSE.
    Supports L1 (Lasso) and L2 (Ridge) regularization.
    """
    model_type = "linear_regression"
    # Initialize the class (model) constructor
    def __init__(self , learning_rate: float = 0.05 , epochs: int = 2000 , 
                 lambda_l1: float = 0.0 , lambda_l2: float = 0.0):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.lambda_l1 = lambda_l1  # L1 regularization strength (Lasso)
        self.lambda_l2 = lambda_l2  # L2 regularization strength (Ridge)
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
            
            # Base MSE loss
            mse_loss = float(np.mean(errors ** 2))
            
            # L1 regularization penalty: λ₁ × Σ|w|
            l1_penalty = self.lambda_l1 * np.sum(np.abs(self.w))
            
            # L2 regularization penalty: λ₂ × Σ(w²)
            l2_penalty = self.lambda_l2 * np.sum(self.w ** 2)
            
            # Total loss: MSE + L1 + L2
            loss = mse_loss + l1_penalty + l2_penalty
            
            # determination of new weights
            self.loss_history.append(loss)
            dw = (2.0 / n_samples) * (X.T @ errors)
            
            # Add L1 gradient penalty: λ₁ × sign(w)
            if self.lambda_l1 > 0:
                dw += self.lambda_l1 * np.sign(self.w)
            
            # Add L2 gradient penalty: 2 × λ₂ × w
            if self.lambda_l2 > 0:
                dw += 2 * self.lambda_l2 * self.w
            
            db = (2.0 / n_samples) * float(np.sum(errors))

            # weight change
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db 
    
    # New method: training with early stopping for acceleration
    def fit_with_early_stopping(self, X_train: np.ndarray, y_train: np.ndarray,
                                X_val: np.ndarray, y_val: np.ndarray,
                                patience: int = 50, verbose: bool = False) -> None:
        """
        Training with Early Stopping for convergence acceleration.
        
        Stops training if validation loss doesn't improve for 'patience' epochs.
        
        Args:
            X_train, y_train: training data
            X_val, y_val: validation data
            patience: number of epochs without improvement before stopping
            verbose: print progress information
        """
        # Initializing initial weights
        n_samples, n_features = X_train.shape
        self.w = np.zeros(n_features, dtype=float)
        self.b = 0.0
        self.loss_history = []
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Training with validation check
        for epoch in range(1, self.epochs + 1):
            # ===== TRAINING =====
            y_pred = X_train @ self.w + self.b
            errors = y_pred - y_train
            
            mse_loss = float(np.mean(errors ** 2))
            l1_penalty = self.lambda_l1 * np.sum(np.abs(self.w))
            l2_penalty = self.lambda_l2 * np.sum(self.w ** 2)
            train_loss = mse_loss + l1_penalty + l2_penalty
            
            self.loss_history.append(train_loss)
            dw = (2.0 / n_samples) * (X_train.T @ errors)
            
            if self.lambda_l1 > 0:
                dw += self.lambda_l1 * np.sign(self.w)
            if self.lambda_l2 > 0:
                dw += 2 * self.lambda_l2 * self.w
            
            db = (2.0 / n_samples) * float(np.sum(errors))
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db
            
            # ===== VALIDATION =====
            y_val_pred = X_val @ self.w + self.b
            val_errors = y_val_pred - y_val
            
            val_mse = float(np.mean(val_errors ** 2))
            val_l1 = self.lambda_l1 * np.sum(np.abs(self.w))
            val_l2 = self.lambda_l2 * np.sum(self.w ** 2)
            val_loss = val_mse + val_l1 + val_l2
            
            # ===== EARLY STOPPING =====
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Stop if no improvement
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}. Best val loss: {best_val_loss:.6f}")
                break
            
            if verbose and epoch % max(1, self.epochs // 10) == 0:
                print(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
            
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
            "w": self.w.tolist() if self.w is not None else None,
            "b": float(self.b),
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "lambda_l1": self.lambda_l1,
            "lambda_l2": self.lambda_l2,
            "loss_history": self.loss_history,
        }
    
    # set parameters from downloaded data
    def set_params(self , params: Dict[str , Any]) -> None:
        if params["w"] is not None:
            self.w = np.array(params["w"] , dtype=float)
            self.b = params["b"]
            self.learning_rate = params["learning_rate"]
            self.epochs = params["epochs"]
            self.lambda_l1 = params.get("lambda_l1" , 0.0)
            self.lambda_l2 = params.get("lambda_l2" , 0.0)
            self.loss_history = params.get("loss_history" , [])