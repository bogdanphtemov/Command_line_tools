import numpy as np 
from typing import Optional, List, Dict, Any

from myclt.ML.base_models import SupervisedModel, BaseModel


class LinearRegressionGD(BaseModel, SupervisedModel):
    """
    y_hat = X @ w + b
    Batch Gradient Descent minimizing MSE.
    Supports L1 (Lasso) and L2 (Ridge) regularization.
    """
    model_type = "linear_regression"
    
    # Initialize the class (model) constructor
    def __init__(self, learning_rate: float = 0.05, epochs: int = 2000, 
                 lambda_l1: float = 0.0, lambda_l2: float = 0.0):
        """
        Initialize Linear Regression with Gradient Descent.
        
        Args:
            learning_rate: Step size for gradient descent (> 0)
            epochs: Maximum number of full-batch training iterations (>= 1)
            lambda_l1: L1 regularization strength (>= 0, 0 = disabled)
            lambda_l2: L2 regularization strength (>= 0, 0 = disabled)

        Raises:
            ValueError: If learning_rate <= 0, epochs < 1, or lambda_* < 0.
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.lambda_l1 = lambda_l1  # L1 regularization strength (Lasso)
        self.lambda_l2 = lambda_l2  # L2 regularization strength (Ridge)
        self.w: Optional[np.ndarray] = None
        self.b: float = 0.0
        self.loss_history: List[float] = []
        self._validate_hyperparams()

    def _validate_hyperparams(self) -> None:
        """Validate constructor / loaded hyperparameters."""
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")
        if self.lambda_l1 < 0:
            raise ValueError(f"lambda_l1 must be >= 0, got {self.lambda_l1}")
        if self.lambda_l2 < 0:
            raise ValueError(f"lambda_l2 must be >= 0, got {self.lambda_l2}")

    @staticmethod
    def _validate_inputs(X, y, *, name_x="X", name_y="y"):
        """Validate input shapes and finiteness."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"{name_x} must be 2D (n_samples, n_features), got {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"{name_y} must be 1D (n_samples,), got {y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"{name_x}/{name_y} sample mismatch: {X.shape[0]} vs {y.shape[0]}")
        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
            raise ValueError(f"{name_x}/{name_y} contain NaN or infinity")
        return X, y

    def _forward(self, X, y):
        """Forward pass: compute predictions, errors, and total loss."""
        y_pred = X @ self.w + self.b
        errors = y_pred - y
        # Base MSE loss
        mse_loss = float(np.mean(errors ** 2))
        # Regularization penalties
        l1_penalty = self.lambda_l1 * np.sum(np.abs(self.w))
        l2_penalty = (self.lambda_l2 / 2) * np.sum(self.w ** 2)
        # Total loss
        loss = mse_loss + l1_penalty + l2_penalty
        return errors, float(loss)

    def _apply_gradient(self, X, errors, n_samples):
        """Apply gradient update; return True if weights remain finite."""
        dw = (2.0 / n_samples) * (X.T @ errors)
        # Add L1 gradient penalty: λ₁ × sign(w)
        if self.lambda_l1 > 0:
            dw += self.lambda_l1 * np.sign(self.w)
        # Add L2 gradient penalty: λ₂ × w
        if self.lambda_l2 > 0:
            dw += self.lambda_l2 * self.w
        db = (2.0 / n_samples) * float(np.sum(errors))
        # Update weights and bias
        self.w -= self.learning_rate * dw
        self.b -= self.learning_rate * db
        # Check finiteness after update
        return bool(np.all(np.isfinite(self.w)) and np.isfinite(self.b))

    # check if the model is trained
    @property
    def is_trained(self) -> bool:
        """Returns True if model has finite trained weights and bias, False otherwise."""
        return (
            self.w is not None
            and bool(np.all(np.isfinite(self.w)))
            and bool(np.isfinite(self.b))
        )

    # model training method
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the model using batch gradient descent.
        
        Uses the shared gradient update logic: 
        - MSE loss with optional L1/L2 regularization
        - Updates: w -= lr * (∇MSE + λ₁⋅sign(w) + λ₂⋅w), b -= lr * ∇b
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training targets, 1D (n_samples,). If you have shape (n, 1), call y.ravel() first.
            
        Raises:
            ValueError: If input shapes mismatch or contain non-finite values,
                or if hyperparameters are invalid (e.g. epochs < 1, learning_rate <= 0).
            
        Note:
            No convergence check - unsuitable learning rates may produce non-finite weights,
            detectable via is_trained property or direct np.isfinite(w) check.
        """
        self._validate_hyperparams()
        X, y = self._validate_inputs(X, y)
        n_samples, n_features = X.shape
        
        # Initialize weights and bias
        self.w = np.zeros(n_features, dtype=float)
        self.b = 0.0 
        self.loss_history = []
        
        # Training loop
        for epoch in range(1, self.epochs + 1):
            # Forward pass and loss calculation
            errors, loss = self._forward(X, y)
            self.loss_history.append(loss)
            
            # Gradient update
            self._apply_gradient(X, errors, n_samples)
            
            # Note: No divergence guard - weights may become non-finite with bad lr 
    
    # New method: training with early stopping for acceleration
    def fit_with_early_stopping(self, X_train: np.ndarray, y_train: np.ndarray,
                                X_val: np.ndarray, y_val: np.ndarray,
                                patience: int = 50, min_delta: float = 1e-6,
                                verbose: bool = False) -> None:
        """
        Training with Early Stopping for convergence acceleration.
        
        Stops training if validation loss doesn't improve for 'patience' epochs.
        Restores the best weights after stopping.
        
        Args:
            X_train, y_train: training data
            X_val, y_val: validation data  
            patience: number of epochs without improvement before stopping
            min_delta: minimum improvement required to reset patience
            verbose: print progress information
            
        Raises:
            ValueError: If input shapes mismatch or contain non-finite values,
                or if hyperparameters are invalid (e.g. epochs < 1, learning_rate <= 0)
            
        Note:
            On successful termination — natural completion, early stopping, or
            divergence after at least one valid epoch (best_w is not None) — the
            best weights are restored, and:
              * loss_history[-1] is the loss of the final (restored) weights
              * len(loss_history) == best_epoch + 1

            If training diverges before any valid epoch (best_w is None), the model
            is left in a non-finite state and loss_history[-1] is NaN.
        """
        # Validate all inputs
        self._validate_hyperparams()
        X_train, y_train = self._validate_inputs(X_train, y_train, name_x="X_train", name_y="y_train")
        X_val, y_val = self._validate_inputs(X_val, y_val, name_x="X_val", name_y="y_val")
        
        if X_train.shape[1] != X_val.shape[1]:
            raise ValueError(f"X_train/X_val feature mismatch: {X_train.shape[1]} vs {X_val.shape[1]}")
            
        n_samples, n_features = X_train.shape
        
        # Initialize weights and bias
        self.w = np.zeros(n_features, dtype=float)
        self.b = 0.0
        self.loss_history = []
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_w = None
        best_b = 0.0
        best_epoch = 0
        
        # Training with validation check
        for epoch in range(1, self.epochs + 1):
            # ===== TRAINING =====
            errors, train_loss = self._forward(X_train, y_train)
            self.loss_history.append(train_loss)
            finite = self._apply_gradient(X_train, errors, n_samples)

            # ===== DIVERGENCE GUARD =====
            if not finite:
                if verbose:
                    print(f"!Training diverged at epoch {epoch} (weights became NaN/inf)!")
                # Restore best weights if available (A1 + D1)
                if best_w is not None:
                    self.w = best_w.copy()
                    self.b = best_b
                    self._trim_loss_history(X_train, y_train, best_epoch)
                else:
                    # No valid weights to restore: final model state is non-finite,
                    # so the last history entry must reflect that (loss of NaN weights).
                    self.loss_history.append(float("nan"))
                    if verbose:
                        print("!Training diverged before any valid weights - model state is non-finite!")
                return
            
            # ===== VALIDATION =====
            y_val_pred = X_val @ self.w + self.b
            val_errors = y_val_pred - y_val
            val_mse = float(np.mean(val_errors ** 2))
            val_l1 = self.lambda_l1 * np.sum(np.abs(self.w))
            val_l2 = (self.lambda_l2 / 2) * np.sum(self.w ** 2)
            val_loss = val_mse + val_l1 + val_l2
            
            # ===== EARLY STOPPING =====
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                best_w = self.w.copy()
                best_b = self.b
                best_epoch = epoch
            else:
                patience_counter += 1
            
            # Stop if no improvement (A1: check best_w before using)
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}. Best val loss: {best_val_loss:.6f}")
                
                # Restore best weights if available (A1)
                if best_w is not None:
                    self.w = best_w.copy()
                    self.b = best_b
                    self._trim_loss_history(X_train, y_train, best_epoch)
                else:
                    if verbose:
                        print("!No improvement observed - keeping current weights!")
                break
            
            if verbose and epoch % max(1, self.epochs // 10) == 0:
                print(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
        
        # Handle case when loop finishes without early stopping
        if best_w is not None and patience_counter < patience:
            self.w = best_w.copy()
            self.b = best_b
            self._trim_loss_history(X_train, y_train, best_epoch)
            if verbose:
                print(f"Training completed at epoch {self.epochs}. Best val loss: {best_val_loss:.6f}")
    
    def _trim_loss_history(self, X_train, y_train, best_epoch):
        """
        Trim loss_history to `best_epoch` entries and append the loss of the
        current weights.

        Callers must set self.w/self.b to the final (already restored best) weights
        before calling. After the call: len(loss_history) == best_epoch + 1 and
        loss_history[-1] == loss(final weights).
        """
        self.loss_history = self.loss_history[:best_epoch]
        _, final_loss = self._forward(X_train, y_train)
        self.loss_history.append(final_loss)
    # method of making predictions         
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Predictions (n_samples,)
            
        Raises:
            RuntimeError: If model is not trained or weights are non-finite
            ValueError: If X has wrong shape or contains NaN/inf
        """
        if self.w is None:
            raise RuntimeError("!Model is not trained yet!")
        if not np.all(np.isfinite(self.w)) or not np.isfinite(self.b):
            raise RuntimeError("!Model weights are not finite (training diverged)!")
        
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != len(self.w):
            raise ValueError(f"X must be 2D with {len(self.w)} features, got shape {X.shape}")
        if not np.all(np.isfinite(X)):
            raise ValueError("X contains NaN or infinity")
        
        return X @ self.w + self.b
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get parameters to save
        Includes weights, biases, and hyperparameters
        """
        return {
            "w": self.w.tolist() if self.w is not None else None,
            "b": float(self.b),
            "learning_rate": float(self.learning_rate),
            "epochs": int(self.epochs),
            "lambda_l1": float(self.lambda_l1),
            "lambda_l2": float(self.lambda_l2),
            "loss_history": [
                float(x) if np.isfinite(x) else None  # NaN/Inf -> JSON null
                for x in self.loss_history
            ],
        }
    
    # set parameters from downloaded data
    def set_params(self, params: Dict[str, Any]) -> None:
        """
        Set model parameters from loaded data.
        
        Symmetric to get_params(); restores untrained state when w is None.
        
        Args:
            params: Dictionary from get_params() with parameters to restore
        """
        w = params.get("w", None)
        self.w = np.array(w, dtype=float) if w is not None else None
        self.b = float(params.get("b", 0.0))
        self.learning_rate = float(params.get("learning_rate", 0.05))
        self.epochs = int(params.get("epochs", 2000))
        self.lambda_l1 = float(params.get("lambda_l1", 0.0))
        self.lambda_l2 = float(params.get("lambda_l2", 0.0))
        self.loss_history = [
            float(x) if x is not None else float("nan")  # JSON null -> NaN
            for x in (params.get("loss_history") or [])
        ]
        self._validate_hyperparams()  # Check restored hyperparameters
