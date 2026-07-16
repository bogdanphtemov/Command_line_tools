import numpy as np 
from typing import Optional, List, Dict, Any, Union

from myclt.ML.base_models import SupervisedModel, BaseModel

class LogisticRegressionGD(BaseModel, SupervisedModel):
    """
    Binary Logistic Regression using Gradient Descent.
    
    Mathematical model:
        z = X @ w + b
        y_hat = sigmoid(z) = 1 / (1 + exp(-z))
        
    Loss function: Binary Cross-Entropy
        L = -[y*log(p) + (1-y)*log(1-p)]
    
    Example:
        >>> model = LogisticRegressionGD(learning_rate=0.01, epochs=1000)
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)  # Returns 0 or 1
        >>> probabilities = model.predict_proba(X_test)  # Returns probabilities [0,1]
    
    Features:
        - L2 (Ridge) regularization support
        - Early stopping for faster convergence
        - Probability predictions via predict_proba()
    """
    
    model_type = "logistic_regression"
    
    def __init__(self, learning_rate: float = 0.01, epochs: int = 1000, 
                 lambda_l2: float = 0.0, threshold: float = 0.5):
        """
        Initialize Logistic Regression model.
        
        Args:
            learning_rate: Step size for gradient descent
            epochs: Maximum number of training iterations
            lambda_l2: L2 regularization strength (Ridge)
            threshold: Classification threshold for binary output (default 0.5)
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.lambda_l2 = lambda_l2  # L2 regularization (Ridge)
        self.threshold = threshold  # Classification threshold
        
        self.w: Optional[np.ndarray] = None
        self.b: float = 0.0
        self.loss_history: List[float] = []
    
    @property
    def is_trained(self) -> bool:
        """Check if model has been trained."""
        return self.w is not None
    
    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """
        Sigmoid activation function: 1 / (1 + exp(-z))
        Numerically stable version to prevent overflow.
        """
        z = np.clip(z, -500, 500)  # Prevent overflow
        return 1.0 / (1.0 + np.exp(-z))
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get probability predictions (continuous [0, 1]).
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Probabilities (n_samples,) where values are in [0, 1]
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained yet!")
        
        z = X @ self.w + self.b
        return self._sigmoid(z)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Get binary predictions (0 or 1).
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Binary predictions (n_samples,) with values 0 or 1
        """
        proba = self.predict_proba(X)
        return (proba >= self.threshold).astype(int)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the logistic regression model using batch gradient descent.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,) with values 0 or 1
        """
        n_samples, n_features = X.shape
        
        # Initialize weights and bias
        self.w = np.zeros(n_features, dtype=float)
        self.b = 0.0
        self.loss_history = []
        
        # Training loop
        for epoch in range(1, self.epochs + 1):
            # Forward pass: predictions
            z = X @ self.w + self.b
            y_pred = self._sigmoid(z)
            
            # Binary cross-entropy loss
            # Avoid log(0) by clipping predictions
            y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
            bce_loss = -np.mean(y * np.log(y_pred_clipped) + (1 - y) * np.log(1 - y_pred_clipped))
            
            # L2 regularization penalty: (λ/2) × Σ(w²)
            l2_penalty = (self.lambda_l2 / 2) * np.sum(self.w ** 2)
            
            # Total loss
            loss = bce_loss + l2_penalty
            self.loss_history.append(loss)
            
            # Backward pass: compute gradients
            errors = y_pred - y
            dw = (1.0 / n_samples) * (X.T @ errors)
            
            # Add L2 gradient penalty: λ × w
            if self.lambda_l2 > 0:
                dw += self.lambda_l2 * self.w
            
            db = (1.0 / n_samples) * np.sum(errors)
            
            # Update weights and bias
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db
    
    def fit_with_early_stopping(self, X_train: np.ndarray, y_train: np.ndarray,
                                X_val: np.ndarray, y_val: np.ndarray,
                                patience: int = 50, verbose: bool = False) -> None:
        """
        Train with early stopping to prevent overfitting.
        
        Stops training if validation loss doesn't improve for 'patience' epochs.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            patience: Number of epochs without improvement before stopping
            verbose: Print progress information
        """
        n_samples, n_features = X_train.shape
        
        # Initialize weights and bias
        self.w = np.zeros(n_features, dtype=float)
        self.b = 0.0
        self.loss_history = []
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(1, self.epochs + 1):
            # Training step
            z_train = X_train @ self.w + self.b
            y_pred_train = self._sigmoid(z_train)
            
            errors_train = y_pred_train - y_train
            dw = (1.0 / n_samples) * (X_train.T @ errors_train)
            if self.lambda_l2 > 0:
                dw += self.lambda_l2 * self.w
            
            db = (1.0 / n_samples) * np.sum(errors_train)
            
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db
            
            # Validation step
            z_val = X_val @ self.w + self.b
            y_pred_val = self._sigmoid(z_val)
            y_pred_val_clipped = np.clip(y_pred_val, 1e-15, 1 - 1e-15)
            
            bce_val = -np.mean(y_val * np.log(y_pred_val_clipped) + (1 - y_val) * np.log(1 - y_pred_val_clipped))
            l2_penalty = (self.lambda_l2 / 2) * np.sum(self.w ** 2)
            val_loss = bce_val + l2_penalty
            
            self.loss_history.append(val_loss)
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}: Val Loss = {val_loss:.6f}")
            
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get all model parameters for saving.
        
        Returns:
            Dictionary with all parameters (weights, bias, hyperparameters)
        """
        return {
            'w': self.w.tolist() if self.w is not None else None,
            'b': float(self.b),
            'learning_rate': float(self.learning_rate),
            'epochs': int(self.epochs),
            'lambda_l2': float(self.lambda_l2),
            'threshold': float(self.threshold),
        }
    
    def set_params(self, params: Dict[str, Any]) -> None:
        """
        Set model parameters from loaded data.
        
        Args:
            params: Dictionary from get_params()
        """
        if params['w'] is not None:
            self.w = np.array(params['w'], dtype=float)
        else:
            self.w = None
        
        self.b = float(params.get('b', 0.0))
        self.learning_rate = float(params.get('learning_rate', 0.01))
        self.epochs = int(params.get('epochs', 1000))
        self.lambda_l2 = float(params.get('lambda_l2', 0.0))
        self.threshold = float(params.get('threshold', 0.5))


class MultinomialLogisticRegression(BaseModel, SupervisedModel):
    """
    Multinomial (Softmax) Logistic Regression using Gradient Descent.
    
    Generalizes binary logistic regression to K > 2 classes.
    
    Mathematical model:
        Z = X @ W + b          # (n_samples, n_classes)
        y_hat = softmax(Z)     # (n_samples, n_classes)
        predicted_class = argmax(y_hat, axis=1)
        
    Loss function: Categorical Cross-Entropy
        L = -Σ(y_onehot * log(y_hat))
    
    Example:
        >>> model = MultinomialLogisticRegression(learning_rate=0.01, epochs=1000)
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)  # Returns 0, 1, ..., K-1
        >>> probabilities = model.predict_proba(X_test)  # Returns (n_samples, n_classes)
    
    Features:
        - Supports any K >= 2 classes
        - L2 (Ridge) regularization support
        - Early stopping for faster convergence
        - Full probability matrix via predict_proba()
    """
    
    model_type = "multinomial_logistic_regression"
    
    def __init__(self, learning_rate: float = 0.01, epochs: int = 1000, 
                 lambda_l2: float = 0.0):
        """
        Initialize Multinomial Logistic Regression model.
        
        Args:
            learning_rate: Step size for gradient descent
            epochs: Maximum number of training iterations
            lambda_l2: L2 regularization strength (Ridge)
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.lambda_l2 = lambda_l2
        
        # Weight matrix (n_features, n_classes) and bias vector (n_classes,)
        self.W: Optional[np.ndarray] = None
        self.b: Optional[np.ndarray] = None
        self.n_classes: int = 0
        self.loss_history: List[float] = []
    
    @property
    def is_trained(self) -> bool:
        """Check if model has been trained."""
        return self.W is not None
    
    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        """
        Softmax activation function.
        
        Numerically stable: subtracts max per row before exponentiation.
        Formula: softmax(z_i) = exp(z_i - max(z)) / Σ(exp(z_j - max(z)))
        
        Args:
            z: Logits matrix (n_samples, n_classes)
        
        Returns:
            Probability matrix (n_samples, n_classes) where rows sum to 1
        """
        z = np.clip(z, -500, 500)  # Prevent overflow
        # Subtract max per row for numerical stability
        z_shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    @staticmethod
    def _to_onehot(y: np.ndarray, n_classes: int) -> np.ndarray:
        """
        Convert class labels to one-hot encoding.
        
        Args:
            y: Class labels (n_samples,) with values 0, 1, ..., K-1
            n_classes: Number of classes K
        
        Returns:
            One-hot matrix (n_samples, n_classes)
        """
        onehot = np.zeros((y.shape[0], n_classes))
        onehot[np.arange(y.shape[0]), y.astype(int)] = 1.0
        return onehot
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get probability predictions for all classes.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Probability matrix (n_samples, n_classes) where each row sums to 1.
            Column k corresponds to probability of class k.
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained yet!")
        
        z = X @ self.W + self.b
        return self._softmax(z)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Get class predictions (0, 1, ..., K-1).
        
        Uses argmax to select the most probable class.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Class predictions (n_samples,) with integer labels 0..K-1
        """
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
    
    def predict_proba_for_class(self, X: np.ndarray, class_idx: int) -> np.ndarray:
        """
        Get probability of a specific class.
        
        Useful for ROC analysis per class (one-vs-rest).
        
        Args:
            X: Feature matrix (n_samples, n_features)
            class_idx: Class index (0..K-1)
        
        Returns:
            Probabilities (n_samples,) for the specified class
        """
        proba = self.predict_proba(X)
        return proba[:, class_idx]
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the multinomial logistic regression model.
        
        Uses batch gradient descent with categorical cross-entropy loss.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,) with integer labels 0, 1, ..., K-1
        """
        n_samples, n_features = X.shape
        
        # Determine number of classes from unique values in y
        unique_classes = np.unique(y)
        self.n_classes = len(unique_classes)
        
        # Validate that classes are 0..K-1; remap if needed
        if not np.array_equal(unique_classes, np.arange(self.n_classes)):
            # Remap arbitrary labels to 0..K-1
            self._class_mapping = {old: new for new, old in enumerate(sorted(unique_classes))}
            self._inverse_mapping = {new: old for old, new in self._class_mapping.items()}
            y_mapped = np.array([self._class_mapping[val] for val in y])
        else:
            self._class_mapping = None
            self._inverse_mapping = None
            y_mapped = y.astype(int)
        
        # One-hot encode targets
        y_onehot = self._to_onehot(y_mapped, self.n_classes)
        
        # Initialize weight matrix (n_features, n_classes) and bias vector (n_classes,)
        self.W = np.zeros((n_features, self.n_classes), dtype=float)
        self.b = np.zeros(self.n_classes, dtype=float)
        self.loss_history = []
        
        # Training loop
        for epoch in range(1, self.epochs + 1):
            # Forward pass: logits -> softmax probabilities
            logits = X @ self.W + self.b  # (n_samples, n_classes)
            y_pred = self._softmax(logits)  # (n_samples, n_classes)
            
            # Categorical cross-entropy loss
            y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
            cce_loss = -np.mean(np.sum(y_onehot * np.log(y_pred_clipped), axis=1))
            
            # L2 regularization penalty: (λ/2) × Σ(W²)
            l2_penalty = (self.lambda_l2 / 2) * np.sum(self.W ** 2)
            
            # Total loss
            loss = cce_loss + l2_penalty
            self.loss_history.append(loss)
            
            # Backward pass: gradient of cross-entropy with softmax
            errors = y_pred - y_onehot  # (n_samples, n_classes)
            
            dW = (1.0 / n_samples) * (X.T @ errors)  # (n_features, n_classes)
            
            # Add L2 gradient penalty: λ × W
            if self.lambda_l2 > 0:
                dW += self.lambda_l2 * self.W
            
            db = (1.0 / n_samples) * np.sum(errors, axis=0)  # (n_classes,)
            
            # Update parameters
            self.W -= self.learning_rate * dW
            self.b -= self.learning_rate * db
    
    def fit_with_early_stopping(self, X_train: np.ndarray, y_train: np.ndarray,
                                X_val: np.ndarray, y_val: np.ndarray,
                                patience: int = 50, verbose: bool = False) -> None:
        """
        Train with early stopping to prevent overfitting.
        
        Stops training if validation loss doesn't improve for 'patience' epochs.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            patience: Number of epochs without improvement before stopping
            verbose: Print progress information
        """
        n_samples, n_features = X_train.shape
        
        # Determine classes from union of train and val
        unique_classes = np.unique(np.concatenate([y_train, y_val]))
        self.n_classes = len(unique_classes)
        
        # Remap if needed (same logic as fit)
        if not np.array_equal(unique_classes, np.arange(self.n_classes)):
            self._class_mapping = {old: new for new, old in enumerate(sorted(unique_classes))}
            self._inverse_mapping = {new: old for old, new in self._class_mapping.items()}
            y_train_mapped = np.array([self._class_mapping[val] for val in y_train])
            y_val_mapped = np.array([self._class_mapping[val] for val in y_val])
        else:
            self._class_mapping = None
            self._inverse_mapping = None
            y_train_mapped = y_train.astype(int)
            y_val_mapped = y_val.astype(int)
        
        y_train_onehot = self._to_onehot(y_train_mapped, self.n_classes)
        y_val_onehot = self._to_onehot(y_val_mapped, self.n_classes)
        
        # Initialize weights
        self.W = np.zeros((n_features, self.n_classes), dtype=float)
        self.b = np.zeros(self.n_classes, dtype=float)
        self.loss_history = []
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(1, self.epochs + 1):
            # Training step
            logits_train = X_train @ self.W + self.b
            y_pred_train = self._softmax(logits_train)
            
            errors_train = y_pred_train - y_train_onehot
            dW = (1.0 / n_samples) * (X_train.T @ errors_train)
            if self.lambda_l2 > 0:
                dW += self.lambda_l2 * self.W
            db = (1.0 / n_samples) * np.sum(errors_train, axis=0)
            
            self.W -= self.learning_rate * dW
            self.b -= self.learning_rate * db
            
            # Validation step
            logits_val = X_val @ self.W + self.b
            y_pred_val = self._softmax(logits_val)
            y_pred_val_clipped = np.clip(y_pred_val, 1e-15, 1 - 1e-15)
            
            cce_val = -np.mean(np.sum(y_val_onehot * np.log(y_pred_val_clipped), axis=1))
            l2_penalty = (self.lambda_l2 / 2) * np.sum(self.W ** 2)
            val_loss = cce_val + l2_penalty
            
            self.loss_history.append(val_loss)
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}: Val Loss = {val_loss:.6f}")
            
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get all model parameters for saving.
        
        Returns:
            Dictionary with all parameters (weights matrix, bias, hyperparameters)
        """
        return {
            'W': self.W.tolist() if self.W is not None else None,
            'b': self.b.tolist() if self.b is not None else None,
            'n_classes': int(self.n_classes),
            'learning_rate': float(self.learning_rate),
            'epochs': int(self.epochs),
            'lambda_l2': float(self.lambda_l2),
            '_class_mapping': self._class_mapping,
            '_inverse_mapping': self._inverse_mapping,
        }
    
    def set_params(self, params: Dict[str, Any]) -> None:
        """
        Set model parameters from loaded data.
        
        Args:
            params: Dictionary from get_params()
        """
        if params.get('W') is not None:
            self.W = np.array(params['W'], dtype=float)
        else:
            self.W = None
        
        if params.get('b') is not None:
            self.b = np.array(params['b'], dtype=float)
        else:
            self.b = None
        
        self.n_classes = int(params.get('n_classes', 0))
        self.learning_rate = float(params.get('learning_rate', 0.01))
        self.epochs = int(params.get('epochs', 1000))
        self.lambda_l2 = float(params.get('lambda_l2', 0.0))
        self._class_mapping = params.get('_class_mapping')
        self._inverse_mapping = params.get('_inverse_mapping')
