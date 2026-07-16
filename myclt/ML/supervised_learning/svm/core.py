"""
Support Vector Machine (SVM) — core algorithms.

Provides from-scratch implementations of:

    CLASSIFICATION:
        ─ LinearSVM:        Binary linear SVM with hinge loss + SGD
        ─ KernelSVM:        Binary non-linear SVM with kernel trick + primal RKHS GD
        ─ OneVsRestSVM:     Multiclass wrapper (OvR) for any binary SVM

    REGRESSION (SVR):
        ─ LinearSVR:        Linear Support Vector Regression (epsilon-insensitive)
        ─ KernelSVR:        Non-linear SVR with kernel trick

All implementations use pure NumPy — no scikit-learn dependency.

Mathematical foundation:
    Hinge loss:     L(y, f(x)) = max(0, 1 - y·f(x))
    ε-insensitive:  L(y, f(x)) = max(0, |y - f(x)| - ε)
    Regularization: R(w) = (λ/2) · ||w||²

Example:
    >>> from myclt.ML.supervised_learning.svm import LinearSVM
    >>> model = LinearSVM(C=1.0, learning_rate=0.001, epochs=1000)
    >>> model.fit(X_train, y_train)
    >>> preds = model.predict(X_test)
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Any, Callable
import warnings


# ============================================================================
# Kernel functions
# ============================================================================

def _linear_kernel(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
    """Linear kernel: K(x, z) = x · z"""
    return X1 @ X2.T


def _polynomial_kernel(X1: np.ndarray, X2: np.ndarray,
                       degree: int = 3, gamma: float = 1.0,
                       coef0: float = 1.0) -> np.ndarray:
    """Polynomial kernel: K(x, z) = (γ · x·z + coef0)^degree"""
    return (gamma * (X1 @ X2.T) + coef0) ** degree


def _rbf_kernel(X1: np.ndarray, X2: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """RBF (Gaussian) kernel: K(x, z) = exp(-γ · ||x - z||²)"""
    # ||x - z||² = ||x||² + ||z||² - 2·x·z
    X1_norm = np.sum(X1 ** 2, axis=1, keepdims=True)  # (n1, 1)
    X2_norm = np.sum(X2 ** 2, axis=1, keepdims=True)  # (n2, 1)
    distances = X1_norm + X2_norm.T - 2.0 * (X1 @ X2.T)
    distances = np.maximum(distances, 0.0)  # Numerical safety
    return np.exp(-gamma * distances)


def _sigmoid_kernel(X1: np.ndarray, X2: np.ndarray,
                    gamma: float = 1.0, coef0: float = 0.0) -> np.ndarray:
    """Sigmoid kernel: K(x, z) = tanh(γ · x·z + coef0)"""
    return np.tanh(gamma * (X1 @ X2.T) + coef0)


# Kernel registry
KERNEL_FUNCTIONS: Dict[str, Callable] = {
    'linear': _linear_kernel,
    'poly': _polynomial_kernel,
    'rbf': _rbf_kernel,
    'sigmoid': _sigmoid_kernel,
}


def get_kernel(name: str, **kernel_params) -> Callable:
    """
    Get a kernel function by name.

    Args:
        name: One of 'linear', 'poly', 'rbf', 'sigmoid'
        **kernel_params: Parameters passed to the kernel (gamma, degree, coef0)

    Returns:
        Kernel function with signature (X1, X2) -> kernel_matrix
    """
    name = name.lower()
    if name not in KERNEL_FUNCTIONS:
        raise ValueError(
            f"Unknown kernel '{name}'. Supported: {list(KERNEL_FUNCTIONS.keys())}"
        )

    base_fn = KERNEL_FUNCTIONS[name]

    if name == 'linear':
        return base_fn
    elif name == 'poly':
        return lambda X1, X2: base_fn(
            X1, X2,
            degree=kernel_params.get('degree', 3),
            gamma=kernel_params.get('gamma', 1.0),
            coef0=kernel_params.get('coef0', 1.0)
        )
    elif name == 'rbf':
        return lambda X1, X2: base_fn(
            X1, X2,
            gamma=kernel_params.get('gamma', 1.0)
        )
    elif name == 'sigmoid':
        return lambda X1, X2: base_fn(
            X1, X2,
            gamma=kernel_params.get('gamma', 1.0),
            coef0=kernel_params.get('coef0', 0.0)
        )


# ============================================================================
# Helper: label validator for binary classification
# ============================================================================

def _validate_binary_labels(y: np.ndarray) -> np.ndarray:
    """
    Validate and convert binary labels to {-1, +1}.

    Accepts {0, 1} or {-1, +1} labels. Converts {0, 1} to {-1, +1}.

    Args:
        y: Target vector (n_samples,)

    Returns:
        Labels in {-1, +1} format
    """
    if y.ndim != 1:
        raise ValueError(
            f"y must be 1-dimensional, got shape {y.shape}"
        )
    if y.shape[0] == 0:
        raise ValueError(
            "y is empty — cannot train binary SVM. "
            "Provide at least one sample."
        )

    unique = np.unique(y)
    if len(unique) < 2:
        raise ValueError(
            f"Binary SVM requires exactly 2 classes, got {len(unique)}. "
            f"Found labels: {unique}. All labels have the same value — "
            f"the dataset contains only one class."
        )
    if len(unique) > 2:
        raise ValueError(
            f"Binary SVM requires exactly 2 classes, got {len(unique)}. "
            f"Found labels: {unique}. For multi-class problems, use "
            f"OneVsRestSVM."
        )

    # If labels are {0, 1}, convert to {-1, +1}
    if set(unique) == {0, 1}:
        return np.where(y == 1, 1.0, -1.0)
    elif set(unique) == {-1, 1}:
        return y.astype(float)
    else:
        # General remap: smallest -> -1, largest -> +1
        sorted_unique = sorted(unique)
        return np.where(y == sorted_unique[1], 1.0, -1.0)


def _to_zero_one(y_svm: np.ndarray) -> np.ndarray:
    """Convert {-1, +1} labels back to {0, 1}."""
    return np.where(y_svm == 1, 1, 0)


# ============================================================================
# Base class for all linear models (SVM classification & SVR regression)
# ============================================================================

class BaseLinearModel:
    """
    Base class for linear SVM / SVR models.

    Provides shared gradient-descent training loop, serialisation,
    and early stopping — subclasses only implement loss/gradient logic.

    Attributes shared by subclasses:
        w, b, C, learning_rate, epochs, batch_size,
        loss_history, support_vectors, n_support_vectors, _fitted
    """

    # List of parameter names for serialisation.
    # Subclasses can extend via ``_linear_params + ['my_param']``.
    _linear_params = [
        'w', 'b', 'C', 'learning_rate', 'epochs', 'batch_size',
        'log_every', 'n_support_vectors', '_fitted',
    ]

    def __init__(self, C: float = 1.0, learning_rate: float = 0.001,
                 epochs: int = 1000, batch_size: int = 0,
                 log_every: int = 10):
        self.C = C
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.log_every = log_every

        self.w: Optional[np.ndarray] = None
        self.b: float = 0.0
        self.loss_history: List[float] = []

        self.support_vectors: Optional[np.ndarray] = None
        self.n_support_vectors: int = 0

        self._fitted: bool = False

    # -- Abstract hooks (must be overridden by subclasses) -----------------

    def _compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute the full objective (data loss + regularisation)."""
        raise NotImplementedError

    def _compute_gradients(self, X: np.ndarray, y: np.ndarray
                           ) -> Tuple[np.ndarray, float]:
        """Compute gradients w.r.t. w and b."""
        raise NotImplementedError

    # -- Shared properties --------------------------------------------------

    @property
    def is_trained(self) -> bool:
        """Check if model has been trained."""
        return self._fitted and self.w is not None

    @property
    def lambda_(self) -> float:
        """Regularization parameter λ = 1/C."""
        return 1.0 / self.C if self.C > 0 else 0.0

    # -- Shared prediction --------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values.

        For classification subclasses: returns {0, 1} labels.
        For regression subclasses: returns continuous values.

        Override in subclass if needed (e.g. LinearSVM has decision_function).

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Predictions (n_samples,)
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained yet! Call fit() first.")
        return X @ self.w + self.b

    # -- Shared training loop -----------------------------------------------

    def _shared_fit_loop(self, X: np.ndarray, y_for_grad: np.ndarray,
                         y_for_loss: np.ndarray,
                         early_stopping: bool = False,
                         X_val: Optional[np.ndarray] = None,
                         y_val_for_loss: Optional[np.ndarray] = None,
                         patience: int = 50,
                         verbose: bool = False) -> None:
        """
        Shared gradient-descent loop used by both fit() and
        fit_with_early_stopping().

        Args:
            X:             Training features
            y_for_grad:    Targets used in gradient computation
            y_for_loss:    Targets used in loss computation
            early_stopping: If True, use validation-based early stopping
            X_val:         Validation features
            y_val_for_loss: Validation targets for loss computation
            patience:      Patience for early stopping
            verbose:       Print progress
        """
        n_samples, n_features = X.shape

        self.w = np.zeros(n_features, dtype=float)
        self.b = 0.0
        self.loss_history = []

        # Mini-batch setup
        if self.batch_size <= 0 or self.batch_size >= n_samples:
            batch_size = n_samples
            use_minibatch = False
        else:
            batch_size = min(self.batch_size, n_samples)
            use_minibatch = True

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(1, self.epochs + 1):
            if use_minibatch:
                indices = np.random.permutation(n_samples)
                X_shuffled = X[indices]
                y_shuffled = y_for_grad[indices]

                if epoch % self.log_every == 0 or epoch == 1 or epoch == self.epochs:
                    loss = self._compute_loss(X, y_for_loss)
                    self.loss_history.append(loss)

                for start_idx in range(0, n_samples, batch_size):
                    end_idx = min(start_idx + batch_size, n_samples)
                    X_batch = X_shuffled[start_idx:end_idx]
                    y_batch = y_shuffled[start_idx:end_idx]
                    dw, db = self._compute_gradients(X_batch, y_batch)
                    self.w -= self.learning_rate * dw
                    self.b -= self.learning_rate * db
            else:
                dw, db = self._compute_gradients(X, y_for_grad)
                self.w -= self.learning_rate * dw
                self.b -= self.learning_rate * db
                loss = self._compute_loss(X, y_for_loss)
                self.loss_history.append(loss)

            # Early stopping logic
            if early_stopping and X_val is not None:
                val_loss = self._compute_loss(X_val, y_val_for_loss)

                if val_loss < best_val_loss - 1e-8:
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

        self._identify_support_vectors(X, y_for_loss)
        self._fitted = True

    def _identify_support_vectors(self, X: np.ndarray,
                                  y: np.ndarray) -> None:
        """
        Identify support vectors after training.

        Override in subclasses for model-specific logic.
        Default: all points (for base class).
        """
        self.support_vectors = X
        self.n_support_vectors = X.shape[0]

    # -- Public fit / early stopping ----------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the model using gradient descent.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
        """
        raise NotImplementedError

    def fit_with_early_stopping(self, X_train: np.ndarray, y_train: np.ndarray,
                                X_val: np.ndarray, y_val: np.ndarray,
                                patience: int = 50,
                                verbose: bool = False) -> None:
        """
        Train with early stopping to prevent overfitting.

        Args:
            X_train, y_train: Training data
            X_val, y_val:     Validation data
            patience:         Number of epochs without improvement before stop
            verbose:          Print progress
        """
        raise NotImplementedError

    # -- Serialisation ------------------------------------------------------

    def get_params(self) -> Dict[str, Any]:
        """Get all model parameters for saving."""
        params = {}
        for key in self._linear_params:
            val = getattr(self, key)
            if isinstance(val, np.ndarray):
                params[key] = val.tolist()
            elif isinstance(val, (np.floating, np.integer)):
                params[key] = val.item()
            else:
                params[key] = val
        return params

    def set_params(self, params: Dict[str, Any]) -> None:
        """Set model parameters from loaded data."""
        for key in self._linear_params:
            if key not in params:
                continue
            val = params[key]
            if key == 'w' and val is not None:
                setattr(self, key, np.array(val, dtype=float))
            elif key in ('_fitted',):
                setattr(self, key, bool(val))
            else:
                setattr(self, key, val)
        self.loss_history = []


# ============================================================================
# LinearSVM — Binary Linear SVM Classification
# ============================================================================

class LinearSVM(BaseLinearModel):
    """
    Binary Linear SVM using hinge loss and gradient descent.

    Mathematical model:
        f(x) = w · x + b
        y_hat = sign(f(x))

    Loss function (primal form):
        L = (1/n) · Σ max(0, 1 - y_i · f(x_i)) + (λ/2) · ||w||²

    where λ = 1/C is the regularization parameter.

    Hyperparameters:
        C:         Regularization strength (smaller = stronger regularization)
        learning_rate: Step size for gradient descent
        epochs:    Maximum number of training iterations

    Example:
        >>> model = LinearSVM(C=1.0, learning_rate=0.001, epochs=1000)
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)       # Returns 0 or 1
        >>> scores = model.decision_function(X_test)  # Raw scores
    """

    model_type = "linear_svm"

    def __init__(self, C: float = 1.0, learning_rate: float = 0.001,
                 epochs: int = 1000, batch_size: int = 0):
        super().__init__(C=C, learning_rate=learning_rate,
                         epochs=epochs, batch_size=batch_size)
        self.support_vector_labels: Optional[np.ndarray] = None
        self._label_map: Optional[Dict] = None

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute raw decision scores f(x) = w·x + b.

        Positive scores → class +1, negative scores → class -1.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Decision scores (n_samples,)
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained yet! Call fit() first.")
        return X @ self.w + self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict binary class labels {0, 1}.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Predictions (n_samples,) with values 0 or 1
        """
        scores = self.decision_function(X)
        return (scores >= 0).astype(int)

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        """
        Predict in {-1, +1} space (internal label format).

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Predictions (n_samples,) with values -1 or +1
        """
        scores = self.decision_function(X)
        return np.where(scores >= 0, 1.0, -1.0)

    def _compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the full primal objective (hinge + regularization).

        Args:
            X: Feature matrix
            y: Labels in {-1, +1}

        Returns:
            Total loss value
        """
        n = X.shape[0]
        scores = X @ self.w + self.b
        margins = y * scores
        hinge = np.maximum(0, 1 - margins)
        hinge_loss = np.mean(hinge)
        reg_loss = (self.lambda_ / 2.0) * np.sum(self.w ** 2)
        return hinge_loss + reg_loss

    def _compute_gradients(self, X: np.ndarray, y: np.ndarray
                           ) -> Tuple[np.ndarray, float]:
        """
        Compute gradients of the primal objective w.r.t. w and b.

        For hinge loss:
            if y_i · f(x_i) < 1:  ∂L/∂w = -y_i·x_i + λ·w,  ∂L/∂b = -y_i
            else:                 ∂L/∂w = λ·w,              ∂L/∂b = 0

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels in {-1, +1}

        Returns:
            Tuple of (dw, db)
        """
        n = X.shape[0]
        scores = X @ self.w + self.b
        margins = y * scores

        # Identify misclassified / margin-violating points
        violations = (margins < 1).astype(float)

        # dw = (1/n) * Σ(-y_i * x_i * violations_i) + λ * w
        # db = (1/n) * Σ(-y_i * violations_i)
        dw = -(1.0 / n) * (X.T @ (y * violations)) + self.lambda_ * self.w
        db = -(1.0 / n) * np.sum(y * violations)

        return dw, db

    def _identify_support_vectors(self, X: np.ndarray,
                                  y: np.ndarray) -> None:
        """Identify support vectors (points with margin ≤ 1)."""
        scores = X @ self.w + self.b
        margins = y * scores
        sv_mask = margins <= 1.0 + 1e-6
        self.support_vectors = X[sv_mask]
        self.support_vector_labels = y[sv_mask]
        self.n_support_vectors = np.sum(sv_mask)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the Linear SVM using gradient descent.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,) with 2 unique values
        """
        y_internal = _validate_binary_labels(y)
        self._label_map = {'original': np.unique(y), 'svm': np.array([-1, 1])}
        self._shared_fit_loop(X, y_internal, y_internal)

    def fit_with_early_stopping(self, X_train: np.ndarray, y_train: np.ndarray,
                                X_val: np.ndarray, y_val: np.ndarray,
                                patience: int = 50,
                                verbose: bool = False) -> None:
        """
        Train with early stopping to prevent overfitting.

        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            patience: Number of epochs without improvement before stopping
            verbose: Print progress information
        """
        y_train_internal = _validate_binary_labels(y_train)
        y_val_internal = _validate_binary_labels(y_val)
        self._label_map = {'original': np.unique(y_train), 'svm': np.array([-1, 1])}
        self._shared_fit_loop(
            X_train, y_train_internal, y_train_internal,
            early_stopping=True,
            X_val=X_val, y_val_for_loss=y_val_internal,
            patience=patience, verbose=verbose
        )

    def get_params(self) -> Dict[str, Any]:
        """Get all model parameters for saving."""
        params = super().get_params()
        params['_label_map'] = self._label_map
        if self.support_vector_labels is not None:
            params['support_vector_labels'] = self.support_vector_labels.tolist()
        else:
            params['support_vector_labels'] = None
        return params

    def set_params(self, params: Dict[str, Any]) -> None:
        """Set model parameters from loaded data."""
        super().set_params(params)
        self._label_map = params.get('_label_map')
        svl = params.get('support_vector_labels')
        self.support_vector_labels = np.array(svl, dtype=float) if svl is not None else None


# ============================================================================
# BaseKernelModel — shared base for KernelSVM & KernelSVR
# ============================================================================

class BaseKernelModel:
    """
    Base class for kernel SVM / SVR models.

    Provides shared gradient-descent training loop in the RKHS (primal
    formulation with representer theorem), serialisation with full state
    restoration, and early stopping — subclasses only implement the
    loss/gradient logic specific to classification or regression.

    Attributes shared by subclasses:
        beta, b, C, kernel_name, gamma, degree, coef0,
        learning_rate, epochs, loss_history,
        X_train_stored, y_train_stored,
        support_vectors, support_vector_indices, support_vector_labels,
        n_support_vectors, _fitted
    """

    def __init__(self, kernel: str = 'rbf', C: float = 1.0,
                 gamma: float = 1.0, degree: int = 3, coef0: float = 1.0,
                 learning_rate: float = 0.001, epochs: int = 1000,
                 grad_clip: float = 10.0, log_every: int = 5):
        self.kernel_name = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.grad_clip = grad_clip
        self.log_every = log_every

        # Kernel function — reconstructed by set_params(), set in __init__
        self.kernel_fn: Callable = get_kernel(
            kernel, gamma=gamma, degree=degree, coef0=coef0
        )

        # Learned parameters (representer-theorem coefficients)
        self.beta: Optional[np.ndarray] = None
        self.b: float = 0.0
        self.loss_history: List[float] = []

        # Training data (needed for kernel predictions)
        self.X_train_stored: Optional[np.ndarray] = None
        self.y_train_stored: Optional[np.ndarray] = None

        # Support vector info
        self.support_vectors: Optional[np.ndarray] = None
        self.support_vector_indices: Optional[np.ndarray] = None
        self.support_vector_labels: Optional[np.ndarray] = None
        self.n_support_vectors: int = 0

        self._fitted: bool = False

    # -- Properties ---------------------------------------------------------

    @property
    def is_trained(self) -> bool:
        """Check if model has been trained."""
        return self._fitted and self.beta is not None

    @property
    def lambda_(self) -> float:
        """Regularization parameter λ = 1/C."""
        return 1.0 / self.C if self.C > 0 else 0.0

    # -- Kernel utility -----------------------------------------------------

    def _compute_kernel_matrix(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute kernel matrix K(X1, X2)."""
        return self.kernel_fn(X1, X2)

    # -- Abstract hooks (must be overridden) --------------------------------

    def _compute_loss(self) -> float:
        """
        Compute the full primal RKHS objective.

        Uses self.X_train_stored, self.y_train_stored, self.beta, self.b.
        """
        raise NotImplementedError

    def _kernel_gradient(self, f: np.ndarray, y: np.ndarray,
                         n_samples: int) -> Tuple[np.ndarray, float]:
        """
        Compute the natural gradient direction for β and bias gradient.

        Args:
            f:          Model outputs f = K β + b  (n_samples,)
            y:          Targets in internal format (n_samples,)
            n_samples:  Number of samples

        Returns:
            Tuple of (direction_for_beta, grad_b)
        """
        raise NotImplementedError

    def _compute_val_loss(self, f_val: np.ndarray,
                          y_val: np.ndarray) -> float:
        """Compute validation loss (data loss + regularisation)."""
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values. Subclasses must override."""
        raise NotImplementedError

    # -- Shared training loop -----------------------------------------------

    def _shared_kernel_fit_loop(
        self, X: np.ndarray, y: np.ndarray,
        early_stopping: bool = False,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        patience: int = 50,
        verbose: bool = False
    ) -> None:
        """
        Shared gradient-descent loop for kernel models.

        Args:
            X:              Training features
            y:              Training targets (internal format)
            early_stopping: If True, use validation-based early stopping
            X_val, y_val:   Validation data
            patience:       Patience for early stopping
            verbose:        Print progress
        """
        n_samples = X.shape[0]

        self.X_train_stored = X.copy()
        self.y_train_stored = y.copy().astype(float)
        self.beta = np.zeros(n_samples, dtype=float)
        self.b = 0.0
        self.loss_history = []

        # Pre-compute kernel matrices
        if n_samples > 5000:
            warnings.warn(
                f"Kernel model with {n_samples} samples stores O(n²) kernel "
                f"matrices (~{n_samples * n_samples * 8 / 1e6:.0f} MB). "
                f"For large datasets, consider LinearSVM / LinearSVR "
                f"or reduce data size."
            )
        K = self._compute_kernel_matrix(X, X)
        K_val = None
        if early_stopping and X_val is not None:
            K_val = self._compute_kernel_matrix(X_val, X)

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(1, self.epochs + 1):
            # Forward pass
            f = K @ self.beta + self.b

            # Compute natural gradient
            g, grad_b = self._kernel_gradient(f, y, n_samples)

            # Clip to prevent instability
            g = np.clip(g, -self.grad_clip, self.grad_clip)
            grad_b = np.clip(grad_b, -self.grad_clip, self.grad_clip)

            # Preconditioned gradient descent update
            self.beta -= self.learning_rate * g
            self.b -= self.learning_rate * grad_b

            # Track loss periodically
            if epoch % self.log_every == 0 or epoch == 1 or epoch == self.epochs:
                loss = self._compute_loss()
                self.loss_history.append(loss)

            # Early stopping
            if early_stopping and K_val is not None:
                f_val = K_val @ self.beta + self.b
                val_loss = self._compute_val_loss(f_val, y_val)

                if val_loss < best_val_loss - 1e-8:
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

        # Identify support vectors
        self._identify_kernel_support_vectors(K)

        self._fitted = True

    def _identify_kernel_support_vectors(self, K: np.ndarray) -> None:
        """
        Identify support vectors after training.

        Uses β coefficients (sparse) first; falls back to margin-based
        selection if all β are near-zero.
        """
        f_final = K @ self.beta + self.b
        sv_mask = np.abs(self.beta) > 1e-6
        if np.sum(sv_mask) == 0:
            # Fallback: use model-specific logic (implemented via subclass hook)
            sv_mask = self._fallback_support_vector_mask(f_final)

        self.support_vector_indices = np.where(sv_mask)[0]
        self.support_vectors = self.X_train_stored[sv_mask]
        self.support_vector_labels = self.y_train_stored[sv_mask]
        self.n_support_vectors = np.sum(sv_mask)

    def _fallback_support_vector_mask(self, f_final: np.ndarray) -> np.ndarray:
        """
        Fallback strategy for identifying support vectors when all β ≈ 0.

        Override in subclasses for model-specific logic.
        Default: all points are support vectors.
        """
        return np.ones(len(f_final), dtype=bool)

    # -- Public fit / early stopping ----------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the kernel model. Subclasses must override."""
        raise NotImplementedError

    def fit_with_early_stopping(self, X_train: np.ndarray, y_train: np.ndarray,
                                X_val: np.ndarray, y_val: np.ndarray,
                                patience: int = 50,
                                verbose: bool = False) -> None:
        """Train with early stopping. Subclasses must override."""
        raise NotImplementedError

    # -- Serialisation ------------------------------------------------------

    def get_params(self, save_training_data: bool = True) -> Dict[str, Any]:
        """
        Get all model parameters for saving.

        Includes training data by default, so predict() works immediately
        after set_params() without calling set_training_data().

        Args:
            save_training_data: If True (default), includes full training
                matrices X_train_stored and y_train_stored in the output.
                Set to False to reduce memory usage for large datasets;
                you will need to call set_training_data() before predict().
        """
        params = {
            'beta': self.beta.tolist() if self.beta is not None else None,
            'b': float(self.b),
            'C': float(self.C),
            'kernel': self.kernel_name,
            'gamma': float(self.gamma),
            'degree': int(self.degree),
            'coef0': float(self.coef0),
            'learning_rate': float(self.learning_rate),
            'epochs': int(self.epochs),
            'n_support_vectors': int(self.n_support_vectors),
            '_fitted': self._fitted,
            'grad_clip': float(self.grad_clip),
            'log_every': int(self.log_every),

            # Training data — enables immediate predict() after restore
            'X_train_stored': (
                self.X_train_stored.tolist()
                if save_training_data and self.X_train_stored is not None else None
            ),
            'y_train_stored': (
                self.y_train_stored.tolist()
                if save_training_data and self.y_train_stored is not None else None
            ),
        }
        if self.support_vector_labels is not None:
            params['support_vector_labels'] = self.support_vector_labels.tolist()
        if self.support_vector_indices is not None:
            params['support_vector_indices'] = self.support_vector_indices.tolist()
        if not save_training_data and self.X_train_stored is not None:
            warnings.warn(
                "Training data NOT saved with get_params(save_training_data=False). "
                "Call set_training_data(X_train, y_train) before predict()."
            )
        return params

    def set_params(self, params: Dict[str, Any]) -> None:
        """
        Set model parameters from loaded data.

        Restores **full** state including training data, so predict()
        works immediately after set_params().

        For memory-constrained scenarios, set ``X_train_stored = None``
        after set_params() and later restore with set_training_data().
        """
        self.beta = (
            np.array(params['beta'], dtype=float)
            if params.get('beta') is not None else None
        )
        self.b = float(params.get('b', 0.0))
        self.C = float(params.get('C', 1.0))
        self.kernel_name = params.get('kernel', 'rbf')
        self.gamma = float(params.get('gamma', 1.0))
        self.degree = int(params.get('degree', 3))
        self.coef0 = float(params.get('coef0', 1.0))
        self.learning_rate = float(params.get('learning_rate', 0.001))
        self.epochs = int(params.get('epochs', 1000))
        self.n_support_vectors = int(params.get('n_support_vectors', 0))
        self._fitted = bool(params.get('_fitted', False))
        self.loss_history = []

        # Recreate kernel function
        self.kernel_fn = get_kernel(
            self.kernel_name,
            gamma=self.gamma,
            degree=self.degree,
            coef0=self.coef0
        )

        # Restore training data (if saved)
        xt = params.get('X_train_stored')
        self.X_train_stored = np.array(xt, dtype=float) if xt is not None else None
        yt = params.get('y_train_stored')
        self.y_train_stored = np.array(yt, dtype=float) if yt is not None else None

        # Restore support vector info
        sv_labels = params.get('support_vector_labels')
        self.support_vector_labels = (
            np.array(sv_labels, dtype=float) if sv_labels is not None else None
        )
        sv_indices = params.get('support_vector_indices')
        self.support_vector_indices = (
            np.array(sv_indices) if sv_indices is not None else None
        )

        # Restore additional params (newer saves may include them)
        if 'grad_clip' in params:
            self.grad_clip = float(params['grad_clip'])
        if 'log_every' in params:
            self.log_every = int(params['log_every'])

        # Reconstruct support_vectors from indices if possible.
        # IMPORTANT: do NOT overwrite support_vector_labels if already
        # restored from params above — the saved labels may differ from
        # those derived from y_train_stored (e.g., after remapping).
        if self.support_vector_indices is not None and self.X_train_stored is not None:
            self.support_vectors = self.X_train_stored[self.support_vector_indices]
            if self.support_vector_labels is None and self.y_train_stored is not None:
                self.support_vector_labels = self.y_train_stored[self.support_vector_indices]
        elif self.X_train_stored is not None and self.beta is not None:
            # Re-identify from scratch
            K = self._compute_kernel_matrix(self.X_train_stored, self.X_train_stored)
            self._identify_kernel_support_vectors(K)
        else:
            self.support_vectors = None
            self.support_vector_indices = None

    def set_training_data(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Restore training data references (needed for kernel predictions).

        Useful when loading a model where training data was NOT stored in
        get_params() (e.g., for memory efficiency).

        Must be called after set_params() and before predict()
        if training data was not saved with get_params().

        Args:
            X_train: Training feature matrix
            y_train: Training target vector
        """
        self.X_train_stored = X_train.copy()
        self.y_train_stored = y_train.copy().astype(float)

        # Re-identify support vectors from restored beta
        if self.beta is not None:
            K = self._compute_kernel_matrix(self.X_train_stored, self.X_train_stored)
            self._identify_kernel_support_vectors(K)


# ============================================================================
# KernelSVM — Binary Non-Linear SVM Classification with Kernel Trick
# ============================================================================

class KernelSVM(BaseKernelModel):
    """
    Binary Non-Linear SVM using the kernel trick (primal RKHS formulation).

    Mathematical model:
        f(x) = Σ β_j · K(x_j, x) + b
        y_hat = sign(f(x))

    where β_j are the representer-theorem coefficients in the RKHS, and K is
    the kernel function.  The model minimises the **primal** objective
    directly in the RKHS via gradient descent:

        J(β, b) = (1/n) · Σ max(0, 1 - y_i · f(x_i)) + (λ/2) · βᵀ K β

    with λ = 1/C.

    .. note::
        This is NOT a dual SVM solver.  It does **not** enforce the
        constraints of the dual QP problem (0 ≤ α_i ≤ C, Σ α_i y_i = 0).
        Use it as a non-linear extension of the hinge-loss model that works
        well in practice with proper regularisation, but treat it as an
        **RKHS gradient-descent classifier**, not a canonical SVM.

    Kernels supported:
        - 'linear':    K(x,z) = x·z
        - 'rbf':       K(x,z) = exp(-γ · ||x - z||²)
        - 'poly':      K(x,z) = (γ · x·z + coef0)^degree
        - 'sigmoid':   K(x,z) = tanh(γ · x·z + coef0)

    Example:
        >>> model = KernelSVM(kernel='rbf', C=1.0, gamma=0.1)
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
    """

    model_type = "kernel_svm"

    def __init__(self, kernel: str = 'rbf', C: float = 1.0,
                 gamma: float = 1.0, degree: int = 3, coef0: float = 1.0,
                 learning_rate: float = 0.001, epochs: int = 1000,
                 grad_clip: float = 10.0, log_every: int = 5):
        super().__init__(
            kernel=kernel, C=C, gamma=gamma, degree=degree,
            coef0=coef0, learning_rate=learning_rate, epochs=epochs,
            grad_clip=grad_clip, log_every=log_every
        )

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute raw decision scores: f(x) = Σ β_j · K(x_j, x) + b

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Decision scores (n_samples,)
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained yet! Call fit() first.")
        if self.X_train_stored is None:
            raise RuntimeError(
                "Training data not available. If you loaded the model from "
                "params, call set_training_data(X_train, y_train) first."
            )
        K = self._compute_kernel_matrix(X, self.X_train_stored)
        return K @ self.beta + self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict binary class labels {0, 1}.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Predictions (n_samples,) with values 0 or 1
        """
        scores = self.decision_function(X)
        return (scores >= 0).astype(int)

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        """Predict in {-1, +1} space."""
        scores = self.decision_function(X)
        return np.where(scores >= 0, 1.0, -1.0)

    def _compute_loss(self) -> float:
        """
        Hinge loss in RKHS.

        J = (1/n) Σ max(0, 1 - y_i · f_i) + (λ/2) · βᵀ K β
        """
        n = self.X_train_stored.shape[0]
        K = self._compute_kernel_matrix(self.X_train_stored, self.X_train_stored)
        f = K @ self.beta + self.b

        margins = self.y_train_stored * f
        hinge = np.mean(np.maximum(0, 1 - margins))
        reg = (self.lambda_ / 2.0) * (self.beta @ K @ self.beta)
        return hinge + reg

    def _kernel_gradient(self, f: np.ndarray, y: np.ndarray,
                         n_samples: int) -> Tuple[np.ndarray, float]:
        """
        Hinge-loss gradient (natural gradient form).

        ∂J/∂β = - (1/n) · Σ y_i · 𝕀[y_i·f_i < 1] + λ·β
        ∂J/∂b = - (1/n) · Σ y_i · 𝕀[y_i·f_i < 1]
        """
        margins = y * f
        violations = (margins < 1).astype(float)
        g = (-y * violations) / n_samples
        if self.lambda_ > 0:
            g += self.lambda_ * self.beta
        grad_b = -np.mean(y * violations)
        return g, grad_b

    def _compute_val_loss(self, f_val: np.ndarray,
                          y_val: np.ndarray) -> float:
        """Validation loss: hinge + regularisation."""
        margins = y_val * f_val
        hinge = np.mean(np.maximum(0, 1 - margins))
        K_train = self._compute_kernel_matrix(
            self.X_train_stored, self.X_train_stored
        )
        reg = (self.lambda_ / 2.0) * (self.beta @ K_train @ self.beta)
        return hinge + reg

    def _fallback_support_vector_mask(self, f_final: np.ndarray) -> np.ndarray:
        """Margin-based fallback: points with margin ≤ 1 are support vectors."""
        margins = self.y_train_stored * f_final
        return margins <= 1.0 + 1e-6

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train Kernel SVM by minimising the primal RKHS objective with GD.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,) with 2 unique values
        """
        y_internal = _validate_binary_labels(y)
        self._shared_kernel_fit_loop(X, y_internal)

    def fit_with_early_stopping(self, X_train: np.ndarray, y_train: np.ndarray,
                                X_val: np.ndarray, y_val: np.ndarray,
                                patience: int = 50,
                                verbose: bool = False) -> None:
        """
        Train with early stopping to prevent overfitting.

        Kernel SVM is prone to overfitting due to the high capacity of
        kernel spaces — early stopping is especially valuable here.

        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            patience: Number of epochs without improvement before stopping
            verbose: Print progress information
        """
        y_train_internal = _validate_binary_labels(y_train)
        y_val_internal = _validate_binary_labels(y_val)
        self._shared_kernel_fit_loop(
            X_train, y_train_internal,
            early_stopping=True,
            X_val=X_val, y_val=y_val_internal,
            patience=patience, verbose=verbose
        )


# ============================================================================
# OneVsRestSVM — Multiclass SVM (One-vs-Rest)
# ============================================================================

class OneVsRestSVM:
    """
    Multiclass SVM using One-vs-Rest (OvR) strategy.

    Trains K binary classifiers (one per class), where each classifier
    distinguishes one class from the rest.

    Supports both LinearSVM and KernelSVM as base estimators.

    Example:
        >>> base_svm = LinearSVM(C=1.0, learning_rate=0.001, epochs=1000)
        >>> model = OneVsRestSVM(base_estimator=base_svm)
        >>> model.fit(X_train, y_train)  # y can have 3+ classes
        >>> preds = model.predict(X_test)  # Returns class indices 0..K-1
        >>> scores = model.decision_function(X_test)  # (n_samples, n_classes)
    """

    model_type = "ovr_svm"

    def __init__(self, base_estimator=None):
        """
        Initialize One-vs-Rest SVM.

        Args:
            base_estimator: A binary SVM instance (LinearSVM or KernelSVM).
                           If None, defaults to LinearSVM(C=1.0).
        """
        if base_estimator is None:
            self.base_estimator_class = LinearSVM
            self.base_estimator_kwargs = {'C': 1.0}
        else:
            self.base_estimator_class = base_estimator.__class__
            self.base_estimator_kwargs = {
                k: getattr(base_estimator, k)
                for k in ['C', 'learning_rate', 'epochs', 'batch_size',
                          'gamma', 'degree', 'coef0', 'kernel']
                if hasattr(base_estimator, k)
            }

        self.estimators: List = []  # List of trained binary classifiers
        self.classes_: Optional[np.ndarray] = None
        self.n_classes: int = 0
        self._fitted: bool = False

    @property
    def is_trained(self) -> bool:
        return self._fitted and len(self.estimators) > 0

    def _create_estimator(self):
        """Create a fresh binary estimator instance."""
        return self.base_estimator_class(**self.base_estimator_kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train one binary SVM per class (One-vs-Rest).

        For each class k, creates binary labels: y_k = +1 if y == k else -1.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,) with integer labels 0..K-1
        """
        self.classes_ = np.unique(y)
        self.n_classes = len(self.classes_)
        self.estimators = []

        # Ensure labels are 0..K-1
        if not np.array_equal(self.classes_, np.arange(self.n_classes)):
            # Remap
            mapping = {old: new for new, old in enumerate(sorted(self.classes_))}
            y_mapped = np.array([mapping[val] for val in y])
            self._inverse_mapping = {new: old for old, new in mapping.items()}
            y_work = y_mapped
        else:
            self._inverse_mapping = None
            y_work = y.astype(int)

        for k in range(self.n_classes):
            # Create binary labels: class k = +1, rest = -1
            y_binary = np.where(y_work == k, 1, -1)

            estimator = self._create_estimator()
            estimator.fit(X, y_binary)
            self.estimators.append(estimator)

        self._fitted = True

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Get decision scores for each class.

        Returns a matrix of shape (n_samples, n_classes) where
        column k is the score for class k.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Score matrix (n_samples, n_classes)
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained yet!")

        scores = np.zeros((X.shape[0], self.n_classes))
        for k, estimator in enumerate(self.estimators):
            scores[:, k] = estimator.decision_function(X)
        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels (0..K-1) using argmax over decision scores.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Predicted class labels (n_samples,)
        """
        scores = self.decision_function(X)
        predictions = np.argmax(scores, axis=1)

        # Inverse mapping if labels were remapped
        if self._inverse_mapping is not None:
            predictions = np.array([self._inverse_mapping[p] for p in predictions])

        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get pseudo-probabilities via softmax of decision scores.

        Note: These are NOT true probabilities, but normalized scores
        that indicate relative confidence.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Probability-like matrix (n_samples, n_classes), rows sum to 1
        """
        scores = self.decision_function(X)
        # Softmax for interpretability
        scores_shifted = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores_shifted)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def get_params(self) -> Dict[str, Any]:
        """Get all model parameters for saving."""
        estimators_params = []
        for est in self.estimators:
            estimators_params.append(est.get_params())

        return {
            'estimators': estimators_params,
            'n_classes': int(self.n_classes),
            'classes_': self.classes_.tolist() if self.classes_ is not None else None,
            'base_estimator_class': self.base_estimator_class.__name__,
            'base_estimator_kwargs': self.base_estimator_kwargs,
            '_fitted': self._fitted,
        }

    def set_params(self, params: Dict[str, Any]) -> None:
        """Set model parameters from loaded data."""
        self.n_classes = int(params.get('n_classes', 0))
        self._fitted = bool(params.get('_fitted', False))

        if params.get('classes_') is not None:
            self.classes_ = np.array(params['classes_'])
        else:
            self.classes_ = None

        self.base_estimator_kwargs = params.get('base_estimator_kwargs', {'C': 1.0})

        # Restore estimators
        self.estimators = []
        est_params_list = params.get('estimators', [])
        for est_params in est_params_list:
            est = self._create_estimator()
            est.set_params(est_params)
            self.estimators.append(est)

    def fit_with_early_stopping(self, X_train: np.ndarray, y_train: np.ndarray,
                                X_val: np.ndarray, y_val: np.ndarray,
                                patience: int = 50,
                                verbose: bool = False) -> None:
        """
        Train multiple binary SVMs with early stopping (One-vs-Rest).

        For each class k, creates binary labels:
            y_k = +1 if y == k else -1
        and trains a binary estimator with early stopping.

        If the base estimator does not support fit_with_early_stopping,
        falls back to regular fit().

        Args:
            X_train, y_train: Training data
            X_val, y_val:     Validation data
            patience:         Patience for early stopping
            verbose:          Print progress information
        """
        self.classes_ = np.unique(y_train)
        self.n_classes = len(self.classes_)
        self.estimators = []

        # Ensure labels are 0..K-1
        if not np.array_equal(self.classes_, np.arange(self.n_classes)):
            mapping = {old: new for new, old in enumerate(sorted(self.classes_))}
            y_train_mapped = np.array([mapping[val] for val in y_train])
            y_val_mapped = np.array([mapping[val] for val in y_val])
            self._inverse_mapping = {new: old for old, new in mapping.items()}
            y_train_work = y_train_mapped
            y_val_work = y_val_mapped
        else:
            self._inverse_mapping = None
            y_train_work = y_train.astype(int)
            y_val_work = y_val.astype(int)

        for k in range(self.n_classes):
            y_binary_train = np.where(y_train_work == k, 1, -1)
            y_binary_val = np.where(y_val_work == k, 1, -1)

            estimator = self._create_estimator()
            if hasattr(estimator, 'fit_with_early_stopping'):
                estimator.fit_with_early_stopping(
                    X_train, y_binary_train, X_val, y_binary_val,
                    patience=patience, verbose=verbose
                )
            else:
                # Fallback: regular fit without early stopping
                estimator.fit(X_train, y_binary_train)
            self.estimators.append(estimator)

        self._fitted = True


# ============================================================================
# LinearSVR — Linear Support Vector Regression
# ============================================================================

class LinearSVR(BaseLinearModel):
    """
    Linear Support Vector Regression (SVR) using epsilon-insensitive loss.

    Mathematical model:
        f(x) = w · x + b

    Loss function (ε-insensitive):
        L(y, f(x)) = max(0, |y - f(x)| - ε)

    Total objective:
        J = (1/n) · Σ max(0, |y_i - f(x_i)| - ε) + (λ/2) · ||w||²

    Where λ = 1/C is the regularization parameter.

    Example:
        >>> model = LinearSVR(C=1.0, epsilon=0.1, learning_rate=0.001, epochs=1000)
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)  # Continuous values
    """

    model_type = "linear_svr"

    def __init__(self, C: float = 1.0, epsilon: float = 0.1,
                 learning_rate: float = 0.001, epochs: int = 1000,
                 batch_size: int = 0):
        super().__init__(C=C, learning_rate=learning_rate,
                         epochs=epochs, batch_size=batch_size)
        self.epsilon = epsilon

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict continuous target values.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Predictions (n_samples,)
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained yet!")
        return X @ self.w + self.b

    def _compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute the ε-insensitive loss objective."""
        f = X @ self.w + self.b
        residual = np.abs(y - f)
        epsilon_loss = np.mean(np.maximum(0, residual - self.epsilon))
        reg_loss = (self.lambda_ / 2.0) * np.sum(self.w ** 2)
        return epsilon_loss + reg_loss

    def _compute_gradients(self, X: np.ndarray, y: np.ndarray
                           ) -> Tuple[np.ndarray, float]:
        """
        Compute gradients for ε-insensitive loss.

        Let r_i = y_i - f(x_i) be the residual.
        For points outside the ε-tube (|r_i| > ε):
            ∂L/∂w = -sign(r_i) · x_i + λ·w
            ∂L/∂b = -sign(r_i)
        For points inside the tube (|r_i| ≤ ε):
            ∂L/∂w = λ·w
            ∂L/∂b = 0
        """
        n = X.shape[0]
        f = X @ self.w + self.b
        residual = y - f

        # direction = sign(r) for outside points, 0 otherwise
        direction = np.where(residual > self.epsilon, 1.0,
                             np.where(residual < -self.epsilon, -1.0, 0.0))

        dw = -(X.T @ direction) / n + self.lambda_ * self.w
        db = -np.mean(direction)

        return dw, db

    def _identify_support_vectors(self, X: np.ndarray,
                                  y: np.ndarray) -> None:
        """Identify support vectors (outside or on tube boundary)."""
        f = X @ self.w + self.b
        residual = np.abs(y - f)
        sv_mask = residual >= self.epsilon - 1e-6
        self.support_vectors = X[sv_mask]
        self.n_support_vectors = np.sum(sv_mask)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the Linear SVR model using gradient descent.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,) — continuous values
        """
        self._shared_fit_loop(X, y, y)

    def fit_with_early_stopping(self, X_train: np.ndarray, y_train: np.ndarray,
                                X_val: np.ndarray, y_val: np.ndarray,
                                patience: int = 50,
                                verbose: bool = False) -> None:
        """
        Train with early stopping to prevent overfitting.

        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            patience: Number of epochs without improvement before stopping
            verbose: Print progress information
        """
        self._shared_fit_loop(
            X_train, y_train, y_train,
            early_stopping=True,
            X_val=X_val, y_val_for_loss=y_val,
            patience=patience, verbose=verbose
        )


# ============================================================================
# KernelSVR — Non-Linear Support Vector Regression with Kernel Trick
# ============================================================================

class KernelSVR(BaseKernelModel):
    """
    Non-Linear Support Vector Regression using the kernel trick
    (primal RKHS formulation).

    Mathematical model:
        f(x) = Σ β_j · K(x_j, x) + b

    Loss: ε-insensitive in kernel space, solved by minimising the primal
    RKHS objective directly.

    .. note::
        This is NOT a dual SVR solver.  It minimises the primal objective
        in the RKHS using gradient descent on the representer coefficients β.

    Example:
        >>> model = KernelSVR(kernel='rbf', C=1.0, epsilon=0.1, gamma=0.5)
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
    """

    model_type = "kernel_svr"

    def __init__(self, kernel: str = 'rbf', C: float = 1.0, epsilon: float = 0.1,
                 gamma: float = 1.0, degree: int = 3, coef0: float = 1.0,
                 learning_rate: float = 0.001, epochs: int = 1000,
                 grad_clip: float = 10.0, log_every: int = 5):
        super().__init__(
            kernel=kernel, C=C, gamma=gamma, degree=degree,
            coef0=coef0, learning_rate=learning_rate, epochs=epochs,
            grad_clip=grad_clip, log_every=log_every
        )
        self.epsilon = epsilon

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict continuous values: f(X) = K(X, X_train) @ β + b

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Predictions (n_samples,)
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained yet!")
        if self.X_train_stored is None:
            raise RuntimeError(
                "Training data not available. If you loaded the model from "
                "params, call set_training_data(X_train, y_train) first."
            )
        K = self._compute_kernel_matrix(X, self.X_train_stored)
        return K @ self.beta + self.b

    def _compute_loss(self) -> float:
        """
        ε-insensitive loss in RKHS.

        J = (1/n) Σ max(0, |y_i - f_i| - ε) + (λ/2) βᵀ K β
        """
        n = self.X_train_stored.shape[0]
        K = self._compute_kernel_matrix(self.X_train_stored, self.X_train_stored)
        f = K @ self.beta + self.b
        residual = np.abs(self.y_train_stored - f)
        epsilon_loss = np.mean(np.maximum(0, residual - self.epsilon))
        reg = (self.lambda_ / 2.0) * (self.beta @ K @ self.beta)
        return epsilon_loss + reg

    def _kernel_gradient(self, f: np.ndarray, y: np.ndarray,
                         n_samples: int) -> Tuple[np.ndarray, float]:
        """
        ε-insensitive gradient (natural gradient form).

        ∂J/∂β = - (1/n) · Σ direction_i + λ·β
        where direction_i = sign(r_i) for |r_i| > ε, 0 otherwise
        """
        residual = y - f
        direction = np.where(residual > self.epsilon, 1.0,
                             np.where(residual < -self.epsilon, -1.0, 0.0))
        g = -direction / n_samples
        if self.lambda_ > 0:
            g += self.lambda_ * self.beta
        grad_b = -np.mean(direction)
        return g, grad_b

    def _compute_val_loss(self, f_val: np.ndarray,
                          y_val: np.ndarray) -> float:
        """Validation loss: ε-insensitive + regularisation."""
        residual_val = np.abs(y_val - f_val)
        val_eps = np.mean(np.maximum(0, residual_val - self.epsilon))
        K_train = self._compute_kernel_matrix(
            self.X_train_stored, self.X_train_stored
        )
        val_reg = (self.lambda_ / 2.0) * (self.beta @ K_train @ self.beta)
        return val_eps + val_reg

    def _fallback_support_vector_mask(self, f_final: np.ndarray) -> np.ndarray:
        """Tube-boundary fallback: points outside/on ε-tube are support vectors."""
        return np.abs(self.y_train_stored - f_final) >= self.epsilon - 1e-6

    def get_params(self, save_training_data: bool = True) -> Dict[str, Any]:
        """Get all model parameters for saving, including epsilon."""
        params = super().get_params(save_training_data=save_training_data)
        params['epsilon'] = float(self.epsilon)
        return params

    def set_params(self, params: Dict[str, Any]) -> None:
        """Set model parameters from loaded data, including epsilon."""
        super().set_params(params)
        self.epsilon = float(params.get('epsilon', 0.1))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train Kernel SVR by minimising the primal RKHS objective with GD.

        w = Σ β_j · φ(x_j)  (representer theorem).
        The coefficients β are optimised directly.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,) — continuous values
        """
        self._shared_kernel_fit_loop(X, y)

    def fit_with_early_stopping(self, X_train: np.ndarray, y_train: np.ndarray,
                                X_val: np.ndarray, y_val: np.ndarray,
                                patience: int = 50,
                                verbose: bool = False) -> None:
        """
        Train with early stopping to prevent overfitting.

        Kernel SVR is prone to overfitting due to the high capacity of
        kernel spaces — early stopping is especially valuable here.

        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            patience: Number of epochs without improvement before stopping
            verbose: Print progress information
        """
        self._shared_kernel_fit_loop(
            X_train, y_train,
            early_stopping=True,
            X_val=X_val, y_val=y_val,
            patience=patience, verbose=verbose
        )
