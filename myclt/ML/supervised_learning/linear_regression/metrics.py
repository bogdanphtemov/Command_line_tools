"""
Evaluation metrics for linear regression: MSE, RMSE and R².

All metrics in this module:
  - accept array-likes (lists, tuples, NumPy arrays) and convert them to float arrays;
  - require both targets to be 1D, non-empty, of equal length and free of NaN/inf
    (same contract as LinearRegressionGD._validate_inputs in core.py);
  - return a plain Python float, so the values are JSON-serializable
    (metrics are persisted in saved sessions via SessionData.metrics).
"""

import warnings
from typing import Sequence, Union

import numpy as np

__all__ = ["mse", "rmse", "r2_score"]

ArrayLike = Union[np.ndarray, Sequence[float]]


def _validate_metric_inputs(y_true: ArrayLike, y_pred: ArrayLike,
                            *, name_true: str = "y_true", name_pred: str = "y_pred"
                            ) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert targets to 1D float arrays and validate them.

    Args:
        y_true: True target values (array-like).
        y_pred: Predicted target values (array-like).
        name_true: Name of the first argument, used in error messages.
        name_pred: Name of the second argument, used in error messages.

    Returns:
        Tuple (y_true, y_pred) of validated 1D float ndarrays.

    Raises:
        ValueError: If an input is not numeric, not 1D, empty, has a different
            length than the other input, or contains NaN/inf.

    Note:
        Column vectors of shape (n, 1) are rejected on purpose: combined with the
        (n,) predictions returned by LinearRegressionGD.predict they would be
        broadcast into an (n, n) matrix and silently produce a wrong metric.
        Call y.reshape(-1) or y.ravel() before passing such arrays.
    """
    try:
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name_true}/{name_pred} must be numeric arrays: {exc}") from exc

    if y_true.ndim != 1:
        raise ValueError(
            f"{name_true} must be 1D (n_samples,), got shape {y_true.shape}. "
            "Use reshape(-1) or ravel() to flatten it."
        )
    if y_pred.ndim != 1:
        raise ValueError(
            f"{name_pred} must be 1D (n_samples,), got shape {y_pred.shape}. "
            "Use reshape(-1) or ravel() to flatten it."
        )
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(
            f"{name_true}/{name_pred} sample mismatch: {y_true.shape[0]} vs {y_pred.shape[0]}"
        )
    if y_true.shape[0] == 0:
        raise ValueError(f"{name_true}/{name_pred} must not be empty")
    if not np.all(np.isfinite(y_true)) or not np.all(np.isfinite(y_pred)):
        raise ValueError(f"{name_true}/{name_pred} contain NaN or infinity")
    return y_true, y_pred


def mse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Mean Squared Error: mean((y_true - y_pred) ** 2).

    Lower is better; errors are squared, so outliers dominate the score.

    Args:
        y_true: True target values, 1D (n_samples,).
        y_pred: Predicted target values, 1D (n_samples,).

    Returns:
        MSE as a Python float (0.0 or greater), in squared units of y.

    Raises:
        ValueError: If inputs are invalid (see _validate_metric_inputs).
    """
    y_true, y_pred = _validate_metric_inputs(y_true, y_pred)
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Root Mean Squared Error: sqrt(MSE).

    Expressed in the same units as the target y, which makes it easier to
    interpret than MSE.

    Args:
        y_true: True target values, 1D (n_samples,).
        y_pred: Predicted target values, 1D (n_samples,).

    Returns:
        RMSE as a Python float (0.0 or greater), in units of y.

    Raises:
        ValueError: If inputs are invalid (see _validate_metric_inputs).
    """
    return float(np.sqrt(mse(y_true, y_pred)))


def r2_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Coefficient of determination: R² = 1 - SS_res / SS_tot.

    Args:
        y_true: True target values, 1D (n_samples,).
        y_pred: Predicted target values, 1D (n_samples,).

    Returns:
        R² as a Python float: 1.0 = perfect fit, 0.0 = no better than
        predicting the mean, negative = worse than the mean.

    Raises:
        ValueError: If inputs are invalid (see _validate_metric_inputs).

    Note:
        Constant y_true (SS_tot == 0) means R² is undefined mathematically.
        This implementation follows scikit-learn: a perfect prediction of a
        constant target returns 1.0; any other prediction returns 0.0 plus a
        UserWarning. For a *nearly* constant y_true SS_tot is tiny but nonzero,
        so R² may be arbitrarily large (positive or negative) - this is a
        property of the formula, not a bug.
    """
    y_true, y_pred = _validate_metric_inputs(y_true, y_pred)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))

    if ss_tot == 0.0:
        if ss_res == 0.0:
            return 1.0
        warnings.warn(
            "R^2 score is not well-defined when y_true is constant and the "
            "prediction is not perfect; returning 0.0.",
            UserWarning,
            stacklevel=2,
        )
        return 0.0

    return float(1.0 - (ss_res / ss_tot))