import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

from .core import LinearRegressionGD
from .preprocessing import standardize_apply
from ...visualization_utils import plot_loss_curve, plot_true_vs_pred

__all__ = ["plot_1d_regression", "plot_loss_curve", "plot_true_vs_pred"]

def plot_1d_regression(
    x_raw: np.ndarray,
    y_true: np.ndarray,
    model: LinearRegressionGD,
    scaler_mean: Optional[np.ndarray],
    scaler_std: Optional[np.ndarray],
) -> plt.Figure:
    """
    Plot a 1D regression: scatter of raw feature values vs target, with the
    model's fitted line overlaid in raw feature coordinates.

    The model is trained on scaled features when standardization was enabled
    during fit, so the line is produced by scaling a raw grid, predicting in
    scaled space, and plotting the predictions against the raw grid values.
    The scatter points are always shown in raw coordinates, so both the points
    and the line live in the same (raw) coordinate system.

    Args:
        x_raw: Raw feature values, 1D (n_samples,) or (n_samples, 1).
            Must contain exactly one feature.
        y_true: True target values, 1D (n_samples,).
        model: Trained LinearRegressionGD instance. The model must be trained
            on exactly one feature.
        scaler_mean: Per-feature mean used during training, or None if scaling
            was disabled.
        scaler_std: Per-feature std used during training, or None if scaling
            was disabled. scaler_mean and scaler_std must be either both provided
            (scaling enabled) or both None (no scaling); providing only one
            raises ValueError. If both are None, the model is assumed to be
            trained on raw (unscaled) features.

    Returns:
        The created matplotlib Figure. The plot is also displayed via plt.show().

    Raises:
        ValueError: If x_raw/y_true have invalid shapes or lengths, or contain
            NaN/inf; if only one of scaler_mean/scaler_std is provided; or if
            the model was not trained on exactly one feature.
        RuntimeError: If model.is_trained is False (untrained model or
            non-finite weights).

    Note:
        If all x_raw values are equal, the fitted line collapses to a single
        point and the plot may be hard to interpret - this is an expected
        property of the data, not an error.
    """
    x_raw = np.asarray(x_raw, dtype=float)
    y_true = np.asarray(y_true, dtype=float)

    if x_raw.ndim == 2 and x_raw.shape[1] == 1:
        x_raw = x_raw.ravel()
    elif x_raw.ndim != 1:
        raise ValueError(
            f"x_raw must be 1D or (n_samples, 1), got shape {x_raw.shape}"
        )

    if y_true.ndim != 1:
        raise ValueError(
            f"y_true must be 1D (n_samples,), got shape {y_true.shape}"
        )
    if x_raw.shape[0] != y_true.shape[0]:
        raise ValueError(
            f"x_raw/y_true sample mismatch: {x_raw.shape[0]} vs {y_true.shape[0]}"
        )
    if x_raw.shape[0] == 0:
        raise ValueError("x_raw/y_true must not be empty")
    if not np.all(np.isfinite(x_raw)) or not np.all(np.isfinite(y_true)):
        raise ValueError("x_raw/y_true contain NaN or infinity")

    if not model.is_trained:
        raise RuntimeError(
            "Cannot plot regression line: model is not trained or has non-finite weights"
        )
    
    # This function only supports single-feature regression models
    if len(model.w) != 1:
        raise ValueError(
            f"This function supports only single-feature regression models. "
            f"The provided model was trained on {len(model.w)} features."
        )

    x_grid = np.linspace(float(x_raw.min()), float(x_raw.max()), 200).reshape(-1, 1)

    # scaler_mean and scaler_std must be either both provided (scaling enabled)
    # or both None (no scaling). A mixed state would silently skip scaling and
    # produce a misleading plot, so it is rejected explicitly.
    if (scaler_mean is None) != (scaler_std is None):
        raise ValueError(
            "scaler_mean and scaler_std must be either both None (no scaling) "
            "or both provided (scaling applied)"
        )

    # The model is trained on scaled features (when scaling is enabled), so the
    # prediction grid must be scaled the same way before calling predict().
    # The scatter and the line are still drawn in raw coordinates.
    if scaler_mean is not None and scaler_std is not None:
        x_grid_scaled = standardize_apply(x_grid, scaler_mean, scaler_std)
    else:
        x_grid_scaled = x_grid

    y_line = model.predict(x_grid_scaled)

    fig = plt.figure(figsize=(10, 6))
    plt.scatter(x_raw.flatten(), y_true)
    plt.plot(x_grid.flatten(), y_line)
    plt.xlabel("Feature (raw)")
    plt.ylabel("Target")
    plt.title("1D Regression: data points + fitted line")
    plt.grid(True)
    plt.show()
    return fig