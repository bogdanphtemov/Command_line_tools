"""
Visualization utilities for SVM models.

Provides:
    - Training loss curves
    - Decision boundary plots (2D)
    - Confusion matrices (binary + multiclass)
    - Support vector visualization
    - True vs Predicted (for regression)
    - Regression residual plots

Example:
    >>> from myclt.ML.supervised_learning.svm.visualization import (
    ...     plot_loss_curve, plot_svm_decision_boundary_2d
    ... )
    >>> plot_loss_curve(model.loss_history)
    >>> plot_svm_decision_boundary_2d(model, X_test, y_test)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List


# ============================================================================
# Loss Curve
# ============================================================================

def plot_loss_curve(history: List[float], ylabel: str = "Loss",
                    title: str = "SVM Training Loss Curve") -> None:
    """
    Plot training loss curve across epochs.

    Args:
        history: List of loss values from each epoch/checkpoint
        ylabel: Label for y-axis
        title: Title for the plot
    """
    if not history:
        print("No loss history to display")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, len(history) + 1), history, 'b-', linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================================
# Decision Boundary (2D)
# ============================================================================

def plot_svm_decision_boundary_2d(model, X: np.ndarray, y: np.ndarray,
                                   title: str = "SVM Decision Boundary",
                                   feature_names: Optional[List[str]] = None,
                                   show_support_vectors: bool = True,
                                   resolution: int = 100,
                                   cmap: str = None) -> None:
    """
    Plot 2D decision boundary for trained SVM model.

    Only works with 2 features (for visualization purposes).

    Args:
        model: Trained SVM model with decision_function() and predict() methods
        X: Feature matrix (n_samples, 2) — must have exactly 2 features
        y: Target labels (0 or 1 for classification)
        title: Plot title
        feature_names: Names of the two features
        show_support_vectors: Highlight support vectors
        resolution: Grid resolution for contour plot
        cmap: Colormap for the contour (auto-selected if None)
    """
    if X.shape[1] != 2:
        print("✗ Decision boundary plot requires exactly 2 features")
        return

    if feature_names is None:
        feature_names = ["Feature 1", "Feature 2"]

    # Auto-select colormap based on number of classes
    unique_labels = np.unique(y)
    n_classes = len(unique_labels)
    if cmap is None:
        cmap = "coolwarm" if n_classes <= 2 else "tab10"

    # Create mesh grid
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )

    # Predict on mesh
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    try:
        scores = model.decision_function(grid_points).reshape(xx.shape)
        Z = model.predict(grid_points).reshape(xx.shape)
    except Exception as e:
        print(f"✗ Could not compute decision boundary: {e}")
        return

    plt.figure(figsize=(10, 8))

    # Plot decision boundary and margins
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap, levels=np.arange(-0.5, n_classes + 0.5, 1))
    if n_classes == 2:
        plt.contour(xx, yy, scores, levels=[-1, 0, 1], colors='k',
                    linestyles=['--', '-', '--'], linewidths=[1, 2, 1])
        plt.contourf(xx, yy, scores, levels=[-1, 0, 1], alpha=0.1, colors=['blue', 'red'])

    # Plot data points
    colors = ['blue', 'red'] if n_classes <= 2 else [f"C{i}" for i in range(n_classes)]
    markers = ['o', 's', '^', 'D', 'v']

    for i, label in enumerate(unique_labels):
        mask = y == label
        plt.scatter(
            X[mask, 0], X[mask, 1],
            c=colors[i % len(colors)],
            marker=markers[i % len(markers)],
            label=f"Class {int(label)}",
            edgecolors='k',
            s=60,
            alpha=0.8
        )

    # Highlight support vectors
    if show_support_vectors and hasattr(model, 'support_vectors') and model.support_vectors is not None:
        sv = model.support_vectors
        plt.scatter(
            sv[:, 0], sv[:, 1],
            facecolors='none',
            edgecolors='green',
            s=200,
            linewidths=2,
            label='Support Vectors'
        )

    plt.xlabel(feature_names[0], fontsize=12)
    plt.ylabel(feature_names[1], fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()


# ============================================================================
# Confusion Matrix
# ============================================================================

def plot_confusion_matrix(cm: np.ndarray,
                          class_names: Optional[List[str]] = None,
                          title: str = "Confusion Matrix") -> None:
    """
    Plot a confusion matrix as a heatmap.

    Args:
        cm: Confusion matrix (n_classes, n_classes)
        class_names: Optional class names for axis labels
        title: Plot title
    """
    n_classes = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title(title, fontsize=14)
    plt.colorbar(shrink=0.8)

    tick_marks = np.arange(n_classes)
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)

    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(n_classes):
        for j in range(n_classes):
            plt.text(j, i, str(cm[i, j]),
                     ha='center', va='center',
                     color='white' if cm[i, j] > thresh else 'black',
                     fontsize=12)

    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.show()


# ============================================================================
# True vs Predicted (Regression)
# ============================================================================

def plot_true_vs_pred(y_true: np.ndarray, y_pred: np.ndarray,
                      title: str = "True vs Predicted (SVR)") -> None:
    """
    Plot true vs predicted values with diagonal reference line.

    Args:
        y_true: True target values
        y_pred: Predicted values
        title: Plot title
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolors='k', s=50)

    mn = min(float(y_true.min()), float(y_pred.min()))
    mx = max(float(y_true.max()), float(y_pred.max()))
    plt.plot([mn, mx], [mn, mx], 'r--', linewidth=2, label='Perfect Prediction')

    plt.xlabel("y_true", fontsize=12)
    plt.ylabel("y_pred", fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================================
# Residual Plot (Regression)
# ============================================================================

def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray,
                   title: str = "Residual Plot") -> None:
    """
    Plot residuals (y_true - y_pred) against predicted values.

    Args:
        y_true: True target values
        y_pred: Predicted values
        title: Plot title
    """
    residuals = y_true - y_pred

    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.6, edgecolors='k', s=50)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)

    plt.xlabel("Predicted Values", fontsize=12)
    plt.ylabel("Residuals (y_true - y_pred)", fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================================
# Epsilon Tube Visualization (SVR)
# ============================================================================

def plot_svr_tube(X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray,
                  epsilon: float, feature_idx: int = 0,
                  title: str = "SVR: ε-Insensitive Tube") -> None:
    """
    Plot SVR predictions with ε-insensitive tube.

    Shows the ε tube around predictions and highlights points outside it.

    Args:
        X: Feature matrix (n_samples, n_features)
        y_true: True target values
        y_pred: Predicted values
        epsilon: Width of ε-insensitive tube
        feature_idx: Which feature to use for x-axis
        title: Plot title
    """
    plt.figure(figsize=(10, 6))

    x_vals = X[:, feature_idx]

    # Sort by x for clean plotting
    sort_idx = np.argsort(x_vals)
    x_sorted = x_vals[sort_idx]
    y_pred_sorted = y_pred[sort_idx]

    plt.plot(x_sorted, y_pred_sorted, 'b-', linewidth=2, label='Prediction')
    plt.fill_between(x_sorted,
                     y_pred_sorted - epsilon,
                     y_pred_sorted + epsilon,
                     alpha=0.2, color='blue', label=f'ε-tube (±{epsilon})')

    # Highlight outside-tube points
    residuals = np.abs(y_true - y_pred)
    inside_mask = residuals <= epsilon
    outside_mask = ~inside_mask

    plt.scatter(x_vals[inside_mask], y_true[inside_mask],
                c='green', marker='o', s=50, alpha=0.6,
                label='Inside tube')
    plt.scatter(x_vals[outside_mask], y_true[outside_mask],
                c='red', marker='x', s=80, alpha=0.8,
                label='Outside tube (support vectors)')

    plt.xlabel(f"Feature {feature_idx}", fontsize=12)
    plt.ylabel("Target", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================================
# Support Vector Count
# ============================================================================

def plot_support_vector_info(model, title: str = "Support Vectors") -> None:
    """
    Display information about support vectors.

    For OneVsRestSVM, shows per-class support vector counts.

    Args:
        model: Trained SVM model
        title: Display title
    """
    if hasattr(model, 'estimators'):
        # OneVsRestSVM
        print(f"\n{'=' * 50}")
        print(f"  {title.upper()}")
        print(f"{'=' * 50}")
        total_sv = 0
        for k, est in enumerate(model.estimators):
            n_sv = getattr(est, 'n_support_vectors', 0)
            print(f"  Class {k} vs Rest: {n_sv} support vectors")
            total_sv += n_sv
        print(f"  Total: {total_sv} support vectors")
        print(f"{'=' * 50}\n")
    elif hasattr(model, 'n_support_vectors'):
        print(f"\n{'=' * 50}")
        print(f"  {title.upper()}")
        print(f"{'=' * 50}")
        print(f"  Support vectors: {model.n_support_vectors}")
        # Determine training data attribute (X_train_stored for kernel, X_train for some others)
        X_train_attr = None
        if hasattr(model, 'X_train_stored') and model.X_train_stored is not None:
            X_train_attr = model.X_train_stored
        elif hasattr(model, 'X_train') and model.X_train is not None:
            X_train_attr = model.X_train
        if X_train_attr is not None:
            try:
                frac = model.n_support_vectors / X_train_attr.shape[0] * 100
                print(f"  Fraction of training data: {frac:.1f}%")
            except (AttributeError, TypeError):
                pass
        print(f"{'=' * 50}\n")
