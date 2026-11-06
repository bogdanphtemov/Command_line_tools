"""
Visualization utilities for K-Nearest Neighbors.

Contains functions for plotting KNN classification boundaries, 
confusion matrices, probability distributions, and regression results.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional

from .classifier import KNNClassifier
from .regressor import KNNRegressor
from ...visualization_utils import plot_true_vs_pred


def plot_classification_boundary_2d(
    model: KNNClassifier, 
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    title: str = "KNN Classification Boundary",
    show: bool = True
) -> None:
    """
    Plot 2D classification boundary for KNN (works only with 2 features).
    
    Args:
        model: Trained KNN classifier
        X_train: Training features (2D)
        y_train: Training labels
        X_test: Test features (optional)
        y_test: Test labels (optional)
        title: Plot title
    """
    if X_train.shape[1] != 2:
        print("2D classification boundary plot requires exactly 2 features")
        return
    
    # Create mesh grid
    x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
    y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5
    
    # Optimized grid size for performance (50x50 instead of 100x100)
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                         np.linspace(y_min, y_max, 50))
    
    # Handle string labels for contourf compatibility
    unique_labels = np.unique(y_train)
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    
    # Convert training labels to numeric if they are strings
    if y_train.dtype.kind in ['U', 'S']:  # Unicode or bytes string
        y_train_numeric = np.array([label_to_int[label] for label in y_train])
        
        # Predict and convert predictions to numeric for contourf
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z_numeric = np.array([label_to_int[label] for label in Z]).reshape(xx.shape)
    else:
        # For numeric labels
        y_train_numeric = y_train
        # Predict and ensure numeric type
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        # Use safe conversion to avoid string conversion issues
        if Z.dtype.kind in ['U', 'S']:  # If model returns string labels
            Z_numeric = np.array([label_to_int[label] for label in Z]).reshape(xx.shape)
        else:
            # For numeric predictions, use directly with float conversion
            Z_numeric = Z.reshape(xx.shape).astype(float)
    
    plt.figure(figsize=(10, 8))
    
    # Plot decision boundary
    plt.contourf(xx, yy, Z_numeric, alpha=0.8, cmap=plt.cm.RdYlBu)
    
    # Plot training points
    scatter = plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train_numeric, 
                         cmap=plt.cm.RdYlBu, edgecolors='black', s=50)
    
    # Plot test points if provided
    if X_test is not None and y_test is not None:
        if y_test.dtype.kind in ['U', 'S']:  # Unicode or bytes string
            y_test_numeric = np.array([label_to_int[label] for label in y_test])
        else:
            y_test_numeric = y_test
        plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test_numeric, 
                   cmap=plt.cm.RdYlBu, edgecolors='black', 
                   marker='s', s=100, alpha=0.7)
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    
    # Create colorbar with proper labels
    cbar = plt.colorbar(scatter)
    if y_train.dtype.kind in ['U', 'S']:  # Show actual labels instead of numbers
        cbar.set_ticks(range(len(unique_labels)))
        cbar.set_ticklabels(unique_labels)
    
    plt.grid(True, alpha=0.3)
    
    if show:
        plt.show()
    elif not show:
        # Clean up figure if not showing to prevent memory leaks
        plt.close()


def plot_confusion_matrix(cm: np.ndarray, 
                           class_names: Optional[List[str]] = None,
                           title: str = "KNN Confusion Matrix") -> None:
    """
    Plot confusion matrix heatmap for KNN classification.
    
    Args:
        cm: Confusion matrix (n_classes, n_classes)
        class_names: Optional list of class names
        title: Plot title
    """
    cm = np.asarray(cm)
    n = cm.shape[0]
    
    if class_names is None:
        class_names = [str(i) for i in range(n)]
    elif len(class_names) != n:
        print(f"Warning: {len(class_names)} class names but matrix has {n} classes. Using default names.")
        class_names = [str(i) for i in range(n)]
    
    fig, ax = plt.subplots(figsize=(max(7, n*1.3), max(6, n*1.1)))
    im = ax.imshow(cm, cmap=plt.cm.Blues, aspect='auto')
    
    # Add text annotations
    threshold = cm.max() / 2 if cm.max() > 0 else 0
    for i in range(n):
        for j in range(n):
            color = "white" if cm[i, j] > threshold else "black"
            ax.text(j, i, str(int(cm[i, j])), 
                   ha="center", va="center", 
                   color=color, fontsize=12, fontweight='bold')
    
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f"Pred {c}" for c in class_names], rotation=45, ha='right')
    ax.set_yticklabels([f"True {c}" for c in class_names])
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Count", fontsize=11)
    
    plt.tight_layout()
    plt.show()


def plot_class_probability_distributions(probabilities: np.ndarray, 
                                        y_true: np.ndarray,
                                        class_names: Optional[List[str]] = None,
                                        title: str = "KNN Class Probability Distributions") -> None:
    """
    Plot probability distributions for each class in KNN classification.
    
    Shows histograms of predicted probabilities for samples where the 
    true label equals each class.
    
    Args:
        probabilities: Predicted probabilities (n_samples, n_classes)
        y_true: True labels (n_samples,)
        class_names: Optional list of class names
        title: Plot title
    """
    if probabilities.ndim != 2:
        print("Error: probabilities must be 2D array (n_samples, n_classes)")
        return
    
    n_classes = probabilities.shape[1]
    
    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]
    elif len(class_names) != n_classes:
        print(f"Warning: {len(class_names)} class names but {n_classes} classes. Using default names.")
        class_names = [f"Class {i}" for i in range(n_classes)]
    
    # Determine grid layout
    n_cols = min(3, n_classes)
    n_rows = (n_classes + n_cols - 1) // n_cols  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_classes > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    # Plot distribution for each class
    for k in range(n_classes):
        ax = axes[k]
        # Get probabilities for samples where true label is class k
        mask = y_true == k
        class_probs = probabilities[mask, k]
        
        if len(class_probs) == 0:
            ax.text(0.5, 0.5, f'No samples\nfor {class_names[k]}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        else:
            ax.hist(class_probs, bins=20, alpha=0.7, color=f'C{k}', edgecolor='black')
            ax.axvline(np.mean(class_probs), color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {np.mean(class_probs):.3f}')
            ax.axvline(0.5, color='gray', linestyle=':', linewidth=1.5, 
                      label='Threshold 0.5')
            ax.set_xlim(0, 1)
            ax.legend(fontsize=9)
        
        ax.set_xlabel("Predicted Probability", fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)
        ax.set_title(f"{class_names[k]} (n={len(class_probs)})", fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for k in range(n_classes, len(axes)):
        axes[k].set_visible(False)
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()


def plot_regression_comparison(
    model: KNNRegressor,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    title: str = "KNN Regression Comparison",
    show: bool = True
) -> None:
    """
    Plot regression results comparison (works with 1 feature for clear visualization).
    
    Args:
        model: Trained KNN regressor
        X_train: Training features 
        y_train: Training values
        X_test: Test features (optional)
        y_test: Test values (optional)
        title: Plot title
    """
    fig = None
    
    if X_train.shape[1] != 1:
        print("Regression comparison plot optimized for 1D data, showing True vs Predicted instead")
        # Always show true vs predicted for multi-dimensional data
        if X_test is not None and y_test is not None:
            # Additional check for consistency
            if X_test.shape[1] != X_train.shape[1]:
                print(f"Error: X_train has {X_train.shape[1]} features, but X_test has {X_test.shape[1]} features")
                return
            y_pred = model.predict(X_test)
            plot_true_vs_pred(y_test, y_pred, 
                             title="KNN: True vs Predicted (Test Set)")
            fig = plt.gcf()  # Get current figure for cleanup
        else:
            y_pred = model.predict(X_train)
            plot_true_vs_pred(y_train, y_pred,
                             title="KNN: True vs Predicted (Training Set)")
            fig = plt.gcf()  # Get current figure for cleanup
        
        if show:
            plt.show()
        elif not show and fig is not None:
            plt.close(fig)  # Clean up figure if not showing to prevent memory leaks
        return
    
    # Additional check for test data consistency
    if X_test is not None and y_test is not None:
        if X_test.shape[1] != 1:
            print("Warning: X_test should have 1 feature for consistent visualization")
            # Try to use first feature column if needed
            if X_test.shape[1] > 1:
                X_test = X_test[:, 0].reshape(-1, 1)
    
    # Plot for 1D case
    fig = plt.figure(figsize=(10, 6))
    
    # Plot training data
    plt.scatter(X_train.flatten(), y_train, color='blue', 
               alpha=0.7, label='Training Data')
    
    # Create prediction line - extract first column for 1D case
    x_min, x_max = X_train[:, 0].min(), X_train[:, 0].max()
    x_line = np.linspace(x_min, x_max, 100).reshape(-1, 1)
    y_line = model.predict(x_line)
    
    plt.plot(x_line.flatten(), y_line, color='red', 
            linewidth=2, label='KNN Prediction')
    
    # Plot test data if available
    if X_test is not None and y_test is not None:
        # Ensure correct dimensionality for flattening
        if X_test.shape[1] != 1:
            X_test_flat = X_test[:, 0].flatten()  # Use first column
        else:
            X_test_flat = X_test.flatten()
        plt.scatter(X_test_flat, y_test, color='green', 
                   alpha=0.7, marker='s', s=80, label='Test Data')
    
    plt.xlabel('Feature')
    plt.ylabel('Target')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if show:
        plt.show()
    elif not show:
        plt.close(fig)  # Clean up figure if not showing to prevent memory leaks


# Define public API
__all__ = [
    'plot_classification_boundary_2d',
    'plot_regression_comparison',
    'plot_confusion_matrix',
    'plot_class_probability_distributions',
    'plot_true_vs_pred'
]