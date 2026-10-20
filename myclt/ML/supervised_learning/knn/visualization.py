"""
Visualization utilities for K-Nearest Neighbors.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional

from .core import KNNClassifier, KNNRegressor
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
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 8))
    
    # Plot decision boundary
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
    
    # Plot training points
    scatter = plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, 
                         cmap=plt.cm.RdYlBu, edgecolors='black', s=50)
    
    # Plot test points if provided
    if X_test is not None and y_test is not None:
        plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, 
                   cmap=plt.cm.RdYlBu, edgecolors='black', 
                   marker='s', s=100, alpha=0.7)
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.colorbar(scatter)
    plt.grid(True, alpha=0.3)
    
    if show:
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
    if X_train.shape[1] != 1:
        print("Regression comparison plot optimized for 1D data, showing True vs Predicted instead")
        # Always show true vs predicted for multi-dimensional data
        if X_test is not None and y_test is not None:
            y_pred = model.predict(X_test)
            plot_true_vs_pred(y_test, y_pred, 
                             title="KNN: True vs Predicted (Test Set)", 
                             show=False)  # Don't show now, control from outer function
        else:
            y_pred = model.predict(X_train)
            plot_true_vs_pred(y_train, y_pred,
                             title="KNN: True vs Predicted (Training Set)",
                             show=False)  # Don't show now, control from outer function
        
        if show:
            plt.show()
        return
    
    # Plot for 1D case
    plt.figure(figsize=(10, 6))
    
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
        plt.scatter(X_test.flatten(), y_test, color='green', 
                   alpha=0.7, marker='s', s=80, label='Test Data')
    
    plt.xlabel('Feature')
    plt.ylabel('Target')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if show:
        plt.show()


def plot_nearest_neighbors_analysis(
    model: KNNClassifier,
    test_point: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    title: str = "Nearest Neighbors Analysis",
    show: bool = True
) -> None:
    """
    Visualize k nearest neighbors for a test point (2D only).
    
    Args:
        model: Trained KNN classifier
        test_point: Single test point to analyze
        X_train: Training features
        y_train: Training labels
        title: Plot title
    """
    if X_train.shape[1] != 2:
        print("Nearest neighbors visualization requires 2D data")
        return
    
    # Get k nearest neighbor indices using the optimized method
    k_indices = model._get_k_nearest_indices(test_point)
    
    if len(k_indices) == 0:
        print("!Could not find nearest neighbors")
        return
    
    plt.figure(figsize=(10, 8))
    
    # Plot all training points
    scatter = plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train,
                         cmap=plt.cm.Set1, alpha=0.3, s=50)
    
    # Get nearest neighbors data directly using indices (O(1) complexity)
    nearest_points = X_train[k_indices]
    nearest_labels = y_train[k_indices]
    
    # Highlight nearest neighbors
    plt.scatter(nearest_points[:, 0], nearest_points[:, 1],
               c=nearest_labels, cmap=plt.cm.Set1, 
               edgecolors='black', s=150, marker='o')
    
    # Mark the test point
    plt.scatter(test_point[0], test_point[1], 
               color='yellow', marker='*', s=300, 
               edgecolors='black', label='Test Point')
    
    # Draw circles around neighbors
    for point in nearest_points:
        circle = plt.Circle((point[0], point[1]), 0.2, 
                           color='gray', fill=False, linestyle='--', alpha=0.7)
        plt.gca().add_patch(circle)
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    actual_k = len(k_indices)
    plt.title(f"{title} (k={model.k}, showing {actual_k})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if show:
        plt.show()


def plot_accuracy_vs_k(
    k_values: List[int],
    accuracies: List[float],
    title: str = "KNN Accuracy vs k Value",
    show: bool = True
) -> None:
    """
    Plot accuracy as a function of k value.
    
    Args:
        k_values: List of k values tested
        accuracies: Corresponding accuracy values
        title: Plot title
    """
    # Check for empty data
    if not k_values or not accuracies or len(k_values) != len(accuracies):
        print("!Invalid data: k_values and accuracies must be non-empty lists of same length")
        return
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(k_values, accuracies, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('k Value')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    # Mark the best k
    try:
        best_idx = np.argmax(accuracies)
        best_k = k_values[best_idx]
        best_acc = accuracies[best_idx]
        
        plt.axvline(x=best_k, color='red', linestyle='--', alpha=0.7,
                    label=f'Best k={best_k} (Acc={best_acc:.3f})')
        plt.legend()
    except (ValueError, IndexError) as e:
        print(f"!Could not identify best k: {e}")
    
    if show:
        plt.show()