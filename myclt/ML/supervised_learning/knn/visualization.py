"""
Visualization utilities for K-Nearest Neighbors.
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