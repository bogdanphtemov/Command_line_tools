"""
Shared visualization utilities for all ML models.

Provides reusable plotting functions that are model-agnostic:
    - Training loss curves
    - General scatter plots for true vs predicted
    - Confusion matrices
    - Performance metrics comparisons
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List


def plot_loss_curve(history: List[float], ylabel: str = "Loss", title: str = "Training Loss Curve") -> None:
    """
    Plot training loss curve across epochs.
    
    Universal function for any model (Linear, Logistic, etc.).
    Automatically detects appropriate label based on loss values.
    
    Args:
        history: List of loss values from each epoch
        ylabel: Label for y-axis (e.g., "MSE Loss", "Cross-Entropy Loss")
        title: Title for the plot
    """
    if not history:
        print("No loss history to display")
        return
    
    plt.figure()
    plt.plot(np.arange(1, len(history) + 1), history)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.show()


def plot_true_vs_pred(y_true: np.ndarray, y_pred: np.ndarray, title: str = "True vs Predicted") -> None:
    """
    Plot true vs predicted values with diagonal reference line.
    
    Useful for regression models to visualize prediction accuracy.
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        title: Title for the plot
    """
    plt.figure()
    plt.scatter(y_true, y_pred)
    
    mn = min(float(y_true.min()), float(y_pred.min()))
    mx = max(float(y_true.max()), float(y_pred.max()))
    plt.plot([mn, mx], [mn, mx])
    
    plt.xlabel("y_true")
    plt.ylabel("y_pred")
    plt.title(title)
    plt.grid(True)
    plt.show()
