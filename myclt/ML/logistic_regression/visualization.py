"""
Visualization utilities for Logistic Regression results.

Uses matplotlib for professional visualizations:
    - Training loss history (shared utility)
    - Confusion matrix heatmap
    - Feature importance (coefficients)
    - Probability distribution (by class)
    - Performance metrics comparison
    - ROC curve
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple
from matplotlib.patches import Rectangle

from .preprocessing import standardize_apply
from ..visualization_utils import plot_loss_curve


def plot_confusion_matrix_heatmap(tp: int, fp: int, fn: int, tn: int) -> None:
    """
    Plot confusion matrix as a heatmap.
    
    Args:
        tp: True Positives
        fp: False Positives
        fn: False Negatives
        tn: True Negatives
    """
    # Create confusion matrix array
    cm = np.array([[tn, fp], 
                   [fn, tp]])
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Plot heatmap
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues, aspect='auto')
    
    # Add text annotations
    threshold = cm.max() / 2
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > threshold else "black"
            ax.text(j, i, str(cm[i, j]), 
                   ha="center", va="center", color=color, fontsize=16, fontweight='bold')
    
    # Labels
    class_labels = ['Negative (0)', 'Positive (1)']
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicted 0', 'Predicted 1'])
    ax.set_yticklabels(['Actual 0', 'Actual 1'])
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Count", fontsize=11)
    
    plt.tight_layout()
    plt.show()


def plot_feature_coefficients(feature_names: list, coefficients: np.ndarray) -> None:
    """
    Plot feature coefficients (weights) as horizontal bar chart.
    
    Positive coefficients push prediction toward 1 (positive class).
    Negative coefficients push prediction toward 0 (negative class).
    
    Args:
        feature_names: Names of features
        coefficients: Model weights for each feature
    """
    if coefficients is None:
        print("No model trained yet")
        return
    
    # Sort by absolute value
    sorted_indices = np.argsort(np.abs(coefficients))
    sorted_names = [feature_names[i] for i in sorted_indices]
    sorted_coefs = coefficients[sorted_indices]
    
    # Color based on sign
    colors = ['#E63946' if c < 0 else '#06A77D' for c in sorted_coefs]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    y_pos = np.arange(len(sorted_names))
    ax.barh(y_pos, sorted_coefs, color=colors, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names)
    ax.set_xlabel("Coefficient Value", fontsize=12)
    ax.set_title("Feature Coefficients (Importance)", fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, v in enumerate(sorted_coefs):
        ax.text(v + 0.005 if v > 0 else v - 0.005, i, f'{v:.4f}', 
               va='center', ha='left' if v > 0 else 'right', fontsize=10)
    
    plt.tight_layout()
    plt.show()


def plot_probability_distribution(probabilities: np.ndarray, y_true: Optional[np.ndarray] = None) -> None:
    """
    Plot histogram of predicted probabilities.
    
    If y_true provided: shows separate distributions for each class.
    Otherwise: shows overall probability distribution.
    
    Args:
        probabilities: Array of predicted probabilities [0, 1]
        y_true: Optional true labels for class-separated analysis
    """
    if y_true is None:
        # Overall distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(probabilities, bins=30, color='#2E86AB', alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(probabilities), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(probabilities):.3f}')
        ax.axvline(np.median(probabilities), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(probabilities):.3f}')
        ax.set_xlabel("Predicted Probability", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title("Probability Distribution (All Predictions)", fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    else:
        # Separate distributions for each class
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        colors = ['#E63946', '#06A77D']
        
        for class_val, ax, color in zip([0, 1], axes, colors):
            class_mask = y_true == class_val
            class_probs = probabilities[class_mask]
            
            if len(class_probs) == 0:
                ax.text(0.5, 0.5, f'No samples for class {class_val}', 
                       ha='center', va='center', transform=ax.transAxes)
                continue
            
            ax.hist(class_probs, bins=30, color=color, alpha=0.7, edgecolor='black')
            ax.axvline(np.mean(class_probs), color='black', linestyle='--', linewidth=2, 
                      label=f'Mean: {np.mean(class_probs):.3f}')
            ax.axvline(0.5, color='gray', linestyle=':', linewidth=1.5, label='Threshold: 0.5')
            
            ax.set_xlabel("Predicted Probability", fontsize=11)
            ax.set_ylabel("Frequency", fontsize=11)
            ax.set_title(f"Class {int(class_val)} (n={len(class_probs)})", fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle("Probability Distribution by True Class", fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()


def plot_metrics_comparison(accuracy: float, precision: float, recall: float, f1: float) -> None:
    """
    Plot performance metrics as bar chart.
    
    Args:
        accuracy: Accuracy score [0, 1]
        precision: Precision score [0, 1]
        recall: Recall score [0, 1]
        f1: F1 score [0, 1]
    """
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    scores = [accuracy, precision, recall, f1]
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(metrics, scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{score:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylim([0, 1.1])
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Classification Performance Metrics", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add reference line at 0.5
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Baseline (0.5)')
    ax.axhline(y=1.0, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Perfect (1.0)')
    ax.legend()
    
    plt.tight_layout()
    plt.show()


def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray) -> None:
    """
    Plot ROC curve for binary classification.
    
    Computes ROC curve by sweeping through different probability thresholds
    and calculating True Positive Rate (TPR) and False Positive Rate (FPR).
    
    Args:
        y_true: True binary labels (0 or 1)
        y_proba: Predicted probabilities [0, 1]
    """
    # Sort by probability in descending order
    sorted_indices = np.argsort(-y_proba)
    y_sorted = y_true[sorted_indices]
    
    # Calculate positive and negative totals
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    
    if n_pos == 0 or n_neg == 0:
        print("ROC curve requires both positive and negative samples")
        return
    
    # Initialize arrays to store FPR and TPR
    fpr_list = [0.0]  # Start at threshold = infinity (no predictions)
    tpr_list = [0.0]
    
    tp = 0
    fp = 0
    
    # Sweep through thresholds
    for i, idx in enumerate(sorted_indices):
        if y_sorted[i] == 1:
            tp += 1
        else:
            fp += 1
        
        # Calculate TPR and FPR at this threshold
        tpr = tp / n_pos if n_pos > 0 else 0
        fpr = fp / n_neg if n_neg > 0 else 0
        
        fpr_list.append(fpr)
        tpr_list.append(tpr)
    
    fpr_list = np.array(fpr_list)
    tpr_list = np.array(tpr_list)
    
    # Calculate AUC using trapezoidal rule
    auc = 0.0
    for i in range(1, len(fpr_list)):
        # Area of trapezoid between points
        auc += (fpr_list[i] - fpr_list[i-1]) * (tpr_list[i] + tpr_list[i-1]) / 2.0
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot ROC curve
    ax.plot(fpr_list, tpr_list, color='#2E86AB', lw=2.5, label=f'ROC Curve (AUC = {auc:.4f})')
    
    # Plot diagonal (random classifier)
    ax.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random Classifier (AUC = 0.5)')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve", fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_1d_logistic_regression(x_raw: np.ndarray, y_true: np.ndarray, 
                                 model, 
                                 scaler_mean: Optional[np.ndarray] = None, 
                                 scaler_std: Optional[np.ndarray] = None) -> None:
    """
    Plot 1D logistic regression (for single feature).
    
    Shows:
        - Scatter of raw x vs y (with jitter for visibility)
        - Logistic curve fitted by model
    
    Args:
        x_raw: Raw feature (n_samples, 1) or (n_samples,)
        y_true: True binary labels (0 or 1)
        model: Trained LogisticRegressionGD model
        scaler_mean: Optional scaling mean
        scaler_std: Optional scaling std
    """
    # Reshape if needed
    if x_raw.ndim == 1:
        x_raw = x_raw.reshape(-1, 1)
    
    x_min, x_max = float(x_raw.min()), float(x_raw.max())
    
    # Create grid for smooth curve
    x_grid = np.linspace(x_min, x_max, 300).reshape(-1, 1)
    
    # Apply scaling if needed
    if scaler_mean is not None and scaler_std is not None:
        x_grid_scaled = standardize_apply(x_grid, scaler_mean, scaler_std)
    else:
        x_grid_scaled = x_grid
    
    # Get predictions
    y_proba = model.predict_proba(x_grid_scaled)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Add jitter to y for better visibility
    jitter = np.random.normal(0, 0.02, size=len(y_true))
    y_jittered = y_true + jitter
    
    # Plot data points by class
    mask_0 = y_true == 0
    mask_1 = y_true == 1
    ax.scatter(x_raw[mask_0], y_jittered[mask_0], alpha=0.5, s=50, color='#E63946', label='Class 0')
    ax.scatter(x_raw[mask_1], y_jittered[mask_1], alpha=0.5, s=50, color='#06A77D', label='Class 1')
    
    # Plot logistic curve
    ax.plot(x_grid, y_proba, color='#2E86AB', linewidth=2.5, label='Logistic Curve')
    
    # Plot decision boundary (threshold = 0.5)
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Decision Boundary (0.5)')
    
    ax.set_xlabel("Feature (raw)", fontsize=12)
    ax.set_ylabel("Probability", fontsize=12)
    ax.set_title("1D Logistic Regression: Data Points + Fitted Curve", fontsize=14, fontweight='bold')
    ax.set_ylim([-0.1, 1.1])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
