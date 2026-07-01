"""
Visualization utilities for Logistic Regression results.

Uses matplotlib for professional visualizations:
    - Training loss history (shared utility)
    - Confusion matrix heatmap (binary & multiclass)
    - Feature importance (coefficients)
    - Probability distribution (by class)
    - Performance metrics comparison
    - ROC curve
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List
from matplotlib.patches import Rectangle

from .preprocessing import standardize_apply
from ..visualization_utils import plot_loss_curve


# ============================================================================
# Binary classification visualizations (original)
# ============================================================================

def plot_confusion_matrix_heatmap(tp: int, fp: int, fn: int, tn: int) -> None:
    """
    Plot confusion matrix as a heatmap.
    
    Args:
        tp: True Positives
        fp: False Positives
        fn: False Negatives
        tn: True Negatives
    """
    cm = np.array([[tn, fp], 
                   [fn, tp]])
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues, aspect='auto')
    
    threshold = cm.max() / 2
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > threshold else "black"
            ax.text(j, i, str(cm[i, j]), 
                   ha="center", va="center", color=color, fontsize=16, fontweight='bold')
    
    class_labels = ['Negative (0)', 'Positive (1)']
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicted 0', 'Predicted 1'])
    ax.set_yticklabels(['Actual 0', 'Actual 1'])
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight='bold')
    
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
    
    sorted_indices = np.argsort(np.abs(coefficients))
    sorted_names = [feature_names[i] for i in sorted_indices]
    sorted_coefs = coefficients[sorted_indices]
    
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
    
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{score:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylim([0, 1.1])
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Classification Performance Metrics", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Baseline (0.5)')
    ax.axhline(y=1.0, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Perfect (1.0)')
    ax.legend()
    
    plt.tight_layout()
    plt.show()


def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray) -> None:
    """
    Plot ROC curve for binary classification.
    
    Args:
        y_true: True binary labels (0 or 1)
        y_proba: Predicted probabilities [0, 1]
    """
    sorted_indices = np.argsort(-y_proba)
    y_sorted = y_true[sorted_indices]
    
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    
    if n_pos == 0 or n_neg == 0:
        print("ROC curve requires both positive and negative samples")
        return
    
    fpr_list = [0.0]
    tpr_list = [0.0]
    
    tp = 0
    fp = 0
    
    for i, idx in enumerate(sorted_indices):
        if y_sorted[i] == 1:
            tp += 1
        else:
            fp += 1
        
        tpr = tp / n_pos if n_pos > 0 else 0
        fpr = fp / n_neg if n_neg > 0 else 0
        
        fpr_list.append(fpr)
        tpr_list.append(tpr)
    
    fpr_list = np.array(fpr_list)
    tpr_list = np.array(tpr_list)
    
    # Calculate AUC
    auc = 0.0
    for i in range(1, len(fpr_list)):
        auc += (fpr_list[i] - fpr_list[i-1]) * (tpr_list[i] + tpr_list[i-1]) / 2.0
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.plot(fpr_list, tpr_list, color='#2E86AB', lw=2.5, label=f'ROC Curve (AUC = {auc:.4f})')
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
    
    Args:
        x_raw: Raw feature (n_samples, 1) or (n_samples,)
        y_true: True binary labels (0 or 1)
        model: Trained LogisticRegressionGD model
        scaler_mean: Optional scaling mean
        scaler_std: Optional scaling std
    """
    if x_raw.ndim == 1:
        x_raw = x_raw.reshape(-1, 1)
    
    x_min, x_max = float(x_raw.min()), float(x_raw.max())
    
    x_grid = np.linspace(x_min, x_max, 300).reshape(-1, 1)
    
    if scaler_mean is not None and scaler_std is not None:
        x_grid_scaled = standardize_apply(x_grid, scaler_mean, scaler_std)
    else:
        x_grid_scaled = x_grid
    
    y_proba = model.predict_proba(x_grid_scaled)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    jitter = np.random.normal(0, 0.02, size=len(y_true))
    y_jittered = y_true + jitter
    
    mask_0 = y_true == 0
    mask_1 = y_true == 1
    ax.scatter(x_raw[mask_0], y_jittered[mask_0], alpha=0.5, s=50, color='#E63946', label='Class 0')
    ax.scatter(x_raw[mask_1], y_jittered[mask_1], alpha=0.5, s=50, color='#06A77D', label='Class 1')
    
    ax.plot(x_grid, y_proba, color='#2E86AB', linewidth=2.5, label='Logistic Curve')
    
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Decision Boundary (0.5)')
    
    ax.set_xlabel("Feature (raw)", fontsize=12)
    ax.set_ylabel("Probability", fontsize=12)
    ax.set_title("1D Logistic Regression: Data Points + Fitted Curve", fontsize=14, fontweight='bold')
    ax.set_ylim([-0.1, 1.1])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# Multiclass classification visualizations
# ============================================================================

def plot_multiclass_confusion_matrix(cm: np.ndarray, class_names: Optional[List[str]] = None) -> None:
    """
    Plot confusion matrix heatmap for multiclass classification.
    
    Args:
        cm: Confusion matrix (K, K) where cm[i,j] = count of class i predicted as class j
        class_names: Optional list of class names for labels
    """
    n_classes = cm.shape[0]
    
    if class_names is None:
        class_names = [f"Class {k}" for k in range(n_classes)]
    
    fig, ax = plt.subplots(figsize=(max(8, n_classes * 1.5), max(7, n_classes * 1.2)))
    
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues, aspect='auto')
    
    # Add text annotations
    threshold = cm.max() / 2 if cm.max() > 0 else 0
    for i in range(n_classes):
        for j in range(n_classes):
            color = "white" if cm[i, j] > threshold else "black"
            ax.text(j, i, str(cm[i, j]), 
                   ha="center", va="center", color=color, fontsize=12, fontweight='bold')
    
    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(class_names, fontsize=10)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_title("Multiclass Confusion Matrix", fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Count", fontsize=11)
    
    plt.tight_layout()
    plt.show()


def plot_multiclass_probability_heatmap(probabilities: np.ndarray, y_true: np.ndarray,
                                         class_names: Optional[List[str]] = None,
                                         n_samples_to_show: int = 20) -> None:
    """
    Plot heatmap of predicted probabilities for a subset of samples.
    
    Each row is a sample, columns are classes. Color intensity shows probability.
    True class is highlighted.
    
    Args:
        probabilities: Predicted probabilities (n_samples, n_classes)
        y_true: True labels (n_samples,)
        class_names: Optional list of class names
        n_samples_to_show: Number of samples to display (default 20)
    """
    n_classes = probabilities.shape[1]
    n_show = min(n_samples_to_show, probabilities.shape[0])
    
    if class_names is None:
        class_names = [f"Class {k}" for k in range(n_classes)]
    
    # Take a random subset if too many samples
    if probabilities.shape[0] > n_show:
        indices = np.random.choice(probabilities.shape[0], n_show, replace=False)
        probs_subset = probabilities[indices]
        y_subset = y_true[indices]
    else:
        probs_subset = probabilities[:n_show]
        y_subset = y_true[:n_show]
    
    fig, ax = plt.subplots(figsize=(max(10, n_classes * 0.8), max(8, n_show * 0.4)))
    
    im = ax.imshow(probs_subset, interpolation='nearest', cmap=plt.cm.YlOrRd, aspect='auto', vmin=0, vmax=1)
    
    # Mark true class with a border
    for i in range(n_show):
        true_k = int(y_subset[i])
        ax.add_patch(Rectangle((true_k - 0.5, i - 0.5), 1, 1,
                                fill=False, edgecolor='blue', linewidth=3))
    
    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_show))
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels([f"Sample {i+1}" for i in range(n_show)], fontsize=8)
    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Sample", fontsize=12)
    ax.set_title("Predicted Probabilities Heatmap\n(blue border = true class)", fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Probability", fontsize=11)
    
    plt.tight_layout()
    plt.show()


def plot_multiclass_feature_importance(feature_names: List[str], W: np.ndarray,
                                        class_names: Optional[List[str]] = None) -> None:
    """
    Plot feature importance for each class in multinomial logistic regression.
    
    Shows the weight contribution of each feature to each class.
    
    Args:
        feature_names: Names of features
        W: Weight matrix (n_features, n_classes)
        class_names: Optional list of class names
    """
    n_features, n_classes = W.shape
    
    if class_names is None:
        class_names = [f"Class {k}" for k in range(n_classes)]
    
    fig, axes = plt.subplots(1, n_classes, figsize=(6 * n_classes, 6), sharey=True)
    if n_classes == 1:
        axes = [axes]
    
    for k in range(n_classes):
        coefs = W[:, k]
        sorted_indices = np.argsort(np.abs(coefs))
        sorted_names = [feature_names[i] for i in sorted_indices]
        sorted_coefs = coefs[sorted_indices]
        
        colors = ['#E63946' if c < 0 else '#06A77D' for c in sorted_coefs]
        
        ax = axes[k]
        y_pos = np.arange(len(sorted_names))
        ax.barh(y_pos, sorted_coefs, color=colors, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_names, fontsize=9)
        ax.set_xlabel("Coefficient", fontsize=10)
        ax.set_title(f"Class: {class_names[k]}", fontsize=12, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(True, alpha=0.3, axis='x')
    
    plt.suptitle("Feature Coefficients per Class", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_class_probability_distributions(probabilities: np.ndarray, y_true: np.ndarray,
                                          class_names: Optional[List[str]] = None) -> None:
    """
    Plot probability distribution for each class (true class probabilities).
    
    For each class, shows histogram of predicted probabilities for that class
    when the true label is that class.
    
    Args:
        probabilities: Predicted probabilities (n_samples, n_classes)
        y_true: True labels (n_samples,)
        class_names: Optional list of class names
    """
    n_classes = probabilities.shape[1]
    
    if class_names is None:
        class_names = [f"Class {k}" for k in range(n_classes)]
    
    n_cols = min(3, n_classes)
    n_rows = int(np.ceil(n_classes / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes = axes.flatten() if n_classes > 1 else [axes]
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    
    for k in range(n_classes):
        ax = axes[k]
        mask = y_true == k
        class_probs = probabilities[mask, k]
        
        if len(class_probs) == 0:
            ax.text(0.5, 0.5, f'No samples for\n{class_names[k]}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            continue
        
        ax.hist(class_probs, bins=30, color=colors[k], alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(class_probs), color='black', linestyle='--', linewidth=2,
                  label=f'Mean: {np.mean(class_probs):.3f}')
        
        ax.set_xlabel("Probability", fontsize=11)
        ax.set_ylabel("Frequency", fontsize=11)
        ax.set_title(f"{class_names[k]} (n={len(class_probs)})", fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for k in range(n_classes, len(axes)):
        axes[k].set_visible(False)
    
    plt.suptitle("Predicted Probability Distributions by True Class", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
