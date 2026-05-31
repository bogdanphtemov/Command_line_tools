"""
Classification metrics for Logistic Regression.

Metrics specifically designed for binary classification problems.
"""

import numpy as np
from typing import Tuple

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate accuracy: proportion of correct predictions.
    
    Formula: (TP + TN) / (TP + TN + FP + FN)
    
    Args:
        y_true: True binary labels (0 or 1)
        y_pred: Predicted binary labels (0 or 1)
    
    Returns:
        Accuracy score in range [0, 1]
    """
    return float(np.mean(y_true == y_pred))


def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate precision: proportion of true positives among predicted positives.
    
    Formula: TP / (TP + FP)
    
    Interpretation: Of all items predicted positive, how many were actually positive?
    
    Args:
        y_true: True binary labels (0 or 1)
        y_pred: Predicted binary labels (0 or 1)
    
    Returns:
        Precision score in range [0, 1], or 0 if no positive predictions
    """
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    
    denominator = tp + fp
    if denominator == 0:
        return 0.0
    
    return float(tp / denominator)


def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate recall (sensitivity): proportion of true positives among actual positives.
    
    Formula: TP / (TP + FN)
    
    Interpretation: Of all items that are actually positive, how many did we identify?
    
    Args:
        y_true: True binary labels (0 or 1)
        y_pred: Predicted binary labels (0 or 1)
    
    Returns:
        Recall score in range [0, 1], or 0 if no actual positives
    """
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    
    denominator = tp + fn
    if denominator == 0:
        return 0.0
    
    return float(tp / denominator)


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate F1 score: harmonic mean of precision and recall.
    
    Formula: 2 * (precision * recall) / (precision + recall)
    
    Interpretation: Balanced metric that considers both false positives and false negatives.
    Range: [0, 1], with 1 being perfect.
    
    Args:
        y_true: True binary labels (0 or 1)
        y_pred: Predicted binary labels (0 or 1)
    
    Returns:
        F1 score in range [0, 1]
    """
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    
    denominator = prec + rec
    if denominator == 0:
        return 0.0
    
    return float(2 * (prec * rec) / denominator)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Calculate confusion matrix for binary classification.
    
    Returns:
        (TP, FP, FN, TN) where:
            - TP: True Positives (correctly predicted 1)
            - FP: False Positives (incorrectly predicted 1)
            - FN: False Negatives (incorrectly predicted 0)
            - TN: True Negatives (correctly predicted 0)
    
    Args:
        y_true: True binary labels (0 or 1)
        y_pred: Predicted binary labels (0 or 1)
    
    Returns:
        Tuple of (TP, FP, FN, TN)
    """
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    
    return tp, fp, fn, tn


def specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate specificity (true negative rate): proportion of true negatives among actual negatives.
    
    Formula: TN / (TN + FP)
    
    Interpretation: Of all items that are actually negative, how many did we correctly identify?
    
    Args:
        y_true: True binary labels (0 or 1)
        y_pred: Predicted binary labels (0 or 1)
    
    Returns:
        Specificity score in range [0, 1], or 0 if no actual negatives
    """
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    
    denominator = tn + fp
    if denominator == 0:
        return 0.0
    
    return float(tn / denominator)


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Print comprehensive classification report.
    
    Displays all major metrics in a formatted table.
    
    Args:
        y_true: True binary labels (0 or 1)
        y_pred: Predicted binary labels (0 or 1)
    """
    acc = accuracy(y_true, y_pred)
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    spec = specificity(y_true, y_pred)
    tp, fp, fn, tn = confusion_matrix(y_true, y_pred)
    
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(f"Accuracy:     {acc:.4f}")
    print(f"Precision:    {prec:.4f}")
    print(f"Recall:       {rec:.4f}")
    print(f"F1 Score:     {f1:.4f}")
    print(f"Specificity:  {spec:.4f}")
    print("\n" + "-" * 60)
    print("CONFUSION MATRIX")
    print("-" * 60)
    print(f"True Positives:  {tp}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Negatives:  {tn}")
    print("=" * 60 + "\n")
