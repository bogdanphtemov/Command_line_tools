"""
Classification metrics for Logistic Regression.

Provides metrics for both:
    - Binary classification (2 classes)
    - Multiclass classification (K > 2 classes)
"""

import numpy as np
from typing import Tuple, Optional

# ============================================================================
# Binary classification metrics (original)
# ============================================================================

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate accuracy: proportion of correct predictions.
    
    Works for both binary and multiclass.
    
    Formula: (TP + TN) / (TP + TN + FP + FN)  [binary]
             correct_predictions / total        [multiclass]
    
    Args:
        y_true: True labels (0 or 1 for binary, 0..K-1 for multiclass)
        y_pred: Predicted labels
    
    Returns:
        Accuracy score in range [0, 1]
    """
    return float(np.mean(y_true == y_pred))


def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate precision: proportion of true positives among predicted positives.
    
    Formula: TP / (TP + FP)
    
    For multiclass: macro-averaged precision (average per-class precision).
    
    Args:
        y_true: True binary labels (0 or 1)
        y_pred: Predicted binary labels (0 or 1)
    
    Returns:
        Precision score in range [0, 1]
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
    
    For multiclass: macro-averaged recall.
    
    Args:
        y_true: True binary labels (0 or 1)
        y_pred: Predicted binary labels (0 or 1)
    
    Returns:
        Recall score in range [0, 1]
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
    
    For multiclass: macro-averaged F1.
    
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
        (TP, FP, FN, TN)
    """
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    
    return tp, fp, fn, tn


def specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate specificity (true negative rate).
    
    Formula: TN / (TN + FP)
    
    Args:
        y_true: True binary labels (0 or 1)
        y_pred: Predicted binary labels (0 or 1)
    
    Returns:
        Specificity score in range [0, 1]
    """
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    
    denominator = tn + fp
    if denominator == 0:
        return 0.0
    
    return float(tn / denominator)


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Print comprehensive classification report for binary.
    
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
    print("CLASSIFICATION REPORT (BINARY)")
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


# ============================================================================
# Multiclass classification metrics
# ============================================================================

def multiclass_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                                 n_classes: Optional[int] = None) -> np.ndarray:
    """
    Calculate full confusion matrix for multiclass classification.
    
    cm[i, j] = number of samples of class i predicted as class j.
    
    Args:
        y_true: True labels (0..K-1)
        y_pred: Predicted labels (0..K-1)
        n_classes: Number of classes (auto-detected if None)
    
    Returns:
        Confusion matrix (K, K) as numpy array
    """
    if n_classes is None:
        n_classes = max(int(y_true.max()), int(y_pred.max())) + 1
    
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    
    return cm


def multiclass_precision(y_true: np.ndarray, y_pred: np.ndarray, 
                         average: str = 'macro') -> float:
    """
    Calculate precision for multiclass classification.
    
    Args:
        y_true: True labels (0..K-1)
        y_pred: Predicted labels (0..K-1)
        average: 'macro' (average per-class) or 'micro' (global)
    
    Returns:
        Precision score in range [0, 1]
    """
    n_classes = max(int(y_true.max()), int(y_pred.max())) + 1
    cm = multiclass_confusion_matrix(y_true, y_pred, n_classes)
    
    if average == 'micro':
        # Global: TP / (TP + FP) across all classes
        tp_total = np.trace(cm)
        fp_total = np.sum(cm) - tp_total
        return float(tp_total / (tp_total + fp_total)) if (tp_total + fp_total) > 0 else 0.0
    
    # Macro: average per-class precision
    precisions = []
    for k in range(n_classes):
        tp = cm[k, k]
        fp = np.sum(cm[:, k]) - tp
        precisions.append(float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0)
    
    return float(np.mean(precisions))


def multiclass_recall(y_true: np.ndarray, y_pred: np.ndarray,
                      average: str = 'macro') -> float:
    """
    Calculate recall for multiclass classification.
    
    Args:
        y_true: True labels (0..K-1)
        y_pred: Predicted labels (0..K-1)
        average: 'macro' (average per-class) or 'micro' (global)
    
    Returns:
        Recall score in range [0, 1]
    """
    n_classes = max(int(y_true.max()), int(y_pred.max())) + 1
    cm = multiclass_confusion_matrix(y_true, y_pred, n_classes)
    
    if average == 'micro':
        # Global: TP / (TP + FN) across all classes
        tp_total = np.trace(cm)
        fn_total = np.sum(cm) - tp_total
        return float(tp_total / (tp_total + fn_total)) if (tp_total + fn_total) > 0 else 0.0
    
    # Macro: average per-class recall
    recalls = []
    for k in range(n_classes):
        tp = cm[k, k]
        fn = np.sum(cm[k, :]) - tp
        recalls.append(float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0)
    
    return float(np.mean(recalls))


def multiclass_f1_score(y_true: np.ndarray, y_pred: np.ndarray,
                        average: str = 'macro') -> float:
    """
    Calculate F1 score for multiclass classification.
    
    Args:
        y_true: True labels (0..K-1)
        y_pred: Predicted labels (0..K-1)
        average: 'macro' or 'micro'
    
    Returns:
        F1 score in range [0, 1]
    """
    if average == 'micro':
        prec = multiclass_precision(y_true, y_pred, average='micro')
        rec = multiclass_recall(y_true, y_pred, average='micro')
    else:
        prec = multiclass_precision(y_true, y_pred, average='macro')
        rec = multiclass_recall(y_true, y_pred, average='macro')
    
    denominator = prec + rec
    if denominator == 0:
        return 0.0
    
    return float(2 * (prec * rec) / denominator)


def print_multiclass_classification_report(y_true: np.ndarray, y_pred: np.ndarray,
                                           class_names: Optional[list] = None) -> None:
    """
    Print detailed multiclass classification report.
    
    Shows per-class precision, recall, F1, support and macro/micro averages.
    
    Args:
        y_true: True labels (0..K-1)
        y_pred: Predicted labels (0..K-1)
        class_names: Optional list of class names for display
    """
    n_classes = max(int(y_true.max()), int(y_pred.max())) + 1
    
    if class_names is None:
        class_names = [f"Class {k}" for k in range(n_classes)]
    
    cm = multiclass_confusion_matrix(y_true, y_pred, n_classes)
    
    print("\n" + "=" * 70)
    print("MULTICLASS CLASSIFICATION REPORT")
    print("=" * 70)
    print(f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 70)
    
    for k in range(n_classes):
        tp = cm[k, k]
        fp = np.sum(cm[:, k]) - tp
        fn = np.sum(cm[k, :]) - tp
        support = int(np.sum(cm[k, :]))
        
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        
        print(f"{class_names[k]:<15} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f} {support:>10}")
    
    print("-" * 70)
    
    # Macro avg
    macro_prec = multiclass_precision(y_true, y_pred, average='macro')
    macro_rec = multiclass_recall(y_true, y_pred, average='macro')
    macro_f1 = multiclass_f1_score(y_true, y_pred, average='macro')
    print(f"{'Macro avg':<15} {macro_prec:>10.4f} {macro_rec:>10.4f} {macro_f1:>10.4f} {'':>10}")
    
    # Micro avg
    micro_prec = multiclass_precision(y_true, y_pred, average='micro')
    micro_rec = multiclass_recall(y_true, y_pred, average='micro')
    micro_f1 = multiclass_f1_score(y_true, y_pred, average='micro')
    print(f"{'Micro avg':<15} {micro_prec:>10.4f} {micro_rec:>10.4f} {micro_f1:>10.4f} {'':>10}")
    
    # Accuracy
    acc = accuracy(y_true, y_pred)
    print(f"\nAccuracy: {acc:.4f}")
    
    # Confusion matrix
    print("\n" + "-" * 70)
    print("CONFUSION MATRIX")
    print("-" * 70)
    
    # Header
    header = "     " + "".join(f"{class_names[k][:6]:>7}" for k in range(n_classes))
    print(header)
    
    for i in range(n_classes):
        row_str = f"{class_names[i][:6]:>5}" + "".join(f"{cm[i, j]:>7}" for j in range(n_classes))
        print(row_str)
    
    print("=" * 70 + "\n")
