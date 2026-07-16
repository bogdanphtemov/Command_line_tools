"""
Evaluation metrics for SVM classification and regression.

Provides both classification metrics (accuracy, precision, recall, F1, confusion matrix)
and regression metrics (MSE, MAE, R², RMSE, MAPE).

Example:
    >>> from myclt.ML.supervised_learning.svm.metrics import accuracy, precision_recall_f1
    >>> from myclt.ML.supervised_learning.svm.metrics import mean_squared_error, r2_score
    >>> acc = accuracy(y_true, y_pred)
    >>> mse = mean_squared_error(y_true, y_pred)
"""

import warnings

import numpy as np
from typing import Optional, List


# ============================================================================
# Classification Metrics
# ============================================================================

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute classification accuracy: correct / total.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Accuracy score (0.0 to 1.0)
    """
    return float(np.mean(y_true == y_pred))


def _precision_per_class(y_true: np.ndarray, y_pred: np.ndarray,
                         class_label: int) -> float:
    """
    Compute precision for a single class (used internally).

    Precision = TP / (TP + FP), where TP/FP are defined with the given
    class_label as the positive class.
    """
    tp = int(np.sum((y_true == class_label) & (y_pred == class_label)))
    fp = int(np.sum((y_true != class_label) & (y_pred == class_label)))
    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)


def _recall_per_class(y_true: np.ndarray, y_pred: np.ndarray,
                      class_label: int) -> float:
    """
    Compute recall for a single class (used internally).

    Recall = TP / (TP + FN), where TP/FN are defined with the given
    class_label as the positive class.
    """
    tp = int(np.sum((y_true == class_label) & (y_pred == class_label)))
    fn = int(np.sum((y_true == class_label) & (y_pred != class_label)))
    if tp + fn == 0:
        return 0.0
    return tp / (tp + fn)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                     labels: Optional[List[int]] = None) -> np.ndarray:
    """
    Compute confusion matrix (supports both binary and multiclass).

    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: List of label values. If None, inferred from data.

    Returns:
        Confusion matrix of shape (n_labels, n_labels).
        For binary [0,1] labels: [[TN, FP], [FN, TP]]
    """
    if labels is None:
        labels = sorted(np.unique(np.concatenate([y_true, y_pred])).astype(int).tolist())

    n_labels = len(labels)
    cm = np.zeros((n_labels, n_labels), dtype=int)

    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            cm[i, j] = int(np.sum((y_true == true_label) & (y_pred == pred_label)))

    return cm


def precision_score(y_true: np.ndarray, y_pred: np.ndarray,
                    pos_label: int = 1) -> float:
    """
    Compute precision: TP / (TP + FP)

    Args:
        y_true: True labels
        y_pred: Predicted labels
        pos_label: The positive class label. Default 1.

    Returns:
        Precision score (0.0 to 1.0)
    """
    return _precision_per_class(y_true, y_pred, pos_label)


def recall_score(y_true: np.ndarray, y_pred: np.ndarray,
                 pos_label: int = 1) -> float:
    """
    Compute recall: TP / (TP + FN)

    Args:
        y_true: True labels
        y_pred: Predicted labels
        pos_label: The positive class label. Default 1.

    Returns:
        Recall score (0.0 to 1.0)
    """
    return _recall_per_class(y_true, y_pred, pos_label)


def f1_score(y_true: np.ndarray, y_pred: np.ndarray,
             pos_label: int = 1) -> float:
    """
    Compute F1 score: 2 * (precision * recall) / (precision + recall)

    Args:
        y_true: True labels
        y_pred: Predicted labels
        pos_label: The positive class label. Default 1.

    Returns:
        F1 score (0.0 to 1.0)
    """
    p = precision_score(y_true, y_pred, pos_label)
    r = recall_score(y_true, y_pred, pos_label)
    if p + r == 0:
        return 0.0
    return 2.0 * p * r / (p + r)


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray,
                        pos_label: int = 1) -> tuple:
    """
    Compute precision, recall, and F1 score in one call.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        pos_label: The positive class label. Default 1.

    Returns:
        Tuple of (precision, recall, f1)
    """
    p = precision_score(y_true, y_pred, pos_label)
    r = recall_score(y_true, y_pred, pos_label)
    f1 = f1_score(y_true, y_pred, pos_label)
    return p, r, f1


def classification_report(y_true: np.ndarray, y_pred: np.ndarray,
                          target_names: Optional[List[str]] = None) -> str:
    """
    Generate a text classification report.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        target_names: Optional names for classes

    Returns:
        Formatted classification report string
    """
    classes = sorted(np.unique(y_true))
    if target_names is None:
        target_names = [str(c) for c in classes]

    lines = []
    lines.append(f"{'':>15} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}")
    lines.append("-" * 55)

    for cls, name in zip(classes, target_names):
        p = precision_score(y_true, y_pred, pos_label=cls)
        r = recall_score(y_true, y_pred, pos_label=cls)
        f1 = f1_score(y_true, y_pred, pos_label=cls)
        support = int(np.sum(y_true == cls))
        lines.append(f"{name:>15} {p:>10.4f} {r:>10.4f} {f1:>10.4f} {support:>10}")

    # Averages
    lines.append("-" * 55)
    macro_p = np.mean([precision_score(y_true, y_pred, pos_label=c) for c in classes])
    macro_r = np.mean([recall_score(y_true, y_pred, pos_label=c) for c in classes])
    macro_f1 = np.mean([f1_score(y_true, y_pred, pos_label=c) for c in classes])
    lines.append(f"{'macro avg':>15} {macro_p:>10.4f} {macro_r:>10.4f} {macro_f1:>10.4f} {len(y_true):>10}")

    acc = accuracy(y_true, y_pred)
    lines.append(f"{'accuracy':>15} {'':>10} {'':>10} {acc:>10.4f} {len(y_true):>10}")

    return "\n".join(lines)


# ============================================================================
# Multiclass Classification Metrics
# ============================================================================

def multiclass_precision(y_true: np.ndarray, y_pred: np.ndarray,
                         average: str = 'macro') -> float:
    """
    Compute multiclass precision.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: 'macro' (unweighted mean) or 'micro' (global average).
                 Note: micro-precision equals accuracy.

    Returns:
        Precision score
    """
    classes = np.unique(y_true)
    if average == 'macro':
        scores = [_precision_per_class(y_true, y_pred, c) for c in classes]
        return float(np.mean(scores))
    else:  # micro: precision = recall = F1 = accuracy
        return accuracy(y_true, y_pred)


def multiclass_recall(y_true: np.ndarray, y_pred: np.ndarray,
                      average: str = 'macro') -> float:
    """
    Compute multiclass recall.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: 'macro' (unweighted mean) or 'micro' (global average).
                 Note: micro-recall equals accuracy.

    Returns:
        Recall score
    """
    classes = np.unique(y_true)
    if average == 'macro':
        scores = [_recall_per_class(y_true, y_pred, c) for c in classes]
        return float(np.mean(scores))
    else:  # micro: precision = recall = F1 = accuracy
        return accuracy(y_true, y_pred)


def multiclass_f1_score(y_true: np.ndarray, y_pred: np.ndarray,
                        average: str = 'macro') -> float:
    """Compute multiclass F1 score."""
    p = multiclass_precision(y_true, y_pred, average)
    r = multiclass_recall(y_true, y_pred, average)
    if p + r == 0:
        return 0.0
    return 2.0 * p * r / (p + r)


def multiclass_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                                n_classes: Optional[int] = None) -> np.ndarray:
    """
    Compute K×K confusion matrix for multiclass classification.

    Args:
        y_true: True labels (0..K-1)
        y_pred: Predicted labels (0..K-1)
        n_classes: Number of classes (K). If None, inferred from data.

    Returns:
        Confusion matrix of shape (n_classes, n_classes)
    """
    if n_classes is None:
        n_classes = len(np.unique(np.concatenate([y_true, y_pred])))
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def print_multiclass_classification_report(y_true: np.ndarray, y_pred: np.ndarray,
                                            class_names: Optional[List[str]] = None) -> None:
    """
    Print a formatted multiclass classification report.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Optional names for classes
    """
    classes = sorted(np.unique(y_true))
    n_classes = len(classes)

    if class_names is None:
        class_names = [f"Class {int(c)}" for c in classes]

    print("\n" + "=" * 70)
    print("MULTICLASS CLASSIFICATION REPORT")
    print("=" * 70)

    header = f"{'':>15} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}"
    print(header)
    print("-" * 55)

    for c, name in zip(classes, class_names):
        # Binary one-vs-rest metrics for each class
        y_true_bin = (y_true == c).astype(int)
        y_pred_bin = (y_pred == c).astype(int)
        p = precision_score(y_true_bin, y_pred_bin, pos_label=1)
        r = recall_score(y_true_bin, y_pred_bin, pos_label=1)
        f1 = f1_score(y_true_bin, y_pred_bin, pos_label=1)
        support = int(np.sum(y_true == c))
        print(f"{name:>15} {p:>10.4f} {r:>10.4f} {f1:>10.4f} {support:>10}")

    print("-" * 55)

    macro_p = multiclass_precision(y_true, y_pred, average='macro')
    macro_r = multiclass_recall(y_true, y_pred, average='macro')
    macro_f1 = multiclass_f1_score(y_true, y_pred, average='macro')
    micro_f1 = multiclass_f1_score(y_true, y_pred, average='micro')
    acc = accuracy(y_true, y_pred)

    print(f"{'macro avg':>15} {macro_p:>10.4f} {macro_r:>10.4f} {macro_f1:>10.4f} {len(y_true):>10}")
    print(f"{'micro avg':>15} {'':>10} {'':>10} {micro_f1:>10.4f} {len(y_true):>10}")
    print(f"{'accuracy':>15} {'':>10} {'':>10} {acc:>10.4f} {len(y_true):>10}")
    print("=" * 70)


# ============================================================================
# Regression Metrics
# ============================================================================

def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Mean Squared Error.

    Args:
        y_true: True target values
        y_pred: Predicted values

    Returns:
        MSE value
    """
    return float(np.mean((y_true - y_pred) ** 2))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Root Mean Squared Error.

    Args:
        y_true: True target values
        y_pred: Predicted values

    Returns:
        RMSE value
    """
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Mean Absolute Error.

    Args:
        y_true: True target values
        y_pred: Predicted values

    Returns:
        MAE value
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute R² (coefficient of determination).

    Formula: R² = 1 - SS_res / SS_tot

    Args:
        y_true: True target values
        y_pred: Predicted values

    Returns:
        R² score (best = 1.0, can be negative)
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Mean Absolute Percentage Error.

    Args:
        y_true: True target values
        y_pred: Predicted values

    Returns:
        MAPE value as percentage (0-100).
        Returns NaN if all true values are zero.
    """
    mask = y_true != 0
    if not np.any(mask):
        warnings.warn(
            "MAPE is undefined when all y_true values are zero. "
            "Returning NaN."
        )
        return float('nan')
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def regression_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """
    Generate a text regression report with key metrics.

    Args:
        y_true: True target values
        y_pred: Predicted values

    Returns:
        Formatted regression report string
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)

    lines = [
        "=" * 70,
        "REGRESSION REPORT",
        "=" * 70,
        f"{'Metric':>20} {'Value':>15}",
        "-" * 35,
        f"{'MSE':>20} {mse:>15.6f}",
        f"{'RMSE':>20} {rmse:>15.6f}",
        f"{'MAE':>20} {mae:>15.6f}",
        f"{'R²':>20} {r2:>15.6f}",
        f"{'MAPE (%)':>20} {mape:>15.2f}",
        "=" * 70,
    ]
    return "\n".join(lines)
