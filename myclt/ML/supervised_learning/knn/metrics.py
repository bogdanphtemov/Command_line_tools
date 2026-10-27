"""
Metrics for evaluating KNN algorithm performance.
"""

import numpy as np

__all__ = [
    'calculate_accuracy',
    'calculate_confusion_matrix', 
    'calculate_mean_squared_error',
    'calculate_r2_score',
    'calculate_mae',
    'calculate_classification_report',
    'print_classification_report',
    'evaluate_knn_model'
]


def calculate_accuracy(y_true, y_pred):
    """
    Calculate classification accuracy.
    
    Args:
        y_true: True labels (numpy array or list)
        y_pred: Predicted labels (numpy array or list)
        
    Returns:
        float: Accuracy score (0.0 to 1.0)
        
    Note:
        Returns NaN for empty arrays (edge case)
        Raises ValueError if arrays have different lengths
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if len(y_true) == 0:
        return np.nan
    
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true and y_pred have different lengths: {len(y_true)} vs {len(y_pred)}")
    
    return np.mean(y_true == y_pred)


def calculate_confusion_matrix(y_true, y_pred):
    """
    Calculate confusion matrix for classification.
    
    Args:
        y_true: True labels (numpy array or list)
        y_pred: Predicted labels (numpy array or list)
        
    Returns:
        dict: Confusion matrix as dictionary with class labels as keys
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true and y_pred have different lengths: {len(y_true)} vs {len(y_pred)}")
    
    unique_classes = np.unique(np.concatenate([y_true, y_pred]))
    cm = {}
    
    for true_class in unique_classes:
        cm[true_class] = {}
        for pred_class in unique_classes:
            cm[true_class][pred_class] = 0
    
    for true, pred in zip(y_true, y_pred):
        cm[true][pred] += 1
    
    return cm


def calculate_mean_squared_error(y_true, y_pred):
    """
    Calculate Mean Squared Error for regression.
    
    Args:
        y_true: True values (numpy array or list)
        y_pred: Predicted values (numpy array or list)
        
    Returns:
        float: Mean Squared Error
        
    Note:
        Returns NaN for empty arrays (edge case)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if len(y_true) == 0:
        return np.nan
    
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true and y_pred have different lengths: {len(y_true)} vs {len(y_pred)}")
    
    return np.mean((y_true - y_pred) ** 2)


def calculate_r2_score(y_true, y_pred):
    """
    Calculate R² score for regression.
    
    Args:
        y_true: True values (numpy array or list)
        y_pred: Predicted values (numpy array or list)
        
    Returns:
        float: R² score
        
    Note:
        Returns 1.0 for perfect predictions when y_true is constant
        Returns 0.0 for constant predictions that don't match y_true mean
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true and y_pred have different lengths: {len(y_true)} vs {len(y_pred)}")
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if np.isclose(ss_tot, 0):
        return 1.0 if np.isclose(ss_res, 0) else 0.0
    
    return 1 - (ss_res / ss_tot)


def calculate_mae(y_true, y_pred):
    """
    Calculate Mean Absolute Error for regression.
    
    Args:
        y_true: True values (numpy array or list)
        y_pred: Predicted values (numpy array or list)
        
    Returns:
        float: Mean Absolute Error
        
    Note:
        Returns NaN for empty arrays (edge case)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if len(y_true) == 0:
        return np.nan
    
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true and y_pred have different lengths: {len(y_true)} vs {len(y_pred)}")
    
    return np.mean(np.abs(y_true - y_pred))


def calculate_classification_report(y_true, y_pred):
    """
    Calculate comprehensive classification metrics per class.
    
    Args:
        y_true: True labels (numpy array or list)
        y_pred: Predicted labels (numpy array or list)
        
    Returns:
        dict: Classification metrics by class
        
    Note:
        Follows sklearn semantics: returns 0.0 for undefined metrics (sklearn compatible)
        F1-score is 0.0 when both precision and recall are 0.0 (by definition)
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true and y_pred have different lengths: {len(y_true)} vs {len(y_pred)}")
    
    unique_classes = np.unique(np.concatenate([y_true, y_pred]))
    
    report = {}
    
    for cls in unique_classes:
        # True positives, false positives, false negatives
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        
        # Precision, recall, F1-score (sklearn compatible)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        report[cls] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'support': np.sum(y_true == cls)
        }
    
    return report


def print_classification_report(y_true, y_pred):
    """
    Print comprehensive classification report to stdout.
    
    Args:
        y_true: True labels (numpy array or list)
        y_pred: Predicted labels (numpy array or list)
        
    Returns:
        None
        
    Note:
        Includes overall accuracy and averages (macro/weighted) similar to sklearn
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true and y_pred have different lengths: {len(y_true)} vs {len(y_pred)}")
    
    report = calculate_classification_report(y_true, y_pred)
    accuracy = calculate_accuracy(y_true, y_pred)
    
    total_support = len(y_true)
    
    # Calculate macro and weighted averages
    precisions = []
    recalls = []
    f1_scores = []
    supports = []
    
    for cls, metrics in report.items():
        precisions.append(metrics['precision'])
        recalls.append(metrics['recall'])
        f1_scores.append(metrics['f1_score'])
        supports.append(metrics['support'])
    
    supports = np.array(supports)
    
    macro_precision = np.mean(precisions)
    macro_recall = np.mean(recalls)
    macro_f1 = np.mean(f1_scores)
    
    weighted_precision = np.average(precisions, weights=supports)
    weighted_recall = np.average(recalls, weights=supports)
    weighted_f1 = np.average(f1_scores, weights=supports)
    
    # Print header
    print(f"\n{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
    print("-" * 70)
    
    # Print metrics for each class
    for cls, metrics in report.items():
        print(f"{str(cls):<15} {metrics['precision']:<12.4f} "
              f"{metrics['recall']:<12.4f} {metrics['f1_score']:<12.4f} "
              f"{metrics['support']:<10}")
    
    print("-" * 70)
    
    # Print averages
    print(f"{'accuracy':<15} {'':<36} {accuracy:<12.4f} {total_support:<10}")
    print(f"{'macro avg':<15} {macro_precision:<12.4f} {macro_recall:<12.4f} {macro_f1:<12.4f} {total_support:<10}")
    print(f"{'weighted avg':<15} {weighted_precision:<12.4f} {weighted_recall:<12.4f} {weighted_f1:<12.4f} {total_support:<10}")

def evaluate_knn_model(model, X_test, y_test, model_type='classification'):
    """
    Evaluate KNN model performance.
    
    Args:
        model: Trained KNN model (classifier or regressor)
        X_test: Test features (numpy array)
        y_test: Test labels/values (numpy array or list)
        model_type: Type of model ('classification' or 'regression')
        
    Returns:
        dict: Dictionary with evaluation metrics
        
    Raises:
        ValueError: If model_type is not 'classification' or 'regression'
    """
    # Validate model_type
    if model_type not in ('classification', 'regression'):
        raise ValueError(f"Invalid model_type: {model_type}. Must be 'classification' or 'regression'")
    
    # Convert y_test to numpy array for consistent processing
    y_test = np.asarray(y_test)
    y_pred = model.predict(X_test)
    
    if model_type == 'classification':
        return {
            'accuracy': calculate_accuracy(y_test, y_pred),
            'confusion_matrix': calculate_confusion_matrix(y_test, y_pred),
            'classification_report': calculate_classification_report(y_test, y_pred)
        }
    else:  # regression
        return {
            'mse': calculate_mean_squared_error(y_test, y_pred),
            'r2_score': calculate_r2_score(y_test, y_pred),
            'mae': calculate_mae(y_test, y_pred)
        }