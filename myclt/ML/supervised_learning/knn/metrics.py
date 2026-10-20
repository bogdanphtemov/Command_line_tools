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
        y_true: True labels (numpy array)
        y_pred: Predicted labels (numpy array)
        
    Returns:
        float: Accuracy score (0.0 to 1.0)
        
    Note:
        Returns NaN for empty arrays (edge case)
    """
    if len(y_true) == 0:
        return np.nan
    return np.mean(y_true == y_pred)


def calculate_confusion_matrix(y_true, y_pred):
    """
    Calculate confusion matrix for classification.
    
    Args:
        y_true: True labels (numpy array)
        y_pred: Predicted labels (numpy array)
        
    Returns:
        dict: Confusion matrix as dictionary with class labels as keys
    """
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
        y_true: True values (numpy array)
        y_pred: Predicted values (numpy array)
        
    Returns:
        float: Mean Squared Error
        
    Note:
        Returns NaN for empty arrays (edge case)
    """
    if len(y_true) == 0:
        return np.nan
    return np.mean((y_true - y_pred) ** 2)


def calculate_r2_score(y_true, y_pred):
    """
    Calculate R² score for regression.
    
    Args:
        y_true: True values (numpy array)
        y_pred: Predicted values (numpy array)
        
    Returns:
        float: R² score
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    
    return 1 - (ss_res / ss_tot)


def calculate_mae(y_true, y_pred):
    """
    Calculate Mean Absolute Error for regression.
    
    Args:
        y_true: True values (numpy array)
        y_pred: Predicted values (numpy array)
        
    Returns:
        float: Mean Absolute Error
        
    Note:
        Returns NaN for empty arrays (edge case)
    """
    if len(y_true) == 0:
        return np.nan
    return np.mean(np.abs(y_true - y_pred))


def calculate_classification_report(y_true, y_pred):
    """
    Calculate comprehensive classification metrics per class.
    
    Args:
        y_true: True labels (numpy array)
        y_pred: Predicted labels (numpy array)
        
    Returns:
        dict: Classification metrics by class
    """
    unique_classes = np.unique(np.concatenate([y_true, y_pred]))
    
    report = {}
    
    for cls in unique_classes:
        # True positives, false positives, false negatives
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        
        # Precision, recall, F1-score
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
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
        y_true: True labels (numpy array)
        y_pred: Predicted labels (numpy array)
        
    Returns:
        None
    """
    report = calculate_classification_report(y_true, y_pred)
    
    # Print header
    print(f"\n{'Class':<10} {'Precision':<12} {'Recall':<10} {'F1':<10} {'Support':<10}")
    print("-" * 60)
    
    # Print metrics for each class
    for cls, metrics in report.items():
        print(f"{str(cls):<10} {metrics['precision']:<12.4f} "
              f"{metrics['recall']:<10.4f} {metrics['f1_score']:<10.4f} "
              f"{metrics['support']:<10}")
    

def evaluate_knn_model(model, X_test, y_test, model_type='classification'):
    """
    Evaluate KNN model performance.
    
    Args:
        model: Trained KNN model (classifier or regressor)
        X_test: Test features (numpy array)
        y_test: Test labels/values (numpy array)
        model_type: Type of model ('classification' or 'regression')
        
    Returns:
        dict: Dictionary with evaluation metrics
        
    Raises:
        ValueError: If model_type is not 'classification' or 'regression'
    """
    # Validate model_type
    if model_type not in ('classification', 'regression'):
        raise ValueError(f"Invalid model_type: {model_type}. Must be 'classification' or 'regression'")
    
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