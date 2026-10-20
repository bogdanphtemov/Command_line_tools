"""
Data handling utilities for KNN algorithm.
"""

import numpy as np
from typing import List, Dict, Any, Optional

def load_csv_dataset(csv_path: str, delimiter: str = ',') -> Dict[str, Any]:
    """
    Load dataset from CSV file.
    
    Args:
        csv_path: Path to CSV file
        delimiter: CSV delimiter character
        
    Returns:
        Dictionary with dataset information
    """
    data = np.genfromtxt(csv_path, delimiter=delimiter)
    
    return {
        'data': data,
        'filename': csv_path,
        'n_samples': data.shape[0],
        'n_features': data.shape[1] - 1  # Assume last column is target
    }

def manual_input_dataset() -> Dict[str, Any]:
    """
    Create dataset from manual input.
    
    Returns:
        Dictionary with dataset information
    """
    print("Enter data manually (one sample per line, comma-separated):")
    print("Example: 1.0,2.0,3.0,0 (where last value is target)")
    print("Enter 'done' when finished")
    
    samples = []
    while True:
        line = input("> ").strip()
        if line.lower() == 'done':
            break
        if line:
            try:
                values = [float(x.strip()) for x in line.split(',')]
                samples.append(values)
            except ValueError:
                print("Invalid input. Please enter numbers separated by commas.")
    
    if not samples:
        raise ValueError("No data entered")
    
    data = np.array(samples)
    return {
        'data': data,
        'filename': 'manual_input',
        'n_samples': data.shape[0],
        'n_features': data.shape[1] - 1
    }

def select_features_and_target(dataset: Dict[str, Any], target_col: Optional[int] = None) -> Dict[str, Any]:
    """
    Select target column and features from dataset.
    
    Args:
        dataset: Dataset dictionary
        target_col: Target column index (default: last column)
        
    Returns:
        Dictionary with prepared data
    """
    data = dataset['data']
    n_features = data.shape[1]
    
    if target_col is None:
        target_col = n_features - 1  # Default to last column
    
    # Validate target column
    if target_col < 0 or target_col >= n_features:
        raise ValueError(f"Target column must be between 0 and {n_features-1}")
    
    feature_cols = [i for i in range(n_features) if i != target_col]
    
    return {
        'X': data[:, feature_cols],
        'y': data[:, target_col],
        'feature_names': [f"Feature_{i+1}" for i in feature_cols],
        'target_name': f"Target_{target_col+1}",
        'dataset_info': dataset
    }

# For backward compatibility
def Dataset():
    """Placeholder for dataset class."""
    return dict

def Prepareddata():
    """Placeholder for prepared data class."""
    return dict