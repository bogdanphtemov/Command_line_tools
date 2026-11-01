"""
Data handling utilities for KNN algorithm.
"""

import numpy as np
from typing import Optional
from ...base.base_data import Dataset, Prepareddata

def load_csv_dataset(csv_path: str, delimiter: str = None) -> Dataset:
    """
    Load dataset from CSV file.
    
    This function uses the universal CSV loader from base_data.py which:
    - Automatically detects headers and uses them as column names
    - Preserves text columns (no conversion to numeric codes)
    - Handles mixed numeric/text data
    
    Args:
        csv_path: Path to CSV file
        delimiter: CSV delimiter character (None = auto-detect)
        
    Returns:
        Dataset object with data and column names
        
    Raises:
        ValueError: If file not found, empty, contains missing data, or invalid format
    """
    # Re-export the function from base_data to ensure consistent behavior across all ML algorithms
    from ...base.base_data import load_csv_dataset as base_load_csv_dataset
    return base_load_csv_dataset(csv_path, delimiter)

def manual_input_dataset() -> Dataset:
    """
    Create dataset from manual input.
    
    Returns:
        Dataset object with data and column names
        
    Raises:
        ValueError: If no data entered or inconsistent dimensions
    """
    max_samples = 1000  # Prevent excessive input
    print("Enter data manually (one sample per line, comma-separated):")
    print(f"Example: 1.0,2.0,3.0,0 (where last value is target)")
    print(f"Maximum {max_samples} samples. Enter 'done' when finished")
    
    samples = []
    expected_cols = None
    
    while len(samples) < max_samples:
        line = input("> ").strip()
        if line.lower() == 'done':
            break
        if not line:
            continue
            
        try:
            values = [float(x.strip()) for x in line.split(',')]
        except ValueError:
            print("Invalid input. Please enter numbers separated by commas.")
            continue
        
        # Validate consistent number of columns
        if expected_cols is None:
            expected_cols = len(values)
            if expected_cols < 2:
                print("Error: Each sample must have at least 2 values (at least 1 feature + target)")
                continue
        elif len(values) != expected_cols:
            print(f"Error: Expected {expected_cols} values, got {len(values)}. Please try again.")
            continue
        
        samples.append(values)
    
    if len(samples) >= max_samples:
        print(f"Maximum sample limit reached ({max_samples}). Stopping input.")
    
    if not samples:
        raise ValueError("No data entered")
    
    data = np.array(samples)
    
    # Create default column names
    n_cols = data.shape[1]
    columns = [f"col_{i+1}" for i in range(n_cols)]
    
    return Dataset(data=data, columns=columns)

def select_features_and_target(dataset: Dataset, target_col: Optional[int] = None) -> Prepareddata:
    """
    Select target column and features from dataset.
    
    This function uses the universal feature/target selector from base_data.py which:
    - Allows interactive selection of target and feature columns
    - Validates that feature columns are numeric
    - Supports text targets for classification
    
    Args:
        dataset: Dataset object
        target_col: Target column index (optional - will ask interactively if None)
        
    Returns:
        Prepareddata object with features and target separated
        
    Raises:
        ValueError: If selection is invalid or dataset has issues
    """
    # For KNN, we want to allow automatic selection when target_col is provided
    # but fall back to interactive selection when parameter is None
    if target_col is not None:
        # Manual target column selection - adapt base logic
        data = dataset.data
        
        # Validate dataset
        if data.ndim != 2:
            raise ValueError(f"Dataset data must be 2D array, got shape {data.shape}")
        if data.shape[0] == 0:
            raise ValueError("Dataset is empty")
        
        n_cols = data.shape[1]
        
        # Validate target column bounds
        if target_col < 0 or target_col >= n_cols:
            raise ValueError(f"Target column must be between 0 and {n_cols-1}")
        
        feature_cols = [i for i in range(n_cols) if i != target_col]
        
        # Validate that feature columns are numeric
        from ...base.base_data import _col_is_numeric
        for idx in feature_cols:
            if not _col_is_numeric([data[r, idx] for r in range(data.shape[0])]):
                col_name = dataset.columns[idx] if hasattr(dataset, 'columns') and len(dataset.columns) > idx else f"column_{idx}"
                raise ValueError(
                    f"Feature column '{col_name}' contains non-numeric values! "
                    f"Feature columns must be numeric. Use this column as TARGET instead."
                )
        
        # Use column names if available and correctly sized
        if (hasattr(dataset, 'columns') and dataset.columns and 
            len(dataset.columns) == n_cols):
            feature_names = [dataset.columns[i] for i in feature_cols]
            target_name = dataset.columns[target_col]
        else:
            # Fallback to default names if columns are missing or incorrect size
            feature_names = [f"Feature_{i+1}" for i in feature_cols]
            target_name = f"Target_{target_col+1}"
        
        X = data[:, feature_cols].astype(float, copy=False)
        Y = data[:, target_col]
        
        return Prepareddata(
            X=X,
            Y=Y,
            feature_names=feature_names,
            target_name=target_name
        )
    else:
        # Use interactive selection from base_data when target_col is None
        from ...base.base_data import select_features_and_target as base_select_features_and_target
        return base_select_features_and_target(dataset)

# Export the main functionality
__all__ = [
    'Dataset',
    'Prepareddata',
    'load_csv_dataset',
    'manual_input_dataset', 
    'select_features_and_target'
]