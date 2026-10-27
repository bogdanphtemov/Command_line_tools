"""
Data handling utilities for KNN algorithm.
"""

import numpy as np
from typing import Optional
from ...base.base_data import Dataset, Prepareddata

def load_csv_dataset(csv_path: str, delimiter: str = ',') -> Dataset:
    """
    Load dataset from CSV file.
    
    Args:
        csv_path: Path to CSV file
        delimiter: CSV delimiter character
        
    Returns:
        Dataset object with data and column names
        
    Raises:
        ValueError: If file not found, empty, contains missing data, or invalid format
    """
    try:
        data = np.genfromtxt(csv_path, delimiter=delimiter, dtype=float)
    except FileNotFoundError:
        raise ValueError(f"File not found: {csv_path}")
    except ValueError as e:
        # Handle string to float conversion errors (e.g., headers in CSV)
        if "could not convert string to float" in str(e):
            raise ValueError(f"CSV contains non-numeric data (possibly headers): {e}")
        else:
            raise ValueError(f"Failed to parse CSV data: {e}")
    except Exception as e:
        raise ValueError(f"Failed to load CSV: {e}")
    
    # Validate data shape and content
    if data.ndim != 2:
        raise ValueError(f"CSV must have at least 2 columns, got shape {data.shape}")
    if data.shape[0] == 0:
        raise ValueError("CSV is empty")
    
    # Check for NaN values and provide more specific error messages
    if np.any(np.isnan(data)):
        # Check first row for potential headers
        first_row_nan = np.any(np.isnan(data[0])) if data.shape[0] > 0 else False
        if first_row_nan:
            error_msg = """
CSV appears to contain header row or non-numeric data in the first row.
If your CSV has headers, please remove them or use a different loading method.
The first row was converted to NaN values.
            """.strip()
            raise ValueError(error_msg)
        else:
            error_msg = """
CSV contains missing or non-numeric values.
Please ensure all values are numeric and there are no empty cells.
Found NaN values in the dataset.
            """.strip()
            raise ValueError(error_msg)
    
    # Create default column names
    n_cols = data.shape[1]
    columns = [f"col_{i+1}" for i in range(n_cols)]
    
    return Dataset(data=data, columns=columns)

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
    
    Args:
        dataset: Dataset object
        target_col: Target column index (default: last column)
        
    Returns:
        Prepareddata object with features and target separated
        
    Raises:
        ValueError: If dataset is invalid, target_col is invalid type, or contains NaN values
    """
    data = dataset.data
    
    # Validate dataset
    if data.ndim != 2:
        raise ValueError(f"Dataset data must be 2D array, got shape {data.shape}")
    if data.shape[0] == 0:
        raise ValueError("Dataset is empty")
    if np.any(np.isnan(data)):
        raise ValueError("Dataset contains NaN values")
    
    n_cols = data.shape[1]
    
    # Validate and set default target column
    if target_col is None:
        target_col = n_cols - 1  # Default to last column
    else:
        # Validate target_col type
        if not isinstance(target_col, int):
            raise ValueError(f"Target column must be integer, got {type(target_col).__name__}")
    
    # Validate target column bounds
    if target_col < 0 or target_col >= n_cols:
        raise ValueError(f"Target column must be between 0 and {n_cols-1}")
    
    feature_cols = [i for i in range(n_cols) if i != target_col]
    
    # Use column names if available and correctly sized
    if (hasattr(dataset, 'columns') and dataset.columns and 
        len(dataset.columns) == n_cols):
        feature_names = [dataset.columns[i] for i in feature_cols]
        target_name = dataset.columns[target_col]
    else:
        # Fallback to default names if columns are missing or incorrect size
        feature_names = [f"Feature_{i+1}" for i in feature_cols]
        target_name = f"Target_{target_col+1}"
    
    return Prepareddata(
        X=data[:, feature_cols],
        Y=data[:, target_col],
        feature_names=feature_names,
        target_name=target_name
    )

# Export the main functionality
__all__ = [
    'Dataset',
    'Prepareddata',
    'load_csv_dataset',
    'manual_input_dataset', 
    'select_features_and_target'
]