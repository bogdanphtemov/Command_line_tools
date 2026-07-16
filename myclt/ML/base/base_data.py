"""
Base data structures and utilities for all ML algorithms.

This module provides universal dataset handling that can be reused
by any supervised learning algorithm (Linear Regression, Logistic Regression, etc.).
"""

import csv
import os
from dataclasses import dataclass
from typing import List
import numpy as np

from myclt.common.input_validation import ask_int, ask_yes_no


@dataclass
class Dataset:
    """
    Holds a numeric table + column names.
    We keep the full table so user can choose features/target later.
    
    This is a universal data structure used by all ML algorithms.
    """

    data: np.ndarray
    columns: List[str]


@dataclass
class Prepareddata:
    """
    Data prepared with features selected and target selected.
    
    This is a universal data structure used by all ML algorithms.
    The algorithm-specific logic (regression vs classification) is handled
    in the model's training/validation code, not in the data structure.
    """
    X: np.ndarray
    Y: np.ndarray
    feature_names: List[str]
    target_name: str


def _detect_csv_delimiter(path: str) -> str:
    """
    Auto-detect CSV delimiter by reading the first line.
    Counts occurrences of ',' and ';' and picks the one with more splits.
    Defaults to ',' if both give the same result.
    
    Args:
        path: Path to CSV file
    
    Returns:
        Detected delimiter character (',' or ';')
    """
    with open(path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
    
    comma_count = first_line.count(",")
    semicolon_count = first_line.count(";")
    
    if semicolon_count > comma_count:
        return ";"
    return ","


def _try_parse_numeric(value: str):
    """
    Try to parse a string as float; return float if possible, else return original string.
    
    Args:
        value: String to parse
    
    Returns:
        float value if parseable, else original string
    """
    try:
        return float(value)
    except ValueError:
        return value


def _col_is_numeric(col_data) -> bool:
    """
    Check if all values in a column are numeric (float/int).
    
    Handles both Python lists and numpy array slices.
    
    Args:
        col_data: Iterable of column values (may be strings, floats, ints, or np.float64)
    
    Returns:
        True if all values are numeric (float, int, np.floating, np.integer),
        False if any value is a string
    """
    import numbers
    for v in col_data:
        if isinstance(v, str):
            return False
        if not isinstance(v, (numbers.Number, np.floating, np.integer)):
            return False
    return True


def load_csv_dataset(path: str, delimiter: str = None) -> Dataset:
    """
    The function reads the csv file, checks if the number of rows and columns matches, 
    and if any values are missing. If everything is correct, the values are stored 
    for further training of the model.
    
    NON-NUMERIC COLUMNS: If a column contains non-numeric values (e.g., string labels
    like 'decline', 'stable'), those values are kept as-is (as strings) in the data.
    Numeric columns are converted to float. This allows loading classification datasets
    with categorical target columns.
    
    Args:
        path: Path to CSV file
        delimiter: CSV delimiter character. If None, auto-detects between ',' and ';'.
    
    Returns:
        Dataset object with loaded data
    
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If CSV format is invalid
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"!File not found error: {path}")

    if delimiter is None:
        delimiter = _detect_csv_delimiter(path)

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=delimiter)
        rows = list(reader)

    if len(rows) < 2:
        raise ValueError("!CSV must contain a header row and at least 1 data row!")
    
    header = [i.strip() for i in rows[0]]
    if any(i == "" for i in header):
        raise ValueError("!Header contains empty column name(s)!")
    
    raw_data = rows[1:]
    n_cols = len(header)
    n_rows = len(raw_data)

    # Step 1: Parse all cells — try float, fall back to string
    # Store as list of lists (transposed: col-wise for easier column processing)
    parsed_cols: List[List] = [[] for _ in range(n_cols)]

    for r_idx, row in enumerate(raw_data, start=2):
        if len(row) != n_cols:
            raise ValueError(f"!Row {r_idx} has {len(row)} columns, expected {n_cols}!")

        for c_idx, cell in enumerate(row):
            cell = cell.strip()

            if cell == "":
                raise ValueError(f"!Empty value at row {r_idx}, col {c_idx+1} ({header[c_idx]})!")

            parsed_cols[c_idx].append(_try_parse_numeric(cell))

    # Step 2: For each column, check if all values are numeric
    # Build result matrix with appropriate dtypes
    result_columns = []
    non_numeric_columns = []

    for c_idx in range(n_cols):
        col = parsed_cols[c_idx]
        if _col_is_numeric(col):
            # Full numeric column — convert to float32 for efficiency
            result_columns.append(np.array(col, dtype=float))
        else:
            # Mixed or string column — keep as object dtype (can hold strings)
            result_columns.append(np.array(col, dtype=object))
            non_numeric_columns.append(header[c_idx])

    # Step 3: Combine into a single 2D array (column_stack)
    data = np.column_stack(result_columns)

    # Step 4: Report non-numeric columns (informational, not error)
    if non_numeric_columns:
        print(f"  ℹ Non-numeric column(s) detected: {', '.join(non_numeric_columns)}")
        print(f"    These columns will be kept as strings. Use them as TARGET for classification.")

    return Dataset(data=data, columns=header)


def manual_input_dataset() -> Dataset:
    """
    Manual mode:
    - user provides column names: x1;x2;y
    - then enters rows like: 1;2;3
    stops on empty line
    
    Returns:
        Dataset object with manually entered data
    
    Raises:
        ValueError: If input is invalid
    """
    print("Manual input mode")
    print("Enter column names separated by ';' (example: area;rooms;price)")

    header_line = input("Columns: ").strip()
    columns = [x.strip() for x in header_line.split(";") if x.strip() != ""]

    if len(columns) < 2:
        raise ValueError("!You need at least 2 columns (features + target)!")

    print("\nNow enter rows using the same delimiter ';'")
    print("Example row: 72;2;125")
    print("Enter empty line to finish...\n")

    numeric_rows = []
    n_cols = len(columns)
    line_no = 1

    while True:
        line = input(f"Row #{line_no}: ").strip()

        if line == "":
            break

        parts = [p.strip() for p in line.split(";")]

        if len(parts) != n_cols:
            print(f"!Expected {n_cols} values, got {len(parts)}! Try again...")
            continue
        
        try: 
            row = [float(p) for p in parts]
        except ValueError:
            print("!All values must be numeric! Try again...")
            continue

        numeric_rows.append(row)
        line_no += 1

    if len(numeric_rows) < 5:
        raise ValueError("!Manual dataset should have at least 5 rows for meaningful training!")

    data = np.array(numeric_rows, dtype=float)
    return Dataset(data=data, columns=columns)


def select_features_and_target(ds: Dataset) -> Prepareddata:
    """
    Lets user select target column and feature columns from dataset columns.
    
    This is a universal function that works for both regression and classification.
    The algorithm-specific validation (e.g., binary target for classification) 
    should be done in the model-specific UI code.
    
    Args:
        ds: Dataset object with loaded data
    
    Returns:
        Prepareddata object with selected features and target
    
    Raises:
        ValueError: If selection is invalid
    """
    cols = ds.columns[:]
    print("\nColumns:")
    for i, name in enumerate(cols, start=1):
        print(f"{i}) {name}")
    
    target_idx = ask_int("\nSelect TARGET column number", 1, len(cols)) - 1
    default_feature_idxs = [i for i in range(len(cols)) if i != target_idx]

    print("\nDefault FEATURES are all columns except target:")
    print(", ".join(cols[i] for i in default_feature_idxs))

    if ask_yes_no("Do you want to manually choose feature columns? (y/n): "):
        raw = input("Features: ").strip()
        parts = [p.strip() for p in raw.split(",") if p.strip() != ""]
        idxs = []
        
        for p in parts:
            try:
                val = int(p) - 1
                if not (0 <= val < len(cols)):
                    raise ValueError
                idxs.append(val)
            except ValueError:
                raise ValueError("!Invalid feature index list!")
        
        if target_idx in idxs:
            raise ValueError("!Target column cannot be included in features!")
        
        if len(idxs) < 1:
            raise ValueError("!You must select at least 1 feature!")
        
        feature_idxs = idxs
    else:
        feature_idxs = default_feature_idxs

    # Validate that feature columns are numeric (not strings)
    for idx in feature_idxs:
        if not _col_is_numeric([ds.data[r, idx] for r in range(ds.data.shape[0])]):
            raise ValueError(
                f"!Column '{cols[idx]}' contains non-numeric values! "
                f"Feature columns must be numeric. Use this column as TARGET instead."
            )

    X = ds.data[:, feature_idxs].astype(float, copy=False)
    Y = ds.data[:, target_idx]

    feature_names = [cols[i] for i in feature_idxs]
    target_name = cols[target_idx]

    # Report if target is non-numeric (for classification)
    if not _col_is_numeric([ds.data[r, target_idx] for r in range(ds.data.shape[0])]):
        print(f"  ℹ Target '{target_name}' contains categorical labels → classification mode")

    return Prepareddata(
        X=X,
        Y=Y,
        feature_names=feature_names,
        target_name=target_name
    )