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

from common.input_validation import ask_int, ask_yes_no


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


def load_csv_dataset(path: str, delimiter: str = ";") -> Dataset:
    """
    The function reads the csv file, checks if the number of rows and columns matches, 
    and if any values are missing. If everything is correct, the values are stored 
    for further training of the model.
    
    Args:
        path: Path to CSV file
        delimiter: CSV delimiter character (default: ";")
    
    Returns:
        Dataset object with loaded data
    
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If CSV format is invalid
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"!File not found error: {path}")

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

    numeric_rows = []
    for r_idx, row in enumerate(raw_data, start=2):
        
        if len(row) != n_cols:
            raise ValueError(f"!Row {r_idx} has {len(row)} columns, expected {n_cols}!")

        clean = []    

        for c_idx, cell in enumerate(row):
            cell = cell.strip()

            if cell == "":
                raise ValueError(f"!Empty value at row {r_idx}, col {c_idx+1} ({header[c_idx]})!")
            
            try:
                clean.append(float(cell))
            except ValueError:
                raise ValueError(f"!Non-numeric value at row {r_idx}, col {c_idx+1} ({header[c_idx]}): '{cell}'!")

        numeric_rows.append(clean)
    
    data = np.array(numeric_rows, dtype=float)
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
    
    target_idx = ask_int("\nSelect TARGET column number: ", 1, len(cols)) - 1
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

    X = ds.data[:, feature_idxs]
    Y = ds.data[:, target_idx]

    feature_names = [cols[i] for i in feature_idxs]
    target_name = cols[target_idx]

    return Prepareddata(
        X=X,
        Y=Y,
        feature_names=feature_names,
        target_name=target_name
    )