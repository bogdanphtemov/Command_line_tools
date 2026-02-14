#/usr/bin/env python3

"""
linear_regression_cli.py

A practical CLI tool for Linear / Multiple Linear Regression using
Batch Gradient Descent (from scratch), with:
- CSV (delimiter ';') or manual input
- feature/target selection
- train/test split
- standardization (z-score)
- metrics: MSE, RMSE, R^2
- plots (matplotlib): 1D regression or y_true vs y_pred
"""

from __future__ import annotations 
import csv
import os
from dataclasses import dataclass
from typing import List , Optional , Tuple
import numpy as np
import matplotlib.pyplot as plt

###==========================================================================

# UI HELPERS (input validation, menus)

###==========================================================================

# cleaning the terminal
def clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")

# print the menu
def print_header(title: str) -> None:
    
    print("=" * 72)
    print(title.center(72))
    print("=" * 72)

# pause after displaying results
def pause(msg: str = "Press Enter to continue...") -> None:
    input(f"\n{msg}")

# Shows numbered choices and returns selected index (0-based)
def ask_choice(prompt: str , choices: List[str]) -> int:
    while True:

        print(prompt)

        for i , c in enumerate(choices , start=1):
            print(f"{i}) {c}")

        raw = input("Select a function: ").strip()

        try:
            
            val = int(raw)
            if 1 <= val <= len(choices):
                return val - 1
        
        except ValueError:
            pass
        print("!Invalid choice! Try again...\n")

# function for inputting integer parameters
def ask_int(prompt: str, min_val: Optional[int] = None , max_val: Optional[int] = None) -> int:
    while True:

        raw = input(prompt).strip()

        try :

            v = int(raw)
            if min_val is not None and v < min_val:
              print(f"! Must be >= {min_val}")
              continue
            if max_val is not None and v > max_val:
                print(f"! Must be <= {max_val}")
                continue
            return v
        
        except ValueError:
            print("!Please enter the data type: Integer")

# function for inputting flout parameters
def ask_flout(prompt: str , min_val: Optional[float] = None , max_val: Optional[float] = None) -> float:
    while True:

        raw = input(prompt).strip()

        try:

            v = float(raw)
            if min_val is not None and v < min_val:
              print(f"! Must be >= {min_val}")
              continue
            if max_val is not None and v > max_val:
                print(f"! Must be <= {max_val}")
                continue
            return v

        except ValueError: 
            print("!Please enter the data type: Flout")

# check user decisions
def ask_yes_no(promt: str) -> bool:
    while True:
        
        raw = input(promt).strip().lower()

        if raw in ("y" , "yes"):
            return True
        if raw in ("n" , "no"):
            return False

        print("!Please enter y/n:")


###==========================================================================

# DATA BLOCK (manual entry , data preparation)

###==========================================================================  

# Data storage object (has no logic)
@dataclass
class Dataset:
    """
    Holds a numeric table + column names.
    We keep the full table so user can choose features/target later.
    """

    data : np.ndarray
    columns: List[str]

# Data storage object (has no logic)
@dataclass
class Prepareddata:
    """
    Data prepared , features selected , target selected.
    """
    X: np.ndarray
    y: np.ndarray

    feature_names: List[str]
    target_name: str 

def load_csv_dataset(path:  str , dilimiter: str = ";") -> Dataset:
    """
    The function reads the csv file, checks if the number of rows and columns matches, and if any values are missing.
    If everything is correct, the values are stored for further training of the model.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"!File not found error: {path}")

    with open(path, "r" , encoding="utf-8") as f:
        reader = csv.reader(f , delimiter=dilimiter)
        rows = list(reader)

    if len (rows) < 2:
        raise ValueError("!CSV must contain a header row and at least 1 data row!")
    
    header = [i.strip() for i in rows[0]]
    if any(i == "" for i in header):
        raise ValueError("!Header contains empty column name(s)!")
    
    raw_data = rows[1:]
    n_cols = len(header)

    numeric_rows = []
    for r_idx , row in enumerate(raw_data , start=2):
        
        if len(row) != n_cols:
            raise ValueError(f"!Row {r_idx} has {len(row)} columns, expected {n_cols}!")

        clean = []    

        for c_idx , cell in enumerate(row):
            cell = cell.strip()

            if cell == "":
                raise ValueError(f"!Empty value at row {r_idx}, col {c_idx+1} ({header[c_idx]})!")
            
            try:

                clean.append(float(cell))

            except ValueError:
                raise ValueError(f"!Non-numeric value at row {r_idx}, col {c_idx+1} ({header[c_idx]}): '{cell}'!")

        numeric_rows.append(clean)
    data = np.array(numeric_rows , dtype=float)
    return Dataset(data = data , columns=header)

def manual_input_dataset() -> Dataset:
    """
    Manual mode:
    - user provides column names: x1;x2;y
    - then enters rows like: 1;2;3
    stops on empty line
    """
    
    print("Manual input mode")
    print("Enter colums names separated by ';' (example: area;rooms;price)")

    header_line = input("Colums: ").strip()
    colums = [x.strip() for x in header_line.split(";") if x.strip() != ""]

    if len(colums) < 2:
        raise ValueError("!You need at least 2 columns (features + target)!")

    print("/Now enter rows using the same delimiter ';'")
    print("Example row: 72;2;125")
    print("Enter empty line to finish...\n")

    numeric_rows = []
    n_cols = len(colums)
    line_no = 1

    while True:
        line = input(f"Row #{line_no}: ").strip()

        if line == "":
            break

        parts = [p.strip() for p in line.split(";")]

        if len(parts) != n_cols:
            print(f"!Expected {n_cols} values , got {len(parts)}! Try again...")
        
        try: 

            row = [float(p) for p in parts]

        except ValueError:
            print("!All values must be numeric! Try again...")
            continue

        numeric_rows.append(row)
        line_no += 1

    if len(numeric_rows) < 5:
        raise ValueError("!Manual dataset should have at least 5 rows for meaningful training!")

    data = np.array(numeric_rows , dtype=float)
    return Dataset(data=data , columns=colums)    