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
import matplotlib
matplotlib.use('TkAgg')
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
def ask_float(prompt: str , min_val: Optional[float] = None , max_val: Optional[float] = None) -> float:
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
            print("!Please enter the data type: Float")

# check user decisions
def ask_yes_no(prompt: str) -> bool:
    while True:
        
        raw = input(prompt).strip().lower()

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
    Y: np.ndarray

    feature_names: List[str]
    target_name: str 

def load_csv_dataset(path:  str , delimiter: str = ";") -> Dataset:
    """
    The function reads the csv file, checks if the number of rows and columns matches, and if any values are missing.
    If everything is correct, the values are stored for further training of the model.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"!File not found error: {path}")

    with open(path, "r" , encoding="utf-8") as f:
        reader = csv.reader(f , delimiter=delimiter)
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

    data = np.array(numeric_rows , dtype=float)
    return Dataset(data=data , columns=colums)    

def select_features_and_target(ds: Dataset) -> Prepareddata:
    """
    Lets user select target column and feature columns from dataset columns.
    """

    cols = ds.columns[:]
    print("\nColums:")
    for i , name in enumerate(cols , start=1):
        print(f"{i}) {name}")
    
    target_idx = ask_int("\nSelect TARGET colums number: " , 1 , len(cols)) - 1
    default_feaure_idxs = [i for i in range(len(cols)) if i != target_idx]

    print("\nDefault FEATURES are all columns except target:")
    print(", ".join(cols[i] for i in default_feaure_idxs))

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
                raise ValueError("!nvalid feature index list!")
        
        if target_idx in idxs:
            raise ValueError("!Target column cannot be included in features!")
        
        if len (idxs) < 1:
            raise ValueError("!You must select at least 1 feature!")
        
        feature_idxs = idxs

    else:

        feature_idxs = default_feaure_idxs

    X = ds.data[: , feature_idxs]
    Y = ds.data[: , target_idx]

    feature_names = [cols[i] for i in feature_idxs]
    target_name = cols[target_idx]

    return Prepareddata(
        X=X,
        Y=Y,
        feature_names=feature_names,
        target_name=target_name
    )

def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float, seed:int) -> Tuple[np.ndarray , np.ndarray , np.ndarray , np.ndarray]:

    """
    Random split with fixed seed for reproducibility.
    This is a standard practice in machine learning 
    that helps detect overfitting of the model.
    """
    if not(0.05 <= test_size <= 0.5):
        raise ValueError("!test_size should be between 0.05 and 0.5 for this tool!")
    
    # we take the number of examples
    n = X.shape[0]
    
    # creating a random number generator with a fixed code for reproducibility of results
    rng = np.random.RandomState(seed)
    
    # randomize indices
    idx = np.arange(n)
    rng.shuffle(idx)

    test_n = int(round(n * test_size))
    test_idx = idx[:test_n]

    train_idx = idx[test_n:]

    X_train = X[train_idx]
    y_train = y[train_idx]

    X_test = X[test_idx]
    y_test = y[test_idx]

    return X_train , X_test , y_train , y_test

def standardize_fit(X_train: np.ndarray) -> Tuple[np.ndarray , np.ndarray , np.ndarray]:
    """
    Fit standardization on TRAIN only:
    mean, std per feature.
    Returns scaled X_train + mean + std.
    """

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)

    # division by zero protection
    std_safe = np.where(std == 0.0 , 1.0, std)
    X_scaled = (X_train - mean) / std_safe

    return X_scaled , mean , std_safe

def standardize_apply(X: np.ndarray , mean: np.ndarray , std: np.ndarray) -> np.ndarray:
    return (X - mean) / std


# ============================================================
# ML LAYER (Linear Regression with Batch Gradient Descent)
# ============================================================

class LinearRegressionGD:
    """
    y_hat = X @ w + b
    Batch Gradient Descent minimizing MSE.
    """
    # Initialize the class (model) constructor
    def __init__(self , learning_rate: float = 0.05 , epochs: int = 2000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w: Optional[np.ndarray] = None
        self.b: float = 0.0
        self.loss_history: List[float] = []
    # model training method
    def fit(self , X: np.ndarray , y: np.ndarray) -> None:
        # Initializing initial weights
        n_samples , n_features = X.shape
        self.w = np.zeros(n_features , dtype=float)
        self.b = 0.0 
        self.loss_history = []
        
        # model training cycle
        for epoch in range(1 , self.epochs + 1):
            # prediction and error detection
            y_pred = X @ self.w + self.b
            errors = y_pred - y
            loss = float(np.mean(errors ** 2))
            
            # determination of new weights
            self.loss_history.append(loss)
            dw = (2.0 / n_samples) * (X.T @ errors)
            db = (2.0 / n_samples) * float(np.sum(errors))

            # weight change
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db 
    # method of making predictions         
    def predict(self , X: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("!Model is not trained yet!")
        
        return X @ self.w + self.b
    
# ============================================================
# METRICS & PLOTS
# ============================================================

# simple model performance testing functions
def mse(y_true: np.ndarray , y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))

def rmse(y_true: np.ndarray , y_pred: np.ndarray) -> float:
    return float(np.sqrt(mse(y_true , y_pred)))

def r2_score(y_true: np.ndarray , y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))

    if ss_tot == 0.0:
        return 0.0
    return 1.0 - (ss_res / ss_tot)
# This function plots the model's training process — 
# that is, how the error (loss) changed during training at each epoch
def plot_loss_curve(history: List[float]) -> None:
    plt.figure()
    plt.plot(np.arange(1 , len(history) + 1) , history)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.show()
    

def plot_true_vs_pred(y_true: np.ndarray , y_pred: np.ndarray , title: str = "True vs Predicted") -> None:
    
    plt.figure()
    plt.scatter(y_true , y_pred)
    
    mn = min(float(y_true.min()) , float(y_pred.min()))
    mx = max(float(y_true.max()) , float(y_pred.max()))

    plt.plot([mn , mx] , [mn , mx])

    plt.xlabel("y_true")
    plt.ylabel("y_pred")
    plt.title(title)
    plt.grid(True)
    plt.show()

def plot_1d_regression(x_raw: np.ndarray , y_true: np.ndarray , model: LinearRegressionGD , scaler_mean: Optional[np.ndarray] , scaler_std: Optional[np.ndarray],) -> None:
    """
    Plot:
    - scatter of raw x vs y
    - regression line using model, respecting scaling if enabled
    """

    # transforming the shape of an array
    x_raw = x_raw.reshape(-1 , 1)

    x_min , x_max = float(x_raw.min()) , float(x_raw.max())

    x_grid = np.linspace(x_min , x_max , 200).reshape(-1 , 1)
    
    # checking whether standardization (scaling) is used
    if scaler_mean is not None and scaler_std is not None:
        """
        If scaling is enabled
        The model was trained on: X_scaled
        Therefore, new values ​​must also be scaled.
        """
        x_grid_scaled = standardize_apply(x_grid , scaler_mean , scaler_std)
        x_scaled = standardize_apply(x_raw , scaler_mean , scaler_std)
    else:
        # if standardization is not enabled leave as is
        x_grid_scaled = x_grid
        x_scaled = x_raw

    y_line = model.predict(x_grid_scaled)
        
    plt.figure()
    plt.scatter(x_raw.flatten() , y_true)
    plt.plot(x_grid.flatten() , y_line)
    plt.xlabel("Feature (raw)")
    plt.ylabel("Target")
    plt.title("1D Regression: data points + fitted line")
    plt.grid(True)
    plt.show()

# ============================================================
# APP STATE + CLI MENUS
# ============================================================

@dataclass
class AppState:
    dataset: Optional[Dataset] = None
    prepareddata: Optional[Prepareddata] = None
    test_size: float = 0.2
    seed: int = 42
    use_scaling: bool = True
    learning_rate: float = 0.05
    epochs: int = 2000
    
    # split data (after features/target chosen)
    X_train: Optional[np.ndarray] = None
    X_test: Optional[np.ndarray] = None
    y_train: Optional[np.ndarray] = None
    y_test: Optional[np.ndarray] = None

    # scaler params (train-fitted)
    scaler_mean: Optional[np.ndarray] = None
    scaled_std: Optional[np.ndarray] = None

    model: Optional[LinearRegressionGD] = None
    
    # last evaluation
    last_mse: Optional[float] = None
    last_rmse: Optional[float] = None
    last_r2: Optional[float] = None

# this function determines which values ​​we have already defined and displays this information to the user
def print_status(s: AppState) -> None:
    ds = "none" if s.dataset is None else f"loaded ({s.dataset.data.shape[0]} rows , {s.dataset.data.shape[1]} cols)"
    sup = "none" if s.prepareddata is None else f"{len(s.prepareddata.feature_names)} features -> target '{s.prepareddata.target_name}'"
    trained = "no" if s.model is None else "yes"
    metrics = "none" if s.last_rmse is None else f"RMSE={s.last_rmse:.4f} | R2={s.last_r2:.4f}"
    
    print(f"Dataset: {ds}")
    print(f"Selection: {sup}")
    print(f"Split: test_size = {s.test_size}; seed =  {s.seed}")
    print(f"Scaling: {'ON' if s.use_scaling else 'OFF'}")
    print(f"Model: trained = {trained} (lr = {s.learning_rate} , epochs = {s.epochs})")
    print(f"Metrics: {metrics}")
    print("=" * 72)

def rebuild_split(s: AppState) -> None:
    """
    Build train/test split from supervised data, apply scaling if enabled.
    """
    if s.prepareddata is None:
        raise RuntimeError("!No features/target selection yet!")
    
    X , y = s.prepareddata.X , s.prepareddata.Y

    X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=s.test_size , seed = s.seed)

    if s.use_scaling:
        X_train_scaled , mean , std = standardize_fit(X_train)
        X_test_scaled = standardize_apply(X_test , mean , std)
        s.X_train , s.X_test = X_train_scaled , X_test_scaled
        s.scaler_mean , s.scaled_std = mean , std
    else:
        s.X_train , s.X_test = X_train , X_test
        s.scaler_mean , s.scaled_std = None , None

    s.y_train , s.y_test = y_train , y_test
    # reset model+metrics because the data pipeline changed
    s.model = None
    s.last_mse = s.last_rmse = s.last_r2 = None

def menu_data(s: AppState) -> None:
    """
    Menu for interacting with data loading functions, manually or from a file, viewing the contents 
    of the dataset, selecting features and target, and splitting into training and learning data
    """
    while True:
        clear_screen()
        print_header("Linear Regression Tool — Data")
        print_status(s)

        options = [
            "Load CSV dataset",
            "Manual input dataset",
            "Show dataset summary",
            "Select features + target",
            "Configure train/test split",
            "Back",
        ]

        choice = ask_choice("" , options)

        if choice == 0:
            path = input("CSV path: ").strip()
            try:
                s.dataset = load_csv_dataset(path , delimiter= ";")  
                s.prepareddata = None
                s.model = None
                s.last_mse = s.last_rmse = s.last_r2 = None
                print("Dataset loaded successfully.")
            except Exception as e:
                print(f"!Error: {e}!")
            pause()

        elif choice == 1:
            try:
                s.dataset = manual_input_dataset()
                s.prepareddata = None
                s.model = None
                s.last_mse = s.last_rmse = s.last_r2 = None
                print("Dataset loaded successfully.")
            except Exception as e:
                print(f"!Error: {e}!")
            pause()
        
        elif choice == 2:
            if s.dataset is None:
                print("!No dataset loaded!")
                pause()
                continue
            ds = s.dataset
            print("\nSummary:")
            print(f"Rows: {ds.data.shape[0]}")
            print(f"Colums: {ds.data.shape[1]}")

            for i , name in enumerate(ds.columns):
                col = ds.data[:,i]
                print(f"- {name}: min = {col.min():.4f} mean = {col.mean():.4f} max = {col.max():.4f}")
            pause()

        elif choice == 3:
            if s.dataset is None:
                print("!Load or create a dataset first!")
                pause()
                continue
            try:
                s.prepareddata = select_features_and_target(s.dataset)
                rebuild_split(s)
                print("\nSelection saved and train/test split rebuilt.")
            except Exception as e:
                print(f"!Error: {e}!")
            pause()

        elif choice == 4:
            s.test_size = ask_float("test_size (0.05-0.5): " , 0.05 , 0.5)
            s.seed = ask_int("seed (integer): ")

            if s.prepareddata is not None:
                try:
                    rebuild_split(s)
                    print("Split rebuilt.")
                except Exception as e:
                    print(f"!Error: {e}")
            pause()

        else:
            return

# menu for interacting with the model training functions, setting hyperparameters and training the model respectively      
def menu_train(s: AppState) -> None:
    while True:
        clear_screen()
        print_header("Linear Regression Tool — Train")
        print_status(s)

        options = [
            "Configure hyperparameters",
            "Train model",
            "Back",
        ]
        
        choice = ask_choice("" , options)

        if choice == 0:
            s.use_scaling = ask_yes_no("Enable standardization scaling? (y/n): ")
            s.learning_rate = ask_float("learning_rate (e.g. 0.01..0.2): ", 1e-6, 10.0)
            s.epochs = ask_int("epochs (e.g. 500..10000): ", 1, 1_000_000)

            if s.prepareddata is not None:
                try:
                    rebuild_split(s)
                    print("Data pipeline rebuilt with new settings.")
                except Exception as e:
                        print(f"! Error: {e}")

            pause()
        
        elif choice == 1:
            if s.prepareddata is None:
                print("!Select features + target first (Data menu)!")
                pause()
                continue
            if s.X_train is None or s.y_train is None:
                print("!Split data not ready!")
                pause()
                continue
            
            model = LinearRegressionGD(learning_rate=s.learning_rate , epochs=s.epochs)
            model.fit(s.X_train , s.y_train)

            s.model = model

            s.last_mse = s.last_rmse = s.last_r2 = None

            final_loss = model.loss_history[-1] if model.loss_history else None

            if final_loss is not None:
                print(f"Training finished. Final train MSE loss: {final_loss:.6f}")
            else:
                print("Training finished")
            pause()
        else:
            return

# menu for using functions to evaluate model performance, such as the mean square error         
def menu_evaluate(s: AppState) -> None:
    while True:
        clear_screen()
        print_header("Linear Regression Tool — Evaluate")
        print_status(s)

        options = [
            "Evaluate on test set",
            "Explain metrics",
            "Back",
        ]

        choice = ask_choice("" , options)

        if choice == 0:
            if s.model is None:
                print("!Train the model first!")
                pause()
                continue
            
            if s.X_test is None or s.y_test is None:
                print("!Test set not ready!")
                pause()
                continue

            y_pred = s.model.predict(s.X_test)
            s.last_mse = mse(s.y_test , y_pred)
            s.last_rmse = rmse(s.y_test , y_pred)
            s.last_r2 = r2_score(s.y_test , y_pred)

            print("\nTest metrics:")
            print(f"MSE : {s.last_mse:.6f}")
            print(f"RMSE : {s.last_rmse:.6f}")
            print(f"R^2 : {s.last_r2:.6f}")
            pause()
        
        elif choice == 1:
            print("\nMetric explanations (short):")
            print("- MSE  : average squared error (penalizes big errors strongly)")
            print("- RMSE : sqrt(MSE), error in the same units as target y (more intuitive)")
            print("- R^2  : how much variance is explained by the model (1.0 is perfect; <0 means worse than predicting mean)")
            pause()
        
        else:
            return

# menu to use the forecasting function, create forecasts on new, previously unseen data
def menu_predict(s: AppState) -> None:
    while True:
        clear_screen()
        print_header("Linear Regression Tool — Predict")
        print_status(s)

        options = [
            "Predict for ONE object (enter feature values)",
            "Back",
        ]

        choice = ask_choice("" , options)

        if choice == 0:
            if s.model is None or s.prepareddata is None:
                print("!Need trained model and selected features!")
                pause()
                continue
            
            vals = []

            print("\nEnter feature values:")

            for name in s.prepareddata.feature_names:
                v = ask_float(f"{name}: ")
                vals.append(v)
            
            X_new = np.array(vals ,dtype=float).reshape(1 , -1)

            if s.use_scaling and s.scaler_mean is not None and s.scaled_std is not None:
                X_new = standardize_apply(X_new , s.scaler_mean , s.scaled_std)

            y_hat = float(s.model.predict(X_new)[0])

            print(f"\nPredicted {s.prepareddata.target_name}: {y_hat:.6f}")

            pause()
        
        else:
            return

# menu for using the visualization function, building various graphs
def menu_visualize(s: AppState) -> None:
    while True:
        clear_screen()
        print_header("Linear Regression Tool - Visualize")
        print_status(s)

        options = [
            "Plot loss curve (needs trained model)",
            "Plot True vs Predicted (test set)",
            "Plot 1D regression (only if 1 feature)",
            "Back",
        ]
                
        choice = ask_choice("" , options)

        if choice == 0:
            if s.model is None:
                print("!Train model first!")
                pause()
                continue

            plot_loss_curve(s.model.loss_history)
            pause()

        elif choice == 1:
            if s.model is None or s.X_test is None or s.y_test is None:
                print("!Need trained model and test set!")
                pause()
                continue

            y_pred = s.model.predict(s.X_test)

            plot_true_vs_pred(s.y_test , y_pred , title="True vs Predicted (test set)")
            pause()

        elif choice == 2:
            if s.model is None or s.prepareddata is None:
                print("!Need trained model and selected features!")
                pause()
                continue
            if len(s.prepareddata.feature_names) != 1:
                print("!1D regression plot works only when you selected exactly 1 feature!")
                pause()
                continue
                
            x_raw = s.prepareddata.X[: , 0]
            y_true = s.prepareddata.Y
          
            plot_1d_regression(
                x_raw=x_raw,
                y_true=y_true,
                model=s.model,
                scaler_mean=s.scaler_mean,
                scaler_std=s.scaled_std,
            )
            pause()

        else:
            return

# "entry point" of the entire CLI tool
def main() -> None:
    state = AppState()
    
    while True:
        clear_screen()
        print_header("Linear Regression Tool (MVP)")
        print_status(state)

        options = [
            "Data",
            "Train",
            "Evaluate",
            "Predict",
            "Visualize",
            "Exit",
        ]

        choice = ask_choice("" , options)

        if choice == 0:
            menu_data(state)
        elif choice == 1:
            menu_train(state)
        elif choice == 2:
            menu_evaluate(state)
        elif choice == 3:
            menu_predict(state)
        elif choice == 4:
            menu_visualize(state)
        else:
            return
        # do not pause here; main.py already pauses after subprocess returns


if __name__ == "__main__":
    main()