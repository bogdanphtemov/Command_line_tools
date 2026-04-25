from dataclasses import dataclass
from typing import Optional 
import numpy as np

from .data import Dataset , Prepareddata
from .core import LinearRegressionGD
from .preprocessing import train_test_split , standardize_fit , standardize_apply

@dataclass
class AppState:
    dataset: Optional[Dataset] = None
    prepareddata: Optional[Prepareddata] = None
    test_size: float = 0.2
    seed: int = 42
    use_scaling: bool = True
    learning_rate: float = 0.05
    epochs: int = 2000
    
    # Regularization parameters
    use_l1: bool = False
    use_l2: bool = False
    lambda_l1: float = 0.01
    lambda_l2: float = 0.01
    
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
    
    # Regularization status
    reg_status = "OFF"
    if s.use_l1 or s.use_l2:
        reg_parts = []
        if s.use_l1:
            reg_parts.append(f"L1(λ={s.lambda_l1})")
        if s.use_l2:
            reg_parts.append(f"L2(λ={s.lambda_l2})")
        reg_status = " + ".join(reg_parts)
    
    print(f"Dataset: {ds}")
    print(f"Selection: {sup}")
    print(f"Split: test_size = {s.test_size}; seed =  {s.seed}")
    print(f"Scaling: {'ON' if s.use_scaling else 'OFF'}")
    print(f"Model: trained = {trained} (lr = {s.learning_rate} , epochs = {s.epochs})")
    print(f"Regularization: {reg_status}")
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
