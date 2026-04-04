import numpy as np
import matplotlib.pyplot as plt
from typing import List , Optional

from .core import LinearRegressionGD
from .preprocessing import standardize_apply

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
        

    y_line = model.predict(x_grid_scaled)
        
    plt.figure()
    plt.scatter(x_raw.flatten() , y_true)
    plt.plot(x_grid.flatten() , y_line)
    plt.xlabel("Feature (raw)")
    plt.ylabel("Target")
    plt.title("1D Regression: data points + fitted line")
    plt.grid(True)
    plt.show()