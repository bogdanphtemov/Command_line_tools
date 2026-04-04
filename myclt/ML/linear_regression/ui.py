import numpy as np

from .app_state import AppState , print_status , rebuild_split
from .data import load_csv_dataset , manual_input_dataset , select_features_and_target 
from .preprocessing import standardize_apply
from .metrics import mse , rmse , r2_score
from .visualization import plot_loss_curve , plot_true_vs_pred , plot_1d_regression
from .core import LinearRegressionGD
from ...common.input_validation import ask_choice , ask_int , ask_float , ask_yes_no
from ...common.ui_helpers import clear_screen , print_header , pause

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
