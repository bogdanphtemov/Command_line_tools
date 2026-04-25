import numpy as np

from .app_state import AppState , print_status , rebuild_split
from .data import load_csv_dataset , manual_input_dataset , select_features_and_target 
from .preprocessing import standardize_apply
from .metrics import mse , rmse , r2_score
from .visualization import plot_loss_curve , plot_true_vs_pred , plot_1d_regression
from .core import LinearRegressionGD
from .hyperparameter_tuning import grid_search_regularization
from common.input_validation import ask_choice , ask_int , ask_float , ask_yes_no
from common.ui_helpers import clear_screen , print_header , pause
from ML.model_storage import save_model , load_model , list_saved_models , delete_model

def menu_save_load(s: AppState) -> None:
    while True:
        clear_screen()
        print_header("Linear Regression Tool — Save/Load")
        print_status(s)

        options = [
            "Save trained model",
            "Load model",
            "List saved models",
            "Delete model",
            "Back",
        ]

        choice = ask_choice("" , options)

        if choice == 0:
            if s.model is None:
                print("!Model not loaded!")
                pause()
                continue

            if not s.model.is_trained:
                print("!Model is not trained yet!")
                pause()
                continue

            name = input("Enter model name (e.g. , 'my_lr_model'): ").strip()
            if not name:
                print("!Invalid name!")
                pause()
                continue

            try:
                filepath = f"./ml_models/{name}.pkl"
                save_model(s.model , filepath)
                print(f"Model '{name}' saved successfully.")
            except Exception as e:
                print(f"!Error: {e}!")
            pause()

        elif choice == 1:
            models = list_saved_models()
            if not models:
                print("!No saved models!")
                pause()
                continue

            print("\nAvailable models:")
            for i , model_name in enumerate(models , 1):
                print(f"{i}. {model_name}")
            
            try:
                idx = int(input("Select model number: ")) - 1
                if 0 <= idx < len(models):
                    filepath = f"./ml_models/{models[idx]}.pkl"
                    s.model = load_model(filepath)
                    print(f"Model '{models[idx]}' loaded successfully.")
                else:
                    print("!Invalid selection!")
            except ValueError:
                print("!Invalid input!")
            pause()
        
        elif choice == 2:
            models = list_saved_models()
            if not models:
                print("!No saved models!")
            else:
                print("\nSaved models: ")
                for name in models:
                    print(f" - {name}")
                    pause()

        elif choice == 3:
            models = list_saved_models()
            if not models:
                print("!No saved models!")
                pause()
                continue

            name = input("Enter model name to delete: ").strip()
            if name in models:
                confim = ask_yes_no(f"Delete model '{name}' (y/n)")
                if confim:
                    delete_model(f"./ml_models/{name}.pkl")
                    print("Model deleted.")
                else:
                    print("Cancelled.")
            else:
                print("!Model not found!")
                pause()
        
        else:
            return

        
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
                    print(f"!Error: {e}!")
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
            "Configure regularization",
            "Auto-tune regularization (Grid Search)",
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
                        print(f"!Error: {e}!")

            pause()
        
        elif choice == 1:
            clear_screen()
            print_header("Linear Regression Tool — Regularization")
            print_status(s)
            
            print("\nL1 Regularization (Lasso):")
            print("- Adds penalty: λ₁ × Σ|w|")
            print("- Can zero out unimportant features (feature selection)")
            print("- Use when: you suspect only some features matter\n")
            
            print("L2 Regularization (Ridge):")
            print("- Adds penalty: λ₂ × Σ(w²)")
            print("- Reduces all weights but doesn't zero them")
            print("- Use when: all features matter but you want smaller weights\n")
            
            s.use_l1 = ask_yes_no("Enable L1 (Lasso) regularization? (y/n): ")
            if s.use_l1:
                s.lambda_l1 = ask_float("L1 strength λ₁ (e.g. 0.001..0.1): ", 1e-6, 10.0)
            
            s.use_l2 = ask_yes_no("Enable L2 (Ridge) regularization? (y/n): ")
            if s.use_l2:
                s.lambda_l2 = ask_float("L2 strength λ₂ (e.g. 0.001..0.1): ", 1e-6, 10.0)
            
            if s.use_l1 or s.use_l2:
                reg_info = []
                if s.use_l1:
                    reg_info.append(f"L1(λ={s.lambda_l1})")
                if s.use_l2:
                    reg_info.append(f"L2(λ={s.lambda_l2})")
                print(f"\nRegularization configured: {' + '.join(reg_info)}")
            else:
                print("\nNo regularization enabled.")
            
            pause()
        
        elif choice == 2:
            # Auto-tune regularization via Grid Search
            if s.prepareddata is None:
                print("!Select features + target first (Data menu)!")
                pause()
                continue
            if s.X_train is None or s.y_train is None:
                print("!Split data not ready!")
                pause()
                continue
            
            clear_screen()
            print_header("Linear Regression Tool — Auto-tune Regularization")
            print("Grid Search with 5-Fold Cross-Validation")
            print("=" * 72)
            
            # Ask user about search ranges
            print("\nConfiguring grid search parameters...\n")
            
            # L1 search range
            use_l1_search = ask_yes_no("Include L1 (Lasso) in search? (y/n): ")
            
            # L2 search range
            use_l2_search = ask_yes_no("Include L2 (Ridge) in search? (y/n): ")
            
            if not use_l1_search and not use_l2_search:
                print("!At least one regularization type must be selected!")
                pause()
                continue
            
            
            # Generate grids based on user selection
            l1_grid = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0] if use_l1_search else [0.0]
            l2_grid = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0] if use_l2_search else [0.0]
            
            print(f"\nL1 grid : {l1_grid}")
            print(f"L2 grid : {l2_grid}")
            print(f"Total combinations: {len(l1_grid) * len(l2_grid)}\n")
            
            try:
                # Run grid search
                results = grid_search_regularization(
                    X=s.X_train,
                    y=s.y_train,
                    learning_rate=s.learning_rate,
                    epochs=s.epochs,
                    lambda_l1_grid=l1_grid,
                    lambda_l2_grid=l2_grid,
                    k_folds=5,
                    seed=s.seed,
                    verbose=False
                )
                
                # Display results
                print("\n" + "=" * 72)
                print("GRID SEARCH RESULTS")
                print("=" * 72)
                print(f"\nBest L1 (Lasso):  {results['best_lambda_l1']:.6f}")
                print(f"Best L2 (Ridge):  {results['best_lambda_l2']:.6f}")
                print(f"Best CV MSE:      {results['best_mse']:.6f}")
                
                # Show top 5 combinations 
                print("\nTop 5 Combinations:")
                for idx, result in enumerate(results['results'][:5], 1):
                    print(f"{idx}. L1={result['l1']:.6f} L2={result['l2']:.6f} | MSE={result['mean_mse']:.6f} ± {result['std_mse']:.6f}")
                
                # Ask if user wants to apply
                apply_result = ask_yes_no("\nApply these parameters to model? (y/n): ")
                if apply_result:
                    s.use_l1 = results['best_lambda_l1'] > 0
                    s.use_l2 = results['best_lambda_l2'] > 0
                    s.lambda_l1 = results['best_lambda_l1']
                    s.lambda_l2 = results['best_lambda_l2']
                    print("Regularization parameters updated.")
                else:
                    print("Parameters not applied.")
                
            except Exception as e:
                print(f"!Error during grid search: {e}!")
            
            pause()
        
        elif choice == 3:
            if s.prepareddata is None:
                print("!Select features + target first (Data menu)!")
                pause()
                continue
            if s.X_train is None or s.y_train is None:
                print("!Split data not ready!")
                pause()
                continue
            
            # Create model with regularization parameters
            model = LinearRegressionGD(
                learning_rate=s.learning_rate , 
                epochs=s.epochs,
                lambda_l1=s.lambda_l1 if s.use_l1 else 0.0,
                lambda_l2=s.lambda_l2 if s.use_l2 else 0.0
            )
            model.fit(s.X_train , s.y_train)

            s.model = model

            s.last_mse = s.last_rmse = s.last_r2 = None

            final_loss = model.loss_history[-1] if model.loss_history else None

            if final_loss is not None:
                print(f"Training finished. Final train loss: {final_loss:.6f}")
            else:
                print("Training finished.")
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
