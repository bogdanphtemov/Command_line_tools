import numpy as np

from .app_state import AppState , print_status , rebuild_split
from .data import load_csv_dataset , manual_input_dataset , select_features_and_target 
from .preprocessing import standardize_apply
from .metrics import mse , rmse , r2_score
from .visualization import plot_loss_curve , plot_true_vs_pred , plot_1d_regression
from .core import LinearRegressionGD
from .hyperparameter_tuning import grid_search_regularization
from .session_adapter import LinearRegressionSessionAdapter
from common.input_validation import ask_choice , ask_int , ask_float , ask_yes_no
from common.ui_helpers import clear_screen , print_header , pause
from ML.session_storage import SessionStorage

def menu_save_load(s: AppState) -> None:
    """
    Save/Load complete ML sessions with:
      - Raw dataset (stored once, no duplication)
      - Feature/target selection
      - Split indices + train/test data
      - Trained model with all parameters
      - Hyperparameters (algorithm-specific)
      - Evaluation metrics
      - Preprocessing state (scaling params)
    
    Format: JSON + NPZ (safe, efficient, no pickle)
    """
    # Initialize storage and adapter
    storage = SessionStorage()
    adapter = LinearRegressionSessionAdapter()
    
    while True:
        clear_screen()
        print_header("Linear Regression Tool — Save/Load Session")
        print_status(s)

        options = [
            "Save complete session",
            "Load session",
            "List saved sessions",
            "Delete session",
            "Back",
        ]

        choice = ask_choice("" , options)

        if choice == 0:
            # SAVE SESSION
            if s.dataset is None:
                print("!No dataset loaded!")
                pause()
                continue
            
            if s.prepareddata is None:
                print("!Features/target not selected!")
                pause()
                continue

            if s.model is None or not s.model.is_trained:
                print("!Model not trained yet!")
                pause()
                continue

            session_name = input("Enter session name (e.g., 'housing_v1'): ").strip()
            if not session_name:
                print("!Invalid name!")
                pause()
                continue

            try:
                # Extract session data using adapter
                session_data, arrays_dict = adapter.extract(s)
                
                # Save session
                session_dir = f"./ml_sessions/{session_name}"
                storage.save_session(session_data, session_dir, arrays_dict, verbose=True)
                
                print(f"\n✓ Complete session '{session_name}' saved successfully!")
                print(f"  - Dataset: {s.dataset.data.shape}")
                print(f"  - Features: {len(s.prepareddata.feature_names)}")
                print(f"  - Model: trained")
                if s.metrics:
                    metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in s.metrics.items())
                    print(f"  - Metrics: {metrics_str}")
            
            except Exception as e:
                print(f"!Error saving session: {e}!")
            
            pause()

        elif choice == 1:
            # LOAD SESSION
            sessions = storage.list_sessions()
            if not sessions:
                print("!No saved sessions!")
                pause()
                continue

            print("\nAvailable sessions:")
            for i, session_name in enumerate(sessions, 1):
                print(f"{i}. {session_name}")
            
            try:
                idx = int(input("Select session number: ")) - 1
                if 0 <= idx < len(sessions):
                    session_dir = f"./ml_sessions/{sessions[idx]}"
                    
                    # Load session
                    session_data, arrays_dict = storage.load_session(session_dir, verbose=True)
                    
                    # Restore state using adapter (adapter will recreate model)
                    adapter.restore(session_data, arrays_dict, s)
                    
                    print(f"\n✓ Session '{sessions[idx]}' loaded successfully!")
                    print(f"  - Dataset: {s.dataset.data.shape}")
                    print(f"  - Features: {len(s.prepareddata.feature_names)}")
                    print(f"  - Model: Ready for predictions")
                    if s.metrics:
                        metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in s.metrics.items())
                        print(f"  - Metrics: {metrics_str}")
                else:
                    print("!Invalid selection!")
            
            except ValueError as e:
                print(f"!Invalid input: {e}!")
            except Exception as e:
                print(f"!Error loading session: {e}!")
            
            pause()
        
        elif choice == 2:
            # LIST SESSIONS
            sessions = storage.list_sessions()
            if not sessions:
                print("!No saved sessions!")
            else:
                print("\nSaved sessions: ")
                for name in sessions:
                    session_dir = f"./ml_sessions/{name}"
                    try:
                        _, arrays = storage.load_session(session_dir, verbose=False)
                        dataset_shape = arrays["dataset"].shape
                        print(f" ✓ {name} (dataset: {dataset_shape})")
                    except:
                        print(f" ? {name} (corrupted)")
            
            pause()

        elif choice == 3:
            # DELETE SESSION
            sessions = storage.list_sessions()
            if not sessions:
                print("!No saved sessions!")
                pause()
                continue

            session_name = input("Enter session name to delete: ").strip()
            if session_name in sessions:
                confirm = ask_yes_no(f"Delete session '{session_name}'? (y/n): ")
                if confirm:
                    session_dir = f"./ml_sessions/{session_name}"
                    storage.delete_session(session_dir, verbose=True)
                    print("Session deleted.")
                else:
                    print("Cancelled.")
            else:
                print("!Session not found!")
            
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
                s.metrics.clear()
                print("Dataset loaded successfully.")
            except Exception as e:
                print(f"!Error: {e}!")
            pause()

        elif choice == 1:
            try:
                s.dataset = manual_input_dataset()
                s.prepareddata = None
                s.model = None
                s.metrics.clear()
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
            
            # Ask about early stopping
            use_early_stopping = ask_yes_no("Use early stopping?", default=False)
            
            # Create model with regularization parameters
            model = LinearRegressionGD(
                learning_rate=s.learning_rate , 
                epochs=s.epochs,
                lambda_l1=s.lambda_l1 if s.use_l1 else 0.0,
                lambda_l2=s.lambda_l2 if s.use_l2 else 0.0
            )
            
            if use_early_stopping:
                # Split training data into train/validation for early stopping
                n_train = len(s.X_train)
                val_size = int(0.2 * n_train)
                X_train_part = s.X_train[val_size:]
                y_train_part = s.y_train[val_size:]
                X_val = s.X_train[:val_size]
                y_val = s.y_train[:val_size]
                patience = ask_int("Patience (epochs without improvement):", min_val=5, max_val=200, default=50)
                
                print(f"\nTraining with early stopping (patience={patience})...")
                model.fit_with_early_stopping(X_train_part, y_train_part, X_val, y_val, patience=patience, verbose=True)
            else:
                print("\nTraining without early stopping...")
                model.fit(s.X_train , s.y_train)

            s.model = model

            s.metrics.clear()

            final_loss = model.loss_history[-1] if model.loss_history else None

            if final_loss is not None:
                print(f"\nTraining finished. Final train loss: {final_loss:.6f}")
                print(f"Total epochs trained: {len(model.loss_history)}")
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
            s.metrics['mse'] = mse(s.y_test , y_pred)
            s.metrics['rmse'] = rmse(s.y_test , y_pred)
            s.metrics['r2'] = r2_score(s.y_test , y_pred)

            print("\n" + "="*70)
            print("TEST SET METRICS")
            print("="*70)
            print(f"\nMSE (Mean Squared Error)")
            print(f"  Value: {s.metrics['mse']:.6f}")
            print(f"  → Average of squared errors (lower is better)")
            print(f"  → Penalizes large errors strongly")
            
            print(f"\nRMSE (Root Mean Squared Error)")
            print(f"  Value: {s.metrics['rmse']:.6f}")
            print(f"  → Square root of MSE")
            print(f"  → In same units as target y (more intuitive)")
            
            print(f"\nR² (Coefficient of Determination)")
            print(f"  Value: {s.metrics['r2']:.6f}")
            print(f"  → Proportion of variance explained by model")
            print(f"  → 1.0 = perfect fit, 0.0 = same as predicting mean")
            print(f"  → <0 = worse than predicting mean")
            print("="*70)
            pause()
        
        elif choice == 1:
            clear_screen()
            print_header("Linear Regression Tool — Metrics Explained")
            print("\nMSE (Mean Squared Error):")
            print("─" * 70)
            print("  • What: Average of squared differences between predictions and actual values")
            print("  • Formula: MSE = Σ(y_actual - y_predicted)² / n")
            print("  • Lower is better: closer to 0 means better predictions")
            print("  • Downside: sensitive to outliers due to squaring")
            print("  • Use when: outliers are important/shouldn't be ignored")
            
            print("\nRMSE (Root Mean Squared Error):")
            print("─" * 70)
            print("  • What: Square root of MSE")
            print("  • Formula: RMSE = √MSE")
            print("  • Advantage: expressed in same units as target y (easier to interpret)")
            print("  • Example: if predicting house price in USD, RMSE is in USD")
            print("  • Use when: need interpretable error in original units")
            
            print("\nR² (Coefficient of Determination):")
            print("─" * 70)
            print("  • What: How well model explains variance in the data")
            print("  • Range: -∞ to 1.0")
            print("  • 1.0 = perfect predictions")
            print("  • 0.0 = predictions no better than predicting mean value")
            print("  • <0 = worse than predicting mean (model is failing)")
            print("  • Use when: need percentage of explained variance")
            
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
            y_true = s.prepareddata.y
          
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
