"""
User Interface for K-Nearest Neighbors algorithm.
"""

import os
import shutil  # Moved from inside function for PEP 8 compliance
import traceback  # For better error reporting

from .app_state import AppState , print_status , rebuild_split
from .data import load_csv_dataset , manual_input_dataset , select_features_and_target, Dataset
from .metrics import evaluate_knn_model, print_classification_report
from .classifier import KNNClassifier
from .regressor import KNNRegressor
from .session_adapter import KNNSessionAdapter
from .visualization import (
    plot_classification_boundary_2d, plot_confusion_matrix,
    plot_class_probability_distributions, plot_regression_comparison
)
from .visualization import plot_true_vs_pred
from myclt.common.input_validation import (
    ask_choice, ask_int, ask_float, ask_yes_no, ask_yes_no_recommended
)
from myclt.common.ui_helpers import clear_screen , print_header , pause
from myclt.ML.session_storage import SessionStorage

import numpy as np

from myclt.ML.batch_predict import batch_predict_interactive
# Constants
SESSIONS_DIR = "./ml_sessions"


def menu_save_load(s: AppState) -> None:
    """Save/Load complete KNN sessions."""
    storage = SessionStorage()
    adapter = KNNSessionAdapter()
    
    while True:
        clear_screen()
        print_header("KNN Tool — Save/Load Session")
        print_status(s)
        
        options = [
            "Save session",
            "Load session",
            "List sessions",
            "Delete session",
            "Back"
        ]
        
        choice = ask_choice("Select action: ", options)
        
        if choice == 0:
            # Save
            try:
                # Check if there's something to save
                if s.model is None or not s.model.is_trained:
                    print("No trained model to save. Please train a model first.")
                    pause()
                    continue
                if s.prepareddata is None:
                    print("No data prepared. Please load and prepare data first.")
                    pause()
                    continue
                
                session_name = input("Session name: ").strip()
                if not session_name:
                    print("Invalid name")
                    pause()
                    continue
                
                # Extract session data
                session_data, arrays_dict = adapter.extract(s)
                
                # Save session
                storage.save_session(session_data, f"{SESSIONS_DIR}/{session_name}", arrays_dict)
                
                print(f"✓ Session '{session_name}' saved successfully!")
                
            except Exception as e:
                print(f"Error saving session: {e}")
                traceback.print_exc()
            pause()
        
        elif choice == 1:
            # Load
            try:
                # List available sessions
                sessions = storage.list_sessions(SESSIONS_DIR)
                if not sessions:
                    print("No sessions found. Please save a session first.")
                    pause()
                    continue
                
                print("Available sessions:")
                for i, session in enumerate(sessions, 1):
                    print(f"  {i}) {session}")
                
                while True:
                    try:
                        session_choice = ask_int("Select session to load (0 to cancel)", 0, len(sessions), default=None)
                        if session_choice == 0:
                            break
                        
                        selected_session = sessions[session_choice - 1]
                        
                        # Load session
                        session_data, arrays_dict = storage.load_session(f"{SESSIONS_DIR}/{selected_session}")
                        
                        # Restore to app state
                        adapter.restore(session_data, arrays_dict, s)
                        
                        # Verify restoration was successful
                        if s.model is None or s.prepareddata is None:
                            raise ValueError(f"Failed to restore session: model or prepared data is None")
                        
                        print(f"✓ Session '{selected_session}' loaded successfully!")
                        
                        # Option to continue with loaded session
                        if ask_yes_no("Do you want to continue with the loaded session?"):
                            break
                        else:
                            print("Session loaded but not applied.")
                            break
                            
                    except Exception as e:
                        print(f"Error loading session: {e}")
                        
            except Exception as e:
                print(f"Error: {e}")
                traceback.print_exc()
            pause()
        
        elif choice == 2:
            # List
            try:
                sessions = storage.list_sessions(SESSIONS_DIR)
                if not sessions:
                    print("No sessions found.")
                else:
                    print("Available sessions:")
                    for i, session in enumerate(sessions, 1):
                        print(f"  {i}) {session}")
            except Exception as e:
                print(f"Error listing sessions: {e}")
                traceback.print_exc()
            pause()
        
        elif choice == 3:
            # Delete
            try:
                sessions = storage.list_sessions(SESSIONS_DIR)
                if not sessions:
                    print("No sessions found.")
                    pause()
                    continue
                
                print("Available sessions:")
                for i, session in enumerate(sessions, 1):
                    print(f"  {i}) {session}")
                
                session_choice = ask_int("Select session to delete (0 to cancel)", 0, len(sessions), default=None)
                if session_choice == 0:
                    print("Deletion cancelled")
                else:
                    selected_session = sessions[session_choice - 1]
                    
                    if ask_yes_no(f"Are you sure you want to delete session '{selected_session}'?", False):
                        storage.delete_session(f"{SESSIONS_DIR}/{selected_session}")
                        print(f"✓ Session '{selected_session}' deleted successfully!")
                    else:
                        print("Deletion cancelled")
                        
            except Exception as e:
                print(f"Error deleting session: {e}")
                traceback.print_exc()
            pause()
        
        else:
            break


def menu_visualize(s: AppState) -> None:
    """Visualization menu for KNN."""
    while True:
        clear_screen()
        print_header("KNN Tool — Visualization")
        print_status(s)
        
        # Build visualization options based on model type and training status
        if s.model is None or not s.model.is_trained:
            print("No trained model available for visualization")
            pause()
            return
            
        if s.model_type == 'classification':
            options = [
                "Plot decision boundary (2D only)",
                "Plot confusion matrix (test set)", 
                "Plot probability distributions (test set)",
                "Back"
            ]
        else:  # regression
            options = [
                "Plot True vs Predicted (test set)",
                "Plot regression comparison (1D only)",
                "Back"
            ]
        
        choice = ask_choice("Select visualization: ", options)
        
        if choice == 0:
            if s.model_type == 'classification':
                # Plot decision boundary (2D only)
                if s.X_train is None or s.X_train.shape[1] != 2:
                    print("Decision boundary plot requires exactly 2 features")
                else:
                    try:
                        plot_classification_boundary_2d(
                            s.model, s.X_train, s.y_train, 
                            s.X_test, s.y_test,
                            title=f"KNN Classification Boundary (k={s.k})"
                        )
                    except Exception as e:
                        print(f"Error plotting decision boundary: {e}")
            else:
                # Plot True vs Predicted
                if s.X_test is None or s.y_test is None:
                    print("No test data available")
                else:
                    try:
                        y_pred = s.model.predict(s.X_test)
                        plot_true_vs_pred(
                            s.y_test, y_pred,
                            title=f"KNN Regression: True vs Predicted (k={s.k})"
                        )
                    except Exception as e:
                        print(f"Error plotting True vs Predicted: {e}")
            pause()
            
        elif choice == 1:
            if s.model_type == 'classification':
                # Plot confusion matrix
                if s.X_test is None or s.y_test is None:
                    print("No test data available")
                    pause()
                    continue
                
                y_pred = s.model.predict(s.X_test)
                metrics = evaluate_knn_model(s.model, s.X_test, s.y_test, 'classification')
                
                # Convert dict confusion matrix to numpy array
                cm_dict = metrics.get('confusion_matrix', {})
                if not cm_dict:
                    print("Could not compute confusion matrix")
                    pause()
                    continue
                
                # Get all unique classes to ensure consistent ordering
                all_classes = sorted(set(s.y_train) | set(s.y_test) | set(y_pred))
                n_classes = len(all_classes)
                cm_array = np.zeros((n_classes, n_classes), dtype=int)
                
                # Create mapping from class to index
                class_to_idx = {cls: idx for idx, cls in enumerate(all_classes)}
                
                # Fill the confusion matrix array
                for i, true_cls in enumerate(all_classes):
                    for j, pred_cls in enumerate(all_classes):
                        cm_array[i, j] = cm_dict.get(str(true_cls), {}).get(str(pred_cls), 0)
                
                # Get class names for display
                if hasattr(s.model, 'classes_'):
                    class_names = [str(cls) for cls in s.model.classes_]
                else:
                    class_names = [str(cls) for cls in all_classes]
                
                try:
                    plot_confusion_matrix(
                        cm_array, class_names,
                        title=f"KNN Confusion Matrix (k={s.k})"
                    )
                except Exception as e:
                    print(f"Error plotting confusion matrix: {e}")
                
            else:
                # Plot regression comparison (1D only)
                if s.X_train is None or s.X_train.shape[1] != 1:
                    print("Regression comparison plot requires exactly 1 feature")
                else:
                    try:
                        plot_regression_comparison(
                            s.model, s.X_train, s.y_train,
                            s.X_test, s.y_test,
                            title=f"KNN Regression (k={s.k})"
                        )
                    except Exception as e:
                        print(f"Error plotting regression comparison: {e}")
            pause()
            
        elif choice == 2:
            if s.model_type == 'classification':
                # Plot probability distributions
                if s.X_test is None or s.y_test is None:
                    print("No test data available")
                    pause()
                    continue
                
                # Get probabilities and predictions
                if hasattr(s.model, 'predict_proba'):
                    try:
                        probabilities = s.model.predict_proba(s.X_test)
                        
                        # Convert y_test to numeric indices for compatibility with plotting function
                        if hasattr(s.model, 'classes_'):
                            class_to_idx = {cls: idx for idx, cls in enumerate(s.model.classes_)}
                            y_test_numeric = np.array([class_to_idx[cls] for cls in s.y_test])
                            class_names = [str(cls) for cls in s.model.classes_]
                        else:
                            # Fallback: use original encoding if model doesn't have classes_
                            unique_classes = sorted(set(s.y_test))
                            class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
                            y_test_numeric = np.array([class_to_idx[cls] for cls in s.y_test])
                            class_names = [str(cls) for cls in unique_classes]
                        
                        plot_class_probability_distributions(
                            probabilities, y_test_numeric, class_names,
                            title=f"KNN Probability Distributions (k={s.k})"
                        )
                    except Exception as e:
                        print(f"Error plotting probability distributions: {e}")
                else:
                    print("Model does not support probability predictions")
            else:
                break  # regression has only 2 options
            pause()
            
        else:
            break


def show_prediction_example(feature_names):
    """
    Interactive single prediction prompt.

    Args:
        feature_names: List of feature column names

    Returns:
        Feature vector for prediction
    """
    print("\n" + "=" * 70)
    print("MAKE A PREDICTION")
    print("=" * 70)
    print("Enter values for each feature (press Enter for default 0.0):\n")

    values = []
    for feature_name in feature_names:
        while True:
            try:
                val_str = input(f"  {feature_name}: ").strip()
                if val_str == "":
                    print(f"    (using default: 0.0)")
                    values.append(0.0)
                    break
                value = float(val_str)
                values.append(value)
                break
            except ValueError:
                print("  Please enter a valid number.")

    return np.array(values, dtype=float)


def predict_single_interactive(s: AppState) -> None:
    """Make a single prediction."""
    if s.model is None or not s.model.is_trained:
        print("✗ No trained model!")
        return
    if s.prepareddata is None:
        print("✗ No feature names available!")
        return

    try:
        X_input = show_prediction_example(s.prepareddata.feature_names)

        # Apply scaling if needed
        if s.use_scaling and s.scaler_mean is not None and s.scaler_std is not None:
            X_input = X_input.reshape(1, -1)
            from .preprocessing import standardize_apply
            X_input = standardize_apply(X_input, s.scaler_mean, s.scaler_std)
        else:
            X_input = X_input.reshape(1, -1)

        # Predict
        pred = s.model.predict(X_input)
        
        print("\n" + "=" * 70)
        print("PREDICTION RESULT")
        print("=" * 70)

        if s.model_type == 'classification':
            class_name = pred[0]  # KNN already returns string labels directly
            print(f"Predicted class: {class_name}")
            # Show probabilities if available
            if hasattr(s.model, 'predict_proba'):
                probs = s.model.predict_proba(X_input)[0]
                print("\nClass probabilities:")
                for i, prob in enumerate(probs):
                    class_label = s.model.classes_[i]
                    print(f"  {class_label}: {prob:.4f}")
        else:
            print(f"Predicted value: {pred[0]:.6f}")

        print("=" * 70 + "\n")

    except Exception as e:
        print(f"✗ Prediction error: {e}")
        traceback.print_exc()
def menu_predict(s: AppState) -> None:
    """Predict menu."""
    while True:
        clear_screen()
        print_header("KNN — Predict")
        print_status(s)
        
        options = [
            "Make a single prediction",
            "Batch predict from CSV file",
            "Back"
        ]
        
        choice = ask_choice("Select action: ", options)
        
        if choice == 0:
            predict_single_interactive(s)
            pause()
        elif choice == 1:
            if s.model is None or not s.model.is_trained:
                print("✗ Model not trained!")
                pause()
                continue
            if s.prepareddata is None:
                print("✗ No features selected!")
                pause()
                continue

            csv_path = input("\nCSV path: ").strip()
            if not csv_path:
                print("✗ Invalid path!")
                pause()
                continue

            base_name = os.path.splitext(os.path.basename(csv_path))[0]
            default_output = f"predictions_{base_name}.csv"
            output_csv = input(f"Output [{default_output}]: ").strip() or default_output

            try:
                model_type = "multinomial" if s.model_type == "classification" else "regression"
                class_names = [str(c) for c in s.model.classes_] if s.model_type == "classification" else None
                
                batch_predict_interactive(
                    csv_path=csv_path,
                    model=s.model,
                    feature_names=s.prepareddata.feature_names,
                    use_scaling=s.use_scaling,
                    scaler_mean=s.scaler_mean,
                    scaler_std=s.scaler_std,
                    output_csv_path=output_csv,
                    model_type=model_type,
                    class_names=class_names,
                )
            except Exception as e:
                print(f"\n  ✗ Error: {e}")
            pause()
        else:
            break

def configure_split_interactive(s: AppState) -> None:
    """Configure train/test split parameters (test_size, seed, scaling)."""
    if s.dataset is None:
        print("✗ No dataset loaded yet!")
        return
    print("\n" + "=" * 70)
    print("CONFIGURE TRAIN/TEST SPLIT")
    print("=" * 70)
    s.test_size = ask_float("Test size (0.05-0.5)", min_val=0.05, max_val=0.5, default=0.2)
    s.seed = ask_int("Random seed (integer)", min_val=0, max_val=10000, default=42)
    s.use_scaling = ask_yes_no_recommended("Use feature scaling (standardization)?", recommended=True)

    if s.prepareddata is not None:
        rebuild_split(s)
        print("✓ Split configuration updated and split rebuilt")
    else:
        print("✓ Split configuration updated (select target to build the split)")


def menu_data(s: AppState) -> None:
    """Data handling menu."""
    while True:
        clear_screen()
        print_header("KNN Tool — Data Management")
        print_status(s)
        
        options = [
            "Load CSV dataset",
            "Manual data input",
            "Select target column",
            "Configure train/test split",
            "Back"
        ]
        
        choice = ask_choice("Select action: ", options)
        
        if choice == 0:
            # Load CSV dataset
            try:
                csv_path = input("Enter CSV file path: ").strip()
                if not csv_path:
                    print("No file path provided")
                    pause()
                    continue
                
                print("Loading dataset...")
                dataset = load_csv_dataset(csv_path)
                print(f"✓ Loaded dataset with {dataset.data.shape[0]} samples and {dataset.data.shape[1]} columns")
                
                # Set dataset
                s.dataset = dataset
                
                # Auto-select target column (last column by default)
                print("Auto-selecting target column (last column)...")
                target_col = dataset.data.shape[1] - 1 if dataset.data.shape[0] > 0 else 0
                prepareddata = select_features_and_target(dataset, target_col)
                s.prepareddata = prepareddata
                
                # Rebuild split with new data
                rebuild_split(s)
                print(f"✓ Features selected: {len(prepareddata.feature_names)} features, target: {prepareddata.target_name}")
                print("✓ Data split created")
                
            except Exception as e:
                print(f"Error loading CSV: {e}")
                traceback.print_exc()
            pause()
            
        elif choice == 1:
            # Manual data input
            try:
                print("Manual data input:")
                dataset = manual_input_dataset()
                print(f"✓ Created dataset with {dataset.data.shape[0]} samples and {dataset.data.shape[1]} columns")
                
                # Set dataset
                s.dataset = dataset
                
                # Auto-select target column (last column by default)
                target_col = dataset.data.shape[1] - 1 if dataset.data.shape[0] > 0 else 0
                prepareddata = select_features_and_target(dataset, target_col)
                s.prepareddata = prepareddata
                
                # Rebuild split with new data
                rebuild_split(s)
                print(f"✓ Features selected: {len(prepareddata.feature_names)} features, target: {prepareddata.target_name}")
                print("✓ Data split created")
                
            except Exception as e:
                print(f"Error in manual input: {e}")
                traceback.print_exc()
            pause()
            
        elif choice == 2:
            # Select target column
            if s.dataset is None:
                print("Error: No dataset loaded. Please load data first.")
                pause()
                continue
                
            try:
                print(f"Dataset has {s.dataset.data.shape[1]} columns")
                if hasattr(s.dataset, 'columns') and s.dataset.columns:
                    for i, col in enumerate(s.dataset.columns):
                        print(f"  {i}: {col}")
                else:
                    for i in range(s.dataset.data.shape[1]):
                        print(f"  {i}: col_{i+1}")
                
                target_col = ask_int("Select target column index", 0, s.dataset.data.shape[1] - 1)
                prepareddata = select_features_and_target(s.dataset, target_col)
                s.prepareddata = prepareddata
                
                # Rebuild split with new target selection
                rebuild_split(s)
                print(f"✓ Target column set to index {target_col}")
                print(f"✓ Features selected: {len(prepareddata.feature_names)} features, target: {prepareddata.target_name}")
                print("✓ Data split rebuilt")
                
            except Exception as e:
                print(f"Error selecting target column: {e}")
                traceback.print_exc()
            pause()
            
        elif choice == 3:
            # Configure train/test split
            configure_split_interactive(s)
            pause()
            
        else:
            break

def menu_train(s: AppState) -> None:
    """Model training."""
    # Check if data is prepared
    if s.prepareddata is None:
        print("Error: No data loaded. Please use 'Data' menu to load and prepare data first.")
        pause()
        return
    
    # Check if we need to rebuild split (no split available)
    if s.X_train is None or s.X_train.size == 0:
        try:
            rebuild_split(s)
            print("✓ Data split rebuilt successfully")
        except RuntimeError as e:
            print(f"Error rebuilding data split: {e}")
            pause()
            return
    
    # Validate split data
    if s.X_train is None or s.X_train.size == 0:
        print("Error: Data split is empty. Please check your data.")
        pause()
        return
    
    # Configure training parameters
    while True:
        clear_screen()
        print_header("KNN Tool — Model Training")
        print_status(s)
        
        print("\n→ Current configuration:")
        print(f"  Model type: {s.model_type}")
        print(f"  k: {s.k}")
        print(f"  Distance metric: {s.distance_metric}")
        if s.model_type == 'regression':
            print(f"  Weighted averaging: {s.weighted}")
        
        options = [
            "Configure parameters",
            "Train with current parameters",
            "Back"
        ]
        
        choice = ask_choice("Select action: ", options)
        
        if choice == 0:
            # Configure parameters
            clear_screen()
            print_header("KNN Tool — Configure Parameters")
            
            # Model type
            model_options = ["classification", "regression"]
            print("\nModel type:")
            for i, opt in enumerate(model_options):
                current = " (current)" if opt == s.model_type else ""
                print(f"  {i+1}) {opt}{current}")
            
            try:
                model_choice = ask_int("Select model type (1-2, or 0 to skip)", 0, 2, default=None)
                if model_choice and model_choice > 0:
                    s.model_type = model_options[model_choice - 1]
                    print(f"✓ Model type set to: {s.model_type}")
            except Exception as e:
                print(f"Error setting model type: {e}")
            
            # K value
            try:
                new_k = ask_int(f"k value [current: {s.k}]", 1, 100, default=s.k)
                s.k = new_k
                print(f"✓ k set to: {s.k}")
            except Exception as e:
                print(f"Error setting k: {e}")
            
            # Distance metric
            metric_options = ['euclidean', 'manhattan', 'minkowski']
            print("\nDistance metric:")
            for i, metric in enumerate(metric_options):
                current = " (current)" if metric == s.distance_metric else ""
                print(f"  {i+1}) {metric}{current}")
            
            try:
                metric_choice = ask_int("Select distance metric (1-3, or 0 to skip)", 0, 3, default=None)
                if metric_choice and metric_choice > 0:
                    s.distance_metric = metric_options[metric_choice - 1]
                    print(f"✓ Distance metric set to: {s.distance_metric}")
            except Exception as e:
                print(f"Error setting distance metric: {e}")
            
            # Weighted (for regression only)
            if s.model_type == 'regression':
                try:
                    weighted_choice = ask_yes_no("Use weighted averaging?") 
                    s.weighted = weighted_choice
                    print(f"✓ Weighted averaging: {'enabled' if s.weighted else 'disabled'}")
                except Exception as e:
                    print(f"Error setting weighted parameter: {e}")
            
            pause()
            
        elif choice == 1:
            # Train with current parameters
            try:
                # Validate app state before training
                s.validate()
                
                print("\n→ Training model...")
                
                # Create appropriate model based on type
                if s.model_type == 'classification':
                    model = KNNClassifier(k=s.k, distance_metric=s.distance_metric)
                else:  # regression
                    model = KNNRegressor(k=s.k, distance_metric=s.distance_metric, weighted=s.weighted)
                
                # Fit the model
                model.fit(s.X_train, s.y_train)
                s.model = model
                
                # Clear previous metrics
                s.metrics.clear()
                
                print(f"✓ Model trained successfully!")
                print(f"  - Type: {s.model_type.title()}")
                print(f"  - k: {s.k}")
                print(f"  - Distance metric: {s.distance_metric}")
                if s.model_type == 'regression':
                    print(f"  - Weighted: {s.weighted}")
                print(f"  - Training samples: {len(s.X_train)}")
                
            except Exception as e:
                print(f"Error training model: {e}")
                traceback.print_exc()
            
            pause()
            break
            
        else:
            break

def menu_evaluate(s: AppState) -> None:
    """Model evaluation."""
    while True:
        clear_screen()
        print_header("KNN Tool — Model Evaluation")
        print_status(s)
        
        options = [
            "Evaluate on test data",
            "Show metrics summary",
            "Back"
        ]
        
        choice = ask_choice("Select action: ", options)
        
        if choice == 0:
            # Check if model is trained
            if s.model is None or not s.model.is_trained:
                print("Error: No trained model. Please use 'Train' menu to train a model first.")
                pause()
                continue
            
            # Check if test data is available
            if s.X_test is None or s.X_test.size == 0:
                print("Error: No test data available. Please ensure data is properly split.")
                pause()
                continue
            
            try:
                clear_screen()
                print_header("KNN Tool — Model Evaluation")
                print_status(s)
                
                print("\n→ Evaluating model on test data...")
                
                # Evaluate the model
                metrics = evaluate_knn_model(s.model, s.X_test, s.y_test, s.model_type)
                
                # Store only scalar metrics for JSON serialization
                if s.model_type == 'classification':
                    s.metrics = {'accuracy': float(metrics.get('accuracy', 0.0))}
                    
                    print(f"\n✓ Model Evaluation Results (Classification):")
                    print(f"  Accuracy: {metrics.get('accuracy', 0.0):.4f}")
                    
                    # Print detailed classification report
                    predictions = s.model.predict(s.X_test)
                    print_classification_report(s.y_test, predictions)
                    
                    confusion_matrix = metrics.get('confusion_matrix', {})
                    if confusion_matrix and len(confusion_matrix) > 0:
                        print(f"\n  Confusion Matrix:")
                        # Convert to matrix for display
                        classes = sorted(confusion_matrix.keys())
                        if len(classes) <= 2:
                            # Simple display for binary classification
                            print("    True\\Pred    ", end="")
                            for cls in classes:
                                print(f"{cls}", end="\t")
                            print()
                            
                            for true_cls in classes:
                                print(f"    {true_cls}", end="\t\t")
                                for pred_cls in classes:
                                    print(f"{confusion_matrix[true_cls].get(pred_cls, 0)}", end="\t")
                                print()
                        else:
                            # Display for multi-class classification
                            # Print header
                            print("    " + "".join(f"{cls:>8}" for cls in classes))
                            # Print matrix rows
                            for true_cls in classes:
                                print(f"    {true_cls:>4}" + "".join(f"{confusion_matrix[true_cls].get(pred_cls, 0):>8}" for pred_cls in classes))
                            print()  # Add newline after matrix
                
                else:  # regression
                    s.metrics = {k: float(v) for k, v in metrics.items()}
                    
                    print(f"\n✓ Model Evaluation Results (Regression):")
                    print(f"  Mean Squared Error (MSE): {metrics.get('mse', 0.0):.4f}")
                    print(f"  R² Score: {metrics.get('r2_score', 0.0):.4f}")
                    print(f"  Mean Absolute Error (MAE): {metrics.get('mae', 0.0):.4f}")
                
                print(f"\n  Test samples: {len(s.X_test)}")
                print("\n✓ Evaluation completed and metrics saved")
                
            except Exception as e:
                print(f"Error evaluating model: {e}")
                traceback.print_exc()
            
            pause()
            
        elif choice == 1:
            # Show metrics summary
            if not s.metrics:
                print("No metrics available. Please evaluate the model first.")
            else:
                clear_screen()
                print_header("KNN Tool — Metrics Summary")
                print_status(s)
                
                print("\n→ Current metrics:")
                for key, value in s.metrics.items():
                    if isinstance(value, dict) and len(value) > 10:
                        print(f"  {key}: <complex structure>")
                    elif isinstance(value, float):
                        print(f"  {key}: {value:.4f}")
                    else:
                        print(f"  {key}: {value}")
                
            pause()
            
        else:
            break