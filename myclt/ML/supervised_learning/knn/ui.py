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
from myclt.common.input_validation import (
    ask_choice, ask_int, ask_float, ask_yes_no
)
from myclt.common.ui_helpers import clear_screen , print_header , pause
from myclt.ML.session_storage import SessionStorage

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
            "Clear current data",
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
            # Clear current data
            if ask_yes_no("Are you sure you want to clear all data?", False):
                s.dataset = None
                s.prepareddata = None
                s.X_train = None
                s.X_test = None
                s.y_train = None
                s.y_test = None
                s.train_idx = None
                s.test_idx = None
                s.model = None
                s.metrics.clear()
                print("✓ All data cleared")
            else:
                print("Clear cancelled")
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