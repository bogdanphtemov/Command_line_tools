"""
User interface helpers for Multinomial Logistic Regression workflow.

Provides interactive prompts and data selection dialogs for multiclass classification.
Mirrors the binary UI pattern from ui.py but adapted for K > 2 classes.
"""

import numpy as np
from typing import List, Tuple, Optional

from .multinomial_app_state import MultinomialAppState, print_status, rebuild_split
from .data import Dataset, Prepareddata, load_csv_dataset, manual_input_dataset
from .core import MultinomialLogisticRegression
from .preprocessing import standardize_apply
from .metrics import (
    accuracy, multiclass_precision, multiclass_recall, multiclass_f1_score,
    multiclass_confusion_matrix, print_multiclass_classification_report
)
from .visualization import (
    plot_loss_curve, plot_multiclass_confusion_matrix, plot_multiclass_feature_importance,
    plot_multiclass_probability_heatmap, plot_class_probability_distributions
)
from .session_adapter import MultinomialSessionAdapter
from ML.session_storage import SessionStorage
from common.input_validation import ask_yes_no, ask_int, ask_float, ask_choice
from common.ui_helpers import clear_screen, print_header, pause


def select_features_and_target_multiclass(dataset: Dataset) -> Prepareddata:
    """
    Interactive feature and target selection for multiclass classification.
    
    Uses the universal select_features_and_target from base_data,
    validates that target has at least 2 classes, and remaps to 0..K-1.
    
    Args:
        dataset: Loaded dataset
    
    Returns:
        Prepareddata with selected X and y (labels remapped to 0..K-1)
    """
    from ..base.base_data import select_features_and_target as universal_select_features_and_target
    
    prepared = universal_select_features_and_target(dataset)
    
    # Validate target has at least 2 classes
    unique_values = np.unique(prepared.Y)
    if len(unique_values) < 2:
        raise ValueError(f"Target must contain at least 2 classes, found {len(unique_values)}")
    
    # REMAP labels from arbitrary values (e.g., 1,2,3) to 0..K-1
    # This is CRITICAL because the model internally uses 0..K-1 for one-hot encoding,
    # and predict() returns 0..K-1 without inverse mapping.
    label_mapping = {old: new for new, old in enumerate(sorted(unique_values))}
    prepared.Y = np.array([label_mapping[val] for val in prepared.Y])
    
    print(f"\n✓ Found {len(unique_values)} classes: {sorted(unique_values.tolist())}")
    print(f"  Original labels remapped to 0..{len(unique_values)-1} for training")
    print(f"  Class distribution: {dict(zip(*np.unique(prepared.Y.astype(int), return_counts=True)))}")
    
    return prepared


def configure_multinomial_model_hyperparameters() -> Tuple[float, int, float]:
    """
    Interactive model hyperparameter configuration for multinomial model.
    
    Returns:
        Tuple of (learning_rate, epochs, lambda_l2)
    """
    print("\n" + "=" * 70)
    print("MULTINOMIAL MODEL HYPERPARAMETER CONFIGURATION")
    print("=" * 70)
    
    learning_rate = ask_float("Learning rate (default 0.01, range 0.001-0.1):", min_val=0.001, max_val=0.1, default=0.01)
    epochs = ask_int("Number of epochs (default 1000, range 100-10000):", min_val=100, max_val=10000, default=1000)
    lambda_l2 = ask_float("L2 regularization strength (default 0.0):", min_val=0.0, max_val=1.0, default=0.0)
    
    print("\n" + "=" * 70)
    print(f"Configuration: lr={learning_rate}, epochs={epochs}, λ2={lambda_l2}")
    print("=" * 70)
    
    return learning_rate, epochs, lambda_l2


def show_prediction_example_multiclass(feature_names: List[str]) -> np.ndarray:
    """
    Interactive single prediction example for multiclass.
    
    Args:
        feature_names: List of feature column names
    
    Returns:
        Feature vector for prediction
    """
    print("\n" + "=" * 70)
    print("MAKE A PREDICTION")
    print("=" * 70)
    print("Enter values for each feature (press Enter for default):\n")
    
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


def detect_class_names(s: MultinomialAppState) -> None:
    """
    Auto-detect and optionally rename class names based on dataset.
    
    If the target column has few unique values and they look like labels,
    use them as class names. Otherwise prompt user.
    """
    if s.prepareddata is None:
        return
    
    unique_vals = np.unique(s.prepareddata.Y)
    
    # By default use "Class 0", "Class 1", etc.
    s.class_names = [f"Class {int(v)}" for v in sorted(unique_vals)]
    
    # Ask user if they want custom names
    if ask_yes_no(f"Use default class names ({', '.join(s.class_names)})?", default=True):
        return
    
    print("\nEnter custom names for each class:")
    s.class_names = []
    for v in sorted(unique_vals):
        name = input(f"  Class {int(v)}: ").strip()
        s.class_names.append(name if name else f"Class {int(v)}")


def load_data_interactive_multinomial(s: MultinomialAppState) -> None:
    """Load data for multinomial classification."""
    print("\n" + "=" * 70)
    print("LOAD DATA")
    print("=" * 70)
    
    print("\n1. Load from CSV file")
    print("2. Manual input")
    
    choice = ask_choice("", ["Load CSV", "Manual input"])
    if choice == 0:
        print("\nEnter CSV file path:")
        file_path = input("> ").strip()
        try:
            s.dataset = load_csv_dataset(file_path)
            print(f"✓ Loaded dataset: {s.dataset.data.shape[0]} rows × {s.dataset.data.shape[1]} columns")
        except FileNotFoundError as e:
            print(f"✗ Error: {e}")
    else:
        try:
            s.dataset = manual_input_dataset()
            print(f"✓ Created dataset: {s.dataset.data.shape[0]} rows × {s.dataset.data.shape[1]} columns")
        except ValueError as e:
            print(f"✗ Error: {e}")


def select_features_interactive_multinomial(s: MultinomialAppState) -> None:
    """Select features and target for multiclass."""
    if s.dataset is None:
        print("✗ No dataset loaded yet!")
        return
    try:
        s.prepareddata = select_features_and_target_multiclass(s.dataset)
        rebuild_split(s)
        detect_class_names(s)
        print("✓ Features and target selected")
    except ValueError as e:
        print(f"✗ Error: {e}")


def configure_split_interactive_multinomial(s: MultinomialAppState) -> None:
    """Configure train/test split."""
    if s.dataset is None:
        print("✗ No dataset loaded yet!")
        return
    print("\n" + "=" * 70)
    print("CONFIGURE TRAIN/TEST SPLIT")
    print("=" * 70)
    s.test_size = ask_float("Test set size (0.05-0.5):", min_val=0.05, max_val=0.5, default=0.2)
    s.seed = ask_int("Random seed:", min_val=0, max_val=10000, default=42)
    s.use_scaling = ask_yes_no("Use feature scaling (standardization)?", default=True)
    rebuild_split(s)
    print("✓ Split configuration updated")


def configure_model_interactive_multinomial(s: MultinomialAppState) -> None:
    """Configure model hyperparameters."""
    print("\n" + "=" * 70)
    print("CONFIGURE MODEL")
    print("=" * 70)
    s.learning_rate, s.epochs, s.lambda_l2 = configure_multinomial_model_hyperparameters()
    print("✓ Model configured")


def train_model_interactive_multinomial(s: MultinomialAppState) -> None:
    """Train multinomial model interactively."""
    if s.X_train is None or s.y_train is None:
        print("✗ No training data prepared yet!")
        return
    print("\n" + "=" * 70)
    print("TRAINING MULTINOMIAL MODEL")
    print("=" * 70)
    try:
        s.model = MultinomialLogisticRegression(
            learning_rate=s.learning_rate,
            epochs=s.epochs,
            lambda_l2=s.lambda_l2
        )
        use_early_stopping = ask_yes_no("Use early stopping?", default=False)
        if use_early_stopping:
            n_train = len(s.X_train)
            val_size = int(0.2 * n_train)
            X_train_part = s.X_train[val_size:]
            y_train_part = s.y_train[val_size:]
            X_val = s.X_train[:val_size]
            y_val = s.y_train[:val_size]
            patience = ask_int("Patience (epochs without improvement):", min_val=5, max_val=200, default=50)
            s.model.fit_with_early_stopping(X_train_part, y_train_part, X_val, y_val, patience=patience)
        else:
            s.model.fit(s.X_train, s.y_train)
        print(f"✓ Training complete ({len(s.model.loss_history)} epochs)")
        print(f"  Model supports {s.model.n_classes} classes")
        
        if ask_yes_no("Show loss history?", default=True):
            plot_loss_curve(s.model.loss_history)
    except Exception as e:
        print(f"✗ Training error: {e}")
        s.model = None


def evaluate_model_interactive_multinomial(s: MultinomialAppState) -> None:
    """Evaluate multinomial model interactively."""
    if s.model is None or not s.model.is_trained:
        print("✗ No trained model!")
        return
    if s.X_test is None or s.y_test is None:
        print("✗ No test data!")
        return
    print("\n" + "=" * 70)
    print("EVALUATING MULTINOMIAL MODEL")
    print("=" * 70)
    try:
        y_pred = s.model.predict(s.X_test)
        y_proba = s.model.predict_proba(s.X_test)
        
        # Compute metrics
        s.metrics['accuracy'] = accuracy(s.y_test, y_pred)
        s.metrics['macro_precision'] = multiclass_precision(s.y_test, y_pred, average='macro')
        s.metrics['macro_recall'] = multiclass_recall(s.y_test, y_pred, average='macro')
        s.metrics['macro_f1'] = multiclass_f1_score(s.y_test, y_pred, average='macro')
        s.metrics['micro_f1'] = multiclass_f1_score(s.y_test, y_pred, average='micro')
        
        # Print report
        class_names = s.class_names if s.class_names else None
        print_multiclass_classification_report(s.y_test, y_pred, class_names)
        
        # Visualizations
        cm = multiclass_confusion_matrix(s.y_test, y_pred, s.model.n_classes)
        plot_multiclass_confusion_matrix(cm, class_names)
        
        if ask_yes_no("Show feature coefficients?", default=True):
            plot_multiclass_feature_importance(
                s.prepareddata.feature_names, s.model.W, class_names
            )
        
        if ask_yes_no("Show probability distributions by class?", default=False):
            plot_class_probability_distributions(y_proba, s.y_test, class_names)
        
        if ask_yes_no("Show probability heatmap?", default=False):
            plot_multiclass_probability_heatmap(y_proba, s.y_test, class_names)
        
        print("✓ Evaluation complete")
    except Exception as e:
        print(f"✗ Evaluation error: {e}")


def predict_single_interactive_multinomial(s: MultinomialAppState) -> None:
    """Make a single prediction with multiclass model."""
    if s.model is None or not s.model.is_trained:
        print("✗ No trained model!")
        return
    if s.prepareddata is None:
        print("✗ No feature names available!")
        return
    if s.scaler_mean is None and s.use_scaling:
        print("✗ Scaling parameters not available!")
        return
    try:
        X_input = show_prediction_example_multiclass(s.prepareddata.feature_names)
        if s.use_scaling and s.scaler_mean is not None:
            X_input = standardize_apply(X_input.reshape(1, -1), s.scaler_mean, s.scaled_std)[0]
        
        proba = s.model.predict_proba(X_input.reshape(1, -1))[0]
        pred = s.model.predict(X_input.reshape(1, -1))[0]
        
        class_names = s.class_names if s.class_names else [f"Class {k}" for k in range(s.model.n_classes)]
        
        print("\n" + "=" * 70)
        print("PREDICTION RESULT")
        print("=" * 70)
        print(f"Predicted class: {int(pred)} ({class_names[int(pred)]})")
        print(f"Confidence:      {proba[int(pred)] * 100:.2f}%")
        print("\nPer-class probabilities:")
        for k in range(s.model.n_classes):
            bar = "█" * int(proba[k] * 50) + "░" * int((1 - proba[k]) * 50)
            print(f"  {class_names[k]:<15} {proba[k]:.4f}  {bar}")
        print("=" * 70 + "\n")
    except Exception as e:
        print(f"✗ Prediction error: {e}")


# ============================================================================
# Menu functions
# ============================================================================

def menu_data_multinomial(s: MultinomialAppState) -> None:
    """Data menu for multinomial logistic regression."""
    while True:
        clear_screen()
        print_header("Multinomial Logistic Regression — Data")
        print_status(s)
        options = [
            "Load CSV dataset",
            "Manual input",
            "Select features + target",
            "Configure train/test split",
            "Back",
        ]
        choice = ask_choice("", options)
        if choice == 0:
            path = input("CSV path: ").strip()
            try:
                s.dataset = load_csv_dataset(path)
                s.prepareddata = None
                s.model = None
                print("Dataset loaded successfully.")
            except Exception as e:
                print(f"✗ Error: {e}")
            pause()
        elif choice == 1:
            try:
                s.dataset = manual_input_dataset()
                s.prepareddata = None
                s.model = None
                print("Dataset created successfully.")
            except Exception as e:
                print(f"✗ Error: {e}")
            pause()
        elif choice == 2:
            if s.dataset is None:
                print("✗ No dataset loaded yet!")
                pause()
                continue
            try:
                s.prepareddata = select_features_and_target_multiclass(s.dataset)
                rebuild_split(s)
                detect_class_names(s)
                print("✓ Features and target selected")
            except Exception as e:
                print(f"✗ Error: {e}")
            pause()
        elif choice == 3:
            if s.dataset is None:
                print("✗ No dataset loaded yet!")
                pause()
                continue
            s.test_size = ask_float("Test set size (0.05-0.5):", min_val=0.05, max_val=0.5, default=0.2)
            s.seed = ask_int("Random seed:", min_val=0, max_val=10000, default=42)
            s.use_scaling = ask_yes_no("Use feature scaling (standardization)?", default=True)
            rebuild_split(s)
            print("✓ Split configuration updated")
            pause()
        else:
            return


def menu_train_multinomial(s: MultinomialAppState) -> None:
    """Train menu for multinomial logistic regression."""
    while True:
        clear_screen()
        print_header("Multinomial Logistic Regression — Train")
        print_status(s)
        options = [
            "Configure model",
            "Train model",
            "Back",
        ]
        choice = ask_choice("", options)
        if choice == 0:
            s.learning_rate, s.epochs, s.lambda_l2 = configure_multinomial_model_hyperparameters()
            print("✓ Model configured")
            pause()
        elif choice == 1:
            train_model_interactive_multinomial(s)
            pause()
        else:
            return


def menu_evaluate_multinomial(s: MultinomialAppState) -> None:
    """Evaluate menu for multinomial logistic regression."""
    while True:
        clear_screen()
        print_header("Multinomial Logistic Regression — Evaluate")
        print_status(s)
        options = [
            "Evaluate on test set",
            "Explain metrics",
            "Back",
        ]
        choice = ask_choice("", options)
        if choice == 0:
            evaluate_model_interactive_multinomial(s)
            pause()
        elif choice == 1:
            clear_screen()
            print_header("Multinomial Logistic Regression — Metrics Explained")
            print("\nACCURACY:")
            print("─" * 70)
            print("  • What: Percentage of all predictions that are correct")
            print("  • Formula: correct / total")
            print("  • Works for any number of classes")
            
            print("\nMACRO-AVERAGED METRICS:")
            print("─" * 70)
            print("  • Calculate metric (precision/recall/F1) for each class separately")
            print("  • Then take the unweighted average")
            print("  • Each class contributes equally regardless of size")
            print("  • Best when: all classes are equally important")
            
            print("\nMICRO-AVERAGED METRICS:")
            print("─" * 70)
            print("  • Aggregate contributions of all classes first")
            print("  • Then calculate metric globally")
            print("  • Larger classes contribute more")
            print("  • Best when: class imbalance is present")
            
            print("\nCONFUSION MATRIX:")
            print("─" * 70)
            print("  • K×K matrix where cm[i,j] = count of class i predicted as class j")
            print("  • Diagonal = correct predictions")
            print("  • Off-diagonal = errors (shows which classes get confused)")
            print("  • Row sums = actual class counts")
            print("  • Column sums = predicted class counts")
            pause()
        else:
            return


def menu_save_load_multinomial(s: MultinomialAppState) -> None:
    """
    Save/Load complete Multinomial sessions.
    """
    storage = SessionStorage()
    adapter = MultinomialSessionAdapter()
    
    while True:
        clear_screen()
        print_header("Multinomial Logistic Regression — Save/Load Session")
        print_status(s)

        options = [
            "Save complete session",
            "Load session",
            "List saved sessions",
            "Delete session",
            "Back",
        ]

        choice = ask_choice("", options)

        if choice == 0:
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

            session_name = input("Enter session name (e.g., 'iris_multinomial_v1'): ").strip()
            if not session_name:
                print("!Invalid name!")
                pause()
                continue

            try:
                session_data, arrays_dict = adapter.extract(s)
                
                session_dir = f"./ml_sessions/{session_name}"
                storage.save_session(session_data, session_dir, arrays_dict, verbose=True)
                
                print(f"\n✓ Complete session '{session_name}' saved successfully!")
                print(f"  - Dataset: {s.dataset.data.shape}")
                print(f"  - Features: {len(s.prepareddata.feature_names)}")
                print(f"  - Classes: {s.model.n_classes}")
                if s.metrics:
                    metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in s.metrics.items())
                    print(f"  - Metrics: {metrics_str}")
            
            except Exception as e:
                print(f"!Error saving session: {e}!")
            
            pause()

        elif choice == 1:
            sessions = storage.list_sessions()
            if not sessions:
                print("!No saved sessions!")
                pause()
                continue

            print("\nAvailable sessions:")
            for i, session_name in enumerate(sessions, 1):
                print(f"{i}. {session_name}")
            
            try:
                idx = ask_int("Select session number: ", min_val=1, max_val=len(sessions)) - 1
                session_name = sessions[idx]
                session_dir = f"./ml_sessions/{session_name}"
                
                session_data, arrays_dict = storage.load_session(session_dir, verbose=True)
                
                adapter.restore(session_data, arrays_dict, s)
                
                print(f"\n✓ Session '{session_name}' loaded successfully!")
                print(f"  - Dataset: {s.dataset.data.shape}")
                print(f"  - Features: {len(s.prepareddata.feature_names)}")
                print(f"  - Classes: {s.model.n_classes}")
                if s.metrics:
                    metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in s.metrics.items())
                    print(f"  - Metrics: {metrics_str}")
            except Exception as e:
                print(f"!Error loading session: {e}!")
            
            pause()
        
        elif choice == 2:
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


def menu_predict_multinomial(s: MultinomialAppState) -> None:
    """Predict menu for multinomial logistic regression."""
    while True:
        clear_screen()
        print_header("Multinomial Logistic Regression — Predict")
        print_status(s)
        options = [
            "Make a single prediction",
            "Back",
        ]
        choice = ask_choice("", options)
        if choice == 0:
            predict_single_interactive_multinomial(s)
            pause()
        else:
            return


def menu_visualize_multinomial(s: MultinomialAppState) -> None:
    """Visualize menu for multinomial logistic regression."""
    while True:
        clear_screen()
        print_header("Multinomial Logistic Regression — Visualize")
        print_status(s)
        options = [
            "Plot loss curve (needs trained model)",
            "Plot multiclass confusion matrix (test set)",
            "Plot feature coefficients per class",
            "Plot class probability distributions",
            "Plot probability heatmap",
            "Back",
        ]
        choice = ask_choice("", options)
        if choice == 0:
            if s.model is None:
                print("✗ No trained model!")
                pause()
                continue
            plot_loss_curve(s.model.loss_history)
            pause()
        elif choice == 1:
            if s.model is None or s.X_test is None or s.y_test is None:
                print("✗ Need trained model and test set!")
                pause()
                continue
            y_pred = s.model.predict(s.X_test)
            cm = multiclass_confusion_matrix(s.y_test, y_pred, s.model.n_classes)
            class_names = s.class_names if s.class_names else None
            plot_multiclass_confusion_matrix(cm, class_names)
            pause()
        elif choice == 2:
            if s.model is None or s.prepareddata is None:
                print("✗ Need trained model and feature names!")
                pause()
                continue
            class_names = s.class_names if s.class_names else None
            plot_multiclass_feature_importance(
                s.prepareddata.feature_names, s.model.W, class_names
            )
            pause()
        elif choice == 3:
            if s.model is None or s.X_test is None or s.y_test is None:
                print("✗ Need trained model and test set!")
                pause()
                continue
            y_proba = s.model.predict_proba(s.X_test)
            class_names = s.class_names if s.class_names else None
            plot_class_probability_distributions(y_proba, s.y_test, class_names)
            pause()
        elif choice == 4:
            if s.model is None or s.X_test is None or s.y_test is None:
                print("✗ Need trained model and test set!")
                pause()
                continue
            y_proba = s.model.predict_proba(s.X_test)
            class_names = s.class_names if s.class_names else None
            plot_multiclass_probability_heatmap(y_proba, s.y_test, class_names)
            pause()
        else:
            return
