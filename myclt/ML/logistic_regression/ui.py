"""
User interface helpers for Logistic Regression workflow.

Provides interactive prompts and data selection dialogs.
"""

import numpy as np
from typing import List, Tuple

from .app_state import AppState, print_status, rebuild_split
from .data import Dataset, Prepareddata, load_csv_dataset
from .core import LogisticRegressionGD
from .metrics import accuracy, precision, recall, f1_score, confusion_matrix, print_classification_report
from .visualization import (
    plot_loss_curve, plot_confusion_matrix_heatmap, plot_feature_coefficients,
    plot_probability_distribution, plot_metrics_comparison
)
from common.input_validation import ask_yes_no, ask_int, ask_float, ask_choice
from common.ui_helpers import clear_screen, print_header, pause


def select_features_and_target(dataset: Dataset) -> Prepareddata:
    """
    Interactive feature and target selection from dataset.
    
    Guides user to:
        1. View available columns
        2. Select features (independent variables)
        3. Select target (dependent variable, must be binary 0/1)
    
    Args:
        dataset: Loaded dataset
    
    Returns:
        Prepareddata with selected X and y
    """
    print("\n" + "=" * 70)
    print("FEATURE & TARGET SELECTION FOR CLASSIFICATION")
    print("=" * 70)
    
    # Show available columns
    print("\nAvailable columns:")
    for i, col in enumerate(dataset.columns, 1):
        print(f"  {i}. {col}")
    
    # Feature selection
    print("\nSelect feature columns (comma-separated indices, e.g., 1,2,3):")
    features_input = input("> ").strip()
    
    try:
        feature_indices = [int(x.strip()) - 1 for x in features_input.split(",")]
        
        # Validate indices
        if any(i < 0 or i >= len(dataset.columns) for i in feature_indices):
            raise ValueError("Invalid column index")
        if len(feature_indices) == 0:
            raise ValueError("Must select at least one feature")
        
        feature_names = [dataset.columns[i] for i in feature_indices]
        X = dataset.data[:, feature_indices]
        
    except (ValueError, IndexError) as e:
        raise ValueError(f"Invalid feature selection: {e}")
    
    # Target selection
    print("\nSelect target column index (must contain binary values 0/1):")
    try:
        target_idx = int(input("> ").strip()) - 1
        
        if target_idx < 0 or target_idx >= len(dataset.columns):
            raise ValueError("Invalid column index")
        
        target_name = dataset.columns[target_idx]
        y = dataset.data[:, target_idx]
        
        # Validate binary target
        unique_values = np.unique(y)
        valid_binary = len(unique_values) == 2 and set(unique_values) == {0, 1}
        
        if not valid_binary:
            # Try to auto-convert to binary
            if len(unique_values) == 2:
                print(f"Warning: Target has values {unique_values}, converting to 0/1...")
                y = (y == unique_values[1]).astype(int)
            else:
                raise ValueError(f"Target must contain exactly 2 classes, found {len(unique_values)}")
        
    except (ValueError, IndexError) as e:
        raise ValueError(f"Invalid target selection: {e}")
    
    # Ensure features and target don't overlap
    if target_idx in feature_indices:
        raise ValueError("Target column cannot also be a feature!")
    
    print(f"\n✓ Selected features: {feature_names}")
    print(f"✓ Target: {target_name}")
    print(f"✓ Data shape: {X.shape[0]} samples × {X.shape[1]} features")
    
    # Display class distribution safely
    unique_classes, class_counts = np.unique(y, return_counts=True)
    class_dist = dict(zip(unique_classes, class_counts))
    print(f"✓ Class distribution: {class_dist}")
    
    return Prepareddata(X=X, Y=y, feature_names=feature_names, target_name=target_name)


def configure_model_hyperparameters() -> Tuple[float, int, float, float]:
    """
    Interactive model hyperparameter configuration.
    
    Guides user to configure:
        - Learning rate
        - Number of epochs
        - L2 regularization strength
        - Classification threshold
    
    Returns:
        Tuple of (learning_rate, epochs, lambda_l2, threshold)
    """
    print("\n" + "=" * 70)
    print("MODEL HYPERPARAMETER CONFIGURATION")
    print("=" * 70)
    
    learning_rate = ask_float("Learning rate (default 0.01, range 0.001-0.1):", min_val=0.001, max_val=0.1, default=0.01)
    epochs = ask_int("Number of epochs (default 1000, range 100-10000):", min_val=100, max_val=10000, default=1000)
    lambda_l2 = ask_float("L2 regularization strength (default 0.0):", min_val=0.0, max_val=1.0, default=0.0)
    threshold = ask_float("Classification threshold (default 0.5, range 0.1-0.9):", min_val=0.1, max_val=0.9, default=0.5)
    
    print("\n" + "=" * 70)
    print(f"Configuration: lr={learning_rate}, epochs={epochs}, λ2={lambda_l2}, threshold={threshold}")
    print("=" * 70)
    
    return learning_rate, epochs, lambda_l2, threshold


def show_prediction_example(feature_names: List[str]) -> np.ndarray:
    """
    Interactive single prediction example.
    
    Example: Rain prediction from weather parameters
        Temperature: 21
        Humidity: 50
        Wind Speed: 15
        → Prediction: Rain (1) or No Rain (0)
    
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
                    # Note: using 0.0 as default - ensure data matches training scale
                    print(f"    (using default: 0.0)")
                    values.append(0.0)
                    break
                value = float(val_str)
                values.append(value)
                break
            except ValueError:
                print("  Please enter a valid number.")
    
    return np.array(values, dtype=float)


def load_data_interactive(s: AppState) -> None:
    print("\n" + "=" * 70)
    print("LOAD DATA")
    print("=" * 70)
    
    print("\n1. Load from CSV file")
    print("2. Manual input (placeholder)")
    
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
        print("Manual input not yet implemented")


def select_features_interactive(s: AppState) -> None:
    if s.dataset is None:
        print("✗ No dataset loaded yet!")
        return
    try:
        s.prepareddata = select_features_and_target(s.dataset)
        rebuild_split(s)
        print("✓ Features and target selected")
    except ValueError as e:
        print(f"✗ Error: {e}")


def configure_split_interactive(s: AppState) -> None:
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


def configure_model_interactive(s: AppState) -> None:
    print("\n" + "=" * 70)
    print("CONFIGURE MODEL")
    print("=" * 70)
    s.learning_rate, s.epochs, s.lambda_l2, s.threshold = configure_model_hyperparameters()
    print("✓ Model configured")


def train_model_interactive(s: AppState) -> None:
    if s.X_train is None or s.y_train is None:
        print("✗ No training data prepared yet!")
        return
    print("\n" + "=" * 70)
    print("TRAINING MODEL")
    print("=" * 70)
    try:
        s.model = LogisticRegressionGD(
            learning_rate=s.learning_rate,
            epochs=s.epochs,
            lambda_l2=s.lambda_l2,
            threshold=s.threshold
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
        if ask_yes_no("Show loss history?", default=True):
            plot_loss_curve(s.model.loss_history)
    except Exception as e:
        print(f"✗ Training error: {e}")
        s.model = None


def evaluate_model_interactive(s: AppState) -> None:
    if s.model is None or not s.model.is_trained:
        print("✗ No trained model!")
        return
    if s.X_test is None or s.y_test is None:
        print("✗ No test data!")
        return
    print("\n" + "=" * 70)
    print("EVALUATING MODEL")
    print("=" * 70)
    try:
        y_pred = s.model.predict(s.X_test)
        y_proba = s.model.predict_proba(s.X_test)
        s.metrics['accuracy'] = accuracy(s.y_test, y_pred)
        s.metrics['precision'] = precision(s.y_test, y_pred)
        s.metrics['recall'] = recall(s.y_test, y_pred)
        s.metrics['f1'] = f1_score(s.y_test, y_pred)
        print_classification_report(s.y_test, y_pred)
        tp, fp, fn, tn = confusion_matrix(s.y_test, y_pred)
        plot_confusion_matrix_heatmap(tp, fp, fn, tn)
        plot_metrics_comparison(s.metrics['accuracy'], s.metrics['precision'], s.metrics['recall'], s.metrics['f1'])
        if ask_yes_no("Show feature coefficients?", default=True):
            plot_feature_coefficients(s.prepareddata.feature_names, s.model.w)
        if ask_yes_no("Show probability distribution?", default=False):
            plot_probability_distribution(y_proba, s.y_test)
        print("✓ Evaluation complete")
    except Exception as e:
        print(f"✗ Evaluation error: {e}")


def predict_single_interactive(s: AppState) -> None:
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
        X_input = show_prediction_example(s.prepareddata.feature_names)
        if s.use_scaling and s.scaler_mean is not None:
            from .preprocessing import standardize_apply
            X_input = standardize_apply(X_input.reshape(1, -1), s.scaler_mean, s.scaled_std)[0]
        proba = s.model.predict_proba(X_input.reshape(1, -1))[0]
        pred = s.model.predict(X_input.reshape(1, -1))[0]
        print("\n" + "=" * 70)
        print("PREDICTION RESULT")
        print("=" * 70)
        print(f"Probability of positive class (1): {proba:.4f}")
        print(f"Prediction: {pred} ({'POSITIVE' if pred == 1 else 'NEGATIVE'})")
        print(f"Confidence: {max(proba, 1 - proba) * 100:.2f}%")
        print("=" * 70 + "\n")
    except Exception as e:
        print(f"✗ Prediction error: {e}")


def menu_data(s: AppState) -> None:
    while True:
        clear_screen()
        print_header("Logistic Regression — Data")
        print_status(s)
        options = [
            "Load CSV dataset",
            "Manual input (placeholder)",
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
            print("Manual input not yet implemented")
            pause()
        elif choice == 2:
            if s.dataset is None:
                print("✗ No dataset loaded yet!")
                pause()
                continue
            try:
                s.prepareddata = select_features_and_target(s.dataset)
                rebuild_split(s)
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


def menu_train(s: AppState) -> None:
    while True:
        clear_screen()
        print_header("Logistic Regression — Train")
        print_status(s)
        options = [
            "Configure model",
            "Train model",
            "Back",
        ]
        choice = ask_choice("", options)
        if choice == 0:
            s.learning_rate, s.epochs, s.lambda_l2, s.threshold = configure_model_hyperparameters()
            print("✓ Model configured")
            pause()
        elif choice == 1:
            train_model_interactive(s)
            pause()
        else:
            return


def menu_evaluate(s: AppState) -> None:
    while True:
        clear_screen()
        print_header("Logistic Regression — Evaluate")
        print_status(s)
        options = [
            "Evaluate on test set",
            "Back",
        ]
        choice = ask_choice("", options)
        if choice == 0:
            evaluate_model_interactive(s)
            pause()
        else:
            return


def menu_predict(s: AppState) -> None:
    while True:
        clear_screen()
        print_header("Logistic Regression — Predict")
        print_status(s)
        options = [
            "Make a single prediction",
            "Back",
        ]
        choice = ask_choice("", options)
        if choice == 0:
            predict_single_interactive(s)
            pause()
        else:
            return


def menu_visualize(s: AppState) -> None:
    while True:
        clear_screen()
        print_header("Logistic Regression — Visualize")
        print_status(s)
        options = [
            "Plot loss curve (needs trained model)",
            "Plot confusion matrix (test set)",
            "Plot feature coefficients",
            "Plot probability distribution",
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
            tp, fp, fn, tn = confusion_matrix(s.y_test, y_pred)
            plot_confusion_matrix_heatmap(tp, fp, fn, tn)
            pause()
        elif choice == 2:
            if s.model is None or s.prepareddata is None:
                print("✗ Need trained model and feature names!")
                pause()
                continue
            plot_feature_coefficients(s.prepareddata.feature_names, s.model.w)
            pause()
        elif choice == 3:
            if s.model is None or s.X_test is None or s.y_test is None:
                print("✗ Need trained model and test set!")
                pause()
                continue
            y_proba = s.model.predict_proba(s.X_test)
            plot_probability_distribution(y_proba, s.y_test)
            pause()
        else:
            return
