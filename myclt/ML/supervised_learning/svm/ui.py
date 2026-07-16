"""
User interface helpers for Binary SVM (Classification + Regression).

Provides interactive prompts and data selection dialogs for both
SVM classification (LinearSVM, KernelSVM) and SVR (LinearSVR, KernelSVR).
"""

import numpy as np
import os
from typing import List, Optional

from .app_state import AppState, print_status, rebuild_split
from .data import Dataset, Prepareddata, load_csv_dataset, manual_input_dataset
from .core import LinearSVM, KernelSVM, LinearSVR, KernelSVR
from .preprocessing import standardize_apply
from .metrics import (
    accuracy, f1_score,
    classification_report, confusion_matrix,
    mean_squared_error, root_mean_squared_error, mean_absolute_error,
    r2_score, regression_report
)
from .visualization import (
    plot_loss_curve, plot_confusion_matrix,
    plot_svm_decision_boundary_2d,
    plot_true_vs_pred, plot_residuals,
    plot_svr_tube, plot_support_vector_info
)
from .session_adapter import (
    LinearSVMSessionAdapter, KernelSVMSessionAdapter,
    LinearSVRSessionAdapter, KernelSVRSessionAdapter
)
from myclt.ML.session_storage import SessionStorage
from myclt.ML.batch_predict import batch_predict_from_csv
from myclt.common.input_validation import ask_yes_no, ask_int, ask_float, ask_choice
from myclt.common.ui_helpers import clear_screen, print_header, pause


# ============================================================================
# Configuration helpers
# ============================================================================

def select_features_and_target_binary(dataset: Dataset, mode: str = 'classifier'
                                       ) -> Prepareddata:
    """
    Interactive feature and target selection for binary SVM.

    Uses universal select_features_and_target.
    For classification, validates that target has exactly 2 classes.

    Args:
        dataset: Loaded dataset
        mode: 'classifier' or 'regressor'

    Returns:
        Prepareddata with selected X and Y
    """
    from myclt.ML.base.base_data import select_features_and_target

    prepared = select_features_and_target(dataset)

    if mode == 'classifier':
        unique_values = np.unique(prepared.Y)
        if len(unique_values) != 2:
            raise ValueError(
                f"Binary SVM requires exactly 2 classes, found {len(unique_values)}. "
                f"For multiclass, use Multiclass SVM."
            )
        print(f"\n✓ Binary classification: classes = {sorted(unique_values.tolist())}")
        print(f"  Class distribution: {dict(zip(*np.unique(prepared.Y, return_counts=True)))}")
    else:
        print(f"\n✓ Regression target selected: {prepared.Y.shape[0]} samples")
        print(f"  Range: [{prepared.Y.min():.4f}, {prepared.Y.max():.4f}]")

    return prepared


def configure_common_hyperparameters(mode: str = 'classifier') -> dict:
    """
    Interactive hyperparameter configuration (common params).

    Args:
        mode: 'classifier' or 'regressor'

    Returns:
        Dictionary with common parameters (C, learning_rate, epochs)
    """
    print("\n" + "=" * 70)
    print("COMMON HYPERPARAMETER CONFIGURATION")
    print("=" * 70)

    C = ask_float("Regularization C (0.001-1000, default=1.0):",
                  min_val=0.001, max_val=1000.0, default=1.0)
    learning_rate = ask_float("Learning rate (0.0001-0.1, default=0.001):",
                              min_val=0.0001, max_val=0.1, default=0.001)
    epochs = ask_int("Epochs (100-10000, default=1000):",
                     min_val=100, max_val=10000, default=1000)

    params = {
        'C': C,
        'learning_rate': learning_rate,
        'epochs': epochs,
    }

    if mode == 'regressor':
        epsilon = ask_float("Epsilon (ε-insensitive tube, 0.001-1.0, default=0.1):",
                            min_val=0.001, max_val=1.0, default=0.1)
        params['epsilon'] = epsilon

    print("\n" + "=" * 70)
    print(f"Configuration: {params}")
    print("=" * 70)

    return params


def configure_kernel_hyperparameters() -> dict:
    """
    Interactive kernel hyperparameter configuration.

    Returns:
        Dictionary with kernel parameters (kernel, gamma, degree, coef0)
    """
    print("\n" + "=" * 70)
    print("KERNEL CONFIGURATION")
    print("=" * 70)

    kernel_options = ['rbf', 'linear', 'poly', 'sigmoid']
    kernel_idx = ask_choice("Select kernel:", kernel_options)
    kernel = kernel_options[kernel_idx]

    kwargs = {'kernel': kernel}

    if kernel in ('rbf', 'poly', 'sigmoid'):
        gamma = ask_float("Gamma (0.0001-10, default=1.0):",
                          min_val=0.0001, max_val=10.0, default=1.0)
        kwargs['gamma'] = gamma

    if kernel == 'poly':
        degree = ask_int("Degree (2-5, default=3):",
                         min_val=2, max_val=5, default=3)
        coef0 = ask_float("Coef0 (0.0-10.0, default=1.0):",
                          min_val=0.0, max_val=10.0, default=1.0)
        kwargs['degree'] = degree
        kwargs['coef0'] = coef0

    if kernel == 'sigmoid':
        coef0 = ask_float("Coef0 (0.0-10.0, default=0.0):",
                          min_val=0.0, max_val=10.0, default=0.0)
        kwargs['coef0'] = coef0

    print(f"\nKernel config: {kwargs}")
    return kwargs


def show_prediction_example(feature_names: List[str]) -> np.ndarray:
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


# ============================================================================
# Core interactive functions
# ============================================================================

def load_data_interactive(s: AppState) -> None:
    """Load data interactively."""
    print("\n" + "=" * 70)
    print("LOAD DATA")
    print("=" * 70)

    options = ["Load CSV", "Manual input"]
    choice = ask_choice("", options)
    if choice == 0:
        path = input("\nCSV path: ").strip()
        try:
            s.dataset = load_csv_dataset(path)
            print(f"✓ Loaded: {s.dataset.data.shape[0]} rows × {s.dataset.data.shape[1]} columns")
        except FileNotFoundError as e:
            print(f"✗ Error: {e}")
    else:
        try:
            s.dataset = manual_input_dataset()
            print(f"✓ Created: {s.dataset.data.shape[0]} rows × {s.dataset.data.shape[1]} columns")
        except ValueError as e:
            print(f"✗ Error: {e}")
    pause()


def select_features_interactive(s: AppState) -> None:
    """Select features and target interactively."""
    if s.dataset is None:
        print("✗ No dataset loaded yet!")
        pause()
        return
    try:
        s.prepareddata = select_features_and_target_binary(s.dataset, s.mode)
        rebuild_split(s)
        print("✓ Features and target selected")
        pause()
    except ValueError as e:
        print(f"✗ Error: {e}")
        pause()


def configure_split_interactive(s: AppState) -> None:
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
    pause()


def configure_model_interactive(s: AppState) -> None:
    """Configure model type and hyperparameters."""
    print("\n" + "=" * 70)
    print("CONFIGURE MODEL")
    print("=" * 70)

    # Choose model type
    if s.mode == 'classifier':
        model_options = ["Linear SVM", "Kernel SVM"]
        model_choice = ask_choice("Select model type:", model_options)
        s.model_type = "linear_svm" if model_choice == 0 else "kernel_svm"
    else:
        model_options = ["Linear SVR", "Kernel SVR"]
        model_choice = ask_choice("Select model type:", model_options)
        s.model_type = "linear_svr" if model_choice == 0 else "kernel_svr"

    # Common params
    common = configure_common_hyperparameters(s.mode)
    s.C = common['C']
    s.learning_rate = common['learning_rate']
    s.epochs = common['epochs']
    if s.mode == 'regressor' and 'epsilon' in common:
        s.epsilon = common['epsilon']

    # Kernel params (if applicable)
    if s.model_type in ("kernel_svm", "kernel_svr"):
        kernel_params = configure_kernel_hyperparameters()
        s.kernel = kernel_params.get('kernel', 'rbf')
        s.gamma = kernel_params.get('gamma', 1.0)
        s.degree = kernel_params.get('degree', 3)
        s.coef0 = kernel_params.get('coef0', 1.0)

    print("✓ Model configured")


def train_model_interactive(s: AppState) -> None:
    """Train SVM model interactively."""
    if s.X_train is None or s.y_train is None:
        print("✗ No training data prepared yet!")
        return

    print("\n" + "=" * 70)
    print(f"TRAINING {s.model_type.upper()}")
    print("=" * 70)

    try:
        # Create model based on type
        if s.model_type == "linear_svm":
            s.model = LinearSVM(
                C=s.C, learning_rate=s.learning_rate, epochs=s.epochs
            )
        elif s.model_type == "kernel_svm":
            s.model = KernelSVM(
                kernel=s.kernel, C=s.C,
                gamma=s.gamma, degree=s.degree, coef0=s.coef0,
                learning_rate=s.learning_rate, epochs=s.epochs
            )
        elif s.model_type == "linear_svr":
            s.model = LinearSVR(
                C=s.C, epsilon=s.epsilon,
                learning_rate=s.learning_rate, epochs=s.epochs
            )
        elif s.model_type == "kernel_svr":
            s.model = KernelSVR(
                kernel=s.kernel, C=s.C, epsilon=s.epsilon,
                gamma=s.gamma, degree=s.degree, coef0=s.coef0,
                learning_rate=s.learning_rate, epochs=s.epochs
            )

        # Optional early stopping
        use_early_stopping = ask_yes_no("Use early stopping? (validation split)", default=False)
        if use_early_stopping and hasattr(s.model, 'fit_with_early_stopping'):
            n_train = len(s.X_train)
            val_size = int(0.2 * n_train)
            X_train_part = s.X_train[val_size:]
            y_train_part = s.y_train[val_size:]
            X_val = s.X_train[:val_size]
            y_val = s.y_train[:val_size]
            patience = ask_int("Patience (epochs without improvement):",
                               min_val=5, max_val=200, default=50)
            s.model.fit_with_early_stopping(
                X_train_part, y_train_part, X_val, y_val,
                patience=patience, verbose=True
            )
        else:
            s.model.fit(s.X_train, s.y_train)

        print(f"✓ Training complete ({len(s.model.loss_history)} checkpoints)")
        if hasattr(s.model, 'n_support_vectors'):
            print(f"  Support vectors: {s.model.n_support_vectors}")

        # Show loss history
        if s.model.loss_history and ask_yes_no("Show loss curve?", default=True):
            if s.mode == 'classifier':
                plot_loss_curve(s.model.loss_history, ylabel="Hinge Loss",
                                title=f"{s.model_type.upper()} Loss Curve")
            else:
                plot_loss_curve(s.model.loss_history, ylabel="ε-Insensitive Loss",
                                title=f"{s.model_type.upper()} Loss Curve")

    except Exception as e:
        print(f"✗ Training error: {e}")
        s.model = None


def evaluate_model_interactive(s: AppState) -> None:
    """Evaluate model interactively."""
    if s.model is None or not s.model.is_trained:
        print("✗ No trained model!")
        return
    if s.X_test is None or s.y_test is None:
        print("✗ No test data!")
        return

    print("\n" + "=" * 70)
    print(f"EVALUATING {s.model_type.upper()}")
    print("=" * 70)

    try:
        y_pred = s.model.predict(s.X_test)
        s.metrics = {}

        if s.mode == 'classifier':
            # Classification metrics
            s.metrics['accuracy'] = accuracy(s.y_test, y_pred)

            # Try F1 score
            try:
                s.metrics['f1'] = f1_score(s.y_test, y_pred)
            except Exception:
                pass

            # Classification report
            print(classification_report(s.y_test, y_pred))

            # Confusion matrix
            cm = confusion_matrix(s.y_test, y_pred)
            print(f"\nConfusion Matrix:\n{cm}")
            if ask_yes_no("Show confusion matrix plot?", default=True):
                plot_confusion_matrix(cm)

            # Decision boundary (if 2 features)
            if s.X_test.shape[1] == 2:
                if ask_yes_no("Show decision boundary plot?", default=True):
                    plot_svm_decision_boundary_2d(
                        s.model, s.X_test, s.y_test,
                        feature_names=s.prepareddata.feature_names if s.prepareddata else None
                    )

            # Support vector info
            if ask_yes_no("Show support vector information?", default=False):
                plot_support_vector_info(s.model)

        else:
            # Regression metrics
            s.metrics['mse'] = mean_squared_error(s.y_test, y_pred)
            s.metrics['rmse'] = root_mean_squared_error(s.y_test, y_pred)
            s.metrics['mae'] = mean_absolute_error(s.y_test, y_pred)
            s.metrics['r2'] = r2_score(s.y_test, y_pred)

            print(regression_report(s.y_test, y_pred))

            # True vs Predicted plot
            if ask_yes_no("Show True vs Predicted plot?", default=True):
                plot_true_vs_pred(s.y_test, y_pred)

            # Residual plot
            if ask_yes_no("Show residual plot?", default=True):
                plot_residuals(s.y_test, y_pred)

            # Epsilon tube (if SVR and 1D feature)
            if s.model_type in ("linear_svr", "kernel_svr") and s.X_test.shape[1] == 1:
                if ask_yes_no("Show ε-tube visualization?", default=False):
                    plot_svr_tube(s.X_test, s.y_test, y_pred,
                                  epsilon=s.epsilon,
                                  feature_idx=0)

        print("✓ Evaluation complete")

    except Exception as e:
        print(f"✗ Evaluation error: {e}")


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
        if s.use_scaling and s.scaler_mean is not None and s.scaled_std is not None:
            X_input = X_input.reshape(1, -1)
            X_input = standardize_apply(X_input, s.scaler_mean, s.scaled_std)
        else:
            X_input = X_input.reshape(1, -1)

        # Predict
        pred = s.model.predict(X_input)

        print("\n" + "=" * 70)
        print("PREDICTION RESULT")
        print("=" * 70)

        if s.mode == 'classifier':
            class_name = f"Class {int(pred[0])}"
            print(f"Predicted class: {class_name}")
            # Show decision function value if available
            if hasattr(s.model, 'decision_function'):
                score = s.model.decision_function(X_input)[0]
                print(f"Decision score: {score:.4f}")
                print(f"Confidence: {abs(score):.4f}")
        else:
            print(f"Predicted value: {pred[0]:.6f}")

        print("=" * 70 + "\n")

    except Exception as e:
        print(f"✗ Prediction error: {e}")


# ============================================================================
# Menu functions
# ============================================================================

def menu_data(s: AppState) -> None:
    """Data menu."""
    while True:
        clear_screen()
        print_header(f"SVM — Data ({'Classification' if s.mode == 'classifier' else 'Regression'})")
        print_status(s)
        options = ["Load dataset (CSV / Manual)",
                    "Select features + target", "Configure train/test split",
                    "Back"]
        choice = ask_choice("", options)
        if choice == 0:
            load_data_interactive(s)
        elif choice == 1:
            select_features_interactive(s)
        elif choice == 2:
            configure_split_interactive(s)
        else:
            return


def menu_train(s: AppState) -> None:
    """Train menu."""
    while True:
        clear_screen()
        print_header("SVM — Train")
        print_status(s)
        options = ["Configure model", "Train model", "Back"]
        choice = ask_choice("", options)
        if choice == 0:
            configure_model_interactive(s)
            pause()
        elif choice == 1:
            train_model_interactive(s)
            pause()
        else:
            return


def menu_evaluate(s: AppState) -> None:
    """Evaluate menu."""
    while True:
        clear_screen()
        print_header("SVM — Evaluate")
        print_status(s)
        options = ["Evaluate on test set",
                    f"Explain {'classification' if s.mode == 'classifier' else 'regression'} metrics",
                    "Back"]
        choice = ask_choice("", options)
        if choice == 0:
            evaluate_model_interactive(s)
            pause()
        elif choice == 1:
            clear_screen()
            if s.mode == 'classifier':
                print_header("SVM Classification — Metrics Explained")
                print("\nACCURACY:")
                print("─" * 70)
                print("  • What: Percentage of correct predictions")
                print("  • Formula: correct / total")

                print("\nPRECISION:")
                print("─" * 70)
                print("  • What: Of predicted positives, how many are correct")
                print("  • Formula: TP / (TP + FP)")

                print("\nRECALL:")
                print("─" * 70)
                print("  • What: Of actual positives, how many were found")
                print("  • Formula: TP / (TP + FN)")

                print("\nF1 SCORE:")
                print("─" * 70)
                print("  • Harmonic mean of precision and recall")
                print("  • Formula: 2 × P × R / (P + R)")

                print("\nCONFUSION MATRIX:")
                print("─" * 70)
                print("  • [[TN, FP], [FN, TP]]")
                print("  • Diagonal = correct, off-diagonal = errors")

                print("\nSUPPORT VECTORS:")
                print("─" * 70)
                print("  • Data points that determine the decision boundary")
                print("  • Points with margin ≤ 1")
                print("  • Only SVs affect the model (sparsity!)")
            else:
                print_header("SVR Regression — Metrics Explained")
                print("\nMSE (Mean Squared Error):")
                print("─" * 70)
                print("  • Average squared difference between true and predicted")
                print("  • Penalizes large errors more")

                print("\nRMSE (Root Mean Squared Error):")
                print("─" * 70)
                print("  • Square root of MSE")
                print("  • Same units as target variable")

                print("\nMAE (Mean Absolute Error):")
                print("─" * 70)
                print("  • Average absolute difference")
                print("  • Less sensitive to outliers than MSE")

                print("\nR² (Coefficient of Determination):")
                print("─" * 70)
                print("  • How well the model explains variance")
                print("  • 1.0 = perfect, 0.0 = mean predictor, < 0 = worse than mean")

                print("\nε-INSENSITIVE TUBE:")
                print("─" * 70)
                print("  • Errors within ±ε are ignored")
                print("  • Points outside the tube = support vectors")
                print("  • Controls sparsity of the solution")
            pause()
        else:
            return


def menu_save_load(s: AppState) -> None:
    """Save/Load binary SVM sessions."""
    # Select appropriate adapter based on model type
    if s.model_type == "linear_svm":
        adapter = LinearSVMSessionAdapter()
    elif s.model_type == "kernel_svm":
        adapter = KernelSVMSessionAdapter()
    elif s.model_type == "linear_svr":
        adapter = LinearSVRSessionAdapter()
    elif s.model_type == "kernel_svr":
        adapter = KernelSVRSessionAdapter()
    else:
        adapter = LinearSVMSessionAdapter()

    storage = SessionStorage()

    while True:
        clear_screen()
        print_header(f"SVM — Save/Load Session ({'Classifier' if s.mode == 'classifier' else 'Regression'})")
        print_status(s)

        options = ["Save complete session", "Load session",
                    "List saved sessions", "Delete session", "Back"]
        choice = ask_choice("", options)

        if choice == 0:
            if s.dataset is None or s.prepareddata is None:
                print("!Need dataset and selected features!")
                pause()
                continue
            if s.model is None or not s.model.is_trained:
                print("!Model not trained yet!")
                pause()
                continue

            session_name = input("Session name: ").strip()
            if not session_name:
                print("!Invalid name!")
                pause()
                continue

            try:
                session_data, arrays_dict = adapter.extract(s)
                session_dir = f"./ml_sessions/{session_name}"
                storage.save_session(session_data, session_dir, arrays_dict, verbose=True)
                print(f"\n✓ Session '{session_name}' saved!")
            except Exception as e:
                print(f"!Error saving: {e}!")
            pause()

        elif choice == 1:
            sessions = storage.list_sessions()
            if not sessions:
                print("!No saved sessions!")
                pause()
                continue

            print("\nAvailable sessions:")
            for i, name in enumerate(sessions, 1):
                print(f"{i}. {name}")

            try:
                idx = ask_int("Select session: ", min_val=1, max_val=len(sessions)) - 1
                session_name = sessions[idx]
                session_dir = f"./ml_sessions/{session_name}"
                session_data, arrays_dict = storage.load_session(session_dir, verbose=True)

                # Determine which adapter to use based on stored algorithm_name
                adapter.restore(session_data, arrays_dict, s)

                print(f"\n✓ Session '{session_name}' loaded!")
                if s.metrics:
                    metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in s.metrics.items())
                    print(f"  Metrics: {metrics_str}")
            except Exception as e:
                print(f"!Error loading: {e}!")
            pause()

        elif choice == 2:
            sessions = storage.list_sessions()
            if not sessions:
                print("!No saved sessions!")
            else:
                print("\nSaved sessions:")
                for name in sessions:
                    session_dir = f"./ml_sessions/{name}"
                    try:
                        _, arrays = storage.load_session(session_dir, verbose=False)
                        ds_shape = arrays.get("dataset", np.array([])).shape
                        print(f" ✓ {name} (dataset: {ds_shape})")
                    except:
                        print(f" ? {name} (corrupted)")
            pause()

        elif choice == 3:
            sessions = storage.list_sessions()
            if not sessions:
                print("!No saved sessions!")
                pause()
                continue

            session_name = input("Session name to delete: ").strip()
            if session_name in sessions:
                if ask_yes_no(f"Delete '{session_name}'? "):
                    storage.delete_session(f"./ml_sessions/{session_name}", verbose=True)
                    print("Deleted.")
                else:
                    print("Cancelled.")
            else:
                print("!Not found!")
            pause()
        else:
            return


def menu_predict(s: AppState) -> None:
    """Predict menu."""
    while True:
        clear_screen()
        print_header("SVM — Predict")
        print_status(s)
        options = ["Make a single prediction",
                    "Batch predict from CSV file", "Back"]
        choice = ask_choice("", options)
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
                model_type = "binary_svm" if s.mode == "classifier" else "svr"
                result = batch_predict_from_csv(
                    csv_path=csv_path,
                    model=s.model,
                    feature_names=s.prepareddata.feature_names,
                    use_scaling=s.use_scaling,
                    scaler_mean=s.scaler_mean,
                    scaler_std=s.scaled_std,
                    output_path=output_csv,
                    model_type=model_type,
                )
                print(f"\n  ✓ Processed {result['n_samples']} rows!")
                print(f"  ✓ Output: {result['output_path']}")
            except Exception as e:
                print(f"\n  ✗ Error: {e}")
            pause()
        else:
            return


def menu_visualize(s: AppState) -> None:
    """Visualize menu."""
    while True:
        clear_screen()
        print_header("SVM — Visualize")
        print_status(s)
        options = [
            "Plot loss curve",
            "Plot decision boundary (2D only)",
            "Plot confusion matrix (classification)",
            "Plot True vs Predicted (regression)",
            "Plot residuals (regression)",
            "Support vector info",
            "Back",
        ]
        choice = ask_choice("", options)
        if choice == 0:
            if s.model is None or not s.model.loss_history:
                print("✗ No loss history!")
                pause()
                continue
            plot_loss_curve(s.model.loss_history)
            pause()
        elif choice == 1:
            if s.model is None or not s.model.is_trained:
                print("✗ No trained model!")
                pause()
                continue
            if s.X_test is None or s.X_test.shape[1] != 2:
                print("✗ Need exactly 2 features!")
                pause()
                continue
            plot_svm_decision_boundary_2d(
                s.model, s.X_test, s.y_test,
                feature_names=s.prepareddata.feature_names if s.prepareddata else None
            )
            pause()
        elif choice == 2:
            if s.model is None or s.mode != 'classifier':
                print("✗ Need trained classifier!")
                pause()
                continue
            if s.X_test is None or s.y_test is None:
                print("✗ No test data!")
                pause()
                continue
            y_pred = s.model.predict(s.X_test)
            cm = confusion_matrix(s.y_test, y_pred)
            plot_confusion_matrix(cm)
            pause()
        elif choice == 3:
            if s.model is None or s.mode != 'regressor':
                print("✗ Need trained regressor!")
                pause()
                continue
            if s.X_test is None or s.y_test is None:
                print("✗ No test data!")
                pause()
                continue
            y_pred = s.model.predict(s.X_test)
            plot_true_vs_pred(s.y_test, y_pred)
            pause()
        elif choice == 4:
            if s.model is None or s.mode != 'regressor':
                print("✗ Need trained regressor!")
                pause()
                continue
            if s.X_test is None or s.y_test is None:
                print("✗ No test data!")
                pause()
                continue
            y_pred = s.model.predict(s.X_test)
            plot_residuals(s.y_test, y_pred)
            pause()
        elif choice == 5:
            if s.model is None or not s.model.is_trained:
                print("✗ No trained model!")
                pause()
                continue
            plot_support_vector_info(s.model)
            pause()
        else:
            return
