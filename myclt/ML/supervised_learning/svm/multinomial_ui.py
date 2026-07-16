"""
User interface helpers for Multiclass SVM (OneVsRestSVM).

Provides interactive prompts and menus for multiclass SVM workflow:
    - Data loading, feature/target selection
    - Model configuration (linear or kernel base estimator)
    - Training with progress
    - Evaluation with multiclass metrics
    - Prediction (single & batch)
    - Session save/load
    - Visualization
"""

import numpy as np
import os
from typing import Optional

from .multinomial_app_state import MultinomialAppState, print_status, rebuild_split
from .data import Dataset, Prepareddata, load_csv_dataset, manual_input_dataset
from .core import OneVsRestSVM, LinearSVM, KernelSVM
from .preprocessing import standardize_apply
from .metrics import (
    accuracy, multiclass_precision, multiclass_recall, multiclass_f1_score,
    print_multiclass_classification_report
)
from .visualization import (
    plot_confusion_matrix,
    plot_svm_decision_boundary_2d,
    plot_support_vector_info
)
from .session_adapter import OneVsRestSVMSessionAdapter
from myclt.ML.session_storage import SessionStorage
from myclt.ML.batch_predict import batch_predict_from_csv
from myclt.common.input_validation import ask_yes_no, ask_int, ask_float, ask_choice
from myclt.common.ui_helpers import clear_screen, print_header, pause
from myclt.ML.base.base_data import select_features_and_target


# ============================================================================
# Configuration helpers
# ============================================================================

def configure_multiclass_hyperparameters() -> dict:
    """
    Interactive hyperparameter configuration for multiclass SVM.

    Returns:
        Dictionary with base_estimator_type and hyperparams
    """
    print("\n" + "=" * 70)
    print("MULTICLASS SVM — HYPERPARAMETER CONFIGURATION")
    print("=" * 70)

    # Choose base estimator type
    est_options = ["Linear SVM (faster)", "Kernel SVM (non-linear)"]
    est_choice = ask_choice("Select base estimator type:", est_options)

    config = {}
    if est_choice == 0:
        config['base_estimator_type'] = 'linear'
    else:
        config['base_estimator_type'] = 'kernel'

    # C parameter
    C = ask_float("Regularization C (0.001-1000, default=1.0):",
                  min_val=0.001, max_val=1000.0, default=1.0)
    config['C'] = C

    # Learning rate
    lr = ask_float("Learning rate (0.0001-0.1, default=0.001):",
                   min_val=0.0001, max_val=0.1, default=0.001)
    config['learning_rate'] = lr

    # Epochs
    epochs = ask_int("Epochs per classifier (100-5000, default=1000):",
                     min_val=100, max_val=5000, default=1000)
    config['epochs'] = epochs

    # Kernel params if kernel chosen
    if config['base_estimator_type'] == 'kernel':
        kernel_options = ['rbf', 'linear', 'poly', 'sigmoid']
        kernel_idx = ask_choice("Select kernel:", kernel_options)
        config['kernel'] = kernel_options[kernel_idx]

        if kernel_options[kernel_idx] in ('rbf', 'poly', 'sigmoid'):
            gamma = ask_float("Gamma (0.0001-10, default=1.0):",
                              min_val=0.0001, max_val=10.0, default=1.0)
            config['gamma'] = gamma

        if kernel_options[kernel_idx] == 'poly':
            degree = ask_int("Degree (2-5, default=3):",
                             min_val=2, max_val=5, default=3)
            coef0 = ask_float("Coef0 (0.0-10.0, default=1.0):",
                              min_val=0.0, max_val=10.0, default=1.0)
            config['degree'] = degree
            config['coef0'] = coef0

        if kernel_options[kernel_idx] == 'sigmoid':
            coef0 = ask_float("Coef0 (0.0-10.0, default=0.0):",
                              min_val=0.0, max_val=10.0, default=0.0)
            config['coef0'] = coef0

    print(f"\nConfiguration: {config}")
    pause()
    return config


# ============================================================================
# Core interactive functions
# ============================================================================

def load_data_interactive(s: MultinomialAppState) -> None:
    """Load dataset interactively."""
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


def select_features_interactive(s: MultinomialAppState) -> None:
    """Select features and target interactively."""
    if s.dataset is None:
        print("✗ No dataset loaded yet!")
        pause()
        return

    try:
        s.prepareddata = select_features_and_target(s.dataset)
        # Validate multiclass
        n_classes = len(np.unique(s.prepareddata.Y))
        if n_classes < 2:
            raise ValueError(f"Need at least 2 classes, found {n_classes}")
        print(f"\n✓ Multiclass target selected: {n_classes} classes")
        print(f"  Classes: {sorted(np.unique(s.prepareddata.Y).tolist())}")
        print(f"  Distribution: {dict(zip(*np.unique(s.prepareddata.Y, return_counts=True)))}")

        # Try to infer class names from column name
        target_col = s.prepareddata.target_column
        if target_col in s.dataset.data.columns:
            s.class_names = sorted(s.dataset.data[target_col].astype(str).unique().tolist())
            print(f"  Class names: {s.class_names}")

        rebuild_split(s)
        print("✓ Split rebuilt")
        pause()
    except ValueError as e:
        print(f"✗ Error: {e}")
        pause()


def configure_split_interactive(s: MultinomialAppState) -> None:
    """Configure train/test split."""
    if s.dataset is None:
        print("✗ No dataset loaded yet!")
        pause()
        return

    print("\n" + "=" * 70)
    print("CONFIGURE TRAIN/TEST SPLIT")
    print("=" * 70)
    s.test_size = ask_float("Test set size (0.05-0.5):", min_val=0.05, max_val=0.5, default=0.2)
    s.seed = ask_int("Random seed:", min_val=0, max_val=10000, default=42)
    s.use_scaling = ask_yes_no("Use feature scaling?", default=True)
    rebuild_split(s)
    print("✓ Split updated")
    pause()


def configure_model_interactive(s: MultinomialAppState) -> None:
    """Configure multiclass SVM model."""
    config = configure_multiclass_hyperparameters()

    s.base_estimator_type = config.get('base_estimator_type', 'linear')
    s.C = config.get('C', 1.0)
    s.learning_rate = config.get('learning_rate', 0.001)
    s.epochs = config.get('epochs', 1000)

    if s.base_estimator_type == 'kernel':
        s.kernel = config.get('kernel', 'rbf')
        s.gamma = config.get('gamma', 1.0)
        s.degree = config.get('degree', 3)
        s.coef0 = config.get('coef0', 1.0)

    print("✓ Model configured")


def train_model_interactive(s: MultinomialAppState) -> None:
    """Train multiclass SVM model."""
    if s.X_train is None or s.y_train is None:
        print("✗ No training data!")
        pause()
        return

    print("\n" + "=" * 70)
    print("TRAINING MULTICLASS SVM (One-vs-Rest)")
    print("=" * 70)

    try:
        if s.base_estimator_type == 'linear':
            base_estimator = LinearSVM(C=s.C, learning_rate=s.learning_rate, epochs=s.epochs)
        else:
            base_estimator = KernelSVM(
                kernel=s.kernel, C=s.C,
                gamma=s.gamma, degree=s.degree, coef0=s.coef0,
                learning_rate=s.learning_rate, epochs=s.epochs
            )

        s.model = OneVsRestSVM(base_estimator=base_estimator)
        s.model.fit(s.X_train, s.y_train)

        print(f"✓ Training complete!")
        print(f"  Number of classifiers: {s.model.n_classes}")
        for k, est in enumerate(s.model.estimators):
            n_sv = getattr(est, 'n_support_vectors', 0)
            print(f"  Class {k}: {n_sv} support vectors")
        pause()
    except Exception as e:
        print(f"✗ Training error: {e}")
        s.model = None
        pause()


def evaluate_model_interactive(s: MultinomialAppState) -> None:
    """Evaluate multiclass SVM model."""
    if s.model is None or not s.model.is_trained:
        print("✗ No trained model!")
        pause()
        return
    if s.X_test is None or s.y_test is None:
        print("✗ No test data!")
        pause()
        return

    print("\n" + "=" * 70)
    print("EVALUATING MULTICLASS SVM")
    print("=" * 70)

    try:
        y_pred = s.model.predict(s.X_test)
        s.metrics = {}

        # Multiclass metrics
        s.metrics['accuracy'] = accuracy(s.y_test, y_pred)
        s.metrics['macro_f1'] = multiclass_f1_score(s.y_test, y_pred, average='macro')
        s.metrics['micro_f1'] = multiclass_f1_score(s.y_test, y_pred, average='micro')

        # Print report
        if s.class_names is not None:
            print_multiclass_classification_report(s.y_test, y_pred, s.class_names)
        else:
            print_multiclass_classification_report(s.y_test, y_pred)

        # Confusion matrix
        from .metrics import multiclass_confusion_matrix
        n_classes = s.model.n_classes
        cm = multiclass_confusion_matrix(s.y_test, y_pred, n_classes)
        print(f"\nConfusion Matrix:\n{cm}")

        if ask_yes_no("Show confusion matrix plot?", default=True):
            from .visualization import plot_confusion_matrix
            names = s.class_names if s.class_names else None
            plot_confusion_matrix(cm, names)

        # Decision boundary if 2D
        if s.X_test.shape[1] == 2:
            if ask_yes_no("Show decision boundary plot?", default=True):
                plot_svm_decision_boundary_2d(
                    s.model, s.X_test, s.y_test,
                    feature_names=s.prepareddata.feature_names if s.prepareddata else None
                )

        print("✓ Evaluation complete")
        pause()
    except Exception as e:
        print(f"✗ Evaluation error: {e}")
        pause()


def predict_single_interactive(s: MultinomialAppState) -> None:
    """Make a single prediction."""
    if s.model is None or not s.model.is_trained:
        print("✗ No trained model!")
        pause()
        return
    if s.prepareddata is None:
        print("✗ No feature names!")
        pause()
        return

    print("\n" + "=" * 70)
    print("MAKE A PREDICTION")
    print("=" * 70)
    print("Enter values for each feature (press Enter for default 0.0):\n")

    values = []
    for name in s.prepareddata.feature_names:
        while True:
            try:
                val_str = input(f"  {name}: ").strip()
                if val_str == "":
                    values.append(0.0)
                    break
                values.append(float(val_str))
                break
            except ValueError:
                print("  Please enter a valid number.")

    X_input = np.array(values, dtype=float).reshape(1, -1)

    # Apply scaling
    if s.use_scaling and s.scaler_mean is not None and s.scaler_std is not None:
        X_input = standardize_apply(X_input, s.scaler_mean, s.scaler_std)

    try:
        pred = s.model.predict(X_input)[0]
        scores = s.model.decision_function(X_input)[0]

        print("\n" + "=" * 70)
        print("PREDICTION RESULT")
        print("=" * 70)
        class_name = s.class_names[pred] if s.class_names and pred < len(s.class_names) else f"Class {pred}"
        print(f"  Predicted class: {class_name}")
        print(f"\n  Per-class scores:")
        for k in range(s.model.n_classes):
            cls_name = s.class_names[k] if s.class_names and k < len(s.class_names) else str(k)
            print(f"    Class {cls_name}: {scores[k]:.4f}")
        print("=" * 70 + "\n")
        pause()
    except Exception as e:
        print(f"✗ Prediction error: {e}")
        pause()


# ============================================================================
# Menu functions
# ============================================================================

def menu_data(s: MultinomialAppState) -> None:
    """Data menu."""
    while True:
        clear_screen()
        print_header("Multiclass SVM — Data")
        print_status(s)
        options = ["Load dataset (CSV / Manual)",
                    "Select features + target",
                    "Configure train/test split", "Back"]
        choice = ask_choice("", options)
        if choice == 0:
            load_data_interactive(s)
        elif choice == 1:
            select_features_interactive(s)
        elif choice == 2:
            configure_split_interactive(s)
        else:
            return


def menu_train(s: MultinomialAppState) -> None:
    """Train menu."""
    while True:
        clear_screen()
        print_header("Multiclass SVM — Train")
        print_status(s)
        options = ["Configure model", "Train model", "Back"]
        choice = ask_choice("", options)
        if choice == 0:
            configure_model_interactive(s)
        elif choice == 1:
            train_model_interactive(s)
        else:
            return


def menu_evaluate(s: MultinomialAppState) -> None:
    """Evaluate menu."""
    while True:
        clear_screen()
        print_header("Multiclass SVM — Evaluate")
        print_status(s)
        options = ["Evaluate on test set",
                    "Explain multiclass metrics", "Back"]
        choice = ask_choice("", options)
        if choice == 0:
            evaluate_model_interactive(s)
        elif choice == 1:
            clear_screen()
            print_header("Metrics Explained")
            print("\nACCURACY:")
            print("─" * 70)
            print("  Overall correct predictions / total")

            print("\nMACRO AVERAGE:")
            print("─" * 70)
            print("  • Average of per-class metrics (unweighted)")
            print("  • Gives equal importance to all classes")
            print("  • Good for balanced datasets")

            print("\nMICRO AVERAGE:")
            print("─" * 70)
            print("  • Global TP / (TP+FP) — aggregates all classes")
            print("  • Same as accuracy for F1-micro")
            print("  • Good for imbalanced datasets")

            print("\nCONFUSION MATRIX:")
            print("─" * 70)
            print("  • Row = true class, Col = predicted class")
            print("  • Diagonal = correct predictions")
            print("  • Off-diagonal = misclassifications")

            print("\nONE-VS-REST STRATEGY:")
            print("─" * 70)
            print("  • Trains K binary SVM classifiers (one per class)")
            print("  • Each: class k vs all others")
            print("  • Prediction = argmax of decision scores")
            pause()
        else:
            return


def menu_save_load(s: MultinomialAppState) -> None:
    """Save/Load multiclass SVM sessions."""
    adapter = OneVsRestSVMSessionAdapter()
    storage = SessionStorage()

    while True:
        clear_screen()
        print_header("Multiclass SVM — Save/Load")
        print_status(s)

        options = ["Save session", "Load session",
                    "List sessions", "Delete session", "Back"]
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

            name = input("Session name: ").strip()
            if not name:
                print("!Invalid name!")
                pause()
                continue

            try:
                session_data, arrays = adapter.extract(s)
                storage.save_session(session_data, f"./ml_sessions/{name}", arrays, verbose=True)
                print(f"✓ Saved: {name}")
            except Exception as e:
                print(f"!Error: {e}")
            pause()

        elif choice == 1:
            sessions = storage.list_sessions()
            if not sessions:
                print("!No sessions!")
                pause()
                continue

            print("\nAvailable sessions:")
            for i, name in enumerate(sessions, 1):
                print(f"{i}. {name}")

            idx = ask_int("Select: ", min_val=1, max_val=len(sessions)) - 1
            name = sessions[idx]
            try:
                sd, arrays = storage.load_session(f"./ml_sessions/{name}", verbose=True)
                adapter.restore(sd, arrays, s)
                print(f"✓ Loaded: {name}")
                if s.metrics:
                    print(f"  Metrics: {', '.join(f'{k}={v:.4f}' for k, v in s.metrics.items())}")
                if s.class_names:
                    print(f"  Classes: {s.class_names}")
            except Exception as e:
                print(f"!Error: {e}")
            pause()

        elif choice == 2:
            sessions = storage.list_sessions()
            if not sessions:
                print("!No sessions!")
            else:
                print("\nSaved sessions:")
                for name in sessions:
                    print(f"  • {name}")
            pause()

        elif choice == 3:
            sessions = storage.list_sessions()
            if not sessions:
                print("!No sessions!")
                pause()
                continue
            name = input("Session name to delete: ").strip()
            if name in sessions and ask_yes_no(f"Delete '{name}'? "):
                storage.delete_session(f"./ml_sessions/{name}", verbose=True)
                print("Deleted.")
            else:
                print("Not found or cancelled.")
            pause()
        else:
            return


def menu_predict(s: MultinomialAppState) -> None:
    """Predict menu."""
    while True:
        clear_screen()
        print_header("Multiclass SVM — Predict")
        print_status(s)
        options = ["Make a single prediction",
                    "Batch predict from CSV", "Back"]
        choice = ask_choice("", options)
        if choice == 0:
            predict_single_interactive(s)
        elif choice == 1:
            if s.model is None or not s.model.is_trained:
                print("!Model not trained!")
                pause()
                continue

            csv_path = input("\nCSV path: ").strip()
            if not csv_path:
                print("!Invalid path!")
                pause()
                continue

            base_name = os.path.splitext(os.path.basename(csv_path))[0]
            default_output = f"predictions_multiclass_svm_{base_name}.csv"
            output_csv = input(f"Output [{default_output}]: ").strip() or default_output

            try:
                result = batch_predict_from_csv(
                    csv_path=csv_path,
                    model=s.model,
                    feature_names=s.prepareddata.feature_names,
                    use_scaling=s.use_scaling,
                    scaler_mean=s.scaler_mean,
                    scaler_std=s.scaler_std,
                    output_path=output_csv,
                    model_type="multiclass_svm",
                )
                print(f"  ✓ Processed {result['n_samples']} rows!")
                print(f"  ✓ Output: {result['output_path']}")
            except Exception as e:
                print(f"  ✗ Error: {e}")
            pause()
        else:
            return


def menu_visualize(s: MultinomialAppState) -> None:
    """Visualize menu."""
    while True:
        clear_screen()
        print_header("Multiclass SVM — Visualize")
        print_status(s)
        options = [
            "Plot confusion matrix",
            "Plot decision boundary (2D only)",
            "Support vector info",
            "Back",
        ]
        choice = ask_choice("", options)
        if choice == 0:
            if s.model is None or not s.model.is_trained:
                print("!No model!")
                pause()
                continue
            if s.X_test is None or s.y_test is None:
                print("!No test data!")
                pause()
                continue
            y_pred = s.model.predict(s.X_test)
            from .metrics import multiclass_confusion_matrix
            cm = multiclass_confusion_matrix(s.y_test, y_pred, s.model.n_classes)
            names = s.class_names if s.class_names else None
            plot_confusion_matrix(cm, names)
            pause()
        elif choice == 1:
            if s.model is None or not s.model.is_trained:
                print("!No model!")
                pause()
                continue
            if s.X_test is None or s.X_test.shape[1] != 2:
                print("!Need exactly 2 features!")
                pause()
                continue
            plot_svm_decision_boundary_2d(
                s.model, s.X_test, s.y_test,
                feature_names=s.prepareddata.feature_names if s.prepareddata else None
            )
            pause()
        elif choice == 2:
            if s.model is None or not s.model.is_trained:
                print("!No model!")
                pause()
                continue
            plot_support_vector_info(s.model)
            pause()
        else:
            return
