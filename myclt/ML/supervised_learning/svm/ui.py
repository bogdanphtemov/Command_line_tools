"""
User interface helpers for Binary SVM (Classification + Regression).

Provides interactive prompts and data selection dialogs for both
SVM classification (LinearSVM, KernelSVM) and SVR (LinearSVR, KernelSVR).
"""

import numpy as np
import os
from typing import List, Optional, Dict, Any

from .app_state import AppState, print_status, rebuild_split
from .data import Dataset, Prepareddata, load_csv_dataset, manual_input_dataset
from .core import LinearSVM, KernelSVM, LinearSVR, KernelSVR
from .preprocessing import standardize_apply
from .hyperparameter_optimization import fast_auto_optimize
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
from myclt.common.input_validation import (
    ask_yes_no, ask_int, ask_float, ask_choice,
    ask_yes_no_recommended, ask_auto_or_float, ask_auto_or_int
)
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
        print(f"\nâœ“ Binary classification: classes = {sorted(unique_values.tolist())}")
        print(f"  Class distribution: {dict(zip(*np.unique(prepared.Y, return_counts=True)))}")
    else:
        print(f"\nâœ“ Regression target selected: {prepared.Y.shape[0]} samples")
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

    C = ask_auto_or_float("Regularization C", min_val=0.001, max_val=1000.0)
    lr = ask_auto_or_float("Learning rate", min_val=0.0001, max_val=1.0)
    epochs = ask_int("Max epochs", min_val=100, max_val=100000, default=10000)

    # Epsilon with auto option for regression
    epsilon_auto = False
    epsilon_value = 0.1
    if mode == 'regressor':
        from myclt.common.input_validation import ask_epsilon_with_auto
        epsilon_input = ask_epsilon_with_auto("Epsilon (Îµ-insensitive tube, 0.001-1.0)")
        epsilon_auto = (epsilon_input is None)
        epsilon_value = epsilon_input or 0.1

    print("\n" + "=" * 70)
    print("Configuration Summary:")
    print(f"  C: {'AUTO' if C is None else C}")
    print(f"  Learning rate: {'AUTO' if lr is None else lr}")
    print(f"  Max epochs: {epochs}")
    if mode == 'regressor':
        print(f"  Epsilon: {'AUTO' if epsilon_auto else epsilon_value}")
    print(f"  Early stopping: ON")
    print("=" * 70)

    params = {
        'C': C,
        'learning_rate': lr,
        'epochs': epochs,
        'C_auto': (C is None),
        'learning_rate_auto': (lr is None),
    }

    if mode == 'regressor':
        params['epsilon'] = epsilon_value
        params['epsilon_auto'] = epsilon_auto

    return params


def configure_kernel_hyperparameters_with_auto() -> dict | None:
    """
    Interactive kernel hyperparameter configuration with auto optimization option.

    Returns:
        Dictionary with kernel parameters or None for auto optimization
    """
    from myclt.common.input_validation import ask_kernel_with_auto
    
    kernel_choice = ask_kernel_with_auto()
    
    if kernel_choice is None:
        return None  # Auto optimization
    
    print("\n" + "=" * 70)
    print("KERNEL CONFIGURATION")
    print("=" * 70)

    kernel = kernel_choice
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


def configure_kernel_hyperparameters() -> dict:
    """
    Interactive kernel hyperparameter configuration.
    
    This is the original function kept for backward compatibility.
    Use configure_kernel_hyperparameters_with_auto() for new features.
    
    Returns:
        Dictionary with kernel parameters (kernel, gamma, degree, coef0)
    """
    auto_result = configure_kernel_hyperparameters_with_auto()
    if auto_result is None:
        # If auto was selected, use default parameters
        return {
            'kernel': 'rbf',
            'gamma': 1.0,
            'degree': 3,
            'coef0': 0.0
        }
    return auto_result


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
            print(f"âœ“ Loaded: {s.dataset.data.shape[0]} rows Ã— {s.dataset.data.shape[1]} columns")
        except FileNotFoundError as e:
            print(f"âœ— Error: {e}")
    else:
        try:
            s.dataset = manual_input_dataset()
            print(f"âœ“ Created: {s.dataset.data.shape[0]} rows Ã— {s.dataset.data.shape[1]} columns")
        except ValueError as e:
            print(f"âœ— Error: {e}")
    pause()


def select_features_interactive(s: AppState) -> None:
    """Select features and target interactively."""
    if s.dataset is None:
        print("âœ— No dataset loaded yet!")
        pause()
        return
    try:
        s.prepareddata = select_features_and_target_binary(s.dataset, s.mode)
        rebuild_split(s)
        print("âœ“ Features and target selected")
        pause()
    except ValueError as e:
        print(f"âœ— Error: {e}")
        pause()


def configure_split_interactive(s: AppState) -> None:
    """Configure train/test split."""
    if s.dataset is None:
        print("âœ— No dataset loaded yet!")
        return
    print("\n" + "=" * 70)
    print("CONFIGURE TRAIN/TEST SPLIT")
    print("=" * 70)
    s.test_size = ask_float("Test size (0.05-0.5)", min_val=0.05, max_val=0.5, default=0.2)
    s.seed = ask_int("Random seed (integer)", min_val=0, max_val=10000, default=42)
    s.use_scaling = ask_yes_no_recommended("Use feature scaling (standardization)?", recommended=True)
    rebuild_split(s)
    print("âœ“ Split configuration updated")
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
    s.C_auto = common['C'] is None
    if common['C'] is not None:
        s.C = common['C']
    s.learning_rate_auto = common['learning_rate'] is None
    if common['learning_rate'] is not None:
        s.learning_rate = common['learning_rate']
    s.epochs = common['epochs']
    s.early_stopping = True
    if s.mode == 'regressor' and 'epsilon' in common:
        s.epsilon = common['epsilon']

    # Kernel params (if applicable) with auto option
    if s.model_type in ("kernel_svm", "kernel_svr"):
        kernel_params = configure_kernel_hyperparameters_with_auto()
        if kernel_params is None:
            # Auto optimization selected
            s.kernel_auto = True
            s.kernel = 'rbf'  # Default until optimized
            s.gamma = 1.0
            s.degree = 3
            s.coef0 = 0.0
        else:
            # Manual configuration
            s.kernel_auto = False
            s.kernel = kernel_params.get('kernel', 'rbf')
            s.gamma = kernel_params.get('gamma', 1.0)
            s.degree = kernel_params.get('degree', 3)
            s.coef0 = kernel_params.get('coef0', 1.0)

    print("âœ“ Model configured")


def auto_optimize_all_params(s: AppState) -> Dict[str, Any]:
    """Optimize all parameters using our fast optimization algorithm."""
    print("\n" + "=" * 70)
    print("ðŸš€ FAST AUTOMATIC OPTIMIZATION")
    print("=" * 70)
    
    # Ask for time limit
    from myclt.common.input_validation import ask_int
    max_minutes = ask_int("Maximum optimization time (minutes)", 
                         min_val=5, max_val=120, default=15)
    
    print(f"\n>> Starting fast optimization (max {max_minutes} minutes)...")
    print(">> Using 3-stage adaptive sampling for best results")
    
    # Use our fast optimizer
    optimized_params = fast_auto_optimize(
        X=s.X_train, 
        y=s.y_train,
        mode=s.mode,
        max_time_minutes=max_minutes
    )
    
    # Update app state with optimized parameters
    if optimized_params:
        s.C = optimized_params.get('C', s.C)
        s.learning_rate = optimized_params.get('learning_rate', s.learning_rate)
        
        # For kernel models
        if hasattr(s, 'kernel') and 'kernel' in optimized_params:
            s.kernel = optimized_params['kernel']
        if hasattr(s, 'gamma') and 'gamma' in optimized_params:
            s.gamma = optimized_params['gamma']
        if hasattr(s, 'epsilon') and 'epsilon' in optimized_params:
            s.epsilon = optimized_params['epsilon']
        
        print("\n" + "=" * 70)
        print("âœ… OPTIMIZATION COMPLETE")
        print("=" * 70)
        metric_name = 'RÂ² Score' if s.mode == 'regressor' else 'Accuracy'
        metric_value = optimized_params.get('score', -float('inf'))
        print(f"Best {metric_name}: {metric_value:.4f}")
        print(f"Target: {'â‰¥ 0.80' if s.mode == 'regressor' else 'â‰¥ 0.85'}")
        print(f"Optimal Parameters:")
        for key, value in optimized_params.items():
            if key != 'score':
                print(f"  {key}: {value}")
        
        return optimized_params
    
    return None


def train_model_interactive(s: AppState) -> None:
    """Train SVM model interactively with auto-tuning."""
    if s.X_train is None or s.y_train is None:
        print("âœ— No training data prepared yet!")
        return

    print("\n" + "=" * 70)
    print(f"TRAINING {s.model_type.upper()}")
    print("=" * 70)

    try:
        # First check if we should use fast optimization (works for both regression and classification)
        auto_mode = False
        
        # Debug info
        print(f"DEBUG: learning_rate_auto={s.learning_rate_auto}")
        print(f"DEBUG: C_auto={s.C_auto}")
        print(f"DEBUG: kernel_auto={getattr(s, 'kernel_auto', False)}")
        print(f"DEBUG: mode={s.mode}")
        
        if (s.learning_rate_auto and s.C_auto and 
            (s.mode == 'regressor' or s.mode == 'classifier') and
            (hasattr(s, 'kernel_auto') and s.kernel_auto)):
            
            from myclt.common.input_validation import ask_yes_no
            auto_mode = ask_yes_no("Use fast automatic optimization (3-stage)?", default=True)
            
            if auto_mode:
                optimized_params = auto_optimize_all_params(s)
                if optimized_params:
                    # Skip individual tuning since we have all optimized params
                    s.learning_rate_auto = False
                    s.C_auto = False
        
        # Auto-tune learning rate if needed (and not done by fast optimization)
        if s.learning_rate_auto and not auto_mode:
            from .hyperparameter_tuning import auto_tune_learning_rate
            model_class = LinearSVM if s.model_type == 'linear_svm' else (
                KernelSVM if s.model_type == 'kernel_svm' else (
                    LinearSVR if s.model_type == 'linear_svr' else KernelSVR
                )
            )
            task = 'classifier' if s.mode == 'classifier' else 'regressor'
            lr_result = auto_tune_learning_rate(
                X=s.X_train, y=s.y_train,
                model_class=model_class, C=s.C,
                max_epochs=2000, k_folds=3, seed=s.seed,  # Reduced for faster testing
                use_scaling=False, task=task, verbose=True
            )
            s.learning_rate = lr_result['best_lr']
            print(f"\n=> Learning rate set to: {s.learning_rate}")

        # Auto-tune C if needed
        if s.C_auto:
            from .hyperparameter_tuning import auto_tune_C
            model_class = LinearSVM if s.model_type == 'linear_svm' else (
                KernelSVM if s.model_type == 'kernel_svm' else (
                    LinearSVR if s.model_type == 'linear_svr' else KernelSVR
                )
            )
            task = 'classifier' if s.mode == 'classifier' else 'regressor'
            C_result = auto_tune_C(
                X=s.X_train, y=s.y_train,
                model_class=model_class, learning_rate=s.learning_rate,
                max_epochs=2000, k_folds=3, seed=s.seed,  # Reduced for faster testing
                use_scaling=False, task=task, verbose=True
            )
            s.C = C_result['best_C']
            print(f"\n=> C set to: {s.C}")

        # Auto-tune kernel and epsilon if needed (kernel-based models and SVR)
        if hasattr(s, 'kernel_auto') and s.kernel_auto:
            # Use new auto optimization for kernel parameters
            print("\nðŸ”§ Starting automatic kernel optimization...")
            from .hyperparameter_optimization import auto_optimize_parameters
            
            # Create temporary dataset for optimization
            class TempPreparedData:
                def __init__(self, X, y):
                    self.X = X
                    self.y = y
            
            temp_data = TempPreparedData(s.X_train, s.y_train)
            
            # Run optimization
            optimized_params = auto_optimize_parameters(
                temp_data, mode=s.mode, max_time_minutes=8
            )
            
            # Apply optimized parameters
            s.kernel = optimized_params['kernel']
            s.gamma = optimized_params['gamma']
            s.degree = optimized_params['degree']
            s.coef0 = optimized_params['coef0']
            if s.mode == 'regressor':
                s.epsilon = optimized_params['epsilon']
            
            print("âœ“ Kernel and parameters optimized")
        elif s.mode == 'regressor' and hasattr(s, 'epsilon_auto') and s.epsilon_auto:
            # Auto-tune only epsilon for SVR
            print("\nðŸ”§ Starting automatic epsilon optimization...")
            from .hyperparameter_optimization import SVMHyperparameterOptimizer
            
            # Create validation split
            from myclt.ML.base_models import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                s.X_train, s.y_train, test_size=0.2, seed=s.seed
            )
            
            optimizer = SVMHyperparameterOptimizer(max_time_minutes=3)
            s.epsilon = optimizer.optimize_epsilon_param(
                X_train, y_train, X_val, y_val, s.mode
            )
            print(f"âœ“ Epsilon optimized to: {s.epsilon}")

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

        # Train with early stopping (always ON)
        n_train = len(s.X_train)
        val_size = int(0.2 * n_train)
        X_train_part = s.X_train[val_size:]
        y_train_part = s.y_train[val_size:]
        X_val = s.X_train[:val_size]
        y_val = s.y_train[:val_size]
        patience = ask_int("Patience (epochs without improvement)",
                           min_val=5, max_val=200, default=50)

        print(f"\nTraining with early stopping (patience={patience}, min_delta=1e-6)...")
        s.model.fit_with_early_stopping(
            X_train_part, y_train_part, X_val, y_val,
            patience=patience, min_delta=1e-6, verbose=True
        )

        print(f"âœ“ Training complete ({len(s.model.loss_history)} epochs / {s.epochs} max)")
        if hasattr(s.model, 'n_support_vectors'):
            print(f"  Support vectors: {s.model.n_support_vectors}")

        # Show loss history
        if s.model.loss_history and ask_yes_no("Show loss curve?", default=True):
            if s.mode == 'classifier':
                plot_loss_curve(s.model.loss_history, ylabel="Hinge Loss",
                                title=f"{s.model_type.upper()} Loss Curve")
            else:
                plot_loss_curve(s.model.loss_history, ylabel="Îµ-Insensitive Loss",
                                title=f"{s.model_type.upper()} Loss Curve")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"âœ— Training error: {e}")
        s.model = None


def evaluate_model_interactive(s: AppState) -> None:
    """Evaluate model interactively."""
    if s.model is None or not s.model.is_trained:
        print("âœ— No trained model!")
        return
    if s.X_test is None or s.y_test is None:
        print("âœ— No test data!")
        return

    print("\n" + "=" * 70)
    print(f"EVALUATING {s.model_type.upper()}")
    print("=" * 70)

    try:
        y_pred = s.model.predict(s.X_test)
        
        # Convert y_test to numeric if it contains non-numeric labels
        y_test_num = s.y_test
        unique_test = list(np.unique(s.y_test))
        if not set(unique_test).issubset({0, 1, -1}):
            # Labels are non-numeric (e.g., strings) -> convert to {0, 1}
            y_test_num = np.where(s.y_test == unique_test[1], 1, 0).astype(int)
        
        s.metrics = {}

        if s.mode == 'classifier':
            # Classification metrics
            s.metrics['accuracy'] = accuracy(y_test_num, y_pred)

            # Try F1 score
            try:
                s.metrics['f1'] = f1_score(y_test_num, y_pred)
            except Exception:
                pass

            # Classification report
            print(classification_report(y_test_num, y_pred))

            # Confusion matrix
            cm = confusion_matrix(y_test_num, y_pred)
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
                if ask_yes_no("Show Îµ-tube visualization?", default=False):
                    plot_svr_tube(s.X_test, s.y_test, y_pred,
                                  epsilon=s.epsilon,
                                  feature_idx=0)

        print("âœ“ Evaluation complete")

    except Exception as e:
        print(f"âœ— Evaluation error: {e}")


def predict_single_interactive(s: AppState) -> None:
    """Make a single prediction."""
    if s.model is None or not s.model.is_trained:
        print("âœ— No trained model!")
        return
    if s.prepareddata is None:
        print("âœ— No feature names available!")
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
        print(f"âœ— Prediction error: {e}")


# ============================================================================
# Menu functions
# ============================================================================

def menu_data(s: AppState) -> None:
    """Data menu."""
    while True:
        clear_screen()
        print_header(f"SVM â€” Data ({'Classification' if s.mode == 'classifier' else 'Regression'})")
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
        print_header("SVM â€” Train")
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
        print_header("SVM â€” Evaluate")
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
                print_header("SVM Classification â€” Metrics Explained")
                print("\nACCURACY:")
                print("â”€" * 70)
                print("  â€¢ What: Percentage of correct predictions")
                print("  â€¢ Formula: correct / total")

                print("\nPRECISION:")
                print("â”€" * 70)
                print("  â€¢ What: Of predicted positives, how many are correct")
                print("  â€¢ Formula: TP / (TP + FP)")

                print("\nRECALL:")
                print("â”€" * 70)
                print("  â€¢ What: Of actual positives, how many were found")
                print("  â€¢ Formula: TP / (TP + FN)")

                print("\nF1 SCORE:")
                print("â”€" * 70)
                print("  â€¢ Harmonic mean of precision and recall")
                print("  â€¢ Formula: 2 Ã— P Ã— R / (P + R)")

                print("\nCONFUSION MATRIX:")
                print("â”€" * 70)
                print("  â€¢ [[TN, FP], [FN, TP]]")
                print("  â€¢ Diagonal = correct, off-diagonal = errors")

                print("\nSUPPORT VECTORS:")
                print("â”€" * 70)
                print("  â€¢ Data points that determine the decision boundary")
                print("  â€¢ Points with margin â‰¤ 1")
                print("  â€¢ Only SVs affect the model (sparsity!)")
            else:
                print_header("SVR Regression â€” Metrics Explained")
                print("\nMSE (Mean Squared Error):")
                print("â”€" * 70)
                print("  â€¢ Average squared difference between true and predicted")
                print("  â€¢ Penalizes large errors more")

                print("\nRMSE (Root Mean Squared Error):")
                print("â”€" * 70)
                print("  â€¢ Square root of MSE")
                print("  â€¢ Same units as target variable")

                print("\nMAE (Mean Absolute Error):")
                print("â”€" * 70)
                print("  â€¢ Average absolute difference")
                print("  â€¢ Less sensitive to outliers than MSE")

                print("\nRÂ² (Coefficient of Determination):")
                print("â”€" * 70)
                print("  â€¢ How well the model explains variance")
                print("  â€¢ 1.0 = perfect, 0.0 = mean predictor, < 0 = worse than mean")

                print("\nÎµ-INSENSITIVE TUBE:")
                print("â”€" * 70)
                print("  â€¢ Errors within Â±Îµ are ignored")
                print("  â€¢ Points outside the tube = support vectors")
                print("  â€¢ Controls sparsity of the solution")
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
        print_header(f"SVM â€” Save/Load Session ({'Classifier' if s.mode == 'classifier' else 'Regression'})")
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
                print(f"\nâœ“ Session '{session_name}' saved!")
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

                print(f"\nâœ“ Session '{session_name}' loaded!")
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
                        print(f" âœ“ {name} (dataset: {ds_shape})")
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
        print_header("SVM â€” Predict")
        print_status(s)
        options = ["Make a single prediction",
                    "Batch predict from CSV file", "Back"]
        choice = ask_choice("", options)
        if choice == 0:
            predict_single_interactive(s)
            pause()
        elif choice == 1:
            if s.model is None or not s.model.is_trained:
                print("âœ— Model not trained!")
                pause()
                continue
            if s.prepareddata is None:
                print("âœ— No features selected!")
                pause()
                continue

            csv_path = input("\nCSV path: ").strip()
            if not csv_path:
                print("âœ— Invalid path!")
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
                print(f"\n  âœ“ Processed {result['n_samples']} rows!")
                print(f"  âœ“ Output: {result['output_path']}")
            except Exception as e:
                print(f"\n  âœ— Error: {e}")
            pause()
        else:
            return


def menu_visualize(s: AppState) -> None:
    """Visualize menu."""
    while True:
        clear_screen()
        print_header("SVM â€” Visualize")
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
                print("âœ— No loss history!")
                pause()
                continue
            plot_loss_curve(s.model.loss_history)
            pause()
        elif choice == 1:
            if s.model is None or not s.model.is_trained:
                print("âœ— No trained model!")
                pause()
                continue
            if s.X_test is None or s.X_test.shape[1] != 2:
                print("âœ— Need exactly 2 features!")
                pause()
                continue
            plot_svm_decision_boundary_2d(
                s.model, s.X_test, s.y_test,
                feature_names=s.prepareddata.feature_names if s.prepareddata else None
            )
            pause()
        elif choice == 2:
            if s.model is None or s.mode != 'classifier':
                print("âœ— Need trained classifier!")
                pause()
                continue
            if s.X_test is None or s.y_test is None:
                print("âœ— No test data!")
                pause()
                continue
            y_pred = s.model.predict(s.X_test)
            cm = confusion_matrix(s.y_test, y_pred)
            plot_confusion_matrix(cm)
            pause()
        elif choice == 3:
            if s.model is None or s.mode != 'regressor':
                print("âœ— Need trained regressor!")
                pause()
                continue
            if s.X_test is None or s.y_test is None:
                print("âœ— No test data!")
                pause()
                continue
            y_pred = s.model.predict(s.X_test)
            plot_true_vs_pred(s.y_test, y_pred)
            pause()
        elif choice == 4:
            if s.model is None or s.mode != 'regressor':
                print("âœ— Need trained regressor!")
                pause()
                continue
            if s.X_test is None or s.y_test is None:
                print("âœ— No test data!")
                pause()
                continue
            y_pred = s.model.predict(s.X_test)
            plot_residuals(s.y_test, y_pred)
            pause()
        elif choice == 5:
            if s.model is None or not s.model.is_trained:
                print("âœ— No trained model!")
                pause()
                continue
            plot_support_vector_info(s.model)
            pause()
        else:
            return
