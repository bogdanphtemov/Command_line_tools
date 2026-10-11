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
import time
from typing import Optional, Dict, Any

from .multinomial_app_state import MultinomialAppState, print_status, rebuild_split
from .data import Dataset, Prepareddata, load_csv_dataset, manual_input_dataset
from .core import OneVsRestSVM, LinearSVM, KernelSVM
from .preprocessing import standardize_apply
from myclt.ML.base_models import standardize_fit
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
from myclt.common.input_validation import (
    ask_yes_no, ask_int, ask_float, ask_choice, 
    ask_yes_no_recommended, ask_auto_or_float, ask_kernel_with_auto
)
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

    # C parameter with AUTO option
    C = ask_auto_or_float("Regularization C (0.001-1000)",
                         min_val=0.001, max_val=1000.0)
    config['C'] = C
    config['C_auto'] = (C is None)

    # Learning rate with AUTO option
    lr = ask_auto_or_float("Learning rate (0.0001-0.1)",
                          min_val=0.0001, max_val=0.1)
    config['learning_rate'] = lr
    config['learning_rate_auto'] = (lr is None)

    # Epochs
    epochs = ask_int("Epochs per classifier (100-5000, default=1000):",
                     min_val=100, max_val=5000, default=1000)
    config['epochs'] = epochs

    # Kernel params with AUTO option if kernel chosen
    if config['base_estimator_type'] == 'kernel':
        kernel_choice = ask_kernel_with_auto()
        
        if kernel_choice is None:
            # Auto optimization selected
            config['kernel_auto'] = True
            config['kernel'] = 'rbf'  # Default until optimized
            config['gamma'] = 1.0
            config['degree'] = 3
            config['coef0'] = 0.0
        else:
            # Manual kernel selection
            config['kernel_auto'] = False
            config['kernel'] = kernel_choice
            
            if kernel_choice in ('rbf', 'poly', 'sigmoid'):
                gamma = ask_float("Gamma (0.0001-10, default=1.0):",
                                  min_val=0.0001, max_val=10.0, default=1.0)
                config['gamma'] = gamma

            if kernel_choice == 'poly':
                degree = ask_int("Degree (2-5, default=3):",
                                 min_val=2, max_val=5, default=3)
                coef0 = ask_float("Coef0 (0.0-10.0, default=1.0):",
                                  min_val=0.0, max_val=10.0, default=1.0)
                config['degree'] = degree
                config['coef0'] = coef0

            if kernel_choice == 'sigmoid':
                coef0 = ask_float("Coef0 (0.0-10.0, default=0.0):",
                                  min_val=0.0, max_val=10.0, default=0.0)
                config['coef0'] = coef0

    print("\n" + "=" * 70)
    print("Configuration Summary:")
    print(f"  C: {'AUTO' if config['C_auto'] else config['C']}")
    print(f"  Learning rate: {'AUTO' if config['learning_rate_auto'] else config['learning_rate']}")
    print(f"  Max epochs: {epochs}")
    if config['base_estimator_type'] == 'kernel':
        print(f"  Kernel: {'AUTO' if config.get('kernel_auto', False) else config['kernel']}")
    print("=" * 70)
    
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
            pause()
        except FileNotFoundError as e:
            print(f"✗ Error: {e}")
            pause()
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
        target_col = s.prepareddata.target_name
        if target_col in s.dataset.columns:
            # Get unique class names directly from the target column in the dataset
            target_col_idx = s.dataset.columns.index(target_col)
            unique_vals = np.unique(s.dataset.data[:, target_col_idx])
            s.class_names = [str(val) for val in sorted(unique_vals)]
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
    s.test_size = ask_float("Test size (0.05-0.5)", min_val=0.05, max_val=0.5, default=0.2)
    s.seed = ask_int("Random seed (integer)", min_val=0, max_val=10000, default=42)
    s.use_scaling = ask_yes_no_recommended("Use feature scaling?", recommended=True)
    rebuild_split(s)
    print("✓ Split updated")
    pause()


def configure_model_interactive(s: MultinomialAppState) -> None:
    """Configure multiclass SVM model."""
    config = configure_multiclass_hyperparameters()

    s.base_estimator_type = config.get('base_estimator_type', 'linear')
    s.C = config.get('C', 1.0)
    s.C_auto = config.get('C_auto', False)
    s.learning_rate = config.get('learning_rate', 0.001)
    s.learning_rate_auto = config.get('learning_rate_auto', False)
    s.epochs = config.get('epochs', 1000)

    if s.base_estimator_type == 'kernel':
        s.kernel = config.get('kernel', 'rbf')
        s.gamma = config.get('gamma', 1.0)
        s.degree = config.get('degree', 3)
        s.coef0 = config.get('coef0', 1.0)
        s.kernel_auto = config.get('kernel_auto', False)

    print("✓ Model configured")


# ============================================================================
# Multi-class specific optimization functions
# ============================================================================

def multiclass_auto_tune_learning_rate(
    X: np.ndarray,
    y: np.ndarray,
    base_estimator_type: str,
    C: float = 1.0,
    kernel: str = "rbf",
    gamma: float = 1.0,
    degree: int = 3,
    coef0: float = 0.0,
    max_epochs: int = 2000,
    k_folds: int = 3,
    seed: int = 42,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Auto-tune learning rate for multiclass SVM using cross-validation.
    Uses OneVsRestSVM internally to handle multiple classes.
    """
    lr_grid = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]

    if verbose:
        print("Auto-tuning learning rate for multiclass SVM...")
        print("=" * 72)
        print(f"Grid: {lr_grid}")
        print(f"Using {k_folds}-fold CV, max_epochs={max_epochs}")
        print("Using OneVsRestSVM for multiclass handling")
        print("=" * 72)

    from .preprocessing import k_fold_split, standardize_fit, standardize_apply

    folds = k_fold_split(X, y, k=k_folds, seed=seed)
    results = []

    for lr in lr_grid:
        fold_accuracies = []
        
        for _, (X_train, X_val, y_train, y_val) in enumerate(folds):
            # Scale data
            X_train_s, mean, std = standardize_fit(X_train)
            X_val_s = standardize_apply(X_val, mean, std)
            
            # Create base estimator based on type
            if base_estimator_type == 'linear':
                base_estimator = LinearSVM(C=C, learning_rate=lr, epochs=max_epochs)
            else:
                base_estimator = KernelSVM(
                    kernel=kernel, C=C, gamma=gamma, degree=degree, coef0=coef0,
                    learning_rate=lr, epochs=max_epochs
                )
            
            # Create and train multiclass model
            model = OneVsRestSVM(base_estimator=base_estimator)
            model.fit(X_train_s, y_train)
            
            # Evaluate on validation set
            y_pred = model.predict(X_val_s)
            accuracy_val = np.mean(y_pred == y_val)
            fold_accuracies.append(accuracy_val)
        
        avg_accuracy = np.mean(fold_accuracies)
        results.append({
            'lr': lr,
            'accuracy': avg_accuracy,
            'std': np.std(fold_accuracies)
        })
        
        if verbose:
            print(f"  LR={lr}: accuracy={avg_accuracy:.4f} (±{np.std(fold_accuracies):.4f})")
    
    # Find best learning rate
    best_result = max(results, key=lambda x: x['accuracy'])
    
    if verbose:
        print("=" * 72)
        print(f"✓ Best learning rate: {best_result['lr']}")
        print(f"  CV Accuracy: {best_result['accuracy']:.4f} (±{best_result['std']:.4f})")
    
    return {
        'best_lr': best_result['lr'],
        'best_accuracy': best_result['accuracy'],
        'results': results
    }


def multiclass_auto_tune_C(
    X: np.ndarray,
    y: np.ndarray,
    base_estimator_type: str,
    learning_rate: float = 0.001,
    kernel: str = "rbf",
    gamma: float = 1.0,
    degree: int = 3,
    coef0: float = 0.0,
    max_epochs: int = 2000,
    k_folds: int = 3,
    seed: int = 42,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Auto-tune C parameter for multiclass SVM using cross-validation.
    Uses OneVsRestSVM internally to handle multiple classes.
    """
    C_grid = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]

    if verbose:
        print("Auto-tuning C for multiclass SVM...")
        print("=" * 72)
        print(f"Grid: {C_grid}")
        print(f"Using {k_folds}-fold CV, max_epochs={max_epochs}")
        print("Using OneVsRestSVM for multiclass handling")
        print("=" * 72)

    from .preprocessing import k_fold_split, standardize_fit, standardize_apply

    folds = k_fold_split(X, y, k=k_folds, seed=seed)
    results = []

    for C_val in C_grid:
        fold_accuracies = []
        
        for _, (X_train, X_val, y_train, y_val) in enumerate(folds):
            # Scale data
            X_train_s, mean, std = standardize_fit(X_train)
            X_val_s = standardize_apply(X_val, mean, std)
            
            # Create base estimator based on type
            if base_estimator_type == 'linear':
                base_estimator = LinearSVM(C=C_val, learning_rate=learning_rate, epochs=max_epochs)
            else:
                base_estimator = KernelSVM(
                    kernel=kernel, C=C_val, gamma=gamma, degree=degree, coef0=coef0,
                    learning_rate=learning_rate, epochs=max_epochs
                )
            
            # Create and train multiclass model
            model = OneVsRestSVM(base_estimator=base_estimator)
            model.fit(X_train_s, y_train)
            
            # Evaluate on validation set
            y_pred = model.predict(X_val_s)
            accuracy_val = np.mean(y_pred == y_val)
            fold_accuracies.append(accuracy_val)
        
        avg_accuracy = np.mean(fold_accuracies)
        results.append({
            'C': C_val,
            'accuracy': avg_accuracy,
            'std': np.std(fold_accuracies)
        })
        
        if verbose:
            print(f"  C={C_val}: accuracy={avg_accuracy:.4f} (±{np.std(fold_accuracies):.4f})")
    
    # Find best C
    best_result = max(results, key=lambda x: x['accuracy'])
    
    if verbose:
        print("=" * 72)
        print(f"✓ Best C: {best_result['C']}")
        print(f"  CV Accuracy: {best_result['accuracy']:.4f} (±{best_result['std']:.4f})")
    
    return {
        'best_C': best_result['C'],
        'best_accuracy': best_result['accuracy'],
        'results': results
    }


def fast_auto_optimize_multiclass(s: MultinomialAppState, max_time_minutes: int = 15) -> Dict[str, Any]:
    """Fast optimization for multiclass SVM using simplified approach."""
    print("\n" + "=" * 70)
    print("🚀 FAST MULTICLASS OPTIMIZATION")
    print("=" * 70)
    
    print(f"\n>> Starting fast optimization (max {max_time_minutes} minutes)...")
    print(">> Using adaptive grid search with OneVsRestSVM")
    
    try:
        from .core import OneVsRestSVM, LinearSVM, KernelSVM
        from myclt.ML.base_models import train_test_split, standardize_fit, standardize_apply
        import time
        
        start_time = time.time()
        time_limit = max_time_minutes * 60
        
        # Prepare data
        # Standardize the data - standardize_fit returns (X_scaled, mean, std_safe)
        X_train_std, _, _ = standardize_fit(s.X_train)
        
        # Adaptive parameter grids based on dataset size
        n_samples = len(X_train_std)
        
        if s.base_estimator_type == 'linear':
            if n_samples > 500:
                param_grid = {
                    'C': [0.1, 1.0, 10.0],  # Reduced for speed
                    'learning_rate': [0.001, 0.01, 0.03, 0.1]
                }
            else:
                param_grid = {
                    'C': [0.1, 0.5, 1.0, 5.0, 10.0],
                    'learning_rate': [0.0001, 0.001, 0.01, 0.03, 0.1]
                }
        else:
            if n_samples > 500:
                param_grid = {
                    'C': [0.1, 1.0],
                    'learning_rate': [0.001, 0.01],
                    'kernel': ['linear', 'rbf'],
                    'gamma': [0.1, 1.0]
                }
            else:
                param_grid = {
                    'C': [0.1, 1.0, 10.0],
                    'learning_rate': [0.001, 0.01, 0.1],
                    'kernel': ['linear', 'rbf'],
                    'gamma': [0.01, 0.1, 1.0]
                }
        
        best_score = -float('inf')
        best_params = {}
        
        total_combinations = (len(param_grid['C']) * len(param_grid['learning_rate']) *
                            len(param_grid.get('kernel', ['linear'])) *
                            len(param_grid.get('gamma', [0.1])))
        
        print(f">> Testing {total_combinations} parameter combinations...")
        print(">> Using small sample (30%) for quick evaluation...")
        
        # Pre-split data for faster evaluation
        from myclt.ML.base_models import train_test_split
        subset_size = min(300, len(X_train_std))  # Use smaller sample for speed
        if len(X_train_std) > subset_size:
            idx = np.random.choice(len(X_train_std), subset_size, replace=False)
            X_subset = X_train_std[idx]
            y_subset = s.y_train[idx] 
            X_train_half, X_val, y_train_half, y_val, _, _ = train_test_split(
                X_subset, y_subset, test_size=0.3, seed=42
            )
        else:
            X_subset, y_subset = X_train_std, s.y_train
            X_train_half, X_val, y_train_half, y_val, _, _ = train_test_split(
                X_subset, y_subset, test_size=0.3, seed=42
            )
        
        counter = 0
        successful_combinations = 0
        
        # Test all parameter combinations
        parameter_combinations = []
        for C_val in param_grid['C']:
            for lr_val in param_grid['learning_rate']:
                for kernel_val in param_grid.get('kernel', ['linear']):
                    for gamma_val in param_grid.get('gamma', [0.1]):
                        parameter_combinations.append((C_val, lr_val, kernel_val, gamma_val))
        
        # Shuffle combinations for more even progress
        np.random.shuffle(parameter_combinations)
        
        # Track remaining time for adaptive optimization
        remaining_time = time_limit
        
        for C_val, lr_val, kernel_val, gamma_val in parameter_combinations:
            counter += 1
            
            # Update time calculations for this iteration
            current_time = time.time()
            elapsed = current_time - start_time
            
            if elapsed > time_limit:
                print(f">> 🕒 TIME LIMIT REACHED - Processed {counter-1} combinations ({elapsed/60:.1f} min)")
                break
            
            # Update progress with time remaining (recalculated)
            if counter % max(2, total_combinations // 10) == 0 or counter == len(parameter_combinations):
                progress = (counter / len(parameter_combinations) * 100)
                current_elapsed = time.time() - start_time
                current_remaining = time_limit - current_elapsed
                print(f">> Progress: {progress:5.1f}% ({counter}/{len(parameter_combinations)}) - Time left: {current_remaining/60:.1f} min")
            
            # SKIP if there's not enough time left (recalculate remaining_time)
            current_elapsed = time.time() - start_time
            current_remaining_time = time_limit - current_elapsed
            
            if current_remaining_time < 30:  # Increased to 30 seconds
                print(f">> ⏰ Skipping remaining {len(parameter_combinations) - counter + 1} combinations (insufficient time: {current_remaining_time:.1f}s)")
                break
                
            # Create and train model
            try:
                # Recalculate time right before starting training
                current_train_time = time.time()
                train_elapsed = current_train_time - start_time
                if train_elapsed > time_limit:
                    print(f">> 🕒 TRAIN TIME LIMIT - Skipping combo {counter} after {train_elapsed/60:.1f} min")
                    continue
                
                # Smart epochs calculation: use more time when available, but stay fast
                dynamic_remaining_time = time_limit - train_elapsed
                # Base epochs: use more if we have plenty of time, but minimum 30
                remaining_combos = max(1, len(parameter_combinations) - counter + 1)
                
                if dynamic_remaining_time > 300:  # More than 5 minutes left
                    epochs = 100 + min(200, int(dynamic_remaining_time / remaining_combos))
                    time_category = "5+ min left"
                elif dynamic_remaining_time > 120:  # 2-5 minutes left
                    epochs = 50 + min(100, int(dynamic_remaining_time / remaining_combos / 2))
                    time_category = "2-5 min left"
                else:  # Less than 2 minutes
                    epochs = 30 + min(50, int(dynamic_remaining_time / remaining_combos / 3))
                    time_category = "<2 min left"
                
                epochs = min(epochs, 300)  # Cap at reasonable maximum
                
                # Log epochs calculation for debugging (every 5 iterations)
                if counter % 5 == 0:
                    print(f">> 🔧 Time: {dynamic_remaining_time:.0f}s left → {time_category} → {epochs} epochs")
                
                if s.base_estimator_type == 'linear':
                    base_estimator = LinearSVM(
                        C=C_val, learning_rate=lr_val, epochs=epochs
                    )
                else:
                    base_estimator = KernelSVM(
                        kernel=kernel_val, C=C_val, gamma=gamma_val,
                        learning_rate=lr_val, epochs=epochs
                    )
                
                model = OneVsRestSVM(base_estimator=base_estimator)
                
                # Quick training on small subset with timing
                train_start_time = time.time()
                model.fit(X_train_half, y_train_half)
                train_end_time = time.time()
                y_pred = model.predict(X_val)
                
                # Calculate accuracy
                accuracy = np.mean(y_val == y_pred)
                successful_combinations += 1
                
                # Debug timing (log every 5 iterations)
                if successful_combinations % 5 == 0:
                    print(f">> Training time: {train_end_time - train_start_time:.2f}s")
                
                if accuracy > best_score:
                    best_score = accuracy
                    best_params = {
                        'C': C_val,
                        'learning_rate': lr_val,
                        'kernel': kernel_val,
                        'gamma': gamma_val,
                        'score': accuracy
                    }
                    
                    # Show top 5 best parameters
                    print(f">> ✅ New best: C={C_val}, lr={lr_val}, kernel={kernel_val}, gamma={gamma_val} → acc={accuracy:.4f}")
                    
            except Exception as e:
                # Skip this parameter combination if it fails
                continue
                
        # Check if we found any valid parameters
        total_elapsed = time.time() - start_time
        
        if successful_combinations == 0:
            print(f"\n⚠️  FAST OPTIMIZATION: NO VALID PARAMETERS")
            print(f"═" * 50)
            print(f"Tested: {total_combinations} combinations")
            print(f"Success rate: {successful_combinations}/{total_combinations}")
            print(f"Time used: {total_elapsed/60:.1f} min (limit: {max_time_minutes} min)")
            print(f"═" * 50)
            print(f"Dataset: {len(s.X_train)} samples")
            print(f"Subset used: {subset_size} samples")
            print(f"Epochs per test: ~{epochs} (adaptive)")
            print(f"═" * 50)
            return None
        
        if best_params:
            s.C = best_params['C']
            s.learning_rate = best_params['learning_rate']
            
            if hasattr(s, 'kernel') and 'kernel' in best_params:
                s.kernel = best_params['kernel']
                s.gamma = best_params.get('gamma', 0.1)
            
            total_elapsed = time.time() - start_time
            
            # Fix total time display - show at least 0.1 min even if very fast
            display_minutes = max(0.1, total_elapsed / 60)
            
            print(f"\n✓ FAST OPTIMIZATION SUCCESSFUL! ({display_minutes:.1f} min / limit: {max_time_minutes} min)")
            print(f"═" * 50)
            print(f"Final parameters (accuracy={best_score:.4f}):")
            for key, value in best_params.items():
                if key != 'score':
                    print(f"  • {key}: {value}")
            print(f"═" * 50)
            print(f"Performance: {successful_combinations} / {total_combinations} successful ({(successful_combinations/total_combinations*100):.1f}%)")
            print(f"Time efficiency: {display_minutes:.1f} min actual vs {max_time_minutes} min limit ({display_minutes/max_time_minutes*100:.1f}% of limit used)")
            print(f"═" * 50)
        else:
            print("⚠️  No valid parameters found, using defaults")
            
        return best_params
        
    except Exception as e:
        print(f"\n⚠️  FAST OPTIMIZATION FAILED")
        print(f"═" * 50)
        print(f"Error: {str(e)[:100]}{'...' if len(str(e)) > 100 else ''}")
        print(f"═" * 50)
        print("Falling back to individual parameter tuning...")
        return None


def ask_fast_optimization_multiclass() -> bool:
    """Ask user if they want to use fast optimization for multiclass."""
    from myclt.common.input_validation import ask_yes_no
    
    print("\n" + "=" * 60)
    print("FAST OPTIMIZATION OPTION")
    print("=" * 60)
    print("""
For multiclass SVM with kernel parameters, you can choose:

🔹 Fast grid search (recommended for large datasets):
   - Tests all parameter combinations in parallel
   - Uses small data subset for speed
   - Shows real-time progress updates
   - Total time: ~5-15 minutes

🔹 Individual parameter tuning:
   - Tunes parameters one by one with cross-validation
   - More precise but much slower (2-3x slower)
   - Can take 20-30+ minutes for full optimization
""")
    
    return ask_yes_no("Use fast grid search optimization?", default=True)


def train_model_interactive(s: MultinomialAppState) -> None:
    """Train multiclass SVM model with automatic parameter tuning."""
    if s.X_train is None or s.y_train is None:
        print("✗ No training data!")
        pause()
        return

    print("\n" + "=" * 70)
    print("TRAINING MULTICLASS SVM (One-vs-Rest)")
    print("=" * 70)

    try:
        # Debug info for development
        if False:  # Set to True for debugging
            print(f"DEBUG: learning_rate_auto={s.learning_rate_auto}")
            print(f"DEBUG: C_auto={s.C_auto}")
            print(f"DEBUG: kernel_auto={getattr(s, 'kernel_auto', False)}")
        
        # Check if all parameters are set to AUTO and dataset is suitable for optimization
        all_params_auto = (s.learning_rate_auto and s.C_auto and 
                          (s.base_estimator_type == 'kernel' and s.kernel_auto))
        
        any_param_auto = (s.learning_rate_auto or s.C_auto or 
                         (s.base_estimator_type == 'kernel' and s.kernel_auto))
        
        # Offer fast optimization for kernel models with all parameters in AUTO mode
        fast_optimization = False
        if s.base_estimator_type == 'kernel' and all_params_auto and len(s.y_train) > 500:
            fast_optimization = ask_fast_optimization_multiclass()
            
            if fast_optimization:
                # Use fast optimization for all parameters
                from myclt.common.input_validation import ask_int
                max_minutes = ask_int("Maximum optimization time (minutes)", 
                                    min_val=5, max_val=60, default=10)
                
                optimized_params = fast_auto_optimize_multiclass(s, max_minutes)
                
                if optimized_params:
                    # Skip individual tuning since fast optimization already did everything
                    s.learning_rate_auto = False
                    s.C_auto = False
                    s.kernel_auto = False
                    print("✓ Using parameters from fast optimization")
                else:
                    print("\n⚠️  FAST OPTIMIZATION: FALLING BACK TO INDIVIDUAL TUNING")
                    print("═" * 50)
                    fast_optimization = False
        
        if any_param_auto and not fast_optimization:
            print("\n📊 Starting multiclass SVM auto-optimization...")
            print("Note: Using individual parameter tuning for proper multiclass support")
            print("=" * 60)
        
        # Auto-tune learning rate if needed
        if s.learning_rate_auto:
            # Use a safe default C value when C is None (auto)
            current_C = s.C if s.C is not None else 1.0
            lr_result = multiclass_auto_tune_learning_rate(
                X=s.X_train, y=s.y_train,
                base_estimator_type=s.base_estimator_type,
                C=current_C, kernel=s.kernel, gamma=s.gamma,
                degree=s.degree, coef0=s.coef0,
                max_epochs=2000, k_folds=3, seed=s.seed,
                verbose=True
            )
            s.learning_rate = lr_result['best_lr']
            print(f"\n=> Learning rate set to: {s.learning_rate}")

        # Auto-tune C if needed
        if s.C_auto:
            # Use a safe default learning rate when learning_rate is None (auto)
            current_lr = s.learning_rate if s.learning_rate is not None else 0.001
            C_result = multiclass_auto_tune_C(
                X=s.X_train, y=s.y_train,
                base_estimator_type=s.base_estimator_type,
                learning_rate=current_lr, kernel=s.kernel,
                gamma=s.gamma, degree=s.degree, coef0=s.coef0,
                max_epochs=2000, k_folds=3, seed=s.seed,
                verbose=True
            )
            s.C = C_result['best_C']
            print(f"\n=> C set to: {s.C}")

        # Auto-tune kernel if needed
        if s.base_estimator_type == 'kernel' and s.kernel_auto and not fast_optimization:
            print("\n⚙️  Starting kernel parameter search...")
            
            # Simple grid search for kernel parameters using current C and learning_rate
            try:
                from .core import OneVsRestSVM, KernelSVM
                from myclt.ML.base_models import train_test_split, standardize_fit
                
                # Prepare data subset for faster search
                subset_size = min(400, len(s.X_train))
                if len(s.X_train) > subset_size:
                    idx = np.random.choice(len(s.X_train), subset_size, replace=False)
                    X_kernel = s.X_train[idx]
                    y_kernel = s.y_train[idx]
                else:
                    X_kernel, y_kernel = s.X_train, s.y_train
                
                X_kernel_std, _, _ = standardize_fit(X_kernel)
                X_train_half, X_val, y_train_half, y_val, _, _ = train_test_split(
                    X_kernel_std, y_kernel, test_size=0.3, seed=42
                )
                
                kernels = ['linear', 'rbf', 'poly']
                gammas = [0.01, 0.1, 1.0]
                
                best_accuracy = 0.0
                best_kernel = s.kernel
                best_gamma = s.gamma
                successful_searches = 0
                
                print(f">> Testing {len(kernels) * len(gammas)} kernel/gamma combinations...")
                
                kernel_search_start_time = time.time()
                
                for kernel_val in kernels:
                    kernel_timeout_triggered = False
                    for gamma_val in gammas:
                        # Check if we have time for this search
                        if time.time() - kernel_search_start_time > 5 * 60:  # 5 min max for kernel search
                            print(f">> ⏰ Time limit reached for kernel search - skipping remaining combinations")
                            kernel_timeout_triggered = True
                            break
                            
                        try:
                            # Use shorter epochs for kernel search (faster)
                            base_estimator = KernelSVM(
                                kernel=kernel_val, C=s.C, gamma=gamma_val,
                                learning_rate=s.learning_rate, epochs=50
                            )
                            
                            model = OneVsRestSVM(base_estimator=base_estimator)
                            model.fit(X_train_half, y_train_half)
                            y_pred = model.predict(X_val)
                            
                            accuracy = np.mean(y_val == y_pred)
                            successful_searches += 1
                            
                            # Show results for all attempts
                            status_icon = "✅" if accuracy > best_accuracy else "  "
                            if accuracy < 0.1:
                                print(f">> {status_icon} kernel={kernel_val}, gamma={gamma_val} → acc={accuracy:.4f} (very low)")
                            else:
                                print(f">> {status_icon} kernel={kernel_val}, gamma={gamma_val} → acc={accuracy:.4f}")
                            
                            if accuracy > best_accuracy:
                                best_accuracy = accuracy
                                best_kernel = kernel_val
                                best_gamma = gamma_val
                                print(f">> 🏆 NEW BEST: kernel={best_kernel}, gamma={best_gamma} → acc={best_accuracy:.4f}")
                                
                        except Exception as e:
                            # Detailed error reporting for debugging
                            import traceback
                            error_summary = f">> ❌ kernel={kernel_val}, gamma={gamma_val} → failed: {type(e).__name__}: {str(e)[:200]}"
                            print(error_summary)
                            # Print full traceback for the first error to avoid noise
                            continue
                    
                    # Break outer loop if timeout was triggered
                    if kernel_timeout_triggered:
                        break
                
                s.kernel = best_kernel
                s.gamma = best_gamma
                
                if successful_searches == 0:
                    print(f"✓ Using default kernel parameters: {s.kernel} (γ={s.gamma})")
                    print(f">> Note: No successful kernel searches, keeping defaults")
                else:
                    print(f"✓ Kernel set to: {s.kernel} (γ={s.gamma})")
                    print(f">> Best accuracy: {best_accuracy:.4f} ({successful_searches}/9 searches successful)")
                    
            except Exception as e:
                # Provide helpful error message
                print(f"⚠️  Kernel search encountered an issue")
                print(f">> Using default parameters: kernel={s.kernel}, gamma={s.gamma}")
                print(f">> Error: {str(e)[:100]}..." if len(str(e)) > 100 else f">> Error: {e}")
                # Keep current/default values

        # Create and train the final model
        if s.base_estimator_type == 'linear':
            from .core import LinearSVM
            base_estimator = LinearSVM(
                C=s.C, learning_rate=s.learning_rate, epochs=s.epochs
            )
        else:
            from .core import KernelSVM
            base_estimator = KernelSVM(
                kernel=s.kernel, C=s.C,
                gamma=s.gamma, degree=s.degree, coef0=s.coef0,
                learning_rate=s.learning_rate, epochs=s.epochs
            )

        from .core import OneVsRestSVM
        s.model = OneVsRestSVM(base_estimator=base_estimator)
        s.model.fit(s.X_train, s.y_train)

        print(f"✓ Training complete!")
        print(f"  Number of classifiers: {s.model.n_classes}")
        total_sv = 0
        for k, est in enumerate(s.model.estimators):
            n_sv = getattr(est, 'n_support_vectors', 0)
            print(f"  Class {k}: {n_sv} support vectors")
            total_sv += n_sv
        print(f"  Total support vectors: {total_sv}")
        
        # Show final parameters if auto-tuning was used
        if any([s.learning_rate_auto, s.C_auto, getattr(s, 'kernel_auto', False)]):
            print("\nFinal parameters after auto-tuning:")
            print(f"  C: {s.C}")
            print(f"  Learning rate: {s.learning_rate}")
            if s.base_estimator_type == 'kernel':
                print(f"  Kernel: {s.kernel}")
                print(f"  Gamma: {s.gamma}")
        
        pause()
    except Exception as e:
        print(f"✗ Training error: {e}")
        import traceback
        traceback.print_exc()
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
        
        # Convert y_test and y_pred to numeric indices if they contain string labels
        y_test_num = s.y_test
        y_pred_num = y_pred
        
        # Check if labels are strings
        unique_labels = np.unique(np.concatenate([s.y_test, y_pred]))
        has_string_labels = any(isinstance(label, str) for label in unique_labels)
        
        # Initialize label_to_idx for all cases
        label_to_idx = {}
        
        if has_string_labels:
            # Create mapping from class names to numeric indices
            label_to_idx = {label: idx for idx, label in enumerate(sorted(unique_labels))}
            
            # Convert both y_test and y_pred to numeric indices
            y_test_num = np.array([label_to_idx[label] for label in s.y_test], dtype=int)
            y_pred_num = np.array([label_to_idx[label] for label in y_pred], dtype=int)
            
            print(f"⚠️  Converted string labels to numeric indices for metric calculation:")
            for label, idx in label_to_idx.items():
                print(f"   '{label}' → {idx}")
        else:
            # Already numeric, use as is
            y_test_num = s.y_test
            y_pred_num = y_pred
            
        s.metrics = {}

        # Multiclass metrics - use numeric versions
        s.metrics['accuracy'] = accuracy(y_test_num, y_pred_num)
        s.metrics['macro_f1'] = multiclass_f1_score(y_test_num, y_pred_num, average='macro')
        s.metrics['micro_f1'] = multiclass_f1_score(y_test_num, y_pred_num, average='micro')

        # Print report - use numeric indices for calculation, but display original names
        if s.class_names is not None:
            # If we had to convert, we need to convert class names to match numeric indices
            if has_string_labels:
                # Reorder class names to match the numeric index mapping
                numeric_class_names = [class_name for class_name, _ in sorted(label_to_idx.items())]
                print_multiclass_classification_report(y_test_num, y_pred_num, numeric_class_names)
            else:
                print_multiclass_classification_report(s.y_test, y_pred, s.class_names)
        else:
            print_multiclass_classification_report(y_test_num, y_pred_num)

        # Confusion matrix - use numeric versions for calculation
        from .metrics import multiclass_confusion_matrix
        n_classes = s.model.n_classes
        cm = multiclass_confusion_matrix(y_test_num, y_pred_num, n_classes)
        print(f"\nConfusion Matrix:\n{cm}")

        if ask_yes_no("Show confusion matrix plot?", default=True):
            from .visualization import plot_confusion_matrix
            names = s.class_names if s.class_names else None
            plot_confusion_matrix(cm, names)

        # Decision boundary if 2D
        if s.X_test.shape[1] == 2:
            if ask_yes_no("Show decision boundary plot?", default=True):
                plot_svm_decision_boundary_2d(
                    s.model, s.X_test, y_test_num if has_string_labels else s.y_test,
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
                    model_type="multiclass",
                    class_names=s.class_names,
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
