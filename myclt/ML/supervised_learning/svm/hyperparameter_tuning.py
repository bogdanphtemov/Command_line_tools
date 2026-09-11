"""
Hyperparameter tuning utilities for SVM models.

Provides:
    - Grid search with cross-validation
    - Random search with cross-validation

Supported parameter ranges (typical):
    C:        [0.001, 0.01, 0.1, 1.0, 10, 100, 1000]
    gamma:    [0.0001, 0.001, 0.01, 0.1, 1.0, 10]
    degree:   [2, 3, 4, 5]
    epsilon:  [0.01, 0.05, 0.1, 0.2, 0.5]
    kernels:  ['linear', 'rbf', 'poly', 'sigmoid']

Example:
    >>> from svm.hyperparameter_tuning import grid_search_cv
    >>> param_grid = {'C': [0.1, 1.0, 10], 'gamma': [0.01, 0.1, 1.0]}
    >>> best_params, best_score = grid_search_cv(X, y, model_class=LinearSVM, param_grid=param_grid)
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Type
from itertools import product

from .core import LinearSVM, KernelSVM, LinearSVR, KernelSVR
from .preprocessing import standardize_fit, standardize_apply
from .metrics import accuracy, multiclass_f1_score, mean_squared_error


def _create_model(model_class: Type, params: Dict[str, Any]):
    """
    Create a model instance with given parameters.

    Filters out irrelevant parameters for the model class.

    Args:
        model_class: LinearSVM, KernelSVM, LinearSVR, or KernelSVR
        params: Hyperparameter dictionary

    Returns:
        Model instance
    """
    # Common params
    valid_params = {}
    for key in ['C', 'learning_rate', 'epochs', 'batch_size']:
        if key in params:
            valid_params[key] = params[key]

    # Kernel params
    if model_class in (KernelSVM, KernelSVR):
        for key in ['kernel', 'gamma', 'degree', 'coef0']:
            if key in params:
                valid_params[key] = params[key]

    # SVR params
    if model_class in (LinearSVR, KernelSVR):
        if 'epsilon' in params:
            valid_params['epsilon'] = params['epsilon']

    return model_class(**valid_params)


def _evaluate_params_cv(X: np.ndarray, y: np.ndarray,
                        model_class: Type, params: Dict[str, Any],
                        cv_folds: int = 5,
                        task: str = 'classifier',
                        verbose: bool = False) -> float:
    """
    Evaluate a single parameter configuration using k-fold CV.

    Uses random permutation of indices for fold assignment, and applies
    stratified sampling for classification tasks to preserve class ratios.

    Args:
        X: Feature matrix
        y: Target vector
        model_class: Model class to evaluate
        params: Hyperparameters
        cv_folds: Number of CV folds
        task: 'classifier' or 'regressor'
        verbose: If True, print fold progress

    Returns:
        Mean score across folds (macro-F1 for classification, -MSE for regression)
    """
    fold_scores = []
    n_samples = len(X)

    if task == 'classifier':
        # Stratified k-fold: preserve class proportions in each fold
        classes = np.unique(y)
        class_indices = {cls: np.where(y == cls)[0] for cls in classes}

        fold_indices = [[] for _ in range(cv_folds)]
        for cls in classes:
            cls_idx = class_indices[cls].copy()
            np.random.shuffle(cls_idx)
            # Distribute class samples across folds
            splits = np.array_split(cls_idx, cv_folds)
            for fold in range(cv_folds):
                fold_indices[fold].extend(splits[fold].tolist())

        for fold in range(cv_folds):
            if verbose:
                print(f"    Fold {fold + 1}/{cv_folds}", end="\r")

            test_idx = np.array(fold_indices[fold], dtype=int)
            train_idx = np.setdiff1d(np.arange(n_samples), test_idx)

            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            # Scale
            X_train_scaled, mean, std = standardize_fit(X_train)
            X_test_scaled = standardize_apply(X_test, mean, std)

            # Train model
            model = _create_model(model_class, params)
            model.fit(X_train_scaled, y_train)

            # Evaluate with macro-averaged F1 (stable across all class distributions)
            y_pred = model.predict(X_test_scaled)
            score = multiclass_f1_score(y_test, y_pred, average='macro')
            fold_scores.append(score)
    else:
        # Random permutation for regression (no stratification needed)
        indices = np.random.permutation(n_samples)
        fold_size = n_samples // cv_folds

        for fold in range(cv_folds):
            if verbose:
                print(f"    Fold {fold + 1}/{cv_folds}", end="\r")

            test_start = fold * fold_size
            test_end = test_start + fold_size if fold < cv_folds - 1 else n_samples

            test_indices = indices[test_start:test_end]
            train_indices = np.concatenate([indices[:test_start], indices[test_end:]])

            X_train, y_train = X[train_indices], y[train_indices]
            X_test, y_test = X[test_indices], y[test_indices]

            # Scale
            X_train_scaled, mean, std = standardize_fit(X_train)
            X_test_scaled = standardize_apply(X_test, mean, std)

            # Train model
            model = _create_model(model_class, params)
            model.fit(X_train_scaled, y_train)

            # Evaluate
            y_pred = model.predict(X_test_scaled)
            score = -mean_squared_error(y_test, y_pred)
            fold_scores.append(score)

    if verbose:
        print()  # Clear the progress line

    return float(np.mean(fold_scores))


def _print_results(best_params: Dict[str, Any], best_score: float,
                   task: str, search_type: str = "SEARCH") -> None:
    """Print formatted search results."""
    metric = 'F1' if task == 'classifier' else '(neg) MSE'
    print(f"{'=' * 70}")
    print(f"Best params: {best_params}")
    print(f"Best {metric}: {best_score:.6f}")
    print(f"{'=' * 70}\n")


def grid_search_cv(X: np.ndarray, y: np.ndarray,
                   model_class: Type,
                   param_grid: Dict[str, List[Any]],
                   cv_folds: int = 5,
                   task: str = 'classifier',
                   verbose: bool = True) -> Tuple[Dict[str, Any], float]:
    """
    Grid search with k-fold cross-validation.

    Tests all parameter combinations and returns best configuration.

    Args:
        X: Feature matrix
        y: Target vector
        model_class: Model class (LinearSVM, KernelSVM, LinearSVR, KernelSVR)
        param_grid: Dictionary of parameter names -> list of values
                    Example: {'C': [0.1, 1.0, 10], 'gamma': [0.01, 0.1, 1.0]}
        cv_folds: Number of CV folds
        task: 'classifier' or 'regressor'
        verbose: Print progress

    Returns:
        Tuple of (best_params, best_score)

    Raises:
        ValueError: If param_grid is empty
    """
    if not param_grid:
        raise ValueError("param_grid cannot be empty")

    param_names = list(param_grid.keys())
    param_values = [param_grid[name] for name in param_names]

    # Calculate total combinations without materialising all of them
    total_combinations = 1
    for v in param_values:
        total_combinations *= len(v)

    best_score = -float('inf')
    best_params = None

    show_fold_progress = verbose and cv_folds > 5

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"GRID SEARCH: {total_combinations} combinations to test")
        print(f"{'=' * 70}")

    # Use generator instead of list(product(...)) to save memory
    for combo_idx, values in enumerate(product(*param_values), 1):
        params = dict(zip(param_names, values))

        avg_score = _evaluate_params_cv(
            X, y, model_class, params, cv_folds, task,
            verbose=show_fold_progress
        )

        if avg_score > best_score:
            best_score = avg_score
            best_params = params.copy()

        if verbose and combo_idx % max(1, total_combinations // 10 + 1) == 0:
            metric = 'F1' if task == 'classifier' else '(neg) MSE'
            print(f"  Progress: {combo_idx}/{total_combinations} | Best {metric}: {best_score:.6f}")

    if verbose:
        _print_results(best_params, best_score, task, "GRID SEARCH")

    return best_params, best_score


def random_search_cv(X: np.ndarray, y: np.ndarray,
                     model_class: Type,
                     param_distributions: Dict[str, List[Any]],
                     n_iter: int = 10,
                     cv_folds: int = 5,
                     task: str = 'classifier',
                     seed: int = 42,
                     verbose: bool = True) -> Tuple[Dict[str, Any], float]:
    """
    Random search with k-fold cross-validation.

    Randomly samples parameter combinations (faster than grid search).

    Args:
        X: Feature matrix
        y: Target vector
        model_class: Model class (LinearSVM, KernelSVM, LinearSVR, KernelSVR)
        param_distributions: Dictionary of parameter names -> list of values
        n_iter: Number of random combinations to test
        cv_folds: Number of CV folds
        task: 'classifier' or 'regressor'
        seed: Random seed for reproducibility
        verbose: Print progress

    Returns:
        Tuple of (best_params, best_score)

    Raises:
        ValueError: If param_distributions is empty
    """
    if not param_distributions:
        raise ValueError("param_distributions cannot be empty")

    rng = np.random.RandomState(seed)
    best_score = -float('inf')
    best_params = None

    show_fold_progress = verbose and cv_folds > 5

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"RANDOM SEARCH: {n_iter} random combinations to test")
        print(f"{'=' * 70}")

    for iteration in range(n_iter):
        params = {}
        for param_name, values in param_distributions.items():
            params[param_name] = rng.choice(values)

        avg_score = _evaluate_params_cv(
            X, y, model_class, params, cv_folds, task,
            verbose=show_fold_progress
        )

        if avg_score > best_score:
            best_score = avg_score
            best_params = params.copy()

        if verbose and (iteration + 1) % max(1, n_iter // 5 + 1) == 0:
            metric = 'F1' if task == 'classifier' else '(neg) MSE'
            print(f"  Progress: {iteration + 1}/{n_iter} | Best {metric}: {best_score:.6f}")

    if verbose:
        _print_results(best_params, best_score, task, "RANDOM SEARCH")

    return best_params, best_score
def auto_tune_learning_rate(
    X: np.ndarray,
    y: np.ndarray,
    model_class: Type,
    C: float = 1.0,
    max_epochs: int = 10000,
    k_folds: int = 3,
    seed: int = 42,
    use_scaling: bool = True,
    early_stopping_patience: int = 50,
    task: str = 'classifier',
    do_refine: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Auto-tune the learning rate for SVM models via K-fold CV.
    Uses validation loss (hinge for classifier, MSE for regressor).
    Two-phase approach: coarse log-scale search + refinement.
    """
    lr_grid = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]

    if verbose:
        print("Auto-tuning learning rate...")
        print("=" * 72)
        print(f"Grid: {lr_grid}")
        print(f"Using {k_folds}-fold CV, max_epochs={max_epochs}")
        if use_scaling:
            print("Scaling inside fold: ON")
        else:
            print("Data already scaled externally")
        print("=" * 72)

    from .preprocessing import k_fold_split, standardize_fit, standardize_apply

    folds = k_fold_split(X, y, k=k_folds, seed=seed)
    results = []

    for lr in lr_grid:
        fold_losses = []
        diverged = False

        for _, (X_train, X_val, y_train, y_val) in enumerate(folds):
            if use_scaling:
                X_train_s, mean, std = standardize_fit(X_train)
                X_val_s = standardize_apply(X_val, mean, std)
            else:
                X_train_s = X_train
                X_val_s = X_val

            model = model_class(C=C, learning_rate=lr, epochs=max_epochs)
            model.fit_with_early_stopping(
                X_train_s, y_train, X_val_s, y_val,
                patience=early_stopping_patience, min_delta=1e-6
            )

            if not model.is_trained or not np.all(np.isfinite(model.w if hasattr(model, 'w') else model.beta)):
                diverged = True
                break

            if task == 'classifier':
                from .metrics import accuracy
                y_pred = model.predict(X_val_s)
                # Convert labels to numeric
                unique_y = np.unique(y_val)
                if np.issubdtype(np.array(y_val).dtype, np.str_) or len(set(unique_y) - {0, 1, -1}) > 0:
                    y_val_cmp = np.where(y_val == unique_y[1], 1, 0).astype(int)
                else:
                    y_val_cmp = y_val.astype(int)
                fold_loss = 1.0 - accuracy(y_val_cmp, y_pred)  # Error Rate (lower is better, ~0.001 means 99.9% accuracy)
            else:
                from .metrics import mean_squared_error
                y_pred = model.predict(X_val_s)
                fold_loss = mean_squared_error(y_val, y_pred)

            fold_losses.append(fold_loss)

        if diverged:
            if verbose:
                print(f"  Testing {lr:.4g} ... DIVERGED")
            results.append({'lr': lr, 'mean_loss': None, 'diverged': True})
        else:
            mean_loss = float(np.mean(fold_losses))
            metric = 'Error Rate' if task == 'classifier' else 'MSE'
            if verbose:
                print(f"  Testing {lr:.4g} ... {metric}: {mean_loss:.4e}")
            results.append({'lr': lr, 'mean_loss': mean_loss, 'diverged': False})

    valid = [r for r in results if not r['diverged']]
    if not valid:
        raise RuntimeError("All learning rates diverged! Cannot auto-tune.")

    valid.sort(key=lambda x: x['mean_loss'])
    best_lr = valid[0]['lr']
    best_loss = valid[0]['mean_loss']

    # Phase 2: refine
    if do_refine and best_lr > 0:
        factor = 2.0
        refine = sorted(set(
            round(best_lr / factor ** (i / 2), 10) for i in range(-2, 3)
        ))
        refine = [lr for lr in refine if 1e-8 <= lr <= 1.0]

        if len(refine) > 1:
            if verbose:
                print("\nRefining search around best LR...")
                print(f"Phase 2 grid: {refine}")
                print("-" * 72)

            for lr in refine:
                fold_losses = []
                diverged = False
                for _, (X_tr, X_v, y_tr, y_v) in enumerate(folds):
                    if use_scaling:
                        X_tr_s, m, s = standardize_fit(X_tr)
                        X_v_s = standardize_apply(X_v, m, s)
                    else:
                        X_tr_s = X_tr
                        X_v_s = X_v

                    model = model_class(C=C, learning_rate=lr, epochs=max_epochs)
                    model.fit_with_early_stopping(
                        X_tr_s, y_tr, X_v_s, y_v,
                        patience=early_stopping_patience, min_delta=1e-6
                    )

                    if not model.is_trained or not np.all(np.isfinite(model.w if hasattr(model, 'w') else model.beta)):
                        diverged = True
                        break

                    if task == 'classifier':
                        from .metrics import accuracy
                        y_pred = model.predict(X_v_s)
                        unique_y2 = np.unique(y_v)
                        if np.issubdtype(np.array(y_v).dtype, np.str_) or len(set(unique_y2) - {0, 1, -1}) > 0:
                            y_v_cmp = np.where(y_v == unique_y2[1], 1, 0).astype(int)
                        else:
                            y_v_cmp = y_v.astype(int)
                        fold_loss = 1.0 - accuracy(y_v_cmp, y_pred)  # Error Rate (lower is better, ~0.001 means 99.9% accuracy)
                    else:
                        from .metrics import mean_squared_error
                        y_pred = model.predict(X_v_s)
                        fold_loss = mean_squared_error(y_v, y_pred)
                    fold_losses.append(fold_loss)

                if not diverged:
                    mean_loss = float(np.mean(fold_losses))
                    if mean_loss < best_loss:
                        best_lr = lr
                        best_loss = mean_loss
                    if verbose:
                        metric = 'Error Rate' if task == 'classifier' else 'MSE'
                        print(f"  Testing {lr:.4g} ... {metric}: {mean_loss:.4e}")

    if verbose:
        print("=" * 72)
        print(f"Best learning rate: {best_lr}")
        print(f"Best validation loss: {best_loss:.6e}")
        print("=" * 72)

    return {'best_lr': best_lr, 'best_loss': best_loss, 'results': results}
def auto_tune_C(
    X: np.ndarray,
    y: np.ndarray,
    model_class: Type,
    learning_rate: float,
    max_epochs: int = 10000,
    k_folds: int = 3,
    seed: int = 42,
    use_scaling: bool = True,
    early_stopping_patience: int = 50,
    task: str = 'classifier',
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Auto-tune the regularization parameter C for SVM models via K-fold CV.
    Uses validation loss (hinge for classifier, MSE for regressor).
    """
    C_grid = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

    if verbose:
        print("\nAuto-tuning regularization C...")
        print("=" * 72)
        print(f"Grid: {C_grid}")
        print(f"Using {k_folds}-fold CV, max_epochs={max_epochs}")
        print(f"Learning rate fixed at: {learning_rate}")
        if use_scaling:
            print("Scaling inside fold: ON")
        else:
            print("Data already scaled externally")
        print("=" * 72)

    from .preprocessing import k_fold_split, standardize_fit, standardize_apply

    folds = k_fold_split(X, y, k=k_folds, seed=seed)
    results = []

    for C_val in C_grid:
        fold_losses = []
        diverged = False

        for _, (X_train, X_val, y_train, y_val) in enumerate(folds):
            if use_scaling:
                X_train_s, mean, std = standardize_fit(X_train)
                X_val_s = standardize_apply(X_val, mean, std)
            else:
                X_train_s = X_train
                X_val_s = X_val

            model = model_class(C=C_val, learning_rate=learning_rate, epochs=max_epochs)
            model.fit_with_early_stopping(
                X_train_s, y_train, X_val_s, y_val,
                patience=early_stopping_patience, min_delta=1e-6
            )

            if not model.is_trained or not np.all(np.isfinite(model.w if hasattr(model, 'w') else model.beta)):
                diverged = True
                break

            if task == 'classifier':
                from .metrics import accuracy
                y_pred = model.predict(X_val_s)
                # Convert labels to numeric
                unique_y = np.unique(y_val)
                if np.issubdtype(np.array(y_val).dtype, np.str_) or len(set(unique_y) - {0, 1, -1}) > 0:
                    y_val_cmp = np.where(y_val == unique_y[1], 1, 0).astype(int)
                else:
                    y_val_cmp = y_val.astype(int)
                fold_loss = 1.0 - accuracy(y_val_cmp, y_pred)  # Error Rate (lower is better, ~0.001 means 99.9% accuracy)
            else:
                from .metrics import mean_squared_error
                y_pred = model.predict(X_val_s)
                fold_loss = mean_squared_error(y_val, y_pred)

            fold_losses.append(fold_loss)

        if diverged:
            if verbose:
                print(f"  Testing C={C_val:.4g} ... DIVERGED")
            results.append({'C': C_val, 'mean_loss': None, 'diverged': True})
        else:
            mean_loss = float(np.mean(fold_losses))
            metric = 'Error Rate' if task == 'classifier' else 'MSE'
            if verbose:
                print(f"  Testing C={C_val:.4g} ... {metric}: {mean_loss:.4e}")
            results.append({'C': C_val, 'mean_loss': mean_loss, 'diverged': False})

    valid = [r for r in results if not r['diverged']]
    if not valid:
        raise RuntimeError("All C values diverged! Cannot auto-tune.")

    valid.sort(key=lambda x: x['mean_loss'])
    best_C = valid[0]['C']
    best_loss = valid[0]['mean_loss']

    if verbose:
        print("=" * 72)
        print(f"Best C: {best_C}")
        print(f"Best validation loss: {best_loss:.6e}")
        print("=" * 72)

    return {'best_C': best_C, 'best_loss': best_loss, 'results': results}
