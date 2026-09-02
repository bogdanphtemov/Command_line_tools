import numpy as np
from typing import Tuple, Dict, Any, List, Optional
from .core import LinearRegressionGD
from .preprocessing import standardize_fit, standardize_apply
from .metrics import mse

def k_fold_split(X: np.ndarray, y: np.ndarray, k: int = 5, seed: int = 42) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    K-Fold Cross-Validation split generator.
    Returns k tuples of (X_train, X_val, y_train, y_val)
    """
    if k < 2:
        raise ValueError("k must be at least 2")
    
    n = X.shape[0]
    rng = np.random.default_rng(seed)  
    
    # Shuffle indices
    idx = np.arange(n)
    rng.shuffle(idx)
    
      
    fold_indices = np.array_split(idx, k)
    
    folds = []
    
    for fold_idx in range(k):
        val_idx = fold_indices[fold_idx]
        train_idx = np.concatenate([fold_indices[i] for i in range(k) if i != fold_idx])
        
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_val = X[val_idx]
        y_val = y[val_idx]
        
        folds.append((X_train, X_val, y_train, y_val))
    
    return folds

def grid_search_regularization(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.05,
    epochs: int = 2000,
    lambda_l1_grid: List[float] = None,
    lambda_l2_grid: List[float] = None,
    k_folds: int = 5,
    seed: int = 42,
    verbose: bool = False,
    use_scaling: bool = True,
    early_stopping: bool = True,
    early_stopping_patience: int = 50
) -> Dict[str, Any]:
    """
    Searches for the best regularization parameters (L1 and L2) over a grid.

    The algorithm iterates over all possible combinations of lambda_l1 and lambda_l2 values, 
    sing K-Fold cross-validation to assess the stability of the model.
    """
   
    if lambda_l1_grid is None:
        lambda_l1_grid = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    
    if lambda_l2_grid is None:
        lambda_l2_grid = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    
    # Generate k-fold splits
    folds = k_fold_split(X, y, k=k_folds, seed=seed)
    
    results = []
    best_mse = float('inf')
    best_params = {'lambda_l1': 0.0, 'lambda_l2': 0.0}
    
    total_combinations = len(lambda_l1_grid) * len(lambda_l2_grid)
    current_combo = 0
    
    print(f"\n{'='*72}")
    print(f"Grid Search: {total_combinations} combinations × {k_folds}-fold CV")
    print(f"Scaling: {'ON' if use_scaling else 'OFF'}")
    print(f"Early Stopping: {'ON' if early_stopping else 'OFF'}")
    print(f"{'='*72}\n")
    
    for l1 in lambda_l1_grid:
        for l2 in lambda_l2_grid:
            current_combo += 1
            fold_mses = []
            
            
            for fold_idx, (X_train, X_val, y_train, y_val) in enumerate(folds):
                
                if use_scaling:
                    X_train_scaled, scaler_mean, scaler_std = standardize_fit(X_train)
                    X_val_scaled = standardize_apply(X_val, scaler_mean, scaler_std)
                else:
                    X_train_scaled = X_train
                    X_val_scaled = X_val
                
                # Create and train model
                model = LinearRegressionGD(
                    learning_rate=learning_rate,
                    epochs=epochs,
                    lambda_l1=l1,
                    lambda_l2=l2
                )
                
                
                if early_stopping:
                    model.fit_with_early_stopping(
                        X_train_scaled, 
                        y_train,
                        X_val_scaled,
                        y_val,
                        patience=early_stopping_patience
                    )
                else:
                    model.fit(X_train_scaled, y_train)
                
                # Evaluate on validation fold
                y_pred = model.predict(X_val_scaled)
                fold_mse = mse(y_val, y_pred)
                fold_mses.append(fold_mse)
            
            # Calculate mean and std across folds
            mean_mse = float(np.mean(fold_mses))
            std_mse = float(np.std(fold_mses))
            
            
            result_dict = {
                'l1': float(l1),
                'l2': float(l2),
                'mean_mse': mean_mse,
                'std_mse': std_mse
            }
            results.append(result_dict)
            
            # Update best params
            if mean_mse < best_mse:
                best_mse = mean_mse
                best_params = {'lambda_l1': l1, 'lambda_l2': l2}
            
            # Verbose output
            if verbose:
                status = "BEST" if mean_mse == best_mse else ""
                print(f"[{current_combo}/{total_combinations}] L1={l1:.4f} L2={l2:.4f} | "
                      f"MSE: {mean_mse:.6f} ± {std_mse:.6f} {status}")
            else:
                # Progress bar
                if current_combo % max(1, total_combinations // 20) == 0:
                    progress = (current_combo / total_combinations) * 100
                    print(f"Progress: {progress:.1f}% completed.")
    
    
    results_sorted = sorted(results, key=lambda x: x['mean_mse'])
    
    return {
        'best_lambda_l1': best_params['lambda_l1'],
        'best_lambda_l2': best_params['lambda_l2'],
        'best_mse': best_mse,
        'results': results_sorted
    }



def auto_tune_learning_rate(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate_grid: Optional[List[float]] = None,
    epochs: int = 10000,
    k_folds: int = 3,
    seed: int = 42,
    use_scaling: bool = True,
    early_stopping_patience: int = 50,
    do_refine: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Auto-tune the learning rate via grid search with K-fold CV.
    
    Two-phase approach:
      Phase 1: Coarse log-scale search
      Phase 2: Fine-grain search around the best Phase 1 value
    
    Detects divergence (NaN/inf) and excludes those candidates.
    """
    if learning_rate_grid is None:
        learning_rate_grid = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
    
    if verbose:
        print("Auto-tuning learning rate...")
        print("=" * 72)
        print(f"Grid: {learning_rate_grid}")
        print(f"Using {k_folds}-fold CV, epochs={epochs}")
        if use_scaling:
            print("Scaling inside fold: ON")
        else:
            print("Data already scaled externally")
        print("=" * 72)
    
    # Phase 1
    phase1_results = _evaluate_lr_grid(
        X, y, learning_rate_grid, epochs, k_folds, seed, use_scaling,
        early_stopping_patience, verbose
    )
    
    valid_results = [r for r in phase1_results if r['mean_mse'] is not None]
    if not valid_results:
        raise RuntimeError("All learning rates diverged! Cannot auto-tune.")
    
    valid_results.sort(key=lambda x: x['mean_mse'])
    best_lr = valid_results[0]['lr']
    best_mse = valid_results[0]['mean_mse']
    
    phase2_results = None
    
    # Phase 2: refine
    if do_refine and best_lr > 0:
        factor = 2.0
        refine_grid = [
            best_lr / factor,
            best_lr / (factor ** 0.5),
            best_lr,
            best_lr * (factor ** 0.5),
            best_lr * factor,
        ]
        refine_grid = sorted(set(round(lr, 10) for lr in refine_grid))
        refine_grid = [lr for lr in refine_grid if 1e-8 <= lr <= 1.0]
        
        if len(refine_grid) > 1:
            if verbose:
                print("\nRefining search around best LR...")
                print(f"Phase 2 grid: {refine_grid}")
                print("-" * 72)
            
            phase2_results = _evaluate_lr_grid(
                X, y, refine_grid, epochs, k_folds, seed, use_scaling,
                early_stopping_patience, verbose
            )
            
            valid_p2 = [r for r in phase2_results if r['mean_mse'] is not None]
            if valid_p2:
                valid_p2.sort(key=lambda x: x['mean_mse'])
                if valid_p2[0]['mean_mse'] < best_mse:
                    best_lr = valid_p2[0]['lr']
                    best_mse = valid_p2[0]['mean_mse']
    
    if verbose:
        print("=" * 72)
        print(f"Best learning rate: {best_lr}")
        print(f"Best validation MSE: {best_mse:.6e}")
        print("=" * 72)
    
    return {
        'best_lr': best_lr,
        'best_mse': best_mse,
        'results': phase1_results,
        'phase2_results': phase2_results,
    }


def _evaluate_lr_grid(
    X: np.ndarray, y: np.ndarray, lr_grid: List[float],
    epochs: int, k_folds: int, seed: int, use_scaling: bool,
    early_stopping_patience: int, verbose: bool
) -> List[Dict[str, Any]]:
    """
    Evaluate a list of learning rates using K-fold CV.
    Returns list of dicts with 'lr', 'mean_mse', 'std_mse', 'n_folds', 'diverged'.
    """
    from .metrics import mse
    from .core import LinearRegressionGD
    from .preprocessing import standardize_fit, standardize_apply
    
    folds = k_fold_split(X, y, k=k_folds, seed=seed)
    results = []
    
    for lr in lr_grid:
        fold_mses = []
        diverged = False
        
        for _, (X_train, X_val, y_train, y_val) in enumerate(folds):
            if use_scaling:
                X_train_scaled, scaler_mean, scaler_std = standardize_fit(X_train)
                X_val_scaled = standardize_apply(X_val, scaler_mean, scaler_std)
            else:
                X_train_scaled = X_train
                X_val_scaled = X_val
            
            model = LinearRegressionGD(
                learning_rate=lr, epochs=epochs,
                lambda_l1=0.0, lambda_l2=0.0
            )
            model.fit_with_early_stopping(
                X_train_scaled, y_train,
                X_val_scaled, y_val,
                patience=early_stopping_patience
            )
            
            # Check divergence
            if model.w is None or not np.all(np.isfinite(model.w)):
                diverged = True
                break
            
            y_pred = model.predict(X_val_scaled)
            fold_mses.append(mse(y_val, y_pred))
        
        if diverged:
            if verbose:
                print(f"  Testing {lr:.4f} ... DIVERGED")
            results.append({
                'lr': lr, 'mean_mse': None, 'std_mse': None,
                'n_folds': 0, 'diverged': True,
            })
        else:
            mean_mse_val = float(np.mean(fold_mses))
            std_mse_val = float(np.std(fold_mses))
            if verbose:
                print(f"  Testing {lr:.4f} ... MSE: {mean_mse_val:.4e}")
            results.append({
                'lr': lr, 'mean_mse': mean_mse_val, 'std_mse': std_mse_val,
                'n_folds': len(fold_mses), 'diverged': False,
            })
    
    return results
