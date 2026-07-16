import numpy as np
from typing import Tuple, Dict, Any, List
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
