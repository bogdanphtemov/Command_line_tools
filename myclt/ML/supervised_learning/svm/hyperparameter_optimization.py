"""
Hyperparameter optimization for SVM models.

Provides automatic parameter search with time constraints and performance metrics.
Designed to achieve R┬▓ ΓëÑ 0.80 within 10 minutes for regression tasks.
"""

import time
import numpy as np
from typing import Dict, Tuple, Optional, Any, List
from myclt.ML.base_models import train_test_split, standardize_fit, standardize_apply
from myclt.ML.base.base_data import Dataset, Prepareddata
from .core import LinearSVM, KernelSVM, LinearSVR, KernelSVR
from .metrics import r2_score, accuracy


class SVMHyperparameterOptimizer:
    """Optimizes SVM hyperparameters with time constraints."""
    
    def __init__(self, max_time_minutes: int = 10):
        self.max_time_minutes = max_time_minutes
        self.best_params = {}
        self.best_score = -float('inf')
    
    def optimize_epsilon_param(self, X_train: np.ndarray, y_train: np.ndarray, 
                             X_val: np.ndarray, y_val: np.ndarray,
                             mode: str = 'classifier') -> float:
        """
        Optimize epsilon parameter for SVR models.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            mode: 'classifier' or 'regressor'
            
        Returns:
            Optimized epsilon value
        """
        if mode != 'regressor':
            return 0.1  # Default for classifiers
        
        epsilon_candidates = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 1.0]
        best_epsilon = 0.1
        best_r2 = -float('inf')
        
        start_time = time.time()
        
        for eps in epsilon_candidates:
            # Check time constraint
            if time.time() - start_time > self.max_time_minutes * 60 - 60:  # Leave 1 min for final training
                break
                
            try:
                model = LinearSVR(C=1.0, epsilon=eps, epochs=100, learning_rate=0.01)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                r2 = r2_score(y_val, y_pred)
                
                if r2 > best_r2:
                    best_r2 = r2
                    best_epsilon = eps
                    print(f"╬╡={eps}: R┬▓={r2:.4f} Γ£ô")
                else:
                    print(f"╬╡={eps}: R┬▓={r2:.4f}")
                    
            except Exception as e:
                print(f"╬╡={eps}: Error - {e}")
                continue
        
        print(f"\nBest ╬╡: {best_epsilon} (R┬▓={best_r2:.4f})")
        return best_epsilon
    
    def optimize_kernel_selection(self, X_train: np.ndarray, y_train: np.ndarray,
                                X_val: np.ndarray, y_val: np.ndarray,
                                mode: str = 'classifier') -> Dict[str, Any]:
        """
        Optimize kernel selection and parameters.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            mode: 'classifier' or 'regressor'
            
        Returns:
            Dictionary with optimal kernel configuration
        """
        kernels = ['rbf', 'linear', 'poly', 'sigmoid']
        gamma_values = [0.1, 0.5, 1.0, 2.0]
        
        best_config = {'kernel': 'rbf', 'gamma': 1.0, 'degree': 3, 'coef0': 0.0}
        best_score = -float('inf')
        
        start_time = time.time()
        
        for kernel in kernels:
            # Check time constraint
            if time.time() - start_time > self.max_time_minutes * 60 - 120:  # Leave 2 mins for other params
                break
                
            gamma_candidates = gamma_values if kernel in ['rbf', 'poly', 'sigmoid'] else [None]
            
            for gamma in gamma_candidates:
                try:
                    if mode == 'classifier':
                        model = KernelSVM(kernel=kernel, C=1.0, gamma=gamma or 1.0, epochs=100)
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_val)
                        score = accuracy(y_val, y_pred)  # Use project's accuracy function
                    else:
                        model = KernelSVR(kernel=kernel, C=1.0, gamma=gamma or 1.0, epsilon=0.1, epochs=100)
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_val)
                        score = r2_score(y_val, y_pred)  # Use project's r2_score function
                    
                    if score > best_score:
                        best_score = score
                        best_config = {
                            'kernel': kernel,
                            'gamma': gamma or 1.0,
                            'degree': 3,
                            'coef0': 0.0
                        }
                        print(f"Kernel {kernel}, gamma: {gamma}: Score={score:.4f} Γ£ô")
                    else:
                        print(f"Kernel {kernel}, gamma: {gamma}: Score={score:.4f}")
                        
                except Exception as e:
                    print(f"Kernel {kernel}, gamma: {gamma}: Error - {e}")
                    continue
        
        # Simple poly/sigmoid params if selected
        if best_config['kernel'] == 'poly':
            best_config['degree'] = 3
            best_config['coef0'] = 1.0
        elif best_config['kernel'] == 'sigmoid':
            best_config['coef0'] = 0.0
        
        print(f"\nBest kernel: {best_config['kernel']} (Score={best_score:.4f})")
        return best_config
    
    def optimize_all_parameters(self, prepared_data: Prepareddata, 
                              mode: str = 'classifier') -> Dict[str, Any]:
        """
        Run complete hyperparameter optimization.
        
        Args:
            prepared_data: Prepared dataset
            mode: 'classifier' or 'regressor'
            
        Returns:
            Dictionary with all optimized parameters
        """
        print("\n" + "=" * 60)
        print("STARTING AUTOMATIC PARAMETER OPTIMIZATION")
        print("=" * 60)
        print(f"Time limit: {self.max_time_minutes} minutes")
        print("Target: R┬▓ ΓëÑ 0.80 for regression\n")
        
        start_time = time.time()
        
        # Prepare data using project's train_test_split (returns 6 values, we need first 4)
        X_train, X_val, y_train, y_val, _, _ = train_test_split(
            prepared_data.X, prepared_data.y, 
            test_size=0.2, seed=42
        )
        
        # Note: For kernel and epsilon optimization we use raw data, 
        # and for C optimization we'll scale it
        mean, std, X_train_scaled, X_val_scaled = None, None, None, None
        
        # Optimize kernel first
        print("\n1. OPTIMIZING KERNEL SELECTION")
        kernel_config = self.optimize_kernel_selection(X_train, y_train, X_val, y_val, mode)
        
        # Optimize epsilon for regression
        epsilon = 0.1
        if mode == 'regressor':
            print("\n2. OPTIMIZING EPSILON (╬╡)")
            epsilon = self.optimize_epsilon_param(X_train, y_train, X_val, y_val, mode)
        
        # Before C optimization, standardize the data
        X_train_scaled, mean, std = standardize_fit(X_train)
        X_val_scaled = standardize_apply(X_val, mean, std)
        
        # Optimize C parameter
        print("\n3. OPTIMIZING REGULARIZATION (C)")
        C_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        best_C = 1.0
        best_final_score = -float('inf')
        
        for C in C_values:
            if time.time() - start_time > self.max_time_minutes * 60 - 30:  # Leave 30s
                break
                
            try:
                if mode == 'classifier':
                    if kernel_config['kernel'] == 'linear':
                        model = LinearSVM(C=C, epochs=200, learning_rate=0.01)
                    else:
                        model = KernelSVM(
                            kernel=kernel_config['kernel'], 
                            C=C, 
                            gamma=kernel_config['gamma'],
                            degree=kernel_config['degree'],
                            coef0=kernel_config['coef0'],
                            epochs=200
                        )
                else:
                    if kernel_config['kernel'] == 'linear':
                        model = LinearSVR(C=C, epsilon=epsilon, epochs=200, learning_rate=0.01)
                    else:
                        model = KernelSVR(
                            kernel=kernel_config['kernel'],
                            C=C,
                            epsilon=epsilon,
                            gamma=kernel_config['gamma'],
                            degree=kernel_config['degree'],
                            coef0=kernel_config['coef0'],
                            epochs=200
                        )
                
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_val_scaled)
                
                if mode == 'classifier':
                    score = accuracy(y_val, y_pred)
                else:
                    score = r2_score(y_val, y_pred)
                
                if score > best_final_score:
                    best_final_score = score
                    best_C = C
                    print(f"C={C}: Score={score:.4f} Γ£ô")
                else:
                    print(f"C={C}: Score={score:.4f}")
                    
            except Exception as e:
                print(f"C={C}: Error - {e}")
                continue
        
        # Compile final results
        final_params = {
            'kernel': kernel_config['kernel'],
            'C': best_C,
            'gamma': kernel_config['gamma'],
            'degree': kernel_config['degree'],
            'coef0': kernel_config['coef0'],
            'epsilon': epsilon if mode == 'regressor' else 0.1,
            'score': best_final_score
        }
        
        elapsed_time = (time.time() - start_time) / 60
        print(f"\n" + "=" * 60)
        print("OPTIMIZATION COMPLETE")
        print("=" * 60)
        print(f"Time elapsed: {elapsed_time:.1f} minutes")
        print(f"Best parameters: {final_params}")
        print(f"Best score: {best_final_score:.4f}")
        
        if mode == 'regressor' and best_final_score >= 0.80:
            print("≡ƒÄ» TARGET ACHIEVED: R┬▓ ΓëÑ 0.80")
        elif mode == 'regressor':
            print("ΓÜá∩╕Å  TARGET NOT MET: Consider increasing time limit or reducing data size")
        
        return final_params


class FastSVMHyperparameterOptimizer:
    """Fast hyperparameter optimization with adaptive sampling and early stopping."""
    
    def __init__(self, max_time_minutes: int = 15, target_r2: float = 0.80, target_accuracy: float = 0.85):
        self.max_time_minutes = max_time_minutes
        self.target_r2 = target_r2
        self.target_accuracy = target_accuracy
        self.best_params = {}
        self.best_score = -float('inf')
    
    def _preprocess_target(self, y: np.ndarray) -> np.ndarray:
        """Preprocess target variable for crypto data."""
        # Detect crypto data for specialized processing
        is_crypto = self._detect_crypto_data(y)
        
        if is_crypto:
            # Enhanced crypto preprocessing
            # 1. Log transformation to handle exponential growth
            y_log = np.log(y + 1)
            
            # 2. Robust scaling using median and IQR approximation
            median = np.median(y_log)
            q75, q25 = np.percentile(y_log, [75, 25])
            iqr = q75 - q25
            y_robust = (y_log - median) / max(iqr, 1e-10)  # Avoid division by zero
            
            # 3. Final standardization
            y_std = (y_robust - np.mean(y_robust)) / np.std(y_robust)
            
            print(f">> Crypto data detected: applying enhanced preprocessing")
            print(f"   Original range: {y.min():.2f} - {y.max():.2f}")
            print(f"   Processed range: {y_std.min():.2f} - {y_std.max():.2f}")
            
            return y_std
        
        # For regular data, use standard preprocessing
        if np.std(y) > 0:
            y_std = (y - np.mean(y)) / np.std(y)
            return y_std
        
        return y
    
    def _detect_crypto_data(self, y: np.ndarray) -> bool:
        """Detect if data has characteristics of crypto prices."""
        # Crypto data typically has:
        # 1. Large price range (min to max ratio > 1000)
        # 2. High volatility (std/mean ratio > 0.5)
        # 3. Positive values only
        if len(y) == 0:
            return False
            
        price_range_ratio = y.max() / max(1, y.min())
        volatility_ratio = np.std(y) / max(1, np.mean(y))
        all_positive = np.all(y >= 0)
        
        # If it looks like crypto data
        crypto_like = (price_range_ratio > 1000 and 
                      volatility_ratio > 0.3 and 
                      all_positive)
        
        return crypto_like
    
    def adaptive_optimize(self, X: np.ndarray, y: np.ndarray, mode: str = 'regressor') -> Dict[str, Any]:
        """
        Adaptive optimization with full 3-stage approach.
        
        Stage 1: Quick screening - coarse grid on small sample
        Stage 2: Medium optimization - refined grid on medium sample  
        Stage 3: Final refinement - full dataset with best parameters
        """
        n_samples = len(X)
        start_time = time.time()
        
        print(f"\n>> Starting 3-STAGE OPTIMIZATION (max {self.max_time_minutes} min)")
        print(f"Dataset size: {n_samples} samples, {X.shape[1]} features")
        
        best_params = None
        
        # Stage 1: Quick Screening (1-2 minutes) - ALWAYS RUN
        print(f"\n>> STAGE 1: Quick Screening (~2 min)")
        stage1_sample_size = min(500, n_samples)
        
        if stage1_sample_size < n_samples:
            idx = np.random.choice(n_samples, stage1_sample_size, replace=False)
            X_stage1 = X[idx]
            y_stage1 = self._preprocess_target(y[idx])
        else:
            X_stage1, y_stage1 = X, self._preprocess_target(y)
        
        stage1_params = self._improved_quick_screening(X_stage1, y_stage1, mode, start_time)
        
        # Check Stage 1 results
        target_threshold = self.target_r2 if mode == 'regressor' else self.target_accuracy
        if stage1_params.get('score', -float('inf')) >= target_threshold:
            metric = 'R┬▓' if mode == 'regressor' else 'Accuracy'
            print(f">> TARGET ACHIEVED in Stage 1! {metric} = {stage1_params['score']:.4f}")
            return stage1_params
        
        best_params = stage1_params
        
        # Stage 2: Medium Optimization (3-5 minutes) - if dataset is large enough
        if n_samples > 300 and (time.time() - start_time) < (self.max_time_minutes * 60 - 300):
            print(f"\n>> STAGE 2: Medium Optimization (~5 min)")
            stage2_sample_size = min(800, n_samples)  # Increased from 1200 to use more data
            
            if stage2_sample_size > stage1_sample_size:
                idx = np.random.choice(n_samples, stage2_sample_size, replace=False)
                X_stage2 = X[idx]
                y_stage2 = self._preprocess_target(y[idx])
            else:
                X_stage2, y_stage2 = X_stage1, y_stage1
            
            stage2_params = self._improved_medium_optimization(X_stage2, y_stage2, mode, 
                                                              start_time, best_params)
            
            # Update best parameters
            if stage2_params.get('score', -float('inf')) > best_params.get('score', -float('inf')):
                best_params = stage2_params
            
            # Check Stage 2 target
            if best_params.get('score', -float('inf')) >= self.target_r2:
                print(f">> TARGET ACHIEVED in Stage 2! R┬▓ = {best_params['score']:.4f}")
                return best_params
        
        # Stage 3: Final Refinement (remaining time)
        remaining_time = max(60, self.max_time_minutes * 60 - (time.time() - start_time))
        print(f"\n>> STAGE 3: Final Refinement (~{remaining_time/60:.1f} min remaining)")
        
        final_params = self._improved_final_refinement(X, y, mode, start_time, best_params)
        
        # Final check
        if final_params.get('score', -float('inf')) >= self.target_r2:
            print(f">> TARGET ACHIEVED in Stage 3! R┬▓ = {final_params['score']:.4f}")
        else:
            print(f">> FINAL RESULT: R┬▓ = {final_params.get('score', -float('inf')):.4f}")
        
        return final_params

    def _improved_quick_screening(self, X: np.ndarray, y: np.ndarray, mode: str,
                                 start_time: float) -> Dict[str, Any]:
        """Improved Stage 1: Fast screening with R┬▓ scoring."""
        from .hyperparameter_tuning import grid_search_cv
        from .metrics import r2_score
        
        metric = 'Accuracy' if mode == 'classifier' else 'R┬▓'
        print(f">> Stage 1: Fast screening with {metric} scoring...")
        
        # Smart parameter grid for quick screening - adapted for classification/regression
        param_grid = {
            'C': [0.1, 0.5, 1.0, 5.0, 10.0],  # Wider range for both tasks
            'kernel': ['rbf', 'linear'],  # Start with simpler kernels
            'gamma': [0.01, 0.1, 1.0],   # Reasonable gamma range
            'epsilon': [0.01, 0.1, 0.2] if mode == 'regressor' else None,  # Epsilon only for regression
            'epochs': [50]  # Increased minimal epochs
        }
        
        # Remove None values
        param_grid = {k: v for k, v in param_grid.items() if v is not None}
        
        try:
            # Use grid search with negative MSE (original behavior)
            best_params, negative_mse = grid_search_cv(
                X, y, 
                model_class=KernelSVR if mode == 'regressor' else KernelSVM,
                param_grid=param_grid,
                cv_folds=2,  # Fast cross-validation
                task=mode,
                verbose=False
            )
            
            # Convert negative MSE to R┬▓ approximation
            # For regression: R┬▓ = 1 - MSE/Var(y), but with negative MSE we need different approach
            # Let's manually calculate R┬▓ for final validation
            from .preprocessing import train_test_split, standardize_fit, standardize_apply
            
            # Quick validation with best parameters
            X_train, X_val, y_train, y_val, _, _ = train_test_split(X, y, test_size=0.3, seed=42)
            X_train_s, mean, std = standardize_fit(X_train)
            X_val_s = standardize_apply(X_val, mean, std)
            
            model_class = KernelSVR if mode == 'regressor' else KernelSVM
            model = model_class(**{k: v for k, v in best_params.items() if k not in ['score']})
            
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_val_s)
            actual_r2 = r2_score(y_val, y_pred)
            
            best_params['score'] = actual_r2
            print(f">> Stage 1 complete. Best R┬▓: {actual_r2:.4f} (from negative MSE: {negative_mse:.4f})")
            
        except Exception as e:
            print(f">> Stage 1 error: {e}, using defaults")
            best_params = {
                'kernel': 'rbf',
                'C': 1.0,
                'gamma': 1.0,
                'epsilon': 0.1 if mode == 'regressor' else 0.1,
                'score': -float('inf')
            }
        
        return best_params
    
    def _quick_screening(self, X: np.ndarray, y: np.ndarray, mode: str,
                        start_time: float) -> Dict[str, Any]:
        """Legacy quick screening (for backward compatibility)."""
        return self._improved_quick_screening(X, y, mode, start_time)

    def _improved_medium_optimization(self, X: np.ndarray, y: np.ndarray, mode: str,
                                    start_time: float, prev_best: Dict[str, Any] = None) -> Dict[str, Any]:
        """Improved Stage 2: Medium optimization with adaptive grid."""
        from .hyperparameter_tuning import grid_search_cv
        
        metric = 'Accuracy' if mode == 'classifier' else 'R┬▓'
        print(f">> Stage 2: Medium optimization with expanded grid...")
        
        # More comprehensive grid based on previous results
        best_kernel = prev_best.get('kernel', 'rbf') if prev_best else 'rbf'
        best_C = prev_best.get('C', 1.0) if prev_best else 1.0
        best_gamma = prev_best.get('gamma', 1.0) if prev_best else 1.0
        
        # Adaptive parameter grid - more comprehensive for Stage 2
        param_grid = {
            'kernel': [best_kernel],  # Focus on best performing kernel
            'C': self._generate_c_grid(best_C),
            'gamma': self._generate_gamma_grid(best_gamma),
            'epsilon': self._generate_epsilon_grid() if mode == 'regressor' else None,
            'epochs': [100, 150]  # More epochs for better convergence
        }
        
        param_grid = {k: v for k, v in param_grid.items() if v is not None}
        
        try:
            best_params, negative_mse = grid_search_cv(
                X, y,
                model_class=KernelSVR if mode == 'regressor' else KernelSVM,
                param_grid=param_grid,
                cv_folds=3,  # More folds for stability
                task=mode,
                verbose=False
            )
            
            # Calculate actual R┬▓ for validation
            from .preprocessing import train_test_split, standardize_fit, standardize_apply
            from .metrics import r2_score
            
            X_train, X_val, y_train, y_val, _, _ = train_test_split(X, y, test_size=0.3, seed=42)
            X_train_s, mean, std = standardize_fit(X_train)
            X_val_s = standardize_apply(X_val, mean, std)
            
            model_class = KernelSVR if mode == 'regressor' else KernelSVM
            model = model_class(**{k: v for k, v in best_params.items() if k not in ['score']})
            
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_val_s)
            actual_r2 = r2_score(y_val, y_pred)
            
            best_params['score'] = actual_r2
            print(f">> Stage 2 complete. Best R┬▓: {actual_r2:.4f} (from negative MSE: {negative_mse:.4f})")
            
            # Time check
            time_elapsed = time.time() - start_time
            time_remaining = self.max_time_minutes * 60 - time_elapsed
            if time_remaining < 120:  # Less than 2 minutes left
                print(f">> Time remaining: {time_remaining/60:.1f} min - proceeding to Stage 3")
            
        except Exception as e:
            print(f">> Stage 2 error: {e}, using previous best")
            best_params = prev_best.copy() if prev_best else {
                'kernel': 'rbf', 'C': 1.0, 'gamma': 1.0, 'epsilon': 0.1, 'score': -float('inf')
            }
        
        return best_params
    
    def _medium_optimization(self, X: np.ndarray, y: np.ndarray, mode: str,
                           start_time: float, prev_best: Dict[str, Any] = None) -> Dict[str, Any]:
        """Legacy medium optimization (for backward compatibility)."""
        return self._improved_medium_optimization(X, y, mode, start_time, prev_best)
    
    def _generate_c_grid(self, best_c: float) -> list:
        """Generate adaptive C parameter grid."""
        return [
            max(0.01, best_c / 100),
            max(0.05, best_c / 10), 
            best_c,
            min(1000.0, best_c * 10),
            min(5000.0, best_c * 100)
        ]
    
    def _generate_gamma_grid(self, best_gamma: float) -> list:
        """Generate adaptive gamma parameter grid."""
        return [
            max(0.001, best_gamma / 100),
            max(0.01, best_gamma / 10),
            best_gamma,
            min(10.0, best_gamma * 10)
        ]
    
    def _generate_epsilon_grid(self) -> list:
        """Generate epsilon parameter grid for SVR."""
        return [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]

    def _improved_final_refinement(self, X: np.ndarray, y: np.ndarray, mode: str,
                                  start_time: float, prev_best: Dict[str, Any] = None) -> Dict[str, Any]:
        """Improved Stage 3: Final refinement with comprehensive tuning."""
        from .hyperparameter_tuning import auto_tune_learning_rate, auto_tune_C
        
        if not prev_best:
            prev_best = {
                'kernel': 'rbf',
                'C': 1.0,
                'gamma': 1.0,
                'epsilon': 0.1,
                'score': -float('inf')
            }
        
        print(">> Stage 3: Final refinement with comprehensive tuning...")
        
        model_class = KernelSVR if mode == 'regressor' else KernelSVM
        
        # Calculate remaining time for adaptive tuning
        time_elapsed = time.time() - start_time
        time_remaining = max(60, self.max_time_minutes * 60 - time_elapsed)  # At least 1 minute
        
        # Adaptive epochs based on remaining time - optimized for crypto data
        final_epochs = min(1500, int(time_remaining * 1.2 * 10))  # 12 epochs/sec, more stable
        tuning_epochs = min(300, int(time_remaining * 0.8 * 10))   # 8 epochs/sec for tuning
        
        print(f">> Remaining time: {time_remaining/60:.1f} min, Using epochs: {tuning_epochs} for tuning, {final_epochs} for final")
        
        try:
            # For Stage 3, use the best parameters from Stage 2 and just optimize learning rate
            crypto_like = self._detect_crypto_data(y)
            
            if crypto_like:
                # For crypto data, use conservative approach - keep Stage 2 parameters
                final_params = prev_best.copy()
                
                # Only tune learning rate for final optimization
                lr_result = auto_tune_learning_rate(
                    X, y,
                    model_class=model_class,
                    C=final_params['C'],
                    task=mode,
                    max_epochs=min(100, tuning_epochs),  # Quick LR tuning
                    k_folds=2,
                    verbose=False
                )
                final_params['learning_rate'] = lr_result['best_lr']
                
                # Ensure reasonable parameter ranges for crypto
                final_params['C'] = max(0.5, min(20.0, final_params.get('C', 1.0)))
                final_params['gamma'] = max(0.01, min(5.0, final_params.get('gamma', 1.0)))
                final_params['epsilon'] = max(0.01, min(0.2, final_params.get('epsilon', 0.1)))
                
            else:
                # For regular data, do full parameter tuning
                lr_result = auto_tune_learning_rate(
                    X, y,
                    model_class=model_class,
                    C=prev_best['C'],
                    task=mode,
                    max_epochs=tuning_epochs,
                    k_folds=3,
                    verbose=False
                )
                
                C_result = auto_tune_C(
                    X, y,
                    model_class=model_class,
                    learning_rate=lr_result['best_lr'],
                    task=mode,
                    max_epochs=tuning_epochs,
                    k_folds=3,
                    verbose=False
                )
                
                final_params = {
                    'kernel': prev_best['kernel'],
                    'C': max(0.1, min(100.0, C_result['best_C'])),
                    'gamma': prev_best['gamma'],
                    'epsilon': prev_best.get('epsilon', 0.1),
                    'learning_rate': lr_result['best_lr'],
                    'score': prev_best['score']
                }
            
        except Exception as e:
            print(f">> Tuning error: {e}, using safe parameters")
            final_params = prev_best.copy()
            final_params['learning_rate'] = 0.01  # More conservative default
            final_params['C'] = max(0.5, min(10.0, final_params.get('C', 1.0)))
            final_params['gamma'] = max(0.01, min(2.0, final_params.get('gamma', 1.0)))
        
        # Final training with optimal epochs
        from .preprocessing import standardize_fit, standardize_apply, train_test_split
        
        # Use 80/20 split for final evaluation
        X_train, X_val, y_train, y_val, _, _ = train_test_split(X, y, test_size=0.2, seed=42)
        
        X_train_scaled, mean, std = standardize_fit(X_train)
        X_val_scaled = standardize_apply(X_val, mean, std)
        
        # Create and train final model with adaptive epochs
        if mode == 'regressor':
            model = KernelSVR(
                kernel=final_params['kernel'],
                C=final_params['C'],
                gamma=final_params['gamma'],
                epsilon=final_params['epsilon'],
                learning_rate=final_params['learning_rate'],
                epochs=final_epochs
            )
        else:
            model = KernelSVM(
                kernel=final_params['kernel'],
                C=final_params['C'],
                gamma=final_params['gamma'],
                learning_rate=final_params['learning_rate'],
                epochs=final_epochs
            )
        
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_val_scaled)
        
        # Final score
        from .metrics import r2_score, accuracy
        if mode == 'regressor':
            try:
                final_score = r2_score(y_val, y_pred)
            except:
                final_score = -float('inf')
        else:
            final_score = accuracy(y_val, y_pred)
        
        final_params['score'] = final_score
        
        # If score is worse than previous stage or extremely bad, keep previous best
        metric = 'R┬▓' if mode == 'regressor' else 'Accuracy'
        target_threshold = self.target_r2 if mode == 'regressor' else self.target_accuracy
        
        if (final_score < (target_threshold * 0.5) or 
            (prev_best and 'score' in prev_best and final_score < prev_best['score'] * 0.95)):
            print(f">> Poor {metric} detected ({final_score:.4f}), keeping previous best result...")
            if prev_best and 'score' in prev_best:
                print(f">> FINAL RESULT: Keeping Stage 2 {metric} = {prev_best['score']:.4f}")
                return prev_best
            else:
                print(">> FINAL RESULT: Using safe defaults")
                return {
                    'kernel': 'rbf',
                    'C': 1.0,
                    'gamma': 0.1,
                    'epsilon': 0.1 if mode == 'regressor' else 0.1,
                    'learning_rate': 0.1,
                    'score': final_score
                }
        
        # Check if target achieved
        if final_score >= target_threshold:
            print(f">> TARGET ACHIEVED in Stage 3! {metric} = {final_score:.4f}")
        else:
            print(f">> FINAL RESULT: {metric} = {final_score:.4f}")
        return final_params
    
    def _final_refinement(self, X: np.ndarray, y: np.ndarray, mode: str,
                         start_time: float, prev_best: Dict[str, Any] = None) -> Dict[str, Any]:
        """Legacy final refinement (for backward compatibility)."""
        return self._improved_final_refinement(X, y, mode, start_time, prev_best)


def fast_auto_optimize(X: np.ndarray, y: np.ndarray, mode: str = 'regressor', 
                      max_time_minutes: int = 15) -> Dict[str, Any]:
    """
    Simplified interface for fast auto-optimization.
    Supports both classification and regression.
    """
    target_accuracy = 0.85 if mode == 'classifier' else 0.80
    optimizer = FastSVMHyperparameterOptimizer(
        max_time_minutes, 
        target_r2=0.80, 
        target_accuracy=target_accuracy
    )
    return optimizer.adaptive_optimize(X, y, mode)


def auto_optimize_parameters(prepared_data: Prepareddata, mode: str = 'classifier',
                           max_time_minutes: int = 10) -> Dict[str, Any]:
    """
    Convenience function for automatic parameter optimization.
    
    Args:
        prepared_data: Prepared dataset
        mode: 'classifier' or 'regressor'
        max_time_minutes: Maximum optimization time
        
    Returns:
        Optimized parameters dictionary
    """
    optimizer = SVMHyperparameterOptimizer(max_time_minutes)
    return optimizer.optimize_all_parameters(prepared_data, mode)
