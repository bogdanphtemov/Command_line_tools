import numpy as np
import pytest
from typing import List, Dict, Any
from unittest.mock import patch

from myclt.ML.supervised_learning.linear_regression.hyperparameter_tuning import (
    k_fold_split, grid_search_regularization, auto_tune_learning_rate, _evaluate_lr_grid
)


class TestKFoldSplit:
    """Test cases for k_fold_split function."""

    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic data for testing."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = 2 * X[:, 0] + 1.5 * X[:, 1] - X[:, 2] + 3 + 0.1 * np.random.randn(100)
        return X, y

    def test_k_must_be_at_least_2(self, synthetic_data):
        """Test that k < 2 raises ValueError."""
        X, y = synthetic_data
        
        with pytest.raises(ValueError, match="k must be at least 2"):
            k_fold_split(X, y, k=1, seed=42)

    def test_n_samples_less_than_k(self):
        """Test that n_samples < k raises ValueError."""
        X = np.random.randn(3, 2)
        y = np.random.randn(3)
        
        with pytest.raises(ValueError, match="n_samples .* must be >= k"):
            k_fold_split(X, y, k=5, seed=42)

    def test_x_not_2d_raises_error(self):
        """Test that 1D X raises ValueError."""
        X = np.random.randn(10)  # 1D instead of 2D
        y = np.random.randn(10)
        
        with pytest.raises(ValueError, match="X must be 2D"):
            k_fold_split(X, y, k=2, seed=42)

    def test_y_not_1d_raises_error(self, synthetic_data):
        """Test that 2D y raises ValueError."""
        X, y = synthetic_data
        y_2d = y.reshape(-1, 1)  # Make y 2D
        
        with pytest.raises(ValueError, match="y must be 1D"):
            k_fold_split(X, y_2d, k=2, seed=42)

    def test_sample_mismatch_raises_error(self, synthetic_data):
        """Test that X/y length mismatch raises ValueError."""
        X, y = synthetic_data
        
        with pytest.raises(ValueError, match=f"X/y sample mismatch"):
            k_fold_split(X, y[:-5], k=2, seed=42)

    def test_nan_in_x_raises_error(self, synthetic_data):
        """Test that NaN in X raises ValueError."""
        X, y = synthetic_data
        X[0, 0] = np.nan
        
        with pytest.raises(ValueError, match="NaN or infinity"):
            k_fold_split(X, y, k=2, seed=42)

    def test_inf_in_y_raises_error(self, synthetic_data):
        """Test that infinity in y raises ValueError."""
        X, y = synthetic_data
        y[0] = np.inf
        
        with pytest.raises(ValueError, match="NaN or infinity"):
            k_fold_split(X, y, k=2, seed=42)

    def test_correct_number_of_folds(self, synthetic_data):
        """Test that function returns correct number of folds."""
        X, y = synthetic_data
        k = 5
        folds = k_fold_split(X, y, k=k, seed=42)
        
        assert len(folds) == k
        
        # Each fold should contain 4 arrays
        for fold in folds:
            assert len(fold) == 4  # X_train, X_val, y_train, y_val

    def test_each_sample_in_exactly_one_validation_fold(self, synthetic_data):
        """Test that each sample appears exactly once in validation set across all folds."""
        X, y = synthetic_data
        n_samples = X.shape[0]
        k = 5
        folds = k_fold_split(X, y, k=k, seed=42)
        
        val_indices = []
        for X_train, X_val, y_train, y_val in folds:
            # Convert arrays back to indices by checking against original data
            for val_sample in X_val:
                # Find index in original X (simple approach for testing)
                for i, sample in enumerate(X):
                    if np.array_equal(sample, val_sample):
                        val_indices.append(i)
                        break
        
        # Check that all validation indices are unique
        assert len(val_indices) == len(set(val_indices)) == n_samples
        
        # Check that all indices from 0 to n_samples-1 are present
        assert set(val_indices) == set(range(n_samples))

    def test_fold_sizes_reasonable(self, synthetic_data):
        """Test that fold sizes are approximately equal."""
        X, y = synthetic_data
        n_samples = X.shape[0]
        k = 5
        folds = k_fold_split(X, y, k=k, seed=42)
        
        val_sizes = []
        for _, X_val, _, _ in folds:
            val_sizes.append(X_val.shape[0])
        
        # Sizes should be approximately n_samples/k
        expected_size = n_samples // k
        assert all(size >= expected_size - 1 and size <= expected_size + 1 
                   for size in val_sizes)

    def test_reproducibility_with_same_seed(self, synthetic_data):
        """Test that same seed produces identical results."""
        X, y = synthetic_data
        
        folds1 = k_fold_split(X, y, k=5, seed=42)
        folds2 = k_fold_split(X, y, k=5, seed=42)
        
        # Check that folds are identical
        for i in range(len(folds1)):
            for j in range(4):  # X_train, X_val, y_train, y_val
                np.testing.assert_array_equal(folds1[i][j], folds2[i][j])

    def test_different_seeds_produce_different_folds(self, synthetic_data):
        """Test that different seeds produce different results (with high probability)."""
        X, y = synthetic_data
        
        folds1 = k_fold_split(X, y, k=5, seed=42)
        folds2 = k_fold_split(X, y, k=5, seed=123)
        
        # They should be different (check first validation set as proxy)
        assert not np.array_equal(folds1[0][1], folds2[0][1])


class TestGridSearchRegularization:
    """Test cases for grid_search_regularization function."""

    @pytest.fixture
    def tiny_dataset(self):
        """Tiny dataset for fast testing."""
        np.random.seed(42)
        X = np.random.randn(20, 2)
        y = 2 * X[:, 0] + 1.5 * X[:, 1] + 1 + 0.1 * np.random.randn(20)
        return X, y

    def test_empty_grid_raises_error(self, tiny_dataset):
        """Test that empty grid raises ValueError."""
        X, y = tiny_dataset
        
        with pytest.raises(ValueError, match="must be non-empty"):
            grid_search_regularization(X, y, lambda_l1_grid=[], lambda_l2_grid=[0.0])
        
        with pytest.raises(ValueError, match="must be non-empty"):
            grid_search_regularization(X, y, lambda_l1_grid=[0.0], lambda_l2_grid=[])

    def test_returns_expected_keys(self, tiny_dataset):
        """Test that function returns expected dictionary keys."""
        X, y = tiny_dataset
        
        result = grid_search_regularization(
            X, y, 
            lambda_l1_grid=[0.0, 0.1],
            lambda_l2_grid=[0.0],
            k_folds=2,
            epochs=10  # Small for speed
        )
        
        expected_keys = {'best_lambda_l1', 'best_lambda_l2', 'best_mse', 'results'}
        assert set(result.keys()) == expected_keys

    def test_results_not_empty_and_sorted(self, tiny_dataset):
        """Test that results list is non-empty and sorted by mean_mse."""
        X, y = tiny_dataset
        
        result = grid_search_regularization(
            X, y,
            lambda_l1_grid=[0.0, 0.1],
            lambda_l2_grid=[0.0],
            k_folds=2,
            epochs=10
        )
        
        assert len(result['results']) > 0
        
        # Check sorting by mean_mse (ascending)
        mean_mses = [r['mean_mse'] for r in result['results']]
        assert mean_mses == sorted(mean_mses)

    def test_best_params_minimize_mse(self, tiny_dataset):
        """Test that best parameters actually minimize MSE."""
        X, y = tiny_dataset
        
        result = grid_search_regularization(
            X, y,
            lambda_l1_grid=[0.0, 0.1, 1.0],
            lambda_l2_grid=[0.0],
            k_folds=2,
            epochs=10
        )
        
        best_mse = result['best_mse']
        min_mse_in_results = min(r['mean_mse'] for r in result['results'])
        
        assert best_mse == pytest.approx(min_mse_in_results)

    def test_best_mse_is_finite(self, tiny_dataset):
        """Test that best_mse is finite (not NaN or inf)."""
        X, y = tiny_dataset
        
        result = grid_search_regularization(
            X, y,
            lambda_l1_grid=[0.0, 0.1],
            lambda_l2_grid=[0.0],
            k_folds=2,
            epochs=10
        )
        
        assert np.isfinite(result['best_mse'])

    @pytest.mark.slow
    def test_large_divergent_learning_rate_raises_error(self):
        """Test that highly divergent learning rate causes RuntimeError (marked as slow)."""
        np.random.seed(42)
        X = np.random.randn(30, 2)
        y = 2 * X[:, 0] + 1.5 * X[:, 1] + 1
        
        # Use extremely large learning rate and disable early stopping to force divergence
        with pytest.raises(RuntimeError, match="All regularization combinations diverged"):
            grid_search_regularization(
                X, y,
                learning_rate=1000.0,  # Extremely large learning rate
                lambda_l1_grid=[0.0],
                lambda_l2_grid=[0.0],
                k_folds=2,
                epochs=10,
                early_stopping=False  # Disable to ensure divergence
            )

    def test_verbose_output(self, tiny_dataset, capsys):
        """Test that verbose mode produces output."""
        X, y = tiny_dataset
        
        # Mock print to speed up test
        with patch('builtins.print') as mock_print:
            grid_search_regularization(
                X, y,
                lambda_l1_grid=[0.0],
                lambda_l2_grid=[0.0],
                k_folds=2,
                epochs=5,
                verbose=True
            )
        
        # Check that banner was printed
        assert mock_print.called

    def test_with_progress_output(self, tiny_dataset, capsys):
        """Test that non-verbose mode produces progress output."""
        X, y = tiny_dataset
        
        result = grid_search_regularization(
            X, y,
            lambda_l1_grid=[0.0, 0.1],
            lambda_l2_grid=[0.0, 0.1],
            k_folds=2,
            epochs=5,
            verbose=False
        )
        
        captured = capsys.readouterr()
        # Should contain grid search banner and progress
        assert "Grid Search" in captured.out or "Progress:" in captured.out


class TestAutoTuneLearningRate:
    """Test cases for auto_tune_learning_rate function."""

    @pytest.fixture
    def tiny_dataset(self):
        """Tiny dataset for fast testing."""
        np.random.seed(42)
        X = np.random.randn(30, 2)
        y = 2 * X[:, 0] + 1.5 * X[:, 1] + 1 + 0.1 * np.random.randn(30)
        return X, y

    def test_returns_expected_keys(self, tiny_dataset):
        """Test that function returns expected dictionary keys."""
        X, y = tiny_dataset
        
        result = auto_tune_learning_rate(
            X, y,
            learning_rate_grid=[0.001, 0.01],
            epochs=10,
            k_folds=2,
            verbose=False
        )
        
        expected_keys = {'best_lr', 'best_mse', 'results', 'phase2_results'}
        assert set(result.keys()) == expected_keys

    def test_no_refine_results_in_none_phase2(self, tiny_dataset):
        """Test that do_refine=False results in phase2_results=None."""
        X, y = tiny_dataset
        
        result = auto_tune_learning_rate(
            X, y,
            learning_rate_grid=[0.001, 0.01],
            epochs=10,
            k_folds=2,
            do_refine=False,
            verbose=False
        )
        
        assert result['phase2_results'] is None

    def test_phase2_results_none_when_no_refine(self, tiny_dataset):
        """Test that phase2_results is None when do_refine=False.""" 
        X, y = tiny_dataset
        
        result = auto_tune_learning_rate(
            X, y,
            learning_rate_grid=[0.001, 0.01],
            epochs=10,
            k_folds=2,
            do_refine=False,
            verbose=False
        )
        
        assert result['phase2_results'] is None

    def test_best_lr_in_grid(self, tiny_dataset):
        """Test that best_lr is either from the original grid or refinement phase.""" 
        X, y = tiny_dataset
        lr_grid = [0.001, 0.01, 0.1]
        
        # Test without refinement to ensure it's from original grid
        result = auto_tune_learning_rate(
            X, y,
            learning_rate_grid=lr_grid,
            epochs=10,
            k_folds=2,
            do_refine=False,  # Disable refinement
            verbose=False
        )
        
        assert result['best_lr'] in lr_grid

    def test_best_mse_is_finite(self, tiny_dataset):
        """Test that best_mse is finite."""
        X, y = tiny_dataset
        
        result = auto_tune_learning_rate(
            X, y,
            learning_rate_grid=[0.001, 0.01],
            epochs=10,
            k_folds=2,
            verbose=False
        )
        
        assert np.isfinite(result['best_mse'])


class TestEvaluateLRGrid:
    """Test cases for _evaluate_lr_grid function."""

    @pytest.fixture
    def tiny_dataset(self):
        """Tiny dataset for fast testing."""
        np.random.seed(42)
        X = np.random.randn(20, 2)
        y = 2 * X[:, 0] + 1.5 * X[:, 1] + 1 + 0.1 * np.random.randn(20)
        return X, y

    def test_lr_convergence_behavior(self, tiny_dataset):
        """Test various learning rates and their convergence behavior."""
        X, y = tiny_dataset
        
        # Test both very small and reasonable learning rates
        lr_grid = [0.0001, 0.01, 1.0]
        
        results = _evaluate_lr_grid(
            X, y, lr_grid,
            epochs=10, k_folds=2, seed=42,
            use_scaling=False, early_stopping_patience=3, verbose=False
        )
        
        assert len(results) == len(lr_grid)
        
        # Check that each result has correct structure
        for result in results:
            assert 'lr' in result
            assert 'mean_mse' in result
            assert 'std_mse' in result
            assert 'n_folds' in result
            assert 'diverged' in result
            
            if result['diverged']:
                assert result['mean_mse'] is None
            else:
                assert result['mean_mse'] is not None
                assert np.isfinite(result['mean_mse'])

    def test_converged_lr_has_finite_mean_mse(self, tiny_dataset):
        """Test that converged learning rates have finite mean_mse."""
        X, y = tiny_dataset
        
        lr_grid = [0.01]  # Should converge
        
        results = _evaluate_lr_grid(
            X, y, lr_grid,
            epochs=10, k_folds=2, seed=42,
            use_scaling=False, early_stopping_patience=5, verbose=False
        )
        
        assert len(results) == 1
        assert results[0]['mean_mse'] is not None
        assert np.isfinite(results[0]['mean_mse'])
        assert results[0]['diverged'] is False
        assert results[0]['n_folds'] == 2

    def test_std_mse_calculation(self, tiny_dataset):
        """Test that std_mse is calculated correctly."""
        X, y = tiny_dataset
        
        lr_grid = [0.01]
        
        results = _evaluate_lr_grid(
            X, y, lr_grid,
            epochs=10, k_folds=2, seed=42,
            use_scaling=False, early_stopping_patience=5, verbose=False
        )
        
        assert len(results) == 1
        result = results[0]
        
        if not result['diverged']:
            # std_mse should be finite for converged results
            assert result['std_mse'] is not None
            assert np.isfinite(result['std_mse'])

    def test_all_lr_in_grid_are_tested(self, tiny_dataset):
        """Test that all learning rates in grid are evaluated."""
        X, y = tiny_dataset
        
        lr_grid = [0.001, 0.01, 0.1]
        
        results = _evaluate_lr_grid(
            X, y, lr_grid,
            epochs=5, k_folds=2, seed=42,
            use_scaling=False, early_stopping_patience=2, verbose=False
        )
        
        # Should return results for all learning rates in grid
        assert len(results) == len(lr_grid)
        
        # Each result should have the correct lr value
        result_lrs = [r['lr'] for r in results]
        assert set(result_lrs) == set(lr_grid)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])