import numpy as np
import pytest
from typing import Tuple

from myclt.ML.supervised_learning.linear_regression.preprocessing import (
    train_test_split, standardize_fit, standardize_apply
)


class TestPreprocessing:
    """Test cases for preprocessing functions: train_test_split, standardize_fit, standardize_apply."""
    
    @pytest.fixture
    def synthetic_data_2d(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create synthetic 2D data with 100 samples."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = 2 * X[:, 0] + 1.5 * X[:, 1] - X[:, 2] + 3
        return X, y
    
    @pytest.fixture
    def synthetic_data_1d(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create synthetic 1D data with 100 samples."""
        np.random.seed(42)
        X = np.random.randn(100, 1)
        y = 2 * X[:, 0] + 1
        return X, y
    
    @pytest.fixture

    def test_train_test_split_dimensions_and_reproducibility(self, synthetic_data_2d):
        """Test split dimensions and that same seed gives same results."""
        X, y = synthetic_data_2d
        n_samples = X.shape[0]
        
        # Test with same seed multiple times
        for test_size in [0.2, 0.3, 0.4]:
            seed = 42
            
            # First call
            X_train1, X_test1, y_train1, y_test1, train_idx1, test_idx1 = train_test_split(
                X, y, test_size, seed
            )
            
            # Second call with same seed - should get identical results
            X_train2, X_test2, y_train2, y_test2, train_idx2, test_idx2 = train_test_split(
                X, y, test_size, seed
            )
            
            # Check reproducibility
            np.testing.assert_array_equal(X_train1, X_train2)
            np.testing.assert_array_equal(X_test1, X_test2)
            np.testing.assert_array_equal(y_train1, y_train2)
            np.testing.assert_array_equal(y_test1, y_test2)
            np.testing.assert_array_equal(train_idx1, train_idx2)
            np.testing.assert_array_equal(test_idx1, test_idx2)
            
            # Check dimensions
            expected_test_n = int(round(n_samples * test_size))
            expected_train_n = n_samples - expected_test_n
            
            assert X_train1.shape[0] == expected_train_n
            assert X_test1.shape[0] == expected_test_n
            assert y_train1.shape[0] == expected_train_n
            assert y_test1.shape[0] == expected_test_n
            assert len(train_idx1) == expected_train_n
            assert len(test_idx1) == expected_test_n
            
            # Check feature dimensions preserved
            assert X_train1.shape[1] == X.shape[1]
            assert X_test1.shape[1] == X.shape[1]
    
    def test_train_test_split_different_seeds_produce_different_results(self, synthetic_data_2d):
        """Test that different seeds produce different splits (with high probability)."""
        X, y = synthetic_data_2d
        test_size = 0.2
        
        # Get split with seed 42
        X_train1, X_test1, y_train1, y_test1, train_idx1, test_idx1 = train_test_split(
            X, y, test_size, seed=42
        )
        
        # Get split with seed 123
        X_train2, X_test2, y_train2, y_test2, train_idx2, test_idx2 = train_test_split(
            X, y, test_size, seed=123
        )
        
        # They should be different (we can check train indices)
        assert not np.array_equal(train_idx1, train_idx2), "Different seeds should produce different splits"
        
        # But dimensions should be the same
        assert X_train1.shape == X_train2.shape
        assert X_test1.shape == X_test2.shape
    
    def test_train_test_split_disjoint_indices_and_coverage(self, synthetic_data_2d):
        """Test that train and test indices are disjoint and cover all indices."""
        X, y = synthetic_data_2d
        n_samples = X.shape[0]
        
        for test_size in [0.1, 0.25, 0.5]:
            X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
                X, y, test_size, seed=42
            )
            
            # Check disjointness
            train_set = set(train_idx)
            test_set = set(test_idx)
            assert train_set.isdisjoint(test_set), "Train and test indices should be disjoint"
            
            # Check coverage - all indices should be accounted for
            all_indices = set(range(n_samples))
            assert train_set.union(test_set) == all_indices, "Train and test should cover all indices"
            
            # Check correspondence between indices and data
            assert np.array_equal(X[train_idx], X_train)
            assert np.array_equal(X[test_idx], X_test)
            assert np.array_equal(y[train_idx], y_train)
            assert np.array_equal(y[test_idx], y_test)
            
            # Check specific correspondence
            assert y_train[0] == y[train_idx[0]], "y_train[0] should equal y[train_idx[0]]"
    

    def test_standardize_fit_basic_properties(self, synthetic_data_2d):
        """Test that standardized data has mean≈0 and std≈1."""
        X, y = synthetic_data_2d
        
        # Split data first
        X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
            X, y, 0.2, seed=42
        )
        
        # Apply standardization
        X_scaled, mean, std_safe = standardize_fit(X_train)
        
        # Check scaled data properties
        np.testing.assert_allclose(X_scaled.mean(axis=0), 0.0, atol=1e-10)
        np.testing.assert_allclose(X_scaled.std(axis=0), 1.0, atol=1e-10)
        
        # Check that mean and std_safe are reasonable
        assert X_scaled.shape == X_train.shape
        assert mean.shape == (X_train.shape[1],)
        assert std_safe.shape == (X_train.shape[1],)
        
        # std_safe should never be zero due to protection
        assert np.all(std_safe > 0)
    
    def test_standardize_fit_zero_variance(self):
        """Test standardization with constant feature (zero variance)."""
        # Create data with one constant feature
        np.random.seed(42)
        X = np.column_stack([
            np.random.randn(100, 1),  # Normal feature
            np.full((100, 1), 5.0)    # Constant feature
        ])
        
        X_scaled, mean, std_safe = standardize_fit(X)
        
        # Constant feature should have std_safe = 1.0 (protection)
        np.testing.assert_allclose(std_safe[1], 1.0)
        
        # Normal feature should be properly standardized
        np.testing.assert_allclose(X_scaled[:, 0].mean(), 0.0, atol=1e-10)
        np.testing.assert_allclose(X_scaled[:, 0].std(), 1.0, atol=1e-10)
        
        # Constant feature should become zeros
        np.testing.assert_allclose(X_scaled[:, 1], 0.0)

    def test_standardize_apply_basic(self, synthetic_data_2d):
        """Test standardized_apply with precomputed parameters."""
        X, y = synthetic_data_2d
        
        # Split data
        X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
            X, y, 0.2, seed=42
        )
        
        # Fit on training data
        X_train_scaled, mean, std_safe = standardize_fit(X_train)
        
        # Apply to test data
        X_test_scaled = standardize_apply(X_test, mean, std_safe)
        
        # Test data should be centered and scaled by training params
        np.testing.assert_allclose(X_test_scaled, (X_test - mean) / std_safe)
        
        # Test data statistics (typically won't be 0 mean, 1 std - that's correct!)
        # But we can verify the manual calculation
        assert X_test_scaled.shape == X_test.shape
    
    def test_standardize_apply_on_train_data(self, synthetic_data_2d):
        """Test that applying training params to training data gives the same result as fit."""
        X, y = synthetic_data_2d
        
        X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
            X, y, 0.2, seed=42
        )
        
        # Fit standardization
        X_scaled_fit, mean, std_safe = standardize_fit(X_train)
        
        # Apply same params to training data
        X_scaled_apply = standardize_apply(X_train, mean, std_safe)
        
        np.testing.assert_allclose(X_scaled_fit, X_scaled_apply, rtol=1e-10, atol=1e-10)
    
    def test_shapes_and_types(self):
        """Test that functions return correct shapes and types."""
        # Create simple dataset
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = np.random.randn(50)
        
        # Test train_test_split shapes
        X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
            X, y, 0.2, 42
        )
        
        assert isinstance(X_train, np.ndarray)
        assert isinstance(X_test, np.ndarray)
        assert isinstance(y_train, np.ndarray) 
        assert isinstance(y_test, np.ndarray)
        assert isinstance(train_idx, np.ndarray)
        assert isinstance(test_idx, np.ndarray)
        
        assert X_train.dtype == X.dtype
        assert X_test.dtype == X.dtype
        assert y_train.dtype == y.dtype
        assert y_test.dtype == y.dtype
        assert train_idx.dtype == np.int64
        assert test_idx.dtype == np.int64
        
        # Test standardize_fit shapes
        X_scaled, mean, std_safe = standardize_fit(X_train)
        
        assert isinstance(X_scaled, np.ndarray)
        assert isinstance(mean, np.ndarray) 
        assert isinstance(std_safe, np.ndarray)
        
        assert X_scaled.shape == X_train.shape
        assert mean.shape == (X_train.shape[1],)
        assert std_safe.shape == (X_train.shape[1],)
        
        # Test standardize_apply shapes
        X_scaled_apply = standardize_apply(X_test, mean, std_safe)
        assert X_scaled_apply.shape == X_test.shape
    
    def test_standardize_fit_roundtrip(self, synthetic_data_2d):
        """Test that (X_scaled * std_safe + mean) ≈ X_train."""
        X, y = synthetic_data_2d
        X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
            X, y, 0.2, seed=42
        )
        
        X_scaled, mean, std_safe = standardize_fit(X_train)
        
        # Reverse the transformation
        X_manual = X_scaled * std_safe + mean
        
        np.testing.assert_allclose(X_manual, X_train, rtol=1e-10, atol=1e-10)
    def test_train_test_split_1d_X(self, synthetic_data_1d):
        """Test that 1D feature matrix works correctly."""
        X, y = synthetic_data_1d
        
        X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
            X, y, 0.2, seed=42
        )
        
        # Check dimensions
        assert X_train.ndim == 2
        assert X_test.ndim == 2
        assert y_train.ndim == 1
        assert y_test.ndim == 1
        
        # Feature dimension should be preserved
        assert X_train.shape[1] == 1
        assert X_test.shape[1] == 1
    
    def test_train_test_split_edge_case_n1(self, single_sample_data):
        """Test single sample edge case."""
        X, y = single_sample_data
        
        # For n=1, test_split doesn't raise error but works differently
        # Let's check that it handles this edge case gracefully
        X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
            X, y, 0.2, seed=42
        )
        
        # Since n=1 and test_size=0.2, we round to 0 test samples
        # So all data should go to training
        assert len(X_train) == 1
        assert len(X_test) == 0
        assert len(y_train) == 1
        assert len(y_test) == 0
        assert len(train_idx) == 1
        assert len(test_idx) == 0
    
    def test_train_test_split_invalid_test_size(self, synthetic_data_2d):
        """Test that invalid test_size raises appropriate ValueError."""
        X, y = synthetic_data_2d
        
        # Test sizes outside bounds
        for invalid_test_size in [-0.1, 0.0, 0.04, 0.51, 1.0]:
            with pytest.raises(ValueError, match=f"test_size should be between"):
                train_test_split(X, y, invalid_test_size, seed=42)
        
        # Valid test sizes should work
        for valid_test_size in [0.05, 0.1, 0.25, 0.5]:
            try:
                train_test_split(X, y, valid_test_size, seed=42)
            except ValueError as e:
                pytest.fail(f"Valid test_size {valid_test_size} should not raise ValueError: {e}")
    def small_dataset_1d(self) -> Tuple[np.ndarray, np.ndarray]:
        """Small dataset with only 10 samples (edge case)."""
        np.random.seed(42)
        X = np.random.randn(10, 1)
        y = 2 * X[:, 0] + 1
        return X, y
    
    @pytest.fixture  
    def single_sample_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Single sample data (extreme edge case)."""
        X = np.array([[5.0]])
        y = np.array([11.0])
        return X, y