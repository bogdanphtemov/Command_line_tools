"""Tests for preprocessing functions - train_test_split, standardize"""

import numpy as np
import pytest
from myclt.ML.base_models import train_test_split, standardize_fit, standardize_apply
from myclt.ML.base_models import TEST_SIZE_MIN, TEST_SIZE_MAX


class TestTrainTestSplit:
    """Test train_test_split function with various scenarios."""
    
    def test_basic_split(self):
        """Test basic functionality with valid inputs."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([10, 20, 30, 40, 50])
        
        X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
            X, y, test_size=0.4, seed=42
        )
        
        # Check shapes (5 samples * 0.4 = 2 test samples)
        assert X_train.shape == (3, 1)
        assert X_test.shape == (2, 1)
        assert y_train.shape == (3,)
        assert y_test.shape == (2,)
        assert len(train_idx) == 3
        assert len(test_idx) == 2
        
        # Check that indices are correct
        assert y_train[0] == y[train_idx[0]]
        assert y_test[0] == y[test_idx[0]]
    
    def test_reproducibility(self):
        """Test that same seed produces same split."""
        X = np.random.randn(100, 2)
        y = np.random.randn(100)
        
        # First split
        X_train1, X_test1, y_train1, y_test1, _, _ = train_test_split(
            X, y, test_size=0.3, seed=123
        )
        
        # Second split with same seed
        X_train2, X_test2, y_train2, y_test2, _, _ = train_test_split(
            X, y, test_size=0.3, seed=123
        )
        
        # Should be identical
        np.testing.assert_array_equal(X_train1, X_train2)
        np.testing.assert_array_equal(X_test1, X_test2)
        np.testing.assert_array_equal(y_train1, y_train2)
        np.testing.assert_array_equal(y_test1, y_test2)
    
    def test_different_seeds(self):
        """Test that different seeds produce different splits."""
        X = np.random.randn(100, 2)
        y = np.random.randn(100)
        
        # Split with seed 1
        X_train1, X_test1, y_train1, y_test1, _, _ = train_test_split(
            X, y, test_size=0.3, seed=1
        )
        
        # Split with seed 2
        X_train2, X_test2, y_train2, y_test2, _, _ = train_test_split(
            X, y, test_size=0.3, seed=2
        )
        
        # Should be different (high probability)
        assert not np.array_equal(X_train1, X_train2)
    
    @pytest.mark.parametrize("test_size", [TEST_SIZE_MIN - 0.01, TEST_SIZE_MAX + 0.01, -0.1, 0.6])
    def test_invalid_test_size(self, test_size):
        """Test that invalid test sizes are rejected."""
        X = np.array([[1], [2], [3]])
        y = np.array([1, 2, 3])
        
        with pytest.raises(ValueError, match=f"between {TEST_SIZE_MIN} and {TEST_SIZE_MAX}"):
            train_test_split(X, y, test_size=test_size, seed=42)
    
    def test_test_size_bounds(self):
        """Test allowed test sizes."""
        X = np.random.randn(100, 2)
        y = np.random.randn(100)
        
        # Test min and max bounds
        for test_size in [TEST_SIZE_MIN, TEST_SIZE_MAX, 0.2, 0.3]:
            X_train, X_test, y_train, y_test, _, _ = train_test_split(
                X, y, test_size=test_size, seed=42
            )
            
            expected_test_size = int(round(100 * test_size))
            assert X_test.shape[0] == expected_test_size
    
    def test_small_dataset(self):
        """Test with minimal allowed dataset size."""
        X = np.array([[1], [2], [3], [4], [5]])  # 5 samples
        y = np.array([1, 2, 3, 4, 5])
        
        # Small test size (0.05 of 5 = 0.25, rounded to 0)
        X_train, X_test, y_train, y_test, _, _ = train_test_split(
            X, y, test_size=TEST_SIZE_MIN, seed=42
        )
        
        assert X_test.shape[0] == 0  # round(5 * 0.05) = nearest int(0.25) = 0
        assert X_train.shape[0] == 5


class TestStandardization:
    """Test standardization functions."""
    
    def test_standardize_fit_basic(self):
        """Test basic standardization."""
        X_train = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        
        X_scaled, mean, std = standardize_fit(X_train)
        
        # Check that scaled data has mean ≈ 0 and std ≈ 1
        np.testing.assert_allclose(X_scaled.mean(axis=0), [0, 0], atol=1e-10)
        np.testing.assert_allclose(X_scaled.std(axis=0), [1, 1], atol=1e-10)
        
        # Check that mean and std are calculated correctly
        np.testing.assert_allclose(mean, [3, 4])
        np.testing.assert_allclose(std, [1.63299316, 1.63299316], rtol=1e-6)
    
    def test_standardize_fit_zero_std(self):
        """Test standardization with zero standard deviation."""
        X_train = np.array([[1, 5], [1, 5], [1, 5]], dtype=float)  # Column 1 has std=0
        
        X_scaled, mean, std_safe = standardize_fit(X_train)
        
        # Column with zero std gets std_safe = 1.0
        np.testing.assert_allclose(std_safe, [1.0, 1.0])
        
        # First column should be all 0 (mean subtraction only)
        np.testing.assert_allclose(X_scaled[:, 0], [0, 0, 0])
        
        # Second column should have zero mean
        np.testing.assert_allclose(X_scaled.mean(axis=0), [0, 0], atol=1e-10)
    
    def test_standardize_apply_consistency(self):
        """Test that fit and apply work consistently."""
        X_train = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        X_test = np.array([[2, 3], [4, 5]], dtype=float)
        
        # Fit on training data
        _, mean, std = standardize_fit(X_train)
        
        # Apply to test data
        X_test_scaled = standardize_apply(X_test, mean, std)
        
        # Test data should be scaled with training parameters
        expected = (X_test - mean) / std
        np.testing.assert_allclose(X_test_scaled, expected)
    
    def test_standardize_round_trip(self):
        """Test that scaling can be reversed."""
        X_train = np.random.randn(100, 3) * 5 + 10  # Some offset and scaling
        
        # Scale
        X_scaled, mean, std = standardize_fit(X_train)
        
        # Reverse scale
        X_recovered = X_scaled * std + mean
        
        # Should get back original (within floating point precision)
        np.testing.assert_allclose(X_recovered, X_train, rtol=1e-10)
    
    def test_standardize_shapes(self):
        """Test that shapes are preserved."""
        n_samples, n_features = 50, 4
        X_train = np.random.randn(n_samples, n_features)
        X_test = np.random.randn(20, n_features)
        
        X_scaled, mean, std = standardize_fit(X_train)
        assert X_scaled.shape == (n_samples, n_features)
        assert mean.shape == (n_features,)
        assert std.shape == (n_features,)
        
        X_test_scaled = standardize_apply(X_test, mean, std)
        assert X_test_scaled.shape == X_test.shape
    
    def test_standardize_with_zero_features(self):
        """Test handling of zero features (no division by zero)."""
        X_train = np.array([[0, 0, 1], [0, 0, 2], [0, 0, 3]], dtype=float)
        
        X_scaled, mean, std_safe = standardize_fit(X_train)
        
        # First two columns have std=0, so std_safe should be 1.0
        assert std_safe[0] == 1.0
        assert std_safe[1] == 1.0
        # Third column: std of [1, 2, 3] = 0.816
        assert np.isclose(std_safe[2], 0.816, atol=0.01)
        
        # First two columns should be all 0 after mean subtraction
        np.testing.assert_allclose(X_scaled[:, :2], [[0, 0], [0, 0], [0, 0]])
    
    def test_standardize_single_sample(self):
        """Test standardization with single sample in training."""
        X_train = np.array([[1.0, 2.0, 3.0]])
        
        X_scaled, mean, std_safe = standardize_fit(X_train)
        
        # All columns should have std=0 -> std_safe=1.0
        np.testing.assert_allclose(std_safe, [1.0, 1.0, 1.0])
        
        # Mean subtracted data should be all 0
        np.testing.assert_allclose(X_scaled, [[0.0, 0.0, 0.0]])


class TestPreprocessingEdgeCases:
    """Test edge cases for preprocessing functions."""
    
    def test_empty_array_split(self):
        """Test handling of empty arrays."""
        X = np.array([]).reshape(0, 2)
        y = np.array([])
        
        # With empty arrays and small test_size, it should work (test_size=0)
        X_train, X_test, y_train, y_test, _, _ = train_test_split(
            X, y, test_size=0.2, seed=42
        )
        
        # Both arrays should be empty
        assert X_train.shape == (0, 2)
        assert X_test.shape == (0, 2)
        assert y_train.shape == (0,)
        assert y_test.shape == (0,)
    
    def test_shape_mismatch_split(self):
        """Test that mismatched X and y shapes cause IndexError."""
        X = np.array([[1], [2], [3]])
        y = np.array([1, 2])  # Different length
        
        # Function doesn't validate, so it will fail with IndexError
        # when trying to access y[train_idx] with mismatched indices
        with pytest.raises(IndexError):
            train_test_split(X, y, test_size=0.2, seed=42)
    
    def test_non_2d_array_split(self):
        """Test that 1D X array works (stays 1D after indexing)."""
        X = np.array([1, 2, 3])  # 1D, function expects 2D but doesn't validate
        y = np.array([1, 2, 3])
        
        # 1D array stays 1D after indexing
        X_train, X_test, y_train, y_test, _, _ = train_test_split(
            X, y, test_size=0.2, seed=42
        )
        
        # Result keeps the same dimension as input
        assert X_train.ndim == 1
        assert X_test.ndim == 1