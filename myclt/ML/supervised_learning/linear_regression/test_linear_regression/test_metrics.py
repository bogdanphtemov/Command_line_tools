"""Tests for metrics module - MSE, RMSE, R²"""

import numpy as np
import pytest
from myclt.ML.supervised_learning.linear_regression.metrics import mse, rmse, r2_score
import warnings


class TestMetrics:
    """Test MSE, RMSE, and R² metrics with various input types and edge cases."""
    
    # Test data for happy path
    Y_TRUE = np.array([3.0, -0.5, 2.0, 7.0])
    Y_PRED = np.array([2.5, 0.0, 2.0, 8.0])
    
    def test_mse_happy_path(self):
        """Test MSE with perfect and imperfect predictions."""
        # Perfect prediction
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        assert mse(y_true, y_pred) == 0.0
        
        # Imperfect prediction
        result = mse(self.Y_TRUE, self.Y_PRED)
        expected = ((0.5)**2 + (-0.5)**2 + (0.0)**2 + (-1.0)**2) / 4
        assert result == pytest.approx(expected)
    
    def test_rmse_happy_path(self):
        """Test RMSE as sqrt(MSE)."""
        mse_value = mse(self.Y_TRUE, self.Y_PRED)
        rmse_value = rmse(self.Y_TRUE, self.Y_PRED)
        assert rmse_value == pytest.approx(np.sqrt(mse_value))
    
    def test_r2_score_happy_path(self):
        """Test R² with perfect, imperfect, and worse-than-mean predictions."""
        # Perfect prediction
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        assert r2_score(y_true, y_pred) == 1.0
        
        # Imperfect but better than mean
        result = r2_score(self.Y_TRUE, self.Y_PRED)
        assert -1.0 < result < 1.0
        assert result > 0.0  # Should be positive since better than mean
    
    def test_r2_score_constant_target(self):
        """Test R² with constant target values."""
        y_true = np.array([5.0, 5.0, 5.0, 5.0])
        
        # Perfect prediction of constant target
        y_pred = np.array([5.0, 5.0, 5.0, 5.0])
        assert r2_score(y_true, y_pred) == 1.0
        
        # Imperfect prediction of constant target
        with pytest.warns(UserWarning, match=r"R\^2 score is not well-defined"):
            y_pred = np.array([4.0, 4.0, 4.0, 4.0])
            assert r2_score(y_true, y_pred) == 0.0
    
    @pytest.mark.parametrize("func", [mse, rmse, r2_score])
    def test_input_types(self, func):
        """Test that metrics work with list, tuple, and array inputs."""
        y_true = [1.0, 2.0, 3.0]
        y_pred = [1.1, 1.9, 3.1]
        
        # Lists
        result_list = func(y_true, y_pred)
        
        # Tuples  
        result_tuple = func(tuple(y_true), tuple(y_pred))
        
        # Arrays
        result_array = func(np.array(y_true), np.array(y_pred))
        
        # All should give the same result
        assert result_list == result_array
        assert result_tuple == result_array
        assert isinstance(result_list, float)
    
    @pytest.mark.parametrize("func", [mse, rmse, r2_score])
    def test_dimension_validation(self, func):
        """Test that 2D arrays are rejected."""
        y_true = np.array([[1.0], [2.0], [3.0]])
        y_pred = np.array([[1.0], [2.0], [3.0]])
        
        with pytest.raises(ValueError, match="must be 1D"):
            func(y_true, y_pred)
    
    @pytest.mark.parametrize("func", [mse, rmse, r2_score])
    def test_length_mismatch(self, func):
        """Test that mismatched lengths are rejected."""
        y_true = np.array([1.0, 2.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        
        with pytest.raises(ValueError, match="sample mismatch"):
            func(y_true, y_pred)
    
    @pytest.mark.parametrize("func", [mse, rmse, r2_score])
    def test_empty_input(self, func):
        """Test that empty arrays are rejected."""
        y_true = np.array([])
        y_pred = np.array([])
        
        with pytest.raises(ValueError, match="must not be empty"):
            func(y_true, y_pred)
    
    @pytest.mark.parametrize("func", [mse, rmse, r2_score])
    def test_nan_inf_validation(self, func):
        """Test that NaN and infinity values are rejected."""
        y_true = np.array([1.0, 2.0, np.nan])
        y_pred = np.array([1.0, 2.0, 3.0])
        
        with pytest.raises(ValueError, match="NaN or infinity"):
            func(y_true, y_pred)
        
        y_true = np.array([1.0, 2.0, np.inf])
        with pytest.raises(ValueError, match="NaN or infinity"):
            func(y_true, y_pred)


class TestR2EdgeCases:
    """Test edge cases specific to R² calculation."""
    
    def test_r2_worse_than_mean(self):
        """Test R² when predictions are worse than predicting the mean."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([10.0, 10.0, 10.0])  # All predictions at 10
        
        result = r2_score(y_true, y_pred)
        assert result < 0.0  # Should be negative
    
    def test_r2_nearly_constant_target(self):
        """Test R² with nearly constant target (tiny but nonzero SS_tot)."""
        y_true = np.array([1.0, 1.0 + 1e-10, 1.0 - 1e-10, 1.0])
        y_pred = np.array([2.0, 2.0, 2.0, 2.0])
        
        # Should not warn (SS_tot is tiny but nonzero)
        result = r2_score(y_true, y_pred)
        # Result can be any finite value due to numerical instability
        assert np.isfinite(result)