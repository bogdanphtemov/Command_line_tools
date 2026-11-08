import numpy as np
import pytest
import warnings
from typing import List, Tuple, Union

from myclt.ML.supervised_learning.linear_regression.metrics import mse, rmse, r2_score


class TestMetrics:
    """Test cases for regression metrics: MSE, RMSE, R²."""
    
    @pytest.fixture
    def perfect_match_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Perfectly matching predictions: y_pred == y_true."""
        np.random.seed(42)
        y_true = np.random.randn(100)
        y_pred = y_true.copy()
        return y_true, y_pred
    
    @pytest.fixture
    def constant_target_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Constant target with perfect and imperfect predictions."""
        y_true = np.full(10, 5.0)
        y_pred_perfect = y_true.copy()
        y_pred_imperfect = y_true + np.random.randn(10) * 0.1
        return y_true, y_pred_perfect, y_pred_imperfect
    
    @pytest.fixture 
    def linear_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Typical linear regression data with some noise."""
        np.random.seed(42)
        y_true = np.random.randn(100)
        # Add small noise to predictions (R² should be high but < 1.0)
        y_pred = y_true + np.random.randn(100) * 0.01
        return y_true, y_pred
    
    @pytest.fixture
    def worse_than_mean_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Predictions worse than mean (negative R²)."""
        np.random.seed(42)
        y_true = np.random.randn(100)
        # Deliberately bad predictions
        y_pred = y_true * (-2) + np.random.randn(100) * 0.5
        return y_true, y_pred
    
    @pytest.mark.parametrize("func_name, func", [
        ("mse", mse),
        ("rmse", rmse), 
        ("r2_score", r2_score)
    ])
    @pytest.mark.parametrize("data_type", [list, tuple, np.ndarray])
    def test_input_types_return_float(self, func_name: str, func, data_type):
        """Test that functions accept list/tuple/array and return float."""
        # Для кожного типу даних потрібен окремий підхід
        if data_type is list:
            y_true = [1.0, 2.0, 3.0]
            y_pred = [1.1, 1.9, 3.2]
        elif data_type is tuple:
            y_true = (1.0, 2.0, 3.0)
            y_pred = (1.1, 1.9, 3.2)
        else:  # np.ndarray
            y_true = np.array([1.0, 2.0, 3.0], dtype=float)
            y_pred = np.array([1.1, 1.9, 3.2], dtype=float)
        
        result = func(y_true, y_pred)
        
        assert isinstance(result, float), f"{func_name} should return float"
        assert not np.isnan(result), f"{func_name} result should not be NaN"
        assert np.isfinite(result), f"{func_name} result should be finite"
    
    def test_mse_perfect_match(self, perfect_match_data):
        """MSE should be 0.0 for perfect predictions."""
        y_true, y_pred = perfect_match_data
        result = mse(y_true, y_pred)
        
        assert result == pytest.approx(0.0), "MSE should be 0.0 for perfect predictions"
    
    def test_mse_manual_calculation(self):
        """Test MSE with manual calculation."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 1.9, 3.2])
        
        errors = y_true - y_pred
        expected = np.mean(errors ** 2)
        result = mse(y_true, y_pred)
        
        assert result == pytest.approx(expected), "MSE should match manual calculation"
    
    def test_rmse_relationship_to_mse(self, linear_data):
        """RMSE should be sqrt(MSE)."""
        y_true, y_pred = linear_data
        
        mse_result = mse(y_true, y_pred)
        rmse_result = rmse(y_true, y_pred)
        
        expected_rmse = np.sqrt(mse_result)
        assert rmse_result == pytest.approx(expected_rmse), "RMSE should equal sqrt(MSE)"
        
        # RMSE should be positive unless MSE is exactly 0
        if mse_result > 0:
            assert rmse_result > 0, "RMSE should be positive for non-zero MSE"
    
    def test_r2_perfect_match(self, perfect_match_data):
        """R² should be 1.0 for perfect predictions."""
        y_true, y_pred = perfect_match_data
        result = r2_score(y_true, y_pred)
        
        assert result == pytest.approx(1.0), "R² should be 1.0 for perfect predictions"

    def test_r2_constant_target_imperfect(self, constant_target_data):
        """R² should be 0.0 + warning for imperfect prediction of constant target."""
        y_true, _, y_pred_imperfect = constant_target_data
        
        with pytest.warns(UserWarning, match=r"R\^2 score is not well-defined"):
            result = r2_score(y_true, y_pred_imperfect)
        
        assert result == pytest.approx(0.0), "R² should be 0.0 for imperfect constant target prediction"
    
    def test_r2_better_than_mean(self, linear_data):
        """R² should be > 0.0 for predictions better than mean."""
        y_true, y_pred = linear_data
        
        result = r2_score(y_true, y_pred)
        
        assert result > 0.0, "R² should be positive for predictions better than mean"
        assert result < 1.0, "R² should be < 1.0 for imperfect predictions"
    
    def test_r2_worse_than_mean(self, worse_than_mean_data):
        """R² should be < 0.0 for predictions worse than mean."""
        y_true, y_pred = worse_than_mean_data
        
        result = r2_score(y_true, y_pred)
        
        assert result < 0.0, "R² should be negative for predictions worse than mean"
    
    @pytest.mark.parametrize("func_name, func", [
        ("mse", mse),
        ("rmse", rmse),
        ("r2_score", r2_score)
    ])
    def test_empty_input_raises_error(self, func_name: str, func):
        """Test that empty arrays raise ValueError."""
        y_true = np.array([])
        y_pred = np.array([])
        
        with pytest.raises(ValueError, match="must not be empty"):
            func(y_true, y_pred)
    
    @pytest.mark.parametrize("func_name, func", [
        ("mse", mse),
        ("rmse", rmse),
        ("r2_score", r2_score)
    ])
    def test_2d_input_raises_error(self, func_name: str, func):
        """Test that 2D arrays raise ValueError."""
        y_true = np.random.randn(10, 1)
        y_pred = np.random.randn(10, 1)
        
        with pytest.raises(ValueError, match="must be 1D"):
            func(y_true, y_pred)
    
    @pytest.mark.parametrize("func_name, func", [
        ("mse", mse),
        ("rmse", rmse),
        ("r2_score", r2_score)
    ])
    def test_length_mismatch_raises_error(self, func_name: str, func):
        """Test that length mismatch raises ValueError."""
        y_true = np.random.randn(10)
        y_pred = np.random.randn(5)
        
        with pytest.raises(ValueError, match="sample mismatch"):
            func(y_true, y_pred)
    
    @pytest.mark.parametrize("func_name, func", [
        ("mse", mse),
        ("rmse", rmse),
        ("r2_score", r2_score)
    ])
    def test_nan_input_raises_error(self, func_name: str, func):
        """Test that NaN values raise ValueError."""
        y_true = np.array([1.0, 2.0, np.nan, 4.0])
        y_pred = np.array([1.1, 1.9, 3.0, 4.1])
        
        with pytest.raises(ValueError, match="NaN or infinity"):
            func(y_true, y_pred)
    
    @pytest.mark.parametrize("func_name, func", [
        ("mse", mse),
        ("rmse", rmse),
        ("r2_score", r2_score)
    ])
    def test_inf_input_raises_error(self, func_name: str, func):
        """Test that infinity values raise ValueError."""
        y_true = np.array([1.0, 2.0, np.inf, 4.0])
        y_pred = np.array([1.1, 1.9, 3.0, 4.1])
        
        with pytest.raises(ValueError, match="NaN or infinity"):
            func(y_true, y_pred)
    
    def test_r2_math_properties(self):
        """Test mathematical properties of R² score."""
        np.random.seed(42)
        
        # Test that R² = 1 - SS_res/SS_tot
        y_true = np.random.randn(100)
        y_pred = y_true + np.random.randn(100) * 0.5
        
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        
        r2_expected = 1.0 - (ss_res / ss_tot)
        r2_actual = r2_score(y_true, y_pred)
        
        assert r2_actual == pytest.approx(r2_expected), "R² formula should match calculation"
    
    def test_edge_case_single_sample(self):
        """Test metrics with single sample edge case."""
        y_true = np.array([5.0])
        y_pred = np.array([5.1])
        
        # Should not raise for valid single sample
        mse_result = mse(y_true, y_pred)
        rmse_result = rmse(y_true, y_pred)
        
        assert mse_result == pytest.approx(0.01)  # (0.1)^2 = 0.01
        assert rmse_result == pytest.approx(0.1)  # sqrt(0.01) = 0.1
        
        # R² for single sample with different values
        # SS_tot = 0 when there's only one sample and it's different from prediction
        y_true_single = np.array([5.0])
        y_pred_single = np.array([6.0])
        
        with pytest.warns(UserWarning):
            r2_result = r2_score(y_true_single, y_pred_single)
        
        assert r2_result == pytest.approx(0.0), "R² = 0.0 for single sample imperfect prediction"
    
    def test_r2_constant_target_perfect(self, constant_target_data):
        """R² should be 1.0 for perfect prediction of constant target."""
        y_true, y_pred_perfect, _ = constant_target_data
        
        result = r2_score(y_true, y_pred_perfect)
        
        assert result == pytest.approx(1.0), "R² should be 1.0 for perfect constant target prediction"