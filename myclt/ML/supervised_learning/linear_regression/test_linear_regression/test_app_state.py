import numpy as np
import pytest
from typing import Any, Dict, List, Optional, Union

from myclt.ML.supervised_learning.linear_regression.app_state import AppState, rebuild_split, print_status


class TestAppStateValidation:
    """Test cases for AppState validation and initialization."""

    def test_default_state_is_valid(self):
        """Test that default AppState is valid and doesn't raise errors."""
        state = AppState()
        
        # Should not raise ValueError
        state.validate()
        
        # Check default values match expectation (from app_state.py default parameters)
        assert state.test_size == 0.2
        assert state.seed == 42
        assert state.use_scaling is True
        assert state.learning_rate == 0.05
        assert state.learning_rate_auto is True
        assert state.epochs == 10000
        assert state.early_stopping is True
        assert state.use_l1 is False
        assert state.use_l2 is False
        assert state.lambda_l1 == 0.01
        assert state.lambda_l2 == 0.01
        assert state.dataset is None
        assert state.prepareddata is None
        assert state.metrics == {}

    @pytest.mark.parametrize("learning_rate_auto_value, expected_bool", [
        (True, True),
        (False, False),
        (1, True),
        (0, False),
        (None, False),
        ("", False),
        ("x", True),
    ])
    def test_learning_rate_auto_normalized_to_bool(self, learning_rate_auto_value: Any, expected_bool: bool):
        """Test that __post_init__ normalizes learning_rate_auto to bool."""
        state = AppState(learning_rate_auto=learning_rate_auto_value)
        
        # Check it's a boolean and has correct value
        assert isinstance(state.learning_rate_auto, bool)
        assert state.learning_rate_auto == expected_bool

    @pytest.mark.parametrize("param_name, invalid_value, expected_error_part", [
        ("test_size", 0.0, "test_size must be in \\[0.05, 0.5\\]"),
        ("test_size", 0.6, "test_size must be in \\[0.05, 0.5\\]"),
        ("test_size", -0.1, "test_size must be in \\[0.05, 0.5\\]"),
        ("seed", True, "must be a non-negative integer"),
        ("seed", -1, "must be a non-negative integer"),
        ("seed", 1.5, "must be a non-negative integer"),
        ("seed", "not_a_number", "must be a non-negative integer"),
        ("learning_rate", 0.0, "learning_rate must be > 0"),
        ("learning_rate", -0.5, "learning_rate must be > 0"),
        ("epochs", 0, "epochs must be >= 1"),
        ("epochs", -10, "epochs must be >= 1"),
        ("lambda_l1", -0.1, "lambda_l1 must be >= 0"),
        ("lambda_l2", -0.1, "lambda_l2 must be >= 0"),
    ])
    def test_invalid_params_raise_value_error(self, param_name: str, invalid_value: Any, expected_error_part: str):
        """Test that invalid parameters raise ValueError with appropriate messages."""
        kwargs = {param_name: invalid_value}
        if param_name == "learning_rate":
            kwargs["learning_rate_auto"] = False
            
        # The exception should be raised during initialization since __post_init__ calls validate()
        with pytest.raises(ValueError, match=expected_error_part):
            AppState(**kwargs)

    def test_learning_rate_auto_bool_branch_in_validate(self):
        """Test that validate properly handles learning_rate_auto being bool in the check."""
        # When learning_rate_auto is True, learning_rate is not validated
        state = AppState(learning_rate_auto=True, learning_rate=-1.0)
        state.validate()  # Should not raise error when learning_rate_auto is True
        
        # When learning_rate_auto is False, learning_rate must be > 0
        # The exception should be raised during initialization since __post_init__ calls validate()
        with pytest.raises(ValueError, match="learning_rate must be > 0"):
            AppState(learning_rate_auto=False, learning_rate=0.0)

    def test_reset_metrics_clears_dict(self):
        """Test that reset_metrics clears all metrics."""
        state = AppState()
        state.metrics = {"mse": 0.1, "rmse": 0.3, "r2": 0.8}
        
        assert len(state.metrics) == 3
        state.reset_metrics()
        
        assert state.metrics == {}
        assert len(state.metrics) == 0


class TestRebuildSplit:
    """Test cases for rebuild_split function."""

    def test_raises_without_prepareddata(self):
        """Test that rebuild_split raises RuntimeError when prepareddata is None."""
        state = AppState()
        
        with pytest.raises(RuntimeError, match="!No features/target selection yet!"):
            rebuild_split(state)

    def test_with_scaling_scales_train_test(self, prepared_state):
        """Test rebuild_split with scaling enabled produces scaled data with mean≈0, std≈1."""
        state = prepared_state
        state.use_scaling = True
        
        rebuild_split(state)
        
        np.testing.assert_allclose(state.X_train.mean(axis=0), 0.0, atol=1e-10)
        np.testing.assert_allclose(state.X_train.std(axis=0), 1.0, atol=1e-10)
        
        assert state.scaler_mean is not None
        assert state.scaler_std is not None
        assert state.scaler_mean.shape == (3,)
        assert state.scaler_std.shape == (3,)
        
        np.testing.assert_allclose(state.X_test, (state.prepareddata.X[state.test_idx] - state.scaler_mean) / state.scaler_std)

    def test_without_scaling_keeps_raw_and_clears_scalers(self):
        """Test rebuild_split without scaling keeps raw data and clears scalers."""
        X = np.random.randn(100, 3)
        y = 2 * X[:, 0] + 1.5 * X[:, 1] - X[:, 2] + 3 + 0.1 * np.random.randn(100)
        
        from myclt.ML.supervised_learning.linear_regression.data import Prepareddata, Dataset
        prepareddata = Prepareddata(
            X=X,
            Y=y,
            feature_names=["feature_1", "feature_2", "feature_3"],
            target_name="target"
        )
        
        state = AppState(prepareddata=prepareddata, use_scaling=False)
        rebuild_split(state)
        
        assert state.scaler_mean is None
        assert state.scaler_std is None
        
        np.testing.assert_array_equal(state.X_train, X[state.train_idx])
        np.testing.assert_array_equal(state.X_test, X[state.test_idx])
        np.testing.assert_array_equal(state.y_train, y[state.train_idx])
        np.testing.assert_array_equal(state.y_test, y[state.test_idx])

    def test_resets_model_and_metrics_and_sets_indices(self, prepared_state):
        """Test rebuild_split resets model, metrics, and sets proper indices."""
        state = prepared_state
        
        from myclt.ML.supervised_learning.linear_regression.core import LinearRegressionGD
        state.model = LinearRegressionGD(learning_rate=0.01, epochs=10)
        state.model.w = np.ones(3)
        state.model.b = 1.0
        state.metrics = {"mse": 0.1, "rmse": 0.3}
        
        assert state.model is not None
        assert len(state.metrics) > 0
        
        rebuild_split(state)
        
        assert state.model is None
        assert state.metrics == {}
        
        assert state.train_idx is not None
        assert state.test_idx is not None
        
        n_samples = state.prepareddata.X.shape[0]
        train_set = set(state.train_idx)
        test_set = set(state.test_idx)
        all_indices = set(range(n_samples))
        
        assert train_set.isdisjoint(test_set)
        assert train_set.union(test_set) == all_indices


class TestPrintStatus:
    """Test cases for print_status function."""

    def test_prints_expected_sections_empty_state(self, capsys):
        """Test print_status shows all sections for empty state."""
        state = AppState()
        
        print_status(state)
        captured = capsys.readouterr()
        output = captured.out
        
        assert "Dataset:" in output
        assert "Selection:" in output
        assert "Split:" in output
        assert "Scaling:" in output
        assert "Model:" in output
        assert "Regularization:" in output
        assert "Metrics:" in output
        
        assert "none" in output
        assert f"test_size={state.test_size}" in output
        assert f"seed={state.seed}" in output
        assert "ON" in output
        assert "OFF" in output
        assert "NONE" in output

    def test_prints_metrics_when_present(self, prepared_state, capsys):
        """Test print_status shows metrics when they exist."""
        state = prepared_state
        state.metrics = {"mse": 0.1234, "rmse": 0.3512, "r2": 0.8567}
        
        print_status(state)
        captured = capsys.readouterr()
        output = captured.out
        
        assert "Metrics:" in output
        assert "0.1234" in output or ".1234" in output
        assert "0.35" in output
        assert "0.8567" in output or "0.857" in output

    def test_prints_regularization_when_enabled(self, capsys):
        """Test print_status shows regularization when enabled."""
        state = AppState(use_l1=True, lambda_l1=0.1, use_l2=True, lambda_l2=0.2)
        
        print_status(state)
        captured = capsys.readouterr()
        output = captured.out
        
        assert "Regularization:" in output
        assert "L1" in output
        assert "L2" in output
        assert "0.1" in output
        assert "0.2" in output