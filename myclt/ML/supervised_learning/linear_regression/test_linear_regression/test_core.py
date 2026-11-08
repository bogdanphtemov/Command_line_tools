import numpy as np
import pytest
import json
from typing import List, Tuple, Union

from myclt.ML.supervised_learning.linear_regression.core import LinearRegressionGD


class TestLinearRegressionGDCore:
    """Test cases for LinearRegressionGD - core functionality and validation."""

    @pytest.mark.parametrize("learning_rate, epochs, lambda_l1, lambda_l2, expected_error", [
        (-0.01, 100, 0.0, 0.0, "learning_rate must be > 0"),
        (0.01, 0, 0.0, 0.0, "epochs must be >= 1"),
        (0.01, 100, -0.1, 0.0, "lambda_l1 must be >= 0"),
        (0.01, 100, 0.0, -0.1, "lambda_l2 must be >= 0"),
    ])
    def test_constructor_validation(self, learning_rate: float, epochs: int, 
                                  lambda_l1: float, lambda_l2: float, expected_error: str):
        """Test that invalid constructor parameters raise ValueError."""
        with pytest.raises(ValueError, match=expected_error):
            LinearRegressionGD(
                learning_rate=learning_rate,
                epochs=epochs,
                lambda_l1=lambda_l1,
                lambda_l2=lambda_l2
            )

    @pytest.mark.parametrize("X_shape, y_shape, y_error_contains", [
        (np.random.randn(10, 2), np.random.randn(5), r"sample mismatch"),  # Different lengths
        (np.random.randn(10), np.random.randn(10), r"2D"),  # X not 2D
        (np.random.randn(10, 2), np.random.randn(10, 1), r"1D"),  # y not 1D
    ])
    def test_validate_inputs_shape_errors(self, X_shape: Union[np.ndarray, Tuple], 
                                         y_shape: Union[np.ndarray, Tuple], 
                                         y_error_contains: str):
        """Test _validate_inputs through fit method for shape validation."""
        if isinstance(X_shape, tuple):
            X = np.random.randn(*X_shape)
        else:
            X = X_shape
            
        if isinstance(y_shape, tuple):
            y = np.random.randn(*y_shape)
        else:
            y = y_shape
            
        model = LinearRegressionGD(learning_rate=0.01, epochs=100)
        
        with pytest.raises(ValueError, match=y_error_contains):
            # The validation happens inside fit, which calls _validate_inputs
            model.fit(X, y)

    @pytest.mark.parametrize("X_has_nan_inf, y_has_nan_inf, expected_error", [
        (True, False, r"NaN or infinity"),
        (False, True, r"NaN or infinity"),
        (True, True, r"NaN or infinity"),
    ])
    def test_validate_inputs_nan_inf_errors(self, X_has_nan_inf: bool, y_has_nan_inf: bool, expected_error: str):
        """Test _validate_inputs through fit method for NaN/inf validation."""
        X = np.random.randn(10, 2)
        y = np.random.randn(10)
        
        if X_has_nan_inf:
            X[0, 0] = np.nan
            
        if y_has_nan_inf:
            y[0] = np.inf
            
        model = LinearRegressionGD(learning_rate=0.01, epochs=100)
        
        with pytest.raises(ValueError, match=expected_error):
            model.fit(X, y)

    def test_fit_basic_properties(self, linear_2d):
        """Test that fit produces correct model state."""
        X, y = linear_2d
        n_samples, n_features = X.shape
        
        model = LinearRegressionGD(learning_rate=0.01, epochs=100)
        model.fit(X, y)
        
        assert model.w is not None, "Weights should be initialized after fit"
        assert model.w.shape == (n_features,), f"w.shape should be ({n_features},), got {model.w.shape}"
        assert np.isfinite(model.b), "Bias should be finite"
        assert len(model.loss_history) == 100, f"loss_history should have {model.epochs} entries"
        assert model.is_trained, "Model should be marked as trained"
        
        # Loss should generally decrease (but allow for small fluctuations)
        first_loss = model.loss_history[0]
        last_loss = model.loss_history[-1]
        assert last_loss <= first_loss * 1.1, f"Loss should decrease or be stable, got {first_loss}->{last_loss}"

    def test_fit_perfect_linear_convergence(self, perfect_linear):
        """Test that model approaches true parameters on perfect linear data."""
        X, y = perfect_linear
        
        model = LinearRegressionGD(learning_rate=0.01, epochs=500)
        model.fit(X, y)
        
        # True parameters are w=2.0, b=1.0 for y = 2x + 1
        np.testing.assert_allclose(model.w, [2.0], atol=0.1, 
                                 err_msg=f"w should be close to 2.0, got {model.w}")
        assert model.b == pytest.approx(1.0, abs=0.1), f"b should be close to 1.0, got {model.b}"

    def test_gradient_descent_logic(self, tiny_xy):
        """Test that a single GD step changes weights in the correct direction."""
        X, y = tiny_xy
        
        model = LinearRegressionGD(learning_rate=0.001, epochs=1)
        init_w = np.zeros(X.shape[1])
        model.w = init_w.copy()
        model.b = 0.0
        
        # Compute gradients manually
        y_pred = X @ model.w + model.b
        errors = y_pred - y
        learning_rate = model.learning_rate
        n_samples = X.shape[0]
        
        expected_dw = (2.0 / n_samples) * (X.T @ errors)
        expected_db = (2.0 / n_samples) * np.sum(errors)
        
        expected_new_w = init_w - learning_rate * expected_dw
        expected_new_b = -learning_rate * expected_db
        
        # Apply one step 
        model._apply_gradient(X, errors, n_samples)
        
        # Weights should move towards optimal solution
        np.testing.assert_allclose(model.w, expected_new_w, rtol=1e-10,
                                 err_msg="Gradient descent step incorrect for w")
        assert model.b == pytest.approx(expected_new_b, abs=1e-10), \
               f"Gradient descent step incorrect for b"

    @pytest.mark.parametrize("lambda_l2", [0.1, 1.0])
    def test_regularization_l2_effect(self, linear_2d, lambda_l2: float):
        """Test that L2 regularization reduces weight magnitudes."""
        X, y = linear_2d
        
        # Fit without regularization
        model_no_reg = LinearRegressionGD(learning_rate=0.01, epochs=100, lambda_l2=0.0)
        model_no_reg.fit(X, y)
        
        # Fit with regularization
        model_reg = LinearRegressionGD(learning_rate=0.01, epochs=100, lambda_l2=lambda_l2)
        model_reg.fit(X, y)
        
        # Regularized model should have smaller weights (in magnitude)
        weight_norm_no_reg = np.linalg.norm(model_no_reg.w)
        weight_norm_reg = np.linalg.norm(model_reg.w)
        
        assert weight_norm_reg < weight_norm_no_reg, \
            f"L2 regularization should reduce weight norm: {weight_norm_reg} < {weight_norm_no_reg}"

    @pytest.mark.parametrize("lambda_l1", [0.1])
    def test_regularization_l1_loss(self, tiny_xy, lambda_l1: float):
        """Test that L1 regularization is included in loss calculation."""
        X, y = tiny_xy
        
        model = LinearRegressionGD(learning_rate=0.01, epochs=10, lambda_l1=lambda_l1)
        model.w = np.ones(X.shape[1])  # Set initial weights
        model.b = 1.0
        
        # Compute loss manually
        errors, manual_loss = model._forward(X, y)
        
        # Loss should include L1 penalty: lambda_l1 * sum(|w|)
        mse_term = np.mean(errors ** 2)
        l1_penalty = lambda_l1 * np.sum(np.abs(model.w))
        expected_loss = mse_term + l1_penalty
        
        assert manual_loss == pytest.approx(expected_loss, rel=1e-10), \
            f"Loss calculation incorrect: {manual_loss} vs {expected_loss}"

    def test_is_trained_property(self, linear_2d):
        """Test is_trained property in various states."""
        # Before training
        model = LinearRegressionGD(learning_rate=0.01, epochs=100)
        assert not model.is_trained, "is_trained should be False before training"
        
        # After successful training
        X, y = linear_2d
        model.fit(X, y)
        assert model.is_trained, "is_trained should be True after successful training"
        
        # Test divergence to non-finite weights (simulated)
        model.w = np.array([np.nan, np.inf])
        model.b = 1.0
        assert not model.is_trained, "is_trained should be False for non-finite weights"

    def test_predict_error_cases(self, linear_2d):
        """Test predict method error conditions."""
        X, y = linear_2d
        model = LinearRegressionGD(learning_rate=0.01, epochs=100)
        
        # Predict before training
        with pytest.raises(RuntimeError, match="!Model is not trained yet!"):
            model.predict(X)
        
        # Train model
        model.fit(X, y)
        
        # Predict with non-finite weights (simulate divergence)
        model.w = np.array([np.inf, 1.0])
        with pytest.raises(RuntimeError, match="!Model weights are not finite"):
            model.predict(X)
        
        # Reset to valid weights
        model.w = np.ones(X.shape[1])
        
        # Predict with wrong feature count
        X_wrong = np.random.randn(5, 3)  # 3 features instead of 2
        with pytest.raises(ValueError, match=r"X must be 2D with 2 features"):
            model.predict(X_wrong)
        
        # Predict with NaN/Inf in X
        X_nan = X.copy()
        X_nan[0, 0] = np.nan
        with pytest.raises(ValueError, match="X contains NaN or infinity"):
            model.predict(X_nan)

    def test_predict_correctness(self, linear_2d):
        """Test that predict produces correct outputs."""
        X, y = linear_2d
        
        model = LinearRegressionGD(learning_rate=0.01, epochs=100)
        model.fit(X, y)
        
        predictions = model.predict(X)
        
        # Check shapes and types
        assert predictions.shape == (X.shape[0],), "Predictions should have same length as X"
        assert np.all(np.isfinite(predictions)), "All predictions should be finite"
        
        # Should be approximately X @ w + b
        expected = X @ model.w + model.b
        np.testing.assert_allclose(predictions, expected, rtol=1e-10,
                                 err_msg="Predictions don't match manual calculation")

    def test_get_params_set_params_roundtrip(self, linear_2d):
        """Test parameter serialization and deserialization."""
        X, y = linear_2d
        
        # Original model
        original = LinearRegressionGD(learning_rate=0.01, epochs=100, lambda_l1=0.1, lambda_l2=0.2)
        original.fit(X, y)
        
        # Get parameters
        params = original.get_params()
        
        # Check JSON compatibility
        json_str = json.dumps(params)
        params_restored = json.loads(json_str)
        
        # Create new model from params
        restored = LinearRegressionGD()
        restored.set_params(params_restored)
        
        # Should have same parameters and produce same predictions
        np.testing.assert_allclose(original.w, restored.w, rtol=1e-10)
        assert original.b == pytest.approx(restored.b, rel=1e-10)
        assert original.learning_rate == restored.learning_rate
        assert original.epochs == restored.epochs
        assert original.lambda_l1 == restored.lambda_l1
        assert original.lambda_l2 == restored.lambda_l2
        
        # Predictions should match
        pred_original = original.predict(X)
        pred_restored = restored.predict(X)
        np.testing.assert_allclose(pred_original, pred_restored, rtol=1e-10)

    def test_set_params_to_untrained_state(self):
        """Test that set_params with w=None resets to untrained state."""
        model = LinearRegressionGD()
        
        # Set parameters with w=None (untrained state)
        params = {
            "w": None,
            "b": 0.0,
            "learning_rate": 0.01,
            "epochs": 100,
            "lambda_l1": 0.0,
            "lambda_l2": 0.0,
            "loss_history": []
        }
        
        model.set_params(params)
        
        assert model.w is None, "w should be None in untrained state"
        assert not model.is_trained, "Model should not be trained"
        assert model.loss_history == [], "Loss history should be empty"

    def test_set_params_invalid_hyperparams(self):
        """Test that set_params validates hyperparameters."""
        model = LinearRegressionGD()
        
        # Try to set invalid hyperparameters
        invalid_params = {
            "w": [1.0],
            "b": 0.0,
            "learning_rate": -0.01,  # Invalid
            "epochs": 100,
            "lambda_l1": 0.0,
            "lambda_l2": 0.0
        }
        
        with pytest.raises(ValueError, match="learning_rate must be > 0"):
            model.set_params(invalid_params)

    def test_loss_history_tracks_training(self, tiny_xy):
        """Test that loss history properly tracks training progress."""
        X, y = tiny_xy
        epochs = 5
        
        model = LinearRegressionGD(learning_rate=0.01, epochs=epochs)
        model.fit(X, y)
        
        # Check loss history properties
        assert len(model.loss_history) == epochs, \
            f"loss_history should have {epochs} entries"
        assert np.all(np.isfinite(model.loss_history)), \
            "All loss values should be finite"
        
        # Loss should generally be positive (or very small for perfect fit)
        assert np.all(np.array(model.loss_history) >= -1e-10), \
            "Loss values should be non-negative (MSE)" 

    def test_forward_pass_calculation(self, linear_2d):
        """Test that _forward computes predictions and loss correctly."""
        X, y = linear_2d
        
        model = LinearRegressionGD(learning_rate=0.01, epochs=10, lambda_l1=0.1, lambda_l2=0.1)
        
        # Use small number of epochs for faster test
        model.fit(X, y)
        
        # Get errors and loss from forward pass
        errors, total_loss = model._forward(X, y)
        
        # Manual calculation
        y_pred = X @ model.w + model.b
        mse_loss = np.mean((y_pred - y) ** 2)
        l1_penalty = model.lambda_l1 * np.sum(np.abs(model.w))
        l2_penalty = (model.lambda_l2 / 2) * np.sum(model.w ** 2)
        expected_loss = mse_loss + l1_penalty + l2_penalty
        expected_errors = y_pred - y
        
        np.testing.assert_allclose(errors, expected_errors, rtol=1e-10)
        assert total_loss == pytest.approx(expected_loss, rel=1e-10), \
            f"Loss calculation incorrect: {total_loss} vs {expected_loss}"
        
        # Note: The loss_history tracks training loss during each epoch, 
        # but the last entry may not match the current forward pass due to 
        # regularization or slight numerical differences
        # Let's just check that loss_history is consistent
        assert len(model.loss_history) == 10, f"Should have {model.epochs} loss entries"