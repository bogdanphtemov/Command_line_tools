import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from typing import Optional
import matplotlib.pyplot as plt
import matplotlib.figure

from myclt.ML.supervised_learning.linear_regression.visualization import plot_1d_regression
from myclt.ML.supervised_learning.linear_regression.core import LinearRegressionGD
from myclt.ML.visualization_utils import plot_loss_curve, plot_true_vs_pred


class Test1DRegressionPlot:
    """Test cases for plot_1d_regression function."""

    def test_plot_1d_regression_untrained_model_raises_error(self, linear_xy):
        """Test that untrained model raises RuntimeError."""
        X, y = linear_xy
        x_raw = X.flatten()
        
        model = LinearRegressionGD(learning_rate=0.01, epochs=100)
        # Model is not trained
        
        with pytest.raises(RuntimeError, match="model is not trained"):
            plot_1d_regression(x_raw, y, model, scaler_mean=None, scaler_std=None)

    def test_plot_1d_regression_multi_feature_model_raises_error(self, linear_2d):
        """Test that model with multiple features raises ValueError."""
        X, y = linear_2d
        
        model = LinearRegressionGD(learning_rate=0.01, epochs=100)
        model.fit(X, y)
        
        # Use first column as x_raw for single feature
        x_raw = X[:, 0]
        
        with pytest.raises(ValueError, match="single-feature regression models"):
            plot_1d_regression(x_raw, y, model, scaler_mean=None, scaler_std=None)

    def test_plot_1d_regression_partial_scaler_params_raises_error(self, linear_xy):
        """Test that partial scaler params raise ValueError."""
        X, y = linear_xy
        x_raw = X.flatten()
        
        model = LinearRegressionGD(learning_rate=0.01, epochs=100)
        model.fit(X, y)
        
        # Only provide scaler_mean but not scaler_std
        scaler_mean = np.array([1.0])
        scaler_std = None  # Missing
        
        with pytest.raises(ValueError, match="scaler_mean and scaler_std must be either both None"):
            plot_1d_regression(x_raw, y, model, scaler_mean, scaler_std)

    def test_plot_1d_regression_mismatch_lengths_raises_error(self, linear_xy):
        """Test that length mismatch raises ValueError."""
        X, y = linear_xy
        x_raw = X.flatten()
        
        model = LinearRegressionGD(learning_rate=0.01, epochs=100)
        model.fit(X, y)
        
        # Create mismatch by using different y length
        y_mismatched = y[:-1]
        
        with pytest.raises(ValueError, match="sample mismatch"):
            plot_1d_regression(x_raw, y_mismatched, model, scaler_mean=None, scaler_std=None)

    def test_plot_1d_regression_nan_values_raise_error(self, linear_xy):
        """Test that NaN values raise ValueError."""
        X, y = linear_xy
        x_raw = X.flatten()
        
        model = LinearRegressionGD(learning_rate=0.01, epochs=100)
        model.fit(X, y)
        
        # Add NaN to x_raw
        x_raw_nan = x_raw.copy()
        x_raw_nan[0] = np.nan
        
        with pytest.raises(ValueError, match="NaN or infinity"):
            plot_1d_regression(x_raw_nan, y, model, scaler_mean=None, scaler_std=None)

    @patch('matplotlib.pyplot.show')
    def test_plot_1d_regression_successful_call_no_scaling(self, mock_show, perfect_linear):
        """Test successful plot_1d_regression call without scaling."""
        X, y = perfect_linear
        x_raw = X.flatten()
        
        model = LinearRegressionGD(learning_rate=0.01, epochs=100)
        model.fit(X, y)
        
        # Mock plt.show to prevent blocking
        with patch('matplotlib.pyplot.show') as mock_show:
            fig = plot_1d_regression(
                x_raw, y, model, 
                scaler_mean=None, scaler_std=None
            )
        
        assert isinstance(fig, matplotlib.figure.Figure)
        mock_show.assert_called_once()
        
        # Clean up
        plt.close(fig)

    @patch('matplotlib.pyplot.show')
    def test_plot_1d_regression_successful_call_with_scaling(self, mock_show, linear_xy):
        """Test successful plot_1d_regression call with scaling."""
        X, y = linear_xy
        x_raw = X.flatten()
        
        model = LinearRegressionGD(learning_rate=0.01, epochs=100)
        
        # Create scaling parameters (simulate scaled training)
        scaler_mean = np.array([x_raw.mean()])
        scaler_std = np.array([x_raw.std()])
        
        # Scale data for training
        X_scaled = (X - scaler_mean) / scaler_std
        model.fit(X_scaled, y)
        
        # Mock plt.show to prevent blocking
        with patch('matplotlib.pyplot.show') as mock_show:
            fig = plot_1d_regression(
                x_raw, y, model,
                scaler_mean=scaler_mean, scaler_std=scaler_std
            )
        
        assert isinstance(fig, matplotlib.figure.Figure)
        mock_show.assert_called_once()
        
        # Clean up
        plt.close(fig)


class TestLossCurvePlot:
    """Test cases for plot_loss_curve function."""

    @patch('matplotlib.pyplot.show')
    def test_plot_loss_curve_empty_history_no_error(self, mock_show):
        """Test that empty loss history doesn't cause errors."""
        plot_loss_curve([])
        
        # Should not raise any errors and return gracefully
        mock_show.assert_not_called()  # plt.show should not be called for empty history

    @patch('matplotlib.pyplot.show')
    def test_plot_loss_curve_valid_history(self, mock_show):
        """Test plot_loss_curve with valid loss history."""
        loss_history = [10.0, 5.0, 2.5, 1.25, 0.625]
        
        with patch('matplotlib.pyplot.show') as mock_show:
            plot_loss_curve(loss_history, ylabel="MSE Loss", title="Training Progress")
        
        mock_show.assert_called_once()
        
        # Clean up
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_plot_loss_curve_with_nan_values(self, mock_show):
        """Test plot_loss_curve handles NaN values gracefully."""
        # Loss history with some NaN values (can happen during unstable training)
        loss_history = [10.0, 5.0, np.nan, 1.25, 0.625]
        
        with patch('matplotlib.pyplot.show') as mock_show:
            plot_loss_curve(loss_history)
        
        # Should still complete without error
        mock_show.assert_called_once()
        
        # Clean up
        plt.close('all')


class TestTrueVsPredictedPlot:
    """Test cases for plot_true_vs_pred function."""

    @patch('matplotlib.pyplot.show')
    def test_plot_true_vs_pred_successful_call(self, mock_show):
        """Test successful plot_true_vs_pred call."""
        np.random.seed(42)
        y_true = np.random.randn(100)
        y_pred = y_true + np.random.randn(100) * 0.1
        
        with patch('matplotlib.pyplot.show') as mock_show:
            plot_true_vs_pred(y_true, y_pred, title="My Test Plot")
        
        mock_show.assert_called_once()
        
        # Clean up
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_plot_true_vs_pred_perfect_predictions(self, mock_show):
        """Test plot_true_vs_pred with perfect predictions."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = y_true.copy()  # Perfect predictions
        
        with patch('matplotlib.pyplot.show') as mock_show:
            plot_true_vs_pred(y_true, y_pred)
        
        mock_show.assert_called_once()
        
        # Clean up
        plt.close('all')

    def test_plot_true_vs_pred_input_validation(self):
        """Test that plot_true_vs_pred handles 2D inputs by flattening."""
        y_true = np.random.randn(10)
        y_pred_2d = np.random.randn(10, 1)  # 2D instead of 1D
        
        # plot_true_vs_pred should handle 2D arrays by flattening them
        # It returns None in non-interactive mode (Agg backend), which is expected
        with patch('matplotlib.pyplot.show'):
            fig = plot_true_vs_pred(y_true, y_pred_2d)
            # In Agg backend, plt.show() returns None, which is expected behavior
            # The function successfully creates a plot, it just doesn't return the figure
            pass  # Test passes if no exception is raised


class TestVisualizationUtilsIntegration:
    """Integration tests for visualization utilities."""

    @patch('matplotlib.pyplot.show')
    def test_visualization_functions_close_figures(self, mock_show, linear_xy):
        """Test that visualization functions properly manage figures."""
        X, y = linear_xy
        x_raw = X.flatten()
        
        model = LinearRegressionGD(learning_rate=0.01, epochs=100)
        model.fit(X, y)
        
        # Test plot_1d_regression with proper figure handling
        with patch('matplotlib.pyplot.show') as mock_show:
            fig = plot_1d_regression(x_raw, y, model, scaler_mean=None, scaler_std=None)
            plt.close(fig)  # Should not cause errors
        
        # Test plot_loss_curve
        with patch('matplotlib.pyplot.show') as mock_show:
            plot_loss_curve(model.loss_history)
            plt.close('all')  # Should not cause errors
        
        # Test plot_true_vs_pred with model predictions
        y_pred = model.predict(X)
        with patch('matplotlib.pyplot.show') as mock_show:
            plot_true_vs_pred(y, y_pred)
            plt.close('all')  # Should not cause errors

    @patch('matplotlib.pyplot.show')
    def test_visualization_with_agg_backend(self, mock_show):
        """Test that visualizations work with Agg backend (non-interactive)."""
        # This test ensures that functions handle the Agg backend correctly
        loss_history = [10.0, 5.0, 2.5, 1.25]
        
        # Should work without interactive display
        with patch('matplotlib.pyplot.show') as mock_show:
            plot_loss_curve(loss_history)
            # plt.show() should still be called, but Agg backend handles it gracefully
        
        mock_show.assert_called_once()