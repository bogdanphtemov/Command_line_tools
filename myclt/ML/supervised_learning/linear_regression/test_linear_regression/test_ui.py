import pytest
import numpy as np
from unittest.mock import patch, MagicMock, call
from pathlib import Path
import tempfile

from myclt.ML.supervised_learning.linear_regression.app_state import AppState
from myclt.ML.supervised_learning.linear_regression.core import LinearRegressionGD
from myclt.ML.supervised_learning.linear_regression.data import Dataset, Prepareddata
from myclt.ML.supervised_learning.linear_regression.visualization import plot_1d_regression
from myclt.ML.visualization_utils import plot_loss_curve, plot_true_vs_pred
from myclt.ML.supervised_learning.linear_regression.test_linear_regression.helpers import UIDriver, make_app_state
from myclt.ML.supervised_learning.linear_regression.ui import (
    menu_data, menu_train, menu_evaluate, menu_predict, menu_visualize, menu_save_load
)
from myclt.ML.supervised_learning.linear_regression.cli import main as cli_main


class TestUIDataMenu:
    """Test cases for menu_data function."""

    def test_menu_data_load_csv_and_back(self, monkeypatch, tmp_path):
        """Test loading CSV dataset and navigating back."""
        # Create test CSV file
        csv_content = """ad_budget_usd,monthly_revenue_usd
                         7803.53,39742.63
                         19038.93,91046.47
                         14773.88,71316.1"""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text(csv_content)
        
        driver = (UIDriver()
                 .choose([0, 5])  # Load CSV -> Back
                 .type([str(csv_path)])  # CSV path
                 .answer([""])  # Continue after success
                 )
        
        state = AppState()
        output = driver.run(menu_data, state, monkeypatch)
        
        assert state.dataset is not None
        assert state.dataset.data.shape == (3, 2)
        assert "Dataset loaded successfully" in output

    def test_menu_data_load_csv_error_no_file(self, monkeypatch):
        """Test error handling when CSV file doesn't exist."""
        driver = (UIDriver()
                 .choose([0, 5])  # Load CSV -> Back (after error)
                 .type(["/nonexistent/file.csv"])  # Invalid path
                 .answer([""])  # Continue after error
                 )
        
        state = AppState()
        output = driver.run(menu_data, state, monkeypatch)
        
        assert "Error" in output

    def test_menu_data_manual_input_and_back(self, monkeypatch):
        """Test manual input dataset entry and navigation."""
        driver = (UIDriver()
                 .choose([1, 5])  # Manual input -> Back
                 .type([
                     "Feature1;Target",
                     "1.0;10.0",
                     "2.0;20.0", 
                     "3.0;30.0",
                     "4.0;40.0",
                     "5.0;50.0",
                     ""
                 ])
                 .answer([""])  # Continue after success
                 )
        
        state = AppState()
        output = driver.run(menu_data, state, monkeypatch)
        
        assert state.dataset is not None
        assert state.dataset.data.shape == (5, 2)
        assert "Dataset loaded successfully" in output


class TestUITrainMenu:
    """Test cases for menu_train function."""

    def test_menu_train_configure_hyperparams_and_back(self, monkeypatch):
        """Test hyperparameter configuration."""
        state = make_app_state(n_features=2, n_samples=20)
        
        driver = (UIDriver()
                 .choose([0, 4, 4])  # Configure hyperparams -> Back -> Back (needs extra Back to exit menu)
                 .answer(["yes", "no"])  # Enable scaling, no L1/L2  # FIX: bool -> string
                 .choose([0, 0])  # Manual learning rate -> Back (twice needed for this menu flow)
                 .give_lr([0.01])  # Learning rate
                 .give_int([1000, 10000])  # Epochs, Max epochs (two ask_int calls)
                 .answer([""])  # Continue
                 )
        
        output = driver.run(menu_train, state, monkeypatch)
        
        assert state.learning_rate == 0.01
        assert state.epochs == 1000
        assert state.use_scaling is True

    def test_menu_train_configure_regularization_and_back(self, monkeypatch):
        """Test regularization configuration."""
        state = make_app_state(n_features=2, n_samples=20)
        
        driver = (UIDriver()
                 .choose([1, 4])  # Configure regularization -> Back
                 .answer(["yes", "yes"])  # Enable both L1 and L2
                 .give_float([0.1, 0.05])  # L1=0.1, L2=0.05
                 .answer([""])  # Continue
                 )
        
        output = driver.run(menu_train, state, monkeypatch)
        output = driver.run(menu_train, state, monkeypatch)
        
        assert state.use_l1 is True
        assert state.lambda_l1 == 0.1
        assert state.use_l2 is True
        assert state.lambda_l2 == 0.05

    def test_menu_train_auto_tune_regularization_error_no_data(self, monkeypatch):
        """Test grid search error when no data prepared."""
        state = AppState()  # Empty state
        
        driver = (UIDriver()
                 .choose([2, 4])  # Auto-tune regularization -> Back
                 )
        
        output = driver.run(menu_train, state, monkeypatch)
        
        assert "Select features + target first" in output

    @pytest.mark.slow
    def test_menu_train_grid_search_with_data(self, monkeypatch):
        """Test grid search workflow with prepared data (slow test)."""
        state = make_app_state(n_features=2, n_samples=30)  # Small dataset
        
        driver = (UIDriver()
                 .choose([2, 4])  # Auto-tune regularization -> Back
                 .answer(["yes", "yes", "yes"])  # Search both L1 and L2, apply results
                 .give_int([5, 2])  # Patience 5, epochs 2 for speed
                 )
        
        # Mock grid_search_regularization to avoid long computation
        with patch("myclt.ML.supervised_learning.linear_regression.ui.grid_search_regularization") as mock_grid:
            mock_grid.return_value = {
                'best_lambda_l1': 0.01,
                'best_lambda_l2': 0.001,
                'best_mse': 0.5,
                'results': [{'l1': 0.01, 'l2': 0.001, 'mean_mse': 0.5, 'std_mse': 0.1}]
            }
            
            output = driver.run(menu_train, state, monkeypatch)
        
        mock_grid.assert_called_once()

    def test_menu_train_successful_auto_lr_training(self, monkeypatch):
        """Test training with automatic learning rate tuning."""
        state = make_app_state(n_features=2, n_samples=20)
        state.learning_rate_auto = True
        
        driver = (UIDriver()
                 .choose([3, 4])  # Train model -> Back
                 .give_int([10])  # Patience
                 .answer([""])  # Continue after training
                 )
        
        # Mock auto_tune_learning_rate for speed
        with patch("myclt.ML.supervised_learning.linear_regression.ui.auto_tune_learning_rate") as mock_auto:
            mock_auto.return_value = {'best_lr': 0.01, 'best_mse': 0.5, 'results': [], 'phase2_results': None}
            
            # Mock fit_with_early_stopping
            with patch.object(LinearRegressionGD, 'fit_with_early_stopping') as mock_fit:
                output = driver.run(menu_train, state, monkeypatch)
        
        assert state.model is not None
        assert "Training finished" in output

    def test_menu_train_error_insufficient_samples(self, monkeypatch):
        """Test error when insufficient training samples."""
        state = make_app_state(n_features=2, n_samples=3)  # Too few samples
        
        driver = (UIDriver()
                 .choose([3, 4])  # Train model -> Back
                 )
        
        output = driver.run(menu_train, state, monkeypatch)
        
        assert "Too few training samples" in output

    def test_menu_train_no_data_error(self, monkeypatch):
        """Test error when no data is prepared."""
        state = AppState()  # Empty state
        
        driver = (UIDriver()
                 .choose([3, 4])  # Train model -> Back
                 )
        
        output = driver.run(menu_train, state, monkeypatch)
        
        assert "Select features + target first" in output
        


class TestUIEvaluateMenu:
    """Test cases for menu_evaluate function."""

    def test_menu_evaluate_no_model_error(self, monkeypatch):
        """Test error when no trained model exists."""
        state = AppState()
        
        driver = (UIDriver()
                 .choose([0, 2])  # Evaluate -> Back
                 )
        
        output = driver.run(menu_evaluate, state, monkeypatch)
        
        assert "trained model" in output or "Train" in output

    def test_menu_evaluate_no_test_set_error(self, monkeypatch):
        """Test error when no test set exists."""
        state = make_app_state(n_features=2, n_samples=20)
        state.model = LinearRegressionGD(learning_rate=0.01, epochs=10)
        state.X_test = None  # No test set
        
        driver = (UIDriver()
                 .choose([0, 2])  # Evaluate -> Back
                 )
        
        output = driver.run(menu_evaluate, state, monkeypatch)
        
        assert "test set" in output

    def test_menu_evaluate_success(self, monkeypatch):
        """Test successful evaluation."""
        state = make_app_state(n_features=2, n_samples=20)
        state.model = LinearRegressionGD(learning_rate=0.01, epochs=10)
        state.model.w = np.array([1.0, 2.0])
        state.model.b = 3.0
        
        driver = (UIDriver()
                 .choose([0, 2])  # Evaluate -> Back
                 .answer([""])  # Continue
                 )
        
        # Mock calculate_metrics
        with patch("myclt.ML.supervised_learning.linear_regression.ui.calculate_metrics") as mock_calc:
            mock_calc.return_value = {"mse": 0.5, "rmse": 0.707, "r2": 0.8}
            
            output = driver.run(menu_evaluate, state, monkeypatch)
        
        mock_calc.assert_called_once()
        assert "MSE" in output or "R²" in output

    def test_menu_evaluate_feature_importance_no_model(self, monkeypatch):
        """Test feature importance error when no model."""
        state = AppState()
        
        driver = (UIDriver()
                 .choose([1, 2])  # Feature importance -> Back
                 )
        
        output = driver.run(menu_evaluate, state, monkeypatch)
        
        assert "trained model" in output

    def test_menu_evaluate_feature_importance_success(self, monkeypatch):
        """Test successful feature importance calculation."""
        state = make_app_state(n_features=3, n_samples=20)
        state.model = LinearRegressionGD(learning_rate=0.01, epochs=10)
        state.model.w = np.array([1.0, 2.0, 0.5])
        
        driver = (UIDriver()
                 .choose([1, 2])  # Feature importance -> Back
                 .answer([""])  # Continue
                 )
        
        output = driver.run(menu_evaluate, state, monkeypatch)
        
        assert "Feature importance" in output


class TestUIVisualizeMenu:
    """Test cases for menu_visualize function."""

    def test_menu_visualize_loss_curve_no_model(self, monkeypatch):
        """Test loss curve plot error when no model."""
        state = AppState()
        
        driver = (UIDriver()
                 .choose([0, 3])  # Plot loss -> Back
                 )
        
        output = driver.run(menu_visualize, state, monkeypatch)
        
        assert "Train model first" in output

    def test_menu_visualize_loss_curve_success(self, monkeypatch):
        """Test successful loss curve plotting."""
        state = make_app_state(n_features=2, n_samples=20)
        state.model = LinearRegressionGD(learning_rate=0.01, epochs=10)
        state.model.loss_history = [10.0, 5.0, 2.5, 1.25]
        
        driver = (UIDriver()
                 .choose([0, 3])  # Plot loss -> Back
                 .answer([""])  # Continue
                 )
        
        with patch("myclt.ML.visualization_utils.plot_loss_curve") as mock_plot:
            output = driver.run(menu_visualize, state, monkeypatch)
        
        mock_plot.assert_called_once_with(state.model.loss_history)

    def test_menu_visualize_true_vs_pred_no_test_set(self, monkeypatch):
        """Test true vs predicted plot error when no test set."""
        state = make_app_state(n_features=2, n_samples=20)
        state.model = LinearRegressionGD(learning_rate=0.01, epochs=10)
        state.X_test = None  # No test set
        
        driver = (UIDriver()
                 .choose([1, 3])  # True vs Pred -> Back
                 )
        
        output = driver.run(menu_visualize, state, monkeypatch)
        
        assert "test set" in output

    def test_menu_visualize_true_vs_pred_success(self, monkeypatch):
        """Test successful true vs predicted plot."""
        state = make_app_state(n_features=2, n_samples=20)
        state.model = LinearRegressionGD(learning_rate=0.01, epochs=10)
        state.model.w = np.array([1.0, 2.0])
        state.model.b = 3.0
        
        driver = (UIDriver()
                 .choose([1, 3])  # True vs Pred -> Back
                 .answer([""])  # Continue
                 )
        
        with patch("myclt.ML.visualization_utils.plot_true_vs_pred") as mock_plot:
            output = driver.run(menu_visualize, state, monkeypatch)
        
        mock_plot.assert_called_once()

    def test_menu_visualize_1d_regression_wrong_feature_count(self, monkeypatch):
        """Test 1D regression plot error with multiple features."""
        state = make_app_state(n_features=2, n_samples=20)  # 2 features, not 1
        state.model = LinearRegressionGD(learning_rate=0.01, epochs=10)
        
        driver = (UIDriver()
                 .choose([2, 3])  # 1D regression -> Back
                 )
        
        output = driver.run(menu_visualize, state, monkeypatch)
        
        assert "exactly 1 feature" in output

    def test_menu_visualize_1d_regression_success(self, monkeypatch):
        """Test successful 1D regression plot."""
        # Create 1-feature dataset
        state = make_app_state(n_features=1, n_samples=20)
        state.model = LinearRegressionGD(learning_rate=0.01, epochs=10)
        state.model.w = np.array([1.0])
        state.model.b = 3.0
        
        driver = (UIDriver()
                 .choose([2, 3])  # 1D regression -> Back
                 .answer([""])  # Continue
                 )
        
        with patch("myclt.ML.supervised_learning.linear_regression.visualization.plot_1d_regression") as mock_plot:
            output = driver.run(menu_visualize, state, monkeypatch)
        
        mock_plot.assert_called_once()


class TestUISaveLoadMenu:
    """Test cases for menu_save_load function."""

    def test_menu_save_load_save_no_dataset(self, monkeypatch):
        """Test error when trying to save without dataset."""
        state = AppState()
        
        driver = (UIDriver()
                 .choose([0, 4])  # Save -> Back
                 )
        
        output = driver.run(menu_save_load, state, monkeypatch)
        
        assert "No dataset" in output

    def test_menu_save_load_save_successful(self, monkeypatch, tmp_path):
        """Test successful session save workflow."""
        state = make_app_state(n_features=2, n_samples=20)
        state.model = LinearRegressionGD(learning_rate=0.01, epochs=10)
        state.model.w = np.array([1.0, 2.0])
        state.model.b = 3.0
        state.metrics = {"mse": 0.5, "rmse": 0.707, "r2": 0.8}
        
        driver = (UIDriver()
                 .choose([0, 4])  # Save -> Back
                 .type(["test_session"])  # Session name
                 .answer([""])  # Continue
                 )
        
        # Mock session storage operations
        with patch("myclt.ML.session_storage.SessionStorage.save_session") as mock_save:
            with patch("myclt.ML.supervised_learning.linear_regression.session_adapter.LinearRegressionSessionAdapter.extract") as mock_extract:
                mock_extract.return_value = (MagicMock(), MagicMock())
                
                output = driver.run(menu_save_load, state, monkeypatch)
        
        assert "saved successfully" in output
        mock_extract.assert_called_once_with(state)
        mock_save.assert_called_once()

    def test_menu_save_load_list_sessions(self, monkeypatch):
        """Test listing saved sessions."""
        state = AppState()
        
        driver = (UIDriver()
                 .choose([2, 4])  # List sessions -> Back
                 .answer([""])  # Continue
                 )
        
        # Mock list_sessions to return some sessions
        with patch("myclt.ML.session_storage.SessionStorage.list_sessions") as mock_list:
            mock_list.return_value = ["session1", "session2"]
            with patch("myclt.ML.session_storage.SessionStorage.load_session") as mock_load:
                mock_load.side_effect = Exception("Mock load error for list test")
                
                output = driver.run(menu_save_load, state, monkeypatch)
        
        assert "Saved sessions" in output

    def test_menu_save_load_delete_session(self, monkeypatch):
        """Test session deletion workflow."""
        state = AppState()
        
        driver = (UIDriver()
                 .choose([3, 4])  # Delete session -> Back
                 .type(["test_session"])  # Session name
                 .answer(["yes"])  # Confirm deletion
                 .answer([""])  # Continue
                 )
        
        # Mock delete_session
        with patch("myclt.ML.session_storage.SessionStorage.delete_session") as mock_delete:
            with patch("myclt.ML.session_storage.SessionStorage.list_sessions") as mock_list:
                mock_list.return_value = ["test_session", "other_session"]
                
                output = driver.run(menu_save_load, state, monkeypatch)
        
        assert "Session deleted" in output or "deleted" in output

    def test_menu_save_load_load_session_success(self, monkeypatch):
        """Test successful session loading."""
        state = AppState()
        
        driver = (UIDriver()
                 .choose([1, 4])  # Load session -> Back
                 .type(["1"])  # Select first session
                 .answer([""])  # Continue
                 )
        
        # Mock load operations
        with patch("myclt.ML.session_storage.SessionStorage.list_sessions") as mock_list:
            mock_list.return_value = ["test_session"]
            with patch("myclt.ML.session_storage.SessionStorage.load_session") as mock_load:
                mock_load.return_value = (MagicMock(), MagicMock())
                with patch("myclt.ML.supervised_learning.linear_regression.session_adapter.LinearRegressionSessionAdapter.restore") as mock_restore:
                    
                    output = driver.run(menu_save_load, state, monkeypatch)
        
        assert "loaded successfully" in output
        mock_restore.assert_called_once()
        

class TestUICLIEntryPoint:
    """Test cases for CLI entry point."""

    def test_cli_main_basic_navigation(self, monkeypatch):
        """Test basic CLI navigation through menus."""
        # Simulate user input: choose Data (0) then Exit (6)
        input_sequence = ["0", "", "6"]
        
        # Mock the UI functions that cli_main would call
        mock_data = MagicMock(return_value=None)
        mock_train = MagicMock(return_value=None)
        mock_eval = MagicMock(return_value=None)
        mock_predict = MagicMock(return_value=None)
        mock_viz = MagicMock(return_value=None)
        mock_save = MagicMock(return_value=None)
        
        # Patch the menu functions in the cli module
        with patch("myclt.ML.supervised_learning.linear_regression.cli.menu_data", mock_data):
            with patch("myclt.ML.supervised_learning.linear_regression.cli.menu_train", mock_train):
                with patch("myclt.ML.supervised_learning.linear_regression.cli.menu_evaluate", mock_eval):
                    with patch("myclt.ML.supervised_learning.linear_regression.cli.menu_predict", mock_predict):
                        with patch("myclt.ML.supervised_learning.linear_regression.cli.menu_visualize", mock_viz):
                            with patch("myclt.ML.supervised_learning.linear_regression.cli.menu_save_load", mock_save):
                                
                                # Mock input to simulate user choices
                                with patch("builtins.input", side_effect=input_sequence):
                                    try:
                                        cli_main()  # Should call Data menu, then exit
                                    except SystemExit:
                                        pass  # Normal exit from choosing 'Exit'
        
        # Verify Data menu was called (the first navigation step)
        mock_data.assert_called_once()


class TestUIPredictMenu:
    """Test cases for menu_predict function."""

    def test_menu_predict_no_model_error(self, monkeypatch):
        """Test error when no trained model exists."""
        state = AppState()
        
        driver = (UIDriver()
                 .choose([0, 2])  # Predict one -> Back
                 )
        
        output = driver.run(menu_predict, state, monkeypatch)
        
        assert "trained model" in output or "Train" in output

    def test_menu_predict_single_object_scaling(self, monkeypatch):
        """Test single object prediction with scaling."""
        state = make_app_state(n_features=2, n_samples=20, use_scaling=True)
        state.model = LinearRegressionGD(learning_rate=0.01, epochs=10)
        state.model.w = np.array([1.0, 2.0])
        state.model.b = 3.0
        state.scaler_mean = np.array([0.0, 0.0])
        state.scaler_std = np.array([1.0, 1.0])
        
        driver = (UIDriver()
                 .choose([0, 2])  # Predict one -> Back
                 .give_float([1.0, 2.0])  # Feature values
                 .answer([""])  # Continue
                 )
        
        output = driver.run(menu_predict, state, monkeypatch)
        
        assert "Predicted" in output
        assert state.prepareddata.target_name in output

    def test_menu_predict_batch_from_csv_error_no_model(self, monkeypatch):
        """Test batch prediction error when no model is trained."""
        state = make_app_state(n_features=2, n_samples=20)
        
        driver = (UIDriver()
                 .choose([1, 2])  # Batch predict -> Back
                 )
        
        output = driver.run(menu_predict, state, monkeypatch)
        
        assert "Train" in output

    def test_menu_predict_batch_from_csv_workflow(self, monkeypatch, tmp_path):
        """Test batch prediction workflow with mock."""
        # Create test CSV
        csv_content = """feature_1,feature_2
1.0,2.0
3.0,4.0"""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text(csv_content)
        
        state = make_app_state(n_features=2, n_samples=20)
        state.model = LinearRegressionGD(learning_rate=0.01, epochs=10)
        state.model.w = np.array([1.0, 2.0])
        state.model.b = 3.0
        
        driver = (UIDriver()
                 .choose([1, 2])  # Batch predict -> Back
                 .type([str(csv_path), ""])  # CSV path, default output
                 .answer([""])  # Continue
                 )
        
        # Mock batch predict function
        with patch("myclt.ML.supervised_learning.linear_regression.ui.batch_predict_from_csv") as mock_batch:
            mock_batch.return_value = {"n_samples": 2, "output_path": "predictions.csv"}
            
            output = driver.run(menu_predict, state, monkeypatch)
        
        assert "Processed" in output
        assert "2" in output
        mock_batch.assert_called_once()
