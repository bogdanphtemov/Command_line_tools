import numpy as np
import pytest
import json
from pathlib import Path
from unittest.mock import patch
from typing import Dict, Any, Optional

from myclt.ML.supervised_learning.linear_regression.app_state import AppState
from myclt.ML.supervised_learning.linear_regression.core import LinearRegressionGD
from myclt.ML.supervised_learning.linear_regression.session_adapter import LinearRegressionSessionAdapter
from myclt.ML.session_storage import SessionStorage, SessionData, TrainingConfig
from myclt.ML.supervised_learning.linear_regression.data import Dataset, Prepareddata
from myclt.ML.supervised_learning.linear_regression.test_linear_regression.helpers import session_roundtrip, make_app_state


class TestSessionRoundtrip:
    """Test cases for session save/load cycle and validation."""

    def test_extract_without_dataset_raises_error(self):
        """Test that extract raises ValueError without dataset."""
        adapter = LinearRegressionSessionAdapter()
        state = AppState()
        state.dataset = None
        state.prepareddata = Prepareddata(
            X=np.random.randn(10, 2),
            Y=np.random.randn(10),
            feature_names=["f1", "f2"],
            target_name="target"
        )
        
        with pytest.raises(ValueError, match="No dataset loaded!"):
            adapter.extract(state)

    def test_extract_without_prepareddata_raises_error(self):
        """Test that extract raises ValueError without prepareddata."""
        adapter = LinearRegressionSessionAdapter()
        state = AppState()
        state.dataset = Dataset(
            data=np.random.randn(10, 3),
            columns=["f1", "f2", "target"]
        )
        state.prepareddata = None
        
        with pytest.raises(ValueError, match="No features/target selected!"):
            adapter.extract(state)

    def test_extract_without_trained_model_raises_error(self):
        """Test that extract raises ValueError with untrained model."""
        adapter = LinearRegressionSessionAdapter()
        state = AppState()
        state.dataset = Dataset(
            data=np.random.randn(10, 3),
            columns=["f1", "f2", "target"]
        )
        state.prepareddata = Prepareddata(
            X=np.random.randn(10, 2),
            Y=np.random.randn(10),
            feature_names=["f1", "f2"],
            target_name="target"
        )
        state.model = LinearRegressionGD()  # Not trained
        
        with pytest.raises(ValueError, match="Model not trained yet!"):
            adapter.extract(state)

    def test_session_roundtrip_predictions_identical(self, prepared_state, tmp_path):
        """Test that roundtrip produces identical predictions and metrics."""
        state = prepared_state
        
        # Train a model
        state.model = LinearRegressionGD(learning_rate=0.01, epochs=100)
        state.model.fit(state.X_train, state.y_train)
        
        # Store original predictions and metrics
        original_predictions = state.model.predict(state.X_test)
        original_metrics = {
            "mse": 0.1234,
            "rmse": 0.3512,
            "r2": 0.8567
        }
        state.metrics = original_metrics.copy()
        
        # Perform roundtrip
        restored_state = session_roundtrip(state, tmp_path / "session")
        
        # Check predictions are identical
        restored_predictions = restored_state.model.predict(restored_state.X_test)
        np.testing.assert_allclose(
            original_predictions, restored_predictions,
            rtol=1e-10, atol=1e-10,
            err_msg="Predictions should be identical after roundtrip"
        )
        
        # Check metrics are identical
        assert state.metrics == restored_state.metrics, "Metrics should be identical after roundtrip"
        
        # Check hyperparameters are preserved
        assert state.learning_rate == pytest.approx(restored_state.learning_rate)
        assert state.epochs == restored_state.epochs
        assert state.use_scaling == restored_state.use_scaling
        
        # Check scaler parameters if scaling was used
        if state.use_scaling:
            assert state.scaler_mean is not None
            assert state.scaler_std is not None
            np.testing.assert_allclose(state.scaler_mean, restored_state.scaler_mean)
            np.testing.assert_allclose(state.scaler_std, restored_state.scaler_std)

    def test_session_roundtrip_hyperparameters_preserved(self, prepared_state, tmp_path):
        """Test that all hyperparameters are preserved in roundtrip."""
        state = prepared_state
        
        # Set custom hyperparameters with regularization
        state.learning_rate = 0.005
        state.learning_rate_auto = False
        state.epochs = 500
        state.use_l1 = True
        state.lambda_l1 = 0.05
        state.use_l2 = True  
        state.lambda_l2 = 0.01
        state.early_stopping = False
        
        # Train model
        state.model = LinearRegressionGD(
            learning_rate=state.learning_rate,
            epochs=state.epochs,
            lambda_l1=state.lambda_l1,
            lambda_l2=state.lambda_l2
        )
        state.model.fit(state.X_train, state.y_train)
        
        # Perform roundtrip
        restored_state = session_roundtrip(state, tmp_path / "session")
        
        # Check hyperparameters
        assert restored_state.learning_rate == pytest.approx(0.005)
        assert restored_state.learning_rate_auto is False
        assert restored_state.epochs == 500
        assert restored_state.use_l1 is True
        assert restored_state.lambda_l1 == pytest.approx(0.05)
        assert restored_state.use_l2 is True
        assert restored_state.lambda_l2 == pytest.approx(0.01)
        assert restored_state.early_stopping is False

    def test_session_files_created(self, prepared_state, tmp_path):
        """Test that session files are created on disk."""
        state = prepared_state
        
        state.model = LinearRegressionGD(learning_rate=0.01, epochs=100)
        state.model.fit(state.X_train, state.y_train)
        
        adapter = LinearRegressionSessionAdapter()
        storage = SessionStorage()
        
        session_dir = tmp_path / "session"
        session_data, arrays_dict = adapter.extract(state)
        storage.save_session(session_data, str(session_dir), arrays_dict, verbose=False)
        
        # Check files exist
        assert (session_dir / "metadata.json").exists()
        assert (session_dir / "config.json").exists()
        assert (session_dir / "data.npz").exists()
        
        # Check metadata content
        with open(session_dir / "metadata.json") as f:
            metadata = json.load(f)
        assert metadata["algorithm"] == "linear_regression"
        assert "timestamp" in metadata
        
        # Check config content
        with open(session_dir / "config.json") as f:
            config = json.load(f)
        assert config["model_type"] == "linear_regression"
        assert config["model_trained"] is True

    def test_storage_list_delete_operations(self, prepared_state, tmp_path):
        """Test SessionStorage list_sessions and delete_session operations."""
        state = prepared_state
        
        state.model = LinearRegressionGD(learning_rate=0.01, epochs=100)
        state.model.fit(state.X_train, state.y_train)
        
        storage = SessionStorage()
        adapter = LinearRegressionSessionAdapter()
        
        session_dir1 = tmp_path / "sessions" / "test1"
        session_dir1.mkdir(parents=True)
        
        session_data, arrays_dict = adapter.extract(state)
        storage.save_session(session_data, str(session_dir1), arrays_dict, verbose=False)
        
        # Test list_sessions
        sessions = storage.list_sessions(str(tmp_path / "sessions"))
        assert "test1" in sessions
        
        # Test delete_session
        storage.delete_session(str(session_dir1), verbose=False)
        sessions_after_delete = storage.list_sessions(str(tmp_path / "sessions"))
        assert "test1" not in sessions_after_delete
        assert not session_dir1.exists()

    def test_validate_session_missing_required_params(self):
        """Test validate_session raises error for missing required parameters."""
        adapter = LinearRegressionSessionAdapter()
        
        # Create session data without required params
        session_data = SessionData(
            dataset_columns=["f1", "f2", "target"],
            feature_names=["f1", "f2"],
            target_name="target",
            use_scaling=True,
            model_type="linear_regression",
            model_trained=True,
            training_config=TrainingConfig(
                hyperparams={}  # Missing learning_rate, epochs
            ),
            metrics={},
            metadata={
                "algorithm": "linear_regression",
                "timestamp": "2023-01-01T00:00:00"
            }
        )
        
        with pytest.raises(ValueError, match="Missing learning_rate"):
            adapter.validate_session(session_data)
    
    def test_validate_session_missing_epochs_param(self):
        """Test validate_session raises error for missing epochs."""
        adapter = LinearRegressionSessionAdapter()
        
        session_data = SessionData(
            dataset_columns=["f1", "f2", "target"],
            feature_names=["f1", "f2"],
            target_name="target",
            use_scaling=True,
            model_type="linear_regression",
            model_trained=True,
            training_config=TrainingConfig(
                hyperparams={
                    "learning_rate": 0.05
                    # Missing epochs
                }
            ),
            metrics={},
            metadata={
                "algorithm": "linear_regression",
                "timestamp": "2023-01-01T00:00:00"
            }
        )
        
        with pytest.raises(ValueError, match="Missing epochs"):
            adapter.validate_session(session_data)

    def test_load_nonexistent_session_raises_filenotfounderror(self):
        """Test that loading non-existent session raises FileNotFoundError."""
        storage = SessionStorage()
        
        with pytest.raises(FileNotFoundError):
            storage.load_session("/path/to/nonexistent/session", verbose=False)

    def test_session_roundtrip_use_scaling_false(self, tmp_path):
        """Test roundtrip when use_scaling is False."""
        # Create state without scaling
        state = make_app_state(use_scaling=False)
        
        # Train model
        state.model = LinearRegressionGD(learning_rate=0.01, epochs=100)
        state.model.fit(state.X_train, state.y_train)
        
        # Perform roundtrip
        restored_state = session_roundtrip(state, tmp_path / "session")
        
        # Check that scaling is preserved as False
        assert restored_state.use_scaling is False
        assert restored_state.scaler_mean is None
        assert restored_state.scaler_std is None
        
        # Check that raw data matches
        np.testing.assert_allclose(state.X_train, restored_state.X_train)
        np.testing.assert_allclose(state.X_test, restored_state.X_test)

    def test_session_roundtrip_with_indices_reconstruction(self, prepared_state, tmp_path):
        """Test roundtrip with index-based reconstruction."""
        state = prepared_state
        
        # Ensure we have indices
        assert state.train_idx is not None
        assert state.test_idx is not None
        
        state.model = LinearRegressionGD(learning_rate=0.01, epochs=100)
        state.model.fit(state.X_train, state.y_train)
        
        # Perform roundtrip
        restored_state = session_roundtrip(state, tmp_path / "session")
        
        # Check indices are preserved
        np.testing.assert_array_equal(state.train_idx, restored_state.train_idx)
        np.testing.assert_array_equal(state.test_idx, restored_state.test_idx)
        
        # Check that reconstructed data matches using indices
        original_X_train_from_indices = state.prepareddata.X[state.train_idx]
        restored_X_train_from_indices = restored_state.prepareddata.X[restored_state.train_idx]
        
        np.testing.assert_allclose(original_X_train_from_indices, restored_X_train_from_indices)