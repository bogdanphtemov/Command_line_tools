"""
Integration test for end-to-end linear regression workflow on real datasets.
Tests loading real CSV data, feature selection, training with early stopping, 
and evaluation metrics with realistic quality expectations.

Marked as slow because training real datasets takes time.
"""

import pytest
import numpy as np
from pathlib import Path
from typing import Dict, List, Any

from myclt.ML.supervised_learning.linear_regression.app_state import AppState, rebuild_split
from myclt.ML.supervised_learning.linear_regression.data import (
    Dataset, Prepareddata, load_csv_dataset
)
from myclt.ML.supervised_learning.linear_regression.core import LinearRegressionGD
from myclt.ML.supervised_learning.linear_regression.metrics import mse, rmse, r2_score

# Path to test datasets (use relative path from this file for CI compatibility)
DATASETS_DIR = Path(__file__).parents[5] / "assets" / "test_datasets" / "linear_regression_datasets(multiple_linear_regression_datasets)"

# Dataset specifications
DATASET_SPECS = {
    "marketing": {
        "file": "basic_marketing_dataset.csv",
        "columns": ["ad_budget_usd", "monthly_revenue_usd"],
        "target": "monthly_revenue_usd"
    },
    "real_estate": {
        "file": "medium_difficulty_real_estate_dataset.csv", 
        "columns": ["area_sqm", "rooms", "distance_to_center_km", "building_age_years", "price_thousand_usd"],
        "target": "price_thousand_usd"
    },
    "lifestyle": {
        "file": "high_difficulty_lifestyle_dataset.csv",
        "columns": ["sleep_hours", "stress_level", "physical_activity_mins", "calories_intake", "daily_energy_score"],
        "target": "daily_energy_score"
    }
}


class TestIntegrationRealDatasets:
    """End-to-end integration tests for real datasets."""

    @pytest.mark.slow
    @pytest.mark.parametrize("dataset_name", ["marketing", "real_estate", "lifestyle"])
    def test_end_to_end_workflow(self, dataset_name: str):
        """
        Full workflow test: load CSV → select features → split → train → predict → evaluate.
        
        Checks that models learn meaningful patterns on real data without being 
        overly sensitive to exact numeric values.
        """
        # Skip if dataset files don't exist
        spec = DATASET_SPECS[dataset_name]
        file_path = DATASETS_DIR / spec["file"]
        if not file_path.exists():
            pytest.skip(f"Dataset file not found: {file_path}")
        
        # Load dataset
        dataset = load_csv_dataset(str(file_path))
        
        # Create AppState for the workflow
        state = AppState(
            test_size=0.2,
            seed=42,
            use_scaling=True,
            learning_rate=0.01,
            learning_rate_auto=False,
            epochs=1000,
            early_stopping=True
        )
        state.dataset = dataset
        
        # Create Prepareddata directly (mimicking select_features_and_target)
        # Use all features except target
        target_col = spec["target"]
        target_idx = state.dataset.columns.index(target_col)
        
        feature_names = [col for col in state.dataset.columns if col != target_col]
        feature_indices = [state.dataset.columns.index(feature) for feature in feature_names]
        
        prepareddata = Prepareddata(
            X=state.dataset.data[:, feature_indices],
            Y=state.dataset.data[:, target_idx],
            feature_names=feature_names,
            target_name=target_col
        )
        state.prepareddata = prepareddata
        
        # Build train/test split
        rebuild_split(state)
        
        # Verify split is correct
        assert state.X_train is not None, "Training features should be set"
        assert state.X_test is not None, "Test features should be set"
        assert state.y_train is not None, "Training targets should be set"
        assert state.y_test is not None, "Test targets should be set"
        
        # Training set should be larger than test set
        assert len(state.X_train) > len(state.X_test), "Training set should be larger than test set"
        assert len(state.X_train) + len(state.X_test) == len(state.prepareddata.X), "Samples should add up"
        
        # Train model with early stopping (reduced epochs for faster testing)
        model = LinearRegressionGD(
            learning_rate=0.01,
            epochs=1000,  # Small epochs for test speed
            lambda_l1=0.0,
            lambda_l2=0.0
        )
        
        # Validation split for early stopping
        val_size = max(1, int(len(state.X_train) * 0.1))  # 10% for validation
        X_train_subset = state.X_train[val_size:]
        y_train_subset = state.y_train[val_size:]
        X_val = state.X_train[:val_size]
        y_val = state.y_train[:val_size]
        
        model.fit_with_early_stopping(
            X_train_subset, y_train_subset, 
            X_val, y_val,
            patience=5,  # Small patience for faster testing
            min_delta=1e-6,
            verbose=False
        )
        
        # Verify model is trained
        assert model.is_trained, "Model should be trained"
        assert model.w is not None and np.all(np.isfinite(model.w)), "Weights should be finite"
        assert np.isfinite(model.b), "Bias should be finite"
        
        # Make predictions on test set
        y_pred = model.predict(state.X_test)
        
        # Verify predictions
        assert y_pred.shape == state.y_test.shape, "Predictions should match test set shape"
        assert np.all(np.isfinite(y_pred)), "All predictions should be finite"
        
        # Calculate metrics
        test_mse = mse(state.y_test, y_pred)
        test_rmse = rmse(state.y_test, y_pred)
        test_r2 = r2_score(state.y_test, y_pred)
        
        # Store metrics in state
        state.metrics = {
            'mse': test_mse,
            'rmse': test_rmse,
            'r2': test_r2
        }
        
        # Verify metrics are finite
        assert np.isfinite(test_mse), f"MSE should be finite, got {test_mse}"
        assert np.isfinite(test_rmse), f"RMSE should be finite, got {test_rmse}"
        assert np.isfinite(test_r2), f"R² should be finite, got {test_r2}"
        
        # Test-specific quality expectations
        if dataset_name == "marketing":
            # Marketing data should have strong linear relationship
            assert test_r2 > 0.8, f"Marketing dataset R² should be > 0.8, got {test_r2:.4f}"
        elif dataset_name in ["real_estate", "lifestyle"]:
            # Real estate and lifestyle should have reasonable but not perfect fit
            # Using more realistic ranges based on dataset difficulty
            expected_range = (0.3, 0.9) if dataset_name == "real_estate" else (0.1, 0.7)
            
            # Check that R² is in reasonable range (not testing specific values to avoid fragility)
            assert test_r2 > expected_range[0], (
                f"{dataset_name} dataset R² should be > {expected_range[0]}, "
                f"got {test_r2:.4f}"
            )
            
            # Upper bound is less critical but should be reasonable
            assert test_r2 < expected_range[1] + 0.2, (
                f"{dataset_name} dataset R² unexpectedly high: {test_r2:.4f}"
            )
        
        # Common checks for all datasets
        assert test_mse >= 0, f"MSE should be non-negative, got {test_mse}"
        assert test_rmse >= 0, f"RMSE should be non-negative, got {test_rmse}"
        
        # RMSE should be sqrt(MSE) within tolerance
        np.testing.assert_allclose(test_rmse, np.sqrt(test_mse), rtol=1e-10)
        
        # Verify loss history properties
        assert len(model.loss_history) <= model.epochs + 1, "Loss history should not exceed epochs + 1"
        if len(model.loss_history) > 0:
            assert np.all(np.isfinite(model.loss_history)), "All loss values should be finite"
        
        print(f"\n=== {dataset_name.upper()} DATASET RESULTS ===")
        print(f"Training samples: {len(state.X_train)}, Test samples: {len(state.X_test)}")
        print(f"Features: {len(feature_names)} ({', '.join(feature_names)})")
        print(f"MSE: {test_mse:.6f}")
        print(f"RMSE: {test_rmse:.6f}")
        print(f"R²: {test_r2:.6f}")
        print(f"Epochs trained: {len(model.loss_history)}")
        
    @pytest.mark.slow
    def test_roundtrip_session_on_real_dataset(self):
        """Test session save/load roundtrip on real dataset."""
        # Use marketing dataset for this test
        spec = DATASET_SPECS["marketing"]
        file_path = DATASETS_DIR / spec["file"]
        if not file_path.exists():
            pytest.skip(f"Dataset file not found: {file_path}")
        
        from myclt.ML.session_storage import SessionStorage
        from myclt.ML.supervised_learning.linear_regression.session_adapter import LinearRegressionSessionAdapter
        from tempfile import TemporaryDirectory
        
        # Load dataset and prepare
        dataset = load_csv_dataset(str(file_path))
        
        state = AppState(
            test_size=0.2,
            seed=42,
            use_scaling=True,
            learning_rate=0.01,
            learning_rate_auto=False,
            epochs=100,
            early_stopping=True
        )
        state.dataset = dataset
        
        # Create Prepareddata
        target_col = spec["target"]
        target_idx = state.dataset.columns.index(target_col)
        feature_names = [col for col in state.dataset.columns if col != target_col]
        feature_indices = [state.dataset.columns.index(feature) for feature in feature_names]
        
        prepareddata = Prepareddata(
            X=state.dataset.data[:, feature_indices],
            Y=state.dataset.data[:, target_idx],
            feature_names=feature_names,
            target_name=target_col
        )
        state.prepareddata = prepareddata
        
        # Build split
        rebuild_split(state)
        
        # Quick training
        model = LinearRegressionGD(learning_rate=0.01, epochs=50)
        model.fit(state.X_train, state.y_train)
        state.model = model
        
        # Calculate and store metrics
        y_pred = model.predict(state.X_test)
        state.metrics = {
            'mse': mse(state.y_test, y_pred),
            'rmse': rmse(state.y_test, y_pred),
            'r2': r2_score(state.y_test, y_pred)
        }
        
        # Roundtrip test
        adapter = LinearRegressionSessionAdapter()
        storage = SessionStorage()
        
        with TemporaryDirectory() as tmpdir:
            session_dir = Path(tmpdir) / "test_session"
            
            # Extract and save
            session_data, arrays_dict = adapter.extract(state)
            storage.save_session(session_data, str(session_dir), arrays_dict, verbose=False)
            
            # Load and restore
            loaded_session_data, loaded_arrays = storage.load_session(str(session_dir), verbose=False)
            new_state = AppState()
            adapter.restore(loaded_session_data, loaded_arrays, new_state)
            
            # Verify state is restored correctly
            assert new_state.dataset is not None
            assert new_state.prepareddata is not None
            assert new_state.model is not None
            assert new_state.model.is_trained
            
            # Verify metrics are restored
            assert 'mse' in new_state.metrics
            assert 'rmse' in new_state.metrics
            assert 'r2' in new_state.metrics
            
            # Predictions should be identical
            original_pred = state.model.predict(state.X_test)
            restored_pred = new_state.model.predict(new_state.X_test)
            np.testing.assert_allclose(original_pred, restored_pred, rtol=1e-10)
            
            print("✓ Session roundtrip successful on real dataset")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-m", "slow"])