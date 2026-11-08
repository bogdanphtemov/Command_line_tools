"""
Test helpers for linear regression module.

This module provides utility functions and constants that can be used
across all test modules to ensure consistency and avoid duplication.
"""

import numpy as np
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
from unittest.mock import patch, Mock

from myclt.ML.supervised_learning.linear_regression.app_state import AppState, rebuild_split
from myclt.ML.supervised_learning.linear_regression.data import Dataset, Prepareddata
from myclt.ML.supervised_learning.linear_regression.core import LinearRegressionGD
from myclt.ML.supervised_learning.linear_regression.metrics import mse, rmse, r2_score


# Paths configuration - test is in test_linear_regression/
PROJECT_ROOT = Path(__file__).parents[5]
DATASETS_DIR = PROJECT_ROOT / "assets" / "test_datasets" / "linear_regression_datasets(multiple_linear_regression_datasets)"

# Dataset specifications for testing
DATASET_SPECS = {
    'marketing': {
        'features': ['ad_budget_usd'],
        'target': 'sales_usd',
        'r2_threshold': 0.80,
        'size': 2500
    },
    'real_estate': {
        'features': ['sqft', 'bedrooms', 'bathrooms', 'year_built', 'distance_to_city_center_miles'],
        'target': 'price_usd',
        'r2_range': (0.63, 0.85),
        'size': 3500
    },
    'lifestyle': {
        'features': ['age', 'income_usd', 'work_hours', 'exercise_hours', 'screen_time_hours'],
        'target': 'happiness_score',
        'r2_range': (0.45, 0.68),
        'size': 4200
    }
}

# Thresholds for model performance validation
MARKETING_R2_THRESHOLD = 0.80
REAL_ESTATE_R2_RANGE = (0.63, 0.85)
LIFESTYLE_R2_RANGE = (0.45, 0.68)


def train_like_ui(state: AppState, patience: int = 50) -> None:
    """
    Train model following the UI workflow with early stopping.
    
    Args:
        state: AppState with prepared data
        patience: Early stopping patience
    """
    if state.model is None:
        state.model = LinearRegressionGD()
    
    # Set regularization if configured
    if state.use_l1:
        state.model.use_l1 = True
        state.model.lambda_l1 = state.lambda_l1
    if state.use_l2:
        state.model.use_l2 = True
        state.model.lambda_l2 = state.lambda_l2
    
    # Configure learning rate
    lr = None if state.learning_rate_auto else state.learning_rate
    
    # Fit with early stopping
    state.model.fit_with_early_stopping(
        state.X_train, 
        state.y_train, 
        epochs=state.epochs,
    """
    if state.model is None or not state.model.is_trained:
        raise RuntimeError("No trained model available")
    if state.X_test is None or state.y_test is None:
        raise RuntimeError("Test set not available")
        
    y_pred = state.model.predict(state.X_test)
    
    return {
        'mse': mse(state.y_test, y_pred),
        'rmse': rmse(state.y_test, y_pred),
        'r2': r2_score(state.y_test, y_pred)
    }


def assert_metrics_match(expected: Dict[str, float], actual: Dict[str, float], 
                        rtol: float = 1e-6) -> None:
    """
    Assert that two metrics dicts match approximately.
    
    Args:
        expected: Expected metrics dictionary
        actual: Actual metrics dictionary
        rtol: Relative tolerance for comparison
    """
    assert set(expected.keys()) == set(actual.keys())
    for key in expected:
        assert actual[key] == pytest.approx(expected[key], rel=rtol)


def session_roundtrip(state: AppState, session_dir: Path) -> AppState:
    """
    Helper to save and load a session.
    
    Args:
        state: AppState to save
        session_dir: Directory for session files
        
    Returns:
        New AppState loaded from session
    """
    from myclt.ML.session_storage import SessionStorage
    from myclt.ML.supervised_learning.linear_regression.session_adapter import LinearRegressionSessionAdapter
    
    storage = SessionStorage()
    adapter = LinearRegressionSessionAdapter()
    
    # Extract and save
    session_data, arrays_dict = adapter.extract(state)
    storage.save_session(session_data, session_dir, arrays_dict, verbose=False)
    
    # Load into new state
    new_state = type(state)()
    loaded_data, loaded_arrays = storage.load_session(session_dir, verbose=False)
    adapter.restore(loaded_data, loaded_arrays, new_state)
    
    return new_state


def load_dataset_xy(name: str) -> tuple:
    """
    Load dataset and return (X, y, features, target).
    
    Args:
        name: Dataset name (marketing, real_estate, lifestyle)
        
    Returns:
        Tuple of (X, y, feature_names, target_name)
    """
    from myclt.ML.supervised_learning.linear_regression.data import load_csv_dataset
    
    spec = DATASET_SPECS[name]
    path = DATASETS_DIR / f"{name}_dataset.csv"
    
    dataset = load_csv_dataset(str(path))
    X = dataset.data[:, :-1].astype(float)
    y = dataset.data[:, -1].astype(float)
    
    return X, y, spec['features'], spec['target']


class UIDriver:
    """Helper class for running UI functions with mocked inputs."""
    
    def __init__(self):
        self.mock_inputs = []
        self.mock_choices = []
        self.mock_answers = []
        self.mock_ints = []
        self.mock_floats = []
        self.mock_lrs = []
    
    def choose(self, *choices):
        """Queue choices for ask_choice calls."""
        self.mock_choices.extend(choices)
        return self
    
    def answer(self, *answers):
        """Queue answers for ask_yes_no calls."""
        self.mock_answers.extend(answers)
        return self
    
    def give_int(self, *ints):
        """Queue integers for ask_int calls."""
        self.mock_ints.extend(ints)
        return self
    
    def give_float(self, *floats):
        """Queue floats for ask_float calls."""
        self.mock_floats.extend(floats)
        return self
    
    def give_lr(self, *lrs):
        """Queue learning rates for ask_learning_rate calls."""
        self.mock_lrs.extend(lrs)
        return self
    
    def type(self, *inputs):
        """Queue strings for input calls."""
        self.mock_inputs.extend(inputs)
        return self
    
    def run(self, func, state, *args, **kwargs):
        """
        Run a UI function with mocked inputs.
        
        Args:
            func: UI function to run
            state: AppState to use
            *args: Additional args for func
            **kwargs: Additional kwargs for func
            
        Returns:
            Captured stdout output
        """
        from io import StringIO
        from contextlib import redirect_stdout
        from unittest.mock import patch
        
        choices = list(self.mock_choices)
        answers = list(self.mock_answers)
        ints = list(self.mock_ints)}

import pytest
        learning_rate=lr,
        patience=patience,
        verbose=False
    )


def evaluate_like_ui(state: AppState) -> Dict[str, float]:
    """
    Evaluate model like ui.menu_evaluate - returns metrics dict.
    
    Args:
        state: AppState with trained model and test data
        
    Returns:
        Dictionary of metrics: mse, rmse, r2