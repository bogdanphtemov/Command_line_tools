import numpy as np
from unittest.mock import patch, MagicMock
from io import StringIO
from typing import List, Dict, Any, Optional, Union
import tempfile
import json

import pytest

from myclt.ML.supervised_learning.linear_regression.app_state import AppState
from myclt.ML.supervised_learning.linear_regression.data import Dataset, Prepareddata
from myclt.ML.supervised_learning.linear_regression.core import LinearRegressionGD
from myclt.ML.supervised_learning.linear_regression.session_adapter import LinearRegressionSessionAdapter
from myclt.ML.session_storage import SessionStorage


class UIDriver:
    """Helper for mocking UI interactions in tests

    Simulates user input for UI functions by predefining a queue of responses.
    """
    
    def __init__(self):
        self.choose_queue: List[int] = []
        self.answer_queue: List[str] = []
        self.int_queue: List[int] = []
        self.float_queue: List[float] = []
        self.lr_queue: List[Optional[float]] = []  # learning_rate can be None for auto
        self.type_queue: List[str] = []  # For raw input() calls
    
    def choose(self, values: List[int]) -> 'UIDriver':
        """Queue menu choices (integers)."""
        self.choose_queue.extend(values)
        return self
    
    def answer(self, values: List[str]) -> 'UIDriver':
        """Queue yes/no answers."""
        self.answer_queue.extend(values)
        return self
    
    def give_int(self, values: List[int]) -> 'UIDriver':
        """Queue integer inputs."""
        self.int_queue.extend(values)
        return self
    
    def give_float(self, values: List[float]) -> 'UIDriver':
        """Queue float inputs."""
        self.float_queue.extend(values)
        return self
    
    def give_lr(self, values: List[Optional[float]]) -> 'UIDriver':
        """Queue learning rate inputs."""
        self.lr_queue.extend(values)
        return self
    
    def type(self, values: List[str]) -> 'UIDriver':
        """Queue raw typed inputs."""
        self.type_queue.extend(values)
        return self
    
    def run(self, func, state: AppState, monkeypatch=None) -> str:
        """Run a UI function with mocked inputs, return captured output.
        
        Requires monkeypatch fixture from test.
        """
        if monkeypatch is None:
            raise ValueError("monkeypatch fixture required")
            
        output = StringIO()
        
        # Mock functions in the ui module
        from myclt.ML.supervised_learning.linear_regression import ui
        
        def mock_ask_choice(prompt, options):
            if not self.choose_queue:
                raise RuntimeError("No more choice responses queued")
            return self.choose_queue.pop(0)
        
        def mock_ask_int(prompt, min_val=None, max_val=None, default=None):
            if not self.int_queue:
                raise RuntimeError("No more int responses queued")
            return self.int_queue.pop(0)
        
        def mock_ask_float(prompt, min_val=None, max_val=None, default=None):
            if not self.float_queue:
                raise RuntimeError("No more float responses queued")
            return self.float_queue.pop(0)
        
        def mock_ask_yes_no(prompt):
            if not self.answer_queue:
                raise RuntimeError("No more yes/no responses queued")
            answer = self.answer_queue.pop(0).lower()
            return answer in ('y', 'yes', 'true')
        
        def mock_ask_yes_no_recommended(prompt, recommended):
            return mock_ask_yes_no(prompt)
        
        def mock_ask_learning_rate(prompt):
            if not self.lr_queue:
                raise RuntimeError("No more learning rate responses queued")
            return self.lr_queue.pop(0)
        
        def mock_input(prompt=""):
            if not self.type_queue:
                raise RuntimeError("No more type responses queued")
            return self.type_queue.pop(0)
        
        # Setup monkeypatching
        monkeypatch.setattr(ui, "ask_choice", mock_ask_choice)
        monkeypatch.setattr(ui, "ask_int", mock_ask_int)
        monkeypatch.setattr(ui, "ask_float", mock_ask_float)
        monkeypatch.setattr(ui, "ask_yes_no", mock_ask_yes_no)
        monkeypatch.setattr(ui, "ask_yes_no_recommended", mock_ask_yes_no_recommended)
        monkeypatch.setattr(ui, "ask_learning_rate", mock_ask_learning_rate)
        
        # Patch input and stdout
        monkeypatch.setattr("builtins.input", mock_input)
        monkeypatch.setattr("sys.stdout", output)
        
        # Run the function
        try:
            func(state)
        except SystemExit:
            pass  # Expected for exit operations
        
        return output.getvalue()


def session_roundtrip(state: AppState, session_dir) -> AppState:
    """Test session save/load cycle.
    
    Extract -> Save -> Load -> Restore into new AppState.
    """
    adapter = LinearRegressionSessionAdapter()
    storage = SessionStorage()
    
    # Extract session data
    session_data, arrays_dict = adapter.extract(state)
    
    # Save session
    storage.save_session(session_data, str(session_dir), arrays_dict, verbose=False)
    
    # Load session
    loaded_session_data, loaded_arrays = storage.load_session(str(session_dir), verbose=False)
    
    # Create new state and restore
    new_state = AppState()
    adapter.restore(loaded_session_data, loaded_arrays, new_state)
    
    return new_state


def build_prepared(n_features: int, n_samples: int = 100, has_target: bool = True, 
                   seed: int = 42) -> Prepareddata:
    """Factory for creating Prepareddata instances for testing."""
    np.random.seed(seed)
    
    # Generate random features
    X = np.random.randn(n_samples, n_features)
    
    # Generate target with linear relationship plus noise
    if has_target:
        coefficients = np.random.randn(n_features)
        intercept = np.random.randn()
        y = X @ coefficients + intercept + 0.1 * np.random.randn(n_samples)
        
        # Create column names
        feature_names = [f"feature_{i+1}" for i in range(n_features)]
        target_name = "target"
    else:
        y = None
        feature_names = [f"feature_{i+1}" for i in range(n_features)]
        target_name = None
    
    return Prepareddata(
        X=X,
        Y=y,
        feature_names=feature_names,
        target_name=target_name
    )


def make_app_state(dataset: Optional[Dataset] = None, prepareddata: Optional[Prepareddata] = None,
                  n_features: int = 3, n_samples: int = 100, test_size: float = 0.2,
                  seed: int = 42, use_scaling: bool = True) -> AppState:
    """Factory for creating AppState instances for testing."""
    state = AppState(
        test_size=test_size,
        seed=seed,
        use_scaling=use_scaling
    )
    
    # Create dataset and prepareddata if not provided
    if dataset is None:
        prepareddata = prepareddata or build_prepared(n_features, n_samples)
        
        # Create synthetic dataset to match prepareddata
        # We need to create the full data matrix including target
        if prepareddata.Y is not None:
            data = np.column_stack([prepareddata.X, prepareddata.Y])
            columns = prepareddata.feature_names + [prepareddata.target_name]
        else:
            data = prepareddata.X
            columns = prepareddata.feature_names
            
        dataset = Dataset(data=data, columns=columns)
    
    state.dataset = dataset
    state.prepareddata = prepareddata or build_prepared(n_features, n_samples)
    
    # Build the split
    from myclt.ML.supervised_learning.linear_regression.app_state import rebuild_split
    rebuild_split(state)
    
    return state


def assert_metric_dict_equal(expected: Dict[str, float], actual: Dict[str, float], 
                            rtol: float = 1e-7, atol: float = 1e-10) -> None:
    """Assert that two metric dictionaries are approximately equal.
    
    Uses np.testing.assert_allclose for numeric comparison.
    """
    assert set(expected.keys()) == set(actual.keys()), \
        f"Key mismatch: expected {set(expected.keys())}, got {set(actual.keys())}"
    
    for key in expected:
        np.testing.assert_allclose(
            expected[key], actual[key], rtol=rtol, atol=atol,
            err_msg=f"Metric '{key}' mismatch"
        )


def make_linear_xy(n_samples: int = 100, n_features: int = 1, noise: float = 0.1,
                   weights: Optional[np.ndarray] = None, bias: float = 1.0, 
                   seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic linear data for testing.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features  
        noise: Standard deviation of Gaussian noise
        weights: Custom weights (if None, random weights are generated)
        bias: Intercept term
        seed: Random seed for reproducibility
        
    Returns:
        tuple: (X, y) where X has shape (n_samples, n_features), y has shape (n_samples,)
    """
    np.random.seed(seed)
    
    # Generate random features
    X = np.random.randn(n_samples, n_features)
    
    # Generate weights if not provided
    if weights is None:
        weights = np.random.randn(n_features)
    
    # Generate target with linear relationship
    y = X @ weights + bias
    
    # Add noise
    if noise > 0:
        y += noise * np.random.randn(n_samples)
    
    return X, y


def create_model_with_params(n_features: int, learning_rate: float = 0.01, 
                            epochs: int = 100, lambda_l1: float = 0.0, 
                            lambda_l2: float = 0.0) -> LinearRegressionGD:
    """Helper to create a LinearRegressionGD model with specific parameters."""
    return LinearRegressionGD(
        learning_rate=learning_rate,
        epochs=epochs,
        lambda_l1=lambda_l1,
        lambda_l2=lambda_l2
    )