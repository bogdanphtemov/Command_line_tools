"""
Shared fixtures and configuration for linear regression tests.

This file is automatically discovered by pytest and provides common fixtures
that can be used across all test modules in this package.
"""

import numpy as np
# Configure matplotlib to use non-interactive backend BEFORE importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pytest
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List, Callable
import tempfile
import shutil
from unittest.mock import Mock
from io import StringIO
import csv

from myclt.ML.supervised_learning.linear_regression.app_state import AppState, rebuild_split
from myclt.ML.supervised_learning.linear_regression.data import Dataset, Prepareddata, load_csv_dataset
from myclt.ML.supervised_learning.linear_regression.core import LinearRegressionGD
from myclt.ML.supervised_learning.linear_regression.preprocessing import train_test_split, standardize_fit, standardize_apply
from myclt.common.input_validation import ask_choice

# Configurable paths - test is in test_linear_regression/
# Path(__file__).parents[5] goes up to CLT directory
PROJECT_ROOT = Path(__file__).parents[5]
DATASETS_DIR = PROJECT_ROOT / "assets" / "test_datasets" / "linear_regression_datasets(multiple_linear_regression_datasets)"


@pytest.fixture(autouse=True)
def _matplotlib_agg():
    """Automatically close all matplotlib figures after each test."""
    yield
    plt.close('all')


@pytest.fixture
def dataset_path(request):
    """Fixture to provide paths to test datasets."""
    datasets = {
        'marketing': DATASETS_DIR / 'basic_marketing_dataset.csv',
        'real_estate': DATASETS_DIR / 'medium_difficulty_real_estate_dataset.csv', 
        'lifestyle': DATASETS_DIR / 'high_difficulty_lifestyle_dataset.csv',
    }
    
    def get_path(name: str) -> Path:
        if name not in datasets:
            raise ValueError(f"Unknown dataset: {name}. Available: {list(datasets.keys())}")
        return datasets[name]
    
    return get_path


@pytest.fixture
def linear_xy():
    """Simple linear data: y = 2x + 1 + noise"""
    np.random.seed(42)
    n_samples = 100
    X = np.random.randn(n_samples, 1) * 10
    y_clean = 2 * X.flatten() + 1
    y = y_clean + np.random.randn(n_samples) * 0.5  # Small noise
    return X, y


@pytest.fixture
def linear_2d():
    """2D linear data: y = 3x1 + 2x2 + 1 + noise"""
    np.random.seed(42)
    n_samples = 100
    X = np.random.randn(n_samples, 2) * 5
    y_clean = 3 * X[:, 0] + 2 * X[:, 1] + 1
    y = y_clean + np.random.randn(n_samples) * 1.0
    return X, y


@pytest.fixture
def multi_xy():
    """3-feature linear data: y = 1.5x1 + 0.8x2 - 0.3x3 + 2 + noise"""
    np.random.seed(42)
    n_samples = 150
    X = np.random.randn(n_samples, 3) * 3
    y_clean = 1.5 * X[:, 0] + 0.8 * X[:, 1] - 0.3 * X[:, 2] + 2
    y = y_clean + np.random.randn(n_samples) * 0.8
    return X, y


@pytest.fixture
def noisy_xy():
    """High-noise linear data, low R² expected"""
    np.random.seed(42)
    n_samples = 100
    X = np.random.randn(n_samples, 1) * 5
    y_clean = 2 * X.flatten() + 1
    y = y_clean + np.random.randn(n_samples) * 8  # High noise
    return X, y


@pytest.fixture
def ill_conditioned_xy():
    """Ill-conditioned data with near-duplicate features"""
    np.random.seed(42)
    n_samples = 100
    X1 = np.random.randn(n_samples) * 3
    X2 = X1 + np.random.randn(n_samples) * 0.01  # Near-duplicate
    X3 = np.random.randn(n_samples) * 4
    X = np.column_stack([X1, X2, X3])
    y = 2 * X[:, 0] + 1
    return X, y


@pytest.fixture
def make_state(linear_xy):
    """Factory fixture to create AppState instances with different configurations."""
    def _make_state(**overrides) -> AppState:
        # Default state with synthetic data
        X, y = linear_xy
        
        dataset = Dataset(
            data=np.column_stack([X, y]),
            columns=['feature', 'target']
        )
        
        prepareddata = Prepareddata(
            X=X.astype(float),
            Y=y.astype(float),
            feature_names=['feature'],
            target_name='target'
        )
        
        state = AppState(
            dataset=dataset,
            prepareddata=prepareddata,
            test_size=0.2,
            seed=42,
            use_scaling=True,
            learning_rate=0.01,
            learning_rate_auto=False,
            epochs=1000,
            early_stopping=True,
            use_l1=False,
            use_l2=False,
            lambda_l1=0.01,
            lambda_l2=0.01,
            **overrides
        )
        
        return state
    
    return _make_state


@pytest.fixture
def session_tmp(tmp_path):
    """Fixture providing temporary session directory."""
    session_dir = tmp_path / "test_session"
    session_dir.mkdir()
    return session_dir


# Quality thresholds for dataset tests (with reasonable tolerances)
MARKETING_R2_THRESHOLD = 0.80
REAL_ESTATE_R2_RANGE = (0.65, 0.75)  # Adjusted for conservative estimate
LIFESTYLE_R2_RANGE = (0.40, 0.65)    # Wide range due to high noise


def pytest_collection_modifyitems(config, items):
    """Customize test collection - mark slow tests."""
    for item in items:
        if "end_to_end" in item.nodeid:
            item.add_marker(pytest.mark.slow)
        if "grid_search" in item.nodeid or "auto_tune" in item.nodeid:
            item.add_marker(pytest.mark.slow)


# ============================================================================
# Additional fixtures and helpers for the extended test suite
# ============================================================================

@pytest.fixture
def feed_inputs(monkeypatch):
    """Fixture to feed inputs to input() in a controlled sequence."""
    inputs = []
    original_input = input
    
    def mock_input(prompt=""):
        if not inputs:
            raise ValueError("No more inputs available!")
        return inputs.pop(0)
    
    monkeypatch.setattr('builtins.input', mock_input)
    
    def add_input(value):
        inputs.append(value)
    
    return add_input


@pytest.fixture
def menu_choices(monkeypatch, request):
    """Fixture to simulate menu choices (ask_choice) and skip clear_screen."""
    choices = []
    
    def mock_ask_choice(prompt, options):
        if not choices:
            # If no choices left, return last item ("Back" equivalent)
            return len(options) - 1
        return choices.pop(0)
    
    def mock_clear_screen():
        pass  # No-op to avoid screen clearing in tests
    
    monkeypatch.setattr('myclt.common.input_validation.ask_choice', mock_ask_choice)
    monkeypatch.setattr('myclt.common.ui_helpers.clear_screen', mock_clear_screen)
    
    def add_choice(choice_index):
        choices.append(choice_index)
    
    return add_choice


@pytest.fixture
def csv_writer(tmp_path):
    """Fixture to create temporary CSV files for testing."""
    def create_csv(header, data, delimiter=";"):
        import numpy as np
        
        # Create a temporary CSV file
        csv_file = tmp_path / "test_data.csv"
        
        # Handle different data types
        data_rows = []
        for row in data:
            if isinstance(row, (list, tuple, np.ndarray)):
                data_rows.append([str(x) for x in row])
            else:
                data_rows.append([str(row)])
        
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter=delimiter)
            writer.writerow(header)
            for row in data_rows:
                writer.writerow(row)
        
        return str(csv_file)
    
    return create_csv


@pytest.fixture
def prepared_state(make_state):
    """Fixture providing AppState with prepared split (X_train/X_test/y_train/y_test)."""
    def _prepared_state(**overrides):
        state = make_state(**overrides)
        rebuild_split(state)
        return state
    
    return _prepared_state


@pytest.fixture
def trained_model(linear_xy):
    """Fixture providing a trained LinearRegressionGD model."""
    X, y = linear_xy
    model = LinearRegressionGD(learning_rate=0.01, epochs=100)
    model.fit(X, y)
    return model


@pytest.fixture
def trained_state(prepared_state):
    """Fixture providing AppState with trained model."""
    def _trained_state(**overrides):
        state = prepared_state(**overrides)
        model = LinearRegressionGD(learning_rate=0.01, epochs=100)
        model.fit(state.X_train, state.y_train)
        state.model = model
        return state
    
    return _trained_state


@pytest.fixture
def session_dir_tmp(request, tmp_path):
    """Fixture providing structured session directory path."""
    def create_session_dir(session_name="test_session"):
        session_dir = tmp_path / "ml_sessions" / session_name
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir
    
    return create_session_dir


def train_like_ui(state: AppState, patience: int = 50) -> LinearRegressionGD:
    """
    Train a model using the UI recipe (to ensure consistency in tests).
    
    This mimics the training logic from ui.menu_train (choice 3) to ensure
    end-to-end tests and UI tests use the same training recipe.
    
    Args:
        state: AppState with prepared split
        patience: Early stopping patience (default 50 as in UI)
        
    Returns:
        Trained LinearRegressionGD model
    """
    # Split training data into train/validation for early stopping (UI pattern)
    n_train = len(state.X_train)
    val_size = int(0.2 * n_train)
    
    X_train_part = state.X_train[val_size:]
    y_train_part = state.y_train[val_size:]
    X_val = state.X_train[:val_size]
    y_val = state.y_train[:val_size]
    
    # Create and train model
    model = LinearRegressionGD(
        learning_rate=state.learning_rate,
        epochs=state.epochs,
        lambda_l1=state.lambda_l1 if state.use_l1 else 0.0,
        lambda_l2=state.lambda_l2 if state.use_l2 else 0.0
    )
    
    # Train with early stopping (always ON in UI, same pattern as menu_train)
    model.fit_with_early_stopping(
        X_train_part, 
        y_train_part, 
        X_val, 
        y_val, 
        patience=patience,
        verbose=False
    )
    
    return model

# ============================================================================
# Extended fixtures for comprehensive test suite  
# ============================================================================

class UIDriver:
    """Helper class to simulate user input and capture output for UI testing."""
    def __init__(self, module):
        self.module = module
        self.choices = []
        self.answers = []
        self.ints = []
        self.floats = []
        self.lrs = []
        self.inputs = []
        self.pause_calls = []
        self.out = ""
        
    def choose(self, *choices):
        """Queue choices for ask_choice calls."""
        self.choices.extend(choices)
        return self
        
    def answer(self, *answers):
        """Queue answers for yes/no questions."""
        self.answers.extend(answers)
        return self
        
    def give_int(self, *ints):
        """Queue integer inputs."""
        self.ints.extend(ints)
        return self
        
    def give_float(self, *floats):
        """Queue float inputs."""
        self.floats.extend(floats)
        return self
        
    def run(self, func, *args, **kwargs):
        """Run function with mocked inputs and capture output."""
        import io
        from unittest.mock import patch
        from contextlib import redirect_stdout, redirect_stderr
        
        def mock_choice(prompt, options):
            if not self.choices:
                raise AssertionError(f"No more choices available. Prompt: {prompt}, Options: {options}")
            return self.choices.pop(0)
            
        def mock_yes_no(prompt, default=None):
            if not self.answers:
                raise AssertionError(f"No more answers available. Prompt: {prompt}")
            answer = self.answers.pop(0).lower()
            if answer in ('y', 'yes', 'true', '1'):
                return True
            elif answer in ('n', 'no', 'false', '0'):
                return False
            raise AssertionError(f"Invalid yes/no answer: {answer}")
            
        def mock_int(prompt, min_val=None, max_val=None, default=None):
            if not self.ints:
                raise AssertionError(f"No more ints available. Prompt: {prompt}")
            val = self.ints.pop(0)
            if min_val is not None and val < min_val:
                raise AssertionError(f"Int {val} < min_val {min_val}")
            if max_val is not None and val > max_val:
                raise AssertionError(f"Int {val} > max_val {max_val}")
            return val
            
        def mock_float(prompt, min_val=None, max_val=None, default=None):
            if not self.floats:
                raise AssertionError(f"No more floats available. Prompt: {prompt}")
            val = self.floats.pop(0)
            if min_val is not None and val < min_val:
                raise AssertionError(f"Float {val} < min_val {min_val}")
            if max_val is not None and val > max_val:
                raise AssertionError(f"Float {val} > max_val {max_val}")
            return val
            
        def mock_lr(prompt, min_val=1e-6, max_val=10.0):
            if not self.lrs:
                raise AssertionError(f"No more learning rates available. Prompt: {prompt}")
            val = self.lrs.pop(0)
            if val == "" or val == "auto" or val is None:
                return None
            if val < min_val:
                raise AssertionError(f"LR {val} < min_val {min_val}")
            if val > max_val:
                raise AssertionError(f"LR {val} > max_val {max_val}")
            return float(val)
            
        def mock_input(prompt=""):
            if not self.inputs:
                raise AssertionError(f"No more inputs available. Prompt: {prompt}")
            return self.inputs.pop(0)
            
        def mock_pause(msg=""):
            self.pause_calls.append(msg)
            
        def mock_clear_screen():
            pass
            
        def mock_print_header(title):
            pass
        
        patches = [
            patch.object(self.module, 'ask_choice', mock_choice),
            patch.object(self.module, 'ask_yes_no', mock_yes_no),
            patch.object(self.module, 'ask_yes_no_recommended', mock_yes_no),
            patch.object(self.module, 'ask_int', mock_int),
            patch.object(self.module, 'ask_float', mock_float),
            patch.object(self.module, 'ask_learning_rate', mock_lr),
            patch('builtins.input', mock_input),
            patch.object(self.module, 'pause', mock_pause),
            patch.object(self.module, 'clear_screen', mock_clear_screen),
            patch.object(self.module, 'print_header', mock_print_header),
        ]
        
        f = io.StringIO()
        with redirect_stdout(f), redirect_stderr(f):
            for p in patches:
                p.start()
            try:
                result = func(*args, **kwargs)
                for p in patches:
                    p.stop()
                self.out = f.getvalue()
                return result
            except Exception:
                for p in patches:
                    p.stop()
                self.out = f.getvalue()
                raise


@pytest.fixture
def ui_driver(monkeypatch):
    """Fixture providing a UI driver for testing UI functions."""
    from myclt.ML.supervised_learning.linear_regression import ui
    
    def create_driver():
        return UIDriver(ui)
    
    return create_driver


@pytest.fixture  
def cli_driver(monkeypatch):
    """Fixture providing a CLI driver for testing CLI functions."""
    from myclt.ML.supervised_learning.linear_regression import cli
    
    def create_driver():
        return UIDriver(cli)
    
    return create_driver


@pytest.fixture
def quiet_show(monkeypatch):
    """Fixture to suppress plt.show() calls."""
    monkeypatch.setattr(plt, 'show', lambda: None)


@pytest.fixture
def capture_output(capsys):
    """Fixture to capture printed output."""
    def capture():
        captured = capsys.readouterr()
        return captured.out + captured.err
    return capture


@pytest.fixture
def chdir_tmp(tmp_path, monkeypatch):
    """Fixture to change to temporary directory."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def make_model():
    """Factory fixture to create LinearRegressionGD models with different configs."""
    def _make_model(**kwargs):
        defaults = {
            'learning_rate': 0.01,
            'epochs': 100,
            'lambda_l1': 0.0,
            'lambda_l2': 0.0,
        }
        defaults.update(kwargs)
        return LinearRegressionGD(**defaults)
    
    return _make_model


DATASET_SPECS = {
    'marketing': {
        'file': 'basic_marketing_dataset.csv',
        'features': ['ad_budget_usd'],
        'target': 'monthly_revenue_usd',
        'r2_threshold': 0.80,
        'rmse_threshold': 7000.0,
    },
    'real_estate': {
        'file': 'medium_difficulty_real_estate_dataset.csv', 
        'features': ['area_sqm', 'rooms', 'distance_to_center_km', 'building_age_years'],
        'target': 'price_thousand_usd',
        'r2_range': (0.63, 0.85),
        'rmse_threshold': 50.0,
    },
    'lifestyle': {
        'file': 'high_difficulty_lifestyle_dataset.csv',
        'features': ['sleep_hours', 'stress_level', 'physical_activity_mins', 'calories_intake'],
        'target': 'daily_energy_score',
        'r2_range': (0.45, 0.68),  
        'rmse_threshold': 18.0,
    }
}


@pytest.fixture
def loaded_dataset(dataset_path):
    """Fixture providing loaded Dataset from CSV."""
    def _loaded_dataset(name):
        path = dataset_path(name)
        return load_csv_dataset(str(path))
    
    return _loaded_dataset


@pytest.fixture
def dataset_state(loaded_dataset, make_state):
    """Fixture providing AppState with dataset loaded from CSV."""
    def _dataset_state(name):
        dataset = loaded_dataset(name)
        spec = DATASET_SPECS[name]
        
        # Create base state and add dataset
        state = make_state()
        state.dataset = dataset
        
        # Prepare data (select columns)
        X = dataset.data[:, :-1].astype(float)
        y = dataset.data[:, -1].astype(float)
        
        prepareddata = Prepareddata(
            X=X,
            Y=y,
            feature_names=spec['features'],
            target_name=spec['target']
        )
        state.prepareddata = prepareddata
        
        # Rebuild split
        rebuild_split(state)
        
        return state
    
    return _dataset_state



def type(self, *inputs):
    """Queue raw input() calls."""
    self.inputs.extend(inputs)
    return self

