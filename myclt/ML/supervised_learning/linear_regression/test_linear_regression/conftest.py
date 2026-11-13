import numpy as np
import pytest
from pathlib import Path
import matplotlib

# Set matplotlib backend to Agg before importing pyplot
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from myclt.ML.supervised_learning.linear_regression.core import LinearRegressionGD
from myclt.ML.supervised_learning.linear_regression.data import Dataset, Prepareddata
from myclt.ML.supervised_learning.linear_regression.app_state import AppState, rebuild_split
from myclt.ML.supervised_learning.linear_regression.preprocessing import (
    train_test_split, standardize_fit, standardize_apply
)

# Project paths
PROJECT_ROOT = Path(__file__).parents[5]
# Note: Directory has parentheses in the name, so we match the actual folder structure
DATASETS_DIR = PROJECT_ROOT / "assets" / "test_datasets" / "linear_regression_datasets(multiple_linear_regression_datasets)"

# Dataset specifications for real datasets (using actual file names in the test directory)
DATASET_SPECS = {
    "marketing": {
        "file": "basic_marketing_dataset.csv",
        "columns": ["TV", "Radio", "Newspaper", "Sales"],
        "target": "Sales"
    },
    "real_estate": {
        "file": "medium_difficulty_real_estate_dataset.csv", 
        "columns": ["house_age", "distance_to_mrt", "num_convenience_stores", "house_price"],
        "target": "house_price"
    },
    "lifestyle": {
        "file": "high_difficulty_lifestyle_dataset.csv",
        "columns": ["Age", "Income", "Spending", "Savings"],
        "target": "Spending"
    }
}

@pytest.fixture(autouse=True)
def _matplotlib_agg():
    """Automatically close all matplotlib figures after each test."""
    yield
    plt.close("all")

# Synthetic datasets

@pytest.fixture
def perfect_linear():
    """Perfect linear relationship: y = 2x + 1 (no noise)"""
    np.random.seed(42)
    X = np.random.randn(100, 1)
    y = 2 * X[:, 0] + 1
    return X, y

@pytest.fixture
def linear_xy():
    """Single feature with small noise"""
    np.random.seed(42)
    X = np.random.randn(100, 1)
    y = 2 * X[:, 0] + 1 + 0.1 * np.random.randn(100)
    return X, y

@pytest.fixture
def linear_2d():
    """Two features with linear relationship"""
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = 3 * X[:, 0] - 2 * X[:, 1] + 5 + 0.1 * np.random.randn(100)
    return X, y

@pytest.fixture
def multi_xy():
    """Three features with linear relationship"""
    np.random.seed(42)
    X = np.random.randn(100, 3)
    y = 2 * X[:, 0] + 1.5 * X[:, 1] - X[:, 2] + 3 + 0.1 * np.random.randn(100)
    return X, y

@pytest.fixture
def noisy_xy():
    """Single feature with high noise (low R²)"""
    np.random.seed(42)
    X = np.random.randn(100, 1)
    y = 2 * X[:, 0] + 1 + 2 * np.random.randn(100)  # High noise
    return X, y

@pytest.fixture
def constant_y():
    """Constant target for R² edge case testing"""
    np.random.seed(42)
    X = np.random.randn(100, 1)
    y = np.full(100, 5.0)  # Constant target
    return X, y

@pytest.fixture
def tiny_xy():
    """Small dataset for fast training"""
    np.random.seed(42)
    X = np.random.randn(10, 1)
    y = 2 * X[:, 0] + 1 + 0.1 * np.random.randn(10)
    return X, y

@pytest.fixture
def dataset_path(request):
    """Get path to real dataset CSV by name."""
    dataset_name = request.param  # Get parameter from @pytest.mark.parametrize
    if dataset_name not in DATASET_SPECS:
        pytest.skip(f"Dataset '{dataset_name}' not found in DATASET_SPECS")
    
    file_path = DATASETS_DIR / DATASET_SPECS[dataset_name]["file"]
    if not file_path.exists():
        pytest.skip(f"Dataset file not found: {file_path}")
    
    return str(file_path)

@pytest.fixture
def prepared_state():
    """AppState with dataset, prepared data and split."""
    # Create synthetic dataset
    X = np.random.randn(100, 3)
    y = 2 * X[:, 0] + 1.5 * X[:, 1] - X[:, 2] + 3 + 0.1 * np.random.randn(100)
    
    # Create Dataset
    dataset = Dataset(
        data=np.column_stack([X, y]),
        columns=["feature_1", "feature_2", "feature_3", "target"]
    )
    
    # Create Prepareddata (assuming all features, last column as target)
    prepareddata = Prepareddata(
        X=X,
        Y=y,
        feature_names=["feature_1", "feature_2", "feature_3"],
        target_name="target"
    )
    
    # Create AppState and setup
    state = AppState()
    state.dataset = dataset
    state.prepareddata = prepareddata
    state.use_scaling = True
    state.test_size = 0.2
    state.seed = 42
    
    # Build split
    rebuild_split(state)
    
    return state

@pytest.fixture
def trained_model(linear_2d):
    """Trained LinearRegressionGD on synthetic data."""
    X, y = linear_2d
    
    model = LinearRegressionGD(
        learning_rate=0.01,
        epochs=100
    )
    
    model.fit(X, y)
    return model

@pytest.fixture
def session_dir(tmp_path):
    """Temporary directory for session storage tests."""
    return tmp_path / "sessions" / "test1"

@pytest.fixture
def skip_without_assets():
    """Skip test if dataset files are not available."""
    if not DATASETS_DIR.exists():
        pytest.skip("Test datasets directory not found")
    
    # Check if any dataset files exist
    dataset_files = list(DATASETS_DIR.glob("*.csv"))
    if not dataset_files:
        pytest.skip("No dataset files found")
    
    return DATASETS_DIR