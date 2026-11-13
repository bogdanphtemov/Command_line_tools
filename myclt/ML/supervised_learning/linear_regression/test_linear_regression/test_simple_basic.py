"""
Very simple basic tests that verify minimum functionality.
"""

import pytest
from myclt.ML.supervised_learning.linear_regression.app_state import AppState


def test_app_state_basic():
    """Basic test for AppState creation."""
    state = AppState()
    assert state is not None
    assert state.dataset is None
    assert state.model is None


def test_ui_import():
    """Test that UI module can be imported."""
    from myclt.ML.supervised_learning.linear_regression import ui
    assert ui is not None


def test_cli_import():
    """Test that CLI module can be imported."""
    from myclt.ML.supervised_learning.linear_regression import cli
    assert cli is not None


def test_data_import():
    """Test that data module can be imported."""
    from myclt.ML.supervised_learning.linear_regression import data
    assert data is not None


def test_core_import():
    """Test that core module (LinearRegressionGD) can be imported."""
    from myclt.ML.supervised_learning.linear_regression import core
    assert core is not None
    # Test that the class exists
    assert hasattr(core, 'LinearRegressionGD')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])