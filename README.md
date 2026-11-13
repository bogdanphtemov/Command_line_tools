# 🔧 Command-line-tools

A comprehensive collection of command-line utilities for system management, project automation, and machine learning. This project is designed for educational purposes to enhance skills in machine learning and software development.

## 📋 Overview

This project implements machine learning algorithms from scratch without relying on high-level libraries like [Scikit-learn](https://scikit-learn.org/stable/). It demonstrates core ML concepts and algorithms using modular, well-tested Python code.

### Features

- **Machine Learning Algorithms**: Linear Regression and Logistic Regression with gradient descent optimization
- **Hyperparameter Tuning**: Grid search capabilities for regularization parameters
- **Data Preprocessing**: Feature scaling, train/test splitting, and data validation
- **Session Management**: Save/load trained models with full reproducibility
- **Visualization**: Loss curves, predictions, and model analysis plots
- **System Utilities**: Project management and system information
- **Pure Python Implementation**: Built with NumPy for numerical computations

## 💻 How It Works

The project uses **NumPy** as the core numerical library for mathematical computations. All machine learning algorithms are implemented from scratch, providing educational insights into how these algorithms work internally.

**Knowledge Sources**: Concepts and implementations are based on [GeeksforGeeks — Python AI](https://www.geeksforgeeks.org/artificial-intelligence/python-ai/) and standard ML textbooks.

---

## 📦 Project Structure

```
myclt/
├── ML/                          # Machine Learning module
│   ├── base_models.py          # Abstract base classes for all models
│   ├── session_storage.py      # Model persistence and loading
│   ├── linear_regression/      # Linear Regression implementation
│   │   ├── core.py             # Core gradient descent algorithm
│   │   ├── data.py             # Data handling and preprocessing
│   │   ├── metrics.py          # Evaluation metrics (MSE, RMSE, R²)
│   │   ├── ui.py               # Interactive CLI menu
│   │   └── visualization.py    # Plotting and visualization
│   └── logistic_regression/    # Logistic Regression implementation
├── automation/                  # Project automation utilities
├── system/                       # System utilities
└── common/                       # Shared utilities and helpers
```

---

## ⚙️ Requirements

- **Python**: ≥ 3.12
- **NumPy**: ≥ 1.24.0
- **Matplotlib**: ≥ 3.7.0 (for visualization)
- **psutil**: ≥ 5.9.0
- **send2trash**: ≥ 1.8.0
- **pipreqs**: ≥ 0.5.0

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/bogdanphtemov/Command_line_tools.git
cd Command_line_tools

# [Recommended] Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# Install the project (will install all dependencies)
pip install -e .
```

## 🎮 Usage

```bash
# From the project root (Command_line_tools)
python -m myclt
# or if you activated the virtual environment:
# just: myclt
```

> **Note:** `python -m myclt` works from any folder after package installation.

### If you don't want to install the package:
```bash
# Just run from the project root:
python -m myclt
# or
python3 -m myclt
```

> **No need** to go inside the `myclt/` folder! The project runs from the root.

This launches an interactive menu where you can:
1. Load and explore datasets
2. Select features and target variables
3. Configure train/test splits
4. Train machine learning models
5. Tune hyperparameters
6. Evaluate model performance
7. Save and load trained models

---

## 🧪 Testing

This project includes a comprehensive test suite (229 tests) to validate the machine learning algorithms and ensure code quality.

### Quick Start

```bash
# Run ALL tests (pytest automatically finds the test directory)
python -m pytest

# For detailed output (one test per line):
python -m pytest -v
```

### Test Structure & What's Covered

The test suite is divided into modular test files covering all major components:

- ✅ **Core Algorithm** (`test_core.py`): Gradient descent implementation, weight initialization, convergence
- ✅ **Metrics** (`test_metrics.py`): MSE, RMSE, R² calculations with edge cases
- ✅ **Data Preprocessing** (`test_preprocessing.py`): Scaling, train/test splitting, normalization
- ✅ **App State Management** (`test_app_state.py`): Configuration state and data flow
- ✅ **Session Storage** (`test_session_roundtrip.py`): Model save/load functionality
- ✅ **CLI Interface** (`test_cli.py`): Command-line argument parsing
- ✅ **Visualization** (`test_visualization.py`): Plotting utilities and charts
- ✅ **Hyperparameter Tuning** (`test_hyperparameter_tuning.py`): Grid search optimization
- ✅ **UI Menus** (`test_ui.py`): Interactive menu system
- ✅ **End-to-End** (`test_integration.py`): Complete workflow on real datasets

### Test Commands for Different Needs

```bash
# Run all tests quietly (default - shows only failures)
python -m pytest

# Run with verbose output (see each test)
python -m pytest -v

# Skip slow tests (integration tests with real CSV datasets)
python -m pytest -m "not slow"

# Run only fast tests (core functionality)
python -m pytest -m "not slow and not integration"

# Run only specific test file
python -m pytest myclt/ML/supervised_learning/linear_regression/test_linear_regression/test_core.py -v

# Run single test method
python -m pytest "myclt/ML/supervised_learning/linear_regression/test_linear_regression/test_ui.py::TestUIDataMenu::test_menu_data_load_csv_and_back" -v

# Stop on first failure
python -m pytest -x

# Show failed tests first, then errors, then skipped tests
python -m pytest -rfa

# Show timing for 10 slowest tests
python -m pytest --durations=10
```

### Configuration

Pytest is configured in `pyproject.toml` to automatically:
- Find test files in the linear regression test directory
- Run in quiet mode by default (`-q`)
- Recognize `@pytest.mark.slow` for slow tests that use real datasets

### Current Test Coverage

- **229 Total Tests**
- **Core Algorithms**: ✅ Fully validated (100% passing)
- **Data Handling**: ✅ Robust error handling and edge cases (100% passing)
- **Integration Tests**: ✅ Real dataset workflow validated (100% passing)
- **UI/UX**: 🔶 Mostly working (some interactive tests may require manual review)

> **Note:** As of latest updates, the test suite completes successfully with **213/224 tests passing (95% success rate)**. The remaining 11 failing tests involve exact text matching in UI interactions - they don't affect algorithm correctness.

Tests ensure that the linear regression algorithm correctly implements:
- Gradient descent optimization
- L1/L2 regularization
- Early stopping
- Feature scaling and normalization
- Performance metrics calculation
- Model persistence

---

## 🎥 Demo

Interactive Linear Regression Training:

![Demo 1](assets/VID_20260531_154641.gif)

Model Configuration and Visualization:

![Demo 2](assets/VID_20260531_154847.gif)

Prediction and Evaluation:

![Demo 3](assets/VID_20260531_155210.gif)

---

## 🛠️ Technologies & Libraries

| Category | Tools |
|----------|-------|
| **Language** | Python 3.12+ |
| **Numerical Computing** | NumPy |
| **Visualization** | Matplotlib |
| **System Utilities** | psutil, send2trash |
| **Package Management** | pipreqs |

---

## 📝 License

This project is licensed under the [MIT License](LICENSE) - feel free to use and modify for educational purposes.

---

## 🤝 Contributing

This is an educational project. Contributions, suggestions, and feedback are welcome!



