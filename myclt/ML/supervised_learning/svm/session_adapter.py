"""
Session adapters for SVM — refactored to use BaseSessionAdapter pattern.

Provides adapters for:
    - Binary LinearSVM
    - Binary KernelSVM
    - OneVsRestSVM (multiclass)
    - LinearSVR
    - KernelSVR

Each adapter handles serialization/extraction/restoration of model state,
hyperparameters, and training data (critical for kernel methods).
"""

import ast
from typing import List, Dict, Any, Optional
import numpy as np

from .core import LinearSVM, KernelSVM, OneVsRestSVM, LinearSVR, KernelSVR
from .data import Dataset, Prepareddata
from . import preprocessing

from myclt.ML.session_storage import SessionData, SessionMetadata, TrainingConfig
from myclt.ML.base_session_adapter import BaseSessionAdapter


class LinearSVMSessionAdapter(BaseSessionAdapter):
    """Adapter for Binary Linear SVM session serialization"""

    algorithm_name = "linear_svm"
    model_class = LinearSVM
    dataset_class = Dataset
    prepareddata_class = Prepareddata
    preprocessing_module = preprocessing

    hyperparam_specs: List[Dict[str, Any]] = [
        {"name": "C", "default": 1.0},
        {"name": "batch_size", "default": 0},
    ]

    def validate_session(self, session_data: SessionData) -> bool:
        return super().validate_session(session_data)


class KernelSVMSessionAdapter(BaseSessionAdapter):
    """
    Adapter for Kernel SVM session serialization.

    KernelSVM needs special handling because it requires storing the training
    data (X_train) for computing kernel predictions on new data.
    """

    algorithm_name = "kernel_svm"
    model_class = KernelSVM
    dataset_class = Dataset
    prepareddata_class = Prepareddata
    preprocessing_module = preprocessing

    hyperparam_specs: List[Dict[str, Any]] = [
        {"name": "C", "default": 1.0},
        {"name": "kernel", "default": "rbf"},
        {"name": "gamma", "default": 1.0},
        {"name": "degree", "default": 3},
        {"name": "coef0", "default": 1.0},
    ]

    def extract(self, app_state: Any) -> tuple[SessionData, Dict[str, np.ndarray]]:
        """Extract with support for kernel training data."""
        if app_state.dataset is None:
            raise ValueError("No dataset loaded!")
        if app_state.prepareddata is None:
            raise ValueError("No features/target selected!")
        if app_state.model is None or not app_state.model.is_trained:
            raise ValueError("Model not trained yet!")

        session_data, arrays_dict = super().extract(app_state)

        # Store the full X_train array (needed by kernel methods for predictions)
        if app_state.X_train is not None:
            arrays_dict["X_train_kernel"] = app_state.X_train.copy()
            # Also store the internal training data from the model if available
            if hasattr(app_state.model, 'X_train_stored') and app_state.model.X_train_stored is not None:
                arrays_dict["X_train_kernel_internal"] = app_state.model.X_train_stored

        # Store kernel-specific model params (beta, etc.) stored via model_w
        # The base adapter stores model params from get_params() which includes beta

        return session_data, arrays_dict

    def restore(self, session_data: SessionData, arrays: Dict[str, np.ndarray], app_state: Any) -> None:
        """Restore with kernel training data support."""
        super().restore(session_data, arrays, app_state)

        # Restore internal training data for kernel methods
        if hasattr(app_state.model, 'set_training_data'):
            X_train_kernel = None
            if "X_train_kernel_internal" in arrays:
                X_train_kernel = arrays["X_train_kernel_internal"]
            elif "X_train_kernel" in arrays:
                X_train_kernel = arrays["X_train_kernel"]

            if X_train_kernel is not None and app_state.y_train is not None:
                app_state.model.set_training_data(X_train_kernel, app_state.y_train)

    def validate_session(self, session_data: SessionData) -> bool:
        return super().validate_session(session_data)


class LinearSVRSessionAdapter(BaseSessionAdapter):
    """Adapter for Linear SVR session serialization."""

    algorithm_name = "linear_svr"
    model_class = LinearSVR
    dataset_class = Dataset
    prepareddata_class = Prepareddata
    preprocessing_module = preprocessing

    hyperparam_specs: List[Dict[str, Any]] = [
        {"name": "C", "default": 1.0},
        {"name": "epsilon", "default": 0.1},
        {"name": "batch_size", "default": 0},
    ]

    def validate_session(self, session_data: SessionData) -> bool:
        return super().validate_session(session_data)


class KernelSVRSessionAdapter(BaseSessionAdapter):
    """Adapter for Kernel SVR session serialization."""

    algorithm_name = "kernel_svr"
    model_class = KernelSVR
    dataset_class = Dataset
    prepareddata_class = Prepareddata
    preprocessing_module = preprocessing

    hyperparam_specs: List[Dict[str, Any]] = [
        {"name": "C", "default": 1.0},
        {"name": "epsilon", "default": 0.1},
        {"name": "kernel", "default": "rbf"},
        {"name": "gamma", "default": 1.0},
        {"name": "degree", "default": 3},
        {"name": "coef0", "default": 1.0},
    ]

    def extract(self, app_state: Any) -> tuple[SessionData, Dict[str, np.ndarray]]:
        session_data, arrays_dict = super().extract(app_state)

        # Store training data for kernel predictions
        if app_state.X_train is not None:
            arrays_dict["X_train_kernel"] = app_state.X_train.copy()
            if hasattr(app_state.model, 'X_train_stored') and app_state.model.X_train_stored is not None:
                arrays_dict["X_train_kernel_internal"] = app_state.model.X_train_stored

        return session_data, arrays_dict

    def restore(self, session_data: SessionData, arrays: Dict[str, np.ndarray], app_state: Any) -> None:
        super().restore(session_data, arrays, app_state)

        if hasattr(app_state.model, 'set_training_data'):
            X_train_kernel = None
            if "X_train_kernel_internal" in arrays:
                X_train_kernel = arrays["X_train_kernel_internal"]
            elif "X_train_kernel" in arrays:
                X_train_kernel = arrays["X_train_kernel"]

            if X_train_kernel is not None and app_state.y_train is not None:
                app_state.model.set_training_data(X_train_kernel, app_state.y_train)

    def validate_session(self, session_data: SessionData) -> bool:
        return super().validate_session(session_data)


class OneVsRestSVMSessionAdapter(BaseSessionAdapter):
    """Adapter for OneVsRestSVM (multiclass) session serialization."""

    algorithm_name = "ovr_svm"
    model_class = OneVsRestSVM
    dataset_class = Dataset
    prepareddata_class = Prepareddata
    preprocessing_module = preprocessing

    hyperparam_specs: List[Dict[str, Any]] = [
        {"name": "C", "default": 1.0},
        {"name": "base_estimator_type", "default": "linear"},
        {"name": "kernel", "default": "rbf", "aliases": ["kernel_name"]},
        {"name": "gamma", "default": 1.0},
        {"name": "degree", "default": 3},
        {"name": "coef0", "default": 1.0},
    ]

    def extract(self, app_state: Any) -> tuple[SessionData, Dict[str, np.ndarray]]:
        """Extract with class_names support."""
        if app_state.dataset is None:
            raise ValueError("No dataset loaded!")
        if app_state.prepareddata is None:
            raise ValueError("No features/target selected!")
        if app_state.model is None or not app_state.model.is_trained:
            raise ValueError("Model not trained yet!")

        session_data, arrays_dict = super().extract(app_state)

        # Store class_names in metadata
        if hasattr(app_state, 'class_names') and app_state.class_names is not None:
            session_data.metadata.description = str(app_state.class_names)

        # Store estimator-specific internal data if kernel-based
        if hasattr(app_state, 'base_estimator_type') and app_state.base_estimator_type == 'kernel':
            if app_state.X_train is not None:
                arrays_dict["X_train_kernel"] = app_state.X_train.copy()
                # Store each estimator's internal training data
                for i, est in enumerate(app_state.model.estimators):
                    if hasattr(est, 'X_train_stored') and est.X_train_stored is not None:
                        arrays_dict[f"X_train_kernel_est_{i}"] = est.X_train_stored

        return session_data, arrays_dict

    def restore(self, session_data: SessionData, arrays: Dict[str, np.ndarray], app_state: Any) -> None:
        """Restore with class_names and kernel training data."""
        super().restore(session_data, arrays, app_state)

        # Restore class_names from metadata
        if session_data.metadata.description and hasattr(app_state, 'class_names'):
            try:
                parsed = ast.literal_eval(session_data.metadata.description)
                if isinstance(parsed, list):
                    app_state.class_names = parsed
            except (ValueError, SyntaxError):
                pass

        # Restore internal training data for kernel estimators
        if hasattr(app_state, 'base_estimator_type') and app_state.base_estimator_type == 'kernel':
            for i, est in enumerate(app_state.model.estimators):
                if hasattr(est, 'set_training_data'):
                    X_train_key = f"X_train_kernel_est_{i}"
                    if X_train_key in arrays and app_state.y_train is not None:
                        est.set_training_data(arrays[X_train_key], app_state.y_train)
                    elif "X_train_kernel" in arrays and app_state.y_train is not None:
                        est.set_training_data(arrays["X_train_kernel"], app_state.y_train)

    def validate_session(self, session_data: SessionData) -> bool:
        return super().validate_session(session_data)
