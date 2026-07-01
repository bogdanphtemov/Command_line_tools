"""
Session adapters for Logistic Regression — refactored to use BaseSessionAdapter.

Provides adapters for both:
    - Binary Logistic Regression (LogisticRegressionGD)
    - Multinomial Logistic Regression (MultinomialLogisticRegression)
"""

from typing import List, Dict, Any
import numpy as np
from .data import Dataset, Prepareddata
from .core import LogisticRegressionGD, MultinomialLogisticRegression
from . import preprocessing

from ML.session_storage import SessionData
from ML.base_session_adapter import BaseSessionAdapter


class LogisticRegressionSessionAdapter(BaseSessionAdapter):
    """Adapter for Binary Logistic Regression session serialization"""

    algorithm_name = "logistic_regression"
    model_class = LogisticRegressionGD
    dataset_class = Dataset
    prepareddata_class = Prepareddata
    preprocessing_module = preprocessing

    hyperparam_specs: List[Dict[str, Any]] = [
        {"name": "lambda_l2", "default": 0.0},
        {"name": "threshold", "default": 0.5, "required": True},
    ]

    def validate_session(self, session_data: SessionData) -> bool:
        """Validate Logistic Regression specific requirements"""
        super().validate_session(session_data)
        return True


class MultinomialSessionAdapter(BaseSessionAdapter):
    """Adapter for Multinomial Logistic Regression session serialization"""

    algorithm_name = "multinomial_logistic_regression"
    model_class = MultinomialLogisticRegression
    dataset_class = Dataset
    prepareddata_class = Prepareddata
    preprocessing_module = preprocessing

    hyperparam_specs: List[Dict[str, Any]] = [
        {"name": "lambda_l2", "default": 0.0},
    ]

    def extract(self, app_state: Any) -> tuple[SessionData, Dict[str, np.ndarray]]:
        """Extract with support for class_names and weight matrix W."""
        session_data, arrays_dict = super().extract(app_state)

        # Store class_names if available
        if hasattr(app_state, 'class_names') and app_state.class_names is not None:
            session_data.metadata.description = str(app_state.class_names)

        # The weight matrix W is stored via model_w key in arrays_dict
        # (handled by base adapter's model_w logic)

        return session_data, arrays_dict

    def restore(self, session_data: SessionData, arrays: Dict[str, np.ndarray], app_state: Any) -> None:
        """Restore with class_names support."""
        super().restore(session_data, arrays, app_state)

        # Restore class_names from metadata description if present
        if session_data.metadata.description and hasattr(app_state, 'class_names'):
            try:
                import ast
                parsed = ast.literal_eval(session_data.metadata.description)
                if isinstance(parsed, list):
                    app_state.class_names = parsed
            except (ValueError, SyntaxError):
                pass

    def validate_session(self, session_data: SessionData) -> bool:
        """Validate Multinomial specific requirements"""
        super().validate_session(session_data)
        return True
