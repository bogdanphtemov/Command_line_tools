"""
Session adapter for Linear Regression — refactored to use BaseSessionAdapter.
"""

from typing import List, Dict, Any
from .data import Dataset, Prepareddata
from .core import LinearRegressionGD
from . import preprocessing

from myclt.ML.session_storage import SessionData
from myclt.ML.base_session_adapter import BaseSessionAdapter


class LinearRegressionSessionAdapter(BaseSessionAdapter):
    """Adapter for Linear Regression session serialization"""

    algorithm_name = "linear_regression"
    model_class = LinearRegressionGD
    dataset_class = Dataset
    prepareddata_class = Prepareddata
    preprocessing_module = preprocessing

    hyperparam_specs: List[Dict[str, Any]] = [
        {"name": "use_l1", "default": False},
        {"name": "lambda_l1", "default": 0.01},
        {"name": "use_l2", "default": False},
        {"name": "lambda_l2", "default": 0.01},
    ]

    def validate_session(self, session_data: SessionData) -> bool:
        """Validate Linear Regression specific requirements"""
        super().validate_session(session_data)
        # No additional required params beyond base
        return True
