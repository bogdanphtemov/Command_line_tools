"""
Session adapter for K-Nearest Neighbors.
"""

from typing import List, Dict, Any
from .data import Dataset, Prepareddata
from .classifier import KNNClassifier
from .regressor import KNNRegressor
from . import preprocessing

from myclt.ML.session_storage import SessionData
from myclt.ML.base_session_adapter import BaseSessionAdapter


class KNNSessionAdapter(BaseSessionAdapter):
    """Adapter for KNN session serialization"""

    algorithm_name = "knn"
    model_class = None  # Handled dynamically based on model_type
    dataset_class = Dataset
    prepareddata_class = Prepareddata
    preprocessing_module = preprocessing

    hyperparam_specs: List[Dict[str, Any]] = [
        {"name": "k", "default": 5},
        {"name": "distance_metric", "default": "euclidean"},
        {"name": "model_type", "default": "classification"},
        {"name": "weighted", "default": False, "applies_to": "regression"},
    ]

    def validate_session(self, session_data: SessionData) -> bool:
        """
        Validate KNN specific requirements.
        
        Side effect:
            Sets self.model_class based on model_type.
        """
        # First validate using base class (which will raise ValueError if invalid)
        super().validate_session(session_data)
        
        # Base class already validated that training_config.hyperparams exists
        hyperparams = session_data.training_config.hyperparams
        
        # Validate that model_type is valid
        model_type = hyperparams.get('model_type')
        if model_type not in ['classification', 'regression']:
            raise ValueError(f"Invalid model_type: {model_type}")
        
        # Set appropriate model class dynamically
        if model_type == 'classification':
            self.model_class = KNNClassifier
        else:
            self.model_class = KNNRegressor
            
        return True
    
    def instantiate_model(self, algorithm_specific: Dict[str, Any]) -> object:
        """Create model instance from saved parameters."""
        # Note: This method is currently not used by BaseSessionAdapter
        # but retained for backward compatibility.
        # BaseSessionAdapter creates models via self.model_class() 
        # and calls set_params() to configure them.
        
        # Use the same source as validate_session for consistency
        model_type = algorithm_specific.get('model_type', 'classification')
        
        # Validate model_type consistency
        if model_type not in ['classification', 'regression']:
            raise ValueError(f"Invalid model_type: {model_type}")
        
        # Extract parameters from algorithm_specific 
        k = algorithm_specific.get('k', 5)
        distance_metric = algorithm_specific.get('distance_metric', 'euclidean')
        
        if model_type == 'classification':
            return KNNClassifier(k=k, distance_metric=distance_metric)
        else:
            weighted = algorithm_specific.get('weighted', False)
            return KNNRegressor(k=k, distance_metric=distance_metric, weighted=weighted)