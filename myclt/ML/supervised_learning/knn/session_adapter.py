"""
Session adapter for K-Nearest Neighbors.
"""

from typing import List, Dict, Any
from .data import Dataset, Prepareddata
from .classifier import KNNClassifier
from .regressor import KNNRegressor
from . import preprocessing

from myclt.ML.session_storage import SessionData, TrainingConfig
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
        {"name": "p", "default": 2, "applies_to": "minkowski"},
    ]

    def _set_model_class(self, model_type: str) -> None:
        """
        Set the appropriate model class based on model_type.
        
        Args:
            model_type: Either 'classification' or 'regression'
        """
        if model_type == 'classification':
            self.model_class = KNNClassifier
        else:
            self.model_class = KNNRegressor
    
    def _build_training_config(self, app_state, model_params_json) -> TrainingConfig:
        """KNN has no learning_rate/epochs - provide neutral placeholders for schema compatibility."""
        hyperparams = {
            "learning_rate": 0.0,
            "epochs": 0,
            "test_size": app_state.test_size,
            "seed": app_state.seed,
            "model_params": model_params_json,
        }
        for spec in self.hyperparam_specs:
            name = spec["name"]
            hyperparams[name] = getattr(app_state, name, spec.get("default"))
        return TrainingConfig(hyperparams=hyperparams)

    def _validate_weighted_parameter(self, model_type: str, hyperparams: Dict[str, Any]) -> None:
        """
        Validate that weighted parameter is only used for regression.
        
        Args:
            model_type: Current model type
            hyperparams: Hyperparameters to validate
            
        Raises:
            ValueError: If weighted parameter is used incorrectly
        """
        weighted = hyperparams.get('weighted', False)
        if model_type == 'classification' and weighted:
            raise ValueError("weighted parameter is only supported for regression")

    def _create_model(self, hyperparams: Dict[str, Any]):
        """
        Create a model instance from hyperparameters.
        
        Args:
            hyperparams: Model hyperparameters
            
        Returns:
            KNNClassifier or KNNRegressor instance
        """
        model_type = hyperparams.get('model_type', 'classification')
        k = hyperparams.get('k', 5)
        distance_metric = hyperparams.get('distance_metric', 'euclidean')
        p = hyperparams.get('p', 2)
        
        if model_type == 'classification':
            return KNNClassifier(k=k, distance_metric=distance_metric, p=p)
        else:
            weighted = hyperparams.get('weighted', False)
            return KNNRegressor(k=k, distance_metric=distance_metric, weighted=weighted, p=p)

    def validate_session(self, session_data: SessionData) -> bool:
        """
        Validate KNN specific requirements.
        
        Side effect:
            Sets self.model_class based on model_type.
        """
        # Ensure base validation passes first
        hyperparams = session_data.training_config.hyperparams
        
        # KNN has no learning_rate/epochs - add placeholders for compatibility
        hyperparams.setdefault("learning_rate", 0.0)
        hyperparams.setdefault("epochs", 0)
        
        # First validate using base class (which will raise ValueError if invalid)
        super().validate_session(session_data)
        
        # Base class already validated that training_config.hyperparams exists
        
        # Validate that model_type is valid
        model_type = hyperparams.get('model_type')
        if model_type not in ['classification', 'regression']:
            raise ValueError(f"Invalid model_type: {model_type}")
        
        # Validate that weighted parameter is only used for regression
        self._validate_weighted_parameter(model_type, hyperparams)
        
        # Set appropriate model class dynamically
        self._set_model_class(model_type)
            
        return True
    def restore(self, session_data, arrays, app_state) -> None:
        """Restore session and clean up unused attributes for KNN."""
        super().restore(session_data, arrays, app_state)
        # Clean up unused gradient-based attributes from base class
        for name in ("learning_rate", "epochs"):
            app_state.__dict__.pop(name, None)
    
    def instantiate_model(self, algorithm_specific: Dict[str, Any]) -> object:
        """Create model instance from saved parameters."""
        # Note: This method is currently not used by BaseSessionAdapter
        # but retained for backward compatibility.
        # BaseSessionAdapter creates models via self.model_class() 
        # and calls set_params() to configure them.
        
        # Use the unified model creation logic
        return self._create_model(algorithm_specific)