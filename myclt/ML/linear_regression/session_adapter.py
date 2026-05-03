"""
LinearRegression-specific SessionAdapter.

Handles conversion between AppState ↔ SessionData.
Keeps storage system clean and algorithm-agnostic.
"""

import numpy as np
from typing import List, Dict, Any
from datetime import datetime

from ML.session_storage import (
    SessionAdapter, SessionData, SessionMetadata, 
    TrainingConfig, _reconstruct_split
)
from .data import Dataset, Prepareddata


class LinearRegressionSessionAdapter(SessionAdapter):
    """Adapter for Linear Regression session serialization"""
    
    def extract(self, app_state: Any) -> SessionData:
        """
        Extract SessionData from AppState.
        
        Efficiently stores dataset once, reconstructs splits from indices.
        """
        # Validate state
        if app_state.dataset is None:
            raise ValueError("No dataset loaded!")
        if app_state.prepareddata is None:
            raise ValueError("No features/target selected!")
        if app_state.model is None or not app_state.model.is_trained:
            raise ValueError("Model not trained yet!")
        
        # Calculate split indices (efficient reconstruction)
        dataset_array = app_state.dataset.data
        X_data = app_state.prepareddata.X
        
        # Find which original rows correspond to prepared features
        train_indices = list(range(app_state.X_train.shape[0]))
        test_indices = list(range(app_state.X_train.shape[0], 
                                   app_state.X_train.shape[0] + app_state.X_test.shape[0]))
        
        # Collect all arrays for NPZ storage
        arrays_dict = {
            "dataset": dataset_array,
            "X": X_data,
            "Y": app_state.prepareddata.Y,
            "X_train": app_state.X_train,
            "X_test": app_state.X_test,
            "y_train": app_state.y_train,
            "y_test": app_state.y_test,
        }
        
        # Scaler parameters
        if app_state.scaler_mean is not None:
            arrays_dict["scaler_mean"] = app_state.scaler_mean
        if app_state.scaled_std is not None:
            arrays_dict["scaler_std"] = app_state.scaled_std
        
        # Model weights/params
        model_params = app_state.model.get_params()
        
        # Store numpy arrays in model_params
        if model_params.get("w") is not None:
            arrays_dict["model_w"] = np.array(model_params["w"])
        
        model_params_json = {k: v for k, v in model_params.items() if k != "w"}
        
        # Training config with algorithm-specific params
        training_config = TrainingConfig(
            hyperparams={
                "learning_rate": app_state.learning_rate,
                "epochs": app_state.epochs,
                "use_l1": app_state.use_l1,
                "lambda_l1": app_state.lambda_l1,
                "use_l2": app_state.use_l2,
                "lambda_l2": app_state.lambda_l2,
                "test_size": app_state.test_size,
                "seed": app_state.seed,
                "model_params": model_params_json,
            }
        )
        
        # Metadata
        metadata = SessionMetadata(
            algorithm="linear_regression",
            timestamp=datetime.now().isoformat(),
            random_seed=app_state.seed,
        )
        
        # Create SessionData
        session_data = SessionData(
            dataset_columns=app_state.dataset.columns,
            feature_names=app_state.prepareddata.feature_names,
            target_name=app_state.prepareddata.target_name,
            train_indices=train_indices,
            test_indices=test_indices,
            use_scaling=app_state.use_scaling,
            model_type="linear_regression",
            model_trained=app_state.model.is_trained,
            training_config=training_config,
            metrics={
                "mse": app_state.last_mse,
                "rmse": app_state.last_rmse,
                "r2": app_state.last_r2,
            },
            metadata=metadata,
            arrays_keys={"model_w": "model_w"},
        )
        
        return session_data, arrays_dict
    
    def restore(self, session_data: SessionData, arrays_dict: Dict[str, np.ndarray], app_state: Any) -> None:
        """
        Restore AppState from SessionData and arrays.
        """
        # Validate
        self.validate_session(session_data)
        
        # Restore dataset
        dataset_array = arrays_dict["dataset"]
        app_state.dataset = Dataset(
            data=dataset_array,
            columns=session_data.dataset_columns,
        )
        
        # Restore prepared data
        X = arrays_dict.get("X")
        Y = arrays_dict.get("Y")
        
        if X is None or Y is None:
            raise ValueError("Missing prepared data arrays!")
        
        app_state.prepareddata = Prepareddata(
            X=X,
            Y=Y,
            feature_names=session_data.feature_names,
            target_name=session_data.target_name,
        )
        
        # Restore preprocessing state
        app_state.use_scaling = session_data.use_scaling
        
        if "scaler_mean" in arrays_dict:
            app_state.scaler_mean = arrays_dict["scaler_mean"]
        if "scaler_std" in arrays_dict:
            app_state.scaled_std = arrays_dict["scaler_std"]
        
        # Restore train/test split
        app_state.X_train = arrays_dict.get("X_train")
        app_state.X_test = arrays_dict.get("X_test")
        app_state.y_train = arrays_dict.get("y_train")
        app_state.y_test = arrays_dict.get("y_test")
        
        # Restore training config
        hyperparams = session_data.training_config.hyperparams
        app_state.learning_rate = hyperparams.get("learning_rate", 0.05)
        app_state.epochs = hyperparams.get("epochs", 2000)
        app_state.test_size = hyperparams.get("test_size", 0.2)
        app_state.seed = hyperparams.get("seed", 42)
        
        # Restore regularization
        app_state.use_l1 = hyperparams.get("use_l1", False)
        app_state.lambda_l1 = hyperparams.get("lambda_l1", 0.01)
        app_state.use_l2 = hyperparams.get("use_l2", False)
        app_state.lambda_l2 = hyperparams.get("lambda_l2", 0.01)
        
        # Restore metrics
        app_state.last_mse = session_data.metrics.get("mse")
        app_state.last_rmse = session_data.metrics.get("rmse")
        app_state.last_r2 = session_data.metrics.get("r2")
        
        # Restore model
        if session_data.model_trained and app_state.model is not None:
            model_params = hyperparams.get("model_params", {})
            
            # Add back the weight array
            if "model_w" in arrays_dict:
                model_params["w"] = arrays_dict["model_w"].tolist()
            
            app_state.model.set_params(model_params)
    
    def validate_session(self, session_data: SessionData) -> bool:
        """Validate Linear Regression specific requirements"""
        # Call parent validation
        super().validate_session(session_data)
        
        # Linear Regression specific checks
        hyperparams = session_data.training_config.hyperparams
        
        if "learning_rate" not in hyperparams:
            raise ValueError("Missing learning_rate hyperparameter!")
        
        if "epochs" not in hyperparams:
            raise ValueError("Missing epochs hyperparameter!")
        
        return True
