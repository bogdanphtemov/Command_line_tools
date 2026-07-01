"""
Base Session Adapter — shared logic for all model session adapters.

Eliminates code duplication between LinearRegressionSessionAdapter
and LogisticRegressionSessionAdapter (and any future adapters).
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
from collections import defaultdict, deque

from ML.session_storage import (
    SessionAdapter, SessionData, SessionMetadata, TrainingConfig
)


def _map_rows_to_indices(full: np.ndarray, subset: np.ndarray) -> List[int]:
    """
    Fast mapping by hashing each row's bytes to queues of indices.

    Complexity: O(n) to build the map + O(m) to assign indices for subset.
    Raises ValueError when a row cannot be matched.
    """
    if full.ndim != 2 or subset.ndim != 2:
        raise ValueError("Expected 2D arrays for row mapping")

    if full.shape[1] != subset.shape[1]:
        raise ValueError("Feature dimension mismatch between full and subset arrays")

    mapping = defaultdict(deque)
    for i, row in enumerate(full):
        mapping[row.tobytes()].append(i)

    indices: List[int] = []
    for row in subset:
        key = row.tobytes()
        if not mapping[key]:
            raise ValueError(
                "Cannot map subset row to prepared X: possible duplicate or mismatch"
            )
        indices.append(mapping[key].popleft())

    return indices


class BaseSessionAdapter(SessionAdapter):
    """
    Configurable base adapter for supervised model session serialization.

    Subclasses only need to provide:
      - algorithm_name (str, e.g., "linear_regression")
      - hyperparams_config (list of dicts describing hyperparams)
      - model_class (the model class to instantiate on restore)
      - dataset_class, prepareddata_class (for reconstructing data objects)
      - preprocessing_module (optional, for standardize_apply import)
    """

    # Override in subclass
    algorithm_name: str = ""
    model_class = None
    dataset_class = None
    prepareddata_class = None
    preprocessing_module = None  # module path for standardize_apply

    # List of hyperparam specs: {"name": str, "default": Any, "aliases": Optional[List[str]]}
    hyperparam_specs: List[Dict[str, Any]] = []

    def extract(self, app_state: Any) -> Tuple[SessionData, Dict[str, np.ndarray]]:
        """Extract SessionData and arrays_dict from AppState."""
        # Validate state
        if app_state.dataset is None:
            raise ValueError("No dataset loaded!")
        if app_state.prepareddata is None:
            raise ValueError("No features/target selected!")
        if app_state.model is None or not app_state.model.is_trained:
            raise ValueError("Model not trained yet!")

        dataset_array = app_state.dataset.data
        X_data = app_state.prepareddata.X
        Y_data = app_state.prepareddata.Y

        # Prefer explicit indices saved in AppState (fast, exact)
        train_indices: Optional[List[int]] = None
        test_indices: Optional[List[int]] = None

        if (
            getattr(app_state, "train_idx", None) is not None
            and getattr(app_state, "test_idx", None) is not None
        ):
            train_indices = [int(x) for x in np.asarray(app_state.train_idx).tolist()]
            test_indices = [int(x) for x in np.asarray(app_state.test_idx).tolist()]
        else:
            try:
                if app_state.X_train is not None and app_state.X_test is not None:
                    if X_data.shape[0] == (app_state.X_train.shape[0] + app_state.X_test.shape[0]):
                        if np.array_equal(
                            np.vstack([app_state.X_train, app_state.X_test]), X_data
                        ):
                            train_indices = list(range(app_state.X_train.shape[0]))
                            test_indices = list(
                                range(app_state.X_train.shape[0], X_data.shape[0])
                            )
                        else:
                            train_indices = _map_rows_to_indices(X_data, app_state.X_train)
                            test_indices = _map_rows_to_indices(X_data, app_state.X_test)
            except Exception:
                return self._fallback_extract(
                    app_state, dataset_array, X_data, Y_data
                )

        # Build arrays dict (canonical X/Y only)
        arrays_dict: Dict[str, np.ndarray] = {
            "dataset": dataset_array,
            "X": X_data,
            "Y": Y_data,
        }

        if getattr(app_state, "scaler_mean", None) is not None:
            arrays_dict["scaler_mean"] = app_state.scaler_mean
        if getattr(app_state, "scaled_std", None) is not None:
            arrays_dict["scaled_std"] = app_state.scaled_std

        # Model params
        model_params = app_state.model.get_params()
        if model_params.get("w") is not None:
            arrays_dict["model_w"] = np.array(model_params["w"])
        model_params_json = {k: v for k, v in model_params.items() if k != "w"}

        training_config = self._build_training_config(app_state, model_params_json)

        metadata = SessionMetadata(
            algorithm=self.algorithm_name,
            timestamp=datetime.now().isoformat(),
            random_seed=app_state.seed,
        )

        session_data = SessionData(
            dataset_columns=app_state.dataset.columns,
            feature_names=app_state.prepareddata.feature_names,
            target_name=app_state.prepareddata.target_name,
            train_indices=train_indices,
            test_indices=test_indices,
            use_scaling=app_state.use_scaling,
            model_type=self.algorithm_name,
            model_trained=app_state.model.is_trained,
            training_config=training_config,
            metrics=app_state.metrics if app_state.metrics else {},
            metadata=metadata,
            arrays_keys={"model_w": "model_w"} if model_params.get("w") is not None else {},
        )

        return session_data, arrays_dict

    def _fallback_extract(
        self,
        app_state: Any,
        dataset_array: np.ndarray,
        X_data: np.ndarray,
        Y_data: np.ndarray,
    ) -> Tuple[SessionData, Dict[str, np.ndarray]]:
        """Fallback when index mapping fails: store explicit split arrays."""
        arrays_dict: Dict[str, np.ndarray] = {
            "dataset": dataset_array,
            "X": X_data,
            "Y": Y_data,
            "X_train": app_state.X_train,
            "X_test": app_state.X_test,
            "y_train": app_state.y_train,
            "y_test": app_state.y_test,
        }

        if getattr(app_state, "scaler_mean", None) is not None:
            arrays_dict["scaler_mean"] = app_state.scaler_mean
        if getattr(app_state, "scaled_std", None) is not None:
            arrays_dict["scaled_std"] = app_state.scaled_std

        model_params = app_state.model.get_params()
        if model_params.get("w") is not None:
            arrays_dict["model_w"] = np.array(model_params["w"])
        model_params_json = {k: v for k, v in model_params.items() if k != "w"}

        training_config = self._build_training_config(app_state, model_params_json)

        metadata = SessionMetadata(
            algorithm=self.algorithm_name,
            timestamp=datetime.now().isoformat(),
            random_seed=app_state.seed,
        )

        session_data = SessionData(
            dataset_columns=app_state.dataset.columns,
            feature_names=app_state.prepareddata.feature_names,
            target_name=app_state.prepareddata.target_name,
            train_indices=None,
            test_indices=None,
            use_scaling=app_state.use_scaling,
            model_type=self.algorithm_name,
            model_trained=app_state.model.is_trained,
            training_config=training_config,
            metrics=app_state.metrics if app_state.metrics else {},
            metadata=metadata,
            arrays_keys={"model_w": "model_w"} if model_params.get("w") is not None else {},
        )

        return session_data, arrays_dict

    def _build_training_config(
        self, app_state: Any, model_params_json: Dict[str, Any]
    ) -> TrainingConfig:
        """Build TrainingConfig from AppState using hyperparam_specs."""
        hyperparams: Dict[str, Any] = {
            "learning_rate": app_state.learning_rate,
            "epochs": app_state.epochs,
            "test_size": app_state.test_size,
            "seed": app_state.seed,
            "model_params": model_params_json,
        }
        # Add algorithm-specific hyperparams
        for spec in self.hyperparam_specs:
            name = spec["name"]
            value = getattr(app_state, name, spec.get("default"))
            hyperparams[name] = value

        return TrainingConfig(hyperparams=hyperparams)

    def _extract_hyperparams(self, hyperparams: Dict[str, Any]) -> Dict[str, Any]:
        """Extract algorithm-specific hyperparams from the config dict."""
        result: Dict[str, Any] = {}
        for spec in self.hyperparam_specs:
            name = spec["name"]
            default = spec.get("default")
            # Check aliases for backward compatibility
            aliases = spec.get("aliases", [])
            value = hyperparams.get(name)
            if value is None:
                for alias in aliases:
                    value = hyperparams.get(alias)
                    if value is not None:
                        break
            result[name] = value if value is not None else default
        return result

    def restore(self, session_data: SessionData, arrays: Dict[str, np.ndarray], app_state: Any) -> None:
        """Restore AppState from SessionData and arrays."""
        # Validate session structure
        self.validate_session(session_data)

        # Restore dataset and prepared data
        dataset_array = arrays.get("dataset")
        if dataset_array is None:
            raise ValueError("Missing 'dataset' in arrays")
        app_state.dataset = self.dataset_class(
            data=dataset_array, columns=session_data.dataset_columns
        )

        X = arrays.get("X")
        Y = arrays.get("Y")
        if X is None or Y is None:
            raise ValueError("Missing prepared arrays X/Y in session data")

        app_state.prepareddata = self.prepareddata_class(
            X=X,
            Y=Y,
            feature_names=session_data.feature_names,
            target_name=session_data.target_name,
        )

        # Preprocessing state
        app_state.use_scaling = session_data.use_scaling
        if "scaler_mean" in arrays:
            app_state.scaler_mean = arrays["scaler_mean"]
        if "scaled_std" in arrays:
            app_state.scaled_std = arrays["scaled_std"]

        # Reconstruct train/test split using indices (preferred)
        if session_data.train_indices is not None and session_data.test_indices is not None:
            train_idx = np.array(session_data.train_indices, dtype=int)
            test_idx = np.array(session_data.test_indices, dtype=int)

            # raw slices from canonical X/Y
            X_train_raw = X[train_idx]
            X_test_raw = X[test_idx]

            # apply scaling if needed
            if session_data.use_scaling and getattr(app_state, "scaler_mean", None) is not None and getattr(app_state, "scaled_std", None) is not None:
                if self.preprocessing_module is not None:
                    app_state.X_train = self.preprocessing_module.standardize_apply(
                        X_train_raw, app_state.scaler_mean, app_state.scaled_std
                    )
                    app_state.X_test = self.preprocessing_module.standardize_apply(
                        X_test_raw, app_state.scaler_mean, app_state.scaled_std
                    )
                else:
                    app_state.X_train = X_train_raw
                    app_state.X_test = X_test_raw
            else:
                app_state.X_train = X_train_raw
                app_state.X_test = X_test_raw

            app_state.y_train = Y[train_idx]
            app_state.y_test = Y[test_idx]
            app_state.train_idx = train_idx
            app_state.test_idx = test_idx
        else:
            # Fallback: session stored explicit split arrays
            if "X_train" in arrays and "X_test" in arrays:
                app_state.X_train = arrays["X_train"]
                app_state.X_test = arrays["X_test"]
                app_state.y_train = arrays.get("y_train")
                app_state.y_test = arrays.get("y_test")
            else:
                raise ValueError(
                    "Missing split info: neither indices nor explicit split arrays "
                    "were found in session"
                )

        # Training config
        hyperparams = session_data.training_config.hyperparams
        app_state.learning_rate = hyperparams.get("learning_rate", 0.05)
        app_state.epochs = hyperparams.get("epochs", 2000)
        app_state.test_size = hyperparams.get("test_size", 0.2)
        app_state.seed = hyperparams.get("seed", 42)

        # Set algorithm-specific hyperparams
        for name, value in self._extract_hyperparams(hyperparams).items():
            setattr(app_state, name, value)

        # Metrics
        app_state.metrics = session_data.metrics if session_data.metrics else {}

        # Restore model instance and params
        model_params = hyperparams.get("model_params", {})
        if "model_w" in arrays:
            model_params["w"] = arrays["model_w"].tolist()

        app_state.model = self.model_class()
        app_state.model.set_params(model_params)

    def validate_session(self, session_data: SessionData) -> bool:
        """Validate algorithm-specific requirements."""
        super().validate_session(session_data)
        hyperparams = session_data.training_config.hyperparams
        if "learning_rate" not in hyperparams:
            raise ValueError("!Missing learning_rate hyperparameter!")
        if "epochs" not in hyperparams:
            raise ValueError("!Missing epochs hyperparameter!")
        # Validate algorithm-specific required params
        for spec in self.hyperparam_specs:
            name = spec["name"]
            if spec.get("required", False) and name not in hyperparams:
                raise ValueError(f"!Missing required hyperparameter: {name}!")
        return True
