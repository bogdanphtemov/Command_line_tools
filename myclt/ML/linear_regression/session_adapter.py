import numpy as np
from typing import List, Dict, Any, Tuple
from datetime import datetime

from ML.session_storage import (
    SessionAdapter, SessionData, SessionMetadata, TrainingConfig
)


def _map_rows_to_indices(full: np.ndarray, subset: np.ndarray) -> List[int]:
    """
    Fast mapping by hashing each row's bytes to queues of indices.

    Complexity: O(n) to build the map + O(m) to assign indices for subset.
    Raises ValueError when a row cannot be matched.
    """
    from collections import defaultdict, deque

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
            raise ValueError("Cannot map subset row to prepared X: possible duplicate or mismatch")
        indices.append(mapping[key].popleft())

    return indices


class LinearRegressionSessionAdapter(SessionAdapter):
    """Adapter for Linear Regression session serialization"""

    def extract(self, app_state: Any) -> Tuple[SessionData, Dict[str, np.ndarray]]:
        """
        Extract SessionData and arrays_dict from AppState.

        Produces:
          - session_data (with train/test indices into prepared `X`)
          - arrays_dict containing `X`, `Y`, optional scalers and model arrays
        """
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
        train_indices = None
        test_indices = None

        if getattr(app_state, "train_idx", None) is not None and getattr(app_state, "test_idx", None) is not None:
            train_indices = [int(x) for x in np.asarray(app_state.train_idx).tolist()]
            test_indices = [int(x) for x in np.asarray(app_state.test_idx).tolist()]
        else:
            # Fast-path: if concatenation of X_train/X_test equals X_data (order preserved)
            try:
                if app_state.X_train is not None and app_state.X_test is not None:
                    if X_data.shape[0] == (app_state.X_train.shape[0] + app_state.X_test.shape[0]):
                        if np.array_equal(np.vstack([app_state.X_train, app_state.X_test]), X_data):
                            train_indices = list(range(app_state.X_train.shape[0]))
                            test_indices = list(range(app_state.X_train.shape[0], X_data.shape[0]))
                        else:
                            # General fast mapping using bytes-hash queues
                            train_indices = _map_rows_to_indices(X_data, app_state.X_train)
                            test_indices = _map_rows_to_indices(X_data, app_state.X_test)
            except Exception:
                # mapping failed -> fallback to storing explicit split arrays
                arrays_dict = {
                    "dataset": dataset_array,
                    "X": X_data,
                    "Y": Y_data,
                    "X_train": app_state.X_train,
                    "X_test": app_state.X_test,
                    "y_train": app_state.y_train,
                    "y_test": app_state.y_test,
                }
                # include scalers and model arrays as before
                if getattr(app_state, "scaler_mean", None) is not None:
                    arrays_dict["scaler_mean"] = app_state.scaler_mean
                if getattr(app_state, "scaled_std", None) is not None:
                    arrays_dict["scaler_std"] = app_state.scaled_std

                model_params = app_state.model.get_params()
                if model_params.get("w") is not None:
                    arrays_dict["model_w"] = np.array(model_params["w"])
                model_params_json = {k: v for k, v in model_params.items() if k != "w"}

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

                metadata = SessionMetadata(
                    algorithm="linear_regression",
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
                    model_type="linear_regression",
                    model_trained=app_state.model.is_trained,
                    training_config=training_config,
                    metrics=app_state.metrics if app_state.metrics else {},
                    metadata=metadata,
                    arrays_keys={"model_w": "model_w"} if model_params.get("w") is not None else {},
                )

                return session_data, arrays_dict

        # Build arrays dict (canonical X/Y only; X_train/X_test included only on fallback)
        arrays_dict: Dict[str, np.ndarray] = {
            "dataset": dataset_array,
            "X": X_data,
            "Y": Y_data,
        }

        if getattr(app_state, "scaler_mean", None) is not None:
            arrays_dict["scaler_mean"] = app_state.scaler_mean
        if getattr(app_state, "scaled_std", None) is not None:
            arrays_dict["scaler_std"] = app_state.scaled_std

        # Model params
        model_params = app_state.model.get_params()
        if model_params.get("w") is not None:
            arrays_dict["model_w"] = np.array(model_params["w"])
        model_params_json = {k: v for k, v in model_params.items() if k != "w"}

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

        metadata = SessionMetadata(
            algorithm="linear_regression",
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
            model_type="linear_regression",
            model_trained=app_state.model.is_trained,
            training_config=training_config,
            metrics=app_state.metrics if app_state.metrics else {},
            metadata=metadata,
            arrays_keys={"model_w": "model_w"} if model_params.get("w") is not None else {},
        )

        return session_data, arrays_dict

    def restore(self, session_data: SessionData, arrays: Dict[str, np.ndarray], app_state: Any) -> None:
        """
        Restore AppState from SessionData and arrays.
        Creates a model instance and sets parameters.
        """
        # Validate session structure
        self.validate_session(session_data)

        # Lazy import to avoid circular imports
        from .data import Dataset, Prepareddata
        from .core import LinearRegressionGD

        # Restore dataset and prepared data
        dataset_array = arrays.get("dataset")
        if dataset_array is None:
            raise ValueError("Missing 'dataset' in arrays")
        app_state.dataset = Dataset(data=dataset_array, columns=session_data.dataset_columns)

        X = arrays.get("X")
        Y = arrays.get("Y")
        if X is None or Y is None:
            raise ValueError("Missing prepared arrays X/Y in session data")

        app_state.prepareddata = Prepareddata(X=X, Y=Y, feature_names=session_data.feature_names, target_name=session_data.target_name)

        # Preprocessing state
        app_state.use_scaling = session_data.use_scaling
        if "scaler_mean" in arrays:
            app_state.scaler_mean = arrays["scaler_mean"]
        if "scaler_std" in arrays:
            app_state.scaled_std = arrays["scaler_std"]

        # Reconstruct train/test split using indices (preferred)
        if session_data.train_indices is not None and session_data.test_indices is not None:
            train_idx = np.array(session_data.train_indices, dtype=int)
            test_idx = np.array(session_data.test_indices, dtype=int)

            # raw slices from canonical X/Y
            X_train_raw = X[train_idx]
            X_test_raw = X[test_idx]

            # apply scaling if needed
            if session_data.use_scaling and getattr(app_state, "scaler_mean", None) is not None and getattr(app_state, "scaled_std", None) is not None:
                from .preprocessing import standardize_apply
                app_state.X_train = standardize_apply(X_train_raw, app_state.scaler_mean, app_state.scaled_std)
                app_state.X_test = standardize_apply(X_test_raw, app_state.scaler_mean, app_state.scaled_std)
            else:
                app_state.X_train = X_train_raw
                app_state.X_test = X_test_raw

            app_state.y_train = Y[train_idx]
            app_state.y_test = Y[test_idx]
            # persist indices on AppState
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
                raise ValueError("Missing split info: neither indices nor explicit split arrays were found in session")

        # Training config
        hyperparams = session_data.training_config.hyperparams
        app_state.learning_rate = hyperparams.get("learning_rate", 0.05)
        app_state.epochs = hyperparams.get("epochs", 2000)
        app_state.test_size = hyperparams.get("test_size", 0.2)
        app_state.seed = hyperparams.get("seed", 42)

        # Regularization
        app_state.use_l1 = hyperparams.get("use_l1", False)
        app_state.lambda_l1 = hyperparams.get("lambda_l1", 0.01)
        app_state.use_l2 = hyperparams.get("use_l2", False)
        app_state.lambda_l2 = hyperparams.get("lambda_l2", 0.01)

        # Metrics
        app_state.metrics = session_data.metrics if session_data.metrics else {}

        # Restore model instance and params
        model_params = hyperparams.get("model_params", {})
        if "model_w" in arrays:
            model_params["w"] = arrays["model_w"].tolist()

        app_state.model = LinearRegressionGD()
        app_state.model.set_params(model_params)

    def validate_session(self, session_data: SessionData) -> bool:
        """Validate Linear Regression specific requirements"""
        super().validate_session(session_data)
        hyperparams = session_data.training_config.hyperparams
        if "learning_rate" not in hyperparams:
            raise ValueError("!Missing learning_rate hyperparameter!")
        if "epochs" not in hyperparams:
            raise ValueError("!Missing epochs hyperparameter!")
        return True
