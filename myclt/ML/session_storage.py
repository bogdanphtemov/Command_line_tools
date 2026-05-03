import json
import shutil
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime
from abc import ABC, abstractmethod
import platform
import sys

# Will be imported from linear_regression module
from .linear_regression.data import Dataset, Prepareddata


@dataclass
class SessionMetadata:
    """Rich metadata for reproducibility and debugging"""
    algorithm: str
    timestamp: str  # ISO format
    version: int = 1
    
    # Reproducibility information
    numpy_version: str = ""
    python_version: str = ""
    random_seed: Optional[int] = None
    
    # Optional tracking
    description: str = ""
    git_commit: Optional[str] = None
    
    def __post_init__(self):
        """Auto-populate version info if not provided"""
        if not self.numpy_version:
            self.numpy_version = np.__version__
        if not self.python_version:
            self.python_version = platform.python_version()


@dataclass
class TrainingConfig:
    """Algorithm-agnostic training hyperparameters"""
    # Will be populated by adapter
    hyperparams: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionData:
    """
    Safe, efficient session snapshot.
    No duplication. Safe serialization format.
    """
    # Dataset (stored once, no duplication)
    dataset_columns: List[str]
    
    # Feature/target selection
    feature_names: List[str]
    target_name: str
    
    # Split indices (efficient reconstruction)
    train_indices: List[int]
    test_indices: List[int]
    
    # Preprocessing state
    use_scaling: bool
    
    # Model info
    model_type: str
    model_trained: bool
    
    # Training configuration
    training_config: TrainingConfig
    
    # Evaluation metrics
    metrics: Dict[str, Optional[float]]
    
    # Metadata
    metadata: SessionMetadata
    
    # Array indices for npz file
    # (dataset_data, X_data, Y_data, X_train, X_test, y_train, y_test, scaler_mean, scaler_std, model_params_arrays)
    arrays_keys: Dict[str, str] = field(default_factory=dict)


class SessionAdapter(ABC):
    """
    Abstract adapter for algorithm-specific session extraction/restoration.
    
    Each algorithm implements this to handle its own AppState ↔ SessionData conversion.
    Storage system stays clean and algorithm-agnostic.
    """
    
    @abstractmethod
    def extract(self, app_state: Any) -> SessionData:
        """
        Extract SessionData from algorithm's AppState.
        
        Args:
            app_state: Algorithm-specific state object
            
        Returns:
            SessionData ready for saving
        """
        raise NotImplementedError()
    
    @abstractmethod
    def restore(self, session_data: SessionData, app_state: Any) -> None:
        """
        Restore algorithm's AppState from SessionData.
        
        Args:
            session_data: Loaded SessionData
            app_state: Algorithm-specific state object to populate
        """
        raise NotImplementedError()
    
    def validate_session(self, session_data: SessionData) -> bool:
        """
        Validate session data integrity.
        Override in subclass for algorithm-specific checks.
        
        Returns:
            True if valid, raises ValueError if not
        """
        # Generic checks
        if session_data.train_indices is None or session_data.test_indices is None:
            raise ValueError("!Missing split indices!")
        
        if not session_data.feature_names:
            raise ValueError("!No features selected!")
        
        if session_data.target_name is None:
            raise ValueError("!No target selected!")
        
        return True


class SessionStorage:
    """
    Safe, efficient session management.
    
    Format: JSON metadata + NPZ arrays (no pickle)
    Directory structure:
      session_name/
        ├── metadata.json
        ├── data.npz
        └── model_params.json
    """
    
    STORAGE_VERSION = 2
    EXTENSION = ".session"
    
    def __init__(self):
        """Initialize adapter registry"""
        self.adapters: Dict[str, SessionAdapter] = {}
    
    def register_adapter(self, model_type: str, adapter: SessionAdapter) -> None:
        """Register adapter for algorithm"""
        self.adapters[model_type] = adapter
        print(f"Adapter registered: {model_type}.")
    
    def _validate_arrays(self, arrays_dict: Dict[str, np.ndarray]) -> None:
        """Verify data integrity after loading"""
        dataset = arrays_dict.get("dataset")
        if dataset is None:
            raise ValueError("!Missing dataset array!")
        
        X_data = arrays_dict.get("X")
        Y_data = arrays_dict.get("Y")
        
        if X_data is not None and Y_data is not None:
            if X_data.shape[0] != Y_data.shape[0]:
                raise ValueError(
                    f"Shape mismatch: X has {X_data.shape[0]} samples, "
                    f"Y has {Y_data.shape[0]} samples!"
                )
        
        # Check split arrays if present
        X_train = arrays_dict.get("X_train")
        X_test = arrays_dict.get("X_test")
        y_train = arrays_dict.get("y_train")
        y_test = arrays_dict.get("y_test")
        
        if all([X_train is not None, X_test is not None, y_train is not None, y_test is not None]):
            if X_train.shape[0] != y_train.shape[0]:
                raise ValueError("!X_train/y_train shape mismatch!")
            if X_test.shape[0] != y_test.shape[0]:
                raise ValueError("!X_test/y_test shape mismatch!")
    
    def save_session(
        self,
        session_data: SessionData,
        filepath: str,
        arrays_dict: Dict[str, np.ndarray],
        verbose: bool = True
    ) -> None:
        """
        Save session safely (JSON + NPZ, no pickle).
        
        Args:
            session_data: SessionData with metadata
            filepath: Base path (extension added automatically)
            arrays_dict: Dict of numpy arrays to save
            verbose: Print status
        """
        # Create directory
        session_dir = Path(filepath)
        session_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # 1. Save metadata as JSON
            metadata_path = session_dir / "metadata.json"
            metadata_dict = asdict(session_data.metadata)
            with open(metadata_path, "w") as f:
                json.dump(metadata_dict, f, indent=2)
            
            # 2. Save session config as JSON (without arrays)
            config_path = session_dir / "config.json"
            config_dict = {
                "dataset_columns": session_data.dataset_columns,
                "feature_names": session_data.feature_names,
                "target_name": session_data.target_name,
                "train_indices": session_data.train_indices,
                "test_indices": session_data.test_indices,
                "use_scaling": session_data.use_scaling,
                "model_type": session_data.model_type,
                "model_trained": session_data.model_trained,
                "training_config": asdict(session_data.training_config),
                "metrics": session_data.metrics,
                "version": self.STORAGE_VERSION,
            }
            with open(config_path, "w") as f:
                json.dump(config_dict, f, indent=2)
            
            # 3. Save all numpy arrays in NPZ (efficient + safe)
            arrays_path = session_dir / "data.npz"
            np.savez_compressed(arrays_path, **arrays_dict)
            
            if verbose:
                print(f"✓ Session saved to {session_dir}/")
                print(f"  - metadata.json: {metadata_path.stat().st_size / 1024:.1f} KB")
                print(f"  - config.json: {config_path.stat().st_size / 1024:.1f} KB")
                print(f"  - data.npz: {arrays_path.stat().st_size / 1024 / 1024:.1f} MB")
        
        except Exception as e:
            raise RuntimeError(f"!Failed to save session: {e}!")
    
    def load_session(
        self,
        filepath: str,
        verbose: bool = True
    ) -> Tuple[SessionData, Dict[str, np.ndarray]]:
        """
        Load session safely (JSON + NPZ, no pickle).
        
        Args:
            filepath: Base path to session directory
            verbose: Print status
            
        Returns:
            (SessionData, arrays_dict) tuple
        """
        session_dir = Path(filepath)
        
        if not session_dir.exists():
            raise FileNotFoundError(f"!Session directory not found: {session_dir}!")
        
        try:
            # 1. Load metadata
            metadata_path = session_dir / "metadata.json"
            with open(metadata_path, "r") as f:
                metadata_dict = json.load(f)
            metadata = SessionMetadata(**metadata_dict)
            
            # 2. Load config
            config_path = session_dir / "config.json"
            with open(config_path, "r") as f:
                config_dict = json.load(f)
            
            version = config_dict.pop("version", 1)
            
            # Migration: handle old versions
            if version < self.STORAGE_VERSION:
                config_dict = self._migrate_session(config_dict, version)
            
            # Parse training config - extract and unpack properly
            tc_dict = config_dict.pop("training_config", {})
            training_config = TrainingConfig(**tc_dict)
            
            # 3. Load arrays
            arrays_path = session_dir / "data.npz"
            arrays_npz = np.load(arrays_path)
            arrays_dict = {key: arrays_npz[key] for key in arrays_npz.files}
            
            # 4. Validate integrity
            self._validate_arrays(arrays_dict)
            
            # 5. Reconstruct SessionData
            session_data = SessionData(
                metadata=metadata,
                training_config=training_config,
                **config_dict
            )
            
            if verbose:
                print(f"✓ Session loaded from {session_dir}/")
                print(f"  - Algorithm: {session_data.model_type}")
                print(f"  - Model trained: {session_data.model_trained}")
                print(f"  - Arrays: {len(arrays_dict)} files")
            
            return session_data, arrays_dict
        
        except Exception as e:
            raise RuntimeError(f"!Failed to load session: {e}!")
    
    def _migrate_session(self, config_dict: Dict, version: int) -> Dict:
        """
        Handle schema migrations for backward compatibility.
        
        Args:
            config_dict: Session config from old version
            version: Version to migrate from
            
        Returns:
            Updated config_dict in current version format
        """
        if version == 1:
            # Migration v1 → v2
            print("!Migrating session from v1 to v2...!")
            
            # v1 had different structure, adapt it
            # (Implement based on actual v1 format if needed)
            
        return config_dict
    
    def list_sessions(self, directory: str = "./ml_sessions") -> List[str]:
        """List all saved sessions"""
        dir_path = Path(directory)
        
        if not dir_path.exists():
            return []
        
        return [
            d.name
            for d in dir_path.iterdir()
            if d.is_dir() and (d / "metadata.json").exists()
        ]
    
    def delete_session(self, filepath: str, verbose: bool = True) -> None:
        """Delete a session directory"""
        
        path = Path(filepath)
        
        if path.exists():
            shutil.rmtree(path)
            if verbose:
                print(f"Session deleted: {filepath}.")
        else:
            if verbose:
                print(f"!Session not found: {filepath}!")


# Helper for efficient dataset reconstruction
def _reconstruct_split(dataset_array: np.ndarray, train_indices: List[int], test_indices: List[int]) -> Tuple:
    """
    Efficiently reconstruct train/test split without duplication.
    
    Args:
        dataset_array: Full dataset (includes features only, not split into X/y)
        train_indices: Indices of training samples
        test_indices: Indices of test samples
        
    Returns:
        (X_train, X_test) - feature matrices for training and test sets
    """
    train_idx = np.array(train_indices, dtype=int)
    test_idx = np.array(test_indices, dtype=int)
    
    X_train = dataset_array[train_idx]
    X_test = dataset_array[test_idx]
    
    return X_train, X_test


# Global storage instance
_global_storage = SessionStorage()
