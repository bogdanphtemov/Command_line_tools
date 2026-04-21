import pickle
from pathlib import Path
from typing import Dict , Type
from .base_models import BaseModel

# model registry (just a dict, no magic)
MODEL_MAP: Dict[str , Type[BaseModel]] = {}

# current version of the model format
MODEL_FORMAT_VERSION = 1

# register the model in the registry
def register_model(model_type: str , model_class: Type[BaseModel]) -> None:
    MODEL_MAP[model_type] = model_class
    print(f"Model registered: {model_type}")

# save model to file
def save_model(model: BaseModel , filepath: str) -> None:
    if not model.is_trained:
        raise ValueError("!Model is not trained yet!")
    if model.model_type is None:
        raise AttributeError("!Model must have model_type attribute!")
    
    model_data = {
        "version": MODEL_FORMAT_VERSION,
        "model_type": model.model_type,
        "params": model.get_params()
    }

    Path(filepath).parent.mkdir(parents=True , exist_ok=True)
    with open(filepath , "wb") as f:
        pickle.dump(model_data , f)

    print(f"Model saved to {filepath}")
# load model from file
def load_model(filepath: str) -> BaseModel:
    print("Warning: loading pickle file can be unsafe if from untrusted source!")

    if not Path(filepath).exists():
        raise FileNotFoundError(f"!Model file not found: {filepath}")
    
    with open(filepath , "rb") as f:
        model_data = pickle.load(f)

    version = model_data.get("version" , 0)
    if version != MODEL_FORMAT_VERSION:
        print(f"!Warning: model was saved with version {version}, current version {MODEL_FORMAT_VERSION}")

    model_type = model_data.get("model_type")
    if model_type is None:
        raise KeyError("!Model file is corrupted: missing model_type!")
        
    params = model_data.get("params")
    if params is None:
        raise KeyError("!Model file is corrupted: missing params!")
    
    if model_type not in MODEL_MAP:
        raise ValueError(f"!Unknown model type: {model_type}!")
    
    model_class = MODEL_MAP[model_type]

    model = model_class()
    model.set_params(params)
    print(f"Model loaded from {filepath}")
    return model

# display all saved models in a folder
def list_saved_models(directory: str = "./ml_models") -> list:
    dir_path = Path(directory)

    if not dir_path.exists():
        return []
    
    return [f.stem for f in dir_path.glob("*.pkl")]

# delete model
def delete_model(filepath: str) -> None:
    path = Path(filepath)

    if path.exists():
        path.unlink()
        print(f"Model deleted: {filepath}")
    else:
        print(f"!Model not found: {filepath}!")


