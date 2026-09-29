from pathlib import Path
import joblib

def save_model(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)

def save_bundle(model, features, path):
    bundle = {"model": model, "features": features}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)

def load_bundle(path):
    return joblib.load(path)
