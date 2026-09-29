from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "raw" / "ncpop"

FILES = {
    "stations": "stations.csv",
    "personnel": "personnel.csv",
    "assets": "assets.csv",
    "asset_telemetry": "asset_telemetry.csv",
    "weather": "weather_observations.csv",
    "inventory": "inventory_master.csv",
    "inventory_consumption": "inventory_consumption.csv",
    "missions": "missions.csv",
    "cargo": "cargo_tracking.csv",
    "personnel_movement": "personnel_movement.csv",
    "emergency": "emergency_incidents.csv",
    "scientific_instruments": "scientific_instruments.csv",
    "scientific_telemetry": "scientific_telemetry.csv",
    "routes": "route_segments.csv",
    "risk_labels": "operational_risk_labels.csv",
}

def load_dataset(filename: str) -> pd.DataFrame:
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)

def load_all():
    return {name: load_dataset(filename) for name, filename in FILES.items()}
