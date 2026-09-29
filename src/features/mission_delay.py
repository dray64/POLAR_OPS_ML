import pandas as pd
import numpy as np

TARGET = "delayed_flag"

def create_features(df: pd.DataFrame):
    x = df.copy()
    x["start_time"] = pd.to_datetime(x["start_time"])

    x["route_condition_code"] = x["route_condition"].astype("category").cat.codes
    x["mission_type_code"] = x["mission_type"].astype("category").cat.codes

    x["cargo_per_person"] = x["cargo_weight_kg"] / x["crew_size"].clip(lower=1)
    x["duration_buffer_h"] = x["planned_duration_h"] - x["distance_km"].clip(lower=0) / 25.0
    x["weather_distance_interaction"] = (
        x["weather_risk_score"] * x["distance_km"]
    )

    features = [
        "distance_km",
        "weather_risk_score",
        "crew_size",
        "cargo_weight_kg",
        "planned_duration_h",
        "route_condition_code",
        "mission_type_code",
        "cargo_per_person",
        "duration_buffer_h",
        "weather_distance_interaction",
    ]
    return x, features
