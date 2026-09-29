import pandas as pd

FEATURES = [
    "engine_temp_c",
    "vibration_mm_s",
    "battery_voltage_v",
    "fuel_level_pct",
    "operating_hours",
    "ambient_temp_c",
    "wind_speed_kmh",
    "health_score",
    "maintenance_required",
]

TARGET = "failure_event"

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["timestamp"] = pd.to_datetime(x["timestamp"])
    x = x.sort_values(["asset_id", "timestamp"])

    g = x.groupby("asset_id", group_keys=False)
    for col in ["engine_temp_c", "vibration_mm_s", "battery_voltage_v",
                "fuel_level_pct", "health_score"]:
        x[f"{col}_delta"] = g[col].diff().fillna(0)

    x["engine_temp_roll6"] = g["engine_temp_c"].transform(
        lambda s: s.rolling(6, min_periods=1).mean()
    )
    x["vibration_roll6"] = g["vibration_mm_s"].transform(
        lambda s: s.rolling(6, min_periods=1).mean()
    )
    x["health_roll6"] = g["health_score"].transform(
        lambda s: s.rolling(6, min_periods=1).mean()
    )

    feature_cols = FEATURES + [
        "engine_temp_c_delta",
        "vibration_mm_s_delta",
        "battery_voltage_v_delta",
        "fuel_level_pct_delta",
        "health_score_delta",
        "engine_temp_roll6",
        "vibration_roll6",
        "health_roll6",
    ]
    return x, feature_cols
