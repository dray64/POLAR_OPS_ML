import pandas as pd

TARGET = "high_risk_label"

def create_features(df: pd.DataFrame):
    x = df.copy()
    x["date"] = pd.to_datetime(x["date"])
    features = [
        "mean_weather_risk",
        "storm_flag",
        "active_missions",
        "mean_stock_ratio",
        "operational_risk_score",
    ]
    return x, features
