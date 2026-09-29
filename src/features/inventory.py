import pandas as pd

TARGET = "next_day_demand"

def create_features(df: pd.DataFrame):
    x = df.copy()
    x["date"] = pd.to_datetime(x["date"])
    x = x.sort_values(["station_id", "item_id", "date"])
    g = x.groupby(["station_id", "item_id"], group_keys=False)

    x["lag_1"] = g["consumption_qty"].shift(1)
    x["lag_7"] = g["consumption_qty"].shift(7)
    x["lag_14"] = g["consumption_qty"].shift(14)
    x["rolling_mean_7"] = g["consumption_qty"].transform(
        lambda s: s.shift(1).rolling(7, min_periods=3).mean()
    )
    x["rolling_mean_30"] = g["consumption_qty"].transform(
        lambda s: s.shift(1).rolling(30, min_periods=7).mean()
    )
    x["rolling_std_7"] = g["consumption_qty"].transform(
        lambda s: s.shift(1).rolling(7, min_periods=3).std()
    ).fillna(0)

    x["next_day_demand"] = g["consumption_qty"].shift(-1)

    # Context variables already present in the dataset.
    features = [
        "consumption_qty", "weather_risk", "storm_flag",
        "personnel_present", "mission_activity_index",
        "lag_1", "lag_7", "lag_14",
        "rolling_mean_7", "rolling_mean_30", "rolling_std_7",
    ]
    return x.dropna(subset=["next_day_demand"]), features
