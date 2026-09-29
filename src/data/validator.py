from pathlib import Path
import pandas as pd

EXPECTED = {
    "asset_telemetry.csv": ["timestamp", "asset_id", "failure_event"],
    "inventory_consumption.csv": ["date", "station_id", "item_id", "consumption_qty"],
    "missions.csv": ["mission_id", "start_time", "delayed_flag"],
    "operational_risk_labels.csv": ["date", "station_id", "high_risk_label"],
}

def validate_dataframe(df: pd.DataFrame, name: str):
    problems = []
    if df.empty:
        problems.append("dataset is empty")
    duplicate_count = int(df.duplicated().sum())
    missing = df.isna().sum()
    return {
        "name": name,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "duplicates": duplicate_count,
        "missing_total": int(missing.sum()),
        "missing_columns": missing[missing > 0].to_dict(),
        "problems": problems,
    }

def validate_all(data):
    results = [validate_dataframe(df, name) for name, df in data.items()]
    return results
