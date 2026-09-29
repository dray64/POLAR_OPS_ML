import json
from pathlib import Path
import numpy as np

def json_safe(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    return value

def save_metrics(metrics, filename):
    path = Path("reports/metrics") / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(json_safe(metrics), f, indent=2)

def save_report(report):
    path = Path("reports/ML_REPORT.json")
    with path.open("w", encoding="utf-8") as f:
        json.dump(json_safe(report), f, indent=2)
