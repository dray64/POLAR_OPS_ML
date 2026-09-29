import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.loader import load_dataset

TARGETS = [
    ("asset_telemetry.csv", "failure_event"),
    ("missions.csv", "delayed_flag"),
    ("operational_risk_labels.csv", "high_risk_label"),
]

def main():
    for filename, target in TARGETS:
        df = load_dataset(filename)
        print(f"\n{filename} → {target}")
        print(df[target].value_counts(dropna=False))
        print((df[target].value_counts(normalize=True) * 100).round(2))

if __name__ == "__main__":
    main()
