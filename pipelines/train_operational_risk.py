import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.loader import load_dataset
from src.models.operational_risk import assess_dataset
from src.evaluation.metrics import save_metrics

def main():
    df = load_dataset("operational_risk_labels.csv")
    result = assess_dataset(df)

    save_metrics(result, "operational_risk.json")

    print("\n=== OPERATIONAL RISK ===")
    for k, v in result.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
