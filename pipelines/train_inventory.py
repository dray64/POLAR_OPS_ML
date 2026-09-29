import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.loader import load_dataset
from src.features.inventory import create_features
from src.models.inventory import train
from src.models.common import save_bundle
from src.evaluation.metrics import save_metrics
from src.evaluation.plots import feature_importance_plot

def main():
    df = load_dataset("inventory_consumption.csv")
    df, features = create_features(df)

    model, metrics, test_df, y, pred = train(df, features)

    save_bundle(model, features, "models/inventory_forecasting/model.pkl")
    save_metrics(metrics, "inventory_forecasting.json")
    feature_importance_plot(
        model, features,
        "Inventory Forecasting Feature Importance",
        "inventory_feature_importance.png",
    )

    print("\n=== INVENTORY FORECASTING ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
