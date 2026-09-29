import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.loader import load_dataset
from src.features.maintenance import create_features
from src.models.maintenance import train
from src.models.common import save_bundle
from src.evaluation.metrics import save_metrics
from src.evaluation.plots import confusion_matrix_plot, feature_importance_plot

def main():
    df = load_dataset("asset_telemetry.csv")
    df, features = create_features(df)

    model, metrics, X_test, y_test, prob, pred = train(df, features)

    save_bundle(model, features, "models/predictive_maintenance/model.pkl")
    save_metrics(metrics, "predictive_maintenance.json")
    confusion_matrix_plot(
        metrics["confusion_matrix"],
        "Predictive Maintenance Confusion Matrix",
        "maintenance_confusion_matrix.png",
    )
    feature_importance_plot(
        model, features,
        "Predictive Maintenance Feature Importance",
        "maintenance_feature_importance.png",
    )

    print("\n=== PREDICTIVE MAINTENANCE ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
