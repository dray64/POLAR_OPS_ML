import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.loader import load_dataset
from src.features.mission_delay import create_features
from src.models.mission_delay import train
from src.models.common import save_bundle
from src.evaluation.metrics import save_metrics
from src.evaluation.plots import confusion_matrix_plot, feature_importance_plot

def main():
    df = load_dataset("missions.csv")
    df, features = create_features(df)

    model, metrics, X_test, y_test, prob, pred = train(df, features)

    save_bundle(model, features, "models/mission_delay/model.pkl")
    save_metrics(metrics, "mission_delay.json")
    confusion_matrix_plot(
        metrics["confusion_matrix"],
        "Mission Delay Confusion Matrix",
        "mission_delay_confusion_matrix.png",
    )
    feature_importance_plot(
        model, features,
        "Mission Delay Feature Importance",
        "mission_delay_feature_importance.png",
    )

    print("\n=== MISSION DELAY ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
