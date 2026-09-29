from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

def confusion_matrix_plot(cm, title, filename):
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(np.asarray(cm)).plot(ax=ax, colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=160)
    plt.close(fig)

def feature_importance_plot(model, features, title, filename):
    values = model.feature_importances_
    order = values.argsort()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([features[i] for i in order], values[order])
    ax.set_title(title)
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=160)
    plt.close(fig)
