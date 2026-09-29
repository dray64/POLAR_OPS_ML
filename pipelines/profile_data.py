import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.loader import load_all

def main():
    data = load_all()
    Path("reports").mkdir(exist_ok=True)
    out = []

    for name, df in data.items():
        for col in df.columns:
            out.append({
                "dataset": name,
                "column": col,
                "dtype": str(df[col].dtype),
                "missing": int(df[col].isna().sum()),
                "unique": int(df[col].nunique(dropna=True)),
            })

    pd.DataFrame(out).to_csv("reports/data_profile.csv", index=False)
    print("Saved reports/data_profile.csv")

if __name__ == "__main__":
    main()
