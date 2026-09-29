import sys
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.loader import load_all
from src.data.validator import validate_all

def main():
    data = load_all()
    results = validate_all(data)

    for item in results:
        print(
            f"{item['name']}: rows={item['rows']:,}, "
            f"columns={item['columns']}, "
            f"duplicates={item['duplicates']}, "
            f"missing={item['missing_total']}"
        )

    Path("reports").mkdir(exist_ok=True)
    with open("reports/data_validation.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

if __name__ == "__main__":
    main()
