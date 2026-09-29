import subprocess
import sys
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]

PIPELINES = [
    "validate_data.py",
    "profile_data.py",
    "check_targets.py",
    "train_maintenance.py",
    "train_inventory.py",
    "train_mission_delay.py",
    "train_operational_risk.py",
]

def main():
    report = {"pipelines": [], "status": "SUCCESS"}

    print("\n==============================================")
    print("             POLAR OPS ML PIPELINE")
    print("==============================================")

    for script in PIPELINES:
        print(f"\n>>> Running {script}")

        result = subprocess.run(
            [sys.executable, str(ROOT / "pipelines" / script)],
            cwd=ROOT,
        )

        report["pipelines"].append({
            "script": script,
            "return_code": result.returncode,
        })

        if result.returncode != 0:
            report["status"] = "FAILED"
            print(f"\nFAILED: {script}")
            break

    with open(ROOT / "reports" / "pipeline_run.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n==============================================")
    print(f"PIPELINE STATUS: {report['status']}")
    print("==============================================")

if __name__ == "__main__":
    main()
