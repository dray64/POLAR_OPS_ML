# POLAR OPS ML

ML-only repository for the POLAR OPS polar expedition logistics and asset-management platform.

## Models

1. Predictive Maintenance
   - Dataset: `asset_telemetry.csv`
   - Target: `failure_event`
   - Model: XGBoost classifier

2. Inventory Forecasting
   - Dataset: `inventory_consumption.csv`
   - Target: next-day demand
   - Model: XGBoost regressor

3. Mission Delay Prediction
   - Dataset: `missions.csv`
   - Target: `delayed_flag`
   - Model: XGBoost classifier

4. Operational Risk
   - Dataset: `operational_risk_labels.csv`
   - Target: `high_risk_label`
   - Current dataset has only 2 positive labels, so the pipeline intentionally marks this model NOT_READY rather than producing misleading metrics.

## Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

## Run everything

```bash
python pipelines/train_all.py
```

## Outputs

- `models/` → trained model bundles
- `reports/metrics/` → JSON evaluation metrics
- `reports/figures/` → confusion matrices and feature-importance plots
- `reports/data_profile.csv` → dataset schema profile
- `reports/data_validation.json` → data quality report
- `reports/pipeline_run.json` → complete pipeline status

## Important validation note

The supplied data is a development/demo dataset. Do not present model metrics as real NCPOR operational performance. The pipeline is designed to be retrained on validated operational data when available.
