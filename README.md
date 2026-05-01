# Algo Ensemble Model — EURUSD ML Trading System

## Project Structure

```
Algo_Ensemble_model/
├── data/                         ← Place your CSV files here
│   ├── eurusd_1h.csv
│   ├── eurusd_4h.csv
│   ├── eurusd_1d.csv
│   ├── dxy_1h.csv
│   ├── vix_1d.csv
│   └── us10y_1d.csv
│
├── features/                     ← Auto-generated (do not edit)
│   ├── eurusd_features.csv
│   └── eurusd_labelled.csv
│
├── models/                       ← Auto-generated model files
│   └── lgbm_baseline.pkl
│
├── results/                      ← All charts and CSV outputs
│
├── src/                          ← Python source modules
│   ├── data_loader.py
│   ├── features.py
│   ├── targets.py
│   ├── models.py
│   └── validation.py
│
├── notebooks/                    ← Run in this order
│   ├── 01_data_validation.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_target_construction.ipynb
│   ├── 04_model_training.ipynb
│   ├── 05_walk_forward_validation.ipynb
│   └── 06_shap_analysis.ipynb
│
├── requirements.txt
└── README.md
```

## Installation

```bash
cd C:\Users\omtej\Projects\Algo_Ensemble_model
pip install -r requirements.txt
```

## Execution Order

Run notebooks **strictly in order, one at a time**:

1. Add CSV files to `data/`
2. `01_data_validation.ipynb`
3. `02_feature_engineering.ipynb`
4. `03_target_construction.ipynb`
5. `04_model_training.ipynb`
6. `05_walk_forward_validation.ipynb`
7. `06_shap_analysis.ipynb`

## Instruments

- Primary: EURUSD
- Timeframe: 1H (model), 4H + Daily (context)
- Target: Triple-barrier classification (TP/SL/Time-out)
- Model: LightGBM Classifier (baseline)

## Performance Targets

| Phase | Target OOS Accuracy |
|-------|---------------------|
| Phase 1 (baseline) | > 52% consistently |
| Phase 2 (macro integration) | > 54% |
| Phase 3 (ensemble) | > 56% |
