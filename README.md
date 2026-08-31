# Predictive Maintenance Analytics

**AI4I2020 Machine Failure Prediction**

A statistical machine learning project for predicting machine failures using the AI4I2020 Predictive Maintenance dataset. Built with classical ML methods, proper evaluation practices, and an interactive dashboard.

## Project Overview

Predictive maintenance aims to detect equipment failures before they happen, allowing maintenance teams to intervene proactively. This project demonstrates a complete end-to-end ML workflow:

- **Data exploration** and careful preprocessing
- **Classical statistical ML models** (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting)
- **Proper train/validation/test splits** with stratified sampling
- **Hyperparameter tuning** with cross-validation
- **Threshold optimization** for the predictive maintenance objective
- **Interactive dashboard** for exploration and prediction

The focus is on **correctness, reproducibility, and clear methodology** — not on achieving the highest possible accuracy on a synthetic benchmark.

## Dataset

The [AI4I2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) contains 10,000 records of machine operating conditions and failure labels.

### Key Variables

| Column | Description |
|--------|-------------|
| `Type` | Machine type: L (Low), M (Medium), H (High) |
| `Air temperature [K]` | Ambient temperature |
| `Process temperature [K]` | Process temperature |
| `Rotational speed [rpm]` | Rotational speed |
| `Torque [Nm]` | Torque |
| `Tool wear [min]` | Tool wear time |
| `Machine failure` | **Target**: 1 = failure, 0 = normal |

### Failure Mode Columns (Excluded from Features)

| Column | Description | Count |
|--------|-------------|-------|
| `TWF` | Tool Wear Failure | 46 |
| `HDF` | Heat Dissipation Failure | 115 |
| `PWF` | Power Failure | 95 |
| `OSF` | Overstrain Failure | 98 |
| `RNF` | Random Failure | 19 |

**Important**: These failure-mode columns are **excluded** from the predictive model. They represent specific failure diagnoses that would only be known *at or after* the time of failure. Including them would constitute data leakage — the model would be "cheating" by using information unavailable at prediction time.

### Target Distribution

- **Normal (0)**: 9,661 records (96.6%)
- **Failure (1)**: 339 records (3.4%)

This is a **highly imbalanced** dataset. Accuracy alone is misleading — a model predicting "normal" for everything achieves 96.6% accuracy but catches zero failures.

## Problem Statement

Build a binary classifier that predicts `Machine failure` from operating conditions. The objective is **early warning**: flag machines at risk of failure so maintenance can be scheduled proactively.

**Cost asymmetry**: Missing a real failure (false negative) is far more costly than a false alarm (false positive). Therefore we prioritize:

- **Recall** — catch as many actual failures as possible
- **F1-score** — balance between recall and precision
- **PR-AUC** — robust metric for imbalanced data
- **ROC-AUC** — overall discriminative ability

## Methodology

### Data Loading & Cleaning

- Load raw CSV from `data/ai4i2020.csv`
- No missing values in the dataset
- Drop identifiers (`UDI`, `Product ID`) — no predictive value
- Drop failure-mode columns (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) — leakage risk

### Feature Preparation

| Feature Type | Features | Preprocessing |
|--------------|----------|---------------|
| Numeric | Air temp, Process temp, Rotational speed, Torque, Tool wear | Median imputation → StandardScaler |
| Categorical | Type (L, M, H) | Most frequent imputation → One-hot encoding (drop first) |

All preprocessing is wrapped in a `ColumnTransformer` inside a `Pipeline` to prevent leakage.

### Train / Validation / Test Split

- **Test set**: 20% (stratified, held out completely until final evaluation)
- **Training set**: 80% (used for CV and hyperparameter tuning)
- **Cross-validation**: 5-fold StratifiedKFold on training set
- **Random seed**: 42 (configurable, fixed default for reproducibility)

### Models Tested

| Model | Key Hyperparameters Tuned |
|-------|---------------------------|
| Logistic Regression | C, penalty, solver |
| Decision Tree | max_depth, min_samples_split, min_samples_leaf |
| Random Forest | n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features |
| Gradient Boosting | n_estimators, learning_rate, max_depth, min_samples_leaf, subsample |

### Hyperparameter Tuning

- **Method**: GridSearchCV
- **Scoring metric**: **Recall** (primary objective: catch failures)
- **Search space**: Small, sensible grids (see `src/train.py`)
- **No test set involvement** during tuning

### Class Imbalance Handling

- Analyzed target distribution (3.4% failure rate)
- Used `class_weight="balanced"` for all supported models
- No oversampling (SMOTE, etc.) — not necessary with balanced class weights
- Threshold optimization on validation probabilities

### Model Selection

Models compared on **test set** using multiple metrics:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|-------|----------|-----------|--------|-----|---------|--------|
| (results populated after training) | | | | | | |

**Selection criterion**: Highest recall, then highest F1. Not highest accuracy.

### Threshold Selection

Default 0.5 threshold is rarely optimal for imbalanced data. We evaluate thresholds from 0.01 to 0.99 and recommend:

1. **F1-optimal threshold**: Maximizes F1 score (balance)
2. **High-recall threshold (≥80% recall)**: Catches most failures, accepts more false alarms

For predictive maintenance, we recommend the **high-recall threshold** because missing a failure is more costly than investigating a false alarm.

### Explainability

- **Tree-based models**: Feature importances from `feature_importances_`
- **Logistic Regression**: Coefficient magnitudes and signs

## Running the Project

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
pip install -r requirements.txt
```

### Training

```bash
python src/train.py
```

This will:
1. Load and preprocess data
2. Run stratified train/test split
3. Tune 4 models with GridSearchCV (5-fold CV, scoring=recall)
4. Evaluate on held-out test set
5. Select best model (highest recall → F1)
6. Save model to `models/final_model.joblib`
7. Save results to `results/`

Expected runtime: 1-3 minutes depending on hardware.

### Launch Dashboard

```bash
streamlit run app.py
```

Open the displayed URL (typically http://localhost:8501).

## Project Structure

```
predictive-maintenance/
├── data/
│   └── ai4i2020.csv           # Raw dataset (not in git)
├── models/
│   └── final_model.joblib     # Trained pipeline (created after training)
├── results/                   # Generated after training
│   ├── model_comparison.csv
│   ├── test_metrics.json
│   ├── threshold_analysis.csv
│   ├── feature_importance.csv
│   ├── training_summary.json
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── precision_recall_curve.png
│   ├── threshold_analysis.png
│   └── feature_importance.png
├── src/
│   ├── data.py               # Data loading
│   ├── preprocessing.py      # Preprocessing pipeline
│   ├── train.py              # Training & tuning pipeline
│   ├── evaluate.py           # Evaluation metrics & plots
│   └── predict.py            # Inference wrapper
├── app.py                    # Streamlit dashboard
├── requirements.txt
├── README.md
├── .gitignore
└── notebooks/
    └── exploration.ipynb     # Optional exploration notebook
```

## Dashboard Pages

| Page | Purpose |
|------|---------|
| **Dataset Overview** | KPIs, class balance, feature distributions, correlations |
| **Failure Analysis** | Failure modes, type breakdown, condition comparisons |
| **Model Performance** | Comparison table, threshold analysis, ROC/PR curves |
| **Feature Importance** | Top features driving predictions |
| **Predict Machine Failure** | Single-machine prediction with risk gauge |
| **Batch Prediction** | CSV upload → predictions + download |
| **Documentation** | This README rendered in-app |

## Results

After training, the `results/` directory contains:

- `model_comparison.csv` — All models on test set
- `test_metrics.json` — Best model metrics
- `threshold_analysis.csv` — Metrics at each threshold
- `feature_importance.csv` — Ranked features
- `training_summary.json` — Complete experiment summary
- PNG plots for confusion matrix, ROC, PR, threshold analysis, feature importance

## Limitations

1. **Synthetic dataset**: AI4I2020 is generated from a simulation, not real sensor data. Patterns may not transfer to real equipment.
2. **Small failure count**: Only 339 failures limits statistical power.
3. **Limited features**: Only 6 predictive features (5 numeric + 1 categorical). Real systems have hundreds of sensors.
4. **No temporal structure**: Records are treated as independent. Real predictive maintenance uses time-series data.
5. **Threshold depends on cost ratio**: The "optimal" threshold depends on the actual cost of false negatives vs false positives, which is domain-specific.
6. **Prediction ≠ Guarantee**: A 72% failure probability means "investigate this machine," not "this machine will fail."

## Future Improvements

- Time-series models (LSTM, Temporal Convolutional Networks) for sequential data
- Anomaly detection for unknown failure modes
- Integration with CMMS/EAM systems
- Real-time streaming inference
- Cost-sensitive threshold optimization with domain-specific cost matrix
- SHAP values for local explanations

## Reproducibility

- Fixed random seed (42) in all stochastic components
- All preprocessing inside sklearn Pipelines
- Version-pinned dependencies in `requirements.txt`
- Complete experiment summary saved to `results/training_summary.json`

## License

MIT License — feel free to use for learning or as a starting point for real projects.

---

*Built as a learning project demonstrating statistical ML best practices. Not intended for production use without extensive validation on real equipment data.*