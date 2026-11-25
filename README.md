# Breast Cancer Subtype Explorer (TCGA-BRCA Teaching Demo)

An accessible MSc Bioinformatics teaching project demonstrating how a researcher directs, reviews, and verifies an agentic machine-learning workflow on a fixed CPU budget.

## Research Question & Dataset Provenance

**Research question:** Can gene expression measurements distinguish the recorded `Basal`, `Her2`, `LumA`, and `LumB` subtype labels in previously unseen primary tumour samples within this TCGA-BRCA teaching table?

- **Dataset:** [`data/tcga_brca_teaching.csv`](data/tcga_brca_teaching.csv) (998 rows × 1,693 columns; 998 unique primary tumours, code `01`; 1,691 numeric gene expression predictors; 0 missing or non-finite values).
- **Provenance & limits:** See [`data/SOURCE.md`](data/SOURCE.md). Upstream normalization, gene-selection criteria for the 1,691 features, and original label-generation steps are unresolved in the supplied table. This project is strictly an **educational demonstration**—not independent clinical validation or biomarker discovery.

## Modelling & Evaluation Design

- **Split (`random_state=42`):** Stratified, patient-disjoint **80% train (798) / 10% validation (100) / 10% locked test (100)**.
- **Cross-validation:** 3-fold stratified CV on the **training set only**, reusing identical folds across all five fixed configurations:
  1. `MajorityBaseline` — `DummyClassifier(strategy='most_frequent')`
  2. `LogisticRegression` — `StandardScaler` + `LogisticRegression(C=0.1, solver='lbfgs', max_iter=2000)`
  3. `RandomForest` — `RandomForestClassifier(n_estimators=150, max_depth=8, min_samples_leaf=2, random_state=42, n_jobs=1)`
  4. `SVM` — `StandardScaler` + `SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)`
  5. `MLP` — `StandardScaler` + `MLPClassifier(hidden_layer_sizes=(32,), alpha=1.0, max_iter=500, early_stopping=False, random_state=42)`
- **Selection & decision rule:** Preferred model (`MLP`) is selected solely by **mean 3-fold CV balanced accuracy** before touching the locked test set. Predicted labels everywhere come from `classes_[argmax(predict_proba)]` in canonical order `['Basal', 'Her2', 'LumA', 'LumB']`.

## Quickstart Commands

```bash
# 1. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Run offline training, CV, and held-out evaluation (writes outputs/)
python analysis.py

# 3. Run automated verification checks
pytest -v

# 4. Launch the interactive dashboard (reads outputs/ only; zero training in UI)
streamlit run app.py
```

## Manual Confusion-Matrix Verification

To independently count a confusion-matrix row from [`outputs/test_predictions.csv`](outputs/test_predictions.csv) (for example, `Her2` under `MLP`):

```bash
python -c "import pandas as pd; df=pd.read_csv('outputs/test_predictions.csv'); print(df[(df.model_name=='MLP') & (df.actual_label=='Her2')]['predicted_label'].value_counts())"
```

Or view the full 4×4 matrix for any model:

```bash
python -c "import pandas as pd; df=pd.read_csv('outputs/test_predictions.csv'); sub=df[df.model_name=='MLP']; print(pd.crosstab(sub['actual_label'], sub['predicted_label']))"
```

## Project Layout

- `analysis.py` — Data validation, 80/10/10 patient-disjoint split, 3-fold CV, model fitting, and artifact export.
- `app.py` — *Breast Cancer Subtype Explorer* Streamlit + Plotly dashboard reading only precomputed `outputs/`.
- `tests/test_checks.py` — Automated checks for leakage, metadata exclusion, scaler statistics, probability mapping, and CSV/JSON agreement.
- `.streamlit/config.toml` — Light ivory/navy/teal theme configuration.
- `outputs/` — Generated `split_manifest.csv`, `cv_results.csv`, `val_predictions.csv`, `test_predictions.csv`, `metrics.json`, and `models/*.joblib`.
