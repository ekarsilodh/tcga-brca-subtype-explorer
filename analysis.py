"""TCGA-BRCA PAM50 teaching pipeline: validation, 80/10/10 split, 3-fold CV, and evaluation."""

from __future__ import annotations

import hashlib
import json
import platform
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

CLASS_ORDER: list[str] = ["Basal", "Her2", "LumA", "LumB"]
RANDOM_STATE: int = 42
DATA_PATH = Path("data/tcga_brca_teaching.csv")
OUTPUT_DIR = Path("outputs")
MODELS_DIR = OUTPUT_DIR / "models"

MODEL_DISPLAY_NAMES: dict[str, str] = {
    "MajorityBaseline": "Majority-Class Baseline",
    "LogisticRegression": "Logistic Regression (L2, C=0.1)",
    "RandomForest": "Random Forest (150 trees, depth=8)",
    "SVM": "SVM (RBF, C=1.0)",
    "MLP": "MLP Neural Network (32 hidden)",
}


def compute_sha256(path: Path) -> str:
    """Return hex SHA-256 checksum for an input file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_and_validate_data(path: Path) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, list[str]]:
    """Load teaching CSV, validate schema/finite values, and separate predictors from labels/IDs."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    required_cols = {"sample", "PAM50"}
    missing_req = required_cols - set(df.columns)
    if missing_req:
        raise ValueError(f"Missing required columns: {sorted(missing_req)}")

    if df["sample"].isna().any() or (df["sample"].astype(str).str.strip() == "").any():
        raise ValueError("Unexpected missing values in 'sample' column.")
    if df["PAM50"].isna().any():
        raise ValueError("Unexpected missing values in 'PAM50' column.")

    observed_labels = set(df["PAM50"].astype(str).unique())
    unexpected_labels = observed_labels - set(CLASS_ORDER)
    if unexpected_labels:
        raise ValueError(f"Unexpected PAM50 labels found: {sorted(unexpected_labels)}")

    if df["sample"].duplicated().any():
        raise ValueError("Duplicate sample IDs detected in dataset.")

    feature_cols = [c for c in df.columns if c not in ("sample", "PAM50")]
    if len(feature_cols) == 0:
        raise ValueError("No numeric gene expression predictors found.")

    X = df[feature_cols].apply(pd.to_numeric, errors="raise").astype(np.float64)
    if X.isna().any().any():
        raise ValueError("Unexpected missing (NaN) gene expression values detected.")
    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("Unexpected non-finite (Inf/-Inf) gene expression values detected.")

    samples = df["sample"].astype(str)
    participants = samples.str.split("-").str[:3].str.join("-")
    y = df["PAM50"].astype(str)
    return X, y, samples, participants, feature_cols


def create_patient_disjoint_splits(
    y: pd.Series,
    samples: pd.Series,
    participants: pd.Series,
) -> pd.DataFrame:
    """Create stratified 80/10/10 train/val/test split and 3-fold stratified CV on train only."""
    if participants.nunique() != len(participants):
        raise ValueError("Expected 1-to-1 participant-to-sample mapping; group splitting required.")

    idx = np.arange(len(y))
    train_idx, holdout_idx = train_test_split(
        idx,
        test_size=0.20,
        stratify=y.iloc[idx],
        random_state=RANDOM_STATE,
    )
    val_idx, test_idx = train_test_split(
        holdout_idx,
        test_size=0.50,
        stratify=y.iloc[holdout_idx],
        random_state=RANDOM_STATE,
    )

    split_labels = np.empty(len(y), dtype=object)
    split_labels[train_idx] = "train"
    split_labels[val_idx] = "val"
    split_labels[test_idx] = "test"

    cv_folds = np.full(len(y), -1, dtype=int)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    for fold_id, (_, fold_val_pos) in enumerate(skf.split(train_idx, y.iloc[train_idx])):
        cv_folds[train_idx[fold_val_pos]] = fold_id

    manifest = pd.DataFrame(
        {
            "sample": samples.to_numpy(),
            "participant_id": participants.to_numpy(),
            "PAM50": y.to_numpy(),
            "split": split_labels,
            "cv_fold": cv_folds,
        }
    )

    # Explicit leakage checks on participant and sample IDs
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        p_a = set(manifest.loc[manifest["split"] == a, "participant_id"])
        p_b = set(manifest.loc[manifest["split"] == b, "participant_id"])
        s_a = set(manifest.loc[manifest["split"] == a, "sample"])
        s_b = set(manifest.loc[manifest["split"] == b, "sample"])
        if p_a & p_b or s_a & s_b:
            raise AssertionError(f"Data leakage detected between {a} and {b} splits.")

    return manifest


def build_model_specs() -> dict[str, Pipeline]:
    """Return the five fixed teaching model configurations."""
    return {
        "MajorityBaseline": Pipeline(
            [("clf", DummyClassifier(strategy="most_frequent"))]
        ),
        "LogisticRegression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(C=0.1, solver="lbfgs", max_iter=2000)),
            ]
        ),
        "RandomForest": Pipeline(
            [
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=150,
                        max_depth=8,
                        min_samples_leaf=2,
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                )
            ]
        ),
        "SVM": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    SVC(
                        kernel="rbf",
                        C=1.0,
                        gamma="scale",
                        probability=True,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "MLP": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(32,),
                        alpha=1.0,
                        max_iter=500,
                        early_stopping=False,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def predict_with_canonical_proba(model: Pipeline, X_eval: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Map predict_proba through model.classes_ into CLASS_ORDER and choose argmax label."""
    raw_proba = model.predict_proba(X_eval)
    model_classes = list(model.classes_)
    missing_classes = set(CLASS_ORDER) - set(model_classes)
    if missing_classes:
        raise ValueError(f"Model missing expected classes: {sorted(missing_classes)}")

    reorder_idx = [model_classes.index(c) for c in CLASS_ORDER]
    proba = raw_proba[:, reorder_idx]
    if not np.isfinite(proba).all():
        raise ValueError("Non-finite predicted probabilities encountered.")

    pred_indices = np.argmax(proba, axis=1)
    y_pred = np.array([CLASS_ORDER[i] for i in pred_indices], dtype=object)
    return y_pred, proba


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba: np.ndarray,
    include_roc: bool = True,
) -> dict:
    """Compute accuracy, balanced accuracy, macro F1, per-class recall, CM, and OvR ROC."""
    acc = float(accuracy_score(y_true, y_pred))
    correct_count = int(np.sum(y_true == y_pred))
    n_samples = int(len(y_true))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, labels=CLASS_ORDER, average="macro", zero_division=0))
    recalls = recall_score(y_true, y_pred, labels=CLASS_ORDER, average=None, zero_division=0)
    per_class_recall = {cls: float(rec) for cls, rec in zip(CLASS_ORDER, recalls)}
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_ORDER).tolist()

    result = {
        "accuracy": acc,
        "correct_count": correct_count,
        "n_samples": n_samples,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "per_class_recall": per_class_recall,
        "confusion_matrix": cm,
    }

    if include_roc:
        roc_ovr: dict[str, dict] = {}
        aucs: list[float] = []
        for idx, cls in enumerate(CLASS_ORDER):
            y_bin = (y_true == cls).astype(int)
            fpr, tpr, _ = roc_curve(y_bin, proba[:, idx])
            cls_auc = float(auc(fpr, tpr))
            aucs.append(cls_auc)
            roc_ovr[cls] = {
                "fpr": [float(v) for v in fpr],
                "tpr": [float(v) for v in tpr],
                "auc": cls_auc,
            }
        result["roc_ovr"] = {
            "per_class": roc_ovr,
            "macro_auc": float(np.mean(aucs)),
        }

    return result


def build_prediction_df(
    samples: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    proba: np.ndarray,
) -> pd.DataFrame:
    """Construct exact prediction output table matching the canonical schema."""
    out = pd.DataFrame(
        {
            "sample": samples.to_numpy(),
            "actual_label": y_true,
            "predicted_label": y_pred,
            "model_name": model_name,
        }
    )
    for idx, cls in enumerate(CLASS_ORDER):
        out[f"proba_{cls}"] = proba[:, idx]
    return out


def fit_and_capture_warnings(model: Pipeline, X_fit: pd.DataFrame, y_fit: np.ndarray) -> tuple[Pipeline, float, list[str]]:
    """Fit pipeline, measure wall-clock seconds, and collect any warning strings."""
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(X_fit, y_fit)
    elapsed = float(time.perf_counter() - t0)
    warn_msgs = [f"{w.category.__name__}: {w.message}" for w in caught]
    return model, elapsed, warn_msgs


def main() -> None:
    total_t0 = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    X, y, samples, participants, feature_cols = load_and_validate_data(DATA_PATH)
    manifest = create_patient_disjoint_splits(y, samples, participants)
    manifest.to_csv(OUTPUT_DIR / "split_manifest.csv", index=False)

    train_mask = manifest["split"] == "train"
    val_mask = manifest["split"] == "val"
    test_mask = manifest["split"] == "test"

    X_train = X.loc[train_mask].reset_index(drop=True)
    y_train = y.loc[train_mask].to_numpy()
    train_folds = manifest.loc[train_mask, "cv_fold"].to_numpy()

    X_val = X.loc[val_mask].reset_index(drop=True)
    y_val = y.loc[val_mask].to_numpy()
    val_samples = samples.loc[val_mask].reset_index(drop=True)

    X_test = X.loc[test_mask].reset_index(drop=True)
    y_test = y.loc[test_mask].to_numpy()
    test_samples = samples.loc[test_mask].reset_index(drop=True)

    # 1. Cross-validation on training set only (using identical folds across all models)
    cv_rows: list[dict] = []
    all_warnings: list[dict] = []
    cv_summaries: dict[str, dict] = {}

    for model_name in build_model_specs().keys():
        fold_accs, fold_bal_accs, fold_f1s, fold_times = [], [], [], []
        for fold_id in (0, 1, 2):
            tr_idx = train_folds != fold_id
            va_idx = train_folds == fold_id
            pipe = build_model_specs()[model_name]
            _, fit_sec, warns = fit_and_capture_warnings(pipe, X_train.iloc[tr_idx], y_train[tr_idx])
            for msg in warns:
                all_warnings.append({"stage": f"cv_fold_{fold_id}", "model_name": model_name, "warning": msg})

            y_fold_pred, _ = predict_with_canonical_proba(pipe, X_train.iloc[va_idx])
            y_fold_true = y_train[va_idx]
            acc = float(accuracy_score(y_fold_true, y_fold_pred))
            bal_acc = float(balanced_accuracy_score(y_fold_true, y_fold_pred))
            macro_f1 = float(f1_score(y_fold_true, y_fold_pred, labels=CLASS_ORDER, average="macro", zero_division=0))

            fold_accs.append(acc)
            fold_bal_accs.append(bal_acc)
            fold_f1s.append(macro_f1)
            fold_times.append(fit_sec)

            cv_rows.append(
                {
                    "model_name": model_name,
                    "fold": fold_id,
                    "accuracy": acc,
                    "balanced_accuracy": bal_acc,
                    "macro_f1": macro_f1,
                    "fit_time_sec": fit_sec,
                }
            )

        cv_summaries[model_name] = {
            "mean_accuracy": float(np.mean(fold_accs)),
            "std_accuracy": float(np.std(fold_accs, ddof=0)),
            "mean_balanced_accuracy": float(np.mean(fold_bal_accs)),
            "std_balanced_accuracy": float(np.std(fold_bal_accs, ddof=0)),
            "mean_macro_f1": float(np.mean(fold_f1s)),
            "std_macro_f1": float(np.std(fold_f1s, ddof=0)),
            "mean_fit_time_sec": float(np.mean(fold_times)),
        }

    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(OUTPUT_DIR / "cv_results.csv", index=False)

    # Select preferred model strictly by mean CV balanced accuracy on training folds only
    preferred_model = max(
        cv_summaries.keys(),
        key=lambda m: cv_summaries[m]["mean_balanced_accuracy"],
    )

    # 2. Fit each model on the full 80% training set and evaluate on val and locked test
    val_pred_frames: list[pd.DataFrame] = []
    test_pred_frames: list[pd.DataFrame] = []
    models_metrics: dict[str, dict] = {}

    for model_name, pipe in build_model_specs().items():
        fitted_pipe, full_fit_sec, warns = fit_and_capture_warnings(pipe, X_train, y_train)
        for msg in warns:
            all_warnings.append({"stage": "full_train_fit", "model_name": model_name, "warning": msg})

        joblib.dump(fitted_pipe, MODELS_DIR / f"{model_name}.joblib")

        y_val_pred, val_proba = predict_with_canonical_proba(fitted_pipe, X_val)
        y_test_pred, test_proba = predict_with_canonical_proba(fitted_pipe, X_test)

        val_pred_frames.append(build_prediction_df(val_samples, y_val, y_val_pred, model_name, val_proba))
        test_pred_frames.append(build_prediction_df(test_samples, y_test, y_test_pred, model_name, test_proba))

        models_metrics[model_name] = {
            "display_name": MODEL_DISPLAY_NAMES[model_name],
            "full_fit_time_sec": full_fit_sec,
            "cv_summary": cv_summaries[model_name],
            "val_metrics": compute_classification_metrics(y_val, y_val_pred, val_proba, include_roc=True),
            "test_metrics": compute_classification_metrics(y_test, y_test_pred, test_proba, include_roc=True),
        }

    pd.concat(val_pred_frames, ignore_index=True).to_csv(OUTPUT_DIR / "val_predictions.csv", index=False)
    pd.concat(test_pred_frames, ignore_index=True).to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

    cohort_counts = {cls: int((y == cls).sum()) for cls in CLASS_ORDER}
    split_counts = {
        split_name: {
            "total": int((manifest["split"] == split_name).sum()),
            "by_class": {
                cls: int(((manifest["split"] == split_name) & (manifest["PAM50"] == cls)).sum())
                for cls in CLASS_ORDER
            },
        }
        for split_name in ("train", "val", "test")
    }

    total_runtime_sec = float(time.perf_counter() - total_t0)
    metrics_payload = {
        "class_order": CLASS_ORDER,
        "random_state": RANDOM_STATE,
        "n_features": len(feature_cols),
        "preferred_model": preferred_model,
        "selection_criterion": "mean_cv_balanced_accuracy",
        "baseline_model": "MajorityBaseline",
        "cohort_counts": cohort_counts,
        "n_cohort_total": int(len(y)),
        "split_counts": split_counts,
        "models": models_metrics,
        "runtime_metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "python_version": platform.python_version(),
            "os": platform.platform(),
            "package_versions": {
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit_learn": sklearn.__version__,
                "joblib": joblib.__version__,
            },
            "dataset_sha256": compute_sha256(DATA_PATH),
            "total_runtime_sec": total_runtime_sec,
            "warnings_count": len(all_warnings),
            "warnings": all_warnings,
        },
    }

    with (OUTPUT_DIR / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    print(f"Completed analysis in {total_runtime_sec:.2f}s. Preferred model (CV balanced acc): {preferred_model}")
    print(f"Warnings recorded: {len(all_warnings)}")
    for m_name, m_info in models_metrics.items():
        cv_b = m_info["cv_summary"]["mean_balanced_accuracy"]
        te_a = m_info["test_metrics"]["accuracy"]
        te_b = m_info["test_metrics"]["balanced_accuracy"]
        print(f"  {m_name:20s} | CV bal_acc={cv_b:.4f} | Test acc={te_a:.4f} | Test bal_acc={te_b:.4f}")


if __name__ == "__main__":
    main()
