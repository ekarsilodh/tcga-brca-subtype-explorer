"""Verification suite for the TCGA-BRCA PAM50 teaching pipeline outputs."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score

CLASS_ORDER = ["Basal", "Her2", "LumA", "LumB"]
PROBA_COLS = [f"proba_{c}" for c in CLASS_ORDER]
EXPECTED_PRED_COLS = ["sample", "actual_label", "predicted_label", "model_name"] + PROBA_COLS

DATA_PATH = Path("data/tcga_brca_teaching.csv")
OUTPUT_DIR = Path("outputs")


def test_patient_disjoint_splits_and_folds() -> None:
    manifest = pd.read_csv(OUTPUT_DIR / "split_manifest.csv")
    assert list(manifest.columns) == ["sample", "participant_id", "PAM50", "split", "cv_fold"]
    assert len(manifest) == 998
    assert set(manifest["split"]) == {"train", "val", "test"}
    assert (manifest["split"] == "train").sum() == 798
    assert (manifest["split"] == "val").sum() == 100
    assert (manifest["split"] == "test").sum() == 100

    for s1, s2 in [("train", "val"), ("train", "test"), ("val", "test")]:
        m1 = manifest[manifest["split"] == s1]
        m2 = manifest[manifest["split"] == s2]
        assert set(m1["sample"]).isdisjoint(set(m2["sample"]))
        assert set(m1["participant_id"]).isdisjoint(set(m2["participant_id"]))

    train_df = manifest[manifest["split"] == "train"]
    holdout_df = manifest[manifest["split"] != "train"]
    assert set(train_df["cv_fold"]) == {0, 1, 2}
    assert (holdout_df["cv_fold"] == -1).all()

    for f1, f2 in [(0, 1), (0, 2), (1, 2)]:
        p1 = set(train_df.loc[train_df["cv_fold"] == f1, "participant_id"])
        p2 = set(train_df.loc[train_df["cv_fold"] == f2, "participant_id"])
        assert p1.isdisjoint(p2)


def test_no_metadata_predictors_and_scaler_matches_train_only() -> None:
    raw_df = pd.read_csv(DATA_PATH)
    manifest = pd.read_csv(OUTPUT_DIR / "split_manifest.csv")
    feature_cols = [c for c in raw_df.columns if c not in ("sample", "PAM50")]
    assert len(feature_cols) == 1691

    train_samples = set(manifest.loc[manifest["split"] == "train", "sample"])
    X_train = raw_df.loc[raw_df["sample"].isin(train_samples), feature_cols].to_numpy(dtype=np.float64)
    X_all = raw_df[feature_cols].to_numpy(dtype=np.float64)

    expected_train_mean = X_train.mean(axis=0)
    expected_train_std = X_train.std(axis=0, ddof=0)
    expected_train_std = np.where(expected_train_std == 0.0, 1.0, expected_train_std)
    full_mean = X_all.mean(axis=0)

    for model_name in ["LogisticRegression", "SVM", "MLP"]:
        pipe = joblib.load(OUTPUT_DIR / "models" / f"{model_name}.joblib")
        assert pipe.n_features_in_ == 1691
        if hasattr(pipe, "feature_names_in_"):
            f_names = list(pipe.feature_names_in_)
            assert "sample" not in f_names
            assert "PAM50" not in f_names
            assert "participant_id" not in f_names
            assert f_names == feature_cols

        scaler = pipe.named_steps["scaler"]
        assert np.allclose(scaler.mean_, expected_train_mean, atol=1e-10)
        assert np.allclose(scaler.scale_, expected_train_std, atol=1e-10)
        assert not np.allclose(scaler.mean_, full_mean, atol=1e-6)

    for no_scaler_model in ["RandomForest", "MajorityBaseline"]:
        pipe = joblib.load(OUTPUT_DIR / "models" / f"{no_scaler_model}.joblib")
        assert pipe.n_features_in_ == 1691
        assert "scaler" not in pipe.named_steps


def test_prediction_csvs_finite_probabilities_and_class_mapping() -> None:
    for fname in ["test_predictions.csv", "val_predictions.csv"]:
        df = pd.read_csv(OUTPUT_DIR / fname)
        assert list(df.columns) == EXPECTED_PRED_COLS
        assert len(df) == 5 * 100

        probs = df[PROBA_COLS].to_numpy(dtype=np.float64)
        assert np.isfinite(probs).all()
        assert (probs >= 0.0).all() and (probs <= 1.0).all()
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-9)

        expected_pred = [CLASS_ORDER[idx] for idx in np.argmax(probs, axis=1)]
        assert list(df["predicted_label"]) == expected_pred
        assert set(df["actual_label"]).issubset(set(CLASS_ORDER))


def test_saved_pipeline_matches_test_predictions_csv() -> None:
    raw_df = pd.read_csv(DATA_PATH)
    manifest = pd.read_csv(OUTPUT_DIR / "split_manifest.csv")
    test_df = pd.read_csv(OUTPUT_DIR / "test_predictions.csv")
    feature_cols = [c for c in raw_df.columns if c not in ("sample", "PAM50")]

    test_samples = manifest.loc[manifest["split"] == "test", "sample"].tolist()
    X_test = raw_df.set_index("sample").loc[test_samples, feature_cols]

    for model_name in ["MajorityBaseline", "LogisticRegression", "RandomForest", "SVM", "MLP"]:
        pipe = joblib.load(OUTPUT_DIR / "models" / f"{model_name}.joblib")
        classes = list(pipe.classes_)
        reorder = [classes.index(c) for c in CLASS_ORDER]
        direct_proba = pipe.predict_proba(X_test)[:, reorder]

        sub = test_df[test_df["model_name"] == model_name].set_index("sample").loc[test_samples]
        saved_proba = sub[PROBA_COLS].to_numpy(dtype=np.float64)
        assert np.allclose(direct_proba, saved_proba, atol=1e-12)


def test_independent_confusion_matrix_and_csv_json_agreement() -> None:
    test_df = pd.read_csv(OUTPUT_DIR / "test_predictions.csv")
    with (OUTPUT_DIR / "metrics.json").open("r", encoding="utf-8") as f:
        metrics = json.load(f)

    assert metrics["class_order"] == CLASS_ORDER
    class_to_idx = {cls: i for i, cls in enumerate(CLASS_ORDER)}

    for model_name, model_info in metrics["models"].items():
        sub = test_df[test_df["model_name"] == model_name]
        assert len(sub) == 100

        # Independent confusion matrix built directly from CSV rows
        independent_cm = [[0 for _ in CLASS_ORDER] for _ in CLASS_ORDER]
        for actual, pred in zip(sub["actual_label"], sub["predicted_label"]):
            independent_cm[class_to_idx[actual]][class_to_idx[pred]] += 1

        assert sum(sum(row) for row in independent_cm) == 100
        tm = model_info["test_metrics"]
        assert independent_cm == tm["confusion_matrix"]

        y_true = sub["actual_label"].to_numpy()
        y_pred = sub["predicted_label"].to_numpy()
        assert np.isclose(accuracy_score(y_true, y_pred), tm["accuracy"], atol=1e-12)
        assert np.isclose(balanced_accuracy_score(y_true, y_pred), tm["balanced_accuracy"], atol=1e-12)
        assert np.isclose(
            f1_score(y_true, y_pred, labels=CLASS_ORDER, average="macro", zero_division=0),
            tm["macro_f1"],
            atol=1e-12,
        )
        recalls = recall_score(y_true, y_pred, labels=CLASS_ORDER, average=None, zero_division=0)
        for cls, rec in zip(CLASS_ORDER, recalls):
            assert np.isclose(rec, tm["per_class_recall"][cls], atol=1e-12)


def test_cv_selection_criterion_matches_csv() -> None:
    cv_df = pd.read_csv(OUTPUT_DIR / "cv_results.csv")
    with (OUTPUT_DIR / "metrics.json").open("r", encoding="utf-8") as f:
        metrics = json.load(f)

    means = cv_df.groupby("model_name")["balanced_accuracy"].mean().to_dict()
    for m_name, mean_bal in means.items():
        assert np.isclose(mean_bal, metrics["models"][m_name]["cv_summary"]["mean_balanced_accuracy"], atol=1e-12)

    expected_preferred = max(means.keys(), key=lambda m: means[m])
    assert metrics["preferred_model"] == expected_preferred
