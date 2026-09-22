"""
Pattern recognition & predictive models for aircraft maintenance data.
Models adapt to whatever sensor / operational columns are present after data prep.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import (
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import (
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Dynamically pick usable feature columns from a prepared dataframe."""
    op_cols = [c for c in df.columns if c.startswith("operational_setting")]
    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    exclude = {
        "engine_id", "cycle", "RUL", "failure_imminent",
        "anomaly_score", "is_anomaly", "pattern_cluster",
    }
    if not sensor_cols and not op_cols:
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        return [c for c in numeric if c not in exclude]
    return op_cols + sensor_cols


def prepare_features(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Optional[pd.Series], Optional[pd.Series], List[str]]:
    """Extract features and optional targets."""
    if feature_cols is None:
        feature_cols = get_feature_columns(df)

    if not feature_cols:
        raise ValueError(
            "No usable feature columns found. Map at least some sensor or operational columns."
        )

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    valid = X.notna().any(axis=1)
    medians = X.median(numeric_only=True)
    X = X.loc[valid].fillna(medians)

    y_rul = None
    y_fail = None
    if "RUL" in df.columns:
        y_rul = pd.to_numeric(df.loc[valid, "RUL"], errors="coerce")
    if "failure_imminent" in df.columns:
        y_fail = (
            pd.to_numeric(df.loc[valid, "failure_imminent"], errors="coerce")
            .fillna(0)
            .astype(int)
        )

    return X, y_rul, y_fail, feature_cols


def train_rul_model(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Train Random Forest regressor for Remaining Useful Life."""
    X, y_rul, _, feature_cols = prepare_features(df)
    if y_rul is None:
        raise ValueError("Column 'RUL' is required for RUL training.")

    mask = y_rul.notna()
    X = X.loc[mask]
    y_rul = y_rul.loc[mask]

    if len(X) < 20:
        raise ValueError("Not enough valid rows with RUL to train (need ≥ 20).")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_rul, test_size=test_size, random_state=random_state
    )

    model = RandomForestRegressor(
        n_estimators=120,
        max_depth=14,
        min_samples_leaf=4,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, preds)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "r2": float(r2_score(y_test, preds)),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    importance = pd.Series(
        model.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)

    return {
        "model": model,
        "metrics": metrics,
        "feature_importance": importance,
        "feature_cols": feature_cols,
        "X_test": X_test,
        "y_test": y_test,
        "preds": preds,
    }


def train_failure_classifier(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Train classifier for imminent failure risk."""
    X, _, y_fail, feature_cols = prepare_features(df)
    if y_fail is None:
        raise ValueError("Column 'failure_imminent' is required.")

    mask = y_fail.notna()
    X = X.loc[mask]
    y_fail = y_fail.loc[mask]

    if len(X) < 20:
        raise ValueError("Not enough valid rows to train classifier (need ≥ 20).")

    stratify = y_fail if y_fail.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_fail, test_size=test_size, random_state=random_state, stratify=stratify
    )

    model = RandomForestClassifier(
        n_estimators=120,
        max_depth=12,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    proba = (
        model.predict_proba(X_test)[:, 1]
        if len(model.classes_) > 1
        else np.zeros(len(X_test))
    )

    report = classification_report(y_test, preds, output_dict=True, zero_division=0)

    return {
        "model": model,
        "metrics": {
            "accuracy": report.get("accuracy", 0),
            "precision_fail": report.get("1", {}).get("precision", 0),
            "recall_fail": report.get("1", {}).get("recall", 0),
            "f1_fail": report.get("1", {}).get("f1-score", 0),
        },
        "feature_cols": feature_cols,
        "X_test": X_test,
        "y_test": y_test,
        "preds": preds,
        "proba": proba,
    }


def detect_anomalies(
    df: pd.DataFrame,
    contamination: float = 0.05,
    random_state: int = 42,
) -> pd.DataFrame:
    """Isolation Forest anomaly detection on available features."""
    X, _, _, feature_cols = prepare_features(df)
    if len(X) < 10:
        raise ValueError("Not enough rows for anomaly detection.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    iso = IsolationForest(
        n_estimators=120,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    preds = iso.fit_predict(X_scaled)
    scores = iso.decision_function(X_scaled)

    result = df.loc[X.index].copy()
    result["anomaly_score"] = scores
    result["is_anomaly"] = (preds == -1).astype(int)
    return result


def find_patterns_kmeans(
    df: pd.DataFrame,
    n_clusters: int = 4,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, KMeans]:
    """Cluster into behavioral patterns."""
    X, _, _, feature_cols = prepare_features(df)
    if len(X) < n_clusters * 3:
        raise ValueError(f"Not enough rows for {n_clusters} clusters.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    result = df.loc[X.index].copy()
    result["pattern_cluster"] = labels
    return result, kmeans


def save_model(model, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str):
    return joblib.load(path)


FEATURE_COLS = (
    [f"operational_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 15)]
)
