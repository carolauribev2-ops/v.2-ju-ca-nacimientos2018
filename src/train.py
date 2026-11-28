"""
Model training and evaluation for birth outcome prediction.

Two binary classifiers are trained:
  - cesarea_pipeline.pkl  : predicts C-section probability
  - bajo_peso_pipeline.pkl: predicts low birth weight risk

Both use a sklearn Pipeline (ColumnTransformer + classifier) so the full
preprocessing is bundled with the model for portable inference. The saved model
wraps that pipeline in isotonic calibration, so predict_proba returns
probabilities on the scale of the observed frequencies.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.preprocessing import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    TARGET_BAJO_PESO,
    TARGET_CESAREA,
)

MODELS_DIR = Path(__file__).parent.parent / "models"

# ── Preprocessor factory ───────────────────────────────────────────────────────


def build_preprocessor() -> ColumnTransformer:
    """
    ColumnTransformer that applies:
    - median imputation + StandardScaler to numeric features
    - most-frequent imputation + OneHotEncoder to categorical features
    """
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


# ── Candidate models ───────────────────────────────────────────────────────────


def _candidate_classifiers() -> dict:
    return {
        "logistic_regression": LogisticRegression(
            C=0.5, max_iter=500, class_weight="balanced", random_state=42
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=100,
            class_weight="balanced", n_jobs=-1, random_state=42
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_depth=6,
            min_samples_leaf=100, class_weight="balanced", random_state=42
        ),
    }


# ── Evaluation ─────────────────────────────────────────────────────────────────


def cross_validate_pipeline(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
) -> dict[str, float]:
    """Return mean ROC-AUC and Average Precision from stratified k-fold CV."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    results = cross_validate(
        pipeline, X, y, cv=skf,
        scoring=["roc_auc", "average_precision"],
        n_jobs=-1,
    )
    return {
        "roc_auc": results["test_roc_auc"].mean(),
        "roc_auc_std": results["test_roc_auc"].std(),
        "avg_precision": results["test_average_precision"].mean(),
        "avg_precision_std": results["test_average_precision"].std(),
    }


# ── Training entry point ───────────────────────────────────────────────────────


def prepare_xy(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    """Model features and integer target, without rows that have missing values."""
    X = df[FEATURES].copy()
    y = df[target].copy()
    valid_mask = X.notna().all(axis=1) & y.notna()
    return X[valid_mask], y[valid_mask].astype(int)



def train_model(
    df: pd.DataFrame,
    target: str,
    model_filename: str,
    cv: int = 5,
) -> tuple[Pipeline, dict]:
    """
    Select and train the best classifier for the given target variable.

    Steps:
    1. Prepare X (FEATURES) and y (target), drop rows with any NaN.
    2. Cross-validate all candidate classifiers.
    3. Measure calibration of the best one on a held-out split.
    4. Refit best on full training data with isotonic calibration.
    5. Persist the calibrated model to models/<model_filename>.pkl.

    Returns the fitted model and a comparison dict of all metrics. The model's
    `calibration_report_` attribute holds the held-out calibration check.
    """
    X, y = prepare_xy(df, target)

    print(
        f"\n[{target}] Training on {len(X):,} samples "
        f"({y.mean():.1%} positive)"
    )

    preprocessor = build_preprocessor()
    candidates = _candidate_classifiers()
    comparison: dict[str, dict] = {}
    best_name, best_score, best_pipeline = "", -1.0, None

    for name, clf in candidates.items():
        pipeline = Pipeline([("prep", preprocessor), ("clf", clf)])
        metrics = cross_validate_pipeline(pipeline, X, y, cv=cv)
        comparison[name] = metrics
        score = metrics["roc_auc"]
        status = ""
        if score > best_score:
            best_score, best_name, best_pipeline = score, name, pipeline
            status = " <-- best"
        print(
            f"  {name:35s}  ROC-AUC={score:.4f} ± {metrics['roc_auc_std']:.4f}"
            f"  AP={metrics['avg_precision']:.4f}{status}"
        )

    print(f"\nBest: {best_name} (ROC-AUC={best_score:.4f})")

    report = calibration_report(best_pipeline, X, y)
    for stage in ("sin_calibrar", "calibrado"):
        print(
            f"  {stage:13s} prob. media={report[stage]['probabilidad_media']:.3f}"
            f" (real {report['tasa_real_prueba']:.3f})"
            f"  Brier={report[stage]['brier']:.4f}"
        )

    model = calibrate(clone(best_pipeline), X, y)
    model.calibration_report_ = report

    MODELS_DIR.mkdir(exist_ok=True)
    out_path = MODELS_DIR / model_filename
    joblib.dump(model, out_path)
    print(f"Calibrated model saved to {out_path}")

    return model, comparison


# ── Calibration ────────────────────────────────────────────────────────────────


def calibrate(
    pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, cv: int = 5
) -> CalibratedClassifierCV:
    """
    Isotonic calibration with cross-fitting.

    class_weight='balanced' improves ranking on rare outcomes but inflates the
    predicted probabilities; calibration maps them back to observed frequencies.
    """
    model = CalibratedClassifierCV(pipeline, method="isotonic", cv=cv, n_jobs=-1)
    return model.fit(X, y)


def calibration_report(
    pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2
) -> dict:
    """Mean probability, Brier score and ROC-AUC on a held-out split, before and after calibration."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    fitted = {
        "sin_calibrar": clone(pipeline).fit(X_train, y_train),
        "calibrado": calibrate(clone(pipeline), X_train, y_train),
    }
    report: dict = {
        "particion_prueba": test_size,
        "tasa_real_prueba": float(y_test.mean()),
    }
    for stage, model in fitted.items():
        probabilities = model.predict_proba(X_test)[:, 1]
        report[stage] = {
            "probabilidad_media": float(probabilities.mean()),
            "brier": float(brier_score_loss(y_test, probabilities)),
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
        }
    return report


def get_feature_importance(model) -> pd.DataFrame | None:
    """
    Extract feature importance from the fitted pipeline's classifier.
    Returns a DataFrame sorted by importance, or None if not supported.
    """
    if hasattr(model, "calibrated_classifiers_"):
        model = model.calibrated_classifiers_[0].estimator
    clf = model.named_steps["clf"]
    prep = model.named_steps["prep"]

    if not hasattr(clf, "feature_importances_"):
        return None

    try:
        feature_names = prep.get_feature_names_out()
    except AttributeError:
        return None

    importances = clf.feature_importances_
    return (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
