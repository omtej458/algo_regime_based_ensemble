"""
models.py
=========
Model definitions and training utilities.

Phase 4 (current): Single LightGBM baseline.
Future phases: Add XGBoost, CatBoost, and meta-learner stacking.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score
)
import joblib
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


# ── LightGBM baseline ────────────────────────────────────────────────────────

# def build_lgbm_baseline() -> lgb.LGBMClassifier:
#     """
#     Returns a LightGBM Classifier with conservative, well-regularised params.
#     These are intentionally conservative to prevent overfitting on the first run.
#     """
#     return lgb.LGBMClassifier(
#         n_estimators=300,
#         learning_rate=0.02,
#         max_depth=5,
#         num_leaves=31,
#         min_child_samples=30,      # at least 30 samples per leaf (regularisation)
#         subsample=0.8,             # use 80% of rows per tree
#         colsample_bytree=0.7,      # use 70% of features per tree
#         reg_alpha=0.1,             # L1 regularisation
#         reg_lambda=0.1,            # L2 regularisation
#         class_weight="balanced",   # handles class imbalance automatically
#         random_state=42,
#         verbose=-1,                # suppress training output
#         n_jobs=-1,
#     )

def build_lgbm_baseline() -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.01,
        max_depth=4,
        num_leaves=15,
        min_child_samples=80,
        subsample=0.7,
        colsample_bytree=0.6,
        reg_alpha=1.0,
        reg_lambda=1.0,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
        n_jobs=-1,
    )

def build_xgb_baseline() -> xgb.XGBClassifier:
    """
    XGBoost classifier — second base model for ensemble.
    Uses exact greedy splitting (different algorithm to LGBM).
    This diversity is what makes the ensemble work.
    """
    return xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.01,
        max_depth=4,
        min_child_weight=80,       # equivalent to min_child_samples
        subsample=0.7,
        colsample_bytree=0.6,
        reg_alpha=1.0,
        reg_lambda=1.0,
        scale_pos_weight=1,        # class balance handled below
        eval_metric='logloss',
        random_state=42,
        verbosity=0,
        n_jobs=-1,
        use_label_encoder=False,
    )

def build_catboost_baseline() -> CatBoostClassifier:
    """
    CatBoost — third base model for ensemble.
    Uses ordered boosting and symmetric trees.
    Fundamentally different inductive bias from LGBM and XGBoost.
    """
    return CatBoostClassifier(
        iterations=200,
        learning_rate=0.01,
        depth=4,
        l2_leaf_reg=3.0,
        min_data_in_leaf=80,
        subsample=0.7,
        colsample_bylevel=0.6,
        auto_class_weights='Balanced',
        random_seed=42,
        verbose=0,
    )

def build_ensemble_prediction(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test:  pd.DataFrame,
    lgbm_weight: float = 0.55,
    xgb_weight:  float = 0.45,
) -> tuple:
    """
    Trains LGBM + XGBoost and returns averaged probability predictions.
    LGBM gets slightly higher weight as it performs better on this data.

    Returns
    -------
    (ensemble_probs, lgbm_probs, xgb_probs)
    """
    # Train LGBM
    lgbm_model = build_lgbm_baseline()
    lgbm_model.fit(X_train, y_train)
    lgbm_probs = lgbm_model.predict_proba(X_test)[:, 1]

    # Train XGBoost
    xgb_model = build_xgb_baseline()
    xgb_model.fit(X_train, y_train)
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]

    # Weighted average
    ensemble_probs = lgbm_weight * lgbm_probs + xgb_weight * xgb_probs

    return ensemble_probs, lgbm_probs, xgb_probs, lgbm_model, xgb_model

# ── Training ─────────────────────────────────────────────────────────────────

def train_model(model, X_train: pd.DataFrame, y_train: pd.Series) -> object:
    """Fits the model and returns it."""
    model.fit(X_train, y_train)
    return model


# ── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_model(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test:  pd.DataFrame,
    y_test:  pd.Series,
    confidence_threshold: float = 0.60,
) -> dict:
    """
    Evaluates model on train and test sets.
    Also evaluates on high-confidence signals only (prob > threshold).

    Returns a dict of all metrics.
    """
    train_preds = model.predict(X_train)
    test_preds  = model.predict(X_test)
    test_probs  = model.predict_proba(X_test)[:, 1]

    train_acc = accuracy_score(y_train, train_preds)
    test_acc  = accuracy_score(y_test,  test_preds)
    overfit_gap = train_acc - test_acc

    # High-confidence predictions only
    hc_mask = (test_probs >= confidence_threshold) | (test_probs <= (1 - confidence_threshold))
    hc_count = hc_mask.sum()

    if hc_count >= 10:
        hc_acc = accuracy_score(y_test[hc_mask], test_preds[hc_mask])
    else:
        hc_acc = np.nan
        hc_count = 0

    # AUC
    try:
        auc = roc_auc_score(y_test, test_probs)
    except Exception:
        auc = np.nan

    metrics = {
        "train_accuracy":    train_acc,
        "test_accuracy":     test_acc,
        "overfit_gap":       overfit_gap,
        "hc_accuracy":       hc_acc,
        "hc_signal_count":   hc_count,
        "total_test_count":  len(y_test),
        "auc_roc":           auc,
    }

    return metrics


def print_evaluation_report(metrics: dict, X_test: pd.DataFrame = None,
                             y_test: pd.Series = None, model=None) -> None:
    """Prints a formatted evaluation report."""
    print("\n" + "=" * 55)
    print("  Model evaluation report")
    print("=" * 55)
    print(f"  Train accuracy        : {metrics['train_accuracy']:.3f}  ({metrics['train_accuracy']*100:.1f}%)")
    print(f"  Test accuracy (OOS)   : {metrics['test_accuracy']:.3f}  ({metrics['test_accuracy']*100:.1f}%)")
    print(f"  Overfit gap           : {metrics['overfit_gap']:.3f}")
    print(f"  AUC-ROC               : {metrics['auc_roc']:.3f}")
    print(f"\n  High-confidence (>{int(metrics.get('hc_threshold', 60))}% prob):")
    print(f"    Signals             : {metrics['hc_signal_count']}")
    hc = metrics['hc_accuracy']
    if not np.isnan(hc):
        print(f"    Accuracy            : {hc:.3f}  ({hc*100:.1f}%)")
    else:
        print(f"    Accuracy            : N/A (fewer than 10 signals)")
    print()

    # Interpretation
    test_acc = metrics["test_accuracy"]
    gap      = metrics["overfit_gap"]
    print("  Interpretation:")

    if test_acc < 0.50:
        print("  [PROBLEM] OOS accuracy < 50% — check for data pipeline bug.")
        print("            Verify that all features are shifted +1 bar.")
    elif test_acc < 0.52:
        print("  [BASELINE] OOS accuracy 50-52% — pipeline is clean.")
        print("             No edge yet. Add 4H/Daily macro context features next.")
    elif test_acc < 0.54:
        print("  [EARLY EDGE] OOS accuracy 52-54% — there is a weak signal here.")
        print("               Add VIX regime, DXY slope features. Run walk-forward next.")
    elif test_acc < 0.57:
        print("  [REAL EDGE] OOS accuracy 54-57% — genuine predictive signal detected.")
        print("              Move to walk-forward validation. Add XGBoost as second model.")
    else:
        print("  [STRONG EDGE] OOS accuracy > 57% — verify no lookahead bias first!")
        print("                Run validate_temporal_integrity() before trusting this.")

    if gap > 0.10:
        print(f"\n  [OVERFIT WARNING] Train-test gap = {gap:.3f}")
        print("  This means the model is memorising training data.")
        print("  Reduce max_depth to 4, increase min_child_samples to 50.")
    elif gap > 0.05:
        print(f"\n  [MILD OVERFIT] Gap = {gap:.3f} — acceptable for now.")
    else:
        print(f"\n  [OK] Overfit gap = {gap:.3f} — well generalised model.")

    print("=" * 55)


# ── Feature importance ────────────────────────────────────────────────────────

def get_feature_importance(model, feature_cols: list) -> pd.DataFrame:
    """Returns a sorted DataFrame of feature importances."""
    importance = pd.DataFrame({
        "feature":    feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    return importance


def print_feature_importance(model, feature_cols: list, top_n: int = 10) -> None:
    """Prints top-N feature importances."""
    imp = get_feature_importance(model, feature_cols)
    print(f"\n  Top {top_n} feature importances:")
    for _, row in imp.head(top_n).iterrows():
        bar = "█" * int(row["importance"] / imp["importance"].max() * 20)
        print(f"    {row['feature']:<20} {bar}")


# ── Save / Load ───────────────────────────────────────────────────────────────

def save_model(model, filename: str = "lgbm_baseline.pkl") -> None:
    """Saves model to models/ directory."""
    MODELS_DIR.mkdir(exist_ok=True)
    path = MODELS_DIR / filename
    joblib.dump(model, path)
    print(f"  [SAVED] Model saved to: {path}")


def load_model(filename: str = "lgbm_baseline.pkl"):
    """Loads model from models/ directory."""
    path = MODELS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    model = joblib.load(path)
    print(f"  [LOADED] Model loaded from: {path}")
    return model
