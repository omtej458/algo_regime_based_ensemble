"""
validation.py
=============
Walk-forward cross-validation engine.

This is the ONLY valid validation methodology for time-series financial data.
Random split and k-fold cross-validation will give you inflated, meaningless metrics.

Protocol:
  - Train window : 2520 bars (~1 year of 1H data)
  - Test window  : 630 bars (~3 months)
  - Step size    : 210 bars (~1 month)
  - Embargo gap  : 24 bars (24-hour gap between train end and test start)
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


# ── Walk-forward engine ───────────────────────────────────────────────────────

def walk_forward_validate(
    df:               pd.DataFrame,
    feature_cols:     list,
    label_col:        str = "label_binary",
    train_bars:       int = 2520,
    test_bars:        int = 630,
    step_bars:        int = 210,
    embargo_bars:     int = 24,
    confidence_thresh: float = 0.60,
    model_builder=None,
) -> pd.DataFrame:
    """
    Runs walk-forward validation over the full dataset.

    Parameters
    ----------
    df                : Clean DataFrame (time-outs removed) with features + label
    feature_cols      : List of feature column names
    label_col         : Binary label column (0 or 1)
    train_bars        : Number of bars in each training window
    test_bars         : Number of bars in each test window
    step_bars         : How many bars to slide forward each iteration
    embargo_bars      : Gap between train end and test start (prevents leakage)
    confidence_thresh : Minimum probability to count as a "high-confidence" signal
    model_builder     : Function that returns a fresh untrained model (default: LGBM baseline)

    Returns
    -------
    pd.DataFrame with one row per fold and all metrics
    """
    from src.models import build_lgbm_baseline

    if model_builder is None:
        model_builder = build_lgbm_baseline

    results = []
    n = len(df)
    start = 0
    fold = 0

    print("=" * 65)
    print("  Walk-forward validation")
    print(f"  Train: {train_bars} bars | Test: {test_bars} bars | "
          f"Step: {step_bars} bars | Embargo: {embargo_bars} bars")
    print("=" * 65)

    while start + train_bars + embargo_bars + test_bars <= n:
        fold += 1
        train_end  = start + train_bars
        test_start = train_end + embargo_bars
        test_end   = test_start + test_bars

        X_tr = df[feature_cols].iloc[start:train_end]
        y_tr = df[label_col].iloc[start:train_end]
        X_te = df[feature_cols].iloc[test_start:test_end]
        y_te = df[label_col].iloc[test_start:test_end]

        # Skip fold if too few samples
        if len(X_tr) < 200 or len(X_te) < 20:
            start += step_bars
            continue

        # Drop any remaining NaN rows within this fold
        tr_mask = ~(X_tr.isna().any(axis=1) | y_tr.isna())
        te_mask = ~(X_te.isna().any(axis=1) | y_te.isna())
        X_tr, y_tr = X_tr[tr_mask], y_tr[tr_mask]
        X_te, y_te = X_te[te_mask], y_te[te_mask]

        if len(X_te) < 10:
            start += step_bars
            continue

        # Train fresh model
        model = model_builder()
        model.fit(X_tr, y_tr)

        # Predict
        preds = model.predict(X_te)
        probs = model.predict_proba(X_te)[:, 1]
        acc   = accuracy_score(y_te, preds)

        # AUC
        try:
            auc = roc_auc_score(y_te, probs)
        except Exception:
            auc = np.nan

        # High-confidence signals
        hc_mask  = (probs >= confidence_thresh) | (probs <= (1 - confidence_thresh))
        hc_count = hc_mask.sum()
        if hc_count >= 10:
            hc_acc = accuracy_score(y_te[hc_mask], preds[hc_mask])
        else:
            hc_acc = np.nan

        # Simulated profit factor (simple)
        # Correct predictions = +RR (reward/risk) | Wrong predictions = -1
        rr = 1.5  # matches TP/SL ratio of 1.5 / 1.0
        gains  = acc * rr
        losses = (1 - acc) * 1.0
        pf = gains / losses if losses > 0 else np.nan

        fold_result = {
            "fold":            fold,
            "test_start":      df.index[test_start].date(),
            "test_end":        df.index[test_end - 1].date(),
            "n_train":         len(y_tr),
            "n_test":          len(y_te),
            "n_hc_signals":    hc_count,
            "accuracy":        round(acc, 4),
            "hc_accuracy":     round(hc_acc, 4) if not np.isnan(hc_acc) else np.nan,
            "auc_roc":         round(auc, 4) if not np.isnan(auc) else np.nan,
            "profit_factor":   round(pf, 3) if not np.isnan(pf) else np.nan,
            "long_pct":        round(y_te.mean(), 3),
        }
        results.append(fold_result)

        # Print fold summary
        hc_str = f"{hc_acc:.3f}" if not np.isnan(hc_acc) else "N/A"
        print(f"  Fold {fold:02d} | {df.index[test_start].date()} → "
              f"{df.index[test_end-1].date()} | "
              f"acc={acc:.3f} | hc_acc={hc_str} | "
              f"n={len(y_te)}")

        start += step_bars

    results_df = pd.DataFrame(results)
    _print_aggregate_report(results_df)
    return results_df


def _print_aggregate_report(results_df: pd.DataFrame) -> None:
    """Prints aggregate metrics across all walk-forward folds."""
    if results_df.empty:
        print("\n[ERROR] No folds completed. Check that your dataset is large enough.")
        print(f"  Need at least {2520 + 24 + 630} rows in the clean training set.")
        return

    print("\n" + "=" * 65)
    print("  Aggregate walk-forward results")
    print("=" * 65)
    print(f"  Folds completed        : {len(results_df)}")
    print(f"  Total test signals     : {results_df['n_test'].sum():,}")
    print()

    acc = results_df["accuracy"]
    print(f"  OOS Direction Accuracy :")
    print(f"    Mean               : {acc.mean():.3f}  ({acc.mean()*100:.1f}%)")
    print(f"    Std                : {acc.std():.3f}")
    print(f"    Best fold          : {acc.max():.3f}")
    print(f"    Worst fold         : {acc.min():.3f}")

    hc = results_df["hc_accuracy"].dropna()
    if len(hc) > 0:
        print(f"\n  High-Confidence Accuracy:")
        print(f"    Mean               : {hc.mean():.3f}  ({hc.mean()*100:.1f}%)")
        print(f"    Folds with HC data : {len(hc)} / {len(results_df)}")

    pf = results_df["profit_factor"].dropna()
    if len(pf) > 0:
        print(f"\n  Simulated Profit Factor:")
        print(f"    Mean               : {pf.mean():.3f}")
        print(f"    Folds > 1.0        : {(pf > 1.0).sum()} / {len(pf)}")

    print()
    _interpret_results(results_df)
    print("=" * 65)


def _interpret_results(results_df: pd.DataFrame) -> None:
    """Gives a concrete diagnosis of the walk-forward results."""
    mean_acc  = results_df["accuracy"].mean()
    std_acc   = results_df["accuracy"].std()
    worst_acc = results_df["accuracy"].min()

    print("  Diagnosis:")

    if mean_acc < 0.50:
        print("  [CRITICAL] Mean OOS accuracy < 50%.")
        print("    Action: Check feature shift (+1 bar) and label construction.")

    elif mean_acc < 0.52:
        print("  [BASELINE] Mean OOS accuracy 50-52% — pipeline is clean, no edge yet.")
        print("    Action: Add 4H/Daily regime features. Encode CB stance and VIX term structure.")

    elif mean_acc < 0.54:
        print("  [EARLY EDGE] Mean OOS accuracy 52-54% — weak predictive signal present.")
        print("    Action: Add XGBoost as second model. Build LGBM+XGB ensemble.")

    elif mean_acc < 0.57:
        print("  [REAL EDGE] Mean OOS accuracy 54-57% — genuine edge. Focus on consistency.")
        print("    Action: Add regime-gated routing. Train separate models per VIX regime.")

    else:
        print("  [STRONG SIGNAL] Mean OOS accuracy > 57%.")
        print("    Action: Verify no lookahead. Then add CatBoost + meta-learner stacking.")

    if std_acc > 0.05:
        print(f"\n  [CONSISTENCY WARNING] Std = {std_acc:.3f} — high variance across folds.")
        print("    This means the model performs very differently in different periods.")
        print("    Action: Add regime features to help the model identify its own edge conditions.")

    if worst_acc < 0.48:
        print(f"\n  [STRESS TEST FAIL] Worst fold = {worst_acc:.3f}.")
        print("    The model falls apart in some market conditions.")
        print("    Action: Identify which dates that fold covers and what regime was active.")


# ── Save results ──────────────────────────────────────────────────────────────

def save_results(results_df: pd.DataFrame, filename: str = "walk_forward_results.csv") -> None:
    """Saves walk-forward results to results/ directory."""
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / filename
    results_df.to_csv(path, index=False)
    print(f"  [SAVED] Results saved to: {path}")
