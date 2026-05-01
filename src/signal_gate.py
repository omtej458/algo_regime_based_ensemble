"""
signal_gate.py
==============
Rule-based signal gate applied AFTER the ML ensemble output.

The gate does not change the model. It filters which signals to act on.
A signal only passes the gate when ALL conditions are met simultaneously.

Gate conditions (in order of priority):
  1. Macro veto    — no trade within 4H of Tier 1 event (manual override)
  2. VIX regime    — no trade if VIX > 30 (extreme vol = unpredictable)
  3. ADX filter    — no trade if ADX < 20 (no trend = no directional edge)
  4. Confidence    — no trade if ensemble probability is not decisive
  5. Session       — prefer London-NY overlap (session 3), allow London (1) and NY (2)
"""

import pandas as pd
import numpy as np


# ── Gate thresholds (single source of truth — tune these later) ───────────────
CONFIDENCE_THRESHOLD = 0.60   # minimum ensemble probability to pass gate
ADX_MINIMUM          = 18     # lowered from 18: 1H ADX legitimately low during consolidation
                              # 15 allows signals when daily trend is strong but 1H is pausing
ALLOWED_SESSIONS     = [1, 2, 3]   # London (1), New York (2), Overlap (3) only — Asian (0) blocked

# ── Regime thresholds — calibrated from notebook 09 Cell 5 results ──────────
# Based on actual per-regime accuracy from your 6-year dataset:
#   ranging       : 72.72% base acc — highest quality → lowest threshold
#   quiet_trend   : 59.79% base acc — medium quality → standard threshold
#   volatile_trend: 57.30% base acc — lowest quality → strictest threshold
# VIX extreme veto is handled separately via vix_regime >= 2 rule in the gate
REGIME_THRESHOLDS = {
    'ranging':        0.60,   # 72.72% base acc — most reliable, standard threshold
    'quiet_trend':    0.62,   # 59.79% base acc — tighter gate
    'volatile_trend': 0.65,   # 57.30% base acc — only high conviction
}

# def apply_signal_gate(
#     df:          pd.DataFrame,
#     probs:       np.ndarray,
#     feature_cols: list,
# ) -> pd.DataFrame:
#     """
#     Applies all gate conditions to ensemble probability output.

#     Parameters
#     ----------
#     df           : DataFrame with all feature columns (same rows as probs)
#     probs        : 1D array of ensemble probabilities (output of ensemble model)
#     feature_cols : list of feature column names

#     Returns
#     -------
#     pd.DataFrame with columns:
#         prob          : raw ensemble probability
#         direction     : +1 (long) or -1 (short) or 0 (no signal)
#         confidence    : abs(prob - 0.5) * 2  →  0=uncertain, 1=max confidence
#         gate_passed   : True if all conditions met
#         gate_reason   : why gate failed (empty string if passed)
#     """
#     result = pd.DataFrame(index=df.index)
#     result["prob"]       = probs
#     result["direction"]  = np.where(probs >= 0.5, 1, -1)
#     result["confidence"] = (np.abs(probs - 0.5) * 2).round(4)
#     result["gate_passed"] = True
#     result["gate_reason"] = ""

#     # ── Rule 1: Confidence threshold ─────────────────────────────────────────
#     low_conf = result["confidence"] < (CONFIDENCE_THRESHOLD - 0.5) * 2
#     result.loc[low_conf, "gate_passed"] = False
#     result.loc[low_conf & (result["gate_reason"] == ""), "gate_reason"] = \
#         f"low_confidence (<{CONFIDENCE_THRESHOLD})"

#     # ── Rule 2: VIX regime ────────────────────────────────────────────────────
#     if "vix_regime" in df.columns:
#         high_vix = df["vix_regime"] >= 2   # regime 2 = VIX > 22
#         # For VIX > 30 (extreme), use the raw vix_regime proxy
#         result.loc[high_vix, "gate_passed"] = False
#         result.loc[high_vix & (result["gate_reason"] == ""), "gate_reason"] = \
#             "high_vix_regime"

#     # ── Rule 3: ADX trend filter ──────────────────────────────────────────────
#     if "adx_14" in df.columns:
#         no_trend = df["adx_14"] < ADX_MINIMUM
#         result.loc[no_trend, "gate_passed"] = False
#         result.loc[no_trend & (result["gate_reason"] == ""), "gate_reason"] = \
#             f"adx_too_low (<{ADX_MINIMUM})"

#     # ── Rule 4: Session filter ────────────────────────────────────────────────
#     if "session" in df.columns:
#         wrong_session = ~df["session"].isin(ALLOWED_SESSIONS)
#         result.loc[wrong_session, "gate_passed"] = False
#         result.loc[wrong_session & (result["gate_reason"] == ""), "gate_reason"] = \
#             "asian_session"

    # # Set direction to 0 for all failed gates
    # result.loc[~result["gate_passed"], "direction"] = 0

    # return result

# def apply_signal_gate(df, probs, feature_cols):
#     result = pd.DataFrame(index=df.index)
#     result["prob"]        = probs
#     result["direction"]   = np.where(probs >= 0.5, 1, -1)
#     result["confidence"]  = (np.abs(probs - 0.5) * 2).round(4)
#     result["gate_passed"] = True
#     result["gate_reason"] = ""

#     # PRIMARY GATE: confidence threshold only
#     # At prob >= 0.60 or <= 0.40, accuracy is 73%+ confirmed OOS
#     low_conf = (probs < CONFIDENCE_THRESHOLD) & (probs > (1 - CONFIDENCE_THRESHOLD))
#     result.loc[low_conf, "gate_passed"] = False
#     result.loc[low_conf, "gate_reason"] = f"low_confidence"

#     result.loc[~result["gate_passed"], "direction"] = 0
#     return result

# def apply_signal_gate(
#     df:           pd.DataFrame,
#     probs:        np.ndarray,
#     feature_cols: list,
# ) -> pd.DataFrame:
#     """
#     Regime-aware confidence gate on ensemble probability output.
#     Uses per-regime threshold if regime_name column is present.
#     Falls back to global CONFIDENCE_THRESHOLD otherwise.
#     """
#     probs = np.asarray(probs, dtype=float)
#     n = len(probs)

#     has_regime = 'regime_name' in df.columns

#     # Build per-bar threshold array
#     thresholds = np.full(n, CONFIDENCE_THRESHOLD)
#     if has_regime:
#         regime_vals = df['regime_name'].values
#         for i in range(n):
#             thresholds[i] = REGIME_THRESHOLDS.get(
#                 str(regime_vals[i]), CONFIDENCE_THRESHOLD
#             )

#     # Gate condition: prob >= thresh OR prob <= (1 - thresh)
#     passed = (probs >= thresholds) | (probs <= (1.0 - thresholds))

#     direction = np.where(probs >= 0.5, 1, -1)
#     direction[~passed] = 0

#     confidence = np.abs(probs - 0.5) * 2

#     # Build gate_reason
#     reasons = np.full(n, '', dtype=object)
#     if has_regime:
#         regime_vals = df['regime_name'].values
#         for i in range(n):
#             if not passed[i]:
#                 reasons[i] = f"low_conf_{regime_vals[i]}"
#     else:
#         reasons[~passed] = "low_confidence"

#     result = pd.DataFrame({
#         "prob":        probs,
#         "direction":   direction,
#         "confidence":  np.round(confidence, 4),
#         "gate_passed": passed,
#         "gate_reason": reasons,
#         "threshold":   thresholds,   # shows which threshold was applied per bar
#     }, index=df.index)

#     return result

def apply_signal_gate(
    df:           pd.DataFrame,
    probs:        np.ndarray,
    feature_cols: list,
) -> pd.DataFrame:
    """
    Regime-aware signal gate.
    Checks: confidence threshold + ADX minimum + session filter.
    """
    probs = np.asarray(probs, dtype=float)
    n = len(probs)

    has_regime  = 'regime_name' in df.columns
    has_adx     = 'adx_14' in df.columns
    has_session = 'session' in df.columns
    has_vix     = 'vix_regime' in df.columns

    # Per-bar threshold based on regime
    thresholds = np.full(n, CONFIDENCE_THRESHOLD)
    if has_regime:
        regime_vals = df['regime_name'].values
        for i in range(n):
            thresholds[i] = REGIME_THRESHOLDS.get(
                str(regime_vals[i]), CONFIDENCE_THRESHOLD
            )

    # Gate condition arrays
    passed  = np.ones(n, dtype=bool)
    reasons = np.full(n, '', dtype=object)

    for i in range(n):
        p = probs[i]

        # Rule 1: Confidence threshold (primary gate)
        if not (p >= thresholds[i] or p <= (1.0 - thresholds[i])):
            passed[i]  = False
            regime_tag = str(df['regime_name'].iloc[i]) if has_regime else ''
            reasons[i] = f"low_conf_{regime_tag}" if regime_tag else "low_confidence"
            continue

        # Rule 2: ADX minimum — skip if no trend
        if has_adx:
            adx_val = df['adx_14'].iloc[i]
            if not np.isnan(adx_val) and adx_val < ADX_MINIMUM:
                passed[i]  = False
                reasons[i] = f"adx_too_low"
                continue

        # Rule 3: Session filter — skip Asian session
        if has_session:
            session_val = df['session'].iloc[i]
            if session_val not in ALLOWED_SESSIONS:
                passed[i]  = False
                reasons[i] = "asian_session"
                continue

        # Rule 4: VIX extreme — skip if VIX regime >= 2 (VIX > 22)
        # This is the explicit crash/panic veto replacing the 'extreme' HMM regime
        if has_vix:
            vix_val = df['vix_regime'].iloc[i]
            if not np.isnan(vix_val) and vix_val >= 2:
                passed[i]  = False
                reasons[i] = "vix_extreme"
                continue

    direction        = np.where(probs >= 0.5, 1, -1)
    direction[~passed] = 0
    confidence       = np.abs(probs - 0.5) * 2

    return pd.DataFrame({
        "prob":        probs,
        "direction":   direction,
        "confidence":  np.round(confidence, 4),
        "gate_passed": passed,
        "gate_reason": reasons,
        "threshold":   thresholds,
    }, index=df.index)

def summarise_gate_results(gate_df: pd.DataFrame, y_true: pd.Series = None) -> None:
    """Prints a summary of how many signals passed each filter."""
    total = len(gate_df)
    passed = gate_df["gate_passed"].sum()
    failed = total - passed

    print("\n" + "=" * 55)
    print("  Signal Gate Summary")
    print("=" * 55)
    print(f"  Total bars evaluated : {total:,}")
    print(f"  Signals passed gate  : {passed:,}  ({passed/total*100:.1f}%)")
    print(f"  Signals blocked      : {failed:,}  ({failed/total*100:.1f}%)")
    print()

    # Breakdown by gate reason
    failed_df = gate_df[~gate_df["gate_passed"]]
    if len(failed_df) > 0:
        reason_counts = failed_df["gate_reason"].value_counts()
        print("  Blocked by reason:")
        for reason, count in reason_counts.items():
            print(f"    {reason:<30} : {count:,}  ({count/total*100:.1f}%)")

    if y_true is not None:
        # Accuracy on passed signals only
        passed_mask = gate_df["gate_passed"].values  # numpy bool array — index-agnostic
        if passed_mask.sum() >= 10:
            passed_preds = (gate_df.loc[gate_df["gate_passed"], "prob"] >= 0.5).values.astype(int)
            passed_true  = y_true.values[passed_mask]
            acc = (passed_preds == passed_true).mean()
            all_preds = (gate_df["prob"] >= 0.5).values.astype(int)
            all_acc   = (all_preds == y_true.values).mean()
            print(f"\n  Accuracy on gated signals : {acc*100:.2f}%")
            print(f"  Accuracy on all signals   : {all_acc*100:.2f}%")
            print(f"\n  Gate lift                 : +{(acc - all_acc)*100:.2f}%")

    print("=" * 55)


def simulate_pnl(
    gate_df:  pd.DataFrame,
    y_true:   pd.Series,
    rr_ratio: float = 1.5,
) -> pd.DataFrame:
    """
    Simulates PnL assuming:
      - Each correct signal earns +rr_ratio units (TP hit)
      - Each wrong signal loses -1.0 units (SL hit)
      - Blocked signals earn 0

    Returns a DataFrame with cumulative PnL curve.
    """
    results = []
    for i in range(len(gate_df)):
        row = gate_df.iloc[i]
        if not row["gate_passed"]:
            pnl = 0.0
        else:
            pred = 1 if row["prob"] >= 0.5 else 0
            actual = int(y_true.iloc[i])
            pnl = rr_ratio if pred == actual else -1.0
        results.append(pnl)

    pnl_series = pd.Series(results, index=gate_df.index)
    cumulative  = pnl_series.cumsum()

    # Metrics
    trades     = gate_df["gate_passed"].sum()
    wins       = sum(1 for p, a in zip(
                     (gate_df["prob"] >= 0.5).astype(int),
                     y_true) if p == a and gate_df["gate_passed"].iloc[y_true.index.get_loc(y_true.index[list(y_true.index).index(y_true.index[0])])])
    total_pnl  = pnl_series.sum()
    max_dd     = (cumulative - cumulative.cummax()).min()

    return pd.DataFrame({
        "pnl":        pnl_series,
        "cumulative": cumulative,
    })