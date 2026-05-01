"""
targets.py
==========
Triple-barrier label construction.

This is the single most important step in the pipeline.
We do NOT predict "did the next bar close up or down?" — that is noise.
We predict: within the next T bars, does price hit the TP first, SL first, or neither?

Label encoding:
  +1  = TP hit first (long trade would have won)
  -1  = SL hit first (long trade would have lost)
   0  = Time barrier hit (no decisive move — excluded from training)

Binary label (for classifier):
   1  = TP hit first (positive class)
   0  = SL hit first (negative class)
"""

import pandas as pd
import numpy as np


# ── Triple-barrier labeller ───────────────────────────────────────────────────

def compute_triple_barrier_labels(
    df: pd.DataFrame,
    n_tp: float = 1.5,
    n_sl: float = 1.0,
    t_bars: int = 12,
    atr_col: str = "atr_14",
) -> pd.DataFrame:
    """
    Applies the triple-barrier labelling method to every bar in df.

    Parameters
    ----------
    df     : DataFrame with at least 'Close' and atr_col columns
    n_tp   : TP multiplier (TP = close + n_tp * ATR)
    n_sl   : SL multiplier (SL = close - n_sl * ATR)
    t_bars : Maximum number of 1H bars to wait for a barrier hit (default 12 = 12 hours)
    atr_col: Column name for ATR values

    Returns
    -------
    df with new columns:
        label_raw    : +1, -1, or 0
        label_binary : 1 (TP hit) or 0 (SL hit) — NaN for time-outs
        tp_price     : where the TP was set for this bar
        sl_price     : where the SL was set for this bar
    """
    if atr_col not in df.columns:
        raise ValueError(
            f"Column '{atr_col}' not found. Run compute_atr() in features.py first."
        )

    closes = df["Close"].values
    atrs   = df[atr_col].values
    n      = len(df)

    labels = np.zeros(n, dtype=float)
    tp_prices = np.full(n, np.nan)
    sl_prices = np.full(n, np.nan)

    print(f"Computing triple-barrier labels on {n:,} bars...")
    print(f"  TP = ATR × {n_tp} | SL = ATR × {n_sl} | Time = {t_bars} bars ({t_bars}H)")

    for i in range(n - t_bars):
        entry = closes[i]
        atr_i = atrs[i]

        if np.isnan(atr_i) or atr_i == 0:
            labels[i] = 0
            continue

        tp = entry + n_tp * atr_i
        sl = entry - n_sl * atr_i
        tp_prices[i] = tp
        sl_prices[i] = sl

        label = 0  # default: time barrier
        for j in range(i + 1, min(i + t_bars + 1, n)):
            if closes[j] >= tp:
                label = 1
                break
            elif closes[j] <= sl:
                label = -1
                break
        labels[i] = label

    df = df.copy()
    df["label_raw"]    = labels
    df["tp_price"]     = tp_prices
    df["sl_price"]     = sl_prices

    # Binary label: 1 = long won, 0 = long lost, NaN = time-out (exclude from training)
    df["label_binary"] = df["label_raw"].replace(0, np.nan)
    df["label_binary"] = (df["label_binary"] == 1).where(df["label_raw"] != 0, np.nan)
    df["label_binary"] = df["label_binary"].astype(float)

    return df


# ── Label summary ─────────────────────────────────────────────────────────────

def summarise_labels(df: pd.DataFrame) -> None:
    """Prints a clear summary of label distribution."""
    if "label_raw" not in df.columns:
        print("[ERROR] Run compute_triple_barrier_labels() first.")
        return

    total    = len(df)
    tp_hits  = (df["label_raw"] == 1).sum()
    sl_hits  = (df["label_raw"] == -1).sum()
    timeouts = (df["label_raw"] == 0).sum()
    usable   = tp_hits + sl_hits

    print("\n" + "=" * 55)
    print("  Triple-barrier label summary")
    print("=" * 55)
    print(f"  Total bars          : {total:,}")
    print(f"  TP hit (+1)         : {tp_hits:,}  ({tp_hits/total*100:.1f}%)")
    print(f"  SL hit (-1)         : {sl_hits:,}  ({sl_hits/total*100:.1f}%)")
    print(f"  Time-out (0)        : {timeouts:,}  ({timeouts/total*100:.1f}%)")
    print(f"  Usable for training : {usable:,}  ({usable/total*100:.1f}%)")

    if usable > 0:
        class_balance = tp_hits / usable * 100
        print(f"\n  Class balance (TP / usable): {class_balance:.1f}% long wins")
        if class_balance < 40 or class_balance > 60:
            print("  [NOTE] Class imbalance detected — LightGBM class_weight='balanced' "
                  "handles this automatically.")
        else:
            print("  [OK] Class balance is within acceptable range (40-60%).")

    print("=" * 55)


def get_clean_training_set(df: pd.DataFrame, feature_cols: list) -> tuple:
    """
    Filters out time-out bars (label_raw == 0) and returns clean X, y arrays.

    Parameters
    ----------
    df           : DataFrame with label_binary column
    feature_cols : List of feature column names

    Returns
    -------
    (X, y, df_clean)
      X        : pd.DataFrame of features
      y        : pd.Series of binary labels (0 or 1)
      df_clean : full filtered DataFrame
    """
    df_clean = df[df["label_raw"] != 0].copy()
    df_clean = df_clean.dropna(subset=feature_cols + ["label_binary"])

    X = df_clean[feature_cols]
    y = df_clean["label_binary"].astype(int)

    print(f"  Clean training set: {len(df_clean):,} rows "
          f"(removed {len(df) - len(df_clean):,} time-outs and NaN rows)")
    return X, y, df_clean
