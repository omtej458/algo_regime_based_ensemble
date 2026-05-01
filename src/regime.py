"""
regime.py
=========
Hidden Markov Model regime classifier.

Identifies 3 market regimes from price data:
  Regime 0 — Quiet trend    (low vol, directional, ADX rising)
  Regime 1 — Volatile trend (high vol, directional)
  Regime 2 — Ranging/chop   (low ADX, mean-reverting, low signal quality)

The regime label is used by the signal gate to:
  - Raise confidence threshold in Regime 2 (ranging = model has less edge)
  - Lower threshold slightly in Regime 0 (strong trend = model is most reliable)
"""

import numpy as np
import pandas as pd
from hmmlearn import hmm
import warnings
warnings.filterwarnings('ignore')


N_REGIMES = 3


# def build_regime_features(df: pd.DataFrame) -> np.ndarray:
#     """
#     Builds the 3-feature input for the HMM:
#       1. Normalised ATR (volatility level)
#       2. ADX (trend strength)
#       3. Absolute 8-bar log return (directional momentum magnitude)

#     All inputs are standardised to zero mean, unit variance.
#     Returns a 2D numpy array of shape (n_bars, 3).
#     """
#     if not all(c in df.columns for c in ['atr_norm', 'adx_14', 'ret_8']):
#         raise ValueError("DataFrame must contain atr_norm, adx_14, ret_8 columns.")

#     feat = pd.DataFrame({
#         'vol':       df['atr_norm'],
#         'trend':     df['adx_14'],
#         'momentum':  df['ret_8'].abs(),
#     }).dropna()

#     # Standardise each feature
#     for col in feat.columns:
#         mean = feat[col].mean()
#         std  = feat[col].std()
#         if std > 0:
#             feat[col] = (feat[col] - mean) / std

#     return feat.values, feat.index

def build_regime_features(df: pd.DataFrame) -> np.ndarray:
    """
    Builds 5-feature input for HMM on multi-year data.
    Adds VIX regime and yield direction to help separate
    macro-driven regimes from pure technical regimes.
    
    Feature order:
      0: atr_norm       — volatility level
      1: adx_14         — trend strength  
      2: ret_8 abs      — momentum magnitude
      3: vix_regime     — macro risk environment
      4: ret_1d_5 abs   — daily momentum magnitude
    """
    required = ['atr_norm', 'adx_14', 'ret_8']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    feat = pd.DataFrame({
        'vol':        df['atr_norm'],
        'trend':      df['adx_14'],
        'momentum':   df['ret_8'].abs(),
        'vix':        df['vix_regime'] if 'vix_regime' in df.columns
                      else pd.Series(1, index=df.index),
        'daily_mom':  df['ret_1d_5'].abs() if 'ret_1d_5' in df.columns
                      else pd.Series(0, index=df.index),
    }).dropna()

    # Standardise each feature
    for col in feat.columns:
        mean = feat[col].mean()
        std  = feat[col].std()
        if std > 0:
            feat[col] = (feat[col] - mean) / std

    return feat.values, feat.index

def fit_hmm(X: np.ndarray, n_regimes: int = N_REGIMES, n_iter: int = 100) -> hmm.GaussianHMM:
    """
    Fits a Gaussian HMM to the regime feature matrix.
    Uses diagonal covariance for stability on small datasets.
    """
    model = hmm.GaussianHMM(
        n_components=n_regimes,
        covariance_type='diag',
        n_iter=n_iter,
        random_state=42,
        verbose=False,
    )
    model.fit(X)
    return model


def predict_regimes(hmm_model: hmm.GaussianHMM, X: np.ndarray) -> np.ndarray:
    """Returns the most likely regime sequence for the given features."""
    return hmm_model.predict(X)


# def label_regimes(hmm_model: hmm.GaussianHMM,
#                   X: np.ndarray,
#                   raw_labels: np.ndarray) -> dict:
#     """
#     Maps HMM integer states (0, 1, 2) to economic regime labels.
#     Identifies which HMM state corresponds to which regime by
#     looking at the mean feature values per state.

#     Returns a dict mapping {hmm_state: regime_name}
#     """
#     means = hmm_model.means_  # shape (n_regimes, n_features)
#     # Feature order: [vol, trend, momentum]
#     # Regime 0 (quiet trend):    low vol, high trend, medium momentum
#     # Regime 1 (volatile trend): high vol, high trend, high momentum
#     # Regime 2 (ranging/chop):   low vol, low trend, low momentum

#     vol_rank   = np.argsort(means[:, 0])  # ascending vol
#     trend_rank = np.argsort(means[:, 1])  # ascending trend

#     state_map = {}
#     # Highest trend + lowest vol = quiet trend (best for model)
#     quiet_trend_state = trend_rank[-1]
#     state_map[quiet_trend_state] = 'quiet_trend'

#     # Highest vol = volatile trend
#     volatile_state = vol_rank[-1]
#     if volatile_state != quiet_trend_state:
#         state_map[volatile_state] = 'volatile_trend'
#     else:
#         volatile_state = vol_rank[-2]
#         state_map[volatile_state] = 'volatile_trend'

#     # Remaining = ranging
#     for s in range(N_REGIMES):
#         if s not in state_map:
#             state_map[s] = 'ranging'

#     return state_map

def label_regimes(hmm_model, X, raw_labels):
    """
    Maps 3 HMM states to economic regime labels using vol + trend.

      quiet_trend  : lowest vol + highest trend  (best for model)
      volatile_trend: highest vol + any trend     (news-driven, fast moves)
      ranging      : remaining state             (chop, consolidation)
    """
    means = hmm_model.means_
    # Feature order: [vol, trend, momentum, vix, daily_mom]

    vol_rank   = np.argsort(means[:, 0])   # ascending: lowest vol first
    trend_rank = np.argsort(means[:, 1])   # ascending: lowest trend first

    state_map = {}

    # volatile_trend: highest vol state
    state_map[vol_rank[-1]] = 'volatile_trend'

    # quiet_trend: highest trend state that is NOT volatile_trend
    for s in trend_rank[::-1]:
        if s not in state_map:
            state_map[s] = 'quiet_trend'
            break

    # ranging: whatever remains
    for s in range(len(means)):
        if s not in state_map:
            state_map[s] = 'ranging'

    return state_map

def add_regime_to_df(df: pd.DataFrame,
                     hmm_model: hmm.GaussianHMM,
                     regime_map: dict) -> pd.DataFrame:
    """
    Adds regime_label and regime_name columns to the DataFrame.
    Bars with missing features get regime_label = -1.
    """
    X, feat_index = build_regime_features(df)
    raw_labels = predict_regimes(hmm_model, X)

    regime_series = pd.Series(raw_labels, index=feat_index)
    regime_name_series = regime_series.map(regime_map)

    df = df.copy()
    df['regime_label'] = regime_series.reindex(df.index).fillna(-1).astype(int)
    df['regime_name']  = regime_name_series.reindex(df.index).fillna('unknown')
    return df


def summarise_regimes(df: pd.DataFrame) -> None:
    """Prints regime distribution and accuracy per regime."""
    if 'regime_name' not in df.columns:
        print("[ERROR] Run add_regime_to_df() first.")
        return

    print("\n" + "=" * 55)
    print("  Regime Distribution")
    print("=" * 55)
    counts = df['regime_name'].value_counts()
    for regime, count in counts.items():
        pct = count / len(df) * 100
        print(f"  {regime:<20} : {count:,}  ({pct:.1f}%)")
    print("=" * 55)