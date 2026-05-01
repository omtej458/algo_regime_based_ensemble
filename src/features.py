"""
features.py
===========
All feature engineering for the EURUSD ML ensemble.
Every function is pure (no side effects) and reproducible.

Feature Tiers:
  Tier A — Price structure (log returns, ATR, body ratio, range position)
  Tier B — Momentum (RSI deviation, ADX, ATR rate of change)
  Tier D — Regime (VIX regime, DXY slope, session label, day of week)
  Tier E — Z-scored return
"""

import pandas as pd
import numpy as np
import ta
from src.data_loader import align_daily_to_1h, align_4h_to_1h

# ── Feature column names (single source of truth) ────────────────────────────
FEATURE_COLS = [
    # Tier A
    "ret_1",
    "ret_4",
    "ret_8",
    "ret_24",
    "atr_norm",
    "body_ratio",
    "range_pos_20",
    # Tier B
    "rsi_dev_14",
    "adx_14",
    "atr_roc_8",
    # Tier D
    "vix_regime",
    "dxy_slope",
    "us10y_direction",
    "us10y_z20",
    "rate_differential",      # ← NEW
    # "rate_diff_direction",    # ← NEW
    "session",
    # "day_of_week",
    # Tier E
    "ret_1_z24",
    # 4H context
    "ret_4h_8",
    "rsi_dev_4h",
    # Daily context
    "ret_1d_5",
]


# ── Tier A: Price structure ───────────────────────────────────────────────────

def compute_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Multi-horizon log returns. All shifted +1 bar."""
    close = df["Close"]
    df["ret_1"]  = np.log(close / close.shift(1))
    df["ret_4"]  = np.log(close / close.shift(4))
    df["ret_8"]  = np.log(close / close.shift(8))
    df["ret_24"] = np.log(close / close.shift(24))
    return df


def compute_atr(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """ATR(14) and normalised ATR."""
    atr_ind = ta.volatility.AverageTrueRange(
        df["High"], df["Low"], df["Close"], window=window
    )
    df["atr_14"]   = atr_ind.average_true_range()
    df["atr_norm"] = df["atr_14"] / df["Close"].replace(0, np.nan)
    return df


def compute_body_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Bar body ratio: |close - open| / (high - low). Measures bar conviction."""
    hl = (df["High"] - df["Low"]).replace(0, np.nan)
    df["body_ratio"] = (df["Close"] - df["Open"]).abs() / hl
    df["body_ratio"] = df["body_ratio"].clip(0, 1)
    return df


def compute_range_position(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Where is close within the last N-bar high/low range. 0=bottom, 1=top."""
    roll_high = df["High"].rolling(window).max()
    roll_low  = df["Low"].rolling(window).min()
    rng = (roll_high - roll_low).replace(0, np.nan)
    df["range_pos_20"] = (df["Close"] - roll_low) / rng
    return df


# ── Tier B: Momentum ──────────────────────────────────────────────────────────

def compute_rsi_deviation(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """RSI deviation from 50. Positive = bullish momentum, negative = bearish."""
    rsi = ta.momentum.RSIIndicator(df["Close"], window=window).rsi()
    df["rsi_dev_14"] = rsi - 50.0
    return df


def compute_adx(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """ADX trend strength indicator."""
    adx_ind = ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"], window=window)
    df["adx_14"] = adx_ind.adx()
    return df


def compute_atr_roc(df: pd.DataFrame, lookback: int = 8) -> pd.DataFrame:
    """Rate of change of ATR over N bars. Positive = expanding vol, negative = contracting."""
    if "atr_14" not in df.columns:
        raise ValueError("Run compute_atr() before compute_atr_roc().")
    df["atr_roc_8"] = df["atr_14"].pct_change(lookback)
    return df


# ── Tier D: Regime features ───────────────────────────────────────────────────

def compute_vix_regime(df: pd.DataFrame, vix_1d: pd.DataFrame) -> pd.DataFrame:
    """
    Maps daily VIX close onto 1H bars.
    regime: 0 = VIX < 15 (calm), 1 = 15-22 (elevated), 2 = >22 (fear)
    Shifted by 1 day to prevent lookahead.
    """
    vix_aligned = align_daily_to_1h(vix_1d["Close"], df.index)
    df["vix_close"] = vix_aligned.values
    df["vix_regime"] = pd.cut(
        df["vix_close"],
        bins=[0, 15, 22, 9999],
        labels=[0, 1, 2]
    ).astype(float)
    df.drop(columns=["vix_close"], inplace=True)
    return df


def compute_dxy_slope(df: pd.DataFrame, dxy_1h: pd.DataFrame, ema_span: int = 20) -> pd.DataFrame:
    """
    DXY 20-bar EMA slope direction: +1 = DXY rising (USD strong), -1 = falling.
    Shifted +1 bar to prevent lookahead.
    """
    dxy_aligned = align_4h_to_1h(dxy_1h["Close"], df.index)
    dxy_ema = dxy_aligned.ewm(span=ema_span, adjust=False).mean()
    df["dxy_slope"] = np.sign(dxy_ema - dxy_ema.shift(5))
    return df

def compute_us10y_features(df: pd.DataFrame, us10y_1d: pd.DataFrame) -> pd.DataFrame:
    """
    US 10-Year yield direction and momentum.
    Aligned daily → 1H via forward-fill with 1-day shift (no lookahead).
    
    us10y_direction: +1 = yields rising (USD bullish, EURUSD bearish)
                     -1 = yields falling (USD bearish, EURUSD bullish)
    us10y_momentum: 5-day rate of change in yield level
    """
    # 5-bar yield change direction
    yield_5d_chg = us10y_1d["Close"].diff(5)
    yield_direction = np.sign(yield_5d_chg)
    
    # Z-score of yield change over 20-day rolling window
    yield_z = (yield_5d_chg - yield_5d_chg.rolling(20).mean()) / \
               yield_5d_chg.rolling(20).std().replace(0, np.nan)

    # Align to 1H with 1-day shift (prevent lookahead)
    df["us10y_direction"] = align_daily_to_1h(yield_direction, df.index).values
    df["us10y_z20"]       = align_daily_to_1h(yield_z, df.index).values
    return df

def compute_rate_differential(df: pd.DataFrame) -> pd.DataFrame:
    """
    ECB minus Fed effective interest rate differential.
    Encodes known central bank rate decisions as a step function.
    
    Positive = ECB rate > Fed rate (EUR bullish bias)
    Negative = Fed rate > ECB rate (USD bullish bias)
    
    Sources: Fed target rate + ECB main refinancing rate
    Update this table after each CB meeting.
    """
    # Historical rate decisions (date = effective from, rates in %)
    # Format: (date, fed_rate, ecb_rate)
    rate_schedule = [
        ("2020-01-01", 1.75, 0.00),
        ("2020-03-04", 1.25, 0.00),
        ("2020-03-16", 0.25, 0.00),
        ("2022-03-17", 0.50, 0.00),
        ("2022-05-05", 1.00, 0.00),
        ("2022-06-16", 1.75, 0.00),
        ("2022-07-27", 2.50, 0.50),
        ("2022-09-08", 2.50, 1.25),
        ("2022-09-22", 3.25, 1.25),
        ("2022-11-03", 4.00, 1.25),
        ("2022-12-15", 4.50, 2.50),
        ("2023-02-02", 4.75, 3.00),
        ("2023-03-23", 5.00, 3.50),
        ("2023-05-04", 5.25, 3.75),
        ("2023-06-15", 5.25, 4.00),
        ("2023-07-27", 5.50, 4.25),
        ("2023-09-14", 5.50, 4.50),
        ("2023-11-02", 5.50, 4.50),
        ("2024-06-06", 5.50, 4.25),  # ECB first cut
        ("2024-09-12", 5.50, 3.65),
        ("2024-09-19", 5.00, 3.65),  # Fed first cut
        ("2024-11-07", 4.75, 3.40),
        ("2024-12-12", 4.75, 3.15),
        ("2024-12-19", 4.50, 3.15),
        ("2025-01-30", 4.50, 3.15),
        ("2025-03-06", 4.50, 2.65),
        ("2025-03-20", 4.50, 2.65),
        ("2025-05-08", 4.50, 2.40),
        ("2026-01-01", 4.50, 2.40),  # placeholder — update as meetings occur
    ]

    schedule_df = pd.DataFrame(rate_schedule, columns=["date", "fed_rate", "ecb_rate"])
    schedule_df["date"] = pd.to_datetime(schedule_df["date"])
    schedule_df = schedule_df.set_index("date")
    schedule_df["rate_diff"] = schedule_df["ecb_rate"] - schedule_df["fed_rate"]

    # Reindex onto the 1H bar dates, then forward-fill
    rate_series = schedule_df["rate_diff"].reindex(
        schedule_df.index.union(df.index)
    ).sort_index().ffill().reindex(df.index)

    df["rate_differential"] = rate_series.values

    # Also add direction of differential change (is spread widening or narrowing?)
    # df["rate_diff_direction"] = np.sign(
    #     df["rate_differential"] - df["rate_differential"].shift(24 * 7)
    # )  # vs 1 week ago
    return df

def compute_session_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Session label based on UTC hour:
      0 = Asian session   (00:00 - 07:59)
      1 = London session  (08:00 - 12:59)
      2 = New York        (13:00 - 20:59)
      3 = London-NY overlap (13:00 - 15:59)  — highest priority
    """
    hour = df.index.hour
    session = pd.Series(0, index=df.index)  # default: Asian
    session[(hour >= 8)  & (hour < 16)] = 1   # London
    session[(hour >= 13) & (hour < 21)] = 2   # New York
    session[(hour >= 13) & (hour < 16)] = 3   # Overlap (overwrites both)
    df["session"] = session.values
    return df


# def compute_day_of_week(df: pd.DataFrame) -> pd.DataFrame:
#     """Day of week: 0=Monday, 4=Friday."""
#     df["day_of_week"] = df.index.dayofweek.astype(float)
#     return df


# ── Tier E: Z-scored return ───────────────────────────────────────────────────

def compute_return_zscore(df: pd.DataFrame, window: int = 24) -> pd.DataFrame:
    """Z-score of 1-bar log return over rolling N-bar window."""
    if "ret_1" not in df.columns:
        raise ValueError("Run compute_log_returns() first.")
    mean = df["ret_1"].rolling(window).mean()
    std  = df["ret_1"].rolling(window).std().replace(0, np.nan)
    df["ret_1_z24"] = (df["ret_1"] - mean) / std
    return df


# ── Multi-timeframe context ───────────────────────────────────────────────────

def compute_4h_features(df_1h: pd.DataFrame, df_4h: pd.DataFrame) -> pd.DataFrame:
    """Adds 4H log return and RSI features aligned to 1H bars."""
    # 4H log return (8 bars = 32 hours)
    ret_4h = np.log(df_4h["Close"] / df_4h["Close"].shift(8))
    df_1h["ret_4h_8"] = align_4h_to_1h(ret_4h, df_1h.index).values

    # 4H RSI deviation
    rsi_4h = ta.momentum.RSIIndicator(df_4h["Close"], window=14).rsi() - 50.0
    df_1h["rsi_dev_4h"] = align_4h_to_1h(rsi_4h, df_1h.index).values
    return df_1h


def compute_daily_features(df_1h: pd.DataFrame, df_1d: pd.DataFrame) -> pd.DataFrame:
    """Adds daily log return over 5 days aligned to 1H bars."""
    ret_1d = np.log(df_1d["Close"] / df_1d["Close"].shift(5))
    df_1h["ret_1d_5"] = align_daily_to_1h(ret_1d, df_1h.index).values
    return df_1h


# ── Master feature builder ────────────────────────────────────────────────────

def build_all_features(
    eurusd_1h: pd.DataFrame,
    eurusd_4h: pd.DataFrame,
    eurusd_1d: pd.DataFrame,
    dxy_1h:    pd.DataFrame,
    vix_1d:    pd.DataFrame,
    us10y_1d:  pd.DataFrame,
) -> pd.DataFrame:
    """
    Runs all feature engineering steps in the correct order.
    All features are shifted +1 bar before being returned.

    Returns
    -------
    pd.DataFrame with raw OHLCV columns + all feature columns
    """
    print("Building features...")
    df = eurusd_1h.copy()

    # Tier A
    df = compute_log_returns(df)
    df = compute_atr(df)
    df = compute_body_ratio(df)
    df = compute_range_position(df)
    print("  [OK] Tier A: price structure features")

    # Tier B
    df = compute_rsi_deviation(df)
    df = compute_adx(df)
    df = compute_atr_roc(df)
    print("  [OK] Tier B: momentum features")

    # Tier D
    df = compute_vix_regime(df, vix_1d)
    df = compute_dxy_slope(df, dxy_1h)
    df = compute_session_label(df)
    # df = compute_day_of_week(df)
    df = compute_us10y_features(df, us10y_1d)
    df = compute_rate_differential(df)    # ← ADD THIS LINE
    print("  [OK] Rate differential (ECB minus Fed) encoded")
    print("  [OK] Tier D: regime features")

    # Tier E
    df = compute_return_zscore(df)
    print("  [OK] Tier E: z-scored return")

    # Multi-timeframe
    df = compute_4h_features(df, eurusd_4h)
    df = compute_daily_features(df, eurusd_1d)
    print("  [OK] Multi-timeframe context (4H + Daily)")

    # ── SHIFT ALL FEATURES +1 BAR (CRITICAL — prevents lookahead) ────────────
    for col in FEATURE_COLS:
        if col in df.columns:
            df[col] = df[col].shift(1)

    df.dropna(subset=FEATURE_COLS, inplace=True)
    print(f"  [OK] All features shifted +1 bar | Final rows: {len(df):,}")
    return df


def validate_features(df: pd.DataFrame) -> None:
    """
    Prints a summary of the feature DataFrame.
    Flags any all-NaN columns or suspicious correlations.
    """
    print("\n" + "=" * 55)
    print("  Feature validation report")
    print("=" * 55)
    print(f"  Shape: {df.shape}")
    print(f"  Date range: {df.index[0].date()} → {df.index[-1].date()}")
    print()

    for col in FEATURE_COLS:
        if col not in df.columns:
            print(f"  [MISSING] {col}")
            continue
        null_pct = df[col].isna().mean() * 100
        status = "[OK]" if null_pct < 5 else "[WARN — high NaN%]"
        print(f"  {status} {col:<20} | null: {null_pct:.1f}% | "
              f"mean: {df[col].mean():.4f} | std: {df[col].std():.4f}")

    print("=" * 55)
