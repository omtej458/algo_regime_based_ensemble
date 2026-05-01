"""
data_loader.py
==============
Loads and validates all raw CSV files.
Handles alignment of different timeframes onto the 1H base frame.

CSV files expected in data/:
  eurusd_1h.csv, eurusd_4h.csv, eurusd_1d.csv
  dxy_1h.csv, vix_1d.csv, us10y_1d.csv
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

# ── Project root path (one level above src/) ──────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
FEATURES_DIR = PROJECT_ROOT / "features"
MODELS_DIR   = PROJECT_ROOT / "models"
RESULTS_DIR  = PROJECT_ROOT / "results"


# ── Required files check ───────────────────────────────────────────────────────
REQUIRED_FILES = [
    "eurusd_1h.csv",
    "eurusd_4h.csv",
    "eurusd_1d.csv",
    "dxy_1h.csv",
    "vix_1d.csv",
    "us10y_1d.csv",
]

OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]


def check_data_files() -> dict:
    """
    Checks which required CSV files are present in the data/ directory.
    Returns a dict: {filename: True/False}
    """
    status = {}
    for fname in REQUIRED_FILES:
        fpath = DATA_DIR / fname
        status[fname] = fpath.exists()
    return status


def load_ohlcv(filename: str, name: str = "") -> pd.DataFrame:
    """
    Loads an OHLCV CSV file from the data/ directory.
    Parses the datetime index, renames columns to standard OHLCV format,
    drops rows with NaN in Close, and sorts ascending by time.

    Parameters
    ----------
    filename : str
        CSV filename (e.g. 'eurusd_1h.csv')
    name : str
        Human-readable name for logging (e.g. 'EURUSD 1H')

    Returns
    -------
    pd.DataFrame with columns: Open, High, Low, Close, Volume
    """
    fpath = DATA_DIR / filename
    if not fpath.exists():
        raise FileNotFoundError(
            f"\n[ERROR] File not found: {fpath}"
            f"\nPlease place {filename} inside the data/ folder."
        )

    # yfinance CSVs have 2 extra metadata rows before the real header:
    #   row 0: "Price", row 1: "Ticker"  — skip both.
    # Try reading with skiprows first; fall back to plain read if that fails.
    try:
        df = pd.read_csv(fpath, index_col=0, skiprows=[1, 2])
        # If the index still contains non-date strings (e.g. 'Ticker', 'Datetime')
        # it means the file does NOT have extra header rows — re-read normally.
        if df.index[0] in ("Ticker", "Datetime", "Price"):
            df = pd.read_csv(fpath, index_col=0)
    except Exception:
        df = pd.read_csv(fpath, index_col=0)

    # Drop any residual header-like rows that leaked into the data
    df = df[~df.index.astype(str).isin(["Ticker", "Datetime", "Price", ""])]

    # Standardise column names — handle yfinance multi-level headers
    df.columns = [str(c).strip().split(" ")[0] for c in df.columns]

    # Map common alternative names to standard OHLCV
    rename_map = {
        "open":   "Open",
        "high":   "High",
        "low":    "Low",
        "close":  "Close",
        "adj":    "Close",   # adj close fallback
        "volume": "Volume",
    }
    df.columns = [rename_map.get(c.lower(), c) for c in df.columns]

    # Keep only OHLCV columns that exist
    available = [c for c in OHLCV_COLS if c in df.columns]
    df = df[available].copy()

    # Volume is optional — fill with 0 if missing
    if "Volume" not in df.columns:
        df["Volume"] = 0.0

    df = df.dropna(subset=["Close"])
    df = df.sort_index()
    # Parse index safely with mixed format to handle timezone-aware strings
    df.index = pd.to_datetime(df.index, format="mixed", utc=True)
    # yfinance returns timezone-aware index (+00:00) — convert then strip
    if df.index.tz is not None:
        df.index = df.index.tz_convert('UTC').tz_localize(None)
    label = name or filename
    print(f"  [OK] {label}: {len(df):,} rows | {df.index[0].date()} → {df.index[-1].date()}")
    return df


def load_all_data() -> dict:
    """
    Loads all 6 required CSV files and returns them as a dictionary.

    Returns
    -------
    dict with keys:
        eurusd_1h, eurusd_4h, eurusd_1d, dxy_1h, vix_1d, us10y_1d
    """
    print("=" * 55)
    print("  Loading raw data files")
    print("=" * 55)

    data = {
        "eurusd_1h": load_ohlcv("eurusd_1h.csv", "EURUSD 1H"),
        "eurusd_4h": load_ohlcv("eurusd_4h.csv", "EURUSD 4H"),
        "eurusd_1d": load_ohlcv("eurusd_1d.csv", "EURUSD Daily"),
        "dxy_1h":    load_ohlcv("dxy_1h.csv",    "DXY 1H"),
        "vix_1d":    load_ohlcv("vix_1d.csv",    "VIX Daily"),
        "us10y_1d":  load_ohlcv("us10y_1d.csv",  "US10Y Daily"),
    }

    print("-" * 55)
    print(f"  All files loaded successfully.")
    print("=" * 55)
    return data


def align_daily_to_1h(daily_series: pd.Series, base_1h_index: pd.DatetimeIndex) -> pd.Series:
    """
    Forward-fills a daily series onto a 1H datetime index.
    This ensures that daily macro data (VIX, US10Y) is available on every 1H bar
    using only the PREVIOUS day's close — no lookahead.

    Parameters
    ----------
    daily_series : pd.Series with DatetimeIndex at daily frequency
    base_1h_index : pd.DatetimeIndex of the 1H EURUSD bars

    Returns
    -------
    pd.Series aligned to base_1h_index, forward-filled
    """
    # Shift daily values by 1 day to avoid lookahead (today's VIX uses yesterday's close)
    daily_shifted = daily_series.shift(1)

    # Reindex onto 1H base, then forward fill within each day
    aligned = daily_shifted.reindex(
        daily_shifted.index.union(base_1h_index)
    ).sort_index().ffill().reindex(base_1h_index)

    return aligned


def align_4h_to_1h(series_4h: pd.Series, base_1h_index: pd.DatetimeIndex) -> pd.Series:
    """
    Forward-fills a 4H series onto the 1H index.
    A 4H bar's value is held constant for the next 4 1H bars.
    Shifted by 1 period to prevent lookahead.
    """
    series_4h_shifted = series_4h.shift(1)
    aligned = series_4h_shifted.reindex(
        series_4h_shifted.index.union(base_1h_index)
    ).sort_index().ffill().reindex(base_1h_index)
    return aligned


def validate_temporal_integrity(df: pd.DataFrame, feature_cols: list) -> bool:
    """
    Asserts that no feature has correlation > 0.99 with the *next* bar's return.
    A perfect correlation with future returns indicates lookahead bias.
    Prints a warning for any suspicious feature.

    Returns True if all clear, False if any lookahead detected.
    """
    if "Close" not in df.columns:
        print("  [SKIP] Close column not found, skipping temporal check.")
        return True

    future_ret = df["Close"].pct_change().shift(-1)  # next bar return
    issues = []

    for col in feature_cols:
        if col not in df.columns:
            continue
        try:
            corr = df[col].corr(future_ret)
            if abs(corr) > 0.95:
                issues.append((col, corr))
        except Exception:
            pass

    if issues:
        print("\n  [WARNING] Possible lookahead bias detected:")
        for col, corr in issues:
            print(f"    {col}: corr with future return = {corr:.4f}")
        return False

    print("  [OK] Temporal integrity check passed — no lookahead detected.")
    return True
