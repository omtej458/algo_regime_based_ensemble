"""
live_engine.py
==============
Main live trading engine.

Runs every 1H bar close:
  1. Fetch latest data from MT5
  2. Build features
  3. Run ensemble prediction
  4. Apply regime gate
  5. Execute signal if gate passes
  6. Check time-stops on open positions
  7. Log everything
"""

import pandas as pd
import numpy as np
import joblib
import time
import os
from datetime import datetime, timedelta
from pathlib import Path

# Project imports
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mt5_connector import (
    connect, disconnect, fetch_ohlcv, fetch_yfinance, fetch_current_spread,
    get_account_balance, calculate_lot_size, send_order,
    close_position_by_ticket, get_open_positions,
    reconnect, is_connected
)
from src.features import build_all_features, FEATURE_COLS
from src.regime import (build_regime_features, predict_regimes,
                         label_regimes, add_regime_to_df)
from src.signal_gate import apply_signal_gate, REGIME_THRESHOLDS
from src.models import build_lgbm_baseline, build_xgb_baseline, build_catboost_baseline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR   = PROJECT_ROOT / "models"
RESULTS_DIR  = PROJECT_ROOT / "results"

# ── Configuration ─────────────────────────────────────────────────────────────

CONFIG = {
    "symbol":         "EURUSD",
    "dxy_symbol":     "USDX",       # check your broker's DXY symbol name
    "risk_pct":       0.005,         # 0.5% account risk per trade
    "atr_tp_mult":    1.5,           # TP = entry ± ATR × 1.5
    "atr_sl_mult":    1.0,           # SL = entry ∓ ATR × 1.0
    "time_stop_bars": 12,            # close if open for 12H
    "max_spread_pips": 2.0,          # skip if spread too wide
    "max_open_trades": 1,            # only 1 trade at a time
    "bars_to_fetch":  3200,          # enough for all features + warmup
    "log_file":       str(RESULTS_DIR / "paper_trade_log.csv"),
    "mode":           "paper",       # "paper" or "live"
                                     # paper mode: signals logged but NOT sent to MT5
}


# ── Ensemble classifier (same as notebook 07) ─────────────────────────────────

from sklearn.base import BaseEstimator, ClassifierMixin

class EnsembleClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, lgbm, xgb, cat):
        self.lgbm = lgbm
        self.xgb  = xgb
        self.cat  = cat
    def predict_proba(self, X):
        lp  = self.lgbm.predict_proba(X)[:, 1]
        xp  = self.xgb.predict_proba(X)[:, 1]
        cp  = self.cat.predict_proba(X)[:, 1]
        avg = 0.45 * lp + 0.30 * xp + 0.25 * cp
        return np.column_stack([1 - avg, avg])
    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def load_models() -> EnsembleClassifier:
    """Loads the saved ensemble models from models/ directory."""
    lgbm = joblib.load(MODELS_DIR / "lgbm_v2.pkl")
    xgb  = joblib.load(MODELS_DIR / "xgb_v1.pkl")
    cat  = joblib.load(MODELS_DIR / "catboost_v1.pkl")
    hmm  = joblib.load(MODELS_DIR / "hmm_regime.pkl")
    rmap = joblib.load(MODELS_DIR / "regime_map.pkl")
    print("[OK] All models loaded.")
    return EnsembleClassifier(lgbm, xgb, cat), hmm, rmap


# ── Trade logging ─────────────────────────────────────────────────────────────

def log_signal(record: dict):
    """Appends one signal record to the trade log CSV."""
    log_path = CONFIG["log_file"]
    df_row = pd.DataFrame([record])
    if os.path.exists(log_path):
        df_row.to_csv(log_path, mode='a', header=False, index=False)
    else:
        df_row.to_csv(log_path, mode='w', header=True, index=False)


def log_trade_close(ticket, exit_price: float,
                    exit_time: str, outcome: str, pnl_pips: float):
    """Updates an existing log entry when trade closes."""
    log_path = CONFIG["log_file"]
    if not os.path.exists(log_path):
        return
    df = pd.read_csv(log_path, dtype={'ticket': str})  # read as str to avoid float mismatch
    ticket_str = str(ticket).split('.')[0]              # '330667600.0' → '330667600'
    mask = df['ticket'].astype(str).str.split('.').str[0] == ticket_str
    if mask.any():
        df.loc[mask, 'exit_datetime'] = exit_time
        df.loc[mask, 'exit_price']    = float(exit_price) if exit_price else None
        df.loc[mask, 'outcome']       = outcome
        df.loc[mask, 'pnl_pips']      = float(pnl_pips) if pnl_pips else None
        df.to_csv(log_path, index=False)


# ── Time-stop manager ─────────────────────────────────────────────────────────

def check_time_stops():
    """
    Closes any position that has been open for >= time_stop_bars hours.
    Only acts on REAL tickets (integer ticket numbers).
    Skips paper-mode PAPER_XXXXXXX tickets and already-closed trades.
    Cross-checks against MT5 actual open positions before attempting close.
    """
    log_path = CONFIG["log_file"]
    if not os.path.exists(log_path):
        return

    df = pd.read_csv(log_path)
    # Only rows where outcome is blank/NaN (still open)
    open_trades = df[
        (df['outcome'].isna() | (df['outcome'] == '')) &
        (df['gate_passed'] == True)
    ].copy()

    if len(open_trades) == 0:
        return

    now = datetime.utcnow()

    # Get actual open positions from MT5 to cross-check
    import MetaTrader5 as mt5
    mt5_positions = mt5.positions_get()
    live_tickets  = set()
    if mt5_positions:
        live_tickets = {p.ticket for p in mt5_positions}

    closed_count = 0
    for _, row in open_trades.iterrows():
        ticket_raw = row.get('ticket')

        # Skip paper mode tickets
        if pd.isna(ticket_raw) or str(ticket_raw).startswith('PAPER'):
            # Mark paper trades as TIME in log so they stop appearing
            if not pd.isna(ticket_raw):
                log_trade_close(str(ticket_raw), 0.0,
                                now.isoformat(), 'TIME_PAPER', 0.0)
            continue

        try:
            ticket    = int(float(ticket_raw))
            entry_str = str(row.get('entry_datetime', ''))
            if not entry_str or entry_str == 'nan':
                continue

            entry_time = datetime.fromisoformat(entry_str.split('.')[0])
            hours_open = (now - entry_time).total_seconds() / 3600

            if hours_open < CONFIG["time_stop_bars"]:
                continue   # not yet expired

            # Only try to close if MT5 actually has this position open
            if ticket not in live_tickets:
                # Position already closed by TP/SL — just update log
                log_trade_close(ticket, 0.0, now.isoformat(),
                                'CLOSED_BY_MT5', 0.0)
                continue

            print(f"  [TIME-STOP] Ticket {ticket} open {hours_open:.1f}H — closing")
            if CONFIG["mode"] == "live":
                success = close_position_by_ticket(ticket)
                if success:
                    log_trade_close(ticket, 0.0, now.isoformat(), 'TIME', 0.0)
                    print(f"  [OK] Ticket {ticket} closed")
                    closed_count += 1
                else:
                    print(f"  [WARN] Could not close ticket {ticket}")
        except Exception as e:
            print(f"  [WARN] Time-stop error on ticket {ticket_raw}: {e}")

    if closed_count > 0:
        print(f"  [TIME-STOP] Closed {closed_count} expired position(s).")


# ── Main signal generation ────────────────────────────────────────────────────

def run_one_bar(ensemble: EnsembleClassifier, hmm_model, regime_map: dict):
    """
    Runs the full pipeline for the current 1H bar close.
    Returns signal dict or None if no signal.
    """
    symbol = CONFIG["symbol"]
    print(f"\n{'='*55}")
    print(f"  Bar close: {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC")
    print(f"{'='*55}")

    # ── Step 1: Fetch data ────────────────────────────────────────────────────
    print("  Fetching live data from MT5...")

    eurusd_1h = fetch_ohlcv(symbol,              '1H', CONFIG["bars_to_fetch"])
    eurusd_4h = fetch_ohlcv(symbol,              '4H', CONFIG["bars_to_fetch"] // 4)
    eurusd_1d = fetch_ohlcv(symbol,              '1D', CONFIG["bars_to_fetch"] // 24)
    dxy_1h    = fetch_ohlcv(CONFIG["dxy_symbol"],'1H', CONFIG["bars_to_fetch"])
    # vix_1d    = fetch_ohlcv("VIX.IDX",           '1D', 500)  # check broker name
    # us10y_1d  = fetch_ohlcv("US10Y.IDX",         '1D', 500)  # check broker name

    if eurusd_1h.empty:
        print("  [ERROR] Failed to fetch EURUSD 1H data. Skipping bar.")
        return None
    
       # yfinance — macro instruments not available on most MT5 brokers
    vix_1d   = fetch_yfinance("^VIX", period="2y", interval="1d")
    us10y_1d = fetch_yfinance("^TNX", period="2y", interval="1d")

    # Fallback to CSV if yfinance fails
    if vix_1d.empty:
        print("  [WARN] yfinance VIX failed — loading from CSV fallback")
        vix_1d = pd.read_csv(
            PROJECT_ROOT / "data" / "vix_1d.csv",
            index_col=0, parse_dates=True
        )
        if vix_1d.index.tz is not None:
            vix_1d.index = vix_1d.index.tz_localize(None)
        print(f"  [OK] VIX loaded from CSV: {len(vix_1d)} rows")

    if us10y_1d.empty:
        print("  [WARN] yfinance US10Y failed — loading from CSV fallback")
        us10y_1d = pd.read_csv(
            PROJECT_ROOT / "data" / "us10y_1d.csv",
            index_col=0, parse_dates=True
        )
        if us10y_1d.index.tz is not None:
            us10y_1d.index = us10y_1d.index.tz_localize(None)
        print(f"  [OK] US10Y loaded from CSV: {len(us10y_1d)} rows")

    if vix_1d.empty or us10y_1d.empty:
        print("  [ERROR] Both yfinance and CSV fallback failed. Skipping bar.")
        return None

    print(f"  [OK] All data fetched. Latest EURUSD bar: {eurusd_1h.index[-1]}")

    # ── Step 2: Check spread ──────────────────────────────────────────────────
    spread = fetch_current_spread(symbol)
    print(f"  Current spread: {spread} pips")
    if spread > CONFIG["max_spread_pips"]:
        print(f"  [SKIP] Spread {spread} > {CONFIG['max_spread_pips']} pips limit")
        return None

    # ── Step 3: Build features ────────────────────────────────────────────────
    print("  Building features...")
    try:
        df_features = build_all_features(
            eurusd_1h=eurusd_1h,
            eurusd_4h=eurusd_4h,
            eurusd_1d=eurusd_1d,
            dxy_1h=dxy_1h,
            vix_1d=vix_1d,
            us10y_1d=us10y_1d,
        )
    except Exception as e:
        print(f"  [ERROR] Feature build failed: {e}")
        return None

    if len(df_features) < 50:
        print("  [SKIP] Not enough bars after feature computation.")
        return None

    # Use only the LAST bar (most recent closed bar)
    latest = df_features.iloc[[-1]]
    X_live = latest[FEATURE_COLS]

    if X_live.isna().any().any():
        print("  [SKIP] NaN in features for latest bar.")
        return None

    print(f"  [OK] Features built. Latest bar: {latest.index[-1]}")

    # ── Step 4: Regime classification ─────────────────────────────────────────
    print("  Classifying market regime...")
    try:
        X_hmm, feat_idx = build_regime_features(df_features)
        raw_labels = predict_regimes(hmm_model, X_hmm)
        rmap_current = label_regimes(hmm_model, X_hmm, raw_labels)
        df_features['regime_label'] = pd.Series(
            raw_labels, index=feat_idx
        ).reindex(df_features.index).fillna(-1).astype(int)
        df_features['regime_name'] = df_features['regime_label'].map(rmap_current)

        current_regime = df_features['regime_name'].iloc[-1]
        print(f"  [OK] Current regime: {current_regime}")
    except Exception as e:
        print(f"  [WARN] Regime classification failed: {e}. Using 'unknown'.")
        current_regime = 'unknown'
        df_features['regime_name'] = 'unknown'

    # ── Step 5: Ensemble prediction ───────────────────────────────────────────
    print("  Running ensemble prediction...")
    prob = ensemble.predict_proba(X_live)[0, 1]
    confidence = abs(prob - 0.5) * 2
    direction = 1 if prob >= 0.5 else -1
    print(f"  Probability: {prob:.4f} | Confidence: {confidence:.4f} | "
          f"Direction: {'LONG' if direction == 1 else 'SHORT'}")

    # ── Step 6: Apply signal gate ─────────────────────────────────────────────
    latest_with_regime = df_features.iloc[[-1]]
    gate_result = apply_signal_gate(
        latest_with_regime,
        np.array([prob]),
        FEATURE_COLS
    )

    gate_passed = gate_result['gate_passed'].iloc[0]
    gate_reason = gate_result['gate_reason'].iloc[0]

    if not gate_passed:
        print(f"  [GATE BLOCKED] Reason: {gate_reason}")
        log_signal({
            'entry_datetime':  latest.index[-1].isoformat(),
            'pair':            symbol,
            'direction':       0,
            'ensemble_prob':   round(prob, 4),
            'confidence':      round(confidence, 4),
            'regime':          current_regime,
            'gate_passed':     False,
            'gate_reason':     gate_reason,
            'entry_price':     None,
            'tp_price':        None,
            'sl_price':        None,
            'lots':            None,
            'ticket':          None,
            'exit_datetime':   None,
            'exit_price':      None,
            'outcome':         'BLOCKED',
            'pnl_pips':        None,
            'notes':           '',
        })
        return None

    print(f"  [GATE PASSED] Signal confirmed.")

    # ── Step 7: Calculate TP/SL ───────────────────────────────────────────────
    atr    = df_features['atr_14'].iloc[-1]
    price  = eurusd_1h['Close'].iloc[-1]
    tp_distance = atr * CONFIG["atr_tp_mult"]
    sl_distance = atr * CONFIG["atr_sl_mult"]

    if direction == 1:  # Long
        tp_price = price + tp_distance
        sl_price = price - sl_distance
    else:               # Short
        tp_price = price - tp_distance
        sl_price = price + sl_distance

    sl_pips = sl_distance * 10000  # convert to pips for EURUSD

    print(f"  Entry: {price:.5f} | TP: {tp_price:.5f} | "
          f"SL: {sl_price:.5f} | SL pips: {sl_pips:.1f}")

    # ── Step 8: Check open trade limit ───────────────────────────────────────
    open_positions = get_open_positions(symbol)
    if len(open_positions) >= CONFIG["max_open_trades"]:
        print(f"  [SKIP] Already have {len(open_positions)} open position(s). "
              f"Max is {CONFIG['max_open_trades']}.")
        return None

    # ── Step 9: Calculate position size ──────────────────────────────────────
    balance = get_account_balance()
    lots    = calculate_lot_size(symbol, sl_pips, CONFIG["risk_pct"], balance)
    print(f"  Balance: {balance:.2f} | Lots: {lots} | "
          f"Risk: {balance * CONFIG['risk_pct']:.2f}")

    # ── Step 10: Execute or paper log ────────────────────────────────────────
    signal = {
        'entry_datetime':  datetime.utcnow().isoformat(),
        'pair':            symbol,
        'direction':       direction,
        'ensemble_prob':   round(prob, 4),
        'confidence':      round(confidence, 4),
        'regime':          current_regime,
        'gate_passed':     True,
        'gate_reason':     '',
        'entry_price':     price,
        'tp_price':        round(tp_price, 5),
        'sl_price':        round(sl_price, 5),
        'lots':            lots,
        'atr_pips':        round(sl_pips, 1),
        'vix_regime':      df_features['vix_regime'].iloc[-1],
        'adx_14':          round(df_features['adx_14'].iloc[-1], 2),
        'session':         df_features['session'].iloc[-1],
        'ticket':          None,
        'exit_datetime':   None,
        'exit_price':      None,
        'outcome':         '',
        'pnl_pips':        None,
        'notes':           f"mode={CONFIG['mode']}",
    }

    if CONFIG["mode"] == "paper":
        print(f"  [PAPER TRADE] Would go {'LONG' if direction==1 else 'SHORT'} "
              f"{lots} lots at {price:.5f}")
        print(f"  TP: {tp_price:.5f} | SL: {sl_price:.5f}")
        signal['ticket'] = f"PAPER_{int(time.time())}"
        signal['outcome'] = ''
        log_signal(signal)
        print(f"  [LOGGED] Signal recorded to {CONFIG['log_file']}")

    elif CONFIG["mode"] == "live":
        print(f"  [LIVE] Sending order: {'BUY' if direction==1 else 'SELL'} "
              f"{lots} lots {symbol}")
        result = send_order(symbol, direction, lots, tp_price, sl_price)
        if result["success"]:
            signal['ticket'] = result['ticket']
            signal['entry_price'] = result['price']
            log_signal(signal)
            print(f"  [OK] Order filled | Ticket: {result['ticket']} | "
                  f"Price: {result['price']:.5f}")
        else:
            print(f"  [ERROR] Order failed: {result.get('error', 'unknown')}")

    return signal