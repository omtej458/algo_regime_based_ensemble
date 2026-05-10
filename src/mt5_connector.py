"""
mt5_connector.py
================
Handles all communication with MetaTrader 5.
Fetches live data, executes orders, manages open positions.

Requirements:
  - MetaTrader 5 terminal must be open and logged in
  - pip install MetaTrader5
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import time


# ── MT5 connection ────────────────────────────────────────────────────────────

def connect(login: int = None, password: str = None,
            server: str = None) -> bool:
    """
    Initialises MT5 connection.
    If login/password/server are None, uses the currently logged-in terminal.
    """
    if not mt5.initialize():
        print(f"[ERROR] MT5 initialise failed: {mt5.last_error()}")
        return False

    if login and password and server:
        authorised = mt5.login(login, password=password, server=server)
        if not authorised:
            print(f"[ERROR] MT5 login failed: {mt5.last_error()}")
            mt5.shutdown()
            return False

    info = mt5.account_info()
    print(f"[OK] Connected to MT5")
    print(f"     Account : {info.login}")
    print(f"     Broker  : {info.company}")
    print(f"     Balance : {info.balance:.2f} {info.currency}")
    print(f"     Leverage: 1:{info.leverage}")
    return True


def reconnect(max_attempts: int = 5, wait_seconds: int = 30) -> bool:
    """
    Attempts to re-establish MT5 connection after IPC failure.
    Tries up to max_attempts times with wait_seconds between each attempt.
    Called automatically when fetch_ohlcv returns empty data.
    """
    import time
    print(f"[RECONNECT] MT5 connection lost. Attempting to reconnect...")
    for attempt in range(1, max_attempts + 1):
        print(f"[RECONNECT] Attempt {attempt}/{max_attempts}...")
        try:
            mt5.shutdown()          # close any dead connection first
            time.sleep(2)
            if mt5.initialize():    # try fresh initialise
                info = mt5.account_info()
                if info is not None:
                    print(f"[RECONNECT] Success. Account: {info.login} | "
                          f"Balance: {info.balance:.2f} {info.currency}")
                    return True
        except Exception as e:
            print(f"[RECONNECT] Exception: {e}")
        print(f"[RECONNECT] Failed. Waiting {wait_seconds}s before retry...")
        time.sleep(wait_seconds)
    print(f"[RECONNECT] All {max_attempts} attempts failed. "
          f"Ensure MT5 is open and logged in.")
    return False


def is_connected() -> bool:
    """Returns True if MT5 is currently connected and responsive."""
    try:
        info = mt5.account_info()
        return info is not None
    except Exception:
        return False


def disconnect():
    mt5.shutdown()
    print("[OK] MT5 disconnected.")


# ── Live data fetching ────────────────────────────────────────────────────────

TIMEFRAME_MAP = {
    '1H':  mt5.TIMEFRAME_H1,
    '4H':  mt5.TIMEFRAME_H4,
    '1D':  mt5.TIMEFRAME_D1,
}


def fetch_ohlcv(symbol: str, timeframe: str, n_bars: int) -> pd.DataFrame:
    """
    Fetches the last N closed bars for a symbol from MT5.
    Automatically attempts reconnection on IPC failure (-10001).
    """
    tf = TIMEFRAME_MAP.get(timeframe)
    if tf is None:
        raise ValueError(f"Unknown timeframe: {timeframe}. Use 1H, 4H, or 1D.")

    rates = mt5.copy_rates_from_pos(symbol, tf, 0, n_bars)

    # IPC send failed — MT5 was closed and reopened
    if rates is None:
        err = mt5.last_error()
        if err[0] == -10001:
            print(f"  [IPC ERROR] MT5 connection dropped. Triggering reconnect...")
            if reconnect(max_attempts=5, wait_seconds=20):
                rates = mt5.copy_rates_from_pos(symbol, tf, 0, n_bars)
            else:
                print(f"  [ERROR] Reconnect failed. Skipping bar.")
                return pd.DataFrame()

    if rates is None or len(rates) == 0:
        print(f"  [ERROR] No data for {symbol} {timeframe}: {mt5.last_error()}")
        return pd.DataFrame()

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df = df.set_index('time')
    df.index = df.index.tz_localize(None)

    df = df.rename(columns={
        'open':        'Open',
        'high':        'High',
        'low':         'Low',
        'close':       'Close',
        'tick_volume': 'Volume',
    })

    df = df.iloc[:-1]  # drop current incomplete bar
    return df[['Open', 'High', 'Low', 'Close', 'Volume']]


def fetch_current_spread(symbol: str) -> float:
    """Returns current spread in pips for the symbol."""
    tick = mt5.symbol_info_tick(symbol)
    info = mt5.symbol_info(symbol)
    if tick is None or info is None:
        return 999.0
    spread_points = tick.ask - tick.bid
    spread_pips   = spread_points / info.point / 10  # convert to pips
    return round(spread_pips, 1)


def get_account_balance() -> float:
    """Returns current account balance."""
    return mt5.account_info().balance


# ── Position sizing ───────────────────────────────────────────────────────────

def calculate_lot_size(
    symbol:       str,
    sl_pips:      float,
    risk_pct:     float = 0.005,   # 0.5% of account per trade
    balance:      float = None,
) -> float:
    """
    Calculates position size based on fixed fractional risk.

    lot_size = (balance × risk_pct) / (sl_pips × pip_value_per_lot)

    Returns lot size rounded to broker's minimum step.
    """
    if balance is None:
        balance = get_account_balance()

    risk_amount = balance * risk_pct
    sym_info    = mt5.symbol_info(symbol)

    if sym_info is None:
        print(f"[ERROR] Symbol info not found for {symbol}")
        return 0.0

    # pip value per standard lot in account currency
    # For most USD-quoted pairs: pip_value = 10 USD per lot
    # MT5 provides contract_size and trade_tick_value
    pip_value_per_lot = sym_info.trade_tick_value * 10  # approximate for most pairs

    raw_lots = risk_amount / (sl_pips * pip_value_per_lot)

    # Round to broker's lot step
    lot_step = sym_info.volume_step
    lots     = round(raw_lots / lot_step) * lot_step
    lots     = max(sym_info.volume_min, min(lots, sym_info.volume_max))

    return round(lots, 2)


# ── Trade execution ───────────────────────────────────────────────────────────

def send_order(
    symbol:    str,
    direction: int,     # 1 = long, -1 = short
    lots:      float,
    tp_price:  float,
    sl_price:  float,
    comment:   str = "algo_ensemble",
) -> dict:
    """
    Sends a market order to MT5.

    Returns dict with order result details.
    """
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return {"success": False, "error": "No tick data"}

    order_type = mt5.ORDER_TYPE_BUY if direction == 1 else mt5.ORDER_TYPE_SELL
    price      = tick.ask if direction == 1 else tick.bid

    request = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       symbol,
        "volume":       lots,
        "type":         order_type,
        "price":        price,
        "sl":           round(sl_price, 5),
        "tp":           round(tp_price, 5),
        "deviation":    10,   # max slippage in points
        "magic":        20250101,   # unique ID for this bot's trades
        "comment":      comment,
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        return {
            "success":  False,
            "retcode":  result.retcode,
            "error":    result.comment,
        }

    return {
        "success":  True,
        "ticket":   result.order,
        "price":    result.price,
        "lots":     result.volume,
        "symbol":   symbol,
        "direction": direction,
    }


def close_position_by_ticket(ticket: int) -> bool:
    """Closes a specific position by ticket number (for time-stop)."""
    position = None
    for pos in mt5.positions_get():
        if pos.ticket == ticket:
            position = pos
            break

    if position is None:
        return False

    tick      = mt5.symbol_info_tick(position.symbol)
    close_type = mt5.ORDER_TYPE_SELL if position.type == 0 else mt5.ORDER_TYPE_BUY
    price     = tick.bid if position.type == 0 else tick.ask

    request = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       position.symbol,
        "volume":       position.volume,
        "type":         close_type,
        "position":     ticket,
        "price":        price,
        "deviation":    10,
        "magic":        20250101,
        "comment":      "time_stop",
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    return result.retcode == mt5.TRADE_RETCODE_DONE


def get_open_positions(symbol: str = "EURUSD") -> list:
    """Returns list of open positions for this bot (magic number filter)."""
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        return []
    return [p for p in positions if p.magic == 20250101]

def fetch_yfinance(
    ticker:   str,
    period:   str = "2y",
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Fetches OHLCV data from Yahoo Finance via yfinance.
    Used for instruments not available on the MT5 broker:
      - VIX  : ticker = "^VIX"
      - US10Y: ticker = "^TNX"

    Parameters
    ----------
    ticker   : Yahoo Finance ticker string
    period   : lookback period — "1y", "2y", "5y"
    interval : bar size — "1d", "1h", "4h"

    Returns
    -------
    pd.DataFrame with columns: Open, High, Low, Close, Volume
    DatetimeIndex in UTC, timezone stripped (matches MT5 format)
    """
    import yfinance as yf

    try:
        # Prefer Ticker.history() — more stable across yfinance versions.
        # Fall back to yf.download() if history() returns empty data.
        tkr = yf.Ticker(ticker)
        raw = tkr.history(period=period, interval=interval, auto_adjust=True)
        if raw.empty:
            raw = yf.download(
                ticker,
                period=period,
                interval=interval,
                progress=False,
                auto_adjust=True,
            )

        if raw.empty:
            print(f"  [ERROR] yfinance returned no data for {ticker}")
            return pd.DataFrame()

        # yfinance multi-level columns → flatten
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [col[0] for col in raw.columns]

        # Standardise column names
        raw.columns = [str(c).strip() for c in raw.columns]
        rename_map = {
            'open':   'Open',
            'high':   'High',
            'low':    'Low',
            'close':  'Close',
            'volume': 'Volume',
        }
        raw.columns = [rename_map.get(c.lower(), c) for c in raw.columns]

        # Keep only OHLCV
        keep = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume']
                if c in raw.columns]
        raw = raw[keep].copy()

        if 'Volume' not in raw.columns:
            raw['Volume'] = 0.0

        # Strip timezone (matches the format your training data uses)
        if raw.index.tz is not None:
            raw.index = raw.index.tz_convert('UTC').tz_localize(None)

        raw = raw.dropna(subset=['Close']).sort_index()

        print(f"  [OK] yfinance {ticker}: {len(raw)} bars | "
              f"{raw.index[0].date()} → {raw.index[-1].date()}")
        return raw

    except Exception as e:
        print(f"  [ERROR] yfinance fetch failed for {ticker}: {e}")
        return pd.DataFrame()