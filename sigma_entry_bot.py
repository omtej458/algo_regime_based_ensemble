# """
# sigma_entry_bot.py
# ==================
# Standalone 5-minute Sigma(EMA 418) execution runner.

# Run this in a second terminal after run_bot.py is writing hourly signals:
#   python sigma_entry_bot.py --mode live
#   python sigma_entry_bot.py --mode paper
# """

# import argparse
# import os
# import sys
# import time
# from datetime import datetime, timedelta
# from pathlib import Path

# import MetaTrader5 as mt5
# import pandas as pd

# sys.path.insert(0, str(Path(__file__).resolve().parent))

# from src.mt5_connector import (
#     close_position_by_ticket,
#     connect,
#     disconnect,
#     fetch_ohlcv,
#     is_connected,
#     reconnect,
#     send_order,
# )


# PROJECT_ROOT = Path(__file__).resolve().parent
# RESULTS_DIR = PROJECT_ROOT / "results"

# CONFIG = {
#     "symbol": "EURUSD",
#     "signal_file": str(RESULTS_DIR / "paper_trade_log.csv"),
#     "trade_file": str(RESULTS_DIR / "sigma_trade_log.csv"),
#     "mode": "paper",
#     "risk_usd": 50.0,
#     "sigma_period": 418,
#     "bars_to_fetch": 650,
#     "first_tp_pips": 32.0,
#     "second_tp_pips": 22.0,
#     "sl_pips": 10.6,
#     "poll_seconds": 1.0,
#     "require_gate_passed": False,
#     "entry_start_utc": 0,
#     "entry_end_utc": 20,
# }

# PIP_SIZE = 0.0001


# def utc_now() -> datetime:
#     return datetime.utcnow().replace(microsecond=0)


# def session_name_at(dt: datetime):
#     hour = dt.hour
#     if 0 <= hour < 9:
#         return "tokyo"
#     if 8 <= hour < 17:
#         return "london"
#     if 13 <= hour < 22:
#         return "new_york"
#     return None


# def in_entry_window(dt: datetime) -> bool:
#     return CONFIG["entry_start_utc"] <= dt.hour < CONFIG["entry_end_utc"]


# def primary_session_at(dt: datetime):
#     hour = dt.hour
#     if 0 <= hour < 8:
#         return "tokyo"
#     if 8 <= hour < 13:
#         return "london"
#     if 13 <= hour < 22:
#         return "new_york"
#     return None


# def third_session_deadline(entry_time: datetime) -> datetime:
#     session = primary_session_at(entry_time) or session_name_at(entry_time)
#     base = entry_time.replace(minute=0, second=0, microsecond=0)
#     if session == "tokyo":
#         return base.replace(hour=22)
#     if session == "london":
#         return (base + timedelta(days=1)).replace(hour=9)
#     if session == "new_york":
#         return (base + timedelta(days=1)).replace(hour=17)
#     return entry_time + timedelta(hours=24)


# def append_csv(path: str, record: dict):
#     os.makedirs(os.path.dirname(path), exist_ok=True)
#     row = pd.DataFrame([record])
#     exists = os.path.exists(path)
#     row.to_csv(path, mode="a" if exists else "w", header=not exists, index=False)


# def update_trade(ticket, updates: dict):
#     path = CONFIG["trade_file"]
#     if not os.path.exists(path):
#         return
#     df = pd.read_csv(path, dtype={"ticket": str})
#     mask = df["ticket"].astype(str).str.split(".").str[0] == str(ticket).split(".")[0]
#     if mask.any():
#         for key, value in updates.items():
#             df.loc[mask, key] = value
#         df.to_csv(path, index=False)


# def read_trade_log() -> pd.DataFrame:
#     path = CONFIG["trade_file"]
#     if not os.path.exists(path):
#         return pd.DataFrame()
#     return pd.read_csv(path, dtype={"ticket": str, "outcome": str})


# def latest_signal():
#     path = CONFIG["signal_file"]
#     if not os.path.exists(path):
#         return None
#     df = pd.read_csv(path)
#     if df.empty or "direction" not in df.columns:
#         return None
#     df = df[df["direction"].isin([1, -1, "1", "-1"])].copy()
#     if CONFIG["require_gate_passed"] and "gate_passed" in df.columns:
#         df = df[df["gate_passed"].astype(str).str.lower().isin(["true", "1"])]
#     if df.empty:
#         return None
#     row = df.iloc[-1].to_dict()
#     row["direction"] = int(row["direction"])
#     return row


# def calculate_sigma():
#     bars = fetch_ohlcv(CONFIG["symbol"], "5M", CONFIG["bars_to_fetch"])
#     if bars.empty or len(bars) < CONFIG["sigma_period"]:
#         return None
#     ema_high = bars["High"].ewm(span=CONFIG["sigma_period"], adjust=False).mean().iloc[-1]
#     ema_low = bars["Low"].ewm(span=CONFIG["sigma_period"], adjust=False).mean().iloc[-1]
#     ema_close = bars["Close"].ewm(span=CONFIG["sigma_period"], adjust=False).mean().iloc[-1]
#     return {
#         "time": bars.index[-1],
#         "high": float(ema_high),
#         "low": float(ema_low),
#         "close": float(ema_close),
#     }


# def tick_price(direction: int):
#     tick = mt5.symbol_info_tick(CONFIG["symbol"])
#     if tick is None:
#         return None
#     return float(tick.ask if direction == 1 else tick.bid)


# def fixed_risk_lots(sl_pips: float) -> float:
#     info = mt5.symbol_info(CONFIG["symbol"])
#     if info is None:
#         return 0.0
#     pip_value_per_lot = info.trade_tick_value * 10
#     if pip_value_per_lot <= 0:
#         return 0.0
#     raw_lots = CONFIG["risk_usd"] / (sl_pips * pip_value_per_lot)
#     lots = round(raw_lots / info.volume_step) * info.volume_step
#     lots = max(info.volume_min, min(lots, info.volume_max))
#     return round(lots, 2)


# def build_order_plan(direction: int, sigma: dict, trade_number: int, price: float):
#     side = "above" if price >= sigma["close"] else "below"
#     tp_pips = CONFIG["first_tp_pips"] if trade_number == 1 else CONFIG["second_tp_pips"]
#     sl_pips = CONFIG["sl_pips"]

#     if side == "above":
#         entry_level = sigma["high"]
#         touched = price <= entry_level
#     else:
#         entry_level = sigma["low"]
#         touched = price >= entry_level

#     if direction == -1 and side == "above":
#         tp_ref = sigma["high"]
#         sl_ref = sigma["high"]
#         tp_price = tp_ref - tp_pips * PIP_SIZE
#         sl_price = sl_ref + sl_pips * PIP_SIZE
#     elif direction == -1 and side == "below":
#         tp_ref = sigma["close"]
#         sl_ref = sigma["close"]
#         tp_price = tp_ref - tp_pips * PIP_SIZE
#         sl_price = sl_ref + sl_pips * PIP_SIZE
#     elif direction == 1 and side == "above":
#         tp_ref = sigma["close"]
#         sl_ref = sigma["close"]
#         tp_price = tp_ref + tp_pips * PIP_SIZE
#         sl_price = sl_ref - sl_pips * PIP_SIZE
#     else:
#         tp_ref = sigma["low"]
#         sl_ref = sigma["low"]
#         tp_price = tp_ref + tp_pips * PIP_SIZE
#         sl_price = sl_ref - sl_pips * PIP_SIZE

#     return {
#         "side": side,
#         "touched": touched,
#         "entry_level": round(entry_level, 5),
#         "tp_ref": round(tp_ref, 5),
#         "sl_ref": round(sl_ref, 5),
#         "tp_price": round(tp_price, 5),
#         "sl_price": round(sl_price, 5),
#         "tp_pips": tp_pips,
#         "sl_pips": sl_pips,
#     }


# def open_position_ticket(ticket: str):
#     positions = mt5.positions_get(symbol=CONFIG["symbol"])
#     if positions is None:
#         return None
#     ticket_str = str(ticket).split(".")[0]
#     for pos in positions:
#         if str(pos.ticket) == ticket_str:
#             return pos
#     return None


# def detect_closed_outcome(ticket: str, entry_time: datetime):
#     start = entry_time - timedelta(days=2)
#     end = utc_now() + timedelta(minutes=5)
#     deals = mt5.history_deals_get(start, end)
#     if not deals:
#         return "CLOSED", 0.0, None

#     ticket_str = str(ticket).split(".")[0]
#     exit_deals = []
#     for deal in deals:
#         if str(getattr(deal, "position_id", "")) == ticket_str:
#             if getattr(deal, "entry", None) == getattr(mt5, "DEAL_ENTRY_OUT", 1):
#                 exit_deals.append(deal)
#     if not exit_deals:
#         return "CLOSED", 0.0, None

#     deal = sorted(exit_deals, key=lambda d: getattr(d, "time", 0))[-1]
#     reason = getattr(deal, "reason", None)
#     profit = float(getattr(deal, "profit", 0.0))
#     price = float(getattr(deal, "price", 0.0))
#     if reason == getattr(mt5, "DEAL_REASON_SL", object()):
#         return "SL", profit, price
#     if reason == getattr(mt5, "DEAL_REASON_TP", object()):
#         return "TP", profit, price
#     return ("SL" if profit < 0 else "TP"), profit, price


# def sync_open_trades():
#     df = read_trade_log()
#     if df.empty:
#         return
#     open_rows = df[df["outcome"].fillna("").isin(["", "OPEN"])]
#     for _, row in open_rows.iterrows():
#         ticket = str(row.get("ticket", ""))
#         if not ticket or ticket == "nan" or ticket.startswith("PAPER"):
#             continue
#         entry_time = datetime.fromisoformat(str(row["entry_datetime"]).split(".")[0])
#         deadline = datetime.fromisoformat(str(row["time_stop_deadline"]).split(".")[0])
#         pos = open_position_ticket(ticket)
#         if pos is not None and utc_now() >= deadline:
#             print(f"  [TIME] Closing ticket {ticket} after 3 sessions.")
#             if CONFIG["mode"] == "live" and close_position_by_ticket(int(float(ticket))):
#                 update_trade(ticket, {
#                     "exit_datetime": utc_now().isoformat(),
#                     "outcome": "TIME",
#                     "notes": "closed_after_3_sessions",
#                 })
#             continue
#         if pos is None:
#             outcome, profit, exit_price = detect_closed_outcome(ticket, entry_time)
#             update_trade(ticket, {
#                 "exit_datetime": utc_now().isoformat(),
#                 "exit_price": exit_price,
#                 "outcome": outcome,
#                 "profit": profit,
#             })
#             print(f"  [CLOSED] Ticket {ticket}: {outcome} | profit {profit:.2f}")


# def daily_trade_plan(signal: dict):
#     today = utc_now().date().isoformat()
#     df = read_trade_log()
#     if df.empty:
#         today_trades = pd.DataFrame()
#     else:
#         today_trades = df[df["trade_date"].astype(str) == today].copy()

#     open_today = today_trades[today_trades["outcome"].fillna("").isin(["", "OPEN"])]
#     if not open_today.empty:
#         return None
#     if len(today_trades) >= 2:
#         return None
#     if len(today_trades) == 0:
#         return 1, int(signal["direction"])

#     first = today_trades.iloc[0]
#     if str(first.get("outcome", "")).upper() == "SL":
#         return 2, -int(first["direction"])
#     return None


# def log_paper_or_live_trade(trade_number, direction, signal, sigma, plan, lots, result):
#     now = utc_now()
#     ticket = result.get("ticket") if result else f"PAPER_SIGMA_{int(time.time())}"
#     entry_price = result.get("price") if result else tick_price(direction)
#     append_csv(CONFIG["trade_file"], {
#         "trade_date": now.date().isoformat(),
#         "trade_number": trade_number,
#         "entry_datetime": now.isoformat(),
#         "time_stop_deadline": third_session_deadline(now).isoformat(),
#         "pair": CONFIG["symbol"],
#         "direction": direction,
#         "direction_text": "LONG" if direction == 1 else "SHORT",
#         "sigma_time": sigma["time"],
#         "sigma_high": round(sigma["high"], 5),
#         "sigma_low": round(sigma["low"], 5),
#         "sigma_close": round(sigma["close"], 5),
#         "price_side": plan["side"],
#         "entry_level": plan["entry_level"],
#         "entry_price": round(float(entry_price), 5) if entry_price else None,
#         "tp_price": plan["tp_price"],
#         "sl_price": plan["sl_price"],
#         "tp_pips": plan["tp_pips"],
#         "sl_pips": plan["sl_pips"],
#         "risk_usd": CONFIG["risk_usd"],
#         "lots": lots,
#         "ticket": ticket,
#         "source_signal_time": signal.get("signal_datetime") or signal.get("entry_datetime"),
#         "source_probability": signal.get("ensemble_prob"),
#         "source_confidence": signal.get("confidence"),
#         "outcome": "OPEN",
#         "exit_datetime": None,
#         "exit_price": None,
#         "profit": None,
#         "notes": f"mode={CONFIG['mode']}; sigma_ema_{CONFIG['sigma_period']}",
#     })


# def maybe_enter_trade():
#     if not in_entry_window(utc_now()):
#         return
#     signal = latest_signal()
#     if signal is None:
#         return
#     plan_key = daily_trade_plan(signal)
#     if plan_key is None:
#         return
#     trade_number, direction = plan_key
#     sigma = calculate_sigma()
#     if sigma is None:
#         print("  [WAIT] Not enough 5M bars for Sigma EMA 418.")
#         return
#     price = tick_price(direction)
#     if price is None:
#         print("  [WAIT] No tick price.")
#         return
#     order_plan = build_order_plan(direction, sigma, trade_number, price)
#     if not order_plan["touched"]:
#         return

#     lots = fixed_risk_lots(CONFIG["sl_pips"])
#     if lots <= 0:
#         print("  [SKIP] Could not calculate lot size.")
#         return

#     print(f"  [ENTRY] Trade {trade_number} "
#           f"{'BUY' if direction == 1 else 'SELL'} touched Sigma {order_plan['side']} "
#           f"| lots {lots} | TP {order_plan['tp_price']} | SL {order_plan['sl_price']}")

#     result = None
#     if CONFIG["mode"] == "live":
#         result = send_order(
#             CONFIG["symbol"],
#             direction,
#             lots,
#             order_plan["tp_price"],
#             order_plan["sl_price"],
#             comment=f"sigma_t{trade_number}",
#         )
#         if not result.get("success"):
#             print(f"  [ERROR] Order failed: {result.get('error', 'unknown')}")
#             return
#     else:
#         result = None

#     log_paper_or_live_trade(trade_number, direction, signal, sigma, order_plan, lots, result)


# def main():
#     parser = argparse.ArgumentParser(description="5M Sigma EMA execution bot")
#     parser.add_argument("--mode", choices=["paper", "live"], default="paper")
#     parser.add_argument("--risk-usd", type=float, default=CONFIG["risk_usd"])
#     args = parser.parse_args()

#     CONFIG["mode"] = args.mode
#     CONFIG["risk_usd"] = args.risk_usd

#     print("=" * 55)
#     print("  SIGMA ENTRY BOT - Starting")
#     print(f"  Mode      : {CONFIG['mode'].upper()}")
#     print(f"  Symbol    : {CONFIG['symbol']}")
#     print(f"  Risk USD  : {CONFIG['risk_usd']:.2f}")
#     print(f"  Timeframe : 5M | Sigma EMA: {CONFIG['sigma_period']}")
#     print(f"  Entries   : {CONFIG['entry_start_utc']:02d}:00-"
#           f"{CONFIG['entry_end_utc']:02d}:00 UTC")
#     print("=" * 55)

#     if CONFIG["mode"] == "live":
#         confirm = input("  Type 'CONFIRM' to allow live Sigma entries: ")
#         if confirm != "CONFIRM":
#             print("  Aborted.")
#             return

#     if not connect():
#         print("[ERROR] Cannot connect to MT5. Ensure terminal is open.")
#         return

#     try:
#         while True:
#             if not is_connected():
#                 print("  [HEALTH] MT5 not responding. Attempting reconnect...")
#                 reconnect(max_attempts=5, wait_seconds=20)
#             sync_open_trades()
#             maybe_enter_trade()
#             time.sleep(CONFIG["poll_seconds"])
#     except KeyboardInterrupt:
#         print("\n  [STOP] Sigma entry bot stopped.")
#     finally:
#         disconnect()


# if __name__ == "__main__":
#     main()

"""
sigma_entry_bot.py
==================
Standalone 5-minute Sigma (EMA 418) execution runner.

Run in a second terminal after run_bot.py is active:
  python sigma_entry_bot.py --mode paper          # log only
  python sigma_entry_bot.py --mode live           # real MT5 orders
  python sigma_entry_bot.py --mode live --risk-usd 100

How it works
------------
1. Polls MT5 5M bars every 1 second, but only processes on new bar close.
2. Reads the most recent gate_passed=True ML direction from paper_trade_log.csv.
3. Computes Sigma: EMA(418) on 5M High, Low, Close.
4. Waits for price to TOUCH the correct Sigma level per trade rules.
5. Trade 1 fires on first touch of the session.
6. Trade 2 fires (opposite direction, 22-pip TP) only if Trade 1 hit SL.
7. 3-session time-stop closes any trade still open after 3 consecutive sessions.
8. Every 5M bar AND every trade event is logged to sigma_log.csv.
"""

import argparse
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import MetaTrader5 as mt5
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.mt5_connector import (
    close_position_by_ticket,
    connect,
    disconnect,
    fetch_ohlcv,
    is_connected,
    reconnect,
    send_order,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR  = PROJECT_ROOT / "results"

# ── Configuration ──────────────────────────────────────────────────────────────
CONFIG = {
    "symbol":           "EURUSD",
    "signal_file":      str(RESULTS_DIR / "paper_trade_log.csv"),
    "trade_file":       str(RESULTS_DIR / "sigma_trade_log.csv"),
    "mode":             "paper",
    "risk_usd":         50.0,        # ← change to 100 manually when needed
    "sigma_period":     418,
    "bars_to_fetch":    650,          # must be > 418 for EMA warmup
    "first_tp_pips":    32.0,
    "second_tp_pips":   22.0,
    "sl_pips":          10.6,
    "poll_seconds":     1.0,          # how often to check for new 5M bar
    "max_signal_age_h": 8,            # max hours to consider an ML signal valid
}

PIP = 0.0001   # 1 pip for EURUSD

# ── Timing helpers ─────────────────────────────────────────────────────────────

def utc_now() -> datetime:
    return datetime.utcnow().replace(microsecond=0)


def session_at(dt: datetime) -> str:
    """Returns the primary trading session name for a UTC datetime."""
    h = dt.hour
    if 0  <= h < 8:  return "tokyo"
    if 8  <= h < 13: return "london"
    if 13 <= h < 21: return "new_york"
    return "off"


def in_trading_session(dt: datetime) -> bool:
    """Returns True if current UTC hour is in Tokyo, London, or New York."""
    return session_at(dt) != "off"


def third_session_deadline(entry_dt: datetime) -> datetime:
    """
    Returns the UTC datetime by which the trade must be closed
    if neither TP nor SL has been hit (3 consecutive sessions).

    Tokyo   entry → close at 21:00 UTC same day   (end of NY)
    London  entry → close at 09:00 UTC next day   (end of next Tokyo)
    NY      entry → close at 17:00 UTC next day   (end of next London)
    """
    sess = session_at(entry_dt)
    d    = entry_dt.date()
    if sess == "tokyo":
        return datetime(d.year, d.month, d.day, 21, 0, 0)
    if sess == "london":
        nd = d + timedelta(days=1)
        return datetime(nd.year, nd.month, nd.day, 9, 0, 0)
    # new_york or off → default to +24h
    nd = d + timedelta(days=1)
    return datetime(nd.year, nd.month, nd.day, 17, 0, 0)


# ── CSV helpers ────────────────────────────────────────────────────────────────

def append_csv(path: str, record: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    row    = pd.DataFrame([record])
    exists = os.path.exists(path)
    row.to_csv(path, mode="a" if exists else "w", header=not exists, index=False)


def update_trade_row(ticket: str, updates: dict):
    path = CONFIG["trade_file"]
    if not os.path.exists(path):
        return
    df = pd.read_csv(path, dtype={"ticket": str, "outcome": str})
    t_str = str(ticket).split(".")[0]
    mask  = df["ticket"].astype(str).str.split(".").str[0] == t_str
    if mask.any():
        for k, v in updates.items():
            df.loc[mask, k] = v
        df.to_csv(path, index=False)


def read_trade_log() -> pd.DataFrame:
    path = CONFIG["trade_file"]
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, dtype={"ticket": str, "outcome": str})


# ── ML signal reader ───────────────────────────────────────────────────────────

def latest_ml_signal() -> dict | None:
    """
    Returns the most recent gate_passed=True ML signal from paper_trade_log.csv,
    or None if none found or the signal is older than max_signal_age_h hours.
    """
    path = CONFIG["signal_file"]
    if not os.path.exists(path):
        return None
    try:
        # Handle completely empty file
        if os.path.getsize(path) == 0:
            return None
        df = pd.read_csv(path, dtype=str)
        if df.empty or "direction" not in df.columns:
            return None
        # Keep only signals with a real direction
        df["_dir"] = pd.to_numeric(df["direction"], errors="coerce")
        df = df[df["_dir"].isin([1, -1])].copy()
        if df.empty:
            return None
        df["_dt"] = pd.to_datetime(df["entry_datetime"], errors="coerce")
        df = df.dropna(subset=["_dt"]).sort_values("_dt")
        latest = df.iloc[-1]
        age_h  = (utc_now() - latest["_dt"]).total_seconds() / 3600
        if age_h > CONFIG["max_signal_age_h"]:
            return None
        row = latest.to_dict()
        row["direction"] = int(latest["_dir"])
        return row
    except Exception:
        return None


# ── Sigma computation (cached per 5M bar) ──────────────────────────────────────

_sigma_cache: dict = {"bar_time": None, "sigma": None}


def get_sigma() -> dict | None:
    """
    Returns Sigma levels (EMA 418 on 5M High/Low/Close) for the latest closed bar.
    Caches the result — only recomputes when a new 5M bar closes.
    """
    global _sigma_cache
    bars = fetch_ohlcv(CONFIG["symbol"], "5M", CONFIG["bars_to_fetch"])
    if bars.empty or len(bars) < CONFIG["sigma_period"]:
        return None

    latest_bar_time = bars.index[-1]
    if latest_bar_time == _sigma_cache["bar_time"]:
        return _sigma_cache["sigma"]     # same bar — return cached value

    n  = CONFIG["sigma_period"]
    sh = float(bars["High"].ewm(span=n, adjust=False).mean().iloc[-1])
    sl = float(bars["Low"].ewm(span=n, adjust=False).mean().iloc[-1])
    sc = float(bars["Close"].ewm(span=n, adjust=False).mean().iloc[-1])
    sigma = {"time": latest_bar_time, "high": sh, "low": sl, "close": sc,
             "bar_close": float(bars["Close"].iloc[-1])}

    _sigma_cache = {"bar_time": latest_bar_time, "sigma": sigma}
    return sigma


def mid_price() -> float | None:
    """Returns mid price (ask+bid)/2 for position relative to Sigma."""
    tick = mt5.symbol_info_tick(CONFIG["symbol"])
    if tick is None:
        return None
    return (tick.ask + tick.bid) / 2


def exec_price(direction: int) -> float | None:
    """Returns the execution price for a new order (ask for buy, bid for sell)."""
    tick = mt5.symbol_info_tick(CONFIG["symbol"])
    if tick is None:
        return None
    return tick.ask if direction == 1 else tick.bid


# ── Trade plan computation ─────────────────────────────────────────────────────

def build_trade_plan(direction: int, sigma: dict, tp_pips: float,
                      mid: float) -> dict:
    """
    Computes entry_level, TP, SL for one trade.

    direction  : 1=BUY, -1=SELL
    sigma      : dict with 'high', 'low', 'close'
    tp_pips    : 32 for trade1, 22 for trade2
    mid        : current mid price (for above/below Sigma determination)

    Returns dict with all levels and the 'touched' flag.

    Touched means:
      above sigma → price must PULL BACK DOWN to sigma_high
                    (i.e. mid <= sigma_high)
      below sigma → price must PUSH UP to sigma_low
                    (i.e. mid >= sigma_low)
    """
    above_sigma  = mid > sigma["close"]
    sh, sl, sc   = sigma["high"], sigma["low"], sigma["close"]
    sl_pips      = CONFIG["sl_pips"]

    if direction == -1:   # SELL
        if above_sigma:
            entry   = sh
            tp      = sh - tp_pips * PIP
            stop    = sh + sl_pips * PIP
            touched = mid <= sh           # price dropped to or below sigma_high
        else:
            entry   = sl
            tp      = sc - tp_pips * PIP
            stop    = sc + sl_pips * PIP
            touched = mid <= sl           # price dropped to or below sigma_low
    else:                 # BUY
        if above_sigma:
            entry   = sh
            tp      = sc + tp_pips * PIP
            stop    = sc - sl_pips * PIP
            touched = mid <= sh           # price pulled back to sigma_high
        else:
            entry   = sl
            tp      = sl + tp_pips * PIP
            stop    = sl - sl_pips * PIP
            touched = mid >= sl           # price pushed up to sigma_low

    return {
        "above_sigma":  above_sigma,
        "entry_level":  round(entry, 5),
        "tp":           round(tp,    5),
        "sl":           round(stop,  5),
        "tp_pips":      tp_pips,
        "sl_pips":      sl_pips,
        "touched":      touched,
    }


# ── Lot sizing ─────────────────────────────────────────────────────────────────

def fixed_risk_lots(sl_pips: float) -> float:
    """Returns lot size that risks exactly CONFIG['risk_usd'] on CONFIG['sl_pips']."""
    info = mt5.symbol_info(CONFIG["symbol"])
    if info is None:
        return 0.0
    # For EURUSD: 1 standard lot = 100,000 units, pip = 0.0001
    # pip_value per lot ≈ contract_size × pip × (1 for USD-quoted pairs)
    pip_value_per_lot = info.trade_contract_size * PIP   # e.g. 100000 × 0.0001 = $10
    if pip_value_per_lot <= 0:
        return 0.0
    raw  = CONFIG["risk_usd"] / (sl_pips * pip_value_per_lot)
    lots = round(raw / info.volume_step) * info.volume_step
    lots = max(info.volume_min, min(lots, info.volume_max))
    return round(lots, 2)


# ── Order execution ────────────────────────────────────────────────────────────

def place_order(direction: int, lots: float, plan: dict,
                comment: str = "sigma") -> dict:
    """Places a market order and returns result dict."""
    return send_order(
        CONFIG["symbol"],
        direction,
        lots,
        plan["tp"],
        plan["sl"],
        comment=comment,
    )


# ── Outcome detection ──────────────────────────────────────────────────────────

def is_ticket_open(ticket: str) -> bool:
    positions = mt5.positions_get(symbol=CONFIG["symbol"]) or []
    t = str(ticket).split(".")[0]
    return any(str(p.ticket) == t for p in positions)


def detect_outcome_from_history(ticket: str,
                                 entry_dt: datetime) -> tuple[str, float, float | None]:
    """
    Looks up the deal history for a closed position.
    Returns (outcome, profit, exit_price).
    outcome is 'TP', 'SL', or 'CLOSED'.
    """
    start  = entry_dt - timedelta(minutes=5)
    end    = utc_now() + timedelta(minutes=5)
    deals  = mt5.history_deals_get(start, end) or []
    t_str  = str(ticket).split(".")[0]
    exits  = [d for d in deals
              if str(getattr(d, "position_id", "")) == t_str
              and getattr(d, "entry", None) == mt5.DEAL_ENTRY_OUT]

    if not exits:
        return "CLOSED", 0.0, None

    deal   = sorted(exits, key=lambda d: getattr(d, "time", 0))[-1]
    profit = float(getattr(deal, "profit", 0.0))
    price  = float(getattr(deal, "price",  0.0))
    reason = getattr(deal, "reason", -1)

    if reason == mt5.DEAL_REASON_SL:
        return "SL", profit, price
    if reason == mt5.DEAL_REASON_TP:
        return "TP", profit, price
    # Fallback: use profit sign
    return ("TP" if profit > 0 else "SL"), profit, price


# ── Daily trade state ──────────────────────────────────────────────────────────

def get_today_trades() -> pd.DataFrame:
    """Returns today's UTC trades from sigma_trade_log.csv."""
    df = read_trade_log()
    if df.empty:
        return pd.DataFrame()
    today = utc_now().date().isoformat()
    return df[df["trade_date"].astype(str) == today].copy()


def trade_slot() -> tuple[int, int] | None:
    """
    Determines which trade slot to attempt.
    Returns (trade_number, direction) or None if no trade should be taken.

    Rules:
      - Max 2 trades per day
      - Trade 2 only if Trade 1 hit SL
      - No new trade if one is currently OPEN
    """
    today_df = get_today_trades()

    if today_df.empty:
        return None   # will be filled after ML signal read

    open_trades = today_df[today_df["outcome"].fillna("").isin(["", "OPEN"])]
    if not open_trades.empty:
        return None   # trade already open

    if len(today_df) >= 2:
        return None   # day limit reached

    if len(today_df) == 1:
        t1 = today_df.iloc[0]
        if str(t1.get("outcome", "")).upper() == "SL":
            return 2, -int(float(t1["direction"]))
        return None   # trade 1 still pending or TP'd — no trade 2

    return None   # 0 trades today — handled in main loop


# ── Sync closed positions ──────────────────────────────────────────────────────

def sync_closed_positions():
    """
    Checks if any open-logged trades were closed by MT5 (TP/SL hit).
    Updates sigma_trade_log.csv with outcome.
    Also fires time-stop if deadline passed.
    """
    df = read_trade_log()
    if df.empty:
        return

    open_rows = df[df["outcome"].fillna("").isin(["", "OPEN"])].copy()
    for _, row in open_rows.iterrows():
        ticket = str(row.get("ticket", ""))
        if not ticket or ticket in ("nan", "None") or ticket.startswith("PAPER"):
            continue

        entry_str = str(row.get("entry_datetime", ""))
        try:
            entry_dt = datetime.fromisoformat(entry_str.split(".")[0])
        except Exception:
            continue

        deadline_str = str(row.get("time_stop_deadline", ""))
        try:
            deadline = datetime.fromisoformat(deadline_str.split(".")[0])
        except Exception:
            deadline = entry_dt + timedelta(hours=24)

        now = utc_now()

        # Time-stop
        if now >= deadline and is_ticket_open(ticket):
            print(f"  [TIME-STOP] Ticket {ticket} past 3-session deadline — closing.")
            if CONFIG["mode"] == "live":
                close_position_by_ticket(int(float(ticket)))
            update_trade_row(ticket, {
                "exit_datetime": now.isoformat(),
                "outcome":       "TIME",
                "profit":        0.0,
            })
            print(f"  [OK] Ticket {ticket} closed by time-stop.")
            continue

        # Check if MT5 already closed it
        if not is_ticket_open(ticket):
            outcome, profit, exit_price = detect_outcome_from_history(
                ticket, entry_dt
            )
            update_trade_row(ticket, {
                "exit_datetime": now.isoformat(),
                "exit_price":    exit_price,
                "outcome":       outcome,
                "profit":        profit,
            })
            dir_txt = "LONG" if int(float(row.get("direction", 1))) == 1 else "SHORT"
            print(f"  [CLOSED] Ticket {ticket} | {dir_txt} | "
                  f"{outcome} | profit {profit:+.2f}")


# ── Hourly heartbeat log ───────────────────────────────────────────────────────

_last_hourly_log: str = ""


def maybe_hourly_log(sigma: dict | None, ml_dir: int,
                      today_df: pd.DataFrame):
    """
    Logs a heartbeat row to sigma_log.csv once per hour.
    """
    global _last_hourly_log
    now    = utc_now()
    key    = now.strftime("%Y-%m-%d %H")
    if key == _last_hourly_log:
        return
    _last_hourly_log = key

    t1_status = ""
    t2_status = ""
    if not today_df.empty:
        rows = today_df.sort_values("trade_number")
        if len(rows) >= 1:
            t1_status = str(rows.iloc[0].get("outcome", "OPEN"))
        if len(rows) >= 2:
            t2_status = str(rows.iloc[1].get("outcome", "OPEN"))

    append_csv(CONFIG["trade_file"].replace("sigma_trade_log", "sigma_log"), {
        "timestamp":      now.isoformat(),
        "session":        session_at(now),
        "sigma_high":     round(sigma["high"],  5) if sigma else None,
        "sigma_low":      round(sigma["low"],   5) if sigma else None,
        "sigma_close":    round(sigma["close"], 5) if sigma else None,
        "bar_close":      round(sigma["bar_close"], 5) if sigma else None,
        "ml_direction":   "LONG" if ml_dir == 1 else "SHORT" if ml_dir == -1 else "",
        "trade1_status":  t1_status,
        "trade2_status":  t2_status,
        "event":          "hourly_heartbeat",
    })


# ── 5M bar log ────────────────────────────────────────────────────────────────

_last_bar_log: datetime | None = None


def maybe_5m_log(sigma: dict, ml_dir: int, event: str = "bar",
                  detail: str = ""):
    """Logs every new 5M bar to sigma_log.csv."""
    global _last_bar_log
    if sigma["time"] == _last_bar_log:
        return
    _last_bar_log = sigma["time"]

    now = utc_now()
    append_csv(CONFIG["trade_file"].replace("sigma_trade_log", "sigma_log"), {
        "timestamp":    now.isoformat(),
        "session":      session_at(now),
        "sigma_high":   round(sigma["high"],     5),
        "sigma_low":    round(sigma["low"],      5),
        "sigma_close":  round(sigma["close"],    5),
        "bar_close":    round(sigma["bar_close"],5),
        "ml_direction": "LONG" if ml_dir == 1 else "SHORT" if ml_dir == -1 else "",
        "trade1_status": "",
        "trade2_status": "",
        "event":        event,
        "detail":       detail,
    })


# ── Main loop ──────────────────────────────────────────────────────────────────

def run(mode: str):
    CONFIG["mode"] = mode

    print("=" * 60)
    print("  SIGMA ENTRY BOT — Starting")
    print(f"  Mode      : {mode.upper()}")
    print(f"  Symbol    : {CONFIG['symbol']}")
    print(f"  Risk USD  : ${CONFIG['risk_usd']:.2f}")
    print(f"  Sigma EMA : {CONFIG['sigma_period']} on 5M")
    print(f"  T1 TP/SL  : {CONFIG['first_tp_pips']}/{CONFIG['sl_pips']} pips")
    print(f"  T2 TP/SL  : {CONFIG['second_tp_pips']}/{CONFIG['sl_pips']} pips")
    print("=" * 60)

    if not connect():
        print("[FATAL] Cannot connect to MT5.")
        sys.exit(1)

    print("\n  Monitoring 5M bars. Press Ctrl+C to stop.\n")

    try:
        while True:

            # ── Health check ──────────────────────────────────────────────────
            if not is_connected():
                print("  [HEALTH] MT5 disconnected — reconnecting...")
                reconnect(max_attempts=5, wait_seconds=20)

            # ── Sync open trades ──────────────────────────────────────────────
            sync_closed_positions()

            # ── Get Sigma (cached per 5M bar) ─────────────────────────────────
            sigma = get_sigma()

            # ── Get ML direction ──────────────────────────────────────────────
            ml_signal = latest_ml_signal()
            ml_dir    = ml_signal["direction"] if ml_signal else 0

            # ── Today's trade state ───────────────────────────────────────────
            today_df  = get_today_trades()

            # ── Hourly heartbeat log ──────────────────────────────────────────
            maybe_hourly_log(sigma, ml_dir, today_df)

            if sigma is None:
                print(f"  [{utc_now().strftime('%H:%M')}] "
                      f"Waiting for {CONFIG['sigma_period']} 5M bars...")
                time.sleep(CONFIG["poll_seconds"])
                continue

            # ── 5M bar log ────────────────────────────────────────────────────
            maybe_5m_log(sigma, ml_dir)

            # ── Session check ─────────────────────────────────────────────────
            now = utc_now()
            if not in_trading_session(now):
                time.sleep(CONFIG["poll_seconds"])
                continue

            if ml_dir == 0:
                time.sleep(CONFIG["poll_seconds"])
                continue

            # ── Determine trade slot ──────────────────────────────────────────
            open_trades = today_df[
                today_df["outcome"].fillna("").isin(["", "OPEN"])
            ] if not today_df.empty else pd.DataFrame()

            if not open_trades.empty:
                time.sleep(CONFIG["poll_seconds"])
                continue   # a trade is already running

            if len(today_df) >= 2:
                time.sleep(CONFIG["poll_seconds"])
                continue   # day limit reached

            if today_df.empty:
                trade_number = 1
                direction    = ml_dir
            else:
                t1 = today_df.sort_values("trade_number").iloc[0]
                if str(t1.get("outcome", "")).upper() != "SL":
                    time.sleep(CONFIG["poll_seconds"])
                    continue   # trade 1 TP'd or still open
                trade_number = 2
                direction    = -int(float(t1["direction"]))   # opposite

            # ── Check Sigma touch ─────────────────────────────────────────────
            mid = mid_price()
            if mid is None:
                time.sleep(CONFIG["poll_seconds"])
                continue

            tp_pips = (CONFIG["first_tp_pips"] if trade_number == 1
                       else CONFIG["second_tp_pips"])
            plan = build_trade_plan(direction, sigma, tp_pips, mid)

            if not plan["touched"]:
                time.sleep(CONFIG["poll_seconds"])
                continue

            # ── Entry triggered ───────────────────────────────────────────────
            lots = fixed_risk_lots(CONFIG["sl_pips"])
            if lots <= 0:
                print("  [SKIP] Lot size calculation failed.")
                time.sleep(CONFIG["poll_seconds"])
                continue

            dir_txt = "BUY" if direction == 1 else "SELL"
            side_txt = "ABOVE" if plan["above_sigma"] else "BELOW"
            print(f"\n  *** TRADE {trade_number} ENTRY ***")
            print(f"  {dir_txt} | Price {side_txt} Sigma | "
                  f"Sigma touched at {plan['entry_level']:.5f}")
            print(f"  TP: {plan['tp']:.5f} ({tp_pips} pips) | "
                  f"SL: {plan['sl']:.5f} ({CONFIG['sl_pips']} pips) | "
                  f"Lots: {lots} | Risk: ${CONFIG['risk_usd']:.0f}")

            ticket = None
            entry_price = None

            if mode == "live":
                result = place_order(
                    direction, lots, plan,
                    comment=f"sigma_t{trade_number}"
                )
                if result.get("success"):
                    ticket      = str(result["ticket"])
                    entry_price = result["price"]
                    print(f"  [OK] Order filled | Ticket: {ticket} | "
                          f"Price: {entry_price:.5f}")
                else:
                    print(f"  [ERROR] Order failed: {result.get('error', 'unknown')}")
                    time.sleep(5)
                    continue
            else:
                ticket      = f"PAPER_T{trade_number}_{int(time.time())}"
                entry_price = exec_price(direction) or plan["entry_level"]
                print(f"  [PAPER] Logged as {ticket} | Price: {entry_price:.5f}")

            # ── Log trade ─────────────────────────────────────────────────────
            entry_dt  = utc_now()
            deadline  = third_session_deadline(entry_dt)
            append_csv(CONFIG["trade_file"], {
                "trade_date":           entry_dt.date().isoformat(),
                "trade_number":         trade_number,
                "entry_datetime":       entry_dt.isoformat(),
                "time_stop_deadline":   deadline.isoformat(),
                "pair":                 CONFIG["symbol"],
                "direction":            direction,
                "direction_text":       dir_txt,
                "above_sigma":          plan["above_sigma"],
                "sigma_time":           str(sigma["time"]),
                "sigma_high":           round(sigma["high"],  5),
                "sigma_low":            round(sigma["low"],   5),
                "sigma_close":          round(sigma["close"], 5),
                "entry_level":          plan["entry_level"],
                "entry_price":          round(float(entry_price), 5),
                "tp_price":             plan["tp"],
                "sl_price":             plan["sl"],
                "tp_pips":              tp_pips,
                "sl_pips":              CONFIG["sl_pips"],
                "risk_usd":             CONFIG["risk_usd"],
                "lots":                 lots,
                "ticket":               ticket,
                "ml_signal_time":       ml_signal.get("entry_datetime", ""),
                "ml_probability":       ml_signal.get("ensemble_prob", ""),
                "ml_confidence":        ml_signal.get("confidence", ""),
                "outcome":              "OPEN",
                "exit_datetime":        None,
                "exit_price":           None,
                "profit":               None,
                "notes":                f"mode={mode}",
            })

            # Log the entry event to sigma_log too
            maybe_5m_log(sigma, ml_dir,
                          event=f"trade{trade_number}_entry",
                          detail=(f"{dir_txt} {side_txt}_sigma "
                                  f"tp={plan['tp']:.5f} sl={plan['sl']:.5f} "
                                  f"ticket={ticket}"))

            time.sleep(CONFIG["poll_seconds"])

    except KeyboardInterrupt:
        print("\n\n  [STOP] Sigma Entry Bot stopped. Open positions unaffected.")
    finally:
        disconnect()


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sigma EMA-418 5M execution bot")
    parser.add_argument("--mode", choices=["paper", "live"], default="paper",
                        help="paper = log only | live = real MT5 orders")
    parser.add_argument("--risk-usd", type=float, default=CONFIG["risk_usd"],
                        help="Dollar risk per trade (default 50, change to 100 when ready)")
    parser.add_argument("--direction", choices=["long", "short"], default=None,
                        help="Manual ML direction override (long or short). "
                             "Use when run_bot.py hasn't fired yet but you know the bias.")
    args = parser.parse_args()

    if args.mode == "live":
        print("\n  WARNING: LIVE MODE — Real trades will be executed on your MT5 account.")
        confirm = input("  Type CONFIRM to proceed: ")
        if confirm.strip() != "CONFIRM":
            print("  Aborted.")
            sys.exit(0)

    CONFIG["risk_usd"]          = args.risk_usd
    CONFIG["manual_direction"]  = (1 if args.direction == "long"
                                   else -1 if args.direction == "short"
                                   else 0)
    run(args.mode)