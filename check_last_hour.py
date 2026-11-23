"""
check_last_hour.py
==================
On-demand version of `run_bot.py --mode live`.

run_bot.py sleeps until 2 minutes after the next 1H bar close before it prints
anything, so if you start it at 10:45 you have to wait until 11:02 to see
numbers. This script does ONE pass immediately, for the hour that has already
closed, and prints exactly the same block run_bot.py would have printed at :02.

Usage:
  python check_last_hour.py                 # read-only: print last closed hour
  python check_last_hour.py --log            # also append the row to paper_trade_log.csv
  python check_last_hour.py --discord        # also push the output to Discord
  python check_last_hour.py --ignore-spread  # don't abort when the live spread is wide
  python check_last_hour.py --check-time-stops   # also run the time-stop pass
  python check_last_hour.py --mode live      # label the record mode=live (no orders are sent)

Nothing here places or closes an order on its own: the signal pipeline in
live_engine is signal-only (execution lives in sigma_entry_bot.py), and the
time-stop pass is skipped unless you ask for it.

Why the numbers match run_bot.py:
  * MT5 data comes from fetch_ohlcv(), which already drops the still-forming
    bar — so "latest bar" is the hour that just closed, whatever time you run.
  * The only other time dependency in run_one_bar() is datetime.utcnow(),
    used for the session-hour gate and the printed timestamps. This script
    pins it to the top of the current UTC hour, which is the same hour
    run_bot.py used when it woke at :02 — so the session gate decides the
    same way, and the printed bar-close timestamp is a round number.

Run it any time after the hour turns. Right at :00 the broker may not have
finalised the bar yet, so give it a minute or two, same as run_bot.py does.
"""

import argparse
import contextlib
import io
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import src.live_engine as live_engine
from src.live_engine import load_models, run_one_bar, check_time_stops, CONFIG
from src.mt5_connector import connect, disconnect, fetch_ohlcv, is_connected
from run_bot import TeeBuffer, send_discord_message


class FrozenClock(datetime):
    """datetime whose utcnow() returns a fixed moment."""

    _frozen = None

    @classmethod
    def freeze(cls, moment: datetime):
        cls._frozen = moment

    @classmethod
    def utcnow(cls):
        return cls._frozen if cls._frozen is not None else datetime.utcnow()


def last_closed_bar_time(symbol: str):
    """Timestamp (index) of the most recent fully closed 1H bar, or None."""
    probe = fetch_ohlcv(symbol, '1H', 5)
    if probe.empty:
        return None
    return probe.index[-1].to_pydatetime()


def main():
    parser = argparse.ArgumentParser(
        description="Print the previous hour's signal data, right now.")
    parser.add_argument('--mode', choices=['paper', 'live'], default='paper',
                        help="label written into the record's notes field "
                             "(no orders are sent either way)")
    parser.add_argument('--log', action='store_true',
                        help="append the signal row to paper_trade_log.csv "
                             "(default: read-only, nothing is written)")
    parser.add_argument('--discord', action='store_true',
                        help="also send the output to the Discord webhook")
    parser.add_argument('--ignore-spread', action='store_true',
                        help="skip the max-spread abort (useful outside "
                             "trading hours, when quotes are wide)")
    parser.add_argument('--check-time-stops', action='store_true',
                        help="also run the time-stop pass on open positions")
    args = parser.parse_args()

    CONFIG["mode"] = args.mode
    if args.ignore_spread:
        CONFIG["max_spread_pips"] = 10_000.0

    print("=" * 55)
    print("  ALGO ENSEMBLE BOT — Last closed hour (on demand)")
    print(f"  Mode    : {CONFIG['mode'].upper()}")
    print(f"  Symbol  : {CONFIG['symbol']}")
    print(f"  Risk    : {CONFIG['risk_pct']*100:.1f}% per trade")
    print(f"  Logging : {'ON (paper_trade_log.csv)' if args.log else 'OFF (read-only)'}")
    print("=" * 55)

    if not connect():
        print("[ERROR] Cannot connect to MT5. Ensure terminal is open.")
        sys.exit(1)

    try:
        if not is_connected():
            print("[ERROR] MT5 is not responding.")
            sys.exit(1)

        bar_time = last_closed_bar_time(CONFIG["symbol"])
        if bar_time is None:
            print("[ERROR] Could not read any closed 1H bar from MT5.")
            sys.exit(1)

        real_now = datetime.utcnow()
        hour_top = real_now.replace(minute=0, second=0, microsecond=0)

        print(f"\n  Clock now     : {real_now.strftime('%Y-%m-%d %H:%M')} UTC")
        print(f"  Latest bar    : {bar_time.strftime('%Y-%m-%d %H:%M')} "
              f"(MT5 server time) — the hour that just closed")
        print(f"  Session hour  : {hour_top.strftime('%H')}:xx UTC "
              f"(the hour run_bot.py would be using right now)\n")

        # Pin live_engine's clock to the top of this UTC hour: same session
        # hour run_bot.py used at :02, minus the ragged minutes.
        FrozenClock.freeze(hour_top)
        live_engine.datetime = FrozenClock

        original_log_signal = live_engine.log_signal
        if not args.log:
            live_engine.log_signal = lambda record: None

        ensemble, hmm_model, regime_map = load_models()

        hourly_output = TeeBuffer(sys.stdout)
        try:
            with contextlib.redirect_stdout(hourly_output):
                if args.check_time_stops:
                    print("\n  Checking time-stops on open positions...")
                    check_time_stops()
                else:
                    print("\n  Skipping time-stop check (pass "
                          "--check-time-stops to run it).")

                signal = run_one_bar(ensemble, hmm_model, regime_map)

                if signal and signal.get('gate_passed'):
                    dir_str = "LONG" if signal['direction'] == 1 else "SHORT"
                    print(f"\n  *** SIGNAL: {dir_str} | "
                          f"Prob: {signal['ensemble_prob']:.3f} | "
                          f"Regime: {signal['regime']} ***")
                else:
                    print(f"\n  No signal this bar.")
        finally:
            live_engine.log_signal = original_log_signal
            live_engine.datetime = datetime

        if args.discord:
            send_discord_message(hourly_output.getvalue())

        if not args.log:
            print("\n  (Read-only run — nothing was written to "
                  "paper_trade_log.csv. Use --log to record it.)")

    except KeyboardInterrupt:
        print("\n\n  [STOP] Stopped by user.")

    finally:
        disconnect()


if __name__ == "__main__":
    main()
