"""
run_bot.py
==========
Main entry point for the automated trading bot.

Usage:
  python run_bot.py --mode paper    (default — no real trades)
  python run_bot.py --mode live     (ONLY after 60 days paper trading)

The bot runs continuously, waking up 2 minutes after each 1H bar close.
It is safe to stop with Ctrl+C — no open positions are affected.
"""

import argparse
import contextlib
import io
import json
import time
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib import request, error

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.mt5_connector import connect, disconnect, reconnect, is_connected
from src.live_engine import load_models, run_one_bar, check_time_stops, CONFIG

DISCORD_WEBHOOK_URL = (
    "https://discord.com/api/webhooks/"
    "1510697350446714970/"
    "At3egL5I13YfUj0jlvZd4BXq2290JxQv8_O3mTXLc3-id14PM0-5C8H8bX1mOddcSCDS"
)


class TeeBuffer(io.StringIO):
    """Capture printed output while still showing it in the terminal."""

    def __init__(self, stream):
        super().__init__()
        self.stream = stream

    def write(self, text):
        self.stream.write(text)
        return super().write(text)

    def flush(self):
        self.stream.flush()
        return super().flush()


def send_discord_message(message: str) -> bool:
    if not DISCORD_WEBHOOK_URL or not message.strip():
        return False

    # Discord has a 2000-character content limit. Keep room for code fences.
    chunks = [message[i:i + 1800] for i in range(0, len(message), 1800)]
    for index, chunk in enumerate(chunks, start=1):
        suffix = f" ({index}/{len(chunks)})" if len(chunks) > 1 else ""
        payload = {
            "content": (
                f"**run_bot.py hourly output{suffix}**\n"
                f"```text\n{chunk.strip()}\n```"
            )
        }
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            DISCORD_WEBHOOK_URL,
            data=data,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "run_bot.py",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=15) as response:
                if response.status >= 300:
                    print(f"  [DISCORD] Webhook returned HTTP {response.status}")
                    return False
        except (error.URLError, TimeoutError, OSError) as exc:
            print(f"  [DISCORD] Failed to send hourly output: {exc}")
            return False
    return True


def seconds_until_next_bar() -> float:
    """
    Returns seconds until 2 minutes after the next 1H bar close.
    1H bars close at :00 each hour. We wait until :02 to ensure
    the bar is fully confirmed in MT5 before fetching.
    """
    now = datetime.utcnow()
    minutes_past = now.minute
    seconds_past = now.second

    # Minutes remaining until next hour + 2 minute buffer
    wait_minutes = (60 - minutes_past) + 2
    wait_seconds = wait_minutes * 60 - seconds_past

    return wait_seconds


def main():
    parser = argparse.ArgumentParser(description="Algo Ensemble Trading Bot")
    parser.add_argument('--mode', choices=['paper', 'live'], default='paper',
                        help="paper = log only, live = execute real trades")
    args = parser.parse_args()

    CONFIG["mode"] = args.mode

    print("=" * 55)
    print("  ALGO ENSEMBLE BOT — Starting")
    print(f"  Mode    : {CONFIG['mode'].upper()}")
    print(f"  Symbol  : {CONFIG['symbol']}")
    print(f"  Risk    : {CONFIG['risk_pct']*100:.1f}% per trade")
    print("=" * 55)

    if CONFIG["mode"] == "live":
        print("\n  WARNING: LIVE MODE — Real trades will be executed.")
        print("  Only proceed if you have completed 60 days paper trading.")
        confirm = input("  Type 'CONFIRM' to proceed: ")
        if confirm != "CONFIRM":
            print("  Aborted.")
            sys.exit(0)

    # Connect to MT5
    if not connect():
        print("[ERROR] Cannot connect to MT5. Ensure terminal is open.")
        sys.exit(1)

    # Load all trained models
    ensemble, hmm_model, regime_map = load_models()

    print(f"\n  Bot is running. Waiting for next 1H bar close...")
    print(f"  Press Ctrl+C to stop safely.\n")

    try:
        while True:
            wait = seconds_until_next_bar()
            next_bar = datetime.utcnow()
            print(f"  [{datetime.utcnow().strftime('%H:%M:%S')}] "
                  f"Next bar check in {wait/60:.1f} min "
                  f"(~{int(wait//60):02d}:{int(wait%60):02d})")
            time.sleep(wait)

            # Health check — reconnect if MT5 was restarted
            if not is_connected():
                print("\n  [HEALTH] MT5 not responding. Attempting reconnect...")
                if not reconnect(max_attempts=5, wait_seconds=20):
                    print("  [HEALTH] Could not reconnect. Skipping this bar.")
                    continue

            hourly_output = TeeBuffer(sys.stdout)
            with contextlib.redirect_stdout(hourly_output):
                # Check time-stops on open positions first
                print("\n  Checking time-stops on open positions...")
                check_time_stops()

                # Run the signal pipeline
                signal = run_one_bar(ensemble, hmm_model, regime_map)

                if signal and signal.get('gate_passed'):
                    dir_str = "LONG" if signal['direction'] == 1 else "SHORT"
                    print(f"\n  *** SIGNAL: {dir_str} | "
                          f"Prob: {signal['ensemble_prob']:.3f} | "
                          f"Regime: {signal['regime']} ***")
                else:
                    print(f"\n  No signal this bar.")

            send_discord_message(hourly_output.getvalue())

    except KeyboardInterrupt:
        print("\n\n  [STOP] Bot stopped by user. Open positions unaffected.")

    finally:
        disconnect()


if __name__ == "__main__":
    main()
