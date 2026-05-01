# Run this once to create the template
import pandas as pd

template = pd.DataFrame(columns=[
    'trade_id',          # sequential number
    'entry_datetime',    # exact bar timestamp
    'pair',              # EURUSD
    'direction',         # 1=long, -1=short
    'entry_price',
    'tp_price',          # entry + ATR × 1.5
    'sl_price',          # entry - ATR × 1.0
    'regime',            # quiet_trend / volatile_trend / ranging
    'ensemble_prob',     # raw probability from model
    'confidence',        # abs(prob - 0.5) × 2
    'vix_regime',        # 0/1/2
    'dxy_slope',         # +1 or -1
    'us10y_direction',   # +1 or -1
    'adx_14',            # value at entry
    'session',           # 0/1/2/3
    'exit_datetime',
    'exit_price',
    'outcome',           # TP / SL / TIME
    'pnl_pips',
    'pnl_units',         # +1.5 or -1.0
    'notes',
])

template.to_csv('../results/paper_trade_log.csv', index=False)
print("Trade log template created.")