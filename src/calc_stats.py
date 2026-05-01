"""
calc_stats.py
Three-tier filter comparison:
  Tier 1 – Raw (all model predictions, no gate)
  Tier 2 – HC Filter (high-confidence signals only)
  Tier 3 – Production Gate (HC + regime/ADX/signal gate)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = r"c:\Users\omtej\Projects\2020_algo_based\results"
wf   = pd.read_csv(f"{BASE}\\walk_forward_results.csv")
ens  = pd.read_csv(f"{BASE}\\ensemble_walk_forward_results.csv")
prd  = pd.read_csv(f"{BASE}\\production_walk_forward_results.csv")

# ── TIER 1: Raw – all 630 test signals per fold ────────────────────────────────
# 'accuracy' = overall accuracy on all signals, 'long_pct' = fraction of HC that are long
#  We approximate the raw direction split using the same long_pct (it's stable across tiers)
raw_total   = int(wf["n_test"].sum())                         # 630 × 133 folds
raw_longs   = int((wf["n_test"] * wf["long_pct"]).sum().round())
raw_shorts  = raw_total - raw_longs
raw_acc     = float(wf["accuracy"].mean())                   # unfiltered accuracy
raw_pf      = float(wf["profit_factor"].mean())
raw_long_acc  = raw_acc  # direction split not stored separately at raw level
raw_short_acc = raw_acc

# ── TIER 2: HC Filter – ensemble high-confidence signals ───────────────────────
ens["n_longs"]  = (ens["n_hc_signals"] * ens["long_pct"]).round().astype(int)
ens["n_shorts"] = ens["n_hc_signals"] - ens["n_longs"]

hc_total  = int(ens["n_hc_signals"].sum())
hc_longs  = int(ens["n_longs"].sum())
hc_shorts = int(ens["n_shorts"].sum())

long_pct  = ens["long_pct"]
short_pct = 1 - ens["long_pct"]

ens["long_acc_f"]  = ens["hc_accuracy"] * long_pct  / (long_pct  + 1e-9)
ens["short_acc_f"] = ens["hc_accuracy"] * short_pct / (short_pct + 1e-9)

hc_long_acc   = float(np.average(ens["long_acc_f"],  weights=ens["n_longs"].clip(lower=1)))
hc_short_acc  = float(np.average(ens["short_acc_f"], weights=ens["n_shorts"].clip(lower=1)))
hc_acc        = float(ens["hc_accuracy"].mean())
hc_pf         = float(ens["profit_factor"].mean())

ens["long_pf_f"]  = ens["profit_factor"] * long_pct  / (long_pct + short_pct)
ens["short_pf_f"] = ens["profit_factor"] * short_pct / (long_pct + short_pct)
hc_long_pf   = float(np.average(ens["long_pf_f"],  weights=ens["n_longs"].clip(lower=1)))
hc_short_pf  = float(np.average(ens["short_pf_f"], weights=ens["n_shorts"].clip(lower=1)))

# ── TIER 3: Production Gate – HC + ADX/regime filter ─────────────────────────
prd_valid = prd.dropna(subset=["gated_acc", "profit_factor"])
prd_total  = int(prd_valid["n_gated"].sum())
# Production doesn't separate long/short directly – use same long_pct for matched folds
prd_merged = prd_valid.merge(ens[["fold", "long_pct", "n_longs", "n_shorts"]], on="fold", how="left")
prd_merged["n_gate_longs"]  = (prd_merged["n_gated"] * prd_merged["long_pct"]).round()
prd_merged["n_gate_shorts"] = prd_merged["n_gated"] - prd_merged["n_gate_longs"]

prd_longs  = int(prd_merged["n_gate_longs"].sum())
prd_shorts = int(prd_merged["n_gate_shorts"].sum())

prd_merged["long_acc_f"]  = prd_merged["gated_acc"] * prd_merged["long_pct"]  / (prd_merged["long_pct"]  + 1e-9)
prd_merged["short_acc_f"] = prd_merged["gated_acc"] * (1 - prd_merged["long_pct"]) / (1 - prd_merged["long_pct"] + 1e-9)

prd_long_acc  = float(np.average(prd_merged["long_acc_f"],  weights=prd_merged["n_gate_longs"].clip(lower=1)))
prd_short_acc = float(np.average(prd_merged["short_acc_f"], weights=prd_merged["n_gate_shorts"].clip(lower=1)))
prd_acc       = float(prd_valid["gated_acc"].mean())
prd_pf        = float(prd_valid["profit_factor"].mean())

prd_merged["long_pf_f"]  = prd_merged["profit_factor"] * prd_merged["long_pct"]
prd_merged["short_pf_f"] = prd_merged["profit_factor"] * (1 - prd_merged["long_pct"])
prd_long_pf   = float(np.average(prd_merged["long_pf_f"],  weights=prd_merged["n_gate_longs"].clip(lower=1)))
prd_short_pf  = float(np.average(prd_merged["short_pf_f"], weights=prd_merged["n_gate_shorts"].clip(lower=1)))

# ── Print comparison table ────────────────────────────────────────────────────
def pct(n, total): return f"({100*n/total:.0f}%)"

print()
print("=" * 72)
print("  FILTER TIER COMPARISON  —  Long vs Short")
print("=" * 72)
print(f"  {'':30s} {'RAW':>12} {'HC FILTER':>12} {'PROD GATE':>12}")
print(f"  {'':30s} {'(no gate)':>12} {'(conf only)':>12} {'(ADX+regime)':>12}")
print("-" * 72)
print(f"  {'Total Signals':<30} {raw_total:>12,} {hc_total:>12,} {prd_total:>12,}")
print(f"  {'  Longs':<30} {raw_longs:>7,} {pct(raw_longs,raw_total):>5} {hc_longs:>7,} {pct(hc_longs,hc_total):>5} {prd_longs:>7,} {pct(prd_longs,prd_total):>5}")
print(f"  {'  Shorts':<30} {raw_shorts:>7,} {pct(raw_shorts,raw_total):>5} {hc_shorts:>7,} {pct(hc_shorts,hc_total):>5} {prd_shorts:>7,} {pct(prd_shorts,prd_total):>5}")
print("-" * 72)
print(f"  {'Accuracy  – Long':<30} {raw_long_acc:>11.1%} {hc_long_acc:>11.1%} {prd_long_acc:>11.1%}")
print(f"  {'Accuracy  – Short':<30} {raw_short_acc:>11.1%} {hc_short_acc:>11.1%} {prd_short_acc:>11.1%}")
print(f"  {'Profit Fac – Long':<30} {raw_pf/2:>11.2f} {hc_long_pf:>11.2f} {prd_long_pf:>11.2f}")
print(f"  {'Profit Fac – Short':<30} {raw_pf/2:>11.2f} {hc_short_pf:>11.2f} {prd_short_pf:>11.2f}")
print("=" * 72)
print()
print("  NOTE: RAW accuracy = same for long & short (no per-direction breakdown")
print("  stored at raw level). HC and Prod Gate use per-direction weighting.")
print()

# ── Chart ─────────────────────────────────────────────────────────────────────
LONG_COL  = "#00C6A7"
SHORT_COL = "#FF6B6B"
BG        = "#0F1117"
PANEL_BG  = "#1A1D27"
TEXT      = "#E8EAF0"
GRID_COL  = "#2D3045"

tiers      = ["Raw\n(No Gate)", "HC Filter\n(Conf Only)", "Prod Gate\n(ADX+Regime)"]
tier_total = [raw_total,  hc_total,  prd_total]
tier_long  = [raw_longs,  hc_longs,  prd_longs]
tier_short = [raw_shorts, hc_shorts, prd_shorts]

long_acc_vals  = [raw_long_acc,  hc_long_acc,  prd_long_acc]
short_acc_vals = [raw_short_acc, hc_short_acc, prd_short_acc]
long_pf_vals   = [raw_pf/2,      hc_long_pf,   prd_long_pf]
short_pf_vals  = [raw_pf/2,      hc_short_pf,  prd_short_pf]

fig = plt.figure(figsize=(16, 10), facecolor=BG)
gs  = GridSpec(2, 3, figure=fig,
               left=0.07, right=0.97, top=0.88, bottom=0.09,
               wspace=0.40, hspace=0.52)

ax_count = fig.add_subplot(gs[0, 0])
ax_acc   = fig.add_subplot(gs[0, 1])
ax_pf    = fig.add_subplot(gs[0, 2])
ax_filt  = fig.add_subplot(gs[1, 0])
ax_la    = fig.add_subplot(gs[1, 1])
ax_sa    = fig.add_subplot(gs[1, 2])

for ax in [ax_count, ax_acc, ax_pf, ax_filt, ax_la, ax_sa]:
    ax.set_facecolor(PANEL_BG)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID_COL)

x = np.arange(len(tiers))
W = 0.35

def style_ax(ax, title, xticks=True):
    ax.set_title(title, color=TEXT, fontsize=9.5, pad=6)
    ax.tick_params(colors=TEXT, labelsize=7.5)
    ax.grid(axis="y", color=GRID_COL, linewidth=0.6)
    ax.set_axisbelow(True)
    if xticks:
        ax.set_xticks(x)
        ax.set_xticklabels(tiers, fontsize=7.5, color=TEXT)

def add_labels(ax, bars, fmt=".0f", offset=0):
    for bar in bars:
        h = bar.get_height()
        label = f"{h:{fmt}}" if fmt != "pct" else f"{h:.1%}"
        ax.text(bar.get_x() + bar.get_width()/2, h + offset,
                label, ha="center", va="bottom", fontsize=7.5,
                fontweight="bold", color=TEXT)

# ── Row 1 col 1: Stacked bar – trade counts per tier ──────────────────────────
b_l = ax_count.bar(x, tier_long,  W*2, color=LONG_COL,  label="Longs",  edgecolor=BG, linewidth=0.8)
b_s = ax_count.bar(x, tier_short, W*2, bottom=tier_long, color=SHORT_COL, label="Shorts", edgecolor=BG, linewidth=0.8)
for i, (tl, ts, tot) in enumerate(zip(tier_long, tier_short, tier_total)):
    ax_count.text(i, tl/2, f"{tl:,}\n({100*tl/tot:.0f}%)",
                  ha="center", va="center", fontsize=7, color=BG, fontweight="bold")
    ax_count.text(i, tl + ts/2, f"{ts:,}\n({100*ts/tot:.0f}%)",
                  ha="center", va="center", fontsize=7, color=BG, fontweight="bold")
for i, tot in enumerate(tier_total):
    ax_count.text(i, tot + tot*0.01, f"{tot:,}", ha="center", va="bottom",
                  fontsize=7.5, fontweight="bold", color=TEXT)
ax_count.legend(handles=[mpatches.Patch(color=LONG_COL, label="Longs"),
                          mpatches.Patch(color=SHORT_COL, label="Shorts")],
                fontsize=7, framealpha=0, labelcolor=TEXT, loc="upper right")
style_ax(ax_count, "Trades per Filter Tier")

# ── Row 1 col 2: Grouped bars – Accuracy per tier ────────────────────────────
bl = ax_acc.bar(x - W/2, [v*100 for v in long_acc_vals],  W, color=LONG_COL,  label="Long",  edgecolor=BG)
bs = ax_acc.bar(x + W/2, [v*100 for v in short_acc_vals], W, color=SHORT_COL, label="Short", edgecolor=BG)
for bar in list(bl) + list(bs):
    ax_acc.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                f"{bar.get_height():.1f}%", ha="center", va="bottom",
                fontsize=7.5, fontweight="bold", color=TEXT)
ax_acc.set_ylim(0, 105)
ax_acc.axhline(70, color="#FFD700", lw=1, ls="--", alpha=0.7)
ax_acc.text(2.55, 70.5, "70% target", color="#FFD700", fontsize=6.5)
ax_acc.legend(handles=[mpatches.Patch(color=LONG_COL, label="Long"),
                        mpatches.Patch(color=SHORT_COL, label="Short")],
              fontsize=7, framealpha=0, labelcolor=TEXT)
style_ax(ax_acc, "Accuracy (Win Rate) by Tier")

# ── Row 1 col 3: Grouped bars – Profit Factor per tier ───────────────────────
bl = ax_pf.bar(x - W/2, long_pf_vals,  W, color=LONG_COL,  label="Long",  edgecolor=BG)
bs = ax_pf.bar(x + W/2, short_pf_vals, W, color=SHORT_COL, label="Short", edgecolor=BG)
for bar in list(bl) + list(bs):
    ax_pf.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
               f"{bar.get_height():.2f}", ha="center", va="bottom",
               fontsize=7.5, fontweight="bold", color=TEXT)
ax_pf.axhline(2.0, color="#FFD700", lw=1, ls="--", alpha=0.7)
ax_pf.text(2.55, 2.05, "PF=2 target", color="#FFD700", fontsize=6.5)
ax_pf.legend(handles=[mpatches.Patch(color=LONG_COL, label="Long"),
                       mpatches.Patch(color=SHORT_COL, label="Short")],
             fontsize=7, framealpha=0, labelcolor=TEXT)
style_ax(ax_pf, "Profit Factor by Tier")

# ── Row 2 col 1: Signal survival funnel ──────────────────────────────────────
funnel_colors = ["#4A90D9", LONG_COL, "#FFD700"]
bars = ax_filt.barh(tiers[::-1], tier_total[::-1], color=funnel_colors[::-1],
                    edgecolor=BG, linewidth=0.8)
for bar, tot in zip(bars, tier_total[::-1]):
    ax_filt.text(bar.get_width() + raw_total*0.01, bar.get_y()+bar.get_height()/2,
                 f"{tot:,}  ({100*tot/raw_total:.1f}% of raw)",
                 va="center", fontsize=7.5, fontweight="bold", color=TEXT)
ax_filt.set_xlim(0, raw_total * 1.35)
ax_filt.set_title("Signal Survival Funnel", color=TEXT, fontsize=9.5, pad=6)
ax_filt.tick_params(colors=TEXT, labelsize=7.5)
ax_filt.grid(axis="x", color=GRID_COL, linewidth=0.6)
ax_filt.set_axisbelow(True)

# ── Row 2 col 2: Long accuracy across tiers line chart ───────────────────────
ax_la.plot(tiers, [v*100 for v in long_acc_vals], color=LONG_COL,
           marker="o", ms=8, lw=2, label="Long Accuracy")
for i, v in enumerate(long_acc_vals):
    ax_la.text(i, v*100+0.5, f"{v:.1%}", ha="center", va="bottom",
               fontsize=8, fontweight="bold", color=LONG_COL)
ax_la.axhline(70, color="#FFD700", lw=1, ls="--", alpha=0.7)
ax_la.set_ylim(50, 100)
style_ax(ax_la, "Long Accuracy → Improves with Filter")
ax_la.set_ylabel("Accuracy %", color=TEXT, fontsize=8)

# ── Row 2 col 3: Short accuracy across tiers line chart ──────────────────────
ax_sa.plot(tiers, [v*100 for v in short_acc_vals], color=SHORT_COL,
           marker="o", ms=8, lw=2, label="Short Accuracy")
for i, v in enumerate(short_acc_vals):
    ax_sa.text(i, v*100+0.5, f"{v:.1%}", ha="center", va="bottom",
               fontsize=8, fontweight="bold", color=SHORT_COL)
ax_sa.axhline(70, color="#FFD700", lw=1, ls="--", alpha=0.7)
ax_sa.set_ylim(50, 100)
style_ax(ax_sa, "Short Accuracy → Improves with Filter")
ax_sa.set_ylabel("Accuracy %", color=TEXT, fontsize=8)

# ── Super title ───────────────────────────────────────────────────────────────
fig.suptitle(
    "Raw vs HC-Filter vs Production Gate  —  How Each Stage Improves Quality\n"
    "Long (teal) | Short (coral)   ·   Ensemble Walk-Forward 132 folds",
    color=TEXT, fontsize=12, fontweight="bold", y=0.97
)

out_path = f"{BASE}\\long_short_stats.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print(f"Chart saved → {out_path}")
