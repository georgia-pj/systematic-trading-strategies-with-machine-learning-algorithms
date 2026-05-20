"""
triple_barrier.py
=================
Triple-barrier labeling for the metamodel coursework.

For each primary signal (date, instrument, direction), we look forward in time
and assign a label based on which barrier the price hits first:

    +1  →  upper barrier hit first  (trade was profitable → metamodel: take it)
    -1  →  lower barrier hit first  (trade lost          → metamodel: skip it)
     0  →  time limit hit first     (inconclusive, dropped by default)

Barrier widths are set as multiples of the 14-day ATR so they
self-scale across instruments with very different price levels.

Usage
-----
Run directly:
    python triple_barrier.py

Or import in your notebook:
    from triple_barrier import compute_atr, label_signals, run_labeling
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ── 0. PATHS ─────────────────────────────────────────────────────────────────

# Adjust these to point at your local copies of the CSV files
OHLCV_PATH   = "ohlcv_data.csv"
SIGNALS_PATH = "primary_signals.csv"


# ── 1. BARRIER PARAMETERS (justify these in your write-up) ───────────────────

UPPER_MULT = 1.5   # upper barrier = entry + UPPER_MULT × ATR
LOWER_MULT = 1.5   # lower barrier = entry − LOWER_MULT × ATR  (symmetric)
MAX_DAYS   = 20    # time limit: 20 trading days ≈ 1 calendar month
ATR_WINDOW = 14    # standard ATR lookback

# Justification notes (include this reasoning in your report):
#   - ATR scaling: each instrument has wildly different price levels (Gold ~$1800,
#     Copper ~$4). Using a fixed dollar barrier would be meaningless across
#     instruments. ATR measures recent daily volatility, so 1.5×ATR is roughly
#     1.5 "typical daily moves" — a reasonable profit/loss threshold.
#   - Symmetric barriers (same upper and lower mult): keeps the labeling unbiased.
#     You could make the upper wider than the lower to reflect a long-biased signal
#     if you have prior reason to.
#   - 20-day time limit: beyond ~1 month the original signal is stale. Shorter (10)
#     creates very few labels; longer (40) risks confounding with new signals.
#   - Sensitivity analysis: run label_signals() with a grid of values and compare
#     the label class distributions — this is what the markers want to see.


# ── 2. ATR CALCULATION ────────────────────────────────────────────────────────

def compute_atr(prices: pd.DataFrame, window: int = ATR_WINDOW) -> pd.Series:
    """
    Compute the Average True Range for a single-instrument price DataFrame.

    Parameters
    ----------
    prices : DataFrame with columns [date, open, high, low, close] sorted by date
    window : rolling lookback for the ATR mean (default 14)

    Returns
    -------
    Series of ATR values aligned to prices.index
    """
    prev_close = prices["close"].shift(1)
    true_range = pd.concat(
        [
            prices["high"] - prices["low"],          # intraday range
            (prices["high"] - prev_close).abs(),      # gap up
            (prices["low"]  - prev_close).abs(),      # gap down
        ],
        axis=1,
    ).max(axis=1)

    return true_range.rolling(window).mean()


# ── 3. SINGLE-TRADE LABELER ───────────────────────────────────────────────────

def label_one_trade(
    signal_date: pd.Timestamp,
    signal_dir: int,          # +1 (long) or -1 (short)
    entry_price: float,
    atr: float,
    prices_df: pd.DataFrame,  # full price history for this instrument
    upper_mult: float = UPPER_MULT,
    lower_mult: float = LOWER_MULT,
    max_days: int    = MAX_DAYS,
) -> int:
    """
    Label a single trade using the triple-barrier method.

    Logic
    -----
    For a LONG signal (+1):
        upper barrier = entry + upper_mult × ATR   (take profit)
        lower barrier = entry − lower_mult × ATR   (stop loss)
        Check each future day's high vs upper and low vs lower.

    For a SHORT signal (-1):
        upper barrier = entry − upper_mult × ATR   (take profit, i.e. price falls)
        lower barrier = entry + lower_mult × ATR   (stop loss, i.e. price rises)
        Barriers are mirrored because we profit when price goes DOWN.

    Returns +1, -1, or 0 (time limit).
    """
    if signal_dir == 1:   # long trade
        upper = entry_price + upper_mult * atr
        lower = entry_price - lower_mult * atr
    else:                 # short trade
        upper = entry_price - upper_mult * atr
        lower = entry_price + lower_mult * atr

    # Only look at the next max_days trading days after the signal
    future_prices = prices_df[prices_df["date"] > signal_date].head(max_days)

    for _, row in future_prices.iterrows():
        if signal_dir == 1:
            if row["high"] >= upper:  return  1   # take profit hit
            if row["low"]  <= lower:  return -1   # stop loss hit
        else:
            if row["low"]  <= upper:  return  1   # take profit hit (price fell)
            if row["high"] >= lower:  return -1   # stop loss hit  (price rose)

    return 0   # time limit hit — inconclusive


# ── 4. VECTORISED LABELER FOR ONE INSTRUMENT ─────────────────────────────────

def label_signals(
    instrument: str,
    ohlcv: pd.DataFrame,
    signals_long: pd.DataFrame,    # long-format signals: columns [date, instrument, signal]
    upper_mult: float = UPPER_MULT,
    lower_mult: float = LOWER_MULT,
    max_days: int    = MAX_DAYS,
    drop_zero: bool  = True,
) -> pd.DataFrame:
    """
    Apply triple-barrier labeling to all signals for one instrument.

    Parameters
    ----------
    instrument    : e.g. 'es1s'
    ohlcv         : full OHLCV DataFrame (all instruments)
    signals_long  : long-format signal DataFrame
    upper_mult    : ATR multiple for upper barrier
    lower_mult    : ATR multiple for lower barrier
    max_days      : maximum holding period in trading days
    drop_zero     : if True, drop inconclusive (time-limit) labels

    Returns
    -------
    DataFrame with columns:
        date, instrument, signal, entry_price, atr14, label
    """
    # ── Price history for this instrument ──
    prices = (
        ohlcv[ohlcv["instrument"] == instrument]
        .copy()
        .sort_values("date")
        .reset_index(drop=True)
    )
    prices["atr14"] = compute_atr(prices)

    # ── Filter to non-zero signals ──
    inst_signals = signals_long[
        (signals_long["instrument"] == instrument) &
        (signals_long["signal"] != 0)
    ].copy()

    # ── Merge entry price and ATR onto signal dates ──
    inst_signals = inst_signals.merge(
        prices[["date", "close", "atr14"]],
        on="date",
        how="left",
    ).rename(columns={"close": "entry_price"})

    # Drop rows where ATR is NaN (very early history before enough data)
    inst_signals = inst_signals.dropna(subset=["atr14"])

    # ── Apply the labeler row by row ──
    # (Row-by-row is necessary because each trade looks forward in the price series)
    labels = []
    for _, row in inst_signals.iterrows():
        lbl = label_one_trade(
            signal_date=row["date"],
            signal_dir=int(row["signal"]),
            entry_price=row["entry_price"],
            atr=row["atr14"],
            prices_df=prices,
            upper_mult=upper_mult,
            lower_mult=lower_mult,
            max_days=max_days,
        )
        labels.append(lbl)

    inst_signals["label"] = labels

    if drop_zero:
        inst_signals = inst_signals[inst_signals["label"] != 0]

    return inst_signals.reset_index(drop=True)


# ── 5. SENSITIVITY ANALYSIS (put this in your write-up) ──────────────────────

def sensitivity_analysis(
    instrument: str,
    ohlcv: pd.DataFrame,
    signals_long: pd.DataFrame,
):
    """
    Run the labeler across a grid of (upper_mult, lower_mult, max_days) values
    and print the label distribution for each. Include this table in your report
    to justify the chosen parameters.
    """
    print(f"\nSensitivity analysis for {instrument.upper()}")
    print(f"{'upper':>6} {'lower':>6} {'days':>6} | {'n_total':>8} {'n_pos':>8} {'n_neg':>8} {'dropped_%':>10}")
    print("-" * 65)

    for upper in [1.0, 1.5, 2.0]:
        for lower in [1.0, 1.5, 2.0]:
            for days in [10, 20, 40]:
                result = label_signals(
                    instrument, ohlcv, signals_long,
                    upper_mult=upper, lower_mult=lower,
                    max_days=days, drop_zero=False,
                )
                n_total = len(result)
                n_pos   = (result["label"] ==  1).sum()
                n_neg   = (result["label"] == -1).sum()
                n_zero  = (result["label"] ==  0).sum()
                drop_pct = 100 * n_zero / n_total if n_total > 0 else 0
                print(
                    f"{upper:>6.1f} {lower:>6.1f} {days:>6d} | "
                    f"{n_total:>8d} {n_pos:>8d} {n_neg:>8d} {drop_pct:>9.1f}%"
                )


# ── 6. VISUALISATION ─────────────────────────────────────────────────────────

def plot_label_distribution(labeled_df: pd.DataFrame, instrument: str):
    """
    Bar chart of label counts — include in your report.
    """
    counts = labeled_df["label"].value_counts().sort_index()
    colors = {-1: "#D85A30", 1: "#1D9E75"}
    fig, ax = plt.subplots(figsize=(5, 3))
    bars = ax.bar(
        ["-1 (loss)", "+1 (profit)"],
        [counts.get(-1, 0), counts.get(1, 0)],
        color=["#D85A30", "#1D9E75"],
        width=0.5,
        edgecolor="none",
    )
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            str(int(bar.get_height())),
            ha="center", va="bottom", fontsize=11,
        )
    ax.set_title(f"{instrument.upper()} — label distribution\n"
                 f"(upper={UPPER_MULT}×ATR, lower={LOWER_MULT}×ATR, T={MAX_DAYS}d)")
    ax.set_ylabel("Count")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"labels_{instrument}.png", dpi=150)
    plt.show()
    print(f"Saved labels_{instrument}.png")


def plot_example_trade(
    instrument: str,
    ohlcv: pd.DataFrame,
    labeled_df: pd.DataFrame,
    row_idx: int = 0,
):
    """
    Plot one example trade showing the three barriers on top of the price path.
    Helps visually confirm the logic is correct.
    """
    row    = labeled_df.iloc[row_idx]
    prices = (
        ohlcv[ohlcv["instrument"] == instrument]
        .sort_values("date")
        .reset_index(drop=True)
    )

    start_idx = prices.index[prices["date"] == row["date"]][0]
    window    = prices.iloc[start_idx : start_idx + MAX_DAYS + 1]

    entry   = row["entry_price"]
    atr     = row["atr14"]
    sig_dir = int(row["signal"])

    if sig_dir == 1:
        upper = entry + UPPER_MULT * atr
        lower = entry - LOWER_MULT * atr
    else:
        upper = entry - UPPER_MULT * atr
        lower = entry + LOWER_MULT * atr

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(window["date"], window["close"], color="#185FA5", lw=1.5, label="Close price")
    ax.axhline(upper, color="#1D9E75", lw=1.2, ls="--", label=f"Upper barrier ({UPPER_MULT}×ATR)")
    ax.axhline(lower, color="#D85A30", lw=1.2, ls="--", label=f"Lower barrier ({LOWER_MULT}×ATR)")
    ax.axvline(window["date"].iloc[-1], color="#888780", lw=1, ls=":", label=f"Time limit ({MAX_DAYS}d)")
    ax.axhline(entry, color="#888780", lw=0.8, ls="-", alpha=0.5, label="Entry price")

    label_str = {1: "+1 (profit)", -1: "−1 (loss)", 0: "0 (time limit)"}[row["label"]]
    signal_str = "LONG" if sig_dir == 1 else "SHORT"
    ax.set_title(
        f"{instrument.upper()} — example {signal_str} trade on {row['date'].date()}\n"
        f"Label: {label_str}"
    )
    ax.legend(fontsize=8, loc="best")
    ax.spines[["top", "right"]].set_visible(False)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(f"example_trade_{instrument}.png", dpi=150)
    plt.show()


# ── 7. MAIN RUNNER ────────────────────────────────────────────────────────────

def run_labeling(instruments: list[str] | None = None) -> pd.DataFrame:
    """
    Run the full labeling pipeline for a list of instruments.

    Parameters
    ----------
    instruments : list of tickers, e.g. ['es1s', 'nq1s']. If None, runs all 11.

    Returns
    -------
    Combined labeled DataFrame for all instruments.
    """
    print("Loading data...")
    ohlcv   = pd.read_csv(OHLCV_PATH,   parse_dates=["date"])
    signals = pd.read_csv(SIGNALS_PATH, parse_dates=["date"])

    # Melt signals from wide to long format
    signals_long = signals.melt(id_vars="date", var_name="instrument", value_name="signal")

    all_instruments = [
        "es1s", "nq1s", "fesx1s",          # equity index
        "cl1s", "ho1s", "rb1s", "ng1s",    # energy
        "gc1s", "si1s", "hg1s", "pl1s",    # metals
    ]
    if instruments is None:
        instruments = all_instruments

    all_labeled = []

    for inst in instruments:
        print(f"  Labeling {inst.upper()}...", end=" ")
        labeled = label_signals(inst, ohlcv, signals_long)
        n_pos = (labeled["label"] ==  1).sum()
        n_neg = (labeled["label"] == -1).sum()
        print(f"{len(labeled)} labels  (+1: {n_pos}, -1: {n_neg})")
        all_labeled.append(labeled)

    combined = pd.concat(all_labeled, ignore_index=True)
    combined.to_csv("labeled_signals.csv", index=False)
    print(f"\nSaved labeled_signals.csv  ({len(combined)} total rows)")
    return combined


# ── 8. ENTRY POINT ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Run labeling for all instruments ──
    labeled = run_labeling()

    print("\nOverall label distribution:")
    print(labeled["label"].value_counts().to_string())

    # ── Sensitivity analysis for one instrument (extend to all in your report) ──
    ohlcv   = pd.read_csv(OHLCV_PATH,   parse_dates=["date"])
    signals = pd.read_csv(SIGNALS_PATH, parse_dates=["date"])
    signals_long = signals.melt(id_vars="date", var_name="instrument", value_name="signal")

    sensitivity_analysis("es1s", ohlcv, signals_long)

    # ── Plot label distribution and an example trade for ES1S ──
    es_labeled = labeled[labeled["instrument"] == "es1s"]
    plot_label_distribution(es_labeled, "es1s")
    plot_example_trade("es1s", ohlcv, es_labeled, row_idx=10)
