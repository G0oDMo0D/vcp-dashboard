"""
Strategy B1: EMA Crossover + BTC Local High

Entry conditions:
1. EMA35_4H crossed above EMA200_4H within last N bars (default 5)
2. BTC is within 7 days of a new 90-day local high (regime gate)

Backtested performance (89 events, 2024-2025):
- Win rate: 70.8%
- Median 14d return: +11.3%
- 95% CI on WR: [61.8%, 79.8%]
- Caveats: edge concentrates in Q4 2024 BTC rallies; Q4 2025 had 10% WR

This module computes signals only — see strategy_b1_backtest.py for trade simulation.
"""
from __future__ import annotations
import os
import glob
from dataclasses import dataclass

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================
B1_CFG = {
    # Regime gate
    "local_high_window_days":     90,   # rolling window for BTC local high
    "local_high_tolerance":       0.01, # 1% — "at local high" if within 1%
    "days_since_local_high_max":  7,    # max days since BTC made new local high

    # Signal — cross detection
    "cross_lookback_bars":        5,    # how many bars back to consider cross "fresh"

    # Trade construction (used by backtest module and Symbol Detail)
    "atr_stop_mult":              2.5,
    "target_R":                   3.0,  # 1:3 risk:reward
    "min_R":                      1.5,
    "max_R":                      6.0,
    "min_bars":                   250,
    "fee_bps":                    4.0,
    "slip_bps":                  10.0,
}


# ============================================================
# Helpers
# ============================================================
def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def atr_series(df: pd.DataFrame, n: int = 14) -> pd.Series:
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False, min_periods=n).mean()


# ============================================================
# BTC regime computation
# ============================================================
def compute_btc_regime(btc_4h: pd.DataFrame, cfg: dict = B1_CFG) -> pd.DataFrame:
    """For each 4H BTC bar, compute:
      - local_high_90d:           rolling 90d max of high
      - at_local_high:            close >= local_high * (1 - tolerance)
      - days_since_new_local_high: time since last "at_local_high" transition
      - regime_active:            days_since <= max
    """
    bars = cfg["local_high_window_days"] * 6  # 90 days * 6 bars/day
    btc = btc_4h.copy()
    btc["local_high"] = btc["high"].rolling(bars, min_periods=20).max()
    btc["at_local_high"] = btc["close"] / btc["local_high"] >= (1.0 - cfg["local_high_tolerance"])

    # Detect transitions to "new local high"
    at_hi_int = btc["at_local_high"].astype(int)
    new_local_high = (at_hi_int.diff() == 1).fillna(at_hi_int.iloc[0] == 1).astype(bool)

    # Track timestamp of last "new local high" transition
    last_hi_ts = pd.Series(index=btc.index, dtype="datetime64[ns, UTC]")
    last = btc.index[0]
    for ts, is_new in zip(btc.index, new_local_high):
        if is_new:
            last = ts
        last_hi_ts.loc[ts] = last

    btc["days_since_new_local_high"] = (
        btc.index - pd.DatetimeIndex(last_hi_ts.values, tz="UTC")
    ).total_seconds() / 86400.0

    btc["regime_active"] = btc["days_since_new_local_high"] <= cfg["days_since_local_high_max"]
    btc["dd_from_local_high"] = btc["close"] / btc["local_high"] - 1
    return btc


def get_btc_regime_status(btc_regime: pd.DataFrame) -> dict:
    """Snapshot of current BTC regime state for header display."""
    last_ts = btc_regime.index[-1]
    last_row = btc_regime.iloc[-1]
    return {
        "as_of":                     last_ts,
        "btc_price":                 float(last_row["close"]),
        "btc_local_high_90d":        float(last_row["local_high"]),
        "btc_dd_from_local_high":    float(last_row["dd_from_local_high"]),
        "days_since_local_high":     float(last_row["days_since_new_local_high"]),
        "regime_active":             bool(last_row["regime_active"]),
    }


# ============================================================
# Per-symbol signal evaluation
# ============================================================
def evaluate_symbol(symbol: str, df: pd.DataFrame, btc_regime: pd.DataFrame,
                    cfg: dict = B1_CFG) -> dict | None:
    """Evaluate B1 strategy state for one symbol at the latest bar."""
    if len(df) < cfg["min_bars"]:
        return None

    e35 = ema(df["close"], 35)
    e200 = ema(df["close"], 200)
    atr14 = atr_series(df, 14)

    # Cross detection
    above = (e35 > e200).astype(int)
    cross_up_series = (above.diff() == 1).fillna(False)

    # Current snapshot
    cur_close = float(df["close"].iloc[-1])
    cur_e35 = float(e35.iloc[-1]) if pd.notna(e35.iloc[-1]) else None
    cur_e200 = float(e200.iloc[-1]) if pd.notna(e200.iloc[-1]) else None
    cur_atr = float(atr14.iloc[-1]) if pd.notna(atr14.iloc[-1]) else None
    if cur_e35 is None or cur_e200 is None or cur_atr is None:
        return None

    cross_aligned = cur_e35 > cur_e200
    ema_spread_pct = abs(cur_e35 / cur_e200 - 1)

    # Find most recent crossover (within last 30 bars)
    recent = cross_up_series.iloc[-30:]
    cross_indices = recent.index[recent]
    if len(cross_indices) > 0:
        last_cross_ts = cross_indices[-1]
        last_cross_idx = df.index.get_loc(last_cross_ts)
        bars_since_cross = len(df) - 1 - last_cross_idx
        cross_active = bars_since_cross <= cfg["cross_lookback_bars"]
    else:
        last_cross_ts = None
        bars_since_cross = None
        cross_active = False

    # BTC regime at current bar
    regime_ts = df.index[-1]
    if regime_ts < btc_regime.index[0]:
        btc_active = False
        btc_days_since_high = None
        btc_dd_from_high = None
    else:
        try:
            btc_active = bool(btc_regime["regime_active"].asof(regime_ts))
            btc_days_since_high = float(btc_regime["days_since_new_local_high"].asof(regime_ts))
            btc_dd_from_high = float(btc_regime["dd_from_local_high"].asof(regime_ts))
        except (KeyError, ValueError):
            btc_active = False
            btc_days_since_high = None
            btc_dd_from_high = None

    # Trade plan: stop / target / R
    stop = cur_close - cfg["atr_stop_mult"] * cur_atr
    target = cur_close + (cur_close - stop) * cfg["target_R"]
    R = (target - cur_close) / max(1e-9, cur_close - stop) if stop < cur_close else 0
    R_ok = cfg["min_R"] <= R <= cfg["max_R"]

    # Tier classification
    if cross_active and cross_aligned and btc_active and R_ok:
        tier = "A"
    elif cross_active and cross_aligned and R_ok and not btc_active:
        tier = "A_pending_regime"
    elif cross_aligned and bars_since_cross is not None and bars_since_cross <= 20:
        tier = "B"
    elif cross_aligned:
        tier = "C"
    else:
        tier = "D"

    return {
        "symbol":                  symbol,
        "price":                   cur_close,
        "ema35":                   cur_e35,
        "ema200":                  cur_e200,
        "atr14":                   cur_atr,
        "ema_spread_pct":          ema_spread_pct,
        "cross_aligned":           cross_aligned,
        "cross_active":            cross_active,
        "bars_since_cross":        bars_since_cross,
        "last_cross_ts":           last_cross_ts,
        "btc_regime_active":       btc_active,
        "btc_days_since_high":     btc_days_since_high,
        "btc_dd_from_high":        btc_dd_from_high,
        "stop":                    stop,
        "target":                  target,
        "R_target":                R,
        "R_ok":                    R_ok,
        "tier":                    tier,
    }


def scan_all(data_dir: str, symbols: list[str] | None = None,
             cfg: dict = B1_CFG) -> tuple[pd.DataFrame, dict]:
    """Scan all symbols for B1 signals. Returns (rows_df, btc_regime_status)."""
    btc_path = os.path.join(data_dir, "BTCUSDT_4h.parquet")
    if not os.path.exists(btc_path):
        return pd.DataFrame(), {"regime_active": False, "as_of": None}

    btc_4h = pd.read_parquet(btc_path)
    btc_regime = compute_btc_regime(btc_4h, cfg)
    regime_status = get_btc_regime_status(btc_regime)

    if symbols is None:
        files = sorted(glob.glob(os.path.join(data_dir, "*_4h.parquet")))
        symbols = [
            os.path.basename(f).replace("_4h.parquet", "")
            for f in files if "BTC" not in os.path.basename(f)
        ]

    rows = []
    for sym in symbols:
        path = os.path.join(data_dir, f"{sym}_4h.parquet")
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_parquet(path)
            result = evaluate_symbol(sym, df, btc_regime, cfg)
            if result is not None:
                rows.append(result)
        except Exception:
            continue

    return pd.DataFrame(rows), regime_status
