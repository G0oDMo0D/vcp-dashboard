"""
VCP-EMA-Stack — signal engine. Pure functions, no UI dependencies.

Outputs DataFrame per scan with all filter values + decision.
Used by both the Streamlit app and any background scanner.
"""
from __future__ import annotations
import os, glob
import numpy as np
import pandas as pd

# Strategy parameters — single source of truth
CFG = {
    "ema_fast":         35,
    "ema_mid":          100,
    "ema_slow":         200,
    "slope_lookback":   20,
    "body_count":       3,
    "atr_ratio_max":    0.75,
    "atr_stop_mult":    2.5,
    "base_days":        30,
    "base_in_range_min":0.60,   # C5 threshold (30d)
    "base_60d_min":     0.75,   # F1 threshold
    "base_range_pct":   0.15,
    "ret_90d_min":      0.0,
    "ret_90d_max":      0.50,
    "vol_spike_min":    1.3,
    "vol_lookback":     360,
    "btc_above_ema_min":0.05,
    "btc_ret_30d_min":  0.05,
    "min_R":            1.5,
    "max_R":            6.0,
    "target_pct_floor": 1.08,
    "min_bars":         540,
}


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def atr(df: pd.DataFrame, n: int) -> pd.Series:
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"]  - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False, min_periods=n).mean()


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with all indicator columns appended."""
    out = df.copy()
    out["ema35"]  = ema(df["close"], CFG["ema_fast"])
    out["ema100"] = ema(df["close"], CFG["ema_mid"])
    out["ema200"] = ema(df["close"], CFG["ema_slow"])
    out["atr14"]  = atr(df, 14)
    out["atr20"]  = atr(df, 20)
    out["atr100"] = atr(df, CFG["ema_slow"])
    out["atr_ratio"] = out["atr20"] / out["atr100"]
    body_low = np.minimum(df["open"], df["close"])
    above = (body_low > out["ema200"]).astype(int)
    grp = above.eq(0).cumsum()
    out["streak"] = above.groupby(grp).cumsum()
    return out


def evaluate_symbol(symbol: str, df: pd.DataFrame,
                    btc_above_pct: float, btc_30d_ret: float) -> dict | None:
    """Evaluate VCP-EMA-Stack conditions at the latest bar.

    Returns dict with all condition values + decision tier, or None if insufficient history.
    """
    if len(df) < CFG["min_bars"]:
        return None

    ind = compute_indicators(df)
    row = ind.iloc[-1]
    px = float(row["close"])

    # --- Core conditions C1-C5 ---
    # C1+C2: trend stack + positive slope
    stack = (row["ema35"] > row["ema100"] > row["ema200"])
    slope_fast = ind["ema35"].iloc[-1] > ind["ema35"].iloc[-1 - CFG["slope_lookback"]]
    slope_slow = ind["ema200"].iloc[-1] > ind["ema200"].iloc[-1 - CFG["slope_lookback"]]
    C1 = bool(stack and slope_fast and slope_slow)

    # C2 alias (we treat C1+C2 as a single trend gate)
    C2 = C1

    # C3: 3+ consecutive bars body > EMA200_4H
    streak = int(row["streak"])
    C3 = streak >= CFG["body_count"]

    # C4: ATR compression
    atr_ratio = float(row["atr_ratio"])
    C4 = (not np.isnan(atr_ratio)) and atr_ratio < CFG["atr_ratio_max"]

    # C5: 30d base — fraction of last 180 bars within ±15% of current
    look30 = CFG["base_days"] * 6
    win30 = df["close"].iloc[-look30:].values
    base_30 = float(np.mean((win30 >= px*0.85) & (win30 <= px*1.15)))
    C5 = base_30 >= CFG["base_in_range_min"]

    # --- Smart filters F1, F2, F5 ---
    # F1: 60d base ≥ 0.75
    look60 = 60 * 6
    if len(df) >= look60:
        win60 = df["close"].iloc[-look60:].values
        base_60 = float(np.mean((win60 >= px*0.85) & (win60 <= px*1.15)))
    else:
        base_60 = float("nan")
    F1 = (not np.isnan(base_60)) and base_60 >= CFG["base_60d_min"]

    # F2: ret_90d in [0%, 50%]
    if len(df) >= 540:
        ret90 = float(df["close"].iloc[-1] / df["close"].iloc[-540] - 1)
    else:
        ret90 = float("nan")
    F2 = (not np.isnan(ret90)) and (CFG["ret_90d_min"] <= ret90 <= CFG["ret_90d_max"])

    # F5: volume spike vs 60d median
    if "volume" in df.columns and df["volume"].notna().sum() >= CFG["vol_lookback"]:
        vol_med = float(df["volume"].iloc[-CFG["vol_lookback"]:].median())
        vol_3   = float(df["volume"].iloc[-3:].mean())
        vol_spike = vol_3 / vol_med if vol_med > 0 else float("nan")
    else:
        vol_spike = float("nan")
    F5 = (not np.isnan(vol_spike)) and vol_spike >= CFG["vol_spike_min"]

    # --- BTC regime (computed globally, passed in) ---
    F3 = btc_above_pct >= CFG["btc_above_ema_min"]
    F4 = btc_30d_ret >= CFG["btc_ret_30d_min"]

    # --- Stop & target ---
    stop_atr = px - CFG["atr_stop_mult"] * float(row["atr14"])
    stop_struct = float(row["ema100"])
    stop = max(stop_atr, stop_struct)
    target = max(float(row["ema200"]), px * CFG["target_pct_floor"])
    R = (target - px) / max(1e-9, px - stop) if px > stop else 0.0

    # --- Decision ---
    symbol_passes = sum([C1, C3, C4, C5, F1, F2, F5])  # 7 local conditions
    regime_passes = sum([F3, F4])
    total = symbol_passes + regime_passes

    if symbol_passes == 7 and regime_passes == 2:
        tier = "A"
    elif symbol_passes == 7:
        tier = "A_pending_regime"
    elif symbol_passes == 6:
        tier = "B"
    elif symbol_passes == 5:
        tier = "C"
    else:
        tier = "D"

    R_ok = CFG["min_R"] <= R <= CFG["max_R"]

    return {
        "symbol":       symbol,
        "as_of":        df.index[-1],
        "price":        px,
        "ema35":        float(row["ema35"]),
        "ema100":       float(row["ema100"]),
        "ema200":       float(row["ema200"]),
        "atr14":        float(row["atr14"]),
        # condition flags
        "C1_stack":     C1,
        "C3_body_streak": streak,
        "C3_pass":      C3,
        "C4_atr_ratio": atr_ratio,
        "C4_pass":      C4,
        "C5_base_30d":  base_30,
        "C5_pass":      C5,
        "F1_base_60d":  base_60,
        "F1_pass":      F1,
        "F2_ret_90d":   ret90,
        "F2_pass":      F2,
        "F3_pass":      F3,
        "F4_pass":      F4,
        "F5_vol_spike": vol_spike,
        "F5_pass":      F5,
        # outcome
        "stop":         stop,
        "target":       target,
        "target_R":     R,
        "R_ok":         R_ok,
        "symbol_pass":  symbol_passes,
        "regime_pass":  regime_passes,
        "total":        total,
        "tier":         tier,
        "actionable":   bool(tier == "A" and R_ok),
    }


def scan_all(data_dir: str, symbols: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
    """Scan all symbols in data_dir/{SYMBOL}_4h.parquet. Returns (scan_df, regime_dict)."""
    # BTC regime — 4H EMA200
    btc_4h_path = f"{data_dir}/BTCUSDT_4h.parquet"
    btc_4h = pd.read_parquet(btc_4h_path)
    btc_ema200 = ema(btc_4h["close"], 200)
    btc_close = float(btc_4h["close"].iloc[-1])
    btc_ema_val = float(btc_ema200.iloc[-1])
    btc_above_pct = btc_close / btc_ema_val - 1

    # BTC daily for 30d return
    btc_1d_path = f"{data_dir}/BTCUSDT_1d.parquet"
    btc_30d_ret = float("nan")
    if os.path.exists(btc_1d_path):
        btc_1d = pd.read_parquet(btc_1d_path)
        if len(btc_1d) >= 31:
            btc_30d_ret = float(btc_1d["close"].iloc[-1] / btc_1d["close"].iloc[-31] - 1)
    if np.isnan(btc_30d_ret):
        # fallback to 4H equivalent (180 bars ≈ 30 days)
        if len(btc_4h) >= 181:
            btc_30d_ret = float(btc_4h["close"].iloc[-1] / btc_4h["close"].iloc[-181] - 1)
        else:
            btc_30d_ret = 0.0

    regime = {
        "btc_price":     btc_close,
        "btc_ema200_4h": btc_ema_val,
        "btc_above_pct": btc_above_pct,
        "btc_30d_ret":   btc_30d_ret,
        "F3_active":     btc_above_pct >= CFG["btc_above_ema_min"],
        "F4_active":     btc_30d_ret   >= CFG["btc_ret_30d_min"],
        "active":        (btc_above_pct >= CFG["btc_above_ema_min"]) and (btc_30d_ret >= CFG["btc_ret_30d_min"]),
        "as_of":         btc_4h.index[-1],
    }

    if symbols is None:
        files = glob.glob(f"{data_dir}/*_4h.parquet")
        symbols = [os.path.basename(f).replace("_4h.parquet", "") for f in files]
        symbols = [s for s in symbols if s != "BTCUSDT"]

    rows = []
    for sym in symbols:
        path = f"{data_dir}/{sym}_4h.parquet"
        if not os.path.exists(path):
            continue
        df = pd.read_parquet(path)
        if len(df) < CFG["min_bars"]:
            continue
        r = evaluate_symbol(sym, df, btc_above_pct, btc_30d_ret)
        if r is not None:
            rows.append(r)
    return pd.DataFrame(rows), regime


if __name__ == "__main__":
    df, reg = scan_all("/home/claude/strategy/data/clean")
    print("Regime:", reg)
    print(f"\nScan ({len(df)} symbols):")
    cols = ["symbol", "price", "tier", "symbol_pass", "regime_pass",
            "C4_atr_ratio", "F1_base_60d", "F2_ret_90d", "F5_vol_spike", "target_R"]
    print(df.sort_values(["symbol_pass", "F1_base_60d"], ascending=[False, False])[cols].to_string(index=False))
