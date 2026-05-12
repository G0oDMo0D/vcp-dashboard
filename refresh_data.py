"""
Refresh 4H OHLC data for all universe symbols from Coinglass.

Usage:
    export COINGLASS_API_KEY=your_key_here
    python refresh_data.py

Requires a Coinglass paid plan that supports /api/futures/price/history.
Writes parquet files to DATA_DIR (default: ./data/clean).

If you don't have a Coinglass key, you can use the bundled parquet files
shipped with the dashboard (they cover Jan 2024 -> May 2026).
"""
from __future__ import annotations
import os, sys, time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent))
from universe import UNIVERSE

DATA_DIR = Path(os.environ.get("VCP_DATA_DIR", "data/clean"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

API_KEY = os.environ.get("COINGLASS_API_KEY")
BASE_URL = "https://open-api-v4.coinglass.com/api/futures/price/history"

# Earliest date you want history from (Jan 1 2024 = 1704067200000 ms)
START_TS_MS = 1704067200000


def fetch_chunk(symbol: str, start_ms: int, limit: int = 3500, interval: str = "4h"):
    if not API_KEY:
        raise RuntimeError("COINGLASS_API_KEY not set in environment")
    headers = {"CG-API-KEY": API_KEY, "accept": "application/json"}
    params = {
        "exchange":   "Binance",
        "symbol":     symbol,
        "interval":   interval,
        "limit":      limit,
        "start_time": start_ms,
    }
    r = requests.get(BASE_URL, headers=headers, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    if data.get("code") != "0":
        raise RuntimeError(f"{symbol}: API error {data.get('code')} {data.get('msg')}")
    return data.get("data", [])


def rows_to_df(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"].astype(int), unit="ms", utc=True)
    for c in ("open", "high", "low", "close", "volume_usd"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.rename(columns={"volume_usd": "volume"})
    df = df.set_index("time").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df[["open", "high", "low", "close", "volume"]]


def fetch_symbol(symbol: str, interval: str = "4h") -> pd.DataFrame:
    """Fetch full history for a single symbol, in chunks of 3500 bars."""
    rows: list[dict] = []
    cursor = START_TS_MS
    while True:
        chunk = fetch_chunk(symbol, cursor, limit=3500, interval=interval)
        if not chunk:
            break
        rows.extend(chunk)
        last_ts = int(chunk[-1]["time"])
        if last_ts <= cursor or len(chunk) < 3500:
            break
        cursor = last_ts + 1
        time.sleep(0.3)  # rate-limit safety
    return rows_to_df(rows)


def main():
    if not API_KEY:
        print("❌ COINGLASS_API_KEY not set.")
        print("   Set it with: export COINGLASS_API_KEY=your_key")
        print("   Or use the bundled parquet files.")
        return 1

    print(f"Refreshing {len(UNIVERSE)} symbols + BTC to {DATA_DIR}/ ...")
    failed = []
    for i, symbol in enumerate(["BTCUSDT"] + list(UNIVERSE.keys()), 1):
        try:
            print(f"  [{i:3d}/{len(UNIVERSE)+1}] {symbol} ... ", end="", flush=True)
            df = fetch_symbol(symbol, "4h")
            if df.empty:
                print("EMPTY")
                failed.append(symbol)
                continue
            out = DATA_DIR / f"{symbol}_4h.parquet"
            df.to_parquet(out)
            print(f"{len(df)} bars, last={df.index[-1].strftime('%Y-%m-%d %H:%M')}")
        except Exception as e:
            print(f"FAIL: {e}")
            failed.append(symbol)
            time.sleep(1)

    # BTC daily for 30d regime
    try:
        print("\n  BTCUSDT (daily) ... ", end="", flush=True)
        rows = fetch_chunk("BTCUSDT", START_TS_MS, limit=900, interval="1d")
        df = rows_to_df(rows)
        df.to_parquet(DATA_DIR / "BTCUSDT_1d.parquet")
        print(f"{len(df)} bars")
    except Exception as e:
        print(f"FAIL: {e}")

    if failed:
        print(f"\n⚠ Failed symbols: {failed}")
    print(f"\n✓ Done. Refresh dashboard to see latest signals.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
