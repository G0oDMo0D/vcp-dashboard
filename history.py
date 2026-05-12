"""
Signal history panel.

Persists each scan as a Parquet snapshot so we can later answer:
- Which Tier-A signals fired in the last 30 days?
- Did regime gate let them through?
- What did the price do 24h / 72h / 7d after the signal?

Snapshots live in data/snapshots/scan_YYYY-MM-DD_HHmm.parquet — one row per
evaluated symbol per scan. Forward-return analysis joins these against the
current 4H parquet files to retrieve out-of-sample price moves.
"""
from __future__ import annotations
import os
from pathlib import Path
from datetime import datetime, timedelta, timezone

import pandas as pd
import numpy as np
import streamlit as st


# ============================================================
# Snapshot writer
# ============================================================
def save_snapshot(scan_df: pd.DataFrame, regime: dict, snapshots_dir: str) -> str | None:
    """Append a snapshot if the current 4H bar hasn't been recorded yet.

    De-dup key: BTC 4H bar timestamp. Multiple page loads within the same 4H
    bar produce only one snapshot — we keep the most recent within that bar.
    """
    if scan_df.empty:
        return None

    Path(snapshots_dir).mkdir(parents=True, exist_ok=True)
    bar_ts = pd.Timestamp(regime["as_of"])
    bar_key = bar_ts.strftime("%Y-%m-%d_%H%M")
    out_path = os.path.join(snapshots_dir, f"scan_{bar_key}.parquet")

    df = scan_df.copy()
    df["scan_as_of"] = bar_ts
    df["btc_above_pct"] = regime["btc_above_pct"]
    df["btc_30d_ret"]   = regime["btc_30d_ret"]
    df["regime_active"] = regime["active"]

    # Keep only the columns we care about for history
    cols = [
        "scan_as_of", "symbol", "price", "tier", "symbol_pass", "regime_pass",
        "C1_stack", "C3_pass", "C4_pass", "C5_pass",
        "F1_pass", "F2_pass", "F3_pass", "F4_pass", "F5_pass",
        "C4_atr_ratio", "F1_base_60d", "F2_ret_90d", "F5_vol_spike",
        "stop", "target", "target_R", "R_ok",
        "btc_above_pct", "btc_30d_ret", "regime_active",
    ]
    cols = [c for c in cols if c in df.columns]
    df = df[cols]

    df.to_parquet(out_path, index=False)
    return out_path


# ============================================================
# Snapshot reader
# ============================================================
@st.cache_data(ttl=60, show_spinner=False)
def load_history(snapshots_dir: str) -> pd.DataFrame:
    """Load all snapshots from disk into one long-format DataFrame."""
    path = Path(snapshots_dir)
    if not path.exists():
        return pd.DataFrame()
    files = sorted(path.glob("scan_*.parquet"))
    if not files:
        return pd.DataFrame()
    parts = []
    for f in files:
        try:
            parts.append(pd.read_parquet(f))
        except Exception:
            continue
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    df["scan_as_of"] = pd.to_datetime(df["scan_as_of"], utc=True)
    return df.sort_values("scan_as_of").reset_index(drop=True)


# ============================================================
# Forward-return analysis
# ============================================================
def _forward_return(price_df: pd.DataFrame, entry_ts: pd.Timestamp, hours: int) -> float | None:
    """Return the price change from entry_ts to entry_ts + hours, as a fraction."""
    if price_df.empty:
        return None
    try:
        entry_idx = price_df.index.searchsorted(entry_ts)
        if entry_idx >= len(price_df):
            return None
        entry_price = float(price_df.iloc[entry_idx]["close"])
        target_ts = entry_ts + pd.Timedelta(hours=hours)
        target_idx = price_df.index.searchsorted(target_ts)
        if target_idx >= len(price_df):
            return None  # forward window not yet realized
        target_price = float(price_df.iloc[target_idx]["close"])
        if entry_price <= 0:
            return None
        return target_price / entry_price - 1
    except Exception:
        return None


def enrich_with_forward_returns(events: pd.DataFrame, data_dir: str) -> pd.DataFrame:
    """Add ret_24h, ret_72h, ret_7d columns by reading each symbol's 4H parquet."""
    if events.empty:
        return events

    out = events.copy()
    out["ret_24h"] = np.nan
    out["ret_72h"] = np.nan
    out["ret_7d"]  = np.nan

    price_cache: dict[str, pd.DataFrame] = {}

    for idx, row in out.iterrows():
        sym = row["symbol"]
        if sym not in price_cache:
            path = os.path.join(data_dir, f"{sym}_4h.parquet")
            if not os.path.exists(path):
                price_cache[sym] = pd.DataFrame()
            else:
                try:
                    price_cache[sym] = pd.read_parquet(path).sort_index()
                except Exception:
                    price_cache[sym] = pd.DataFrame()
        pdf = price_cache[sym]
        ts = row["scan_as_of"]
        out.at[idx, "ret_24h"] = _forward_return(pdf, ts, 24)
        out.at[idx, "ret_72h"] = _forward_return(pdf, ts, 72)
        out.at[idx, "ret_7d"]  = _forward_return(pdf, ts, 168)
    return out


# ============================================================
# Renderer
# ============================================================
def render_history(snapshots_dir: str, data_dir: str, universe: dict) -> None:
    history = load_history(snapshots_dir)

    if history.empty:
        st.info(
            "📜 **No history yet.** Snapshots are recorded on every scan. "
            "Open this app daily — within a few weeks you'll see forward-return analysis here."
        )
        return

    first_ts = history["scan_as_of"].min()
    last_ts  = history["scan_as_of"].max()
    span_days = (last_ts - first_ts).total_seconds() / 86400
    n_snapshots = history["scan_as_of"].nunique()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Snapshots", n_snapshots)
    c2.metric("History span", f"{span_days:.1f}d")
    c3.metric("Symbol-events", f"{len(history):,}")
    c4.metric("First scan", first_ts.strftime("%Y-%m-%d"))

    if span_days < 1.0:
        st.info(
            f"📜 Only {span_days*24:.1f} hours of history accumulated. "
            "Forward-return analysis needs at least 24-72h. "
            "Come back tomorrow."
        )
        return

    # --- Time-range filter ---
    days_back = st.slider("Show events from last N days", 1, 90, min(30, int(span_days) + 1))
    cutoff = last_ts - pd.Timedelta(days=days_back)
    h = history[history["scan_as_of"] >= cutoff].copy()
    h["display"] = h["symbol"].map(lambda s: universe.get(s, (s, "?"))[0])
    h["sector"]  = h["symbol"].map(lambda s: universe.get(s, (s, "?"))[1])

    # --- Tier-A events focus ---
    st.markdown("---")
    st.subheader("Tier A actionable signals")
    tier_a = h[(h["tier"] == "A") & h.get("R_ok", True)].copy()
    if tier_a.empty:
        st.caption("No Tier-A signals fired in the selected window.")
    else:
        # De-dup: keep first occurrence per (symbol, day) — repeated fires
        # within hours of each other are one event
        tier_a["day"] = tier_a["scan_as_of"].dt.floor("D")
        tier_a = tier_a.sort_values("scan_as_of").drop_duplicates(subset=["symbol", "day"], keep="first")
        tier_a = tier_a.drop(columns=["day"])
        with st.spinner(f"Computing forward returns for {len(tier_a)} events..."):
            tier_a = enrich_with_forward_returns(tier_a, data_dir)

        def fmt_ret(v):
            if pd.isna(v): return "—"
            return f"{v:+.1%}"

        tbl = pd.DataFrame({
            "When":      tier_a["scan_as_of"].dt.strftime("%Y-%m-%d %H:%M"),
            "Symbol":    tier_a["display"],
            "Sector":    tier_a["sector"],
            "Entry":     tier_a["price"].map(lambda x: f"${x:.4g}"),
            "Stop":      tier_a["stop"].map(lambda x: f"${x:.4g}"),
            "Target":    tier_a["target"].map(lambda x: f"${x:.4g}"),
            "R":         tier_a["target_R"].map(lambda x: f"{x:.1f}"),
            "ret 24h":   tier_a["ret_24h"].map(fmt_ret),
            "ret 72h":   tier_a["ret_72h"].map(fmt_ret),
            "ret 7d":    tier_a["ret_7d"].map(fmt_ret),
        })
        st.dataframe(tbl, hide_index=True, use_container_width=True)

        # Summary stats
        completed = tier_a.dropna(subset=["ret_72h"])
        if len(completed):
            wr = (completed["ret_72h"] > 0).mean()
            avg = completed["ret_72h"].mean()
            median = completed["ret_72h"].median()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Completed events", len(completed))
            c2.metric("Hit rate (ret 72h > 0)", f"{wr:.0%}")
            c3.metric("Avg 72h return", f"{avg:+.2%}")
            c4.metric("Median 72h return", f"{median:+.2%}")

    # --- A pending regime events ---
    st.markdown("---")
    st.subheader("A pending regime — what we missed")
    tier_ap = h[h["tier"] == "A_pending_regime"].copy()
    if tier_ap.empty:
        st.caption(
            "No A-pending events in window. Either regime was on the whole time, "
            "or there were no perfect 7/7 setups."
        )
    else:
        tier_ap["day"] = tier_ap["scan_as_of"].dt.floor("D")
        tier_ap = tier_ap.sort_values("scan_as_of").drop_duplicates(subset=["symbol", "day"], keep="first")
        tier_ap = tier_ap.drop(columns=["day"])
        with st.spinner(f"Computing forward returns for {len(tier_ap)} pending events..."):
            tier_ap = enrich_with_forward_returns(tier_ap, data_dir)

        def fmt_ret(v):
            if pd.isna(v): return "—"
            return f"{v:+.1%}"
        tbl = pd.DataFrame({
            "When":     tier_ap["scan_as_of"].dt.strftime("%Y-%m-%d %H:%M"),
            "Symbol":   tier_ap["display"],
            "Sector":   tier_ap["sector"],
            "Entry":    tier_ap["price"].map(lambda x: f"${x:.4g}"),
            "BTC reg":  tier_ap["btc_above_pct"].map(lambda x: f"{x:+.1%}"),
            "ret 24h":  tier_ap["ret_24h"].map(fmt_ret),
            "ret 72h":  tier_ap["ret_72h"].map(fmt_ret),
            "ret 7d":   tier_ap["ret_7d"].map(fmt_ret),
        })
        st.dataframe(tbl, hide_index=True, use_container_width=True)
        st.caption(
            "Setups that scored 7/7 locally but regime gate (F3/F4) was off. "
            "If many of these have positive ret_7d, your regime filter may be too strict."
        )

    # --- Tier transitions (B → A) ---
    st.markdown("---")
    st.subheader("Tier transitions")
    if h["scan_as_of"].nunique() < 2:
        st.caption("Need at least 2 snapshots to detect transitions.")
        return
    # For each symbol, look at sequential snapshots and detect tier change
    transitions = []
    for sym, grp in h.sort_values("scan_as_of").groupby("symbol"):
        prev_tier = None
        prev_ts = None
        for _, r in grp.iterrows():
            if prev_tier is not None and r["tier"] != prev_tier:
                transitions.append({
                    "When":    r["scan_as_of"],
                    "Symbol":  universe.get(sym, (sym, "?"))[0],
                    "From":    prev_tier,
                    "To":      r["tier"],
                    "Price":   r["price"],
                })
            prev_tier = r["tier"]
            prev_ts = r["scan_as_of"]
    if not transitions:
        st.caption("No tier transitions in this window — universe state has been stable.")
        return
    tr = pd.DataFrame(transitions).sort_values("When", ascending=False)
    # Highlight upgrades
    rank = {"D": 0, "C": 1, "B": 2, "A_pending_regime": 3, "A": 4}
    tr["direction"] = tr.apply(
        lambda r: "↑" if rank.get(r["To"], 0) > rank.get(r["From"], 0) else "↓",
        axis=1,
    )
    upgrades = tr[tr["direction"] == "↑"].head(20)
    if not upgrades.empty:
        st.markdown("**Recent upgrades** (most recent first):")
        tbl = pd.DataFrame({
            "When":   upgrades["When"].dt.strftime("%Y-%m-%d %H:%M"),
            "Symbol": upgrades["Symbol"],
            "From":   upgrades["From"],
            "To":     upgrades["To"],
            "Price":  upgrades["Price"].map(lambda x: f"${x:.4g}"),
        })
        st.dataframe(tbl, hide_index=True, use_container_width=True)
    else:
        st.caption("No upward tier transitions in the window.")
