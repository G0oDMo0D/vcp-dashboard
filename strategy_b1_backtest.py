"""
Strategy B1 — per-symbol historical backtest with visualization.

Walks through each bar; opens a trade when:
1. EMA35 just crossed above EMA200 (within last N bars)
2. BTC was within N days of a new 90d local high at time of cross

Exit: trailing stop (2.5×ATR) + initial fixed 1:3 target.

Diagnostic only — no portfolio constraints (each entry = fixed $10k risk).
"""
from __future__ import annotations
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from strategy_b1 import (
    B1_CFG, compute_btc_regime, ema, atr_series,
)


@dataclass
class _Pos:
    entry_time:   pd.Timestamp
    entry_price:  float
    qty:          float
    stop:         float
    initial_stop: float
    target:       float
    bars_held:    int = 0
    peak:         float = 0.0
    half_off:     bool = False
    R_high_water: float = 0.0


def simulate_symbol(
    df: pd.DataFrame,
    btc_regime: pd.DataFrame,
    cfg: dict = B1_CFG,
    use_regime_gate: bool = True,
    nav_per_trade: float = 10_000.0,
) -> tuple[pd.DataFrame, list[dict], pd.DataFrame]:
    """Walk-forward simulation on a single symbol.

    Returns (trades_df, missed_events_list, signal_series_df).
    """
    if len(df) < cfg["min_bars"]:
        return pd.DataFrame(), [], pd.DataFrame()

    e35 = ema(df["close"], 35)
    e200 = ema(df["close"], 200)
    atr14 = atr_series(df, 14)

    above = (e35 > e200).astype(int)
    cross_up_series = (above.diff() == 1).fillna(False)

    # Build a frame of indicators for chart
    sig = pd.DataFrame({
        "open":   df["open"],
        "high":   df["high"],
        "low":    df["low"],
        "close":  df["close"],
        "volume": df.get("volume", pd.Series(np.nan, index=df.index)),
        "ema35":  e35,
        "ema200": e200,
        "atr14":  atr14,
    })

    trades = []
    missed = []
    pos: _Pos | None = None
    last_entry_ts = None

    for i in range(len(df)):
        t = df.index[i]
        row = df.iloc[i]
        atr_v = float(atr14.iloc[i]) if pd.notna(atr14.iloc[i]) else None

        # ===== Manage open position =====
        if pos is not None and atr_v is not None:
            pos.bars_held += 1
            pos.peak = max(pos.peak, float(row["high"]))
            R_init = pos.entry_price - pos.initial_stop
            R_pnl_high = (pos.peak - pos.entry_price) / max(1e-9, R_init)
            pos.R_high_water = max(pos.R_high_water, R_pnl_high)

            exit_price = None
            reason = None

            # Initial hard stop
            if row["low"] <= pos.stop:
                exit_price = pos.stop
                reason = "stop"

            # Target hit (50% off, trail rest)
            elif (not pos.half_off) and row["high"] >= pos.target:
                half_qty = pos.qty * 0.5
                pnl_half = (pos.target - pos.entry_price) * half_qty
                cost_half = pos.target * half_qty * (cfg["fee_bps"] + cfg["slip_bps"]) / 1e4
                trades.append({
                    "entry_time":  pos.entry_time, "exit_time": t,
                    "entry":       pos.entry_price, "exit": pos.target,
                    "qty":         half_qty, "pnl": pnl_half - cost_half,
                    "ret_pct":     pos.target / pos.entry_price - 1,
                    "R":           (pos.target - pos.entry_price) / R_init if R_init > 0 else 0,
                    "reason":      "target50", "bars": pos.bars_held,
                })
                pos.qty *= 0.5
                pos.half_off = True
                pos.stop = max(pos.stop, pos.entry_price)  # move to BE

            # Trailing stop after target hit
            if exit_price is None and pos.half_off:
                trail_stop = pos.peak - 2.5 * atr_v
                if trail_stop > pos.stop:
                    pos.stop = trail_stop

            # Time stop (60 bars max)
            if exit_price is None and pos.bars_held >= 60 and not pos.half_off:
                exit_price = float(row["close"])
                reason = "time"

            if exit_price is not None:
                pnl = (exit_price - pos.entry_price) * pos.qty
                cost = exit_price * pos.qty * (cfg["fee_bps"] + cfg["slip_bps"]) / 1e4
                trades.append({
                    "entry_time": pos.entry_time, "exit_time": t,
                    "entry":      pos.entry_price, "exit": exit_price,
                    "qty":        pos.qty, "pnl": pnl - cost,
                    "ret_pct":    exit_price / pos.entry_price - 1,
                    "R":          (exit_price - pos.entry_price) / R_init if R_init > 0 else 0,
                    "reason":     reason, "bars": pos.bars_held,
                })
                last_entry_ts = t
                pos = None

        # ===== New entry =====
        if pos is None and atr_v is not None and bool(cross_up_series.iloc[i]):
            # Check BTC regime at this time
            regime_ok = True
            btc_active = None
            if use_regime_gate and t >= btc_regime.index[0]:
                try:
                    btc_active = bool(btc_regime["regime_active"].asof(t))
                    regime_ok = btc_active
                except (KeyError, ValueError):
                    regime_ok = False
                    btc_active = False

            if not regime_ok:
                missed.append({
                    "time":  t, "price": float(row["close"]),
                    "reason": "regime_off",
                    "btc_active": btc_active,
                })
                continue

            # Cooldown after exit
            if last_entry_ts is not None:
                if (t - last_entry_ts).total_seconds() < 5 * 4 * 3600:
                    continue

            # Open on next bar
            if i + 1 < len(df):
                next_row = df.iloc[i + 1]
                entry_price = float(next_row["open"])
                stop = entry_price - cfg["atr_stop_mult"] * atr_v
                target = entry_price + (entry_price - stop) * cfg["target_R"]
                if entry_price > stop and entry_price > 0:
                    risk_per_unit = entry_price - stop
                    qty = nav_per_trade / risk_per_unit
                    pos = _Pos(
                        entry_time=df.index[i + 1], entry_price=entry_price,
                        qty=qty, stop=stop, initial_stop=stop, target=target,
                        peak=float(next_row["high"]),
                    )

    # Close any final open position
    if pos is not None:
        last = df.iloc[-1]
        exit_price = float(last["close"])
        R_init = pos.entry_price - pos.initial_stop
        pnl = (exit_price - pos.entry_price) * pos.qty
        cost = exit_price * pos.qty * (cfg["fee_bps"] + cfg["slip_bps"]) / 1e4
        trades.append({
            "entry_time": pos.entry_time, "exit_time": df.index[-1],
            "entry":      pos.entry_price, "exit": exit_price,
            "qty":        pos.qty, "pnl": pnl - cost,
            "ret_pct":    exit_price / pos.entry_price - 1,
            "R":          (exit_price - pos.entry_price) / R_init if R_init > 0 else 0,
            "reason":     "open_at_end", "bars": pos.bars_held,
        })

    return pd.DataFrame(trades), missed, sig


def render_chart(sig: pd.DataFrame, trades: pd.DataFrame, missed: list[dict],
                 symbol_display: str, show_missed: bool = True):
    """Render Plotly chart with entries, exits, missed signals."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.78, 0.22], vertical_spacing=0.03,
        subplot_titles=(f"{symbol_display} — Trend strategy backtest (4H)", "Volume"),
    )

    fig.add_trace(go.Candlestick(
        x=sig.index, open=sig["open"], high=sig["high"],
        low=sig["low"], close=sig["close"],
        increasing_line_color="#0F6E56",
        decreasing_line_color="#A32D2D",
        showlegend=False, name="Price",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(x=sig.index, y=sig["ema35"],
                              line=dict(color="#FFFFFF", width=1.3),
                              name="EMA 35"), row=1, col=1)
    fig.add_trace(go.Scatter(x=sig.index, y=sig["ema200"],
                              line=dict(color="#378ADD", width=1.6),
                              name="EMA 200"), row=1, col=1)

    if not trades.empty:
        # Unique entries
        unique_entries = trades[["entry_time", "entry"]].drop_duplicates(subset=["entry_time"])
        entry_y = unique_entries["entry_time"].map(
            lambda t: sig.loc[t, "low"] * 0.97 if t in sig.index else np.nan
        )
        fig.add_trace(go.Scatter(
            x=unique_entries["entry_time"], y=entry_y,
            mode="markers",
            marker=dict(symbol="triangle-up", size=13, color="#0F6E56",
                        line=dict(width=1.5, color="white")),
            name="Entry",
            hovertext=[f"Entry ${e:.4g}<br>{t.strftime('%Y-%m-%d %H:%M')}"
                       for t, e in zip(unique_entries["entry_time"], unique_entries["entry"])],
            hoverinfo="text",
        ), row=1, col=1)

        # Exits by reason
        reason_colors = {
            "target50": "#0F6E56", "stop": "#A32D2D",
            "time":     "#888780", "open_at_end": "#BA7517",
        }
        for reason, color in reason_colors.items():
            mask = trades["reason"] == reason
            if not mask.any(): continue
            xs = trades.loc[mask, "exit_time"]
            ys = xs.map(lambda t: sig.loc[t, "high"] * 1.02 if t in sig.index else np.nan)
            hover = [f"Exit ${e:.4g}<br>R = {r:+.2f}<br>{reason}"
                     for e, r in zip(trades.loc[mask, "exit"], trades.loc[mask, "R"])]
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="markers",
                marker=dict(symbol="triangle-down", size=11, color=color,
                            line=dict(width=1, color="white")),
                name=f"Exit: {reason}",
                hovertext=hover, hoverinfo="text",
            ), row=1, col=1)

    if show_missed and missed:
        m_xs = [m["time"] for m in missed]
        m_ys = [sig.loc[m["time"], "low"] * 0.94 if m["time"] in sig.index else np.nan
                for m in missed]
        fig.add_trace(go.Scatter(
            x=m_xs, y=m_ys, mode="markers",
            marker=dict(symbol="line-ns-open", size=12, color="#BA7517",
                        line=dict(width=2)),
            name="Cross (regime off)",
            hovertext=[f"Cross @ ${m['price']:.4g}<br>BTC regime off" for m in missed],
            hoverinfo="text",
        ), row=1, col=1)

    if "volume" in sig.columns and sig["volume"].notna().any():
        vol_colors = ["#0F6E56" if c >= o else "#A32D2D"
                      for c, o in zip(sig["close"], sig["open"])]
        fig.add_trace(go.Bar(
            x=sig.index, y=sig["volume"],
            marker_color=vol_colors, showlegend=False, opacity=0.7,
        ), row=2, col=1)

    fig.update_layout(
        height=680, xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font=dict(size=10)),
        margin=dict(l=10, r=10, t=40, b=10),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Price (USDT)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1, showgrid=False)
    fig.update_xaxes(showgrid=True, gridcolor="rgba(120,120,120,0.1)")

    st.plotly_chart(fig, use_container_width=True)


@st.cache_data(ttl=600, show_spinner=False)
def _cached_simulate(symbol: str, data_dir: str, use_regime: bool):
    """Cached wrapper for the full B1 backtest pipeline."""
    path = os.path.join(data_dir, f"{symbol}_4h.parquet")
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    if len(df) < B1_CFG["min_bars"]:
        return None

    btc_path = os.path.join(data_dir, "BTCUSDT_4h.parquet")
    if not os.path.exists(btc_path):
        return None
    btc_4h = pd.read_parquet(btc_path)
    btc_regime = compute_btc_regime(btc_4h)

    trades, missed, sig = simulate_symbol(df, btc_regime, use_regime_gate=use_regime)
    return sig, trades, missed


def render_symbol_backtest(data_dir: str, universe: dict):
    """Main render function — to be called from app.py."""
    st.markdown(
        "Backtest the Trend strategy (EMA crossover + BTC near 90d local high) on a single symbol. "
        "**Diagnostic mode** — fixed $10k risk per trade, no portfolio constraints."
    )

    display_options = sorted(universe.keys(), key=lambda k: universe[k][0])
    if not display_options:
        st.info("Universe is empty.")
        return

    col1, col2 = st.columns([3, 1])
    with col1:
        sym = st.selectbox(
            "Symbol",
            options=display_options,
            format_func=lambda s: f"{universe[s][0]} ({s}) — {universe[s][1]}",
            key="b1_bt_picker",
        )
    with col2:
        use_regime = st.checkbox(
            "Apply BTC regime gate",
            value=True,
            help="If unchecked, shows all crossovers regardless of BTC state. "
                 "Useful to see how many setups are filtered out.",
        )

    with st.spinner(f"Running Trend backtest on {sym}..."):
        result = _cached_simulate(sym, data_dir, use_regime)

    if result is None:
        st.warning(
            f"Insufficient data for {sym}. Strategy needs ≥{B1_CFG['min_bars']} 4H bars."
        )
        return

    sig, trades, missed = result

    if trades.empty:
        st.info(
            f"No Trend trades on {sym} over its full history. "
            "Either no crossovers happened, or all were filtered by BTC regime."
        )
        render_chart(sig, trades, missed, universe[sym][0], show_missed=bool(missed))
        return

    # Stats
    wins = trades[trades["pnl"] > 0]
    losses = trades[trades["pnl"] <= 0]
    n = len(trades)
    wr = len(wins) / n if n else 0
    avg_R_win = wins["R"].mean() if len(wins) else 0
    avg_R_loss = losses["R"].mean() if len(losses) else 0
    pf = (abs(wins["pnl"].sum() / losses["pnl"].sum())
          if (len(losses) and losses["pnl"].sum() != 0) else float("inf"))
    total_pnl = trades["pnl"].sum()
    target_hit = (trades["reason"] == "target50").mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total trades", n)
    c2.metric("Win rate", f"{wr:.1%}")
    c3.metric("Profit factor", f"{pf:.2f}" if pf != float("inf") else "∞")
    c4.metric("Net P&L ($10k risk)", f"${total_pnl:,.0f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Avg win", f"+{avg_R_win:.2f}R")
    c6.metric("Avg loss", f"{avg_R_loss:.2f}R")
    c7.metric("Target hit rate", f"{target_hit:.0%}")
    c8.metric("Filtered by regime", len(missed))

    render_chart(sig, trades, missed, universe[sym][0], show_missed=bool(missed))

    st.markdown("### Trades")
    tbl = trades.copy()
    tbl["When entered"] = tbl["entry_time"].dt.strftime("%Y-%m-%d %H:%M")
    tbl["When exited"]  = tbl["exit_time"].dt.strftime("%Y-%m-%d %H:%M")
    tbl["Entry"]   = tbl["entry"].map(lambda x: f"${x:.4g}")
    tbl["Exit"]    = tbl["exit"].map(lambda x: f"${x:.4g}")
    tbl["Bars"]    = tbl["bars"].astype(int)
    tbl["R"]       = tbl["R"].map(lambda x: f"{x:+.2f}")
    tbl["P&L"]     = tbl["pnl"].map(lambda x: f"${x:+,.0f}")
    tbl["Reason"]  = tbl["reason"]
    st.dataframe(
        tbl[["When entered", "When exited", "Entry", "Exit", "Bars", "R", "P&L", "Reason"]],
        hide_index=True, use_container_width=True,
    )

    st.markdown("### Exit reasons")
    reason_summary = trades.groupby("reason").agg(
        n=("pnl", "count"), total_pnl=("pnl", "sum"), avg_R=("R", "mean"),
    ).round(2)
    reason_summary.columns = ["Count", "Total P&L", "Avg R"]
    st.dataframe(reason_summary, use_container_width=True)
