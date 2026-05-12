"""
Per-symbol historical backtest.

Runs the full VCP-EMA-Stack v1.2 entry/exit logic on a single symbol
across its entire price history. Ignores portfolio constraints
(max_pos, per_name_cap, portfolio heat) — this is diagnostic per-symbol,
not a portfolio simulation. The intent is to show "how does the strategy
behave on THIS symbol historically".

Outputs:
- DataFrame of trades with entry/exit/pnl/R/reason
- Plotly chart with EMA overlay, entry/exit markers, missed setups
- Summary stats specific to this symbol
"""
from __future__ import annotations
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from engine import CFG, compute_indicators, ema, atr


# ============================================================
# Signal computation (mirrors backtest_v3 logic)
# ============================================================
def make_signals_for_backtest(df: pd.DataFrame) -> pd.DataFrame:
    """Return full indicator + signal dataframe for the symbol."""
    s = compute_indicators(df).copy()

    # Trend + slope
    stack = (s["ema35"] > s["ema100"]) & (s["ema100"] > s["ema200"])
    slope = (s["ema200"] > s["ema200"].shift(20)) & (s["ema35"] > s["ema35"].shift(20))
    s["S1"] = (stack & slope).astype(int)
    s["S2"] = (s["streak"] >= CFG["body_count"]).astype(int)
    s["S3"] = (s["atr_ratio"] < CFG["atr_ratio_max"]).astype(int)

    # Bases
    win30 = CFG["base_days"] * 6
    win60 = 360
    def bf(arr):
        cur = arr[-1]
        if cur == 0: return 0.0
        return float(np.mean((arr >= cur * 0.85) & (arr <= cur * 1.15)))
    s["base_30d"] = s["close"].rolling(win30).apply(bf, raw=True)
    s["S4"] = (s["base_30d"] >= CFG["base_in_range_min"]).astype(int)
    s["base_60d"] = s["close"].rolling(win60).apply(bf, raw=True)
    s["F1"] = (s["base_60d"] >= CFG["base_60d_min"]).astype(int)

    s["ret_90d"] = s["close"] / s["close"].shift(540) - 1
    s["F2"] = ((s["ret_90d"] >= CFG["ret_90d_min"]) &
               (s["ret_90d"] <= CFG["ret_90d_max"])).astype(int)

    if "volume" in df.columns:
        vm = df["volume"].rolling(360).median()
        v3 = df["volume"].rolling(3).mean()
        s["vol_spike"] = v3 / vm
        s["F5"] = (s["vol_spike"] >= CFG["vol_spike_min"]).astype(int)
    else:
        s["F5"] = 0
        s["vol_spike"] = np.nan

    s["stop_atr"] = s["close"] - CFG["atr_stop_mult"] * s["atr14"]
    s["stop_lvl"] = np.maximum(s["stop_atr"], s["ema100"])
    s["target_lvl"] = np.maximum(s["ema200"], s["close"] * CFG["target_pct_floor"])

    R = (s["target_lvl"] - s["close"]) / np.maximum(1e-9, s["close"] - s["stop_lvl"])
    s["R_target"] = R

    s["score_core"] = s["S1"] + s["S2"] + s["S3"] + s["S4"]
    s["score_smart"] = s["F1"] + s["F2"] + s["F5"]
    s["signal"] = (
        (s["S1"] == 1) & (s["S2"] == 1) & (s["score_core"] >= 3) &
        (s["F1"] == 1) & (s["F2"] == 1) & (s["F5"] == 1) &
        (s["stop_lvl"] < s["close"]) &
        (R >= CFG["min_R"]) & (R <= CFG["max_R"])
    )
    return s


# ============================================================
# Single-symbol trade simulator
# ============================================================
@dataclass
class _Pos:
    entry_time: pd.Timestamp
    entry_price: float
    qty: float
    stop: float
    initial_stop: float
    target: float
    initial_atr: float
    half_off: bool = False
    bars_held: int = 0
    peak: float = 0.0


def simulate_symbol(
    signals: pd.DataFrame,
    btc_regime: pd.Series,
    nav_per_trade: float = 10_000.0,
    use_regime_gate: bool = True,
    fee_bps: float = 4.0,
    slip_bps: float = 10.0,
) -> tuple[pd.DataFrame, list[dict]]:
    """Walk-forward simulation. Returns (trades_df, missed_setups_list).

    nav_per_trade — fixed dollars allocated each trade (no portfolio constraints).
    use_regime_gate — if False, ignore BTC regime filter (see all possible setups).
    """
    trades = []
    missed = []
    pos: _Pos | None = None

    for i, (t, row) in enumerate(signals.iterrows()):
        # 1. Update open position
        if pos is not None:
            pos.bars_held += 1
            pos.peak = max(pos.peak, row["high"])
            exit_price = None
            reason = None

            if row["low"] <= pos.stop:
                exit_price = pos.stop
                reason = "stop"
            elif (not pos.half_off) and row["high"] >= pos.target:
                # Take half off at target
                half_qty = pos.qty * 0.5
                pnl_half = (pos.target - pos.entry_price) * half_qty
                cost_half = pos.target * half_qty * (fee_bps + slip_bps) / 1e4
                R_init = pos.entry_price - pos.initial_stop
                trades.append({
                    "entry_time": pos.entry_time, "exit_time": t,
                    "entry": pos.entry_price, "exit": pos.target,
                    "qty": half_qty, "pnl": pnl_half - cost_half,
                    "reason": "target50", "bars": pos.bars_held,
                    "R": (pos.target - pos.entry_price) / R_init if R_init > 0 else 0,
                })
                pos.qty *= 0.5
                pos.half_off = True
                pos.stop = max(pos.stop, pos.entry_price)
            elif pos.bars_held >= CFG.get("time_stop_bars", 60) and not pos.half_off:
                exit_price = row["close"]
                reason = "time"

            # Trail update if still in position
            if exit_price is None:
                R_init = pos.entry_price - pos.initial_stop
                R_pnl = (row["close"] - pos.entry_price) / max(1e-9, R_init)
                if pos.half_off or R_pnl >= 2.0:
                    new_stop = pos.peak - 3.0 * row["atr14"]
                    if new_stop > pos.stop:
                        pos.stop = new_stop

            if exit_price is not None:
                pnl = (exit_price - pos.entry_price) * pos.qty
                cost = exit_price * pos.qty * (fee_bps + slip_bps) / 1e4
                R_init = pos.entry_price - pos.initial_stop
                trades.append({
                    "entry_time": pos.entry_time, "exit_time": t,
                    "entry": pos.entry_price, "exit": exit_price,
                    "qty": pos.qty, "pnl": pnl - cost,
                    "reason": reason, "bars": pos.bars_held,
                    "R": (exit_price - pos.entry_price) / R_init if R_init > 0 else 0,
                })
                pos = None

        # 2. Check for new entry on next bar
        if pos is None and bool(row["signal"]):
            # Check regime if required
            regime_ok = True
            if use_regime_gate and btc_regime is not None:
                rv = btc_regime.asof(t)
                regime_ok = bool(rv) if pd.notna(rv) else False
            if not regime_ok:
                missed.append({
                    "time": t, "price": float(row["close"]),
                    "stop": float(row["stop_lvl"]), "target": float(row["target_lvl"]),
                    "R": float(row["R_target"]), "reason": "regime_off",
                })
                continue

            # Open on next bar open
            try:
                next_idx = signals.index[i + 1]
            except IndexError:
                continue
            next_row = signals.loc[next_idx]
            entry_price = float(next_row["open"])
            stop = float(row["stop_lvl"])
            target = float(row["target_lvl"])
            if entry_price <= stop or target <= entry_price:
                continue

            risk_per_unit = entry_price - stop
            qty = nav_per_trade / risk_per_unit  # fixed-dollar risk
            cost = entry_price * qty * (fee_bps + slip_bps) / 1e4

            pos = _Pos(
                entry_time=next_idx, entry_price=entry_price, qty=qty,
                stop=stop, initial_stop=stop, target=target,
                initial_atr=float(row["atr14"]), peak=float(next_row["high"]),
            )

    return pd.DataFrame(trades), missed


# ============================================================
# Plotly visualization
# ============================================================
def render_backtest_chart(
    signals: pd.DataFrame, trades: pd.DataFrame, missed: list[dict],
    symbol_display: str, show_missed: bool = True,
):
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.78, 0.22], vertical_spacing=0.03,
        subplot_titles=(f"{symbol_display} — historical backtest (4H)", "Volume"),
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=signals.index,
        open=signals["open"], high=signals["high"],
        low=signals["low"], close=signals["close"],
        increasing_line_color="#0F6E56",
        decreasing_line_color="#A32D2D",
        showlegend=False, name="Price",
    ), row=1, col=1)

    # EMA overlays
    fig.add_trace(go.Scatter(
        x=signals.index, y=signals["ema35"],
        line=dict(color="#378ADD", width=1.2), name="EMA 35",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=signals.index, y=signals["ema100"],
        line=dict(color="#EF9F27", width=1.2), name="EMA 100",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=signals.index, y=signals["ema200"],
        line=dict(color="#E24B4A", width=1.6), name="EMA 200",
    ), row=1, col=1)

    # Entry markers (green up-triangle below the bar low)
    if not trades.empty:
        entries = trades[["entry_time", "entry", "reason"]].copy()
        # First half of trade has entry, second half (target50) shares entry_time
        # Take unique entries
        unique_entries = entries.drop_duplicates(subset=["entry_time"])
        entry_y = unique_entries["entry_time"].map(
            lambda t: signals.loc[t, "low"] * 0.985 if t in signals.index else np.nan
        )
        fig.add_trace(go.Scatter(
            x=unique_entries["entry_time"], y=entry_y,
            mode="markers",
            marker=dict(symbol="triangle-up", size=12, color="#0F6E56",
                       line=dict(width=1.5, color="white")),
            name="Entry",
            hovertext=[f"Entry @ ${e:.4g}<br>{t.strftime('%Y-%m-%d %H:%M')}"
                       for t, e in zip(unique_entries["entry_time"], unique_entries["entry"])],
            hoverinfo="text",
        ), row=1, col=1)

        # Exit markers — color by reason
        reason_colors = {"target50": "#0F6E56", "stop": "#A32D2D",
                         "time": "#888780", "trail": "#BA7517"}
        for reason, color in reason_colors.items():
            mask = trades["reason"] == reason
            if not mask.any():
                continue
            xs = trades.loc[mask, "exit_time"]
            ys = trades.loc[mask, "exit_time"].map(
                lambda t: signals.loc[t, "high"] * 1.015 if t in signals.index else np.nan
            )
            label_map = {"target50": "Exit: target", "stop": "Exit: stop",
                         "time": "Exit: time", "trail": "Exit: trail"}
            fig.add_trace(go.Scatter(
                x=xs, y=ys,
                mode="markers",
                marker=dict(symbol="triangle-down", size=10, color=color,
                           line=dict(width=1, color="white")),
                name=label_map[reason],
                hovertext=[f"Exit @ ${e:.4g}<br>R = {r:+.2f}<br>{reason}"
                           for e, r in zip(trades.loc[mask, "exit"], trades.loc[mask, "R"])],
                hoverinfo="text",
            ), row=1, col=1)

    # Missed setups — yellow dashes
    if show_missed and missed:
        m_xs = [m["time"] for m in missed]
        m_ys = [signals.loc[m["time"], "low"] * 0.96 if m["time"] in signals.index else np.nan
                for m in missed]
        fig.add_trace(go.Scatter(
            x=m_xs, y=m_ys,
            mode="markers",
            marker=dict(symbol="line-ns-open", size=14, color="#BA7517",
                       line=dict(width=2)),
            name="Setup (regime off)",
            hovertext=[f"Setup pending<br>${m['price']:.4g}<br>R = {m['R']:.1f}"
                       for m in missed],
            hoverinfo="text",
        ), row=1, col=1)

    # Volume
    if "volume" in signals.columns:
        vol_colors = ["#0F6E56" if c >= o else "#A32D2D"
                      for c, o in zip(signals["close"], signals["open"])]
        fig.add_trace(go.Bar(
            x=signals.index, y=signals["volume"],
            marker_color=vol_colors, name="Volume", showlegend=False,
            opacity=0.7,
        ), row=2, col=1)

    fig.update_layout(
        height=680,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font=dict(size=11)),
        margin=dict(l=10, r=10, t=40, b=10),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Price (USDT)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1, showgrid=False)
    fig.update_xaxes(showgrid=True, gridcolor="rgba(120,120,120,0.1)")

    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# Main render — to be called from app.py
# ============================================================
@st.cache_data(ttl=600, show_spinner=False)
def _cached_simulate(symbol: str, data_dir: str, use_regime: bool):
    """Cached wrapper for the full backtest pipeline."""
    path = os.path.join(data_dir, f"{symbol}_4h.parquet")
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    if len(df) < CFG["min_bars"]:
        return None

    btc_path = os.path.join(data_dir, "BTCUSDT_4h.parquet")
    btc_regime = None
    if os.path.exists(btc_path):
        btc = pd.read_parquet(btc_path)
        btc_ema = ema(btc["close"], 200)
        btc_regime = btc["close"] > btc_ema * (1 + CFG["btc_above_ema_min"])

    signals = make_signals_for_backtest(df)
    trades, missed = simulate_symbol(
        signals, btc_regime, use_regime_gate=use_regime,
    )
    return signals, trades, missed


def render_symbol_backtest(data_dir: str, universe: dict):
    st.markdown(
        "Run the full v1.2 strategy on a single symbol across its entire history. "
        "**Diagnostic mode** — ignores portfolio constraints (max 5 positions, "
        "per-name cap, portfolio heat). Each trade gets fixed $10k risk regardless. "
        "Real portfolio results would be lower because of slot competition."
    )

    # Symbol picker
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
            key="symbt_picker",
        )
    with col2:
        use_regime = st.checkbox(
            "Apply BTC regime gate",
            value=True,
            help="If unchecked, shows all setups regardless of BTC regime. "
                 "Useful to see how many setups were filtered out.",
        )

    with st.spinner(f"Running backtest on {sym}..."):
        result = _cached_simulate(sym, data_dir, use_regime)

    if result is None:
        st.warning(
            f"Insufficient data for {sym}. The strategy needs at least "
            f"{CFG['min_bars']} 4H bars (~3 months) of price history."
        )
        return

    signals, trades, missed = result

    if trades.empty:
        st.info(
            f"No trades fired on {sym} over its full history "
            f"({len(signals)} bars from "
            f"{signals.index[0].strftime('%Y-%m-%d')} to "
            f"{signals.index[-1].strftime('%Y-%m-%d')}). "
            "This is a valid result — strategy is selective on purpose."
        )
        # Still show chart so user can see why
        render_backtest_chart(signals, trades, missed, universe[sym][0],
                              show_missed=bool(missed))
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
    expectancy_R = trades["R"].mean()
    total_pnl = trades["pnl"].sum()
    target_hit = (trades["reason"] == "target50").mean()

    # Top stat row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total trades", n)
    c2.metric("Win rate", f"{wr:.1%}")
    c3.metric("Profit factor", f"{pf:.2f}" if pf != float("inf") else "∞")
    c4.metric("Net P&L (10k risk/trade)", f"${total_pnl:,.0f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Avg win", f"+{avg_R_win:.2f}R")
    c6.metric("Avg loss", f"{avg_R_loss:.2f}R")
    c7.metric("Expectancy", f"{expectancy_R:+.2f}R")
    c8.metric("Target hit rate", f"{target_hit:.0%}")

    if missed:
        st.caption(
            f"📍 {len(missed)} setup(s) appeared but were filtered out by the "
            "BTC regime gate (F3/F4). Yellow ticks on the chart."
        )

    # Chart
    render_backtest_chart(signals, trades, missed, universe[sym][0],
                          show_missed=bool(missed))

    # Trades table
    st.markdown("### Trades")
    if not trades.empty:
        # Group target50 + final exits by entry_time
        tbl = trades.copy()
        tbl["When entered"] = tbl["entry_time"].dt.strftime("%Y-%m-%d %H:%M")
        tbl["When exited"]  = tbl["exit_time"].dt.strftime("%Y-%m-%d %H:%M")
        tbl["Entry"]   = tbl["entry"].map(lambda x: f"${x:.4g}")
        tbl["Exit"]    = tbl["exit"].map(lambda x: f"${x:.4g}")
        tbl["Bars"]    = tbl["bars"].astype(int)
        tbl["R"]       = tbl["R"].map(lambda x: f"{x:+.2f}")
        tbl["P&L"]     = tbl["pnl"].map(lambda x: f"${x:+,.0f}")
        tbl["Reason"]  = tbl["reason"]
        display_cols = ["When entered", "When exited", "Entry", "Exit",
                        "Bars", "R", "P&L", "Reason"]
        st.dataframe(tbl[display_cols], hide_index=True, use_container_width=True)

    # Exit-reason breakdown
    st.markdown("### Exit reasons")
    reason_summary = trades.groupby("reason").agg(
        n=("pnl", "count"),
        total_pnl=("pnl", "sum"),
        avg_R=("R", "mean"),
    ).round(2)
    reason_summary.columns = ["Count", "Total P&L", "Avg R"]
    st.dataframe(reason_summary, use_container_width=True)
