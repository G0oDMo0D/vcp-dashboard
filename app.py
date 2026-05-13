"""
VCP-EMA-Stack Dashboard

Web UI for monitoring strategy signals across the configured universe.
Run with: streamlit run app.py
"""
from __future__ import annotations
import os, sys
from datetime import datetime

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add this directory to path so engine.py is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine import scan_all, compute_indicators, CFG
from universe import UNIVERSE, SECTORS
from heatmap import render_sector_heatmap
from history import save_snapshot, render_history
from symbol_backtest import render_symbol_backtest

# Strategy B1 (EMA crossover + BTC local high)
import strategy_b1 as b1
from strategy_b1_backtest import render_symbol_backtest as render_b1_symbol_backtest

# ============================================================
# Configuration
# ============================================================
DATA_DIR = os.environ.get("VCP_DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "clean"))
RESULTS_DIR = os.environ.get("VCP_RESULTS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
SNAPSHOTS_DIR = os.environ.get("VCP_SNAPSHOTS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "snapshots"))

st.set_page_config(
    page_title="VCP-EMA-Stack Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Styling
# ============================================================
st.markdown("""
<style>
    .main .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1400px; }
    div[data-testid="stMetricValue"] { font-size: 22px; }
    div[data-testid="stMetricLabel"] { font-size: 12px; color: rgba(255,255,255,0.6); }
    .tier-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 13px;
        margin-right: 8px;
    }
    .tier-A { background: rgba(15,110,86,0.2); color: #0F6E56; border: 1px solid #0F6E56; }
    .tier-B { background: rgba(186,117,23,0.15); color: #BA7517; border: 1px solid #BA7517; }
    .tier-C { background: rgba(120,120,120,0.15); color: #888; border: 1px solid #888; }
    .stDataFrame { font-size: 13px; }
    [data-testid="stSidebar"] { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Data loading (cached)
# ============================================================
@st.cache_data(ttl=300, show_spinner=False)
def run_scan(_cache_key: str = ""):
    """Cached scan. _cache_key allows manual invalidation by changing the key."""
    symbols = list(UNIVERSE.keys())
    df, regime = scan_all(DATA_DIR, symbols)
    return df, regime


@st.cache_data(ttl=600, show_spinner=False)
def load_symbol_data(symbol: str):
    path = f"{DATA_DIR}/{symbol}_4h.parquet"
    if not os.path.exists(path):
        return None
    return pd.read_parquet(path)


@st.cache_data(ttl=3600, show_spinner=False)
def load_backtest_equity():
    for fname in ["equity_v3.csv", "equity_v2.csv", "equity.csv"]:
        path = f"{RESULTS_DIR}/{fname}"
        if os.path.exists(path):
            eq = pd.read_csv(path, parse_dates=[0], index_col=0)
            eq.columns = ["equity"]
            return eq, fname
    return None, None


@st.cache_data(ttl=3600, show_spinner=False)
def load_backtest_trades():
    for fname in ["trades_v3.csv", "trades_v2.csv", "trades.csv"]:
        path = f"{RESULTS_DIR}/{fname}"
        if os.path.exists(path):
            return pd.read_csv(path, parse_dates=["entry_time", "exit_time"]), fname
    return None, None


# ============================================================
# Header / Regime
# ============================================================
def render_title(regime: dict):
    """Top-of-page title + refresh button. Always visible regardless of tab."""
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("📈 VCP-EMA-Stack Dashboard")
        st.caption(f"v1.2 spec · BTC 4H bar as of {regime['as_of'].strftime('%Y-%m-%d %H:%M UTC')}")
    with col2:
        st.write("")
        if st.button("🔄 Refresh scan", use_container_width=True):
            run_scan.clear()
            load_symbol_data.clear()
            st.rerun()


def render_vcp_regime(regime: dict):
    """VCP strategy regime banner + 4 metric columns. Placed inside VCP tabs only."""
    active = regime["active"]
    if active:
        st.success("✅ REVERSAL STRATEGY ACTIVE — both regime gates satisfied. Tier-A signals are tradable.")
    else:
        reasons = []
        if not regime["F3_active"]:
            reasons.append(f"F3 fail (BTC {regime['btc_above_pct']:+.2%} vs ≥+5%)")
        if not regime["F4_active"]:
            reasons.append(f"F4 fail (BTC 30d {regime['btc_30d_ret']:+.2%} vs ≥+5%)")
        st.warning(f"⏸ REVERSAL STRATEGY PAUSED — {' AND '.join(reasons)}. Tier-A symbols stay on watchlist.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("BTC price", f"${regime['btc_price']:,.0f}")
    col2.metric("BTC EMA200 (4H)", f"${regime['btc_ema200_4h']:,.0f}")
    col3.metric(
        "BTC vs EMA200 (F3)",
        f"{regime['btc_above_pct']:+.2%}",
        delta=f"{(regime['btc_above_pct']-0.05)*100:+.1f}pp vs +5% gate",
        delta_color="normal",
    )
    col4.metric(
        "BTC 30-day return (F4)",
        f"{regime['btc_30d_ret']:+.2%}",
        delta=f"{(regime['btc_30d_ret']-0.05)*100:+.1f}pp vs +5% gate",
        delta_color="normal",
    )


# Keep render_header as alias for backwards compat — calls both
def render_header(regime: dict):
    render_title(regime)
    render_vcp_regime(regime)


# ============================================================
# Tier summary cards
# ============================================================
def render_tier_summary(scan_df: pd.DataFrame, regime_active: bool):
    st.subheader("Signal tiers")
    tier_a   = scan_df[(scan_df["symbol_pass"] == 7) & (scan_df["regime_pass"] == 2) & (scan_df["R_ok"])]
    tier_a_p = scan_df[(scan_df["symbol_pass"] == 7) & (scan_df["regime_pass"] < 2)]
    tier_b   = scan_df[scan_df["symbol_pass"] == 6]
    tier_c   = scan_df[scan_df["symbol_pass"] == 5]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("**Tier A — Actionable**")
        st.metric("Pass 7/7 + regime + R ok", len(tier_a))
        st.caption("Ready to enter on next bar")
    with c2:
        st.markdown("**A pending regime**")
        st.metric("Pass 7/7, waiting BTC", len(tier_a_p))
        st.caption("Setup ready, regime gate off")
    with c3:
        st.markdown("**Tier B — Watching**")
        st.metric("Pass 6/7", len(tier_b))
        st.caption("Missing one condition")
    with c4:
        st.markdown("**Tier C — Tracking**")
        st.metric("Pass 5/7", len(tier_c))
        st.caption("Forming")


# ============================================================
# Watchlist table
# ============================================================
def render_watchlist(scan_df: pd.DataFrame, sector_filter: list, tier_filter: list, min_score: int):
    if scan_df.empty:
        st.info("No symbols evaluated. Check data directory.")
        return None

    df = scan_df.copy()
    df["display"] = df["symbol"].map(lambda s: UNIVERSE.get(s, (s, "?"))[0])
    df["sector"]  = df["symbol"].map(lambda s: UNIVERSE.get(s, (s, "?"))[1])

    if sector_filter:
        df = df[df["sector"].isin(sector_filter)]
    if tier_filter:
        df = df[df["tier"].isin(tier_filter)]
    df = df[df["symbol_pass"] >= min_score]

    if df.empty:
        st.info("No symbols match the current filters.")
        return None

    # Compact display columns
    def pass_glyph(b: bool) -> str:
        return "✓" if b else "·"

    df_view = pd.DataFrame({
        "Symbol":   df["display"],
        "Sector":   df["sector"],
        "Tier":     df["tier"].map({"A":"🟢 A", "A_pending_regime":"🟡 A*", "B":"🟠 B", "C":"⚪ C", "D":"○ D"}),
        "Pass":     df.apply(lambda r: f"{r['symbol_pass']}/7", axis=1),
        "Price":    df["price"].map(lambda x: f"${x:.4g}"),
        "C1":       df["C1_stack"].map(pass_glyph),
        "C3":       df.apply(lambda r: f"{pass_glyph(r['C3_pass'])} ({int(r['C3_body_streak'])})", axis=1),
        "C4 ATRr":  df.apply(lambda r: f"{pass_glyph(r['C4_pass'])} {r['C4_atr_ratio']:.2f}", axis=1),
        "C5 30d":   df.apply(lambda r: f"{pass_glyph(r['C5_pass'])} {r['C5_base_30d']:.2f}", axis=1),
        "F1 60d":   df.apply(lambda r: f"{pass_glyph(r['F1_pass'])} {r['F1_base_60d']:.2f}" if not pd.isna(r['F1_base_60d']) else "·", axis=1),
        "F2 ret90": df.apply(lambda r: f"{pass_glyph(r['F2_pass'])} {r['F2_ret_90d']:+.1%}" if not pd.isna(r['F2_ret_90d']) else "·", axis=1),
        "F5 vol":   df.apply(lambda r: f"{pass_glyph(r['F5_pass'])} {r['F5_vol_spike']:.1f}x" if not pd.isna(r['F5_vol_spike']) else "·", axis=1),
        "R":        df["target_R"].map(lambda x: f"{x:.1f}"),
        "Stop":     df["stop"].map(lambda x: f"${x:.4g}"),
        "Target":   df["target"].map(lambda x: f"${x:.4g}"),
        "_sym":     df["symbol"],
        "_pass":    df["symbol_pass"],
        "_base60":  df["F1_base_60d"].fillna(-1),
    })
    df_view = df_view.sort_values(["_pass", "_base60"], ascending=[False, False])

    st.dataframe(
        df_view.drop(columns=["_sym", "_pass", "_base60"]),
        hide_index=True,
        use_container_width=True,
        height=min(600, 50 + 35 * len(df_view)),
    )
    return df_view


# ============================================================
# Per-symbol drill-down with Plotly chart
# ============================================================
def render_symbol_detail(symbol: str, scan_row: pd.Series):
    df = load_symbol_data(symbol)
    if df is None or len(df) < CFG["min_bars"]:
        st.warning(f"Insufficient data for {symbol}")
        return

    # Compute indicators on last N bars (chart window)
    chart_bars = 360  # last 60 days at 4H
    plot_df = compute_indicators(df).iloc[-chart_bars:].copy()

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.04,
        subplot_titles=(f"{UNIVERSE.get(symbol, (symbol,'?'))[0]} — 4H with EMA stack", "Volume"),
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=plot_df.index,
        open=plot_df["open"], high=plot_df["high"],
        low=plot_df["low"], close=plot_df["close"],
        name="Price",
        increasing_line_color="#0F6E56",
        decreasing_line_color="#A32D2D",
        showlegend=False,
    ), row=1, col=1)

    # EMA overlays
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["ema35"],
                             line=dict(color="#3B82F6", width=1.5), name="EMA 35"), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["ema100"],
                             line=dict(color="#F59E0B", width=1.5), name="EMA 100"), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["ema200"],
                             line=dict(color="#EF4444", width=2), name="EMA 200"), row=1, col=1)

    # Current stop / target lines
    fig.add_hline(y=scan_row["stop"], line_dash="dash", line_color="#A32D2D",
                  annotation_text=f"Stop ${scan_row['stop']:.4g}",
                  annotation_position="right", row=1, col=1)
    fig.add_hline(y=scan_row["target"], line_dash="dash", line_color="#0F6E56",
                  annotation_text=f"Target ${scan_row['target']:.4g}",
                  annotation_position="right", row=1, col=1)
    fig.add_hline(y=scan_row["price"], line_dash="dot", line_color="rgba(150,150,150,0.6)",
                  annotation_text=f"Last ${scan_row['price']:.4g}",
                  annotation_position="right", row=1, col=1)

    # Mark entry-eligible bars in the visible window (where signal would have fired)
    sig_idx = []
    sig_close = []
    last = plot_df.iloc[-1]
    # Simple retrospective signal marker — bars where the basic stack + streak + compression + base hold
    stack_hist = (plot_df["ema35"] > plot_df["ema100"]) & (plot_df["ema100"] > plot_df["ema200"])
    streak_hist = plot_df["streak"] >= CFG["body_count"]
    atr_hist = plot_df["atr_ratio"] < CFG["atr_ratio_max"]
    candidate = stack_hist & streak_hist & atr_hist
    for ts in plot_df.index[candidate]:
        sig_idx.append(ts)
        sig_close.append(plot_df.loc[ts, "low"] * 0.985)
    if sig_idx:
        fig.add_trace(go.Scatter(
            x=sig_idx, y=sig_close,
            mode="markers",
            marker=dict(symbol="triangle-up", size=10, color="#0F6E56", line=dict(width=1, color="white")),
            name="Setup forming",
            hovertext=[f"Setup candidate {t}" for t in sig_idx],
        ), row=1, col=1)

    # Volume
    vol_colors = ["#0F6E56" if c >= o else "#A32D2D"
                  for c, o in zip(plot_df["close"], plot_df["open"])]
    fig.add_trace(go.Bar(
        x=plot_df.index, y=plot_df["volume"],
        marker_color=vol_colors, name="Volume", showlegend=False,
    ), row=2, col=1)

    fig.update_layout(
        height=620,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=40, b=10),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Price (USDT)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1, showgrid=False)
    fig.update_xaxes(showgrid=True, gridcolor="rgba(120,120,120,0.1)")

    st.plotly_chart(fig, use_container_width=True)

    # Decision panel below chart
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Trade plan**")
        st.write(f"Entry (next 4H open): ~${scan_row['price']:.4g}")
        st.write(f"Stop: ${scan_row['stop']:.4g} ({(scan_row['stop']/scan_row['price']-1)*100:+.1f}%)")
        st.write(f"Target: ${scan_row['target']:.4g} ({(scan_row['target']/scan_row['price']-1)*100:+.1f}%)")
        st.write(f"R-multiple: **{scan_row['target_R']:.2f}**")
    with c2:
        st.markdown("**Position sizing (per VCP-EMA-Stack v1.2)**")
        st.write("Risk per trade: **1% of NAV**")
        st.write("Per-name cap: **8% of NAV** notional")
        risk_per_share = scan_row["price"] - scan_row["stop"]
        st.write(f"Risk per token: ${risk_per_share:.4g}")
        st.write(f"Position formula: `qty = NAV × 0.01 / {risk_per_share:.4g}`")
    with c3:
        st.markdown("**Condition status**")
        conds = [
            ("C1 Stack",     scan_row["C1_stack"]),
            ("C3 Body ≥3",   scan_row["C3_pass"]),
            ("C4 ATR <0.75", scan_row["C4_pass"]),
            ("C5 30d base",  scan_row["C5_pass"]),
            ("F1 60d base",  scan_row["F1_pass"]),
            ("F2 ret 90d",   scan_row["F2_pass"]),
            ("F5 Vol spike", scan_row["F5_pass"]),
            ("F3 BTC reg",   scan_row["F3_pass"]),
            ("F4 BTC mom",   scan_row["F4_pass"]),
        ]
        for label, ok in conds:
            st.write(f"{'✅' if ok else '❌'} {label}")


# ============================================================
# Backtest equity panel
# ============================================================
def render_equity():
    eq_data = load_backtest_equity()
    trades_data = load_backtest_trades()

    if eq_data[0] is None:
        st.info("No backtest equity file found in results directory.")
        return

    eq, eq_fname = eq_data
    trades, tr_fname = trades_data if trades_data[0] is not None else (None, None)

    # Stats
    nav0 = float(eq["equity"].iloc[0])
    nav1 = float(eq["equity"].iloc[-1])
    cum = eq["equity"] / nav0
    peak = cum.cummax()
    dd = (cum - peak) / peak
    days = (eq.index[-1] - eq.index[0]).total_seconds() / 86400
    cagr = cum.iloc[-1] ** (365 / max(1, days)) - 1
    max_dd = dd.min()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Final equity", f"${nav1:,.0f}", f"{(nav1/nav0-1)*100:+.1f}%")
    col2.metric("CAGR", f"{cagr:+.1%}")
    col3.metric("Max drawdown", f"{max_dd:.1%}")
    if trades is not None:
        wr = (trades["pnl"] > 0).mean()
        pf = abs(trades[trades["pnl"]>0]["pnl"].sum() / trades[trades["pnl"]<=0]["pnl"].sum()) if (trades["pnl"]<=0).sum() else float("inf")
        col4.metric("Win rate / PF", f"{wr:.1%} / {pf:.2f}")
    st.caption(f"Source: `{eq_fname}`" + (f" + `{tr_fname}` ({len(trades)} trades)" if trades is not None else ""))

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.7, 0.3], vertical_spacing=0.05,
                        subplot_titles=("Cumulative return", "Drawdown"))
    fig.add_trace(go.Scatter(
        x=eq.index, y=(cum.values - 1) * 100,
        line=dict(color="#0F6E56", width=2), name="Return",
        fill="tozeroy", fillcolor="rgba(15,110,86,0.1)",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=eq.index, y=dd.values * 100,
        line=dict(color="#A32D2D", width=1.5), name="Drawdown",
        fill="tozeroy", fillcolor="rgba(163,45,45,0.15)", showlegend=False,
    ), row=2, col=1)
    fig.update_layout(height=480, margin=dict(l=10, r=10, t=40, b=10), hovermode="x unified")
    fig.update_yaxes(title_text="Return %", row=1, col=1)
    fig.update_yaxes(title_text="DD %", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# Strategy B1 — render function
# ============================================================
@st.cache_data(ttl=300, show_spinner=False)
def _cached_b1_scan(data_dir: str):
    """Cache B1 scan results for 5 minutes."""
    return b1.scan_all(data_dir, list(UNIVERSE.keys()))


def render_strategy_b1_tab():
    """Render the Strategy B1 dashboard tab.

    Strategy: EMA35×EMA200 golden cross + BTC near 90d local high.
    Backtest: WR 70.8% (n=89, CI [62%, 80%]).
    """
    with st.spinner("Computing B1 signals..."):
        b1_df, b1_regime = _cached_b1_scan(DATA_DIR)

    # ===== Regime status banner =====
    if b1_regime.get("regime_active"):
        days = b1_regime.get("days_since_local_high", 0)
        st.success(
            f"✅ **TREND STRATEGY ACTIVE** — BTC made new 90d local high "
            f"{days:.1f} days ago. Tier-A signals are tradable."
        )
    else:
        days = b1_regime.get("days_since_local_high", 0)
        dd = b1_regime.get("btc_dd_from_local_high", 0)
        st.warning(
            f"⏸ **TREND STRATEGY PAUSED** — BTC last 90d local high was "
            f"**{days:.0f} days** ago (need ≤7). Currently {dd:+.1%} from local high. "
            f"Setups stay on watchlist but don't trigger entries."
        )

    # ===== Regime metrics =====
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("BTC price", f"${b1_regime.get('btc_price', 0):,.0f}")
    c2.metric("BTC 90d local high", f"${b1_regime.get('btc_local_high_90d', 0):,.0f}")
    c3.metric(
        "BTC vs local high",
        f"{b1_regime.get('btc_dd_from_local_high', 0):+.1%}",
    )
    c4.metric(
        "Days since new local high",
        f"{b1_regime.get('days_since_local_high', 0):.1f}",
        delta=f"{(7 - b1_regime.get('days_since_local_high', 0)):.1f}d to gate close",
        delta_color="normal",
    )

    # ===== Strategy info =====
    with st.expander("ℹ️ About Trend strategy"):
        st.markdown("""
**Trend strategy — EMA Crossover + BTC Local High**

**Entry conditions:**
1. EMA35_4H crossed above EMA200_4H within the last 5 bars (fresh golden cross)
2. BTC made a new 90-day local high within the last 7 days (regime gate)
3. R-target between 1.5 and 6 (computed from 2.5×ATR stop and 1:3 target)

**Exit:**
- Hard stop at 2.5×ATR14 below entry (catastrophic loss protection)
- 50% off at 1:3 target, then chandelier trail (2.5×ATR) on remainder
- Time stop: 60 bars max

**Historical performance (89 events, 2024-2025):**
- Win rate: 70.8%
- Median 14d return: +11.3%
- 95% CI on win rate: [61.8%, 79.8%]
- Best sector: L1 (WR 82%, n=45)
- Worst sector: Meme (WR 40%), Oracle (WR 50%)

**Known caveats:**
- Half of edge concentrates in Q4 2024 BTC rally — sample skew risk
- Q4 2025 showed bull-trap behavior: WR dropped to 10% (n=10)
- Strategy currently has 0 events in 2026 (BTC hasn't made new local highs)
- Sensitive to local-high window: 30d gives WR 47%, 90d gives 71% (p-hacking risk)
""")

    st.divider()

    # ===== Inner tabs =====
    sub1, sub2, sub3, sub4 = st.tabs([
        "📋 Watchlist",
        "🔍 Symbol detail",
        "🔬 Backtest",
        "🔥 Sector heatmap",
    ])

    # ----- Sub-tab 1: Watchlist -----
    with sub1:
        if b1_df.empty:
            st.info("No symbols evaluated.")
        else:
            # Tier counts
            t_a   = (b1_df["tier"] == "A").sum()
            t_ap  = (b1_df["tier"] == "A_pending_regime").sum()
            t_b   = (b1_df["tier"] == "B").sum()
            t_c   = (b1_df["tier"] == "C").sum()

            cc1, cc2, cc3, cc4 = st.columns(4)
            cc1.metric("Tier A — Actionable", t_a, help="Fresh cross + BTC regime ok")
            cc2.metric("A pending regime",   t_ap, help="Fresh cross, BTC regime off")
            cc3.metric("Tier B — In trend",  t_b,  help="EMA35>EMA200 but cross was >5 bars ago")
            cc4.metric("Tier C — Below",     t_c,  help="EMA35>EMA200 but cross was long ago")

            st.markdown("---")
            tier_filter_b1 = st.multiselect(
                "Filter by tier",
                options=["A", "A_pending_regime", "B", "C", "D"],
                default=["A", "A_pending_regime", "B"],
                key="b1_tier_filter",
            )
            view = b1_df[b1_df["tier"].isin(tier_filter_b1)].copy() if tier_filter_b1 else b1_df.copy()
            view["display"] = view["symbol"].apply(lambda s: UNIVERSE.get(s, (s, "?"))[0])
            view["sector"]  = view["symbol"].apply(lambda s: UNIVERSE.get(s, (s, "?"))[1])

            # Format for display
            view_disp = pd.DataFrame({
                "Symbol":      view["display"],
                "Sector":      view["sector"],
                "Tier":        view["tier"],
                "Price":       view["price"].map(lambda x: f"${x:.4g}"),
                "EMA35":       view["ema35"].map(lambda x: f"${x:.4g}"),
                "EMA200":      view["ema200"].map(lambda x: f"${x:.4g}"),
                "Cross aligned": view["cross_aligned"].map({True: "✓", False: "·"}),
                "Bars since cross": view["bars_since_cross"].apply(
                    lambda x: f"{int(x)}" if pd.notna(x) else "—"
                ),
                "Stop":        view["stop"].map(lambda x: f"${x:.4g}" if pd.notna(x) else "—"),
                "Target":      view["target"].map(lambda x: f"${x:.4g}" if pd.notna(x) else "—"),
                "R":           view["R_target"].map(lambda x: f"{x:.2f}"),
            })
            # Sort by tier rank
            tier_rank = {"A": 0, "A_pending_regime": 1, "B": 2, "C": 3, "D": 4}
            view_disp["_rank"] = view["tier"].map(tier_rank)
            view_disp = view_disp.sort_values("_rank").drop(columns=["_rank"])
            st.dataframe(view_disp, hide_index=True, use_container_width=True)
            st.caption(
                f"Showing {len(view_disp)} of {len(b1_df)} symbols. "
                "Tiers: A = fresh cross + BTC regime, A* = cross ok / BTC off, "
                "B = aligned but cross >5 bars ago, C = aligned but old, D = below."
            )

    # ----- Sub-tab 2: Symbol detail -----
    with sub2:
        if b1_df.empty:
            st.info("No symbols available.")
        else:
            # Sort by tier rank, then by bars_since_cross
            tier_rank = {"A": 0, "A_pending_regime": 1, "B": 2, "C": 3, "D": 4}
            b1_sorted = b1_df.copy()
            b1_sorted["_rank"] = b1_sorted["tier"].map(tier_rank)
            b1_sorted = b1_sorted.sort_values(["_rank", "bars_since_cross"], na_position="last")
            options = b1_sorted["symbol"].tolist()
            display_options = [
                f"{UNIVERSE.get(s,(s,'?'))[0]} ({s}) — {r['tier']}, "
                f"bars since cross: {int(r['bars_since_cross']) if pd.notna(r['bars_since_cross']) else '—'}"
                for s, r in zip(options, b1_sorted.to_dict("records"))
            ]
            choice = st.selectbox(
                "Select symbol",
                range(len(options)),
                format_func=lambda i: display_options[i],
                key="b1_detail_picker",
            )
            sym = options[choice]
            row = b1_df[b1_df["symbol"] == sym].iloc[0]

            # Summary metrics
            cc1, cc2, cc3, cc4 = st.columns(4)
            cc1.metric("Tier",        row["tier"])
            cc2.metric("Price",       f"${row['price']:.4g}")
            cc3.metric("EMA35/EMA200 spread", f"{row['ema_spread_pct']*100:.1f}%")
            cc4.metric("R target",    f"{row['R_target']:.2f}")

            cc5, cc6, cc7, cc8 = st.columns(4)
            cc5.metric("Stop",        f"${row['stop']:.4g}")
            cc6.metric("Target",      f"${row['target']:.4g}")
            cc7.metric("Risk %",      f"{(row['price']/row['stop']-1)*100:.1f}%")
            cc8.metric("Bars since cross",
                       f"{int(row['bars_since_cross']) if pd.notna(row['bars_since_cross']) else '—'}")

            # Conditions check
            st.markdown("**Entry conditions:**")
            checks = [
                ("EMA35 > EMA200 (cross aligned)", row["cross_aligned"]),
                ("Recent cross (≤5 bars)",         row["cross_active"]),
                ("BTC regime active (<7d since 90d high)", row["btc_regime_active"]),
                ("R target in [1.5, 6.0]",         row["R_ok"]),
            ]
            for label, ok in checks:
                glyph = "✅" if ok else "❌"
                st.markdown(f"- {glyph} {label}")

            # Action recommendation
            if row["tier"] == "A":
                st.success(
                    f"**🚀 TRADABLE — Tier A.** Entry on next bar at market. "
                    f"Stop at ${row['stop']:.4g}, target at ${row['target']:.4g}, R = {row['R_target']:.2f}."
                )
            elif row["tier"] == "A_pending_regime":
                st.warning(
                    "**⏸ Watch only — A pending regime.** Setup is ready but BTC has not "
                    "made a new 90d local high within 7 days. Wait for BTC regime to activate."
                )
            elif row["tier"] == "B":
                st.info(
                    "**👀 Already in trend — Tier B.** Crossover happened more than 5 bars ago. "
                    "Don't chase — wait for next pullback or new cross."
                )
            else:
                st.info(f"**Tier {row['tier']}** — not actionable.")

    # ----- Sub-tab 3: B1 Backtest -----
    with sub3:
        st.markdown(
            "Per-symbol historical backtest of the Trend strategy. "
            "Walks through all history, opens trades on every cross + BTC regime match, "
            "exits via stop/target/trail."
        )
        render_b1_symbol_backtest(DATA_DIR, UNIVERSE)

    # ----- Sub-tab 4: Sector heatmap (B1-specific) -----
    with sub4:
        if b1_df.empty:
            st.info("No data for heatmap.")
        else:
            st.markdown("**Trend signal heat across sectors** — where crossovers are forming right now.")
            # Reuse the existing heatmap renderer, passing B1 tier data
            render_sector_heatmap(b1_df, UNIVERSE)


# ============================================================
# Main app flow
# ============================================================
def main():
    # Load scan
    with st.spinner("Computing signals..."):
        scan_df, regime = run_scan()

    # Write snapshot for history tracking (one per unique 4H bar, idempotent)
    try:
        save_snapshot(scan_df, regime, SNAPSHOTS_DIR)
    except Exception as e:
        st.sidebar.caption(f"⚠ Snapshot write failed: {type(e).__name__}")

    # Header title + refresh button (always visible at top)
    render_title(regime)

    # Sidebar filters (always visible, used by Watchlist tab)
    with st.sidebar:
        st.header("Filters")
        sector_filter = st.multiselect(
            "Sector",
            options=SECTORS,
            default=[],
            help="Empty = all sectors. Affects the Watchlist tab.",
        )
        tier_filter = st.multiselect(
            "Tier",
            options=["A", "A_pending_regime", "B", "C"],
            default=["A", "A_pending_regime", "B"],
            help="A = ready, A* = setup ok but regime off, B/C = forming",
        )
        min_score = st.slider("Min pass count (0-7)", 0, 7, 5)
        st.divider()
        st.caption(f"Universe: {len(UNIVERSE)} symbols")
        st.caption(f"Evaluated: {len(scan_df)}")
        st.caption(f"Data: `{DATA_DIR}`")
        if regime["as_of"]:
            age = (datetime.now(regime["as_of"].tz) - regime["as_of"]).total_seconds() / 3600
            st.caption(f"Data age: {age:.1f}h")

    # ============================================================
    # Top-level: STRATEGY SWITCHER
    # Each top-level tab is a complete strategy with its own sub-tabs
    # ============================================================
    strat_vcp, strat_b1 = st.tabs([
        "🔄 Reversal",
        "📈 Trend",
    ])

    # ------------------------------------------------------------
    # STRATEGY 1 — VCP-EMA-Stack
    # ------------------------------------------------------------
    with strat_vcp:
        # VCP regime banner
        render_vcp_regime(regime)
        st.divider()

        # Tier summary cards
        render_tier_summary(scan_df, regime["active"])
        st.divider()

        # Sub-tabs for VCP strategy views
        vcp_tab1, vcp_tab2, vcp_tab3, vcp_tab4, vcp_tab5 = st.tabs([
            "📋 Watchlist",
            "🔍 Symbol detail",
            "📊 Portfolio backtest",
            "🔬 Strategy backtest",
            "🔥 Sector heatmap",
        ])

        with vcp_tab1:
            df_view = render_watchlist(scan_df, sector_filter, tier_filter, min_score)
            if df_view is not None:
                st.caption(
                    f"Showing {len(df_view)} of {len(scan_df)} evaluated symbols. "
                    "Glyphs: ✓ pass · · fail. Sort by clicking column headers."
                )

        with vcp_tab2:
            sorted_df = scan_df.sort_values(["symbol_pass", "F1_base_60d"],
                                             ascending=[False, False])
            options = sorted_df["symbol"].tolist()
            if options:
                display_options = [
                    f"{UNIVERSE.get(s,(s,'?'))[0]} ({s}) — {r['tier']}, {r['symbol_pass']}/7"
                    for s, r in zip(options, sorted_df.to_dict("records"))
                ]
                choice = st.selectbox(
                    "Select symbol for detail view",
                    range(len(options)),
                    format_func=lambda i: display_options[i],
                    index=0,
                    key="vcp_symbol_picker",
                )
                sym = options[choice]
                row = scan_df[scan_df["symbol"] == sym].iloc[0]
                render_symbol_detail(sym, row)
            else:
                st.info("No symbols available.")

        with vcp_tab3:
            render_equity()

        with vcp_tab4:
            st.subheader("Reversal — per-symbol historical backtest")
            render_symbol_backtest(DATA_DIR, UNIVERSE)

        with vcp_tab5:
            st.subheader("Sector heat — where setups are concentrated right now")
            render_sector_heatmap(scan_df, UNIVERSE)
            st.divider()
            st.subheader("📜 Signal history")
            render_history(SNAPSHOTS_DIR, DATA_DIR, UNIVERSE)

    # ------------------------------------------------------------
    # STRATEGY 2 — B1 (EMA Crossover + BTC Local High)
    # ------------------------------------------------------------
    with strat_b1:
        render_strategy_b1_tab()


if __name__ == "__main__" or True:  # streamlit runs scripts top-level
    main()
