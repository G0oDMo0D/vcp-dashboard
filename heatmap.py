"""
Sector heatmap — visualizes which sectors are 'heating up' right now.

Renders a Plotly heatmap with sectors on Y-axis and tier categories on X-axis.
Cell intensity = number of symbols in that bucket.
Reveals sector rotation in progress (e.g. 4 DeFi in Tier B = DeFi forming).
"""
from __future__ import annotations
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


def render_sector_heatmap(scan_df: pd.DataFrame, universe: dict) -> None:
    if scan_df.empty:
        st.info("No scan data available for heatmap.")
        return

    df = scan_df.copy()
    df["sector"] = df["symbol"].map(lambda s: universe.get(s, (s, "Other"))[1])

    tier_order = ["A", "A_pending_regime", "B", "C", "D"]
    tier_labels = ["A (7/7+regime)", "A* (7/7 pending)", "B (6/7)", "C (5/7)", "D (<5)"]

    # Pivot to sector x tier counts
    pivot = pd.pivot_table(
        df, index="sector", columns="tier",
        values="symbol", aggfunc="count", fill_value=0,
    ).reindex(columns=tier_order, fill_value=0)

    # Sort sectors by sum of A+A_pending+B counts (where action is most likely)
    pivot["_heat"] = pivot["A"] + pivot["A_pending_regime"] + pivot["B"]
    pivot = pivot.sort_values("_heat", ascending=True)
    heat_scores = pivot["_heat"].copy()
    pivot = pivot.drop(columns=["_heat"])

    z = pivot.values
    sectors = pivot.index.tolist()

    # Per-cell symbol lists for hover
    sym_per_cell = []
    for sector in sectors:
        row_text = []
        for tier in tier_order:
            cell_syms = df[(df["sector"] == sector) & (df["tier"] == tier)]["symbol"].tolist()
            cell_displays = [universe.get(s, (s, "?"))[0] for s in cell_syms]
            if cell_displays:
                row_text.append("<br>".join(cell_displays[:10]) + ("..." if len(cell_displays) > 10 else ""))
            else:
                row_text.append("(none)")
        sym_per_cell.append(row_text)

    # Color scale — green for hot tiers (A), grayscale for cold (D)
    colorscale = [
        [0.0,  "rgba(50,50,50,0.05)"],
        [0.15, "rgba(120,120,120,0.25)"],
        [0.4,  "rgba(186,117,23,0.55)"],
        [0.7,  "rgba(15,110,86,0.75)"],
        [1.0,  "rgba(15,110,86,1.0)"],
    ]

    fig = go.Figure(data=go.Heatmap(
        z=z, x=tier_labels, y=sectors,
        text=z, texttemplate="%{text}",
        textfont=dict(size=13, color="white"),
        customdata=sym_per_cell,
        hovertemplate="<b>%{y} × %{x}</b><br>%{z} symbols<br><br>%{customdata}<extra></extra>",
        colorscale=colorscale,
        showscale=False,
        zmin=0, zmax=max(1, int(z.max())),
    ))
    fig.update_layout(
        height=max(280, 50 + 38 * len(sectors)),
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(side="top", tickfont=dict(size=11)),
        yaxis=dict(tickfont=dict(size=12)),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Sector "heat" leaderboard below
    hot_sectors = heat_scores[heat_scores > 0].sort_values(ascending=False).head(5)
    if len(hot_sectors):
        st.markdown("**Hottest sectors right now** (count of Tier A + A* + B):")
        cols = st.columns(min(len(hot_sectors), 5))
        for (sector, count), col in zip(hot_sectors.items(), cols):
            col.metric(sector, int(count))
    else:
        st.caption("No sectors with active setups right now. All quiet.")
