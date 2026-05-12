# VCP-EMA-Stack Dashboard

Streamlit + Plotly web UI for monitoring the VCP-EMA-Stack v1.2 momentum strategy across a configurable crypto universe.

## What it shows

- **Regime widget** — current BTC vs EMA200_4H position, 30-day return, and strategy ACTIVE/PAUSED status (F3/F4 gates).
- **Tier summary** — count of Tier A (actionable), A* (setup ready but regime off), B (6/7 conditions), C (5/7 forming) candidates.
- **Watchlist table** — every evaluated symbol with all 7 local condition flags (C1, C3, C4, C5, F1, F2, F5) + computed stop / target / R-multiple. Filterable by sector, tier, minimum pass count.
- **Symbol detail tab** — interactive Plotly 4H candlestick with EMA 35 / 100 / 200 overlays, volume panel, stop and target horizontal lines, retrospective "setup forming" markers.
- **Backtest tab** — equity curve, drawdown, win rate, profit factor from the most recent backtest in `results/`.

## Quick start (local)

```bash
# 1. Clone or copy the dashboard folder
cd dashboard

# 2. Install dependencies
pip install -r requirements.txt

# 3. Point at your data directory (parquet files: SYMBOL_4h.parquet, BTCUSDT_1d.parquet)
export VCP_DATA_DIR=/path/to/data/clean
export VCP_RESULTS_DIR=/path/to/backtest/results   # optional, for equity tab

# 4. Run
streamlit run app.py
```

The app opens at `http://localhost:8501`. First load takes ~5 seconds while the engine evaluates the universe.

## File layout

```
dashboard/
├── app.py              # Streamlit UI — header, tabs, charts
├── engine.py           # Pure-function signal computation (importable, no UI)
├── universe.py         # Symbol list + sector tags (per v1.2 spec)
├── refresh_data.py     # Pulls fresh 4H OHLC from Coinglass into data/clean
├── requirements.txt
└── README.md
```

`engine.py` has no Streamlit dependency. You can import it from a cron job or jupyter notebook:

```python
from engine import scan_all
df, regime = scan_all("data/clean")
print(df.sort_values("symbol_pass", ascending=False))
```

## Refreshing data

The dashboard reads parquet files from `VCP_DATA_DIR`. Each symbol needs `{SYMBOL}_4h.parquet` (4H OHLC + volume) and `BTCUSDT_1d.parquet` for the daily 30-day return gate.

If you have a Coinglass paid plan:

```bash
export COINGLASS_API_KEY=your_key
python refresh_data.py
```

This pulls history from Jan 2024 onwards for every symbol in `universe.py` (and BTC daily). Takes 5–10 minutes on first run. Schedule it via cron every 4 hours to keep signals fresh:

```cron
# Every 4 hours at minute 5 (5 min after 4H bar close)
5 */4 * * * cd /path/to/dashboard && python refresh_data.py >> refresh.log 2>&1
```

Without an API key, the dashboard works fine on whatever parquet files you have on disk — but the data will get stale.

## Editing the universe

Open `universe.py` and edit the `UNIVERSE` dict. Each entry is:

```python
"SYMBOLUSDT": ("DisplayName", "Sector"),
```

The v1.2 spec excludes BTC, ETH, SOL, HYPE plus all stablecoins and wrapped/staked tokens. Don't add them — they're not what the strategy is designed for.

Adding a new symbol requires (a) entry in `universe.py`, (b) corresponding parquet file in `VCP_DATA_DIR`. Run `refresh_data.py` to pull it.

## Deploying to Streamlit Cloud

Streamlit Cloud (1 private app free) works well for personal use.

1. Push the `dashboard/` folder to a private GitHub repo. Include your parquet files (size them — git LFS if >100MB total).
2. Go to https://share.streamlit.io, connect the repo, point to `app.py`.
3. In Streamlit Cloud "Settings → Secrets", set:
   ```toml
   COINGLASS_API_KEY = "your_key_if_using_refresh"
   ```
4. App is live at `https://your-app.streamlit.app`.

For periodic data refresh on Streamlit Cloud, you have two options:

- **Manual**: click the 🔄 Refresh button in the app header (clears cache, re-reads parquet)
- **Background cron** (recommended): run `refresh_data.py` on a separate VPS or GitHub Actions schedule, commit the updated parquets back to the repo. Streamlit Cloud auto-redeploys on push.

GitHub Actions snippet (`.github/workflows/refresh.yml`):

```yaml
name: Refresh data
on:
  schedule:
    - cron: '5 */4 * * *'   # every 4 hours
  workflow_dispatch:
jobs:
  refresh:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - run: pip install -r dashboard/requirements.txt
      - env: { COINGLASS_API_KEY: ${{ secrets.COINGLASS_API_KEY }} }
        run: cd dashboard && python refresh_data.py
      - run: |
          git config user.name "github-actions"
          git config user.email "actions@github.com"
          git add dashboard/data
          git diff --staged --quiet || git commit -m "Refresh data $(date -u +%F-%H)"
          git push
```

## How to use it day-to-day

1. **Morning routine**: open dashboard, check the regime status banner. If ACTIVE (green), proceed. If PAUSED, you're in observation mode only.
2. **Watchlist tab**: scan the Tier A row. If any 7/7 rows appear, those are tradable per v1.2 spec.
3. **Tier B (6/7) rows**: monitor — they could become A on any 4H close. Common pattern: missing C4 (ATR not yet compressed) — wait for volatility to dry up.
4. **Symbol detail tab**: pick the candidate, check the candlestick. Verify the EMA stack is clean (no whipsaws) and the base is visually tight on the chart. The dashboard's quantitative signal is necessary but not sufficient — eyeball confirms.
5. **Position sizing**: dashboard shows `qty = NAV × 1% / (entry − stop)`. Cap at 8% of NAV notional.

## Limitations & honest disclaimers

- The "setup forming" triangles in the symbol detail chart show **historical bars where the basic stack + streak + compression aligned**. This is a quick-and-dirty visualization, not a backtest signal — it skips the smart filters (F1, F2, F5) for chart speed. Trust the watchlist table for actionable decisions.
- Cached scan refreshes every 5 minutes by default (`@st.cache_data(ttl=300)`). Click 🔄 Refresh to force.
- `refresh_data.py` requires a Coinglass paid plan for the `/futures/price/history` endpoint. Free tier is metadata-only.
- The strategy spec is in the parent project's `VCP_EMA_STACK_v1.2.md` — keep that in sync with `engine.CFG` if you change parameters.
