"""
VCP-EMA-Stack — Universe configuration.

Per v1.2 spec:
- INCLUDE: all USDT-perps on Binance/Bybit/OKX with sufficient liquidity,
  any market cap (including top-10 alts).
- EXCLUDE: BTC, ETH, SOL, HYPE + all stablecoins + wrapped/staked tokens.

Sector tagging is for dashboard filtering only — does not affect strategy logic.
"""

# Symbol -> (display_name, sector)
UNIVERSE = {
    # Major L1s (top mcap alts — kept per v1.2)
    "XRPUSDT":      ("XRP",      "L1"),
    "BNBUSDT":      ("BNB",      "L1"),
    "DOGEUSDT":     ("DOGE",     "Meme"),
    "ADAUSDT":      ("ADA",      "L1"),
    "TRXUSDT":      ("TRX",      "L1"),
    "AVAXUSDT":     ("AVAX",     "L1"),
    "LINKUSDT":     ("LINK",     "Oracle"),
    "BCHUSDT":      ("BCH",      "L1"),
    "LTCUSDT":      ("LTC",      "L1"),
    "XLMUSDT":      ("XLM",      "L1"),
    "DOTUSDT":      ("DOT",      "L1"),
    "TONUSDT":      ("TON",      "L1"),
    "NEARUSDT":     ("NEAR",     "L1"),
    "APTUSDT":      ("APT",      "L1"),
    "ATOMUSDT":     ("ATOM",     "L1"),
    "ICPUSDT":      ("ICP",      "L1"),
    "HBARUSDT":     ("HBAR",     "L1"),
    "FILUSDT":      ("FIL",      "Storage"),
    "SUIUSDT":      ("SUI",      "L1"),
    "SEIUSDT":      ("SEI",      "L1"),
    "TIAUSDT":      ("TIA",      "L1"),
    # L2 / scaling
    "ARBUSDT":      ("ARB",      "L2"),
    "OPUSDT":       ("OP",       "L2"),
    "POLUSDT":      ("POL",      "L2"),
    "IMXUSDT":      ("IMX",      "L2"),
    # DeFi
    "AAVEUSDT":     ("AAVE",     "DeFi"),
    "UNIUSDT":      ("UNI",      "DeFi"),
    "CRVUSDT":      ("CRV",      "DeFi"),
    "LDOUSDT":      ("LDO",      "DeFi"),
    "PENDLEUSDT":   ("PENDLE",   "DeFi"),
    "MORPHOUSDT":   ("MORPHO",   "DeFi"),
    "ENAUSDT":      ("ENA",      "DeFi"),
    "ONDOUSDT":     ("ONDO",     "RWA"),
    "INJUSDT":      ("INJ",      "DeFi"),
    "JUPUSDT":      ("JUP",      "DeFi"),
    "CAKEUSDT":     ("CAKE",     "DeFi"),
    "AEROUSDT":     ("AERO",     "DeFi"),
    "ETHFIUSDT":    ("ETHFI",    "DeFi"),
    # AI / Data
    "TAOUSDT":      ("TAO",      "AI"),
    "RENDERUSDT":   ("RENDER",   "AI"),
    "FETUSDT":      ("FET",      "AI"),
    "WLDUSDT":      ("WLD",      "AI"),
    "VIRTUALUSDT":  ("VIRTUAL",  "AI"),
    # Memes
    "1000PEPEUSDT": ("PEPE",     "Meme"),
    "1000SHIBUSDT": ("SHIB",     "Meme"),
    "WIFUSDT":      ("WIF",      "Meme"),
    "1000BONKUSDT": ("BONK",     "Meme"),
    "1000FLOKIUSDT":("FLOKI",    "Meme"),
    "POPCATUSDT":   ("POPCAT",   "Meme"),
    "PENGUUSDT":    ("PENGU",    "Meme"),
    "PUMPUSDT":     ("PUMP",     "Meme"),
    "SPXUSDT":      ("SPX6900",  "Meme"),
    # Other
    "JTOUSDT":      ("JTO",      "DeFi"),
    "JASMYUSDT":    ("JASMY",    "Data"),
    "PYTHUSDT":     ("PYTH",     "Oracle"),
    "ENSUSDT":      ("ENS",      "Infra"),
    "QNTUSDT":      ("QNT",      "Infra"),
}

SECTORS = sorted(set(s for _, s in UNIVERSE.values()))

if __name__ == "__main__":
    print(f"Universe: {len(UNIVERSE)} symbols across {len(SECTORS)} sectors")
    for sector in SECTORS:
        count = sum(1 for _, s in UNIVERSE.values() if s == sector)
        print(f"  {sector}: {count}")
