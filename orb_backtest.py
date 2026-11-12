"""
Opening Range + Inside Candles Breakout - Backtest (Nifty 200, 5-min)

STRATEGY RULES
1. First 5-min candle of the day (09:15-09:20) -> mark High (RH) and Low (RL).
2. Next candles (2..5 of the day onwards) must stay fully INSIDE the range
   (high <= RH and low >= RL, wicks included). Count must be between
   MIN_INSIDE and MAX_INSIDE (default 2 to 5).
3. Then a 5-min candle CLOSES above RH (LONG) or below RL (SHORT) -> SIGNAL.
4. Wick pokes outside but closes inside, or breakout comes too early/late
   -> setup is invalid for that day.

TRADE SIMULATION
- Entry : close of breakout candle
- SL    : opposite side of the opening range (or midpoint, see SL_MODE)
- Target: RR x risk
- If SL and target both hit in the same candle -> SL assumed (conservative)
- Square off at 15:15 candle close
- One trade per stock per day

USAGE
    pip install yfinance pandas numpy requests
    python orb_backtest.py --demo                 # test logic on synthetic data
    python orb_backtest.py                        # real run, yfinance (last ~58 days only)
    python orb_backtest.py --symbols RELIANCE TCS # specific stocks

NOTE: yfinance gives only ~60 days of 5-min data. For longer history
use Angel One SmartAPI / Upstox / Zerodha Kite and plug it into load_data().
"""
import argparse
import datetime as dt
import io
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd


# ----------------------------------------------------------------- PARAMS
@dataclass
class Params:
    min_inside: int = 2          # minimum inside candles after first candle
    max_inside: int = 5          # maximum inside candles after first candle
    rr: float = 2.0              # target = rr * risk
    sl_mode: str = "range"       # "range" = opposite edge, "mid" = range midpoint
    min_range_pct: float = 0.0   # skip if first candle range % smaller than this
    max_range_pct: float = 100.0  # skip if larger than this
    cost_pct: float = 0.05       # round-trip cost+slippage as % of entry price
    squareoff: str = "15:15"     # last candle start time to hold
    market_open: str = "09:15"


# ----------------------------------------------------------------- SIGNAL
def detect_signal(day: pd.DataFrame, p: Params):
    """
    day: one trading day of 5-min candles (index = timestamps, sorted),
         columns: open, high, low, close.
    Returns dict(side, idx, time, entry, rh, rl, inside) or None.
    SAME function will be reused by the live bot.
    """
    if len(day) < p.min_inside + 2:
        return None
    if day.index[0].strftime("%H:%M") != p.market_open:
        return None  # first candle missing -> skip day

    first = day.iloc[0]
    rh, rl = first["high"], first["low"]
    rng_pct = (rh - rl) / first["close"] * 100
    if rng_pct < p.min_range_pct or rng_pct > p.max_range_pct:
        return None

    inside = 0
    for i in range(1, len(day)):
        c = day.iloc[i]
        if c["high"] <= rh and c["low"] >= rl:
            inside += 1
            if inside > p.max_inside:
                return None
            continue
        if c["close"] > rh or c["close"] < rl:
            if inside < p.min_inside:
                return None
            side = "LONG" if c["close"] > rh else "SHORT"
            return dict(side=side, idx=i, time=day.index[i], entry=c["close"],
                        rh=rh, rl=rl, inside=inside)
        return None  # wick outside but closed inside -> invalid
    return None


# ------------------------------------------------------------- SIMULATION
def simulate_trade(day: pd.DataFrame, sig: dict, p: Params):
    entry, rh, rl = sig["entry"], sig["rh"], sig["rl"]
    if sig["side"] == "LONG":
        sl = rl if p.sl_mode == "range" else (rh + rl) / 2
        risk = entry - sl
        tgt = entry + p.rr * risk
    else:
        sl = rh if p.sl_mode == "range" else (rh + rl) / 2
        risk = sl - entry
        tgt = entry - p.rr * risk
    if risk <= 0:
        return None

    exit_px, reason, exit_time = None, "EOD", None
    for j in range(sig["idx"] + 1, len(day)):
        c = day.iloc[j]
        t = day.index[j]
        if sig["side"] == "LONG":
            hit_sl, hit_tg = c["low"] <= sl, c["high"] >= tgt
        else:
            hit_sl, hit_tg = c["high"] >= sl, c["low"] <= tgt
        if hit_sl:                       # SL first if both hit (conservative)
            exit_px, reason, exit_time = sl, "SL", t
            break
        if hit_tg:
            exit_px, reason, exit_time = tgt, "TARGET", t
            break
        if t.strftime("%H:%M") >= p.squareoff:
            exit_px, reason, exit_time = c["close"], "EOD", t
            break
    if exit_px is None:                  # data ended early
        exit_px, exit_time = day.iloc[-1]["close"], day.index[-1]

    pnl_pts = (exit_px - entry) if sig["side"] == "LONG" else (entry - exit_px)
    gross_pct = pnl_pts / entry * 100
    net_pct = gross_pct - p.cost_pct
    r_mult = pnl_pts / risk - (p.cost_pct / 100 * entry) / risk
    return dict(entry_time=sig["time"], exit_time=exit_time, side=sig["side"],
                entry=round(entry, 2), sl=round(sl, 2), target=round(tgt, 2),
                exit=round(exit_px, 2), reason=reason, inside=sig["inside"],
                range_high=rh, range_low=rl,
                gross_pct=round(gross_pct, 3), net_pct=round(net_pct, 3),
                r_mult=round(r_mult, 3))


# ----------------------------------------------------------------- DATA
FALLBACK = ["RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "SBIN", "ITC",
            "LT", "AXISBANK", "BHARTIARTL", "KOTAKBANK", "TATAMOTORS", "MARUTI",
            "SUNPHARMA", "BAJFINANCE", "TITAN", "ADANIENT", "NTPC", "ONGC", "WIPRO"]


def get_nifty200_symbols():
    """Download official Nifty 200 list from NSE; else local nifty200.csv; else fallback."""
    import requests
    url = "https://nsearchives.nseindia.com/content/indices/ind_nifty200list.csv"
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        df.to_csv("nifty200.csv", index=False)
        return df["Symbol"].tolist()
    except Exception as e:
        print(f"[warn] NSE download failed ({e})")
    if os.path.exists("nifty200.csv"):
        return pd.read_csv("nifty200.csv")["Symbol"].tolist()
    print("[warn] using small fallback list; put NSE's nifty200.csv in this folder")
    return FALLBACK


def load_data(symbol: str) -> pd.DataFrame:
    """5-min OHLC in IST. Replace this function to use any broker API."""
    import yfinance as yf
    df = yf.download(symbol + ".NS", period="58d", interval="5m",
                     progress=False, auto_adjust=False)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)[["open", "high", "low", "close"]].dropna()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert("Asia/Kolkata").tz_localize(None)
    return df


def synthetic_data(seed: int, days: int = 60) -> pd.DataFrame:
    """Random-walk 5-min data (09:15-15:30) to sanity-test the code."""
    rng = np.random.default_rng(seed)
    rows, price = [], 1000.0
    for d in pd.bdate_range(end=dt.date.today(), periods=days):
        times = pd.date_range(d + pd.Timedelta(hours=9, minutes=15), periods=75, freq="5min")
        vol = rng.uniform(0.0008, 0.003)
        for t in times:
            o = price
            c = o * (1 + rng.normal(0, vol))
            h = max(o, c) * (1 + abs(rng.normal(0, vol / 2)))
            l = min(o, c) * (1 - abs(rng.normal(0, vol / 2)))
            rows.append((t, o, h, l, c))
            price = c
    return pd.DataFrame(rows, columns=["time", "open", "high", "low", "close"]).set_index("time")


# --------------------------------------------------------------- BACKTEST
def backtest_symbol(symbol, df, p):
    trades = []
    for _, day in df.groupby(df.index.date):
        day = day.sort_index()
        sig = detect_signal(day, p)
        if sig:
            tr = simulate_trade(day, sig, p)
            if tr:
                tr["symbol"] = symbol
                trades.append(tr)
    return trades


def report(tr: pd.DataFrame):
    if tr.empty:
        print("No trades generated.")
        return
    wins = tr[tr.net_pct > 0]
    losses = tr[tr.net_pct <= 0]
    pf = wins.net_pct.sum() / abs(losses.net_pct.sum()) if len(losses) and losses.net_pct.sum() != 0 else float("inf")
    eq = tr.sort_values("entry_time").net_pct.cumsum()
    dd = (eq - eq.cummax()).min()
    print("\n=========== BACKTEST SUMMARY ===========")
    print(f"Trades            : {len(tr)}  (Long {sum(tr.side=='LONG')}, Short {sum(tr.side=='SHORT')})")
    print(f"Win rate          : {len(wins)/len(tr)*100:.1f}%")
    print(f"Avg R / trade     : {tr.r_mult.mean():.3f}")
    print(f"Avg net % / trade : {tr.net_pct.mean():.3f}%")
    print(f"Profit factor     : {pf:.2f}")
    print(f"Total net % (sum) : {tr.net_pct.sum():.2f}%")
    print(f"Max drawdown (%)  : {dd:.2f}%  (sum-of-returns basis)")
    print("Exit reasons      :", tr.reason.value_counts().to_dict())
    print("\nTop 5 symbols by net %:")
    print(tr.groupby("symbol").net_pct.sum().sort_values(ascending=False).head(5).round(2).to_string())
    print("\nBottom 5 symbols:")
    print(tr.groupby("symbol").net_pct.sum().sort_values().head(5).round(2).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="use synthetic data")
    ap.add_argument("--symbols", nargs="*")
    ap.add_argument("--rr", type=float, default=2.0)
    ap.add_argument("--sl", choices=["range", "mid"], default="range")
    ap.add_argument("--min-inside", type=int, default=2)
    ap.add_argument("--max-inside", type=int, default=5)
    a = ap.parse_args()

    p = Params(min_inside=a.min_inside, max_inside=a.max_inside, rr=a.rr, sl_mode=a.sl)
    symbols = a.symbols or (FALLBACK[:10] if a.demo else get_nifty200_symbols())

    all_trades = []
    for k, s in enumerate(symbols, 1):
        try:
            df = synthetic_data(k) if a.demo else load_data(s)
            if df.empty:
                print(f"[{k}/{len(symbols)}] {s}: no data")
                continue
            t = backtest_symbol(s, df, p)
            all_trades += t
            print(f"[{k}/{len(symbols)}] {s}: {len(t)} trades")
        except Exception as e:
            print(f"[{k}/{len(symbols)}] {s}: error {e}")

    tr = pd.DataFrame(all_trades)
    if not tr.empty:
        tr.sort_values("entry_time").to_csv("backtest_trades.csv", index=False)
        print("Saved trades -> backtest_trades.csv")
    report(tr)


if __name__ == "__main__":
    main()
