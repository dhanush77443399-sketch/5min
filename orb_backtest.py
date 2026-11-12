"""
Opening Range + Inside Candles Breakout - Backtest (Nifty 200, 5-min)

STRATEGY RULES
1. First 5-min candle of the day (09:15-09:20) -> mark High (RH) and Low (RL).
2. Next candles must stay fully INSIDE the range (high <= RH and low >= RL,
   wicks included). Number of inside candles must be MIN_INSIDE..MAX_INSIDE
   (default 2 to 4).
3. Then a 5-min candle CLOSES above RH (LONG) or below RL (SHORT) -> SIGNAL.
4. Invalid for the day if: a wick pokes outside but candle closes inside,
   breakout comes with fewer than MIN_INSIDE inside candles, or more than
   MAX_INSIDE inside candles happen before the breakout.

TRADE SIMULATION
- Entry : close of breakout candle
- SL    : opposite side of opening range (or midpoint with --sl mid)
- Target: RR x risk
- If SL and target hit in the same candle -> SL assumed (conservative)
- Square off at 15:15 candle close. One trade per stock per day.

DATA SOURCES
- upstox  (default) : 5-min data from Jan 2022. Free. Uses ISIN from nifty200.csv.
                      Optional env var UPSTOX_TOKEN (sent as Bearer token if set).
- yfinance          : only ~58 days.
Downloaded data is cached in ./data_cache so reruns are fast.

USAGE
    pip install pandas numpy requests yfinance
    python orb_backtest.py --demo                       # logic test, fake data
    python orb_backtest.py --days 365                   # 1 year, Nifty 200
    python orb_backtest.py --days 1700                  # maximum (~Jan 2022)
    python orb_backtest.py --symbols RELIANCE TCS --days 365
    python orb_backtest.py --days 365 --telegram        # send results to Telegram
Telegram needs env vars TG_TOKEN and TG_CHAT_ID.
"""
import argparse
import datetime as dt
import io
import os
import time
from dataclasses import dataclass
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests

CACHE_DIR = "data_cache"


# ----------------------------------------------------------------- PARAMS
@dataclass
class Params:
    min_inside: int = 2          # minimum inside candles after first candle
    max_inside: int = 4          # maximum inside candles after first candle
    rr: float = 2.0              # target = rr * risk
    sl_mode: str = "range"       # "range" = opposite edge, "mid" = midpoint
    min_range_pct: float = 0.0   # skip if first-candle range % smaller than this
    max_range_pct: float = 100.0
    cost_pct: float = 0.05       # round-trip cost + slippage, % of entry price
    squareoff: str = "15:15"
    market_open: str = "09:15"


# ----------------------------------------------------------------- SIGNAL
def detect_signal(day: pd.DataFrame, p: Params):
    """
    day: one trading day of 5-min candles (sorted index), cols open/high/low/close.
    Returns dict(side, idx, time, entry, rh, rl, inside) or None.
    The live bot will reuse this same function.
    """
    if len(day) < p.min_inside + 2:
        return None
    if day.index[0].strftime("%H:%M") != p.market_open:
        return None

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
        if hit_sl:
            exit_px, reason, exit_time = sl, "SL", t
            break
        if hit_tg:
            exit_px, reason, exit_time = tgt, "TARGET", t
            break
        if t.strftime("%H:%M") >= p.squareoff:
            exit_px, reason, exit_time = c["close"], "EOD", t
            break
    if exit_px is None:
        exit_px, exit_time = day.iloc[-1]["close"], day.index[-1]

    pnl_pts = (exit_px - entry) if sig["side"] == "LONG" else (entry - exit_px)
    gross_pct = pnl_pts / entry * 100
    net_pct = gross_pct - p.cost_pct
    r_mult = (pnl_pts - p.cost_pct / 100 * entry) / risk
    return dict(entry_time=sig["time"], exit_time=exit_time, side=sig["side"],
                entry=round(entry, 2), sl=round(sl, 2), target=round(tgt, 2),
                exit=round(exit_px, 2), reason=reason, inside=sig["inside"],
                range_high=rh, range_low=rl,
                gross_pct=round(gross_pct, 3), net_pct=round(net_pct, 3),
                r_mult=round(r_mult, 3))


# ----------------------------------------------------------- SYMBOL LIST
FALLBACK = [("RELIANCE", "INE002A01018"), ("TCS", "INE467B01029"),
            ("HDFCBANK", "INE040A01034"), ("ICICIBANK", "INE090A01021"),
            ("INFY", "INE009A01021"), ("SBIN", "INE062A01020"),
            ("ITC", "INE154A01025"), ("LT", "INE018A01030"),
            ("AXISBANK", "INE238A01034"), ("BHARTIARTL", "INE397D01024")]


def get_nifty200():
    """Returns list of (symbol, isin). Tries NSE, then local nifty200.csv."""
    url = "https://nsearchives.nseindia.com/content/indices/ind_nifty200list.csv"
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        df.to_csv("nifty200.csv", index=False)
    except Exception as e:
        print(f"[warn] NSE download failed ({e})")
        if os.path.exists("nifty200.csv"):
            df = pd.read_csv("nifty200.csv")
        else:
            print("[warn] using 10-stock fallback. Upload NSE's nifty200.csv to the repo.")
            return FALLBACK
    df.columns = [c.strip() for c in df.columns]
    return list(zip(df["Symbol"], df["ISIN Code"]))


# ------------------------------------------------------------------ DATA
def _get_json(url, retries=4):
    headers = {"Accept": "application/json"}
    token = os.environ.get("UPSTOX_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    for a in range(1, retries + 1):
        try:
            r = requests.get(url, headers=headers, timeout=30)
            if r.status_code == 429:
                time.sleep(30 * a)
                continue
            if r.status_code == 200:
                return r.json()
            if r.status_code == 400:
                print(f"    [400] {r.text[:120]}")
                return None
            time.sleep(2 * a)
        except requests.RequestException:
            time.sleep(2 * a)
    return None


def parse_upstox_candles(candles) -> pd.DataFrame:
    """candles: [[ts, o, h, l, c, vol, oi], ...] (newest first)."""
    if not candles:
        return pd.DataFrame()
    df = pd.DataFrame(candles).iloc[:, :5]
    df.columns = ["time", "open", "high", "low", "close"]
    t = pd.to_datetime(df["time"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    df = df.drop(columns="time").set_index(t).astype(float)
    df = df[~df.index.duplicated()].sort_index()
    mins = df.index.hour * 60 + df.index.minute
    return df[(mins >= 9 * 60 + 15) & (mins <= 15 * 60 + 25)]


def fetch_upstox(isin, start: dt.date, end: dt.date, chunk_days=27, pause=0.3):
    key = quote(f"NSE_EQ|{isin}", safe="")
    frames, cur_end = [], end
    while cur_end >= start:
        cur_start = max(start, cur_end - dt.timedelta(days=chunk_days))
        url = (f"https://api.upstox.com/v3/historical-candle/{key}/minutes/5/"
               f"{cur_end:%Y-%m-%d}/{cur_start:%Y-%m-%d}")
        js = _get_json(url)
        if js:
            frames.append(parse_upstox_candles(js.get("data", {}).get("candles", [])))
        cur_end = cur_start - dt.timedelta(days=1)
        time.sleep(pause)
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames).sort_index().loc[lambda d: ~d.index.duplicated()]


def fetch_yfinance(symbol):
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
    return df.tz_convert("Asia/Kolkata").tz_localize(None)


def load_data(symbol, isin, days, source):
    """Cached loader. Cache is reused if it already covers the requested range."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{symbol}_{source}.csv")
    end = dt.date.today()
    start = end - dt.timedelta(days=days)
    if os.path.exists(path):
        c = pd.read_csv(path, index_col=0, parse_dates=True)
        if not c.empty and c.index[0].date() <= start + dt.timedelta(days=7) \
                and c.index[-1].date() >= end - dt.timedelta(days=4):
            return c[c.index.date >= start]
    df = fetch_upstox(isin, start, end) if source == "upstox" else fetch_yfinance(symbol)
    if not df.empty:
        df.to_csv(path)
    return df


def synthetic_data(seed: int, days: int = 120) -> pd.DataFrame:
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


def build_report(tr: pd.DataFrame, p: Params, days_label: str) -> str:
    if tr.empty:
        return "No trades generated."
    tr = tr.sort_values("entry_time")
    wins, losses = tr[tr.net_pct > 0], tr[tr.net_pct <= 0]
    lsum = abs(losses.net_pct.sum())
    pf = wins.net_pct.sum() / lsum if lsum else float("inf")
    eq = tr.net_pct.cumsum()
    dd = (eq - eq.cummax()).min()
    monthly = tr.groupby(pd.to_datetime(tr.entry_time).dt.to_period("M")).net_pct.sum().round(2)
    sym = tr.groupby("symbol").net_pct.sum().sort_values()
    L = [
        "ORB INSIDE-CANDLE BACKTEST",
        f"Period: {tr.entry_time.min():%Y-%m-%d} to {tr.entry_time.max():%Y-%m-%d} ({days_label})",
        f"Rules: {p.min_inside}-{p.max_inside} inside candles, RR 1:{p.rr}, SL={p.sl_mode}, cost {p.cost_pct}%",
        "-" * 38,
        f"Trades        : {len(tr)} (Long {sum(tr.side=='LONG')}, Short {sum(tr.side=='SHORT')})",
        f"Win rate      : {len(wins)/len(tr)*100:.1f}%",
        f"Avg R / trade : {tr.r_mult.mean():.3f}",
        f"Avg net %/trd : {tr.net_pct.mean():.3f}%",
        f"Profit factor : {pf:.2f}",
        f"Total net %   : {tr.net_pct.sum():.2f}% (sum of trade returns)",
        f"Max drawdown  : {dd:.2f}%",
        f"Exits         : {tr.reason.value_counts().to_dict()}",
        f"Winning months: {(monthly>0).sum()}/{len(monthly)}",
        "",
        "Top 5 symbols : " + ", ".join(f"{s} {v:.1f}" for s, v in sym.tail(5)[::-1].items()),
        "Worst 5       : " + ", ".join(f"{s} {v:.1f}" for s, v in sym.head(5).items()),
    ]
    return "\n".join(L)


# --------------------------------------------------------------- TELEGRAM
def send_telegram(text, file_path=None):
    token, chat = os.environ.get("TG_TOKEN"), os.environ.get("TG_CHAT_ID")
    if not token or not chat:
        print("[telegram] TG_TOKEN / TG_CHAT_ID not set - skipping")
        return
    base = f"https://api.telegram.org/bot{token}"
    r = requests.post(f"{base}/sendMessage", data={"chat_id": chat, "text": text[:4000]}, timeout=30)
    print("[telegram] message:", r.status_code, r.json().get("description", "ok"))
    if file_path and os.path.exists(file_path):
        with open(file_path, "rb") as f:
            r = requests.post(f"{base}/sendDocument", data={"chat_id": chat, "caption": "All backtest trades"},
                              files={"document": f}, timeout=120)
        print("[telegram] file:", r.status_code, r.json().get("description", "ok"))


# ------------------------------------------------------------------- MAIN
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="synthetic data test")
    ap.add_argument("--symbols", nargs="*")
    ap.add_argument("--days", type=int, default=365, help="calendar days of history (max ~1700 for Upstox)")
    ap.add_argument("--source", choices=["upstox", "yfinance"], default="upstox")
    ap.add_argument("--rr", type=float, default=2.0)
    ap.add_argument("--sl", choices=["range", "mid"], default="range")
    ap.add_argument("--min-inside", type=int, default=2)
    ap.add_argument("--max-inside", type=int, default=4)
    ap.add_argument("--telegram", action="store_true")
    a = ap.parse_args()

    p = Params(min_inside=a.min_inside, max_inside=a.max_inside, rr=a.rr, sl_mode=a.sl)
    universe = get_nifty200() if not a.demo else FALLBACK
    if a.symbols:
        want = {s.upper() for s in a.symbols}
        universe = [(s, i) for s, i in universe if s.upper() in want]

    all_trades = []
    for k, (s, isin) in enumerate(universe, 1):
        try:
            df = synthetic_data(k) if a.demo else load_data(s, isin, a.days, a.source)
            if df.empty:
                print(f"[{k}/{len(universe)}] {s}: no data")
                continue
            t = backtest_symbol(s, df, p)
            all_trades += t
            print(f"[{k}/{len(universe)}] {s}: {len(df)//75} days, {len(t)} trades")
        except Exception as e:
            print(f"[{k}/{len(universe)}] {s}: error {e}")

    tr = pd.DataFrame(all_trades)
    if not tr.empty:
        tr.sort_values("entry_time").to_csv("backtest_trades.csv", index=False)
    text = build_report(tr, p, f"{a.days}d requested")
    print("\n" + text)
    if a.telegram:
        send_telegram(text, "backtest_trades.csv")


if __name__ == "__main__":
    main()
