#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

S3_LIST = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
DV = "https://data.binance.vision/data/futures/um/daily/aggTrades"
UA = "V2-DataVision-Validation/1.0"


@dataclass
class SymbolResult:
    symbol: str
    lead_symbol: str
    test_seconds: int
    signal_count: int
    signal_coverage: float
    threshold: float
    m1_bps: float
    m5_bps: float
    m30_bps: float
    hit5: float
    lead_trades: int
    exec_trades: int
    lead_zip_mb: float
    exec_zip_mb: float
    status: str


def get_bytes(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def head_size(url: str, timeout: int = 30) -> int:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return int(r.headers.get("Content-Length") or 0)


def list_aggtrade_symbols() -> list[str]:
    prefix = "data/futures/um/daily/aggTrades/"
    marker = ""
    found: set[str] = set()
    while True:
        qs = {"delimiter": "/", "prefix": prefix, "max-keys": "1000"}
        if marker:
            qs["marker"] = marker
        root = ET.fromstring(get_bytes(S3_LIST + "?" + urllib.parse.urlencode(qs), 60))
        ns = ""
        if root.tag.startswith("{"):
            ns = root.tag.split("}")[0] + "}"
        for cp in root.findall(f"{ns}CommonPrefixes"):
            p = cp.findtext(f"{ns}Prefix") or ""
            if p.startswith(prefix):
                s = p[len(prefix):].strip("/")
                if s:
                    found.add(s.upper())
        truncated = (root.findtext(f"{ns}IsTruncated") or "false").lower() == "true"
        if not truncated:
            break
        marker = root.findtext(f"{ns}NextMarker") or ""
        if not marker:
            keys = [k.text or "" for k in root.findall(f"{ns}Contents/{ns}Key")]
            prefixes = [p.findtext(f"{ns}Prefix") or "" for p in root.findall(f"{ns}CommonPrefixes")]
            marker = max(keys + prefixes) if (keys or prefixes) else ""
        if not marker:
            break
    return sorted(found)


def csv_from_zip_bytes(raw: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise ValueError("zip contains no CSV")
        with zf.open(names[0]) as fh:
            df = pd.read_csv(
                fh,
                header=None,
                usecols=[1, 2, 5, 6],
                names=["price", "qty", "ts", "is_buyer_maker"],
                dtype={"price": "string", "qty": "string", "ts": "string", "is_buyer_maker": "string"},
                low_memory=False,
            )
    for c in ("price", "qty", "ts"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    bm = df["is_buyer_maker"].astype(str).str.lower().str.strip()
    df["is_buyer_maker"] = bm.map({"true": True, "false": False, "1": True, "0": False})
    df = df.dropna(subset=["price", "qty", "ts", "is_buyer_maker"])
    if df.empty:
        return df
    df["ts"] = df["ts"].astype(np.int64)
    med = float(df["ts"].median())
    if med > 1e15:
        df["ts"] = (df["ts"] // 1000).astype(np.int64)
    return df


def load_day(symbol: str, date: str) -> tuple[pd.DataFrame, float]:
    url = f"{DV}/{symbol}/{symbol}-aggTrades-{date}.zip"
    try:
        size = head_size(url)
    except Exception:
        size = 0
    raw = get_bytes(url, 180)
    mb = (size or len(raw)) / 1_048_576
    return csv_from_zip_bytes(raw), mb


def aggregate_lead(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    sec = (df["ts"].to_numpy(np.int64) // 1000).astype(np.int64)
    px = df["price"].to_numpy(float)
    qty = df["qty"].to_numpy(float)
    bm = df["is_buyer_maker"].to_numpy(bool)
    notional = px * qty
    signed = np.where(bm, -notional, notional)
    x = pd.DataFrame({"sec": sec, "px": px, "signed": signed, "absn": notional})
    return x.groupby("sec", sort=True).agg(px=("px", "last"), signed=("signed", "sum"), absn=("absn", "sum"))


def aggregate_exec(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    sec = (df["ts"].to_numpy(np.int64) // 1000).astype(np.int64)
    px = df["price"].to_numpy(float)
    return pd.DataFrame({"sec": sec, "px": px}).groupby("sec", sort=True).agg(px=("px", "last"))


def build_frame(lead: pd.DataFrame, exe: pd.DataFrame, max_stale: int) -> pd.DataFrame:
    if lead.empty or exe.empty:
        return pd.DataFrame()
    start = max(int(lead.index.min()), int(exe.index.min()))
    end = min(int(lead.index.max()), int(exe.index.max()))
    if end - start < 600:
        return pd.DataFrame()
    idx = pd.Index(np.arange(start, end + 1, dtype=np.int64), name="sec")
    lp_raw = lead["px"].reindex(idx)
    ep_raw = exe["px"].reindex(idx)
    lp = lp_raw.ffill(limit=max_stale)
    ep = ep_raw.ffill(limit=max_stale)
    ltrade_sec = pd.Series(np.where(lp_raw.notna(), idx.to_numpy(), np.nan), index=idx).ffill()
    etrade_sec = pd.Series(np.where(ep_raw.notna(), idx.to_numpy(), np.nan), index=idx).ffill()
    lead_age = idx.to_numpy() - ltrade_sec.to_numpy()
    exec_age = idx.to_numpy() - etrade_sec.to_numpy()
    signed = lead["signed"].reindex(idx).fillna(0.0)
    absn = lead["absn"].reindex(idx).fillna(0.0)
    flow = signed.rolling(2, min_periods=1).sum() / absn.rolling(2, min_periods=1).sum().replace(0.0, np.nan)
    lr1 = np.log(lp / lp.shift(1)) * 1e4
    mom2 = np.log(lp / lp.shift(2)) * 1e4
    vol60 = lr1.rolling(60, min_periods=20).std(ddof=0)
    mom_z = mom2 / (vol60 * math.sqrt(2)).replace(0.0, np.nan)
    score = 0.65 * flow.clip(-1, 1) + 0.35 * (mom_z.clip(-3, 3) / 3.0)
    out = pd.DataFrame({
        "lead_px": lp, "exec_px": ep, "lead_age": lead_age, "exec_age": exec_age,
        "flow2": flow, "mom2_bps": mom2, "score": score,
    }, index=idx)
    for h in (1, 5, 30):
        fut = ep.shift(-h)
        fut_age = pd.Series(exec_age, index=idx).shift(-h)
        r = np.log(fut / ep) * 1e4
        valid = (out["exec_age"] <= max_stale) & (fut_age <= max_stale) & (out["lead_age"] <= max_stale)
        out[f"r{h}"] = r.where(valid)
    return out.replace([np.inf, -np.inf], np.nan)


def choose_threshold(val: pd.DataFrame, candidates: list[float], min_signals: int) -> float:
    best_thr = candidates[0]
    best_obj = -1e18
    for thr in candidates:
        s = val[val["score"].abs() >= thr].dropna(subset=["r5", "score"])
        if len(s) < min_signals:
            continue
        m = float((np.sign(s["score"]) * s["r5"]).mean())
        coverage = len(s) / max(len(val), 1)
        obj = m * math.sqrt(max(coverage, 1e-6))
        if obj > best_obj:
            best_obj = obj
            best_thr = thr
    return best_thr


def evaluate_symbol(base: str, date: str, max_stale: int, min_exec_trades: int):
    lead_sym, exec_sym = base + "USDT", base + "USDC"
    try:
        lead_raw, lmb = load_day(lead_sym, date)
        exec_raw, emb = load_day(exec_sym, date)
    except urllib.error.HTTPError as e:
        return None, None, f"HTTP {e.code}"
    except Exception as e:
        return None, None, f"{type(e).__name__}: {e}"
    if len(exec_raw) < min_exec_trades:
        return None, None, f"exec trades {len(exec_raw):,} < {min_exec_trades:,}"
    fr = build_frame(aggregate_lead(lead_raw), aggregate_exec(exec_raw), max_stale)
    if len(fr) < 1800:
        return None, None, "insufficient aligned seconds"
    n = len(fr)
    i1, i2 = int(n * 0.60), int(n * 0.80)
    val, test = fr.iloc[i1:i2], fr.iloc[i2:].copy()
    threshold = choose_threshold(val, [0.20, 0.30, 0.40, 0.50, 0.60], max(30, int(len(val) * 0.005)))
    sig = test[test["score"].abs() >= threshold].dropna(subset=["r1", "r5", "r30", "score"]).copy()
    if len(sig) < 30:
        return None, None, f"only {len(sig)} OOS signals"
    sign = np.sign(sig["score"].to_numpy(float))
    for h in (1, 5, 30):
        sig[f"pm{h}"] = sign * sig[f"r{h}"].to_numpy(float)
    sig["base"] = base
    sig["bucket5m"] = (sig.index.to_numpy(np.int64) // 300).astype(np.int64)
    res = SymbolResult(
        symbol=exec_sym, lead_symbol=lead_sym, test_seconds=len(test), signal_count=len(sig),
        signal_coverage=len(sig) / max(len(test), 1), threshold=threshold,
        m1_bps=float(sig["pm1"].mean()), m5_bps=float(sig["pm5"].mean()), m30_bps=float(sig["pm30"].mean()),
        hit5=float((sig["pm5"] > 0).mean()), lead_trades=len(lead_raw), exec_trades=len(exec_raw),
        lead_zip_mb=lmb, exec_zip_mb=emb,
        status="PASS_SIGNAL" if float(sig["pm5"].mean()) > 0 else "FAIL_SIGNAL",
    )
    return res, sig[["base", "bucket5m", "pm1", "pm5", "pm30"]], None


def block_bootstrap_lcb(pooled: pd.DataFrame, n_boot: int, seed: int) -> dict:
    if pooled.empty:
        return {"mean_bps": None, "lcb90_bps": None, "blocks": 0, "signals": 0}
    b = pooled.groupby("bucket5m").agg(mean5=("pm5", "mean"), n=("pm5", "size"))
    vals = b["mean5"].to_numpy(float)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    n = len(vals)
    for i in range(n_boot):
        boots[i] = rng.choice(vals, size=n, replace=True).mean()
    return {"mean_bps": float(pooled["pm5"].mean()), "lcb90_bps": float(np.quantile(boots, 0.05)), "blocks": int(n), "signals": int(len(pooled))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--output-dir", default="v2_validation_out")
    ap.add_argument("--max-stale-sec", type=int, default=2)
    ap.add_argument("--min-exec-trades", type=int, default=1000)
    ap.add_argument("--max-pairs", type=int, default=0)
    ap.add_argument("--bootstrap", type=int, default=3000)
    args = ap.parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    syms = list_aggtrade_symbols()
    sset = set(syms)
    bases = sorted({s[:-4] for s in syms if s.endswith("USDC") and (s[:-4] + "USDT") in sset})
    if args.max_pairs > 0:
        bases = bases[:args.max_pairs]
    print(f"DataVision symbols={len(syms)} matched USDC/USDT bases={len(bases)} date={args.date}")
    results, pooled_parts, skipped = [], [], []
    t0 = time.time()
    for i, base in enumerate(bases, 1):
        print(f"[{i}/{len(bases)}] {base}...", flush=True)
        res, sig, err = evaluate_symbol(base, args.date, args.max_stale_sec, args.min_exec_trades)
        if res is None:
            skipped.append({"base": base, "reason": err or "unknown"})
            print(f"  SKIP {err}")
            continue
        results.append(res)
        pooled_parts.append(sig)
        print(f"  {res.symbol}: signals={res.signal_count:,} cov={res.signal_coverage:.1%} pm5={res.m5_bps:+.4f}bps pm30={res.m30_bps:+.4f}bps")
    rdf = pd.DataFrame([r.__dict__ for r in results])
    rdf.to_csv(outdir / "v2_signal_validation.csv", index=False)
    pd.DataFrame(skipped).to_csv(outdir / "v2_skipped.csv", index=False)
    pooled = pd.concat(pooled_parts, ignore_index=True) if pooled_parts else pd.DataFrame()
    boot = block_bootstrap_lcb(pooled, args.bootstrap, 20260818)
    if not rdf.empty:
        weighted = {
            "pm1_bps": float(np.average(rdf["m1_bps"], weights=rdf["signal_count"])),
            "pm5_bps": float(np.average(rdf["m5_bps"], weights=rdf["signal_count"])),
            "pm30_bps": float(np.average(rdf["m30_bps"], weights=rdf["signal_count"])),
            "hit5": float(np.average(rdf["hit5"], weights=rdf["signal_count"])),
        }
        positive = int((rdf["m5_bps"] > 0).sum())
    else:
        weighted = {"pm1_bps": None, "pm5_bps": None, "pm30_bps": None, "hit5": None}
        positive = 0
    summary = {
        "validation_type": "V2 lead-signal proxy validation (aggTrades only; not maker queue replay)",
        "date_utc": args.date,
        "matched_pairs_discovered": len(bases), "eligible_pairs_analyzed": len(results), "skipped_pairs": len(skipped),
        "total_oos_signals": int(rdf["signal_count"].sum()) if not rdf.empty else 0,
        "positive_5s_symbols": positive, "positive_5s_symbol_share": positive / len(results) if results else None,
        "weighted": weighted, "block_bootstrap_5m": boot,
        "signal_pass": bool(results and positive / len(results) >= 0.60 and boot.get("lcb90_bps") is not None and boot["lcb90_bps"] > 0.10),
        "elapsed_sec": round(time.time() - t0, 2),
        "notes": [
            "Score uses fixed 65% 2s aggressive-flow imbalance + 35% volatility-normalized 2s USDT momentum.",
            "Threshold is selected per symbol on the validation segment only; reported metrics are the final chronological 20% OOS segment.",
            "USDC future returns use aggregate-trade last prices with <=2s freshness at signal and horizon.",
            "5-minute wall-clock block bootstrap clusters simultaneous cross-symbol events.",
            "This tests whether USDT lead signals anticipate USDC moves/toxicity. It does NOT estimate actual maker fill probability, queue position, spread capture, or net maker EV."
        ],
    }
    (outdir / "v2_validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n=== V2 LEAD-SIGNAL PROXY RESULT ===")
    print(json.dumps(summary, indent=2))
    if not rdf.empty:
        cols = ["symbol", "signal_count", "signal_coverage", "threshold", "m1_bps", "m5_bps", "m30_bps", "hit5", "status"]
        print("\n=== PER SYMBOL ===")
        print(rdf.sort_values("m5_bps", ascending=False)[cols].to_string(index=False))


if __name__ == "__main__":
    main()
