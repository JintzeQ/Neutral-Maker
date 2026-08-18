#!/usr/bin/env python3
from __future__ import annotations

# V2 live paper scan: public market data only; never sends orders.
import concurrent.futures as cf
import json
import math
import statistics
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any

BASE = "https://fapi.binance.com"
UA = "Neutral-Maker-V2-LiveScan/1.0"


def get_json(path: str, params: dict[str, Any] | None = None, timeout: int = 20):
    url = BASE + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def median(xs):
    ys = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return statistics.median(ys) if ys else 0.0


def mad(xs):
    m = median(xs)
    return median([abs(float(x) - m) for x in xs if x is not None and math.isfinite(float(x))])


def rz(x, xs):
    m = median(xs); d = mad(xs)
    return 0.0 if d <= 1e-12 else 0.67448975 * (float(x) - m) / d


def logret_bps(a, b):
    return math.log(a / b) * 1e4 if a > 0 and b > 0 else 0.0


def book_features(depth):
    bids = [(float(p), float(q)) for p, q in depth.get("bids", [])[:10]]
    asks = [(float(p), float(q)) for p, q in depth.get("asks", [])[:10]]
    if not bids or not asks:
        raise ValueError("empty book")
    bb, bq = bids[0]; ba, aq = asks[0]; mid = (bb + ba) / 2.0
    spread_bps = (ba - bb) / mid * 1e4 if mid > 0 else 0.0
    bid_n = sum(p * q for p, q in bids[:5]); ask_n = sum(p * q for p, q in asks[:5]); den = bid_n + ask_n
    obi5 = (bid_n - ask_n) / den if den > 0 else 0.0
    micro = (ba * bq + bb * aq) / (bq + aq) if bq + aq > 0 else mid
    micro_bps = (micro - mid) / mid * 1e4 if mid > 0 else 0.0
    return {"mid": mid, "spread_bps": spread_bps, "obi5": obi5, "micro_bps": micro_bps}


def agg_flow(symbol):
    rows = get_json("/fapi/v1/aggTrades", {"symbol": symbol, "limit": 500})
    signed = total = 0.0
    for r in rows:
        n = float(r["p"]) * float(r["q"])
        signed += -n if bool(r.get("m")) else n
        total += n
    return signed / total if total > 0 else 0.0


def kline_features(symbol):
    rows = get_json("/fapi/v1/klines", {"symbol": symbol, "interval": "1m", "limit": 10})
    closes = [float(r[4]) for r in rows]
    if len(closes) < 5:
        return {"mom3_bps": 0.0, "vol1m_bps": 0.0}
    rs = [logret_bps(closes[i], closes[i-1]) for i in range(1, len(closes))]
    return {"mom3_bps": logret_bps(closes[-1], closes[-4]), "vol1m_bps": statistics.pstdev(rs) if len(rs) > 1 else 0.0}


def scan_pair(base):
    usdc = base + "USDC"; usdt = base + "USDT"
    try:
        with cf.ThreadPoolExecutor(max_workers=4) as ex:
            f_du = ex.submit(get_json, "/fapi/v1/depth", {"symbol": usdt, "limit": 20})
            f_dc = ex.submit(get_json, "/fapi/v1/depth", {"symbol": usdc, "limit": 20})
            f_fl = ex.submit(agg_flow, usdt); f_kl = ex.submit(kline_features, usdt)
            bu = book_features(f_du.result()); bc = book_features(f_dc.result()); flow = f_fl.result(); kl = f_kl.result()
        return {"base": base, "usdc": usdc, "usdt": usdt, "flow": flow, "mom3_bps": kl["mom3_bps"], "vol1m_bps": kl["vol1m_bps"], "usdt_obi": bu["obi5"], "usdt_micro_bps": bu["micro_bps"], "usdc_obi": bc["obi5"], "usdc_micro_bps": bc["micro_bps"], "usdc_spread_bps": bc["spread_bps"], "basis_bps": logret_bps(bc["mid"], bu["mid"]), "error": None}
    except Exception as e:
        return {"base": base, "usdc": usdc, "usdt": usdt, "error": f"{type(e).__name__}: {e}"}


def classify(rows):
    good = [r for r in rows if not r.get("error")]
    if not good: return good
    cols = ["flow","mom3_bps","vol1m_bps","usdt_obi","usdt_micro_bps","usdc_obi","usdc_micro_bps","usdc_spread_bps","basis_bps"]
    pop = {c:[r[c] for r in good] for c in cols}
    for r in good:
        zflow=rz(r["flow"],pop["flow"]); zmom=rz(r["mom3_bps"],pop["mom3_bps"]); zobi_u=rz(r["usdt_obi"],pop["usdt_obi"]); zmicro_u=rz(r["usdt_micro_bps"],pop["usdt_micro_bps"])
        zobi_c=rz(r["usdc_obi"],pop["usdc_obi"]); zmicro_c=rz(r["usdc_micro_bps"],pop["usdc_micro_bps"]); zspr=rz(r["usdc_spread_bps"],pop["usdc_spread_bps"]); zbasis=rz(r["basis_bps"],pop["basis_bps"]); zvol=rz(r["vol1m_bps"],pop["vol1m_bps"])
        lead=0.35*zflow+0.30*zmom+0.25*zobi_u+0.10*zmicro_u; local=0.70*zobi_c+0.30*zmicro_c; direction=0.75*lead+0.25*local; disagreement=abs(lead-local)
        tox=0.35*max(zspr,0.0)+0.25*abs(zbasis)+0.25*max(zvol,0.0)+0.15*disagreement
        severe=(zspr>2.5 or abs(zbasis)>3.0 or zvol>3.0 or disagreement>3.0 or r["usdc_spread_bps"]>8.0)
        bid_q=direction-0.85*tox; ask_q=-direction-0.85*tox
        def gate(q):
            if severe: return "OFF" if q < 1.0 else "RETREAT"
            if q >= 0.55: return "ON"
            if q >= -0.75: return "RETREAT"
            return "OFF"
        r.update({"lead_score":lead,"local_score":local,"direction_score":direction,"toxicity_score":tox,"severe_toxic":severe,"bid_quality":bid_q,"ask_quality":ask_q,"bid_gate":gate(bid_q),"ask_gate":gate(ask_q)})
    return good


def main():
    t0=time.time(); exinfo=get_json("/fapi/v1/exchangeInfo"); symbols=exinfo.get("symbols",[])
    tradable={s.get("symbol"):s for s in symbols if s.get("status")=="TRADING" and s.get("contractType")=="PERPETUAL"}
    bases=sorted(set(s.get("baseAsset") for s in tradable.values() if (s.get("quoteAsset")=="USDC" or s.get("marginAsset")=="USDC") and s.get("baseAsset") and s.get("baseAsset")+"USDT" in tradable))
    print(f"matched_tradable_pairs={len(bases)} bases={bases}")
    rows=[]
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        futs={ex.submit(scan_pair,b):b for b in bases}
        for f in cf.as_completed(futs):
            r=f.result(); rows.append(r); print(json.dumps(r,ensure_ascii=False),flush=True)
    good=classify(rows); good.sort(key=lambda r:r["base"]); failed=[r for r in rows if r.get("error")]; n=len(good)
    bullish=sum(r["direction_score"]>0.55 for r in good); bearish=sum(r["direction_score"]<-0.55 for r in good); neutral=n-bullish-bearish; toxic=sum(bool(r["severe_toxic"]) for r in good)
    gates={"BID_ON":sum(r["bid_gate"]=="ON" for r in good),"BID_RETREAT":sum(r["bid_gate"]=="RETREAT" for r in good),"BID_OFF":sum(r["bid_gate"]=="OFF" for r in good),"ASK_ON":sum(r["ask_gate"]=="ON" for r in good),"ASK_RETREAT":sum(r["ask_gate"]=="RETREAT" for r in good),"ASK_OFF":sum(r["ask_gate"]=="OFF" for r in good)}
    pos=sum(r["direction_score"]>0 for r in good); neg=n-pos; dominant="BULLISH" if pos>=max(neg*1.5,1) else ("BEARISH" if neg>=max(pos*1.5,1) else "MIXED")
    strongest=sorted(good,key=lambda r:abs(r["direction_score"]),reverse=True)[:5]; most_toxic=sorted(good,key=lambda r:r["toxicity_score"],reverse=True)[:5]; worst_quality=sorted(good,key=lambda r:max(r["bid_quality"],r["ask_quality"]))[:5]
    summary={"timestamp_utc":datetime.now(timezone.utc).isoformat(),"matched_tradable_pairs":len(bases),"evaluated":n,"failed":len(failed),"direction_breadth":{"bullish_strong":bullish,"bearish_strong":bearish,"neutral":neutral,"positive_sign":pos,"negative_sign":neg,"dominant":dominant},"toxic_markets":toxic,"toxic_share":toxic/max(n,1),"gates":gates,"median_usdc_spread_bps":median([r["usdc_spread_bps"] for r in good]),"median_abs_basis_bps":median([abs(r["basis_bps"]) for r in good]),"median_vol1m_bps":median([r["vol1m_bps"] for r in good]),"strongest":[{k:r[k] for k in ["base","direction_score","lead_score","toxicity_score","bid_gate","ask_gate","usdc_spread_bps","basis_bps"]} for r in strongest],"most_toxic":[{k:r[k] for k in ["base","toxicity_score","severe_toxic","bid_gate","ask_gate","usdc_spread_bps","basis_bps","vol1m_bps"]} for r in most_toxic],"worst_quality":[{k:r[k] for k in ["base","bid_quality","ask_quality","toxicity_score","bid_gate","ask_gate"]} for r in worst_quality],"failed_rows":failed,"elapsed_sec":time.time()-t0,"method_note":"Cross-sectional robust-z snapshot; gates are V2 paper eligibility only, not live execution approval. Raw fills across symbols are not treated as independent statistical samples."}
    with open("v2_live_scan.json","w",encoding="utf-8") as f: json.dump({"summary":summary,"rows":good},f,ensure_ascii=False,indent=2)
    print("=== SUMMARY ==="); print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
