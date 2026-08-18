#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import v2_datavision_validation as dv
import v2_maker_proxy as mp

PLACEMENT_LATENCY_MS = 75
PAPER_CAPITAL_USDC = 1000.0


@dataclass(frozen=True)
class V3Profile:
    name: str
    quote_usdc: float = 5.0
    half_spread_bps: float = 1.0
    lead_shift_bps_per_score: float = 0.0
    vol_mult: float = 0.0
    inventory_shift_bps: float = 0.75
    inventory_skew: bool = True
    exec_flow_floor: float = -1.1
    exec_mom_floor_bps: float = -999.0
    penetration_ticks: int = 1
    queue_ahead_multiple: float = 3.0
    friction_bps: float = 0.20
    inventory_cap_usdc: float = 20.0
    quote_life_sec: int = 1


def make_profiles() -> list[V3Profile]:
    profiles: list[V3Profile] = []
    for hs in (0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0):
        profiles.append(V3Profile(name=f"HB_S{hs:g}", half_spread_bps=hs))
    for hs in (0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0):
        profiles.append(V3Profile(name=f"LEAD_S{hs:g}", half_spread_bps=hs, lead_shift_bps_per_score=0.50))
    for hs in (0.75, 1.0, 1.25, 1.5, 2.0):
        profiles.append(V3Profile(name=f"LV_S{hs:g}", half_spread_bps=hs, lead_shift_bps_per_score=0.50, vol_mult=0.50))
    for hs in (0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0):
        profiles.append(V3Profile(name=f"TOX_S{hs:g}", half_spread_bps=hs, lead_shift_bps_per_score=0.50,
                                  vol_mult=0.50, exec_flow_floor=-0.25, exec_mom_floor_bps=-0.75))
    for hs in (1.0, 1.25, 1.5, 2.0, 2.5):
        profiles.append(V3Profile(name=f"TOX2_S{hs:g}", half_spread_bps=hs, lead_shift_bps_per_score=0.35,
                                  vol_mult=0.50, exec_flow_floor=0.0, exec_mom_floor_bps=-0.25))
    return profiles


def add_exec_features(frame: pd.DataFrame, exec_raw: pd.DataFrame) -> pd.DataFrame:
    f = frame.copy()
    ep = f["exec_px"].astype(float)
    f["exec_mom1_bps"] = np.log(ep / ep.shift(1)) * 1e4
    f["exec_vol60_bps"] = (np.log(ep / ep.shift(1)) * 1e4).rolling(60, min_periods=20).std(ddof=0)
    ea = dv.aggregate_lead(exec_raw)
    idx = f.index
    signed = ea["signed"].reindex(idx).fillna(0.0)
    absn = ea["absn"].reindex(idx).fillna(0.0)
    f["exec_flow2"] = signed.rolling(2, min_periods=1).sum() / absn.rolling(2, min_periods=1).sum().replace(0.0, np.nan)
    return f.replace([np.inf, -np.inf], np.nan)


def simulate_symbol(symbol, frame, exec_raw, signal_threshold, start_sec, end_sec, profile):
    f = frame.loc[(frame.index >= start_sec) & (frame.index <= end_sec)].copy()
    f = f[(f["score"].abs() >= signal_threshold) & (f["lead_age"] <= 2) & (f["exec_age"] <= 2)]
    f = f.dropna(subset=["score", "exec_px", "exec_flow2", "exec_mom1_bps", "exec_vol60_bps"])
    if f.empty:
        return {}, pd.DataFrame()
    raw = exec_raw[(exec_raw["ts"] >= start_sec * 1000) & (exec_raw["ts"] <= (end_sec + 35) * 1000)].copy()
    if raw.empty:
        return {}, pd.DataFrame()
    ts = raw["ts"].to_numpy(np.int64)
    px = raw["price"].to_numpy(float)
    qty = raw["qty"].to_numpy(float)
    bm = raw["is_buyer_maker"].to_numpy(bool)
    notion = px * qty
    tick = mp.infer_tick(px)
    if not np.isfinite(tick) or tick <= 0:
        return {}, pd.DataFrame()

    lots = deque()
    pos_qty = 0.0
    realized_gross = 0.0
    matched_entry_notional = 0.0
    holding_seconds = []
    fills = []
    quotes = 0
    max_inventory = 0.0

    for sec, row in f.iterrows():
        sec = int(sec)
        score = float(row["score"])
        side = 1 if score > 0 else -1
        p0 = float(row["exec_px"])
        inv_u = pos_qty * p0
        inv_ratio = float(np.clip(inv_u / profile.inventory_cap_usdc, -1.0, 1.0))

        if side * float(row["exec_flow2"]) < profile.exec_flow_floor:
            continue
        if side * float(row["exec_mom1_bps"]) < profile.exec_mom_floor_bps:
            continue

        size_mult = float(np.clip(1.0 - side * inv_ratio, 0.25, 1.75)) if profile.inventory_skew else 1.0
        quote_u = profile.quote_usdc * size_mult
        if side > 0 and inv_u + quote_u > profile.inventory_cap_usdc:
            quote_u = max(0.0, profile.inventory_cap_usdc - inv_u)
        elif side < 0 and inv_u - quote_u < -profile.inventory_cap_usdc:
            quote_u = max(0.0, profile.inventory_cap_usdc + inv_u)
        if quote_u < 1.0:
            continue

        lead_shift = profile.lead_shift_bps_per_score * score
        inv_shift = -profile.inventory_shift_bps * inv_ratio
        vol_component = profile.vol_mult * min(float(row["exec_vol60_bps"]), 8.0)
        half_spread = profile.half_spread_bps + max(0.0, vol_component)
        quote_shift_bps = lead_shift + inv_shift - side * half_spread
        quote_px = p0 * math.exp(quote_shift_bps / 1e4)
        quote_px = min(quote_px, p0 - tick) if side > 0 else max(quote_px, p0 + tick)
        if quote_px <= 0:
            continue
        quotes += 1

        active_ms = (sec + 1) * 1000 + PLACEMENT_LATENCY_MS
        expire_ms = active_ms + profile.quote_life_sec * 1000
        i0 = int(np.searchsorted(ts, active_ms, side="left"))
        i1 = int(np.searchsorted(ts, expire_ms, side="left"))
        if i1 <= i0:
            continue
        sl = slice(i0, i1)
        if side > 0:
            qualifying = bm[sl] & (px[sl] <= quote_px - profile.penetration_ticks * tick + tick * 1e-6)
        else:
            qualifying = (~bm[sl]) & (px[sl] >= quote_px + profile.penetration_ticks * tick - tick * 1e-6)
        loc = np.flatnonzero(qualifying)
        if len(loc) == 0:
            continue
        required = quote_u * (1.0 + profile.queue_ahead_multiple)
        c = np.cumsum(notion[sl][loc])
        k = int(np.searchsorted(c, required, side="left"))
        if k >= len(loc):
            continue
        j = i0 + int(loc[k])
        fill_ms = int(ts[j])
        fill_px = float(quote_px)
        q = quote_u / fill_px
        signed_q = side * q

        rem = abs(signed_q)
        while rem > 1e-15 and lots and np.sign(lots[0]["qty"]) != side:
            lot = lots[0]
            m = min(rem, abs(float(lot["qty"])))
            if lot["qty"] > 0 and side < 0:
                realized_gross += (fill_px - float(lot["px"])) * m
                matched_entry_notional += float(lot["px"]) * m
            elif lot["qty"] < 0 and side > 0:
                realized_gross += (float(lot["px"]) - fill_px) * m
                matched_entry_notional += float(lot["px"]) * m
            holding_seconds.append(max(0.0, (fill_ms - int(lot["ts"])) / 1000.0))
            lot["qty"] = float(lot["qty"]) - math.copysign(m, float(lot["qty"]))
            rem -= m
            if abs(float(lot["qty"])) < 1e-15:
                lots.popleft()
        if rem > 1e-15:
            lots.append({"qty": side * rem, "px": fill_px, "ts": fill_ms})
        pos_qty += signed_q
        max_inventory = max(max_inventory, abs(pos_qty * fill_px))

        marks = {}
        for h in (1, 5, 30):
            fp = mp.last_price_at(ts, px, fill_ms + h * 1000, 2000)
            marks[h] = None if fp is None else side * math.log(fp / fill_px) * 1e4
        fills.append({"symbol": symbol, "fill_ms": fill_ms, "bucket5m": fill_ms // 300000,
                      "side": "BUY" if side > 0 else "SELL", "score": score,
                      "exec_flow2": float(row["exec_flow2"]), "exec_mom1_bps": float(row["exec_mom1_bps"]),
                      "exec_vol60_bps": float(row["exec_vol60_bps"]), "half_spread_bps": half_spread,
                      "fill_px": fill_px, "quote_usdc": quote_u,
                      "mark1_bps": marks[1], "mark5_bps": marks[5], "mark30_bps": marks[30],
                      "net5_bps": None if marks[5] is None else marks[5] - profile.friction_bps,
                      "inventory_usdc": pos_qty * fill_px})

    ff = pd.DataFrame(fills)
    final_px = mp.last_price_at(ts, px, end_sec * 1000, 5000)
    unreal = 0.0
    if final_px is not None:
        for lot in lots:
            if lot["qty"] > 0:
                unreal += (final_px - float(lot["px"])) * float(lot["qty"])
            else:
                unreal += (float(lot["px"]) - final_px) * abs(float(lot["qty"]))
    fill_notional = float(ff["quote_usdc"].sum()) if not ff.empty else 0.0
    friction_cost = fill_notional * profile.friction_bps / 1e4
    mtm_net = realized_gross + unreal - friction_cost
    rt_net = realized_gross - (2.0 * profile.friction_bps / 1e4) * matched_entry_notional
    rt_bps = rt_net / matched_entry_notional * 1e4 if matched_entry_notional > 0 else float("nan")
    duration_days = max((end_sec - start_sec) / 86400.0, 1e-9)
    summary = {
        "symbol": symbol, "signal_threshold": signal_threshold, "tick_inferred": tick,
        "quotes": int(quotes), "fills": int(len(ff)), "fill_ratio": float(len(ff) / quotes) if quotes else 0.0,
        "maker_notional_usdc": fill_notional, "daily_maker_notional_usdc": fill_notional / duration_days,
        "capital_turnover_x_day_1000u": fill_notional / PAPER_CAPITAL_USDC / duration_days,
        "mark1_bps": float(ff["mark1_bps"].mean()) if not ff.empty else float("nan"),
        "mark5_bps": float(ff["mark5_bps"].mean()) if not ff.empty else float("nan"),
        "mark30_bps": float(ff["mark30_bps"].mean()) if not ff.empty else float("nan"),
        "net5_bps": float(ff["net5_bps"].mean()) if not ff.empty else float("nan"),
        "positive_net5_share": float((ff["net5_bps"] > 0).mean()) if not ff.empty else float("nan"),
        "roundtrip_net_bps": float(rt_bps), "mtm_net_pnl_usdc": float(mtm_net),
        "max_inventory_usdc": float(max_inventory),
        "avg_holding_sec": float(np.mean(holding_seconds)) if holding_seconds else float("nan"),
        "matched_entry_notional_usdc": float(matched_entry_notional)}
    return summary, ff


def pooled_profile(frames, raws, thresholds, period, profile, n_boot, seed, symbol_allow=None):
    rows, fills = [], []
    for sym in sorted(frames):
        if symbol_allow is not None and sym not in symbol_allow:
            continue
        s, f = simulate_symbol(sym, frames[sym], raws[sym], thresholds[sym], period[0], period[1], profile)
        if s: rows.append(s)
        if not f.empty: fills.append(f)
    rdf = pd.DataFrame(rows)
    fdf = pd.concat(fills, ignore_index=True) if fills else pd.DataFrame()
    boot = mp.block_bootstrap(fdf, n_boot, seed)
    maker = float(rdf["maker_notional_usdc"].sum()) if not rdf.empty else 0.0
    duration_days = max((period[1] - period[0]) / 86400.0, 1e-9)
    matched = float(rdf["matched_entry_notional_usdc"].sum()) if not rdf.empty else 0.0
    rt_pnl = 0.0
    if not rdf.empty:
        good = rdf[np.isfinite(rdf["roundtrip_net_bps"]) & (rdf["matched_entry_notional_usdc"] > 0)]
        rt_pnl = float(((good["roundtrip_net_bps"] / 1e4) * good["matched_entry_notional_usdc"]).sum())
    result = {"profile": profile.name, **asdict(profile),
              "symbols": int((rdf["fills"] > 0).sum()) if not rdf.empty else 0,
              "fills": int(len(fdf)), "maker_notional_usdc": maker,
              "daily_maker_notional_usdc": maker / duration_days,
              "capital_turnover_x_day_1000u": maker / PAPER_CAPITAL_USDC / duration_days,
              "mean_net5_bps": boot["mean_bps"], "lcb90_net5_bps": boot["lcb90_bps"],
              "bootstrap_blocks": boot["blocks"],
              "mark30_bps": float(fdf["mark30_bps"].mean()) if not fdf.empty else None,
              "roundtrip_net_bps": rt_pnl / matched * 1e4 if matched > 0 else None,
              "mtm_net_pnl_usdc": float(rdf["mtm_net_pnl_usdc"].sum()) if not rdf.empty else 0.0,
              "max_symbol_inventory_usdc": float(rdf["max_inventory_usdc"].max()) if not rdf.empty else 0.0,
              "positive_net5_share": float((fdf["net5_bps"] > 0).mean()) if not fdf.empty else None}
    return result, rdf, fdf


def pass_gate(r):
    return bool(r["fills"] >= 500 and r["symbols"] >= 5
                and r["mean_net5_bps"] is not None and r["mean_net5_bps"] >= 0.30
                and r["lcb90_net5_bps"] is not None and r["lcb90_net5_bps"] >= 0.10
                and r["roundtrip_net_bps"] is not None and r["roundtrip_net_bps"] > 0
                and r["mark30_bps"] is not None and r["mark30_bps"] >= 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-date", required=True); ap.add_argument("--end-date", required=True)
    ap.add_argument("--output-dir", default="v3_hb_optimized_out")
    ap.add_argument("--min-exec-trades", type=int, default=3000)
    ap.add_argument("--bootstrap", type=int, default=3000); ap.add_argument("--max-pairs", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    dates = mp.date_range(args.start_date, args.end_date)
    syms = dv.list_aggtrade_symbols(); sset = set(syms)
    bases = sorted({s[:-4] for s in syms if s.endswith("USDC") and (s[:-4] + "USDT") in sset})
    if args.max_pairs: bases = bases[:args.max_pairs]
    print(f"dates={dates} matched_pairs={len(bases)} V3=Hummingbot-inspired+USDT-lead paper_capital=1000USDC", flush=True)

    frames, raws, thresholds, skipped = {}, {}, {}, []
    t0 = time.time()
    for i, base in enumerate(bases, 1):
        lead_sym, exec_sym = base + "USDT", base + "USDC"
        print(f"[{i}/{len(bases)}] load {base}", flush=True)
        try:
            lead = mp.load_multi(lead_sym, dates); exe = mp.load_multi(exec_sym, dates)
        except Exception as e:
            skipped.append({"base": base, "reason": f"{type(e).__name__}: {e}"}); print(f"  SKIP {e}"); continue
        if len(exe) < args.min_exec_trades:
            skipped.append({"base": base, "reason": f"exec trades {len(exe)} < {args.min_exec_trades}"}); continue
        fr = dv.build_frame(dv.aggregate_lead(lead), dv.aggregate_exec(exe), 2)
        if len(fr) < 10800:
            skipped.append({"base": base, "reason": "insufficient aligned seconds"}); continue
        fr = add_exec_features(fr, exe)
        n = len(fr); i1, i2 = int(n * .60), int(n * .80)
        val = fr.iloc[i1:i2]
        thr = dv.choose_threshold(val, [0.20,0.30,0.40,0.50,0.60], max(50, int(len(val)*.003)))
        frames[exec_sym], raws[exec_sym], thresholds[exec_sym] = fr, exe, thr
        print(f"  OK seconds={len(fr):,} trades={len(exe):,} signal_thr={thr:.2f}", flush=True)
    if not frames: raise SystemExit("no eligible symbols")

    common_start = max(int(f.index.min()) for f in frames.values()); common_end = min(int(f.index.max()) for f in frames.values())
    span = common_end-common_start
    val_period = (common_start+int(span*.60), common_start+int(span*.80)); test_period = (val_period[1]+1, common_end-31)
    print(f"eligible={len(frames)} val_h={(val_period[1]-val_period[0])/3600:.2f} test_h={(test_period[1]-test_period[0])/3600:.2f}")

    profiles = make_profiles(); val_results=[]; val_per_symbol={}
    for p in profiles:
        print(f"VALIDATE {p.name}", flush=True)
        r, ps, _ = pooled_profile(frames, raws, thresholds, val_period, p, 500, 20260818)
        r["pass_gate"] = pass_gate(r); val_results.append(r); val_per_symbol[p.name]=ps
        print(json.dumps({k:r[k] for k in ("profile","fills","symbols","capital_turnover_x_day_1000u","mean_net5_bps","lcb90_net5_bps","mark30_bps","roundtrip_net_bps","pass_gate")}, ensure_ascii=False), flush=True)
    vr=pd.DataFrame(val_results); vr.to_csv(out/"validation_profiles.csv", index=False)
    passing=vr[vr.pass_gate==True]
    if not passing.empty:
        chosen_name=str(passing.sort_values("capital_turnover_x_day_1000u", ascending=False).iloc[0].profile)
        reason="max validation turnover among profiles passing EV/LCB/roundtrip/30s gates"
    else:
        chosen_name=str(vr.sort_values(["lcb90_net5_bps","mean_net5_bps","capital_turnover_x_day_1000u"], ascending=False).iloc[0].profile)
        reason="no profile passed; diagnostic profile with strongest validation statistical edge"
    chosen=next(p for p in profiles if p.name==chosen_name)
    print(f"CHOSEN_PROFILE={chosen_name} reason={reason}")

    vps=val_per_symbol[chosen_name].copy(); selected=set()
    if not vps.empty:
        eligible_sel=vps[(vps.fills>=100) & (vps.net5_bps>=0.30) & (vps.mark30_bps>=0)]
        selected=set(eligible_sel.symbol.tolist())
    print(f"VALIDATION_SELECTED_SYMBOLS={len(selected)} {sorted(selected)}")

    all_r, all_ps, all_ff = pooled_profile(frames, raws, thresholds, test_period, chosen, args.bootstrap, 20260819)
    all_r["pass_gate"] = pass_gate(all_r)
    sel_r, sel_ps, sel_ff = pooled_profile(frames, raws, thresholds, test_period, chosen, args.bootstrap, 20260820,
                                           symbol_allow=selected if selected else None)
    sel_r["pass_gate"] = pass_gate(sel_r) if selected else False; sel_r["selection_fallback_all"] = not bool(selected)
    final = sel_r if selected else all_r
    final.update({"status":"PASS_MAKER_PROXY" if final["pass_gate"] else "FAIL_MAKER_PROXY",
                  "chosen_profile":chosen_name, "selection_reason":reason,
                  "validation_selected_symbols":sorted(selected),
                  "oos_hours":(test_period[1]-test_period[0])/3600.0,
                  "matched_pairs_discovered":len(bases), "eligible_pairs":len(frames),
                  "paper_capital_usdc":PAPER_CAPITAL_USDC, "elapsed_sec":time.time()-t0,
                  "limitations":["Binance DataVision aggTrades proxy; no exact L2 price-time queue reconstruction.",
                                 "Quotes activate one full signal-second later plus 75ms synthetic latency; no same-second lookahead.",
                                 "Post-only is approximated by keeping quote at least one inferred tick behind last USDC trade.",
                                 "Fill requires aggressive trade-through by >=1 tick and synthetic queue-ahead volume.",
                                 "Maker fee assumed 0; 0.20 bps execution friction is subtracted from 5s markout.",
                                 "Validation selects profile and optional symbol universe; final test segment is untouched OOS.",
                                 "A proxy PASS is evidence for further L2/live-paper validation, not live-trading approval."]})

    all_ps.to_csv(out/"oos_all_symbols.csv", index=False); all_ff.to_csv(out/"oos_all_fills.csv", index=False)
    sel_ps.to_csv(out/"oos_selected_symbols.csv", index=False); sel_ff.to_csv(out/"oos_selected_fills.csv", index=False)
    vps.to_csv(out/"validation_per_symbol_chosen.csv", index=False); pd.DataFrame(skipped).to_csv(out/"skipped.csv", index=False)
    (out/"oos_all_summary.json").write_text(json.dumps(all_r, ensure_ascii=False, indent=2), encoding="utf-8")
    (out/"oos_final_summary.json").write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== V3 HUMMINGBOT-INSPIRED OPTIMIZED OOS FINAL ==="); print(json.dumps(final, ensure_ascii=False, indent=2))
    print("\n=== OOS ALL-SYMBOL CONTROL ==="); print(json.dumps(all_r, ensure_ascii=False, indent=2))
    z=sel_ps if selected else all_ps
    if not z.empty:
        cols=["symbol","fills","daily_maker_notional_usdc","net5_bps","mark30_bps","roundtrip_net_bps","mtm_net_pnl_usdc","max_inventory_usdc"]
        print("\n=== PER SYMBOL ==="); print(z[cols].sort_values("daily_maker_notional_usdc", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
