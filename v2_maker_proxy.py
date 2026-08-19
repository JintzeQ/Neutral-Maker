#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

import v2_datavision_validation as dv


@dataclass(frozen=True)
class Profile:
    name: str
    quote_usdc: float
    penetration_ticks: int
    queue_ahead_multiple: float
    friction_bps: float
    inventory_cap_usdc: float
    quote_life_sec: int = 3


PROFILES = [
    Profile("SAFE", 5.0, 2, 3.0, 0.30, 14.0),
    Profile("MID", 5.0, 1, 3.0, 0.20, 14.0),
    Profile("BALANCED", 7.0, 1, 3.0, 0.20, 18.0),
    Profile("TURNOVER", 9.0, 1, 1.0, 0.20, 21.0),
]


def date_range(start: str, end: str) -> list[str]:
    return [d.strftime("%Y-%m-%d") for d in pd.date_range(start, end, freq="D")]


def load_multi(symbol: str, dates: list[str]) -> pd.DataFrame:
    parts = []
    for d in dates:
        x, _ = dv.load_day(symbol, d)
        if not x.empty:
            parts.append(x)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True).sort_values("ts", kind="stable").reset_index(drop=True)


def infer_tick(prices: np.ndarray) -> float:
    if len(prices) < 2:
        return float("nan")
    # Consecutive trade-price differences are enough to infer a conservative grid.
    p = np.round(prices.astype(float), 12)
    d = np.abs(np.diff(p))
    d = d[d > 1e-12]
    if len(d) == 0:
        return float("nan")
    x = float(np.quantile(d, 0.01))
    power = 10.0 ** math.floor(math.log10(x))
    m = max(1.0, round(x / power))
    return float(m * power)


def last_price_at(ts_ms: np.ndarray, px: np.ndarray, target_ms: int, max_age_ms: int = 2000) -> float | None:
    j = int(np.searchsorted(ts_ms, target_ms, side="right") - 1)
    if j < 0 or target_ms - int(ts_ms[j]) > max_age_ms:
        return None
    return float(px[j])


def block_bootstrap(fill_df: pd.DataFrame, n_boot: int, seed: int) -> dict:
    x = fill_df.dropna(subset=["net5_bps"]).copy()
    if x.empty:
        return {"mean_bps": None, "lcb90_bps": None, "blocks": 0, "fills": 0}
    b = x.groupby("bucket5m").agg(mean5=("net5_bps", "mean"), n=("net5_bps", "size"))
    vals = b["mean5"].to_numpy(float)
    if len(vals) == 0:
        return {"mean_bps": None, "lcb90_bps": None, "blocks": 0, "fills": int(len(x))}
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        boots[i] = rng.choice(vals, size=len(vals), replace=True).mean()
    return {
        "mean_bps": float(x["net5_bps"].mean()),
        "lcb90_bps": float(np.quantile(boots, 0.05)),
        "blocks": int(len(vals)),
        "fills": int(len(x)),
    }


def simulate_symbol(
    symbol: str,
    frame: pd.DataFrame,
    exec_raw: pd.DataFrame,
    threshold: float,
    start_sec: int,
    end_sec: int,
    profile: Profile,
) -> tuple[dict, pd.DataFrame]:
    f = frame.loc[(frame.index >= start_sec) & (frame.index <= end_sec)].copy()
    f = f[(f["score"].abs() >= threshold) & (f["lead_age"] <= 2) & (f["exec_age"] <= 2)]
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
    tick = infer_tick(px)
    if not np.isfinite(tick) or tick <= 0:
        return {}, pd.DataFrame()

    lots: deque[dict] = deque()
    pos_qty = 0.0
    realized_gross = 0.0
    matched_entry_notional = 0.0
    holding_seconds: list[float] = []
    fills: list[dict] = []
    quotes = 0
    max_inventory = 0.0
    next_free_sec = start_sec

    for sec, row in f.iterrows():
        sec = int(sec)
        if sec < next_free_sec:
            continue
        score = float(row["score"])
        side = 1 if score > 0 else -1  # +1 BUY safe-side, -1 SELL safe-side
        p0 = float(row["exec_px"])
        inv_u = pos_qty * p0
        if side > 0 and inv_u + profile.quote_usdc > profile.inventory_cap_usdc:
            continue
        if side < 0 and inv_u - profile.quote_usdc < -profile.inventory_cap_usdc:
            continue

        quote_px = p0 - tick if side > 0 else p0 + tick
        if quote_px <= 0:
            continue
        quotes += 1
        t0 = sec * 1000
        t1 = (sec + profile.quote_life_sec) * 1000
        i0 = int(np.searchsorted(ts, t0, side="left"))
        i1 = int(np.searchsorted(ts, t1, side="right"))
        if i1 <= i0:
            next_free_sec = sec + profile.quote_life_sec
            continue

        sl = slice(i0, i1)
        if side > 0:
            # BUY maker: only aggressive sells count, and touch does not fill.
            qualifying = bm[sl] & (px[sl] <= quote_px - profile.penetration_ticks * tick + tick * 1e-6)
        else:
            # SELL maker: only aggressive buys count, and touch does not fill.
            qualifying = (~bm[sl]) & (px[sl] >= quote_px + profile.penetration_ticks * tick - tick * 1e-6)
        loc = np.flatnonzero(qualifying)
        if len(loc) == 0:
            next_free_sec = sec + profile.quote_life_sec
            continue

        # Require trade-through volume to consume a synthetic queue ahead plus our order.
        required = profile.quote_usdc * (1.0 + profile.queue_ahead_multiple)
        c = np.cumsum(notion[sl][loc])
        k = int(np.searchsorted(c, required, side="left"))
        if k >= len(loc):
            next_free_sec = sec + profile.quote_life_sec
            continue
        j = i0 + int(loc[k])
        fill_ms = int(ts[j])
        fill_px = float(quote_px)
        q = profile.quote_usdc / fill_px
        signed_q = side * q

        # FIFO realized maker-to-maker PnL.
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
            fp = last_price_at(ts, px, fill_ms + h * 1000, 2000)
            marks[h] = None if fp is None else side * math.log(fp / fill_px) * 1e4

        fills.append({
            "symbol": symbol,
            "fill_ms": fill_ms,
            "bucket5m": fill_ms // 300000,
            "side": "BUY" if side > 0 else "SELL",
            "score": score,
            "fill_px": fill_px,
            "quote_usdc": profile.quote_usdc,
            "mark1_bps": marks[1],
            "mark5_bps": marks[5],
            "mark30_bps": marks[30],
            "net5_bps": None if marks[5] is None else marks[5] - profile.friction_bps,
            "inventory_usdc": pos_qty * fill_px,
        })
        next_free_sec = max(sec + 1, int(fill_ms // 1000) + 1)

    ff = pd.DataFrame(fills)
    final_px = last_price_at(ts, px, end_sec * 1000, 5000)
    unreal = 0.0
    if final_px is not None:
        for lot in lots:
            if lot["qty"] > 0:
                unreal += (final_px - float(lot["px"])) * float(lot["qty"])
            else:
                unreal += (float(lot["px"]) - final_px) * abs(float(lot["qty"]))
    fill_notional = len(ff) * profile.quote_usdc
    friction_cost = fill_notional * profile.friction_bps / 1e4
    mtm_net = realized_gross + unreal - friction_cost
    rt_net = realized_gross - (2.0 * profile.friction_bps / 1e4) * matched_entry_notional
    rt_bps = (rt_net / matched_entry_notional * 1e4) if matched_entry_notional > 0 else float("nan")
    duration_days = max((end_sec - start_sec) / 86400.0, 1e-9)
    summary = {
        "symbol": symbol,
        "threshold": threshold,
        "tick_inferred": tick,
        "quotes": int(quotes),
        "fills": int(len(ff)),
        "fill_ratio": float(len(ff) / quotes) if quotes else 0.0,
        "maker_notional_usdc": float(fill_notional),
        "daily_maker_notional_usdc": float(fill_notional / duration_days),
        "capital_turnover_x_day_1000u": float(fill_notional / 1000.0 / duration_days),
        "mark1_bps": float(ff["mark1_bps"].mean()) if not ff.empty else float("nan"),
        "mark5_bps": float(ff["mark5_bps"].mean()) if not ff.empty else float("nan"),
        "mark30_bps": float(ff["mark30_bps"].mean()) if not ff.empty else float("nan"),
        "net5_bps": float(ff["net5_bps"].mean()) if not ff.empty else float("nan"),
        "positive_net5_share": float((ff["net5_bps"] > 0).mean()) if not ff.empty else float("nan"),
        "roundtrip_net_bps": float(rt_bps),
        "mtm_net_pnl_usdc": float(mtm_net),
        "max_inventory_usdc": float(max_inventory),
        "avg_holding_sec": float(np.mean(holding_seconds)) if holding_seconds else float("nan"),
        "matched_entry_notional_usdc": float(matched_entry_notional),
    }
    return summary, ff


def pooled_profile(
    frames: dict[str, pd.DataFrame],
    raws: dict[str, pd.DataFrame],
    thresholds: dict[str, float],
    period: tuple[int, int],
    profile: Profile,
    n_boot: int,
    seed: int,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    rows, fills = [], []
    for sym in sorted(frames):
        s, f = simulate_symbol(sym, frames[sym], raws[sym], thresholds[sym], period[0], period[1], profile)
        if s:
            rows.append(s)
        if not f.empty:
            fills.append(f)
    rdf = pd.DataFrame(rows)
    fdf = pd.concat(fills, ignore_index=True) if fills else pd.DataFrame()
    boot = block_bootstrap(fdf, n_boot, seed)
    maker = float(rdf["maker_notional_usdc"].sum()) if not rdf.empty else 0.0
    duration_days = max((period[1] - period[0]) / 86400.0, 1e-9)
    matched = float(rdf["matched_entry_notional_usdc"].sum()) if not rdf.empty else 0.0
    # Weighted roundtrip PnL reconstructed from each symbol's bps and matched notional.
    rt_pnl = 0.0
    if not rdf.empty:
        good = rdf[np.isfinite(rdf["roundtrip_net_bps"]) & (rdf["matched_entry_notional_usdc"] > 0)]
        rt_pnl = float(((good["roundtrip_net_bps"] / 1e4) * good["matched_entry_notional_usdc"]).sum())
    result = {
        "profile": profile.name,
        **asdict(profile),
        "symbols": int((rdf["fills"] > 0).sum()) if not rdf.empty else 0,
        "fills": int(len(fdf)),
        "maker_notional_usdc": maker,
        "daily_maker_notional_usdc": maker / duration_days,
        "capital_turnover_x_day_1000u": maker / 1000.0 / duration_days,
        "mean_net5_bps": boot["mean_bps"],
        "lcb90_net5_bps": boot["lcb90_bps"],
        "bootstrap_blocks": boot["blocks"],
        "mark30_bps": float(fdf["mark30_bps"].mean()) if not fdf.empty else None,
        "roundtrip_net_bps": (rt_pnl / matched * 1e4) if matched > 0 else None,
        "mtm_net_pnl_usdc": float(rdf["mtm_net_pnl_usdc"].sum()) if not rdf.empty else 0.0,
        "max_symbol_inventory_usdc": float(rdf["max_inventory_usdc"].max()) if not rdf.empty else 0.0,
        "positive_net5_share": float((fdf["net5_bps"] > 0).mean()) if not fdf.empty else None,
    }
    return result, rdf, fdf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-date", required=True)
    ap.add_argument("--end-date", required=True)
    ap.add_argument("--output-dir", default="v2_maker_proxy_out")
    ap.add_argument("--min-exec-trades", type=int, default=3000)
    ap.add_argument("--bootstrap", type=int, default=3000)
    ap.add_argument("--max-pairs", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dates = date_range(args.start_date, args.end_date)
    syms = dv.list_aggtrade_symbols()
    sset = set(syms)
    bases = sorted({s[:-4] for s in syms if s.endswith("USDC") and (s[:-4] + "USDT") in sset})
    if args.max_pairs:
        bases = bases[: args.max_pairs]
    print(f"dates={dates} matched_pairs={len(bases)} paper_capital=1000USDC", flush=True)

    frames: dict[str, pd.DataFrame] = {}
    raws: dict[str, pd.DataFrame] = {}
    thresholds: dict[str, float] = {}
    skipped = []
    t0 = time.time()

    for i, base in enumerate(bases, 1):
        lead_sym, exec_sym = base + "USDT", base + "USDC"
        print(f"[{i}/{len(bases)}] load {base}", flush=True)
        try:
            lead = load_multi(lead_sym, dates)
            exe = load_multi(exec_sym, dates)
        except Exception as e:
            skipped.append({"base": base, "reason": f"{type(e).__name__}: {e}"})
            print(f"  SKIP {type(e).__name__}: {e}", flush=True)
            continue
        if len(exe) < args.min_exec_trades:
            skipped.append({"base": base, "reason": f"exec trades {len(exe)} < {args.min_exec_trades}"})
            print(f"  SKIP low trades {len(exe):,}", flush=True)
            continue
        fr = dv.build_frame(dv.aggregate_lead(lead), dv.aggregate_exec(exe), 2)
        if len(fr) < 10800:
            skipped.append({"base": base, "reason": "insufficient aligned seconds"})
            print("  SKIP insufficient alignment", flush=True)
            continue
        n = len(fr)
        i1, i2 = int(n * 0.60), int(n * 0.80)
        val = fr.iloc[i1:i2]
        thr = dv.choose_threshold(val, [0.20, 0.30, 0.40, 0.50, 0.60], max(50, int(len(val) * 0.003)))
        frames[exec_sym] = fr
        raws[exec_sym] = exe
        thresholds[exec_sym] = thr
        print(f"  OK seconds={len(fr):,} exec_trades={len(exe):,} threshold={thr:.2f}", flush=True)

    if not frames:
        raise SystemExit("no eligible symbols")

    common_start = max(int(f.index.min()) for f in frames.values())
    common_end = min(int(f.index.max()) for f in frames.values())
    span = common_end - common_start
    val_period = (common_start + int(span * 0.60), common_start + int(span * 0.80))
    test_period = (val_period[1] + 1, common_end - 31)
    print(f"eligible={len(frames)} val_hours={(val_period[1]-val_period[0])/3600:.2f} test_hours={(test_period[1]-test_period[0])/3600:.2f}", flush=True)

    val_results = []
    for p in PROFILES:
        print(f"VALIDATE profile={p.name}", flush=True)
        r, _, _ = pooled_profile(frames, raws, thresholds, val_period, p, 800, 20260818)
        r["pass_gate"] = bool(
            r["fills"] >= 500
            and r["symbols"] >= 5
            and r["mean_net5_bps"] is not None and r["mean_net5_bps"] >= 0.30
            and r["lcb90_net5_bps"] is not None and r["lcb90_net5_bps"] >= 0.10
            and r["roundtrip_net_bps"] is not None and r["roundtrip_net_bps"] > 0
            and r["mark30_bps"] is not None and r["mark30_bps"] >= 0
        )
        val_results.append(r)
        print(json.dumps(r, ensure_ascii=False), flush=True)

    vr = pd.DataFrame(val_results)
    passing = vr[vr["pass_gate"] == True]
    if not passing.empty:
        chosen_name = str(passing.sort_values("daily_maker_notional_usdc", ascending=False).iloc[0]["profile"])
        selection_reason = "max turnover among validation profiles passing EV/LCB/roundtrip/30s gates"
    else:
        chosen_name = str(vr.sort_values(["lcb90_net5_bps", "mean_net5_bps"], ascending=False).iloc[0]["profile"])
        selection_reason = "no validation profile passed all gates; chose strongest statistical profile for diagnostic OOS only"
    chosen = next(p for p in PROFILES if p.name == chosen_name)
    print(f"CHOSEN={chosen_name} reason={selection_reason}", flush=True)

    test_result, per_symbol, fills = pooled_profile(frames, raws, thresholds, test_period, chosen, args.bootstrap, 20260819)
    test_result["validation_selected"] = chosen_name
    test_result["selection_reason"] = selection_reason
    test_result["oos_hours"] = (test_period[1] - test_period[0]) / 3600.0
    test_result["matched_pairs_discovered"] = len(bases)
    test_result["eligible_pairs"] = len(frames)
    test_result["paper_capital_usdc"] = 1000.0
    test_result["pass_gate"] = bool(
        test_result["fills"] >= 500
        and test_result["symbols"] >= 5
        and test_result["mean_net5_bps"] is not None and test_result["mean_net5_bps"] >= 0.30
        and test_result["lcb90_net5_bps"] is not None and test_result["lcb90_net5_bps"] >= 0.10
        and test_result["roundtrip_net_bps"] is not None and test_result["roundtrip_net_bps"] > 0
        and test_result["mark30_bps"] is not None and test_result["mark30_bps"] >= 0
    )
    test_result["status"] = "PASS_MAKER_PROXY" if test_result["pass_gate"] else "FAIL_MAKER_PROXY"
    test_result["elapsed_sec"] = time.time() - t0
    test_result["limitations"] = [
        "DataVision aggTrades proxy; no exact L2 price-time queue reconstruction.",
        "Quote starts one inferred tick behind current execution last trade; touch never fills.",
        "Fill requires penetration plus aggressive traded notional sufficient to consume a synthetic queue ahead.",
        "0 maker-fee scenario with explicit execution-friction stress already subtracted.",
        "1000 USDC paper capital; per-symbol inventory caps keep theoretical aggregate inventory below capital.",
        "PASS here is a conservative maker-proxy gate, not live-trading approval.",
    ]

    vr.to_csv(out / "validation_profiles.csv", index=False)
    per_symbol.to_csv(out / "oos_per_symbol.csv", index=False)
    fills.to_csv(out / "oos_fills.csv", index=False)
    pd.DataFrame(skipped).to_csv(out / "skipped.csv", index=False)
    (out / "oos_summary.json").write_text(json.dumps(test_result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== V2 CONSERVATIVE MAKER PROXY OOS ===")
    print(json.dumps(test_result, ensure_ascii=False, indent=2))
    if not per_symbol.empty:
        cols = ["symbol", "fills", "fill_ratio", "daily_maker_notional_usdc", "net5_bps", "mark30_bps", "roundtrip_net_bps", "mtm_net_pnl_usdc", "max_inventory_usdc"]
        print("\n=== PER SYMBOL ===")
        print(per_symbol[cols].sort_values("daily_maker_notional_usdc", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
