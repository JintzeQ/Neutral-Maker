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
FRICTION_BPS = 0.20
PAPER_CAPITAL_USDC = 1000.0
RETREAT_STEPS_BPS = (0.0, 0.5, 1.0, 2.0, 3.0)
MODEL_SPREADS_BPS = (0.5, 1.0, 2.0, 3.0, 4.0)


@dataclass(frozen=True)
class V3Config:
    name: str
    base_spread_bps: float
    vol_mult: float
    toxicity_mult: float
    inventory_skew_bps: float
    ev_gate_bps: float
    off_vulnerability: float
    quote_usdc: float
    queue_ahead_multiple: float
    inventory_cap_usdc: float


CONFIGS = [
    V3Config("DYN_LOOSE", 0.75, 0.25, 0.75, 0.35, -0.10, 1.10, 5.0, 3.0, 20.0),
    V3Config("DYN_GATE0", 0.75, 0.35, 1.00, 0.45, 0.00, 1.00, 5.0, 3.0, 20.0),
    V3Config("DYN_GATE10", 1.00, 0.40, 1.25, 0.50, 0.10, 0.95, 5.0, 3.0, 20.0),
    V3Config("DYN_GATE30", 1.00, 0.50, 1.50, 0.60, 0.30, 0.90, 5.0, 3.0, 20.0),
    V3Config("WIDE_GATE10", 1.50, 0.60, 1.50, 0.65, 0.10, 0.95, 5.0, 3.0, 20.0),
    V3Config("WIDE_GATE30", 1.50, 0.75, 2.00, 0.75, 0.30, 0.90, 5.0, 3.0, 20.0),
    V3Config("SAFE_GATE30", 2.00, 1.00, 2.50, 1.00, 0.30, 0.85, 5.0, 3.0, 18.0),
    V3Config("TURN_GATE10", 0.75, 0.25, 1.00, 0.40, 0.10, 0.95, 7.0, 1.0, 21.0),
    V3Config("TURN_GATE30", 1.00, 0.35, 1.25, 0.50, 0.30, 0.90, 7.0, 1.0, 21.0),
]


@dataclass
class RidgeModel:
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    train_rows: int
    train_symbols: int
    rmse_bps: float

    def predict(self, x: np.ndarray) -> np.ndarray:
        z = (x - self.mean) / self.scale
        return np.c_[np.ones(len(z)), z] @ self.coef


def add_features(fr: pd.DataFrame) -> pd.DataFrame:
    x = fr.copy()
    x["basis_bps"] = np.log(x["exec_px"] / x["lead_px"]) * 1e4
    x["basis_dev_bps"] = x["basis_bps"] - x["basis_bps"].rolling(300, min_periods=30).median()
    x["exec_mom1_bps"] = np.log(x["exec_px"] / x["exec_px"].shift(1)) * 1e4
    x["exec_vol30_bps"] = x["exec_mom1_bps"].rolling(30, min_periods=15).std(ddof=0)
    return x.replace([np.inf, -np.inf], np.nan)


def feature_vector(row: pd.Series, side: int, distance_bps: float) -> np.ndarray:
    score = float(row["score"])
    flow = float(row["flow2"]) if np.isfinite(row["flow2"]) else 0.0
    mom = float(row["mom2_bps"]) if np.isfinite(row["mom2_bps"]) else 0.0
    emom = float(row["exec_mom1_bps"]) if np.isfinite(row["exec_mom1_bps"]) else 0.0
    basis = float(row["basis_dev_bps"]) if np.isfinite(row["basis_dev_bps"]) else 0.0
    vol = float(row["exec_vol30_bps"]) if np.isfinite(row["exec_vol30_bps"]) else 0.0
    ss = side * score
    return np.array([
        ss,
        abs(score),
        side * flow,
        math.tanh(side * mom / 3.0),
        math.tanh(side * emom / 2.0),
        math.tanh(side * basis / 5.0),
        min(max(vol, 0.0), 10.0),
        distance_bps,
        distance_bps * ss,
        distance_bps * distance_bps,
    ], dtype=float)


def fit_ridge(records: pd.DataFrame, ridge: float = 8.0) -> RidgeModel:
    feats = [c for c in records.columns if c.startswith("x")]
    X = records[feats].to_numpy(float)
    y = records["mark5_bps"].to_numpy(float)
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale[scale < 1e-9] = 1.0
    Z = (X - mean) / scale
    A = np.c_[np.ones(len(Z)), Z]
    pen = np.eye(A.shape[1]) * ridge
    pen[0, 0] = 0.0
    coef = np.linalg.solve(A.T @ A + pen, A.T @ y)
    pred = A @ coef
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    return RidgeModel(mean=mean, scale=scale, coef=coef, train_rows=len(records),
                      train_symbols=int(records["symbol"].nunique()), rmse_bps=rmse)


def infer_lead_beta(frames: dict[str, pd.DataFrame], thresholds: dict[str, float], period: tuple[int, int]) -> float:
    xs, ys = [], []
    for sym, fr in frames.items():
        f = fr.loc[(fr.index >= period[0]) & (fr.index <= period[1])]
        f = f[(f["score"].abs() >= thresholds[sym]) & f["r1"].notna()]
        if len(f) == 0:
            continue
        # Cap each symbol so BTC/ETH do not dominate the pooled regression.
        if len(f) > 5000:
            f = f.iloc[np.linspace(0, len(f) - 1, 5000).astype(int)]
        xs.append(f["score"].to_numpy(float))
        ys.append(f["r1"].to_numpy(float))
    if not xs:
        return 0.0
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    den = float(np.dot(x, x))
    beta = float(np.dot(x, y) / den) if den > 1e-12 else 0.0
    return float(np.clip(beta, -2.0, 2.0))


def candidate_fill(
    ts: np.ndarray,
    px: np.ndarray,
    bm: np.ndarray,
    notion: np.ndarray,
    side: int,
    quote_px: float,
    quote_usdc: float,
    tick: float,
    active_ms: int,
    life_ms: int,
    queue_multiple: float,
) -> tuple[int, float] | None:
    expire_ms = active_ms + life_ms
    i0 = int(np.searchsorted(ts, active_ms, side="left"))
    i1 = int(np.searchsorted(ts, expire_ms, side="left"))
    if i1 <= i0:
        return None
    sl = slice(i0, i1)
    # Conservative: touch does not fill. Require at least one inferred tick of trade-through.
    if side > 0:
        qual = bm[sl] & (px[sl] <= quote_px - tick + tick * 1e-6)
    else:
        qual = (~bm[sl]) & (px[sl] >= quote_px + tick - tick * 1e-6)
    loc = np.flatnonzero(qual)
    if len(loc) == 0:
        return None
    required = quote_usdc * (1.0 + queue_multiple)
    cs = np.cumsum(notion[sl][loc])
    k = int(np.searchsorted(cs, required, side="left"))
    if k >= len(loc):
        return None
    j = i0 + int(loc[k])
    return int(ts[j]), float(quote_px)


def collect_training_records(
    symbol: str,
    fr: pd.DataFrame,
    raw: pd.DataFrame,
    threshold: float,
    period: tuple[int, int],
    max_rows_symbol: int = 2500,
) -> pd.DataFrame:
    f = fr.loc[(fr.index >= period[0]) & (fr.index <= period[1])].copy()
    f = f[(f["score"].abs() >= threshold) & (f["lead_age"] <= 2) & (f["exec_age"] <= 2)]
    if f.empty:
        return pd.DataFrame()
    # Deterministic thinning keeps training cost bounded and avoids symbol dominance.
    if len(f) > 4500:
        f = f.iloc[np.linspace(0, len(f) - 1, 4500).astype(int)]
    rr = raw[(raw["ts"] >= period[0] * 1000) & (raw["ts"] <= (period[1] + 10) * 1000)]
    if rr.empty:
        return pd.DataFrame()
    ts = rr["ts"].to_numpy(np.int64)
    px = rr["price"].to_numpy(float)
    bm = rr["is_buyer_maker"].to_numpy(bool)
    notion = px * rr["qty"].to_numpy(float)
    tick = mp.infer_tick(px)
    if not np.isfinite(tick) or tick <= 0:
        return pd.DataFrame()
    rows = []
    for sec, row in f.iterrows():
        sec = int(sec)
        p0 = float(row["exec_px"])
        active_ms = (sec + 1) * 1000 + PLACEMENT_LATENCY_MS
        for side in (1, -1):
            for dist in MODEL_SPREADS_BPS:
                qp = p0 * math.exp((-side * dist) / 1e4)
                qp = min(qp, p0 - tick) if side > 0 else max(qp, p0 + tick)
                actual_dist = -side * math.log(qp / p0) * 1e4
                hit = candidate_fill(ts, px, bm, notion, side, qp, 5.0, tick, active_ms, 1000, 1.0)
                if hit is None:
                    continue
                fill_ms, fill_px = hit
                fp5 = mp.last_price_at(ts, px, fill_ms + 5000, 2000)
                if fp5 is None:
                    continue
                mark5 = side * math.log(fp5 / fill_px) * 1e4
                fv = feature_vector(row, side, actual_dist)
                d = {f"x{i}": float(v) for i, v in enumerate(fv)}
                d.update({"symbol": symbol, "mark5_bps": mark5})
                rows.append(d)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    if len(out) > max_rows_symbol:
        out = out.iloc[np.linspace(0, len(out) - 1, max_rows_symbol).astype(int)]
    return out


def choose_quote(
    row: pd.Series,
    side: int,
    p0: float,
    tick: float,
    inventory_usdc: float,
    cfg: V3Config,
    model: RidgeModel,
    lead_beta: float,
) -> tuple[float, float, float, float] | None:
    score = float(row["score"])
    vulnerability = max(0.0, -side * score)
    if vulnerability >= cfg.off_vulnerability:
        return None
    vol = float(row["exec_vol30_bps"]) if np.isfinite(row["exec_vol30_bps"]) else 0.0
    vol = float(np.clip(vol, 0.0, 10.0))
    inv_norm = float(np.clip(inventory_usdc / cfg.inventory_cap_usdc, -1.0, 1.0))
    # Hummingbot/Avellaneda-style reservation center: lead fair-value shift plus inventory penalty.
    center_shift_bps = lead_beta * score - cfg.inventory_skew_bps * inv_norm
    center = p0 * math.exp(center_shift_bps / 1e4)
    base_half = cfg.base_spread_bps + cfg.vol_mult * vol + cfg.toxicity_mult * vulnerability
    for extra in RETREAT_STEPS_BPS:
        half = max(0.05, base_half + extra)
        qp = center * math.exp((-side * half) / 1e4)
        # AggTrades cannot prove BBO. Enforce at least one tick away from last execution price.
        qp = min(qp, p0 - tick) if side > 0 else max(qp, p0 + tick)
        if qp <= 0:
            continue
        dist = -side * math.log(qp / p0) * 1e4
        pred_mark5 = float(model.predict(feature_vector(row, side, dist)[None, :])[0])
        pred_net5 = pred_mark5 - FRICTION_BPS
        if pred_net5 >= cfg.ev_gate_bps:
            size_mult = float(np.clip(1.0 - side * inv_norm, 0.0, 1.5))
            quote_u = cfg.quote_usdc * size_mult
            if quote_u < 1.0:
                return None
            return qp, quote_u, pred_net5, dist
    return None


def fifo_apply(lots: deque, side: int, qty: float, fill_px: float, fill_ms: int) -> tuple[float, float, list[float]]:
    realized = 0.0
    matched = 0.0
    holds = []
    rem = abs(qty)
    while rem > 1e-15 and lots and np.sign(lots[0]["qty"]) != side:
        lot = lots[0]
        m = min(rem, abs(float(lot["qty"])))
        if lot["qty"] > 0 and side < 0:
            realized += (fill_px - float(lot["px"])) * m
            matched += float(lot["px"]) * m
        elif lot["qty"] < 0 and side > 0:
            realized += (float(lot["px"]) - fill_px) * m
            matched += float(lot["px"]) * m
        holds.append(max(0.0, (fill_ms - int(lot["ts"])) / 1000.0))
        lot["qty"] = float(lot["qty"]) - math.copysign(m, float(lot["qty"]))
        rem -= m
        if abs(float(lot["qty"])) < 1e-15:
            lots.popleft()
    if rem > 1e-15:
        lots.append({"qty": side * rem, "px": fill_px, "ts": fill_ms})
    return realized, matched, holds


def simulate_symbol(
    symbol: str,
    fr: pd.DataFrame,
    raw: pd.DataFrame,
    threshold: float,
    period: tuple[int, int],
    cfg: V3Config,
    model: RidgeModel,
    lead_beta: float,
) -> tuple[dict, pd.DataFrame]:
    f = fr.loc[(fr.index >= period[0]) & (fr.index <= period[1])].copy()
    f = f[(f["score"].abs() >= threshold) & (f["lead_age"] <= 2) & (f["exec_age"] <= 2)]
    if f.empty:
        return {}, pd.DataFrame()
    rr = raw[(raw["ts"] >= period[0] * 1000) & (raw["ts"] <= (period[1] + 35) * 1000)]
    if rr.empty:
        return {}, pd.DataFrame()
    ts = rr["ts"].to_numpy(np.int64)
    px = rr["price"].to_numpy(float)
    bm = rr["is_buyer_maker"].to_numpy(bool)
    notion = px * rr["qty"].to_numpy(float)
    tick = mp.infer_tick(px)
    if not np.isfinite(tick) or tick <= 0:
        return {}, pd.DataFrame()

    lots = deque()
    pos_qty = 0.0
    realized_gross = 0.0
    matched_notional = 0.0
    holding = []
    fills = []
    quotes = 0
    rejected_ev = 0
    max_inventory = 0.0

    for sec, row in f.iterrows():
        sec = int(sec)
        p0 = float(row["exec_px"])
        inv_u = pos_qty * p0
        pending = []
        for side in (1, -1):
            qspec = choose_quote(row, side, p0, tick, inv_u, cfg, model, lead_beta)
            if qspec is None:
                rejected_ev += 1
                continue
            qp, quote_u, pred_net5, dist = qspec
            if side > 0 and inv_u + quote_u > cfg.inventory_cap_usdc:
                continue
            if side < 0 and inv_u - quote_u < -cfg.inventory_cap_usdc:
                continue
            quotes += 1
            active_ms = (sec + 1) * 1000 + PLACEMENT_LATENCY_MS
            hit = candidate_fill(ts, px, bm, notion, side, qp, quote_u, tick, active_ms, 1000, cfg.queue_ahead_multiple)
            if hit is not None:
                pending.append((hit[0], side, hit[1], quote_u, pred_net5, dist, float(row["score"])))

        # Both maker sides can be live. Apply any fills in event-time order.
        pending.sort(key=lambda z: z[0])
        for fill_ms, side, fill_px, quote_u, pred_net5, dist, score in pending:
            q = quote_u / fill_px
            inv_before = pos_qty * fill_px
            # Orders were already resting, but reject pathological overshoot in the conservative simulator.
            if side > 0 and inv_before + quote_u > cfg.inventory_cap_usdc * 1.10:
                continue
            if side < 0 and inv_before - quote_u < -cfg.inventory_cap_usdc * 1.10:
                continue
            rg, mn, hs = fifo_apply(lots, side, q, fill_px, fill_ms)
            realized_gross += rg
            matched_notional += mn
            holding.extend(hs)
            pos_qty += side * q
            max_inventory = max(max_inventory, abs(pos_qty * fill_px))
            marks = {}
            for h in (1, 5, 30):
                fp = mp.last_price_at(ts, px, fill_ms + h * 1000, 2000)
                marks[h] = None if fp is None else side * math.log(fp / fill_px) * 1e4
            fills.append({
                "symbol": symbol,
                "fill_ms": fill_ms,
                "bucket5m": fill_ms // 300000,
                "side": "BUY" if side > 0 else "SELL",
                "score": score,
                "fill_px": fill_px,
                "quote_usdc": quote_u,
                "distance_bps": dist,
                "pred_net5_bps": pred_net5,
                "mark1_bps": marks[1],
                "mark5_bps": marks[5],
                "mark30_bps": marks[30],
                "net5_bps": None if marks[5] is None else marks[5] - FRICTION_BPS,
                "inventory_usdc": pos_qty * fill_px,
            })

    ff = pd.DataFrame(fills)
    final_px = mp.last_price_at(ts, px, period[1] * 1000, 5000)
    unreal = 0.0
    if final_px is not None:
        for lot in lots:
            if lot["qty"] > 0:
                unreal += (final_px - float(lot["px"])) * float(lot["qty"])
            else:
                unreal += (float(lot["px"]) - final_px) * abs(float(lot["qty"]))
    fill_notional = float(ff["quote_usdc"].sum()) if not ff.empty else 0.0
    friction_cost = fill_notional * FRICTION_BPS / 1e4
    mtm_net = realized_gross + unreal - friction_cost
    rt_net = realized_gross - (2.0 * FRICTION_BPS / 1e4) * matched_notional
    rt_bps = rt_net / matched_notional * 1e4 if matched_notional > 0 else float("nan")
    days = max((period[1] - period[0]) / 86400.0, 1e-9)
    summary = {
        "symbol": symbol,
        "quotes": int(quotes),
        "fills": int(len(ff)),
        "fill_ratio": float(len(ff) / quotes) if quotes else 0.0,
        "maker_notional_usdc": fill_notional,
        "daily_maker_notional_usdc": fill_notional / days,
        "capital_turnover_x_day_1000u": fill_notional / PAPER_CAPITAL_USDC / days,
        "mark1_bps": float(ff["mark1_bps"].mean()) if not ff.empty else float("nan"),
        "mark5_bps": float(ff["mark5_bps"].mean()) if not ff.empty else float("nan"),
        "mark30_bps": float(ff["mark30_bps"].mean()) if not ff.empty else float("nan"),
        "net5_bps": float(ff["net5_bps"].mean()) if not ff.empty else float("nan"),
        "pred_net5_bps": float(ff["pred_net5_bps"].mean()) if not ff.empty else float("nan"),
        "roundtrip_net_bps": float(rt_bps),
        "mtm_net_pnl_usdc": float(mtm_net),
        "max_inventory_usdc": float(max_inventory),
        "avg_holding_sec": float(np.mean(holding)) if holding else float("nan"),
        "matched_entry_notional_usdc": float(matched_notional),
        "rejected_or_off_quotes": int(rejected_ev),
    }
    return summary, ff


def pooled(
    frames: dict[str, pd.DataFrame],
    raws: dict[str, pd.DataFrame],
    thresholds: dict[str, float],
    period: tuple[int, int],
    cfg: V3Config,
    model: RidgeModel,
    lead_beta: float,
    bootstrap: int,
    seed: int,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    rows, fills = [], []
    for sym in sorted(frames):
        s, f = simulate_symbol(sym, frames[sym], raws[sym], thresholds[sym], period, cfg, model, lead_beta)
        if s:
            rows.append(s)
        if not f.empty:
            fills.append(f)
    rdf = pd.DataFrame(rows)
    fdf = pd.concat(fills, ignore_index=True) if fills else pd.DataFrame()
    boot = mp.block_bootstrap(fdf, bootstrap, seed) if not fdf.empty else {"mean_bps": None, "lcb90_bps": None, "blocks": 0, "fills": 0}
    maker = float(rdf["maker_notional_usdc"].sum()) if not rdf.empty else 0.0
    days = max((period[1] - period[0]) / 86400.0, 1e-9)
    matched = float(rdf["matched_entry_notional_usdc"].sum()) if not rdf.empty else 0.0
    rt_pnl = 0.0
    if not rdf.empty:
        good = rdf[np.isfinite(rdf["roundtrip_net_bps"]) & (rdf["matched_entry_notional_usdc"] > 0)]
        if not good.empty:
            rt_pnl = float(((good["roundtrip_net_bps"] / 1e4) * good["matched_entry_notional_usdc"]).sum())
    result = {
        **asdict(cfg),
        "symbols": int((rdf["fills"] > 0).sum()) if not rdf.empty else 0,
        "fills": int(len(fdf)),
        "maker_notional_usdc": maker,
        "daily_maker_notional_usdc": maker / days,
        "capital_turnover_x_day_1000u": maker / PAPER_CAPITAL_USDC / days,
        "mean_net5_bps": boot["mean_bps"],
        "lcb90_net5_bps": boot["lcb90_bps"],
        "bootstrap_blocks": boot["blocks"],
        "mark30_bps": float(fdf["mark30_bps"].mean()) if not fdf.empty else None,
        "roundtrip_net_bps": rt_pnl / matched * 1e4 if matched > 0 else None,
        "mtm_net_pnl_usdc": float(rdf["mtm_net_pnl_usdc"].sum()) if not rdf.empty else 0.0,
        "max_symbol_inventory_usdc": float(rdf["max_inventory_usdc"].max()) if not rdf.empty else 0.0,
        "prediction_calibration_gap_bps": float((fdf["net5_bps"] - fdf["pred_net5_bps"]).mean()) if not fdf.empty else None,
    }
    return result, rdf, fdf


def pass_gate(r: dict) -> bool:
    return bool(
        r["fills"] >= 500
        and r["symbols"] >= 5
        and r["mean_net5_bps"] is not None and r["mean_net5_bps"] >= 0.30
        and r["lcb90_net5_bps"] is not None and r["lcb90_net5_bps"] >= 0.10
        and r["roundtrip_net_bps"] is not None and r["roundtrip_net_bps"] > 0
        and r["mark30_bps"] is not None and r["mark30_bps"] >= 0
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-date", default="2026-08-12")
    ap.add_argument("--end-date", default="2026-08-17")
    ap.add_argument("--output-dir", default="v3_out")
    ap.add_argument("--min-exec-trades", type=int, default=5000)
    ap.add_argument("--bootstrap", type=int, default=3000)
    ap.add_argument("--max-pairs", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dates = mp.date_range(args.start_date, args.end_date)
    syms = dv.list_aggtrade_symbols()
    sset = set(syms)
    bases = sorted({s[:-4] for s in syms if s.endswith("USDC") and (s[:-4] + "USDT") in sset})
    if args.max_pairs:
        bases = bases[:args.max_pairs]
    print(f"V3 dates={dates} matched_pairs={len(bases)} paper_capital={PAPER_CAPITAL_USDC:.0f}USDC", flush=True)

    frames, raws, thresholds = {}, {}, {}
    skipped = []
    t0 = time.time()
    for i, base in enumerate(bases, 1):
        lead_sym, exec_sym = base + "USDT", base + "USDC"
        print(f"[{i}/{len(bases)}] load {base}", flush=True)
        try:
            lead = mp.load_multi(lead_sym, dates)
            exe = mp.load_multi(exec_sym, dates)
        except Exception as e:
            skipped.append({"base": base, "reason": f"{type(e).__name__}: {e}"})
            print(f"  SKIP {type(e).__name__}: {e}", flush=True)
            continue
        if len(exe) < args.min_exec_trades:
            skipped.append({"base": base, "reason": f"exec trades {len(exe)} < {args.min_exec_trades}"})
            print(f"  SKIP low trades {len(exe):,}", flush=True)
            continue
        fr = add_features(dv.build_frame(dv.aggregate_lead(lead), dv.aggregate_exec(exe), 2))
        if len(fr) < 86400:
            skipped.append({"base": base, "reason": "insufficient aligned seconds"})
            print("  SKIP insufficient alignment", flush=True)
            continue
        frames[exec_sym] = fr
        raws[exec_sym] = exe
        print(f"  OK seconds={len(fr):,} exec_trades={len(exe):,}", flush=True)

    if not frames:
        raise SystemExit("no eligible symbols")
    common_start = max(int(f.index.min()) for f in frames.values())
    common_end = min(int(f.index.max()) for f in frames.values()) - 31
    span = common_end - common_start
    train_end = common_start + int(span * 2 / 3)
    val_end = common_start + int(span * 5 / 6)
    train_period = (common_start, train_end)
    val_period = (train_end + 1, val_end)
    test_period = (val_end + 1, common_end)
    print(f"eligible={len(frames)} train_h={(train_end-common_start)/3600:.1f} val_h={(val_end-train_end)/3600:.1f} test_h={(common_end-val_end)/3600:.1f}", flush=True)

    # Thresholds are learned ONLY from train directional predictiveness.
    for sym, fr in frames.items():
        tr = fr.loc[(fr.index >= train_period[0]) & (fr.index <= train_period[1])]
        thresholds[sym] = dv.choose_threshold(tr, [0.20, 0.30, 0.40, 0.50, 0.60], max(100, int(len(tr) * 0.002)))

    lead_beta = infer_lead_beta(frames, thresholds, train_period)
    print(f"lead_beta_bps_per_score={lead_beta:+.4f}", flush=True)

    # Build a pooled, symbol-balanced fill-conditioned markout model on TRAIN only.
    train_parts = []
    for i, sym in enumerate(sorted(frames), 1):
        rec = collect_training_records(sym, frames[sym], raws[sym], thresholds[sym], train_period)
        if not rec.empty:
            train_parts.append(rec)
        print(f"MODEL [{i}/{len(frames)}] {sym} rows={len(rec):,}", flush=True)
    if not train_parts:
        raise SystemExit("no training fill records")
    train_records = pd.concat(train_parts, ignore_index=True)
    model = fit_ridge(train_records)
    print(json.dumps({"model_rows": model.train_rows, "model_symbols": model.train_symbols, "model_rmse_bps": model.rmse_bps}, ensure_ascii=False), flush=True)

    # Select quoting/risk parameters on validation only.
    val_rows = []
    for cfg in CONFIGS:
        print(f"VALIDATE {cfg.name}", flush=True)
        r, _, _ = pooled(frames, raws, thresholds, val_period, cfg, model, lead_beta, 800, 20260820)
        r["pass_gate"] = pass_gate(r)
        val_rows.append(r)
        print(json.dumps(r, ensure_ascii=False), flush=True)
    vr = pd.DataFrame(val_rows)
    passing = vr[vr["pass_gate"] == True]
    if not passing.empty:
        chosen_name = str(passing.sort_values("daily_maker_notional_usdc", ascending=False).iloc[0]["name"])
        reason = "max validation turnover among configs passing EV/LCB/roundtrip/30s gates"
    else:
        chosen_name = str(vr.sort_values(["lcb90_net5_bps", "mean_net5_bps"], ascending=False).iloc[0]["name"])
        reason = "no validation config passed; diagnostic choice by strongest LCB then mean EV"
    chosen = next(c for c in CONFIGS if c.name == chosen_name)
    print(f"CHOSEN={chosen_name} reason={reason}", flush=True)

    # One-shot research OOS. Note: 2026-08-17 aggregate behavior was previously inspected in V2,
    # so this is useful evidence but not a pristine final holdout for V3.
    test_result, per_symbol, fills = pooled(frames, raws, thresholds, test_period, chosen, model, lead_beta, args.bootstrap, 20260821)
    test_result["validation_selected"] = chosen_name
    test_result["selection_reason"] = reason
    test_result["train_hours"] = (train_period[1] - train_period[0]) / 3600.0
    test_result["validation_hours"] = (val_period[1] - val_period[0]) / 3600.0
    test_result["test_hours"] = (test_period[1] - test_period[0]) / 3600.0
    test_result["eligible_pairs"] = len(frames)
    test_result["paper_capital_usdc"] = PAPER_CAPITAL_USDC
    test_result["lead_beta_bps_per_score"] = lead_beta
    test_result["model_train_rows"] = model.train_rows
    test_result["model_train_symbols"] = model.train_symbols
    test_result["model_train_rmse_bps"] = model.rmse_bps
    test_result["pass_gate"] = pass_gate(test_result)
    test_result["status"] = "PASS_V3_RESEARCH_OOS" if test_result["pass_gate"] else "FAIL_V3_RESEARCH_OOS"
    test_result["elapsed_sec"] = time.time() - t0
    test_result["limitations"] = [
        "Binance DataVision aggTrades proxy; no exact L2 price-time queue reconstruction or true BBO spread.",
        "Quotes activate only after the signal second closes plus 75ms synthetic placement latency.",
        "Touch never fills; fill requires one inferred tick of trade-through and synthetic queue depletion.",
        "Fill-conditioned ridge model is trained only on the train segment; quoting configs are selected only on validation.",
        "The final 2026-08-17 regime was previously observed at aggregate V2 level, so this V3 test is research OOS, not a pristine final holdout.",
        "A future unseen day and live paper L2 validation are required before any live-trading approval.",
    ]

    vr.to_csv(out / "v3_validation_configs.csv", index=False)
    train_records.to_csv(out / "v3_model_training_sample.csv", index=False)
    per_symbol.to_csv(out / "v3_oos_per_symbol.csv", index=False)
    fills.to_csv(out / "v3_oos_fills.csv", index=False)
    pd.DataFrame(skipped).to_csv(out / "v3_skipped.csv", index=False)
    (out / "v3_oos_summary.json").write_text(json.dumps(test_result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== V3 HUMMINGBOT-STYLE FILL-CONDITIONED MAKER ===")
    print(json.dumps(test_result, ensure_ascii=False, indent=2))
    if not per_symbol.empty:
        cols = ["symbol", "fills", "fill_ratio", "daily_maker_notional_usdc", "net5_bps", "mark30_bps", "roundtrip_net_bps", "mtm_net_pnl_usdc", "max_inventory_usdc"]
        print("\n=== PER SYMBOL ===")
        print(per_symbol[cols].sort_values("daily_maker_notional_usdc", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
