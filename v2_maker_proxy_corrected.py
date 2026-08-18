#!/usr/bin/env python3
from __future__ import annotations

import math
from collections import deque

import numpy as np
import pandas as pd

import v2_maker_proxy as mp

# Correct the first proxy's timing: a 1-second signal is only actionable AFTER
# that second closes. Quotes therefore become active on the next second plus a
# small synthetic placement latency, and are refreshed every second.
mp.PROFILES = [
    mp.Profile("SAFE", 5.0, 2, 3.0, 0.30, 14.0, 1),
    mp.Profile("MID", 5.0, 1, 3.0, 0.20, 14.0, 1),
    mp.Profile("BALANCED", 7.0, 1, 3.0, 0.20, 18.0, 1),
    mp.Profile("TURNOVER", 9.0, 1, 1.0, 0.20, 21.0, 1),
]
PLACEMENT_LATENCY_MS = 75


def simulate_symbol_corrected(symbol, frame, exec_raw, threshold, start_sec, end_sec, profile):
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
        if side > 0 and inv_u + profile.quote_usdc > profile.inventory_cap_usdc:
            continue
        if side < 0 and inv_u - profile.quote_usdc < -profile.inventory_cap_usdc:
            continue

        quote_px = p0 - tick if side > 0 else p0 + tick
        if quote_px <= 0:
            continue
        quotes += 1

        # IMPORTANT: score/exec_px for second `sec` are only known when that
        # second ends. Do not let the quote fill earlier than sec+1.
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

        required = profile.quote_usdc * (1.0 + profile.queue_ahead_multiple)
        c = np.cumsum(notion[sl][loc])
        k = int(np.searchsorted(c, required, side="left"))
        if k >= len(loc):
            continue
        j = i0 + int(loc[k])
        fill_ms = int(ts[j])
        fill_px = float(quote_px)
        q = profile.quote_usdc / fill_px
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

    ff = pd.DataFrame(fills)
    final_px = mp.last_price_at(ts, px, end_sec * 1000, 5000)
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


mp.simulate_symbol = simulate_symbol_corrected

if __name__ == "__main__":
    mp.main()
