#!/usr/bin/env python3
import argparse, gzip, json, math, statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

PROFILES = [
    {"name": "PP025", "target_bps": 0.25, "queue_mult": 1.0, "reprice_ms": 3000},
    {"name": "PP050", "target_bps": 0.50, "queue_mult": 1.0, "reprice_ms": 3000},
    {"name": "PP100", "target_bps": 1.00, "queue_mult": 1.0, "reprice_ms": 3000},
    {"name": "PP200", "target_bps": 2.00, "queue_mult": 1.0, "reprice_ms": 3000},
    {"name": "PP050_Q2", "target_bps": 0.50, "queue_mult": 2.0, "reprice_ms": 3000},
    {"name": "PP100_Q2", "target_bps": 1.00, "queue_mult": 2.0, "reprice_ms": 3000},
    {"name": "PP050_SLOW", "target_bps": 0.50, "queue_mult": 1.0, "reprice_ms": 8000},
    {"name": "PP100_SLOW", "target_bps": 1.00, "queue_mult": 1.0, "reprice_ms": 8000},
]
COST_STRESS_BPS = [0.0, 0.10, 0.25, 0.50, 1.00]

@dataclass
class Order:
    side: str
    price: float
    qty: float
    queue_ahead: float | None
    activate_ms: int
    placed_ms: int

class State:
    def __init__(self):
        self.bid = self.ask = None
        self.bid_qty = self.ask_qty = 0.0
        self.mid = self.last_mid = None
        self.order_bid = self.order_ask = None
        self.pending_bid = self.pending_ask = None
        self.last_reprice = {"BUY": -10**18, "SELL": -10**18}
        self.inv_qty = 0.0
        self.avg_entry = None
        self.entry_ms = None
        self.cash = 0.0
        self.fills = []
        self.cycles = []
        self.max_abs_inv_usdc = 0.0
        self.inv_area = 0.0
        self.last_event_ms = None


def load_events(path):
    events = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("type") != "EVENT":
                continue
            d = obj["data"]
            if d.get("e") not in ("bookTicker", "trade"):
                continue
            events.append((int(obj["recv_ts_ms"]), d))
    events.sort(key=lambda x: x[0])
    return events


def bps(a, b):
    return (b - a) / ((a + b) / 2.0) * 1e4


def replay(events, profile, start_ms, end_ms, quote_usdc=5.0, latency_ms=75):
    states = defaultdict(State)
    qm = profile["queue_mult"]
    target_bps = profile["target_bps"]
    reprice_ms = profile["reprice_ms"]

    def update_inv_area(st, now):
        if st.last_event_ms is not None and st.last_mid is not None:
            dt = max(0, now - st.last_event_ms)
            st.inv_area += abs(st.inv_qty * st.last_mid) * dt
        st.last_event_ms = now
        if st.last_mid is not None:
            st.max_abs_inv_usdc = max(st.max_abs_inv_usdc, abs(st.inv_qty * st.last_mid))

    def desired(st, side):
        if st.mid is None:
            return None, 0.0
        flat = abs(st.inv_qty * st.mid) < 1e-7
        if flat:
            px = st.bid if side == "BUY" else st.ask
            return px, quote_usdc / st.mid
        if st.inv_qty > 0:
            if side == "BUY":
                return None, 0.0
            floor = st.avg_entry * (1.0 + target_bps / 1e4)
            px = max(st.ask, floor)
            qty = min(st.inv_qty, quote_usdc / st.mid)
            return px, qty
        if side == "SELL":
            return None, 0.0
        ceil = st.avg_entry * (1.0 - target_bps / 1e4)
        px = min(st.bid, ceil)
        qty = min(-st.inv_qty, quote_usdc / st.mid)
        return px, qty

    def deactivate_side(st, side):
        if side == "BUY":
            st.pending_bid = None
            st.order_bid = None
        else:
            st.pending_ask = None
            st.order_ask = None

    def schedule(st, side, now):
        px, qty = desired(st, side)
        if px is None or qty <= 1e-15:
            deactivate_side(st, side)
            return
        cur = st.order_bid if side == "BUY" else st.order_ask
        pend = st.pending_bid if side == "BUY" else st.pending_ask
        if cur is not None and abs(cur.price - px) <= max(1e-12, abs(px) * 1e-12):
            return
        if pend is not None and abs(pend.price - px) <= max(1e-12, abs(px) * 1e-12):
            return
        if now - st.last_reprice[side] < reprice_ms:
            return
        at_bbo = (side == "BUY" and abs(px - st.bid) <= max(1e-12, abs(px) * 1e-12)) or \
                 (side == "SELL" and abs(px - st.ask) <= max(1e-12, abs(px) * 1e-12))
        queue = (st.bid_qty if side == "BUY" else st.ask_qty) * qm if at_bbo else None
        od = Order(side, px, qty, queue, now + latency_ms, now)
        if side == "BUY":
            st.pending_bid = od
        else:
            st.pending_ask = od
        st.last_reprice[side] = now

    def activate(st, now):
        for side in ("BUY", "SELL"):
            p = st.pending_bid if side == "BUY" else st.pending_ask
            if p is None or now < p.activate_ms or st.bid is None or st.ask is None:
                continue
            post_only_ok = (side == "BUY" and p.price < st.ask) or (side == "SELL" and p.price > st.bid)
            if post_only_ok:
                if side == "BUY":
                    st.order_bid = p
                else:
                    st.order_ask = p
            if side == "BUY":
                st.pending_bid = None
            else:
                st.pending_ask = None

    def record_cycle(st, now, buy_px, sell_px, qty, entry_ms):
        notional = ((buy_px + sell_px) / 2.0) * qty
        pnl = (sell_px - buy_px) * qty
        st.cycles.append({
            "ts_ms": now,
            "buy_px": buy_px,
            "sell_px": sell_px,
            "qty": qty,
            "notional": notional,
            "gross_pnl": pnl,
            "gross_bps": pnl / notional * 1e4 if notional else 0.0,
            "hold_ms": max(0, now - entry_ms) if entry_ms is not None else None,
        })

    def execute_fill(st, od, qty, now):
        qty = min(max(0.0, qty), od.qty)
        if qty <= 1e-15:
            return
        side, px = od.side, od.price
        st.fills.append({"ts_ms": now, "side": side, "price": px, "qty": qty, "notional": px * qty})
        old_inv = st.inv_qty
        old_entry = st.avg_entry
        old_entry_ms = st.entry_ms
        if side == "BUY":
            st.cash -= px * qty
            if old_inv < -1e-15:
                matched = min(qty, -old_inv)
                record_cycle(st, now, px, old_entry, matched, old_entry_ms)
                new_inv = old_inv + qty
                if new_inv < -1e-15:
                    st.inv_qty = new_inv
                elif new_inv > 1e-15:
                    st.inv_qty = new_inv
                    st.avg_entry = px
                    st.entry_ms = now
                else:
                    st.inv_qty = 0.0
                    st.avg_entry = None
                    st.entry_ms = None
            else:
                new_inv = old_inv + qty
                if old_inv > 1e-15 and old_entry is not None:
                    st.avg_entry = (old_entry * old_inv + px * qty) / new_inv
                else:
                    st.avg_entry = px
                    st.entry_ms = now
                st.inv_qty = new_inv
        else:
            st.cash += px * qty
            if old_inv > 1e-15:
                matched = min(qty, old_inv)
                record_cycle(st, now, old_entry, px, matched, old_entry_ms)
                new_inv = old_inv - qty
                if new_inv > 1e-15:
                    st.inv_qty = new_inv
                elif new_inv < -1e-15:
                    st.inv_qty = new_inv
                    st.avg_entry = px
                    st.entry_ms = now
                else:
                    st.inv_qty = 0.0
                    st.avg_entry = None
                    st.entry_ms = None
            else:
                short_abs = -old_inv
                new_abs = short_abs + qty
                if short_abs > 1e-15 and old_entry is not None:
                    st.avg_entry = (old_entry * short_abs + px * qty) / new_abs
                else:
                    st.avg_entry = px
                    st.entry_ms = now
                st.inv_qty = old_inv - qty
        od.qty -= qty

    for now, d in events:
        if now < start_ms:
            continue
        if now >= end_ms:
            break
        sym = d.get("s")
        if not sym:
            continue
        st = states[sym]
        update_inv_area(st, now)
        activate(st, now)
        typ = d.get("e")
        if typ == "bookTicker":
            st.bid = float(d["b"]); st.ask = float(d["a"])
            st.bid_qty = float(d["B"]); st.ask_qty = float(d["A"])
            st.mid = st.last_mid = (st.bid + st.ask) / 2.0
            # If a resting shadow order is now crossed by the observed BBO, it would have executed.
            if st.order_bid is not None and st.ask <= st.order_bid.price:
                execute_fill(st, st.order_bid, st.order_bid.qty, now)
                st.order_bid = None
            if st.order_ask is not None and st.bid >= st.order_ask.price:
                execute_fill(st, st.order_ask, st.order_ask.qty, now)
                st.order_ask = None
            # When an off-BBO order becomes BBO, initialize conservative queue-ahead as if we were last.
            if st.order_bid is not None and st.order_bid.queue_ahead is None and abs(st.order_bid.price - st.bid) <= max(1e-12, abs(st.bid)*1e-12):
                st.order_bid.queue_ahead = st.bid_qty * qm
            if st.order_ask is not None and st.order_ask.queue_ahead is None and abs(st.order_ask.price - st.ask) <= max(1e-12, abs(st.ask)*1e-12):
                st.order_ask.queue_ahead = st.ask_qty * qm
            schedule(st, "BUY", now)
            schedule(st, "SELL", now)
        elif typ == "trade":
            px = float(d["p"]); tq = float(d["q"]); buyer_is_maker = bool(d["m"])
            # m=True => buyer was maker => aggressive seller hit bids.
            if st.order_bid is not None and buyer_is_maker:
                od = st.order_bid
                if px < od.price:
                    execute_fill(st, od, od.qty, now)
                    st.order_bid = None
                elif abs(px - od.price) <= max(1e-12, abs(px)*1e-12) and od.queue_ahead is not None:
                    used = min(tq, od.queue_ahead)
                    od.queue_ahead -= used
                    residual = tq - used
                    if residual > 1e-15:
                        execute_fill(st, od, min(residual, od.qty), now)
                    if od.qty <= 1e-15:
                        st.order_bid = None
            # m=False => buyer was taker => aggressive buyer lifted asks.
            tq = float(d["q"])
            if st.order_ask is not None and not buyer_is_maker:
                od = st.order_ask
                if px > od.price:
                    execute_fill(st, od, od.qty, now)
                    st.order_ask = None
                elif abs(px - od.price) <= max(1e-12, abs(px)*1e-12) and od.queue_ahead is not None:
                    used = min(tq, od.queue_ahead)
                    od.queue_ahead -= used
                    residual = tq - used
                    if residual > 1e-15:
                        execute_fill(st, od, min(residual, od.qty), now)
                    if od.qty <= 1e-15:
                        st.order_ask = None
            schedule(st, "BUY", now)
            schedule(st, "SELL", now)

    duration_ms = max(1, end_ms - start_ms)
    total_fills = sum(len(st.fills) for st in states.values())
    total_cycles = sum(len(st.cycles) for st in states.values())
    maker_notional = sum(x["notional"] for st in states.values() for x in st.fills)
    cycle_notional = sum(x["notional"] for st in states.values() for x in st.cycles)
    gross_cycle_pnl = sum(x["gross_pnl"] for st in states.values() for x in st.cycles)
    gross_cycle_bps = gross_cycle_pnl / cycle_notional * 1e4 if cycle_notional else float("nan")
    mtm_gross = 0.0
    max_inv = 0.0
    mean_inv_num = 0.0
    holds = []
    for st in states.values():
        if st.last_mid is not None:
            mtm_gross += st.cash + st.inv_qty * st.last_mid
        max_inv = max(max_inv, st.max_abs_inv_usdc)
        mean_inv_num += st.inv_area
        holds.extend([c["hold_ms"] / 1000.0 for c in st.cycles if c["hold_ms"] is not None])
    mean_abs_inv = mean_inv_num / duration_ms / max(1, len(states))
    turnover = maker_notional / (duration_ms / 86400000.0) / 1000.0
    cost = {}
    for c in COST_STRESS_BPS:
        matched_cost = cycle_notional * 2.0 * c / 1e4
        all_fill_cost = maker_notional * c / 1e4
        net_cycle = gross_cycle_pnl - matched_cost
        cost[str(c)] = {
            "roundtrip_net_bps": net_cycle / cycle_notional * 1e4 if cycle_notional else float("nan"),
            "mtm_net_pnl_usdc": mtm_gross - all_fill_cost,
        }
    be_cost = gross_cycle_bps / 2.0 if math.isfinite(gross_cycle_bps) else float("nan")
    per_symbol = []
    for sym, st in sorted(states.items()):
        cn = sum(c["notional"] for c in st.cycles)
        cp = sum(c["gross_pnl"] for c in st.cycles)
        mn = sum(f["notional"] for f in st.fills)
        mtm = st.cash + (st.inv_qty * st.last_mid if st.last_mid is not None else 0.0)
        per_symbol.append({
            "symbol": sym,
            "fills": len(st.fills),
            "cycles": len(st.cycles),
            "gross_roundtrip_bps": cp / cn * 1e4 if cn else None,
            "turnover_x_day_1000u": mn / (duration_ms / 86400000.0) / 1000.0,
            "mtm_gross_pnl_usdc": mtm,
            "end_inventory_usdc": st.inv_qty * st.last_mid if st.last_mid is not None else 0.0,
            "max_abs_inventory_usdc": st.max_abs_inv_usdc,
        })
    return {
        "profile": profile,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "hours": duration_ms / 3600000.0,
        "symbols": len(states),
        "fills": total_fills,
        "cycles": total_cycles,
        "maker_notional_usdc": maker_notional,
        "turnover_x_day_1000u": turnover,
        "gross_roundtrip_bps": gross_cycle_bps,
        "gross_cycle_pnl_usdc": gross_cycle_pnl,
        "mtm_gross_pnl_usdc": mtm_gross,
        "break_even_per_fill_cost_bps": be_cost,
        "max_symbol_inventory_usdc": max_inv,
        "mean_abs_inventory_usdc": mean_abs_inv,
        "median_hold_sec": statistics.median(holds) if holds else None,
        "cost_stress": cost,
        "per_symbol": per_symbol,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output-dir", default="v6_pingpong_out")
    ap.add_argument("--quote-usdc", type=float, default=5.0)
    ap.add_argument("--latency-ms", type=int, default=75)
    args = ap.parse_args()
    events = load_events(args.input)
    if len(events) < 100:
        raise SystemExit("not enough events")
    t0, t1 = events[0][0], events[-1][0]
    split = t0 + int((t1 - t0) * 0.60)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    validation = []
    for p in PROFILES:
        r = replay(events, p, t0, split, args.quote_usdc, args.latency_ms)
        validation.append(r)
        print("VALIDATION", json.dumps({k:v for k,v in r.items() if k != "per_symbol"}, allow_nan=True))
    eligible = [r for r in validation if r["cycles"] >= 10 and r["mtm_gross_pnl_usdc"] >= 0 and r["gross_roundtrip_bps"] > 0]
    if eligible:
        chosen = max(eligible, key=lambda r: (r["turnover_x_day_1000u"], r["gross_roundtrip_bps"]))
        reason = "positive gross cycle EV + nonnegative MTM on validation; max turnover"
    else:
        chosen = max(validation, key=lambda r: ((r["mtm_gross_pnl_usdc"] >= 0), r["gross_roundtrip_bps"] if math.isfinite(r["gross_roundtrip_bps"]) else -1e9, r["turnover_x_day_1000u"]))
        reason = "no validation profile passed; selected strongest diagnostic profile"
    test = replay(events, chosen["profile"], split, t1 + 1, args.quote_usdc, args.latency_ms)
    summary = {
        "status": "PILOT_PASS" if test["cycles"] >= 10 and test["gross_roundtrip_bps"] > 0 and test["mtm_gross_pnl_usdc"] >= 0 else "PILOT_FAIL",
        "selection_reason": reason,
        "chosen_profile": chosen["profile"],
        "capture_hours": (t1 - t0) / 3600000.0,
        "validation": {k:v for k,v in chosen.items() if k != "per_symbol"},
        "oos": test,
        "limitations": [
            "Direct Binance Futures bookTicker + raw trade capture; no directional signal used.",
            "BBO displayed size is used as queue-ahead only when shadow quote is at BBO; no full L2 reconstruction.",
            "Off-BBO inventory-recycling quotes only fill on strict price-through, or after becoming BBO and then depleting a conservative queue.",
            "This is a short pilot, not production evidence. Longer multi-regime live-paper capture is required before deployment.",
            "Maker fees are not assumed; cost-stress table reports net EV under several per-fill cost assumptions.",
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")
    (out / "validation_all.json").write_text(json.dumps(validation, indent=2, allow_nan=True), encoding="utf-8")
    print("\n=== V6 BBO+TRADE PING-PONG OOS ===")
    print(json.dumps({k:v for k,v in summary.items() if k != "oos"}, indent=2, allow_nan=True))
    print(json.dumps({k:v for k,v in test.items() if k != "per_symbol"}, indent=2, allow_nan=True))
    print("PER_SYMBOL")
    for row in sorted(test["per_symbol"], key=lambda x: x["turnover_x_day_1000u"], reverse=True):
        print(json.dumps(row, allow_nan=True))

if __name__ == "__main__":
    main()
