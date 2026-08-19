#!/usr/bin/env python3
"""
Capture Binance USDⓈ-M public market data for offline V2 strategy replay.

Network-enabled side only. No API key and no orders.
It discovers every TRADING USDC perpetual with a matching USDT perpetual,
then records:
  - 100ms top-of-book snapshots derived from depth5@100ms for lead + exec,
  - all aggTrade events for lead + exec,
  - exec markPrice@1s funding updates.

Output is gzip JSONL with an embedded pair/spec manifest. The matching V2 paper
engine can replay it offline with --replay, so the analysis environment itself
does not need outbound Binance access.
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import math
import signal
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import websockets

REST_HOSTS = [
    "https://fapi.binance.com",
    "https://fapi1.binance.com",
    "https://fapi2.binance.com",
    "https://fapi3.binance.com",
]
WS_URL = "wss://fstream.binance.com/ws"


def now_ms() -> int:
    return int(time.time() * 1000)


@dataclass(frozen=True)
class PairSpec:
    base: str
    lead_symbol: str
    exec_symbol: str
    tick_size: float
    qty_step: float
    min_qty: float
    min_notional: float


def filter_value(filters: list[dict], filter_type: str, key: str, default: float = 0.0) -> float:
    for f in filters:
        if f.get("filterType") == filter_type and key in f:
            try:
                return float(f[key])
            except (TypeError, ValueError):
                return default
    return default


def fetch_exchange_info_sync() -> dict:
    last: Optional[Exception] = None
    for host in REST_HOSTS:
        try:
            req = urllib.request.Request(
                host + "/fapi/v1/exchangeInfo",
                headers={"User-Agent": "V2-Capture/1.0"},
            )
            with urllib.request.urlopen(req, timeout=15) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:
            last = e
    raise RuntimeError(f"Unable to fetch Binance Futures exchangeInfo: {last}")


def discover_specs(info: dict, include: set[str], exclude: set[str]) -> list[PairSpec]:
    symbols = info.get("symbols", [])
    usdt = {
        s.get("baseAsset"): s for s in symbols
        if s.get("status") == "TRADING"
        and s.get("contractType") == "PERPETUAL"
        and s.get("quoteAsset") == "USDT"
    }
    out: list[PairSpec] = []
    for s in symbols:
        base = str(s.get("baseAsset") or "").upper()
        if not base or base in exclude or (include and base not in include):
            continue
        if not (
            s.get("status") == "TRADING"
            and s.get("contractType") == "PERPETUAL"
            and s.get("quoteAsset") == "USDC"
            and s.get("marginAsset") == "USDC"
            and base in usdt
        ):
            continue
        fs = s.get("filters", [])
        tick = filter_value(fs, "PRICE_FILTER", "tickSize")
        step = filter_value(fs, "LOT_SIZE", "stepSize")
        if tick <= 0 or step <= 0:
            continue
        min_qty = filter_value(fs, "LOT_SIZE", "minQty")
        min_notional = filter_value(fs, "MIN_NOTIONAL", "notional")
        if min_notional <= 0:
            min_notional = filter_value(fs, "NOTIONAL", "minNotional")
        out.append(PairSpec(
            base=base,
            lead_symbol=usdt[base]["symbol"],
            exec_symbol=s["symbol"],
            tick_size=tick,
            qty_step=step,
            min_qty=min_qty,
            min_notional=min_notional,
        ))
    return sorted(out, key=lambda x: x.base)


def build_streams(specs: list[PairSpec]) -> tuple[list[str], dict[str, str]]:
    streams: list[str] = []
    symbol_role: dict[str, str] = {}
    for s in specs:
        symbol_role[s.lead_symbol] = "LEAD"
        symbol_role[s.exec_symbol] = "EXEC"
        for sym in (s.lead_symbol, s.exec_symbol):
            lo = sym.lower()
            streams.append(f"{lo}@depth5@100ms")
            streams.append(f"{lo}@aggTrade")
        streams.append(f"{s.exec_symbol.lower()}@markPrice@1s")
    return streams, symbol_role


async def subscribe(ws, streams: list[str]) -> None:
    req_id = 1
    for i in range(0, len(streams), 100):
        await ws.send(json.dumps({"method": "SUBSCRIBE", "params": streams[i:i+100], "id": req_id}))
        req_id += 1
        await asyncio.sleep(0.30)


def normalize_event(d: dict) -> Optional[dict]:
    if "result" in d and "id" in d:
        return None

    e = d.get("e", "")
    if e == "depthUpdate" or (not e and ("b" in d or "bids" in d) and ("a" in d or "asks" in d)):
        bids = d.get("b") or d.get("bids") or []
        asks = d.get("a") or d.get("asks") or []
        if not bids or not asks:
            return None
        bp, bq = bids[0]
        ap, aq = asks[0]
        return {
            "e": "bookTicker",
            "E": int(d.get("E") or d.get("T") or now_ms()),
            "T": int(d.get("T") or d.get("E") or now_ms()),
            "s": d.get("s", ""),
            "b": str(bp), "B": str(bq),
            "a": str(ap), "A": str(aq),
        }
    if e in {"aggTrade", "markPriceUpdate", "bookTicker"}:
        return d
    return None


async def capture(args: argparse.Namespace) -> None:
    info = await asyncio.to_thread(fetch_exchange_info_sync)
    include = {x.strip().upper() for x in args.include.split(",") if x.strip()}
    exclude = {x.strip().upper() for x in args.exclude.split(",") if x.strip()}
    specs = discover_specs(info, include, exclude)
    if args.max_symbols > 0:
        specs = specs[:args.max_symbols]
    if not specs:
        raise SystemExit("No eligible USDC/USDT matched perpetual pairs found")

    streams, _ = build_streams(specs)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    stop = False

    def request_stop(*_):
        nonlocal stop
        stop = True

    try:
        signal.signal(signal.SIGINT, request_stop)
        signal.signal(signal.SIGTERM, request_stop)
    except Exception:
        pass

    deadline = math.inf if args.minutes <= 0 else time.monotonic() + args.minutes * 60
    event_count = 0
    reconnects = 0
    last_flush = time.monotonic()
    last_status = time.monotonic()
    started = now_ms()

    with gzip.open(out, "wt", encoding="utf-8", compresslevel=args.gzip_level) as f:
        meta = {
            "kind": "META",
            "capture_version": 1,
            "created_ms": started,
            "book_source": "depth5@100ms->BBO",
            "trade_source": "aggTrade realtime",
            "funding_source": "markPrice@1s",
            "specs": [asdict(x) for x in specs],
        }
        f.write(json.dumps(meta, separators=(",", ":")) + "\n")
        f.flush()

        delay = 1.0
        while not stop and time.monotonic() < deadline:
            try:
                async with websockets.connect(
                    WS_URL,
                    ping_interval=20,
                    ping_timeout=60,
                    close_timeout=5,
                    max_queue=100000,
                ) as ws:
                    await subscribe(ws, streams)
                    delay = 1.0
                    while not stop and time.monotonic() < deadline:
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=5)
                        except asyncio.TimeoutError:
                            continue
                        recv = now_ms()
                        obj = json.loads(raw)
                        if isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], dict):
                            obj = obj["data"]
                        d = normalize_event(obj)
                        if d is None:
                            continue
                        f.write(json.dumps({"kind": "EVENT", "recv_ts_ms": recv, "data": d}, separators=(",", ":")) + "\n")
                        event_count += 1
                        now_mono = time.monotonic()
                        if event_count % 5000 == 0 or now_mono - last_flush >= 5:
                            f.flush()
                            last_flush = now_mono
                        if now_mono - last_status >= 60:
                            mb = out.stat().st_size / 1_048_576 if out.exists() else 0
                            print(f"pairs={len(specs)} streams={len(streams)} events={event_count:,} file={mb:.1f}MiB reconnects={reconnects}")
                            last_status = now_mono
            except Exception as e:
                reconnects += 1
                print(f"[reconnect] {type(e).__name__}: {e}")
                await asyncio.sleep(delay)
                delay = min(30.0, delay * 2)
        f.flush()

    elapsed_h = max((now_ms() - started) / 3_600_000, 1e-9)
    print(f"Capture complete: {out} | pairs={len(specs)} events={event_count:,} elapsed={elapsed_h:.3f}h")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Capture Binance market data for V2 offline replay")
    p.add_argument("--minutes", type=float, default=1440, help="duration; 0 = until Ctrl-C")
    p.add_argument("--output", default="v2_market_capture.jsonl.gz")
    p.add_argument("--include", default="")
    p.add_argument("--exclude", default="")
    p.add_argument("--max-symbols", type=int, default=0)
    p.add_argument("--gzip-level", type=int, default=3, choices=range(1, 10))
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(capture(parse_args()))
