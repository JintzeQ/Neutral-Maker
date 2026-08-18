from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from math import log
from typing import List, Literal, Optional

Side = Literal["BUY", "SELL"]


@dataclass
class QuoteLevel:
    side: Side
    price: Decimal
    amount_base: Decimal
    level: int


@dataclass
class ActiveQuote:
    side: Side
    price: Decimal
    created_ts: float


@dataclass
class PMMConfig:
    buy_spreads: List[Decimal]
    sell_spreads: List[Decimal]
    quote_amounts_usdc: List[Decimal]
    inventory_target: Decimal = Decimal("0")
    inventory_range_usdc: Decimal = Decimal("20")
    min_spread: Decimal = Decimal("0")
    refresh_time_s: float = 5.0
    refresh_tolerance_pct: Decimal = Decimal("0.0001")
    max_order_age_s: float = 60.0


@dataclass
class AvellanedaInputs:
    mid: Decimal
    inventory_normalized: Decimal
    gamma: Decimal
    volatility_abs: Decimal
    time_left_fraction: Decimal
    kappa: Decimal
    min_spread_abs: Decimal = Decimal("0")


def quantize_floor(x: Decimal, tick: Decimal) -> Decimal:
    return (x // tick) * tick


def quantize_ceil(x: Decimal, tick: Decimal) -> Decimal:
    q = x // tick
    return q * tick if q * tick == x else (q + 1) * tick


def inventory_size_multipliers(
    inventory_usdc: Decimal,
    target_usdc: Decimal,
    range_usdc: Decimal,
) -> tuple[Decimal, Decimal]:
    """Clean-room piecewise inventory skew.

    At target: bid=1, ask=1.
    At target-range: bid=2, ask=0.
    At target+range: bid=0, ask=2.
    """
    if range_usdc <= 0:
        return Decimal("1"), Decimal("1")
    z = (inventory_usdc - target_usdc) / range_usdc
    z = max(Decimal("-1"), min(Decimal("1"), z))
    bid_mult = Decimal("1") - z
    ask_mult = Decimal("1") + z
    return max(Decimal("0"), bid_mult), max(Decimal("0"), ask_mult)


def build_pmm_quotes(
    mid: Decimal,
    best_bid: Decimal,
    best_ask: Decimal,
    tick: Decimal,
    inventory_usdc: Decimal,
    cfg: PMMConfig,
    reference_price: Optional[Decimal] = None,
    spread_multiplier: Decimal = Decimal("1"),
) -> List[QuoteLevel]:
    """Hummingbot-style PMM proposal pipeline, rewritten from first principles.

    1) reference price
    2) multi-level spread ladder
    3) inventory size skew
    4) maker-only / non-crossing filter
    """
    ref = reference_price if reference_price is not None else mid
    bid_mult, ask_mult = inventory_size_multipliers(
        inventory_usdc, cfg.inventory_target, cfg.inventory_range_usdc
    )
    out: List[QuoteLevel] = []

    for i, spread in enumerate(cfg.buy_spreads):
        quote_usdc = cfg.quote_amounts_usdc[min(i, len(cfg.quote_amounts_usdc) - 1)] * bid_mult
        if quote_usdc <= 0:
            continue
        px = ref * (Decimal("1") - spread * spread_multiplier)
        px = quantize_floor(px, tick)
        # Explicit post-only protection.
        px = min(px, best_ask - tick)
        if px <= 0:
            continue
        amount = quote_usdc / px
        out.append(QuoteLevel("BUY", px, amount, i))

    for i, spread in enumerate(cfg.sell_spreads):
        quote_usdc = cfg.quote_amounts_usdc[min(i, len(cfg.quote_amounts_usdc) - 1)] * ask_mult
        if quote_usdc <= 0:
            continue
        px = ref * (Decimal("1") + spread * spread_multiplier)
        px = quantize_ceil(px, tick)
        px = max(px, best_bid + tick)
        if px <= 0:
            continue
        amount = quote_usdc / px
        out.append(QuoteLevel("SELL", px, amount, i))

    return out


def should_refresh(active: ActiveQuote, target_price: Decimal, now: float, cfg: PMMConfig) -> bool:
    age = now - active.created_ts
    if age >= cfg.max_order_age_s:
        return True
    if age < cfg.refresh_time_s:
        return False
    if active.price <= 0:
        return True
    drift = abs(target_price / active.price - Decimal("1"))
    return drift > cfg.refresh_tolerance_pct


def avellaneda_reservation_and_spread(x: AvellanedaInputs) -> tuple[Decimal, Decimal, Decimal, Decimal]:
    """Inventory-aware Avellaneda-Stoikov style quote center and spread.

    This mirrors the mathematical structure used by Hummingbot's Avellaneda
    strategy, but is an independent implementation for research/backtesting.
    """
    if x.gamma <= 0 or x.kappa <= 0:
        raise ValueError("gamma and kappa must be > 0")
    reservation = x.mid - (
        x.inventory_normalized * x.gamma * x.volatility_abs * x.time_left_fraction
    )
    spread = (
        x.gamma * x.volatility_abs * x.time_left_fraction
        + Decimal("2") * Decimal(str(log(1.0 + float(x.gamma / x.kappa)))) / x.gamma
    )
    spread = max(spread, x.min_spread_abs)
    bid = reservation - spread / Decimal("2")
    ask = reservation + spread / Decimal("2")
    return reservation, spread, bid, ask


def apply_lead_toxicity_overlay(
    quotes: List[QuoteLevel],
    toxicity_score: Decimal,
    off_threshold: Decimal = Decimal("0.65"),
    retreat_threshold: Decimal = Decimal("0.35"),
    tick: Decimal = Decimal("0.00001"),
) -> List[QuoteLevel]:
    """Our V2 overlay; not part of Hummingbot.

    Positive toxicity = bullish lead flow => SELL side is vulnerable.
    Negative toxicity = bearish lead flow => BUY side is vulnerable.
    """
    out: List[QuoteLevel] = []
    for q in quotes:
        vulnerable = (toxicity_score < 0 and q.side == "BUY") or (toxicity_score > 0 and q.side == "SELL")
        a = abs(toxicity_score)
        if vulnerable and a >= off_threshold:
            continue
        if vulnerable and a >= retreat_threshold:
            price = q.price - tick if q.side == "BUY" else q.price + tick
            out.append(QuoteLevel(q.side, price, q.amount_base, q.level))
        else:
            out.append(q)
    return out
