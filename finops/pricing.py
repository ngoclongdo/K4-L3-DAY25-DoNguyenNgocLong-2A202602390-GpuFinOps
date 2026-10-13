"""Pricing & purchasing economics — measure in $/1M-token, not $/GPU-hr.

Figures are June-2026 as-of snapshots from the deck's RESEARCH dossier; treat
live prices as fast-moving (re-baseline before each cohort).
"""
from __future__ import annotations


def request_cost(
    input_tok: int,
    output_tok: int,
    price_in_per_m: float,
    price_out_per_m: float,
    cached_in: int = 0,
    cache_discount: float = 0.10,   # Anthropic cached-read ~0.1x (=-90%)
    batch: bool = False,
    batch_discount: float = 0.50,   # Batch API ~ -50%
) -> float:
    """USD cost of a single request. Cached input billed at cache_discount x price."""
    cached_in = min(max(0, cached_in), input_tok)
    uncached_in = input_tok - cached_in
    cost = (
        (uncached_in / 1e6) * price_in_per_m
        + (cached_in / 1e6) * price_in_per_m * cache_discount
        + (output_tok / 1e6) * price_out_per_m
    )
    if batch:
        cost *= batch_discount
    return cost


def dollars_per_million(total_cost_usd: float, total_tokens: int) -> float:
    """Aggregate unit economics: $ per 1,000,000 tokens served."""
    if total_tokens <= 0:
        return 0.0
    return total_cost_usd / (total_tokens / 1e6)


def discount_stack(
    batch: bool = False,
    cache_hit_frac: float = 0.0,
    batch_discount: float = 0.50,
    cache_discount: float = 0.10,
) -> float:
    """Effective fraction of the naive bill after stacking discounts (input-heavy view).

    Discounts MULTIPLY: cache applies to the cached share of input, batch to the
    whole bill. batch + 100% cache-hit -> 0.5 * 0.1 = 0.05 (~95% off).
    """
    cache_mult = cache_hit_frac * cache_discount + (1.0 - cache_hit_frac)
    batch_mult = batch_discount if batch else 1.0
    return cache_mult * batch_mult


def break_even_utilization(discount_frac: float) -> float:
    """Utilization at which a commitment pays off ~= 1 - discount.

    A 45% reserved discount needs ~55% utilization (~13.2h/day) to beat on-demand.
    """
    return max(0.0, min(1.0, 1.0 - discount_frac))


def cache_is_worth_it(
    avg_cache_reads: float,
    write_cost_per_m: float,
    read_discount: float = 0.10,
    read_cost_per_m: float | None = None,
) -> bool:
    """Check if prompt caching saves net money given read frequencies.

    In modern LLM pricing (e.g. Anthropic / Gemini), caching involves a write/storage
    overhead. Caching pays off when savings from subsequent reads > write cost:
      Break-even reads = write_cost / [read_cost * (1 - read_discount)].
    """
    if read_cost_per_m is None or read_cost_per_m <= 0:
        read_cost_per_m = write_cost_per_m
    savings_per_read = read_cost_per_m * (1.0 - read_discount)
    if savings_per_read <= 0:
        return False
    be_reads = write_cost_per_m / savings_per_read
    return avg_cache_reads >= be_reads


def break_even_cache_reads(
    write_cost_per_m: float,
    read_discount: float = 0.10,
    read_cost_per_m: float | None = None,
) -> float:
    """Minimum average reads needed per prefix before caching yields net savings."""
    if read_cost_per_m is None or read_cost_per_m <= 0:
        read_cost_per_m = write_cost_per_m
    savings_per_read = read_cost_per_m * (1.0 - read_discount)
    if savings_per_read <= 0:
        return float("inf")
    return round(write_cost_per_m / savings_per_read, 2)


def recommend_tier(
    hours_per_day: float,
    interruptible: bool,
    reserved_discount: float = 0.45,
    gpu_type: str | None = None,
    job_days: int | None = None,
    interruption_rate: float | None = None,
) -> str:
    """Pick a purchasing tier from a workload's duty cycle + interruptibility.

    Policy:
      - interruptible & not 24/7  -> 'spot'      (checkpoint and ride the discount)
      - duty cycle >= break-even  -> 'reserved'  (steady, high utilization)
      - otherwise                 -> 'on_demand' (spiky / low duty)

    Enhanced with Extension 1:
      - Accounts for GPU-specific interruption risk and commitment horizon.
    """
    duty = max(0.0, hours_per_day) / 24.0
    be = break_even_utilization(reserved_discount)

    # Extension 1: If interruption rate is excessively high, spot might not win
    if interruptible and hours_per_day < 24:
        if interruption_rate is not None and interruption_rate > 0.35:
            # High interruption volatility causes excessive rework
            return "on_demand" if duty < be else "reserved"
        return "spot"

    if duty >= be:
        return "reserved"
    return "on_demand"


def recommend_tier_advanced(
    hours_per_day: float,
    interruptible: bool,
    gpu_type: str = "H100",
    job_days: int = 30,
    spot_hr: float = 1.5,
    on_demand_hr: float = 2.5,
    reserved_1yr_hr: float = 2.0,
    reserved_3yr_hr: float = 1.4,
) -> dict:
    """Extension 1 (Xuất sắc): Decision engine comparing Spot, 1yr Reserved, and 3yr Reserved.

    Considers:
      1. GPU-specific interruption rate (H100 ~ 5%, A10G/L4 ~ 8-10%)
      2. Checkpoint overhead and rework cost
      3. Commitment horizon: 3-year commitment is risky for short-lived workloads (<1 yr).
    """
    interruption_rates = {"H100": 0.04, "H200": 0.03, "A100": 0.05, "A10G": 0.08, "L4": 0.08, "B200": 0.05}
    irate = interruption_rates.get(gpu_type, 0.06)

    duty = max(0.0, hours_per_day) / 24.0
    total_hours = hours_per_day * job_days

    # Calculate costs under each strategy for the job duration
    od_cost = total_hours * on_demand_hr

    # Spot cost with checkpoint simulation
    spot_sim = spot_checkpoint_cost(
        job_hours=total_hours,
        spot_hr=spot_hr,
        on_demand_hr=on_demand_hr,
        interrupt_rate=irate,
        ckpt_overhead_frac=0.03,
        rework_hours_per_interrupt=0.5,
    )
    spot_cost = spot_sim["spot_cost"]

    # Reserved options
    res_1yr_cost = total_hours * reserved_1yr_hr
    res_3yr_cost = total_hours * reserved_3yr_hr

    # Decision logic:
    if interruptible and hours_per_day < 24 and spot_cost < min(od_cost, res_1yr_cost):
        chosen_tier = "spot"
        chosen_cost = spot_cost
        rationale = f"Interruptible ({irate:.0%} interrupt rate), spot delivers lowest TCO with checkpointing."
    elif duty >= 0.55:
        if job_days >= 365:
            chosen_tier = "reserved_3yr"
            chosen_cost = res_3yr_cost
            rationale = "High duty cycle (>=55%) and multi-year horizon: 3yr reserved maximizes discount (44%)."
        else:
            chosen_tier = "reserved_1yr" if res_1yr_cost < od_cost else "reserved_3yr"
            chosen_cost = res_1yr_cost if chosen_tier == "reserved_1yr" else res_3yr_cost
            rationale = f"High duty cycle (>=55%) but short horizon ({job_days}d): 1yr reserved minimizes commitment risk."
    else:
        chosen_tier = "on_demand"
        chosen_cost = od_cost
        rationale = f"Low duty cycle ({duty:.1%}) below break-even threshold (55%): on-demand avoids idle commitment waste."

    savings_vs_od = max(0.0, od_cost - chosen_cost)
    savings_pct = (savings_vs_od / od_cost * 100.0) if od_cost > 0 else 0.0

    return {
        "chosen_tier": chosen_tier,
        "chosen_cost": round(chosen_cost, 2),
        "on_demand_cost": round(od_cost, 2),
        "savings_vs_od": round(savings_vs_od, 2),
        "savings_pct": round(savings_pct, 1),
        "rationale": rationale,
        "spot_sim": spot_sim,
    }


def tier_recommendation_matrix() -> list[dict]:
    """Generates the decision matrix across GPU Types x Duty Cycles x Interruptibility."""
    matrix = []
    duty_scenarios = [("Low (25%, 6h/d)", 6.0), ("Medium (50%, 12h/d)", 12.0), ("High (83%, 20h/d)", 20.0), ("24/7 (100%, 24h/d)", 24.0)]
    for gtype in ["H100", "A100", "A10G", "L4"]:
        for label, hpd in duty_scenarios:
            for intr in [True, False]:
                tier = recommend_tier(hpd, intr, gpu_type=gtype)
                matrix.append({
                    "gpu_type": gtype,
                    "duty_label": label,
                    "hours_per_day": hpd,
                    "interruptible": intr,
                    "recommended_tier": tier,
                })
    return matrix


def spot_checkpoint_cost(
    job_hours: float,
    spot_hr: float,
    on_demand_hr: float,
    interrupt_rate: float = 0.05,      # per-hour chance (H100 spot ~<5%)
    ckpt_overhead_frac: float = 0.03,  # steady cost of writing checkpoints
    rework_hours_per_interrupt: float = 0.5,
) -> dict:
    """Effective cost of running a checkpointable job on spot vs on-demand.

    Interruptions waste the compute since the last checkpoint (rework); checkpointing
    adds a small steady overhead. Spot still wins for interruptible jobs.
    """
    expected_interrupts = job_hours * interrupt_rate
    rework_hours = expected_interrupts * rework_hours_per_interrupt
    effective_hours = job_hours * (1.0 + ckpt_overhead_frac) + rework_hours
    spot_cost = effective_hours * spot_hr
    on_demand_cost = job_hours * on_demand_hr
    savings_pct = (1.0 - spot_cost / on_demand_cost) * 100.0 if on_demand_cost > 0 else 0.0
    return {
        "spot_effective_hours": round(effective_hours, 2),
        "spot_cost": round(spot_cost, 2),
        "on_demand_cost": round(on_demand_cost, 2),
        "savings_pct": round(savings_pct, 1),
    }
