"""M2 — Inference Cost Levers: $/1M-token, batch x cache x cascade (deck §7).

Run: python missions/m2_inference_levers.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num
from finops import pricing

# $/1M tokens (input, output) — illustrative 2026.
MODEL_PRICES = {"small": (0.20, 0.40), "large": (3.00, 15.00)}


def run(verbose: bool = True) -> dict:
    rows = load_csv("token_usage.csv")
    base_cost = opt_cost = 0.0
    total_tokens = 0
    for r in rows:
        inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
        cached = int(num(r["cached_input_tokens"]))
        is_batch = bool(int(num(r["is_batch"])))
        total_tokens += inp + out
        # BASELINE: naive deployment — everything on the large model, no cache, no batch
        lin, lout = MODEL_PRICES["large"]
        base_cost += pricing.request_cost(inp, out, lin, lout)
        # OPTIMIZED: cascade (route_tier), prompt caching, batch API
        pin, pout = MODEL_PRICES[r["route_tier"]]
        opt_cost += pricing.request_cost(inp, out, pin, pout, cached_in=cached, batch=is_batch)

    base_pm = pricing.dollars_per_million(base_cost, total_tokens)
    opt_pm = pricing.dollars_per_million(opt_cost, total_tokens)
    savings_pct = (1 - opt_cost / base_cost) * 100 if base_cost else 0.0

    # --- EXTENSION 3: Prompt Caching Worth-It Audit ---
    ext3_cache = audit_cache_worth(rows)

    # --- EXTENSION 4: Reasoning Energy & Cost Budgeting ---
    ext4_reasoning = audit_reasoning_budget(rows)

    if verbose:
        print("== M2 Inference Cost Levers ==")
        print(f"requests={len(rows)}  tokens={total_tokens:,}")
        print(f"baseline  : ${base_cost:,.2f}/day   ${base_pm:.3f}/1M-token")
        print(f"optimized : ${opt_cost:,.2f}/day   ${opt_pm:.3f}/1M-token")
        print(f"savings   : {savings_pct:.1f}%  (cascade + caching + batch)")
        print(f"discount stack (batch + 100% cache): {pricing.discount_stack(batch=True, cache_hit_frac=1.0):.3f} of naive")

        print("\n-- [Extension 3] Cache-is-worth-it Analysis --")
        print(f"  Total Input Tokens : {ext3_cache['total_input_tokens']:,}")
        print(f"  Cached Input Tokens: {ext3_cache['cached_input_tokens']:,} ({ext3_cache['cache_token_ratio']:.1%})")
        print(f"  Break-even Reads (Large): {ext3_cache['be_reads_large']} reads  |  Actual Avg Reads: ~{ext3_cache['estimated_reads']:.1f}")
        print(f"  Cache is worth it? {ext3_cache['is_worth_it']} (Net savings: ${ext3_cache['cache_savings_usd']:.2f}/day)")

        print("\n-- [Extension 4] Reasoning Energy & Cost Budget --")
        print(f"  Reasoning Requests : {ext4_reasoning['reasoning_reqs']} / {len(rows)} ({ext4_reasoning['traffic_pct']:.1f}% traffic)")
        print(f"  Reasoning Cost ($) : ${ext4_reasoning['reasoning_cost']:.2f}/day ({ext4_reasoning['cost_pct']:.1f}% of optimized spend)")
        print(f"  Reasoning Energy   : {ext4_reasoning['reasoning_kwh']:.2f} kWh/day ({ext4_reasoning['energy_pct']:.1f}% of total inference power!)")
        print(f"  Routing Cap Impact : Restricting 50% non-critical reasoning queries saves ${ext4_reasoning['cap_savings_usd_mo']:.2f}/mo & {ext4_reasoning['cap_energy_kwh_mo']:.1f} kWh/mo.")

    return {
        "baseline_daily": round(base_cost, 2), "optimized_daily": round(opt_cost, 2),
        "baseline_per_m": round(base_pm, 3), "optimized_per_m": round(opt_pm, 3),
        "savings_pct": round(savings_pct, 1), "total_tokens": total_tokens,
        "cache_audit": ext3_cache,
        "reasoning_audit": ext4_reasoning,
    }


def audit_cache_worth(rows: list[dict]) -> dict:
    """Extension 3: Evaluates whether prompt caching achieves break-even."""
    tot_in = sum(int(num(r["input_tokens"])) for r in rows)
    tot_cached = sum(int(num(r["cached_input_tokens"])) for r in rows)
    cache_ratio = tot_cached / tot_in if tot_in else 0.0

    # Large model write cost = $3.00, read discount = 0.10 (90% off) -> savings per read = $2.70
    be_large = pricing.break_even_cache_reads(3.00, 0.10)
    be_small = pricing.break_even_cache_reads(0.20, 0.10)

    # In typical chat/RAG prompts with recurring system prompts, avg reads per cached prefix is ~3-5
    estimated_reads = 3.8
    worth = pricing.cache_is_worth_it(estimated_reads, 3.00, 0.10)

    # Calculate actual daily dollars saved by cache
    cache_savings = 0.0
    for r in rows:
        cached = int(num(r["cached_input_tokens"]))
        pin, _ = MODEL_PRICES[r["route_tier"]]
        # Normal read price vs cached read price (0.90 discount)
        cache_savings += (cached / 1e6) * pin * 0.90
        if bool(int(num(r["is_batch"]))):
            cache_savings *= 0.50

    return {
        "total_input_tokens": tot_in,
        "cached_input_tokens": tot_cached,
        "cache_token_ratio": round(cache_ratio, 3),
        "be_reads_large": be_large,
        "be_reads_small": be_small,
        "estimated_reads": estimated_reads,
        "is_worth_it": worth,
        "cache_savings_usd": round(cache_savings, 2),
    }


def audit_reasoning_budget(rows: list[dict]) -> dict:
    """Extension 4: Tách riêng chi phí $ và năng lượng Wh cho reasoning queries."""
    from finops.sustainability import wh_per_query

    r_cost = nr_cost = 0.0
    r_wh = nr_wh = 0.0
    r_count = nr_count = 0

    for r in rows:
        inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
        cached = int(num(r["cached_input_tokens"]))
        is_batch = bool(int(num(r["is_batch"])))
        is_reason = bool(int(num(r["is_reasoning"])))

        pin, pout = MODEL_PRICES[r["route_tier"]]
        cost = pricing.request_cost(inp, out, pin, pout, cached_in=cached, batch=is_batch)
        wh = wh_per_query(inp + out, is_reasoning=is_reason)

        if is_reason:
            r_cost += cost
            r_wh += wh
            r_count += 1
        else:
            nr_cost += cost
            nr_wh += wh
            nr_count += 1

    total_cost = r_cost + nr_cost
    total_wh = r_wh + nr_wh
    total_reqs = r_count + nr_count

    traffic_pct = (r_count / total_reqs * 100) if total_reqs else 0.0
    cost_pct = (r_cost / total_cost * 100) if total_cost else 0.0
    energy_pct = (r_wh / total_wh * 100) if total_wh else 0.0

    # Policy simulation: Cap/filter 50% of reasoning queries via speculative classifier
    capped_monthly_savings_usd = (r_cost * 0.50) * 30
    capped_monthly_energy_kwh = (r_wh * 0.50 / 1000.0) * 30

    return {
        "reasoning_reqs": r_count,
        "non_reasoning_reqs": nr_count,
        "traffic_pct": round(traffic_pct, 1),
        "reasoning_cost": round(r_cost, 2),
        "cost_pct": round(cost_pct, 1),
        "reasoning_kwh": round(r_wh / 1000.0, 2),
        "energy_pct": round(energy_pct, 1),
        "cap_savings_usd_mo": round(capped_monthly_savings_usd, 2),
        "cap_energy_kwh_mo": round(capped_monthly_energy_kwh, 1),
    }


if __name__ == "__main__":
    run()
