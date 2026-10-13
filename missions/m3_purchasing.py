"""M3 — Purchasing Strategy: break-even, tier choice, spot-checkpoint sim (deck §4).

Run: python missions/m3_purchasing.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num, catalog_by_type
from finops import pricing

DAYS = 30


def run(verbose: bool = True) -> dict:
    jobs = load_csv("workloads.csv")
    cat = catalog_by_type()
    on_demand_monthly = optimized_monthly = 0.0
    recs = []
    for j in jobs:
        gtype = j["gpu_type"]
        ngpu = int(num(j["num_gpus"]))
        hpd = num(j["hours_per_day"])
        interruptible = bool(int(num(j["interruptible"])))
        c = cat[gtype]
        gpu_hours = hpd * DAYS * ngpu
        od = num(c["on_demand_hr"])
        on_demand_cost = gpu_hours * od

        tier = pricing.recommend_tier(hpd, interruptible)
        if tier == "spot":
            sim = pricing.spot_checkpoint_cost(gpu_hours, num(c["spot_hr"]), od)
            opt_cost = sim["spot_cost"]
        elif tier == "reserved":
            opt_cost = gpu_hours * num(c["reserved_3yr_hr"])
        else:
            opt_cost = on_demand_cost

        on_demand_monthly += on_demand_cost
        optimized_monthly += opt_cost
        recs.append({"job_id": j["job_id"], "gpu_type": gtype, "tier": tier,
                     "on_demand": round(on_demand_cost), "optimized": round(opt_cost)})

    savings = on_demand_monthly - optimized_monthly
    savings_pct = savings / on_demand_monthly * 100 if on_demand_monthly else 0.0

    # --- EXTENSION 1: Advanced Tier Matrix & Policy Comparison ---
    ext1_analysis = audit_advanced_purchasing(jobs, cat)

    # --- EXTENSION 5: Carbon-Aware Scheduling for Interruptible Workloads ---
    ext5_carbon = audit_carbon_aware_scheduling(jobs, cat)

    if verbose:
        print("== M3 Purchasing Strategy ==")
        print(f"break-even utilization @ 45% reserved discount = {pricing.break_even_utilization(0.45):.0%}")
        print(f"{'job':18}{'gpu':7}{'tier':11}{'on-demand':>12}{'optimized':>12}")
        for r in recs:
            print(f"{r['job_id']:18}{r['gpu_type']:7}{r['tier']:11}${r['on_demand']:>11,}${r['optimized']:>11,}")
        print(f"\nmonthly: on-demand ${on_demand_monthly:,.0f} -> optimized ${optimized_monthly:,.0f}  ({savings_pct:.1f}% saved)")

        print("\n-- [Extension 1] Advanced Tier Matrix & Commitment Horizon Analysis --")
        print(f"{'job_id':18}{'Basic Tier':12}{'Adv Tier':15}{'Horizon':10}{'Adv Cost':>10}{'Rationale'}")
        for adv in ext1_analysis["job_evaluations"]:
            print(f"{adv['job_id']:18}{adv['basic_tier']:12}{adv['adv_tier']:15}{adv['days']:>3}d       ${adv['adv_cost']:>9,d}  {adv['rationale']}")
        print(f"Advanced Purchasing Total: ${ext1_analysis['adv_monthly_total']:,.0f} ({ext1_analysis['adv_savings_pct']:.1f}% saved)")

        print("\n-- [Extension 5] Carbon-Aware Scheduling (Interruptible Workloads) --")
        print(f"Total interruptible jobs: {ext5_carbon['interruptible_job_count']}")
        print(f"Total energy consumed   : {ext5_carbon['total_kwh']:,.1f} kWh/month")
        print(f"Carbon in us-east-1     : {ext5_carbon['carbon_us_east_1_kg']:,.1f} kg CO2e")
        print(f"Carbon in europe-north1 : {ext5_carbon['carbon_europe_north1_kg']:,.1f} kg CO2e")
        print(f"Carbon Reduction        : {ext5_carbon['saved_carbon_kg']:,.1f} kg CO2e ({ext5_carbon['reduction_pct']:.1f}% cleaner!)")

        print("\nRegional Comparison Across 5 Data Center Regions:")
        print(f"{'Region':18}{'gCO2/kWh':>10}{'$/kWh':>8}{'Est. Carbon(kg)':>18}{'Est. Elec Cost':>16}")
        for reg in ext5_carbon["regional_comparison"]:
            print(f"{reg['region']:18}{reg['gco2_kwh']:>10}{reg['price_kwh']:>8.3f}{reg['total_carbon_kg']:>18.1f}  ${reg['total_elec_cost']:>14.2f}")

    return {
        "recommendations": recs,
        "on_demand_monthly": round(on_demand_monthly),
        "optimized_monthly": round(optimized_monthly),
        "savings_pct": round(savings_pct, 1),
        "advanced_purchasing": ext1_analysis,
        "carbon_scheduling": ext5_carbon,
    }


def audit_advanced_purchasing(jobs: list[dict], cat: dict) -> dict:
    """Extension 1: Evaluates 1yr vs 3yr reserved and GPU-specific spot interruption risk."""
    adv_evals = []
    adv_total = 0.0
    od_total = 0.0

    for j in jobs:
        gtype = j["gpu_type"]
        ngpu = int(num(j["num_gpus"]))
        hpd = num(j["hours_per_day"])
        days = int(num(j["days"]))
        intr = bool(int(num(j["interruptible"])))
        c = cat[gtype]

        basic_tier = pricing.recommend_tier(hpd, intr)
        adv_res = pricing.recommend_tier_advanced(
            hours_per_day=hpd,
            interruptible=intr,
            gpu_type=gtype,
            job_days=days,
            spot_hr=num(c["spot_hr"]),
            on_demand_hr=num(c["on_demand_hr"]),
            reserved_1yr_hr=num(c["reserved_1yr_hr"]),
            reserved_3yr_hr=num(c["reserved_3yr_hr"]),
        )
        adv_cost_scaled = adv_res["chosen_cost"] * ngpu * (30 / days if days else 1.0)
        od_cost_scaled = adv_res["on_demand_cost"] * ngpu * (30 / days if days else 1.0)

        adv_total += adv_cost_scaled
        od_total += od_cost_scaled

        adv_evals.append({
            "job_id": j["job_id"],
            "basic_tier": basic_tier,
            "adv_tier": adv_res["chosen_tier"],
            "days": days,
            "adv_cost": round(adv_cost_scaled),
            "rationale": adv_res["rationale"],
        })

    adv_savings_pct = ((od_total - adv_total) / od_total * 100.0) if od_total > 0 else 0.0

    return {
        "job_evaluations": adv_evals,
        "adv_monthly_total": round(adv_total),
        "adv_savings_pct": round(adv_savings_pct, 1),
    }


def audit_carbon_aware_scheduling(jobs: list[dict], cat: dict) -> dict:
    """Extension 5: Computes carbon emissions & optimal region placement for interruptible jobs."""
    from finops import sustainability

    total_kwh = 0.0
    intr_count = 0

    for j in jobs:
        if bool(int(num(j["interruptible"]))):
            intr_count += 1
            gtype = j["gpu_type"]
            ngpu = int(num(j["num_gpus"]))
            hpd = num(j["hours_per_day"])
            days = int(num(j["days"]))
            watts = num(cat[gtype]["watts"])
            gpu_hours = hpd * days * ngpu
            total_kwh += (watts * gpu_hours) / 1000.0

    # Regional carbon and cost
    c_us_east_1 = (total_kwh * sustainability.REGION_CARBON["us-east-1"]) / 1000.0  # kg CO2
    c_norway = (total_kwh * sustainability.REGION_CARBON["europe-north1"]) / 1000.0    # kg CO2
    saved_carbon = c_us_east_1 - c_norway
    reduction_pct = (saved_carbon / c_us_east_1 * 100.0) if c_us_east_1 else 0.0

    comparison = []
    for reg, gco2 in sustainability.REGION_CARBON.items():
        price = sustainability.REGION_PRICE_KWH.get(reg, 0.10)
        carb_kg = (total_kwh * gco2) / 1000.0
        cost = total_kwh * price
        comparison.append({
            "region": reg,
            "gco2_kwh": gco2,
            "price_kwh": price,
            "total_carbon_kg": round(carb_kg, 1),
            "total_elec_cost": round(cost, 2),
        })

    return {
        "interruptible_job_count": intr_count,
        "total_kwh": round(total_kwh, 1),
        "carbon_us_east_1_kg": round(c_us_east_1, 1),
        "carbon_europe_north1_kg": round(c_norway, 1),
        "saved_carbon_kg": round(saved_carbon, 1),
        "reduction_pct": round(reduction_pct, 1),
        "regional_comparison": comparison,
    }


if __name__ == "__main__":
    run()
