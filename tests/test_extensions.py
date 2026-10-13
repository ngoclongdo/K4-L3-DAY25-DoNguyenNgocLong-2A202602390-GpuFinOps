"""Unit tests for Lab 25 Extensions (Extensions 1-5)."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from finops import pricing, metrics, sustainability
from missions import m1_efficiency_audit, m2_inference_levers, m3_purchasing


def test_cache_is_worth_it():
    # Break-even reads for write=3.00, read_discount=0.10 (90% savings):
    # savings per read = 3.00 * 0.90 = 2.70
    # break-even = 3.00 / 2.70 = 1.11 reads
    be = pricing.break_even_cache_reads(3.00, 0.10)
    assert abs(be - 1.11) < 0.05

    # If average reads is below break-even, caching is net loss
    assert pricing.cache_is_worth_it(0.8, 3.00, 0.10) is False

    # If average reads is above break-even, caching saves money
    assert pricing.cache_is_worth_it(2.5, 3.00, 0.10) is True


def test_recommend_tier_advanced_decisions():
    # Job 1: Interruptible training (20h/d, 14 days, H100) -> spot
    res_spot = pricing.recommend_tier_advanced(
        hours_per_day=20,
        interruptible=True,
        gpu_type="H100",
        job_days=14,
        spot_hr=1.5,
        on_demand_hr=2.5,
        reserved_1yr_hr=2.0,
        reserved_3yr_hr=1.4,
    )
    assert res_spot["chosen_tier"] == "spot"
    assert res_spot["savings_pct"] > 30

    # Job 2: 24/7 service long horizon (365 days) -> reserved_3yr
    res_res = pricing.recommend_tier_advanced(
        hours_per_day=24,
        interruptible=False,
        gpu_type="A100",
        job_days=365,
        spot_hr=1.1,
        on_demand_hr=1.79,
        reserved_1yr_hr=1.4,
        reserved_3yr_hr=1.0,
    )
    assert res_res["chosen_tier"] == "reserved_3yr"
    assert res_res["savings_pct"] > 40


def test_tier_matrix_covers_scenarios():
    matrix = pricing.tier_recommendation_matrix()
    assert len(matrix) >= 16
    tiers = {m["recommended_tier"] for m in matrix}
    assert "spot" in tiers
    assert "reserved" in tiers
    assert "on_demand" in tiers


def test_mbu_rightsizing_audit():
    cat = {
        "H100": {"on_demand_hr": 2.5, "hbm_gb": 80, "peak_bw_tbs": 3.35},
        "A100": {"on_demand_hr": 1.79, "hbm_gb": 80, "peak_bw_tbs": 2.0},
    }
    summary = [
        {"gpu_id": "gpu-test-1", "gpu_type": "H100", "gpu_util_pct": 98.0, "mfu": 0.19, "mbu": 0.20, "idle_hours": 0}
    ]
    res = m1_efficiency_audit.audit_mbu_rightsizing(summary, cat)
    assert len(res["recommendations"]) == 1
    assert res["recommendations"][0]["proposed_type"] == "A100"
    assert res["recommendations"][0]["monthly_savings"] > 500


def test_reasoning_energy_multiplier():
    wh_standard = sustainability.wh_per_query(1000, is_reasoning=False)
    wh_reasoning = sustainability.wh_per_query(1000, is_reasoning=True)
    assert abs(wh_reasoning / wh_standard - sustainability.REASONING_ENERGY_MULTIPLIER) < 1e-5


def test_carbon_aware_reduction():
    # Compare emissions for 1000 kWh
    kwh = 1000.0
    carbon_us = (kwh * sustainability.REGION_CARBON["us-east-1"]) / 1000.0
    carbon_no = (kwh * sustainability.REGION_CARBON["europe-north1"]) / 1000.0
    pct_reduction = (carbon_us - carbon_no) / carbon_us * 100.0
    assert pct_reduction > 90.0
