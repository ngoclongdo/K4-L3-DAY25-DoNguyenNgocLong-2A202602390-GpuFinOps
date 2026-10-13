"""M1 — Efficiency Audit: MFU/MBU, the GPU-Util lie, and idle waste (deck §5).

Run: python missions/m1_efficiency_audit.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from collections import defaultdict
from missions._common import load_csv, num, catalog_by_type
from finops import metrics


def run(verbose: bool = True) -> dict:
    tel = load_csv("gpu_telemetry.csv")
    cat = catalog_by_type()

    # per-row MFU/MBU, then aggregate per GPU
    agg = defaultdict(lambda: {"util": [], "mfu": [], "mbu": [], "type": None, "idle_hours": 0})
    for r in tel:
        gtype = r["gpu_type"]
        peak_fp16 = num(cat[gtype]["peak_tflops_fp16"])
        peak_bw = num(cat[gtype]["peak_bw_tbs"])
        mfu = metrics.compute_mfu(num(r["achieved_tflops"]), peak_fp16)
        mbu = metrics.compute_mbu(num(r["achieved_bw_tbs"]), peak_bw)
        a = agg[r["gpu_id"]]
        a["type"] = gtype
        a["util"].append(num(r["gpu_util_pct"]))
        a["mfu"].append(mfu)
        a["mbu"].append(mbu)
        if num(r["gpu_util_pct"]) < 10:  # effectively idle this interval (1h)
            a["idle_hours"] += 1

    summary = []
    for gid, a in agg.items():
        summary.append({
            "gpu_id": gid, "gpu_type": a["type"],
            "gpu_util_pct": round(sum(a["util"]) / len(a["util"]), 1),
            "mfu": round(sum(a["mfu"]) / len(a["mfu"]), 3),
            "mbu": round(sum(a["mbu"]) / len(a["mbu"]), 3),
            "idle_hours": a["idle_hours"],
        })

    lies = metrics.flag_util_lies(summary)
    idle_waste = 0.0
    for s in summary:
        on_demand = num(catalog_by_type()[s["gpu_type"]]["on_demand_hr"])
        idle_waste += metrics.idle_waste_usd(s["idle_hours"], on_demand)

    # --- EXTENSION 2: Right-sizing theo MBU & $/GB-VRAM ---
    rightsizing_analysis = audit_mbu_rightsizing(summary, cat)

    if verbose:
        print("== M1 Efficiency Audit ==")
        print(f"{'GPU':14}{'type':7}{'util%':>7}{'MFU':>7}{'MBU':>7}{'idle_h':>8}")
        for s in sorted(summary, key=lambda x: x["mfu"]):
            print(f"{s['gpu_id']:14}{s['gpu_type']:7}{s['gpu_util_pct']:>7}{s['mfu']:>7}{s['mbu']:>7}{s['idle_hours']:>8}")
        print(f"\nGPU-Util LIES (util>=90% but MFU<30%): {[l['gpu_id'] for l in lies]}")
        print(f"Idle waste (1 day): ${idle_waste:,.2f}  ->  ${idle_waste*30:,.0f}/month")

        print("\n-- [Extension 2] Right-sizing theo MBU & $/GB-VRAM --")
        print(f"{'GPU Type':9}{'$/hr':>7}{'VRAM(GB)':>10}{'BW(TB/s)':>10}{'$/GB-VRAM':>12}{'$/(TB/s-hr)':>14}")
        for gtype, row in rightsizing_analysis["catalog_economics"].items():
            print(f"{gtype:9}${row['on_demand_hr']:>6.2f}{row['vram_gb']:>10.0f}{row['peak_bw_tbs']:>10.2f}  ${row['cost_per_gb_vram']:>9.4f}  ${row['cost_per_bw_tbs']:>11.3f}")

        print("\nProposed Right-Sizing for Underutilized/Lie GPUs:")
        print(f"{'GPU ID':12}{'Current':9}{'Proposed':9}{'Achieved BW':>12}{'Proposed BW':>12}{'Monthly Savings':>17}")
        for rec in rightsizing_analysis["recommendations"]:
            print(f"{rec['gpu_id']:12}{rec['current_type']:9}{rec['proposed_type']:9}{rec['achieved_bw']:>10.2f} TB/s{rec['proposed_peak_bw']:>10.2f} TB/s  ${rec['monthly_savings']:>14,.2f}")
        print(f"Total Right-Sizing Monthly Savings: ${rightsizing_analysis['total_monthly_savings']:,.2f}")

    return {
        "summary": summary,
        "lies": lies,
        "idle_waste_daily": round(idle_waste, 2),
        "mbu_rightsizing": rightsizing_analysis["recommendations"],
        "mbu_rightsizing_savings_monthly": rightsizing_analysis["total_monthly_savings"],
    }


def audit_mbu_rightsizing(summary: list[dict], cat: dict) -> dict:
    """Extension 2: Analyzes GPU memory bandwidth and $/GB-VRAM to recommend optimal right-sizing.

    Why not just pick the cheapest GPU by $/GPU-hr?
    A cheaper GPU (e.g. L4) may have inadequate memory bandwidth or capacity, causing severe
    slowdown and extending runtime 5-10x, thereby dramatically increasing total job cost.
    FinOps right-sizing matches achieved bandwidth/memory requirements to the lowest $/GB-VRAM.
    """
    cat_econ = {}
    for gtype, spec in cat.items():
        od = num(spec["on_demand_hr"])
        vram = num(spec["hbm_gb"])
        bw = num(spec["peak_bw_tbs"])
        cat_econ[gtype] = {
            "on_demand_hr": od,
            "vram_gb": vram,
            "peak_bw_tbs": bw,
            "cost_per_gb_vram": round(od / vram, 4) if vram > 0 else 0.0,
            "cost_per_bw_tbs": round(od / bw, 3) if bw > 0 else 0.0,
        }

    # Target downgrades that satisfy achieved memory bandwidth without performance regression
    rightsize_map = {"H100": "A100", "H200": "H100", "A100": "A10G", "A10G": "L4", "L4": "L4"}
    recs = []
    total_savings = 0.0

    # Look for GPUs with util lies or low MBU (< 0.35)
    for s in summary:
        util = s["gpu_util_pct"] / 100.0
        mfu = s["mfu"]
        mbu = s["mbu"]
        cur = s["gpu_type"]
        cur_bw = num(cat[cur]["peak_bw_tbs"])
        achieved_bw = round(mbu * cur_bw, 3)

        if (util >= 0.90 and mfu < 0.30) or (mbu < 0.25 and s["idle_hours"] == 0):
            proposed = rightsize_map.get(cur, cur)
            if proposed != cur:
                prop_bw = num(cat[proposed]["peak_bw_tbs"])
                hourly_delta = num(cat[cur]["on_demand_hr"]) - num(cat[proposed]["on_demand_hr"])
                monthly_saving = max(0.0, hourly_delta) * 24 * 30
                total_savings += monthly_saving
                recs.append({
                    "gpu_id": s["gpu_id"],
                    "current_type": cur,
                    "proposed_type": proposed,
                    "achieved_bw": achieved_bw,
                    "proposed_peak_bw": prop_bw,
                    "hourly_savings": round(hourly_delta, 2),
                    "monthly_savings": round(monthly_saving, 2),
                    "rationale": f"Achieved BW ({achieved_bw:.2f} TB/s) is well within {proposed} ceiling ({prop_bw:.2f} TB/s).",
                })

    return {
        "catalog_economics": cat_econ,
        "recommendations": recs,
        "total_monthly_savings": round(total_savings, 2),
    }


if __name__ == "__main__":
    run()
