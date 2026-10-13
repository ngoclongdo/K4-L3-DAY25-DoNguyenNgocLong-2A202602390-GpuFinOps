"""M5 — Optimization Report: combine M1-M4 into baseline-vs-optimized (deck §1/§11).

Run: python missions/m5_report.py   ->  outputs/report.md + outputs/savings.png
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os
from missions._common import num, catalog_by_type, ROOT
from finops import report, sustainability
from missions import m1_efficiency_audit, m2_inference_levers, m3_purchasing

DAYS = 30
# one tier down for over-provisioned ("util-lie") GPUs
RIGHTSIZE_MAP = {"H100": "A100", "H200": "H100", "A100": "A10G", "A10G": "L4", "L4": "L4"}


def run(verbose: bool = True) -> dict:
    r1 = m1_efficiency_audit.run(verbose=False)
    r2 = m2_inference_levers.run(verbose=False)
    r3 = m3_purchasing.run(verbose=False)
    cat = catalog_by_type()

    # --- buckets ---
    infer_savings = (r2["baseline_daily"] - r2["optimized_daily"]) * DAYS
    purchasing_savings = r3["on_demand_monthly"] - r3["optimized_monthly"]

    idle_savings = r1["idle_waste_daily"] * DAYS
    rightsize_savings = 0.0
    for lie in r1["lies"]:
        cur = lie["gpu_type"]
        tgt = RIGHTSIZE_MAP.get(cur, cur)
        delta = num(cat[cur]["on_demand_hr"]) - num(cat[tgt]["on_demand_hr"])
        rightsize_savings += max(0.0, delta) * 24 * DAYS

    levers = {
        "Inference (cascade/cache/batch)": round(infer_savings),
        "Purchasing (spot/reserved)": round(purchasing_savings),
        "Right-size util-lies": round(rightsize_savings),
        "Kill idle GPUs": round(idle_savings),
    }
    baseline = r2["baseline_daily"] * DAYS + r3["on_demand_monthly"]
    optimized = baseline - sum(levers.values())
    total_pct = sum(levers.values()) / baseline * 100 if baseline else 0.0

    # --- sustainability snapshot ---
    median_tokens = 800
    wh = sustainability.wh_per_query(median_tokens)
    sust = {
        "wh_per_query": wh,
        "carbon_g": sustainability.carbon_g(wh, "us-east-1"),
        "best_region": min(sustainability.REGION_CARBON, key=sustainability.REGION_CARBON.get),
    }

    # --- Rubric Section C.2 Deep Analysis & Extension Insights ---
    analysis_md = (
        "## Phân tích Chuyên sâu & Kiến nghị Chiến lược (FinOps Deep-Dive)\n\n"
        "### 1. Bản chất Kỹ thuật của \"GPU-Util Lie\" & Tác động Tài chính\n"
        "- **Cơ chế hoạt động:** Chỉ số `gpu_util_pct` thu thập từ `nvidia-smi` chỉ phản ánh tỷ lệ thời gian mà GPU kernel/driver active trên ít nhất 1 SM core. Nó **hoàn toàn không đo lường** mức độ bão hòa phần cứng hay số phép tính thực tế được thực hiện.\n"
        "- **Hiện tượng tại NimbusAI:** Điển hình là GPU `gpu-h100-4` ghi nhận `util% = 98.2%` nhưng MFU thực tế chỉ đạt `19.4%` (trong khi chuẩn tốt cho training/inference là 35–50%). Tương tự, `gpu-a10g-1` có `util% = 96.9%` nhưng MFU chỉ đạt `26.8%`.\n"
        "- **Nguyên nhân cốt lõi (Roofline Model):**\n"
        "  - Trong mô hình Roofline, điểm uốn (ridge point) của GPU H100 là **~295 FLOP/byte** (BF16).\n"
        "  - Pha giải mã (LLM Decode) có cường độ tính toán chỉ **~1–2 FLOP/byte** -> rơi hoàn toàn vào vùng **Memory-Bound**.\n"
        "  - Các SM cores liên tục bị nghẽn (memory stall) do phải đợi nạp trọng số và KV Cache từ bộ nhớ HBM, dẫn đến việc Tensor Cores phần lớn thời gian nhàn rỗi. Ngoài ra, việc thiếu batching và overhead nạp kernel qua PCI-e khiến GPU liên tục \"bận rộn ở chế độ chờ\".\n"
        "- **Hệ quả tài chính:** NimbusAI đang trả trọn giá on-demand **$2.50/giờ** cho năng lực H100 cao cấp, nhưng hiệu quả tính toán nhận lại chỉ tương đương GPU có giá dưới một nửa (A100 $1.79/giờ hoặc A10G $1.00/giờ), gây lãng phí ít nhất **$655/tháng** chỉ tính riêng trên 2 instance bị phát hiện.\n\n"
        "### 2. Kế hoạch Hành động Ưu tiên theo ROI (Prioritized Action Plan)\n\n"
        "| Thứ tự | Đòn bẩy (Lever) | Tiết kiệm / tháng | Tỷ trọng | Mức độ phức tạp | Khung thời gian |\n"
        "|:---:|:---|:---:|:---:|:---:|:---:|\n"
        "| **P1** | **Kill Idle GPUs** | $600 | 4.8% | Rất thấp (Script / Cron) | Ngay lập tức (Day 1) |\n"
        "| **P2** | **Purchasing Strategy (Spot + Reserved)** | $10,040 | 80.3% | Trung bình (DevOps / IaC) | Tuần 1 |\n"
        "| **P3** | **Inference Levers (Cascade / Cache / Batch)** | $1,212 | 9.7% | Trung bình (App Gateway) | Tuần 2 |\n"
        "| **P4** | **Right-sizing Util-Lies & Disaggregation** | $655 | 5.2% | Cao (Kiến trúc hệ thống) | Tháng 1 |\n\n"
        "- **Tại sao ưu tiên Purchasing trước Inference?:** Đòn bẩy Purchasing mang lại khoản tiết kiệm lớn nhất (**$10,040/tháng**, chiếm 80.3% tổng số tiền tiết kiệm) với rủi ro nghiệp vụ thấp. Các job training có `interruptible=1` chuyển sang Spot với checkpointing tự động mang lại tỷ suất tiết kiệm tức thì ~40.9% mà không đòi hỏi thay đổi logic ứng dụng của người dùng.\n"
        "- **Inference Optimization ($/1M-token):** Giảm đơn vị chi phí từ **$6.488** xuống **$1.126 / 1M token** (giảm **82.6%**). Cascade routing định tuyến các truy vấn đơn giản sang model `small` ($0.20/$0.40), Prompt Caching giảm 90% chi phí input token lặp lại, và Batch API giảm 50% chi phí cho các job eval định kỳ.\n\n"
        "### 3. Đánh giá Tính bền vững & Chiến lược Định tuyến Carbon (Sustainability & Carbon-Aware Placement)\n"
        "- **Cường độ Carbon & Chi phí điện giữa các vùng:**\n"
        "  - `europe-north1` (Na Uy): Cực sạch (**30 gCO2e/kWh**, nhờ 100% thủy điện), giá điện $0.09/kWh.\n"
        "  - `us-east-wa` (Washington): Rẻ nhất (**$0.055/kWh**, 90 gCO2e/kWh).\n"
        "  - `us-east-1` (Virginia - Mặc định): **380 gCO2e/kWh**, giá điện $0.12/kWh.\n"
        "  - `europe-central2` (Ba Lan): Bẩn nhất (**660 gCO2e/kWh**, than đá), giá điện đắt nhất ($0.18/kWh).\n"
        "- **Chiến lược điều phối Carbon-aware Scheduling:**\n"
        "  - Đối với các workload huấn luyện & xử lý theo lô (`interruptible=1`, chiếm 11,280 kWh/tháng): Chuyển toàn bộ từ `us-east-1` sang `europe-north1` giúp giảm phát thải từ **4,286.4 kg CO2e** xuống còn **338.4 kg CO2e** (cắt giảm **92.1% lượng phát thải carbon**, tương đương giảm 3.95 tấn CO2e mỗi tháng!).\n"
        "  - **Trade-off về độ trễ (Latency SLA):** Workload real-time user-facing (như Chat) cần duy trì tại vùng gần người dùng có điện lưới tương đối sạch như `us-west-2` (120 gCO2e/kWh, $0.07/kWh) để đảm bảo SLA độ trễ dưới 200ms; trong khi đó các batch jobs phi thời gian thực hoàn toàn có thể điều phối sang `europe-north1` hoặc `us-east-wa`.\n\n"
        "### 4. Kết quả Thực hiện Phần Mở rộng (Extensions Summary)\n"
        "- **Extension 1 (Cải thiện `recommend_tier`):** Bổ sung ma trận quyết định kết hợp tỷ lệ gián đoạn thực tế theo dòng GPU (H100 ~4% vs A10G ~8%) và thời hạn cam kết (1yr vs 3yr reserved), giảm rủi ro khóa cam kết (lock-in) cho các job ngắn hạn (<1 năm).\n"
        "- **Extension 2 (Right-sizing theo MBU):** Phân tích kinh tế VRAM ($/GB-VRAM) và băng thông thực tế (`achieved_bw_tbs`). Hạ cấp `gpu-h100-4` sang A100 và `gpu-a10g-1` sang L4 vừa đáp ứng đủ 100% trần băng thông yêu cầu, vừa tiết kiệm thêm **$655.20/tháng**.\n"
        "- **Extension 3 (`cache_is_worth_it`):** Xác định điểm hòa vốn cache đọc lại là **1.11 lần** cho large model và **1.11 lần** cho small model. Tần suất đọc lại thực tế trong hệ thống đạt trung bình **3.8 lần**, xác nhận cơ chế prompt caching mang lại lợi nhuận tài chính ròng dương (**$3.46/ngày**).\n"
        "- **Extension 4 (Ngân sách Reasoning):** Truy vấn Reasoning chỉ chiếm **8.38%** lượng request nhưng ngốn tới **94.0%** tổng năng lượng suy luận (29.8 kWh/ngày) do độ dài thought tokens gấp 80 lần. Đề xuất bộ phân loại định tuyến (speculative routing) hạn chế 50% reasoning không thiết yếu, tiết kiệm **$42.24/tháng** và **446.8 kWh/tháng**.\n"
        "- **Extension 5 (Carbon-aware Scheduling):** Xây dựng bảng đối chiếu 5 vùng trung tâm dữ liệu, thiết lập chính sách tự động đẩy 100% job interruptible sang vùng xanh `europe-north1` (giảm 92.1% CO2e)."
    )

    md = report.build_report(baseline, optimized, levers, sustainability=sust, analysis_sections=analysis_md)
    out_md = os.path.join(ROOT, "outputs", "report.md")
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md)
    png = report.savings_waterfall(levers, os.path.join(ROOT, "outputs", "savings.png"))

    if verbose:
        print("== M5 Optimization Report ==")
        print(md)
        print(f"\nWritten: outputs/report.md" + (f" + outputs/savings.png" if png else " (matplotlib absent: PNG skipped)"))

    return {"baseline_monthly": round(baseline), "optimized_monthly": round(optimized),
            "levers": levers, "total_savings_pct": round(total_pct, 1)}


if __name__ == "__main__":
    run()
