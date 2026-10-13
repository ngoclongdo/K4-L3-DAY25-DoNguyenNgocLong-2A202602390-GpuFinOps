# Báo Cáo Kỹ Thuật FinOps — Tối Ưu Hóa Chi Phí GPU (NimbusAI)

> **Học viên:** Đỗ Nguyễn Ngọc Long  
> **Khóa học:** Lab 25 — GPU FinOps Optimization (VinUni AI20k)  
> **Thời gian:** Tháng 10/2026  
> **Tài liệu nộp đính kèm:** `outputs/report.md`, `outputs/savings.png`, `outputs/focus_export.csv`

---

## 1. Baseline vs. Optimized: So Sánh Toàn Diện

| Chỉ Số Kinh Tế | Baseline (Trước tối ưu) | Optimized (Sau tối ưu) | Mức Cắt Giảm / Tiết Kiệm |
|---|:---:|:---:|:---:|
| **Tổng Chi Phí Hàng Tháng** | **$27,133** | **$14,626** | **-$12,507/tháng (-46.1%)** |
| - *Chi phí Inference hàng tháng* | $1,466 ($48.87/ngày) | $254 ($8.48/ngày) | -$1,212/tháng (-82.6%) |
| - *Chi phí GPU Compute (Purchasing)* | $25,667 | $15,627 | -$10,040/tháng (-39.1%) |
| - *Lãng phí Idle & Util-Lie thu hồi* | $1,255/tháng | $0/tháng | -$1,255/tháng (-100%) |
| **Đơn Vị Kinh Tế: $/1M-Token** | **$6.488** | **$1.126** | **-82.6%** |
| **Mức Xếp Chồng Chiết Khấu (Discount Stack)** | 1.000 (100% naive) | 0.050 (Stack 95% off) | Tiết kiệm tới 95% trên input cached + batch |

### Nhận định:
Thông qua việc chuyển đổi hệ quy chiếu từ đo lường theo **`$/GPU-giờ`** sang **`$/1M-token`**, hiệu quả phục vụ truy vấn tăng gấp **5.76 lần** cùng một ngân sách, giúp NimbusAI thoát khỏi tình trạng thâm hụt tài chính do chi phí GPU tăng phi mã.

---

## 2. Phân Tích Chuyên Sâu Từng Đòn Bẩy Tiết Kiệm (FinOps Levers)

Biểu đồ Waterfall (`outputs/savings.png`) ghi nhận 4 đòn bẩy tiết kiệm chủ chốt:

```
[Baseline: $27,133]
  ├── (-$10,040) Purchasing Strategy  ─────────────► [80.3% tổng savings]
  ├── (-$1,212)  Inference Levers     ─────────────► [9.7% tổng savings]
  ├── (-$655)    Right-sizing Util-Lies ───────────► [5.2% tổng savings]
  └── (-$600)    Kill Idle GPUs        ────────────► [4.8% tổng savings]
                                      ─────────────► [Optimized: $14,626]
```

### 2.1. Đòn bẩy đóng góp lớn nhất: Purchasing Strategy ($10,040/tháng — 80.3%)
- **Tại sao vượt trội?:** Chi phí hạ tầng điện toán thô (Compute capacity) chiếm hơn 94% hóa đơn tổng. Danh mục workload của NimbusAI gồm 8 jobs, trong đó có tới **5 jobs** (`job-train-llm`, `job-train-embed`, `job-finetune`, `job-dev-sandbox`, `job-batch-eval`) mang tính chất gián đoạn được (`interruptible=1`).
- **Cơ chế thực thi:**
  - Chuyển 5 jobs trên sang **Spot Instances** kết hợp cơ chế checkpoint định kỳ. Giá H100 Spot ($1.50/h) rẻ hơn 40% so với on-demand ($2.50/h). Ngay cả khi trừ chi phí overhead ghi checkpoint (3%) và thời gian chạy lại (rework ~0.5h khi có gián đoạn), Spot vẫn mang lại tỷ suất tiết kiệm ròng **40.9%**.
  - Đối với 3 dịch vụ inference chạy 24/7 (`job-infer-chat`, `job-infer-rag`, `job-infer-search`), duty cycle đạt 75%–100%, vượt xa điểm hòa vốn **55%** (tương ứng mức chiết khấu 45% của Reserved Instance 3 năm), giúp tiết kiệm ngay $4,000–$5,000/tháng mà không gặp bất kỳ rủi ro gián đoạn nào.

### 2.2. Đòn bẩy Inference ($1,212/tháng — Cắt giảm 82.6% đơn vị chi phí)
- **Cascade Routing:** Định tuyến 70% truy vấn đơn giản sang model `small` ($0.20/$0.40 per 1M tokens), chỉ dùng model `large` ($3.00/$15.00) cho truy vấn phức tạp $\rightarrow$ giảm 70–80% chi phí cơ sở.
- **Prompt Caching:** Tận dụng tiền tố hệ thống và ngữ cảnh RAG lặp lại, chiết khấu 90% giá input token đã cache (chỉ trả 10%).
- **Batch API:** Gom nhóm các yêu cầu không cần phản hồi thời gian thực (như `nightly-eval`) $\rightarrow$ giảm thêm 50% chi phí.

---

## 3. Bản Chất Kỹ Thuật của "GPU-Util Lie" & Tác Động Tài Chính

### 3.1. Các GPU bị phát hiện "nói dối":
1. **`gpu-h100-4` (H100):** `gpu_util_pct = 98.2%`, nhưng **MFU chỉ đạt 19.4%** và **MBU đạt 20.7%**.
2. **`gpu-a10g-1` (A10G):** `gpu_util_pct = 96.9%`, nhưng **MFU chỉ đạt 26.8%** và **MBU đạt 30.2%**.

### 3.2. Cơ chế kỹ thuật gốc rễ (Root Cause & Roofline Model):
- Lệnh `nvidia-smi` báo cáo `GPU-Util %` dựa trên mẫu thời gian mà driver kernel active trên GPU. Chỉ số này **không phản ánh số lượng phép tính FLOPs** hay công suất thực của Tensor Cores.
- **Mô hình Roofline:** H100 có điểm uốn (ridge point) là **295 FLOP/byte** (BF16).
  - Pha Prefill là compute-bound (intensity ~455 FLOP/byte), GPU khai thác tốt Tensor Cores.
  - Tuy nhiên, pha **Decode** là memory-bound (intensity chỉ **1–2 FLOP/byte**). Các SM cores liên tục bị **Memory Stall** do phải chờ nạp trọng số từ bộ nhớ HBM qua bus. Tensor Cores phần lớn thời gian phải chờ (idle), nhưng clock vẫn chạy $\rightarrow$ `nvidia-smi` báo 98%.
- **Overhead I/O & batch-size nhỏ:** Workload chạy batch=1 khiến GPU bị nghẽn giao tiếp host-device qua PCI-e, gây lãng phí năng lực tính toán.

### 3.3. Tác động tài chính:
NimbusAI đang trả trọn giá on-demand **$2.50/giờ** cho `gpu-h100-4`, nhưng chỉ nhận được thông lượng tương đương GPU phân khúc giá rẻ ($1.00–$1.79/h). Lãng phí thực tế là **$0.71/giờ** trên H100 và **$0.20/giờ** trên A10G, gây thất thoát **$655.20/tháng** (hơn **$7,860/năm**) chỉ trên 2 chiếc GPU đơn lẻ.

---

## 4. Kết Quả Thực Hiện Phần Mở Rộng (Hoàn Thành Cả 5 Extensions)

| Extension | Mục tiêu & Vị trí thực hiện | Kết quả định lượng đo được | Insight quan trọng nhất |
|---|---|---|---|
| **D.1 — Cải thiện `recommend_tier`** | Ma trận quyết định kết hợp Interruption Rate & Commitment Horizon (`finops/pricing.py`) | Mô phỏng chính xác spot rework (H100 ~4%, A10G ~8%). Khuyên dùng 1yr Reserved cho job ngắn hạn (<1 năm) thay vì khóa 3 năm. | Tránh rủi ro "stranded commitment" (cam kết chết) khi công nghệ model thay đổi nhanh. |
| **D.2 — Right-sizing theo MBU** | Phân tích kinh tế VRAM $/GB và băng thông đạt được (`missions/m1_efficiency_audit.py`) | Hạ `gpu-h100-4` xuống A100 và `gpu-a10g-1` xuống L4; tiết kiệm **$655.20/tháng** mà vẫn đảm bảo 100% trần băng thông yêu cầu. | Không thể chỉ chọn GPU rẻ nhất theo $/h (như L4); nếu không đủ VRAM hoặc BW thấp, thời gian chạy tăng 5x khiến tổng chi phí đội lên gấp bội. |
| **D.3 — `cache_is_worth_it`** | Kiểm toán ngưỡng hoàn vốn của Prompt Caching (`finops/pricing.py`, `m2`) | Điểm hòa vốn: **1.11 lần đọc**. Tần suất đọc lại thực tế trong log đạt trung bình **3.8 lần**, mang lại lợi nhuận ròng **$3.46/ngày**. | Prompt caching chỉ có lợi khi prefix được tái sử dụng $\ge 2$ lần; với hệ thống chat/RAG của NimbusAI, đây là lever siêu sinh lời. |
| **D.4 — Ngân sách Reasoning** | Tách bạch chi phí $ và điện năng Wh cho truy vấn Reasoning (`m2`, `m5`) | Reasoning chiếm **8.38% traffic** nhưng tiêu thụ tới **94.0% tổng năng lượng** (29.8 kWh/ngày). Cắt giảm 50% reasoning thừa tiết kiệm **$42.24/tháng** & **446.8 kWh/tháng**. | Thought tokens ngầm dài gấp 80x là "quả bom năng lượng". Cần speculative router để chỉ kích hoạt reasoning khi task phức tạp. |
| **D.5 — Carbon-aware Scheduling** | Tối ưu hóa điều phối theo cường độ phát thải 5 vùng (`missions/m3_purchasing.py`) | Chuyển toàn bộ 5 job interruptible (1,789 kWh) từ `us-east-1` sang `europe-north1` giúp giảm **92.1% CO2e** (cắt giảm 626.2 kg CO2e/tháng). | Workload huấn luyện không nhạy cảm độ trễ nên chạy tại vùng xanh (Na Uy) hoặc vùng điện rẻ `us-east-wa` ($0.055/kWh). |

---

## 5. Khuyến Nghị Chiến Lược Cho NimbusAI (Top 3 Hành Động của FinOps Lead)

Nếu đảm nhiệm vai trò FinOps Lead tại NimbusAI, tôi sẽ triển khai lộ trình 3 bước hành động ngay lập tức:

### Hành động 1: Thu hồi lãng phí tức thì — "Day-1 Automated Hygiene"
- **Nội dung:** Thiết lập script giám sát tự động (Cron Daemon / CloudWatch Event) kiểm tra telemetry GPU mỗi 15 phút. Tự động chuyển trạng thái hoặc dừng (Stop/Hibernate) bất kỳ GPU nào có `util% < 10%` trong 2 giờ liên tục.
- **Tác động:** Tiết kiệm ngay lập tức **$600/tháng** ($7,200/năm) từ GPU `gpu-h100-5` đang bị bỏ không mà không gây ảnh hưởng tới bất kỳ kỹ sư nào.

### Hành động 2: Tái cơ cấu mua sắm & Kỷ luật Tagging — "Week-1 Purchasing & Governance"
- **Nội dung:** 
  - Kích hoạt chính sách mua sắm lưỡng phân: Toàn bộ 5 pipeline training/batch chuyển sang sử dụng Spot Instances tích hợp thư viện lưu checkpoint tự động mỗi 30 phút. Đối với 3 cụm server inference chạy 24/7, thực hiện ký hợp đồng 1-năm hoặc 3-năm Reserved.
  - Thiết lập GitOps Pre-commit Hook và API Gateway bắt buộc kiểm tra trường tags `team` và `project`. Duy trì độ phủ tag trên 90% (hiện tại đạt 92%) để chính thức kích hoạt cổng thu phí nội bộ (**Chargeback Gate**), gắn trách nhiệm chi phí về từng trưởng bộ phận (Assistant, Search, Eval, RAG).
- **Tác động:** Thu hồi **$10,040/tháng** (80.3% tổng ngân sách lãng phí), chuyển đổi văn hóa kỹ thuật từ dùng chùa sang tự chủ ngân sách.

### Hành động 3: Hiện đại hóa tầng Gateway & Tách rời Kiến trúc — "Month-1 Architecture Optimization"
- **Nội dung:**
  - Cài đặt LiteLLM / Semantic Router tại cổng API để tự động thực thi Cascade Routing: 70% prompt phân loại là đơn giản được xử lý bằng model `small`, áp dụng Prompt Caching cho system prompts và chuyển batch ban đêm cho nightly eval.
  - Phân tách kiến trúc suy luận (**Prefill / Decode Disaggregation**): Đưa pha Prefill (compute-bound) lên các GPU tính toán mạnh và pha Decode (memory-bound) lên cụm GPU tối ưu băng thông bộ nhớ hoặc GPU tầm trung (A100/L4), triệt tiêu hoàn toàn hiện tượng GPU-Util Lie.
  - Triển khai **Carbon-aware Scheduler**: Tự động chuyển các container training sang vùng `europe-north1` hoặc `us-east-wa`.
- **Tác động:** Giảm đơn vị chi phí $/1M-token từ $6.488 xuống $1.126 (giảm 82.6%), giảm 92.1% khí thải carbon cho batch workloads, bảo đảm NimbusAI vận hành hiệu quả và bền vững lâu dài.
