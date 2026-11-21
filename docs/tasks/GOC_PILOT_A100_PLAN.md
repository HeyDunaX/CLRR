# Kế hoạch Pilot A100 — Gold-Ordered Asymmetric Dropout Consistency (GOC) với 2 mức hệ số ($\beta \in \{1.25, 2.50\}$)

Ngày lập: **21/11/2026**
Nhánh làm việc: `work/goc-zero-step`
Quyết định của tác giả sau khi review báo cáo 0-step ([GOC_ZERO_STEP_REPORT.md](reports/GOC_ZERO_STEP_REPORT.md)):
- Báo cáo 0-step đạt đủ 6 tiêu chí định lượng (`failed_conditions: []`) và kết thúc ở trạng thái `HOLD` chờ local review.
- **Tác giả đã review kết quả 0-step và chính thức duyệt mở Pilot A100 mới gồm 9 runs** (2 mức $\beta \in \{1.25, 2.50\} \times 4$ objectives $\{\text{W}, \text{GOC}, \text{SYM-SG}, \text{INV}\} + \text{Baseline}$, Seed 42, validation-only), thay cho kế hoạch 3 runs RTX ban đầu.
- **Triển khai theo 2 pha tách biệt (Cách A):**
  - **Pha 1 (Đã duyệt chạy ngay):** Chạy CPU unit tests, `inspect`, `build`, `cpu` gate, và chạy riêng **`--stage gate` (2 optimizer steps + save/reload + resume check trên A100) cho từng ID trong cả 9 runs**. Báo cáo trạng thái 9/9 GPU gates và **peak VRAM ở `microbatch=32`** rồi **DỪNG** chờ lệnh mở Pha 2.
  - **Pha 2 (Chờ lệnh sau khi 9/9 GPU gates PASS):** Chạy tuần tự `--stage train` (20 epochs, ES10 theo `eval_bleu` trên validation) cho cả 9 runs và xuất báo cáo phân tích validation. **Tuyệt đối không dùng `--account 1 --stage gate-train`** để tránh việc một model vào train dài khi các model còn lại chưa qua GPU gate.
- **Quy định về tập `test.csv`:** Chỉ cho phép runner đọc file `data_processed/amis_mandarin/test.csv` để **kiểm tra SHA256 và đóng gói source/data bundle** theo protocol (`config.py` / `engine.py`). **Không dùng tập test để sinh bản dịch, tính điểm hay chọn checkpoint.**

---

## 1. Mục tiêu, Cơ sở từ Chẩn đoán 0-Step và Thiết kế Thực nghiệm $2 \times 4 + 1$

### 1.1. Kết quả chẩn đoán 0-step dẫn tới pilot hai mức hệ số $\beta$
Từ báo cáo chẩn đoán 0-step trên A100 ([GOC_ZERO_STEP_REPORT.md](reports/GOC_ZERO_STEP_REPORT.md), receipt [goc_zero_step_results.json](reports/goc_zero_step_results.json), commit `bfcda3f5514d6d1fe3dda3f4c4265974070fe312`):
1. Cả 6 điều kiện định lượng đều **PASS** (`failed_conditions: []`):
   - Trên nhóm token validation hiếm (`train freq <= 10`) và bất đồng mạnh (`|p1(y)-p2(y)| > 0.05`) của `W_S42` (`619` quan sát token×mask trên 83 câu): gradient phụ của `W` kéo tụt khoảng cách `gold - top_competitor` ở view thắng trong **88.69%** trường hợp (`[85.43%, 91.55%]`).
   - Đạo hàm khoảng cách trung bình hai view $\dot{m}_t$: `GOC1.25` đạt **`+0.2195`**, `GOC2.50` đạt **`+0.4390`** so với `W` (**`+0.0618`**), `SYM2.50` (`+0.0436`), `RAND2.50` (`+0.0503`), `INV2.50` (`-0.3519`). Lợi thế `GOC > SYM` lặp lại trên **8/8 cặp dropout masks**.
   - View thắng $w_t$ có tổng xác suất non-gold rơi vào top-100 subword tần suất cao **thấp hơn** view thua $\ell_t$ là **`-0.0723`** (`[-0.0937, -0.0515]`, âm trên 8/8 masks), bác bỏ giả thuyết $w_t$ chỉ là nhiễu prior tần suất cao.
2. **Vì sao so sánh hai mức hệ số thử $\beta = 1.25$ và $\beta = 2.50$:**
   - Trong Part B (full-backbone parameter VJP trên A100):
     - Tại **pretrained initialization**: tỷ số norm gradient tham số toàn mạng $\|\nabla_\theta L_{\text{aux}}^{\text{GOC}(1.25)}\|_2 / \|\nabla_\theta L_{\text{aux}}^{W}\|_2$ là **`0.909–1.033× W`** ở `B=32` (và `0.946–0.996× W` ở `B=4`), trong khi `GOC(2.50) / W` là **`1.818–2.066× W`** ở `B=32`.
     - Tại **trained checkpoint `W_S42`**: tỷ số norm gradient tham số toàn mạng của `GOC(1.25) / W` là **`0.567–0.783× W`** (`0.567–0.618×` ở `B=32`, `0.625–0.783×` ở `B=4`), còn của `GOC(2.50) / W` là **`1.134–1.566× W`** (`1.134–1.236×` ở `B=32`, `1.250–1.566×` ở `B=4`).
   - Như vậy, không một hằng số $\beta$ cố định nào khớp chính xác norm của `W` ở cả đầu lẫn cuối quá trình train: $\beta = 1.25$ gần quy mô norm của `W` tại khởi tạo nhưng nhỏ hơn tại `W42`, còn $\beta = 2.50$ là mức hệ số thứ hai cao hơn (`1.134–1.566× W` tại `W42`, chưa thể gọi là *norm-matched*).
   - Việc chạy thiết kế tổ hợp **$2 \times 4 + 1$** ($\beta \in \{1.25, 2.50\} \times \{\text{W}, \text{GOC}, \text{SYM-SG}, \text{INV}\} + \text{Baseline}$) trong cùng một cohort A100 cho phép tách bạch rõ ảnh hưởng của **mức hệ số $\beta$** và **quy tắc chọn hướng teacher theo từng token**.

---

## 2. Cấu trúc Module và Protocol Khóa Chống Trôi Dạt Baseline

### 2.1. Quy tắc thư mục method mới
Method mới được đóng gói cô lập hoàn toàn theo chuẩn dự án tại:
- `src/methods/gold_ordered_consistency/__init__.py`
- `src/methods/gold_ordered_consistency/v1/__init__.py`
- `src/methods/gold_ordered_consistency/v1/model.py` (`GoldOrderedConsistencyMT`, kế thừa trực tiếp `PairedRouteMT` với `'paired-native-rdrop-v1'`, giữ nguyên 100% cơ chế hai lượt forward dropout, RNG, reentrant gradient checkpointing, FP32 chunked loss reduction 32 tokens, valid-token mean `labels != -100`, và native single-pass inference).
- `src/methods/gold_ordered_consistency/v1/validation.py` (`validate_recipe`, `require_admission` kiểm tra khóa protocol, design review SHA256 và biên nhận 0-step SHA256).
- `tests/test_gold_ordered_consistency.py` và `tests/smoke_gold_ordered_consistency.py`.

### 2.2. Khóa Protocol A100 (`configs/protocols/amis_a100_bleu.json`)
Toàn bộ 9 runs trong `configs/experiments/amis_goc_pilot_a100.json` dùng chung khóa `configs/protocols/amis_a100_bleu.json`:
- **Dataset:** `amis_mandarin` (`tl_XX -> zh_CN`), `train.csv` (4600 câu, SHA256 `4e13d83f035a18c67e1b2e3990baff3aa92154a34b754a4d0cdef9a71d85b8e6`), `validation.csv` (576 câu, SHA256 `eac078ab237b55c10614f4db5ff026dc3a96a09a69c648ced29b5cdd8d4d11b4`), `test.csv` (575 câu, SHA256 `2e3cea3f9011ef6ee237a437b57a1ac21bd9fae9682473bce00c99a80afde0aa` — chỉ kiểm tra SHA256 khi build/validate).
- **Backbone:** `facebook/mbart-large-50-many-to-many-mmt`, revision `e30b6cb8eb0d43a0b73cab73c7676b9863223a30`.
- **Training recipe (cohort A100 ES10):**
  - `seed = 42`, `data_seed = 42`
  - `microbatch = 32`, `accumulation = 4` (effective batch = 128; 36 optimizer steps/epoch, 720 steps tối đa cho 20 epochs)
  - `learning_rate = 5e-5`, `optim = "adamw_torch_fused"`, `scheduler = "linear"`, `warmup_ratio = 0.06`, `weight_decay = 0.0`, `max_grad_norm = 1.0`
  - `bf16 = true`, `tf32_training = true`, `attention_implementation = "sdpa"`
  - `gradient_checkpointing = true`, `gradient_checkpointing_use_reentrant = true`
  - `epochs = 20`, `patience = 10` (`early_stopping_enabled = true`)
- **Validation & Selection recipe:**
  - `selection_metric = "eval_bleu"` (`"original-reference validation BLEU; beam1 FP32"`)
  - `generation_precision = "fp32"`, `tf32_generation = false`, `validation_beams = 1`, `eval_batch = 8`, `bleu_tokenizer = "zh"`, `log_validation_chrf0 = true`.
  - `test_barrier = "No test stage in this pilot"`.

---

## 3. Bảng 9 Runs trong `configs/experiments/amis_goc_pilot_a100.json`

| Thứ tự | Run ID | Method ID | $\beta$ | Công thức $\mathcal{L}_{\text{aux}}(t)$ (cộng vào $\frac{1}{2}(\text{CE}_1 + \text{CE}_2)$) | Vai trò khoa học |
|---:|---|---|---:|---|---|
| 1 | `BASELINE_S42` | `baseline` | `0.00` | $0$ (1 view CE) | Baseline CE chuẩn trên A100 dưới selector `eval_bleu` |
| 2 | `W_B125_S42` | `rdrop-consistency-v1` | `1.25` | $1.25 \cdot \frac{1}{2}[\text{KL}(p_1\|p_2) + \text{KL}(p_2\|p_1)]$ | Đối chứng R-Drop đối xứng ở mức hệ số $\beta=1.25$ |
| 3 | `W_B250_S42` | `rdrop-consistency-v1` | `2.50` | $2.50 \cdot \frac{1}{2}[\text{KL}(p_1\|p_2) + \text{KL}(p_2\|p_1)]$ | Đối chứng R-Drop đối xứng ở mức hệ số $\beta=2.50$ |
| 4 | **`GOC_B125_S42`** | `gold-ordered-consistency-v1` | **`1.25`** | $1.25 \cdot \text{KL}(\operatorname{sg}(p_{w_t}) \| p_{\ell_t})$ | **Ứng viên GOC ở mức hệ số $\beta=1.25$** |
| 5 | **`GOC_B250_S42`** | `gold-ordered-consistency-v1` | **`2.50`** | $2.50 \cdot \text{KL}(\operatorname{sg}(p_{w_t}) \| p_{\ell_t})$ | **Ứng viên GOC ở mức hệ số $\beta=2.50$** |
| 6 | `GOC_SYM_B125_S42` | `symmetric-stopgrad-consistency-v1` | `1.25` | $1.25 \cdot \frac{1}{2}[\text{KL}(\operatorname{sg}(p_1)\|p_2) + \text{KL}(\operatorname{sg}(p_2)\|p_1)]$ | Đối chứng EDWARD-NMT Eq. 13 tất định ở $\beta=1.25$ |
| 7 | `GOC_SYM_B250_S42` | `symmetric-stopgrad-consistency-v1` | `2.50` | $2.50 \cdot \frac{1}{2}[\text{KL}(\operatorname{sg}(p_1)\|p_2) + \text{KL}(\operatorname{sg}(p_2)\|p_1)]$ | Đối chứng EDWARD-NMT Eq. 13 tất định ở $\beta=2.50$ |
| 8 | `GOC_INV_B125_S42` | `inverted-gold-consistency-v1` | `1.25` | $1.25 \cdot \text{KL}(\operatorname{sg}(p_{\ell_t}) \| p_{w_t})$ | Đối chứng bác bỏ nhân quả (yếu dạy mạnh) ở $\beta=1.25$ |
| 9 | `GOC_INV_B250_S42` | `inverted-gold-consistency-v1` | `2.50` | $2.50 \cdot \text{KL}(\operatorname{sg}(p_{\ell_t}) \| p_{w_t})$ | Đối chứng bác bỏ nhân quả (yếu dạy mạnh) ở $\beta=2.50$ |

---

## 4. Hướng dẫn Thực thi Pha 1 & Pha 2 trên Server A100 (`/workspace/long/CLRR`)

### Pha 1 (Đã duyệt chạy ngay): Đồng bộ, CPU Gate và 9 GPU 2-Step Gates (`--stage gate`)

#### Bước 1.1: Đồng bộ nhánh `work/goc-zero-step`
```bash
cd /workspace/long/CLRR
git status --short --branch
git fetch origin work/goc-zero-step
git switch work/goc-zero-step
git merge --ff-only origin/work/goc-zero-step
git rev-parse HEAD
```
- Xác nhận `HEAD` khớp với commit được giao. Không đụng tới các file ngoài phạm vi (`docs/README.md`, `outputs_rebuttal/...`, `tmp/`).

#### Bước 1.2: Chạy CPU Unit Tests, `inspect`, `build`, và `cpu` gate
```bash
conda run -n clrr python -X utf8 -m unittest tests/test_goc_zero_step.py tests/test_gold_ordered_consistency.py -v
conda run -n clrr python -X utf8 scripts/experiment.py inspect --config configs/experiments/amis_goc_pilot_a100.json
conda run -n clrr python -X utf8 scripts/experiment.py build --config configs/experiments/amis_goc_pilot_a100.json
conda run -n clrr python -X utf8 scripts/experiment.py cpu --config configs/experiments/amis_goc_pilot_a100.json
```

#### Bước 1.3: Chạy riêng `--stage gate` cho cả 9 ID (kiểm tra 2 bước optimizer, save/reload, resume và peak VRAM ở `B=32`)
```bash
for id in BASELINE_S42 W_B125_S42 W_B250_S42 GOC_B125_S42 GOC_B250_S42 GOC_SYM_B125_S42 GOC_SYM_B250_S42 GOC_INV_B125_S42 GOC_INV_B250_S42; do
  conda run -n clrr python -X utf8 scripts/experiment.py run --config configs/experiments/amis_goc_pilot_a100.json --id "$id" --stage gate
done
```
- **Điểm dừng cuối Pha 1:** Sau khi vòng lặp `--stage gate` kết thúc, đọc `outputs/experiments/amis_goc_pilot_a100/gates/<ID>/PASS.json` của cả 9 ID, báo cáo bảng tổng hợp (`ID | passed | max_memory_gb | seconds | save_reload_logits_match | optimizer_scheduler_rng_resume_matches`) và trạng thái `HF_TOKEN` (có sẵn hay chưa, không in giá trị token), rồi **DỪNG trước `--stage train`** chờ người dùng xác nhận mở Pha 2.

---

### Pha 2 (Chỉ chạy sau khi 9/9 GPU Gates PASS và người dùng ra lệnh mở train)

```bash
for id in BASELINE_S42 W_B125_S42 W_B250_S42 GOC_B125_S42 GOC_B250_S42 GOC_SYM_B125_S42 GOC_SYM_B250_S42 GOC_INV_B125_S42 GOC_INV_B250_S42; do
  conda run -n clrr python -X utf8 scripts/experiment.py run --config configs/experiments/amis_goc_pilot_a100.json --id "$id" --stage train
done
```
- **Theo dõi thanh tiến trình (Progress Bar & Live Board) khi train:**
  - Trong luồng chạy `train`, callback `Progress` tự động in thanh tiến trình trực tiếp ra stdout mỗi 10 bước:
    `PROGRESS GOC_B125_S42 [######--------------] 30.6% step 220/720 (ep 6.11) loss=3.1420 best=16.8500 ETA=8.2m`
  - Ở một tab Terminal JupyterLab khác (hoặc khi kiểm tra tiến độ), chạy bảng theo dõi toàn bộ 9 runs:
    - Xem 1 lần: `conda run -n clrr python -X utf8 scripts/experiment.py watch --config configs/experiments/amis_goc_pilot_a100.json`
    - Tự động cập nhật mỗi 15 giây: `conda run -n clrr python -X utf8 scripts/experiment.py watch --config configs/experiments/amis_goc_pilot_a100.json --interval 15`

---

## 5. Phân tích Hậu kiểm trên Tập Validation (sau Pha 2)

Sau khi 9 runs hoàn tất `VAL_COMPLETE.json` ở Pha 2, tổng hợp báo cáo vào:
- `docs/tasks/reports/GOC_PILOT_A100_REPORT.md`
- `docs/tasks/reports/goc_pilot_a100_summary.json`
- `docs/tasks/reports/goc_pilot_a100_epoch_trajectories.csv`

Nội dung bắt buộc:
1. **Bảng kết quả tổng hợp 9 runs trên tập Validation (576 câu, beam 1 FP32):**
   - `Run ID | Method | beta | Best Epoch / Step | Stop Reason | Val BLEU | Val chrF0 | Val chrF++ | Val Loss (at best step) | Peak GPU GiB | Wall Time (s) | Archive SHA256 | HF Revision`.
2. **Phân tích bảng $2 \times 4$ ($\beta \in \{1.25, 2.50\} \times \{\text{W}, \text{GOC}, \text{SYM-SG}, \text{INV}\}$ + `BASELINE_S42`)** và đối chiếu ngưỡng quyết định đã khóa trước (`BLEU(GOC) - BLEU(W_best) >= +0.30`, `BLEU(GOC(beta)) - BLEU(SYM-SG(beta)) >= +0.20`, `BLEU(GOC(beta)) > BLEU(INV(beta))`).
3. **Phân tích cơ chế trên `validation_predictions.csv` (576 câu):** Rare CJK Character Recall (`<=10`), Medium (`11–100`), Frequent (`>100`), tần suất ký tự thay thế trên các câu bất đồng, và quỹ đạo 20 epochs.
