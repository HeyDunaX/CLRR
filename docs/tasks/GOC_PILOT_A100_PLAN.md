# Kế hoạch Pilot A100 — Gold-Ordered Asymmetric Dropout Consistency (GOC) với 2 trọng số ($\beta \in \{1.25, 2.50\}$)

Ngày lập: **21/11/2026**
Nhánh làm việc: `work/goc-zero-step`
Phạm vi được phép: **CPU Gate + GPU 2-step Gate + Validation-Only Training (Seed 42) trên A100 40GB cho cả 2 trọng số $\beta = 1.25$ và $\beta = 2.50$ cùng các đối chứng khớp protocol.**
Phạm vi **KHÔNG** được phép: **CẤM mở hoặc đánh giá tập `test.csv` dưới mọi hình thức; CẤM chạy sang seed khác (43, 44) trước khi local review báo cáo validation; CẤM sửa receipts lịch sử hoặc đổi siêu tham số ngoài file config đã khóa.**

---

## 1. Mục tiêu, Cơ sở từ Chẩn đoán 0-Step và Thiết kế Thực nghiệm $2 \times 4 + 1$

### 1.1. Kết quả chẩn đoán 0-step dẫn tới pilot hai trọng số
Từ báo cáo chẩn đoán 0-step trên A100 ([GOC_ZERO_STEP_REPORT.md](reports/GOC_ZERO_STEP_REPORT.md), receipt [goc_zero_step_results.json](reports/goc_zero_step_results.json), commit `bfcda3f5514d6d1fe3dda3f4c4265974070fe312`):
1. Cả 6 điều kiện định lượng đều **PASS** (`failed_conditions: []`):
   - Trên nhóm token validation hiếm (`train freq <= 10`) và bất đồng mạnh (`|p1(y)-p2(y)| > 0.05`) của `W_S42` (`619` quan sát token×mask trên 83 câu): gradient phụ của `W` kéo tụt khoảng cách `gold - top_competitor` ở view thắng trong **88.69%** trường hợp (`[85.43%, 91.55%]`).
   - Đạo hàm khoảng cách trung bình hai view $\dot{m}_t$: `GOC1.25` đạt **`+0.2195`**, `GOC2.50` đạt **`+0.4390`** so với `W` (**`+0.0618`**), `SYM2.50` (`+0.0436`), `RAND2.50` (`+0.0503`), `INV2.50` (`-0.3519`). Lợi thế `GOC > SYM` lặp lại trên **8/8 cặp dropout masks**.
   - View thắng $w_t$ có tổng xác suất non-gold rơi vào top-100 subword tần suất cao **thấp hơn** view thua $\ell_t$ là **`-0.0723`** (`[-0.0937, -0.0515]`, âm trên 8/8 masks), bác bỏ giả thuyết $w_t$ chỉ là nhiễu prior tần suất cao.
2. **Vì sao chạy cả hai trọng số $\beta = 1.25$ và $\beta = 2.50$:**
   - Trong Part B (full-backbone parameter VJP ở `B=32` trên A100):
     - Tại **pretrained initialization (`B=32`)**: tỷ số norm gradient tham số toàn mạng $\|\nabla_\theta L_{\text{aux}}^{\text{GOC}(1.25)}\|_2 / \|\nabla_\theta L_{\text{aux}}^{W}\|_2 \in [0.909, 1.033]$ (trung vị **`0.999× W`**), tức **$\beta = 1.25$ khớp chuẩn lực cập nhật tham số với `W` ở đầu quá trình huấn luyện**.
     - Tại **trained checkpoint `W_S42` (`B=32`)**: tỷ số norm gradient tham số toàn mạng của `GOC(2.50) / W` nằm trong $[1.134, 1.236]$ (trung vị **`1.167× W`**), trong khi `GOC(1.25) / W` là $[0.567, 0.618]$. Đồng thời ở mức công thức logit, $\beta = 2.50$ giữ tổng trọng số KL bằng hai nhánh của `W` ($1.25 + 1.25 = 2.50$).
   - Do đó, tác giả đã duyệt chạy **cả 2 trọng số $\beta \in \{1.25, 2.50\}$** trong một bảng tổ hợp đối chứng hoàn chỉnh trên A100 để tách bạch hoàn toàn **(a) quy tắc chọn hướng teacher theo từng token (`GOC` vs `W` vs `SYM-SG` vs `INV`)** và **(b) cường độ regularization ($\beta = 1.25$ vs $\beta = 2.50$)**.

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
Để tránh mọi hiện tượng trôi dạt baseline khi chạy trên A100 (`/workspace/long/CLRR`), toàn bộ 9 runs trong `configs/experiments/amis_goc_pilot_a100.json` dùng chung khóa `configs/protocols/amis_a100_bleu.json`:
- **Dataset:** `amis_mandarin` (`tl_XX -> zh_CN`), `train.csv` (4600 câu, SHA256 `4e13d83f035a18c67e1b2e3990baff3aa92154a34b754a4d0cdef9a71d85b8e6`), `validation.csv` (576 câu, SHA256 `eac078ab237b55c10614f4db5ff026dc3a96a09a69c648ced29b5cdd8d4d11b4`).
- **Backbone:** `facebook/mbart-large-50-many-to-many-mmt`, revision `e30b6cb8eb0d43a0b73cab73c7676b9863223a30`.
- **Training recipe (khớp cohort A100 ES10):**
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

Tất cả 9 runs đều ở `account: 1`, `seed: 42`, chạy qua cùng `scripts/experiment.py`:

| Thứ tự | Run ID | Method ID | $\beta$ | Công thức $\mathcal{L}_{\text{aux}}(t)$ (cộng vào $\frac{1}{2}(\text{CE}_1 + \text{CE}_2)$) | Vai trò khoa học |
|---:|---|---|---:|---|---|
| 1 | `BASELINE_S42` | `baseline` | `0.00` | $0$ (1 view CE) | Baseline CE chuẩn trên A100 dưới selector `eval_bleu` |
| 2 | `W_B125_S42` | `rdrop-consistency-v1` | `1.25` | $1.25 \cdot \frac{1}{2}[\text{KL}(p_1\|p_2) + \text{KL}(p_2\|p_1)]$ | Đối chứng chính R-Drop chuẩn ($\beta=1.25$, khớp `W` trên RTX) |
| 3 | `W_B250_S42` | `rdrop-consistency-v1` | `2.50` | $2.50 \cdot \frac{1}{2}[\text{KL}(p_1\|p_2) + \text{KL}(p_2\|p_1)]$ | Đối chứng R-Drop đối xứng ở $\beta=2.50$ (khớp `B` trên RTX) |
| 4 | **`GOC_B125_S42`** | `gold-ordered-consistency-v1` | **`1.25`** | $1.25 \cdot \text{KL}(\operatorname{sg}(p_{w_t}) \| p_{\ell_t})$ | **Ứng viên GOC ở $\beta=1.25$** (khớp norm gradient toàn mạng với `W` tại init) |
| 5 | **`GOC_B250_S42`** | `gold-ordered-consistency-v1` | **`2.50`** | $2.50 \cdot \text{KL}(\operatorname{sg}(p_{w_t}) \| p_{\ell_t})$ | **Ứng viên GOC ở $\beta=2.50$** (khớp tổng hệ số KL hai chiều và norm tại `W42`) |
| 6 | `GOC_SYM_B125_S42` | `symmetric-stopgrad-consistency-v1` | `1.25` | $1.25 \cdot \frac{1}{2}[\text{KL}(\operatorname{sg}(p_1)\|p_2) + \text{KL}(\operatorname{sg}(p_2)\|p_1)]$ | Đối chứng EDWARD-NMT Eq. 13 tất định ở $\beta=1.25$ (có `sg`, không chọn theo gold) |
| 7 | `GOC_SYM_B250_S42` | `symmetric-stopgrad-consistency-v1` | `2.50` | $2.50 \cdot \frac{1}{2}[\text{KL}(\operatorname{sg}(p_1)\|p_2) + \text{KL}(\operatorname{sg}(p_2)\|p_1)]$ | Đối chứng EDWARD-NMT Eq. 13 tất định ở $\beta=2.50$ (cùng kỳ vọng gradient với `RAND2.50`) |
| 8 | `GOC_INV_B125_S42` | `inverted-gold-consistency-v1` | `1.25` | $1.25 \cdot \text{KL}(\operatorname{sg}(p_{\ell_t}) \| p_{w_t})$ | Đối chứng bác bỏ nhân quả (falsifier: yếu dạy mạnh) ở $\beta=1.25$ |
| 9 | `GOC_INV_B250_S42` | `inverted-gold-consistency-v1` | `2.50` | $2.50 \cdot \text{KL}(\operatorname{sg}(p_{\ell_t}) \| p_{w_t})$ | Đối chứng bác bỏ nhân quả (falsifier: yếu dạy mạnh) ở $\beta=2.50$ |

---

## 4. Hướng dẫn Thực thi trên Server A100 (`/workspace/long/CLRR`)

### Bước 1: Đồng bộ nhánh `work/goc-zero-step` và kiểm tra môi trường
```bash
cd /workspace/long/CLRR
git status --short --branch
git fetch origin work/goc-zero-step
git switch work/goc-zero-step
git merge --ff-only origin/work/goc-zero-step
git rev-parse HEAD
```
- Xác nhận `HEAD` khớp với commit được giao. Không ghi đè hoặc commit các file ngoài phạm vi (`docs/README.md`, `outputs_rebuttal/...`, `tmp/`).
- Kiểm tra `HF_TOKEN` trong môi trường shell trước khi chạy `train`:
  - `engine.py:train_run()` dùng `os.environ['HF_TOKEN']` để upload và verify `best.zip` cùng receipts lên `FiveC/amis-rewire-checkpoints`.
  - Nếu `HF_TOKEN` đã có quyền write lên `FiveC/amis-rewire-checkpoints`, chạy trực tiếp quy trình chuẩn.
  - Nếu trong phiên terminal hiện tại **chưa** export `HF_TOKEN`, hãy kiểm tra xem token có sẵn trong cấu hình Hugging Face CLI (`~/.cache/huggingface/token` hoặc `$HF_HOME/token`) không; nếu có, export `export HF_TOKEN="$(cat ~/.cache/huggingface/token)"` (tuyệt đối không in giá trị token ra log/report/git). Nếu hoàn toàn không có `HF_TOKEN` trên server, dừng trước bước `train` (sau khi chạy xong `inspect`, `build`, `cpu`, và `gate` cho cả 9 runs) và báo ngay để người dùng cấp `HF_TOKEN` vào terminal.

### Bước 2: Chạy CPU Unit Tests, `inspect`, `build`, và `cpu` gate
```bash
conda run -n clrr python -X utf8 -m unittest tests/test_gold_ordered_consistency.py -v
conda run -n clrr python -X utf8 scripts/experiment.py inspect --config configs/experiments/amis_goc_pilot_a100.json
conda run -n clrr python -X utf8 scripts/experiment.py build --config configs/experiments/amis_goc_pilot_a100.json
conda run -n clrr python -X utf8 scripts/experiment.py cpu --config configs/experiments/amis_goc_pilot_a100.json
```
- Lưu ý về tokenizer trên server A100 (`sentencepiece 0.2.3` + `protobuf 6.33.6`): `engine.py:load_model()` đã được bổ sung fallback an toàn sang `use_fast=False` nếu `use_fast=True` gặp lỗi chuyển đổi protobuf trên bản snapshot gốc (đã chứng minh khớp 100% token IDs trên toàn bộ 4.600 câu train và 576 câu validation trong `outputs/experiments/goc_zero_step_a100/tokenizer_verification.json`).

### Bước 3: Chạy GPU 2-Step Gates và Validation-Only Training cho 9 Runs
Có thể chạy toàn bộ queue `--account 1 --stage gate-train`:
```bash
conda run -n clrr python -X utf8 scripts/experiment.py run --config configs/experiments/amis_goc_pilot_a100.json --account 1 --stage gate-train
```
Hoặc chạy tường minh toàn bộ 9 GPU gates trước rồi mới train tuần tự từng run:
```bash
for id in BASELINE_S42 W_B125_S42 W_B250_S42 GOC_B125_S42 GOC_B250_S42 GOC_SYM_B125_S42 GOC_SYM_B250_S42 GOC_INV_B125_S42 GOC_INV_B250_S42; do
  conda run -n clrr python -X utf8 scripts/experiment.py run --config configs/experiments/amis_goc_pilot_a100.json --id "$id" --stage gate
done

for id in BASELINE_S42 W_B125_S42 W_B250_S42 GOC_B125_S42 GOC_B250_S42 GOC_SYM_B125_S42 GOC_SYM_B250_S42 GOC_INV_B125_S42 GOC_INV_B250_S42; do
  conda run -n clrr python -X utf8 scripts/experiment.py run --config configs/experiments/amis_goc_pilot_a100.json --id "$id" --stage train
done
```

---

## 5. Phân tích Hậu kiểm trên Tập Validation & Báo cáo Bàn giao

Sau khi 9 runs hoàn tất `VAL_COMPLETE.json`, tổng hợp báo cáo chi tiết vào:
- `docs/tasks/reports/GOC_PILOT_A100_REPORT.md`
- `docs/tasks/reports/goc_pilot_a100_summary.json`
- `docs/tasks/reports/goc_pilot_a100_epoch_trajectories.csv`

Nội dung bắt buộc trong báo cáo:
1. **Bảng kết quả tổng hợp 9 runs trên tập Validation (576 câu, beam 1 FP32):**
   - `Run ID | Method | beta | Best Epoch / Step | Stop Reason | Val BLEU | Val chrF0 | Val chrF++ | Val Loss (at best step) | Peak GPU GiB | Wall Time (s) | Archive SHA256 | HF Revision`.
2. **Phân tích hiệu ứng chính và tương tác $2 \times 4$ ($\beta \in \{1.25, 2.50\} \times \{\text{W}, \text{GOC}, \text{SYM-SG}, \text{INV}\}$):**
   - So sánh `GOC_B125_S42` vs `W_B125_S42`, `GOC_SYM_B125_S42`, `GOC_INV_B125_S42`, và `BASELINE_S42`.
   - So sánh `GOC_B250_S42` vs `W_B250_S42`, `GOC_SYM_B250_S42`, `GOC_INV_B250_S42`, và `W_B125_S42`.
   - Đối chiếu với ngưỡng quyết định đã khóa trước:
     - **Quality screen:** $\text{BLEU}(\text{GOC}) - \text{BLEU}(W_{\text{best}}) \ge +0.30$ và không tụt `chrF0`.
     - **Mechanism screen:** $\text{BLEU}(\text{GOC}(\beta)) - \text{BLEU}(\text{SYM-SG}(\beta)) \ge +0.20$ ở cùng $\beta$, và $\text{BLEU}(\text{GOC}(\beta)) > \text{BLEU}(\text{INV}(\beta))$.
3. **Phân tích cơ chế trên `validation_predictions.csv` (576 câu) của cả 9 runs:**
   - Dùng cùng định nghĩa tần suất ký tự Hán trong `train.csv` như báo cáo Mục 3 (`<=10` là hiếm: 789 lượt tham chiếu trên validation; `11–100` là trung bình; `>100` là phổ biến):
     - Tính **Rare CJK Character Recall (`<=10`)**, **Medium CJK Recall (`11–100`)**, **Frequent CJK Recall (`>100`)** cho từng run trong 9 runs.
     - Tính tần suất train trung vị / trung bình của các ký tự dự đoán sai hoặc thay thế trên các câu bất đồng giữa `GOC` và `W` / `SYM-SG`.
     - Đếm số câu `GOC` thắng / hòa / thua `W` và `SYM-SG` theo sentence-level BLEU / chrF0, kèm ví dụ câu cụ thể.
4. **Động học huấn luyện theo epoch (`final_trainer_state.json` & `loss_components.jsonl`):**
   - Quỹ đạo `eval_bleu`, `eval_chrf++`, `eval_loss`, `ce`, `kl`, `winner_view1_fraction`, `tie_fraction` qua 20 epochs cho từng run.

### Bước 4: Bàn giao kết quả về Local qua Git / Git Bundle
- Chỉ stage các file báo cáo và JSON/CSV tổng hợp nhỏ trong `docs/tasks/reports/`:
  - `docs/tasks/reports/GOC_PILOT_A100_REPORT.md`
  - `docs/tasks/reports/goc_pilot_a100_summary.json`
  - `docs/tasks/reports/goc_pilot_a100_epoch_trajectories.csv`
- **Không** đưa weights, `.cache/`, hay toàn bộ thư mục `outputs/experiments/amis_goc_pilot_a100/` vào Git.
- Commit trên nhánh `work/goc-zero-step`:
  - Thử `git push origin work/goc-zero-step`.
  - Nếu server vẫn bị chặn `HTTP 403` khi push GitHub, tạo file bundle `goc_pilot_a100_handoff.bundle` tại `/workspace/long/CLRR/goc_pilot_a100_handoff.bundle` bằng lệnh:
    ```bash
    git bundle create goc_pilot_a100_handoff.bundle origin/work/goc-zero-step..work/goc-zero-step
    ```
    rồi báo commit SHA và kích thước bundle để local tải về merge.
