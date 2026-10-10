from pathlib import Path

root = Path(__file__).resolve().parents[2]
path = root / 'docs/operations/NEXT_EXPERIMENTS_HANDOFF_20261005.md'
text = path.read_text(encoding='utf-8')
start = text.index('## Đợt ES10 đã được cho chạy')
end = text.index('## Lưu checkpoint hiện hành', start)
current = '''## Đợt ES10 đang chạy — 08/10/2026

**APPROVED_14_RUNS; ORIGINAL_12_CPU_AND_GPU_GATES_PASS; TWO_SUPPLEMENT_CPU_GATES_PASS.** Cả ba runtime là A100 Standard 40GB/no high-mem, chạy conda `clrr`. Tác giả đã duyệt thêm Strong LoRA mBART A/B và giao hai run này cho Colab3 sau mT5 Middle. Không thay bundle 12 runs đang hoạt động.

| Máy | Queue theo thứ tự | Số runs |
|---|---|---:|
| Colab1 | B01–B05: NLLB Baseline, Middle, Strong A, Strong B, BitFit | 5 |
| Colab2 | A01–A04: mBART Baseline46, Middle42, Narrow LoRA42, BitFit42; D01–D02: ByT5 Baseline/Middle42 | 6 |
| Colab3 | C01: mT5 Middle42 → A05: mBART Strong LoRA A42 → A06: mBART Strong LoRA B42 | 3 |

Colab3 còn 20.94 CU khi kiểm tra trước khi đặt queue bổ sung, rate 5.30 CU/giờ. Controller đã xác minh sống và ở trạng thái `WAITING_FOR_C01_VERIFIED_BACKUP`; chỉ chạy hai GPU gates rồi training sau khi queue C01 hoàn tất và `VAL_COMPLETE` có backup verified. Không tắt Colab3 ngay sau C01. Không suy ETA từ size model.

- A05/A06: LR 2e-4/1e-4, microbatch4 × accumulation32, lengths256, validation beam1; r16/alpha32/dropout.05, all-linear targets q/k/v/out/fc1/fc2; B thêm tied embeddings/lm_head trainable. Cùng ES10/max20epochs/raw CSV references/FP32 generation với cohort mới.
- Run cũ Strong mBART đủ20epochs nhưng source đã xác minh chọn checkpoint theo decoded references và Torch2.11; cohort mới dùng raw references và Torch2.6. Đây là lý do rerun, không kết luận số cũ sai.
- Worker gốc: `scripts/revalidation/run_standardized_controls.py`; supplemental adapter: `run_standardized_strong_mbart.py`; preparation: `prepare_strong_mbart_supplement.py`; scheduler: `schedule_strong_mbart_supplement.py`.
- Original plan/status: `outputs_rebuttal/amis_standardized_controls/`; bổ sung: `outputs_rebuttal/amis_standardized_strong_mbart/`, gồm PLAN, manifest, CPU_GATE, launch receipt, schedule probe.
- Mỗi model xong upload public vào `FiveC/amis-rewire-checkpoints/checkpoints/amis_mandarin/<backbone>/<method>/seed-<n>/es10/`; xác minh hash trước khi dọn weights runtime. Không tải weights về PC, không overwrite số/file cũ.
- A01 đã backup verified; replay độc lập public validation PASS: BLEU16.68918323, chrF++14.06492586, chrF0 16.12515646. Đây là validation, không phải test. Colab2 đang chạy A02.
- **Test chỉ mở sau khi đủ14 validation selections và backup verified độc lập.** Worker gốc có barrier12; không được dùng riêng barrier đó để test sớm. Supplemental worker hiện chỉ có CPU/gate/train, chưa có test stage.
- Gate attempt1 đã bắt lỗi input-gradient ở frozen embeddings với reentrant checkpointing trước khi có research training. Đã bổ sung hooks trên encoder/decoder embeddings thực sự được gọi, không unfreeze weights hoặc đổi LR/batch/rank. CPU checkpoint/reload và cả12 GPU gates đã PASS; giữ failure snapshot ở `gate_attempt1/`.
- BF16 train, FP32/TF32-off generation, validation NLLB beam4/các backbone khác beam1, test beam4. Không pool phương pháp cũ vào cohort đối sánh ES10 mới; không mở thêm runs ngoài14 nếu chưa được duyệt.

Trạng thái này thay các câu `PLAN_NOT_LAUNCHED` trong phần kế hoạch lịch sử phía dưới. Theo dõi và báo chat mỗi10 phút; ghi scores cuối sau khi replay public artifacts.

'''
path.write_text(text[:start] + current + text[end:], encoding='utf-8')
print('UPDATED_HANDOFF_CURRENT_14_RUNS_COLAB3')
