# Bàn giao: tìm cấu hình CLRR trên mBART Amis

Ngày: 03/10/2026. Workspace: `D:\Code\CLRR`.

## 1. Yêu cầu mới nhất và điểm dừng

Tác giả yêu cầu **chạy xong model hiện tại rồi tạm ngưng**, viết hướng dẫn để chat khác
tiếp tục. Yêu cầu này thay thế việc tự động chạy hết suite trong phiên hiện tại.
Không tự chạy tiếp khi đọc file này; đợi tác giả yêu cầu tiếp tục.

- Model hiện tại: `mbart-amis-full_a020_l010-seed42`, Full CLRR+LSR, α=0.20, λ=0.10.
- Điều phối đã bị chặn trước lượt tiếp theo; model hiện tại tiếp tục train và backup.
- Trạng thái cuối, kết quả và xác nhận tắt Colab sẽ được cập nhật dưới đây khi hoàn tất.
- **Không chạy Turkish**, không chạy NLLB và chưa nâng cấp kiến trúc trong vòng này.

## 2. Mục tiêu khoa học

Tìm cấu hình tốt **trong ngân sách đã khai báo**, rồi kiểm tra CLRR có đóng góp vượt
LSR-only được tune hay không. Không chọn cấu hình bằng test và không bảo đảm kết quả tốt.
Xem `CLRR_GO_NOGO_PLAN.md`, `EXPERIMENTS_UPDATED.md`, `RESEARCH_INSIGHTS.md`, `METRICS.md`.
Paper hiện tại là `docs/clrr_main.tex`; chưa đưa các trial tuning mới vào paper.

Nhóm C cũ: Baseline, CLRR-only, LSR-only, Full × seeds42/43/44. Protocol matched đã chạy.
Vòng D mới: tối đa20 lượt train Amis, gồm8 screening + tối đa6 confirmation + tối đa6 controls.
Đã chạy cũng phải tính vào ngân sách, không loại trial có kết quả thấp.

## 3. Kết quả đã có và điều cần cảnh giác

### Audit C không train — đã hoàn tất

Artifacts: `outputs_rebuttal/amis_confirmation_20261003/metric_influence/`.
12systems ×575rows; predictions/reference hashes và thứ tự khớp; chrF++ tái tính khớp <1e-8.

| Character-only chrF | Seed42 | Seed43 | Seed44 | Mean |
| --- | ---: | ---: | ---: | ---: |
| Full−Baseline | +0.2654 | +1.1001 | −0.3865 | +0.3263 |
| Full−LSR | +0.3046 | −0.0676 | +0.2655 | +0.1675 |

Seed43 Full−Baseline chrF++ +5.3162 còn +0.5803 khi bỏ riêng row170.
Character-only gap cùng seed còn +0.8602, nên gain chrF++ lớn không chứng minh gain rộng.
Đây là sensitivity, **không sửa điểm chính, không xóa câu**. Không có p-value mới từ audit này.

### Screening D — chỉ validation

| Trial | α | λ | Seed | Validation BLEU | Validation chrF++ | Validation chrF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full gốc, reuse metadata C | 0.10 | 0.10 | 42 | 17.6437 | 14.7317 | Chưa có validation predictions |
| `full_a005_l010` | 0.05 | 0.10 | 42 | 17.1225 | 14.4092 | 16.4998 |
| `full_a020_l010` | 0.20 | 0.10 | 42 | Đang chờ kết thúc | Đang chờ kết thúc | Đang chờ kết thúc |

α=0.05 chưa cải thiện so cấu hình gốc ở validation seed42. Chưa kết luận toàn bộ phương pháp.
Validation576rows chỉ có **1 word-bigram reference ở row197**; test575rows có3 word-bigrams.
Trial α0.05 không match bigram đó; bỏ row197 chrF++=14.4440, chrF=16.5395.
Phải kiểm tra thứ hạng candidate bằng character-only và leave-one-out trước confirmation;
không tự đổi primary metric chỉ để lấy kết quả đẹp.

Các audit mới:

- `scripts/revalidation/audit_configuration_validation.py` → `reference_word_statistics.json`.
- `scripts/revalidation/audit_configuration_predictions.py` → `validation_influence.json`.
- Thư mục D: `outputs_rebuttal/amis_configuration_20261003_v1/`.
- `inputs.json`: input/source hashes và metadata gốc dùng để reuse.
- `runs/<run>/metrics.json`, `COMPLETE.json`, `validation_predictions.csv`: kết quả từng trial.
- `status_latest.json`: snapshot từ Colab, không tự coi snapshot cũ là tiến độ đang chạy.

## 4. Cấu hình phải giữ để so công bằng

| Hạng mục | Giá trị |
| --- | --- |
| Backbone | `facebook/mbart-large-50-many-to-many-mmt` |
| Revision pretrained | `e30b6cb8eb0d43a0b73cab73c7676b9863223a30` |
| Data | `data_processed/amis_mandarin/{train,validation,test}.csv`, `source,target` |
| Proxy source / target | `tl_XX` / `zh_CN` |
| LR / weight decay / warmup | `5e-5` / `0` / `0.06` |
| Train batch / accumulation / eval batch | `32` / `4` / `8` |
| Max source / target length | `256` / `256` |
| Max epochs / patience | `20` / `4` |
| Routing | Encoder, distance2, cached source detach mặc định |
| Selection | Raw-reference validation corpus chrF++, beam1 |
| Final test | Beam4, FP32, TF32 tắt trong evaluation |
| Train | BF16, TF32, gradient checkpointing |
| Library | torch2.6.0+cu124, transformers4.57.6, sacrebleu2.6.0, hub0.36.0, accelerate1.15.0 |
| Params | 610,879,488, tất cả trainable; extra params0 |
| GPU / environment | Colab A10040GB Standard, **không high-mem**, conda `clrr` |

`src/amis_rewire/train.py --validation-only` không load actual test dataset; đã smoke test
với test.csv bị bỏ khỏi fixture. GPU smoke bốn nhánh cũng đã đạt.
Generation FP32 là tường minh, không dựa vào BF16 autocast mặc định của Trainer.

## 5. Checkpoint, backup và phục hồi

Repo: `FiveC/amis-rewire-checkpoints`. Prefix mới, không trùng C/lịch sử:

```text
amis_configuration_20261003_v1/<run>/<run>-best.zip
amis_configuration_20261003_v1/<run>/metrics.json
amis_configuration_20261003_v1/<run>/COMPLETE.json
amis_configuration_20261003_v1/<run>/validation_predictions.csv
```

`COMPLETE.json` chứa archive SHA256, bytes, commit revision, prediction SHA256.
Chỉ xóa weights/archive trên Colab **sau khi HF size và etag khớp SHA256**.
Token đã được tác giả cho phép dùng trong launcher trước; helper lấy token an toàn,
không in token, không đưa token vào Markdown hoặc bundle.

Hai weights **Full/LSR seed42 của C gốc** không còn; chỉ còn test predictions/metrics.
Không thay bằng epoch2 hoặc checkpoint lịch sử khác rồi gọi là best C.
10weights C còn lại có receipts tại `outputs_rebuttal/amis_confirmation_20261003/HF_UPLOAD.json`.

### Phục hồi screening trên A100 mới khi được yêu cầu chạy tiếp

Chạy PowerShell tại `D:\Code\CLRR`. Python local luôn dùng `conda run -n clrr`.
Nếu `colab status` báo phiên cũ vẫn tồn tại, kiểm tra trước; không tạo thêm A100 song song.

```powershell
colab status -s colab
colab new -s colab --gpu A100
colab upload -s colab scratch/amis_configuration_code.zip /content/amis_configuration_code.zip
colab exec -s colab -f scratch/start_configuration_setup.py --timeout 60
```

Setup chạy nền. Kiểm tra đến khi `ENV_READY` tồn tại và CUDA báo A100:

```powershell
colab exec -s colab -f scratch/configuration_setup_status.py --timeout 60
```

**Không chạy lại `scratch/package_configuration_search.py` để thay zip hiện có.**
Zip ban đầu cùng `inputs.json` khóa source/data của screening; source mới có thể đổi hashes.
Phục hồi receipts trước để không train lại hai trial đã xong:

```powershell
conda run -n clrr python -X utf8 scratch/set_configuration_colab_auth.py
colab exec -s colab -f scratch/restore_configuration_search_remote.py --timeout 120
colab upload -s colab scratch/monitor_configuration_remote.py /content/monitor_configuration_remote.py
colab exec -s colab -f scratch/launch_configuration_search_remote.py --timeout 43200
```

Restore giải nén zip, đối chiếu protocol HF với bundle, tải small artifacts, xác minh
archive metadata pinned revision và prediction hash. **Không tải weights và không truy cập test.**
Runner bỏ qua trial có COMPLETE; dừng nếu gặp run folder dở dang để kiểm tra thay vì overwrite.
Lệnh launch giữ kernel hoạt động; cần theo dõi trong lúc chạy, không tắt máy trước backup.
Trong Codex, lỗi truy cập profile Colab/SSH do sandbox cần chạy công cụ có quyền network/profile;
không diễn giải lỗi profile là runtime đã mất.

### Giám sát và lưu artifacts local

```powershell
conda run -n clrr python -X utf8 scratch/snapshot_configuration_local.py
conda run -n clrr python -X utf8 scratch/sync_configuration_predictions.py
conda run -n clrr python -X utf8 scripts/revalidation/audit_configuration_predictions.py
```

Snapshot lấy metrics/receipts; sync lấy validation predictions có SHA đúng. Báo tiến độ chat mỗi10phút.
SSH monitor cần `LD_LIBRARY_PATH=/usr/lib64-nvidia:/usr/local/cuda/lib64`; helper đã đặt.
Thiếu libnvidia-ml trong SSH không tự chứng minh CUDA training bị lỗi.

## 6. Chính xác còn phải chạy gì

### A. Hoàn thành sáu trial screening còn lại, seed42

Runner `scripts/run_amis_configuration_search.py` đã thực hiện được phần này:

| Thứ tự | Run label | Method CLI | α | λ | Câu hỏi |
| --- | --- | --- | ---: | ---: | --- |
| 3 | `full_a010_l003` | `jepa-clrr-enc` | 0.10 | 0.03 | Giảm LSR có giúp Full? |
| 4 | `lsr_l003` | `jepa` | 0 | 0.03 | Full có vượt matched LSR λ0.03? |
| 5 | `full_a010_l030` | `jepa-clrr-enc` | 0.10 | 0.30 | Tăng LSR có giúp Full? |
| 6 | `lsr_l030` | `jepa` | 0 | 0.30 | Full có vượt matched LSR λ0.30? |
| 7 | `lsr_l001` | `jepa` | 0 | 0.01 | LSR nhẹ hơn có tốt hơn? |
| 8 | `lsr_l060` | `jepa` | 0 | 0.60 | LSR mạnh hơn có tốt hơn? |

Cả Full và LSR có5trial gồm cấu hình gốc λ0.1 reuse; chưa chạy grid3×3.
Sau mỗi trial: HF verify → sync local → audit validation → ghi ledger/insight và báo tốt/xấu.
Runner hiện tại **chỉ làm screening**, không tự làm B/C/D bên dưới.

### B. Khóa candidate từ validation, rồi xác nhận seeds43/44

1. Gộp validation của8trial mới và2trial gốc từ `inputs.json`.
2. Chọn một Full và một best-tuned LSR bằng validation chrF++ đã khai báo.
3. Kiểm tra BLEU, character-only và ảnh hưởng row197; ghi rõ thiếu validation predictions
   của Full/LSR gốc seed42. Nếu ranking phụ thuộc rare bigram, phân tích trước khi tiếp tục;
   không đổi metric ngầm hoặc dùng test chọn lại.
4. Viết `selection_lock.json`: α/λ, best LSR λ, hashes, ranking, seed list, protocol,
   lý do chọn, scope và thời điểm khóa **trước khi xem test candidate**.
5. Chạy Full candidate ×seeds43/44, matched-λ LSR ×seeds43/44, best-tuned LSR ×seeds43/44.
   Tối đa6lượt; nếu hai LSR trùng thì chỉ train một lần; nếu đúng cấu hình C đã có thì reuse.
6. So validation giữa Full và hai controls qua seed; nếu lợi ích không lặp, dừng mở rộng
   và báo NO-GO/tín hiệu chưa đủ. Không tiếp tục chỉ để đủ20runs.

Để tạo command cho các lượt mới, dùng toàn bộ argv trong `current_command.json`
hoặc runner hiện tại; chỉ đổi `--run-name`, `--seed`, `--method`, `--rewire-strength`,
`--jepa-weight` phù hợp. Giữ `--validation-only` và mọi hạng mục ở mục4.
Lưu archive/receipt giống runner; code điều phối B/C chưa được viết, chat tiếp phải bổ sung.

### C. Controls CLRR-only và bỏ detach

- Nếu α cuối khác0.1: CLRR-only `--method clrr-enc --jepa-weight 0`
  với α cuối, seeds42/43/44. Tối đa3runs. Nếu α0.1, dùng đúng C CLRR-only hiện có.
- Full bỏ detach **chỉ ở cached CLRR source**, giữ LSR target no-grad, cùng α/λ cuối,
  seeds42/43/44, tối đa3runs.
- Entry có sẵn: `scripts/revalidation/train_amis_no_detach.py`, nhận các args của trainer.
- Implementation: `src/amis_rewire/ablation.py`.
- **Không chỉ xóa `.detach()` trong hook**: với reentrant gradient checkpointing,
  hook forward chạy no-grad. Attached control đặt routing ngoài checkpoint call từng layer.
- CPU smoke `tests/smoke_attached_routing.py` đã đạt: forward/loss như detached,
  plain/checkpoint gradients khớp, source gradient bổ sung thực sự hoạt động.
- **GPU gate chưa chạy**. Trước train control, upload source/test/entry trong bundle riêng,
  không sửa source screening đang chạy; chạy:

```bash
LD_LIBRARY_PATH=/usr/lib64-nvidia:/usr/local/cuda/lib64 \
  /content/miniforge3/envs/clrr/bin/python tests/smoke_attached_routing.py --cuda
```

Kiểm tra thêm tiny BF16 training/checkpoint reload bằng entry mới trước train thật.
Không có bằng chứng gradient của detached implementation cũ bị lỗi; đây là control cơ chế mới.

### D. Đánh giá test cuối và quyết định

Sau khóa candidate và đạt gates, load **đúng archive/commit/SHA**. Strict-check missing,
unexpected và mismatched keys; tokenizer/proxy đúng; FP32 beam4/raw references/TF32off.
Nguồn helper inference/extraction: `reevaluate_checkpoints.py`, `measure_checkpoint_mechanisms.py`.
Scripts cũ có manifest mặc định nhóm B; phải truyền/viết manifest D riêng, không chạy mặc định
rồi gọi là D. No-detach inference forward như detached, nhưng weights/provenance phải đúng.

- Báo từng seed, mean/std, paired deltas; bootstrap sentences không thay training-seed evidence.
- Family chính D: Full−Baseline, Full−matchedLSR, Full−bestLSR ×2metrics ×3seeds =Holm18.
  Nếu hai LSR trùng, giữ18slots với unused p=1, không giảm family để tăng significance.
- Family controls riêng Holm18: CLRR−Baseline, Full−CLRR, Full−noDetach.
- **Không sửa A/B/C raw CSV/JSON, không gộp các protocol và không tính lại Holm24 C.**
- GO dự án: mean Full−Baseline≥0.5BLEU, Full−matchedLSR≥0.3,
  Full thắng matchedLSR3/3seeds, baseline≥2/3, mean Full−bestLSR>0,
  không harm hệ thống trên chrF/chrF++.
- GO có điều kiện: gain nhỏ hơn nhưng lặp3/3 vs matchedLSR, vượt bestLSR trung bình.
- NO-GO: không vượt controls, chỉ một seed thắng, gain chỉ do rare-ngram hoặc harm rõ.
  Đây là tiêu chí dự án, **không phải chuẩn acceptance ACL**.
- Nếu chưa ổn: báo nguyên nhân/giả thuyết tối ưu cho tác giả; không tự sweep nâng cấp,
  không chạy Turkish để né kết quả Amis. Turkish cần tác giả cho phép riêng.

## 7. Lưu tài liệu, dừng GPU và các file không được chạy nhầm

- Số mới → `EXPERIMENTS_UPDATED.md`; diễn giải → `RESEARCH_INSIGHTS.md` hiện hành.
- Số cũ/paper ban đầu đã ở archive, không phục hồi vào file active.
- File này là bàn giao thao tác, không tạo thêm một ledger số liệu cạnh tranh.
- Sau backup và sync:

```powershell
colab stop -s colab
colab status -s colab
```

`scratch/finish_current_configuration_remote.py` là helper **một lần** để dừng đúng phiên
hiện tại, chứa PID cụ thể; **không chạy lại helper này ở runtime mới**.
`scratch/launch_configuration_search_remote.py` là launcher screening có thể chạy lại
sau restore; không thay bằng các launcher Turkish/PEFT/historical có tên tương tự.

## 8. Câu giao việc cho chat tiếp

> Đọc docs/CONFIGURATION_SEARCH_HANDOFF.md, docs/CLRR_GO_NOGO_PLAN.md và hai file
> EXPERIMENTS_UPDATED/RESEARCH_INSIGHTS. Tiếp tục sáu trial screening mBART Amis còn lại,
> phục hồi receipts HF để không train lại trial đã xong, giữ protocol validation-only,
> conda clrr, Colab A100 Standard không high-mem. Sau screening chọn và khóa candidate
> từ validation, rồi confirmation và controls theo gates/ngân sách; không chạy Turkish.
> Báo chat mỗi10phút, backup FiveC prefix riêng và chỉ dọn weights sau verify.
