# DRM: kiểm chứng cơ chế sau audit novelty

Ngày 2026-11-18. Chẩn đoán GPU được tác giả duyệt qua yêu cầu “vậy làm đi”. Không optimizer step, không test, không đổi method đã train.

## Trạng thái cuối: COMPLETE, chưa mở rộng training

Bản hợp lệ là `outputs/experiments/drm_direction_probe_v3/REPORT.md`, `SUMMARY.json`, `paired_differences.json`, `COMPLETE.json` và `download_receipt.json`. Đủ32 TRAIN rows trên mỗi W42/DRM42; 576 records can thiệp; parameters unchanged; peak2484.89 MiB allocated. Dữ liệu nhỏ đã tải về và hash khớp.

|Checkpoint cố định|DRM trajectory − sequential-zero NLL|CI95% theo câu TRAIN|
|---|---:|---|
|W42|−0.000004607|[−0.000013412,+0.000002902]|
|DRM42|−0.000003316|[−0.000013017,+0.000005761]|

NLL thấp hơn tốt hơn. Hướng gradient tốt hơn context/scalar khoảng1e-5 NLL trên mẫu này, nhưng so với không update thì cả hai CI chứa0. DRM-checkpoint so random-axis control: −0.000006614, CI[−0.000017482,+0.000003355]. Single-update trên DRM: −0.000001618 so không update, CI[−0.000006655,+0.000002765]. Đây là CI exploratory pointwise, không phải multiple-testing-corrected significance hay uncertainty giữa training seeds.

**Quyết định:** chưa có bằng chứng về lợi ích thực tế đủ mạnh của decision-conditioned memory; không mở replication/A100 hoặc training controls dài. Điều này không chứng minh mọi biến thể DRM bất khả. Nếu tiếp tục nghiên cứu cơ chế thì cần validation diagnostic/annotation phù hợp; TRAIN proxy không đủ để nhận semantic consumption. Gate chất lượng cũ vẫn chưa đạt.

V3 max norm-match error2.3164e-7, max contraction coefficient.554311; zero/native max logit error2.2888e-5 trong tolerance FP32 đã khóa. NLL effects dùng sequential-zero để loại offset giữa kernels native/tuần tự. Không diễn giải norm contraction thành bảo toàn nghĩa.

### Sửa phép đo, không sửa method

- V1 `drm_direction_probe`: random half-space suy biến, norm-match error lên1 và hệ số>1. Random comparisons không hợp lệ; raw và SOURCE.py giữ để truy vết; INVALIDATION.md đánh dấu.
- V2 `drm_direction_probe_v2`: guard dừng ở hệ số4.6361; không có complete scientific result. Không tính là một run training thất bại.
- V3 `drm_direction_probe_v3`, PID3754633: random **axis** chọn tốt nhất trong8 axes và hai dấu theo support memory/attention, không dùng nhãn hoặc gradient để chọn hướng. Đây là state-conditioned random-axis control, không phải isotropic signed random control hay faithful QED/Mi replication. Các điều kiện norm/contraction giữ nguyên; 3/3 CPU tests PASS, bao gồm fixture mọi source slots giống nhau.
- Source method DRM SHA giữ nguyên `0f87d1a7049eead8388832e21e924c3ca99339f2bd3f8b67694655f0ef939508`. Không đổi checkpoint, objective, eta hoặc protocol training. Chỉ sửa diagnostic controls và baseline NLL.

## Phạm vi

- Script: `scripts/analysis/probe_drm_directions.py`; kiểm thử: `tests/test_drm_direction_probe.py` (3/3 PASS CPU).
- RTX output hợp lệ: `/workspace/ura/thanhdat/projects/CLRR/outputs/experiments/drm_direction_probe_v3`.
- PID V3: 3754633, đã COMPLETE. PID3751113 thuộc V1 không hợp lệ. GPU trước đợt đo: 23952 MiB free, 257 MiB used, utilization0%; disk free190.94 GiB; project lock rảnh.
- Launcher: `outputs/experiments/drm_direction_probe_dispatch/launch.py`; project lock giữ suốt queue; không can thiệp process khác.
- Hai checkpoint public được khóa revision và SHA: W42 và DRM42; chi tiết lưu trong PLAN.json của phép đo. Restore tạm chỉ trong output riêng, xóa sau khi đã đọc và xác minh archive public.
- 32 câu TRAIN chọn bằng permutation seed1729, data hash theo recipe cũ; không chọn mẫu theo kết quả. FP32 SDPA TF32off, eval/dropoutoff, max length256. B1 cho phép đo, không nhận đây là B4/GA32 training hoặc một lần đánh giá chất lượng dịch.

## Can thiệp

1. Native eta0 và sequential-zero: kiểm tra logits khớp (tolerance FP32 đã khóa trong script).
2. Full trajectory: DRM, context direction, deterministic random direction và scalar decay.
3. Single update tại vị trí supervised giữa câu; các update khác bằng0. Kiểm tra logits ở vị trí đó/trước đó không đổi; đo gold NLL ở token sau và toàn phần sau.
4. Context/random/scalar được khớp norm update của DRM từng head tại cùng trạng thái đang đo. Đây là can thiệp chẩn đoán, **không phải recipe train**. Nếu train control sau này cần khóa calibration TRAIN riêng, không copy adaptive norm-matching làm control rẻ hơn.
5. Attention-argmax reuse chỉ là proxy, không phải gold one-to-many alignment. Không kết luận semantic preservation từ proxy này.
6. Báo norm-match error và hệ số co tối đa; hệ số>1 có thể phá tính co, cần đọc trước khi kết luận. Random/context controls vẫn tính gradient để cô lập khác biệt direction, thời gian này không phản ánh implementation control được tối ưu.
7. Xác minh parameter versions và .grad không đổi, không optimizer. NLL thấp hơn tốt hơn; bootstrap theo câu TRAIN chỉ mô tả uncertainty trên mẫu, không thay thế seed/generalization inference.

## Gate cũ cần giữ nguyên

Nguồn `configs/experiments/amis_drm_validation_fix_rtx.json` / screening:

| Tiêu chí đã khóa | DRM42 hiện có | Kết luận |
|---|---:|---|
|Validation BLEU−W >=+.30|+.200029806|Chưa đạt|
|Validation chrF++−W >=0|−.017085220|Chưa đạt|

Vì vậy chưa mở replication nhiều seed hoặc A100 confirmation. Kết quả test đã xem không dùng để bỏ gate này. Phép đo cơ chế phục vụ quyết định giữ/sửa/bỏ giả thuyết; nó không tự chứng minh một DRM phiên bản mới sẽ có chất lượng tốt hơn.

## Lệnh

Local CPU (conda clrr): `python -m unittest discover -s tests -p test_drm_direction_probe.py -v` với PYTHONPATH=src.

RTX (conda clrr dưới project, launcher giữ lock và environment):

```bash
python -u scripts/analysis/probe_drm_directions.py \
  --config configs/experiments/amis_drm_validation_fix_rtx.json \
  --output outputs/experiments/drm_direction_probe_v3 --rows 32
```

Không chạy trùng lệnh khi queue còn sống. PLAN/status/measurements/SUMMARY/COMPLETE là artifacts chính; chỉ tải JSON/CSV/Markdown nhỏ về local.

## Nghĩ tiếp sau phép đo — chưa nhận method mới

Audit tiếp hai hướng, primary papers và hai vòng NotebookLM đã lưu ở `outputs/experiments/post_drm_design_review/DESIGN_REVIEW.md`. Shared-error negative loss trùng unlikelihood; shared-prefix cross-source ranking là logistic assignment ranking, còn chồng lấn họ natural-pair đã khảo sát. Không mở pilot hai hướng này. CPU recheck CSV cũ khớp counts/means; không coi 165/165 Amis TRAIN source margins dương là W validation accuracy hoặc semantic correctness.

Câu hỏi còn đáng phân biệt: có sửa được quyết định token bằng cách đọc lại chính source values hiện có hay không? Phép đo sau đó đã được tác giả duyệt và hoàn tất như dưới đây; không mở training/test.

## W42 read recoverability — 2026-11-18 COMPLETE

**[ĐO ĐƯỢC]** `outputs/experiments/w_read_recoverability_rtx/REPORT.md`, raw `measurements.json`, `PLAN.json`, `GATES.json`, `SUMMARY.json` và `download_receipt.json`. W42 public revision/hash giữ nguyên. 32 câu VALIDATION chọn bằng permutation1729; mỗi câu lấy token gold không-special ở giữa, không chọn theo lỗi. Có12 native-correct và20 native-wrong. Gold prefixes cố định; last decoder cross-attention only; source keys/values và toàn bộ parameters cố định. 576 interventions =32×2 target-label controls×3 read spaces×3 budgets. FP32 SDPA TF32off eval, max256; hai restarts, tối đa24 bước local search mỗi restart. Tối ưu biến can thiệp theo margin gold/alternative so fixed native rival, **không cập nhật weights**.

|Read space|Sửa thành gold top1, budget3%|Budget10%|Budget30%|Mean gold NLL delta trên20 lỗi, budget30%|
|---|---:|---:|---:|---:|
|Query, giữ nguyên K/V|0/20|0/20|0/20|−0.304510|
|Attention simplex, giữ nguyên V|0/20|0/20|0/20|−0.759516|
|Direct context, tập khả thi rộng hơn|0/20|0/20|2/20|−2.378260|

Budget là norm thay đổi context ghép các heads trước out-projection, chia native-context norm; không phải tỷ lệ sửa token. Actual mean displacement ở30%: query.288256, simplex.300000, direct.300000. Native source slots vẫn bao gồm special tokens; không nhận lexical/semantic attribution từ phép này.

**[ĐO ĐƯỢC]** Trên18 vị trí mà gold và alternative đều native-wrong, rank chênh tối đa1: direct-context30% đưa gold lên top1 ở1/18 và alternative ở1/18 (cùng vị trí); query/simplex đều0/18. Chênh gold−alternative margin gain của direct30% +.192066, sentence-bootstrap10000 CI95%[−.328638,+.716139]. Cả9 read/budget comparisons đều có CI margin-gain contrast chứa0. Bootstrap paired-recovery CI suy biến[0,0] vì outcomes giống nhau, **không chứng minh equivalence**. 0/20 recovery có exact-binomial CI95% upper khoảng16.8%, không phải xác suất thật bằng0. Các CI exploratory pointwise, không seed-level confirmation/multiple-testing evidence.

**[SUY LUẬN]** Trong phạm vi mẫu và local search này, chưa có tín hiệu sửa lỗi gold có chọn lọc để biện minh training một cơ chế đọc nguồn mới. Direct-context NLL cải thiện mạnh hơn không đủ: arbitrary alternative cũng steer được. Hai vị trí direct30% vượt fixed native rival nhưng vẫn không thành gold top1 cho thấy margin một đối thủ không đại diện toàn bộ quyết định vocabulary.

**[ĐO ĐƯỢC]** Trong20 native-wrong tokens,17 token có gold rank>5; median gold rank65.5 và median native gold−rival margin−5.653001. Mẫu này chứa nhiều lỗi xa biên quyết định hiện tại, không chỉ nhầm hai token sát nhau. Chỉ last decoder cross-attention được can thiệp; các cơ chế ở tầng trước chưa được đo bởi phép này.

**[GIỚI HẠN / QUYẾT ĐỊNH]** Không kết luận encoder thiếu nghĩa, mọi attention intervention thất bại, hay retraining không thể cải thiện. Đây là bounded local oracle search, không global optimum; chỉ32 teacher-forced tokens trên một checkpoint. Chưa chốt method và không tự mở train/grid/budget lớn hơn. Hướng tiếp cần giải thích lỗi có phân loại và dự đoán chọn lọc vượt arbitrary-label steering, thay vì suy từ oracle NLL giảm thành gain dịch.

Technical:2/2 CPU tests PASS; context/native và downstream/native replay PASS, max logit error2.2888184e-5; parameter versions unchanged, .grad absent; peak2394.90 MiB. PID3762972 đã thoát; RTX free23952 MiB trước/sau; own restore đã xóa sau public SHA verification. Runtime Python3.13.16/Torch2.6.0+cu124/Transformers4.57.6. Loader phát cảnh báo Mistral regex khi đọc local checkpoint; không đổi tokenizer/flag của protocol. `RUNTIME_CHECK.json` lưu nhánh guard nguồn Transformers: nhánh local config4.57.6 có thể phát cảnh báo này cho model không phải Mistral. So sánh can thiệp dùng chung tokenizer archive khóa; không nhận cảnh báo là bằng chứng số lịch sử sai.

Code/lệnh (conda clrr):

```bash
python -m unittest discover -s tests -p test_read_recoverability_probe.py -v
python scripts/analysis/run_read_recoverability.py status
python scripts/analysis/run_read_recoverability.py collect
python scripts/analysis/summarize_read_recoverability.py
```

Source SHA `b72c9837db19a230552a54763e78fa4b7390152ad8969a2bfc02a3b739cd3994`. Không chạy lại `launch`: output đã COMPLETE. Chỉ JSON nhỏ được tải về; report/summary sinh từ chúng, không tải weights và chưa upload artifacts mới lên HF trong lượt này.
