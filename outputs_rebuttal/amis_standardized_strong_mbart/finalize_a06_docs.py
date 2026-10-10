from pathlib import Path
import json,re,requests,hashlib
root=Path(r'D:\Code\CLRR')
base=root/'outputs_rebuttal/amis_standardized_controls'
supp=root/'outputs_rebuttal/amis_standardized_strong_mbart'
lock=json.loads((base/'selection_lock_14.json').read_text(encoding='utf-8'))
assert lock['status']=='ALL_14_VALIDATION_VERIFIED' and len(lock['runs'])==14
summary='''## Đối sánh ES10 đã hoàn tất — 09/10/2026

**ALL_14_VALIDATION_VERIFIED.** Đủ 14 runs đã hoàn tất, replay độc lập public predictions/metrics và kiểm tra hash checkpoint PASS. Bảng validation đầy đủ ở cuối file; đây chưa phải điểm test.

A06 mBART Strong LoRA B seed42 chạy lại trên Colab1 A100 Standard 40GB, hoàn tất 20 epochs; chọn checkpoint epoch19/step684 theo raw-reference validation chrF++. Validation: BLEU **10.9850**, chrF++ **10.3825**, chrF0 **12.0560**. Dừng vì epoch cap. Các 13 run trước không train lại và scores replay không đổi.

Checkpoint public: `checkpoints/amis_mandarin/mbart-large-50/strong-lora-b/seed-42/es10/best.zip`; SHA256 `4cf6ccac866db298b93134a69b015545271a9ef18793dcda483e42cf718a5a94`. Joint selection lock public: `registry/plans/amis_standardized_controls/selection_lock_14.json`. Index public đã cập nhật đủ checkpoint ES10; tổng104 archives, giữ nguyên weights cũ.

Colab1/2/3 đều xác nhận **Active assignments=0**, **Usage rate=0.00/hr**; balance tương ứng **52.92 / 151.73 / 8.57 CU**. Không tải weights về PC. Receipt audit: `outputs_rebuttal/amis_standardized_strong_mbart/status/completion_audit.json`.

**Bước tiếp theo:** evaluation test beam4/FP32 từ đúng 14 checkpoints đã khóa, rồi replay test metrics và ghi bảng so sánh. **Chưa chạy test trong task này.** Những ghi chú đang chạy/tạm dừng ở phần lịch sử bên dưới đã được trạng thái này thay thế.

'''
p=root/'docs/operations/NEXT_EXPERIMENTS_HANDOFF_20261005.md'
t=p.read_text(encoding='utf-8')
start=t.index('## Đợt ES10 đang chạy — 08/10/2026')
end=t.index('### Phân bổ bổ sung A06 vì unit Colab3',start)
t=t[:start]+summary+'### Lịch sử phân bổ bổ sung A06 vì unit Colab3'+t[end+len('### Phân bổ bổ sung A06 vì unit Colab3'):]
pause='''<!-- standardized-es10-user-pause -->
## Lịch sử tạm dừng A06 — 09/10/2026

Theo yêu cầu tác giả, Colab1 đã dừng sau lần chạy đầu chưa hoàn tất; kiểm tra Active assignments=0, Usage rate=0.00/hr, balance65.29 CU. Quan sát training cuối đã lưu: 00:00 ngày09/10, epoch9.7235/20, step350/720; không coi đây là trạng thái cuối khi tắt. Receipt: `outputs_rebuttal/amis_standardized_strong_mbart/status/user_pause_receipt.json`. Trạng thái này đã được lần chạy lại hoàn tất ở đầu file thay thế.
<!-- end-standardized-es10-user-pause -->'''
restart='''<!-- standardized-es10-a06-restart -->
## Lịch sử chạy lại riêng A06 — 09/10/2026

Public HF khi kiểm tra trước restart: A05 PASS, A06 chưa có VAL_COMPLETE hợp lệ. Đã mở lại Colab1 A100 Standard40GB và chạy A06 từ đầu bằng bundle đã khóa. Tiny GPU gate PASS: BF16, accumulation32, checkpointing, save/reload logits và best-checkpoint reload. Lúc08:46, epoch0.2783/20, step10/720 đã được xác nhận live. Đợt chạy này đã hoàn tất, xem trạng thái đầu file.

Launcher operational: `outputs_rebuttal/amis_standardized_strong_mbart/restart_a06_local.py`; receipt: `restart_a06_receipt.json` cùng thư mục. Monitor: `scripts/revalidation/monitor_standardized_controls.py --account 1 --supplement`; remote log: `/content/strong_mbart_b_account1.log`.
<!-- end-standardized-es10-a06-restart -->'''
t=re.sub(r'<!-- standardized-es10-user-pause -->.*?<!-- end-standardized-es10-user-pause -->',lambda _:pause,t,flags=re.S)
t=re.sub(r'<!-- standardized-es10-a06-restart -->.*?<!-- end-standardized-es10-a06-restart -->',lambda _:restart,t,flags=re.S)
p.write_text(t,encoding='utf-8')
p=root/'docs/experiments/EXPERIMENTS_UPDATED.md'
t=p.read_text(encoding='utf-8')
marker='<!-- standardized-es10-final-status -->'
if marker not in t:
    cut=t.index('\n')+1
    t=t[:cut]+'\n'+marker+'\n'+summary+'<!-- end-standardized-es10-final-status -->\n'+t[cut:]
p.write_text(t,encoding='utf-8')
for relative in ['docs/operations/NEXT_EXPERIMENTS_HANDOFF_20261005.md','docs/experiments/EXPERIMENTS_UPDATED.md']:
    t=(root/relative).read_text(encoding='utf-8')
    table=t.split('<!-- standardized-es10-validation-progress -->',1)[1].split('<!-- end-standardized-es10-validation-progress -->',1)[0]
    assert len(re.findall(r'^\| [ABCD]\d\d \|',table,re.M))==14
    assert '| A06 | mbart-large-50 | strong-lora-b | 42 | 20 | 19 | 10.9850 | 10.3825 | 12.0560 | epoch_cap |' in table
    assert summary in t
assert (supp/'status/completion_audit.json').exists()
print('COMPLETION_AUDIT_PASS_ALL_14_PUBLIC_LOCK_INDEX_DOCS_RUNTIME_OFF')
