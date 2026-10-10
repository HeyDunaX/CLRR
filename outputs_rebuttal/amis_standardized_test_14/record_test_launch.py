from pathlib import Path
import json
root=Path(r'D:\Code\CLRR')
plan=json.loads((root/'outputs_rebuttal/amis_standardized_test_14/PLAN.json').read_text())
assert set(plan['partitions'])=={'1','2'} and sum(map(len,plan['partitions'].values()))==14
p=root/'docs/operations/NEXT_EXPERIMENTS_HANDOFF_20261005.md'
t=p.read_text(encoding='utf-8')
section='''<!-- standardized-es10-test-current -->
## Test ES10 đang chạy — 09/10/2026

Tác giả đã cho chạy test đủ14 checkpoint được khóa và yêu cầu chỉ dùng Colab1/2 A100 Standard40GB. Không train lại, không dùng Colab3. Beam4, FP32 generation, TF32-off, raw CSV test575 câu; checkpoint được lấy ở đúng pinned HF revision và SHA256 trong joint selection lock.

| Máy | Thứ tự test | Số model |
|---|---|---:|
| Colab1 | A01, A02, A03, A04, A05, A06, D01 | 7 |
| Colab2 | B01, B02, B03, B04, B05, C01, D02 | 7 |

Launcher: `scripts/revalidation/launch_standardized_14_test.py --account all`; monitor: `scripts/revalidation/monitor_standardized_14_test.py --account 1` hoặc `--account 2`. Plan, bundle và launch receipts: `outputs_rebuttal/amis_standardized_test_14/`. Bundle giữ nguyên frozen worker/data; thêm riêng test worker và joint selection lock. Trước launch, public collector xác nhận chưa có test receipt cho14 model này.

Mỗi model upload test predictions/metrics và COMPLETE receipt vào cùng folder checkpoint public. Collector test replay metrics, kiểm tra selection bằng joint lock và checksum. Chỉ ghi số đã verify vào bảng ES10 mới trong `EXPERIMENTS_UPDATED.md`; bảng validation cũ giữ nguyên. Tắt mỗi runtime khi queue và backup hoàn tất; theo dõi chat mỗi10phút.
<!-- end-standardized-es10-test-current -->

'''
if '<!-- standardized-es10-test-current -->' not in t:
    i=t.index('\n')+1
    t=t[:i]+'\n'+section+t[i:]
p.write_text(t,encoding='utf-8')
print('TEST14_HANDOFF_RECORDED')
