from pathlib import Path

root = Path(__file__).resolve().parents[2]
path = root / 'docs/operations/NEXT_EXPERIMENTS_HANDOFF_20261005.md'
text = path.read_text(encoding='utf-8')
marker = '### Phân bổ bổ sung A06 vì unit Colab3'
assert marker not in text
section = '''### Phân bổ bổ sung A06 vì unit Colab3

Tác giả đã đồng ý chuyển B sang Colab1/2 nếu thiếu unit. Lúc20:07 Colab3 còn17.40CU; tốc độ A khoảng9.4giây/step cộng validation làm dự báo A+B vượt ngân sách. Đã giữ dispatcher Colab3 bằng SIGSTOP **chỉ process cha queue**, xác minh child A05 train PID15010 tiếp tục và chưa có A06 training. Không dừng A05, không sửa frozen bundle. Guard tự đợi A05 train/export/upload/cleanup xong; nếu VAL_COMPLETE verified thì kết thúc dispatcher và ghi `VALIDATION_ASSIGNED_QUEUE_COMPLETE`, completed A05, migrated A06. Nếu A05 lỗi, giữ weights và ghi failure, không mở B trùng trên máy này.

A06 đã được đặt queue Colab1 sau B01–B05, qua GPU preflight trên A100 đích rồi train. Script: `scripts/revalidation/schedule_strong_mbart_b_account1.py`; local assignment receipt: `outputs_rebuttal/amis_standardized_strong_mbart/A06_execution_assignment.json`; remote `/content/strong_mbart_b_account1.pid`, `.log`, `queue_account1.json`. Frozen PLAN vẫn ghi account3 ban đầu; actual execution account1 được ghi riêng, các hyperparameters giữ nguyên. Monitor: `monitor_standardized_controls.py --account 1 --supplement`; account3 dùng `--account 3 --supplement`. Colab3 tắt sau khi A05 backup và replay độc lập xong; không chờ A06 trên Colab3 nữa. Tổng14runs không đổi.

'''
position = text.index('## Lưu checkpoint hiện hành')
path.write_text(text[:position] + section + text[position:], encoding='utf-8')
print('MIGRATION_HANDOFF_UPDATED')
