"""Refresh an explicitly validation-only progress table from replay receipts."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MARKER = '<!-- standardized-es10-validation-progress -->'
END = '<!-- end-standardized-es10-validation-progress -->'


def main():
    rows = []
    for cohort in ['amis_standardized_controls', 'amis_standardized_strong_mbart']:
        folder = ROOT / 'outputs_rebuttal' / cohort
        report = folder / 'validation_verification.json'
        if not report.exists():
            continue
        plans = {s['id']: s for s in json.loads((folder / 'PLAN.json').read_text())['runs']}
        for ident, result in json.loads(report.read_text())['runs'].items():
            spec = plans[ident]
            score = result['scores']
            rows.append(f"| {ident} | {spec['backbone']} | {spec['method']} | {spec['seed']} | {result['completed_epoch']:g} | {result['selected_epoch']:g} | {score['bleu']:.4f} | {score['chrf++']:.4f} | {score['chrf0']:.4f} | {result['stop_reason']} |")
    section = '\n\n' + MARKER + '''
## Tiến độ đối sánh chuẩn hóa ES10 — validation đã kiểm chứng

**Bảng này chỉ là validation, chưa phải kết quả test và không dùng để kết luận hơn/kém phương pháp.** Chọn best theo raw-reference validation chrF++; mọi dòng đã replay độc lập từ public predictions và xác minh hash checkpoint mà không tải weights về PC. Cohort được duyệt gồm14 runs; số cũ giữ nguyên.

| ID | Backbone | Nhánh | Seed | Epoch cuối | Epoch best | Val BLEU | Val chrF++ | Val chrF0 | Dừng vì |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
''' + '\n'.join(sorted(rows)) + '\n' + END + '\n'
    for relative in ['docs/experiments/EXPERIMENTS_UPDATED.md',
                     'docs/operations/NEXT_EXPERIMENTS_HANDOFF_20261005.md']:
        path = ROOT / relative
        original = path.read_text(encoding='utf-8')
        if MARKER in original:
            start = original.index(MARKER)
            end = original.index(END, start) + len(END) if END in original[start:] else len(original)
            updated = original[:start].rstrip() + section + original[end:]
        else:
            updated = original.rstrip() + section
        path.write_text(updated, encoding='utf-8')
    print('DOCUMENTED_VALIDATION_REPLAY_ROWS', len(rows))


if __name__ == '__main__':
    main()
