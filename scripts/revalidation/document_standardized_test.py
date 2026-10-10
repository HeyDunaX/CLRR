"""Document independently replayed locked-checkpoint test scores."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
START = '<!-- standardized-es10-test-results -->'
END = '<!-- end-standardized-es10-test-results -->'


def main():
    rows = []
    for cohort, count in [('amis_standardized_controls', 12), ('amis_standardized_strong_mbart', 2)]:
        folder = ROOT / 'outputs_rebuttal' / cohort
        report = json.loads((folder / 'test_verification.json').read_text(encoding='utf-8'))
        assert report['status'] == f'ALL_{count}_TEST_VERIFIED' and not report['pending']
        specs = {r['id']: r for r in json.loads((folder / 'PLAN.json').read_text())['runs']}
        lock = json.loads((folder / 'validation_verification.json').read_text())['runs']
        for ident, result in report['runs'].items():
            spec, selected, score = specs[ident], lock[ident], result['scores']
            link = f"https://huggingface.co/FiveC/amis-rewire-checkpoints/tree/main/{spec['remote_prefix']}"
            rows.append((ident, f"| {ident} | {spec['backbone']} | {spec['method']} | {spec['seed']} | {selected['selected_epoch']:g} | {score['bleu']:.4f} | {score['chrf++']:.4f} | {score['chrf0']:.4f} | [Artifacts]({link}) |"))
    assert len(rows) == 14 and len({ident for ident, _ in rows}) == 14
    section = '\n\n' + START + '''
## Đối sánh ES10 — kết quả test đã xác minh (14/14)

Test Amis→Mandarin gồm 575 câu, raw CSV references, BLEU tokenize=zh, chrF++ word_order=2 và chrF0 word_order=0; beam 4, FP32 generation, TF32 off. Checkpoint được khóa trước khi test, chọn bằng validation chrF++ greedy (beam 1). Training tối đa 20 epochs, early-stopping patience 10. Chỉ chạy test trên Colab 1/2 A100 Standard; không train lại trong đợt đánh giá này.

Mọi dòng đã replay metrics độc lập từ public predictions, kiểm tra thứ tự source/target, hash artifacts và selection lock 14 checkpoints. Số cũ được giữ để truy vết; bảng này là kết quả đối sánh ES10 mới. Mỗi dòng là một seed, không phải trung bình đa seed; baseline mBART seed 46 khác seed 42 của đối sánh, nên không dùng các chênh lệch này như paired-seed significance.

| ID | Backbone | Nhánh | Seed | Epoch best | Test BLEU | Test chrF++ | Test chrF0 | Nguồn |
|---|---|---|---:|---:|---:|---:|---:|---|
''' + '\n'.join(row for _, row in sorted(rows)) + '\n\nProofs: `outputs_rebuttal/amis_standardized_controls/test_verification.json`, `outputs_rebuttal/amis_standardized_strong_mbart/test_verification.json`; public copies under `registry/plans/` in the checkpoint repository.\n' + END + '\n'
    for relative in ['docs/experiments/EXPERIMENTS_UPDATED.md', 'docs/operations/NEXT_EXPERIMENTS_HANDOFF_20261005.md']:
        path = ROOT / relative
        original = path.read_text(encoding='utf-8')
        if START in original:
            begin = original.index(START)
            finish = original.index(END, begin) + len(END)
            updated = original[:begin].rstrip() + section + original[finish:]
        else:
            updated = original.rstrip() + section
        updated = updated.replace('chọn bằng validation chrF++ greedy (beam 1)', 'chọn bằng validation chrF++ với validation_beams đã khóa trong từng PLAN (mBART beam 1; NLLB beam 4)')
        if relative.startswith('docs/operations/'):
            marker = '<!-- standardized-es10-test-current -->'
            ending = '<!-- end-standardized-es10-test-current -->'
            if marker in updated:
                begin = updated.index(marker)
                finish = updated.index(ending, begin) + len(ending)
                current = marker + '\n## Test ES10 hoàn tất — 09/10/2026\n\n**14/14 TEST_VERIFIED.** Colab 1 và 2 hoàn tất queue, backup và replay độc lập rồi đã tắt. Colab 3 không dùng. Cả ba tài khoản được kiểm tra: active assignments 0, usage rate 0.00/hr. Bảng test đầy đủ ở cuối file; checkpoint selection giữ đúng joint lock trước test. Predictions, metrics và COMPLETE receipts public trong từng folder checkpoint ES10. Index public đánh dấu cả 14 checkpoint test_verified=true. Chỉ tải artifacts nhỏ về PC, không tải weights.\n\nEvaluator ban đầu lỗi vì kế thừa eval_strategy=epoch mà không có eval_dataset; đã đặt eval_strategy=no cho test-only. Không đổi model, training recipe hoặc generation protocol; chưa có test receipt trước khi sửa.\n\n**Tiếp theo:** sử dụng bảng đối sánh ES10 đã xác minh để cập nhật paper và lập kế hoạch phương pháp mới; không tự mở thêm training. Một seed mỗi nhánh chưa đủ kết luận thống kê đa seed.\n' + ending
                updated = updated[:begin] + current + updated[finish:]
            updated = updated.replace('## Đối sánh ES10 đã hoàn tất — 09/10/2026', '## Lịch sử: hoàn tất validation ES10 — trước đợt test 09/10/2026', 1)
        path.write_text(updated, encoding='utf-8')
    print('DOCUMENTED_TEST_VERIFIED_ROWS', len(rows))


if __name__ == '__main__':
    main()
