"""Read-only split overlap audit; no training, model loading, or data filtering."""
import csv
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
import unicodedata

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data_processed/amis_mandarin'
OUT = ROOT / 'outputs_rebuttal/amis_split_overlap_20261116'
SPLITS = ('train', 'validation', 'test')


def normalize(text):
    text = unicodedata.normalize('NFKC', text).casefold()
    text = ''.join(c for c in text if not unicodedata.category(c).startswith('P'))
    return ' '.join(text.split())


def similarity(a, b):
    if not a or not b:
        return float(a == b)
    # SequenceMatcher can depend on argument order. Use the larger direction.
    return max(SequenceMatcher(None, a, b, autojunk=False).ratio(),
               SequenceMatcher(None, b, a, autojunk=False).ratio())


def candidate_pairs(left, right, field, threshold):
    """Exhaustive safe screening by character-multiset overlap upper bound.

    Every matching subsequence uses at most min(count_a[c], count_b[c])
    occurrences of each character. Therefore no pair reaching the specified
    SequenceMatcher ratio can be removed by this screen (apart from 1e-12).
    """
    alphabet = sorted(set(''.join(r[field] for r in left + right)))
    index = {c: i for i, c in enumerate(alphabet)}

    def counts(rows):
        matrix = np.zeros((len(rows), len(alphabet)), dtype=np.int32)
        for i, row in enumerate(rows):
            for c, n in Counter(row[field]).items():
                matrix[i, index[c]] = n
        return matrix

    a, b = counts(left), counts(right)
    lengths_b = b.sum(axis=1)
    for start in range(0, len(left), 32):
        block = a[start:start + 32]
        total = block.sum(axis=1)[:, None] + lengths_b[None, :]
        distance = np.abs(block[:, None, :] - b[None, :, :]).sum(axis=2)
        upper = np.divide(total - distance, total,
                          out=np.ones(total.shape, dtype=float), where=total != 0)
        for i, j in np.argwhere(upper + 1e-12 >= threshold):
            yield start + int(i), int(j)


def write_csv(name, records, fields):
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def write_brief(result, near):
    lines = ['# Kiểm tra overlap dữ liệu Amis–Mandarin', '',
             'Ngày thực hiện: 16/11/2026. Mục đích: cung cấp thông tin trả lời reviewer #3.', '',
             '## Kết luận để gửi người viết bài', '',
             '- Dữ liệu hiện tại có 4.600 / 576 / 575 cặp train / validation / test.',
             '- Không phát hiện câu nguồn hoặc cả cặp dịch trùng nguyên văn giữa bất kỳ hai split nào.',
             '- Khi chuẩn hóa NFKC, chữ thường, bỏ dấu câu và gộp khoảng trắng: có 2 cặp train–test trùng cả hai phía.',
             '- Dấu nháy có thể có giá trị ngôn ngữ học trong Amis. Hai cặp này chỉ trùng dưới phép chuẩn hóa bỏ dấu câu; không tự động gọi là câu tương đương hoặc leakage đã xác nhận.',
             '- Có 13 cặp train–test đạt source similarity ≥ 0,85 và target similarity ≥ 0,90, ứng với 13/575 câu test (2,26%). Bao gồm 2 cặp trùng sau chuẩn hóa nêu trên, không cộng thêm 2 vào 13.',
             '- Trong đó 5 cặp đạt ≥ 0,95 ở cả hai phía (5/575 = 0,87%), bao gồm 2 cặp trùng sau chuẩn hóa.',
             '- Các số là phép sàng lọc tương đồng chuỗi, chưa phải phân loại ngôn ngữ học hoặc bằng chứng tác động đến điểm model.',
             '- Giữ nguyên dữ liệu và split gốc; không train, chấm lại metric hay lọc test trong kiểm tra này.', '',
             '## Kết quả theo cặp split', '',
             'Mỗi số dưới đây là số cặp dòng, không phải số mục từ điển.', '',
             '| Split | Cặp dịch raw trùng | Cặp dịch trùng sau chuẩn hóa | Cặp target raw trùng | Gần trùng hai phía (0,85/0,90) | Hai phía ≥ 0,95 |',
             '|---|---:|---:|---:|---:|---:|']
    for s in result['cross_split']:
        lines.append(f"| {s['left_split']}–{s['right_split']} | {s['exact']['raw_pair']['row_pairs']} | {s['exact']['normalized_pair']['row_pairs']} | {s['exact']['raw_target']['row_pairs']} | {s['near']['bilingual_src085_tgt090']['row_pairs']} | {s['near']['bilingual_both095']['row_pairs']} |")
    lines += ['', 'Riêng train–test, 19 cặp target nguyên văn trùng ứng với 18 câu test; một target test khớp hai dòng train. Target trùng riêng lẻ không đủ chứng minh cả ví dụ dịch bị lặp.', '',
              '## Ví dụ cần xem xét thủ công', '',
              'Chỉ số là thứ tự bản ghi bắt đầu từ 1, không tính header. Không mặc định là số dòng vật lý nếu CSV có trường nhiều dòng.', '']
    for r in near:
        if r['left_split'] == 'train' and r['right_split'] == 'test' and min(r['source_similarity'], r['target_similarity']) >= .95:
            lines += [f"### Train {r['left_row']} ↔ Test {r['right_row']}", '',
                      f"- Amis train: {r['source_left']}", f"- Amis test: {r['source_right']}",
                      f"- Mandarin train: {r['target_left']}", f"- Mandarin test: {r['target_right']}",
                      f"- Similarity nguồn/đích: {r['source_similarity']:.6f} / {r['target_similarity']:.6f}.", '']
    lines += ['## Phương pháp và giới hạn', '',
              '- Kiểm tra toàn bộ 5.625.800 cặp dòng giữa các split. Trong từng split chỉ thống kê exact duplicates, không chạy near-duplicate.',
              '- Raw exact = bằng nhau trên chuỗi CSV đã đọc, không strip/casefold.',
              '- Normalized = Unicode NFKC → casefold → bỏ ký tự Unicode loại P* → gộp khoảng trắng. Không đổi dấu ^, không kiểm tra đồng nghĩa hoặc phiên âm.',
              '- Similarity = max của hai chiều difflib.SequenceMatcher.ratio, autojunk=False, trên ký tự của chuỗi đã chuẩn hóa; thang 0–1. Đây không phải Levenshtein hay semantic similarity.',
              '- Sàng lọc bằng giới hạn trên từ số lượng ký tự, sau đó tính similarity; không lấy top-k hoặc mẫu ngẫu nhiên. Ngưỡng source ≥ 0,85 hoặc target ≥ 0,90 dùng để lưu ứng viên. Ngưỡng hai phía chỉ là heuristic được chọn cho audit này, không được hiệu chỉnh bằng nhãn chuyên gia.',
              '- CSV ứng viên giữ cả câu ngắn và exact matches. Cờ short_text = nguồn ngắn hơn 20 ký tự hoặc đích ngắn hơn 6 ký tự ở ít nhất một bên; không loại các cặp này khỏi số liệu.',
              '- CSV chỉ có source,target; không có ID mục từ hoặc provenance ở cấp entry. Chưa thể xác định overlap theo mục từ điển, quan hệ nguồn gốc hoặc toàn bộ các cách diễn đạt tương đương.',
              '- Không có exact overlap không chứng minh split hoàn toàn độc lập. Near-overlap không chứng minh model đã ghi nhớ hay tạo ra mức tăng chất lượng.',
              '- Nếu báo cáo dataset/split từ nghiên cứu trước, nêu rõ đây là kiểm tra bổ sung của split được sử dụng, không phải kết luận về toàn bộ dataset gốc.', '',
              '## Đoạn tiếng Anh có thể gửi người viết (cần dẫn nguồn dataset trong bài)', '',
              '> We audited the provided 4,600/576/575 train/validation/test records for cross-split string overlap. No verbatim source sentence or bilingual pair was shared across splits. After NFKC normalization, case folding, punctuation removal, and whitespace normalization, two train–test pairs became identical on both sides. A heuristic character-similarity screen (source ≥ 0.85 and target ≥ 0.90) flagged 13 train–test pairs involving 13 test sentences (2.26%), including those two normalized matches; five pairs reached 0.95 on both sides. These are string-overlap candidates rather than confirmed semantic duplicates. In particular, apostrophe differences may be linguistically meaningful in Amis. Dictionary-entry overlap could not be established because entry identifiers were unavailable in the processed CSVs. We retained the original records and splits; this audit does not quantify the effect of overlap on model scores.', '',
              '## Artifacts và tái chạy', '',
              '- `summary.json`: số liệu đầy đủ và SHA-256 dữ liệu.',
              '- `exact_matches.csv`: từng match raw/normalized, theo source/target/pair; cùng một cặp có thể xuất hiện ở nhiều loại.',
              '- `near_candidates.csv`: toàn bộ 80 cặp ứng viên qua ít nhất một ngưỡng.',
              '- `train_test_bilingual_candidates.csv`: 13 cặp train–test qua ngưỡng cả hai phía.',
              '- Lệnh: `conda run --no-capture-output -n clrr python scripts/analysis/audit_amis_split_overlap.py`.', '',
              '## Định danh dữ liệu', '']
    for split, info in result['files'].items():
        lines.append(f"- {split}: {info['rows']} dòng; SHA-256 `{info['sha256']}`.")
    (OUT / 'WRITER_BRIEF_VI.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, manifest = {}, {}
    for split in SPLITS:
        path = DATA / f'{split}.csv'
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            assert reader.fieldnames == ['source', 'target'], reader.fieldnames
            rows[split] = list(reader)
        assert all(r['source'].strip() and r['target'].strip() for r in rows[split])
        for r in rows[split]:
            r['norm_source'], r['norm_target'] = normalize(r['source']), normalize(r['target'])
        manifest[split] = dict(path=str(path), rows=len(rows[split]), sha256=digest)

    within, exact, near, summaries = {}, [], [], []
    detail_fields = ['left_split', 'left_row', 'right_split', 'right_row',
                     'source_left', 'target_left', 'source_right', 'target_right']

    def details(ls, i, rs, j):
        a, b = rows[ls][i], rows[rs][j]
        return dict(left_split=ls, left_row=i + 1, right_split=rs, right_row=j + 1,
                    source_left=a['source'], target_left=a['target'],
                    source_right=b['source'], target_right=b['target'])

    modes = [('raw', 'source', 'target'), ('normalized', 'norm_source', 'norm_target')]
    for split in SPLITS:
        within[split] = {}
        for mode, sf, tf in modes:
            for kind, fields in [('source', (sf,)), ('target', (tf,)), ('pair', (sf, tf))]:
                counts = Counter(tuple(r[f] for f in fields) for r in rows[split])
                within[split][f'{mode}_{kind}'] = dict(
                    duplicate_groups=sum(n > 1 for n in counts.values()),
                    excess_rows=sum(n - 1 for n in counts.values()),
                    row_pairs=sum(n * (n - 1) // 2 for n in counts.values()))

    for ls, rs in itertools.combinations(SPLITS, 2):
        summary = dict(left_split=ls, right_split=rs,
                       comparisons=len(rows[ls]) * len(rows[rs]), exact={})
        for mode, sf, tf in modes:
            for kind, fields in [('source', (sf,)), ('target', (tf,)), ('pair', (sf, tf))]:
                lookup = defaultdict(list)
                for j, r in enumerate(rows[rs]):
                    lookup[tuple(r[f] for f in fields)].append(j)
                matches = [(i, j) for i, r in enumerate(rows[ls])
                           for j in lookup[tuple(r[f] for f in fields)]]
                summary['exact'][f'{mode}_{kind}'] = dict(
                    row_pairs=len(matches), left_rows=len({i for i, j in matches}),
                    right_rows=len({j for i, j in matches}))
                exact.extend(dict(**details(ls, i, rs, j), mode=mode, kind=kind)
                             for i, j in matches)

        candidates = set(candidate_pairs(rows[ls], rows[rs], 'norm_source', .85))
        candidates.update(candidate_pairs(rows[ls], rows[rs], 'norm_target', .90))
        local = []
        for i, j in sorted(candidates):
            a, b = rows[ls][i], rows[rs][j]
            src = similarity(a['norm_source'], b['norm_source'])
            tgt = similarity(a['norm_target'], b['norm_target'])
            if src + 1e-12 < .85 and tgt + 1e-12 < .90:
                continue
            record = dict(**details(ls, i, rs, j), source_similarity=src,
                          target_similarity=tgt,
                          normalized_pair_exact=a['norm_source'] == b['norm_source']
                          and a['norm_target'] == b['norm_target'],
                          short_text=min(len(a['norm_source']), len(b['norm_source'])) < 20
                          or min(len(a['norm_target']), len(b['norm_target'])) < 6)
            near.append(record)
            local.append(record)
        summary['near'] = {}
        for label, predicate in [
            ('source_ge_085', lambda r: r['source_similarity'] + 1e-12 >= .85),
            ('source_ge_090', lambda r: r['source_similarity'] + 1e-12 >= .90),
            ('source_ge_095', lambda r: r['source_similarity'] + 1e-12 >= .95),
            ('target_ge_090', lambda r: r['target_similarity'] + 1e-12 >= .90),
            ('target_ge_095', lambda r: r['target_similarity'] + 1e-12 >= .95),
            ('bilingual_src085_tgt090', lambda r: r['source_similarity'] + 1e-12 >= .85
             and r['target_similarity'] + 1e-12 >= .90),
            ('bilingual_both095', lambda r: min(r['source_similarity'], r['target_similarity']) + 1e-12 >= .95),
        ]:
            selected = [r for r in local if predicate(r)]
            summary['near'][label] = dict(
                row_pairs=len(selected), left_rows=len({r['left_row'] for r in selected}),
                right_rows=len({r['right_row'] for r in selected}),
                nonexact_row_pairs=sum(not r['normalized_pair_exact'] for r in selected),
                short_text_pairs=sum(r['short_text'] for r in selected))
        summaries.append(summary)
        print(f'{ls}/{rs}: {len(local)} candidate pairs retained', flush=True)

    result = dict(dataset='amis_mandarin', files=manifest, within_split=within,
                  cross_split=summaries, row_numbering='1-based data record; header excluded',
                  normalization='NFKC; casefold; remove Unicode P* punctuation; collapse whitespace',
                  similarity='max of both directions of difflib.SequenceMatcher ratio, autojunk=False, character-level',
                  thresholds='source >= .85 OR target >= .90; includes exact and short strings',
                  method='All cross-split pairs; safe character-multiset upper-bound screen before exact similarity',
                  limitations=['String similarity is not semantic equivalence or confirmed leakage.',
                               'No dictionary-entry IDs in CSV; entry-level overlap unverified.',
                               'Normalization may conflate linguistically distinct punctuation/case.',
                               'Short strings and recurring sentence templates inflate similarity.',
                               'No retraining, evaluation filtering, or data changes.'])
    write_csv('exact_matches.csv', exact, detail_fields + ['mode', 'kind'])
    near.sort(key=lambda r: (-min(r['source_similarity'], r['target_similarity']),
                             -r['source_similarity'], r['left_split'], r['left_row'], r['right_row']))
    write_csv('near_candidates.csv', near, detail_fields + [
        'source_similarity', 'target_similarity', 'normalized_pair_exact', 'short_text'])
    write_csv('train_test_bilingual_candidates.csv', [r for r in near if
        r['left_split'] == 'train' and r['right_split'] == 'test'
        and r['source_similarity'] + 1e-12 >= .85 and r['target_similarity'] + 1e-12 >= .90],
        detail_fields + ['source_similarity', 'target_similarity', 'normalized_pair_exact', 'short_text'])
    (OUT / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    write_brief(result, near)
    for split in SPLITS:
        assert hashlib.sha256((DATA / f'{split}.csv').read_bytes()).hexdigest() == manifest[split]['sha256']
    print(json.dumps(summaries, ensure_ascii=True), flush=True)


if __name__ == '__main__':
    main()
