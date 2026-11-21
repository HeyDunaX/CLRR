"""Train-only lexical candidate discovery. Similar spelling is not semantics."""
from collections import defaultdict
from itertools import combinations
import unicodedata

TOKENIZATION_POLICY = 'nfkc-casefold-unicode-alnum-marks-internal-apostrophe-v2'


def words(text):
    """Lexical tokens retaining combining marks; not linguistic segmentation."""
    text = unicodedata.normalize('NFKC', text).casefold()
    tokens, current = [], []
    for index, char in enumerate(text):
        if char.isalnum() or (current and unicodedata.category(char).startswith('M')):
            current.append(char)
        elif (current and char in "'’" and index + 1 < len(text)
              and text[index + 1].isalnum()):
            current.append(char)
        elif current:
            tokens.append(''.join(current))
            current = []
    if current:
        tokens.append(''.join(current))
    return tuple(tokens)


def discover_pairs(rows, minimum_words=3, maximum_words=50):
    """Find single-word substitutions in otherwise identical source sequences.

    Neither CSV contents nor target strings are altered. Target differences
    are only eligibility heuristics; no semantic/morphological labels are inferred.
    """
    if (type(minimum_words) is not int or type(maximum_words) is not int
            or minimum_words < 1 or maximum_words < minimum_words):
        raise ValueError('Expected positive ordered integer word bounds')
    groups = defaultdict(list)
    tokens = [words(row['source']) for row in rows]
    for index, sequence in enumerate(tokens):
        if minimum_words <= len(sequence) <= maximum_words:
            for position in range(len(sequence)):
                groups[(position, sequence[:position], sequence[position + 1:])].append(index)
    pairs = {}
    for (position, _, _), indices in groups.items():
        for left, right in combinations(indices, 2):
            if tokens[left][position] == tokens[right][position]:
                continue
            if not rows[left]['target'].strip() or not rows[right]['target'].strip():
                continue
            if rows[left]['target'].strip() == rows[right]['target'].strip():
                continue
            pairs[(left, right)] = dict(left=left, right=right,
                source_position=position, source_left=tokens[left][position],
                source_right=tokens[right][position], semantic_verified=False)
    return [pairs[key] for key in sorted(pairs)]
