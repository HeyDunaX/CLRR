"""Audit and Venn Breakdown of Morphological Voice Affix Overlap in Amis Test Set.

Analyzes the co-occurrence of verbal voice prefixes:
- mi- (Actor Voice)
- ma- (Patient / Stative Voice)
- pa- (Causative Prefix)
- Root / Simple (Complement control set: No mi/ma/pa)

Outputs exact single, pairwise, and three-way overlap statistics to resolve
reviewer questions regarding test subset construction.
"""

from __future__ import annotations

import re
from pathlib import Path
import pandas as pd


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    test_csv = repo_root / "data" / "processed" / "test.csv"
    output_dir = repo_root / "outputs_rebuttal"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not test_csv.is_file():
        raise FileNotFoundError(f"Missing test dataset at {test_csv}")

    df_test = pd.read_csv(test_csv)
    sources = [str(s).strip() for s in df_test["source"]]
    total_samples = len(sources)

    mi_pattern = re.compile(r"\b(mi|Mi)[a-z']+", re.IGNORECASE)
    ma_pattern = re.compile(r"\b(ma|Ma)[a-z']+", re.IGNORECASE)
    pa_pattern = re.compile(r"\b(pa|Pa)[a-z']+", re.IGNORECASE)

    # Boolean flags for each sentence
    has_mi = [bool(mi_pattern.search(s)) for s in sources]
    has_ma = [bool(ma_pattern.search(s)) for s in sources]
    has_pa = [bool(pa_pattern.search(s)) for s in sources]

    set_mi = set(i for i, v in enumerate(has_mi) if v)
    set_ma = set(i for i, v in enumerate(has_ma) if v)
    set_pa = set(i for i, v in enumerate(has_pa) if v)
    set_any = set_mi | set_ma | set_pa
    set_root = set(range(total_samples)) - set_any

    # Exact Venn disjoint components
    only_mi = set_mi - set_ma - set_pa
    only_ma = set_ma - set_mi - set_pa
    only_pa = set_pa - set_mi - set_ma

    mi_and_ma_only = (set_mi & set_ma) - set_pa
    mi_and_pa_only = (set_mi & set_pa) - set_ma
    ma_and_pa_only = (set_ma & set_pa) - set_mi

    all_three = set_mi & set_ma & set_pa

    print(f"\n{'='*70}")
    print(f"AMIS MORPHOLOGICAL SUBSET AUDIT (Test Set: N = {total_samples})")
    print(f"{'='*70}")
    print(f"Total Test Sentences:                   {total_samples}")
    print(f"Unique sentences with >=1 voice prefix: {len(set_any)} ({len(set_any)/total_samples*100:.1f}%)")
    print(f"Root/Simple sentences (Strict Control): {len(set_root)} ({len(set_root)/total_samples*100:.1f}%)")
    print(f"Check Partition Sum (Affixed + Root):   {len(set_any) + len(set_root)} (Must be {total_samples})")
    print(f"{'-'*70}")
    print(f"CATEGORY MARGINAL COUNTS (Non-mutually exclusive):")
    print(f"  * mi- (Actor Voice):                  {len(set_mi):>3} sentences")
    print(f"  * ma- (Patient/Stative):              {len(set_ma):>3} sentences")
    print(f"  * pa- (Causative):                    {len(set_pa):>3} sentences")
    print(f"  * Sum of Marginals:                   {len(set_mi) + len(set_ma) + len(set_pa):>3} (Implies {len(set_mi) + len(set_ma) + len(set_pa) - len(set_any)} co-occurrences)")
    print(f"{'-'*70}")
    print(f"DISJOINT VENN PARTITIONS:")
    print(f"  * Exclusively mi- only:               {len(only_mi):>3}")
    print(f"  * Exclusively ma- only:               {len(only_ma):>3}")
    print(f"  * Exclusively pa- only:               {len(only_pa):>3}")
    print(f"  * Co-occurrence (mi + ma only):       {len(mi_and_ma_only):>3}")
    print(f"  * Co-occurrence (mi + pa only):       {len(mi_and_pa_only):>3}")
    print(f"  * Co-occurrence (ma + pa only):       {len(ma_and_pa_only):>3}")
    print(f"  * Co-occurrence (all 3: mi + ma + pa):{len(all_three):>3}")
    print(f"  * Root / Simple (0 voice prefixes):   {len(set_root):>3}")
    total_partition = (
        len(only_mi)
        + len(only_ma)
        + len(only_pa)
        + len(mi_and_ma_only)
        + len(mi_and_pa_only)
        + len(ma_and_pa_only)
        + len(all_three)
        + len(set_root)
    )
    print(f"  -> Total Disjoint Sum:                {total_partition:>3} (Matches N = {total_samples}: {total_partition == total_samples})")
    print(f"{'='*70}\n")

    # Export records
    matrix_data = [
        {"Category": "Total Test Set", "Count": total_samples, "Percent": 100.0, "Type": "Full"},
        {"Category": "Affixed Sentences (>=1 prefix)", "Count": len(set_any), "Percent": len(set_any)/total_samples*100, "Type": "Superset"},
        {"Category": "Root / Simple (No voice prefix)", "Count": len(set_root), "Percent": len(set_root)/total_samples*100, "Type": "Disjoint Control"},
        {"Category": "Marginal: mi- (Actor Voice)", "Count": len(set_mi), "Percent": len(set_mi)/total_samples*100, "Type": "Marginal"},
        {"Category": "Marginal: ma- (Patient/Stative)", "Count": len(set_ma), "Percent": len(set_ma)/total_samples*100, "Type": "Marginal"},
        {"Category": "Marginal: pa- (Causative)", "Count": len(set_pa), "Percent": len(set_pa)/total_samples*100, "Type": "Marginal"},
        {"Category": "Venn: Exclusive mi- only", "Count": len(only_mi), "Percent": len(only_mi)/total_samples*100, "Type": "Venn Cell"},
        {"Category": "Venn: Exclusive ma- only", "Count": len(only_ma), "Percent": len(only_ma)/total_samples*100, "Type": "Venn Cell"},
        {"Category": "Venn: Exclusive pa- only", "Count": len(only_pa), "Percent": len(only_pa)/total_samples*100, "Type": "Venn Cell"},
        {"Category": "Venn: mi- & ma- co-occurrence", "Count": len(mi_and_ma_only), "Percent": len(mi_and_ma_only)/total_samples*100, "Type": "Venn Cell"},
        {"Category": "Venn: mi- & pa- co-occurrence", "Count": len(mi_and_pa_only), "Percent": len(mi_and_pa_only)/total_samples*100, "Type": "Venn Cell"},
        {"Category": "Venn: ma- & pa- co-occurrence", "Count": len(ma_and_pa_only), "Percent": len(ma_and_pa_only)/total_samples*100, "Type": "Venn Cell"},
        {"Category": "Venn: mi- & ma- & pa- all three", "Count": len(all_three), "Percent": len(all_three)/total_samples*100, "Type": "Venn Cell"},
    ]
    df_matrix = pd.DataFrame(matrix_data)
    matrix_csv_path = output_dir / "morphological_overlap_matrix.csv"
    df_matrix.to_csv(matrix_csv_path, index=False)
    print(f"[Done] Exported overlap matrix to {matrix_csv_path}")


if __name__ == "__main__":
    main()
