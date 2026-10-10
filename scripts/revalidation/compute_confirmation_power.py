"""Statistical power and sample size analysis for CLRR Fresh-Seed Confirmation.
Pure numpy implementation (no scipy dependency).
Calculates exact Monte Carlo empirical power for paired t-test and exact sign test across N = 6, 8, 10, 12 seeds.
"""
import math
import json
from pathlib import Path
import numpy as np

# Observed parameters from Corrected Controls (seeds 42, 43, 44, n=3):
# Full - LSR:
# Seed 42: +1.2259
# Seed 43: +0.3587
# Seed 44: +0.6273
# Mean: +0.7373 BLEU
# Sample SD (ddof=1): 0.4435
deltas_lsr = np.array([1.225855, 0.358744, 0.627250])
mean_lsr = float(np.mean(deltas_lsr))
sd_lsr = float(np.std(deltas_lsr, ddof=1))
d_lsr = mean_lsr / sd_lsr

# Full - FT (Baseline):
# Seed 42: +0.5932
# Seed 43: +2.0402
# Seed 44: -0.4084
# Mean: +0.7417 BLEU
# Sample SD (ddof=1): 1.2335
deltas_ft = np.array([0.593214, 2.040163, -0.408392])
mean_ft = float(np.mean(deltas_ft))
sd_ft = float(np.std(deltas_ft, ddof=1))
d_ft = mean_ft / sd_ft

def monte_carlo_paired_t_power(n, true_mean, true_sd, alpha=0.05, n_sims=100000, seed=42):
    rng = np.random.default_rng(seed)
    # Generate samples under H1: N(true_mean, true_sd^2)
    samples = rng.normal(loc=true_mean, scale=true_sd, size=(n_sims, n))
    sample_means = samples.mean(axis=1)
    sample_sds = samples.std(axis=1, ddof=1)
    t_stats = sample_means / (sample_sds / np.sqrt(n))
    
    # Critical value under H0 for df = n - 1
    # Generate H0 samples to get empirical critical t value
    h0_samples = rng.normal(loc=0.0, scale=true_sd, size=(n_sims, n))
    h0_means = h0_samples.mean(axis=1)
    h0_sds = h0_samples.std(axis=1, ddof=1)
    h0_t = np.abs(h0_means / (h0_sds / np.sqrt(n)))
    crit_val = np.quantile(h0_t, 1.0 - alpha)
    
    # Power = proportion of |t_stats| >= crit_val
    power_two_sided = np.mean(np.abs(t_stats) >= crit_val)
    # Also proportion of t_stats >= crit_val_onesided
    crit_val_onesided = np.quantile(h0_means / (h0_sds / np.sqrt(n)), 1.0 - alpha)
    power_one_sided = np.mean(t_stats >= crit_val_onesided)
    
    return float(power_two_sided), float(power_one_sided), float(crit_val)

def exact_permutation_sign_flip_p(diffs):
    """Exact magnitude-based permutation sign-flip test.
    Under H0, differences are symmetric around 0, so each diff has random sign +/-1.
    Evaluates all 2^N sign assignments and computes two-sided p-value.
    """
    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)
    obs_stat = abs(np.sum(diffs))
    total_assignments = 2 ** n
    count_as_or_more_extreme = 0
    import itertools
    for signs in itertools.product([-1, 1], repeat=n):
        s = np.array(signs)
        if abs(np.sum(s * diffs)) >= obs_stat - 1e-12:
            count_as_or_more_extreme += 1
    return count_as_or_more_extreme / total_assignments

def binomial_sign_test_thresholds(n):
    total_perms = 2 ** n
    min_two_sided_p = 2.0 / total_perms
    k_req = None
    cum_prob_k = 1.0
    for k in range(n, n // 2, -1):
        cum_prob = 2 * sum(math.comb(n, i) for i in range(k, n + 1)) / total_perms
        if cum_prob <= 0.05:
            k_req = k
            cum_prob_k = cum_prob
        else:
            break
    # Probability of 6/8 wins (investment criterion)
    p_at_6_of_8 = 2 * sum(math.comb(8, i) for i in range(6, 9)) / (2**8) if n == 8 else None
    return {
        'total_permutations': total_perms,
        'min_possible_two_sided_p': min_two_sided_p,
        'required_positive_seeds_for_p_lt_005': k_req,
        'exact_p_at_required_k': cum_prob_k,
        'p_at_6_of_8_investment_criterion': p_at_6_of_8
    }

print("=== STATISTICAL POWER ANALYSIS FOR CONFIRMATION COHORT ===")
print("NOTE: Power estimates assume normal differences and stable variance.")
print(f"Full - LSR (Primary): Mean Delta = {mean_lsr:.4f} BLEU, SD = {sd_lsr:.4f}, Cohen's d = {d_lsr:.4f}")
print(f"Full - FT (Secondary): Mean Delta = {mean_ft:.4f} BLEU, SD = {sd_ft:.4f}, Cohen's d = {d_ft:.4f}")
print(f"Exact 3-seed Sign-Flip p-value (Full-LSR): {exact_permutation_sign_flip_p(deltas_lsr):.4f}")
print(f"Exact 3-seed Sign-Flip p-value (Full-FT): {exact_permutation_sign_flip_p(deltas_ft):.4f}")
print()

table = []
for n in [6, 8, 10, 12]:
    # Full - LSR with observed parameters
    p_lsr_2s, p_lsr_1s, crit = monte_carlo_paired_t_power(n, mean_lsr, sd_lsr)
    # Full - LSR with conservative MDE = +0.30 BLEU (keeping observed SD = 0.4435)
    p_lsr_mde_2s, p_lsr_mde_1s, _ = monte_carlo_paired_t_power(n, 0.30, sd_lsr)
    
    # Full - FT with observed parameters
    p_ft_2s, p_ft_1s, _ = monte_carlo_paired_t_power(n, mean_ft, sd_ft)
    # Full - FT with conservative MDE = +0.50 BLEU (keeping observed SD = 1.2335)
    p_ft_mde_2s, p_ft_mde_1s, _ = monte_carlo_paired_t_power(n, 0.50, sd_ft)
    
    sf = binomial_sign_test_thresholds(n)
    
    table.append({
        'n_seeds': n,
        't_power_full_lsr_observed_two_sided': p_lsr_2s,
        't_power_full_lsr_observed_one_sided': p_lsr_1s,
        't_power_full_lsr_mde03_two_sided': p_lsr_mde_2s,
        't_power_full_ft_observed_two_sided': p_ft_2s,
        't_power_full_ft_mde05_two_sided': p_ft_mde_2s,
        'binomial_sign_test_min_p': sf['min_possible_two_sided_p'],
        'binomial_sign_test_req_k': sf['required_positive_seeds_for_p_lt_005'],
        'binomial_sign_test_exact_p': sf['exact_p_at_required_k'],
        'binomial_6_of_8_investment_p': sf.get('p_at_6_of_8_investment_criterion')
    })
    
    print(f"--- N = {n} seeds ---")
    print(f"  Full - LSR (Primary):")
    print(f"    Observed effect (Delta=+0.74, d={d_lsr:.2f}): Power (two-sided) = {p_lsr_2s*100:.1f}%, Power (one-sided) = {p_lsr_1s*100:.1f}%")
    print(f"    Conservative MDE (Delta=+0.30, d=0.68): Power (two-sided) = {p_lsr_mde_2s*100:.1f}%")
    print(f"  Full - FT (Secondary):")
    print(f"    Observed effect (Delta=+0.74, d={d_ft:.2f}): Power (two-sided) = {p_ft_2s*100:.1f}%")
    print(f"    Conservative MDE (Delta=+0.50, d=0.41): Power (two-sided) = {p_ft_mde_2s*100:.1f}%")
    print(f"  Binomial Sign Test (number of wins):")
    print(f"    Min possible two-sided p: {sf['min_possible_two_sided_p']:.5f}")
    if sf['required_positive_seeds_for_p_lt_005'] is not None:
        print(f"    Required positive seeds for p < 0.05: {sf['required_positive_seeds_for_p_lt_005']}/{n} (p = {sf['exact_p_at_required_k']:.4f})")
    else:
        print(f"    Required positive seeds for p < 0.05: IMPOSSIBLE (n too small, min p = {sf['min_possible_two_sided_p']:.4f} > 0.05)")
    if n == 8:
        print(f"    Note: 6/8 wins (investment criterion) has binomial p = {sf['p_at_6_of_8_investment_criterion']:.4f} (NOT p < 0.05)")
    print()

out_path = Path(__file__).resolve().parents[2] / 'outputs_rebuttal/amis_corrected_four_way_20261005'
out_path.mkdir(parents=True, exist_ok=True)
(out_path / 'power_analysis.json').write_text(json.dumps({
    'observed_parameters': {
        'full_minus_lsr': {'mean': mean_lsr, 'sd': sd_lsr, 'cohen_d': d_lsr},
        'full_minus_ft': {'mean': mean_ft, 'sd': sd_ft, 'cohen_d': d_ft}
    },
    'power_table': table,
    'methodological_notes': {
        'binomial_sign_test_vs_sign_flip': 'Binomial sign test counts wins/losses only. Permutation sign-flip test uses exact difference magnitudes across 2^N sign flips.',
        'investment_criterion_vs_significance': '6/8 wins is an internal project investment rule to avoid wasting effort, NOT a p < 0.05 statistical significance claim.',
        'power_caveat': 'Power of 97.9% for N=8 is conditional on true effect remaining at +0.74 BLEU with SD 0.44. Under MDE=+0.30 BLEU, power is 37.7%.'
    }
}, indent=2))
print("Saved power analysis to outputs_rebuttal/amis_corrected_four_way_20261005/power_analysis.json")

