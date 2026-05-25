"""
Compare two VLM score files for statistical significance.

Loads per-sample primary scores from two score JSON files and runs:
  1. Paired bootstrap test (difference of means with CI)
  2. McNemar's test (contingency on correct/incorrect at threshold)
  3. Holm-Bonferroni correction for multiple comparisons (top-N)

Usage:
    python compare_models.py score_a.json score_b.json
    python compare_models.py score_a.json score_b.json --alpha 0.01
    python compare_models.py --top-n scores_dir/  # compare top N models pairwise
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_per_sample_scores(score_path: Path) -> Dict[str, float]:
    """Load per-sample primary scores from a score JSON file.

    Returns dict mapping sample_key -> primary_score.
    """
    with open(score_path) as f:
        data = json.load(f)

    per_sample = data.get("per_sample", data.get("samples", []))
    if isinstance(per_sample, list):
        return {
            f"{s.get('image_path', s.get('sample_id', i))}_{s.get('question_index', '')}": s.get("primary_score", 0.0)
            for i, s in enumerate(per_sample)
        }
    elif isinstance(per_sample, dict):
        return {k: v.get("primary_score", 0.0) if isinstance(v, dict) else float(v)
                for k, v in per_sample.items()}
    return {}


def paired_bootstrap_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict:
    """Paired bootstrap test for difference of means.

    Returns dict with mean difference, CI, and p-value.
    """
    rng = np.random.default_rng(seed)
    n = len(scores_a)
    observed_diff = float(np.mean(scores_a) - np.mean(scores_b))

    boot_diffs = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_diffs[i] = np.mean(scores_a[idx]) - np.mean(scores_b[idx])

    ci_lower = float(np.percentile(boot_diffs, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_diffs, 100 * (1 - alpha / 2)))

    # Two-sided p-value: proportion of bootstrap samples where diff has opposite sign
    if observed_diff >= 0:
        p_value = float(np.mean(boot_diffs <= 0)) * 2
    else:
        p_value = float(np.mean(boot_diffs >= 0)) * 2
    p_value = min(p_value, 1.0)

    return {
        "test": "paired_bootstrap",
        "mean_diff": observed_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_value,
        "significant": p_value < alpha,
        "n_bootstrap": n_bootstrap,
        "alpha": alpha,
    }


def mcnemar_from_scores(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    threshold: float = 0.5,
) -> Dict:
    """McNemar's test on binarized scores (above/below threshold).

    Returns dict with test statistic, p-value, contingency table.
    """
    correct_a = scores_a >= threshold
    correct_b = scores_b >= threshold

    # Contingency: [both_correct, a_only] / [b_only, both_wrong]
    both_correct = int(np.sum(correct_a & correct_b))
    a_only = int(np.sum(correct_a & ~correct_b))
    b_only = int(np.sum(~correct_a & correct_b))
    both_wrong = int(np.sum(~correct_a & ~correct_b))

    table = [[both_correct, a_only], [b_only, both_wrong]]

    # Use exact test for small discordant counts
    discordant = a_only + b_only
    if discordant < 25:
        p_value = float(stats.binom_test(a_only, discordant, 0.5))
        test_stat = None
    else:
        # Chi-squared approximation with continuity correction
        test_stat = float((abs(a_only - b_only) - 1) ** 2 / (a_only + b_only))
        p_value = float(1 - stats.chi2.cdf(test_stat, df=1))

    return {
        "test": "mcnemar",
        "statistic": test_stat,
        "p_value": p_value,
        "contingency_table": table,
        "threshold": threshold,
        "significant": p_value < 0.05,
    }


def holm_bonferroni(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    """Apply Holm-Bonferroni correction for multiple comparisons.

    Returns list of booleans indicating which tests remain significant.
    """
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    significant = [False] * n

    for rank, (orig_idx, p) in enumerate(indexed):
        adjusted_alpha = alpha / (n - rank)
        if p <= adjusted_alpha:
            significant[orig_idx] = True
        else:
            break  # All subsequent are non-significant

    return significant


def compare_two(
    path_a: Path,
    path_b: Path,
    alpha: float = 0.05,
) -> Dict:
    """Compare two score files and return statistical results."""
    samples_a = load_per_sample_scores(path_a)
    samples_b = load_per_sample_scores(path_b)

    # Align on common samples
    common_keys = sorted(set(samples_a.keys()) & set(samples_b.keys()))
    if not common_keys:
        return {"error": "No common samples found between score files"}

    scores_a = np.array([samples_a[k] for k in common_keys])
    scores_b = np.array([samples_b[k] for k in common_keys])

    bootstrap = paired_bootstrap_test(scores_a, scores_b, alpha=alpha)
    mcnemar = mcnemar_from_scores(scores_a, scores_b)

    return {
        "model_a": path_a.stem,
        "model_b": path_b.stem,
        "n_common_samples": len(common_keys),
        "mean_a": float(np.mean(scores_a)),
        "mean_b": float(np.mean(scores_b)),
        "paired_bootstrap": bootstrap,
        "mcnemar": mcnemar,
    }


def compare_top_n(
    score_dir: Path,
    top_n: int = 5,
    alpha: float = 0.05,
) -> Dict:
    """Compare top-N models pairwise with Holm-Bonferroni correction."""
    score_files = sorted(score_dir.glob("*.json"))
    if len(score_files) < 2:
        return {"error": f"Need at least 2 score files, found {len(score_files)}"}

    # Rank by overall mean
    ranked = []
    for sf in score_files:
        with open(sf) as f:
            data = json.load(f)
        mean = data.get("overall", {}).get("primary_score_mean", 0)
        ranked.append((mean, sf))
    ranked.sort(reverse=True)
    top_files = [sf for _, sf in ranked[:top_n]]

    # Pairwise comparisons
    comparisons = []
    p_values = []
    for i in range(len(top_files)):
        for j in range(i + 1, len(top_files)):
            result = compare_two(top_files[i], top_files[j], alpha=alpha)
            comparisons.append(result)
            p_values.append(result.get("paired_bootstrap", {}).get("p_value", 1.0))

    # Holm-Bonferroni correction
    if p_values:
        corrected = holm_bonferroni(p_values, alpha=alpha)
        for comp, sig in zip(comparisons, corrected):
            comp["significant_after_correction"] = sig

    return {
        "top_n": top_n,
        "n_comparisons": len(comparisons),
        "alpha": alpha,
        "correction": "holm_bonferroni",
        "comparisons": comparisons,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare VLM score files for statistical significance")
    parser.add_argument("paths", nargs="+", help="Two score JSON files, or a directory for top-N comparison")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level (default: 0.05)")
    parser.add_argument("--top-n", type=int, default=5, help="Number of top models for pairwise comparison")
    parser.add_argument("--output", type=str, help="Output JSON path (default: stdout)")
    args = parser.parse_args()

    if len(args.paths) == 1 and Path(args.paths[0]).is_dir():
        result = compare_top_n(Path(args.paths[0]), top_n=args.top_n, alpha=args.alpha)
    elif len(args.paths) == 2:
        result = compare_two(Path(args.paths[0]), Path(args.paths[1]), alpha=args.alpha)
    else:
        parser.error("Provide exactly 2 score files or 1 directory for top-N comparison")

    output = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(output)
        print(f"Results written to {args.output}")
    else:
        print(output)

    # Summary
    if "comparisons" in result:
        print(f"\n{'='*60}")
        print(f"Pairwise comparisons: {result['n_comparisons']}")
        sig_count = sum(1 for c in result["comparisons"] if c.get("significant_after_correction"))
        print(f"Significant after Holm-Bonferroni: {sig_count}/{result['n_comparisons']}")
    elif "paired_bootstrap" in result:
        bs = result["paired_bootstrap"]
        print(f"\n{'='*60}")
        print(f"{result['model_a']} vs {result['model_b']} (n={result['n_common_samples']})")
        print(f"Mean A: {result['mean_a']:.4f}  Mean B: {result['mean_b']:.4f}")
        print(f"Diff: {bs['mean_diff']:.4f}  CI: [{bs['ci_lower']:.4f}, {bs['ci_upper']:.4f}]")
        print(f"Bootstrap p={bs['p_value']:.4f}  {'SIGNIFICANT' if bs['significant'] else 'not significant'}")
        mc = result["mcnemar"]
        print(f"McNemar p={mc['p_value']:.4f}  {'SIGNIFICANT' if mc['significant'] else 'not significant'}")


if __name__ == "__main__":
    main()
