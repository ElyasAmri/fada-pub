"""
Aggregate framework comparison results into a summary table.

Scans all run directories for run_manifest.json and scores.json,
produces a markdown table and JSON summary.

Usage:
    python experiments/framework_comparison/aggregate_results.py
    python experiments/framework_comparison/aggregate_results.py --runs-dir /path/to/runs
"""

import argparse
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent


def collect_results(runs_dir: Path) -> list:
    """Collect results from all run directories."""
    results = []

    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        manifest_path = run_dir / "run_manifest.json"
        if not manifest_path.exists():
            continue

        with open(manifest_path) as f:
            manifest = json.load(f)

        entry = {
            "run_dir": run_dir.name,
            "model": manifest.get("model", ""),
            "framework": manifest.get("framework", ""),
            "status": manifest.get("status", "unknown"),
            "training_time_seconds": manifest.get("training_time_seconds"),
            "final_loss": manifest.get("final_loss"),
            "gpu_memory_peak_gb": manifest.get("gpu_memory_peak_gb"),
        }

        # Load scores if available
        scores_path = run_dir / "scores.json"
        if scores_path.exists():
            with open(scores_path) as f:
                scores = json.load(f)

            # Extract primary score (embedding similarity)
            entry["primary_score"] = scores.get("overall", {}).get("embedding_similarity")
            entry["f2_score"] = scores.get("overall", {}).get("f2_score")

            # Per-question scores
            per_q = scores.get("per_question", {})
            entry["per_question_scores"] = {
                q: v.get("embedding_similarity") for q, v in per_q.items()
            } if per_q else {}
        else:
            entry["primary_score"] = None
            entry["f2_score"] = None
            entry["per_question_scores"] = {}

        results.append(entry)

    return results


def format_time(seconds):
    """Format seconds as HH:MM:SS."""
    if seconds is None:
        return "-"
    h, m = divmod(int(seconds), 3600)
    m, s = divmod(m, 60)
    if h > 0:
        return f"{h}h{m:02d}m"
    return f"{m}m{s:02d}s"


def format_pct(value):
    """Format a float as percentage."""
    if value is None:
        return "-"
    return f"{value * 100:.1f}%"


def generate_markdown(results: list) -> str:
    """Generate markdown comparison table."""
    lines = ["# Framework Comparison Results\n"]

    # Separate completed and failed
    completed = [r for r in results if r["status"] == "complete" and r["primary_score"] is not None]
    trained_only = [r for r in results if r["status"] == "complete" and r["primary_score"] is None]
    failed = [r for r in results if r["status"] != "complete"]

    if completed:
        # Build pivot table: models as rows, frameworks as columns
        models = sorted(set(r["model"] for r in completed))
        frameworks = sorted(set(r["framework"] for r in completed))

        # Score table
        lines.append("## Scores (Embedding Similarity)\n")
        header = "| Model | " + " | ".join(frameworks) + " |"
        sep = "|" + "|".join(["---"] * (len(frameworks) + 1)) + "|"
        lines.append(header)
        lines.append(sep)

        for model in models:
            row = f"| {model} |"
            for fw in frameworks:
                match = [r for r in completed if r["model"] == model and r["framework"] == fw]
                if match:
                    row += f" {format_pct(match[0]['primary_score'])} |"
                else:
                    row += " - |"
            lines.append(row)

        # Training time table
        lines.append("\n## Training Time\n")
        lines.append(header)
        lines.append(sep)

        for model in models:
            row = f"| {model} |"
            for fw in frameworks:
                match = [r for r in completed if r["model"] == model and r["framework"] == fw]
                if match:
                    row += f" {format_time(match[0]['training_time_seconds'])} |"
                else:
                    row += " - |"
            lines.append(row)

        # GPU memory table
        lines.append("\n## GPU Memory Peak (GB)\n")
        lines.append(header)
        lines.append(sep)

        for model in models:
            row = f"| {model} |"
            for fw in frameworks:
                match = [r for r in completed if r["model"] == model and r["framework"] == fw]
                if match and match[0]["gpu_memory_peak_gb"]:
                    row += f" {match[0]['gpu_memory_peak_gb']:.1f} |"
                else:
                    row += " - |"
            lines.append(row)

        # Best results
        lines.append("\n## Best Results\n")
        best = max(completed, key=lambda r: r["primary_score"] or 0)
        lines.append(f"- **Best score**: {format_pct(best['primary_score'])} "
                     f"({best['model']} + {best['framework']})")
        fastest = min(completed, key=lambda r: r["training_time_seconds"] or float("inf"))
        lines.append(f"- **Fastest**: {format_time(fastest['training_time_seconds'])} "
                     f"({fastest['model']} + {fastest['framework']})")

    if trained_only:
        lines.append("\n## Trained (Awaiting Evaluation)\n")
        for r in trained_only:
            lines.append(f"- {r['model']} + {r['framework']} "
                         f"(loss: {r['final_loss']}, time: {format_time(r['training_time_seconds'])})")

    if failed:
        lines.append("\n## Failed Runs\n")
        for r in failed:
            lines.append(f"- {r['model']} + {r['framework']}: {r['status']}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Aggregate framework comparison results")
    parser.add_argument("--runs-dir", type=Path, default=SCRIPT_DIR / "runs")
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    if not args.runs_dir.exists():
        print(f"No runs directory found at {args.runs_dir}")
        return

    results = collect_results(args.runs_dir)
    if not results:
        print("No results found.")
        return

    print(f"Found {len(results)} runs")

    # Generate markdown
    md = generate_markdown(results)
    md_path = args.output_md or args.runs_dir / "comparison_results.md"
    with open(md_path, "w") as f:
        f.write(md)
    print(f"Markdown table saved to {md_path}")
    print(md)

    # Save JSON summary
    json_path = args.output_json or args.runs_dir / "comparison_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON summary saved to {json_path}")


if __name__ == "__main__":
    main()
