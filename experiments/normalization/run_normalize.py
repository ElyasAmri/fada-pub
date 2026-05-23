"""
Apply annotation normalization and generate cleaned dataset.

Usage:
    # Full normalization with report
    python experiments/normalization/run_normalize.py

    # Dry run (report only, no output file)
    python experiments/normalization/run_normalize.py --dry-run

    # Disable specific layers for ablation
    python experiments/normalization/run_normalize.py --no-semantic

    # Custom input/output paths
    python experiments/normalization/run_normalize.py \
        --input "data/Fetal Ultrasound Annotations Final.xlsx" \
        --output "data/Fetal Ultrasound Annotations Normalized.xlsx"
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.normalize_annotations import (
    AnnotationNormalizer,
    NormalizationReport,
    QUESTION_COLUMNS,
)

DEFAULT_INPUT = PROJECT_ROOT / "data" / "Fetal Ultrasound Annotations Final.xlsx"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "Fetal Ultrasound Annotations Normalized.xlsx"
DEFAULT_CHANGELOG = PROJECT_ROOT / "data" / "normalization_changelog.json"


def save_changelog(report: NormalizationReport, path: Path) -> None:
    """Save the detailed changelog as JSON."""
    changes = []
    for c in report.changelog:
        changes.append({
            "row_index": int(c.row_index),
            "column": c.column,
            "original": c.original,
            "normalized": c.normalized,
            "layer": c.layer,
            "rule": c.rule,
        })

    data = {
        "total_cells": report.total_cells,
        "cells_changed": report.cells_changed,
        "changes_by_layer": report.changes_by_layer,
        "changes_by_question": report.changes_by_question,
        "unique_values_before": report.unique_values_before,
        "unique_values_after": report.unique_values_after,
        "unmapped_values": {
            col: [{"value": v, "count": c} for v, c in vals]
            for col, vals in report.unmapped_values.items()
        },
        "changelog": changes,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(
        description="Normalize fetal ultrasound annotation answers"
    )
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT,
        help="Input Excel file (default: data/Fetal Ultrasound Annotations Final.xlsx)",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help="Output Excel file (default: data/Fetal Ultrasound Annotations Normalized.xlsx)",
    )
    parser.add_argument(
        "--changelog", type=Path, default=DEFAULT_CHANGELOG,
        help="Changelog JSON output path",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report only, do not write output files",
    )
    parser.add_argument(
        "--no-basic", action="store_true",
        help="Disable layer 1 (basic text cleanup)",
    )
    parser.add_argument(
        "--no-spelling", action="store_true",
        help="Disable layer 2 (spelling correction)",
    )
    parser.add_argument(
        "--no-semantic", action="store_true",
        help="Disable layer 3 (semantic unification)",
    )
    args = parser.parse_args()

    # Validate input
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    # Load
    print(f"Loading annotations from: {args.input}")
    df = pd.read_excel(args.input)
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Verify expected columns exist
    missing = [c for c in QUESTION_COLUMNS if c not in df.columns]
    if missing:
        print(f"Error: Missing expected columns: {missing}")
        sys.exit(1)

    # Normalize
    normalizer = AnnotationNormalizer(
        enable_basic=not args.no_basic,
        enable_spelling=not args.no_spelling,
        enable_semantic=not args.no_semantic,
    )

    layers = []
    if not args.no_basic:
        layers.append("basic")
    if not args.no_spelling:
        layers.append("spelling")
    if not args.no_semantic:
        layers.append("semantic")
    print(f"  Enabled layers: {', '.join(layers)}")
    print("  Normalizing...")

    normalized_df, report = normalizer.normalize_dataframe(df)

    # Print summary
    print()
    print("=" * 60)
    print("NORMALIZATION REPORT")
    print("=" * 60)
    print(report.summary())
    print("=" * 60)

    if args.dry_run:
        print("\n[DRY RUN] No output files written.")
        return

    # Save normalized Excel
    print(f"\nSaving normalized annotations to: {args.output}")
    normalized_df.to_excel(args.output, index=False, engine="openpyxl")
    print(f"  Written {len(normalized_df):,} rows")

    # Save changelog
    print(f"Saving changelog to: {args.changelog}")
    save_changelog(report, args.changelog)
    print(f"  Written {len(report.changelog):,} change records")

    print("\nDone.")


if __name__ == "__main__":
    main()
