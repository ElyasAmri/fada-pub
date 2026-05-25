"""Expand all abbreviations in the normalized annotations Excel."""
import re
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_root))

import pandas as pd

INPUT = _root / "data" / "Fetal Ultrasound Annotations Normalized.xlsx"
OUTPUT = INPUT  # overwrite in place

# Abbreviation -> expansion (word-boundary safe, case-sensitive)
ABBREVIATIONS = {
    "IVC": "inferior vena cava",
    "AC": "abdominal circumference",
    "HC": "head circumference",
    "BPD": "biparietal diameter",
    "CRL": "crown-rump length",
    "NT": "nuchal translucency",
    "FL": "femur length",
    "FMF": "frontomaxillary facial angle",
    "IT": "intracranial translucency",
    "VSD": "ventricular septal defect",
    "ASD": "atrial septal defect",
    "US": "ultrasound",
    "MRI": "magnetic resonance imaging",
    "OFD": "occipitofrontal diameter",
    "TCD": "transcerebellar diameter",
    "CL": "cervical length",
    "NF": "nuchal fold",
    "NB": "nasal bone",
    "AOP": "angle of progression",
}

# Typo corrections found in the data
TYPO_FIXES = {
    "NY": "nuchal translucency",   # misspelling of NT
    "CTL": "crown-rump length",    # misspelling of CRL
    "CFL": "crown-rump length",    # misspelling of CRL
    "OF": "occipitofrontal diameter",  # incomplete OFD
}

# Columns to process
Q_COLUMNS = [
    "Q1: Anatomical Structures",
    "Q2: Fetal Orientation",
    "Q3: Imaging Plane",
    "Q4: Biometric Measurements",
    "Q5: Gestational Age",
    "Q6: Image Quality",
    "Q7: Normality Assessment",
    "Q8: Clinical Recommendations",
]


def expand_text(text: str) -> str:
    """Expand abbreviations in a single text value."""
    if not isinstance(text, str) or not text.strip():
        return text

    result = text

    # Apply typo fixes first (they're more specific)
    for abbr, expansion in TYPO_FIXES.items():
        result = re.sub(r'\b' + re.escape(abbr) + r'\b', expansion, result)

    # Apply standard abbreviation expansions
    # Sort by length descending to match longer abbreviations first
    for abbr, expansion in sorted(ABBREVIATIONS.items(), key=lambda x: -len(x[0])):
        result = re.sub(r'\b' + re.escape(abbr) + r'\b', expansion, result)

    return result


def main():
    print(f"Loading {INPUT}")
    df = pd.read_excel(INPUT)
    print(f"Loaded {len(df)} rows")

    total_changes = 0
    for col in Q_COLUMNS:
        if col not in df.columns:
            print(f"  WARNING: Column {col} not found")
            continue

        changes = 0
        for idx in df.index:
            val = df.at[idx, col]
            if pd.isna(val):
                continue
            original = str(val)
            expanded = expand_text(original)
            if expanded != original:
                df.at[idx, col] = expanded
                changes += 1

        print(f"  {col}: {changes} cells expanded")
        total_changes += changes

    print(f"\nTotal changes: {total_changes}")

    # Verify no abbreviations remain
    print("\nVerification - remaining uppercase abbreviations:")
    remaining = 0
    for col in Q_COLUMNS:
        if col not in df.columns:
            continue
        vals = df[col].dropna().astype(str)
        for v in vals:
            abbrs = re.findall(r'\b[A-Z]{2,5}\b', v)
            # Filter out acceptable uppercase words
            abbrs = [a for a in abbrs if a not in ('AORTA', 'LOW')]
            if abbrs:
                remaining += 1
                if remaining <= 10:
                    print(f"  {col}: {abbrs} in: {v[:100]}")

    if remaining == 0:
        print("  None found - all abbreviations expanded!")
    else:
        print(f"  {remaining} cells still have abbreviations")

    # Save
    print(f"\nSaving to {OUTPUT}")
    df.to_excel(OUTPUT, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
