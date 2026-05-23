"""
Export Gemini annotations from JSON to Excel format.

Converts the Gemini API annotations JSON file to Excel format matching
the existing annotation structure.
"""

import json
import argparse
from pathlib import Path
import pandas as pd


def load_json(json_path):
    """Load JSON file."""
    print(f"Loading JSON from {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def process_annotations(data):
    """Convert JSON annotations to DataFrame rows."""
    print("Processing annotations...")

    completed_images = data.get('completed_images', {})
    model_name = data.get('model', 'gemini-3-flash-preview')

    rows = []
    skipped = 0

    for img_key, img_data in completed_images.items():
        questions = img_data.get('questions', [])

        # Skip if all questions errored
        if all(q.get('error') is not None for q in questions):
            skipped += 1
            continue

        # Build question responses map (index -> response)
        responses = {}
        evaluations = []

        for q in questions:
            idx = q.get('question_idx')
            response = q.get('response', '')
            error = q.get('error')

            # Use empty string if error, otherwise use response
            responses[idx] = '' if error else response

            # Collect evaluation data
            eval_data = q.get('evaluation')
            if not error and eval_data is not None:
                evaluations.append({
                    'overall_score': eval_data.get('overall_score', 0),
                    'word_count': eval_data.get('word_count', 0),
                    'hallucination_penalty': eval_data.get('hallucination_penalty', 0)
                })

        # Check if all 8 questions have responses
        status = 'complete' if all(
            responses.get(i, '').strip() != '' for i in range(8)
        ) else 'partial'

        # Calculate evaluation metrics
        overall_score_avg = (
            sum(e['overall_score'] for e in evaluations) / len(evaluations)
            if evaluations else 0
        )
        word_count_total = sum(e['word_count'] for e in evaluations)
        hallucination_penalty_avg = (
            sum(e['hallucination_penalty'] for e in evaluations) / len(evaluations)
            if evaluations else 0
        )

        # Build row
        row = {
            'Folder Name': img_data.get('category', ''),
            'Image Name': img_data.get('image', ''),
            'Q1: Anatomical Structures': responses.get(0, ''),
            'Q2: Fetal Orientation': responses.get(1, ''),
            'Q3: Imaging Plane': responses.get(2, ''),
            'Q4: Biometric Measurements': responses.get(3, ''),
            'Q5: Gestational Age': responses.get(4, ''),
            'Q6: Image Quality': responses.get(5, ''),
            'Q7: Normality Assessment': responses.get(6, ''),
            'Q8: Clinical Recommendations': responses.get(7, ''),
            'Annotator': model_name,
            'Status': status,
            'Overall Score (Avg)': round(overall_score_avg, 3),
            'Word Count (Total)': word_count_total,
            'Hallucination Penalty (Avg)': round(hallucination_penalty_avg, 3)
        }

        rows.append(row)

    print(f"Processed {len(rows)} images ({skipped} skipped due to errors)")

    return pd.DataFrame(rows)


def save_excel(df, output_path):
    """Save DataFrame to Excel."""
    print(f"Saving Excel to {output_path}...")

    # Sort by Folder Name then Image Name
    df = df.sort_values(['Folder Name', 'Image Name'])

    # Save to Excel
    df.to_excel(output_path, index=False, engine='openpyxl')

    print(f"Saved {len(df)} rows to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Export Gemini annotations from JSON to Excel'
    )

    # Get script directory for relative paths
    script_dir = Path(__file__).parent
    default_input = script_dir / 'results' / 'gemini_annotations_complete.json'
    default_output = script_dir / 'results' / 'gemini_annotations.xlsx'

    parser.add_argument(
        '--input',
        type=Path,
        default=default_input,
        help=f'Input JSON file (default: {default_input})'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=default_output,
        help=f'Output Excel file (default: {default_output})'
    )

    args = parser.parse_args()

    # Load JSON
    data = load_json(args.input)

    # Process annotations
    df = process_annotations(data)

    # Save Excel
    save_excel(df, args.output)

    print("Done!")


if __name__ == '__main__':
    main()
