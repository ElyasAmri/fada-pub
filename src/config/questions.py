"""
Single source of truth for the 8 clinical VQA questions.

All question text, short names, and column names are defined here.
Other modules should import from this file rather than defining their own copies.

No torch or heavy dependencies -- safe to import anywhere.
"""

from typing import List

# Full question texts (matches Excel annotation column wording)
QUESTIONS: List[str] = [
    "Anatomical Structures Identification: Identify and describe all anatomical structures visible in the image.",
    "Fetal Orientation: Determine the orientation of the fetus based on the image (e.g., head up/down, front/back view).",
    "Plane Evaluation: Assess if the image is taken at a standard diagnostic plane and describe its diagnostic relevance.",
    "Biometric Measurements: Identify any measurable biometric parameters (e.g., femur length, head circumference) from the image.",
    "Gestational Age: Estimate the gestational age of the fetus based on the visible features.",
    "Image Quality: Assess the quality of the ultrasound image, mentioning any factors that might affect its interpretation (e.g., clarity, artifacts).",
    "Normality / Abnormality: Determine whether the observed structures appear normal or identify any visible abnormalities or concerns.",
    "Clinical Recommendations: Provide any relevant clinical recommendations or suggested next steps based on your interpretation.",
]

# Short display names (without Q# prefix)
QUESTION_SHORT_NAMES: List[str] = [
    "Anatomical Structures",
    "Fetal Orientation",
    "Imaging Plane",
    "Biometric Measurements",
    "Gestational Age",
    "Image Quality",
    "Normality Assessment",
    "Clinical Recommendations",
]

# Column names matching Excel/JSONL format ("Q1: ...", "Q2: ...", etc.)
QUESTION_COLUMNS: List[str] = [
    "Q1: Anatomical Structures",
    "Q2: Fetal Orientation",
    "Q3: Imaging Plane",
    "Q4: Biometric Measurements",
    "Q5: Gestational Age",
    "Q6: Image Quality",
    "Q7: Normality Assessment",
    "Q8: Clinical Recommendations",
]
