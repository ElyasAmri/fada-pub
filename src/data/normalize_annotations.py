"""
Annotation Normalizer for Fetal Ultrasound VQA Dataset.

Provides layered text normalization for the 8 clinical questions (Q1-Q8)
in the FADA annotation dataset. Reduces ~7,000 unique answer variants to
~2,000 canonical forms without loss of clinical meaning.

Normalization layers (applied in order):
  1. Basic cleanup: whitespace collapse, trailing punctuation
  2. Spelling correction: systematic misspelling fixes (word-boundary safe)
  3. Semantic unification: per-question canonical mapping

The original text casing is preserved when only basic cleanup applies.
Semantic mappings produce consistently cased canonical forms.

No torch or heavy dependencies -- safe to import anywhere.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import pandas as pd

from src.config.questions import QUESTION_COLUMNS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Layer 2: Spelling corrections (whole-word, case-insensitive)
# ---------------------------------------------------------------------------
# Each entry: misspelling -> correction
# Counts from analysis of 18,936 rows (2026-03-02)
SPELLING_CORRECTIONS: Dict[str, str] = {
    "probablity": "probability",       # 1,588 occurrences
    "symphsis": "symphysis",            # 1,354
    "montoring": "monitoring",          # 1,104
    "vertebrea": "vertebrae",           # 985
    "calverial": "calvarial",           # 819
    "moniotoring": "monitoring",        # 774
    "veiw": "view",                     # 735
    "delivary": "delivery",             # 648
    "longituidinal": "longitudinal",    # 546
    "palne": "plane",                   # 401
    "longituidnal": "longitudinal",     # 402
    "arota": "aorta",                   # 296
    "cephalaic": "cephalic",            # 162
    "chammber": "chamber",              # 104
    "amnitoic": "amniotic",             # 87
    "cervicx": "cervix",               # 72
    "qualtiy": "quality",              # 64
    "abnormall": "abnormal",            # 51
    "rouitne": "routine",              # ~100
    "mointoring": "monitoring",         # 13
    "pernatal": "perinatal",           # 20
    "resoulation": "resolution",       # ~10
    "thoaracic": "thoracic",           # ~75
    "longtuidinal": "longitudinal",    # 77
    "transvers": "transverse",         # ~75 (standalone, not prefix)
    "monitering": "monitoring",        # 8
    "centrlized": "centralized",       # ~29
    "penetraion": "penetration",       # ~44
    "viwe": "view",                    # 22
    "measrured": "measured",           # 12
    "folliw": "follow",               # 4
    "nucal": "nuchal",                 # ~230 (in longer phrases)
    "bocket": "pocket",               # ~53
    "occpital": "occipital",          # ~2
    "nirmal": "normal",                # 2
    "thickned": "thickened",           # ~4
    "anomlaies": "anomalies",         # ~2
    "lengh": "length",                # ~8
    # Q2-specific misspellings (safe globally -- these words don't appear elsewhere)
    "occipit": "occiput",             # ~600 Q2 CRL-view
    "presntation": "presentation",    # ~50 Q2
    "presentaton": "presentation",    # Q2
    "presention": "presentation",     # Q2
    "breach": "breech",               # ~620 Q2 (always breech in this dataset)
    "breacj": "breech",               # 6 Q2
    "breaach": "breech",              # 1 Q2
    "brach": "breech",                # 1 Q2
    "brrech": "breech",               # 1 Q2
    "ceohalic": "cephalic",           # 3 Q2
    "cehalic": "cephalic",            # 4 Q2
    "cephaic": "cephalic",            # 2 Q2
    "cehphalic": "cephalic",          # 2 Q2
    "cephlaic": "cephalic",           # 2 Q2
    "screeen": "screen",              # 37 Q2
    "columon": "column",              # 78 Q2
    "transvere": "transverse",        # 12 Q2
    "iamge": "image",                 # 3 Q2/Q4
    # Q1-specific misspellings (safe globally -- not real words)
    "cerberi": "cerebri",             # ~1,038 Q1 (falx cerberi)
    "cereberi": "cerebri",            # ~359 Q1 variant
    "umblical": "umbilical",          # ~970 Q1
    "pareito": "parieto",             # ~726 Q1 (parieto-occipital)
    "frontoparital": "frontoparietal",  # ~366 Q1
    "pellucidium": "pellucidum",      # ~252 Q1 (cavum septum)
    "thakami": "thalami",             # ~87 Q1
    "stuctures": "structures",        # ~624 Q1 (intracranial structures)
    "abdomne": "abdomen",             # ~10 Q1
    "verterbal": "vertebral",         # ~5 Q1
    "suprenal": "suprarenal",         # ~3 Q1
    "vertbrea": "vertebrae",          # ~80 Q1 (another variant)
    "ceberi": "cerebri",              # ~66 Q1 (another variant)
    "amnoitic": "amniotic",           # ~59 Q1
    "gentalia": "genitalia",          # ~107 Q1
    "genatalia": "genitalia",         # Q1 variant
    # Q3-specific misspellings (sagittal variants)
    "sagital": "sagittal",             # ~200 Q3
    "saggital": "sagittal",            # ~50 Q3
    "dagital": "sagittal",             # ~30 Q3 (typo, d near s on keyboard)
    "midsagital": "midsagittal",       # ~15 Q3
}

# Pre-compile spelling regex patterns (word-boundary, case-insensitive)
_SPELLING_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\b" + re.escape(wrong) + r"\b", re.IGNORECASE), correct)
    for wrong, correct in SPELLING_CORRECTIONS.items()
]


# ---------------------------------------------------------------------------
# Layer 3: Semantic canonical mappings (per question)
# ---------------------------------------------------------------------------
# Keys are lowercase, whitespace-collapsed, period-stripped forms.
# Values are the canonical output string.

# Q2: Fetal Orientation (838 raw -> ~35 canonical via regex)
# Most normalization done by _normalize_q2_regex(); dict entries optimize top values.
Q2_CANONICAL: Dict[str, str] = {
    # Simple presentations
    "cephalic presentation": "Cephalic presentation",
    "breech presentation": "Breech presentation",
    "cephalic position": "Cephalic presentation",
    "cephalic": "Cephalic presentation",
    "head presentation": "Cephalic presentation",
    "fetal presentation": "Cephalic presentation",
    "cephalic view": "Cephalic presentation",
    # Anatomical views
    "longitudinal view of femur": "Longitudinal view of femur",
    "coronal view of aorta": "Coronal view of aorta",
    "sagittal view of aorta": "Sagittal view of aorta",
    # Fetus not seen
    "fetus is not seen in the image": "Fetus not seen",
    "fetus not seen in the image": "Fetus not seen",
    "fetus is not seen": "Fetus not seen",
    "fetus not seen": "Fetus not seen",
    "fetus is not in the images": "Fetus not seen",
    "fetus is not seen image": "Fetus not seen",
    "fetus not seen in the iamge": "Fetus not seen",
    "fetus is not seen in the iamge": "Fetus not seen",
    # Cephalic transverse (no laterality)
    "cephalic and transverse plane": "Cephalic transverse",
    "cephalic and transverse view": "Cephalic transverse",
    "cephalic and transverse": "Cephalic transverse",
    "cephalic transverse view": "Cephalic transverse",
    "cephalic and transverse plan": "Cephalic transverse",
    "cephalic and transverse image": "Cephalic transverse",
    "cephalic and transverse plane": "Cephalic transverse",
    "transverse and cephalic view": "Cephalic transverse",
    "cephalic and coronal view": "Cephalic coronal",
    # Axial abdomen (no vertebral column direction)
    "axial view of upper abdomen": "Axial upper abdomen",
    "axial view for upper abdomen": "Axial upper abdomen",
    "axial view of mid abdomen": "Axial mid abdomen",
    # Specific terms
    "occipitoposterior": "Occipitoposterior",
    "longitudinal orientation, cephalic presentation": "Cephalic presentation",
}

# Q3: Imaging Plane (229 raw -> ~15 canonical)
Q3_CANONICAL: Dict[str, str] = {
    # Sagittal views
    "sagittal": "Sagittal",
    "sagittal view": "Sagittal",
    "sagittal view of fetus": "Sagittal",
    "mid sagittal": "Mid-sagittal",
    "mid-sagittal": "Mid-sagittal",
    # Sagittal plane of head
    "sagittal plane of head": "Sagittal plane of head",
    "sagittal plane of the head": "Sagittal plane of head",
    "sagittal palne of head": "Sagittal plane of head",
    "axial plane of the head": "Axial plane of head",
    # Sagittal of head + symphysis
    "sagittal palne of head and symphysis pubis": "Sagittal plane of head and symphysis pubis",
    "sagittal plane of head and symphysis pubis": "Sagittal plane of head and symphysis pubis",
    # Mid-sagittal view of head and neck
    "mid sagittal view of head and neck": "Mid-sagittal view of head and neck",
    "mid-sagittal view of head and neck": "Mid-sagittal view of head and neck",
    "mid sagittal view for head and neck": "Mid-sagittal view of head and neck",
    "mid -sagittal view for head and neck": "Mid-sagittal view of head and neck",
    "mid -sagittal view of head and neck": "Mid-sagittal view of head and neck",
    # Para-sagittal view of head and neck
    "para-sagittal view of head and neck": "Para-sagittal view of head and neck",
    "para-sagittal view for head and neck": "Para-sagittal view of head and neck",
    "para sagittal view of head and neck": "Para-sagittal view of head and neck",
    # Sagittal with detailed profile (drrehab style)
    "sagittal frontal ones and nasal bone, upper lip in profile corpus callosum, cavum septum pellucidi are likely in plane": "Sagittal profile (frontal bone, nasal bone, corpus callosum)",
    "sagittal frontal ones and nasal bone, upper lip in profile": "Sagittal profile (frontal bone, nasal bone)",
    # Trans-thalamic
    "trans-thalamic view": "Trans-thalamic view",
    "trans-thalamic": "Trans-thalamic view",
    "trans thalamic view": "Trans-thalamic view",
    # Trans-cerebellar
    "trans-cerebellar view": "Trans-cerebellar view",
    "transcerebellar view": "Trans-cerebellar view",
    "trans cerebellar view": "Trans-cerebellar view",
    "trans-cerebellar": "Trans-cerebellar view",
    # Trans-ventricular
    "trans-ventricular view": "Trans-ventricular view",
    "trans ventricular view": "Trans-ventricular view",
    "transventricular view": "Trans-ventricular view",
    # Transverse trans-abdominal
    "transverse and trans-abdominal plane": "Transverse trans-abdominal plane",
    "transverse and transabdominal plane": "Transverse trans-abdominal plane",
    "transverse and trans-abdominal view": "Transverse trans-abdominal plane",
    "transverse trans-abdominal plane": "Transverse trans-abdominal plane",
    "transverse transabdominal plane": "Transverse trans-abdominal plane",
    # Transverse 4-chamber heart
    "transverse plane with 4 chamber heart view": "Transverse 4-chamber heart view",
    "transverse plane with 4 chamber view of the heart": "Transverse 4-chamber heart view",
    "transverse plane with 4 chamber of heart view": "Transverse 4-chamber heart view",
    "transverse plane with 4 chambers heart view": "Transverse 4-chamber heart view",
    "transverse thoracic plane with 4 chambers heart view": "Transverse 4-chamber heart view",
    "transverse thorax plane with 4 chamber heart view": "Transverse 4-chamber heart view",
    "transverse plane of the thorax and 4-chamber of heart": "Transverse 4-chamber heart view",
    "transverse thoaracic plane with 4 chambers heart view": "Transverse 4-chamber heart view",
    # Longitudinal femur
    "longitudinal view of femur": "Longitudinal view of femur",
    "longitudinal veiw of femur": "Longitudinal view of femur",
    # Coronal aorta
    "coronal view of aorta": "Coronal view of aorta",
    "coronal view of arota": "Coronal view of aorta",
    "coronal": "Coronal",
    # Sagittal aorta
    "sagittal view of aorta": "Sagittal view of aorta",
    # Longitudinal cervix
    "longitudinal view of the cervix": "Longitudinal view of cervix",
    "longitudinal view of cervix": "Longitudinal view of cervix",
    "longituidinal view of the cervix": "Longitudinal view of cervix",
    "longituidinal view of cervix": "Longitudinal view of cervix",
    "longituidnal view of cervix": "Longitudinal view of cervix",
    "longituidnal view of the cervix": "Longitudinal view of cervix",
    "longituidinal veiw of cervix": "Longitudinal view of cervix",
    "longtuidinal view of the cervix": "Longitudinal view of cervix",
    "longitudinal veiw of the cervix": "Longitudinal view of cervix",
    "longitudinal veiw of cervix": "Longitudinal view of cervix",
    # Sagittal cervix
    "sagittal plane of cervix": "Sagittal plane of cervix",
    "sagittal cut of cervix": "Sagittal plane of cervix",
    # Transverse 4-chamber heart (hyphenated variants)
    "transverse plane with 4-chamber heart view": "Transverse 4-chamber heart view",
    "transverse plane with 4-chamber view of the heart": "Transverse 4-chamber heart view",
    "4-chamber heart view": "Transverse 4-chamber heart view",
    "4- chamber heart view": "Transverse 4-chamber heart view",
    "transverse thoracic plane with 4-chamber heart view": "Transverse 4-chamber heart view",
    "transverse thorax plane with 4-chamber heart view": "Transverse 4-chamber heart view",
    # Axial plane of head
    "axial plane of head": "Axial plane of head",
    "axial plane of the head": "Axial plane of head",
    # Longitudinal cervix (additional variants)
    "longitudinal view cervix": "Longitudinal view of cervix",
    "longitudinal viwe of the cervix": "Longitudinal view of cervix",
    "longitudinal viwe of cervix": "Longitudinal view of cervix",
    "longtuidinal view of cervix": "Longitudinal view of cervix",
    # Combined planes (cervical folder)
    "axial cut of abdomen sagittal cut of cervix": "Axial abdomen with sagittal cervix",
    "coronal cut of fetal head sagittal cut of cervix": "Coronal head with sagittal cervix",
    "sagittal cut of cervix coronal cut of fetal head": "Coronal head with sagittal cervix",
    "axial cuts of spine sagittal cut of cervix": "Axial spine with sagittal cervix",
    # Sagittal profile (additional variants)
    "sagittal frontal bones and nasal bone, upper lip in profile": "Sagittal profile (frontal bone, nasal bone)",
    # Mid-sagittal with hyphen variants
    "mid-sagittal view for head and neck": "Mid-sagittal view of head and neck",
    "mid-sagittal view of head and neck region": "Mid-sagittal view of head and neck",
    # Trans-cerebellar additional
    "transcebellar view": "Trans-cerebellar view",
    # Para-sagittal additional
    "parasagittal": "Para-sagittal view of head and neck",
    "para -sagittal view of head and neck": "Para-sagittal view of head and neck",
    "para -sagittal view for head and neck": "Para-sagittal view of head and neck",
    # Mid-sagittal CRL
    "mid sagittal of crl": "Mid-sagittal view of CRL",
    "mid sagittal plane of crl": "Mid-sagittal view of CRL",
    # Axial with heart
    "axial plane with 4 chamber of heart view": "Transverse 4-chamber heart view",
    # Combined cervical planes (additional)
    "sagittal cut of fetal head sagittal cut of cervix": "Sagittal head with sagittal cervix",
    "sagittal cut of cervix sagittal cut of fetal head": "Sagittal head with sagittal cervix",
    "axial cut of abdominal circumference sagittal cut of cervix": "Axial abdomen with sagittal cervix",
    "sagittal cut of cervix axial cuts of fetal abdomen": "Axial abdomen with sagittal cervix",
    "axial cut of fetal head sagittal cut of cervix": "Coronal head with sagittal cervix",
    # Coronal additional
    "coronal view": "Coronal",
    "coronal frontal ones and nasal bone, upper lip in profile": "Sagittal profile (frontal bone, nasal bone)",
    "coronal frontal ones and nasal bone, upper lip in profile corpus callosum, cavum septum pellucidi are likely in plane": "Sagittal profile (frontal bone, nasal bone, corpus callosum)",
    # Mid-sagittal additional
    "mid sagittal plane": "Mid-sagittal",
    "mid sagital": "Mid-sagittal",
    "mid sagital of crl": "Mid-sagittal view of CRL",
    "midsagittal of crl": "Mid-sagittal view of CRL",
    "mid sagittal view of crl": "Mid-sagittal view of CRL",
    # Sagittal view of head and neck
    "sagittal view for head and neck": "Mid-sagittal view of head and neck",
    # Trans-supraventricular
    "trans-supraventricular view": "Trans-supraventricular view",
    "trans-supra-ventricular view": "Trans-supraventricular view",
    # Trans-cerebellar additional
    "trans-cerebellar plane": "Trans-cerebellar view",
    "trans- cerebellar view": "Trans-cerebellar view",
    "transverse trans-cerebellar view": "Trans-cerebellar view",
    # Transverse trans-abdominal additional
    "transverse and transabdominal view": "Transverse trans-abdominal plane",
    "transverse and trans-abdomenal plane": "Transverse trans-abdominal plane",
    # Sagittal + heart
    "sagittal plane with 4 chamber of heart view": "Transverse 4-chamber heart view",
    # Longitudinal cervix typo
    "longitudinal view of cervixb": "Longitudinal view of cervix",
}

# Q4: Biometric Measurements (361 raw -> ~15 canonical)
Q4_CANONICAL: Dict[str, str] = {
    # NT thickness
    "nt thickness": "NT thickness",
    "nt measurement": "NT thickness",
    "nuchal translucency": "NT thickness",
    # NT not measurable
    "nt thickness can not be measured because the image is not in mid sagittal plane": "NT not measurable (not mid-sagittal)",
    "nt thickness can not be measured in this image because it is not in mid sagittal plane": "NT not measurable (not mid-sagittal)",
    "nuchal translucency can not be measured in this image because it is not in mid sagittal plane": "NT not measurable (not mid-sagittal)",
    "nt thickness can not be measured in this image because it is not in the center": "NT not measurable (not centered)",
    "nt thickness can not measured from this image because it is not centrlized": "NT not measurable (not centered)",
    # NT with specific measurements
    "nuchal translucency measures 1.2 mm": "NT measurement (specific value)",
    "nuchal translucency measures 1.3 mm": "NT measurement (specific value)",
    "nuchal translucency measures 1.5 mm": "NT measurement (specific value)",
    "nuchal translucency measures 1.6 mm": "NT measurement (specific value)",
    # CRL + NT
    "crl nt": "CRL and NT",
    "crl nt can be measured. nasal bone length": "CRL, NT, and nasal bone length",
    "crl nt can be measured at it's maximum diameter": "CRL and NT",
    "nasal bone length crl nt": "CRL, NT, and nasal bone length",
    "nasal bone length crl nt frontomaxillary facial angle fmf intracranial translucency": "CRL, NT, nasal bone, FMF angle, IT",
    "nasal bone length crl nt frontomaxillary facial angle intracranial translucency": "CRL, NT, nasal bone, FMF angle, IT",
    # CRL alone
    "crl": "CRL",
    # AC
    "ac": "AC",
    # BPD and HC
    "bpd and hc": "BPD and HC",
    "bpd and hc measurements": "BPD and HC",
    # Femur length
    "femur length": "Femur length",
    # Cervical length
    "cervical length": "Cervical length",
    "cervical canal length": "Cervical length",
    "cervical canal length amniotic fluid bocket": "Cervical length and amniotic fluid",
    "cervical canal length ac": "Cervical length and AC",
    # Aortic diameter
    "aortic transverse diameter": "Aortic transverse diameter",
    # Angle of progression
    "angle of progression - angle between long axis of symphsis pubis and fetal head": "Angle of progression",
    "angle of progression - angle between long axis of symphysis pubis and fetal head": "Angle of progression",
    # Cardiac measurements
    "cardiac chamber dimensions asd and vsd abnormalities lung maturation": "Cardiac chamber dimensions",
    "cardiac chamber dimensions vsd and asd abnormalities lung maturation": "Cardiac chamber dimensions",
    "cardiac chambers dimensions vsd and asd abnormalities lung maturation": "Cardiac chamber dimensions",
    "cardiac chambers dimensions asd and vsd abnormalities lung maturation": "Cardiac chamber dimensions",
    "cardiac chambers dimensions. vsd and asd abnormalities lung maturation": "Cardiac chamber dimensions",
    "cardiac chamber dimensions. vsd and asd abnormalities lung maturation": "Cardiac chamber dimensions",
    "heart chambers dimensions. vsd ad asd abnormalities lung maturation": "Cardiac chamber dimensions",
    # Cerebellar measurements
    "cerebellar and cisterna magna measurements": "Cerebellar and cisterna magna",
    "cerebellar and cisterna magna measurement": "Cerebellar and cisterna magna",
    "cerebellar, cisterna magna measurements": "Cerebellar and cisterna magna",
    "cerebellar, cisterna magna measurement": "Cerebellar and cisterna magna",
    "cerebellar, cisterna magna a measurements": "Cerebellar and cisterna magna",
    "cerebellar, and cisterna magna measurements": "Cerebellar and cisterna magna",
    "cerebellar measurement": "Cerebellar and cisterna magna",
    "cerebellar measurements": "Cerebellar and cisterna magna",
    "cerebellar, cisterna magna and nuchal fold measurements": "Cerebellar, cisterna magna, and nuchal fold",
    "cerebellar, cisterna magna and nuchal fold measurement": "Cerebellar, cisterna magna, and nuchal fold",
    "cerebellar, and cisterna magna and nuchal fold measurements": "Cerebellar, cisterna magna, and nuchal fold",
    "cerebellar, and cisterna magna as well as nuchal fold measurements": "Cerebellar, cisterna magna, and nuchal fold",
    "cerebellar and cisterna magna as well as nuchal fold measurements": "Cerebellar, cisterna magna, and nuchal fold",
    # Lateral ventricle
    "lateral ventricle dimensions": "Lateral ventricle dimensions",
    # NT not measurable (additional variants)
    "nt thickness can not measured from this image because it is not centralized": "NT not measurable (not centered)",
    "nt thickness can not be measured in this image due to it is not in mid sagittal plane": "NT not measurable (not mid-sagittal)",
    "nt thickness can not be measrured in this image due to it is not in mid sagittal plane": "NT not measurable (not mid-sagittal)",
    "nuchal translucency can not be measured in this image due to it is not in mid sagittal plane": "NT not measurable (not mid-sagittal)",
    # Cardiac (additional variants with periods/caps)
    "cardiac chamber dimensions asd and vsd abnormalities. lung maturation": "Cardiac chamber dimensions",
    "cardiac chamber dimensions asd and vasd abnormalities. lung maturation": "Cardiac chamber dimensions",
    # Cerebellar (additional variants)
    "cerebellar and cisterna magna and nuchal fold measurements": "Cerebellar, cisterna magna, and nuchal fold",
    # Combined measurements
    "ac cervical canal length": "AC and cervical length",
    "cervical canal length ac": "Cervical length and AC",
    # Nothing measurable
    "nothing": "No measurable parameters",
    # Cervical + amniotic (post-spelling-correction form: bocket->pocket)
    "cervical canal length amniotic fluid pocket": "Cervical length and amniotic fluid",
    "cervical canal length amniotic fluid bocket": "Cervical length and amniotic fluid",
    # NT specific measurements (regex handles most, but catch common forms)
    "nuchal translucency measures 1.4 mm": "NT measurement (specific value)",
    "nuchal translucency measures 1.1 mm": "NT measurement (specific value)",
    "nuchal translucency measures 2 mm": "NT measurement (specific value)",
    "nuchal translucency measures 1.8 mm": "NT measurement (specific value)",
    "nuchal translucency measures 1.3mm": "NT measurement (specific value)",
    "nuchal translucency measures 1.6mm": "NT measurement (specific value)",
    # BPD/HC variant
    "hc bpd": "BPD and HC",
    "bpd hc": "BPD and HC",
    # Femur (with spelling correction)
    "femur lengh": "Femur length",
    "femur length": "Femur length",
    # Bare NT
    "nt": "NT thickness",
    "nthtickness": "NT thickness",
    # NT not measurable (additional)
    "nt thickness can not measures from this image because it is not centrlize": "NT not measurable (not centered)",
    "nt thickness can not measures because the image is not in mid sagittal plane": "NT not measurable (not mid-sagittal)",
    "nt can not be measured in this image because it is not in mid sagittal plane. crl": "NT not measurable (not mid-sagittal)",
    "nt thickness can not be assessed from this image": "NT not measurable (not mid-sagittal)",
    "nuchal translucency can not be measured in this image because it is not in mid sagittal plane crl": "NT not measurable (not mid-sagittal)",
    "nuchal translucency can not be measured in this image due to it is not in mid sagittal plane crl": "NT not measurable (not mid-sagittal)",
    # CRL + NT variants
    "crl, nt": "CRL and NT",
    "crl nt can be measured": "CRL and NT",
    "nuchal translucency crl": "CRL and NT",
    "crl nt nasal bone length": "CRL, NT, and nasal bone length",
    "crl nt measures 1mm": "CRL and NT",
    # Heart/cardiac (additional)
    "cardiac chambers dimensions vsd and asd abnormlaities lung maturation": "Cardiac chamber dimensions",
    "heart chambers dimensions. detect vsd and asd lung maturation": "Cardiac chamber dimensions",
    "heart chambers dimensions. vsd and asd abnormalities lung maturation": "Cardiac chamber dimensions",
    "heart chambers dimensions. asd and vsd abnormalities lung maturation": "Cardiac chamber dimensions",
    "heart chambers dimensions vsd and asd abnormalities lung maturation": "Cardiac chamber dimensions",
    "heart chambers dimensions. vsd and asd abnormalities. lung maturation": "Cardiac chamber dimensions",
    # Cerebellar (additional typo variants)
    "cerebellar, cisterna magna meaurements": "Cerebellar and cisterna magna",
    "cerebellar, cisterna magna and nuchal fold meaurements": "Cerebellar, cisterna magna, and nuchal fold",
    "cerebellar, cisterna magna as well as nuchal fold measurements": "Cerebellar, cisterna magna, and nuchal fold",
    "cerebellar, cisterna magna as well as nuchal fold meaurements": "Cerebellar, cisterna magna, and nuchal fold",
    "cerbellar, cisterna magna and nuchal fold measurements": "Cerebellar, cisterna magna, and nuchal fold",
    "cerebellar, cisterna magna and measurements": "Cerebellar and cisterna magna",
    "cerebellar and cisterna magna as well nuchal fold measurements": "Cerebellar, cisterna magna, and nuchal fold",
    "cerebellar and cisterna magna as well as nuchal fold measurement": "Cerebellar, cisterna magna, and nuchal fold",
    "cerebellar , cisterna magna and nuchal fold measurement": "Cerebellar, cisterna magna, and nuchal fold",
    "cerebellar and cisterna magna and nuchal fold measurement": "Cerebellar, cisterna magna, and nuchal fold",
    "cerebellar and cisterna measurements": "Cerebellar and cisterna magna",
    "cerebellar and cisterna magna": "Cerebellar and cisterna magna",
    "cerebellar, cisterna magna and nuchal fold": "Cerebellar, cisterna magna, and nuchal fold",
    "cerebellum measurement": "Cerebellar and cisterna magna",
    # Lateral ventricle variants
    "lateral ventricle measurements": "Lateral ventricle dimensions",
    "lateral ventricular dimensions": "Lateral ventricle dimensions",
    # BPD + HC variants
    "bpd and hc a": "BPD and HC",
    "hc bpd": "BPD and HC",
    "bpd and hc and occipital horn of lateral ventricle": "BPD, HC, and lateral ventricle",
    "bpd and hc as well as occipital horn of lateral ventricle": "BPD, HC, and lateral ventricle",
    # Cervical + other
    "hc cervical length": "HC and cervical length",
    "cervical canal length hc bpd": "Cervical length, HC, and BPD",
    "cervical lenght": "Cervical length",
    # Amniotic fluid
    "amniotic fluid pocket": "Amniotic fluid pocket",
    "amniotic fluid pocket ac": "Amniotic fluid pocket and AC",
}

# Q5: Gestational Age (164 raw -> ~8 canonical ranges)
Q5_CANONICAL: Dict[str, str] = {
    # First trimester ranges
    "8-10 weeks": "8-13 weeks",
    "8-11 weeks": "8-13 weeks",
    "9-11 weeks": "8-13 weeks",
    "10-13 weeks": "8-13 weeks",
    "10-15 weeks": "8-13 weeks",
    "11-13 weeks": "8-13 weeks",
    "11-13w": "8-13 weeks",
    "11-13 w": "8-13 weeks",
    "11-13weeks": "8-13 weeks",
    "11-13 week": "8-13 weeks",
    "12 weeks": "8-13 weeks",
    "13 weeks": "8-13 weeks",
    "13w": "8-13 weeks",
    # Second trimester early
    "15-20 weeks": "15-20 weeks",
    "15-20weeks": "15-20 weeks",
    "12-18 weeks": "15-20 weeks",
    "13-18 weeks": "15-20 weeks",
    # Second trimester mid
    "18-22 weeks": "20-25 weeks",
    "18-25 weeks": "20-25 weeks",
    "17-25 weeks": "20-25 weeks",
    "17-27 weeks": "20-25 weeks",
    "20-25 weeks": "20-25 weeks",
    "20-25weeks": "20-25 weeks",
    "20-25n weeks": "20-25 weeks",
    "20-24 w": "20-25 weeks",
    "20-30 weeks": "20-25 weeks",
    "22-30 weeks": "25-30 weeks",
    # Third trimester early
    "25-30 weeks": "25-30 weeks",
    "25 -30 weeks": "25-30 weeks",
    "25-30` weeks": "25-30 weeks",
    "25-30": "25-30 weeks",
    # Third trimester mid
    "30-35 weeks": "30-35 weeks",
    # Third trimester late
    "35-38 weeks": "35-38 weeks",
    "35-38+ weeks": "35-38 weeks",
    # Edge: 7-10 weeks
    "7-10 weeks": "8-13 weeks",
    # Additional edge ranges
    "17-22 weeks": "20-25 weeks",
    "17-20 weeks": "15-20 weeks",
    "18-20 weeks": "20-25 weeks",
    "13-16 weeks": "15-20 weeks",
    "13-14 weeks": "15-20 weeks",
    "14 weeks": "15-20 weeks",
    "11-23 weeks": "20-25 weeks",
    "13 week": "8-13 weeks",
    "12 w": "8-13 weeks",
    "12w +6d": "8-13 weeks",
}

# Q6: Image Quality (226 raw -> ~4 canonical)
Q6_CANONICAL: Dict[str, str] = {
    # Good quality
    "good image quality": "Good image quality",
    "good quality of image": "Good image quality",
    "good quality": "Good image quality",
    "good quality image": "Good image quality",
    "good quality images": "Good image quality",
    "good image qualtiy": "Good image quality",
    "good image qualityq": "Good image quality",
    "good": "Good image quality",
    "high image quality": "Good image quality",
    "excellent quality of image": "Good image quality",
    "good despite some blurriness": "Good image quality",
    # Good quality with detailed assessment (drrehab)
    "quality apper good, correct for measure nt. magnification: image zoomed on fetal head and thorax. position :apper in neutral position. clarity: nasal bone and skin is distinct. differentiation :fetus is clearly distinguished from amnion": "Good image quality (detailed assessment)",
    "quality apper good. magnification: image zoomed on fetal head and thorax. position :apper in neutral position. clarity: nasal bone and skin is distinct. differentiation :fetus is clearly distinguished from amnion": "Good image quality (detailed assessment)",
    "quality apper good, correct for measure nt": "Good image quality",
    # Low quality
    "low image quality": "Low image quality",
    "low quality": "Low image quality",
    "low quality image": "Low image quality",
    "low image quality with image darkness": "Low image quality",
    "lowimage quality": "Low image quality",
    "low resolution image": "Low image quality",
    "low resolution details": "Low image quality",
    "low resoulation image": "Low image quality",
    "bad quality of image": "Low image quality",
    "low image quality with mid pulsation artefact": "Low image quality",
    "low image quality with marked image darkness": "Low image quality",
    "low image quality with low contrast and resolution details": "Low image quality",
    "low image quality with low resoulation": "Low image quality",
    "low quality image with low contrast and resolution details": "Low image quality",
    "low quality image with low resolution details": "Low image quality",
    "low image quality with marked darkness": "Low image quality",
    "low image quality with marked image darkenss": "Low image quality",
    "low resolution and contrast details": "Low image quality",
    "low image quality with loss contrast and resolution details": "Low image quality",
    "low contrast and resolution image": "Low image quality",
    "low image quality with low contrast and resolution": "Low image quality",
    "low image quality with more darkness and acoustic shadowing": "Low image quality",
    "low quality image with low resolution and contrast details": "Low image quality",
    "low quality image with low resolution details and mionr motion artifact": "Low image quality",
    "dark image with low contrast and resolution details": "Low image quality",
    "low image quality with loss contrast and resolution with minor motion artefact": "Low image quality",
    "low contrast and resolution details of image": "Low image quality",
    "significant acoustic shadowing": "Low image quality",
    "low image quality with low resolution details": "Low image quality",
    "low image quality with acoustic shadowing": "Low image quality",
    # Medium/acceptable quality
    "medium": "Medium image quality",
    "medium quality of image": "Medium image quality",
    "acceptable image quality": "Medium image quality",
    "acceptable quality image": "Medium image quality",
    "acceptable": "Medium image quality",
    "acceptable image": "Medium image quality",
    "acceptable image quality with low resolution details": "Medium image quality",
    # Good with delivery note (drrehab pattern)
    "good image quality with high probablity to normal delivery": "Good image quality",
    "good image quality with high probability to normal delivery": "Good image quality",
    # Low quality (additional variants)
    "low image quality with low resolution": "Low image quality",
    "bad image quality": "Low image quality",
    "low image quality with significant darkness in the image": "Low image quality",
    "low image quality with loss of contrast and resolution details": "Low image quality",
    "significant image darkness with loss of contrast and resolution": "Low image quality",
    "low image quality with motion artefact": "Low image quality",
    "low image quality with low contrast": "Low image quality",
    "low image quality with darkness": "Low image quality",
    # Good with notes
    "good with few acoustic shadowing": "Good image quality",
    "good but make the scan more bright": "Good image quality",
}

# Q7: Normality Assessment (208 raw -> ~8 canonical)
Q7_CANONICAL: Dict[str, str] = {
    # Normal (basic)
    "normal": "Normal",
    "normal finding": "Normal",
    "normal findings": "Normal",
    # Normal for gestational age
    "normal for the age": "Normal for gestational age",
    "normal for age": "Normal for gestational age",
    "normal for its age": "Normal for gestational age",
    "normal for the ag": "Normal for gestational age",
    "normal for the age e": "Normal for gestational age",
    # Within normal limits
    "within normal": "Within normal limits",
    "with normal": "Within normal limits",
    "within normal for the age": "Within normal limits",
    "within normal for age": "Within normal limits",
    "within normal according to age": "Within normal limits",
    "the visible structures are within normal": "Within normal limits",
    "the visible structure are within normal": "Within normal limits",
    "the visible structures are grossly within normal": "Within normal limits",
    "the visible structure are grossly within normal": "Within normal limits",
    "the visible structures within normal": "Within normal limits",
    "grossly within normal": "Within normal limits",
    # Normal closed cervix
    "normal closed cervix": "Normal closed cervix",
    "closed normal cervix": "Normal closed cervix",
    "closed normal cervicx": "Normal closed cervix",
    "normal closed cervix low lying placenta": "Normal closed cervix with low-lying placenta",
    # Normal with favorable delivery prognosis
    "normal for the age with high probablity of normal delivery": "Normal for gestational age, favorable prognosis",
    "normal for the age with high probablity of normal delivary": "Normal for gestational age, favorable prognosis",
    "normal for the age with high probablity to normal delivery": "Normal for gestational age, favorable prognosis",
    "normal for the age with high probablity to normal delivary": "Normal for gestational age, favorable prognosis",
    "normal for the age with high probability of normal delivery": "Normal for gestational age, favorable prognosis",
    "normal for the age with high probability to normal delivery": "Normal for gestational age, favorable prognosis",
    # Normal with guarded delivery prognosis
    "normal for the age with low probablity to normal delivary": "Normal for gestational age, guarded prognosis",
    "normal for the age with low probablity of normal delivary": "Normal for gestational age, guarded prognosis",
    "normal for the age with low probablity to normal delivery": "Normal for gestational age, guarded prognosis",
    "normal for the age with low probablity of normal delivery": "Normal for gestational age, guarded prognosis",
    "normal for the age with low probability of normal delivery": "Normal for gestational age, guarded prognosis",
    "normal for the age with low probability to normal delivery": "Normal for gestational age, guarded prognosis",
    # Normal intracranial anatomy (detailed CRL assessment)
    # Both pre- and post-spelling-correction forms needed (nucal -> nuchal)
    "normal bone presentation and well opacified normal facial profile normal intracranial anatomy no evidence of major brain anomaly no thick nucal translucency or cystic hygroma": "Normal intracranial anatomy",
    "normal bone presentation and well opacified normal facial profile normal intracranial anatomy no evidence of major brain anomaly no thick nuchal translucency or cystic hygroma": "Normal intracranial anatomy",
    "normal bone presentation andp well opacified normal facial profile normal intracranial anatomy no evidence of major brain anomaly no thick nucal translucency or cystic hygroma": "Normal intracranial anatomy",
    "normal bone presentation andp well opacified normal facial profile normal intracranial anatomy no evidence of major brain anomaly no thick nuchal translucency or cystic hygroma": "Normal intracranial anatomy",
    # Abnormal - NT thickening
    "mild abnormal thickening of nt with high risk of congenital anomalies": "Abnormal NT thickening (mild)",
    "mild abnormal thickening of nt with high risk for congenital anomalies": "Abnormal NT thickening (mild)",
    "mild abnormal thickening of nt with risk of congenital anomalies": "Abnormal NT thickening (mild)",
    "mid abnormal thickening of nt with high risk of congenital anomalies": "Abnormal NT thickening (mild)",
    "mild thickening of nt with high risk of congenital anomalies": "Abnormal NT thickening (mild)",
    "mild nt thickening": "Abnormal NT thickening (mild)",
    "mildly thickened nt with high risk of congenital anomalies": "Abnormal NT thickening (mild)",
    "mildly thickened nt": "Abnormal NT thickening (mild)",
    "abnormal thickening of nt with high risk of congenital anomalies": "Abnormal NT thickening",
    "abnormal thickening of nt": "Abnormal NT thickening",
    "abnormal thickening of nt with high risk of anomalies": "Abnormal NT thickening",
    "abnormally thickened nt with high risk of congenital anomalies": "Abnormal NT thickening",
    "abnormally thickened nt with high risk congenital anomalies": "Abnormal NT thickening",
    "marked abnormal thickening of nt with high risk of congenital anomalies": "Abnormal NT thickening (marked)",
    # Abnormal - structural
    "abnormal vsd": "Abnormal VSD",
    "abnormally dilated occipital horn of lateral ventricle": "Abnormal dilated lateral ventricle",
    "abnormal dilation of occipital horn of lateral ventricle": "Abnormal dilated lateral ventricle",
    # Abnormal - increased NT for further workup
    "increased nt for biochemistry for exclusion of down syndrome": "Increased NT (requires biochemistry)",
    "increased nuchal translucency thickness likely cystic hygroma": "Increased NT (cystic hygroma)",
    # Within normal (additional variants)
    "the visible structures grossly within normal": "Within normal limits",
    "the visible structure is within normal": "Within normal limits",
    "normal of the age": "Normal for gestational age",
    # Abnormal - other
    "abnormal low laying placenta partially open cervix": "Abnormal low-lying placenta",
    "deficient posterior element of vertebra likely spina bifida": "Abnormal spina bifida",
    "abnormally dilated occpital horn of lateral ventricle": "Abnormal dilated lateral ventricle",
    "occipital horn of lateral ventricle seem enlarged in diameter": "Abnormal dilated lateral ventricle",
    "posterior subchronoc hematoma is noted ** no thick nucal translucency or cystic hygroma": "Subchorionic hematoma",
    "posterior subchronoc hematoma is noted ** no thick nuchal translucency or cystic hygroma": "Subchorionic hematoma",
    # Abnormal NT (additional post-spelling-correction variants)
    "mildly thickened nt with high risk of congenital anomalies": "Abnormal NT thickening (mild)",
    "abnormal nt thickening with high risk of congenital anomalies": "Abnormal NT thickening",
    "abnormally thickened nt with high risk of congenital anomalies": "Abnormal NT thickening",
    "mild abnormal thickening of nt with high risk of congenital anomalies": "Abnormal NT thickening (mild)",
    "abnormal thickening of nt for high risk of congenital anomalies": "Abnormal NT thickening",
    # Abnormal - VSD
    "abnormal ventricular septal defect": "Abnormal VSD",
    # Cervix with placenta
    "closed normal cervix low laying placenta suggestive of palcenta previa": "Normal closed cervix with placenta previa",
}

# Q8: Clinical Recommendations (280 raw -> ~12 canonical)
Q8_CANONICAL: Dict[str, str] = {
    # Prenatal routine monitoring (dominant)
    "prenatal routine monitoring": "Prenatal routine monitoring",
    "prenatal montoring": "Prenatal routine monitoring",
    "prenatal routine moniotoring": "Prenatal routine monitoring",
    "prenatal rouitne monitoring": "Prenatal routine monitoring",
    "prenatal routine mointoring": "Prenatal routine monitoring",
    "prenatal monitoring": "Prenatal routine monitoring",
    "perinatal routine monitoring": "Prenatal routine monitoring",
    "routine perinatal monitoring": "Prenatal routine monitoring",
    "routine perinatal monitering": "Prenatal routine monitoring",
    "pernatal routine monitoring": "Prenatal routine monitoring",
    "routine monitoring": "Prenatal routine monitoring",
    "routine peri-natal monitoring": "Prenatal routine monitoring",
    # Prenatal monitoring + NT remeasurement
    "prenatal routine monitoring measures nt thickness in mid sagittal plane": "Prenatal monitoring with NT remeasurement",
    "prenatal routine monitoring measures nt in mid sagittal plane": "Prenatal monitoring with NT remeasurement",
    "prenatal routine monitoring measures nt thickness in mid sagittal view": "Prenatal monitoring with NT remeasurement",
    "prenatal routine monitoring. measures nt thickness in mid sagittal plane": "Prenatal monitoring with NT remeasurement",
    "prenatal rouitne monitoring measures nt thickness in mid sagittal plane": "Prenatal monitoring with NT remeasurement",
    "measures nt in mid sagittal view. prenatal routine monitoring": "Prenatal monitoring with NT remeasurement",
    # Remeasure NT + prenatal monitoring
    "remeasures of nt in mid sagittal plane prenatal routine monitoring": "Remeasure NT in mid-sagittal plane",
    "remeasure of nt in mid sagittal view prenatal routine monitoring": "Remeasure NT in mid-sagittal plane",
    "remeasure of nt in mid sagittal plane prenatal routine monitoring": "Remeasure NT in mid-sagittal plane",
    "remeasure nt in mid sagittal plane prenatal routine monitoring": "Remeasure NT in mid-sagittal plane",
    "remeasures of nt in mid sagittal place prenatal routine monitoring": "Remeasure NT in mid-sagittal plane",
    "remeasures nt in mid sagittal plane prenatal routine monitoring": "Remeasure NT in mid-sagittal plane",
    "remeasures s of nt in mid sagittal plane. prenatal routine monitoring": "Remeasure NT in mid-sagittal plane",
    "remeasures of nta in mid sagittal view prenatal routine monitoring": "Remeasure NT in mid-sagittal plane",
    "remeasures of nt in mid sagittal plane. prenatal routine monitoring": "Remeasure NT in mid-sagittal plane",
    # Prenatal monitoring + high power US
    "prenatal routine monitoring use high power us probe": "Prenatal monitoring (use high-power probe)",
    "prenatal routine monitoring use high power us probes": "Prenatal monitoring (use high-power probe)",
    "use high power us prenatal routine monitoring": "Prenatal monitoring (use high-power probe)",
    "use high power us probe. prenatal routine monitoring": "Prenatal monitoring (use high-power probe)",
    # Clinical assessment for delivery
    "clinical assessment to determine the mode of delivery": "Clinical assessment for delivery mode",
    "clinical assessment to determine mode of delivery": "Clinical assessment for delivery mode",
    "clinical assessment for determine mode of delivery": "Clinical assessment for delivery mode",
    # Follow up (general)
    "follow up": "Follow up",
    "follow up with antenatal care": "Follow up with antenatal care",
    "follow up after 1 month": "Follow up after 1 month",
    "follow up after 4 weeks": "Follow up after 1 month",
    "follow up use probe with high penetraion power": "Follow up (use high-power probe)",
    "follow up use probe with high penetration power": "Follow up (use high-power probe)",
    "recommend clinical assessment and follow up": "Clinical assessment and follow up",
    "recommend clinical correlation and follow up": "Clinical assessment and follow up",
    # Follow up + anatomy scan
    "follow up after 1 month anatomy scan at 18-22 weeks": "Follow up with anatomy scan at 18-22 weeks",
    "follow up, prenatal care anatomy scan at 18-22 weeks": "Follow up with anatomy scan at 18-22 weeks",
    "follow up anatomy scan at 18-22 weeks": "Follow up with anatomy scan at 18-22 weeks",
    "anatomy scan at 18-22 weeks": "Follow up with anatomy scan at 18-22 weeks",
    # Follow up + NT measurement
    "follow up with antenatal care measures nt thickness in mid sagittal plane": "Follow up with NT remeasurement",
    "follow up with antenatal care measures nt in mid sagittal plane": "Follow up with NT remeasurement",
    # Complete anomaly scan
    "complete anomaly scan follow up": "Complete anomaly scan and follow up",
    "complete anomaly scan. follow up": "Complete anomaly scan and follow up",
    "do complete anomaly scan and follow up": "Complete anomaly scan and follow up",
    "complete anomaly scan": "Complete anomaly scan and follow up",
    # Follow up (additional variants)
    "follow up after one month": "Follow up after 1 month",
    "follow up after month": "Follow up after 1 month",
    "folliw up after 1 month": "Follow up after 1 month",
    # Prenatal monitoring (additional variants)
    "prenatal routine monitoring increase image brightness": "Prenatal monitoring (use high-power probe)",
    "prenatal routine monitoring use high power us and scan at other plane": "Prenatal monitoring (use high-power probe)",
    "prenatal routine monitoring use high power us plane": "Prenatal monitoring (use high-power probe)",
    "prenatal routine monitoring. measure nt thickness in mid sagittal plane": "Prenatal monitoring with NT remeasurement",
    # Remeasure NT + anomaly scan
    "remeasures of nt in mid sagittal plane complete anomaly scan follow up": "Remeasure NT with complete anomaly scan",
    "remeasure of nt in mid sagittal plane. prenatal routine monitoring": "Remeasure NT in mid-sagittal plane",
    "remeasure of nt in mid sagittal view complete anomaly scan follow up": "Remeasure NT with complete anomaly scan",
    # Additional Q8 variants (count >= 2)
    "routine prenatal monitoring": "Prenatal routine monitoring",
    "complete routine monitoring": "Prenatal routine monitoring",
    "complete routine monitoring follow up": "Prenatal routine monitoring",
    "prenatal fetal monitoring": "Prenatal routine monitoring",
    "routine perinatal monitor": "Prenatal routine monitoring",
    "prenatal monitoring and follow up": "Prenatal routine monitoring",
    "follis up after 1 month": "Follow up after 1 month",
    "follow up with antenatal care use probe with high penetration power": "Follow up (use high-power probe)",
    "prenatal monitoring anatomy scan at 18-22 weeks": "Follow up with anatomy scan at 18-22 weeks",
    "prenatal monitoring follow up for nuchal translucency thickness": "Follow up with NT remeasurement",
    "prenatal monitoring biochemistry for exclude down syndrome": "Prenatal monitoring with biochemistry",
    "prenatal monitoring better visualization of fetal part by using high penetration power probe": "Prenatal monitoring (use high-power probe)",
    "complete anomalies scan follow up": "Complete anomaly scan and follow up",
    # Prenatal monitoring + NT measurement (many wording variants)
    "prenatal routine monitoring measures of nt thickness in mid sagittal plane": "Prenatal monitoring with NT remeasurement",
    "prenatal routine monitoring. measures of nt thickness in mid sagittal plane": "Prenatal monitoring with NT remeasurement",
    "prenatal routine monitoring measures nt thikcness in mid sagittal plane": "Prenatal monitoring with NT remeasurement",
    "prenatal routine monitoring measures nt thikness in mid sagittal plane": "Prenatal monitoring with NT remeasurement",
    "prenatal routine monitoring. measures nt thickenss in mid sagittal plane": "Prenatal monitoring with NT remeasurement",
    "prenatal routine monitoring. measures nt thickeness in mid sagittal plane": "Prenatal monitoring with NT remeasurement",
    "prenatal routine monitoring. measures nuchal translucency in mid sagittal plane": "Prenatal monitoring with NT remeasurement",
    "prenatal routine monitoring. measures nt thickness from mid sagittal plane": "Prenatal monitoring with NT remeasurement",
    "prenatal routine monitoring measuries nt thickness in mid sagittal plane": "Prenatal monitoring with NT remeasurement",
    "prenatal routine monitoring measure nt thickness in mid sagittal plane": "Prenatal monitoring with NT remeasurement",
    "prenatal routine monitoring. measure nt in mid sagittal plane": "Prenatal monitoring with NT remeasurement",
    "prenatal routine monitoring. measure nt in mid sagittal view": "Prenatal monitoring with NT remeasurement",
    # Prenatal monitoring + probe/brightness
    "prenatal routine monitoring use high power us probed": "Prenatal monitoring (use high-power probe)",
    "prenatal routine monitoringuse high power us probe": "Prenatal monitoring (use high-power probe)",
    "prenatal routine monitoring using high power us probe": "Prenatal monitoring (use high-power probe)",
    "prenatal routine monitoring increase image brightenss": "Prenatal monitoring (use high-power probe)",
    "increase brightness of us prenatal routine monitoring": "Prenatal monitoring (use high-power probe)",
    "use high power us probes and rescan": "Prenatal monitoring (use high-power probe)",
    "use high power us. look at other views prenatal routine monitoring": "Prenatal monitoring (use high-power probe)",
    "prenatal routine monitoring use high power us probe and increase image brightness": "Prenatal monitoring (use high-power probe)",
    "prenatal routine monitoring scan at other plane": "Prenatal monitoring (use high-power probe)",
    "prenatal routine monitoring scan in other plane": "Prenatal monitoring (use high-power probe)",
    # Remeasure NT + anomaly scan (additional)
    "remeasures nt in mid sagittal view. complete anomaly scan follow up": "Remeasure NT with complete anomaly scan",
    "remeasures of nta in mid sagittal view complete anomaly scan follow up": "Remeasure NT with complete anomaly scan",
    "remeasures nt in mid sagittal view. prenatal routine monitoring": "Remeasure NT in mid-sagittal plane",
    "remeasures of nt in mid-sagittal plane prenatal routine monitoring": "Remeasure NT in mid-sagittal plane",
    # Follow up + NT
    "follow up with antenatal care measures nt thikcness in mid sagittal plane": "Follow up with NT remeasurement",
}

# ---------------------------------------------------------------------------
# Q1: Anatomical Structures - Phrase-based set normalization
# ---------------------------------------------------------------------------
# Extract individual anatomical structures from free-text lists via greedy
# longest-match against a synonym dictionary, output as sorted canonical set.
#
# Pipeline:
#   1. Strip ordering prefix ("the order of structures from ... of screen:")
#   2. Strip noise phrases ("to the right/left of screen", etc.)
#   3. Normalize delimiters (commas, periods, ampersands, hyphens -> spaces)
#   4. Greedy longest-match scan against Q1_STRUCTURE_SYNONYMS
#   5. Collect canonical structure names into a set
#   6. Sort alphabetically, join with ", "

# Synonym dictionary: variant phrase (lowercase) -> canonical structure name
# Keys must be lowercase with hyphens replaced by spaces (text is normalized
# the same way before matching).  ~130 entries -> ~55 canonical names.
Q1_STRUCTURE_SYNONYMS: Dict[str, str] = {
    # --- Skull / Head ---
    "calvarial bones": "calvarial bones",
    "calvarial bone": "calvarial bones",
    "calvarium": "calvarial bones",
    "calvaria": "calvarial bones",
    "skull of the fetus": "fetal skull",
    "skull of fetus": "fetal skull",
    "fetal skull": "fetal skull",
    "skull": "fetal skull",

    # --- Brain ---
    "cavum septum pellucidum": "cavum septum pellucidum",
    "cavum septum pellucidi": "cavum septum pellucidum",
    "cavum septi pellucidi": "cavum septum pellucidum",
    "septum pellucidum": "cavum septum pellucidum",
    "intracranial structures": "intracranial structures",
    "intra cranial structures": "intracranial structures",
    "interhemispheric fissure": "interhemispheric fissure",
    "inter hemispheric fissure": "interhemispheric fissure",
    "interhemospheric fissure": "interhemispheric fissure",
    "superior sagittal sinus": "superior sagittal sinus",
    "frontoparietal lobe": "frontoparietal lobe",
    "frontoparietal lobes": "frontoparietal lobe",
    "parieto occipital lobe": "parieto-occipital lobe",
    "parietooccipital lobe": "parieto-occipital lobe",
    "parieto occipital lobes": "parieto-occipital lobe",
    "parieto occipital": "parieto-occipital lobe",
    "brain parenchyma": "brain parenchyma",
    "brain tissue": "brain parenchyma",
    "cerebral peduncle": "cerebral peduncle",
    "cerebral peduncles": "cerebral peduncle",
    "cerebral cortex": "cerebral cortex",
    "cerebellar vermis": "vermis",
    "occipital horn": "occipital horn",
    "occipital horns": "occipital horn",
    "lateral ventricle": "lateral ventricle",
    "lateral ventricles": "lateral ventricle",
    "third ventricle": "third ventricle",
    "3rd ventricle": "third ventricle",
    "fourth ventricle": "fourth ventricle",
    "4th ventricle": "fourth ventricle",
    "choroid plexus": "choroid plexus",
    "choroid plexuses": "choroid plexus",
    "cisterna magna": "cisterna magna",
    "falx cerebri": "falx cerebri",
    "cerebral falx": "falx cerebri",
    "parietal lobe": "parietal lobe",
    "parietal lobes": "parietal lobe",
    "frontal lobe": "frontal lobe",
    "frontal lobes": "frontal lobe",
    "occipital lobe": "occipital lobe",
    "temporal lobe": "temporal lobe",
    "cerebellum": "cerebellum",
    "thalami": "thalami",
    "thalamus": "thalami",
    "vermis": "vermis",
    "falx": "falx cerebri",
    "brain": "brain",
    "csp": "cavum septum pellucidum",

    # --- Face ---
    "facial profile": "facial profile",
    "fascial profile": "facial profile",   # common typo for "facial"
    "fascial bones": "facial bones",       # common typo for "facial"
    "facial bones": "facial bones",
    "face bones": "facial bones",
    "nasal bones": "nasal bone",
    "nasal bone": "nasal bone",
    "frontal bones": "frontal bone",
    "frontal bone": "frontal bone",
    "frontal ones": "frontal bone",         # drrehab typo
    "upper lip": "upper lip",
    "corpus callosum": "corpus callosum",
    "corpus callosal": "corpus callosum",
    "mandible": "mandible",
    "orbits": "orbit",
    "orbit": "orbit",
    "chin": "chin",
    "forehead": "forehead",
    "nose": "nose",
    "mouth": "mouth",

    # --- Neck / NT ---
    "nuchal translucency": "nuchal translucency",
    "nuchal fold": "nuchal fold",
    "head and neck": "head and neck",
    "skin line": "skin line",
    "neck": "neck",
    "nt": "nuchal translucency",

    # --- Thorax ---
    "four chambers of the heart": "four chambers of heart",
    "four chambers of heart": "four chambers of heart",
    "4 chambers of the heart": "four chambers of heart",
    "4 chambers of heart": "four chambers of heart",
    "four chamber of heart": "four chambers of heart",
    "4 chamber of heart": "four chambers of heart",
    "four heart chambers": "four chambers of heart",
    "cardiac chambers": "four chambers of heart",
    "heart chambers": "four chambers of heart",
    "four chambers": "four chambers of heart",
    "4 chambers": "four chambers of heart",
    "descending aorta": "descending aorta",
    "ascending aorta": "ascending aorta",
    "aortic arch": "aortic arch",
    "thoracic vertebrae": "thoracic vertebrae",
    "vertebral column": "vertebral column",
    "spinal column": "vertebral column",
    "rib cage": "rib cage",
    "chest cage": "rib cage",
    "diaphragm": "diaphragm",
    "chest": "chest",
    "lungs": "lungs",
    "lung": "lungs",
    "aorta": "aorta",
    "ribs": "ribs",
    "ivc": "IVC",
    "inferior vena cava": "IVC",
    "heart": "heart",

    # --- Abdomen ---
    "abdominal circumference": "abdominal circumference",
    "suprarenal gland": "suprarenal gland",
    "adrenal gland": "suprarenal gland",
    "abdominal wall": "abdominal wall",
    "fetal abdomen": "abdomen",
    "umbilical cord insertion": "cord insertion",
    "cord insertion": "cord insertion",
    "umbilical vein": "umbilical vein",
    "umbilical artery": "umbilical artery",
    "umbilical cord": "umbilical cord",
    "lumbar vertebrae": "lumbar vertebrae",
    "lumbar vertebra": "lumbar vertebrae",
    "gallbladder": "gallbladder",
    "gall bladder": "gallbladder",
    "liver": "liver",
    "stomach": "stomach",
    "spleen": "spleen",
    "kidneys": "kidney",
    "kidney": "kidney",
    "urinary bladder": "urinary bladder",
    "bowels": "bowel",
    "bowel": "bowel",
    "intestine": "bowel",
    "genitalia": "genitalia",
    "external genitalia": "genitalia",
    "abdomen": "abdomen",

    # --- Femur ---
    "skin and subcutaneous tissue": "skin and subcutaneous tissue",
    "diaphysis of the femur": "diaphysis of femur",
    "diaphysis of femur": "diaphysis of femur",
    "femoral diaphysis": "diaphysis of femur",
    "femoral shaft": "diaphysis of femur",
    "subcutaneous tissue": "subcutaneous tissue",
    "subcutaneous": "subcutaneous tissue",
    "thigh muscles": "thigh muscles",
    "thigh muscle": "thigh muscles",
    "soft tissues": "soft tissue",
    "soft tissue": "soft tissue",
    "diaphysis": "diaphysis of femur",
    "femur": "femur",
    "skin": "skin",

    # --- Cervix ---
    "endocervical canal": "endocervical canal",
    "cervical stroma": "cervical stroma",
    "cervical canal": "cervical canal",
    "internal os": "internal os",
    "external os": "external os",
    "cervix": "cervix",

    # --- Abbreviations ---
    "ac": "abdominal circumference",
    "bpd": "biparietal diameter",
    "hc": "head circumference",
    "crl": "crown-rump length",
    "nt": "nuchal translucency",
    "it": "intracranial translucency",
    "ivc": "inferior vena cava",
    "fmf": "frontomaxillary facial angle",

    # --- General ---
    "amniotic fluid": "amniotic fluid",
    "amniotic membrane": "amnion",
    "amniotic sac": "amniotic sac",
    "gestational sac": "gestational sac",
    "yolk sac": "yolk sac",
    "symphysis pubis": "symphysis pubis",
    "pubic symphysis": "symphysis pubis",
    "fetal spine": "fetal spine",
    "spine of fetus": "fetal spine",
    "fetal extremities": "extremities",
    "extremities": "extremities",
    "fetal limbs": "limbs",
    "fetal back": "fetal back",
    "fetal head": "fetal head",
    "head of fetus": "fetal head",
    "head of the fetus": "fetal head",
    "fetal body": "fetal body",
    "fundal placenta": "placenta",
    "anterior placenta": "placenta",
    "posterior placenta": "placenta",
    "fetal pole": "fetal pole",
    "placenta": "placenta",
    "amnion": "amnion",
    "pelvis": "pelvis",
    "limbs": "limbs",
    "limb": "limbs",
    "spine": "spine",
    "head": "fetal head",
    "vertebrae": "vertebrae",
    "vertebra": "vertebrae",
}

# Pre-sort synonyms by key length descending for greedy longest-match
_Q1_SYNONYMS_SORTED: List[Tuple[str, str]] = sorted(
    Q1_STRUCTURE_SYNONYMS.items(), key=lambda x: len(x[0]), reverse=True
)

# Pre-compile regex pattern for each synonym (word-boundary match)
_Q1_SYNONYM_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\b" + re.escape(syn) + r"\b"), canonical)
    for syn, canonical in _Q1_SYNONYMS_SORTED
]

# Regex to strip ordering prefix
# Matches: "the order of structures from above to below of screen:"
#          "the order of structures from right to left of the screen."
_Q1_PREFIX_PATTERN: re.Pattern = re.compile(
    r"(?:the\s+)?order\s+(?:of\s+)?structures?\s+"
    r"(?:from\s+\w+\s+to\s+\w+\s+)?"
    r"(?:of\s+)?(?:the\s+)?(?:screen|image)\s*[.:]*\s*",
    re.IGNORECASE,
)

# Noise phrases to strip (positional/directional metadata, not structures)
_Q1_NOISE_PATTERNS: List[re.Pattern] = [
    re.compile(r"to\s+the\s+(?:right|left)\s+(?:side\s+)?of\s+(?:the\s+)?screen", re.IGNORECASE),
    re.compile(r"(?:right|left)\s+side\s+of\s+(?:the\s+)?screen", re.IGNORECASE),
    re.compile(r"of\s+the\s+screen", re.IGNORECASE),
    re.compile(r"in\s+the\s+image", re.IGNORECASE),
    re.compile(r"from\s+(?:above|below|top|bottom|right|left)", re.IGNORECASE),
]

# Q1 has no direct canonical dict (all handled by _normalize_q1_structures)
Q1_CANONICAL: Dict[str, str] = {}


# Aggregate all per-question canonical mappings
CANONICAL_MAPPINGS: Dict[str, Dict[str, str]] = {
    "Q1: Anatomical Structures": Q1_CANONICAL,
    "Q2: Fetal Orientation": Q2_CANONICAL,
    "Q3: Imaging Plane": Q3_CANONICAL,
    "Q4: Biometric Measurements": Q4_CANONICAL,
    "Q5: Gestational Age": Q5_CANONICAL,
    "Q6: Image Quality": Q6_CANONICAL,
    "Q7: Normality Assessment": Q7_CANONICAL,
    "Q8: Clinical Recommendations": Q8_CANONICAL,
}

# Q5 regex patterns for gestational age normalization
# Pattern 1: specific GA like "12w+4d", "12w +4d", "13w"
_Q5_SPECIFIC_GA_PATTERN = re.compile(
    r"(\d{1,2})\s*w\s*(?:\+\s*(\d{1,2})\s*d)?", re.IGNORECASE
)
# Pattern 2: range like "20-25 weeks", "20 -25 weeks", "20- 25 weeks",
#             "20-25weeks", "20-25 w", "20-25", "20-25 weeksl", "20-25 weeeks"
_Q5_RANGE_PATTERN = re.compile(
    r"(\d{1,2})\s*-\s*(\d{1,2})\s*(?:weeks?|weeeks?|weeksl?|w)?",
    re.IGNORECASE,
)
# Pattern 3: single week like "14 weeks", "13 week", "12 w"
_Q5_SINGLE_WEEK_PATTERN = re.compile(
    r"(\d{1,2})\s*(?:weeks?|w)", re.IGNORECASE
)


def _weeks_to_bin(weeks: int) -> str:
    """Map a week number to a canonical gestational age range bin."""
    if weeks <= 13:
        return "8-13 weeks"
    elif weeks <= 20:
        return "15-20 weeks"
    elif weeks <= 25:
        return "20-25 weeks"
    elif weeks <= 30:
        return "25-30 weeks"
    elif weeks <= 35:
        return "30-35 weeks"
    else:
        return "35-38 weeks"


def _normalize_q5_regex(text: str) -> Optional[str]:
    """Normalize Q5 gestational age using regex patterns.

    Handles:
    - Specific GA: "12w+4d", "13w", "12w +4d"
    - Ranges: "20-25 weeks", "20- 25 weeks", "20-25weeksl", ". 20-25 weeks"
    - Single weeks: "14 weeks", "12 w"
    - Extra text after valid pattern is stripped
    """
    clean = text.strip()

    # Try specific GA (e.g., "12w+4d") -- search anywhere in text
    m = _Q5_SPECIFIC_GA_PATTERN.search(clean)
    if m:
        return _weeks_to_bin(int(m.group(1)))

    # Try range (e.g., "20-25 weeks") -- search anywhere in text
    m = _Q5_RANGE_PATTERN.search(clean)
    if m:
        low, high = int(m.group(1)), int(m.group(2))
        midpoint = (low + high) / 2
        # NOTE: int(midpoint) truncates toward zero, creating systematic bias toward lower GA bins.
        # Wide ranges (e.g., "11-23 weeks") collapse to a single bin. See issue #46.
        return _weeks_to_bin(int(midpoint))

    # Try single week (e.g., "14 weeks") -- search anywhere in text
    m = _Q5_SINGLE_WEEK_PATTERN.search(clean)
    if m:
        return _weeks_to_bin(int(m.group(1)))

    return None


# Q4 regex: "nuchal translucency measures X.X mm" -> "NT measurement (specific value)"
_Q4_NT_MEASURE_PATTERN = re.compile(
    r"^(?:nuchal translucency|nt)\s+(?:measures?|thickness)\s+\d",
    re.IGNORECASE,
)


def _normalize_q4_regex(text: str) -> Optional[str]:
    """Normalize Q4 NT measurement values via regex.

    Catches patterns like:
    - "Nuchal translucency measures 1.5 mm"
    - "Nuchal translucency measures 1.5mm"
    - "NT measures 2 mm"
    """
    if _Q4_NT_MEASURE_PATTERN.match(text.strip()):
        return "NT measurement (specific value)"
    return None


# Q8 regex: catch "prenatal routine monitoring" variants with additional instructions
_Q8_PRENATAL_MONITOR_PATTERN = re.compile(
    r"^(?:prenatal|perinatal|pernatal)\s+(?:routine\s+)?(?:monitoring|montoring|moniotoring|mointoring)",
    re.IGNORECASE,
)
_Q8_REMEASURE_NT_PATTERN = re.compile(
    r"remeasure", re.IGNORECASE,
)
_Q8_ANOMALY_SCAN_PATTERN = re.compile(
    r"(?:complete\s+)?anomal(?:y|ies)\s+scan", re.IGNORECASE,
)


def _normalize_q8_regex(text: str) -> Optional[str]:
    """Normalize Q8 clinical recommendations via regex.

    Catches the many variants of:
    - "prenatal routine monitoring + additional instruction"
    - "remeasure NT + prenatal monitoring"
    """
    clean = text.strip()

    has_prenatal = bool(_Q8_PRENATAL_MONITOR_PATTERN.search(clean))
    has_remeasure = bool(_Q8_REMEASURE_NT_PATTERN.search(clean))
    has_anomaly_scan = bool(_Q8_ANOMALY_SCAN_PATTERN.search(clean))
    has_nt = bool(re.search(r"\bnt\b|\bnuchal\b", clean, re.IGNORECASE))
    has_probe = bool(re.search(r"high\s+power|probe|bright", clean, re.IGNORECASE))

    if has_remeasure and has_anomaly_scan:
        return "Remeasure NT with complete anomaly scan"
    if has_remeasure and has_nt:
        return "Remeasure NT in mid-sagittal plane"
    if has_prenatal and has_nt and not has_probe:
        return "Prenatal monitoring with NT remeasurement"
    if has_prenatal and has_probe:
        return "Prenatal monitoring (use high-power probe)"
    if has_prenatal:
        return "Prenatal routine monitoring"

    return None


# ---------------------------------------------------------------------------
# Q2 regex: Fetal Orientation
# ---------------------------------------------------------------------------

def _q2_extract_laterality(text: str) -> Optional[str]:
    """Extract left/right laterality from screen-position phrases."""
    m = re.search(r"\b(left|right|lt|rt|light)\b", text)
    if m:
        raw = m.group(1).lower()
        return "left" if raw in ("left", "lt") else "right"
    # Vertical positions (rare)
    m = re.search(r"\b(bottom|bottum|top|down|posterior|anterior)\b", text)
    if m:
        raw = m.group(1).lower()
        if raw in ("bottom", "bottum", "down"):
            return "bottom"
        return raw
    return None


def _q2_extract_presentation(text: str) -> Optional[str]:
    """Extract presentation type from Q2 text (post-spelling-correction)."""
    if re.search(r"\bcephalic\b|\bcranial\b", text):
        return "Cephalic"
    if re.search(r"\bbreech\b", text):
        return "Breech"
    if re.search(r"\btransverse\b", text):
        return "Transverse"
    if re.search(r"\blongitudinal\b", text):
        return "Longitudinal"
    return None


def _normalize_q2_regex(text: str) -> Optional[str]:
    """Normalize Q2 (Fetal Orientation) using regex pattern extraction.

    Handles structured orientation descriptions:
    - Anatomical views (femur, aorta)
    - Fetus not visible
    - Cephalic transverse plane + occiput direction
    - Fetal head/pole screen position
    - Sagittal plane with skull/head direction
    - Axial abdomen with vertebral column direction
    - Axial thorax with cardiac apex direction
    - Occiput-oriented descriptions (CRL views)
    - Presentation + cervix position (cervical folder)
    - Maternal side orientation (drrehab annotator)
    - Simple presentation-only
    """
    tl = text.strip().lower()

    # 1. Fetus not seen
    if re.search(r"fetus\s+(?:is\s+)?not\s+(?:seen|in\b)", tl):
        return "Fetus not seen"

    # 2. Longitudinal view of femur
    if "femur" in tl:
        return "Longitudinal view of femur"

    # 3. View of aorta
    if "aorta" in tl:
        m = re.search(r"(coronal|sagittal)", tl)
        if m:
            return f"{m.group(1).capitalize()} view of aorta"

    # 4. Maternal side orientation (drrehab: "Transverse, oriented the right maternal side")
    #    Must check before cervical patterns to avoid false matches.
    if "maternal" in tl or "material" in tl:
        pres = _q2_extract_presentation(tl)
        side = _q2_extract_laterality(tl)
        if pres and side and side in ("left", "right"):
            return f"{pres}, {side} maternal side"

    # 5. Cervical patterns (presentation + cervix/cx position)
    if re.search(r"\bcervi[xc]\b|\bcx\b", tl):
        # Extract cervix side from text AFTER the cervix keyword
        cx_m = re.search(r"\bcervi[xc]\b|\bcx\b", tl)
        cx_side = _q2_extract_laterality(tl[cx_m.start():]) or "right"
        pres = _q2_extract_presentation(tl)
        if pres:
            return f"{pres} presentation, cervix to {cx_side}"
        return f"Cervix to {cx_side}"

    # 6. Axial abdomen with vertebral column direction
    if "abdomen" in tl:
        region = "mid" if "mid" in tl else "upper"
        col_match = re.search(r"vertebr\w*\s+colum\w*(.*)", tl)
        if col_match:
            col_side = _q2_extract_laterality(col_match.group(1))
            if col_side:
                return f"Axial {region} abdomen, vertebral column to {col_side}"
        return f"Axial {region} abdomen"

    # 7. Thorax / chest / cardiac apex views
    if re.search(r"\bthorax\b|\bchest\b", tl) or re.search(
        r"cardiac\s+apex|heart\s+apex", tl
    ):
        side = _q2_extract_laterality(tl)
        if side and side in ("left", "right"):
            return f"Axial thorax, cardiac apex to {side}"

    # 8. Cephalic transverse plane + occiput (trans-thalamic/ventricular/cerebellar)
    #    "cephalic and transverse plane with occiput of skull to right side of screen"
    if "occiput" in tl and re.search(r"transverse\s+(?:plane|view|plan)\b", tl):
        side = _q2_extract_laterality(tl)
        if side and side in ("left", "right"):
            return f"Cephalic transverse, occiput to {side}"
        return "Cephalic transverse"

    # 9. Occiput/head-oriented descriptions (CRL-view patterns)
    #    "Breech Occiput is oriented to left side of screen"
    #    "Cephalic Head is oriented the right side of screen"
    if "occiput" in tl or re.search(r"head\s+is\s+orient", tl):
        pres = _q2_extract_presentation(tl)
        side = _q2_extract_laterality(tl)
        if pres and side and side in ("left", "right"):
            return f"{pres}, occiput to {side}"
        if pres:
            return f"{pres} presentation"

    # 10. Sagittal plane + skull/head/pole direction
    if re.search(r"sagittal\s+(?:plane|view)", tl):
        landmark = "fetal pole" if "pole" in tl else "skull"
        side = _q2_extract_laterality(tl)
        if side and side in ("left", "right"):
            return f"Sagittal plane, {landmark} to {side}"

    # 11. Fetal head/pole screen position (NT views)
    if re.search(r"fetal\s+(?:head|pole)|head\s+of\s+fetus", tl):
        what = "pole" if "pole" in tl else "head"
        side = _q2_extract_laterality(tl)
        if side and side in ("left", "right"):
            return f"Fetal {what} to {side}"

    # 12. Cephalic transverse/coronal (no occiput, no other details)
    if re.search(r"cephalic\s+(?:and\s+)?(?:transverse|coronal)", tl):
        return "Cephalic transverse"

    # 13. Simple presentation-only
    pres = _q2_extract_presentation(tl)
    if pres:
        if re.search(r"presentation|position", tl) or len(tl.split()) <= 2:
            return f"{pres} presentation"

    return None


# ---------------------------------------------------------------------------
# Q1: Anatomical structure extraction
# ---------------------------------------------------------------------------

def _normalize_q1_structures(text: str) -> Optional[str]:
    """Normalize Q1 (Anatomical Structures) via phrase-based set extraction.

    Pipeline:
      1. Strip ordering prefix ("the order of structures from ... of screen:")
      2. Strip noise phrases ("to the right/left of screen", etc.)
      3. Normalize delimiters (commas, periods, ampersands, hyphens -> spaces)
      4. Greedy longest-match scan against Q1_STRUCTURE_SYNONYMS
      5. Collect canonical structure names into a set
      6. Sort alphabetically, join with ", "

    Returns None if no known structures are found (value stays unchanged).
    """
    clean = text.strip().lower()

    # 1. Strip ordering prefix
    clean = _Q1_PREFIX_PATTERN.sub("", clean)

    # 2. Strip noise phrases
    for pattern in _Q1_NOISE_PATTERNS:
        clean = pattern.sub(" ", clean)

    # 3. Normalize delimiters and whitespace
    clean = re.sub(r"[,;.&()\-/]+", " ", clean)
    clean = re.sub(r"\s+", " ", clean).strip()

    if not clean:
        return None

    # 4. Greedy longest-match scan (patterns pre-sorted by length descending)
    found: set = set()
    remaining = clean
    for pattern, canonical in _Q1_SYNONYM_PATTERNS:
        if pattern.search(remaining):
            found.add(canonical)
            remaining = pattern.sub(" ", remaining)

    if not found:
        return None

    # 5-6. Sort and join
    return ", ".join(sorted(found))


# ---------------------------------------------------------------------------
# Data classes for change tracking
# ---------------------------------------------------------------------------

@dataclass
class NormalizationChange:
    """Record of a single normalization applied to one cell."""
    row_index: int
    column: str
    original: str
    normalized: str
    layer: str  # "basic", "spelling", "semantic"
    rule: str   # e.g., "collapse_whitespace", "probablity->probability", "canonical:good quality"


@dataclass
class NormalizationReport:
    """Aggregated statistics from a normalization run."""
    total_cells: int = 0
    cells_changed: int = 0
    changes_by_layer: Dict[str, int] = field(default_factory=lambda: {
        "basic": 0, "spelling": 0, "semantic": 0
    })
    changes_by_question: Dict[str, int] = field(default_factory=dict)
    unique_values_before: Dict[str, int] = field(default_factory=dict)
    unique_values_after: Dict[str, int] = field(default_factory=dict)
    unmapped_values: Dict[str, List[Tuple[str, int]]] = field(default_factory=dict)
    changelog: List[NormalizationChange] = field(default_factory=list)

    def summary(self) -> str:
        """Return a human-readable summary string."""
        lines = [
            f"Total cells processed: {self.total_cells:,}",
            f"Cells changed: {self.cells_changed:,} ({100*self.cells_changed/max(self.total_cells,1):.1f}%)",
            "",
            "Changes by layer:",
        ]
        for layer, count in self.changes_by_layer.items():
            lines.append(f"  {layer}: {count:,}")
        lines.append("")
        lines.append("Unique values before -> after:")
        for col in QUESTION_COLUMNS:
            before = self.unique_values_before.get(col, 0)
            after = self.unique_values_after.get(col, 0)
            pct = 100 * (1 - after / max(before, 1))
            changed = self.changes_by_question.get(col, 0)
            lines.append(f"  {col}: {before} -> {after} ({pct:.0f}% reduction, {changed:,} cells changed)")
        lines.append("")
        # Top unmapped values
        for col in QUESTION_COLUMNS:
            unmapped = self.unmapped_values.get(col, [])
            if unmapped:
                lines.append(f"Top unmapped for {col}:")
                for val, cnt in unmapped[:10]:
                    lines.append(f"    {cnt:>5}  |  {val}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main normalizer class
# ---------------------------------------------------------------------------

class AnnotationNormalizer:
    """
    Layered normalizer for fetal ultrasound annotation answers.

    Applies three normalization layers in sequence:
    1. Basic cleanup: whitespace collapse, trailing punctuation
    2. Spelling correction: fixes systematic misspellings
    3. Semantic unification: maps to canonical forms per question

    Each layer is independently toggleable for ablation studies.
    """

    def __init__(
        self,
        enable_basic: bool = True,
        enable_spelling: bool = True,
        enable_semantic: bool = True,
        spelling_corrections: Optional[Dict[str, str]] = None,
        canonical_mappings: Optional[Dict[str, Dict[str, str]]] = None,
    ):
        self.enable_basic = enable_basic
        self.enable_spelling = enable_spelling
        self.enable_semantic = enable_semantic

        if spelling_corrections is not None:
            self._spelling_patterns = [
                (re.compile(r"\b" + re.escape(w) + r"\b", re.IGNORECASE), c)
                for w, c in spelling_corrections.items()
            ]
        else:
            self._spelling_patterns = _SPELLING_PATTERNS

        self._canonical = canonical_mappings if canonical_mappings is not None else CANONICAL_MAPPINGS

    def normalize_text_basic(self, text: str) -> Tuple[str, List[str]]:
        """
        Layer 1: Basic text cleanup.
        - Strip leading/trailing whitespace
        - Collapse internal multiple spaces to single space
        - Remove trailing period(s)

        Preserves original casing.
        Returns (cleaned_text, list_of_rules_applied).
        """
        rules = []
        result = text

        # Strip
        stripped = result.strip()
        if stripped != result:
            rules.append("strip_whitespace")
            result = stripped

        # Collapse internal whitespace
        collapsed = re.sub(r"\s+", " ", result)
        if collapsed != result:
            rules.append("collapse_whitespace")
            result = collapsed

        # Remove trailing periods
        if result.endswith("."):
            result = result.rstrip(".")
            if result != text.strip():
                rules.append("strip_trailing_period")

        return result, rules

    def normalize_text_spelling(self, text: str) -> Tuple[str, List[str]]:
        """
        Layer 2: Fix systematic misspellings.
        Applies whole-word replacements using word boundaries.
        Case of the correction follows the dictionary (lowercase).

        Returns (corrected_text, list_of_corrections_applied).
        """
        rules = []
        result = text

        for pattern, correction in self._spelling_patterns:
            new_result = pattern.sub(correction, result)
            if new_result != result:
                rules.append(f"{pattern.pattern}->{correction}")
                result = new_result

        return result, rules

    def normalize_text_semantic(
        self, text: str, question_column: str
    ) -> Tuple[str, List[str]]:
        """
        Layer 3: Map to canonical form.
        Looks up the lowercased, whitespace-collapsed text in the
        canonical mapping for this question.

        Returns (canonical_form_or_original, list_of_mappings_applied).
        Falls back to the input text if no canonical mapping is found.
        """
        mapping = self._canonical.get(question_column)
        if mapping is None:
            return text, []

        # Prepare lookup key: lowercase, collapse whitespace, strip periods
        key = re.sub(r"\s+", " ", text.strip().lower()).rstrip(".")

        canonical = mapping.get(key)
        if canonical is not None:
            return canonical, [f"canonical:{key}"]

        # Special handling for Q1: anatomical structure extraction
        if question_column == "Q1: Anatomical Structures":
            result = _normalize_q1_structures(key)
            if result is not None:
                return result, [f"q1_structures:{key[:80]}"]

        # Special handling for Q2: fetal orientation regex
        if question_column == "Q2: Fetal Orientation":
            result = _normalize_q2_regex(key)
            if result is not None:
                return result, [f"q2_regex:{key}"]

        # Special handling for Q5: regex-based GA normalization
        if question_column == "Q5: Gestational Age":
            ga = _normalize_q5_regex(key)
            if ga is not None:
                return ga, [f"ga_regex:{key}"]

        # Special handling for Q4: NT measurement values
        if question_column == "Q4: Biometric Measurements":
            result = _normalize_q4_regex(key)
            if result is not None:
                return result, [f"q4_regex:{key}"]

        # Special handling for Q8: prenatal monitoring pattern
        if question_column == "Q8: Clinical Recommendations":
            result = _normalize_q8_regex(key)
            if result is not None:
                return result, [f"q8_regex:{key}"]

        return text, []

    def normalize_answer(
        self,
        text: str,
        question_column: str,
        row_index: int = -1,
    ) -> Tuple[str, List[NormalizationChange]]:
        """
        Apply all enabled normalization layers to a single answer.
        Returns (final_text, list_of_changes).
        """
        changes: List[NormalizationChange] = []
        current = text

        # Layer 1: Basic cleanup
        if self.enable_basic:
            cleaned, rules = self.normalize_text_basic(current)
            if cleaned != current:
                changes.append(NormalizationChange(
                    row_index=row_index, column=question_column,
                    original=current, normalized=cleaned,
                    layer="basic", rule="; ".join(rules),
                ))
                current = cleaned

        # Layer 2: Spelling correction
        if self.enable_spelling:
            corrected, rules = self.normalize_text_spelling(current)
            if corrected != current:
                changes.append(NormalizationChange(
                    row_index=row_index, column=question_column,
                    original=current, normalized=corrected,
                    layer="spelling", rule="; ".join(rules),
                ))
                current = corrected

        # Layer 3: Semantic unification
        if self.enable_semantic:
            canonical, rules = self.normalize_text_semantic(current, question_column)
            if canonical != current:
                changes.append(NormalizationChange(
                    row_index=row_index, column=question_column,
                    original=current, normalized=canonical,
                    layer="semantic", rule="; ".join(rules),
                ))
                current = canonical

        return current, changes

    def normalize_dataframe(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, NormalizationReport]:
        """
        Normalize all Q1-Q8 columns in a DataFrame.

        Returns a copy of the DataFrame with normalized values,
        plus a NormalizationReport with full statistics.

        The original DataFrame is NOT modified.
        """
        result_df = df.copy()
        report = NormalizationReport()

        for col in QUESTION_COLUMNS:
            if col not in result_df.columns:
                logger.warning("Column %s not found in DataFrame, skipping", col)
                continue

            # Track unique values before
            raw_values = result_df[col].astype(str)
            report.unique_values_before[col] = raw_values.nunique()
            report.changes_by_question[col] = 0

            for idx in result_df.index:
                raw = str(result_df.at[idx, col])
                if pd.isna(result_df.at[idx, col]) or raw == "nan":
                    continue

                report.total_cells += 1
                normalized, changes = self.normalize_answer(
                    raw, col, row_index=idx
                )

                if normalized != raw:
                    result_df.at[idx, col] = normalized
                    report.cells_changed += 1
                    report.changes_by_question[col] += 1
                    for change in changes:
                        report.changes_by_layer[change.layer] += 1
                    report.changelog.extend(changes)

            # Track unique values after
            report.unique_values_after[col] = result_df[col].astype(str).nunique()

            # Track unmapped values (values not changed by semantic layer)
            if col in self._canonical and self.enable_semantic:
                after_values = result_df[col].astype(str)
                # Canonical forms: dict values + regex-produced forms
                canonical_values = set(self._canonical[col].values())
                for change in report.changelog:
                    if change.column == col and change.layer == "semantic":
                        canonical_values.add(change.normalized)
                unmapped = after_values[~after_values.isin(canonical_values)]
                unmapped_counts = unmapped.value_counts().head(20)
                report.unmapped_values[col] = [
                    (val, int(cnt)) for val, cnt in unmapped_counts.items()
                ]

        return result_df, report

    def normalize_single(self, question_column: str, raw_text: str) -> str:
        """
        Convenience method: normalize a single answer string.
        Returns just the canonical text (no change tracking).
        For use in training/inference pipelines.
        """
        result, _ = self.normalize_answer(raw_text, question_column)
        return result
