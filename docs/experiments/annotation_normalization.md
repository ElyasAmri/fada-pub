# Annotation Normalization

## Motivation

Analysis of the master annotations file (`data/Fetal Ultrasound Annotations Normalized.xlsx`, 18,936 rows) revealed extreme answer fragmentation across the 8 clinical questions (Q1-Q8). The dataset contained 6,988 unique answer values where approximately 120 canonical forms should logically exist.

This fragmentation directly harms VLM training quality: the model learns to associate the same clinical concept with hundreds of superficially different strings, diluting gradient signal and inflating the effective vocabulary.

### Root Causes

| Cause                   | Scope                     | Example                                              |
| ----------------------- | ------------------------- | ---------------------------------------------------- |
| Extra whitespace        | 19-33% of rows            | `"good  image quality"` (double space)               |
| Capitalization          | Pervasive                 | `"Normal"` vs `"NORMAL"` vs `"normal"`               |
| Trailing periods        | ~5% of rows               | `"good image quality."`                              |
| Systematic misspellings | 10,616 cells, 33 patterns | `"probablity"`, `"symphsis"`, `"veiw"`               |
| Word order variation    | ~8%                       | `"normal closed cervix"` vs `"closed normal cervix"` |
| Synonyms                | Pervasive                 | `"within normal"` vs `"normal for the age"`          |

### Annotator Characteristics

Two annotators contributed with non-overlapping folder assignments:

| Annotator | Rows           | Folders                                 | Style                             |
| --------- | -------------- | --------------------------------------- | --------------------------------- |
| drshalal  | 15,682 (82.8%) | 12 folders                              | Shorter, consistent phrasing      |
| drrehab   | 3,254 (17.2%)  | 3 folders (CRL-View, NT-View, Cervical) | Longer, more detailed assessments |

Inter-annotator reliability cannot be computed (only 1 shared image).

## Methodology

Three normalization layers applied in sequence. Each layer is independently toggleable for ablation studies.

### Layer 1: Basic Text Cleanup

- Strip leading/trailing whitespace
- Collapse internal multiple spaces to single space
- Remove trailing periods

This layer preserves original casing and does not alter clinical meaning. Applied to all 8 questions.

**Cells affected**: 44,016 (29.1% of total)

### Layer 2: Spelling Correction

Whole-word replacement using regex word boundaries (`\b`), case-insensitive. Prevents substring corruption (e.g., "normal" inside "abnormal" is not affected).

33 systematic misspelling patterns identified and corrected:

| Misspelling   | Correction   | Occurrences |
| ------------- | ------------ | ----------- |
| probablity    | probability  | 1,588       |
| symphsis      | symphysis    | 1,354       |
| montoring     | monitoring   | 1,104       |
| vertebrea     | vertebrae    | 985         |
| calverial     | calvarial    | 819         |
| moniotoring   | monitoring   | 774         |
| veiw          | view         | 735         |
| delivary      | delivery     | 648         |
| longituidinal | longitudinal | 546         |
| longituidnal  | longitudinal | 402         |
| palne         | plane        | 401         |
| arota         | aorta        | 296         |
| cephalaic     | cephalic     | 162         |
| chammber      | chamber      | 104         |
| amnitoic      | amniotic     | 87          |
| rouitne       | routine      | ~100        |
| longtuidinal  | longitudinal | 77          |
| thoaracic     | thoracic     | ~75         |
| transvers     | transverse   | ~75         |
| cervicx       | cervix       | 72          |
| qualtiy       | quality      | 64          |
| abnormall     | abnormal     | 51          |
| penetraion    | penetration  | ~44         |
| centrlized    | centralized  | ~29         |
| viwe          | view         | 22          |
| pernatal      | perinatal    | 20          |
| mointoring    | monitoring   | 13          |
| measrured     | measured     | 12          |
| resoulation   | resolution   | ~10         |
| lengh         | length       | ~8          |
| monitering    | monitoring   | 8           |
| folliw        | follow       | 4           |
| nirmal        | normal       | 2           |

**Cells affected**: 10,616

### Layer 3: Semantic Unification

Per-question canonical mapping dictionaries that collapse synonyms and variants to standard forms. Applied to all 8 questions. Q1 uses phrase-based set extraction (greedy longest-match against a synonym dictionary). Q2 uses regex-based pattern extraction to decompose structured orientation descriptions into canonical components. Q3-Q8 use dictionary lookups with regex fallbacks for specific patterns.

**Cells affected**: 124,722

## Per-Question Results

### Q1: Anatomical Structures

- **Before**: 4,682 unique values
- **After**: 1,363 unique values (71% reduction, 18,935 cells changed)
- **Layers applied**: Basic cleanup + spelling + phrase-based set normalization
- **Method**: Greedy longest-match extraction against a synonym dictionary (~160 entries mapping to ~65 canonical structure names). Raw Q1 text is pre-processed by stripping ordering prefixes ("the order of structures from above to below of screen:"), removing positional noise phrases, and normalizing delimiters. Matched structures are collected into a sorted set and joined with ", ". Word order variation, punctuation differences, and subset variation are all eliminated. 16 new Q1-specific spelling corrections added to the global layer (cerberi/cerebri, umblical/umbilical, pareito/parieto, etc.).
- **Canonical structures by region**: Skull (fetal skull, calvarial bones), Brain (thalami, third ventricle, lateral ventricle, cerebral cortex, cerebral peduncle, cerebellum, cisterna magna, cavum septum pellucidum, falx cerebri, vermis, parieto-occipital lobe, frontoparietal lobe, interhemispheric fissure, brain parenchyma, choroid plexus, superior sagittal sinus, intracranial structures), Face (facial profile, facial bones, nasal bone, frontal bone, orbit, nose, mouth, mandible, upper lip, corpus callosum), Neck/NT (nuchal translucency, nuchal fold, head and neck, neck), Thorax (four chambers of heart, lungs, aorta, IVC, vertebrae, diaphragm, rib cage, chest), Abdomen (liver, umbilical vein, stomach, spleen, kidney, lumbar vertebrae, umbilical cord, gallbladder, suprarenal gland, abdomen), Femur (diaphysis of femur, skin and subcutaneous tissue, thigh muscles, soft tissue), Cervix (internal os, endocervical canal, external os, cervical canal), General (amniotic fluid, amnion, amniotic sac, placenta, fetal head, fetal spine, symphysis pubis, pelvis, limbs, extremities).
- **Per-folder reduction**: Public_Symphysis 1->1, Aorta 2->2, Thorax 26->11, Non_standard_NT 59->17, Cervix 98->23, Standard_NT 125->22, Abdomen 555->111, Femur 354->85, NT-View 109->67, Cervical 170->96, Trans-ventricular 498->168, Trans-thalamic 1235->240, Trans-cerebellum 533->246, CRL-View 917->352.

### Q2: Fetal Orientation

- **Before**: 838 unique values
- **After**: 73 unique values (91% reduction)
- **Layers applied**: Basic cleanup + spelling + regex-based semantic normalization
- **Canonical forms**: Cephalic/Breech/Transverse presentation, Cephalic transverse (occiput to left/right), Fetal head to left/right, Sagittal plane (skull/fetal pole to left/right), Axial upper/mid abdomen (vertebral column to left/right/bottom/top), Axial thorax (cardiac apex to left/right), Presentation + occiput direction, Presentation + cervix position, Presentation + maternal side orientation, Longitudinal view of femur, Coronal/Sagittal view of aorta, Fetus not seen.
- **Method**: Dictionary for common exact matches plus regex pattern extraction that decomposes structured descriptions into presentation type, anatomical view, and laterality. Q2-specific spelling corrections (breach/breech, cephalic variants, screen typos) added to global spelling layer. 18 new spelling entries.

### Q3: Imaging Plane

- **Before**: 229 unique values
- **After**: 92 unique values (60% reduction)
- **Canonical forms**: Sagittal, Mid-sagittal, Para-sagittal, Trans-thalamic view, Trans-cerebellar view, Trans-ventricular view, Trans-supraventricular view, Transverse trans-abdominal plane, Transverse 4-chamber heart view, Longitudinal view of femur, Longitudinal view of cervix, Coronal view of aorta, Sagittal view of aorta, Axial plane of head, Mid-sagittal view of head and neck, Mid-sagittal view of CRL, Sagittal profile variants, and combined cervical planes.

### Q4: Biometric Measurements

- **Before**: 361 unique values
- **After**: 128 unique values (65% reduction)
- **Canonical forms**: NT thickness, NT not measurable (not mid-sagittal / not centered), NT measurement (specific value), CRL, CRL and NT, CRL/NT/nasal bone length, AC, BPD and HC, Femur length, Cervical length, Aortic transverse diameter, Angle of progression, Cardiac chamber dimensions, Cerebellar and cisterna magna, Cerebellar/cisterna magna/nuchal fold, Lateral ventricle dimensions, and combined measurements.
- **Method**: Dictionary mappings plus regex for "Nuchal translucency measures X.X mm" patterns (any specific NT measurement value).

### Q5: Gestational Age

- **Before**: 164 unique values
- **After**: 15 unique values (91% reduction)
- **Canonical ranges**: 8-13 weeks, 15-20 weeks, 20-25 weeks, 25-30 weeks, 30-35 weeks, 35-38 weeks.
- **Method**: Comprehensive regex normalization handles range patterns ("20-25 weeks"), specific GA ("12w+4d"), and single-week values ("14 weeks"). Catches typos like "weeksl", "weeeks", leading periods, misplaced spaces around hyphens. All valid patterns are mapped to the nearest canonical bin by midpoint.
- **Remaining**: 9 count-1 values that are genuine data entry errors (e.g., a timestamp, "11-13 month", "18-200 weeks").

### Q6: Image Quality

- **Before**: 226 unique values
- **After**: 107 unique values (53% reduction)
- **Canonical forms**: Good image quality, Low image quality, Medium image quality, Good image quality (detailed assessment).

### Q7: Normality Assessment

- **Before**: 208 unique values
- **After**: 105 unique values (50% reduction)
- **Canonical forms**: Normal, Normal for gestational age, Within normal limits, Normal closed cervix, Normal for gestational age (favorable prognosis), Normal for gestational age (guarded prognosis), Normal intracranial anatomy, Abnormal NT thickening, Abnormal NT thickening (mild), Abnormal NT thickening (marked), Abnormal VSD, Abnormal dilated lateral ventricle, Increased NT (requires biochemistry), Subchorionic hematoma.

### Q8: Clinical Recommendations

- **Before**: 280 unique values
- **After**: 101 unique values (64% reduction)
- **Canonical forms**: Prenatal routine monitoring, Prenatal monitoring with NT remeasurement, Prenatal monitoring (use high-power probe), Remeasure NT in mid-sagittal plane, Remeasure NT with complete anomaly scan, Clinical assessment for delivery mode, Follow up, Follow up with antenatal care, Follow up after 1 month, Follow up with anatomy scan at 18-22 weeks, Complete anomaly scan and follow up, Prenatal monitoring with biochemistry.
- **Method**: Dictionary mappings plus regex fallback that detects prenatal monitoring, NT remeasurement, anomaly scan, and probe/brightness keywords to classify remaining variants.

## Aggregate Statistics

| Metric                     | Value           |
| -------------------------- | --------------- |
| Total cells processed      | 151,488         |
| Cells changed              | 126,588 (83.6%) |
| Layer 1 (basic) changes    | 44,016          |
| Layer 2 (spelling) changes | 15,608          |
| Layer 3 (semantic) changes | 124,722         |
| Total unique values before | 6,988           |
| Total unique values after  | 1,984           |
| Overall reduction          | 71.6%           |

Q1 was the last question to receive semantic normalization. With phrase-based set extraction, Q1 dropped from 4,682 to 1,363 unique values (71% reduction). The remaining Q1 variation represents legitimate differences in structure sets visible across different anatomical views. For Q2-Q8 combined, unique values dropped from 2,306 to 621 (73% reduction). Q2 and Q5 achieved the highest reduction (91%) through regex-based normalization, followed by Q1 (71%), Q4 (65%), and Q8 (64%).

## Unmapped Values

Values that remain unmapped after all three layers are logged in `data/normalization_changelog.json` under the `unmapped_values` key. These are primarily:

- Low-frequency variants (count 1-5) representing one-off typos or unique phrasing
- Clinically specific descriptions that do not fit a general canonical form
- Combined/compound answers spanning multiple clinical concepts

Review of the unmapped tail confirmed no high-frequency patterns were missed.

## Files

| File                                                | Description                                    |
| --------------------------------------------------- | ---------------------------------------------- |
| `src/data/normalize_annotations.py`                 | Core normalizer class and mapping dictionaries |
| `experiments/normalization/run_normalize.py`        | CLI runner script                              |
| `data/Fetal Ultrasound Annotations Normalized.xlsx` | Original annotations (unchanged)               |
| `data/Fetal Ultrasound Annotations Normalized.xlsx` | Normalized output                              |
| `data/normalization_changelog.json`                 | Per-cell change log with layer/rule tracking   |

## Reproducibility

```bash
# Full normalization
./venv/Scripts/python.exe experiments/normalization/run_normalize.py

# Dry run (report only)
./venv/Scripts/python.exe experiments/normalization/run_normalize.py --dry-run

# Ablation: basic cleanup only
./venv/Scripts/python.exe experiments/normalization/run_normalize.py --no-spelling --no-semantic

# Ablation: basic + spelling, no semantic
./venv/Scripts/python.exe experiments/normalization/run_normalize.py --no-semantic
```

## Future Work

1. **Annotator calibration**: Harmonize Q6 quality rating differences between drshalal (81.2% good) and drrehab (96.4% good)
2. **Automated discovery**: Run fuzzy matching to identify new misspelling patterns as annotations grow
3. **Q1 dictionary expansion**: Review remaining singleton Q1 values (668 count-1 combinations) to identify additional structure synonyms, particularly in CRL-View (352 unique) and Trans-cerebellum (246 unique) folders
