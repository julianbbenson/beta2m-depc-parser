# β2m DEPC Covalent Labeling Pipeline

Computational pipeline for processing DEPC (diethylpyrocarbonate) footprinting LC-MS/MS
data on β-2-microglobulin (β2m).

---

## Background

β-2-microglobulin (β2m, UniProt P61769) is a 114-residue serum protein that forms amyloid
fibrils under physiological conditions in patients undergoing long-term hemodialysis
(Dialysis-Related Amyloidosis, DRA). Cu(II) ions accelerate fibril formation by binding to
histidine residues His13, His31, and His51 and remodeling the protein's native structure.

### What DEPC footprinting tells us

DEPC reacts covalently with solvent-exposed nucleophilic residues (His, Lys, Tyr, Ser, Thr,
Cys, and the protein N-terminus), adding a +72.0211 Da carbethoxyl group per modification.
By comparing labeling extents between a Cu(II)-free control and a Cu(II)-containing sample,
residues that become *less* labeled in the presence of Cu(II) indicate sites of metal binding
or Cu(II)-induced structural occlusion.

**Labeling extent** for a residue = (labeled peptide intensity) / (labeled + unlabeled
peptide intensity), extracted from MS1 chromatographic peak areas.

---

## What this pipeline does

```
Raw .mzML files
      │
      ▼
mzml_loader    — parse spectra from Orbitrap/ion trap instruments
      │
      ▼
peak_picker    — detect monoisotopic MS1 peaks, determine charge states
      │
      ▼
depc_hunter    — find +72.0211 Da shifts, confirm via MS2 fragment ions
      │
      ▼
sequence_mapper — assign modified residues to β2m positions 1-114
      │
      ▼
labeling_quantifier — compute per-residue labeling extents (mean ± SD)
      │
      ▼
cu_protection  — compare +Cu vs -Cu; flag protected residues (t-test + Bonferroni)
      │
      ▼
array_formatter — numpy arrays (114,) or (114, N_conditions) → .npy / .csv
```

Output arrays are directly ingestible by the `beta2m-deep-aggregation` PyTorch CNN for
predicting aggregation kinetics.

---

## Installation

```bash
# Recommended: create a dedicated conda environment
conda env create -f environment.yml
conda activate depc-parser

# Or with pip:
pip install -r requirements.txt
```

**Python 3.10+ required.**

---

## Quick start

### 1. Place your .mzML files in `data/raw/`

Name files so that the condition is identifiable:
- No-Cu(II) controls: include `no_cu` in the filename (e.g., `b2m_no_cu_rep1.mzML`)
- Cu(II) samples: include `cu` in the filename (e.g., `b2m_cu_50uM_rep1.mzML`)

### 2. Run the pipeline

```bash
python -m src.pipeline \
    --input  data/raw/ \
    --output data/arrays/ \
    --mode   high_res        # or low_res, or auto
```

**Mode selection:**
- `high_res` — Orbitrap instruments (±0.02 Da mass tolerance)
- `low_res`  — Ion trap instruments (±0.5 Da)
- `auto`     — reads instrument metadata from mzML and chooses automatically

### 3. Outputs

| File | Description |
|------|-------------|
| `data/arrays/<stem>.npy` | 1D numpy array (114,) per file, float32 labeling extents |
| `data/arrays/<stem>.csv` | CSV with residue position, amino acid, susceptibility flag, and labeling extent |
| `data/arrays/all_conditions.npy` | 2D array (114, N\_conditions) |
| `data/arrays/labeling_extents.csv` | Full labeling extent table, all conditions |
| `data/arrays/cu_protection.csv` | Protection scores, raw and Bonferroni p-values, per residue |
| `data/arrays/cu_protection_scores.npy` | Protection score array (114,) |

Array indexing: position 1 (N-terminus Ile) → array index 0. Values are labeling extents
in [0, 1]; non-susceptible residues are set to 0.0.

---

## Testing with synthetic data

If no real .mzML files are available, generate a realistic synthetic dataset:

```bash
python scripts/generate_synthetic.py \
    --output data/raw/ \
    --conditions no_cu cu
```

This simulates tryptic LC-MS/MS data with:
- His residues: ~60% labeling extent without Cu, ~20% with Cu (protection effect)
- Lys residues: ~45% (no Cu), ~42% (Cu) — minimal protection
- Tyr/Ser/Thr/Cys: ~20-30% (both conditions)
- 5% Gaussian noise on all intensities

Then run the pipeline as above.

---

## Replicate design recommendations

The Cu(II) protection analysis uses a two-tailed Welch's t-test with Bonferroni correction
(114 comparisons). For adequate statistical power:

- **Minimum**: 3 independent LC-MS/MS runs per condition
- **Recommended**: 4-5 biological replicates
- Name files consistently: `b2m_no_cu_rep1.mzML`, `b2m_no_cu_rep2.mzML`, etc.

The pipeline aggregates all `no_cu`-pattern files into one condition and all `cu`-pattern
files into another. Use `--no-cu-pattern` and `--cu-pattern` flags to change the matching
strings.

---

## Running tests

```bash
python -m pytest tests/ -v
```

101 tests covering constants, DEPC mass matching, fragment ion calculation, sequence mapping,
labeling quantification, and output array formatting.

---

## Expected biological results

After processing real experimental data, the Cu(II) protection analysis should show:

- **His13, His31, His51**: protection scores > 0.20, Bonferroni p < 0.05
  (these are the primary Cu(II) binding histidines)
- **His78**: possible partial protection (secondary coordination)
- **Lys, Tyr, Ser, Thr**: protection scores near zero (no Cu(II) interaction)

The pipeline logs a warning if none of His13/31/51 show significant protection — this is a
built-in biological sanity check to catch swapped condition labels or data quality issues.

---

## Sequence reference

β2m mature form (114 residues, UniProt P61769):

```
Position:  1         11        21        31        41
           IQRTPKIQVY SRHPAENGKS NFLNCYVSGF HPSDIEVDLL KNGERIEKVE
           51        61        71        81        91
           HSDLSFSKDW SFYLLYYTEF TPTELKPHQN LVFQNLSSTP NVKVEELSST
           101       111
           HPFPFDLNLN PKKK
```

DEPC-susceptible residues: **H** (His), **K** (Lys), **Y** (Tyr), **S** (Ser), **T** (Thr),
**C** (Cys), and the N-terminal amine.

---

## Citation / acknowledgments

Developed in support of research into Cu(II)-driven β2m amyloid formation in
Dialysis-Related Amyloidosis. If you use this pipeline in published work, please
cite the relevant DEPC footprinting and β2m structural literature.
