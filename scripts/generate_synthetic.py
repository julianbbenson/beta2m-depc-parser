"""
Generate synthetic .mzML files for testing the β2m DEPC pipeline.

Simulates a tryptic LC-MS/MS experiment with realistic DEPC labeling extents:
  - His residues:  ~0.60 extent (no Cu), ~0.20 extent (+Cu)
  - Lys residues:  ~0.45 extent (no Cu), ~0.40 extent (+Cu)
  - Tyr/Ser/Thr/Cys: ~0.25 extent (both conditions)

Adds 5% Gaussian noise to all intensities.
Outputs valid mzML files (base64-encoded binary peak arrays).

Usage:
    python scripts/generate_synthetic.py --output data/raw/ --conditions no_cu cu
"""

from __future__ import annotations

import argparse
import base64
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---- import pipeline constants ----
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.constants import (
    AA_MONOISOTOPIC_MASSES,
    B2M_SEQUENCE,
    DEPC_MASS_SHIFT,
    PROTON_MASS,
    SUSCEPTIBLE_RESIDUES,
    TRYPTIC_PEPTIDES,
    WATER_MASS,
    compute_fragment_ions,
    compute_peptide_mass,
)

RNG = np.random.default_rng(42)

# ── Realistic labeling extents per residue type ──────────────────────────────
LABELING_EXTENTS_NO_CU: dict[str, float] = {
    "H": 0.60, "K": 0.45, "Y": 0.25, "S": 0.20, "T": 0.22, "C": 0.30,
}
LABELING_EXTENTS_CU: dict[str, float] = {
    "H": 0.20, "K": 0.42, "Y": 0.24, "S": 0.19, "T": 0.21, "C": 0.29,
}
NOISE_FRACTION = 0.05
BASE_INTENSITY = 1e6


# ---------------------------------------------------------------------------
# mzML XML helpers
# ---------------------------------------------------------------------------

def _encode_array(arr: np.ndarray) -> str:
    """Encode array as little-endian 64-bit floats, base64-encoded (no compression)."""
    raw = np.asarray(arr, dtype="<f8").tobytes()
    return base64.b64encode(raw).decode()


def _noisy(intensity: float) -> float:
    return max(0.0, intensity * (1.0 + RNG.normal(0, NOISE_FRACTION)))


def _make_isotope_envelope(
    mono_mz: float, charge: int, intensity: float, n_isotopes: int = 4
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a realistic isotope envelope centred on *mono_mz*."""
    spacing = 1.003355 / charge
    mzs = [mono_mz + k * spacing for k in range(n_isotopes)]
    # Isotope intensity ratios (approximate binomial distribution)
    ratios = [1.0, 0.55, 0.18, 0.04]
    ints = [_noisy(intensity * r) for r in ratios[:n_isotopes]]
    return np.array(mzs), np.array(ints)


def _peptide_mz(mass: float, charge: int) -> float:
    return (mass + charge * PROTON_MASS) / charge


# ---------------------------------------------------------------------------
# Fragment ion generation for MS2
# ---------------------------------------------------------------------------

def _make_ms2_peaks(
    peptide: str,
    depc_offset: int | None,
    base_intensity: float = 5e4,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate b/y fragment ions, optionally with DEPC shift on one residue."""
    masses = [AA_MONOISOTOPIC_MASSES[aa] for aa in peptide]
    if depc_offset is not None:
        masses[depc_offset] += DEPC_MASS_SHIFT

    b_ions, y_ions = [], []
    running = 0.0
    for m in masses[:-1]:
        running += m
        b_ions.append(running + PROTON_MASS)
    running = 0.0
    for m in reversed(masses[1:]):
        running += m
        y_ions.append(running + WATER_MASS + PROTON_MASS)

    all_mz = b_ions + y_ions
    # Vary intensities — y-ions tend to be more intense for tryptic peptides
    all_int = (
        [_noisy(base_intensity * 0.6) for _ in b_ions]
        + [_noisy(base_intensity) for _ in y_ions]
    )
    idx = np.argsort(all_mz)
    return np.array(all_mz)[idx], np.array(all_int)[idx]


# ---------------------------------------------------------------------------
# Scan construction
# ---------------------------------------------------------------------------

def _ms1_scan_xml(scan_id: int, rt: float, peaks_xml: str) -> str:
    return f"""    <spectrum index="{scan_id}" id="scan={scan_id}" defaultArrayLength="0">
      <cvParam accession="MS:1000511" name="ms level" value="1"/>
      <cvParam accession="MS:1000579" name="MS1 spectrum"/>
      <scanList count="1">
        <scan>
          <cvParam accession="MS:1000016" name="scan start time" value="{rt:.4f}" unitAccession="UO:0000031" unitName="minute"/>
        </scan>
      </scanList>
{peaks_xml}
    </spectrum>
"""


def _ms2_scan_xml(
    scan_id: int,
    rt: float,
    precursor_mz: float,
    precursor_charge: int,
    peaks_xml: str,
) -> str:
    return f"""    <spectrum index="{scan_id}" id="scan={scan_id}" defaultArrayLength="0">
      <cvParam accession="MS:1000511" name="ms level" value="2"/>
      <cvParam accession="MS:1000580" name="MSn spectrum"/>
      <scanList count="1">
        <scan>
          <cvParam accession="MS:1000016" name="scan start time" value="{rt:.4f}" unitAccession="UO:0000031" unitName="minute"/>
        </scan>
      </scanList>
      <precursorList count="1">
        <precursor>
          <selectedIonList count="1">
            <selectedIon>
              <cvParam accession="MS:1000744" name="selected ion m/z" value="{precursor_mz:.6f}"/>
              <cvParam accession="MS:1000041" name="charge state" value="{precursor_charge}"/>
            </selectedIon>
          </selectedIonList>
        </precursor>
      </precursorList>
{peaks_xml}
    </spectrum>
"""


def _peaks_xml(mz_arr: np.ndarray, int_arr: np.ndarray) -> str:
    mz_b64 = _encode_array(mz_arr)
    int_b64 = _encode_array(int_arr)
    n = len(mz_arr)
    return f"""      <binaryDataArrayList count="2">
        <binaryDataArray encodedLength="{len(mz_b64)}">
          <cvParam cvRef="MS" accession="MS:1000514" name="m/z array" value=""/>
          <cvParam cvRef="MS" accession="MS:1000523" name="64-bit float" value=""/>
          <cvParam cvRef="MS" accession="MS:1000576" name="no compression" value=""/>
          <binary>{mz_b64}</binary>
        </binaryDataArray>
        <binaryDataArray encodedLength="{len(int_b64)}">
          <cvParam cvRef="MS" accession="MS:1000515" name="intensity array" value=""/>
          <cvParam cvRef="MS" accession="MS:1000523" name="64-bit float" value=""/>
          <cvParam cvRef="MS" accession="MS:1000576" name="no compression" value=""/>
          <binary>{int_b64}</binary>
        </binaryDataArray>
      </binaryDataArrayList>"""


# ---------------------------------------------------------------------------
# Synthetic experiment generation
# ---------------------------------------------------------------------------

def generate_synthetic_mzml(
    output_path: Path,
    condition: str = "no_cu",
    n_replicates: int = 1,
) -> None:
    """
    Write a synthetic mzML simulating one LC-MS/MS run of β2m + DEPC.

    *condition* should be 'no_cu' or 'cu' to select the appropriate extents.
    """
    extents_map = (
        LABELING_EXTENTS_NO_CU if "no_cu" in condition else LABELING_EXTENTS_CU
    )

    spectra_xml: list[str] = []
    scan_id = 1
    rt = 5.0  # start retention time (minutes)

    # Work through DEPC-susceptible residues in tryptic peptides
    covered_residues: set[int] = set()

    for peptide, (start, end) in sorted(TRYPTIC_PEPTIDES.items(), key=lambda x: x[1][0]):
        if len(peptide) < 6:
            continue  # skip very short peptides

        pep_mass = compute_peptide_mass(peptide)
        charge = 2 if pep_mass < 2000 else 3

        # Advance RT per peptide
        rt += RNG.uniform(0.2, 0.8)

        # Susceptible offsets in this peptide
        sus_offsets = [
            (i, aa)
            for i, aa in enumerate(peptide)
            if aa in SUSCEPTIBLE_RESIDUES
        ]

        for offset, aa in sus_offsets:
            abs_pos = start + offset
            if abs_pos in covered_residues:
                continue
            covered_residues.add(abs_pos)

            extent = extents_map.get(aa, 0.20)
            labeled_int = _noisy(BASE_INTENSITY * extent)
            unlabeled_int = _noisy(BASE_INTENSITY * (1.0 - extent))

            labeled_mass = pep_mass + DEPC_MASS_SHIFT
            labeled_mz = _peptide_mz(labeled_mass, charge)
            unlabeled_mz = _peptide_mz(pep_mass, charge)

            # MS1 scan — both labeled and unlabeled peaks present
            mz_l, int_l = _make_isotope_envelope(labeled_mz, charge, labeled_int)
            mz_u, int_u = _make_isotope_envelope(unlabeled_mz, charge, unlabeled_int)
            ms1_mz = np.concatenate([mz_u, mz_l])
            ms1_int = np.concatenate([int_u, int_l])
            idx = np.argsort(ms1_mz)

            spectra_xml.append(
                _ms1_scan_xml(
                    scan_id, rt, _peaks_xml(ms1_mz[idx], ms1_int[idx])
                )
            )
            ms1_scan_id = scan_id
            scan_id += 1

            # MS2 scan for LABELED peptide
            ms2_mz, ms2_int = _make_ms2_peaks(peptide, offset, base_intensity=5e4)
            spectra_xml.append(
                _ms2_scan_xml(
                    scan_id,
                    rt + 0.01,
                    labeled_mz,
                    charge,
                    _peaks_xml(ms2_mz, ms2_int),
                )
            )
            scan_id += 1

    n_spectra = len(spectra_xml)
    spectra_block = "\n".join(spectra_xml)

    mzml_content = f"""<?xml version="1.0" encoding="utf-8"?>
<mzML xmlns="http://psi.hupo.org/ms/mzml"
      xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
      xsi:schemaLocation="http://psi.hupo.org/ms/mzml http://psidev.info/files/ms/mzML/xsd/mzML1.1.0.xsd"
      version="1.1.0">
  <cvList count="2">
    <cv id="MS" fullName="Proteomics Standards Initiative Mass Spectrometry Ontology" version="4.1.30"
        URI="https://raw.githubusercontent.com/HUPO-PSI/psi-ms-CV/master/psi-ms.obo"/>
    <cv id="UO" fullName="Unit Ontology"
        URI="https://raw.githubusercontent.com/bio-ontology-research-group/unit-ontology/master/unit.obo"/>
  </cvList>
  <fileDescription>
    <fileContent>
      <cvParam cvRef="MS" accession="MS:1000580" name="MSn spectrum" value=""/>
    </fileContent>
  </fileDescription>
  <softwareList count="1">
    <software id="synthetic_generator" version="1.0">
      <cvParam cvRef="MS" accession="MS:1000799" name="custom unreleased software tool" value="beta2m-depc-synthetic"/>
    </software>
  </softwareList>
  <instrumentConfigurationList count="1">
    <instrumentConfiguration id="IC1">
      <cvParam cvRef="MS" accession="MS:1000556" name="LTQ Orbitrap XL" value=""/>
    </instrumentConfiguration>
  </instrumentConfigurationList>
  <dataProcessingList count="1">
    <dataProcessing id="DP1">
      <processingMethod order="0" softwareRef="synthetic_generator">
        <cvParam cvRef="MS" accession="MS:1000544" name="Conversion to mzML" value=""/>
      </processingMethod>
    </dataProcessing>
  </dataProcessingList>
  <run>
    <spectrumList count="{n_spectra}" defaultDataProcessingRef="DP1">
{spectra_block}
    </spectrumList>
  </run>
</mzML>
"""
    output_path.write_text(mzml_content, encoding="utf-8")
    logger.info(
        "Wrote %s  (%d spectra, %d unique residues covered)",
        output_path,
        n_spectra,
        len(covered_residues),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic β2m DEPC mzML files")
    parser.add_argument("--output", default="data/raw", help="Output directory")
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["no_cu", "cu"],
        help="Condition names (must contain 'no_cu' or 'cu')",
    )
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    for cond in args.conditions:
        out_path = out_dir / f"b2m_depc_{cond}.mzML"
        generate_synthetic_mzml(out_path, condition=cond)
        logger.info("Generated: %s", out_path)


if __name__ == "__main__":
    main()
