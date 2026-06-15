"""
Pipeline-wide QC and validation.

Call validate_* at each stage; raises ValueError on hard failures,
logs warnings on biological soft-checks.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np

from .constants import (
    B2M_LENGTH,
    B2M_RESIDUES,
    B2M_SEQUENCE,
    CU_BINDING_HIS,
    DEPC_MASS_SHIFT,
    SUSCEPTIBLE_RESIDUES,
    TOLERANCE_HIGH_RES,
    TOLERANCE_LOW_RES,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sequence integrity
# ---------------------------------------------------------------------------

def validate_residue_position(position: int, amino_acid: str) -> None:
    """Assert that position is valid in β2m and matches expected amino acid."""
    if position < 1 or position > B2M_LENGTH:
        raise ValueError(
            f"Residue position {position} out of range [1, {B2M_LENGTH}]"
        )
    expected = B2M_RESIDUES[position]
    if expected != amino_acid:
        raise ValueError(
            f"Position {position}: expected {expected}, got {amino_acid}"
        )


def validate_susceptible_residue(position: int) -> None:
    """Raise if the residue at *position* cannot react with DEPC."""
    aa = B2M_RESIDUES.get(position)
    if aa is None:
        raise ValueError(f"Position {position} not in β2m sequence")
    if aa not in SUSCEPTIBLE_RESIDUES:
        raise ValueError(
            f"Position {position} ({aa}) is not a DEPC-susceptible residue. "
            f"Susceptible: {sorted(SUSCEPTIBLE_RESIDUES)}"
        )


# ---------------------------------------------------------------------------
# Mass accuracy
# ---------------------------------------------------------------------------

def validate_depc_mass_shift(
    observed_shift: float,
    high_res: bool = True,
    allow_double: bool = False,
) -> str:
    """
    Confirm that *observed_shift* is a valid DEPC modification.

    Returns 'single' or 'double'; raises ValueError if neither match.
    """
    tol = TOLERANCE_HIGH_RES if high_res else TOLERANCE_LOW_RES
    if abs(observed_shift - DEPC_MASS_SHIFT) <= tol:
        return "single"
    if allow_double:
        from .constants import DOUBLE_DEPC
        if abs(observed_shift - DOUBLE_DEPC) <= tol:
            return "double"
    raise ValueError(
        f"Mass shift {observed_shift:.4f} Da not within ±{tol} Da of "
        f"DEPC (+{DEPC_MASS_SHIFT}) or double-DEPC (+{DOUBLE_DEPC if allow_double else 'N/A'})"
    )


# ---------------------------------------------------------------------------
# Labeling extents
# ---------------------------------------------------------------------------

def validate_labeling_extent(value: float, context: str = "") -> None:
    """Labeling extent must be in [0.0, 1.0]."""
    if not (0.0 <= value <= 1.0):
        raise ValueError(
            f"Labeling extent {value:.4f} out of [0, 1]{f' ({context})' if context else ''}"
        )


def validate_labeling_extent_array(
    extents: np.ndarray, label: str = ""
) -> None:
    """Vectorised check for a full array of labeling extents."""
    bad = np.where((extents < 0.0) | (extents > 1.0))[0]
    if len(bad):
        raise ValueError(
            f"{'Array ' + label + ': ' if label else ''}"
            f"{len(bad)} labeling extents outside [0,1] at indices: {bad[:10]}"
        )


# ---------------------------------------------------------------------------
# Replicate consistency
# ---------------------------------------------------------------------------

def validate_replicate_cv(
    values: list[float],
    position: int,
    cv_threshold: float = 0.30,
) -> bool:
    """
    Flag if coefficient of variation exceeds threshold.

    Returns True if within bounds, False (and logs warning) if not.
    Doesn't raise — noisy replicates are flagged, not fatal.
    """
    if len(values) < 2:
        return True
    arr = np.asarray(values, dtype=float)
    mean = arr.mean()
    if mean == 0.0:
        return True
    cv = arr.std(ddof=1) / mean
    if cv > cv_threshold:
        logger.warning(
            "Residue %d: CV=%.1f%% exceeds %.0f%% threshold (values: %s)",
            position,
            cv * 100,
            cv_threshold * 100,
            [f"{v:.3f}" for v in values],
        )
        return False
    return True


# ---------------------------------------------------------------------------
# Output array
# ---------------------------------------------------------------------------

def validate_output_array(arr: np.ndarray) -> None:
    """Check shape and value bounds of a final output array."""
    if arr.ndim != 1 or arr.shape[0] != B2M_LENGTH:
        raise ValueError(
            f"Output array shape {arr.shape} != ({B2M_LENGTH},)"
        )
    validate_labeling_extent_array(arr, label="output")


# ---------------------------------------------------------------------------
# Biological sanity check
# ---------------------------------------------------------------------------

def sanity_check_cu_protection(
    protection_scores: dict[int, float],
    p_values: dict[int, float],
    p_threshold: float = 0.05,
    score_threshold: float = 0.20,
) -> bool:
    """
    Warn (don't crash) if none of His13/His31/His51 show Cu(II) protection.

    Returns True if at least one CU_BINDING_HIS site is protected.
    """
    protected_his = [
        h for h in CU_BINDING_HIS
        if protection_scores.get(h, 0.0) > score_threshold
        and p_values.get(h, 1.0) < p_threshold
    ]
    if not protected_his:
        warnings.warn(
            "Biological sanity check FAILED: none of His13/His31/His51 show "
            "statistically significant Cu(II) protection "
            f"(score>{score_threshold}, p<{p_threshold}). "
            "Check your +Cu/-Cu condition labels and data quality.",
            UserWarning,
            stacklevel=2,
        )
        return False
    logger.info(
        "Biological sanity check PASSED: protected His sites = %s", protected_his
    )
    return True


# ---------------------------------------------------------------------------
# Peptide → sequence consistency
# ---------------------------------------------------------------------------

def validate_peptide_in_sequence(peptide: str, start: int) -> None:
    """Confirm that *peptide* actually occurs at *start* (1-indexed) in β2m."""
    s = start - 1  # 0-indexed
    expected = B2M_SEQUENCE[s : s + len(peptide)]
    if expected != peptide:
        raise ValueError(
            f"Peptide '{peptide}' does not match β2m sequence "
            f"at position {start}: found '{expected}'"
        )
