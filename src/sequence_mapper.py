"""
Map confirmed DEPC-labelled peptides → absolute β2m residue positions.

Handles missed cleavages and overlapping peptides (averaging).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import NamedTuple

from .constants import (
    B2M_RESIDUES,
    SUSCEPTIBLE_RESIDUES,
    TRYPTIC_PEPTIDES,
)
from .validator import validate_peptide_in_sequence, validate_susceptible_residue

logger = logging.getLogger(__name__)


class DEPCEvent(NamedTuple):
    peptide: str
    residue_position: int       # 1-indexed in β2m
    residue_type: str           # single-letter AA
    labeling_intensity: float   # intensity of labelled form
    unlabeled_intensity: float  # intensity of unlabelled form
    scan_id: int
    is_double_depc: bool = False


def _labeling_extent(labeled: float, unlabeled: float) -> float:
    """(labeled) / (labeled + unlabeled), guarded against zero-division."""
    denom = labeled + unlabeled
    if denom <= 0.0:
        return 0.0
    return labeled / denom


def map_events_to_residues(
    events: list[DEPCEvent],
) -> dict[int, list[float]]:
    """
    Convert a list of DEPCEvent objects into per-residue labeling extents.

    When multiple peptides cover the same residue, all their extents are
    collected; the caller (labeling_quantifier) computes mean ± SD.

    Returns:
        {residue_position: [extent1, extent2, ...]}
    """
    residue_extents: dict[int, list[float]] = defaultdict(list)

    for event in events:
        if event.is_double_depc:
            logger.debug(
                "Skipping double-DEPC event at position %d (excluded from primary analysis)",
                event.residue_position,
            )
            continue

        try:
            validate_peptide_in_sequence(event.peptide, _find_peptide_start(event.peptide))
            validate_susceptible_residue(event.residue_position)
        except ValueError as exc:
            logger.warning("Skipping invalid event: %s", exc)
            continue

        extent = _labeling_extent(event.labeling_intensity, event.unlabeled_intensity)
        residue_extents[event.residue_position].append(extent)

    return dict(residue_extents)


def _find_peptide_start(peptide: str) -> int:
    """Return the 1-indexed start of *peptide* in the tryptic map (or 1 if unknown)."""
    entry = TRYPTIC_PEPTIDES.get(peptide)
    return entry[0] if entry else 1


def resolve_residue_in_peptide(
    peptide: str,
    peptide_start: int,
    modified_residue_offset: int,
) -> tuple[int, str]:
    """
    Convert a within-peptide offset (0-indexed) to an absolute β2m position.

    Returns (position_1indexed, amino_acid).
    """
    abs_pos = peptide_start + modified_residue_offset  # 1-indexed
    aa = B2M_RESIDUES.get(abs_pos, "?")
    return abs_pos, aa


def find_susceptible_sites_in_peptide(
    peptide: str, peptide_start: int
) -> list[tuple[int, str]]:
    """
    Return all (position, aa) pairs in *peptide* that are DEPC-susceptible.

    Includes N-terminus of the FULL protein (position 1) but not internal N-termini.
    """
    sites: list[tuple[int, str]] = []
    for offset, aa in enumerate(peptide):
        abs_pos = peptide_start + offset
        if aa in SUSCEPTIBLE_RESIDUES:
            sites.append((abs_pos, aa))
    # N-terminal amine of the intact protein is DEPC-susceptible regardless of residue type.
    # Only add if the first residue isn't already listed (it would be if it's in SUSCEPTIBLE_RESIDUES).
    if peptide_start == 1:
        first_pos = 1
        already_covered = any(s[0] == first_pos for s in sites)
        if not already_covered:
            sites.insert(0, (first_pos, "N-term"))
    return sites
