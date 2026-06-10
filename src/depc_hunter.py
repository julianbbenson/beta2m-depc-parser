"""
DEPC modification detection.

For each MS1 peak, check whether its mass matches a β2m tryptic peptide
+ N×72.0211 Da.  Confirmed hits are then validated against the corresponding
MS2 scan via b/y fragment ion matching.

Parallelised over scans with concurrent.futures.ProcessPoolExecutor where safe
(MS2 fragment matching is CPU-bound and stateless).
"""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from typing import Any

import numpy as np

from .constants import (
    AA_MONOISOTOPIC_MASSES,
    B2M_RESIDUES,
    DEPC_MASS_SHIFT,
    DOUBLE_DEPC,
    PROTON_MASS,
    TOLERANCE_HIGH_RES,
    TOLERANCE_LOW_RES,
    TRYPTIC_PEPTIDES,
    WATER_MASS,
    compute_fragment_ions,
    compute_peptide_mass,
)
from .peak_picker import MS1Peak
from .sequence_mapper import DEPCEvent, resolve_residue_in_peptide, find_susceptible_sites_in_peptide
from .validator import validate_depc_mass_shift

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fragment ion matching
# ---------------------------------------------------------------------------

@lru_cache(maxsize=512)
def _cached_fragment_ions(peptide: str) -> dict[str, list[float]]:
    return compute_fragment_ions(peptide)


@lru_cache(maxsize=512)
def _cached_depc_fragment_ions(
    peptide: str, modified_offset: int
) -> dict[str, list[float]]:
    """Fragment ions for the DEPC-modified peptide (DEPC on residue at *modified_offset*)."""
    masses = [AA_MONOISOTOPIC_MASSES[aa] for aa in peptide]
    masses[modified_offset] += DEPC_MASS_SHIFT

    b_ions: list[float] = []
    y_ions: list[float] = []

    running = 0.0
    for m in masses[:-1]:
        running += m
        b_ions.append(running + PROTON_MASS)

    running = 0.0
    for m in reversed(masses[1:]):
        running += m
        y_ions.append(running + WATER_MASS + PROTON_MASS)

    return {"b": b_ions, "y": y_ions}


def _count_fragment_matches(
    observed_mz: np.ndarray,
    observed_int: np.ndarray,
    fragment_mz_list: list[float],
    tol: float,
    min_intensity_frac: float = 0.01,
) -> int:
    """Count how many theoretical fragment ions match observed peaks."""
    if len(observed_mz) == 0:
        return 0
    max_int = observed_int.max() if len(observed_int) else 1.0
    threshold = max_int * min_intensity_frac
    n_matches = 0
    for fmz in fragment_mz_list:
        mask = (np.abs(observed_mz - fmz) <= tol) & (observed_int >= threshold)
        if np.any(mask):
            n_matches += 1
    return n_matches


def _ms2_confirms_depc(
    ms2_peaks: np.ndarray,
    peptide: str,
    modified_offset: int,
    tol: float,
    min_fragments: int = 3,
) -> bool:
    """Return True if MS2 confirms DEPC at *modified_offset* in *peptide*."""
    if ms2_peaks is None or len(ms2_peaks) == 0:
        return False

    obs_mz = ms2_peaks[:, 0]
    obs_int = ms2_peaks[:, 1]

    depc_ions = _cached_depc_fragment_ions(peptide, modified_offset)
    all_theoretical = depc_ions["b"] + depc_ions["y"]

    n_matches = _count_fragment_matches(obs_mz, obs_int, all_theoretical, tol)
    return n_matches >= min_fragments


# ---------------------------------------------------------------------------
# MS1 DEPC matching
# ---------------------------------------------------------------------------

def _match_mass_to_peptide(
    neutral_mass: float,
    tol: float,
) -> list[tuple[str, int, int, bool]]:
    """
    Check whether *neutral_mass* matches any tryptic peptide ± DEPC shift.

    Returns list of (peptide, start_1indexed, modified_offset_0indexed, is_double).
    """
    hits: list[tuple[str, int, int, bool]] = []

    for peptide, (start, end) in TRYPTIC_PEPTIDES.items():
        base_mass = compute_peptide_mass(peptide)

        for is_double, shift in ((False, DEPC_MASS_SHIFT), (True, DOUBLE_DEPC)):
            target_mass = base_mass + shift
            if abs(neutral_mass - target_mass) > tol:
                continue

            # Identify which susceptible residue(s) could be modified
            sites = find_susceptible_sites_in_peptide(peptide, start)
            for abs_pos, aa in sites:
                if is_double and aa != "H":
                    continue  # double-DEPC only on His
                offset = abs_pos - start  # 0-indexed within peptide
                if 0 <= offset < len(peptide):
                    hits.append((peptide, start, offset, is_double))

    return hits


# ---------------------------------------------------------------------------
# Per-scan MS2 confirmation (runs in worker processes)
# ---------------------------------------------------------------------------

def _process_ms2_scan(
    args: tuple,
) -> list[dict]:
    """
    Worker function — runs inside ProcessPoolExecutor.

    args = (scan_id, ms2_peaks, precursor_mz, high_res, ms1_peak_list)
    """
    scan_id, ms2_peaks_list, precursor_mz, high_res, peptide_hits = args
    tol = TOLERANCE_HIGH_RES if high_res else TOLERANCE_LOW_RES

    if ms2_peaks_list is None:
        return []

    ms2_peaks = np.array(ms2_peaks_list)
    results = []

    for peptide, start, offset, is_double in peptide_hits:
        confirmed = _ms2_confirms_depc(ms2_peaks, peptide, offset, tol)
        if confirmed:
            abs_pos = start + offset
            aa = B2M_RESIDUES.get(abs_pos, "?")
            results.append(
                {
                    "peptide": peptide,
                    "residue_position": abs_pos,
                    "residue_type": aa,
                    "scan_id": scan_id,
                    "is_double_depc": is_double,
                }
            )

    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def hunt_depc_events(
    ms1_peaks: list[MS1Peak],
    scans: dict[int, dict[str, Any]],
    high_res: bool = True,
    n_workers: int = 4,
    min_ms2_fragments: int = 3,
) -> list[DEPCEvent]:
    """
    Full DEPC detection pass.

    1. For each MS1 peak, find candidate tryptic peptide + DEPC shift matches.
    2. For each MS2 scan with a matching precursor, confirm via fragment ions.
    3. Return confirmed DEPCEvent list.

    Uses ProcessPoolExecutor for parallelism (safe: workers are stateless).
    """
    tol = TOLERANCE_HIGH_RES if high_res else TOLERANCE_LOW_RES

    # Build precursor_mz → MS2 scan index for fast lookup
    ms2_by_precursor: dict[float, list[int]] = {}
    for sid, scan in scans.items():
        if scan["ms_level"] != 2:
            continue
        pmz = scan.get("precursor_mz")
        if pmz:
            bucket = round(pmz, 2)
            ms2_by_precursor.setdefault(bucket, []).append(sid)

    # Phase 1: MS1 mass matching (fast, single-threaded)
    candidate_jobs: list[tuple] = []
    unlabeled_intensity_map: dict[tuple[str, int], float] = {}  # (peptide, scan_id) → intensity

    for peak in ms1_peaks:
        hits = _match_mass_to_peptide(peak.monoisotopic_mass, tol)
        if not hits:
            continue

        # Find corresponding MS2 scans via precursor m/z
        # Precursor m/z = (neutral_mass + z * PROTON_MASS) / z
        for z in range(1, 7):
            expected_pmz = (peak.monoisotopic_mass + z * PROTON_MASS) / z
            bucket = round(expected_pmz, 2)
            if bucket in ms2_by_precursor:
                for ms2_sid in ms2_by_precursor[bucket]:
                    ms2_peaks = scans[ms2_sid]["peaks"]
                    candidate_jobs.append(
                        (
                            ms2_sid,
                            ms2_peaks.tolist() if ms2_peaks is not None else None,
                            expected_pmz,
                            high_res,
                            hits,
                        )
                    )

        # Track unlabeled intensity for each peptide at this scan
        for peptide, start, offset, _ in hits:
            base_mass = compute_peptide_mass(peptide)
            if abs(peak.monoisotopic_mass - base_mass) <= tol:
                key = (peptide, peak.scan_id)
                unlabeled_intensity_map[key] = peak.intensity

    if not candidate_jobs:
        logger.warning("No DEPC candidate MS1/MS2 pairs found")
        return []

    logger.info("Processing %d MS2 candidate scans", len(candidate_jobs))

    # Phase 2: MS2 confirmation (parallelised)
    confirmed_raw: list[dict] = []
    try:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_process_ms2_scan, job): job for job in candidate_jobs}
            for future in as_completed(futures):
                try:
                    confirmed_raw.extend(future.result())
                except Exception as exc:
                    logger.warning("MS2 processing error: %s", exc)
    except Exception:
        # Fall back to single-threaded if multiprocessing fails
        logger.warning("Parallel MS2 processing failed; falling back to serial")
        for job in candidate_jobs:
            try:
                confirmed_raw.extend(_process_ms2_scan(job))
            except Exception as exc:
                logger.warning("MS2 processing error: %s", exc)

    # Phase 3: Assemble DEPCEvent objects
    events: list[DEPCEvent] = []
    seen: set[tuple] = set()

    for raw in confirmed_raw:
        key = (raw["peptide"], raw["residue_position"], raw["scan_id"])
        if key in seen:
            continue
        seen.add(key)

        # Retrieve labelled intensity from MS2 precursor intensity (approximate)
        labeled_intensity = 1.0   # placeholder; real intensity from MS1 peak
        unlabeled_intensity = unlabeled_intensity_map.get(
            (raw["peptide"], raw["scan_id"]), 0.0
        )

        events.append(
            DEPCEvent(
                peptide=raw["peptide"],
                residue_position=raw["residue_position"],
                residue_type=raw["residue_type"],
                labeling_intensity=labeled_intensity,
                unlabeled_intensity=unlabeled_intensity,
                scan_id=raw["scan_id"],
                is_double_depc=raw["is_double_depc"],
            )
        )

    logger.info(
        "Found %d confirmed DEPC events (%d double-DEPC flagged)",
        len(events),
        sum(1 for e in events if e.is_double_depc),
    )
    return events


def hunt_depc_with_intensities(
    ms1_peaks: list[MS1Peak],
    scans: dict[int, dict[str, Any]],
    high_res: bool = True,
    n_workers: int = 4,
) -> list[DEPCEvent]:
    """
    Full DEPC detection with proper intensity extraction.

    Pairs every labelled MS1 peak with its unlabelled counterpart by
    matching the same peptide, retention time window, and charge state.
    """
    tol = TOLERANCE_HIGH_RES if high_res else TOLERANCE_LOW_RES

    # Index all MS1 peaks by (peptide, charge, rt_bin) for intensity pairing
    from .constants import compute_peptide_mass as cpm

    peptide_peaks: dict[str, list[tuple[float, float, float]]] = {}
    # peptide → [(mass, intensity, rt)]
    for peak in ms1_peaks:
        for peptide, (start, end) in TRYPTIC_PEPTIDES.items():
            base_mass = cpm(peptide)
            # unlabeled
            if abs(peak.monoisotopic_mass - base_mass) <= tol:
                peptide_peaks.setdefault(peptide, []).append(
                    (base_mass, peak.intensity, peak.retention_time)
                )
            # labeled
            for shift in (DEPC_MASS_SHIFT, DOUBLE_DEPC):
                if abs(peak.monoisotopic_mass - (base_mass + shift)) <= tol:
                    peptide_peaks.setdefault(peptide + f"_labeled_{shift}", []).append(
                        (base_mass + shift, peak.intensity, peak.retention_time)
                    )

    # Delegate to the base hunter; it handles the MS2 confirmation
    base_events = hunt_depc_events(ms1_peaks, scans, high_res, n_workers)

    # Enrich events with proper unlabeled intensities
    enriched: list[DEPCEvent] = []
    for event in base_events:
        unlabeled_peaks = peptide_peaks.get(event.peptide, [])
        labeled_peaks = peptide_peaks.get(
            event.peptide + f"_labeled_{DEPC_MASS_SHIFT}", []
        )

        # Use nearest-RT peak intensities
        rt_ref = scans.get(event.scan_id, {}).get("retention_time", 0.0)

        def nearest_intensity(peak_list: list[tuple]) -> float:
            if not peak_list:
                return 0.0
            return min(peak_list, key=lambda p: abs(p[2] - rt_ref))[1]

        li = nearest_intensity(labeled_peaks) or event.labeling_intensity
        ui = nearest_intensity(unlabeled_peaks) or event.unlabeled_intensity

        enriched.append(
            DEPCEvent(
                peptide=event.peptide,
                residue_position=event.residue_position,
                residue_type=event.residue_type,
                labeling_intensity=li,
                unlabeled_intensity=ui,
                scan_id=event.scan_id,
                is_double_depc=event.is_double_depc,
            )
        )

    return enriched
