"""
MS1 peak picking: isotope envelope detection, charge state determination,
monoisotopic mass computation.

Filters: top-1000 peaks per scan by intensity, S/N > 3.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# C13 - C12 spacing used for isotope envelope detection
ISOTOPE_SPACING = 1.003355  # Da


@dataclass
class MS1Peak:
    monoisotopic_mass: float   # Da (neutral)
    charge: int
    retention_time: float      # minutes
    intensity: float
    mz: float                  # observed m/z of monoisotopic peak
    scan_id: int = 0


def _estimate_noise(intensities: np.ndarray) -> float:
    """Robust noise estimate: median of the lower 50% of intensities."""
    if len(intensities) == 0:
        return 1.0
    sorted_i = np.sort(intensities)
    return float(np.median(sorted_i[: max(1, len(sorted_i) // 2)])) or 1.0


def _find_isotope_envelope(
    mz_arr: np.ndarray,
    intensity_arr: np.ndarray,
    seed_idx: int,
    charge: int,
    n_isotopes: int = 5,
    tol_ppm: float = 20.0,
) -> float:
    """
    Walk up the isotope ladder from *seed_idx* and return the summed intensity.
    """
    seed_mz = mz_arr[seed_idx]
    step = ISOTOPE_SPACING / charge
    total_intensity = intensity_arr[seed_idx]

    for k in range(1, n_isotopes):
        target_mz = seed_mz + k * step
        tol_da = target_mz * tol_ppm * 1e-6
        mask = np.abs(mz_arr - target_mz) <= tol_da
        if not np.any(mask):
            break
        best = np.argmax(intensity_arr * mask)
        total_intensity += intensity_arr[best]

    return float(total_intensity)


def _mz_to_neutral_mass(mz: float, charge: int) -> float:
    from .constants import PROTON_MASS
    return mz * charge - charge * PROTON_MASS


def pick_ms1_peaks(
    scan_id: int,
    peaks: np.ndarray,
    retention_time: float,
    charge_range: tuple[int, int] = (1, 6),
    sn_threshold: float = 3.0,
    max_peaks: int = 1000,
    tol_ppm: float = 20.0,
) -> list[MS1Peak]:
    """
    Detect monoisotopic peaks in an MS1 scan.

    Returns a list of MS1Peak sorted by descending intensity.
    """
    if peaks is None or len(peaks) == 0:
        return []

    mz_arr = peaks[:, 0]
    int_arr = peaks[:, 1]

    # Filter to top-N by intensity
    if len(int_arr) > max_peaks:
        top_idx = np.argpartition(int_arr, -max_peaks)[-max_peaks:]
        mz_arr = mz_arr[top_idx]
        int_arr = int_arr[top_idx]

    noise = _estimate_noise(int_arr)
    sort_idx = np.argsort(mz_arr)
    mz_arr = mz_arr[sort_idx]
    int_arr = int_arr[sort_idx]

    results: list[MS1Peak] = []
    used = set()

    for seed_idx in np.argsort(int_arr)[::-1]:
        if seed_idx in used:
            continue
        if int_arr[seed_idx] / noise < sn_threshold:
            break  # remaining peaks below S/N

        seed_mz = mz_arr[seed_idx]
        best_charge: int | None = None
        best_score = -1.0

        for z in range(charge_range[0], charge_range[1] + 1):
            step = ISOTOPE_SPACING / z
            next_target = seed_mz + step
            tol_da = next_target * tol_ppm * 1e-6
            mask = np.abs(mz_arr - next_target) <= tol_da
            if np.any(mask):
                score = float(np.max(int_arr[mask]))
                if score > best_score:
                    best_score = score
                    best_charge = z

        if best_charge is None:
            best_charge = 1  # assume singly charged if no isotope found

        neutral_mass = _mz_to_neutral_mass(seed_mz, best_charge)
        env_intensity = _find_isotope_envelope(
            mz_arr, int_arr, seed_idx, best_charge, tol_ppm=tol_ppm
        )

        results.append(
            MS1Peak(
                monoisotopic_mass=neutral_mass,
                charge=best_charge,
                retention_time=retention_time,
                intensity=env_intensity,
                mz=seed_mz,
                scan_id=scan_id,
            )
        )
        used.add(seed_idx)

    results.sort(key=lambda p: p.intensity, reverse=True)
    return results[:max_peaks]


def pick_all_ms1_peaks(
    scans: dict[int, dict[str, Any]],
    **kwargs: Any,
) -> list[MS1Peak]:
    """Apply *pick_ms1_peaks* to every MS1 scan in *scans*."""
    all_peaks: list[MS1Peak] = []
    for scan_id, scan in scans.items():
        if scan["ms_level"] != 1:
            continue
        peaks = pick_ms1_peaks(
            scan_id=scan_id,
            peaks=scan["peaks"],
            retention_time=scan["retention_time"],
            **kwargs,
        )
        all_peaks.extend(peaks)
    logger.info("Picked %d MS1 peaks across all scans", len(all_peaks))
    return all_peaks
