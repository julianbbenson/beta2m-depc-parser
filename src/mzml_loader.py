"""
Raw .mzML file ingestion via pymzml.

Auto-detects instrument resolution from file metadata (Orbitrap → high-res,
ion trap → low-res).  Returns a structured dict keyed by scan ID.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pymzml

logger = logging.getLogger(__name__)

# Terms that indicate high-resolution (Orbitrap / FT-based) instruments
_HIGH_RES_TERMS = frozenset(
    {
        "orbitrap",
        "fourier transform",
        "ft-icr",
        "fticr",
        "ftms",
        "q exactive",
        "exploris",
        "astral",
        "lumos",
    }
)
_LOW_RES_TERMS = frozenset(
    {"ion trap", "iontrap", "quadrupole", "linear trap", "ltq"}
)


def _detect_high_res(run: pymzml.run.Reader) -> bool:
    """Inspect mzML metadata to guess if the instrument is high-resolution."""
    try:
        info = run.info
        instrument_str = str(info).lower()
        if any(t in instrument_str for t in _HIGH_RES_TERMS):
            return True
        if any(t in instrument_str for t in _LOW_RES_TERMS):
            return False
    except Exception:
        pass
    warnings.warn(
        "Could not determine instrument resolution from mzML metadata. "
        "Defaulting to high-res (±0.02 Da) tolerance.",
        UserWarning,
        stacklevel=3,
    )
    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

ScanDict = dict[int, dict[str, Any]]


def load_mzml(filepath: str | Path, ms_level_filter: tuple[int, ...] = (1, 2)) -> tuple[ScanDict, bool]:
    """
    Parse an mzML file and return structured scan data plus a high_res flag.

    Returns:
        (scans, high_res)
        scans: {scan_id: {
            "ms_level": int,
            "precursor_mz": float | None,
            "precursor_charge": int | None,
            "retention_time": float,        # minutes
            "peaks": np.ndarray (shape N×2, columns mz/intensity)
        }}
        high_res: bool — True if Orbitrap-class instrument detected
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"mzML file not found: {filepath}")

    logger.info("Loading %s", filepath.name)

    # pymzml precision hints for MS1/MS2
    ms_precisions = {1: 5e-6, 2: 20e-6}
    run = pymzml.run.Reader(
        str(filepath),
        MS_precisions=ms_precisions,
        obo_version="4.1.30",
    )

    high_res = _detect_high_res(run)
    logger.info("Instrument resolution: %s", "high" if high_res else "low")

    scans: ScanDict = {}
    n_skipped = 0

    for spectrum in run:
        ms_level: int = spectrum.ms_level
        if ms_level not in ms_level_filter:
            continue

        try:
            peaks_raw = spectrum.peaks("raw")
        except Exception:
            peaks_raw = None

        if peaks_raw is None or len(peaks_raw) == 0:
            n_skipped += 1
            continue

        peaks = np.asarray(peaks_raw, dtype=np.float64)
        if peaks.ndim != 2 or peaks.shape[1] != 2:
            n_skipped += 1
            continue

        # Retention time
        try:
            rt = float(spectrum.scan_time_in_minutes())
        except Exception:
            rt = float("nan")

        # Precursor info (MS2 only)
        precursor_mz: float | None = None
        precursor_charge: int | None = None
        if ms_level == 2:
            try:
                prec = spectrum.selected_precursors
                if prec:
                    precursor_mz = float(prec[0].get("mz", 0.0))
                    precursor_charge = int(prec[0].get("charge", 0)) or None
            except Exception:
                pass

        scan_id = spectrum.ID
        scans[scan_id] = {
            "ms_level": ms_level,
            "precursor_mz": precursor_mz,
            "precursor_charge": precursor_charge,
            "retention_time": rt,
            "peaks": peaks,
        }

    logger.info(
        "Loaded %d scans (%d MS1, %d MS2); skipped %d empty",
        len(scans),
        sum(1 for s in scans.values() if s["ms_level"] == 1),
        sum(1 for s in scans.values() if s["ms_level"] == 2),
        n_skipped,
    )
    return scans, high_res


def get_ms2_scans(scans: ScanDict) -> ScanDict:
    return {sid: s for sid, s in scans.items() if s["ms_level"] == 2}


def get_ms1_scans(scans: ScanDict) -> ScanDict:
    return {sid: s for sid, s in scans.items() if s["ms_level"] == 1}
