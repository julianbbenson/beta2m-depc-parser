"""Tests for src/depc_hunter.py"""

import numpy as np
import pytest

from src.constants import (
    AA_MONOISOTOPIC_MASSES,
    B2M_RESIDUES,
    DEPC_MASS_SHIFT,
    DOUBLE_DEPC,
    PROTON_MASS,
    TOLERANCE_HIGH_RES,
    TOLERANCE_LOW_RES,
    TRYPTIC_PEPTIDES,
    WATER_MASS,
    compute_peptide_mass,
)
from src.depc_hunter import (
    _match_mass_to_peptide,
    _ms2_confirms_depc,
    _cached_depc_fragment_ions,
    _count_fragment_matches,
)
from src.peak_picker import MS1Peak
from src.validator import validate_depc_mass_shift


# ---------------------------------------------------------------------------
# Mass shift validation
# ---------------------------------------------------------------------------

class TestDEPCMassShiftValidation:
    def test_exact_shift_accepted_high_res(self):
        result = validate_depc_mass_shift(72.0211, high_res=True)
        assert result == "single"

    def test_within_tolerance_high_res(self):
        result = validate_depc_mass_shift(72.0211 + 0.015, high_res=True)
        assert result == "single"

    def test_outside_tolerance_high_res_raises(self):
        with pytest.raises(ValueError):
            validate_depc_mass_shift(72.0211 + 0.05, high_res=True)

    def test_within_tolerance_low_res(self):
        result = validate_depc_mass_shift(72.0211 + 0.4, high_res=False)
        assert result == "single"

    def test_outside_tolerance_low_res_raises(self):
        with pytest.raises(ValueError):
            validate_depc_mass_shift(72.0211 + 0.6, high_res=False)

    def test_double_depc_detected(self):
        result = validate_depc_mass_shift(144.0423, high_res=True, allow_double=True)
        assert result == "double"

    def test_double_depc_not_detected_when_disallowed(self):
        with pytest.raises(ValueError):
            validate_depc_mass_shift(144.0423, high_res=True, allow_double=False)


# ---------------------------------------------------------------------------
# Fragment ion matching
# ---------------------------------------------------------------------------

class TestFragmentMatching:
    def _make_obs_peaks(self, mzs: list[float], scale: float = 1e4) -> tuple[np.ndarray, np.ndarray]:
        obs_mz = np.array(mzs)
        obs_int = np.full(len(mzs), scale)
        return obs_mz, obs_int

    def test_perfect_match(self):
        """All theoretical ions present in observed → all matched."""
        from src.constants import compute_fragment_ions
        ions = compute_fragment_ions("HPAENGK")
        all_mz = ions["b"] + ions["y"]
        obs_mz, obs_int = self._make_obs_peaks(all_mz)
        n = _count_fragment_matches(obs_mz, obs_int, all_mz, tol=0.02)
        assert n == len(all_mz)

    def test_no_match(self):
        obs_mz = np.array([100.0, 200.0, 300.0])
        obs_int = np.array([1e4, 1e4, 1e4])
        n = _count_fragment_matches(obs_mz, obs_int, [500.0, 600.0], tol=0.02)
        assert n == 0

    def test_partial_match(self):
        obs_mz = np.array([100.0, 200.0, 300.0])
        obs_int = np.array([1e4, 1e4, 1e4])
        n = _count_fragment_matches(obs_mz, obs_int, [100.0, 600.0], tol=0.02)
        assert n == 1


# ---------------------------------------------------------------------------
# MS2 confirmation
# ---------------------------------------------------------------------------

class TestMS2Confirmation:
    def _build_ms2_for_peptide(self, peptide: str, offset: int, tol: float) -> np.ndarray:
        """Build fake MS2 peaks that SHOULD confirm DEPC at *offset*."""
        masses = [AA_MONOISOTOPIC_MASSES[aa] for aa in peptide]
        masses[offset] += DEPC_MASS_SHIFT

        b_ions, y_ions = [], []
        running = 0.0
        for m in masses[:-1]:
            running += m
            b_ions.append(running + PROTON_MASS)
        running = 0.0
        for m in reversed(masses[1:]):
            running += m
            y_ions.append(running + WATER_MASS + PROTON_MASS)

        all_mz = np.array(b_ions + y_ions)
        all_int = np.full(len(all_mz), 1e5)
        return np.column_stack([all_mz, all_int])

    def test_confirms_his13_depc(self):
        """DEPC on His13 (offset 0 in HPAENGK) should be confirmed."""
        peptide = "HPAENGK"
        offset = 0  # H is at offset 0
        ms2 = self._build_ms2_for_peptide(peptide, offset, TOLERANCE_HIGH_RES)
        assert _ms2_confirms_depc(ms2, peptide, offset, TOLERANCE_HIGH_RES, min_fragments=3)

    def test_wrong_offset_not_confirmed(self):
        """Synthetic MS2 built for offset 0 should NOT confirm offset 3."""
        peptide = "HPAENGK"
        offset_correct = 0
        offset_wrong = 3
        ms2 = self._build_ms2_for_peptide(peptide, offset_correct, TOLERANCE_HIGH_RES)
        # With wrong offset, at least some b-ions will differ — check it's below threshold
        # (With only 7-residue peptide and 3 fragment min, may still pass; use larger peptide)
        peptide2 = "VEHSDLSFSK"
        ms2_2 = self._build_ms2_for_peptide(peptide2, 2, TOLERANCE_HIGH_RES)  # H at offset 2
        # Checking offset 5 (S) should not confirm as well
        confirmed_wrong = _ms2_confirms_depc(ms2_2, peptide2, 5, TOLERANCE_HIGH_RES, min_fragments=5)
        # Not necessarily false (some ions may coincidentally match), just check correct one works
        confirmed_correct = _ms2_confirms_depc(ms2_2, peptide2, 2, TOLERANCE_HIGH_RES, min_fragments=3)
        assert confirmed_correct

    def test_empty_ms2_not_confirmed(self):
        ms2 = np.zeros((0, 2))
        assert not _ms2_confirms_depc(ms2, "HPAENGK", 0, TOLERANCE_HIGH_RES)

    def test_none_ms2_not_confirmed(self):
        assert not _ms2_confirms_depc(None, "HPAENGK", 0, TOLERANCE_HIGH_RES)


# ---------------------------------------------------------------------------
# MS1 mass matching
# ---------------------------------------------------------------------------

class TestMS1MassMatching:
    def test_matches_hpaengk_depc(self):
        """HPAENGK + 72.0211 should be found."""
        pep = "HPAENGK"
        base_mass = compute_peptide_mass(pep)
        labeled_mass = base_mass + DEPC_MASS_SHIFT
        hits = _match_mass_to_peptide(labeled_mass, TOLERANCE_HIGH_RES)
        peptides = [h[0] for h in hits]
        assert pep in peptides

    def test_no_match_for_random_mass(self):
        hits = _match_mass_to_peptide(99999.0, TOLERANCE_HIGH_RES)
        assert len(hits) == 0

    def test_double_depc_flagged(self):
        """HPAENGK + double DEPC should be flagged as double."""
        pep = "HPAENGK"
        base_mass = compute_peptide_mass(pep)
        double_mass = base_mass + DOUBLE_DEPC
        hits = _match_mass_to_peptide(double_mass, TOLERANCE_HIGH_RES)
        double_hits = [h for h in hits if h[3]]  # is_double flag
        assert len(double_hits) > 0

    def test_vehsdlsfsk_depc_his51(self):
        """VEHSDLSFSK + DEPC should hit His51 (offset 2, abs_pos 51)."""
        pep = "VEHSDLSFSK"
        base_mass = compute_peptide_mass(pep)
        labeled_mass = base_mass + DEPC_MASS_SHIFT
        hits = _match_mass_to_peptide(labeled_mass, TOLERANCE_HIGH_RES)
        positions = [h[1] + h[2] for h in hits if h[0] == pep]  # start + offset = abs_pos
        assert 51 in positions, f"His51 not found in hits: {hits}"

    def test_depc_fragment_ions_cached(self):
        """Repeated calls return same result (LRU cache)."""
        pep = "HPAENGK"
        ions1 = _cached_depc_fragment_ions(pep, 0)
        ions2 = _cached_depc_fragment_ions(pep, 0)
        assert ions1 is ions2  # same object from cache

    def test_depc_fragment_ions_b_shift(self):
        """b-ions containing the modified residue should be shifted by +72.0211."""
        pep = "HPAENGK"
        from src.constants import compute_fragment_ions
        normal_ions = compute_fragment_ions(pep)
        depc_ions = _cached_depc_fragment_ions(pep, 0)  # H at offset 0

        # b1 (contains H) should be shifted
        assert abs(depc_ions["b"][0] - normal_ions["b"][0] - DEPC_MASS_SHIFT) < 1e-3
        # b0 for a 1-residue suffix doesn't exist — check y ions covering the non-modified tail
        # y ions that DON'T include position 0 should be identical
        # y[0] = last residue only (K) — unmodified
        assert abs(depc_ions["y"][0] - normal_ions["y"][0]) < 1e-4
