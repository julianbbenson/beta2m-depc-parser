"""Tests for src/sequence_mapper.py"""

import pytest

from src.constants import B2M_RESIDUES, SUSCEPTIBLE_RESIDUES, TRYPTIC_PEPTIDES
from src.sequence_mapper import (
    DEPCEvent,
    find_susceptible_sites_in_peptide,
    map_events_to_residues,
    resolve_residue_in_peptide,
    _labeling_extent,
)


class TestLabelingExtent:
    def test_half_labeled(self):
        assert abs(_labeling_extent(1.0, 1.0) - 0.5) < 1e-6

    def test_fully_labeled(self):
        assert abs(_labeling_extent(1.0, 0.0) - 1.0) < 1e-6

    def test_unlabeled(self):
        assert abs(_labeling_extent(0.0, 1.0) - 0.0) < 1e-6

    def test_zero_denominator(self):
        assert _labeling_extent(0.0, 0.0) == 0.0

    def test_within_bounds(self):
        for li in [0.1, 0.5, 1.0, 100.0]:
            for ui in [0.1, 0.5, 1.0, 100.0]:
                ext = _labeling_extent(li, ui)
                assert 0.0 <= ext <= 1.0


class TestResolvePeptidePosition:
    def test_his13_offset_in_hpaengk(self):
        # HPAENGK starts at 13; H is at offset 0
        pos, aa = resolve_residue_in_peptide("HPAENGK", 13, 0)
        assert pos == 13
        assert aa == "H"

    def test_his51_offset_in_vehsdlsfsk(self):
        # VEHSDLSFSK starts at 49; H is at offset 2
        pos, aa = resolve_residue_in_peptide("VEHSDLSFSK", 49, 2)
        assert pos == 51
        assert aa == "H"

    def test_lys41_in_long_peptide(self):
        # SNFLNCYVSGFHPSDIEVDLLK starts at 20; K is at offset 21
        pos, aa = resolve_residue_in_peptide("SNFLNCYVSGFHPSDIEVDLLK", 20, 21)
        assert pos == 41
        assert aa == "K"


class TestFindSusceptibleSites:
    def test_hpaengk_sites(self):
        sites = find_susceptible_sites_in_peptide("HPAENGK", 13)
        positions = [s[0] for s in sites]
        aas = [s[1] for s in sites]
        assert 13 in positions  # H
        assert 19 in positions  # K
        assert "H" in aas
        assert "K" in aas

    def test_vehsdlsfsk_sites(self):
        sites = find_susceptible_sites_in_peptide("VEHSDLSFSK", 49)
        positions = [s[0] for s in sites]
        assert 51 in positions  # H (His51)
        assert 52 in positions  # S
        assert 55 in positions  # S
        assert 57 in positions  # S
        assert 58 in positions  # K

    def test_non_susceptible_not_included(self):
        # IQR: I, Q, R — none are in SUSCEPTIBLE_RESIDUES by side chain.
        # However position 1 N-terminus is marked as "N-term" (always susceptible).
        sites = find_susceptible_sites_in_peptide("IQR", 1)
        aas = [s[1] for s in sites]
        for aa in aas:
            assert aa in SUSCEPTIBLE_RESIDUES or aa == "N-term", (
                f"Unexpected AA '{aa}' in sites (only susceptible AAs or 'N-term' allowed)"
            )

    def test_sites_within_sequence_bounds(self):
        for pep, (start, end) in TRYPTIC_PEPTIDES.items():
            sites = find_susceptible_sites_in_peptide(pep, start)
            for pos, aa in sites:
                assert 1 <= pos <= len(B2M_RESIDUES), f"Position {pos} out of bounds"


class TestMapEventsToResidues:
    def _make_event(self, peptide, pep_start, offset, li=0.6, ui=0.4, double=False):
        pos = pep_start + offset
        aa = B2M_RESIDUES.get(pos, "?")
        return DEPCEvent(
            peptide=peptide,
            residue_position=pos,
            residue_type=aa,
            labeling_intensity=li,
            unlabeled_intensity=ui,
            scan_id=1,
            is_double_depc=double,
        )

    def test_single_event_his13(self):
        event = self._make_event("HPAENGK", 13, 0, li=0.6, ui=0.4)
        result = map_events_to_residues([event])
        assert 13 in result
        assert abs(result[13][0] - 0.6) < 0.01

    def test_double_depc_excluded(self):
        event = self._make_event("HPAENGK", 13, 0, li=0.9, ui=0.1, double=True)
        result = map_events_to_residues([event])
        assert 13 not in result

    def test_multiple_events_same_residue_collected(self):
        e1 = self._make_event("HPAENGK", 13, 0, li=0.6, ui=0.4)
        e2 = self._make_event("HPAENGK", 13, 0, li=0.7, ui=0.3)
        result = map_events_to_residues([e1, e2])
        assert len(result[13]) == 2

    def test_overlapping_peptides_both_collected(self):
        # HPAENGK (13-19) and IQVYSRHPAENGK (7-19) both cover His13
        e1 = self._make_event("HPAENGK", 13, 0, li=0.6, ui=0.4)
        e2 = self._make_event("IQVYSRHPAENGK", 7, 6, li=0.5, ui=0.5)  # H at offset 6
        result = map_events_to_residues([e1, e2])
        assert 13 in result
        assert len(result[13]) == 2

    def test_extents_in_bounds(self):
        events = [
            self._make_event("HPAENGK", 13, 0, li=1.0, ui=0.0),
            self._make_event("VEHSDLSFSK", 49, 2, li=0.0, ui=1.0),
        ]
        result = map_events_to_residues(events)
        for pos, extents in result.items():
            for ext in extents:
                assert 0.0 <= ext <= 1.0
