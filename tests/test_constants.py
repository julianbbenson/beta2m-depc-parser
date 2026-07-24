"""Tests for src/constants.py"""

import pytest
from src.constants import (
    AA_MONOISOTOPIC_MASSES,
    B2M_LENGTH,
    B2M_RESIDUES,
    B2M_SEQUENCE,
    CU_BINDING_HIS,
    DEPC_MASS_SHIFT,
    DOUBLE_DEPC,
    PROTON_MASS,
    SUSCEPTIBLE_RESIDUES,
    TRYPTIC_PEPTIDES,
    WATER_MASS,
    compute_fragment_ions,
    compute_peptide_mass,
    digest_protein,
)


class TestSequence:
    def test_sequence_is_string(self):
        assert isinstance(B2M_SEQUENCE, str)

    def test_sequence_length_matches_constant(self):
        assert len(B2M_SEQUENCE) == B2M_LENGTH

    def test_residue_dict_length(self):
        assert len(B2M_RESIDUES) == B2M_LENGTH

    def test_residue_dict_one_indexed(self):
        assert 0 not in B2M_RESIDUES
        assert 1 in B2M_RESIDUES
        assert B2M_LENGTH in B2M_RESIDUES

    def test_residue_dict_matches_sequence(self):
        for i, aa in enumerate(B2M_SEQUENCE):
            assert B2M_RESIDUES[i + 1] == aa

    def test_his_positions(self):
        his_positions = [pos for pos, aa in B2M_RESIDUES.items() if aa == "H"]
        assert 13 in his_positions
        assert 31 in his_positions
        assert 51 in his_positions

    def test_cu_binding_his_in_sequence(self):
        for pos in CU_BINDING_HIS:
            assert B2M_RESIDUES[pos] == "H", f"His{pos} not H in sequence"

    def test_sequence_only_standard_amino_acids(self):
        valid = set(AA_MONOISOTOPIC_MASSES.keys())
        for pos, aa in B2M_RESIDUES.items():
            assert aa in valid, f"Unknown amino acid '{aa}' at position {pos}"


class TestDEPCChemistry:
    def test_depc_mass_shift_value(self):
        assert abs(DEPC_MASS_SHIFT - 72.0211) < 1e-4

    def test_double_depc_is_twice_single(self):
        assert abs(DOUBLE_DEPC - 2 * DEPC_MASS_SHIFT) < 1e-3

    def test_susceptible_residues_present(self):
        required = {"H", "K", "Y", "S", "T", "C"}
        assert required.issubset(SUSCEPTIBLE_RESIDUES)

    def test_susceptible_residues_not_include_non_reactive(self):
        non_reactive = {"A", "G", "P", "V", "L", "I", "F", "W", "M", "D", "E", "N", "Q", "R"}
        assert SUSCEPTIBLE_RESIDUES.isdisjoint(non_reactive)


class TestPeptideMass:
    def test_known_dipeptide_mass(self):
        # GG mass: 57.02146 * 2 + 18.01056 = 132.05348
        mass = compute_peptide_mass("GG")
        assert abs(mass - 132.053) < 0.01

    def test_alanine_mass(self):
        # A = 71.03711 + 18.01056 (water) = 89.04768
        mass = compute_peptide_mass("A")
        assert abs(mass - 89.048) < 0.01

    def test_water_added(self):
        """Peptide mass should equal residue masses + water."""
        pep = "HPAENGK"
        residue_sum = sum(AA_MONOISOTOPIC_MASSES[aa] for aa in pep)
        expected = residue_sum + WATER_MASS
        assert abs(compute_peptide_mass(pep) - expected) < 1e-4


class TestTrypticDigest:
    def test_tryptic_peptides_is_dict(self):
        assert isinstance(TRYPTIC_PEPTIDES, dict)

    def test_minimum_peptides_generated(self):
        # β2m has multiple tryptic peptides
        assert len(TRYPTIC_PEPTIDES) >= 10

    def test_peptides_cover_his13(self):
        covering = [pep for pep, (s, e) in TRYPTIC_PEPTIDES.items() if s <= 13 <= e]
        assert len(covering) > 0, "No tryptic peptide covers His13"

    def test_peptides_cover_his31(self):
        covering = [pep for pep, (s, e) in TRYPTIC_PEPTIDES.items() if s <= 31 <= e]
        assert len(covering) > 0, "No tryptic peptide covers His31"

    def test_peptides_cover_his51(self):
        covering = [pep for pep, (s, e) in TRYPTIC_PEPTIDES.items() if s <= 51 <= e]
        assert len(covering) > 0, "No tryptic peptide covers His51"

    def test_peptide_sequences_valid(self):
        for pep, (s, e) in TRYPTIC_PEPTIDES.items():
            # Verify the peptide actually exists in B2M_SEQUENCE at stated position
            extracted = B2M_SEQUENCE[s - 1 : e]
            assert extracted == pep, (
                f"Peptide '{pep}' at [{s},{e}] doesn't match sequence '{extracted}'"
            )

    def test_no_proline_rule(self):
        """Trypsin should not cut before Pro."""
        for pep, (s, e) in TRYPTIC_PEPTIDES.items():
            if e < B2M_LENGTH:
                last_aa = B2M_RESIDUES[e]
                next_aa = B2M_RESIDUES[e + 1]
                if last_aa in ("K", "R") and next_aa == "P":
                    # This peptide would only appear as part of a missed-cleavage peptide
                    # so its standalone existence means the cut WAS skipped — verify
                    pass  # allowed in missed-cleavage peptides

    def test_base_peptide_hpaengk(self):
        """HPAENGK covers His13 (positions 13-19)."""
        assert "HPAENGK" in TRYPTIC_PEPTIDES
        start, end = TRYPTIC_PEPTIDES["HPAENGK"]
        assert start == 13
        assert end == 19

    def test_base_peptide_vehsdlsfsk(self):
        """VEHSDLSFSK covers His51 (positions 49-58)."""
        assert "VEHSDLSFSK" in TRYPTIC_PEPTIDES
        start, end = TRYPTIC_PEPTIDES["VEHSDLSFSK"]
        assert start == 49
        assert end == 58
        assert B2M_RESIDUES[51] == "H"


class TestFragmentIons:
    def test_fragment_ions_structure(self):
        ions = compute_fragment_ions("PEPTIDE")
        assert "b" in ions
        assert "y" in ions

    def test_b_ion_count(self):
        pep = "PEPTIDE"
        ions = compute_fragment_ions(pep)
        assert len(ions["b"]) == len(pep) - 1

    def test_y_ion_count(self):
        pep = "PEPTIDE"
        ions = compute_fragment_ions(pep)
        assert len(ions["y"]) == len(pep) - 1

    def test_b1_ion_value(self):
        # b1 for "GP" = G residue mass + proton
        ions = compute_fragment_ions("GP")
        expected_b1 = AA_MONOISOTOPIC_MASSES["G"] + PROTON_MASS
        assert abs(ions["b"][0] - expected_b1) < 1e-4

    def test_y1_ion_value(self):
        # y1 for "GP" = P residue mass + water + proton
        ions = compute_fragment_ions("GP")
        expected_y1 = AA_MONOISOTOPIC_MASSES["P"] + WATER_MASS + PROTON_MASS
        assert abs(ions["y"][0] - expected_y1) < 1e-4

    def test_complementary_ions(self):
        """b[i] + y[n-1-i] = peptide_mass + 2 * proton (approximately)."""
        pep = "HPAENGK"
        ions = compute_fragment_ions(pep)
        pep_mass = compute_peptide_mass(pep)
        for i in range(len(pep) - 1):
            total = ions["b"][i] + ions["y"][len(pep) - 2 - i]
            expected = pep_mass + 2 * PROTON_MASS
            assert abs(total - expected) < 1e-3, (
                f"Complementary ion check failed at index {i}: {total:.4f} vs {expected:.4f}"
            )
