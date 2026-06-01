"""
Constants for the β2m DEPC covalent labeling pipeline.

β2m = β-2-microglobulin, UniProt P61769, mature form.
His13/His31/His51 are Cu(II) binding sites.
"""

# ---------------------------------------------------------------------------
# β2m sequence
# ---------------------------------------------------------------------------

B2M_SEQUENCE: str = (
    "IQRTPKIQVYSRHPAENGKSNFLNCYVSGFHPSDIEVDLLK"
    "NGERIEKVEHSDLSFSKDWSFYLLYYTEFTPTELKPHQNLVF"
    "QNLSSTPNVKVEELSSTHPFPFDLNLNPKKK"
)

B2M_LENGTH: int = len(B2M_SEQUENCE)  # 114 — UniProt P61769 mature form including C-terminal tail

# 1-indexed residue dict
B2M_RESIDUES: dict[int, str] = {i + 1: aa for i, aa in enumerate(B2M_SEQUENCE)}

# Known Cu(II) binding histidines (1-indexed)
CU_BINDING_HIS: tuple[int, ...] = (13, 31, 51)

# ---------------------------------------------------------------------------
# DEPC chemistry
# ---------------------------------------------------------------------------

DEPC_MASS_SHIFT: float = 72.0211    # monoisotopic, C4H4O2 carbethoxyl
DOUBLE_DEPC: float = 144.0423       # double carbethoxylation on His (excluded from primary analysis)

SUSCEPTIBLE_RESIDUES: frozenset[str] = frozenset({"H", "K", "Y", "S", "T", "C"})
# N-terminus is also susceptible; handled separately in sequence_mapper

# Instrument-mode mass tolerances (Da)
TOLERANCE_HIGH_RES: float = 0.02   # Orbitrap
TOLERANCE_LOW_RES: float = 0.50    # ion trap

# ---------------------------------------------------------------------------
# Amino acid monoisotopic residue masses (Da)
# ---------------------------------------------------------------------------

AA_MONOISOTOPIC_MASSES: dict[str, float] = {
    "A": 71.03711,
    "R": 156.10111,
    "N": 114.04293,
    "D": 115.02694,
    "C": 103.00919,
    "E": 129.04259,
    "Q": 128.05858,
    "G": 57.02146,
    "H": 137.05891,
    "I": 113.08406,
    "L": 113.08406,
    "K": 128.09496,
    "M": 131.04049,
    "F": 147.06841,
    "P": 97.05276,
    "S": 87.03203,
    "T": 101.04768,
    "W": 186.07931,
    "Y": 163.06333,
    "V": 99.06841,
}

WATER_MASS: float = 18.01056
PROTON_MASS: float = 1.007276

# ---------------------------------------------------------------------------
# Tryptic digest utilities
# ---------------------------------------------------------------------------

def compute_peptide_mass(sequence: str) -> float:
    """Monoisotopic mass of a peptide (residue sum + water)."""
    return sum(AA_MONOISOTOPIC_MASSES[aa] for aa in sequence) + WATER_MASS


def _trypsin_cut_sites(sequence: str) -> list[int]:
    """
    Returns 0-indexed positions AFTER which trypsin cuts.
    Rule: cut after K or R unless the NEXT residue is P.
    """
    sites: list[int] = []
    for i, aa in enumerate(sequence):
        if aa in ("K", "R"):
            next_aa = sequence[i + 1] if i + 1 < len(sequence) else None
            if next_aa != "P":
                sites.append(i)
    return sites


def digest_protein(
    sequence: str,
    max_missed_cleavages: int = 2,
) -> dict[str, tuple[int, int]]:
    """
    In-silico tryptic digest.

    Returns:
        {peptide_sequence: (start_1indexed, end_1indexed)}

    Generates peptides for 0 .. max_missed_cleavages missed cleavages.
    """
    cuts = _trypsin_cut_sites(sequence)
    # Boundary positions (0-indexed, exclusive ends)
    boundaries = [0] + [c + 1 for c in cuts] + [len(sequence)]

    peptides: dict[str, tuple[int, int]] = {}
    n = len(boundaries) - 1
    for i in range(n):
        for j in range(i + 1, min(i + max_missed_cleavages + 2, n + 1)):
            start = boundaries[i]   # 0-indexed inclusive
            end = boundaries[j]     # 0-indexed exclusive
            pep = sequence[start:end]
            if len(pep) >= 4:       # ignore very short peptides
                start_1 = start + 1
                end_1 = end         # 1-indexed inclusive
                if pep not in peptides:
                    peptides[pep] = (start_1, end_1)
    return peptides


# Pre-computed tryptic peptide map (0–2 missed cleavages)
TRYPTIC_PEPTIDES: dict[str, tuple[int, int]] = digest_protein(
    B2M_SEQUENCE, max_missed_cleavages=2
)

# Reverse lookup: position → set of peptides that cover it
def _build_position_coverage() -> dict[int, list[str]]:
    cov: dict[int, list[str]] = {i: [] for i in range(1, B2M_LENGTH + 1)}
    for pep, (s, e) in TRYPTIC_PEPTIDES.items():
        for pos in range(s, e + 1):
            cov[pos].append(pep)
    return cov


POSITION_COVERAGE: dict[int, list[str]] = _build_position_coverage()

# ---------------------------------------------------------------------------
# Fragment ion tables (cached per peptide)
# ---------------------------------------------------------------------------

def compute_fragment_ions(
    peptide: str,
) -> dict[str, list[float]]:
    """
    Returns b-ion and y-ion series for a peptide (singly charged, +1).

    b[i] covers residues 0..i (i from 0 to len-2)
    y[i] covers residues len-1-i..len-1 (i from 0 to len-2)
    """
    masses = [AA_MONOISOTOPIC_MASSES[aa] for aa in peptide]
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
