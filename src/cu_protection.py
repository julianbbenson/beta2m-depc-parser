"""
Cu(II) protection analysis.

Compares labeling extents between +Cu(II) and -Cu(II) conditions.
Two-tailed t-test with Bonferroni correction.
Protection score = (no_Cu - with_Cu) / no_Cu.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from .constants import B2M_LENGTH, CU_BINDING_HIS
from .validator import sanity_check_cu_protection

logger = logging.getLogger(__name__)


@dataclass
class ProtectionResult:
    residue_position: int
    residue_type: str
    labeling_no_cu: float
    labeling_cu: float
    protection_score: float          # (no_cu - cu) / no_cu; positive = protected
    p_value_raw: float
    p_value_bonferroni: float
    is_protected: bool               # score > threshold AND p_bonf < alpha
    n_cu: int = 0
    n_no_cu: int = 0


def compute_protection_scores(
    df_no_cu: pd.DataFrame,
    df_cu: pd.DataFrame,
    protection_threshold: float = 0.20,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Compare +Cu vs -Cu labeling extents per residue.

    Parameters:
        df_no_cu: DataFrame of labeling extents without Cu (rows=positions, cols=replicates)
        df_cu:    DataFrame of labeling extents with Cu    (rows=positions, cols=replicates)

    Returns a DataFrame with one row per residue and columns:
        residue_position, residue_type, mean_no_cu, mean_cu, protection_score,
        p_raw, p_bonferroni, is_protected
    """
    from .constants import B2M_RESIDUES

    positions = sorted(set(df_no_cu.index) | set(df_cu.index))
    n_tests = len(positions)

    records = []
    protection_scores: dict[int, float] = {}
    p_values: dict[int, float] = {}

    for pos in positions:
        row_no_cu = df_no_cu.loc[pos].dropna().values if pos in df_no_cu.index else np.array([])
        row_cu = df_cu.loc[pos].dropna().values if pos in df_cu.index else np.array([])

        mean_no_cu = float(row_no_cu.mean()) if len(row_no_cu) else np.nan
        mean_cu = float(row_cu.mean()) if len(row_cu) else np.nan

        if np.isnan(mean_no_cu) or mean_no_cu == 0.0:
            score = np.nan
            p_raw = 1.0
        else:
            score = (mean_no_cu - mean_cu) / mean_no_cu if not np.isnan(mean_cu) else np.nan
            if len(row_no_cu) >= 2 and len(row_cu) >= 2:
                _, p_raw = stats.ttest_ind(row_no_cu, row_cu, equal_var=False)
            else:
                p_raw = 1.0  # insufficient data

        protection_scores[pos] = float(score) if not np.isnan(score) else 0.0
        p_values[pos] = float(p_raw)

        records.append(
            {
                "residue_position": pos,
                "residue_type": B2M_RESIDUES.get(pos, "?"),
                "mean_no_cu": mean_no_cu,
                "mean_cu": mean_cu,
                "protection_score": score,
                "p_raw": p_raw,
                "n_no_cu": len(row_no_cu),
                "n_cu": len(row_cu),
            }
        )

    result = pd.DataFrame(records).set_index("residue_position")

    # Bonferroni correction
    result["p_bonferroni"] = (result["p_raw"] * n_tests).clip(upper=1.0)

    result["is_protected"] = (
        (result["protection_score"] > protection_threshold)
        & (result["p_bonferroni"] < alpha)
    )

    # Biological sanity check (warns, doesn't raise)
    sanity_check_cu_protection(protection_scores, p_values, alpha, protection_threshold)

    logger.info(
        "Protection analysis: %d protected residues (score>%.2f, p_bonf<%.2f)",
        int(result["is_protected"].sum()),
        protection_threshold,
        alpha,
    )
    protected_his = [
        p for p in CU_BINDING_HIS if result.loc[p, "is_protected"] if p in result.index
    ]
    logger.info("Protected His sites (expected 13,31,51): %s", protected_his)

    return result


def summarise_protection(result: pd.DataFrame) -> dict:
    """High-level summary dict for logging / reporting."""
    protected = result[result["is_protected"]]
    return {
        "n_protected": len(protected),
        "protected_positions": sorted(protected.index.tolist()),
        "protected_his": [p for p in CU_BINDING_HIS if p in protected.index],
        "max_protection_score": float(result["protection_score"].max()),
    }


def make_replicate_dataframes(
    events_by_condition_replicate: dict[str, list[list]],
) -> dict[str, pd.DataFrame]:
    """
    Convert {condition: [events_rep1, events_rep2, ...]} into
    {condition: DataFrame(rows=positions, cols=replicates)}.
    """
    from .labeling_quantifier import quantify_condition

    result: dict[str, pd.DataFrame] = {}
    for condition, replicate_lists in events_by_condition_replicate.items():
        series = [
            quantify_condition(ev_list, condition_label=f"{condition}_r{i}")
            for i, ev_list in enumerate(replicate_lists)
        ]
        result[condition] = pd.concat(series, axis=1) if series else pd.DataFrame()
    return result
