"""
Compute per-residue labeling extents from confirmed DEPC events.

Labeling extent = labeled_intensity / (labeled_intensity + unlabeled_intensity)
Aggregates technical replicates; reports mean ± SD and CV%.
"""

from __future__ import annotations

import logging
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd

from .constants import B2M_LENGTH, SUSCEPTIBLE_RESIDUES, B2M_RESIDUES
from .sequence_mapper import DEPCEvent, map_events_to_residues
from .validator import validate_labeling_extent, validate_replicate_cv

logger = logging.getLogger(__name__)


def _labeling_extent(labeled: float, unlabeled: float) -> float:
    denom = labeled + unlabeled
    if denom <= 0.0:
        return 0.0
    return float(labeled / denom)


# ---------------------------------------------------------------------------
# Single-condition quantification
# ---------------------------------------------------------------------------

def quantify_condition(
    events: list[DEPCEvent],
    condition_label: str = "unknown",
    cv_threshold: float = 0.30,
) -> pd.Series:
    """
    Aggregate DEPC events for one experimental condition into per-residue
    labeling extents.

    Returns a pd.Series indexed 1..B2M_LENGTH with float values in [0, 1].
    Non-susceptible or undetected residues → NaN.
    """
    # Collect labeling extents by position
    extents_by_pos: dict[int, list[float]] = defaultdict(list)

    for event in events:
        if event.is_double_depc:
            continue
        extent = _labeling_extent(event.labeling_intensity, event.unlabeled_intensity)
        try:
            validate_labeling_extent(extent, context=f"{condition_label}/pos{event.residue_position}")
        except ValueError as exc:
            logger.warning("Clamping invalid labeling extent: %s", exc)
            extent = float(np.clip(extent, 0.0, 1.0))
        extents_by_pos[event.residue_position].append(extent)

    # Validate CV% and compute means
    result = pd.Series(np.nan, index=pd.RangeIndex(1, B2M_LENGTH + 1), name=condition_label)

    for pos, vals in extents_by_pos.items():
        validate_replicate_cv(vals, pos, cv_threshold)
        result[pos] = float(np.mean(vals))

    return result


# ---------------------------------------------------------------------------
# Multi-condition quantification
# ---------------------------------------------------------------------------

def quantify_all_conditions(
    events_by_condition: dict[str, list[DEPCEvent]],
    cv_threshold: float = 0.30,
) -> pd.DataFrame:
    """
    Build a DataFrame of labeling extents: rows = residue positions (1..N),
    columns = experimental conditions.

    Values are mean labeling extents; NaN where no data.
    """
    series_list = []
    for condition, events in events_by_condition.items():
        s = quantify_condition(events, condition_label=condition, cv_threshold=cv_threshold)
        series_list.append(s)

    if not series_list:
        return pd.DataFrame(
            index=pd.RangeIndex(1, B2M_LENGTH + 1),
            columns=pd.Index([], name="condition"),
        )

    df = pd.concat(series_list, axis=1)
    df.index.name = "residue_position"
    logger.info(
        "Quantified %d conditions; %d residues with any data",
        len(df.columns),
        int(df.notna().any(axis=1).sum()),
    )
    return df


# ---------------------------------------------------------------------------
# Replicate summary
# ---------------------------------------------------------------------------

def summarize_replicates(
    replicate_events: list[list[DEPCEvent]],
    condition_label: str = "condition",
) -> pd.DataFrame:
    """
    Given a list of replicate event lists (each a separate LC-MS/MS run),
    compute mean ± SD per residue.

    Returns DataFrame with columns: [condition_label, f"{condition_label}_sd", "n_replicates"].
    """
    rep_series = [
        quantify_condition(ev, condition_label=f"{condition_label}_rep{i}")
        for i, ev in enumerate(replicate_events)
    ]
    if not rep_series:
        return pd.DataFrame()

    stacked = pd.concat(rep_series, axis=1)
    summary = pd.DataFrame(index=stacked.index)
    summary[condition_label] = stacked.mean(axis=1)
    summary[f"{condition_label}_sd"] = stacked.std(axis=1, ddof=1)
    summary["n_replicates"] = stacked.notna().sum(axis=1)

    return summary


# ---------------------------------------------------------------------------
# CV reporting
# ---------------------------------------------------------------------------

def compute_cv_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame of replicates (rows=residues, columns=replicates),
    compute CV% per residue.
    """
    means = df.mean(axis=1)
    stds = df.std(axis=1, ddof=1)
    cv = (stds / means.replace(0, np.nan)) * 100
    result = pd.DataFrame({"mean": means, "sd": stds, "cv_pct": cv})
    result.index.name = "residue_position"
    return result
