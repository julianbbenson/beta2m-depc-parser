"""Tests for src/labeling_quantifier.py"""

import numpy as np
import pandas as pd
import pytest

from src.constants import B2M_LENGTH, B2M_RESIDUES, SUSCEPTIBLE_RESIDUES
from src.labeling_quantifier import (
    compute_cv_table,
    quantify_all_conditions,
    quantify_condition,
    summarize_replicates,
)
from src.sequence_mapper import DEPCEvent


def _make_events(residue_positions: list[int], li: float = 0.6, ui: float = 0.4) -> list[DEPCEvent]:
    events = []
    for pos in residue_positions:
        aa = B2M_RESIDUES.get(pos, "H")
        pep_start = pos  # minimal peptide starts at the residue
        events.append(
            DEPCEvent(
                peptide="HPAENGK",
                residue_position=pos,
                residue_type=aa,
                labeling_intensity=li,
                unlabeled_intensity=ui,
                scan_id=1,
                is_double_depc=False,
            )
        )
    return events


class TestQuantifyCondition:
    def test_returns_series_of_correct_length(self):
        events = _make_events([13, 31, 51])
        result = quantify_condition(events, "test")
        assert len(result) == B2M_LENGTH

    def test_index_is_one_based(self):
        events = _make_events([13])
        result = quantify_condition(events, "test")
        assert result.index[0] == 1
        assert result.index[-1] == B2M_LENGTH

    def test_his13_correct_extent(self):
        # li=0.6, ui=0.4 → extent = 0.6/(0.6+0.4) = 0.6
        events = _make_events([13], li=0.6, ui=0.4)
        result = quantify_condition(events, "test")
        assert abs(result[13] - 0.6) < 1e-6

    def test_unlabeled_residue_is_nan(self):
        events = _make_events([13])
        result = quantify_condition(events, "test")
        # Position 1 (I) has no events → NaN
        assert np.isnan(result[1])

    def test_double_depc_excluded(self):
        events = [
            DEPCEvent(
                peptide="HPAENGK",
                residue_position=13,
                residue_type="H",
                labeling_intensity=1.0,
                unlabeled_intensity=0.0,
                scan_id=1,
                is_double_depc=True,  # should be excluded
            )
        ]
        result = quantify_condition(events, "test")
        assert np.isnan(result[13])

    def test_multiple_events_averaged(self):
        e1 = DEPCEvent("HPAENGK", 13, "H", 0.6, 0.4, 1, False)
        e2 = DEPCEvent("HPAENGK", 13, "H", 0.8, 0.2, 2, False)
        result = quantify_condition([e1, e2], "test")
        # extents: 0.6 and 0.8 → mean 0.7
        assert abs(result[13] - 0.7) < 1e-6

    def test_all_values_in_bounds(self):
        events = _make_events([13, 31, 51], li=0.5, ui=0.5)
        result = quantify_condition(events, "test")
        valid = result.dropna()
        assert (valid >= 0.0).all()
        assert (valid <= 1.0).all()


class TestQuantifyAllConditions:
    def test_returns_dataframe(self):
        no_cu = _make_events([13, 31, 51], li=0.6, ui=0.4)
        cu = _make_events([13, 31, 51], li=0.2, ui=0.8)
        result = quantify_all_conditions({"no_cu": no_cu, "cu": cu})
        assert isinstance(result, pd.DataFrame)

    def test_columns_are_conditions(self):
        no_cu = _make_events([13])
        cu = _make_events([13])
        result = quantify_all_conditions({"no_cu": no_cu, "cu": cu})
        assert set(result.columns) == {"no_cu", "cu"}

    def test_empty_input(self):
        result = quantify_all_conditions({})
        assert result.empty

    def test_cu_lower_than_no_cu_for_his(self):
        no_cu = _make_events([13, 31, 51], li=0.6, ui=0.4)
        cu = _make_events([13, 31, 51], li=0.2, ui=0.8)
        result = quantify_all_conditions({"no_cu": no_cu, "cu": cu})
        for pos in [13, 31, 51]:
            assert result.loc[pos, "no_cu"] > result.loc[pos, "cu"]


class TestSummarizeReplicates:
    def test_returns_dataframe_with_mean_sd(self):
        rep1 = _make_events([13], li=0.6, ui=0.4)
        rep2 = _make_events([13], li=0.7, ui=0.3)
        rep3 = _make_events([13], li=0.65, ui=0.35)
        result = summarize_replicates([rep1, rep2, rep3], "test")
        assert "test" in result.columns
        assert "test_sd" in result.columns
        assert "n_replicates" in result.columns

    def test_mean_correct(self):
        rep1 = _make_events([13], li=0.6, ui=0.4)   # extent 0.6
        rep2 = _make_events([13], li=0.8, ui=0.2)   # extent 0.8
        result = summarize_replicates([rep1, rep2], "test")
        assert abs(result.loc[13, "test"] - 0.7) < 1e-6

    def test_sd_correct(self):
        rep1 = _make_events([13], li=0.6, ui=0.4)
        rep2 = _make_events([13], li=0.8, ui=0.2)
        result = summarize_replicates([rep1, rep2], "test")
        expected_sd = np.std([0.6, 0.8], ddof=1)
        assert abs(result.loc[13, "test_sd"] - expected_sd) < 1e-6


class TestCVTable:
    def test_cv_computation(self):
        df = pd.DataFrame(
            {"r0": [0.6, 0.4], "r1": [0.8, 0.4]},
            index=[13, 31],
        )
        result = compute_cv_table(df)
        assert "mean" in result.columns
        assert "cv_pct" in result.columns

    def test_cv_zero_mean(self):
        df = pd.DataFrame({"r0": [0.0], "r1": [0.0]}, index=[13])
        result = compute_cv_table(df)
        assert np.isnan(result.loc[13, "cv_pct"])
