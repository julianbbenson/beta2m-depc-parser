"""Tests for src/array_formatter.py"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.array_formatter import (
    build_condition_array,
    build_multi_condition_array,
    build_protection_array,
    load_array,
    read_csv_metadata,
    save_array,
)
from src.constants import B2M_LENGTH, B2M_RESIDUES, SUSCEPTIBLE_RESIDUES


def _make_series(values: dict[int, float]) -> pd.Series:
    s = pd.Series(np.nan, index=pd.RangeIndex(1, B2M_LENGTH + 1))
    for pos, val in values.items():
        s[pos] = val
    return s


class TestBuildConditionArray:
    def test_output_shape(self):
        s = _make_series({13: 0.6, 31: 0.5, 51: 0.4})
        arr = build_condition_array(s)
        assert arr.shape == (B2M_LENGTH,)

    def test_output_dtype_float32(self):
        s = _make_series({13: 0.6})
        arr = build_condition_array(s)
        assert arr.dtype == np.float32

    def test_his13_value(self):
        s = _make_series({13: 0.6})
        arr = build_condition_array(s)
        assert abs(arr[12] - 0.6) < 1e-6  # 0-indexed: position 13 → index 12

    def test_non_susceptible_filled_with_zero(self):
        s = _make_series({13: 0.6})
        arr = build_condition_array(s, fill_non_susceptible=0.0)
        # Position 1 (I) is not susceptible
        assert arr[0] == 0.0

    def test_undetected_susceptible_filled_with_zero(self):
        # K at position 6 — not in series → fill_undetected
        s = _make_series({})
        arr = build_condition_array(s, fill_undetected=0.0)
        # All susceptible residues with no data → 0
        for pos in range(1, B2M_LENGTH + 1):
            aa = B2M_RESIDUES[pos]
            if aa in SUSCEPTIBLE_RESIDUES:
                assert arr[pos - 1] == 0.0

    def test_all_values_in_bounds(self):
        s = _make_series({13: 0.6, 31: 0.55, 51: 0.3})
        arr = build_condition_array(s)
        assert np.all(arr >= 0.0)
        assert np.all(arr <= 1.0)

    def test_validation_raises_for_wrong_length(self):
        """Manually construct a wrong-length array to verify validator fires."""
        from src.validator import validate_output_array
        bad = np.zeros(50, dtype=np.float32)
        with pytest.raises(ValueError):
            validate_output_array(bad)

    def test_validation_raises_for_out_of_bounds(self):
        from src.validator import validate_output_array
        bad = np.full(B2M_LENGTH, 2.0, dtype=np.float32)
        with pytest.raises(ValueError):
            validate_output_array(bad)


class TestBuildMultiConditionArray:
    def test_shape_multiple_conditions(self):
        s1 = _make_series({13: 0.6})
        s2 = _make_series({13: 0.2})
        df = pd.DataFrame({"no_cu": s1, "cu": s2})
        arr = build_multi_condition_array(df)
        assert arr.shape == (B2M_LENGTH, 2)

    def test_correct_values_in_columns(self):
        s1 = _make_series({13: 0.6})
        s2 = _make_series({13: 0.2})
        df = pd.DataFrame({"no_cu": s1, "cu": s2})
        arr = build_multi_condition_array(df)
        assert abs(arr[12, 0] - 0.6) < 1e-6  # no_cu, His13
        assert abs(arr[12, 1] - 0.2) < 1e-6  # cu, His13


class TestSaveLoad:
    def test_save_and_load_npy(self):
        s = _make_series({13: 0.6, 31: 0.5})
        arr = build_condition_array(s)
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = save_array(arr, tmpdir, "test_run", conditions=["test"])
            loaded = load_array(paths["npy"])
            assert np.allclose(arr, loaded)

    def test_save_creates_csv(self):
        s = _make_series({13: 0.6})
        arr = build_condition_array(s)
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = save_array(arr, tmpdir, "test_run", conditions=["test"])
            assert paths["csv"].exists()

    def test_csv_has_correct_rows(self):
        s = _make_series({13: 0.6})
        arr = build_condition_array(s)
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = save_array(arr, tmpdir, "test_run", conditions=["test"])
            lines = [
                l for l in paths["csv"].read_text().splitlines()
                if not l.startswith("#")
            ]
            assert len(lines) == B2M_LENGTH

    def test_csv_metadata_parseable(self):
        s = _make_series({13: 0.6})
        arr = build_condition_array(s)
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = save_array(arr, tmpdir, "test_run", conditions=["test"])
            meta = read_csv_metadata(paths["csv"])
            assert meta["protein"] == "beta-2-microglobulin"
            assert meta["sequence_length"] == B2M_LENGTH

    def test_save_2d_array(self):
        s1 = _make_series({13: 0.6})
        s2 = _make_series({13: 0.2})
        df = pd.DataFrame({"no_cu": s1, "cu": s2})
        arr = build_multi_condition_array(df)
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = save_array(arr, tmpdir, "multi", conditions=["no_cu", "cu"])
            loaded = load_array(paths["npy"])
            assert loaded.shape == (B2M_LENGTH, 2)


class TestProtectionArray:
    def test_protection_array_shape(self):
        prot_df = pd.DataFrame(
            {"protection_score": {13: 0.7, 31: 0.5, 51: 0.65}},
        )
        arr = build_protection_array(prot_df)
        assert arr.shape == (B2M_LENGTH,)

    def test_his13_protection_value(self):
        prot_df = pd.DataFrame({"protection_score": {13: 0.7}})
        arr = build_protection_array(prot_df)
        assert abs(arr[12] - 0.7) < 1e-6

    def test_non_susceptible_zero(self):
        prot_df = pd.DataFrame({"protection_score": {13: 0.7}})
        arr = build_protection_array(prot_df, fill=0.0)
        # Position 1 (I) — not susceptible
        assert arr[0] == 0.0
