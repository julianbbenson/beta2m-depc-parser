"""
Format pipeline results as numpy arrays for PyTorch ML consumption.

Output:
  - 1D array (B2M_LENGTH,) for a single condition
  - 2D array (B2M_LENGTH, N_conditions) for all conditions
  - .npy binary files and .csv with metadata header
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import B2M_LENGTH, B2M_RESIDUES, SUSCEPTIBLE_RESIDUES
from .validator import validate_output_array

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Array construction
# ---------------------------------------------------------------------------

def build_condition_array(
    labeling_series: pd.Series,
    fill_non_susceptible: float = 0.0,
    fill_undetected: float = 0.0,
) -> np.ndarray:
    """
    Convert a per-residue pd.Series (index 1..B2M_LENGTH) into a numpy array.

    Parameters:
        labeling_series: Series indexed 1..B2M_LENGTH with labeling extents.
        fill_non_susceptible: value for non-DEPC-susceptible residues (default 0).
        fill_undetected: value for susceptible-but-not-detected residues (default 0).

    Returns array of shape (B2M_LENGTH,).
    """
    arr = np.full(B2M_LENGTH, fill_non_susceptible, dtype=np.float32)

    for pos in range(1, B2M_LENGTH + 1):
        aa = B2M_RESIDUES.get(pos, "?")
        if aa in SUSCEPTIBLE_RESIDUES:
            val = labeling_series.get(pos, np.nan)
            arr[pos - 1] = float(val) if not np.isnan(val) else fill_undetected

    validate_output_array(arr)
    return arr


def build_multi_condition_array(
    labeling_df: pd.DataFrame,
    **kwargs: Any,
) -> np.ndarray:
    """
    Build (B2M_LENGTH, N_conditions) array from a multi-condition DataFrame.

    Columns of *labeling_df* are condition labels; rows are residue positions.
    Returns shape (B2M_LENGTH, N) float32.
    """
    arrays = [
        build_condition_array(labeling_df[col], **kwargs)
        for col in labeling_df.columns
    ]
    return np.stack(arrays, axis=1)  # (B2M_LENGTH, N)


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def _build_metadata(
    conditions: list[str],
    extra: dict | None = None,
) -> dict:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protein": "beta-2-microglobulin",
        "uniprot": "P61769",
        "sequence_length": B2M_LENGTH,
        "modification": "DEPC_carbethoxylation_+72.0211Da",
        "conditions": conditions,
        "array_shape": f"({B2M_LENGTH}, {len(conditions)})" if len(conditions) > 1 else f"({B2M_LENGTH},)",
        **(extra or {}),
    }


def save_array(
    arr: np.ndarray,
    output_dir: str | Path,
    name: str,
    conditions: list[str] | None = None,
    metadata: dict | None = None,
) -> dict[str, Path]:
    """
    Save *arr* as both .npy and .csv.

    The CSV has a JSON metadata header in comment lines (starting with #)
    followed by comma-separated data.

    Returns dict with keys 'npy' and 'csv' pointing to saved paths.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npy_path = out_dir / f"{name}.npy"
    csv_path = out_dir / f"{name}.csv"

    np.save(str(npy_path), arr)
    logger.info("Saved %s", npy_path)

    # CSV: metadata in comment header, then data
    conds = conditions or [f"cond_{i}" for i in range(arr.shape[1] if arr.ndim == 2 else 1)]
    meta = _build_metadata(conds, metadata)

    with open(csv_path, "w") as f:
        f.write(f"# METADATA: {json.dumps(meta)}\n")
        f.write("# residue_position,residue,susceptible," + ",".join(conds) + "\n")

        data_2d = arr if arr.ndim == 2 else arr[:, np.newaxis]
        for i in range(B2M_LENGTH):
            pos = i + 1
            aa = B2M_RESIDUES.get(pos, "?")
            susc = "1" if aa in SUSCEPTIBLE_RESIDUES else "0"
            row_vals = ",".join(f"{data_2d[i, j]:.6f}" for j in range(data_2d.shape[1]))
            f.write(f"{pos},{aa},{susc},{row_vals}\n")

    logger.info("Saved %s", csv_path)
    return {"npy": npy_path, "csv": csv_path}


def load_array(path: str | Path) -> np.ndarray:
    """Load a .npy array saved by *save_array*."""
    return np.load(str(path))


def read_csv_metadata(path: str | Path) -> dict:
    """Parse the JSON metadata from the comment header of a pipeline CSV."""
    with open(path) as f:
        for line in f:
            if line.startswith("# METADATA:"):
                return json.loads(line[len("# METADATA:"):].strip())
            if not line.startswith("#"):
                break
    return {}


# ---------------------------------------------------------------------------
# Protection-score array
# ---------------------------------------------------------------------------

def build_protection_array(
    protection_df: pd.DataFrame,
    column: str = "protection_score",
    fill: float = 0.0,
) -> np.ndarray:
    """
    Build a (B2M_LENGTH,) float32 array of protection scores.

    Non-susceptible residues → *fill*.
    """
    arr = np.full(B2M_LENGTH, fill, dtype=np.float32)
    for pos in range(1, B2M_LENGTH + 1):
        aa = B2M_RESIDUES.get(pos, "?")
        if aa in SUSCEPTIBLE_RESIDUES and pos in protection_df.index:
            val = protection_df.loc[pos, column]
            if not np.isnan(val):
                arr[pos - 1] = float(val)
    return arr
