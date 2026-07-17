"""
β2m DEPC parser — end-to-end pipeline runner.

Usage:
    python -m src.pipeline --input data/raw/ --output data/arrays/ --mode high_res
    python -m src.pipeline --input data/raw/ --output data/arrays/ --mode auto
    python -m src.pipeline --help
"""

from __future__ import annotations

import cProfile
import logging
import pstats
import sys
import time
from pathlib import Path

import click
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


@click.command()
@click.option(
    "--input", "input_dir",
    required=True,
    type=click.Path(exists=True),
    help="Directory containing .mzML files.",
)
@click.option(
    "--output", "output_dir",
    required=True,
    type=click.Path(),
    help="Directory for output arrays (.npy/.csv).",
)
@click.option(
    "--mode",
    default="auto",
    type=click.Choice(["auto", "high_res", "low_res"]),
    show_default=True,
    help="Mass tolerance mode. 'auto' detects from instrument metadata.",
)
@click.option(
    "--no-cu-pattern",
    default="no_cu",
    show_default=True,
    help="Substring in filename identifying no-Cu(II) control files.",
)
@click.option(
    "--cu-pattern",
    default="cu",
    show_default=True,
    help="Substring in filename identifying +Cu(II) files.",
)
@click.option(
    "--workers",
    default=4,
    type=int,
    show_default=True,
    help="Number of parallel workers for MS2 processing.",
)
@click.option(
    "--profile",
    is_flag=True,
    default=False,
    help="Run cProfile and print top-20 slowest functions.",
)
def main(
    input_dir: str,
    output_dir: str,
    mode: str,
    no_cu_pattern: str,
    cu_pattern: str,
    workers: int,
    profile: bool,
) -> None:
    """β2m DEPC covalent labeling LC-MS/MS pipeline."""
    if profile:
        pr = cProfile.Profile()
        pr.enable()

    t0 = time.perf_counter()
    _run_pipeline(input_dir, output_dir, mode, no_cu_pattern, cu_pattern, workers)
    elapsed = time.perf_counter() - t0
    logger.info("Pipeline complete in %.1f s", elapsed)

    if profile:
        pr.disable()
        stats = pstats.Stats(pr, stream=sys.stdout)
        stats.sort_stats("cumulative")
        stats.print_stats(20)


def _run_pipeline(
    input_dir: str,
    output_dir: str,
    mode: str,
    no_cu_pattern: str,
    cu_pattern: str,
    workers: int,
) -> None:
    from .mzml_loader import load_mzml, get_ms1_scans, get_ms2_scans
    from .peak_picker import pick_all_ms1_peaks
    from .depc_hunter import hunt_depc_with_intensities
    from .labeling_quantifier import quantify_condition
    from .cu_protection import compute_protection_scores
    from .array_formatter import (
        build_condition_array,
        build_multi_condition_array,
        build_protection_array,
        save_array,
    )

    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    mzml_files = sorted(in_path.glob("*.mzML")) + sorted(in_path.glob("*.mzml"))
    if not mzml_files:
        logger.error("No .mzML files found in %s", input_dir)
        sys.exit(1)

    logger.info("Found %d mzML files", len(mzml_files))

    no_cu_events: list = []
    cu_events: list = []
    condition_events: dict[str, list] = {}
    all_series: dict[str, pd.Series] = {}

    for fpath in mzml_files:
        stem = fpath.stem
        logger.info("── Processing %s", stem)

        scans, high_res = load_mzml(fpath)

        if mode == "high_res":
            high_res = True
        elif mode == "low_res":
            high_res = False

        ms1_scans = get_ms1_scans(scans)
        ms1_peaks = pick_all_ms1_peaks(ms1_scans)
        logger.info("  %d MS1 peaks", len(ms1_peaks))

        events = hunt_depc_with_intensities(ms1_peaks, scans, high_res, workers)
        logger.info("  %d DEPC events confirmed", len(events))

        series = quantify_condition(events, condition_label=stem)
        all_series[stem] = series
        condition_events[stem] = events

        is_no_cu = no_cu_pattern.lower() in stem.lower()
        is_cu = cu_pattern.lower() in stem.lower() and not is_no_cu
        if is_no_cu:
            no_cu_events.extend(events)
        elif is_cu:
            cu_events.extend(events)

        arr_1d = build_condition_array(series)
        save_array(arr_1d, out_path, name=stem, conditions=[stem])

    # Multi-condition array
    if all_series:
        df_all = pd.DataFrame(all_series)
        df_all.index = pd.RangeIndex(1, len(df_all) + 1)
        arr_2d = build_multi_condition_array(df_all)
        save_array(
            arr_2d,
            out_path,
            name="all_conditions",
            conditions=list(all_series.keys()),
        )
        df_all.to_csv(out_path / "labeling_extents.csv")
        logger.info("Saved labeling_extents.csv")

    # Cu(II) protection analysis
    if no_cu_events and cu_events:
        from .labeling_quantifier import quantify_condition
        from .cu_protection import compute_protection_scores

        no_cu_series = quantify_condition(no_cu_events, "no_cu")
        cu_series = quantify_condition(cu_events, "cu")

        df_no_cu = no_cu_series.to_frame(name="r0")
        df_cu = cu_series.to_frame(name="r0")

        prot_df = compute_protection_scores(df_no_cu, df_cu)
        prot_df.to_csv(out_path / "cu_protection.csv")
        logger.info("Saved cu_protection.csv")

        prot_arr = build_protection_array(prot_df)
        save_array(prot_arr, out_path, name="cu_protection_scores", conditions=["protection_score"])
    else:
        logger.warning(
            "Could not find matching +Cu / -Cu file pairs "
            "(patterns: '%s' / '%s'). Skipping protection analysis.",
            no_cu_pattern, cu_pattern,
        )


if __name__ == "__main__":
    main()
