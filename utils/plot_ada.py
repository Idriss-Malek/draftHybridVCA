#!/usr/bin/env python3
"""
CSV Energy Plotter
------------------
Given a folder containing CSVs named in the format {algo}{seed}_{param}.csv, this script:

1) Aggregates (per {algo}_{param}) the columns `minE` and `meanE` across all seeds:
   - Seeds may have different row counts; shorter seeds are padded with their last value
     up to the longest seed length BEFORE averaging.
   - Plots averages as two multi-curve figures: `minEPlot.png` and `meanEPlot.png`.
     Each curve is labeled `{algo}_{param}`. Curves may end earlier than others.
     Y-axis is logarithmic base 10. Each curve is shown as a moving-average (window=20)
     with the raw (non-smoothed) curve overlaid in the same color at low opacity.

2) For each CSV that contains `DeltaT`, creates a twin-y plot named
   `VTT_{algo}{seed}_{param}.png`, where x=Temperature, left y=varE, right y=DeltaT.

All plots are saved to a folder named `plots` at the SAME LEVEL as the CSV folder.

Usage
-----
python csv_energy_plotter.py /path/to/csv_folder

Dependencies
------------
pip install pandas matplotlib numpy
"""

import argparse
import re
import os
import glob
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


FILENAME_RE = re.compile(r"^(?P<algo>[A-Za-z][A-Za-z0-9\-]*)?(?P<seed>\d+)_+(?P<param>[^\.]+)\.csv$")


def parse_filename(filename: str) -> Tuple[str, str, str]:
    """Parse {algo}{seed}_{param}.csv -> (algo, seed, param).
    Accepts algo with letters/digits/hyphens, numeric seed, and any param (no dots).
    Raises ValueError if pattern doesn't match.
    """
    name = os.path.basename(filename)
    m = FILENAME_RE.match(name)
    if not m:
        raise ValueError(f"Filename does not match {{algo}}{{seed}}_{{param}}.csv pattern: {name}")
    algo = m.group("algo") or ""  # algo may be empty in edge-cases; treat as empty string
    seed = m.group("seed")
    param = m.group("param")
    return algo, seed, param


def find_column(df: pd.DataFrame, target: str) -> str:
    """Find a column by case-insensitive match; return actual column name or ''."""
    target_l = target.lower()
    for c in df.columns:
        if c.lower() == target_l:
            return c
    return ""


def pad_series_last_value(s: pd.Series, target_len: int) -> pd.Series:
    """Pad a Series with its last value to target_len (assumes len(s) > 0)."""
    if len(s) == 0:
        # degenerate case: if empty, pad zeros
        return pd.Series([0] * target_len, dtype=float)
    if len(s) >= target_len:
        return s.iloc[:target_len].reset_index(drop=True)
    last_val = s.iloc[-1]
    pad_len = target_len - len(s)
    padded = pd.concat([s.reset_index(drop=True), pd.Series([last_val] * pad_len, dtype=float)], ignore_index=True)
    return padded


def average_over_seeds(series_list: List[pd.Series]) -> pd.Series:
    """Given a list of Series (possibly of different lengths), pad each with its last
    value up to the max length, then compute the mean across seeds (row-wise)."""
    if not series_list:
        return pd.Series(dtype=float)
    max_len = max(len(s) for s in series_list)
    padded = [pad_series_last_value(s.astype(float), max_len) for s in series_list]
    df = pd.concat(padded, axis=1)
    return df.mean(axis=1)


def ensure_plots_dir(csv_dir: Path) -> Path:
    parent = csv_dir.parent
    plots_dir = parent / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir


def plot_aggregated_curves(curves: Dict[str, pd.Series], ylabel: str, out_path: Path, title: str, smooth_window: int = 3):
    """Plot smoothed (moving-average) curves on a log10 y-axis, with faint raw overlays.

    - `curves` maps label -> aggregated Series (already averaged across seeds).
    - Each curve is smoothed with a moving average of `smooth_window` (default 20).
    - Raw (non-smoothed) curve is plotted in the same color with high transparency.
    - Non-positive values are masked (not plotted) due to log scaling.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for label, series in sorted(curves.items()):  # consistent legend order
        x = np.arange(len(series))
        raw = pd.Series(series, dtype=float).reset_index(drop=True)

        # Mask non-finite and non-positive values for log-scale plotting
        raw_masked = raw.copy()
        raw_masked[~np.isfinite(raw_masked)] = np.nan
        raw_masked[raw_masked <= 0] = np.nan

        # Moving average smoothing
        smoothed = raw_masked.rolling(window=smooth_window, min_periods=1).mean()

        # Plot smoothed first to set the color
        (line_smoothed,) = ax.plot(x, smoothed.values, linewidth=2, label=label)
        color = line_smoothed.get_color()
        # Overlay raw in same color, high transparency
        ax.plot(x, raw_masked.values, linewidth=1, alpha=0.25, color=color)

    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    ax.set_yscale("log", base=10)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if curves:
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_vtt(df: pd.DataFrame, algo: str, seed: str, param: str, out_dir: Path):
    # Locate required columns (case-insensitive)
    col_temp = find_column(df, "Temperature")
    col_varE = find_column(df, "varE")
    col_delta = find_column(df, "DeltaT")
    if not (col_temp and col_varE and col_delta):
        return  # silently skip if any missing; upstream checked for DeltaT

    # Sort by Temperature to make the plot tidy
    d = df[[col_temp, col_varE, col_delta]].dropna()
    d = d.sort_values(by=col_temp)

    plt.figure(figsize=(10, 6))
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    ax1.plot(d[col_temp].values, d[col_varE].values, linewidth=2, label="varE")
    ax2.plot(d[col_temp].values, d[col_delta].values, linewidth=2, linestyle=":", label="DeltaT")

    ax1.set_xlabel("Temperature")
    ax1.set_ylabel("varE")
    ax2.set_ylabel("DeltaT")
    plt.title(f"varE & DeltaT vs Temperature — {algo}{seed}_{param}")

    # Build a combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    ax1.grid(True, alpha=0.3)
    plt.tight_layout()

    out_file = out_dir / f"VTT_{algo}{seed}_{param}.png"
    plt.savefig(out_file, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot aggregated minE/meanE and per-file VTT plots from CSVs.")
    parser.add_argument("csv_folder", type=str, help="Path to folder containing the CSV files")
    parser.add_argument("--offset", type=str, help="Offset to substract by")
    parser.add_argument("--factor", type=str, help="Factor to divide by")
    args = parser.parse_args()

    csv_dir = Path(args.csv_folder).expanduser().resolve()
    if not csv_dir.is_dir():
        raise SystemExit(f"Not a directory: {csv_dir}")

    plots_dir = ensure_plots_dir(csv_dir)

    # Gather files
    files = sorted(glob.glob(str(csv_dir / "*.csv")))
    if not files:
        raise SystemExit(f"No CSV files found in: {csv_dir}")

    # Group data per (algo, param) across seeds
    groups: Dict[Tuple[str, str], Dict[str, List[pd.Series]]] = {}
    # Track which files have DeltaT for VTT plots
    files_with_delta: List[Tuple[str, str, str, Path]] = []  # (algo, seed, param, path)

    problems = []

    for f in files:
        try:
            algo, seed, param = parse_filename(f)
        except ValueError as e:
            problems.append(str(e))
            continue

        try:
            df = pd.read_csv(f)
        except Exception as e:
            problems.append(f"Failed to read {os.path.basename(f)}: {e}")
            continue

        col_minE = find_column(df, "minE")
        col_meanE = find_column(df, "meanE")        


        if not (col_minE and col_meanE):
            problems.append(f"Missing minE/meanE in {os.path.basename(f)}; skipping for aggregation.")
        else:
            if args.offset: df[[col_minE, col_meanE]] = df[[col_minE, col_meanE]].astype(float) - float(args.offset)
            if args.factor: df[[col_minE, col_meanE]] = df[[col_minE, col_meanE]].astype(float) / float(args.factor)
            df[[col_minE, col_meanE]] = df[[col_minE, col_meanE]].clip(lower=10**-10)
            key = (algo, param)
            entry = groups.setdefault(key, {"minE": [], "meanE": []})
            entry["minE"].append(df[col_minE].astype(float).reset_index(drop=True))
            entry["meanE"].append(df[col_meanE].astype(float).reset_index(drop=True))

        # Track DeltaT plots
        if find_column(df, "DeltaT"):
            files_with_delta.append((algo, seed, param, Path(f)))

    # Build aggregated curves
    minE_curves: Dict[str, pd.Series] = {}
    meanE_curves: Dict[str, pd.Series] = {}

    for (algo, param), series_dict in groups.items():
        avg_minE = average_over_seeds(series_dict["minE"]) if series_dict["minE"] else pd.Series(dtype=float)
        avg_meanE = average_over_seeds(series_dict["meanE"]) if series_dict["meanE"] else pd.Series(dtype=float)
        label = f"{algo}_{param}" if algo else f"_{param}"
        if len(avg_minE) > 0:
            minE_curves[label] = avg_minE
        if len(avg_meanE) > 0:
            meanE_curves[label] = avg_meanE

    # Save aggregated plots
    plot_aggregated_curves(minE_curves, ylabel="minE", out_path=plots_dir / "minEPlot.png", title="Average minE per {algo}_{param}")
    plot_aggregated_curves(meanE_curves, ylabel="meanE", out_path=plots_dir / "meanEPlot.png", title="Average meanE per {algo}_{param}")

    # Per-file VTT plots where DeltaT exists
    for algo, seed, param, path in files_with_delta:
        try:
            df = pd.read_csv(path)
            plot_vtt(df, algo, seed, param, plots_dir)
        except Exception as e:
            problems.append(f"Failed VTT plot for {os.path.basename(path)}: {e}")

    # Print a small summary
    print(f"Saved aggregated plots to: {plots_dir / 'minEPlot.png'} and {plots_dir / 'meanEPlot.png'}")
    print(f"Saved {len(files_with_delta)} VTT plots (where DeltaT present) to: {plots_dir}")
    if problems:
        print("\nWarnings/Skips:")
        for p in problems:
            print(" - ", p)


if __name__ == "__main__":
    main()
