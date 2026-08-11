#!/usr/bin/env python3
"""Validate and plot completed chromosomes from a Gamma-SMC sanity run."""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import os
import tempfile
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

# Notebook kernels commonly export ``module://matplotlib_inline.backend_inline``.
# Batch plotting must not inherit a backend that may be absent from the uv environment.
os.environ["MPLBACKEND"] = "Agg"

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from phase2_map import parse_chroms  # noqa: E402
from plot_gamma_smc_sanity import (  # noqa: E402
    ProfileBundle,
    _plot_style,
    _quantile_summary,
    _save_figure,
    load_cutoff,
    load_profiles,
)
from run_sim import atomic_text, sha256_file  # noqa: E402

ANALYSIS_SCHEMA = "simulatephase2.gamma-smc-sanity-genomewide-analysis/v1"
FIGURE_SIZE = (11.0, 8.5)


@dataclass(frozen=True)
class GenomeBundle:
    chromosomes: tuple[str, ...]
    chromosome_lengths: np.ndarray
    boundaries: np.ndarray
    chromosome_slices: tuple[slice, ...]
    positions: np.ndarray
    ends: np.ndarray
    values: np.ndarray
    cutoff: np.ndarray
    n_pairs: int
    decode_seconds: np.ndarray
    profile_paths: tuple[Path, ...]
    cutoff_h5_path: Path
    cutoff_group_attributes: dict[str, dict[str, object]]


def load_genome(
    sanity_dir: Path,
    *,
    population: str,
    chromosomes: tuple[str, ...],
    n_sims: int,
) -> GenomeBundle:
    """Load every requested chromosome using the chromosome-level validator."""
    profile_bundles: list[ProfileBundle] = []
    cutoff_arrays: list[np.ndarray] = []
    positions: list[np.ndarray] = []
    ends: list[np.ndarray] = []
    lengths: list[int] = []
    boundaries = [0]
    slices: list[slice] = []
    profile_paths: list[Path] = []
    decode_seconds: list[np.ndarray] = []
    group_attributes: dict[str, dict[str, object]] = {}
    pair_counts: set[int] = set()
    cutoff_paths: set[Path] = set()
    offset = 0
    output_start = 0

    for chromosome in chromosomes:
        print(
            f"[gamma-sanity-genomewide] loading={population}/{chromosome} profiles={n_sims}",
            flush=True,
        )
        profiles = load_profiles(
            sanity_dir,
            population=population,
            chromosome=chromosome,
            n_sims=n_sims,
        )
        cutoff = load_cutoff(
            sanity_dir,
            population=population,
            chromosome=chromosome,
            profiles=profiles,
        )
        if (
            len(profiles.positions) == 0
            or cutoff.ends.shape != profiles.positions.shape
            or np.any(np.diff(profiles.positions) <= 0)
            or np.any(cutoff.ends <= profiles.positions)
            or np.any(np.diff(cutoff.ends) <= 0)
        ):
            raise ValueError(f"invalid position/end geometry for {population} {chromosome}")
        chromosome_length = int(cutoff.ends[-1])
        if chromosome_length <= 0:
            raise ValueError(f"invalid chromosome length for {population} {chromosome}")

        count = len(profiles.positions)
        profile_bundles.append(profiles)
        cutoff_arrays.append(cutoff.cutoff)
        positions.append(profiles.positions + offset)
        ends.append(cutoff.ends + offset)
        lengths.append(chromosome_length)
        slices.append(slice(output_start, output_start + count))
        profile_paths.extend(profiles.paths)
        decode_seconds.append(profiles.decode_seconds)
        group_attributes[chromosome] = cutoff.group_attributes
        pair_counts.add(profiles.n_pairs)
        cutoff_paths.add(cutoff.h5_path)

        offset += chromosome_length
        output_start += count
        boundaries.append(offset)

    if len(pair_counts) != 1:
        raise ValueError("chromosomes do not share one positive pair count")
    if len(cutoff_paths) != 1:
        raise ValueError("chromosomes do not share one cutoff HDF5")

    return GenomeBundle(
        chromosomes=chromosomes,
        chromosome_lengths=np.asarray(lengths, dtype=np.int64),
        boundaries=np.asarray(boundaries, dtype=np.int64),
        chromosome_slices=tuple(slices),
        positions=np.concatenate(positions),
        ends=np.concatenate(ends),
        values=np.concatenate([bundle.values for bundle in profile_bundles], axis=1),
        cutoff=np.concatenate(cutoff_arrays),
        n_pairs=next(iter(pair_counts)),
        decode_seconds=np.stack(decode_seconds),
        profile_paths=tuple(profile_paths),
        cutoff_h5_path=next(iter(cutoff_paths)),
        cutoff_group_attributes=group_attributes,
    )


def _title(bundle: GenomeBundle, population: str) -> str:
    return (
        f"{population} genome-wide Gamma-SMC sanity calibration — "
        f"{len(bundle.chromosomes)} chromosomes, {len(bundle.values)} simulations "
        f"× {bundle.n_pairs:,} pairs"
    )


def _separate_chromosomes(
    bundle: GenomeBundle, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Insert NaNs so plotted lines never connect adjacent chromosomes."""
    values = np.asarray(values)
    if values.shape[-1] != len(bundle.positions):
        raise ValueError("plot values do not match the genome-wide position grid")
    x_parts: list[np.ndarray] = []
    value_parts: list[np.ndarray] = []
    for index, chromosome_slice in enumerate(bundle.chromosome_slices):
        x_parts.append(bundle.positions[chromosome_slice] / 1_000_000.0)
        value_parts.append(values[..., chromosome_slice])
        if index + 1 < len(bundle.chromosome_slices):
            x_parts.append(np.asarray([np.nan]))
            separator_shape = values.shape[:-1] + (1,)
            value_parts.append(np.full(separator_shape, np.nan, dtype=np.float64))
    return np.concatenate(x_parts), np.concatenate(value_parts, axis=-1)


def _configure_genome_axis(
    axis: plt.Axes,
    bundle: GenomeBundle,
    *,
    shade: bool = True,
) -> None:
    boundaries = bundle.boundaries / 1_000_000.0
    if shade:
        for index, (left, right) in enumerate(zip(boundaries[:-1], boundaries[1:], strict=True)):
            if index % 2:
                axis.axvspan(left, right, color="black", alpha=0.025, zorder=0)
    for boundary in boundaries[1:-1]:
        axis.axvline(boundary, color="grey", linewidth=0.45, alpha=0.55, zorder=1)
    midpoints = (boundaries[:-1] + boundaries[1:]) / 2.0
    axis.set_xlim(float(boundaries[0]), float(boundaries[-1]))
    axis.set_xticks(midpoints, labels=[chromosome[3:] for chromosome in bundle.chromosomes])
    axis.set_xlabel("Chromosome (GRCh38; concatenated)")


def plot_null_profiles(
    output_dir: Path,
    *,
    population: str,
    bundle: GenomeBundle,
    dpi: int,
) -> list[Path]:
    x, values = _separate_chromosomes(bundle, bundle.values)
    figure, axis = plt.subplots(figsize=FIGURE_SIZE, constrained_layout=True)
    _configure_genome_axis(axis, bundle)
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 1.0, len(bundle.values)))
    for simulation, (row, color) in enumerate(zip(values, colors, strict=True)):
        axis.plot(
            x,
            row,
            color=color,
            linewidth=0.45,
            alpha=0.75,
            label=f"sim {simulation}",
            rasterized=True,
        )
    axis.set_ylabel("Mean posterior P(TMRCA < 4,500 y)")
    figure.suptitle(_title(bundle, population))
    axis.set_title("Ten independently simulated null profiles")
    axis.legend(ncol=5, frameon=False, loc="upper right")
    axis.grid(axis="y", alpha=0.2, linewidth=0.5)
    return _save_figure(figure, output_dir, "genomewide_null_profiles", dpi)


def plot_across_simulation_summary(
    output_dir: Path,
    *,
    population: str,
    bundle: GenomeBundle,
    dpi: int,
) -> list[Path]:
    mean = bundle.values.mean(axis=0)
    q05, median, q95 = np.quantile(bundle.values, [0.05, 0.5, 0.95], axis=0)
    x, separated = _separate_chromosomes(
        bundle,
        np.stack([q05, q95, mean, median, bundle.cutoff]),
    )
    q05_plot, q95_plot, mean_plot, median_plot, cutoff_plot = separated
    figure, axis = plt.subplots(figsize=FIGURE_SIZE, constrained_layout=True)
    _configure_genome_axis(axis, bundle)
    axis.fill_between(
        x,
        q05_plot,
        q95_plot,
        alpha=0.25,
        label="5th–95th percentile",
        rasterized=True,
    )
    axis.plot(x, mean_plot, linewidth=0.7, label="mean", rasterized=True)
    axis.plot(x, median_plot, linewidth=0.7, label="median", rasterized=True)
    axis.plot(
        x,
        cutoff_plot,
        color="black",
        linewidth=0.6,
        label="p≤0.1 cutoff (maximum)",
        rasterized=True,
    )
    axis.set_ylabel("Posterior probability")
    figure.suptitle(_title(bundle, population))
    axis.set_title("Across-simulation summary and exact pointwise cutoff")
    axis.legend(ncol=2, frameon=False, loc="upper right")
    axis.grid(axis="y", alpha=0.2, linewidth=0.5)
    return _save_figure(figure, output_dir, "genomewide_across_simulation_summary", dpi)


def plot_heatmap(
    output_dir: Path,
    *,
    population: str,
    bundle: GenomeBundle,
    dpi: int,
) -> list[Path]:
    total_mb = float(bundle.boundaries[-1] / 1_000_000.0)
    upper = max(float(np.quantile(bundle.values, 0.995)), np.finfo(float).eps)
    figure, axis = plt.subplots(figsize=FIGURE_SIZE, constrained_layout=True)
    image = axis.imshow(
        bundle.values,
        aspect="auto",
        interpolation="nearest",
        extent=[0.0, total_mb, len(bundle.values) - 0.5, -0.5],
        cmap="viridis",
        vmin=0.0,
        vmax=upper,
        rasterized=True,
    )
    _configure_genome_axis(axis, bundle, shade=False)
    axis.set_ylabel("Simulation index")
    axis.set_yticks(np.arange(len(bundle.values)))
    figure.suptitle(_title(bundle, population))
    axis.set_title("Spatial consistency across null profiles")
    colorbar = figure.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label("Mean posterior P(TMRCA < 4,500 y)")
    return _save_figure(figure, output_dir, "genomewide_profile_heatmap", dpi)


def plot_distributions(
    output_dir: Path,
    *,
    population: str,
    bundle: GenomeBundle,
    dpi: int,
) -> list[Path]:
    figure, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE, constrained_layout=True)
    axes[0].boxplot([row for row in bundle.values], showfliers=False)
    axes[0].set_xticks(
        np.arange(1, len(bundle.values) + 1),
        labels=[str(index) for index in range(len(bundle.values))],
    )
    axes[0].set_xlabel("Simulation index")
    axes[0].set_ylabel("Posterior probability across genome")
    axes[0].set_title("Per-simulation distributions")
    axes[0].grid(axis="y", alpha=0.2, linewidth=0.5)

    axes[1].hist(
        bundle.values.ravel(),
        bins=80,
        density=True,
        alpha=0.55,
        label="all null profile values",
    )
    axes[1].hist(
        bundle.cutoff,
        bins=80,
        density=True,
        histtype="step",
        linewidth=1.8,
        color="black",
        label="pointwise p≤0.1 cutoff",
    )
    axes[1].set_xlabel("Mean posterior P(TMRCA < 4,500 y)")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Null values versus maximum-null cutoff")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.2, linewidth=0.5)
    figure.suptitle(_title(bundle, population))
    return _save_figure(figure, output_dir, "genomewide_profile_distributions", dpi)


def _atomic_gzip_rows(path: Path, rows: Iterable[Sequence[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                    writer = csv.writer(text, delimiter="\t", lineterminator="\n")
                    writer.writerows(rows)
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def write_tables(
    output_dir: Path,
    *,
    bundle: GenomeBundle,
) -> tuple[Path, Path, Path, list[dict[str, object]], list[dict[str, object]]]:
    per_simulation: list[dict[str, object]] = []
    simulation_lines = [
        "\t".join(
            [
                "simulation",
                "n_chromosomes",
                "n_positions",
                "n_pairs",
                "decode_process_wall_seconds_sum",
                "minimum",
                "q01",
                "q05",
                "median",
                "mean",
                "sd",
                "q95",
                "q99",
                "maximum",
            ]
        )
    ]
    for simulation, values in enumerate(bundle.values):
        summary: dict[str, object] = _quantile_summary(values)
        seconds = bundle.decode_seconds[:, simulation]
        finite_seconds = seconds[np.isfinite(seconds)]
        decode_sum = float(finite_seconds.sum()) if len(finite_seconds) else None
        summary.update(
            {
                "simulation": simulation,
                "decode_process_wall_seconds_sum": decode_sum,
            }
        )
        per_simulation.append(summary)
        simulation_lines.append(
            "\t".join(
                map(
                    str,
                    [
                        simulation,
                        len(bundle.chromosomes),
                        len(bundle.positions),
                        bundle.n_pairs,
                        f"{decode_sum:.6f}" if decode_sum is not None else "nan",
                        *[
                            f"{float(summary[key]):.9g}"
                            for key in (
                                "min",
                                "q01",
                                "q05",
                                "median",
                                "mean",
                                "sd",
                                "q95",
                                "q99",
                                "max",
                            )
                        ],
                    ],
                )
            )
        )
    per_simulation_path = output_dir / "per_simulation_summary.tsv"
    atomic_text(per_simulation_path, "\n".join(simulation_lines) + "\n")

    per_chromosome: list[dict[str, object]] = []
    chromosome_lines = [
        "\t".join(
            [
                "chromosome",
                "length_bp",
                "n_positions",
                "null_mean",
                "null_median",
                "null_q99",
                "null_max",
                "cutoff_mean",
                "cutoff_q99",
                "cutoff_max",
            ]
        )
    ]
    for index, (chromosome, chromosome_slice) in enumerate(
        zip(bundle.chromosomes, bundle.chromosome_slices, strict=True)
    ):
        values = bundle.values[:, chromosome_slice]
        cutoff = bundle.cutoff[chromosome_slice]
        record: dict[str, object] = {
            "chromosome": chromosome,
            "length_bp": int(bundle.chromosome_lengths[index]),
            "n_positions": int(values.shape[1]),
            "null_mean": float(values.mean()),
            "null_median": float(np.median(values)),
            "null_q99": float(np.quantile(values, 0.99)),
            "null_max": float(values.max()),
            "cutoff_mean": float(cutoff.mean()),
            "cutoff_q99": float(np.quantile(cutoff, 0.99)),
            "cutoff_max": float(cutoff.max()),
        }
        per_chromosome.append(record)
        chromosome_lines.append(
            "\t".join(
                [
                    chromosome,
                    str(record["length_bp"]),
                    str(record["n_positions"]),
                    *[
                        f"{float(record[key]):.9g}"
                        for key in (
                            "null_mean",
                            "null_median",
                            "null_q99",
                            "null_max",
                            "cutoff_mean",
                            "cutoff_q99",
                            "cutoff_max",
                        )
                    ],
                ]
            )
        )
    per_chromosome_path = output_dir / "per_chromosome_summary.tsv"
    atomic_text(per_chromosome_path, "\n".join(chromosome_lines) + "\n")

    null_min = bundle.values.min(axis=0)
    q05, median, q95 = np.quantile(bundle.values, [0.05, 0.5, 0.95], axis=0)
    null_mean = bundle.values.mean(axis=0)
    null_max = bundle.values.max(axis=0)

    def position_rows() -> Iterator[list[object]]:
        yield [
            "chromosome",
            "position_0based",
            "end",
            "null_min",
            "null_q05",
            "null_median",
            "null_mean",
            "null_q95",
            "null_max",
            "p_le_0.1_cutoff",
        ]
        for chromosome_index, (chromosome, chromosome_slice) in enumerate(
            zip(bundle.chromosomes, bundle.chromosome_slices, strict=True)
        ):
            offset = int(bundle.boundaries[chromosome_index])
            for global_index in range(chromosome_slice.start, chromosome_slice.stop):
                yield [
                    chromosome,
                    int(bundle.positions[global_index] - offset),
                    int(bundle.ends[global_index] - offset),
                    f"{float(null_min[global_index]):.9g}",
                    f"{float(q05[global_index]):.9g}",
                    f"{float(median[global_index]):.9g}",
                    f"{float(null_mean[global_index]):.9g}",
                    f"{float(q95[global_index]):.9g}",
                    f"{float(null_max[global_index]):.9g}",
                    f"{float(bundle.cutoff[global_index]):.9g}",
                ]

    position_path = output_dir / "position_summary.tsv.gz"
    _atomic_gzip_rows(position_path, position_rows())
    return (
        per_simulation_path,
        per_chromosome_path,
        position_path,
        per_simulation,
        per_chromosome,
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    result.add_argument("--sanity-dir", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, default=None)
    result.add_argument("--population", default="AFR")
    result.add_argument("--chroms", default="1-22")
    result.add_argument("--n-sims", type=int, default=10)
    result.add_argument("--dpi", type=int, default=220)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.n_sims != 10 or args.dpi <= 0:
        raise SystemExit("--n-sims must equal 10 for the p<=0.1 maximum-null audit; --dpi > 0")
    population = args.population.strip().upper()
    try:
        chromosomes = tuple(parse_chroms(args.chroms))
    except ValueError as error:
        raise SystemExit(f"Gamma-SMC genome-wide analysis failed: {error}") from error
    sanity_dir = args.sanity_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else sanity_dir / "diagnostics" / f"{population.lower()}_genomewide"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        bundle = load_genome(
            sanity_dir,
            population=population,
            chromosomes=chromosomes,
            n_sims=args.n_sims,
        )
        print("[gamma-sanity-genomewide] phase=tables", flush=True)
        (
            per_simulation_path,
            per_chromosome_path,
            position_path,
            per_simulation,
            per_chromosome,
        ) = write_tables(output_dir, bundle=bundle)
        print("[gamma-sanity-genomewide] phase=figures", flush=True)
        _plot_style()
        figure_paths = [
            *plot_null_profiles(
                output_dir,
                population=population,
                bundle=bundle,
                dpi=args.dpi,
            ),
            *plot_across_simulation_summary(
                output_dir,
                population=population,
                bundle=bundle,
                dpi=args.dpi,
            ),
            *plot_heatmap(
                output_dir,
                population=population,
                bundle=bundle,
                dpi=args.dpi,
            ),
            *plot_distributions(
                output_dir,
                population=population,
                bundle=bundle,
                dpi=args.dpi,
            ),
        ]
    except (FileNotFoundError, KeyError, OSError, TypeError, ValueError) as error:
        raise SystemExit(f"Gamma-SMC genome-wide analysis failed: {error}") from error

    finite_seconds = bundle.decode_seconds[np.isfinite(bundle.decode_seconds)]
    print("[gamma-sanity-genomewide] phase=input-provenance", flush=True)
    profile_inputs: list[dict[str, object]] = []
    path_index = 0
    for chromosome in bundle.chromosomes:
        for simulation in range(len(bundle.values)):
            path = bundle.profile_paths[path_index]
            path_index += 1
            profile_inputs.append(
                {
                    "chromosome": chromosome,
                    "simulation": simulation,
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    manifest = {
        "schema": ANALYSIS_SCHEMA,
        "sanity_dir": str(sanity_dir),
        "output_dir": str(output_dir),
        "population": population,
        "chromosomes": list(bundle.chromosomes),
        "chromosome_lengths_bp": {
            chromosome: int(length)
            for chromosome, length in zip(
                bundle.chromosomes, bundle.chromosome_lengths, strict=True
            )
        },
        "n_chromosomes": len(bundle.chromosomes),
        "n_simulations": len(bundle.values),
        "n_positions": len(bundle.positions),
        "n_pairs_per_simulation": bundle.n_pairs,
        "validation": {
            "all_requested_profiles_present_and_valid": True,
            "all_requested_cutoff_groups_complete": True,
            "shared_pair_count": True,
            "each_cutoff_equals_its_pointwise_profile_maximum": True,
            "chromosomes_concatenated_in_requested_order": True,
        },
        "profile_distribution": _quantile_summary(bundle.values.ravel()),
        "pointwise_cutoff_distribution": _quantile_summary(bundle.cutoff),
        "decode_process_wall_seconds": {
            "note": "per-process wall times overlap when decode workers are greater than one",
            "available_count": int(len(finite_seconds)),
            "minimum": float(finite_seconds.min()) if len(finite_seconds) else None,
            "median": float(np.median(finite_seconds)) if len(finite_seconds) else None,
            "maximum": float(finite_seconds.max()) if len(finite_seconds) else None,
            "sum_not_job_wall_time": float(finite_seconds.sum()) if len(finite_seconds) else None,
        },
        "per_simulation_distribution": per_simulation,
        "per_chromosome_distribution": per_chromosome,
        "inputs": {
            "profiles": profile_inputs,
            "cutoff_h5": {
                "path": str(bundle.cutoff_h5_path),
                "sha256": sha256_file(bundle.cutoff_h5_path),
                "size_bytes": bundle.cutoff_h5_path.stat().st_size,
                "group_attributes": bundle.cutoff_group_attributes,
            },
        },
        "outputs": {
            "per_simulation_summary": str(per_simulation_path),
            "per_chromosome_summary": str(per_chromosome_path),
            "position_summary": str(position_path),
            "figures": [str(path) for path in figure_paths],
        },
    }
    manifest_path = output_dir / "analysis.json"
    atomic_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    checksum_paths = [
        per_simulation_path,
        per_chromosome_path,
        position_path,
        *figure_paths,
        manifest_path,
    ]
    checksum_text = "".join(
        f"{sha256_file(path)}  {path.relative_to(output_dir).as_posix()}\n"
        for path in checksum_paths
    )
    checksum_path = output_dir / "checksums.sha256"
    atomic_text(checksum_path, checksum_text)
    print(
        f"[gamma-sanity-genomewide] validated={len(bundle.chromosomes)} chromosomes "
        f"profiles={len(bundle.values)} positions={len(bundle.positions):,} "
        f"pairs={bundle.n_pairs:,}"
    )
    print(f"[gamma-sanity-genomewide] output_dir={output_dir}")
    print(f"[gamma-sanity-genomewide] manifest={manifest_path}")
    print(f"[gamma-sanity-genomewide] checksums={checksum_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
