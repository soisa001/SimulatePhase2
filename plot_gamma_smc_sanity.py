#!/usr/bin/env python3
"""Validate and plot one completed chromosome from a Gamma-SMC sanity run."""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import h5py  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from phase2_map import canonical_chrom  # noqa: E402
from run_sim import atomic_text, sha256_file  # noqa: E402

STATISTIC = "mean_p_tmrca_lt_threshold"
ANALYSIS_SCHEMA = "simulatephase2.gamma-smc-sanity-chromosome-analysis/v2"
FIGURE_SIZE = (11.0, 8.5)


@dataclass(frozen=True)
class ProfileBundle:
    positions: np.ndarray
    values: np.ndarray
    n_pairs: int
    decode_seconds: np.ndarray
    paths: tuple[Path, ...]
    metadata: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class CutoffBundle:
    cutoff: np.ndarray
    ends: np.ndarray
    source: str
    h5_path: Path
    group_attributes: dict[str, object]


def _json_value(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def load_profiles(
    sanity_dir: Path,
    *,
    population: str,
    chromosome: str,
    n_sims: int,
) -> ProfileBundle:
    """Load and cross-check every atomic restart profile for one chromosome."""
    positions: np.ndarray | None = None
    values: list[np.ndarray] = []
    pair_counts: set[int] = set()
    seconds: list[float] = []
    paths: list[Path] = []
    metadata_records: list[dict[str, object]] = []
    for simulation in range(n_sims):
        path = (
            sanity_dir
            / "profiles"
            / population.lower()
            / f"sim_{simulation:05d}"
            / f"{chromosome}.npz"
        )
        if not path.is_file():
            raise FileNotFoundError(f"missing Gamma-SMC restart profile: {path}")
        try:
            with np.load(path, allow_pickle=False) as data:
                current_positions = data["position_0based"].astype(np.int64, copy=False)
                current_values = data[STATISTIC].astype(np.float32, copy=False)
                n_pairs = int(data["n_pairs"].item())
                metadata = json.loads(str(data["metadata_json"].item()))
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid Gamma-SMC restart profile {path}: {error}") from error
        if not isinstance(metadata, dict):
            raise ValueError(f"profile metadata is not a mapping: {path}")
        expected_metadata = {"simulation": simulation, "chromosome": chromosome}
        mismatches = {
            key: (metadata.get(key), expected)
            for key, expected in expected_metadata.items()
            if metadata.get(key) != expected
        }
        if mismatches:
            raise ValueError(f"profile metadata mismatch {mismatches}: {path}")
        if positions is None:
            positions = current_positions.copy()
        elif not np.array_equal(positions, current_positions):
            raise ValueError(f"profile stride grid differs: {path}")
        if (
            current_values.shape != current_positions.shape
            or not np.isfinite(current_values).all()
            or np.any((current_values < 0.0) | (current_values > 1.0))
            or n_pairs <= 0
        ):
            raise ValueError(f"profile values are invalid: {path}")
        pair_counts.add(n_pairs)
        decode_seconds = float(metadata.get("decode_seconds", np.nan))
        values.append(current_values.copy())
        seconds.append(decode_seconds)
        paths.append(path)
        metadata_records.append(metadata)
    if positions is None or len(pair_counts) != 1:
        raise ValueError("profiles do not share one positive pair count")
    return ProfileBundle(
        positions=positions,
        values=np.stack(values),
        n_pairs=next(iter(pair_counts)),
        decode_seconds=np.asarray(seconds, dtype=np.float64),
        paths=tuple(paths),
        metadata=tuple(metadata_records),
    )


def _read_cutoff_h5(
    path: Path,
    *,
    population: str,
    chromosome: str,
    profiles: ProfileBundle,
) -> CutoffBundle:
    with h5py.File(path, "r") as handle:
        if str(handle.attrs.get("population", "")) != population:
            raise ValueError(f"cutoff population metadata mismatch: {path}")
        if "p_value" not in handle or not np.array_equal(handle["p_value"][:], [0.1]):
            raise ValueError(f"cutoff HDF5 does not contain the sanity p=0.1 row: {path}")
        if chromosome not in handle:
            raise KeyError(f"cutoff group is not available yet: {population} {chromosome}")
        group = handle[chromosome]
        if not bool(group.attrs.get("complete", False)):
            raise ValueError(f"cutoff group is incomplete: {population} {chromosome}")
        if int(group.attrs.get("n_simulations", -1)) != len(profiles.values):
            raise ValueError(f"cutoff simulation count mismatch: {path}")
        if int(group.attrs.get("n_pairs", -1)) != profiles.n_pairs:
            raise ValueError(f"cutoff pair count mismatch: {path}")
        positions = group["position_0based"][:].astype(np.int64, copy=False)
        ends = group["end"][:].astype(np.int64, copy=False)
        cutoff_matrix = group["cutoff"][:].astype(np.float32, copy=False)
        if not np.array_equal(positions, profiles.positions):
            raise ValueError(f"cutoff positions differ from restart profiles: {path}")
        if cutoff_matrix.shape != (1, len(positions)):
            raise ValueError(f"expected one p=0.1 cutoff row: {path}")
        cutoff = cutoff_matrix[0].copy()
        attributes = {str(key): _json_value(value) for key, value in group.attrs.items()}
    expected = profiles.values.max(axis=0)
    if not np.array_equal(cutoff, expected):
        raise ValueError("p<=0.1 cutoff is not the pointwise maximum of the 10 profiles")
    return CutoffBundle(
        cutoff=cutoff,
        ends=ends.copy(),
        source="completed_hdf5_group_verified_against_profiles",
        h5_path=path,
        group_attributes=attributes,
    )


def load_cutoff(
    sanity_dir: Path,
    *,
    population: str,
    chromosome: str,
    profiles: ProfileBundle,
    h5_attempts: int = 8,
) -> CutoffBundle:
    """Read the completed HDF5 group, tolerating brief concurrent writer locks."""
    if len(profiles.values) != 10:
        raise ValueError("this p<=0.1 maximum-null audit requires exactly 10 simulations")
    path = sanity_dir / "cutoffs" / f"{population.lower()}.gamma_smc_cutoffs.10kb.h5"
    last_error: Exception | None = None
    for attempt in range(h5_attempts):
        try:
            return _read_cutoff_h5(
                path,
                population=population,
                chromosome=chromosome,
                profiles=profiles,
            )
        except (KeyError, OSError) as error:
            last_error = error
            if attempt + 1 < h5_attempts:
                time.sleep(0.25)
    if last_error is not None:
        message = f"could not read cutoff HDF5 after {h5_attempts} attempts: {path}"
        raise OSError(message) from last_error
    raise RuntimeError("cutoff read failed without an exception")


def _quantile_summary(values: np.ndarray) -> dict[str, float]:
    quantiles = np.quantile(values, [0.01, 0.05, 0.5, 0.95, 0.99])
    return {
        "min": float(np.min(values)),
        "q01": float(quantiles[0]),
        "q05": float(quantiles[1]),
        "median": float(quantiles[2]),
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "q95": float(quantiles[3]),
        "q99": float(quantiles[4]),
        "max": float(np.max(values)),
    }


def _atomic_gzip_tsv(path: Path, rows: list[list[object]]) -> None:
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
    profiles: ProfileBundle,
    cutoff: CutoffBundle,
) -> tuple[Path, Path, list[dict[str, float]]]:
    per_simulation: list[dict[str, float]] = []
    header = [
        "simulation",
        "n_positions",
        "n_pairs",
        "decode_wall_seconds",
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
    lines = ["\t".join(header)]
    for simulation, values in enumerate(profiles.values):
        summary = _quantile_summary(values)
        per_simulation.append(summary)
        fields: list[object] = [
            simulation,
            len(profiles.positions),
            profiles.n_pairs,
            f"{profiles.decode_seconds[simulation]:.6f}",
            f"{summary['min']:.9g}",
            f"{summary['q01']:.9g}",
            f"{summary['q05']:.9g}",
            f"{summary['median']:.9g}",
            f"{summary['mean']:.9g}",
            f"{summary['sd']:.9g}",
            f"{summary['q95']:.9g}",
            f"{summary['q99']:.9g}",
            f"{summary['max']:.9g}",
        ]
        lines.append("\t".join(map(str, fields)))
    per_simulation_path = output_dir / "per_simulation_summary.tsv"
    atomic_text(per_simulation_path, "\n".join(lines) + "\n")

    q05, median, q95 = np.quantile(profiles.values, [0.05, 0.5, 0.95], axis=0)
    rows: list[list[object]] = [
        [
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
    ]
    for index, position in enumerate(profiles.positions):
        rows.append(
            [
                int(position),
                int(cutoff.ends[index]),
                f"{float(profiles.values[:, index].min()):.9g}",
                f"{float(q05[index]):.9g}",
                f"{float(median[index]):.9g}",
                f"{float(profiles.values[:, index].mean()):.9g}",
                f"{float(q95[index]):.9g}",
                f"{float(profiles.values[:, index].max()):.9g}",
                f"{float(cutoff.cutoff[index]):.9g}",
            ]
        )
    position_path = output_dir / "position_summary.tsv.gz"
    _atomic_gzip_tsv(position_path, rows)
    return per_simulation_path, position_path, per_simulation


def _plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 11,
            "figure.titlesize": 20,
            "savefig.bbox": "tight",
        }
    )


def _save_figure(figure: plt.Figure, output_dir: Path, stem: str, dpi: int) -> list[Path]:
    paths = [output_dir / f"{stem}.png", output_dir / f"{stem}.pdf"]
    figure.savefig(paths[0], dpi=dpi)
    figure.savefig(paths[1], dpi=dpi)
    plt.close(figure)
    return paths


def _remove_legacy_combined_figure(output_dir: Path) -> None:
    """Remove obsolete generated plots after their replacements are complete."""
    for suffix in ("png", "pdf"):
        path = output_dir / f"profiles_and_cutoff.{suffix}"
        if path.is_file():
            path.unlink()


def _calibration_title(*, population: str, chromosome: str, profiles: ProfileBundle) -> str:
    return (
        f"{population} {chromosome} Gamma-SMC sanity calibration — "
        f"{len(profiles.values)} simulations × {profiles.n_pairs:,} pairs"
    )


def plot_null_profiles(
    output_dir: Path,
    *,
    population: str,
    chromosome: str,
    profiles: ProfileBundle,
    dpi: int,
) -> list[Path]:
    x = profiles.positions / 1_000_000.0
    figure, axis = plt.subplots(figsize=FIGURE_SIZE, constrained_layout=True)
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 1.0, len(profiles.values)))
    for simulation, (values, color) in enumerate(zip(profiles.values, colors, strict=True)):
        axis.plot(
            x,
            values,
            color=color,
            linewidth=0.65,
            alpha=0.8,
            label=f"sim {simulation}",
        )
    axis.set_xlabel(f"{chromosome} position (Mb)")
    axis.set_ylabel("Mean posterior P(TMRCA < 4,500 y)")
    axis.set_title(
        f"{_calibration_title(population=population, chromosome=chromosome, profiles=profiles)}\n"
        "Ten independently simulated null profiles"
    )
    axis.legend(ncol=5, frameon=False, loc="upper right")
    axis.grid(alpha=0.2, linewidth=0.5)
    return _save_figure(figure, output_dir, "null_profiles", dpi)


def plot_across_simulation_summary(
    output_dir: Path,
    *,
    population: str,
    chromosome: str,
    profiles: ProfileBundle,
    cutoff: CutoffBundle,
    dpi: int,
) -> list[Path]:
    x = profiles.positions / 1_000_000.0
    mean = profiles.values.mean(axis=0)
    q05, median, q95 = np.quantile(profiles.values, [0.05, 0.5, 0.95], axis=0)
    figure, axis = plt.subplots(figsize=FIGURE_SIZE, constrained_layout=True)
    axis.fill_between(x, q05, q95, alpha=0.25, label="5th–95th percentile")
    axis.plot(x, mean, linewidth=1.0, label="mean")
    axis.plot(x, median, linewidth=1.0, label="median")
    axis.plot(
        x,
        cutoff.cutoff,
        color="black",
        linewidth=0.9,
        label="p≤0.1 cutoff (maximum)",
    )
    axis.set_xlabel(f"{chromosome} position (Mb)")
    axis.set_ylabel("Posterior probability")
    axis.set_title(
        f"{_calibration_title(population=population, chromosome=chromosome, profiles=profiles)}\n"
        "Across-simulation summary and exact pointwise cutoff"
    )
    axis.legend(ncol=2, frameon=False, loc="upper right")
    axis.grid(alpha=0.2, linewidth=0.5)
    return _save_figure(figure, output_dir, "across_simulation_summary", dpi)


def plot_profiles(
    output_dir: Path,
    *,
    population: str,
    chromosome: str,
    profiles: ProfileBundle,
    cutoff: CutoffBundle,
    dpi: int,
) -> list[Path]:
    """Write null traces and across-simulation summaries as separate figures."""
    return [
        *plot_null_profiles(
            output_dir,
            population=population,
            chromosome=chromosome,
            profiles=profiles,
            dpi=dpi,
        ),
        *plot_across_simulation_summary(
            output_dir,
            population=population,
            chromosome=chromosome,
            profiles=profiles,
            cutoff=cutoff,
            dpi=dpi,
        ),
    ]


def plot_heatmap(
    output_dir: Path,
    *,
    population: str,
    chromosome: str,
    profiles: ProfileBundle,
    dpi: int,
) -> list[Path]:
    extent = [
        float(profiles.positions[0] / 1_000_000.0),
        float(profiles.positions[-1] / 1_000_000.0),
        len(profiles.values) - 0.5,
        -0.5,
    ]
    figure, axis = plt.subplots(figsize=FIGURE_SIZE, constrained_layout=True)
    image = axis.imshow(
        profiles.values,
        aspect="auto",
        interpolation="nearest",
        extent=extent,
        cmap="viridis",
        vmin=0.0,
        vmax=float(np.quantile(profiles.values, 0.995)),
    )
    axis.set_xlabel(f"{chromosome} position (Mb)")
    axis.set_ylabel("Simulation index")
    axis.set_yticks(np.arange(len(profiles.values)))
    axis.set_title(f"{population} {chromosome}: spatial consistency across null Gamma-SMC profiles")
    colorbar = figure.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label("Mean posterior P(TMRCA < 4,500 y)")
    return _save_figure(figure, output_dir, "profile_heatmap", dpi)


def plot_distributions(
    output_dir: Path,
    *,
    population: str,
    chromosome: str,
    profiles: ProfileBundle,
    cutoff: CutoffBundle,
    dpi: int,
) -> list[Path]:
    figure, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE, constrained_layout=True)
    axes[0].boxplot(
        [row for row in profiles.values],
        showfliers=False,
    )
    axes[0].set_xticks(
        np.arange(1, len(profiles.values) + 1),
        labels=[str(index) for index in range(len(profiles.values))],
    )
    axes[0].set_xlabel("Simulation index")
    axes[0].set_ylabel("Posterior probability across 10 kb positions")
    axes[0].set_title("Per-simulation distributions")
    axes[0].grid(axis="y", alpha=0.2, linewidth=0.5)

    axes[1].hist(
        profiles.values.ravel(),
        bins=80,
        density=True,
        alpha=0.55,
        label="all null profile values",
    )
    axes[1].hist(
        cutoff.cutoff,
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
    figure.suptitle(f"{population} {chromosome} Gamma-SMC sanity summary")
    return _save_figure(figure, output_dir, "profile_distributions", dpi)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    result.add_argument("--sanity-dir", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, default=None)
    result.add_argument("--population", default="AFR")
    result.add_argument("--chromosome", default="1")
    result.add_argument("--n-sims", type=int, default=10)
    result.add_argument("--dpi", type=int, default=220)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.n_sims != 10 or args.dpi <= 0:
        raise SystemExit("--n-sims must equal 10 for the p<=0.1 maximum-null audit; --dpi > 0")
    population = args.population.strip().upper()
    chromosome = canonical_chrom(args.chromosome)
    sanity_dir = args.sanity_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else sanity_dir / "diagnostics" / f"{population.lower()}_{chromosome}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        profiles = load_profiles(
            sanity_dir,
            population=population,
            chromosome=chromosome,
            n_sims=args.n_sims,
        )
        cutoff = load_cutoff(
            sanity_dir,
            population=population,
            chromosome=chromosome,
            profiles=profiles,
        )
        per_simulation_path, position_path, per_simulation = write_tables(
            output_dir, profiles=profiles, cutoff=cutoff
        )
        _plot_style()
        figure_paths = [
            *plot_profiles(
                output_dir,
                population=population,
                chromosome=chromosome,
                profiles=profiles,
                cutoff=cutoff,
                dpi=args.dpi,
            ),
            *plot_heatmap(
                output_dir,
                population=population,
                chromosome=chromosome,
                profiles=profiles,
                dpi=args.dpi,
            ),
            *plot_distributions(
                output_dir,
                population=population,
                chromosome=chromosome,
                profiles=profiles,
                cutoff=cutoff,
                dpi=args.dpi,
            ),
        ]
        _remove_legacy_combined_figure(output_dir)
    except (FileNotFoundError, KeyError, OSError, TypeError, ValueError) as error:
        raise SystemExit(f"Gamma-SMC chromosome analysis failed: {error}") from error

    finite_seconds = profiles.decode_seconds[np.isfinite(profiles.decode_seconds)]
    manifest = {
        "schema": ANALYSIS_SCHEMA,
        "sanity_dir": str(sanity_dir),
        "output_dir": str(output_dir),
        "population": population,
        "chromosome": chromosome,
        "n_simulations": len(profiles.values),
        "n_positions": len(profiles.positions),
        "n_pairs_per_simulation": profiles.n_pairs,
        "validation": {
            "all_profiles_present_and_valid": True,
            "shared_stride_grid": True,
            "shared_pair_count": True,
            "completed_hdf5_group": True,
            "p_le_0.1_cutoff_equals_pointwise_profile_maximum": True,
            "cutoff_source": cutoff.source,
        },
        "profile_distribution": _quantile_summary(profiles.values.ravel()),
        "pointwise_cutoff_distribution": _quantile_summary(cutoff.cutoff),
        "decode_process_wall_seconds": {
            "note": "per-process wall times overlap when decode_workers is greater than one",
            "available_count": int(len(finite_seconds)),
            "minimum": float(finite_seconds.min()) if len(finite_seconds) else None,
            "median": float(np.median(finite_seconds)) if len(finite_seconds) else None,
            "maximum": float(finite_seconds.max()) if len(finite_seconds) else None,
            "sum_not_job_wall_time": float(finite_seconds.sum()) if len(finite_seconds) else None,
        },
        "per_simulation_distribution": per_simulation,
        "inputs": {
            "profiles": [
                {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
                for path in profiles.paths
            ],
            "cutoff_h5": str(cutoff.h5_path),
            "cutoff_group_attributes": cutoff.group_attributes,
        },
        "outputs": {
            "per_simulation_summary": str(per_simulation_path),
            "position_summary": str(position_path),
            "figures": [str(path) for path in figure_paths],
        },
    }
    manifest_path = output_dir / "analysis.json"
    atomic_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    checksum_paths = [per_simulation_path, position_path, *figure_paths, manifest_path]
    checksum_text = "".join(
        f"{sha256_file(path)}  {path.relative_to(output_dir).as_posix()}\n"
        for path in checksum_paths
    )
    checksum_path = output_dir / "checksums.sha256"
    atomic_text(checksum_path, checksum_text)
    print(
        f"[gamma-sanity-analysis] validated={len(profiles.values)} profiles "
        f"positions={len(profiles.positions):,} pairs={profiles.n_pairs:,}"
    )
    print(f"[gamma-sanity-analysis] output_dir={output_dir}")
    print(f"[gamma-sanity-analysis] manifest={manifest_path}")
    print(f"[gamma-sanity-analysis] checksums={checksum_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
