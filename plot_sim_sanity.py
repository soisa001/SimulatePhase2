#!/usr/bin/env python3
"""Plot read-only sanity summaries for completed calibrated simulations.

For the first N simulations of each requested population, this script computes
windowed segregating sites, Tajima's D, nucleotide diversity, and a folded SFS.
Window geometry, population names, and raw S targets come from the map artifact
produced by ``generate_map.py``; actual sample counts and target scaling come
from the simulation-root contract.

Outputs in ``--out-dir`` are:

* ``{pop}_genomewide.png``: realized versus target segregating sites, Tajima's D,
  and nucleotide diversity along the concatenated requested chromosomes;
* ``{pop}_sfs.png``: folded SFS versus a neutral reference;
* ``summary_by_pop.png`` and ``summary.tsv``: cross-population summaries.

Examples::

    python plot_sim_sanity.py
    python plot_sim_sanity.py --n-sims 5 --workers 4 --pops AFR,EUR --chroms 1-5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from collections.abc import Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path

# Each process handles one tree sequence at a time. Prevent numerical libraries
# from creating another thread pool inside every worker.
for _variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_variable, "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import h5py  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tszip  # noqa: E402

from phase2_map import SCHEMA, canonical_chrom, map_sample_counts, parse_chroms  # noqa: E402

DEFAULT_MAP = Path("data/snv_theta_map.10kb.h5")
DEFAULT_SIM_DIR = Path("sims")
DEFAULT_OUT_DIR = Path("sim_sanity_plots")
MAX_DISPLAY_BINS = 2_500
MAX_DISTRIBUTION_VALUES = 200_000


@dataclass(frozen=True)
class ChromosomeLayout:
    edges: np.ndarray
    output_slice: slice


@dataclass(frozen=True)
class GenomeLayout:
    window_size: int
    chromosomes: tuple[str, ...]
    chromosome_layouts: dict[str, ChromosomeLayout]
    genome_midpoints: np.ndarray
    boundaries: np.ndarray
    labels: tuple[str, ...]
    targets: dict[str, np.ndarray]

    @property
    def n_windows(self) -> int:
        return len(self.genome_midpoints)


@dataclass
class PopulationAccumulator:
    target: np.ndarray
    binned_sum: dict[str, np.ndarray]
    binned_count: dict[str, np.ndarray]
    traces: dict[str, list[np.ndarray]]
    distribution_values: list[np.ndarray]
    afs: np.ndarray | None
    sum_stat: dict[str, float]
    count_stat: dict[str, int]
    used_haplotypes: list[int]
    available_haplotypes: list[int]
    distribution_count: int = 0
    units_ok: int = 0
    sims_with_data: int = 0


_WORKER_SIM_DIR: Path | None = None
_WORKER_EDGES: dict[str, np.ndarray] = {}
_WORKER_PANEL_HAPLOTYPES: dict[str, int] = {}
_WORKER_MAP_SHA256 = ""


def contract_signature(contract: dict[str, object]) -> str:
    """Match the canonical unit-contract signature written by ``run_sim.py``."""
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def init_worker(
    sim_dir: str,
    edges: dict[str, np.ndarray],
    panel_sample_counts: dict[str, int],
    map_sha256: str,
) -> None:
    """Install immutable analysis metadata once in each spawned worker."""
    global _WORKER_SIM_DIR, _WORKER_EDGES, _WORKER_PANEL_HAPLOTYPES, _WORKER_MAP_SHA256
    _WORKER_SIM_DIR = Path(sim_dir)
    _WORKER_EDGES = edges
    _WORKER_PANEL_HAPLOTYPES = {
        population: 2 * count for population, count in panel_sample_counts.items()
    }
    _WORKER_MAP_SHA256 = map_sha256


def validate_sidecar(
    path: Path,
    *,
    population: str,
    simulation: int,
    chromosome: str,
    expected_diploids: int,
) -> None:
    """Reject incomplete, stale, or internally inconsistent simulation metadata."""
    sidecar = path.with_suffix(path.suffix + ".json")
    if not sidecar.is_file():
        raise ValueError(f"required completion sidecar is missing: {sidecar}")
    try:
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read completion sidecar {sidecar}: {error}") from error
    if metadata.get("status") != "complete":
        raise ValueError(f"sidecar status is not complete: {sidecar}")
    contract = metadata.get("contract")
    if not isinstance(contract, dict):
        raise ValueError(f"sidecar has no simulation contract: {sidecar}")
    signature = metadata.get("signature")
    if not isinstance(signature, str) or not signature:
        raise ValueError(f"sidecar has no signature: {sidecar}")
    expected_signature = contract_signature(contract)
    if signature != expected_signature:
        raise ValueError(f"sidecar signature does not match its contract: {sidecar}")
    expected_contract = {
        "map_sha256": _WORKER_MAP_SHA256,
        "population": population,
        "simulation": simulation,
        "chromosome": chromosome,
        "diploid_samples": expected_diploids,
    }
    mismatches = {
        key: (contract.get(key), expected)
        for key, expected in expected_contract.items()
        if contract.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"sidecar contract mismatch {mismatches}: {sidecar}")
    if int(metadata.get("size_bytes", -1)) != path.stat().st_size:
        raise ValueError(f"sidecar size does not match tree-sequence file: {sidecar}")


def analyze_unit(job: tuple[str, int, str, str]) -> dict[str, object]:
    """Analyze one population/simulation/chromosome tree sequence."""
    population, simulation, chromosome, size_mode = job
    if _WORKER_SIM_DIR is None:
        raise RuntimeError("worker was not initialized")
    path = _WORKER_SIM_DIR / population.lower() / f"sim_{simulation:05d}" / f"{chromosome}.tsz"
    if not path.is_file():
        return {
            "population": population,
            "simulation": simulation,
            "chromosome": chromosome,
            "status": "missing",
            "message": str(path),
        }
    try:
        expected_haplotypes = _WORKER_PANEL_HAPLOTYPES[population]
        validate_sidecar(
            path,
            population=population,
            simulation=simulation,
            chromosome=chromosome,
            expected_diploids=expected_haplotypes // 2,
        )
        tree_sequence = tszip.decompress(str(path))
        edges = _WORKER_EDGES[chromosome]
        if not np.isclose(tree_sequence.sequence_length, edges[-1]):
            raise ValueError(
                f"sequence length {tree_sequence.sequence_length:g} does not match "
                f"map length {edges[-1]:g}"
            )
        samples = tree_sequence.samples()
        available = int(len(samples))
        if available < expected_haplotypes:
            raise ValueError(
                f"sample shortfall: map requires {expected_haplotypes // 2} diploids "
                f"({expected_haplotypes} haplotypes), but tree sequence has "
                f"{available // 2} diploids ({available} haplotypes)"
            )
        wanted = available if size_mode == "sim" else expected_haplotypes
        keep = samples[:wanted]
        if len(keep) < 2:
            raise ValueError("fewer than two haploid samples are available")

        segregating = np.asarray(
            tree_sequence.segregating_sites(
                sample_sets=[keep], windows=edges, mode="site", span_normalise=False
            )
        ).ravel()
        tajimas_d = np.asarray(
            tree_sequence.Tajimas_D(sample_sets=[keep], windows=edges, mode="site")
        ).ravel()
        diversity = np.asarray(
            tree_sequence.diversity(
                sample_sets=[keep], windows=edges, mode="site", span_normalise=True
            )
        ).ravel()
        afs = np.asarray(
            tree_sequence.allele_frequency_spectrum(
                sample_sets=[keep],
                mode="site",
                polarised=False,
                span_normalise=False,
            )
        ).ravel()
        return {
            "population": population,
            "simulation": simulation,
            "chromosome": chromosome,
            "status": "ok",
            "segregating": segregating.astype(np.float32, copy=False),
            "tajimas_d": tajimas_d.astype(np.float32, copy=False),
            "diversity": diversity.astype(np.float32, copy=False),
            "afs": afs.astype(np.float64, copy=False),
            "used": int(len(keep)),
            "available": available,
            "wanted": int(wanted),
        }
    except Exception:
        return {
            "population": population,
            "simulation": simulation,
            "chromosome": chromosome,
            "status": "error",
            "message": traceback.format_exc().splitlines()[-1],
        }


def bounded_results(
    executor: ProcessPoolExecutor,
    jobs: Iterable[tuple[str, int, str, str]],
    max_pending: int,
) -> Iterator[dict[str, object]]:
    """Yield worker results while retaining at most ``max_pending`` futures."""
    iterator = iter(jobs)
    pending: set[Future[dict[str, object]]] = set()
    for _ in range(max_pending):
        try:
            pending.add(executor.submit(analyze_unit, next(iterator)))
        except StopIteration:
            break
    while pending:
        completed, pending = wait(pending, return_when=FIRST_COMPLETED)
        for future in completed:
            yield future.result()
            try:
                pending.add(executor.submit(analyze_unit, next(iterator)))
            except StopIteration:
                pass


def decode_strings(values: np.ndarray) -> list[str]:
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_map_header(path: Path) -> tuple[int, list[str], dict[str, int]]:
    """Read and validate map-wide metadata used by both plotting and simulation."""
    with h5py.File(path, "r") as handle:
        schema = str(handle.attrs.get("schema", ""))
        if schema != SCHEMA:
            raise ValueError(f"{path} has schema {schema!r}, expected {SCHEMA!r}")
        if not bool(handle.attrs.get("complete", False)):
            raise ValueError(f"{path} is not marked complete")
        window_size = int(handle.attrs.get("window_size", 0))
        if window_size <= 0:
            raise ValueError("map root has no positive window_size")
        if "populations" not in handle or "samples" not in handle:
            raise ValueError("map is missing embedded populations or samples")
        populations = [value.upper() for value in decode_strings(handle["populations"][...])]
    sample_counts = map_sample_counts(path)
    missing = [population for population in populations if sample_counts.get(population, 0) <= 0]
    if missing:
        raise ValueError(f"map has no embedded samples for {missing}")
    return window_size, populations, sample_counts


def read_simulation_samples(
    sim_dir: Path,
    map_sha256: str,
    populations: list[str],
    map_sample_counts: dict[str, int],
) -> tuple[dict[str, int], dict[str, float]]:
    """Read the root contract so plots use the actual simulated targets and sample sizes."""
    path = sim_dir / "simulation_contract.json"
    try:
        contract = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read simulation contract {path}: {error}") from error
    global_contract = contract.get("global")
    population_contracts = contract.get("populations")
    if not isinstance(global_contract, dict) or not isinstance(population_contracts, dict):
        raise ValueError(f"invalid simulation contract: {path}")
    if str(global_contract.get("map_sha256", "")) != map_sha256:
        raise ValueError("simulation contract map SHA256 differs from the requested map")
    simulation_counts: dict[str, int] = {}
    scales: dict[str, float] = {}
    for population in populations:
        entry = population_contracts.get(population)
        if not isinstance(entry, dict):
            raise ValueError(f"simulation contract has no entry for {population}")
        map_count = int(entry.get("map_diploid_samples", 0))
        simulation_count = int(entry.get("diploid_samples", 0))
        scale = float(entry.get("S_scale", float("nan")))
        if map_count != map_sample_counts[population] or simulation_count <= 0:
            raise ValueError(f"invalid sample counts in simulation contract for {population}")
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"invalid S scale in simulation contract for {population}")
        simulation_counts[population] = simulation_count
        scales[population] = scale
    return simulation_counts, scales


def load_genome_layout(
    path: Path,
    chromosomes: list[str],
    populations: list[str],
    sample_counts: dict[str, int],
) -> GenomeLayout:
    """Load compact map geometry and target vectors once in the parent process."""
    layouts: dict[str, ChromosomeLayout] = {}
    midpoints: list[np.ndarray] = []
    boundaries = [0.0]
    targets_by_chrom: dict[str, dict[str, np.ndarray]] = {
        population: {} for population in populations
    }
    offset = 0.0
    output_start = 0
    with h5py.File(path, "r") as handle:
        window_size = int(handle.attrs["window_size"])
        map_populations = [value.upper() for value in decode_strings(handle["populations"][...])]
        population_indices = {
            population: map_populations.index(population) for population in populations
        }
        absent = [chromosome for chromosome in chromosomes if chromosome not in handle]
        if absent:
            raise ValueError(f"map is missing requested chromosomes: {absent}")
        for chromosome in chromosomes:
            group = handle[chromosome]
            length_bp = int(group.attrs["length_bp"])
            count = int(group.attrs["n_windows"])
            starts = np.arange(count, dtype=np.int64) * window_size
            ends = np.minimum(starts + window_size, length_bp)
            if (
                len(starts) == 0
                or starts.shape != ends.shape
                or starts[0] != 0
                or ends[-1] != length_bp
                or not np.array_equal(starts[1:], ends[:-1])
            ):
                raise ValueError(f"invalid window geometry for {chromosome}")
            widths = ends - starts
            if np.any(widths <= 0) or np.any(widths > window_size):
                raise ValueError(f"invalid window widths for {chromosome}")
            layouts[chromosome] = ChromosomeLayout(
                edges=np.concatenate(([0.0], ends.astype(float))),
                output_slice=slice(output_start, output_start + count),
            )
            midpoints.append((starts + (ends - starts) / 2.0) + offset)
            output_start += count
            offset += length_bp
            boundaries.append(offset)
            matrix = group["S"]
            if matrix.shape != (len(map_populations), count):
                raise ValueError(f"invalid S matrix shape for {chromosome}: {matrix.shape}")
            for population in populations:
                theta = np.asarray(matrix[population_indices[population]], dtype=np.float32)
                if theta.shape != starts.shape:
                    raise ValueError(f"S shape mismatch for {population}/{chromosome}")
                targets_by_chrom[population][chromosome] = theta

    targets = {
        population: np.concatenate(
            [targets_by_chrom[population][chromosome] for chromosome in chromosomes]
        )
        for population in populations
    }
    return GenomeLayout(
        window_size=window_size,
        chromosomes=tuple(chromosomes),
        chromosome_layouts=layouts,
        genome_midpoints=np.concatenate(midpoints),
        boundaries=np.asarray(boundaries),
        labels=tuple(chromosomes),
        targets=targets,
    )


def bin_mean(indices: np.ndarray, values: np.ndarray, n_bins: int) -> np.ndarray:
    """Return the finite-value mean in each precomputed display bin."""
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    sums = np.bincount(indices[valid], weights=values[valid], minlength=n_bins)
    counts = np.bincount(indices[valid], minlength=n_bins)
    result = np.full(n_bins, np.nan)
    nonempty = counts > 0
    result[nonempty] = sums[nonempty] / counts[nonempty]
    return result


def add_binned(accumulator: PopulationAccumulator, name: str, values: np.ndarray) -> None:
    valid = np.isfinite(values)
    accumulator.binned_sum[name][valid] += values[valid]
    accumulator.binned_count[name][valid] += 1


def statistic_mean(accumulator: PopulationAccumulator, name: str) -> float:
    count = accumulator.count_stat[name]
    return accumulator.sum_stat[name] / count if count else float("nan")


def binned_average(accumulator: PopulationAccumulator, name: str) -> np.ndarray:
    result = np.full_like(accumulator.binned_sum[name], np.nan)
    valid = accumulator.binned_count[name] > 0
    result[valid] = accumulator.binned_sum[name][valid] / accumulator.binned_count[name][valid]
    return result


def format_window_size(window_size: int) -> str:
    if window_size % 1_000_000 == 0:
        return f"{window_size / 1_000_000:g} Mb"
    if window_size % 1_000 == 0:
        return f"{window_size / 1_000:g} kb"
    return f"{window_size:,} bp"


def draw_chromosome_guides(axis: plt.Axes, boundaries: np.ndarray, labels: tuple[str, ...]) -> None:
    ymax = axis.get_ylim()[1]
    for boundary in boundaries:
        axis.axvline(boundary / 1e6, color="grey", linewidth=0.4, alpha=0.5)
    mids = (boundaries[:-1] + boundaries[1:]) / 2.0
    for midpoint, label in zip(mids, labels, strict=True):
        axis.text(
            midpoint / 1e6,
            ymax,
            canonical_chrom(label)[3:],
            ha="center",
            va="bottom",
            fontsize=6,
            color="grey",
            clip_on=False,
        )


def sample_range(values: list[int], *, diploid: bool = False) -> str:
    if not values:
        return "unknown"
    low, high = min(values), max(values)
    if diploid:
        low, high = low // 2, high // 2
    return str(low) if low == high else f"{low}-{high}"


def plot_population_genomewide(
    population: str,
    accumulator: PopulationAccumulator,
    centers: np.ndarray,
    layout: GenomeLayout,
    panel_count: int,
    out_dir: Path,
    size_mode: str,
) -> Path:
    figure, axes = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
    x = centers / 1e6
    colors = {"segregating": "C0", "tajimas_d": "C1", "diversity": "C2"}
    for name, axis in zip(colors, axes, strict=True):
        for trace in accumulator.traces[name]:
            axis.plot(x, trace, linewidth=0.4, alpha=0.3, color=colors[name])
        axis.plot(
            x,
            binned_average(accumulator, name),
            linewidth=1.3,
            color=colors[name],
            label=(
                f"realized (mean of {accumulator.sims_with_data} sims)"
                if name == "segregating"
                else None
            ),
        )
    axes[0].plot(
        x,
        binned_average(accumulator, "target"),
        linewidth=1.0,
        linestyle="--",
        color="black",
        label="target S",
    )
    window_label = format_window_size(layout.window_size)
    axes[0].set_ylabel(f"seg sites / {window_label}")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[1].axhline(0.0, color="black", linewidth=0.6, linestyle=":")
    axes[1].set_ylabel("Tajima's D")
    axes[2].set_ylabel(r"$\pi$ / bp")
    axes[2].set_xlabel("genome position (Mb; requested chromosomes concatenated)")
    for axis in axes:
        draw_chromosome_guides(axis, layout.boundaries, layout.labels)
    used = sample_range(accumulator.used_haplotypes, diploid=True)
    available = sample_range(accumulator.available_haplotypes, diploid=True)
    figure.suptitle(
        f"{population} genome-wide sanity (size={size_mode}; used {used} diploids, "
        f"available {available}, map panel {panel_count})",
        fontsize=11,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    output = out_dir / f"{population.lower()}_genomewide.png"
    figure.savefig(output, dpi=130)
    plt.close(figure)
    return output


def plot_population_sfs(
    population: str, afs: np.ndarray | None, used_haplotypes: list[int], out_dir: Path
) -> Path | None:
    if afs is None or afs.size < 3:
        return None
    n_haplotypes = len(afs) - 1
    counts = np.arange(1, n_haplotypes // 2 + 1)
    observed = afs[1 : n_haplotypes // 2 + 1]
    reference = 1.0 / counts + 1.0 / (n_haplotypes - counts)
    if n_haplotypes % 2 == 0:
        reference[-1] = 1.0 / counts[-1]
    reference = reference / reference.sum() * np.nansum(observed)
    figure, axis = plt.subplots(figsize=(6, 4.5))
    axis.loglog(counts, observed, ".", markersize=3, label="observed (all analyzed units)")
    axis.loglog(
        counts,
        reference,
        "-",
        linewidth=1,
        color="black",
        alpha=0.7,
        label=r"folded neutral $\propto 1/i + 1/(n-i)$",
    )
    axis.set_xlabel("minor allele count")
    axis.set_ylabel("number of sites")
    axis.set_title(f"{population} folded SFS ({sample_range(used_haplotypes)} haplotypes)")
    axis.legend(fontsize=8)
    figure.tight_layout()
    output = out_dir / f"{population.lower()}_sfs.png"
    figure.savefig(output, dpi=130)
    plt.close(figure)
    return output


def plot_summary(
    rows: list[dict[str, object]],
    distributions: dict[str, np.ndarray],
    window_size: int,
    out_dir: Path,
) -> Path:
    populations = [str(row["pop"]) for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].boxplot(
        [
            distributions[population] if len(distributions[population]) else np.asarray([np.nan])
            for population in populations
        ],
        tick_labels=populations,
        showfliers=False,
    )
    axes[0].axhline(0.0, color="black", linewidth=0.6, linestyle=":")
    axes[0].set_ylabel("Tajima's D (sampled windows across sims)")
    axes[0].set_title("Tajima's D by population")

    positions = np.arange(len(populations))
    width = 0.38
    axes[1].bar(
        positions - width / 2,
        [float(row["mean_S_realized"]) for row in rows],
        width,
        label="realized",
        color="C0",
    )
    axes[1].bar(
        positions + width / 2,
        [float(row["mean_theta_target"]) for row in rows],
        width,
        label="target",
        color="grey",
    )
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(populations)
    axes[1].set_ylabel(f"mean seg sites / {format_window_size(window_size)}")
    axes[1].set_title("realized versus target S")
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    output = out_dir / "summary_by_pop.png"
    figure.savefig(output, dpi=130)
    plt.close(figure)
    return output


def make_display_axis(layout: GenomeLayout) -> tuple[np.ndarray, np.ndarray, int]:
    n_bins = min(MAX_DISPLAY_BINS, layout.n_windows)
    low = float(layout.genome_midpoints.min())
    high = float(layout.genome_midpoints.max())
    span = high - low
    if span == 0:
        return np.zeros(layout.n_windows, dtype=int), np.asarray([low]), 1
    indices = np.clip(((layout.genome_midpoints - low) / span * n_bins).astype(int), 0, n_bins - 1)
    centers = (np.arange(n_bins) + 0.5) / n_bins * span + low
    return indices, centers, n_bins


def new_accumulator(target: np.ndarray, indices: np.ndarray, n_bins: int) -> PopulationAccumulator:
    names = ("segregating", "tajimas_d", "diversity", "target")
    return PopulationAccumulator(
        target=target,
        binned_sum={name: np.zeros(n_bins) for name in names},
        binned_count={name: np.zeros(n_bins, dtype=np.int64) for name in names},
        traces={name: [] for name in names if name != "target"},
        distribution_values=[],
        afs=None,
        sum_stat={name: 0.0 for name in names},
        count_stat={name: 0 for name in names},
        used_haplotypes=[],
        available_haplotypes=[],
    )


def add_statistic(
    accumulator: PopulationAccumulator,
    name: str,
    values: np.ndarray,
    indices: np.ndarray,
    n_bins: int,
    keep_trace: bool,
) -> np.ndarray:
    valid = np.isfinite(values)
    accumulator.sum_stat[name] += float(np.sum(values[valid], dtype=np.float64))
    accumulator.count_stat[name] += int(valid.sum())
    binned = bin_mean(indices, values, n_bins)
    add_binned(accumulator, name, binned)
    if keep_trace:
        accumulator.traces[name].append(binned)
    return valid


def parse_populations(spec: str | None, embedded: list[str]) -> list[str]:
    if spec is None:
        return embedded
    populations = [value.strip().upper() for value in spec.split(",") if value.strip()]
    if not populations:
        raise ValueError("--pops selected no populations")
    unknown = [population for population in populations if population not in embedded]
    if unknown or len(set(populations)) != len(populations):
        raise ValueError(f"invalid, duplicate, or absent populations: {unknown or populations}")
    return populations


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--n-sims", type=int, default=5, help="Analyze sim_00000 through N-1")
    result.add_argument("--pops", help="Comma-separated populations; default: all in map")
    result.add_argument("--chroms", default="1-22")
    result.add_argument(
        "--size",
        choices=("panel", "sim"),
        default="panel",
        help="panel: simulation-contract sample count; sim: all samples in each tree sequence",
    )
    result.add_argument("--workers", type=int, default=4)
    result.add_argument("--max-pending", type=int, default=0, help="Default: twice --workers")
    result.add_argument(
        "--max-traces",
        type=int,
        default=50,
        help="Maximum individual simulation traces per population (means still use every sim)",
    )
    result.add_argument("--sim-dir", type=Path, default=DEFAULT_SIM_DIR)
    result.add_argument("--h5", "--map", dest="map_path", type=Path, default=DEFAULT_MAP)
    result.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.n_sims <= 0 or args.workers <= 0 or args.max_traces < 0:
        raise SystemExit(
            "--n-sims and --workers must be positive; --max-traces must be nonnegative"
        )
    max_pending = args.max_pending or 2 * args.workers
    if max_pending < args.workers:
        raise SystemExit("--max-pending must be zero or at least --workers")

    map_path = args.map_path.expanduser().resolve()
    sim_dir = args.sim_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    if not map_path.is_file():
        raise SystemExit(f"map does not exist: {map_path}; run generate_map.py first")
    map_sha256 = sha256_file(map_path)
    try:
        window_size, embedded_populations, map_sample_counts = read_map_header(map_path)
        populations = parse_populations(args.pops, embedded_populations)
        chromosomes = parse_chroms(args.chroms)
        layout = load_genome_layout(map_path, chromosomes, populations, map_sample_counts)
        sample_counts, target_scales = read_simulation_samples(
            sim_dir, map_sha256, populations, map_sample_counts
        )
        for population in populations:
            layout.targets[population] = np.floor(
                layout.targets[population].astype(np.float64) * target_scales[population] + 0.5
            ).astype(np.float32)
    except (KeyError, OSError, ValueError) as error:
        raise SystemExit(f"invalid map or selection: {error}") from error
    out_dir.mkdir(parents=True, exist_ok=True)
    indices, centers, n_bins = make_display_axis(layout)
    worker_edges = {
        chromosome: chromosome_layout.edges
        for chromosome, chromosome_layout in layout.chromosome_layouts.items()
    }
    total_units = args.n_sims * len(populations) * len(chromosomes)
    panel_summary = ", ".join(f"{pop}: {map_sample_counts[pop]}" for pop in populations)
    simulation_summary = ", ".join(f"{pop}: {sample_counts[pop]}" for pop in populations)
    print(
        f"map={map_path} sha256={map_sha256} window={format_window_size(window_size)}\n"
        f"pops={populations} map_samples={{{panel_summary}}} "
        f"simulation_samples={{{simulation_summary}}}\n"
        f"chroms={chromosomes} sims=first {args.n_sims} size={args.size}\n"
        f"units={total_units:,} workers={args.workers} max_pending={max_pending}\n"
        f"sim_dir={sim_dir} out={out_dir}",
        flush=True,
    )

    rows: list[dict[str, object]] = []
    distributions: dict[str, np.ndarray] = {}
    completed_units = missing_units = error_units = 0
    started = time.monotonic()
    executor = ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=init_worker,
        initargs=(str(sim_dir), worker_edges, sample_counts, map_sha256),
    )
    try:
        for population in populations:
            accumulator = new_accumulator(layout.targets[population], indices, n_bins)
            distribution_cap_per_sim = max(
                1, (MAX_DISTRIBUTION_VALUES + args.n_sims - 1) // args.n_sims
            )
            for simulation in range(args.n_sims):
                arrays = {
                    name: np.full(layout.n_windows, np.nan, dtype=np.float32)
                    for name in ("segregating", "tajimas_d", "diversity")
                }
                successful_windows = np.zeros(layout.n_windows, dtype=bool)
                units_this_sim = 0
                jobs = (
                    (population, simulation, chromosome, args.size) for chromosome in chromosomes
                )
                for result in bounded_results(executor, jobs, max_pending):
                    completed_units += 1
                    status = str(result["status"])
                    chromosome = str(result["chromosome"])
                    if status == "missing":
                        missing_units += 1
                        print(
                            f"[{completed_units}/{total_units}] MISSING {result['message']}",
                            flush=True,
                        )
                        continue
                    if status == "error":
                        error_units += 1
                        print(
                            f"[{completed_units}/{total_units}] ERROR "
                            f"{population}/sim_{simulation:05d}/{chromosome}: "
                            f"{result['message']}",
                            flush=True,
                        )
                        continue
                    output_slice = layout.chromosome_layouts[chromosome].output_slice
                    expected = output_slice.stop - output_slice.start
                    for name in arrays:
                        values = np.asarray(result[name])
                        if len(values) != expected:
                            raise RuntimeError(
                                f"worker returned {len(values)} {name} windows for "
                                f"{chromosome}; expected {expected}"
                            )
                        arrays[name][output_slice] = values
                    successful_windows[output_slice] = True
                    accumulator.used_haplotypes.append(int(result["used"]))
                    accumulator.available_haplotypes.append(int(result["available"]))
                    accumulator.units_ok += 1
                    units_this_sim += 1
                    afs = np.asarray(result["afs"], dtype=np.float64)
                    if accumulator.afs is None:
                        accumulator.afs = np.zeros_like(afs)
                    if accumulator.afs.shape == afs.shape:
                        accumulator.afs += afs
                    else:
                        print(
                            f"[warn] {population}/sim_{simulation:05d}/{chromosome}: "
                            "SFS sample size differs; excluding this unit from the SFS",
                            flush=True,
                        )
                    if completed_units % 25 == 0 or completed_units == total_units:
                        print(
                            f"... {completed_units}/{total_units} analyzed in "
                            f"{time.monotonic() - started:.0f}s",
                            flush=True,
                        )
                if units_this_sim == 0:
                    continue
                accumulator.sims_with_data += 1
                keep_trace = len(accumulator.traces["segregating"]) < args.max_traces
                for name, values in arrays.items():
                    valid = add_statistic(accumulator, name, values, indices, n_bins, keep_trace)
                    if name == "tajimas_d":
                        finite_values = values[valid]
                        remaining = MAX_DISTRIBUTION_VALUES - accumulator.distribution_count
                        retain = min(distribution_cap_per_sim, remaining)
                        if retain <= 0:
                            continue
                        if len(finite_values) > retain:
                            chosen = np.linspace(
                                0,
                                len(finite_values) - 1,
                                retain,
                                dtype=int,
                            )
                            finite_values = finite_values[chosen]
                        if len(finite_values):
                            accumulator.distribution_values.append(finite_values.copy())
                            accumulator.distribution_count += len(finite_values)
                matched_target = np.where(successful_windows, accumulator.target, np.nan)
                add_statistic(
                    accumulator,
                    "target",
                    matched_target,
                    indices,
                    n_bins,
                    keep_trace=False,
                )

            if accumulator.units_ok == 0:
                print(f"[warn] {population}: no usable units; skipping plots", flush=True)
                continue
            genome_plot = plot_population_genomewide(
                population,
                accumulator,
                centers,
                layout,
                sample_counts[population],
                out_dir,
                args.size,
            )
            sfs_plot = plot_population_sfs(
                population, accumulator.afs, accumulator.used_haplotypes, out_dir
            )
            distributions[population] = (
                np.concatenate(accumulator.distribution_values)
                if accumulator.distribution_values
                else np.empty(0)
            )
            used_min = min(accumulator.used_haplotypes)
            used_max = max(accumulator.used_haplotypes)
            available_min = min(accumulator.available_haplotypes)
            available_max = max(accumulator.available_haplotypes)
            expected_haplotypes = 2 * sample_counts[population]
            rows.append(
                {
                    "pop": population,
                    "n_used_hap": used_min,
                    "n_used_dip": used_min // 2,
                    "panel_dip": map_sample_counts[population],
                    "sim_dip": available_min // 2,
                    "shortfall_dip": max(0, expected_haplotypes - available_min) // 2,
                    "mean_S_realized": statistic_mean(accumulator, "segregating"),
                    "mean_theta_target": statistic_mean(accumulator, "target"),
                    "mean_TajD": statistic_mean(accumulator, "tajimas_d"),
                    "mean_pi": statistic_mean(accumulator, "diversity"),
                    "window_size": window_size,
                    "n_sims_requested": args.n_sims,
                    "n_sims_analyzed": accumulator.sims_with_data,
                    "n_units_analyzed": accumulator.units_ok,
                    "n_used_hap_max": used_max,
                    "sim_dip_max": available_max // 2,
                }
            )
            names = [genome_plot.name]
            if sfs_plot is not None:
                names.append(sfs_plot.name)
            print(f"[{population}] wrote {', '.join(names)}", flush=True)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    if not rows:
        print("No populations produced plots; check --sim-dir and requested simulations.")
        return 1
    summary_plot = plot_summary(rows, distributions, window_size, out_dir)
    summary_path = out_dir / "summary.tsv"
    columns = [
        "pop",
        "n_used_hap",
        "n_used_dip",
        "panel_dip",
        "sim_dip",
        "shortfall_dip",
        "mean_S_realized",
        "mean_theta_target",
        "mean_TajD",
        "mean_pi",
        "window_size",
        "n_sims_requested",
        "n_sims_analyzed",
        "n_units_analyzed",
        "n_used_hap_max",
        "sim_dip_max",
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("\t".join(columns) + "\n")
        for row in rows:
            handle.write(
                "\t".join(
                    f"{row[column]:.6g}" if isinstance(row[column], float) else str(row[column])
                    for column in columns
                )
                + "\n"
            )
    print(
        f"DONE in {time.monotonic() - started:.0f}s. Wrote {summary_plot.name}, "
        f"{summary_path.name}, and per-population PNGs to {out_dir} "
        f"(missing={missing_units}, errors={error_units}).",
        flush=True,
    )
    return 0 if error_units == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
