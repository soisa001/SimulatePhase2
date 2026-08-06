#!/usr/bin/env python3
"""Measure seeded MVN-draw error against the stored PHLASH bootstrap curves."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from statistics import NormalDist

import numpy as np

from phase2_map import DEFAULT_POPS
from run_sim import load_mvn_draws, mvn_path, population_seed, sha256_file

REPORT_SCHEMA = "simulatephase2.mvn-error/v1"
ARTIFACT_SCHEMA = "phlash.aou.log-ne-mvn/v1"
PLOT_QUANTILES = np.asarray([0.025, 0.5, 0.975])
DISTRIBUTION_QUANTILES = np.linspace(0.025, 0.975, 39)
QUANTILE_LABELS = ("2.5% quantile", "median", "97.5% quantile")
QUANTILE_STYLES = (
    {"color": "#1b9e77", "linestyle": "--", "linewidth": 1.8},
    {"color": "#222222", "linestyle": "-", "linewidth": 2.4},
    {"color": "#d95f02", "linestyle": ":", "linewidth": 2.2},
)


def _require_curves(curves: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(curves, dtype=float)
    if value.ndim != 2 or value.shape[0] < 2 or value.shape[1] < 2:
        raise ValueError(f"{name} must have shape (at least 2, at least 2)")
    if not np.isfinite(value).all() or np.any(value <= 0.0):
        raise ValueError(f"{name} must contain only finite positive values")
    return value


def _quantile_metrics(
    reference_quantiles: np.ndarray,
    candidate_quantiles: np.ndarray,
    times: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    relative_error = 100.0 * (candidate_quantiles / reference_quantiles - 1.0)
    median_error = relative_error[1]
    absolute_median_error = np.abs(median_error)
    maximum_index = int(np.argmax(absolute_median_error))
    metrics = {
        "median_mean_error_percent": float(np.mean(median_error)),
        "median_mean_absolute_error_percent": float(np.mean(absolute_median_error)),
        "median_root_mean_square_error_percent": float(np.sqrt(np.mean(np.square(median_error)))),
        "median_maximum_absolute_error_percent": float(absolute_median_error[maximum_index]),
        "median_maximum_absolute_error_time_generations": float(times[maximum_index]),
        "lower_2_5_mean_absolute_error_percent": float(np.mean(np.abs(relative_error[0]))),
        "upper_97_5_mean_absolute_error_percent": float(np.mean(np.abs(relative_error[2]))),
        "central_95_endpoint_mean_absolute_error_percent": float(
            np.mean(np.abs(relative_error[[0, 2]]))
        ),
    }
    return relative_error, metrics


def compare_curves(
    reference: np.ndarray,
    candidate: np.ndarray,
    times: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Compare pointwise distributions of positive candidate and reference curves."""
    reference = _require_curves(reference, "reference")
    candidate = _require_curves(candidate, "candidate")
    times = np.asarray(times, dtype=float)
    if reference.shape[1] != candidate.shape[1] or times.shape != (reference.shape[1],):
        raise ValueError("reference, candidate, and time dimensions do not agree")
    if not np.isfinite(times).all() or times[0] <= 0.0 or np.any(np.diff(times) <= 0.0):
        raise ValueError("times must be finite, positive, and strictly increasing")

    reference_quantiles = np.quantile(reference, PLOT_QUANTILES, axis=0)
    candidate_quantiles = np.quantile(candidate, PLOT_QUANTILES, axis=0)
    relative_error, metrics = _quantile_metrics(reference_quantiles, candidate_quantiles, times)

    reference_log_quantiles = np.quantile(np.log(reference), DISTRIBUTION_QUANTILES, axis=0)
    candidate_log_quantiles = np.quantile(np.log(candidate), DISTRIBUTION_QUANTILES, axis=0)
    pointwise_log_rmse = np.sqrt(
        np.mean(np.square(candidate_log_quantiles - reference_log_quantiles), axis=0)
    )
    pointwise_multiplicative_error = 100.0 * np.expm1(pointwise_log_rmse)
    candidate_low, candidate_high = candidate_quantiles[[0, 2]]
    metrics.update(
        {
            "central_quantile_log_rmse_mean": float(np.mean(pointwise_log_rmse)),
            "central_quantile_multiplicative_error_percent_mean": float(
                np.mean(pointwise_multiplicative_error)
            ),
            "reference_curve_coverage_by_candidate_95_interval": float(
                np.mean((reference >= candidate_low) & (reference <= candidate_high))
            ),
        }
    )
    arrays = {
        "reference_quantiles": reference_quantiles,
        "candidate_quantiles": candidate_quantiles,
        "relative_error_percent": relative_error,
        "central_quantile_log_rmse": pointwise_log_rmse,
    }
    return arrays, metrics


def compare_analytic_mvn(
    reference: np.ndarray,
    mean_log_ne: np.ndarray,
    covariance_factor: np.ndarray,
    times: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """Compare exact pointwise MVN marginals with empirical reference quantiles."""
    reference = _require_curves(reference, "reference")
    mean = np.asarray(mean_log_ne, dtype=float)
    factor = np.asarray(covariance_factor, dtype=float)
    if mean.shape != (reference.shape[1],) or factor.shape[1:] != mean.shape:
        raise ValueError("MVN mean/factor dimensions do not agree with the reference")
    if not np.isfinite(mean).all() or not np.isfinite(factor).all():
        raise ValueError("MVN mean and factor must be finite")
    standard_deviation = np.sqrt(np.sum(np.square(factor), axis=0))
    z = NormalDist().inv_cdf(0.975)
    candidate_quantiles = np.exp(
        mean[None, :] + np.asarray([-z, 0.0, z])[:, None] * standard_deviation[None, :]
    )
    reference_quantiles = np.quantile(reference, PLOT_QUANTILES, axis=0)
    return _quantile_metrics(reference_quantiles, candidate_quantiles, times)


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _atomic_figure(fig, path: Path, *, dpi: int = 220) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=path.suffix, dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        if path.suffix == ".pdf":
            metadata = {
                "Creator": "SimulatePhase2/evaluate_mvn_error.py",
                "Producer": "matplotlib",
                "CreationDate": None,
                "ModDate": None,
            }
        else:
            metadata = {"Software": "SimulatePhase2/evaluate_mvn_error.py"}
        fig.savefig(
            temporary,
            format=path.suffix.removeprefix("."),
            dpi=dpi,
            metadata=metadata,
        )
        with temporary.open("r+b") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _parse_populations(value: str) -> tuple[str, ...]:
    populations = tuple(part.strip().upper() for part in value.split(",") if part.strip())
    unknown = sorted(set(populations) - set(DEFAULT_POPS))
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown populations: {','.join(unknown)}")
    if not populations or len(populations) != len(set(populations)):
        raise argparse.ArgumentTypeError("population list must be nonempty and unique")
    return populations


def _source_manifest(metadata: dict[str, object]) -> tuple[int, int]:
    sources = metadata.get("sources")
    if not isinstance(sources, list):
        raise ValueError("MVN sidecar has no source manifest")
    total_bytes = 0
    for source in sources:
        if not isinstance(source, dict) or int(source.get("size_bytes", 0)) <= 0:
            raise ValueError("MVN sidecar has an invalid source entry")
        total_bytes += int(source["size_bytes"])
    return len(sources), total_bytes


def _plot(
    populations: tuple[str, ...],
    times: np.ndarray,
    results: dict[str, dict[str, object]],
    output_dir: Path,
    n_draws: int,
) -> tuple[Path, Path, float]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 15,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 13,
        }
    )
    columns = 2
    rows = math.ceil(len(populations) / columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(11.0, 8.5),
        sharex=True,
        sharey=True,
    )
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.105, top=0.815, hspace=0.32, wspace=0.16)
    axes_array = np.atleast_1d(axes).ravel()
    maximum = max(
        float(np.max(np.abs(results[population]["relative_error_percent"])))
        for population in populations
    )
    limit = max(5.0, 5.0 * math.ceil(maximum * 1.08 / 5.0))
    handles = []
    for axis, population in zip(axes_array, populations, strict=False):
        errors = np.asarray(results[population]["relative_error_percent"])
        for index, (label, style) in enumerate(zip(QUANTILE_LABELS, QUANTILE_STYLES, strict=True)):
            (line,) = axis.plot(times, errors[index], label=label, **style)
            if len(handles) < len(QUANTILE_LABELS):
                handles.append(line)
        axis.axhline(0.0, color="#777777", linewidth=1.0, alpha=0.65)
        axis.set_xscale("log")
        axis.set_xlim(float(times[0]), float(times[-1]))
        axis.set_ylim(-limit, limit)
        axis.margins(x=0)
        axis.grid(True, which="both", alpha=0.16)
        metrics = results[population]["simulation_metrics"]
        axis.set_title(
            f"{population}  |  median MAE {metrics['median_mean_absolute_error_percent']:.2f}%"
        )
    for axis in axes_array[len(populations) :]:
        axis.set_visible(False)
    fig.suptitle(
        f"MVN simulation error relative to stored PHLASH curves ({n_draws:,} seeded draws)",
        fontsize=17,
        y=0.98,
    )
    fig.supxlabel("generations ago (log scale)", fontsize=15, y=0.025)
    fig.supylabel("signed relative error in $N_e$ (%)", fontsize=15, x=0.015)
    fig.legend(
        handles,
        QUANTILE_LABELS,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncols=3,
        frameon=False,
    )
    png = output_dir / "all_populations.mvn_error.png"
    pdf = output_dir / "all_populations.mvn_error.pdf"
    _atomic_figure(fig, png)
    _atomic_figure(fig, pdf)
    plt.close(fig)
    return png, pdf, limit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mvn-dir", type=Path, default=Path("mvn"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--pops", type=_parse_populations, default=DEFAULT_POPS)
    parser.add_argument("--n-draws", type=int, default=1_000)
    parser.add_argument("--base-seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.n_draws < 2:
        raise SystemExit("--n-draws must be at least 2")
    if args.base_seed < 0:
        raise SystemExit("--base-seed must be nonnegative")
    mvn_dir = args.mvn_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve() if args.output_dir is not None else mvn_dir / "plots"
    )

    times: np.ndarray | None = None
    results: dict[str, dict[str, object]] = {}
    report_populations: dict[str, object] = {}
    source_file_count = 0
    source_total_bytes = 0
    aggregate_median_errors: list[np.ndarray] = []
    aggregate_endpoint_errors: list[np.ndarray] = []
    aggregate_distribution_errors: list[np.ndarray] = []
    aggregate_coverage_numerator = 0
    aggregate_coverage_denominator = 0

    for population in args.pops:
        artifact = mvn_path(mvn_dir, population)
        sidecar = mvn_dir / f"{population}.json"
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        if metadata.get("schema") != ARTIFACT_SCHEMA:
            raise ValueError(f"unexpected sidecar schema for {population}")
        population_source_count, population_source_bytes = _source_manifest(metadata)
        with np.load(artifact, allow_pickle=False) as data:
            if str(data["schema"]) != ARTIFACT_SCHEMA or str(data["population"]) != population:
                raise ValueError(f"unexpected MVN identity for {population}")
            artifact_times = np.asarray(data["time"], dtype=float)
            reference = _require_curves(data["bootstrap_ne"], "bootstrap_ne")
            mean = np.asarray(data["mean_log_ne"], dtype=float)
            factor = np.asarray(data["covariance_factor"], dtype=float)
            jitter = float(data["jitter"])
        if jitter != 0.0:
            raise ValueError("analytic comparison currently requires zero MVN jitter")
        population_times, draws, draw_metadata = load_mvn_draws(
            artifact,
            population=population,
            n_sims=args.n_draws,
            epochs=len(artifact_times),
            seed=args.base_seed,
        )
        if not np.array_equal(artifact_times, population_times):
            raise ValueError(f"simulation changed the time grid for {population}")
        if times is None:
            times = population_times
        elif not np.array_equal(times, population_times):
            raise ValueError(f"{population} does not use the shared MVN time grid")

        arrays, simulation_metrics = compare_curves(reference, draws, population_times)
        analytic_errors, analytic_metrics = compare_analytic_mvn(
            reference, mean, factor, population_times
        )
        log_reference = np.log(reference)
        reconstructed_mean = log_reference.mean(axis=0)
        reconstructed_factor = (log_reference - reconstructed_mean) / np.sqrt(
            reference.shape[0] - 1.0
        )
        moment_checks = {
            "log_mean_maximum_absolute_reconstruction_error": float(
                np.max(np.abs(mean - reconstructed_mean))
            ),
            "covariance_factor_maximum_absolute_reconstruction_error": float(
                np.max(np.abs(factor - reconstructed_factor))
            ),
        }
        results[population] = {
            "relative_error_percent": arrays["relative_error_percent"],
            "simulation_metrics": simulation_metrics,
        }
        report_populations[population] = {
            "analytic_mvn_vs_empirical_reference": analytic_metrics,
            "artifact": artifact.relative_to(mvn_dir.parent).as_posix(),
            "artifact_sha256": str(draw_metadata["source_sha256"]),
            "covariance_factor_shape": list(factor.shape),
            "covariance_rank_at_most": int(metadata["covariance_rank_at_most"]),
            "draw_seed": population_seed(args.base_seed, population),
            "empirical_reference_curves": int(reference.shape[0]),
            "moment_reconstruction_checks": moment_checks,
            "seeded_simulation_vs_empirical_reference": simulation_metrics,
            "sidecar": sidecar.relative_to(mvn_dir.parent).as_posix(),
            "sidecar_sha256": sha256_file(sidecar),
            "source_pkl_count": population_source_count,
            "source_pkl_total_bytes": population_source_bytes,
        }
        source_file_count += population_source_count
        source_total_bytes += population_source_bytes
        aggregate_median_errors.append(arrays["relative_error_percent"][1])
        aggregate_endpoint_errors.append(arrays["relative_error_percent"][[0, 2]].ravel())
        aggregate_distribution_errors.append(100.0 * np.expm1(arrays["central_quantile_log_rmse"]))
        candidate_low, candidate_high = arrays["candidate_quantiles"][[0, 2]]
        aggregate_coverage_numerator += int(
            np.count_nonzero((reference >= candidate_low) & (reference <= candidate_high))
        )
        aggregate_coverage_denominator += int(reference.size)
        print(
            f"[mvn-error] {population}: median MAE="
            f"{simulation_metrics['median_mean_absolute_error_percent']:.3f}%; "
            f"95% endpoint MAE="
            f"{simulation_metrics['central_95_endpoint_mean_absolute_error_percent']:.3f}%",
            flush=True,
        )

    assert times is not None
    png, pdf, y_limit = _plot(args.pops, times, results, output_dir, args.n_draws)
    median_errors = np.concatenate(aggregate_median_errors)
    endpoint_errors = np.concatenate(aggregate_endpoint_errors)
    distribution_errors = np.concatenate(aggregate_distribution_errors)
    worst_index = int(np.argmax(np.abs(median_errors)))
    points_per_population = len(times)
    worst_population_index, worst_time_index = divmod(worst_index, points_per_population)
    aggregate = {
        "central_95_endpoint_mean_absolute_error_percent": float(np.mean(np.abs(endpoint_errors))),
        "central_quantile_multiplicative_error_percent_mean": float(np.mean(distribution_errors)),
        "median_maximum_absolute_error_percent": float(abs(median_errors[worst_index])),
        "median_maximum_absolute_error_population": args.pops[worst_population_index],
        "median_maximum_absolute_error_time_generations": float(times[worst_time_index]),
        "median_mean_absolute_error_percent": float(np.mean(np.abs(median_errors))),
        "median_root_mean_square_error_percent": float(np.sqrt(np.mean(np.square(median_errors)))),
        "reference_curve_coverage_by_candidate_95_interval": float(
            aggregate_coverage_numerator / aggregate_coverage_denominator
        ),
    }
    report = {
        "aggregate_seeded_simulation_vs_empirical_reference": aggregate,
        "base_seed": args.base_seed,
        "complete": True,
        "definitions": {
            "analytic_mvn": (
                "Exact pointwise log-normal marginals implied by mean_log_ne and "
                "covariance_factor; this isolates Gaussian approximation error."
            ),
            "central_quantile_error": (
                "At 39 equally spaced quantiles from 0.025 through 0.975, the pointwise "
                "root-mean-square log-quantile difference, converted to 100*(exp(error)-1)."
            ),
            "empirical_reference": (
                "The 100 stored posterior-median PHLASH bootstrap Ne curves per population; "
                "these are an empirical reference, not known biological truth."
            ),
            "plot": (
                "Pointwise signed relative errors for the 2.5%, 50%, and 97.5% quantiles: "
                "100*(seeded MVN-draw quantile/reference quantile - 1)."
            ),
            "seeded_simulation": (
                "The same run_sim.load_mvn_draws path, full 10,000 epochs, population seed "
                "schedule, and base seed used to prepare production demographies."
            ),
        },
        "n_draws_per_population": args.n_draws,
        "plot_axis_signed_error_percent": [-y_limit, y_limit],
        "plots": {
            "pdf": {
                "path": pdf.relative_to(mvn_dir.parent).as_posix(),
                "sha256": sha256_file(pdf),
            },
            "png": {
                "path": png.relative_to(mvn_dir.parent).as_posix(),
                "sha256": sha256_file(png),
            },
        },
        "populations": report_populations,
        "quantiles_plotted": PLOT_QUANTILES.tolist(),
        "schema": REPORT_SCHEMA,
        "source_pkl_manifest": {
            "count": source_file_count,
            "total_bytes": source_total_bytes,
        },
        "time_grid": {
            "maximum_generations": float(times[-1]),
            "minimum_generations": float(times[0]),
            "points": int(len(times)),
            "sha256_float64": hashlib.sha256(times.astype(np.float64).tobytes()).hexdigest(),
        },
    }
    report_path = output_dir / "mvn_error_report.json"
    _atomic_json(report_path, report)
    print(f"[mvn-error] wrote {png}, {pdf}, and {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
