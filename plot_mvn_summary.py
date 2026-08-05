#!/usr/bin/env python3
"""Plot pointwise medians and central 95% intervals from production MVN draws."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from phase2_map import DEFAULT_POPS
from run_sim import load_mvn_draws, mvn_path, population_seed, sha256_file

REPORT_SCHEMA = "simulatephase2.mvn-plot-summary/v1"
COLOURS = {
    "AFR": "#d95f02",
    "EUR": "#1f78b4",
    "AMR": "#1b9e77",
    "SAS": "#e6ab02",
    "MID": "#66c2e5",
    "EAS": "#e7298a",
}


def summarize(draws: np.ndarray) -> dict[str, np.ndarray]:
    """Return pointwise sample median and central 95% interval."""
    draws = np.asarray(draws, dtype=float)
    if draws.ndim != 2 or draws.shape[0] < 2 or draws.shape[1] < 2:
        raise ValueError("draws must be a two-dimensional array with at least 2 x 2 values")
    if not np.isfinite(draws).all() or np.any(draws <= 0.0):
        raise ValueError("draws must contain only finite positive values")
    median, low, high = np.percentile(draws, [50.0, 2.5, 97.5], axis=0)
    if np.any(low > median) or np.any(median > high):
        raise RuntimeError("invalid MVN summary interval")
    return {"median": median, "low": low, "high": high}


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


def _atomic_figure(fig, path: Path, *, dpi: int = 180) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=path.suffix, dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        if path.suffix == ".pdf":
            metadata = {
                "Creator": "SimulatePhase2/plot_mvn_summary.py",
                "Producer": "matplotlib",
                "CreationDate": None,
                "ModDate": None,
            }
        else:
            metadata = {"Software": "SimulatePhase2/plot_mvn_summary.py"}
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


def _style_axis(axis, args: argparse.Namespace) -> None:
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("generations ago")
    axis.set_ylabel("$N_e$ (diploid)")
    axis.set_xlim(args.min_time, args.max_time)
    axis.set_ylim(args.min_ne, args.max_ne)
    axis.margins(x=0)
    axis.grid(True, which="both", alpha=0.15)
    secondary = axis.secondary_xaxis(
        "top",
        functions=(
            lambda generations: generations * args.generation_time,
            lambda years: years / args.generation_time,
        ),
    )
    secondary.set_xlabel(f"years ago ({args.generation_time:g} years/generation)")


def _parse_populations(value: str) -> tuple[str, ...]:
    populations = tuple(part.strip().upper() for part in value.split(",") if part.strip())
    unknown = sorted(set(populations) - set(DEFAULT_POPS))
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown populations: {','.join(unknown)}")
    if not populations or len(set(populations)) != len(populations):
        raise argparse.ArgumentTypeError("population list must be nonempty and unique")
    return populations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mvn-dir", type=Path, default=Path("mvn"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--pops", type=_parse_populations, default=DEFAULT_POPS)
    parser.add_argument("--n-draws", type=int, default=1_000)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--min-time", type=float, default=100.0)
    parser.add_argument("--max-time", type=float, default=40_000.0)
    parser.add_argument("--min-ne", type=float, default=1_000.0)
    parser.add_argument("--max-ne", type=float, default=80_000.0)
    parser.add_argument("--generation-time", type=float, default=25.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.n_draws < 2:
        raise SystemExit("--n-draws must be at least 2")
    if not 0.0 < args.min_time < args.max_time:
        raise SystemExit("require 0 < --min-time < --max-time")
    if not 0.0 < args.min_ne < args.max_ne:
        raise SystemExit("require 0 < --min-ne < --max-ne")
    if args.generation_time <= 0.0:
        raise SystemExit("--generation-time must be positive")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 15,
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
        }
    )

    output_dir = args.output_dir or args.mvn_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    times: np.ndarray | None = None
    summaries: dict[str, dict[str, np.ndarray]] = {}
    manifest_populations: dict[str, object] = {}

    for population in args.pops:
        artifact = mvn_path(args.mvn_dir, population)
        population_times, draws, metadata = load_mvn_draws(
            artifact,
            population=population,
            n_sims=args.n_draws,
            epochs=10_000,
            seed=args.base_seed,
        )
        if times is None:
            times = population_times
        elif not np.array_equal(times, population_times):
            raise ValueError(f"{population} does not use the shared MVN time grid")
        summary = summarize(draws)
        summaries[population] = summary

        fig, axis = plt.subplots(figsize=(8.5, 6.5))
        colour = COLOURS[population]
        axis.fill_between(
            population_times,
            summary["low"],
            summary["high"],
            color=colour,
            alpha=0.25,
            linewidth=0,
            label="MVN 95% interval",
        )
        axis.plot(
            population_times,
            summary["median"],
            color=colour,
            linewidth=2.5,
            label="MVN median",
        )
        _style_axis(axis, args)
        axis.set_title(
            f"{population}: MVN median and 95% interval "
            f"({args.n_draws:,} seeded draws)"
        )
        axis.legend(frameon=False)
        fig.tight_layout()
        png = output_dir / f"{population}.mvn.png"
        pdf = output_dir / f"{population}.mvn.pdf"
        _atomic_figure(fig, png)
        _atomic_figure(fig, pdf)
        plt.close(fig)

        sidecar = args.mvn_dir / f"{population}.json"
        sidecar_value = (
            json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.is_file() else {}
        )
        manifest_populations[population] = {
            "artifact": str(artifact.as_posix()),
            "artifact_sha256": str(metadata["source_sha256"]),
            "covariance_factor_rows": int(metadata["covariance_rank"]),
            "covariance_rank_at_most": sidecar_value.get("covariance_rank_at_most"),
            "draw_seed": population_seed(args.base_seed, population),
            "json_sha256": sha256_file(sidecar) if sidecar.is_file() else None,
            "median_ne_range": [
                float(np.min(summary["median"])),
                float(np.max(summary["median"])),
            ],
            "pointwise_95_percent_ne_range": [
                float(np.min(summary["low"])),
                float(np.max(summary["high"])),
            ],
            "plots": {
                "pdf": {"path": str(pdf.as_posix()), "sha256": sha256_file(pdf)},
                "png": {"path": str(png.as_posix()), "sha256": sha256_file(png)},
            },
        }
        draw_seed = population_seed(args.base_seed, population)
        print(f"{population}: seed={draw_seed}; wrote {png} and {pdf}")

    assert times is not None
    fig, axis = plt.subplots(figsize=(10.5, 7.5))
    for population in args.pops:
        summary = summaries[population]
        colour = COLOURS[population]
        axis.fill_between(
            times,
            summary["low"],
            summary["high"],
            color=colour,
            alpha=0.12,
            linewidth=0,
        )
        axis.plot(times, summary["median"], color=colour, linewidth=2.5, label=population)
    _style_axis(axis, args)
    axis.set_title(
        "MVN median and 95% interval by population "
        f"({args.n_draws:,} seeded draws each)"
    )
    axis.legend(frameon=False)
    fig.tight_layout()
    combined_png = output_dir / "all_populations.mvn.png"
    combined_pdf = output_dir / "all_populations.mvn.pdf"
    _atomic_figure(fig, combined_png)
    _atomic_figure(fig, combined_pdf)
    plt.close(fig)

    manifest = {
        "axis_limits": {
            "diploid_ne": [args.min_ne, args.max_ne],
            "generations": [args.min_time, args.max_time],
        },
        "base_seed": args.base_seed,
        "combined_plots": {
            "pdf": {
                "path": str(combined_pdf.as_posix()),
                "sha256": sha256_file(combined_pdf),
            },
            "png": {
                "path": str(combined_png.as_posix()),
                "sha256": sha256_file(combined_png),
            },
        },
        "complete": True,
        "generation_time_years": args.generation_time,
        "n_draws_per_population": args.n_draws,
        "populations": manifest_populations,
        "schema": REPORT_SCHEMA,
        "time_grid": {
            "maximum": float(times[-1]),
            "minimum": float(times[0]),
            "points": int(len(times)),
        },
    }
    report = output_dir / "plot_manifest.json"
    _atomic_json(report, manifest)
    print(f"combined: wrote {combined_png} and {combined_pdf}")
    print(f"provenance: wrote {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
