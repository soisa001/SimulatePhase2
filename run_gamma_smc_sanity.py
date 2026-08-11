#!/usr/bin/env python3
"""Run an isolated 10-replicate, 100,000-pair Gamma-SMC sanity calibration.

Only already-published simulation units are read. The reducer uses a separate
HDF5/profile root, so this workflow can run while the full simulation producer
continues writing later replicates. Exact position-specific p<=0.1 cutoffs and
descriptive per-chromosome/population summaries are written with provenance.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import math
import os
import platform
import shlex
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

from generate_cutoff_gamma_smc import (
    DEFAULT_CACHE_SIZE,
    DEFAULT_MUTATION_RATE,
    DEFAULT_PAIR_BLOCK,
    DEFAULT_PAIRS_SEED,
    DEFAULT_RHO_OVER_THETA,
    DEFAULT_THETA,
    GAMMA_CUTOFF_SCHEMA,
    STATISTIC_COLUMN,
)
from generate_cutoffs import ALL_AUTOSOMES, DEFAULT_GENERATION_TIME, DEFAULT_THRESHOLD_YEARS
from phase2_map import DEFAULT_GAMMA_SMC_REPO, DEFAULT_POPS, parse_chroms
from resource_budget import cpu_resource_plan
from run_sim import DEFAULT_SIM_DIR, atomic_text, sha256_file
from simulation_outputs import completed_units, validate_completed_unit

SANITY_SCHEMA = "simulatephase2.gamma-smc-sanity/v1"
DEFAULT_N_SIMULATIONS = 10
DEFAULT_N_RANDOM_PAIRS = 100_000
DEFAULT_P_VALUE = 0.1
DETAIL_REPORT_NAME = "p_le_0.1_cutoffs.tsv.gz"
SUMMARY_REPORT_NAME = "p_le_0.1_cutoff_summary.tsv"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _git_revision(repository: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _parse_populations(spec: str) -> list[str]:
    populations = [part.strip().upper() for part in spec.split(",") if part.strip()]
    unknown = [population for population in populations if population not in DEFAULT_POPS]
    if unknown or not populations or len(populations) != len(set(populations)):
        raise ValueError(f"invalid or duplicate populations: {unknown or populations}")
    return populations


def _default_output_dir(sim_dir: Path, n_sims: int, n_pairs: int) -> Path:
    return sim_dir / "sanity" / f"gamma_smc_{n_sims}sims_{n_pairs}pairs"


def _monte_carlo_rank(n_sims: int, p_value: float) -> tuple[int, int]:
    max_exceedances = math.floor(np.nextafter(p_value * (n_sims + 1) - 1.0, np.inf))
    if max_exceedances < 0:
        raise ValueError(
            f"p={p_value:g} is below the {1.0 / (n_sims + 1):.8g} Monte Carlo resolution"
        )
    max_exceedances = min(max_exceedances, n_sims - 1)
    return max_exceedances, max_exceedances + 1


def preflight_completed_units(
    *,
    sim_dir: Path,
    populations: list[str],
    chromosomes: list[str],
    n_sims: int,
    n_pairs: int,
    exclude_within: bool,
) -> dict[str, object]:
    """Validate every selected input before launching any expensive decode."""
    digests: dict[str, dict[str, str]] = {}
    population_samples: dict[str, int] = {}
    for population in populations:
        digests[population] = {}
        expected_samples: int | None = None
        for chromosome in chromosomes:
            paths, digest = completed_units(sim_dir, population, chromosome, n_sims)
            first = validate_completed_unit(
                paths[0], population=population, simulation=0, chromosome=chromosome
            )
            contract = first.metadata.get("contract")
            if not isinstance(contract, dict):
                raise ValueError(f"invalid unit contract: {first.sidecar}")
            samples = int(contract["diploid_samples"])
            if expected_samples is None:
                expected_samples = samples
            elif samples != expected_samples:
                raise ValueError(
                    f"{population} sample count differs across chromosomes: "
                    f"{expected_samples} != {samples}"
                )
            digests[population][chromosome] = digest
        if expected_samples is None:
            raise ValueError(f"no completed units selected for {population}")
        n_haplotypes = 2 * expected_samples
        available = n_haplotypes * (n_haplotypes - 1) // 2
        if exclude_within:
            available -= expected_samples
        if n_pairs > available:
            raise ValueError(
                f"{population} has only {available:,} eligible pairs from "
                f"{expected_samples:,} diploids; requested {n_pairs:,}"
            )
        population_samples[population] = expected_samples
        print(
            f"[sanity-preflight] {population}: simulations=0-{n_sims - 1} "
            f"chromosomes={len(chromosomes)} diploids={expected_samples:,} "
            f"eligible_pairs={available:,}",
            flush=True,
        )
    return {
        "simulation_indices": list(range(n_sims)),
        "population_diploid_samples": population_samples,
        "unit_manifest_digests": digests,
    }


def reducer_command(
    *,
    args: argparse.Namespace,
    sim_dir: Path,
    output_root: Path,
    populations: list[str],
    chromosomes: list[str],
) -> list[str]:
    script = Path(__file__).resolve().with_name("generate_cutoff_gamma_smc.py")
    command = [
        sys.executable,
        "-u",
        str(script),
        "--sim-dir",
        str(sim_dir),
        "--output-dir",
        str(output_root / "cutoffs"),
        "--profile-dir",
        str(output_root / "profiles"),
        "--keep-profiles",
        "--pops",
        ",".join(populations),
        "--chroms",
        ",".join(chromosome.removeprefix("chr") for chromosome in chromosomes),
        "--n-sims",
        str(args.n_sims),
        "--p-values",
        str(DEFAULT_P_VALUE),
        "--n-random-pairs",
        str(args.n_random_pairs),
        "--pairs-seed",
        str(args.pairs_seed),
        "--gamma-smc-repo",
        str(args.gamma_smc_repo.expanduser().resolve()),
        "--theta",
        str(args.theta),
        "--rho-over-theta",
        str(args.rho_over_theta),
        "--mutation-rate",
        str(args.mutation_rate),
        "--threshold-years",
        str(args.threshold_years),
        "--generation-time",
        str(args.generation_time),
        "--stride",
        str(args.stride),
        "--cache-size",
        str(args.cache_size),
        "--pair-block",
        str(args.pair_block),
        "--decode-workers",
        str(args.decode_workers),
        "--decode-threads",
        str(args.decode_threads),
        "--reserved-cpus",
        str(args.reserved_cpus),
        "--progress-every",
        str(args.progress_every),
    ]
    if args.gamma_smc_aou is not None:
        command.extend(["--gamma-smc-aou", str(args.gamma_smc_aou.expanduser().resolve())])
    if args.gamma_smc_executable is not None:
        command.extend(
            ["--gamma-smc-executable", str(args.gamma_smc_executable.expanduser().resolve())]
        )
    if args.hardmask is not None:
        command.extend(["--hardmask", str(args.hardmask.expanduser().resolve())])
    if args.exclude_within:
        command.append("--exclude-within")
    if args.fresh:
        command.append("--fresh")
    if args.allow_cpu_oversubscription:
        command.append("--allow-cpu-oversubscription")
    return command


def _atomic_gzip_tsv(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                    writer = csv.writer(text, delimiter="\t", lineterminator="\n")
                    for row in rows:
                        writer.writerow(row)
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _summary_row(
    *,
    population: str,
    scope: str,
    values: np.ndarray,
    n_chromosomes: int,
    n_sims: int,
    n_pairs: int,
    max_exceedances: int,
    rank: int,
) -> list[object]:
    quantiles = np.quantile(values, [0.05, 0.5, 0.95])
    return [
        population,
        scope,
        n_chromosomes,
        len(values),
        f"{DEFAULT_P_VALUE:.9g}",
        n_sims,
        n_pairs,
        max_exceedances,
        rank,
        f"{float(values.min()):.9g}",
        f"{float(quantiles[0]):.9g}",
        f"{float(quantiles[1]):.9g}",
        f"{float(values.mean()):.9g}",
        f"{float(quantiles[2]):.9g}",
        f"{float(values.max()):.9g}",
    ]


def write_sanity_reports(
    *,
    output_root: Path,
    populations: list[str],
    chromosomes: list[str],
    n_sims: int,
    n_pairs: int,
    pairs_seed: int = DEFAULT_PAIRS_SEED,
    exclude_within: bool = False,
    theta: float = DEFAULT_THETA,
    rho_over_theta: float = DEFAULT_RHO_OVER_THETA,
    mutation_rate: float = DEFAULT_MUTATION_RATE,
    threshold_years: float = DEFAULT_THRESHOLD_YEARS,
    generation_time: float = DEFAULT_GENERATION_TIME,
    stride: int = 10_000,
) -> dict[str, object]:
    """Validate reducer outputs and export exact plus descriptive cutoff reports."""
    cutoff_dir = output_root / "cutoffs"
    detail_path = output_root / DETAIL_REPORT_NAME
    summary_path = output_root / SUMMARY_REPORT_NAME
    expected_max_exceedances, expected_rank = _monte_carlo_rank(n_sims, DEFAULT_P_VALUE)
    cutoff_files: dict[str, dict[str, object]] = {}
    summaries: list[list[object]] = []

    def detail_rows():
        yield [
            "population",
            "chromosome",
            "position_0based",
            "end",
            "p_value",
            "cutoff",
            "n_simulations",
            "n_pairs",
            "max_null_exceedances",
            "rank_from_largest",
            "significance_rule",
        ]
        for population in populations:
            path = cutoff_dir / f"{population.lower()}.gamma_smc_cutoffs.10kb.h5"
            population_values: list[np.ndarray] = []
            with h5py.File(path, "r") as handle:
                if str(handle.attrs.get("schema", "")) != GAMMA_CUTOFF_SCHEMA:
                    raise ValueError(f"unexpected cutoff schema: {path}")
                if str(handle.attrs.get("population", "")) != population:
                    raise ValueError(f"population metadata mismatch: {path}")
                if int(handle.attrs.get("n_simulations", -1)) != n_sims:
                    raise ValueError(f"simulation count mismatch: {path}")
                if tuple(chromosomes) == ALL_AUTOSOMES and not bool(
                    handle.attrs.get("complete", False)
                ):
                    raise ValueError(f"full-autosome cutoff root is incomplete: {path}")
                if str(handle.attrs.get("pair_selection", "")) != "random_haplotype_pairs":
                    raise ValueError(f"random-pair metadata missing: {path}")
                if str(handle.attrs.get("statistic", "")) != STATISTIC_COLUMN:
                    raise ValueError(f"statistic metadata mismatch: {path}")
                if str(handle.attrs.get("significance_rule", "")) != "observed > cutoff":
                    raise ValueError(f"significance rule mismatch: {path}")
                if not np.array_equal(handle["p_value"][:], [DEFAULT_P_VALUE]):
                    raise ValueError(f"p-value metadata mismatch: {path}")
                decoder = json.loads(str(handle.attrs["decoder_contract_json"]))
                expected_decoder = {
                    "n_random_pairs": n_pairs,
                    "pairs_seed": pairs_seed,
                    "exclude_within": exclude_within,
                    "output_at_stride": stride,
                }
                if any(decoder.get(key) != value for key, value in expected_decoder.items()):
                    raise ValueError(f"random-pair decoder contract mismatch: {path}")
                expected_floats = {
                    "scaled_mutation_rate_theta": theta,
                    "recombination_to_mutation_ratio": rho_over_theta,
                    "unscaled_mutation_rate": mutation_rate,
                    "threshold_years": threshold_years,
                    "generation_time_years": generation_time,
                }
                if any(
                    not math.isclose(float(decoder.get(key, math.nan)), value)
                    for key, value in expected_floats.items()
                ):
                    raise ValueError(f"Gamma-SMC rate/time contract mismatch: {path}")
                for chromosome in chromosomes:
                    if chromosome not in handle:
                        raise ValueError(f"missing {population} {chromosome}: {path}")
                    group = handle[chromosome]
                    if not bool(group.attrs.get("complete", False)):
                        raise ValueError(f"incomplete {population} {chromosome}: {path}")
                    if int(group.attrs.get("n_simulations", -1)) != n_sims:
                        raise ValueError(f"simulation count mismatch in {population} {chromosome}")
                    if int(group.attrs.get("n_pairs", -1)) != n_pairs:
                        raise ValueError(f"pair count mismatch in {population} {chromosome}")
                    positions = group["position_0based"][:]
                    ends = group["end"][:]
                    cutoffs = group["cutoff"][:]
                    max_exceedances = group["max_null_exceedances"][:]
                    ranks = group["rank_from_largest"][:]
                    if cutoffs.shape != (1, len(positions)) or len(ends) != len(positions):
                        raise ValueError(f"cutoff shape mismatch in {population} {chromosome}")
                    values = cutoffs[0].astype(np.float64)
                    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
                        raise ValueError(f"invalid cutoff values in {population} {chromosome}")
                    if not np.array_equal(max_exceedances, [expected_max_exceedances]):
                        raise ValueError(f"exceedance rank mismatch in {population} {chromosome}")
                    if not np.array_equal(ranks, [expected_rank]):
                        raise ValueError(f"cutoff rank mismatch in {population} {chromosome}")
                    if expected_rank == 1 and not np.array_equal(
                        group["cutoff"][0], group["null_max"][:]
                    ):
                        raise ValueError(f"p<=0.1 cutoff is not the null maximum: {path}")
                    population_values.append(values)
                    summaries.append(
                        _summary_row(
                            population=population,
                            scope=chromosome,
                            values=values,
                            n_chromosomes=1,
                            n_sims=n_sims,
                            n_pairs=n_pairs,
                            max_exceedances=expected_max_exceedances,
                            rank=expected_rank,
                        )
                    )
                    for position, end, cutoff in zip(positions, ends, values, strict=True):
                        yield [
                            population,
                            chromosome,
                            int(position),
                            int(end),
                            f"{DEFAULT_P_VALUE:.9g}",
                            f"{float(cutoff):.9g}",
                            n_sims,
                            n_pairs,
                            expected_max_exceedances,
                            expected_rank,
                            "observed > cutoff",
                        ]
            all_values = np.concatenate(population_values)
            summaries.append(
                _summary_row(
                    population=population,
                    scope="all_requested_chromosomes_descriptive",
                    values=all_values,
                    n_chromosomes=len(chromosomes),
                    n_sims=n_sims,
                    n_pairs=n_pairs,
                    max_exceedances=expected_max_exceedances,
                    rank=expected_rank,
                )
            )
            print(
                f"[sanity-report] {population}: position-specific cutoff "
                f"min={all_values.min():.6g} median={np.median(all_values):.6g} "
                f"max={all_values.max():.6g} positions={len(all_values):,}",
                flush=True,
            )
            cutoff_files[population] = {
                "path": str(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }

    _atomic_gzip_tsv(detail_path, detail_rows())
    summary_header = [
        "population",
        "scope",
        "n_chromosomes",
        "n_positions",
        "p_value",
        "n_simulations",
        "n_pairs",
        "max_null_exceedances",
        "rank_from_largest",
        "cutoff_min",
        "cutoff_q05",
        "cutoff_median",
        "cutoff_mean",
        "cutoff_q95",
        "cutoff_max",
    ]
    summary_text = (
        "\t".join(summary_header)
        + "\n"
        + "".join("\t".join(map(str, row)) + "\n" for row in summaries)
    )
    atomic_text(summary_path, summary_text)
    return {
        "cutoff_files": cutoff_files,
        "detail_report": {
            "path": str(detail_path),
            "sha256": sha256_file(detail_path),
            "size_bytes": detail_path.stat().st_size,
        },
        "summary_report": {
            "path": str(summary_path),
            "sha256": sha256_file(summary_path),
            "size_bytes": summary_path.stat().st_size,
        },
        "max_null_exceedances": expected_max_exceedances,
        "rank_from_largest": expected_rank,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    result.add_argument("--sim-dir", type=Path, default=DEFAULT_SIM_DIR)
    result.add_argument("--output-dir", type=Path, default=None)
    result.add_argument("--pops", default=",".join(DEFAULT_POPS))
    result.add_argument("--chroms", default="1-22")
    result.add_argument("--n-sims", type=int, default=DEFAULT_N_SIMULATIONS)
    result.add_argument("--n-random-pairs", type=int, default=DEFAULT_N_RANDOM_PAIRS)
    result.add_argument("--pairs-seed", type=int, default=DEFAULT_PAIRS_SEED)
    result.add_argument("--exclude-within", action="store_true")
    result.add_argument("--gamma-smc-repo", type=Path, default=DEFAULT_GAMMA_SMC_REPO)
    result.add_argument("--gamma-smc-aou", type=Path, default=None)
    result.add_argument("--gamma-smc-executable", type=Path, default=None)
    result.add_argument("--hardmask", type=Path, default=None)
    result.add_argument("--theta", type=float, default=DEFAULT_THETA)
    result.add_argument("--rho-over-theta", type=float, default=DEFAULT_RHO_OVER_THETA)
    result.add_argument("--mutation-rate", type=float, default=DEFAULT_MUTATION_RATE)
    result.add_argument("--threshold-years", type=float, default=DEFAULT_THRESHOLD_YEARS)
    result.add_argument("--generation-time", type=float, default=DEFAULT_GENERATION_TIME)
    result.add_argument("--stride", type=int, default=10_000)
    result.add_argument("--cache-size", type=int, default=DEFAULT_CACHE_SIZE)
    result.add_argument("--pair-block", type=int, default=DEFAULT_PAIR_BLOCK)
    result.add_argument("--decode-workers", type=int, default=4)
    result.add_argument("--decode-threads", type=int, default=4)
    result.add_argument(
        "--reserved-cpus",
        type=int,
        default=0,
        help="CPUs used by a concurrent simulation producer in this allocation",
    )
    result.add_argument(
        "--allow-cpu-oversubscription",
        action="store_true",
        help="permit native decoder threads to exceed the unreserved CPU budget",
    )
    result.add_argument("--progress-every", type=int, default=1)
    result.add_argument("--fresh", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    positive = {
        "n_sims": args.n_sims,
        "n_random_pairs": args.n_random_pairs,
        "theta": args.theta,
        "rho_over_theta": args.rho_over_theta,
        "mutation_rate": args.mutation_rate,
        "threshold_years": args.threshold_years,
        "generation_time": args.generation_time,
        "stride": args.stride,
        "cache_size": args.cache_size,
        "pair_block": args.pair_block,
        "decode_workers": args.decode_workers,
        "decode_threads": args.decode_threads,
        "progress_every": args.progress_every,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid or args.pairs_seed < 0:
        raise SystemExit(f"invalid arguments: {invalid or ['pairs_seed']}")
    if args.reserved_cpus < 0:
        raise SystemExit("--reserved-cpus must be nonnegative")
    try:
        resource_plan = cpu_resource_plan(
            workers=args.decode_workers,
            threads_per_worker=args.decode_threads,
            producer_slots_per_worker=1,
            reserved_cpus=args.reserved_cpus,
            allow_oversubscription=args.allow_cpu_oversubscription,
            label="Gamma-SMC sanity decode",
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    try:
        populations = _parse_populations(args.pops)
        chromosomes = parse_chroms(args.chroms)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    sim_dir = args.sim_dir.expanduser().resolve()
    output_root = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else _default_output_dir(sim_dir, args.n_sims, args.n_random_pairs)
    )
    output_root.mkdir(parents=True, exist_ok=True)
    started_utc = _utc_now()
    try:
        gamma_repository = args.gamma_smc_repo.expanduser().resolve()
        gamma_aou = (
            args.gamma_smc_aou.expanduser().resolve()
            if args.gamma_smc_aou is not None
            else gamma_repository / "scripts" / "aou.sh"
        )
        gamma_executable = (
            args.gamma_smc_executable.expanduser().resolve()
            if args.gamma_smc_executable is not None
            else gamma_repository / "bin" / "gamma_smc"
        )
        if not gamma_aou.is_file() or not gamma_executable.is_file():
            raise FileNotFoundError(
                f"Gamma-SMC launcher/binary missing: {gamma_aou}, {gamma_executable}"
            )
        print(f"[sanity-resources] {json.dumps(resource_plan, sort_keys=True)}", flush=True)
        preflight = preflight_completed_units(
            sim_dir=sim_dir,
            populations=populations,
            chromosomes=chromosomes,
            n_sims=args.n_sims,
            n_pairs=args.n_random_pairs,
            exclude_within=args.exclude_within,
        )
        command = reducer_command(
            args=args,
            sim_dir=sim_dir,
            output_root=output_root,
            populations=populations,
            chromosomes=chromosomes,
        )
        print(f"[sanity] reducer={shlex.join(command)}", flush=True)
        subprocess.run(command, check=True)
        reports = write_sanity_reports(
            output_root=output_root,
            populations=populations,
            chromosomes=chromosomes,
            n_sims=args.n_sims,
            n_pairs=args.n_random_pairs,
            pairs_seed=args.pairs_seed,
            exclude_within=args.exclude_within,
            theta=args.theta,
            rho_over_theta=args.rho_over_theta,
            mutation_rate=args.mutation_rate,
            threshold_years=args.threshold_years,
            generation_time=args.generation_time,
            stride=args.stride,
        )
    except (KeyError, OSError, TypeError, ValueError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"Gamma-SMC sanity workflow failed: {error}") from error

    manifest = {
        "schema": SANITY_SCHEMA,
        "started_utc": started_utc,
        "completed_utc": _utc_now(),
        "host": platform.node(),
        "python": platform.python_version(),
        "simulation_root": str(sim_dir),
        "output_root": str(output_root),
        "software": {
            "simulatephase2_repository": str(Path(__file__).resolve().parent),
            "simulatephase2_git_revision": _git_revision(Path(__file__).resolve().parent),
            "sanity_script_sha256": sha256_file(Path(__file__).resolve()),
            "gamma_cutoff_reducer_sha256": sha256_file(
                Path(__file__).resolve().with_name("generate_cutoff_gamma_smc.py")
            ),
            "gamma_smc_repository": str(gamma_repository),
            "gamma_smc_git_revision": _git_revision(gamma_repository),
            "gamma_smc_aou_sha256": sha256_file(gamma_aou),
            "gamma_smc_executable_sha256": sha256_file(gamma_executable),
            "numpy": np.__version__,
            "h5py": h5py.__version__,
        },
        "populations": populations,
        "chromosomes": chromosomes,
        "resources": resource_plan,
        "preflight": preflight,
        "decoder": {
            "n_random_pairs": args.n_random_pairs,
            "pairs_seed": args.pairs_seed,
            "exclude_within": args.exclude_within,
            "theta": args.theta,
            "rho_over_theta": args.rho_over_theta,
            "mutation_rate": args.mutation_rate,
            "threshold_years": args.threshold_years,
            "generation_time": args.generation_time,
            "stride": args.stride,
            "cache_size": args.cache_size,
            "pair_block": args.pair_block,
            "decode_workers": args.decode_workers,
            "decode_threads": args.decode_threads,
        },
        "statistical_contract": {
            "statistic": STATISTIC_COLUMN,
            "p_value": DEFAULT_P_VALUE,
            "tail": "upper",
            "monte_carlo_method": "(1 + count(null >= observed)) / (R + 1)",
            "minimum_attainable_p_value": 1.0 / (args.n_sims + 1),
            "max_null_exceedances": reports["max_null_exceedances"],
            "rank_from_largest": reports["rank_from_largest"],
            "significance_rule": "observed > position-specific cutoff",
            "pooled_summary_is_descriptive_only": True,
        },
        "reducer_command": command,
        "outputs": reports,
    }
    manifest_path = output_root / "manifest.json"
    atomic_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    checksum_paths = [
        *(Path(record["path"]) for record in reports["cutoff_files"].values()),
        Path(reports["detail_report"]["path"]),
        Path(reports["summary_report"]["path"]),
        manifest_path,
    ]
    checksum_text = "".join(
        f"{sha256_file(path)}  {path.relative_to(output_root).as_posix()}\n"
        for path in checksum_paths
    )
    checksum_path = output_root / "checksums.sha256"
    atomic_text(checksum_path, checksum_text)
    print(f"[sanity] exact_cutoffs={output_root / DETAIL_REPORT_NAME}", flush=True)
    print(f"[sanity] descriptive_summary={output_root / SUMMARY_REPORT_NAME}", flush=True)
    print(f"[sanity] manifest={manifest_path}", flush=True)
    print(f"[sanity] checksums={checksum_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
