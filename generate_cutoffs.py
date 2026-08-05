#!/usr/bin/env python
"""Generate compact per-window TMRCA cutoffs from completed tree-sequence nulls.

The statistic is tree truth: the span-averaged fraction of all unordered sample
pairs whose local TMRCA is below the requested threshold. It is deliberately
labelled as truth and must not be presented as a Gamma-SMC posterior cutoff
unless the empirical and null statistics have first been shown to be calibrated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import tszip

from phase2_map import DEFAULT_POPS, parse_chroms
from run_sim import (
    DEFAULT_SIM_DIR,
    SIM_ROOT_CONTRACT_NAME,
    SIM_ROOT_CONTRACT_SCHEMA,
    output_lock,
)
from simulation_outputs import completed_units

CUTOFF_SCHEMA = "simulatephase2.tmrca-cutoffs/v1"
DEFAULT_WINDOW_SIZE = 10_000
DEFAULT_THRESHOLD_YEARS = 4_500.0
DEFAULT_GENERATION_TIME = 25.0
DEFAULT_P_VALUES = (0.01, 0.05)
ALL_AUTOSOMES = tuple(f"chr{index}" for index in range(1, 23))


def parse_p_values(spec: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in spec.split(",") if part.strip())
    if not values or any(not 0.0 < value < 1.0 for value in values):
        raise ValueError("p-values must be a nonempty comma-separated list inside (0, 1)")
    if len(values) != len(set(values)):
        raise ValueError("p-values must not contain duplicates")
    return tuple(sorted(values))


def json_digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def load_population_contract(sim_dir: Path, population: str) -> tuple[dict[str, object], str]:
    path = sim_dir / SIM_ROOT_CONTRACT_NAME
    try:
        contract = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise ValueError(f"cannot read simulation contract {path}: {error}") from error
    if not isinstance(contract, dict) or contract.get("schema") != SIM_ROOT_CONTRACT_SCHEMA:
        raise ValueError(f"invalid simulation contract: {path}")
    global_contract = contract.get("global")
    populations = contract.get("populations")
    if not isinstance(global_contract, dict) or not isinstance(populations, dict):
        raise ValueError(f"invalid simulation contract mappings: {path}")
    population_contract = populations.get(population)
    if not isinstance(population_contract, dict):
        raise ValueError(f"simulation contract has no {population} entry: {path}")
    selected = {"global": global_contract, "population": population_contract}
    return selected, json_digest(selected)


def window_breaks(sequence_length: float, window_size: int) -> np.ndarray:
    length = int(round(sequence_length))
    if length <= 0 or not np.isclose(sequence_length, length):
        raise ValueError(f"tree sequence length must be a positive integer: {sequence_length}")
    breaks = np.arange(0, length, window_size, dtype=np.float64)
    if len(breaks) == 0 or breaks[-1] != length:
        breaks = np.append(breaks, float(length))
    return breaks


def truth_recent_pair_fraction(
    path: Path,
    *,
    window_size: int,
    threshold_generations: float,
) -> tuple[int, int, np.ndarray]:
    ts = tszip.decompress(str(path))
    if ts.num_samples < 2:
        raise ValueError(f"tree sequence has fewer than two samples: {path}")
    breaks = window_breaks(ts.sequence_length, window_size)
    counts = ts.pair_coalescence_counts(
        windows=breaks,
        time_windows=np.asarray([0.0, threshold_generations, np.inf]),
        span_normalise=True,
        pair_normalise=True,
    )
    recent = np.asarray(counts[:, 0], dtype=np.float64)
    if not np.isfinite(recent).all() or np.any((recent < 0.0) | (recent > 1.0)):
        raise ValueError(f"invalid recent-coalescence fractions from {path}")
    return int(round(ts.sequence_length)), ts.num_samples, recent


def _profile_task(payload: tuple[str, int, float]) -> tuple[int, int, np.ndarray]:
    path, window_size, threshold_generations = payload
    return truth_recent_pair_fraction(
        Path(path),
        window_size=window_size,
        threshold_generations=threshold_generations,
    )


def collect_profiles(
    paths: list[Path],
    *,
    window_size: int,
    threshold_generations: float,
    workers: int,
    progress_every: int,
) -> tuple[int, int, np.ndarray]:
    started = time.monotonic()
    payloads = [(str(path), window_size, threshold_generations) for path in paths]
    if workers == 1:
        iterator = map(_profile_task, payloads)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        iterator = executor.map(_profile_task, payloads, chunksize=1)
    matrix: np.ndarray | None = None
    expected_length = expected_samples = -1
    try:
        for index, (length, samples, profile) in enumerate(iterator):
            if matrix is None:
                expected_length, expected_samples = length, samples
                matrix = np.empty((len(paths), len(profile)), dtype=np.float32)
            if (
                length != expected_length
                or samples != expected_samples
                or len(profile) != matrix.shape[1]
            ):
                raise ValueError(
                    "simulation units do not share sequence length, samples, and windows"
                )
            matrix[index] = profile
            if index < 4 or (index + 1) % progress_every == 0 or index + 1 == len(paths):
                print(
                    f"[cutoffs] profiles={index + 1:,}/{len(paths):,} "
                    f"elapsed={(time.monotonic() - started) / 60:.1f} min",
                    flush=True,
                )
    finally:
        if executor is not None:
            executor.shutdown()
    if matrix is None:
        raise ValueError("no simulation profiles were collected")
    return expected_length, expected_samples, matrix


def monte_carlo_cutoffs(
    null: np.ndarray, p_values: tuple[float, ...]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return conservative tie-safe cutoffs for plus-one upper-tail p-values.

    An observation is significant when it is strictly greater than the stored
    cutoff. For alpha, the cutoff is the (C+1)-th largest null value, where
    C=floor(alpha * (R+1) - 1) is the largest allowed null exceedance count.
    """
    values = np.asarray(null, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 1 or not np.isfinite(values).all():
        raise ValueError("null matrix must be a finite simulations-by-windows matrix")
    n_sims = values.shape[0]
    max_exceedances = np.asarray(
        [math.floor(np.nextafter(alpha * (n_sims + 1) - 1.0, np.inf)) for alpha in p_values],
        dtype=np.int64,
    )
    if np.any(max_exceedances < 0):
        minimum = 1.0 / (n_sims + 1)
        raise ValueError(
            f"requested p-value is below the Monte Carlo resolution {minimum:.8g}"
        )
    max_exceedances = np.minimum(max_exceedances, n_sims - 1)
    ranks = max_exceedances + 1
    cutoffs = np.empty((len(p_values), values.shape[1]), dtype=np.float32)
    for index, rank in enumerate(ranks):
        partition_index = n_sims - int(rank)
        cutoffs[index] = np.partition(values, partition_index, axis=0)[partition_index]
    return cutoffs, max_exceedances, ranks


def _root_compatible(
    handle: h5py.File,
    *,
    population: str,
    contract_key: str,
    window_size: int,
    threshold_years: float,
    generation_time: float,
    p_values: tuple[float, ...],
    n_sims: int,
) -> bool:
    return (
        str(handle.attrs.get("schema", "")) == CUTOFF_SCHEMA
        and str(handle.attrs.get("population", "")) == population
        and str(handle.attrs.get("simulation_contract_key", "")) == contract_key
        and int(handle.attrs.get("window_size", -1)) == window_size
        and float(handle.attrs.get("threshold_years", np.nan)) == threshold_years
        and float(handle.attrs.get("generation_time_years", np.nan)) == generation_time
        and int(handle.attrs.get("n_simulations", -1)) == n_sims
        and "p_value" in handle
        and np.array_equal(handle["p_value"][:], np.asarray(p_values))
    )


def _initialise_root(
    handle: h5py.File,
    *,
    population: str,
    contract: dict[str, object],
    contract_key: str,
    window_size: int,
    threshold_years: float,
    generation_time: float,
    p_values: tuple[float, ...],
    n_sims: int,
) -> None:
    handle.attrs.update(
        {
            "schema": CUTOFF_SCHEMA,
            "complete": False,
            "population": population,
            "simulation_contract_key": contract_key,
            "simulation_contract_json": json.dumps(contract, sort_keys=True),
            "window_size": window_size,
            "threshold_years": threshold_years,
            "generation_time_years": generation_time,
            "threshold_generations": threshold_years / generation_time,
            "n_simulations": n_sims,
            "source_kind": "tree_truth",
            "statistic": "span_mean_fraction_all_unordered_sample_pairs_tmrca_lt_threshold",
            "tail": "upper",
            "monte_carlo_method": "(1 + count(null >= observed)) / (R + 1)",
            "significance_rule": "observed > cutoff",
        }
    )
    handle.create_dataset("p_value", data=np.asarray(p_values, dtype=np.float64))


def _group_complete(
    group: h5py.Group,
    *,
    source_digest: str,
    n_sims: int,
    n_windows: int | None = None,
) -> bool:
    required = {"start", "end", "cutoff", "null_mean", "null_sd", "null_min", "null_max"}
    if (
        not bool(group.attrs.get("complete", False))
        or str(group.attrs.get("source_manifest_sha256", "")) != source_digest
        or int(group.attrs.get("n_simulations", -1)) != n_sims
        or not required <= set(group.keys())
    ):
        return False
    return n_windows is None or len(group["start"]) == n_windows


def write_population_cutoffs(
    *,
    sim_dir: Path,
    output: Path,
    population: str,
    chromosomes: list[str],
    n_sims: int,
    window_size: int,
    threshold_years: float,
    generation_time: float,
    p_values: tuple[float, ...],
    workers: int,
    progress_every: int,
    fresh: bool,
) -> Path:
    contract, contract_key = load_population_contract(sim_dir, population)
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.with_suffix(output.suffix + ".lock")
    with output_lock(lock):
        if fresh:
            output.unlink(missing_ok=True)
        with h5py.File(output, "a") as handle:
            if not handle.attrs:
                _initialise_root(
                    handle,
                    population=population,
                    contract=contract,
                    contract_key=contract_key,
                    window_size=window_size,
                    threshold_years=threshold_years,
                    generation_time=generation_time,
                    p_values=p_values,
                    n_sims=n_sims,
                )
            elif not _root_compatible(
                handle,
                population=population,
                contract_key=contract_key,
                window_size=window_size,
                threshold_years=threshold_years,
                generation_time=generation_time,
                p_values=p_values,
                n_sims=n_sims,
            ):
                raise ValueError(
                    f"incompatible cutoff output; use --fresh or another path: {output}"
                )
            for name in list(handle):
                if name.startswith("__tmp_"):
                    del handle[name]
            handle.flush()

        for chromosome in chromosomes:
            paths, source_digest = completed_units(sim_dir, population, chromosome, n_sims)
            with h5py.File(output, "r") as handle:
                if chromosome in handle and _group_complete(
                    handle[chromosome], source_digest=source_digest, n_sims=n_sims
                ):
                    print(
                        f"[cutoffs] {population} {chromosome}: compatible output exists",
                        flush=True,
                    )
                    continue
            with h5py.File(output, "a") as handle:
                handle.attrs["complete"] = False
                handle.flush()
            print(f"[cutoffs] {population} {chromosome}: reading {n_sims:,} nulls", flush=True)
            length, samples, null = collect_profiles(
                paths,
                window_size=window_size,
                threshold_generations=threshold_years / generation_time,
                workers=workers,
                progress_every=progress_every,
            )
            cutoffs, max_exceedances, ranks = monte_carlo_cutoffs(null, p_values)
            starts = np.arange(0, length, window_size, dtype=np.int64)
            ends = np.minimum(starts + window_size, length)
            if null.shape[1] != len(starts):
                raise RuntimeError("TMRCA profile does not match the expected window geometry")
            temporary_name = f"__tmp_{chromosome}_{os.getpid()}"
            with h5py.File(output, "a") as handle:
                if temporary_name in handle:
                    del handle[temporary_name]
                group = handle.create_group(temporary_name)
                group.attrs.update(
                    {
                        "complete": False,
                        "chromosome": chromosome,
                        "sequence_length": length,
                        "n_samples": samples,
                        "n_simulations": n_sims,
                        "source_manifest_sha256": source_digest,
                    }
                )
                integer_options = {
                    "compression": "gzip",
                    "compression_opts": 6,
                    "shuffle": True,
                    "fletcher32": True,
                }
                float_options = integer_options | {"dtype": np.float32}
                group.create_dataset("start", data=starts, **integer_options)
                group.create_dataset("end", data=ends, **integer_options)
                group.create_dataset("cutoff", data=cutoffs, **float_options)
                group.create_dataset("null_mean", data=null.mean(axis=0), **float_options)
                group.create_dataset("null_sd", data=null.std(axis=0, ddof=1), **float_options)
                group.create_dataset("null_min", data=null.min(axis=0), **float_options)
                group.create_dataset("null_max", data=null.max(axis=0), **float_options)
                group.create_dataset("max_null_exceedances", data=max_exceedances)
                group.create_dataset("rank_from_largest", data=ranks)
                group.attrs["complete"] = True
                if chromosome in handle:
                    del handle[chromosome]
                handle.move(temporary_name, chromosome)
                handle.attrs["complete"] = all(
                    chrom in handle and bool(handle[chrom].attrs.get("complete", False))
                    for chrom in ALL_AUTOSOMES
                )
                handle.flush()
            with output.open("r+b") as handle:
                os.fsync(handle.fileno())
            del null
            print(
                f"[cutoffs] {population} {chromosome}: {len(starts):,} windows -> {output}",
                flush=True,
            )
    return output


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--sim-dir", type=Path, default=DEFAULT_SIM_DIR)
    result.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: write one HDF5 under each <sim-dir>/<pop>/ directory",
    )
    result.add_argument("--pops", default=",".join(DEFAULT_POPS))
    result.add_argument("--chroms", default="1-22")
    result.add_argument("--n-sims", type=int, default=1_000)
    result.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    result.add_argument("--threshold-years", type=float, default=DEFAULT_THRESHOLD_YEARS)
    result.add_argument("--generation-time", type=float, default=DEFAULT_GENERATION_TIME)
    result.add_argument("--p-values", default=",".join(map(str, DEFAULT_P_VALUES)))
    result.add_argument("--workers", type=int, default=4)
    result.add_argument("--progress-every", type=int, default=50)
    result.add_argument("--fresh", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    positive = {
        "n_sims": args.n_sims,
        "window_size": args.window_size,
        "threshold_years": args.threshold_years,
        "generation_time": args.generation_time,
        "workers": args.workers,
        "progress_every": args.progress_every,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise SystemExit(f"invalid nonpositive arguments: {invalid}")
    try:
        p_values = parse_p_values(args.p_values)
        chromosomes = parse_chroms(args.chroms)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    populations = [part.strip().upper() for part in args.pops.split(",") if part.strip()]
    unknown = [population for population in populations if population not in DEFAULT_POPS]
    if unknown or not populations or len(populations) != len(set(populations)):
        raise SystemExit(f"invalid or duplicate populations: {unknown or populations}")
    sim_dir = args.sim_dir.expanduser().resolve()
    for population in populations:
        output = (
            args.output_dir.expanduser().resolve() / f"{population.lower()}.tmrca_cutoffs.10kb.h5"
            if args.output_dir is not None
            else sim_dir / population.lower() / "tmrca_cutoffs.10kb.h5"
        )
        try:
            write_population_cutoffs(
                sim_dir=sim_dir,
                output=output,
                population=population,
                chromosomes=chromosomes,
                n_sims=args.n_sims,
                window_size=args.window_size,
                threshold_years=args.threshold_years,
                generation_time=args.generation_time,
                p_values=p_values,
                workers=args.workers,
                progress_every=args.progress_every,
                fresh=args.fresh,
            )
        except Exception as error:
            raise SystemExit(f"{population} cutoff generation failed: {error}") from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
