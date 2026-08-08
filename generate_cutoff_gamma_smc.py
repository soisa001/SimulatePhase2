#!/usr/bin/env python3
"""Decode completed TSZip nulls with Gamma-SMC and write compact 10 kb cutoffs.

The decoder contract matches the empirical within-individual scan: fixed
theta/rho/mutation time scale, one homolog pair per diploid, stride-only
output, and the mean posterior ``P(TMRCA < threshold)`` statistic. Decoder TSVs
are reduced immediately to restartable compressed profiles and are not kept.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np

from generate_cutoffs import (
    ALL_AUTOSOMES,
    DEFAULT_GENERATION_TIME,
    DEFAULT_P_VALUES,
    DEFAULT_THRESHOLD_YEARS,
    DEFAULT_WINDOW_SIZE,
    json_digest,
    load_population_contract,
    monte_carlo_cutoffs,
    parse_p_values,
)
from phase2_map import (
    DEFAULT_GAMMA_SMC_REPO,
    DEFAULT_HARDMASK_PATH,
    DEFAULT_POPS,
    clip_merged_mask,
    load_mask,
    parse_chroms,
)
from run_sim import DEFAULT_SIM_DIR, atomic_text, output_lock, sha256_file
from simulation_outputs import completed_units, validate_completed_unit

GAMMA_CUTOFF_SCHEMA = "simulatephase2.gamma-smc-cutoffs/v1"
DEFAULT_THETA = 0.00075
DEFAULT_RHO_OVER_THETA = 0.8
DEFAULT_MUTATION_RATE = 1.29e-8
DEFAULT_CACHE_SIZE = 1_000
DEFAULT_PAIR_BLOCK = 256
STATISTIC_COLUMN = "mean_p_tmrca_lt_threshold"
POSITION_COLUMN = "position_0based"


def _command_prefix(path: Path) -> list[str]:
    if path.suffix.lower() == ".py":
        return [sys.executable, str(path)]
    if path.suffix.lower() == ".sh":
        return ["bash", str(path)]
    return [str(path)]


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


def _interface_sha256(aou: Path) -> str:
    repository = aou.parent.parent
    candidates = [
        aou,
        repository / "python" / "gamma_smc_aou" / "cli.py",
        repository / "python" / "gamma_smc_aou" / "decoder.py",
        repository / "python" / "gamma_smc_aou" / "defaults.py",
    ]
    digest = hashlib.sha256()
    for path in candidates:
        if path.is_file():
            digest.update(str(path.relative_to(repository)).replace("\\", "/").encode())
            digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def decoder_contract(
    *,
    aou: Path,
    executable: Path,
    mask_sha256: str,
    theta: float,
    rho_over_theta: float,
    mutation_rate: float,
    threshold_years: float,
    generation_time: float,
    stride: int,
    cache_size: int,
    threads: int,
    pair_block: int,
) -> dict[str, object]:
    return {
        "schema": "simulatephase2.gamma-smc-decode/v1",
        "aou_launcher_sha256": sha256_file(aou),
        "gamma_smc_interface_sha256": _interface_sha256(aou),
        "gamma_smc_executable_sha256": sha256_file(executable),
        "input_format": "tsz",
        "pair_selection": "only_within",
        "scaled_mutation_rate_theta": theta,
        "recombination_to_mutation_ratio": rho_over_theta,
        "unscaled_mutation_rate": mutation_rate,
        "threshold_years": threshold_years,
        "generation_time_years": generation_time,
        "output_at_stride": stride,
        "output_at_hets": False,
        "recent_call": "mean",
        "statistic": STATISTIC_COLUMN,
        "cache_size_bp": cache_size,
        "threads_per_decode": threads,
        "pair_block": pair_block,
        "exp10": "accurate",
        "backward_alignment": "fixed",
        "callable_mask_sha256": mask_sha256,
    }


def _full_callable_mask(length: int, excluded: np.ndarray) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    cursor = 0
    for start, end in np.asarray(excluded, dtype=np.int64).reshape(-1, 2):
        if cursor < int(start):
            result.append((cursor, int(start)))
        cursor = max(cursor, int(end))
    if cursor < length:
        result.append((cursor, length))
    return result


def prepare_callable_masks(
    *,
    sim_dir: Path,
    chromosomes: list[str],
    sequence_lengths: dict[str, int],
    hardmask: Path | None,
    expected_hardmask_sha256: str,
) -> dict[str, Path]:
    no_mask_sha = hashlib.sha256(b"NO_MASK").hexdigest()
    if hardmask is None:
        if expected_hardmask_sha256 != no_mask_sha:
            raise ValueError(
                "--hardmask is required: simulations used an exclusion mask, and Gamma-SMC "
                "must use its callable complement"
            )
        excluded = {
            chromosome: np.empty((0, 2), dtype=np.int64) for chromosome in chromosomes
        }
    else:
        hardmask = hardmask.expanduser().resolve()
        if not hardmask.is_file():
            raise FileNotFoundError(hardmask)
        observed = sha256_file(hardmask)
        if observed != expected_hardmask_sha256:
            raise ValueError(
                f"hardmask SHA256 differs from simulation contract: "
                f"{observed} != {expected_hardmask_sha256}"
            )
        excluded = load_mask(hardmask, chromosomes)

    directory = sim_dir / ".gamma_smc" / "callable_masks" / expected_hardmask_sha256
    directory.mkdir(parents=True, exist_ok=True)
    result: dict[str, Path] = {}
    for chromosome in chromosomes:
        length = sequence_lengths[chromosome]
        clipped = (
            clip_merged_mask(excluded[chromosome], length)
            if len(excluded[chromosome])
            else excluded[chromosome]
        )
        intervals = _full_callable_mask(length, clipped)
        if not intervals:
            raise ValueError(f"hardmask leaves no callable bases on {chromosome}")
        # Direct tree-sequence conversion writes a single-contig VCF whose ID
        # is "1". Genomic chromosome identity remains in the HDF5 group.
        text = "".join(f"1\t{start}\t{end}\n" for start, end in intervals)
        digest = hashlib.sha256(text.encode()).hexdigest()
        path = directory / f"{chromosome}.{length}.{digest}.callable.bed"
        lock = path.with_suffix(path.suffix + ".lock")
        with output_lock(lock):
            if not path.is_file() or sha256_file(path) != digest:
                atomic_text(path, text)
        result[chromosome] = path
    return result


def decode_command(
    *,
    aou: Path,
    executable: Path,
    input_path: Path,
    output: Path,
    callable_mask: Path,
    contract: dict[str, object],
) -> list[str]:
    return [
        *_command_prefix(aou),
        "decode",
        "--executable",
        str(executable),
        "--input",
        str(input_path),
        "--input-format",
        "tsz",
        "--output",
        str(output),
        "--mask",
        str(callable_mask),
        "--theta",
        str(contract["scaled_mutation_rate_theta"]),
        "--rho-over-theta",
        str(contract["recombination_to_mutation_ratio"]),
        "--mutation-rate",
        str(contract["unscaled_mutation_rate"]),
        "--threshold-years",
        str(contract["threshold_years"]),
        "--generation-time",
        str(contract["generation_time_years"]),
        "--recent-call",
        "mean",
        "--no-output-at-hets",
        "--output-at-stride",
        str(contract["output_at_stride"]),
        "--threads",
        str(contract["threads_per_decode"]),
        "--cache-size",
        str(contract["cache_size_bp"]),
        "--pair-block",
        str(contract["pair_block"]),
        "--exp10",
        "accurate",
        "--backward-alignment",
        "fixed",
    ]


def _read_summary(path: Path, expected_positions: np.ndarray) -> tuple[np.ndarray, int]:
    positions: list[int] = []
    values: list[float] = []
    pair_counts: set[int] = set()
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = set(reader.fieldnames or [])
        required = {POSITION_COLUMN, STATISTIC_COLUMN, "n_pairs"}
        if not required <= fields:
            raise ValueError(f"Gamma-SMC summary lacks {sorted(required - fields)}: {path}")
        for row in reader:
            positions.append(int(row[POSITION_COLUMN]))
            values.append(float(row[STATISTIC_COLUMN]))
            pair_counts.add(int(row["n_pairs"]))
    position_array = np.asarray(positions, dtype=np.int64)
    profile = np.asarray(values, dtype=np.float64)
    if not np.array_equal(position_array, expected_positions):
        raise ValueError(f"Gamma-SMC output positions do not match the stride grid: {path}")
    if (
        len(pair_counts) != 1
        or next(iter(pair_counts), 0) <= 0
        or not np.isfinite(profile).all()
        or np.any((profile < 0.0) | (profile > 1.0))
    ):
        raise ValueError(f"Gamma-SMC summary contains invalid posterior values: {path}")
    return profile.astype(np.float32), next(iter(pair_counts))


def _valid_profile(
    path: Path,
    *,
    key: str,
    expected_positions: np.ndarray,
) -> tuple[np.ndarray, int] | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"].item()))
            positions = data["position_0based"].astype(np.int64, copy=False)
            values = data["mean_p_tmrca_lt_threshold"].astype(np.float32, copy=False)
            n_pairs = int(data["n_pairs"].item())
        if (
            metadata.get("decode_key") != key
            or not np.array_equal(positions, expected_positions)
            or len(values) != len(expected_positions)
            or n_pairs <= 0
            or not np.isfinite(values).all()
            or np.any((values < 0.0) | (values > 1.0))
        ):
            return None
        return values, n_pairs
    except Exception:
        return None


def decode_profile(
    *,
    input_path: Path,
    profile_path: Path,
    simulation: int,
    chromosome: str,
    source_signature: str,
    aou: Path,
    executable: Path,
    callable_mask: Path,
    contract: dict[str, object],
    expected_positions: np.ndarray,
) -> tuple[np.ndarray, int, bool, float]:
    key = json_digest(
        {
            "source_signature": source_signature,
            "decoder_contract": contract,
            "chromosome": chromosome,
            "simulation": simulation,
        }
    )
    existing = _valid_profile(profile_path, key=key, expected_positions=expected_positions)
    if existing is not None:
        return existing[0], existing[1], True, 0.0

    profile_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_summary = profile_path.parent / f".{profile_path.stem}.{uuid.uuid4().hex}.tsv"
    failed = profile_path.with_suffix(profile_path.suffix + ".failed.json")
    command = decode_command(
        aou=aou,
        executable=executable,
        input_path=input_path,
        output=temporary_summary,
        callable_mask=callable_mask,
        contract=contract,
    )
    started = time.monotonic()
    completed = subprocess.run(command, text=True, capture_output=True)
    seconds = time.monotonic() - started
    if completed.returncode != 0:
        atomic_text(
            failed,
            json.dumps(
                {
                    "command": command,
                    "returncode": completed.returncode,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                    "seconds": seconds,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )
        temporary_summary.unlink(missing_ok=True)
        raise RuntimeError(f"Gamma-SMC decode failed; see {failed}")
    try:
        values, n_pairs = _read_summary(temporary_summary, expected_positions)
        metadata = {
            "schema": "simulatephase2.gamma-smc-profile/v1",
            "decode_key": key,
            "source_signature": source_signature,
            "chromosome": chromosome,
            "simulation": simulation,
            "n_pairs": n_pairs,
            "decode_seconds": seconds,
            "command": command,
            "gamma_run": json.loads(
                temporary_summary.with_suffix(temporary_summary.suffix + ".run.json").read_text(
                    encoding="utf-8"
                )
            ),
        }
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{profile_path.name}.", dir=profile_path.parent
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                np.savez_compressed(
                    handle,
                    position_0based=expected_positions,
                    mean_p_tmrca_lt_threshold=values,
                    n_pairs=np.asarray(n_pairs, dtype=np.int64),
                    metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, profile_path)
        finally:
            Path(temporary_name).unlink(missing_ok=True)
        failed.unlink(missing_ok=True)
        return values, n_pairs, False, seconds
    finally:
        temporary_summary.unlink(missing_ok=True)
        temporary_summary.with_suffix(temporary_summary.suffix + ".run.json").unlink(
            missing_ok=True
        )


def _group_complete(
    group: h5py.Group,
    *,
    source_digest: str,
    n_sims: int,
    n_positions: int,
) -> bool:
    required = {"position_0based", "end", "cutoff", "null_mean", "null_sd", "null_min", "null_max"}
    return (
        bool(group.attrs.get("complete", False))
        and str(group.attrs.get("source_manifest_sha256", "")) == source_digest
        and int(group.attrs.get("n_simulations", -1)) == n_sims
        and required <= set(group.keys())
        and len(group["position_0based"]) == n_positions
    )


def _write_root(
    handle: h5py.File,
    *,
    population: str,
    simulation_contract: dict[str, object],
    simulation_contract_key: str,
    decode_contract: dict[str, object],
    p_values: tuple[float, ...],
    n_sims: int,
) -> None:
    handle.attrs.update(
        {
            "schema": GAMMA_CUTOFF_SCHEMA,
            "complete": False,
            "population": population,
            "simulation_contract_key": simulation_contract_key,
            "simulation_contract_json": json.dumps(simulation_contract, sort_keys=True),
            "decoder_contract_key": json_digest(decode_contract),
            "decoder_contract_json": json.dumps(decode_contract, sort_keys=True),
            "n_simulations": n_sims,
            "source_kind": "gamma_smc_posterior",
            "statistic": STATISTIC_COLUMN,
            "pair_selection": "within_individual_homolog_pair",
            "tail": "upper",
            "monte_carlo_method": "(1 + count(null >= observed)) / (R + 1)",
            "significance_rule": "observed > cutoff",
        }
    )
    handle.create_dataset("p_value", data=np.asarray(p_values, dtype=np.float64))


def _root_compatible(
    handle: h5py.File,
    *,
    population: str,
    simulation_contract_key: str,
    decode_contract: dict[str, object],
    p_values: tuple[float, ...],
    n_sims: int,
) -> bool:
    return (
        str(handle.attrs.get("schema", "")) == GAMMA_CUTOFF_SCHEMA
        and str(handle.attrs.get("population", "")) == population
        and str(handle.attrs.get("simulation_contract_key", "")) == simulation_contract_key
        and str(handle.attrs.get("decoder_contract_key", "")) == json_digest(decode_contract)
        and int(handle.attrs.get("n_simulations", -1)) == n_sims
        and "p_value" in handle
        and np.array_equal(handle["p_value"][:], np.asarray(p_values))
    )


def write_population_cutoffs(
    *,
    sim_dir: Path,
    output: Path,
    population: str,
    chromosomes: list[str],
    n_sims: int,
    p_values: tuple[float, ...],
    aou: Path,
    executable: Path,
    callable_masks: dict[str, Path],
    decode_contracts: dict[str, dict[str, object]],
    decode_workers: int,
    progress_every: int,
    keep_profiles: bool,
    fresh: bool,
) -> Path:
    simulation_contract, simulation_contract_key = load_population_contract(sim_dir, population)
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.with_suffix(output.suffix + ".lock")
    with output_lock(lock):
        if fresh:
            output.unlink(missing_ok=True)
        with h5py.File(output, "a") as handle:
            first_contract = decode_contracts[chromosomes[0]]
            initialize_root = not handle.attrs
            if handle.attrs and not _root_compatible(
                handle,
                population=population,
                simulation_contract_key=simulation_contract_key,
                decode_contract=first_contract,
                p_values=p_values,
                n_sims=n_sims,
            ):
                completed_groups = [
                    name
                    for name in handle
                    if isinstance(handle[name], h5py.Group)
                    and bool(handle[name].attrs.get("complete", False))
                ]
                if bool(handle.attrs.get("complete", False)) or completed_groups:
                    raise ValueError(f"incompatible cutoff output; use --fresh: {output}")
                print(
                    f"[gamma-cutoffs] reinitializing empty incompatible output: {output}",
                    flush=True,
                )
                for name in list(handle):
                    del handle[name]
                for name in list(handle.attrs):
                    del handle.attrs[name]
                initialize_root = True
            if initialize_root:
                _write_root(
                    handle,
                    population=population,
                    simulation_contract=simulation_contract,
                    simulation_contract_key=simulation_contract_key,
                    decode_contract=first_contract,
                    p_values=p_values,
                    n_sims=n_sims,
                )
                handle.attrs.update(
                    {
                        "gamma_smc_repository_path": str(aou.parent.parent),
                        "gamma_smc_git_revision": _git_revision(aou.parent.parent),
                        "gamma_smc_aou_launcher_path": str(aou),
                        "gamma_smc_executable_path": str(executable),
                    }
                )
            for name in list(handle):
                if name.startswith("__tmp_"):
                    del handle[name]
            handle.flush()

        for chromosome in chromosomes:
            paths, unit_digest = completed_units(sim_dir, population, chromosome, n_sims)
            first_unit = validate_completed_unit(
                paths[0],
                population=population,
                simulation=0,
                chromosome=chromosome,
            )
            unit_contract = first_unit.metadata["contract"]
            if not isinstance(unit_contract, dict):
                raise ValueError(f"invalid unit contract: {first_unit.sidecar}")
            length = int(unit_contract["sequence_length"])
            diploid_samples = int(unit_contract["diploid_samples"])
            contract = decode_contracts[chromosome]
            expected_positions = np.arange(
                0, length, int(contract["output_at_stride"]), dtype=np.int64
            )
            source_digest = json_digest(
                {
                    "unit_manifest_sha256": unit_digest,
                    "decoder_contract": contract,
                    "callable_mask_sha256": sha256_file(callable_masks[chromosome]),
                }
            )
            profile_root = sim_dir / population.lower() / ".gamma_smc_profiles"
            profile_paths = [
                profile_root / f"sim_{simulation:05d}" / f"{chromosome}.npz"
                for simulation in range(n_sims)
            ]
            with h5py.File(output, "r") as handle:
                if chromosome in handle and _group_complete(
                    handle[chromosome],
                    source_digest=source_digest,
                    n_sims=n_sims,
                    n_positions=len(expected_positions),
                ):
                    print(f"[gamma-cutoffs] {population} {chromosome}: compatible output exists")
                    if not keep_profiles:
                        for profile_path in profile_paths:
                            profile_path.unlink(missing_ok=True)
                    continue

            null = np.empty((n_sims, len(expected_positions)), dtype=np.float32)
            pair_counts = np.empty(n_sims, dtype=np.int64)
            started = time.monotonic()
            with ThreadPoolExecutor(max_workers=decode_workers) as executor:
                futures = {}
                for simulation, (path, profile_path) in enumerate(
                    zip(paths, profile_paths, strict=True)
                ):
                    unit = validate_completed_unit(
                        path,
                        population=population,
                        simulation=simulation,
                        chromosome=chromosome,
                    )
                    futures[
                        executor.submit(
                            decode_profile,
                            input_path=path,
                            profile_path=profile_path,
                            simulation=simulation,
                            chromosome=chromosome,
                            source_signature=unit.signature,
                            aou=aou,
                            executable=executable,
                            callable_mask=callable_masks[chromosome],
                            contract=contract,
                            expected_positions=expected_positions,
                        )
                    ] = simulation
                done = reused = 0
                for future in as_completed(futures):
                    simulation = futures[future]
                    values, n_pairs, was_reused, _seconds = future.result()
                    null[simulation] = values
                    pair_counts[simulation] = n_pairs
                    done += 1
                    reused += int(was_reused)
                    if done <= 4 or done % progress_every == 0 or done == n_sims:
                        print(
                            f"[gamma-cutoffs] {population} {chromosome}: "
                            f"profiles={done:,}/{n_sims:,} reused={reused:,} "
                            f"elapsed={(time.monotonic() - started) / 60:.1f} min",
                            flush=True,
                        )
            if not np.all(pair_counts == diploid_samples):
                raise ValueError(
                    "Gamma-SMC n_pairs does not equal simulated diploid samples "
                    f"({diploid_samples})"
                )
            cutoffs, max_exceedances, ranks = monte_carlo_cutoffs(null, p_values)
            ends = np.minimum(expected_positions + int(contract["output_at_stride"]), length)
            temporary_name = f"__tmp_{chromosome}_{os.getpid()}"
            with h5py.File(output, "a") as handle:
                handle.attrs["complete"] = False
                if temporary_name in handle:
                    del handle[temporary_name]
                group = handle.create_group(temporary_name)
                group.attrs.update(
                    {
                        "complete": False,
                        "chromosome": chromosome,
                        "input_tree_sequence_vcf_contig": "1",
                        "sequence_length": length,
                        "n_pairs": diploid_samples,
                        "n_simulations": n_sims,
                        "source_manifest_sha256": source_digest,
                        "decoder_contract_json": json.dumps(contract, sort_keys=True),
                        "callable_mask": str(callable_masks[chromosome]),
                        "callable_mask_sha256": sha256_file(callable_masks[chromosome]),
                    }
                )
                options = {
                    "compression": "gzip",
                    "compression_opts": 6,
                    "shuffle": True,
                    "fletcher32": True,
                }
                group.create_dataset("position_0based", data=expected_positions, **options)
                group.create_dataset("end", data=ends, **options)
                for name, values in (
                    ("cutoff", cutoffs),
                    ("null_mean", null.mean(axis=0)),
                    ("null_sd", null.std(axis=0, ddof=1)),
                    ("null_min", null.min(axis=0)),
                    ("null_max", null.max(axis=0)),
                ):
                    group.create_dataset(name, data=values, dtype=np.float32, **options)
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
            if not keep_profiles:
                for profile_path in profile_paths:
                    profile_path.unlink(missing_ok=True)
                    failed = profile_path.with_suffix(profile_path.suffix + ".failed.json")
                    failed.unlink(missing_ok=True)
                for directory in sorted(profile_root.glob("sim_*"), reverse=True):
                    try:
                        directory.rmdir()
                    except OSError:
                        pass
            print(
                f"[gamma-cutoffs] {population} {chromosome}: "
                f"{len(expected_positions):,} positions -> {output}",
                flush=True,
            )
    return output


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--sim-dir", type=Path, default=DEFAULT_SIM_DIR)
    result.add_argument("--output-dir", type=Path, default=None)
    result.add_argument("--pops", default=",".join(DEFAULT_POPS))
    result.add_argument("--chroms", default="1-22")
    result.add_argument("--n-sims", type=int, default=1_000)
    result.add_argument("--p-values", default=",".join(map(str, DEFAULT_P_VALUES)))
    result.add_argument("--gamma-smc-repo", type=Path, default=DEFAULT_GAMMA_SMC_REPO)
    result.add_argument("--gamma-smc-aou", type=Path, default=None)
    result.add_argument("--gamma-smc-executable", type=Path, default=None)
    result.add_argument("--hardmask", type=Path, default=None)
    result.add_argument("--theta", type=float, default=DEFAULT_THETA)
    result.add_argument("--rho-over-theta", type=float, default=DEFAULT_RHO_OVER_THETA)
    result.add_argument("--mutation-rate", type=float, default=DEFAULT_MUTATION_RATE)
    result.add_argument("--threshold-years", type=float, default=DEFAULT_THRESHOLD_YEARS)
    result.add_argument("--generation-time", type=float, default=DEFAULT_GENERATION_TIME)
    result.add_argument("--stride", type=int, default=DEFAULT_WINDOW_SIZE)
    result.add_argument("--cache-size", type=int, default=DEFAULT_CACHE_SIZE)
    result.add_argument("--pair-block", type=int, default=DEFAULT_PAIR_BLOCK)
    result.add_argument("--decode-workers", type=int, default=4)
    result.add_argument("--decode-threads", type=int, default=1)
    result.add_argument("--progress-every", type=int, default=25)
    result.add_argument("--keep-profiles", action="store_true")
    result.add_argument("--fresh", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    positive = {
        "n_sims": args.n_sims,
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
    if invalid:
        raise SystemExit(f"invalid nonpositive arguments: {invalid}")
    try:
        chromosomes = parse_chroms(args.chroms)
        p_values = parse_p_values(args.p_values)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    populations = [part.strip().upper() for part in args.pops.split(",") if part.strip()]
    unknown = [population for population in populations if population not in DEFAULT_POPS]
    if unknown or not populations or len(populations) != len(set(populations)):
        raise SystemExit(f"invalid or duplicate populations: {unknown or populations}")

    sim_dir = args.sim_dir.expanduser().resolve()
    repository = args.gamma_smc_repo.expanduser().resolve()
    aou = (
        args.gamma_smc_aou.expanduser().resolve()
        if args.gamma_smc_aou is not None
        else repository / "scripts" / "aou.sh"
    )
    executable = (
        args.gamma_smc_executable.expanduser().resolve()
        if args.gamma_smc_executable is not None
        else repository / "bin" / "gamma_smc"
    )
    if not aou.is_file() or not executable.is_file():
        raise SystemExit(
            f"Gamma-SMC launcher/binary missing: {aou}, {executable}; "
            "run gamma_smc_ts/scripts/bootstrap_uv.sh"
        )

    population_contract, _ = load_population_contract(sim_dir, populations[0])
    global_contract = population_contract["global"]
    if not isinstance(global_contract, dict):
        raise SystemExit("simulation contract has no global mapping")
    expected_hardmask_sha256 = str(global_contract.get("mask_sha256", ""))
    hardmask = args.hardmask
    no_mask_sha256 = hashlib.sha256(b"NO_MASK").hexdigest()
    if hardmask is None and expected_hardmask_sha256 != no_mask_sha256:
        if (
            DEFAULT_HARDMASK_PATH.is_file()
            and sha256_file(DEFAULT_HARDMASK_PATH) == expected_hardmask_sha256
        ):
            hardmask = DEFAULT_HARDMASK_PATH
    sequence_lengths: dict[str, int] = {}
    for chromosome in chromosomes:
        paths, _ = completed_units(sim_dir, populations[0], chromosome, args.n_sims)
        first = validate_completed_unit(
            paths[0], population=populations[0], simulation=0, chromosome=chromosome
        )
        contract = first.metadata["contract"]
        if not isinstance(contract, dict):
            raise SystemExit(f"invalid unit contract: {first.sidecar}")
        sequence_lengths[chromosome] = int(contract["sequence_length"])
    try:
        callable_masks = prepare_callable_masks(
            sim_dir=sim_dir,
            chromosomes=chromosomes,
            sequence_lengths=sequence_lengths,
            hardmask=hardmask,
            expected_hardmask_sha256=expected_hardmask_sha256,
        )
        decode_contracts = {
            chromosome: decoder_contract(
                aou=aou,
                executable=executable,
                mask_sha256=sha256_file(callable_masks[chromosome]),
                theta=args.theta,
                rho_over_theta=args.rho_over_theta,
                mutation_rate=args.mutation_rate,
                threshold_years=args.threshold_years,
                generation_time=args.generation_time,
                stride=args.stride,
                cache_size=args.cache_size,
                threads=args.decode_threads,
                pair_block=args.pair_block,
            )
            for chromosome in chromosomes
        }
        print(
            "[gamma-cutoffs] empirical decode contract "
            f"theta={args.theta:g} rho/theta={args.rho_over_theta:g} "
            f"mu={args.mutation_rate:g} threshold={args.threshold_years:g} years "
            f"stride={args.stride:,} only_within=true statistic={STATISTIC_COLUMN}",
            flush=True,
        )
        for population in populations:
            output = (
                args.output_dir.expanduser().resolve()
                / f"{population.lower()}.gamma_smc_cutoffs.10kb.h5"
                if args.output_dir is not None
                else sim_dir / population.lower() / "gamma_smc_cutoffs.10kb.h5"
            )
            write_population_cutoffs(
                sim_dir=sim_dir,
                output=output,
                population=population,
                chromosomes=chromosomes,
                n_sims=args.n_sims,
                p_values=p_values,
                aou=aou,
                executable=executable,
                callable_masks=callable_masks,
                decode_contracts=decode_contracts,
                decode_workers=args.decode_workers,
                progress_every=args.progress_every,
                keep_profiles=args.keep_profiles,
                fresh=args.fresh,
            )
    except Exception as error:
        raise SystemExit(f"Gamma-SMC cutoff generation failed: {error}") from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
