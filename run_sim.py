#!/usr/bin/env python3
"""Run map-calibrated msprime simulations with exact 10 kb SNV targets.

``S`` in the input HDF5 is an empirical count of segregating SNV sites, not an
msprime mutation probability.  For each chromosome this runner creates
candidate mutations at 5e-8, removes masked/recurrent/non-segregating sites,
thins to the per-window targets, and fills only deficient windows with repeated
1e-7 candidate draws.  An output is published only after exact validation.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import functools
import gc
import hashlib
import inspect
import json
import os
import shutil
import subprocess
import tempfile
import time
import traceback
import uuid
from collections.abc import Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

# Avoid BLAS oversubscription before numpy is imported in either parent or workers.
for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import h5py
import msprime
import numpy as np
import tskit
import tszip

from phase2_map import (
    DEFAULT_POPS,
    SCHEMA,
    ChromosomeTarget,
    canonical_chrom,
    clip_merged_mask,
    load_mask,
    map_sample_counts,
    map_watterson_a_n,
    parse_chroms,
    watterson_a_n,
    window_geometry,
)
from simulation_outputs import quick_tsz_archive

DEFAULT_RECOMBINATION_RATE = 1e-8
DEFAULT_INITIAL_RATE = 5e-8
DEFAULT_RETRY_RATE = 1e-7
DEFAULT_SIM_DIR = Path("/scratch.global/soisa001/sims")
DEFAULT_DEMOGRAPHY_CACHE = DEFAULT_SIM_DIR / "demographies"
DEFAULT_DEMOGRAPHY_EPOCHS = 10_000
ALGORITHM_VERSION = "simulatephase2.calibrated-snv/v5-runtime-mask"
DETERMINISTIC_PROVENANCE_TIMESTAMP = "1970-01-01T00:00:00Z"
SIM_ROOT_CONTRACT_SCHEMA = "simulatephase2.sim-root-contract/v2"
SIM_ROOT_CONTRACT_NAME = "simulation_contract.json"
TARGET_POLICY = "raw-S-to-target-by-watterson-a_n-round-half-up/v1"
SEED_MODULUS = 4_294_967_291  # largest prime below 2**32
SEED_CHANNELS = 256


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class MapSnapshot:
    """An immutable local, content-addressed copy of the input HDF5 map."""

    path: Path
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class MaskSnapshot:
    """Verified local copy of the mask named by the map contract."""

    path: Path | None
    source: str
    sha256: str


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


@contextmanager
def output_lock(path: Path):
    """Cross-process advisory lock; held for skip-check through publication."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    try:
        if os.name == "nt":
            import msvcrt

            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns


def snapshot_map_h5(source: Path, directory: Path, *, chunk_size: int = 8 << 20) -> MapSnapshot:
    """Copy and hash ``source`` in one pass, then atomically publish by SHA256.

    Every caller first makes a private temporary copy. Atomic create-once
    publication means concurrent launchers either publish the same bytes once
    or reuse the already verified immutable snapshot.
    """
    source = source.expanduser().resolve()
    directory = directory.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if chunk_size <= 0:
        raise ValueError("snapshot chunk size must be positive")
    directory.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".map-snapshot.", dir=directory)
    temporary = Path(temporary_name)
    digest = hashlib.sha256()
    copied = 0
    try:
        with os.fdopen(descriptor, "wb") as output_handle, source.open("rb") as input_handle:
            before = os.fstat(input_handle.fileno())
            while chunk := input_handle.read(chunk_size):
                digest.update(chunk)
                output_handle.write(chunk)
                copied += len(chunk)
            after_open = os.fstat(input_handle.fileno())
            output_handle.flush()
            os.fsync(output_handle.fileno())
        after_path = source.stat()
        if (
            copied != before.st_size
            or _file_identity(before) != _file_identity(after_open)
            or _file_identity(before) != _file_identity(after_path)
        ):
            raise RuntimeError(f"map changed while it was being snapshotted: {source}")

        hexdigest = digest.hexdigest()
        snapshot = directory / f"snv_theta_map.{hexdigest}.h5"
        try:
            # Same-directory hard-link publication is atomic and never
            # overwrites an existing content-addressed snapshot.
            os.link(temporary, snapshot)
        except FileExistsError:
            pass
        except OSError:
            # Some local filesystems disable hard links. Retain the same
            # create-once behavior under a cross-process advisory lock.
            lock = directory / f".{hexdigest}.lock"
            with output_lock(lock):
                if not snapshot.exists():
                    os.replace(temporary, snapshot)
        if snapshot.stat().st_size != copied or sha256_file(snapshot) != hexdigest:
            raise RuntimeError(f"corrupt content-addressed map snapshot: {snapshot}")
        return MapSnapshot(path=snapshot, sha256=hexdigest, size_bytes=copied)
    finally:
        temporary.unlink(missing_ok=True)


def snapshot_mask(
    source: str,
    directory: Path,
    *,
    expected_sha256: str,
    gcloud: str = "gcloud",
    billing_project: str | None = None,
) -> MaskSnapshot:
    """Localize a mask once and reject content that differs from the map contract."""
    source = str(source).strip()
    expected_sha256 = str(expected_sha256).strip().lower()
    if source == "NONE":
        observed = hashlib.sha256(b"NO_MASK").hexdigest()
        if expected_sha256 != observed:
            raise ValueError("unmasked map has an invalid hardmask SHA256")
        return MaskSnapshot(path=None, source=source, sha256=observed)
    if len(expected_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in expected_sha256
    ):
        raise ValueError("map does not contain a valid hardmask SHA256")
    directory = directory.expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    suffix = ".bed.gz" if source.lower().endswith(".gz") else ".bed"
    destination = directory / f"hardmask.{expected_sha256}{suffix}"
    local_source: Path | None = None
    if not source.startswith("gs://"):
        local_source = Path(source).expanduser().resolve()
        if not local_source.is_file():
            raise FileNotFoundError(local_source)
        if local_source == destination:
            observed = sha256_file(local_source)
            if observed != expected_sha256:
                raise ValueError(
                    f"mask SHA256 differs from the map contract: {observed} != {expected_sha256}"
                )
            return MaskSnapshot(path=destination, source=source, sha256=observed)
    else:
        if shutil.which(gcloud) is None:
            raise FileNotFoundError(f"gcloud executable not found: {gcloud}")
    lock = directory / f".{destination.name}.lock"
    with output_lock(lock):
        if destination.is_file() and sha256_file(destination) == expected_sha256:
            return MaskSnapshot(destination, source, expected_sha256)
        destination.unlink(missing_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=directory)
        temporary = Path(temporary_name)
        try:
            if local_source is None:
                os.close(descriptor)
                command = [gcloud]
                if billing_project:
                    command.extend(["--billing-project", billing_project])
                command.extend(["storage", "cp", "--quiet", source, str(temporary)])
                subprocess.run(command, check=True)
                observed = sha256_file(temporary)
            else:
                digest = hashlib.sha256()
                with (
                    os.fdopen(descriptor, "wb") as output_handle,
                    local_source.open("rb") as input_handle,
                ):
                    before = os.fstat(input_handle.fileno())
                    while chunk := input_handle.read(8 << 20):
                        digest.update(chunk)
                        output_handle.write(chunk)
                    after_open = os.fstat(input_handle.fileno())
                    output_handle.flush()
                    os.fsync(output_handle.fileno())
                after_path = local_source.stat()
                if _file_identity(before) != _file_identity(after_open) or _file_identity(
                    before
                ) != _file_identity(after_path):
                    raise RuntimeError(
                        f"mask changed while it was being snapshotted: {local_source}"
                    )
                observed = digest.hexdigest()
            if observed != expected_sha256:
                raise ValueError(
                    f"mask SHA256 differs from the map contract: {observed} != {expected_sha256}"
                )
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    return MaskSnapshot(destination, source, expected_sha256)


def simulation_software_versions() -> dict[str, str]:
    return {
        "msprime": msprime.__version__,
        "numpy": np.__version__,
        "tskit": tskit.__version__,
        "tszip": tszip.__version__,
    }


class SimRootContractMismatch(RuntimeError):
    pass


def requested_sim_root_contract(config: dict[str, object]) -> dict[str, object]:
    demography_cache = config.get("demography_cache")
    sample_counts = config.get("sample_counts")
    map_sample_counts = config.get("map_sample_counts")
    map_a_n = config.get("map_watterson_a_n")
    simulation_a_n = config.get("simulation_watterson_a_n")
    if not all(
        isinstance(value, dict)
        for value in (demography_cache, sample_counts, map_sample_counts, map_a_n, simulation_a_n)
    ):
        raise ValueError("simulation config lacks demography or sample-size mappings")
    populations: dict[str, dict[str, object]] = {}
    for raw_population, raw_count in sample_counts.items():
        population = str(raw_population).upper()
        demography = demography_cache.get(population)
        if not isinstance(demography, dict) or "key" not in demography:
            raise ValueError(f"simulation config lacks a demography key for {population}")
        populations[population] = {
            "demography_key": str(demography["key"]),
            "diploid_samples": int(raw_count),
            "map_diploid_samples": int(map_sample_counts[population]),
            "map_watterson_a_n": float(map_a_n[population]),
            "simulation_watterson_a_n": float(simulation_a_n[population]),
            "S_scale": float(simulation_a_n[population]) / float(map_a_n[population]),
        }
    if not populations:
        raise ValueError("simulation config selects no populations")
    return {
        "schema": SIM_ROOT_CONTRACT_SCHEMA,
        "global": {
            "algorithm": ALGORITHM_VERSION,
            "software_versions": simulation_software_versions(),
            "map_sha256": str(config["map_sha256"]),
            "mask_sha256": str(config["mask_sha256"]),
            "target_policy": TARGET_POLICY,
            "recombination_rate": float(config["recombination_rate"]),
            "initial_rate": float(config["initial_rate"]),
            "retry_rate": float(config["retry_rate"]),
            "max_retries": int(config["max_retries"]),
            "base_seed": int(config["base_seed"]),
        },
        "populations": populations,
    }


def _contract_differences(left: dict[str, object], right: dict[str, object]) -> list[str]:
    return sorted(key for key in left.keys() | right.keys() if left.get(key) != right.get(key))


def ensure_sim_root_contract(config: dict[str, object]) -> Path:
    """Create or compatibly extend the locked contract for one simulation root."""
    sim_dir = Path(str(config["sim_dir"])).expanduser().resolve()
    sim_dir.mkdir(parents=True, exist_ok=True)
    manifest = sim_dir / SIM_ROOT_CONTRACT_NAME
    lock = sim_dir / f".{SIM_ROOT_CONTRACT_NAME}.lock"
    requested = requested_sim_root_contract(config)
    with output_lock(lock):
        if not manifest.exists():
            existing_output = next(sim_dir.rglob("*.tsz"), None)
            if existing_output is not None:
                raise SimRootContractMismatch(
                    f"simulation output exists without {SIM_ROOT_CONTRACT_NAME}: "
                    f"{existing_output}; use a new --sim-dir"
                )
            atomic_text(manifest, json.dumps(requested, indent=2, sort_keys=True) + "\n")
            return manifest
        try:
            existing = json.loads(manifest.read_text(encoding="utf-8"))
        except Exception as error:
            raise SimRootContractMismatch(f"cannot read simulation contract: {manifest}") from error
        if not isinstance(existing, dict) or existing.get("schema") != SIM_ROOT_CONTRACT_SCHEMA:
            raise SimRootContractMismatch(f"invalid simulation contract schema: {manifest}")
        existing_global = existing.get("global")
        requested_global = requested["global"]
        if not isinstance(existing_global, dict) or existing_global != requested_global:
            fields = (
                _contract_differences(existing_global, requested_global)
                if isinstance(existing_global, dict) and isinstance(requested_global, dict)
                else ["global"]
            )
            raise SimRootContractMismatch(
                f"simulation root global contract mismatch in {fields}: {manifest}"
            )
        existing_populations = existing.get("populations")
        requested_populations = requested["populations"]
        if not isinstance(existing_populations, dict) or not isinstance(
            requested_populations, dict
        ):
            raise SimRootContractMismatch(f"invalid population contract mapping: {manifest}")
        merged = dict(existing_populations)
        for population, entry in requested_populations.items():
            if population in merged and merged[population] != entry:
                raise SimRootContractMismatch(
                    f"simulation root population contract mismatch for {population}: {manifest}"
                )
            merged[population] = entry
        if merged != existing_populations:
            existing["populations"] = merged
            atomic_text(manifest, json.dumps(existing, indent=2, sort_keys=True) + "\n")
    return manifest


def stable_seed(
    base: int,
    population: str,
    simulation: int,
    chromosome: str,
    stream: int,
    attempt: int = 0,
) -> int:
    """Collision-free uint32 seed for every supported production random stream."""
    population = population.upper()
    if population not in DEFAULT_POPS or simulation < 0:
        raise ValueError("invalid population or negative simulation index")
    chrom = canonical_chrom(chromosome)
    try:
        chrom_index = int(chrom[3:]) - 1
    except ValueError as error:
        raise ValueError(f"invalid autosome: {chromosome}") from error
    if not 0 <= chrom_index < 22:
        raise ValueError(f"invalid autosome: {chromosome}")
    if attempt == 0 and stream in (1, 2, 3):
        channel = stream - 1
    elif stream == 4 and 1 <= attempt <= SEED_CHANNELS - 3:
        channel = attempt + 2
    else:
        raise ValueError(f"unsupported random stream/attempt: {stream}/{attempt}")
    unit = ((simulation * len(DEFAULT_POPS) + DEFAULT_POPS.index(population)) * 22) + chrom_index
    raw = unit * SEED_CHANNELS + channel
    if raw >= SEED_MODULUS:
        raise ValueError("simulation index is too large for collision-free uint32 seeds")
    # Multiplication by any nonzero number is a permutation modulo this prime.
    return int((2_654_435_761 * raw + int(base) % SEED_MODULUS) % SEED_MODULUS) + 1


def population_seed(base: int, population: str) -> int:
    population = population.upper()
    if population not in DEFAULT_POPS:
        raise ValueError(f"unknown population: {population}")
    return int((int(base) + DEFAULT_POPS.index(population)) % SEED_MODULUS) + 1


def mvn_path(directory: Path, population: str) -> Path:
    candidates = (
        directory / f"{population.upper()}.npz",
        directory / f"{population.lower()}.npz",
        directory / f"{population.lower()}_mvn.npz",
        directory / f"{population.upper()}_mvn.npz",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"no MVN artifact for {population} under {directory}")


def coarsen_indices(times: np.ndarray, epochs: int) -> np.ndarray:
    """Choose exactly ``epochs`` time-spaced points, including both endpoints."""
    times = np.asarray(times, dtype=float)
    count = min(len(times), max(2, epochs))
    if count == len(times):
        return np.arange(len(times), dtype=np.int64)
    targets = (
        np.geomspace(times[0], times[-1], count)
        if times[0] > 0
        else np.linspace(times[0], times[-1], count)
    )
    insertion = np.searchsorted(times, targets)
    result = np.empty(count, dtype=np.int64)
    previous = -1
    for index, (target, right) in enumerate(zip(targets, insertion, strict=True)):
        right = int(np.clip(right, 0, len(times) - 1))
        left = max(0, right - 1)
        candidate = left if abs(times[left] - target) <= abs(times[right] - target) else right
        lower = previous + 1
        upper = len(times) - (count - index)
        candidate = int(np.clip(candidate, lower, upper))
        result[index] = candidate
        previous = candidate
    result[0], result[-1] = 0, len(times) - 1
    if len(np.unique(result)) != count:
        raise RuntimeError("could not select the requested number of demographic epochs")
    return result


def load_mvn_draws(
    path: Path,
    *,
    population: str,
    n_sims: int,
    epochs: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Support current low-rank phlash MVNs and the repository's legacy dense MVNs."""
    with np.load(path, allow_pickle=False) as data:
        files = set(data.files)
        rng = np.random.default_rng(population_seed(seed, population))
        if {"time", "mean_log_ne", "covariance_factor"} <= files:
            times = np.asarray(data["time"], dtype=float)
            mean = np.asarray(data["mean_log_ne"], dtype=float)
            factor = np.asarray(data["covariance_factor"], dtype=float)
            if mean.ndim != 1 or not np.isfinite(mean).all():
                raise ValueError(f"invalid mean_log_ne in {path}")
            if (
                factor.ndim != 2
                or factor.shape[1] != len(mean)
                or not np.isfinite(factor).all()
            ):
                raise ValueError(f"invalid covariance factor in {path}")
            latent = rng.standard_normal((n_sims, factor.shape[0]))
            log_ne = mean + latent @ factor
            jitter = float(data["jitter"]) if "jitter" in files else 0.0
            if not np.isfinite(jitter) or jitter < 0.0:
                raise ValueError(f"invalid MVN jitter in {path}")
            if jitter:
                log_ne += rng.normal(0.0, jitter, size=log_ne.shape)
            source_schema = str(data["schema"]) if "schema" in files else "low-rank"
            rank = factor.shape[0]
        elif {"mean", "cov"} <= files:
            mean = np.asarray(data["mean"], dtype=float)
            covariance = np.asarray(data["cov"], dtype=float)
            if covariance.shape != (len(mean), len(mean)):
                raise ValueError(f"invalid dense covariance in {path}")
            times = np.geomspace(100.0, 40_000.0, len(mean))
            covariance = (covariance + covariance.T) / 2.0
            diagonal_scale = max(float(np.diag(covariance).max(initial=0.0)), 1.0)
            try:
                # Much faster than an eigendecomposition for the legacy 1000 x
                # 1000 files. Tiny jitter only repairs PSD-boundary roundoff.
                lower = np.linalg.cholesky(covariance + np.eye(len(mean)) * diagonal_scale * 1e-10)
                factor = lower.T
                factor_method = "cholesky-with-1e-10-relative-jitter"
            except np.linalg.LinAlgError:
                values, vectors = np.linalg.eigh(covariance)
                tolerance = max(float(values.max(initial=0.0)), 1.0) * 1e-10
                keep = values > tolerance
                factor = (vectors[:, keep] * np.sqrt(values[keep])).T
                factor_method = "positive-eigenspace"
            latent = rng.standard_normal((n_sims, factor.shape[0]))
            log_ne = mean + latent @ factor
            source_schema = f"legacy-dense-log-ne-mvn/{factor_method}"
            rank = factor.shape[0]
        else:
            raise ValueError(f"unrecognized MVN schema in {path}: {sorted(files)}")
    if (
        times.ndim != 1
        or len(times) != log_ne.shape[1]
        or not np.isfinite(times).all()
        or times[0] <= 0.0
        or np.any(np.diff(times) <= 0)
    ):
        raise ValueError(f"invalid MVN time grid in {path}")
    indices = coarsen_indices(times, epochs)
    ne = np.exp(log_ne[:, indices])
    if not np.isfinite(ne).all() or np.any(ne <= 0):
        raise ValueError(f"MVN produced invalid Ne values: {path}")
    metadata = {
        "source": str(path.resolve()),
        "source_sha256": sha256_file(path),
        "source_schema": source_schema,
        "covariance_rank": int(rank),
        "epochs": int(len(indices)),
    }
    return times[indices], ne, metadata


def prepare_demography_cache(
    directory: Path,
    mvn_directory: Path,
    populations: Iterable[str],
    *,
    n_sims: int,
    epochs: int,
    seed: int,
) -> dict[str, dict[str, str]]:
    directory.mkdir(parents=True, exist_ok=True)
    result: dict[str, dict[str, str]] = {}
    for population in populations:
        source = mvn_path(mvn_directory, population)
        source_digest = sha256_file(source)
        key = hashlib.sha256(
            json.dumps(
                {
                    "algorithm": ALGORITHM_VERSION,
                    "numpy_version": np.__version__,
                    "population": population,
                    "source_sha256": source_digest,
                    "n_sims": n_sims,
                    "epochs": epochs,
                    "seed": seed,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        output = directory / f"{population.upper()}.{key[:20]}.npz"
        lock = output.with_suffix(output.suffix + ".lock")
        with output_lock(lock):
            valid = False
            if output.is_file():
                try:
                    with np.load(output, allow_pickle=False) as cached:
                        valid = (
                            str(cached["cache_key"]) == key
                            and str(cached["population"]) == population.upper()
                            and cached["ne"].shape[0] == n_sims
                            and cached["ne"].ndim == 2
                            and cached["times"].shape == (cached["ne"].shape[1],)
                        )
                except Exception:
                    valid = False
            if not valid:
                print(f"[demography] drawing {n_sims:,} {population} histories", flush=True)
                times, ne, metadata = load_mvn_draws(
                    source,
                    population=population,
                    n_sims=n_sims,
                    epochs=epochs,
                    seed=seed,
                )
                temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
                try:
                    with temporary.open("wb") as handle:
                        np.savez_compressed(
                            handle,
                            cache_key=np.asarray(key),
                            population=np.asarray(population.upper()),
                            times=times.astype(np.float64),
                            ne=ne.astype(np.float32),
                            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
                        )
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.replace(temporary, output)
                finally:
                    temporary.unlink(missing_ok=True)
        result[population.upper()] = {"path": str(output.resolve()), "key": key}
        print(f"[demography] {population}: {output}", flush=True)
    return result


@functools.lru_cache(maxsize=16)
def cached_demographies(path: str, expected_key: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        if str(data["cache_key"]) != expected_key:
            raise ValueError(f"demography cache key mismatch: {path}")
        times = np.asarray(data["times"], dtype=float)
        ne = np.asarray(data["ne"], dtype=float)
    if (
        times.ndim != 1
        or ne.ndim != 2
        or ne.shape[1] != len(times)
        or not np.isfinite(times).all()
        or not np.isfinite(ne).all()
        or np.any(np.diff(times) <= 0.0)
        or np.any(ne <= 0.0)
    ):
        raise ValueError(f"invalid demography cache contents: {path}")
    return times, ne


def make_demography(population: str, times: np.ndarray, ne: np.ndarray) -> msprime.Demography:
    name = population.lower()
    demography = msprime.Demography()
    demography.add_population(name=name, initial_size=float(ne[0]))
    for time_point, size in zip(times[1:], ne[1:], strict=True):
        demography.add_population_parameters_change(
            time=float(time_point), initial_size=float(size), population=name
        )
    return demography


@functools.lru_cache(maxsize=4)
def cached_mask_by_chrom(path: str) -> dict[str, np.ndarray]:
    """Load and merge the external mask once in each simulation process."""
    chromosomes = tuple(f"chr{index}" for index in range(1, 23))
    return load_mask(Path(path), chromosomes) if path else load_mask(None, chromosomes)


@functools.lru_cache(maxsize=32)
def cached_map_chrom(path: str, mask_path: str, chromosome: str) -> dict[str, object]:
    """Read one compact S matrix and reconstruct geometry from the verified mask."""
    chromosome = canonical_chrom(chromosome)
    with h5py.File(path, "r") as handle:
        if str(handle.attrs.get("schema", "")) != SCHEMA:
            raise ValueError(f"{path} is not a current {SCHEMA} map")
        if not bool(handle.attrs.get("complete", False)):
            raise ValueError(f"map is not marked complete: {path}")
        window_size = int(handle.attrs["window_size"])
        populations = [
            value.decode() if isinstance(value, bytes) else str(value)
            for value in handle["populations"][...]
        ]
        group = handle[chromosome]
        length_bp = int(group.attrs["length_bp"])
        n_windows = int(group.attrs["n_windows"])
        matrix = np.asarray(group["S"], dtype=np.int64)
        if matrix.shape != (len(populations), n_windows):
            raise ValueError(f"invalid S matrix shape for {chromosome}: {matrix.shape}")
        raw_s = {population.upper(): matrix[index] for index, population in enumerate(populations)}
        mask = clip_merged_mask(cached_mask_by_chrom(mask_path)[chromosome], length_bp)
        starts, ends, callable_bp = window_geometry(length_bp, window_size, mask)
        if len(starts) != n_windows:
            raise ValueError(f"window count disagrees with chromosome length in {chromosome}")
        result: dict[str, object] = {
            "window_size": window_size,
            "length_bp": length_bp,
            "starts": starts,
            "ends": ends,
            "callable_bp": callable_bp,
            "mask": mask,
            "raw_s": raw_s,
        }
    if not raw_s:
        raise ValueError(f"no population targets in {chromosome}")
    first_population = next(iter(raw_s))
    ChromosomeTarget(
        chromosome=chromosome,
        population=first_population,
        length_bp=length_bp,
        window_size=window_size,
        starts=starts,
        ends=ends,
        callable_bp=callable_bp,
        theta=raw_s[first_population],
        mask_intervals=mask,
    ).validate()
    for population, values in raw_s.items():
        if values.shape != starts.shape or np.any(values < 0) or np.any(values > callable_bp):
            raise ValueError(f"invalid S vector for {population}/{chromosome}")
    return result


def runtime_target_from_raw_s(
    raw_s: np.ndarray,
    callable_bp: np.ndarray,
    *,
    map_a_n: float,
    simulation_a_n: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return target S, theta_W/callable-bp, and the a_n rescaling factor."""
    raw_s = np.asarray(raw_s, dtype=np.int64)
    callable_bp = np.asarray(callable_bp, dtype=np.int64)
    if raw_s.shape != callable_bp.shape or np.any(raw_s < 0) or np.any(raw_s > callable_bp):
        raise ValueError("raw S must be a valid count for every callable window")
    if not np.isfinite(map_a_n) or map_a_n <= 0:
        raise ValueError("map a_n must be finite and positive")
    if not np.isfinite(simulation_a_n) or simulation_a_n <= 0:
        raise ValueError("simulation a_n must be finite and positive")

    density = np.zeros(raw_s.shape, dtype=np.float64)
    available = callable_bp > 0
    if np.any(raw_s[~available] != 0):
        raise ValueError("raw S is nonzero in a fully masked window")
    density[available] = raw_s[available] / (float(map_a_n) * callable_bp[available])
    scale = float(simulation_a_n) / float(map_a_n)
    if simulation_a_n == map_a_n:
        target = raw_s.copy()
    else:
        # Round-half-up is explicit and platform-independent for nonnegative S.
        target = np.floor(raw_s.astype(np.float64) * scale + 0.5).astype(np.int64)
    if np.any(target > callable_bp):
        raise ValueError(
            "sample-size-rescaled target exceeds callable positions; use fewer simulation "
            "samples or a different target policy"
        )
    return target, density, scale


def positions_masked(positions: np.ndarray, mask: np.ndarray) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.int64)
    if len(mask) == 0 or len(positions) == 0:
        return np.zeros(len(positions), dtype=bool)
    indices = np.searchsorted(mask[:, 0], positions, side="right") - 1
    result = np.zeros(len(positions), dtype=bool)
    valid = indices >= 0
    result[valid] = positions[valid] < mask[indices[valid], 1]
    return result


def merge_runs(windows: np.ndarray, window_size: int, length_bp: int) -> list[tuple[int, int]]:
    windows = np.unique(np.asarray(windows, dtype=np.int64))
    if len(windows) == 0:
        return []
    result: list[tuple[int, int]] = []
    first = previous = int(windows[0])
    for value in map(int, windows[1:]):
        if value != previous + 1:
            result.append((first * window_size, min((previous + 1) * window_size, length_bp)))
            first = value
        previous = value
    result.append((first * window_size, min((previous + 1) * window_size, length_bp)))
    return result


def subtract_mask(intervals: Iterable[tuple[int, int]], mask: np.ndarray) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    mask_index = 0
    for start, end in intervals:
        cursor = int(start)
        while mask_index < len(mask) and mask[mask_index, 1] <= cursor:
            mask_index += 1
        index = mask_index
        while index < len(mask) and mask[index, 0] < end:
            left, right = map(int, mask[index])
            if left > cursor:
                result.append((cursor, min(left, end)))
            cursor = max(cursor, right)
            if cursor >= end:
                break
            index += 1
        if cursor < end:
            result.append((cursor, int(end)))
    return [(start, end) for start, end in result if end > start]


def mutation_rate_map(
    *,
    length_bp: int,
    window_size: int,
    active_windows: np.ndarray,
    mask: np.ndarray,
    rate: float,
) -> msprime.RateMap:
    active = merge_runs(active_windows, window_size, length_bp)
    callable_intervals = subtract_mask(active, mask)
    if not callable_intervals:
        return msprime.RateMap(position=[0.0, float(length_bp)], rate=[0.0])
    segments: list[tuple[int, int, float]] = []
    cursor = 0
    for start, end in callable_intervals:
        if start > cursor:
            segments.append((cursor, start, 0.0))
        segments.append((start, end, rate))
        cursor = end
    if cursor < length_bp:
        segments.append((cursor, length_bp, 0.0))
    merged: list[list[float]] = []
    for start, end, value in segments:
        if merged and merged[-1][2] == value and merged[-1][1] == start:
            merged[-1][1] = end
        else:
            merged.append([float(start), float(end), float(value)])
    positions = [merged[0][0]] + [item[1] for item in merged]
    rates = [item[2] for item in merged]
    return msprime.RateMap(position=positions, rate=rates)


TableView = tskit.TableCollection | tskit.ImmutableTableCollection


@dataclass(frozen=True)
class RetainedCandidateColumns:
    """Compact copies of only the candidate rows retained after thinning."""

    site_position: np.ndarray
    site_ancestral_state: np.ndarray
    site_ancestral_state_offset: np.ndarray
    site_metadata: np.ndarray
    site_metadata_offset: np.ndarray
    mutation_node: np.ndarray
    mutation_time: np.ndarray
    mutation_derived_state: np.ndarray
    mutation_derived_state_offset: np.ndarray
    mutation_parent: np.ndarray
    mutation_metadata: np.ndarray
    mutation_metadata_offset: np.ndarray
    site_metadata_schema: tskit.MetadataSchema
    mutation_metadata_schema: tskit.MetadataSchema

    @property
    def size(self) -> int:
        return len(self.site_position)


def eligible_table_site_ids(tables: TableView, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raw_positions = np.asarray(tables.sites.position)
    if np.any(raw_positions != np.floor(raw_positions)):
        raise ValueError("candidate mutations are not on integer coordinates")
    positions = raw_positions.astype(np.int64)
    if len(tables.sites) == 0:
        return np.empty(0, dtype=np.int64), positions
    mutation_sites = np.asarray(tables.mutations.site, dtype=np.int64)
    mutation_count = np.bincount(mutation_sites, minlength=len(tables.sites))
    valid = mutation_count == 1
    valid &= ~positions_masked(positions, mask)
    return np.flatnonzero(valid), positions


def eligible_site_ids(mts: tskit.TreeSequence, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compatibility/test helper using the tree sequence's zero-copy table view."""
    return eligible_table_site_ids(mts.tables, mask)


def initial_selection(
    tables: TableView,
    target: np.ndarray,
    window_size: int,
    mask: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[int, int], dict[int, set[int]], int]:
    eligible, positions = eligible_table_site_ids(tables, mask)
    eligible_windows = positions[eligible] // window_size
    keep = np.zeros(len(tables.sites), dtype=bool)
    needed: dict[int, int] = {}
    occupied: dict[int, set[int]] = {}
    for window in np.flatnonzero(target):
        left = int(np.searchsorted(eligible_windows, window, side="left"))
        right = int(np.searchsorted(eligible_windows, window, side="right"))
        candidates = eligible[left:right]
        required = int(target[window])
        if len(candidates) > required:
            candidates = np.asarray(
                rng.choice(candidates, size=required, replace=False), dtype=np.int64
            )
        keep[candidates] = True
        if len(candidates) < required:
            needed[int(window)] = required - len(candidates)
            occupied[int(window)] = set(map(int, positions[candidates]))
    return keep, needed, occupied, int(keep.sum())


def retain_candidate_columns(tables: TableView, keep_sites: np.ndarray) -> RetainedCandidateColumns:
    """Copy only retained site/mutation columns from an immutable candidate table view."""
    keep_sites = np.asarray(keep_sites, dtype=bool)
    if keep_sites.shape != (len(tables.sites),):
        raise ValueError("site-retention mask has the wrong shape")
    mutation_sites = np.asarray(tables.mutations.site, dtype=np.int64)
    keep_mutations = keep_sites[mutation_sites]
    retained_sites = int(keep_sites.sum())
    if int(keep_mutations.sum()) != retained_sites:
        raise ValueError("retained candidate sites must each have exactly one mutation")
    retained_mutation_sites = mutation_sites[keep_mutations]
    if not np.array_equal(retained_mutation_sites, np.flatnonzero(keep_sites)):
        raise ValueError("retained candidate mutations are not ordered one per site")

    ancestral_state, ancestral_state_offset = tskit.keep_with_offset(
        keep_sites,
        np.asarray(tables.sites.ancestral_state),
        np.asarray(tables.sites.ancestral_state_offset),
    )
    site_metadata, site_metadata_offset = tskit.keep_with_offset(
        keep_sites,
        np.asarray(tables.sites.metadata),
        np.asarray(tables.sites.metadata_offset),
    )
    derived_state, derived_state_offset = tskit.keep_with_offset(
        keep_mutations,
        np.asarray(tables.mutations.derived_state),
        np.asarray(tables.mutations.derived_state_offset),
    )
    mutation_metadata, mutation_metadata_offset = tskit.keep_with_offset(
        keep_mutations,
        np.asarray(tables.mutations.metadata),
        np.asarray(tables.mutations.metadata_offset),
    )
    return RetainedCandidateColumns(
        site_position=np.asarray(tables.sites.position)[keep_sites].copy(),
        site_ancestral_state=np.asarray(ancestral_state),
        site_ancestral_state_offset=np.asarray(ancestral_state_offset),
        site_metadata=np.asarray(site_metadata),
        site_metadata_offset=np.asarray(site_metadata_offset),
        mutation_node=np.asarray(tables.mutations.node)[keep_mutations].copy(),
        mutation_time=np.asarray(tables.mutations.time)[keep_mutations].copy(),
        mutation_derived_state=np.asarray(derived_state),
        mutation_derived_state_offset=np.asarray(derived_state_offset),
        mutation_parent=np.asarray(tables.mutations.parent)[keep_mutations].copy(),
        mutation_metadata=np.asarray(mutation_metadata),
        mutation_metadata_offset=np.asarray(mutation_metadata_offset),
        site_metadata_schema=tables.sites.metadata_schema,
        mutation_metadata_schema=tables.mutations.metadata_schema,
    )


def append_retained_columns(
    tables: tskit.TableCollection, retained: RetainedCandidateColumns
) -> None:
    """Append one compact candidate chunk to mutable ancestry tables in bulk."""
    if len(tables.sites) != len(tables.mutations):
        raise ValueError("calibrated tables must contain one mutation per site")
    first_site = len(tables.sites)
    if first_site == 0:
        tables.sites.metadata_schema = retained.site_metadata_schema
        tables.mutations.metadata_schema = retained.mutation_metadata_schema
    elif (
        tables.sites.metadata_schema != retained.site_metadata_schema
        or tables.mutations.metadata_schema != retained.mutation_metadata_schema
    ):
        raise ValueError("candidate metadata schemas differ between mutation draws")
    if np.any(retained.mutation_parent != tskit.NULL):
        raise ValueError("retained candidate mutations unexpectedly have parents")
    tables.sites.append_columns(
        position=retained.site_position,
        ancestral_state=retained.site_ancestral_state,
        ancestral_state_offset=retained.site_ancestral_state_offset,
        metadata=retained.site_metadata,
        metadata_offset=retained.site_metadata_offset,
    )
    tables.mutations.append_columns(
        site=np.arange(first_site, first_site + retained.size, dtype=np.int32),
        node=retained.mutation_node,
        time=retained.mutation_time,
        derived_state=retained.mutation_derived_state,
        derived_state_offset=retained.mutation_derived_state_offset,
        parent=retained.mutation_parent,
        metadata=retained.mutation_metadata,
        metadata_offset=retained.mutation_metadata_offset,
    )


def select_retry_candidates(
    candidate_tables: TableView,
    needed: dict[int, int],
    occupied: dict[int, set[int]],
    *,
    window_size: int,
    mask: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    """Select retry sites without materializing or mutating ancestry tables."""
    eligible, positions = eligible_table_site_ids(candidate_tables, mask)
    eligible_windows = positions[eligible] // window_size
    keep = np.zeros(len(candidate_tables.sites), dtype=bool)
    added_total = 0
    for window in list(needed):
        left = int(np.searchsorted(eligible_windows, window, side="left"))
        right = int(np.searchsorted(eligible_windows, window, side="right"))
        used = occupied.setdefault(window, set())
        candidates = [
            site_id for site_id in eligible[left:right] if int(positions[site_id]) not in used
        ]
        if not candidates:
            continue
        required = needed[window]
        if len(candidates) > required:
            candidates = list(rng.choice(np.asarray(candidates), size=required, replace=False))
        for site_id in candidates:
            position = int(positions[int(site_id)])
            keep[int(site_id)] = True
            used.add(position)
            added_total += 1
        deficit = required - len(candidates)
        if deficit:
            needed[window] = deficit
        else:
            del needed[window]
    return keep, added_total


def validate_calibrated(
    ts: tskit.TreeSequence,
    *,
    target: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    mask: np.ndarray,
    expected_haploids: int,
) -> None:
    if ts.num_samples != expected_haploids:
        raise ValueError(f"sample count {ts.num_samples} != {expected_haploids}")
    positions = np.asarray(ts.tables.sites.position)
    if np.any(positions != np.floor(positions)):
        raise ValueError("simulated sites are not on integer coordinates")
    integer_positions = positions.astype(np.int64)
    if len(integer_positions) > 1 and np.any(np.diff(integer_positions) <= 0):
        raise ValueError("site positions are duplicated or unsorted")
    if np.any(positions_masked(integer_positions, mask)):
        raise ValueError("a simulated site lies in the embedded mask")
    mutation_count = np.bincount(
        np.asarray(ts.tables.mutations.site, dtype=np.int64), minlength=ts.num_sites
    )
    if len(mutation_count) != ts.num_sites or np.any(mutation_count != 1):
        raise ValueError("every retained site must contain exactly one mutation event")
    windows = np.concatenate((starts, [ends[-1]])).astype(float)
    realized = np.asarray(ts.segregating_sites(windows=windows, mode="site", span_normalise=False))
    rounded = np.rint(realized).astype(np.int64)
    if not np.allclose(realized, rounded) or not np.array_equal(rounded, target):
        bad = np.flatnonzero(rounded != target)
        preview = [(int(index), int(target[index]), int(rounded[index])) for index in bad[:10]]
        raise ValueError(f"realized segregating sites do not equal target S: {preview}")
    if ts.num_sites != int(target.sum()):
        raise ValueError("site count differs from target-S sum despite biallelic validation")


class TargetDeficit(RuntimeError):
    pass


def calibrate_mutations(
    ancestry: tskit.TreeSequence,
    *,
    target: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    mask: np.ndarray,
    unit_contract: dict[str, object],
) -> tuple[tskit.TreeSequence, dict[str, int]]:
    window_size = int(ends[0] - starts[0])
    length_bp = int(ends[-1])
    contract_geometry = (
        int(unit_contract.get("sequence_length", -1)),
        int(unit_contract.get("window_size", -1)),
        int(unit_contract.get("target_sites", -1)),
        2 * int(unit_contract.get("diploid_samples", -1)),
    )
    observed_geometry = (length_bp, window_size, int(target.sum()), ancestry.num_samples)
    if contract_geometry != observed_geometry:
        raise ValueError(
            f"unit contract geometry/sample mismatch: {contract_geometry} != {observed_geometry}"
        )
    initial_rate = float(unit_contract["initial_rate"])
    retry_rate = float(unit_contract["retry_rate"])
    max_retries = int(unit_contract["max_retries"])
    first_seed, thinning_seed, retry_seeds = contract_random_seeds(unit_contract, max_retries)
    if int(target.sum()) == 0:
        stats = {"initial_retained": 0, "retry_attempts": 0, "retry_added": 0}
        tables = ancestry.dump_tables()
        add_unit_provenance(tables, unit_contract, **stats)
        result = tables.tree_sequence()
        validate_calibrated(
            result,
            target=target,
            starts=starts,
            ends=ends,
            mask=mask,
            expected_haploids=ancestry.num_samples,
        )
        return result, stats
    active = np.flatnonzero(target)
    first_map = mutation_rate_map(
        length_bp=length_bp,
        window_size=window_size,
        active_windows=active,
        mask=mask,
        rate=initial_rate,
    )
    mts = msprime.sim_mutations(
        ancestry,
        rate=first_map,
        model=msprime.BinaryMutationModel(),
        discrete_genome=True,
        random_seed=first_seed,
        record_provenance=False,
    )
    del first_map
    rng = np.random.default_rng(thinning_seed)
    candidate_tables = mts.tables
    keep_sites, needed, occupied, initial_retained = initial_selection(
        candidate_tables, target, window_size, mask, rng
    )
    retained_chunks = [retain_candidate_columns(candidate_tables, keep_sites)]
    del candidate_tables, keep_sites, mts
    gc.collect()
    retry_added = 0
    attempts = 0
    while needed and attempts < max_retries:
        attempts += 1
        retry_map = mutation_rate_map(
            length_bp=length_bp,
            window_size=window_size,
            active_windows=np.asarray(sorted(needed), dtype=np.int64),
            mask=mask,
            rate=retry_rate,
        )
        retry_ts = msprime.sim_mutations(
            ancestry,
            rate=retry_map,
            model=msprime.BinaryMutationModel(),
            discrete_genome=True,
            random_seed=retry_seeds[attempts - 1],
            record_provenance=False,
        )
        del retry_map
        retry_tables = retry_ts.tables
        retry_keep, added = select_retry_candidates(
            retry_tables,
            needed,
            occupied,
            window_size=window_size,
            mask=mask,
            rng=rng,
        )
        retry_added += added
        if added:
            retained_chunks.append(retain_candidate_columns(retry_tables, retry_keep))
        del retry_tables, retry_keep, retry_ts
    if needed:
        preview = ", ".join(f"w{window}:{count}" for window, count in list(needed.items())[:10])
        raise TargetDeficit(
            f"{len(needed)} windows remain short after {max_retries} retries at "
            f"{retry_rate:g}; deficits: {preview}"
        )
    del active, needed, occupied, rng
    gc.collect()
    tables = ancestry.dump_tables()
    for retained in retained_chunks:
        append_retained_columns(tables, retained)
    del retained_chunks
    tables.sort()
    tables.build_index()
    tables.compute_mutation_parents()
    stats = {
        "initial_retained": initial_retained,
        "retry_attempts": attempts,
        "retry_added": retry_added,
    }
    add_unit_provenance(tables, unit_contract, **stats)
    result = tables.tree_sequence()
    validate_calibrated(
        result,
        target=target,
        starts=starts,
        ends=ends,
        mask=mask,
        expected_haploids=ancestry.num_samples,
    )
    return result, stats


def unit_signature(
    *,
    config: dict[str, object],
    population: str,
    simulation: int,
    chromosome: str,
    sample_count: int,
    length_bp: int,
    window_size: int,
    callable_bp: int,
    source_sites: int,
    target_sites: int,
    map_sample_count: int,
    map_a_n: float,
    simulation_a_n: float,
    target_scale: float,
) -> tuple[str, dict[str, object]]:
    chromosome = canonical_chrom(chromosome)
    base_seed = int(config["base_seed"])
    max_retries = int(config["max_retries"])
    seeds = {
        "ancestry": stable_seed(base_seed, population, simulation, chromosome, 1),
        "mutation_initial": stable_seed(base_seed, population, simulation, chromosome, 2),
        "thinning": stable_seed(base_seed, population, simulation, chromosome, 3),
        "mutation_retries": [
            stable_seed(base_seed, population, simulation, chromosome, 4, attempt)
            for attempt in range(1, max_retries + 1)
        ],
    }
    contract = {
        "schema": ALGORITHM_VERSION,
        "software_versions": simulation_software_versions(),
        "map_sha256": config["map_sha256"],
        "mask_sha256": config["mask_sha256"],
        "target_policy": TARGET_POLICY,
        "demography_key": config["demography_cache"][population]["key"],
        "population": population,
        "simulation": simulation,
        "chromosome": chromosome,
        "diploid_samples": sample_count,
        "sequence_length": int(length_bp),
        "window_size": int(window_size),
        "callable_bp": int(callable_bp),
        "source_segregating_sites": int(source_sites),
        "target_sites": int(target_sites),
        "map_diploid_samples": int(map_sample_count),
        "map_watterson_a_n": float(map_a_n),
        "simulation_watterson_a_n": float(simulation_a_n),
        "S_scale": float(target_scale),
        "base_seed": base_seed,
        "recombination_rate": float(config["recombination_rate"]),
        "initial_rate": float(config["initial_rate"]),
        "retry_rate": float(config["retry_rate"]),
        "max_retries": max_retries,
        "seeds": seeds,
    }
    signature = contract_signature(contract)
    return signature, contract


def contract_signature(contract: dict[str, object]) -> str:
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def contract_random_seeds(
    contract: dict[str, object], max_retries: int
) -> tuple[int, int, list[int]]:
    if contract.get("schema") != ALGORITHM_VERSION:
        raise ValueError("unit contract does not match the simulation algorithm")
    if int(contract.get("max_retries", -1)) != max_retries:
        raise ValueError("unit contract retry count does not match calibration")
    raw = contract.get("seeds")
    if not isinstance(raw, dict):
        raise ValueError("unit contract has no random-stream seeds")
    retries = raw.get("mutation_retries")
    if not isinstance(retries, list) or len(retries) != max_retries:
        raise ValueError("unit contract has an invalid retry-seed schedule")
    return int(raw["mutation_initial"]), int(raw["thinning"]), list(map(int, retries))


def add_unit_provenance(
    tables: tskit.TableCollection,
    contract: dict[str, object],
    *,
    initial_retained: int,
    retry_attempts: int,
    retry_added: int,
) -> None:
    record = {
        "schema_version": "1.0.0",
        "software": {"name": "SimulatePhase2/run_sim.py", "version": ALGORITHM_VERSION},
        "environment": {},
        "parameters": {
            "unit_signature": contract_signature(contract),
            "unit_contract": contract,
            "outcome": {
                "initial_retained": int(initial_retained),
                "retry_attempts": int(retry_attempts),
                "retry_added": int(retry_added),
                "realized_sites": int(initial_retained + retry_added),
            },
        },
    }
    tables.provenances.add_row(
        record=json.dumps(record, sort_keys=True, separators=(",", ":")),
        timestamp=DETERMINISTIC_PROVENANCE_TIMESTAMP,
    )


def deep_validate_file(
    path: Path,
    *,
    target: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    mask: np.ndarray,
    expected_haploids: int,
) -> bool:
    try:
        ts = tszip.decompress(str(path))
        validate_calibrated(
            ts,
            target=target,
            starts=starts,
            ends=ends,
            mask=mask,
            expected_haploids=expected_haploids,
        )
        return True
    except Exception:
        return False


def existing_complete(
    output: Path,
    sidecar: Path,
    signature: str,
    *,
    verify: bool,
    target: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    mask: np.ndarray,
    expected_haploids: int,
) -> bool:
    if not output.is_file() or not sidecar.is_file():
        return False
    try:
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        quick = (
            metadata.get("signature") == signature
            and metadata.get("status") == "complete"
            and int(metadata.get("size_bytes", -1)) == output.stat().st_size
            and int(metadata.get("target_sites", -1)) == int(target.sum())
        )
    except Exception:
        return False
    quick = quick and quick_tsz_archive(output)
    return quick and (
        not verify
        or deep_validate_file(
            output,
            target=target,
            starts=starts,
            ends=ends,
            mask=mask,
            expected_haploids=expected_haploids,
        )
    )


def simulate_unit(payload: tuple[dict[str, object], str, int, str]) -> dict[str, object]:
    config, population, simulation, chromosome = payload
    started = time.monotonic()
    output = (
        Path(str(config["sim_dir"]))
        / population.lower()
        / f"sim_{simulation:05d}"
        / f"{chromosome}.tsz"
    )
    sidecar = output.with_suffix(output.suffix + ".json")
    lock = output.with_suffix(output.suffix + ".lock")
    try:
        map_data = cached_map_chrom(str(config["map_path"]), str(config["mask_path"]), chromosome)
        raw_s_by_pop = map_data["raw_s"]
        raw_s = np.asarray(raw_s_by_pop[population], dtype=np.int64)
        starts = np.asarray(map_data["starts"], dtype=np.int64)
        ends = np.asarray(map_data["ends"], dtype=np.int64)
        callable_bp = np.asarray(map_data["callable_bp"], dtype=np.int64)
        mask = np.asarray(map_data["mask"], dtype=np.int64)
        sample_count = int(config["sample_counts"][population])
        map_sample_count = int(config["map_sample_counts"][population])
        map_a_n = float(config["map_watterson_a_n"][population])
        simulation_a_n = float(config["simulation_watterson_a_n"][population])
        target, effective_density, target_scale = runtime_target_from_raw_s(
            raw_s,
            callable_bp,
            map_a_n=map_a_n,
            simulation_a_n=simulation_a_n,
        )
        signature, contract = unit_signature(
            config=config,
            population=population,
            simulation=simulation,
            chromosome=chromosome,
            sample_count=sample_count,
            length_bp=int(map_data["length_bp"]),
            window_size=int(map_data["window_size"]),
            callable_bp=int(callable_bp.sum()),
            source_sites=int(raw_s.sum()),
            target_sites=int(target.sum()),
            map_sample_count=map_sample_count,
            map_a_n=map_a_n,
            simulation_a_n=simulation_a_n,
            target_scale=target_scale,
        )
        with output_lock(lock):
            if not config["fresh"] and existing_complete(
                output,
                sidecar,
                signature,
                verify=bool(config["verify_existing"]),
                target=target,
                starts=starts,
                ends=ends,
                mask=mask,
                expected_haploids=2 * sample_count,
            ):
                return {"status": "skip", "unit": (population, simulation, chromosome)}
            times, histories = cached_demographies(
                str(config["demography_cache"][population]["path"]),
                str(config["demography_cache"][population]["key"]),
            )
            demography = make_demography(population, times, histories[simulation])
            stream_seeds = contract["seeds"]
            if not isinstance(stream_seeds, dict):
                raise ValueError("unit contract has no random-stream seeds")
            ancestry = msprime.sim_ancestry(
                samples={population.lower(): sample_count},
                demography=demography,
                sequence_length=int(map_data["length_bp"]),
                recombination_rate=float(config["recombination_rate"]),
                ploidy=2,
                discrete_genome=True,
                random_seed=int(stream_seeds["ancestry"]),
                record_provenance=False,
            )
            calibrated, stats = calibrate_mutations(
                ancestry,
                target=target,
                starts=starts,
                ends=ends,
                mask=mask,
                unit_contract=contract,
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_name(f".{output.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
            try:
                tszip.compress(calibrated, str(temporary))
                if not quick_tsz_archive(temporary):
                    raise RuntimeError(f"TSZip publication check failed: {temporary}")
                with temporary.open("r+b") as handle:
                    os.fsync(handle.fileno())
                os.replace(temporary, output)
            finally:
                temporary.unlink(missing_ok=True)
            metadata = {
                "status": "complete",
                "signature": signature,
                "contract": contract,
                "source_segregating_sites": int(raw_s.sum()),
                "target_sites": int(target.sum()),
                "realized_sites": int(calibrated.num_sites),
                "callable_bp": int(callable_bp.sum()),
                "effective_theta_w_per_callable_bp": {
                    "definition": "S / (map_watterson_a_n * callable_bp)",
                    "genomewide": (
                        float(raw_s.sum()) / (map_a_n * float(callable_bp.sum()))
                        if callable_bp.sum()
                        else 0.0
                    ),
                    "minimum_nonzero": (
                        float(effective_density[effective_density > 0].min())
                        if np.any(effective_density > 0)
                        else 0.0
                    ),
                    "maximum": float(effective_density.max(initial=0.0)),
                    "nonzero_windows": int(np.count_nonzero(effective_density)),
                    "available_windows": int(np.count_nonzero(callable_bp)),
                },
                "size_bytes": output.stat().st_size,
                "seconds": time.monotonic() - started,
                **stats,
            }
            atomic_text(sidecar, json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        del calibrated, ancestry
        gc.collect()
        return {
            "status": "ok",
            "unit": (population, simulation, chromosome),
            "sites": int(metadata["realized_sites"]),
            "seconds": float(metadata["seconds"]),
            "retry_attempts": int(metadata["retry_attempts"]),
        }
    except Exception as error:
        return {
            "status": "error",
            "unit": (population, simulation, chromosome),
            "message": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }


def interleaved_units(
    populations: list[str], simulations: range, chromosomes: list[str]
) -> Iterator[tuple[str, int, str]]:
    offsets = [
        round(index * len(chromosomes) / len(populations)) for index in range(len(populations))
    ]
    for simulation in simulations:
        for round_index in range(len(chromosomes)):
            for population, offset in zip(populations, offsets, strict=True):
                yield population, simulation, chromosomes[(round_index + offset) % len(chromosomes)]


def bounded_results(
    executor: ProcessPoolExecutor,
    config: dict[str, object],
    units: Iterable[tuple[str, int, str]],
    max_pending: int,
    heartbeat_seconds: float,
) -> Iterator[dict[str, object]]:
    iterator = iter(units)
    pending = set()
    while len(pending) < max_pending:
        try:
            population, simulation, chromosome = next(iterator)
        except StopIteration:
            break
        pending.add(executor.submit(simulate_unit, (config, population, simulation, chromosome)))
    while pending:
        completed, pending = wait(pending, timeout=heartbeat_seconds, return_when=FIRST_COMPLETED)
        if not completed:
            yield {"status": "heartbeat", "pending": len(pending)}
            continue
        for future in completed:
            yield future.result()
        while len(pending) < max_pending:
            try:
                population, simulation, chromosome = next(iterator)
            except StopIteration:
                break
            pending.add(
                executor.submit(simulate_unit, (config, population, simulation, chromosome))
            )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--map",
        "--h5",
        dest="map_path",
        type=Path,
        default=Path("data/snv_theta_map.10kb.h5"),
    )
    result.add_argument(
        "--map-snapshot-dir",
        type=Path,
        default=None,
        help="Local immutable map cache (default: <demography-cache>/map_snapshots)",
    )
    result.add_argument(
        "--mask",
        default=None,
        help=(
            "Hard-mask BED/BED.gz path or gs:// URI. Default: the source recorded in the map; "
            "the content SHA256 must match the map contract"
        ),
    )
    result.add_argument(
        "--mask-cache-dir",
        type=Path,
        default=None,
        help="Local content-addressed mask cache (default: <demography-cache>/mask_snapshots)",
    )
    result.add_argument("--gcloud", default="gcloud")
    result.add_argument("--billing-project", default=None)
    result.add_argument("--mvn-dir", type=Path, default=Path("mvn"))
    result.add_argument(
        "--demography-cache", "--demog-dir", type=Path, default=DEFAULT_DEMOGRAPHY_CACHE
    )
    result.add_argument("--sim-dir", type=Path, default=DEFAULT_SIM_DIR)
    result.add_argument("--n-sims", type=int, default=1_000)
    result.add_argument(
        "--samples-per-population",
        "--samples-per-pop",
        type=int,
        default=0,
        help=(
            "Diploid simulation samples per selected population; 0 uses each map sample count. "
            "A nonzero override rescales S by the ratio of Watterson a_n values"
        ),
    )
    result.add_argument("--pops", default=",".join(DEFAULT_POPS))
    result.add_argument("--chroms", default="1-22")
    result.add_argument("--workers", type=int, default=4)
    result.add_argument("--max-pending", type=int, default=0, help="Default: twice --workers")
    result.add_argument("--max-tasks-per-worker", type=int, default=0)
    result.add_argument("--demography-epochs", type=int, default=DEFAULT_DEMOGRAPHY_EPOCHS)
    result.add_argument("--recombination-rate", type=float, default=DEFAULT_RECOMBINATION_RATE)
    result.add_argument("--initial-rate", type=float, default=DEFAULT_INITIAL_RATE)
    result.add_argument("--retry-rate", type=float, default=DEFAULT_RETRY_RATE)
    result.add_argument("--max-retries", type=int, default=8)
    result.add_argument("--base-seed", type=int, default=42)
    result.add_argument("--verify-existing", action="store_true")
    result.add_argument("--fresh", action="store_true")
    result.add_argument("--heartbeat-seconds", type=float, default=30.0)
    result.add_argument("--progress-every", type=int, default=25)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    positive = {
        "n_sims": args.n_sims,
        "workers": args.workers,
        "demography_epochs": args.demography_epochs,
        "recombination_rate": args.recombination_rate,
        "initial_rate": args.initial_rate,
        "retry_rate": args.retry_rate,
        "heartbeat_seconds": args.heartbeat_seconds,
        "progress_every": args.progress_every,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid or args.max_retries < 0:
        raise SystemExit(f"invalid nonpositive arguments: {invalid}")
    if args.max_retries > SEED_CHANNELS - 3:
        raise SystemExit(f"--max-retries must not exceed {SEED_CHANNELS - 3}")
    if args.base_seed < 0:
        raise SystemExit("--base-seed must be nonnegative")
    if args.samples_per_population < 0:
        raise SystemExit("--samples-per-population must be nonnegative")
    map_source = args.map_path.expanduser().resolve()
    if not map_source.is_file():
        raise SystemExit(f"map does not exist: {map_source}; run generate_map.py first")
    snapshot_directory = (
        args.map_snapshot_dir.expanduser().resolve()
        if args.map_snapshot_dir is not None
        else args.demography_cache.expanduser().resolve() / "map_snapshots"
    )
    try:
        map_snapshot = snapshot_map_h5(map_source, snapshot_directory)
    except Exception as error:
        raise SystemExit(f"could not snapshot map {map_source}: {error}") from error
    map_path = map_snapshot.path
    with h5py.File(map_path, "r") as handle:
        schema = str(handle.attrs.get("schema", ""))
        complete = bool(handle.attrs.get("complete", False))
        if schema != SCHEMA:
            raise SystemExit(
                f"legacy/incompatible map schema {schema!r}; regenerate with generate_map.py"
            )
        if not complete:
            raise SystemExit("map is not marked complete")
        mask_source = str(args.mask or handle.attrs.get("hardmask_source", "")).strip()
        mask_sha256 = str(handle.attrs.get("hardmask_sha256", "")).strip()
    if not mask_source:
        raise SystemExit("map has no hardmask source; regenerate it with generate_map.py")
    mask_cache_directory = (
        args.mask_cache_dir.expanduser().resolve()
        if args.mask_cache_dir is not None
        else args.demography_cache.expanduser().resolve() / "mask_snapshots"
    )
    try:
        mask_snapshot = snapshot_mask(
            mask_source,
            mask_cache_directory,
            expected_sha256=mask_sha256,
            gcloud=args.gcloud,
            billing_project=args.billing_project,
        )
    except Exception as error:
        raise SystemExit(f"could not localize and verify mask {mask_source}: {error}") from error
    map_counts = map_sample_counts(map_path)
    map_a_n = map_watterson_a_n(map_path)
    populations = [part.strip().upper() for part in args.pops.split(",") if part.strip()]
    unknown = [pop for pop in populations if pop not in DEFAULT_POPS or pop not in map_counts]
    if unknown or len(set(populations)) != len(populations):
        raise SystemExit(f"invalid, duplicate, or absent populations: {unknown or populations}")
    chromosomes = parse_chroms(args.chroms)
    terminal_stream = (4, max(1, args.max_retries)) if args.max_retries else (3, 0)
    stable_seed(
        args.base_seed,
        max(populations, key=DEFAULT_POPS.index),
        args.n_sims - 1,
        max(chromosomes, key=lambda value: int(value[3:])),
        terminal_stream[0],
        terminal_stream[1],
    )
    simulation_counts = {pop: args.samples_per_population or map_counts[pop] for pop in populations}
    simulation_a_n = {pop: watterson_a_n(simulation_counts[pop]) for pop in populations}
    for chromosome in chromosomes:
        cached_map_chrom(
            str(map_path),
            str(mask_snapshot.path) if mask_snapshot.path is not None else "",
            chromosome,
        )

    demography_cache = prepare_demography_cache(
        args.demography_cache.expanduser().resolve(),
        args.mvn_dir.expanduser().resolve(),
        populations,
        n_sims=args.n_sims,
        epochs=args.demography_epochs,
        seed=args.base_seed,
    )
    config: dict[str, object] = {
        "map_path": str(map_path),
        "map_sha256": map_snapshot.sha256,
        "mask_path": str(mask_snapshot.path) if mask_snapshot.path is not None else "",
        "mask_source": mask_snapshot.source,
        "mask_sha256": mask_snapshot.sha256,
        "sim_dir": str(args.sim_dir.expanduser().resolve()),
        "demography_cache": demography_cache,
        "map_sample_counts": {pop: map_counts[pop] for pop in populations},
        "map_watterson_a_n": {pop: map_a_n[pop] for pop in populations},
        "sample_counts": simulation_counts,
        "simulation_watterson_a_n": simulation_a_n,
        "base_seed": args.base_seed,
        "recombination_rate": args.recombination_rate,
        "initial_rate": args.initial_rate,
        "retry_rate": args.retry_rate,
        "max_retries": args.max_retries,
        "verify_existing": args.verify_existing,
        "fresh": args.fresh,
    }
    try:
        sim_root_contract = ensure_sim_root_contract(config)
    except (OSError, ValueError, SimRootContractMismatch) as error:
        raise SystemExit(f"simulation root contract rejected: {error}") from error
    total = args.n_sims * len(populations) * len(chromosomes)
    max_pending = args.max_pending or 2 * args.workers
    if max_pending < args.workers:
        raise SystemExit("--max-pending must be zero or at least --workers")
    scale_summary = ", ".join(
        f"{pop}: {simulation_a_n[pop] / map_a_n[pop]:.6g}" for pop in populations
    )
    print(
        f"map_source={map_source}\n"
        f"map_snapshot={map_path} sha256={map_snapshot.sha256} "
        f"bytes={map_snapshot.size_bytes:,}\n"
        f"mask={mask_snapshot.source} sha256={mask_snapshot.sha256} "
        f"local={mask_snapshot.path or 'NONE'}\n"
        f"simulation_contract={sim_root_contract}\n"
        f"pops={populations} map_samples={config['map_sample_counts']} "
        f"simulation_samples={config['sample_counts']} chroms={chromosomes}\n"
        f"S_scales={{{scale_summary}}}\n"
        f"units={total:,} workers={args.workers} max_pending={max_pending} "
        f"mu={args.initial_rate:g} retry_mu={args.retry_rate:g} retries={args.max_retries}",
        flush=True,
    )

    executor_options: dict[str, object] = {"max_workers": args.workers}
    if args.max_tasks_per_worker:
        if "max_tasks_per_child" not in inspect.signature(ProcessPoolExecutor).parameters:
            raise SystemExit("this Python does not support --max-tasks-per-worker")
        executor_options["max_tasks_per_child"] = args.max_tasks_per_worker
    started = time.monotonic()
    last_heartbeat = started
    done = errors = 0
    tally: dict[str, int] = {}
    units = interleaved_units(populations, range(args.n_sims), chromosomes)
    with ProcessPoolExecutor(**executor_options) as executor:
        for result in bounded_results(executor, config, units, max_pending, args.heartbeat_seconds):
            if result["status"] == "heartbeat":
                now = time.monotonic()
                last_heartbeat = now
                print(
                    f"RUNNING {done:,}/{total:,}; {result['pending']} tasks in flight; "
                    f"elapsed={(now - started) / 60:.1f} min; {tally}",
                    flush=True,
                )
                continue
            done += 1
            status = str(result["status"])
            tally[status] = tally.get(status, 0) + 1
            if status == "error":
                errors += 1
                print(
                    f"ERROR {result['unit']}: {result['message']}\n{result['traceback']}",
                    flush=True,
                )
            elif status == "ok" and (tally["ok"] <= 10 or done % max(1, args.progress_every) == 0):
                print(
                    f"[{done:,}/{total:,}] {result['unit']} sites={result['sites']:,} "
                    f"{result['seconds']:.1f}s retries={result['retry_attempts']}",
                    flush=True,
                )
            now = time.monotonic()
            if now - last_heartbeat >= args.heartbeat_seconds:
                last_heartbeat = now
                print(
                    f"RUNNING {done:,}/{total:,}; elapsed={(now - started) / 60:.1f} min; {tally}",
                    flush=True,
                )
    elapsed = time.monotonic() - started
    print(f"DONE {done:,}/{total:,} in {elapsed / 3600:.2f} h; {tally}", flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
