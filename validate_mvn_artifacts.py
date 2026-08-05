#!/usr/bin/env python3
"""Validate the six PHLASH low-rank log-Ne MVNs and their JSON sidecars."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np

from phase2_map import DEFAULT_POPS
from run_sim import sha256_file

SCHEMA = "phlash.aou.log-ne-mvn/v1"
REPORT_SCHEMA = "simulatephase2.mvn-validation/v1"
EXPECTED_TIMES = np.geomspace(100.0, 40_000.0, 10_000).astype(np.float32)
EXPECTED_ARRAYS = {
    "schema",
    "population",
    "time",
    "mean_log_ne",
    "covariance_factor",
    "bootstrap_ne",
    "jitter",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


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


def _source_fit_indices(sources: object, population: str) -> list[int]:
    _require(isinstance(sources, list) and len(sources) == 100, "expected 100 sources")
    indices: list[int] = []
    paths: set[str] = set()
    for source in sources:
        _require(isinstance(source, dict), "source entry is not a mapping")
        path = str(source.get("path", "")).replace("\\", "/")
        _require(path not in paths, f"duplicate source path: {path}")
        paths.add(path)
        name = PurePosixPath(path).name
        _require(
            name.startswith("fit") and name.endswith(".pkl") and name[3:6].isdigit(),
            f"invalid source fit name: {name}",
        )
        _require(
            f"/{population}/" in path,
            f"source path does not identify population {population}: {path}",
        )
        _require(int(source.get("size_bytes", 0)) > 0, f"invalid source size: {path}")
        _require(int(source.get("mtime_ns", 0)) > 0, f"invalid source mtime: {path}")
        indices.append(int(name[3:6]))
    _require(sorted(indices) == list(range(100)), "sources are not exactly fit000--fit099")
    return indices


def validate_population(directory: Path, population: str, draw_seed: int) -> dict[str, object]:
    artifact = directory / f"{population}.npz"
    sidecar = directory / f"{population}.json"
    _require(artifact.is_file(), f"missing artifact: {artifact}")
    _require(sidecar.is_file(), f"missing sidecar: {sidecar}")

    with zipfile.ZipFile(artifact) as archive:
        bad_member = archive.testzip()
    _require(bad_member is None, f"NPZ CRC failure in {bad_member}: {artifact}")

    artifact_sha256 = sha256_file(artifact)
    artifact_size = artifact.stat().st_size
    try:
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception as error:
        raise ValueError(f"cannot parse sidecar {sidecar}: {error}") from error
    _require(isinstance(metadata, dict), f"sidecar is not a mapping: {sidecar}")

    with np.load(artifact, allow_pickle=False) as data:
        _require(set(data.files) == EXPECTED_ARRAYS, f"unexpected NPZ arrays: {data.files}")
        schema = str(data["schema"])
        embedded_population = str(data["population"])
        times = np.asarray(data["time"])
        mean = np.asarray(data["mean_log_ne"])
        factor = np.asarray(data["covariance_factor"])
        bootstrap = np.asarray(data["bootstrap_ne"])
        jitter = float(data["jitter"])

    _require(schema == SCHEMA, f"wrong schema for {population}: {schema}")
    _require(embedded_population == population, f"wrong embedded population: {population}")
    _require(times.dtype == np.float32, f"time is not float32: {population}")
    _require(mean.dtype == np.float32, f"mean_log_ne is not float32: {population}")
    _require(factor.dtype == np.float32, f"covariance_factor is not float32: {population}")
    _require(bootstrap.dtype == np.float32, f"bootstrap_ne is not float32: {population}")
    _require(np.array_equal(times, EXPECTED_TIMES), f"time grid differs: {population}")
    _require(mean.shape == (10_000,), f"wrong mean shape: {population}")
    _require(factor.shape == (100, 10_000), f"wrong factor shape: {population}")
    _require(bootstrap.shape == (100, 10_000), f"wrong bootstrap shape: {population}")
    _require(np.isfinite(mean).all(), f"nonfinite mean: {population}")
    _require(np.isfinite(factor).all(), f"nonfinite factor: {population}")
    _require(np.isfinite(bootstrap).all(), f"nonfinite bootstrap curves: {population}")
    _require(np.all(bootstrap > 0.0), f"nonpositive bootstrap Ne: {population}")
    _require(np.isfinite(jitter) and jitter == 0.0, f"unexpected jitter: {population}")

    log_bootstrap = np.log(bootstrap.astype(np.float64))
    reconstructed_mean = log_bootstrap.mean(axis=0)
    reconstructed_factor = (log_bootstrap - reconstructed_mean) / np.sqrt(99.0)
    mean_error = float(np.max(np.abs(mean - reconstructed_mean)))
    factor_error = float(np.max(np.abs(factor - reconstructed_factor)))
    factor_center_error = float(np.max(np.abs(factor.sum(axis=0, dtype=np.float64))))
    _require(mean_error <= 1e-6, f"mean reconstruction failed: {population}: {mean_error}")
    _require(factor_error <= 1e-6, f"factor reconstruction failed: {population}: {factor_error}")
    _require(
        factor_center_error <= 1e-5,
        f"factor is not centered across fits: {population}: {factor_center_error}",
    )

    _require(metadata.get("schema") == schema, f"sidecar schema mismatch: {population}")
    _require(metadata.get("population") == population, f"sidecar population mismatch: {population}")
    _require(int(metadata.get("num_fits", -1)) == 100, f"sidecar fit count: {population}")
    _require(
        int(metadata.get("num_time_points", -1)) == 10_000,
        f"sidecar time count: {population}",
    )
    _require(
        float(metadata.get("min_time_generations", np.nan)) == 100.0,
        f"sidecar minimum time: {population}",
    )
    _require(
        float(metadata.get("max_time_generations", np.nan)) == 40_000.0,
        f"sidecar maximum time: {population}",
    )
    _require(
        int(metadata.get("covariance_rank_at_most", -1)) == 99,
        f"sidecar covariance rank: {population}",
    )
    _require(
        float(metadata.get("jitter_log_ne_sd", np.nan)) == jitter,
        f"sidecar jitter mismatch: {population}",
    )
    _require(
        PurePosixPath(str(metadata.get("artifact", "")).replace("\\", "/")).name
        == artifact.name,
        f"sidecar artifact name mismatch: {population}",
    )
    _require(
        int(metadata.get("artifact_size_bytes", -1)) == artifact_size,
        f"sidecar artifact size mismatch: {population}",
    )
    _require(
        metadata.get("artifact_sha256") == artifact_sha256,
        f"sidecar artifact SHA256 mismatch: {population}",
    )
    _source_fit_indices(metadata.get("sources"), population)

    rng = np.random.default_rng(draw_seed)
    latent = rng.standard_normal((32, factor.shape[0]))
    draws = np.exp(mean.astype(np.float64) + latent @ factor.astype(np.float64))
    _require(np.isfinite(draws).all() and np.all(draws > 0.0), f"invalid draws: {population}")

    median_curve = np.median(bootstrap, axis=0)
    return {
        "complete": True,
        "artifact": artifact.name,
        "sidecar": sidecar.name,
        "artifact_sha256": artifact_sha256,
        "artifact_size_bytes": artifact_size,
        "schema": schema,
        "num_source_fits": 100,
        "num_time_points": 10_000,
        "time_min_generations": float(times[0]),
        "time_max_generations": float(times[-1]),
        "covariance_factor_shape": list(factor.shape),
        "covariance_rank_at_most": 99,
        "jitter_log_ne_sd": jitter,
        "bootstrap_ne_min": float(bootstrap.min()),
        "bootstrap_ne_max": float(bootstrap.max()),
        "bootstrap_median_curve_min": float(median_curve.min()),
        "bootstrap_median_curve_max": float(median_curve.max()),
        "mean_log_ne_reconstruction_max_abs_error": mean_error,
        "covariance_factor_reconstruction_max_abs_error": factor_error,
        "covariance_factor_centering_max_abs_error": factor_center_error,
        "seeded_draw_check": {
            "seed": draw_seed,
            "n_draws": 32,
            "ne_min": float(draws.min()),
            "ne_max": float(draws.max()),
            "all_finite_positive": True,
        },
        "npz_crc_check": "passed",
        "json_npz_identity_check": "passed",
    }


def validate_directory(directory: Path) -> dict[str, object]:
    directory = directory.expanduser().resolve()
    results = {
        population: validate_population(directory, population, 43 + index)
        for index, population in enumerate(DEFAULT_POPS)
    }
    reference_time_sha = hashlib.sha256(EXPECTED_TIMES.tobytes()).hexdigest()
    total_size = sum(int(value["artifact_size_bytes"]) for value in results.values())
    return {
        "schema": REPORT_SCHEMA,
        "complete": True,
        "validator": "validate_mvn_artifacts.py",
        "validation_contract": {
            "populations": list(DEFAULT_POPS),
            "artifact_schema": SCHEMA,
            "num_source_fits_per_population": 100,
            "num_time_points": 10_000,
            "time_min_generations": 100.0,
            "time_max_generations": 40_000.0,
            "time_grid": "numpy.geomspace float32",
            "time_grid_sha256": reference_time_sha,
            "mean_space": "natural_log_diploid_ne",
            "covariance": "factor.T @ factor; unbiased sample covariance of log Ne",
            "expected_covariance_rank_at_most": 99,
            "jitter_log_ne_sd": 0.0,
            "seeded_draws_per_population": 32,
            "seed_schedule": "43 + DEFAULT_POPS index (production base_seed=42 mapping)",
        },
        "checks": [
            "NPZ ZIP CRC",
            "required array names, dtypes, shapes, and finite values",
            "exact shared 100--40000 generation geomspace grid",
            "positive bootstrap Ne curves",
            "mean and covariance-factor reconstruction from bootstrap curves",
            "JSON schema, population, dimensions, jitter, size, and SHA256",
            "exact unique fit000--fit099 source manifest",
            "fixed-seed finite positive demographic draws",
        ],
        "total_artifact_size_bytes": total_size,
        "populations": results,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--mvn-dir", type=Path, default=Path("mvn"))
    result.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Default: <mvn-dir>/validation_report.json",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    directory = args.mvn_dir.expanduser().resolve()
    report_path = (
        args.report.expanduser().resolve()
        if args.report is not None
        else directory / "validation_report.json"
    )
    try:
        report = validate_directory(directory)
    except Exception as error:
        raise SystemExit(f"MVN validation failed: {error}") from error
    _atomic_json(report_path, report)
    for population, result in report["populations"].items():
        print(
            f"[mvn] {population}: {result['num_source_fits']} fits x "
            f"{result['num_time_points']:,} times; "
            f"Ne={result['bootstrap_ne_min']:,.1f}--{result['bootstrap_ne_max']:,.1f}; "
            f"sha256={result['artifact_sha256']}",
            flush=True,
        )
    print(f"[mvn] validation complete: {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
