#!/usr/bin/env python3
"""Validate the bundled 10 kb SNV map, hardmask, checksums, and all matrix rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import h5py
import numpy as np

from phase2_map import (
    DEFAULT_HARDMASK_PATH,
    DEFAULT_MAP_PATH,
    DEFAULT_POPS,
    SCHEMA,
    iter_targets,
    load_mask,
    map_sample_counts,
    map_watterson_a_n,
    parse_chroms,
    populations,
)

REPORT_SCHEMA = "simulatephase2.snv-count-map-validation/v1"


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
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


def checksum_sidecar(path: Path, expected_name: str) -> str:
    fields = path.read_text(encoding="utf-8").strip().split()
    if len(fields) != 2 or fields[1].lstrip("*") != expected_name:
        raise ValueError(f"invalid checksum sidecar: {path}")
    digest = fields[0].lower()
    if len(digest) != 64 or any(value not in "0123456789abcdef" for value in digest):
        raise ValueError(f"invalid SHA256 in {path}")
    return digest


def validate(
    map_path: Path,
    mask_path: Path,
    *,
    chromosomes: list[str],
    expected_pops: tuple[str, ...],
    expected_window_size: int,
) -> dict[str, object]:
    map_path = map_path.expanduser().resolve()
    mask_path = mask_path.expanduser().resolve()
    if not map_path.is_file() or not mask_path.is_file():
        raise FileNotFoundError(f"map/mask missing: {map_path}, {mask_path}")

    map_sha256 = sha256_file(map_path)
    mask_sha256 = sha256_file(mask_path)
    map_checksum = checksum_sidecar(
        map_path.with_suffix(map_path.suffix + ".sha256"), map_path.name
    )
    mask_checksum = checksum_sidecar(
        mask_path.with_suffix(mask_path.suffix + ".sha256"), mask_path.name
    )
    if map_checksum != map_sha256:
        raise ValueError(f"map SHA256 sidecar mismatch: {map_checksum} != {map_sha256}")
    if mask_checksum != mask_sha256:
        raise ValueError(f"mask SHA256 sidecar mismatch: {mask_checksum} != {mask_sha256}")

    generation_sidecar = map_path.with_suffix(map_path.suffix + ".json")
    generation = json.loads(generation_sidecar.read_text(encoding="utf-8"))
    if generation.get("sha256") != map_sha256:
        raise ValueError("map JSON sidecar SHA256 does not match the HDF5")

    observed_pops = tuple(populations(map_path))
    if observed_pops != expected_pops:
        raise ValueError(f"population order differs: {observed_pops} != {expected_pops}")
    sample_counts = map_sample_counts(map_path)
    watterson = map_watterson_a_n(map_path)
    masks = load_mask(mask_path, chromosomes)

    rows_checked = 0
    chromosome_summary: dict[str, dict[str, int]] = {}
    with h5py.File(map_path, "r") as handle:
        if str(handle.attrs.get("schema", "")) != SCHEMA:
            raise ValueError(f"map schema is not {SCHEMA}")
        if not bool(handle.attrs.get("complete", False)):
            raise ValueError("map is not marked complete")
        if int(handle.attrs.get("window_size", 0)) != expected_window_size:
            raise ValueError("map window size differs from requested 10 kb")
        if str(handle.attrs.get("hardmask_sha256", "")) != mask_sha256:
            raise ValueError("map hardmask SHA256 differs from the bundled mask")
        stored_chromosomes = {
            value.decode() if isinstance(value, bytes) else str(value)
            for value in handle["chromosomes"][...]
        }
        if stored_chromosomes != set(chromosomes):
            raise ValueError("map chromosome set differs from the requested chromosomes")
        for chromosome in chromosomes:
            group = handle[chromosome]
            matrix = np.asarray(group["S"])
            expected_shape = (len(expected_pops), int(group.attrs["n_windows"]))
            if matrix.shape != expected_shape:
                raise ValueError(f"invalid S shape for {chromosome}: {matrix.shape}")
            chromosome_summary[chromosome] = {
                "length_bp": int(group.attrs["length_bp"]),
                "windows": matrix.shape[1],
                "segregating_site_total": int(matrix.sum()),
            }

    for _target in iter_targets(
        map_path,
        chromosomes,
        expected_pops,
        mask_by_chrom=masks,
    ):
        rows_checked += 1
    if rows_checked != len(chromosomes) * len(expected_pops):
        raise AssertionError("not all chromosome/population rows were checked")

    return {
        "schema": REPORT_SCHEMA,
        "valid": True,
        "map": {
            "path": str(map_path),
            "sha256": map_sha256,
            "size_bytes": map_path.stat().st_size,
            "schema": SCHEMA,
            "window_size": expected_window_size,
        },
        "hardmask": {
            "path": str(mask_path),
            "sha256": mask_sha256,
            "size_bytes": mask_path.stat().st_size,
        },
        "populations": list(expected_pops),
        "sample_counts": sample_counts,
        "watterson_a_n": watterson,
        "chromosomes": chromosome_summary,
        "rows_checked": rows_checked,
        "checks": {
            "hdf5_fletcher32_rows_read": True,
            "map_and_mask_sidecars_match": True,
            "generation_sidecar_matches": True,
            "mask_geometry_matches_every_row": True,
        },
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--map", type=Path, default=DEFAULT_MAP_PATH)
    result.add_argument("--mask", type=Path, default=DEFAULT_HARDMASK_PATH)
    result.add_argument("--chroms", default="1-22")
    result.add_argument("--pops", default=",".join(DEFAULT_POPS))
    result.add_argument("--expected-window-size", type=int, default=10_000)
    result.add_argument("--report", type=Path, default=None)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    chromosomes = parse_chroms(args.chroms)
    expected_pops = tuple(value.strip().upper() for value in args.pops.split(",") if value.strip())
    if expected_pops != DEFAULT_POPS:
        raise SystemExit(f"expected population order must be {DEFAULT_POPS}")
    if args.expected_window_size <= 0:
        raise SystemExit("--expected-window-size must be positive")
    try:
        report = validate(
            args.map,
            args.mask,
            chromosomes=chromosomes,
            expected_pops=expected_pops,
            expected_window_size=args.expected_window_size,
        )
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise SystemExit(f"map validation failed: {error}") from error
    report_path = args.report or args.map.with_suffix(args.map.suffix + ".validation.json")
    atomic_json(report_path.expanduser().resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
