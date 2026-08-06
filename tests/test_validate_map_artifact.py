from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

import validate_map_artifact
from phase2_map import DEFAULT_POPS, SCHEMA, watterson_a_n


def write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    mask = tmp_path / "mask.bed"
    mask.write_text("chr1\t10\t20\n", encoding="utf-8")
    mask_sha = validate_map_artifact.sha256_file(mask)
    mask.with_suffix(".bed.sha256").write_text(
        f"{mask_sha}  {mask.name}\n", encoding="utf-8"
    )

    path = tmp_path / "map.h5"
    with h5py.File(path, "w") as handle:
        handle.attrs.update(
            {
                "schema": SCHEMA,
                "complete": True,
                "window_size": 10_000,
                "hardmask_sha256": mask_sha,
            }
        )
        handle.create_dataset("populations", data=np.asarray(DEFAULT_POPS, dtype="S3"))
        handle.create_dataset("chromosomes", data=np.asarray(["chr1"], dtype="S4"))
        handle.create_dataset("sample_counts", data=np.full(len(DEFAULT_POPS), 2))
        handle.create_dataset(
            "watterson_a_n",
            data=np.full(len(DEFAULT_POPS), watterson_a_n(2)),
        )
        group = handle.create_group("chr1")
        group.attrs["length_bp"] = 25_000
        group.attrs["n_windows"] = 3
        group.create_dataset(
            "S",
            data=np.ones((len(DEFAULT_POPS), 3), dtype=np.uint16),
            chunks=(1, 3),
            compression="gzip",
            fletcher32=True,
        )
    map_sha = validate_map_artifact.sha256_file(path)
    path.with_suffix(".h5.sha256").write_text(
        f"{map_sha}  {path.name}\n", encoding="utf-8"
    )
    path.with_suffix(".h5.json").write_text(
        json.dumps({"sha256": map_sha}), encoding="utf-8"
    )
    return path, mask


def test_validate_reads_every_population_row_and_sidecar(tmp_path: Path) -> None:
    path, mask = write_fixture(tmp_path)
    report = validate_map_artifact.validate(
        path,
        mask,
        chromosomes=["chr1"],
        expected_pops=DEFAULT_POPS,
        expected_window_size=10_000,
    )
    assert report["valid"] is True
    assert report["rows_checked"] == len(DEFAULT_POPS)
    assert report["checks"]["mask_geometry_matches_every_row"] is True


def test_validate_rejects_corrupt_checksum_sidecar(tmp_path: Path) -> None:
    path, mask = write_fixture(tmp_path)
    path.with_suffix(".h5.sha256").write_text(
        f"{'0' * 64}  {path.name}\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="sidecar mismatch"):
        validate_map_artifact.validate(
            path,
            mask,
            chromosomes=["chr1"],
            expected_pops=DEFAULT_POPS,
            expected_window_size=10_000,
        )
