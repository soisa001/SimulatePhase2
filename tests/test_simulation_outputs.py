from __future__ import annotations

import hashlib
import json
from pathlib import Path

import msprime
import numpy as np
import tszip

import check_sim_completeness
import run_sim
from simulation_outputs import quick_tsz_archive, validate_completed_unit


def write_unit(root: Path) -> tuple[Path, str]:
    path = root / "afr" / "sim_00000" / "chr1.tsz"
    path.parent.mkdir(parents=True)
    ts = msprime.sim_ancestry(
        samples=2,
        ploidy=2,
        population_size=1_000,
        sequence_length=1_000,
        random_seed=17,
        record_provenance=False,
    )
    tszip.compress(ts, str(path))
    signature = hashlib.sha256(b"unit").hexdigest()
    sidecar = {
        "status": "complete",
        "signature": signature,
        "size_bytes": path.stat().st_size,
        "target_sites": ts.num_sites,
        "realized_sites": ts.num_sites,
        "contract": {
            "population": "AFR",
            "simulation": 0,
            "chromosome": "chr1",
        },
    }
    path.with_suffix(".tsz.json").write_text(json.dumps(sidecar), encoding="utf-8")
    return path, signature


def test_quick_completion_rejects_truncated_tsz_even_if_size_sidecar_is_changed(
    tmp_path: Path,
) -> None:
    path, signature = write_unit(tmp_path)
    assert quick_tsz_archive(path)
    validate_completed_unit(path, population="AFR", simulation=0, chromosome="chr1")

    data = path.read_bytes()
    path.write_bytes(data[:-64])
    sidecar_path = path.with_suffix(".tsz.json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["size_bytes"] = path.stat().st_size
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")

    assert not quick_tsz_archive(path)
    assert not run_sim.existing_complete(
        path,
        sidecar_path,
        signature,
        verify=False,
        target=np.empty(0, dtype=np.int64),
        callable_bp=np.empty(0, dtype=np.int64),
        starts=np.empty(0, dtype=np.int64),
        ends=np.empty(0, dtype=np.int64),
        mask=np.empty((0, 2), dtype=np.int64),
        expected_haploids=4,
    )


def test_completeness_cli_writes_manifest(tmp_path: Path) -> None:
    sim_dir = tmp_path / "sims"
    write_unit(sim_dir)
    contract = {
        "schema": run_sim.SIM_ROOT_CONTRACT_SCHEMA,
        "global": {"base_seed": 42},
        "populations": {"AFR": {"diploid_samples": 2}},
    }
    (sim_dir / run_sim.SIM_ROOT_CONTRACT_NAME).write_text(
        json.dumps(contract), encoding="utf-8"
    )
    assert (
        check_sim_completeness.main(
            [
                "--sim-dir",
                str(sim_dir),
                "--pops",
                "AFR",
                "--chroms",
                "1",
                "--n-sims",
                "1",
            ]
        )
        == 0
    )
    report = json.loads(
        (sim_dir / "simulation_completeness.json").read_text(encoding="utf-8")
    )
    assert report["complete"] is True
    assert report["n_units"] == 1
    assert report["validation"] == "sidecar_size_and_tsz_zip_footer"
