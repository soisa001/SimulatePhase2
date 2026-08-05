from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import msprime
import numpy as np
import pytest
import tszip

import generate_cutoffs
import run_sim


def test_monte_carlo_cutoffs_use_plus_one_tie_safe_rule() -> None:
    null = np.arange(1, 1001, dtype=float)[:, None]
    cutoffs, max_exceedances, ranks = generate_cutoffs.monte_carlo_cutoffs(
        null, (0.01, 0.05)
    )
    np.testing.assert_array_equal(max_exceedances, [9, 49])
    np.testing.assert_array_equal(ranks, [10, 50])
    np.testing.assert_array_equal(cutoffs[:, 0], [991.0, 951.0])
    for alpha, cutoff in zip((0.01, 0.05), cutoffs[:, 0], strict=True):
        observed = np.nextafter(float(cutoff), np.inf)
        p_value = (1 + np.count_nonzero(null[:, 0] >= observed)) / (len(null) + 1)
        assert p_value <= alpha


def test_cutoff_resolution_rejects_unattainable_p_value() -> None:
    with pytest.raises(ValueError, match="resolution"):
        generate_cutoffs.monte_carlo_cutoffs(np.ones((20, 2)), (0.01,))


def write_completed_unit(root: Path, simulation: int) -> None:
    output = root / "afr" / f"sim_{simulation:05d}" / "chr1.tsz"
    output.parent.mkdir(parents=True, exist_ok=True)
    ts = msprime.sim_ancestry(
        samples=6,
        ploidy=2,
        population_size=100 + 5 * simulation,
        sequence_length=25_000,
        recombination_rate=2e-7,
        random_seed=simulation + 1,
        record_provenance=False,
    )
    tszip.compress(ts, str(output))
    signature = hashlib.sha256(f"unit-{simulation}".encode()).hexdigest()
    metadata = {
        "status": "complete",
        "signature": signature,
        "size_bytes": output.stat().st_size,
        "target_sites": ts.num_sites,
        "realized_sites": ts.num_sites,
        "contract": {
            "population": "AFR",
            "simulation": simulation,
            "chromosome": "chr1",
        },
    }
    output.with_suffix(".tsz.json").write_text(
        json.dumps(metadata, sort_keys=True), encoding="utf-8"
    )


def test_compact_cutoff_hdf5_is_restartable(tmp_path: Path) -> None:
    sim_dir = tmp_path / "sims"
    contract = {
        "schema": run_sim.SIM_ROOT_CONTRACT_SCHEMA,
        "global": {"base_seed": 42, "map_sha256": "toy"},
        "populations": {"AFR": {"diploid_samples": 6, "demography_key": "toy"}},
    }
    sim_dir.mkdir()
    (sim_dir / run_sim.SIM_ROOT_CONTRACT_NAME).write_text(
        json.dumps(contract, sort_keys=True), encoding="utf-8"
    )
    for simulation in range(20):
        write_completed_unit(sim_dir, simulation)

    output_dir = tmp_path / "cutoffs"
    args = [
        "--sim-dir",
        str(sim_dir),
        "--output-dir",
        str(output_dir),
        "--pops",
        "AFR",
        "--chroms",
        "1",
        "--n-sims",
        "20",
        "--p-values",
        "0.1",
        "--workers",
        "1",
        "--progress-every",
        "20",
    ]
    assert generate_cutoffs.main(args) == 0
    output = output_dir / "afr.tmrca_cutoffs.10kb.h5"
    first_mtime = output.stat().st_mtime_ns
    assert generate_cutoffs.main(args) == 0
    assert output.stat().st_mtime_ns >= first_mtime

    with h5py.File(output, "r") as handle:
        assert handle.attrs["schema"] == generate_cutoffs.CUTOFF_SCHEMA
        assert handle.attrs["source_kind"] == "tree_truth"
        assert handle.attrs["significance_rule"] == "observed > cutoff"
        np.testing.assert_array_equal(handle["p_value"][:], [0.1])
        group = handle["chr1"]
        assert bool(group.attrs["complete"])
        np.testing.assert_array_equal(group["start"][:], [0, 10_000, 20_000])
        np.testing.assert_array_equal(group["end"][:], [10_000, 20_000, 25_000])
        assert group["cutoff"].shape == (1, 3)
        assert group["null_mean"].shape == (3,)
        assert np.isfinite(group["cutoff"][:]).all()
