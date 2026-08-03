import json
from pathlib import Path

import plot_sim_sanity
import run_sim


def test_plotter_accepts_v4_canonical_contract_signature(tmp_path: Path) -> None:
    output = tmp_path / "sims/afr/sim_00000/chr1.tsz"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"synthetic-tsz-placeholder")
    contract = {
        "schema": run_sim.ALGORITHM_VERSION,
        "map_sha256": "map-sha",
        "population": "AFR",
        "simulation": 0,
        "chromosome": "chr1",
        "diploid_samples": 3,
        "nested": {"b": 2, "a": 1},
    }
    metadata = {
        "status": "complete",
        "signature": run_sim.contract_signature(contract),
        "contract": contract,
        "size_bytes": output.stat().st_size,
    }
    output.with_suffix(".tsz.json").write_text(json.dumps(metadata), encoding="utf-8")
    plot_sim_sanity.init_worker(str(tmp_path / "sims"), {}, {"AFR": 3}, "map-sha")
    plot_sim_sanity.validate_sidecar(
        output,
        population="AFR",
        simulation=0,
        chromosome="chr1",
        expected_diploids=3,
    )
    assert plot_sim_sanity.contract_signature(contract) == metadata["signature"]
