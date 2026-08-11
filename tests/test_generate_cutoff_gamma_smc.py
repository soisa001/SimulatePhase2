from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import msprime
import numpy as np
import pytest
import tszip

import generate_cutoff_gamma_smc
import run_sim


def test_gamma_callable_mask_is_complement_of_simulation_hardmask(tmp_path: Path) -> None:
    hardmask = tmp_path / "hardmask.bed"
    hardmask.write_text("chr1\t100\t200\nchr1\t400\t500\n", encoding="utf-8")
    result = generate_cutoff_gamma_smc.prepare_callable_masks(
        sim_dir=tmp_path / "sims",
        chromosomes=["chr1"],
        sequence_lengths={"chr1": 600},
        hardmask=hardmask,
        expected_hardmask_sha256=run_sim.sha256_file(hardmask),
    )
    assert result["chr1"].read_text(encoding="utf-8") == ("1\t0\t100\n1\t200\t400\n1\t500\t600\n")


def write_unit(root: Path, simulation: int) -> None:
    output = root / "afr" / f"sim_{simulation:05d}" / "chr1.tsz"
    output.parent.mkdir(parents=True, exist_ok=True)
    ts = msprime.sim_ancestry(
        samples=4,
        ploidy=2,
        population_size=1_000 + simulation,
        sequence_length=25_000,
        recombination_rate=1e-8,
        random_seed=simulation + 1,
        record_provenance=False,
    )
    tszip.compress(ts, str(output))
    metadata = {
        "status": "complete",
        "signature": hashlib.sha256(f"unit-{simulation}".encode()).hexdigest(),
        "size_bytes": output.stat().st_size,
        "target_sites": ts.num_sites,
        "realized_sites": ts.num_sites,
        "contract": {
            "population": "AFR",
            "simulation": simulation,
            "chromosome": "chr1",
            "sequence_length": 25_000,
            "diploid_samples": 4,
        },
    }
    output.with_suffix(".tsz.json").write_text(json.dumps(metadata), encoding="utf-8")


def fake_gamma(repository: Path) -> tuple[Path, Path]:
    aou = repository / "scripts" / "aou.py"
    binary = repository / "bin" / "gamma_smc"
    aou.parent.mkdir(parents=True)
    binary.parent.mkdir(parents=True)
    aou.write_text(
        """#!/usr/bin/env python3
import json, os, pathlib, sys
args = sys.argv[1:]
assert args[0] == 'decode'
def value(flag): return args[args.index(flag) + 1]
output = pathlib.Path(value('--output'))
source = pathlib.Path(value('--input'))
simulation = int(source.parent.name.split('_')[1])
n_pairs = int(value('--n-random-pairs')) if '--n-random-pairs' in args else 4
output.parent.mkdir(parents=True, exist_ok=True)
with output.open('w', encoding='utf-8') as handle:
    handle.write('position_0based\\tn_pairs\\tmean_p_tmrca_lt_threshold\\n')
    for index, position in enumerate((0, 10000, 20000)):
        handle.write(f'{position}\\t{n_pairs}\\t{0.01 * simulation + 0.001 * index}\\n')
if '--n-random-pairs' in args:
    pair_manifest = pathlib.Path(str(output) + '.pairs.tsv')
    pair_manifest.write_text(
        '# gamma_smc_pair_manifest_v1\\n'
        f'# n_pairs\\t{n_pairs}\\n'
        f'# pairs_seed\\t{value("--pairs-seed")}\\n'
        '# pairs_digest\\t0xtestdigest\\n'
        '# hap_i\\thap_j\\n'
        + ''.join(f'{index}\\t{index + 1}\\n' for index in range(n_pairs)),
        encoding='utf-8',
    )
output.with_suffix(output.suffix + '.run.json').write_text(
    json.dumps({
        'command': sys.argv,
        'decode_seconds': 0.01,
        'n_pairs_recorded': n_pairs,
    }), encoding='utf-8')
counter = os.environ.get('FAKE_GAMMA_COUNTER')
if counter:
    pathlib.Path(counter).mkdir(parents=True, exist_ok=True)
    (pathlib.Path(counter) / f'{simulation}.done').write_text('1\\n', encoding='utf-8')
""",
        encoding="utf-8",
    )
    binary.write_bytes(b"fake gamma binary")
    return aou, binary


def test_gamma_smc_cutoffs_decode_tsz_and_restart_from_compact_hdf5(
    tmp_path: Path, monkeypatch
) -> None:
    sim_dir = tmp_path / "sims"
    no_mask_sha = hashlib.sha256(b"NO_MASK").hexdigest()
    contract = {
        "schema": run_sim.SIM_ROOT_CONTRACT_SCHEMA,
        "global": {"base_seed": 42, "mask_sha256": no_mask_sha},
        "populations": {"AFR": {"diploid_samples": 4, "demography_key": "toy"}},
    }
    sim_dir.mkdir()
    (sim_dir / run_sim.SIM_ROOT_CONTRACT_NAME).write_text(json.dumps(contract), encoding="utf-8")
    for simulation in range(20):
        write_unit(sim_dir, simulation)
    repository = tmp_path / "gamma_smc_ts"
    aou, binary = fake_gamma(repository)
    counter = tmp_path / "decode_counter"
    monkeypatch.setenv("FAKE_GAMMA_COUNTER", str(counter))
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
        "--gamma-smc-repo",
        str(repository),
        "--gamma-smc-aou",
        str(aou),
        "--gamma-smc-executable",
        str(binary),
        "--decode-workers",
        "2",
        "--progress-every",
        "20",
    ]
    output = output_dir / "afr.gamma_smc_cutoffs.10kb.h5"
    output.parent.mkdir(parents=True)
    with h5py.File(output, "w") as handle:
        handle.attrs.update(
            {
                "schema": generate_cutoff_gamma_smc.GAMMA_CUTOFF_SCHEMA,
                "complete": False,
                "decoder_contract_key": "failed-run-contract",
            }
        )
        handle.create_dataset("p_value", data=[0.1])
    assert generate_cutoff_gamma_smc.main(args) == 0
    assert len(list(counter.glob("*.done"))) == 20
    assert generate_cutoff_gamma_smc.main(args) == 0
    assert len(list(counter.glob("*.done"))) == 20

    with h5py.File(output, "r") as handle:
        assert handle.attrs["schema"] == generate_cutoff_gamma_smc.GAMMA_CUTOFF_SCHEMA
        assert handle.attrs["source_kind"] == "gamma_smc_posterior"
        assert handle.attrs["pair_selection"] == "within_individual_homolog_pair"
        decoder = json.loads(handle.attrs["decoder_contract_json"])
        assert decoder["input_format"] == "tsz"
        assert decoder["recent_call"] == "mean"
        assert decoder["scaled_mutation_rate_theta"] == 0.00075
        group = handle["chr1"]
        assert bool(group.attrs["complete"])
        assert group.attrs["n_pairs"] == 4
        assert group.attrs["decoded_profiles"] == 20
        assert group.attrs["reused_profiles"] == 0
        assert group.attrs["decode_group_wall_seconds"] >= 0
        assert group["cutoff"].shape == (1, 3)
        assert group["position_0based"][:].tolist() == [0, 10_000, 20_000]
    assert not list((sim_dir / "afr" / ".gamma_smc_profiles").rglob("*.npz"))

    binary.write_bytes(b"changed gamma binary")
    with pytest.raises(SystemExit, match="incompatible cutoff output"):
        generate_cutoff_gamma_smc.main(args)


def test_random_pair_cutoffs_are_seeded_counted_and_isolated(tmp_path: Path, monkeypatch) -> None:
    sim_dir = tmp_path / "sims"
    no_mask_sha = hashlib.sha256(b"NO_MASK").hexdigest()
    contract = {
        "schema": run_sim.SIM_ROOT_CONTRACT_SCHEMA,
        "global": {"base_seed": 42, "mask_sha256": no_mask_sha},
        "populations": {"AFR": {"diploid_samples": 4, "demography_key": "toy"}},
    }
    sim_dir.mkdir()
    (sim_dir / run_sim.SIM_ROOT_CONTRACT_NAME).write_text(json.dumps(contract), encoding="utf-8")
    for simulation in range(10):
        write_unit(sim_dir, simulation)
    repository = tmp_path / "gamma_smc_ts"
    aou, binary = fake_gamma(repository)
    counter = tmp_path / "decode_counter"
    monkeypatch.setenv("FAKE_GAMMA_COUNTER", str(counter))
    output_dir = tmp_path / "sanity" / "cutoffs"
    profile_dir = tmp_path / "sanity" / "profiles"
    args = [
        "--sim-dir",
        str(sim_dir),
        "--output-dir",
        str(output_dir),
        "--profile-dir",
        str(profile_dir),
        "--keep-profiles",
        "--pops",
        "AFR",
        "--chroms",
        "1",
        "--n-sims",
        "10",
        "--p-values",
        "0.1",
        "--n-random-pairs",
        "6",
        "--pairs-seed",
        "42",
        "--gamma-smc-repo",
        str(repository),
        "--gamma-smc-aou",
        str(aou),
        "--gamma-smc-executable",
        str(binary),
        "--decode-workers",
        "2",
    ]
    assert generate_cutoff_gamma_smc.main(args) == 0
    assert len(list(counter.glob("*.done"))) == 10

    output = output_dir / "afr.gamma_smc_cutoffs.10kb.h5"
    with h5py.File(output, "r") as handle:
        assert handle.attrs["pair_selection"] == "random_haplotype_pairs"
        decoder = json.loads(handle.attrs["decoder_contract_json"])
        assert decoder["n_random_pairs"] == 6
        assert decoder["pairs_seed"] == 42
        assert decoder["exclude_within"] is False
        group = handle["chr1"]
        assert group.attrs["n_pairs"] == 6
        assert group["max_null_exceedances"][:].tolist() == [0]
        assert group["rank_from_largest"][:].tolist() == [1]
        np.testing.assert_array_equal(group["cutoff"][:], group["null_max"][:][None, :])

    profiles = sorted(profile_dir.rglob("*.npz"))
    assert len(profiles) == 10
    with np.load(profiles[0], allow_pickle=False) as profile:
        metadata = json.loads(str(profile["metadata_json"].item()))
    assert metadata["n_pairs"] == 6
    assert metadata["pair_manifest"]["schema"] == "gamma_smc_pair_manifest_v1"
    assert metadata["pair_manifest"]["header"]["pairs_seed"] == "42"
    assert "--n-random-pairs" in metadata["command"]
    assert not list(profile_dir.rglob("*.pairs.tsv"))

    assert generate_cutoff_gamma_smc.main(args) == 0
    assert len(list(counter.glob("*.done"))) == 10


def test_random_pair_count_rejects_an_impossible_panel() -> None:
    contract = {
        "pair_selection": "random_haplotype_pairs",
        "n_random_pairs": 7,
        "pairs_seed": 42,
        "exclude_within": True,
    }
    with pytest.raises(ValueError, match="only 4 are available"):
        generate_cutoff_gamma_smc.expected_pair_count(contract, diploid_samples=2)
