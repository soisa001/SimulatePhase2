from __future__ import annotations

import gzip
import json
from pathlib import Path

import h5py
import numpy as np

import run_gamma_smc_sanity


def write_cutoff_fixture(output_root: Path) -> None:
    path = output_root / "cutoffs" / "afr.gamma_smc_cutoffs.10kb.h5"
    path.parent.mkdir(parents=True)
    with h5py.File(path, "w") as handle:
        handle.attrs.update(
            {
                "schema": "simulatephase2.gamma-smc-cutoffs/v1",
                "population": "AFR",
                "n_simulations": 10,
                "pair_selection": "random_haplotype_pairs",
                "statistic": "mean_p_tmrca_lt_threshold",
                "significance_rule": "observed > cutoff",
                "decoder_contract_json": json.dumps(
                    {
                        "pair_selection": "random_haplotype_pairs",
                        "n_random_pairs": 100_000,
                        "pairs_seed": 1729,
                        "exclude_within": False,
                        "scaled_mutation_rate_theta": 0.00075,
                        "recombination_to_mutation_ratio": 0.8,
                        "unscaled_mutation_rate": 1.29e-8,
                        "threshold_years": 4500,
                        "generation_time_years": 25,
                        "output_at_stride": 10_000,
                    }
                ),
            }
        )
        handle.create_dataset("p_value", data=[0.1])
        group = handle.create_group("chr1")
        group.attrs.update({"complete": True, "n_simulations": 10, "n_pairs": 100_000})
        group.create_dataset("position_0based", data=[0, 10_000, 20_000])
        group.create_dataset("end", data=[10_000, 20_000, 25_000])
        values = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)
        group.create_dataset("cutoff", data=values[None, :])
        group.create_dataset("null_max", data=values)
        group.create_dataset("max_null_exceedances", data=[0])
        group.create_dataset("rank_from_largest", data=[1])


def test_ten_simulation_p_le_point_one_cutoff_is_largest_null() -> None:
    assert run_gamma_smc_sanity._monte_carlo_rank(10, 0.1) == (0, 1)


def test_sanity_report_exports_exact_position_specific_cutoffs(tmp_path: Path) -> None:
    output_root = tmp_path / "sanity"
    write_cutoff_fixture(output_root)
    result = run_gamma_smc_sanity.write_sanity_reports(
        output_root=output_root,
        populations=["AFR"],
        chromosomes=["chr1"],
        n_sims=10,
        n_pairs=100_000,
    )

    detail = Path(result["detail_report"]["path"])
    with gzip.open(detail, "rt", encoding="utf-8") as handle:
        lines = handle.read().splitlines()
    assert len(lines) == 4
    assert lines[0].startswith("population\tchromosome\tposition_0based")
    fields = lines[-1].split("\t")
    assert fields[0:5] == ["AFR", "chr1", "20000", "25000", "0.1"]
    assert np.isclose(float(fields[5]), 0.3)

    summary = Path(result["summary_report"]["path"]).read_text(encoding="utf-8")
    assert "AFR\tchr1\t1\t3\t0.1\t10\t100000\t0\t1\t0.1" in summary
    assert "all_requested_chromosomes_descriptive" in summary
    assert result["rank_from_largest"] == 1
    assert result["max_null_exceedances"] == 0


def test_sanity_reducer_uses_separate_cutoff_and_profile_roots(tmp_path: Path) -> None:
    args = run_gamma_smc_sanity.parser().parse_args(
        [
            "--sim-dir",
            str(tmp_path / "sims"),
            "--gamma-smc-repo",
            str(tmp_path / "gamma_smc_ts"),
        ]
    )
    output_root = tmp_path / "sanity"
    command = run_gamma_smc_sanity.reducer_command(
        args=args,
        sim_dir=tmp_path / "sims",
        output_root=output_root,
        populations=["AFR"],
        chromosomes=["chr1"],
    )
    assert command[command.index("--output-dir") + 1] == str(output_root / "cutoffs")
    assert command[command.index("--profile-dir") + 1] == str(output_root / "profiles")
    assert command[command.index("--n-random-pairs") + 1] == "100000"
    assert command[command.index("--pairs-seed") + 1] == "1729"
    assert command[command.index("--reserved-cpus") + 1] == "0"
    assert "--keep-profiles" in command
