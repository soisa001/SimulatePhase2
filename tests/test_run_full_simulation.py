from __future__ import annotations

import sys
from pathlib import Path

import run_full_simulation


def test_default_full_runner_is_three_phases_with_compact_cutoffs(tmp_path) -> None:
    args = run_full_simulation.parser().parse_args(
        [
            "--map",
            str(tmp_path / "map.h5"),
            "--mvn-dir",
            str(tmp_path / "mvn"),
            "--sim-dir",
            str(tmp_path / "sims"),
        ]
    )
    commands = run_full_simulation.phase_commands(args, sys.executable)
    names = [name for name, _ in commands]
    assert names == [
        "phase1_demography",
        "phase2_simulate",
        "phase2_check",
        "phase3_compact_cutoffs",
    ]
    assert "generate_cutoff_gamma_smc.py" not in " ".join(
        value for _, command in commands for value in command
    )


def test_gamma_cutoff_mode_runs_completeness_first(tmp_path) -> None:
    args = run_full_simulation.parser().parse_args(
        [
            "--phase",
            "cutoffs",
            "--cutoff-mode",
            "gamma-smc",
            "--mask",
            str(tmp_path / "hardmask.bed"),
        ]
    )
    commands = run_full_simulation.phase_commands(args, sys.executable)
    assert [name for name, _ in commands] == ["phase2_check", "phase3_gamma_smc_cutoffs"]
    gamma = commands[-1][1]
    assert "generate_cutoff_gamma_smc.py" in " ".join(gamma)
    assert gamma[gamma.index("--decode-workers") + 1] == "4"
    assert gamma[gamma.index("--decode-threads") + 1] == "1"


def test_repository_inputs_are_defaults() -> None:
    args = run_full_simulation.parser().parse_args([])
    repository = Path(run_full_simulation.__file__).resolve().parent
    assert args.map == repository / "data" / "snv_theta_map.10kb.h5"
    assert args.mask == repository / "data" / "hardmask.hg38.v4.over99.bed.gz"
    assert args.mvn_dir == repository / "mvn"
    assert args.gamma_smc_repo == repository.parent / "gamma_smc_ts"
