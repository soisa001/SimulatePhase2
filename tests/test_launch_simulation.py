from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import launch_simulation


def _resolved(tmp_path: Path, *extra: str):
    arguments = [
        "--map",
        str(tmp_path / "map.h5"),
        "--mask",
        str(tmp_path / "mask.bed"),
        "--mvn-dir",
        str(tmp_path / "mvn"),
        "--gamma-smc-repo",
        str(tmp_path / "gamma"),
        "--sim-dir",
        str(tmp_path / "sims"),
        *extra,
    ]
    return launch_simulation.resolve(launch_simulation.parser().parse_args(arguments))


def test_repository_inputs_are_the_no_argument_defaults() -> None:
    args = launch_simulation.resolve(launch_simulation.parser().parse_args([]))
    repository = Path(launch_simulation.__file__).resolve().parent
    assert args.map == repository / "data" / "snv_theta_map.10kb.h5"
    assert args.mask == repository / "data" / "hardmask.hg38.v4.over99.bed.gz"
    assert args.mvn_dir == repository / "mvn"
    assert args.gamma_smc_repo == repository.parent / "gamma_smc_ts"


def test_test_profile_is_local_eur100_with_deep_check(tmp_path: Path) -> None:
    args = _resolved(tmp_path)
    assert args.profile == "test"
    assert args.mode == "local"
    assert args.pops == "EUR"
    assert args.n_sims == 100
    assert args.workers == 4
    assert args.decode_workers == 4
    assert args.skip_low_callable_bp == 50
    assert args.skip_low_callable_after_retries == 5
    assert args.deep_completeness_check is True
    command = launch_simulation.runner_command(args)
    assert command[command.index("--cutoff-mode") + 1] == "both"
    assert command[command.index("--chroms") + 1] == "1-22"
    assert command[command.index("--skip-low-callable-bp") + 1] == "50"
    assert command[command.index("--skip-low-callable-after-retries") + 1] == "5"
    assert "--deep-completeness-check" in command


def test_full_profile_remains_local_by_default(tmp_path: Path) -> None:
    args = _resolved(tmp_path, "--profile", "full")
    assert args.mode == "local"
    assert args.pops == "AFR,EUR,AMR,SAS,MID,EAS"
    assert args.n_sims == 1_000
    assert args.deep_completeness_check is False


def test_slurm_defaults_to_sioux_and_safe_parallelism(tmp_path: Path) -> None:
    args = _resolved(tmp_path, "--profile", "full", "--mode", "slurm")
    assert args.partition == "sioux"
    assert args.cpus == 50
    assert args.workers == 32
    assert args.decode_workers == 32
    assert args.decode_threads == 1
    assert args.mem == "384G"
    assert args.tmp == "300G"
    assert args.time == "14-00:00:00"
    command = launch_simulation.runner_command(args)
    contract = launch_simulation.launch_contract(args, command)
    key = launch_simulation.contract_key(contract)
    script = tmp_path / "job.sbatch"
    submit = launch_simulation.sbatch_command(args, script, key)
    assert "--partition=sioux" in submit
    assert "--nodes=1" in submit
    assert "--ntasks=1" in submit
    assert "--cpus-per-task=50" in submit
    assert "--mem=384G" in submit
    assert "--tmp=300G" in submit


def test_slurm_rejects_oversubscription(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        _resolved(
            tmp_path,
            "--mode",
            "slurm",
            "--decode-workers",
            "30",
            "--decode-threads",
            "2",
        )


def test_slurm_submission_is_recorded_and_active_job_is_not_duplicated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _resolved(tmp_path, "--mode", "slurm")
    command = launch_simulation.runner_command(args)
    contract = launch_simulation.launch_contract(args, command)
    calls: list[list[str]] = []

    monkeypatch.setattr(launch_simulation.shutil, "which", lambda name: f"/bin/{name}")

    def fake_run(command, **_kwargs):
        calls.append(command)
        if command[0].endswith("squeue"):
            return SimpleNamespace(stdout="RUNNING\n", stderr="", returncode=0)
        return SimpleNamespace(stdout="12345;cluster\n", stderr="", returncode=0)

    monkeypatch.setattr(launch_simulation.subprocess, "run", fake_run)
    assert launch_simulation.submit_slurm(args, command, contract) == 0
    assert launch_simulation.submit_slurm(args, command, contract) == 0
    assert sum(call[0].endswith("sbatch") for call in calls) == 1
    submission = next((args.sim_dir / "launches").glob("*.submission.json"))
    assert '"job_id": "12345"' in submission.read_text(encoding="utf-8")


@pytest.mark.parametrize("mode", ["local", "slurm"])
def test_dry_run_does_not_require_inputs_or_create_output(
    tmp_path: Path, mode: str
) -> None:
    sim_dir = tmp_path / "sims"
    result = launch_simulation.main(
        [
            "--map",
            str(tmp_path / "missing.h5"),
            "--mask",
            str(tmp_path / "missing.bed"),
            "--mvn-dir",
            str(tmp_path / "missing-mvn"),
            "--gamma-smc-repo",
            str(tmp_path / "missing-gamma"),
            "--sim-dir",
            str(sim_dir),
            "--mode",
            mode,
            "--dry-run",
        ]
    )
    assert result == 0
    assert not sim_dir.exists()
