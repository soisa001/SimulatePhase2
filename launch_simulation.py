#!/usr/bin/env python3
"""Launch test or full simulations locally (default) or through Slurm."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py

from phase2_map import (
    DEFAULT_GAMMA_SMC_REPO,
    DEFAULT_HARDMASK_PATH,
    DEFAULT_MAP_PATH,
    DEFAULT_MVN_DIR,
    DEFAULT_POPS,
    SCHEMA,
    parse_chroms,
)
from run_sim import DEFAULT_SIM_DIR, sha256_file

LAUNCH_SCHEMA = "simulatephase2.launch/v1"
DEFAULT_TEST_SIM_DIR = Path("/scratch.global/soisa001/sims_eur100_test")
DEFAULT_SLURM_PARTITION = "sioux"
DEFAULT_SLURM_CPUS = 50
DEFAULT_SLURM_MEMORY = "384G"
DEFAULT_SLURM_TMP = "300G"


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _atomic_json(path: Path, value: object) -> None:
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--profile", choices=("test", "full"), default="test")
    result.add_argument("--mode", choices=("local", "slurm"), default="local")
    result.add_argument(
        "--phase",
        choices=("all", "demography", "simulate", "check", "cutoffs"),
        default="all",
    )
    result.add_argument(
        "--cutoff-mode", choices=("compact", "gamma-smc", "both"), default="both"
    )
    result.add_argument(
        "--map",
        type=Path,
        default=None,
        help=f"SNV count map (default: bundled {DEFAULT_MAP_PATH.name})",
    )
    result.add_argument(
        "--mask",
        type=Path,
        default=None,
        help=f"Exclusion mask (default: bundled {DEFAULT_HARDMASK_PATH.name})",
    )
    result.add_argument("--mvn-dir", type=Path, default=None)
    result.add_argument("--gamma-smc-repo", type=Path, default=None)
    result.add_argument("--sim-dir", type=Path, default=None)
    result.add_argument("--demography-cache", type=Path, default=None)
    result.add_argument("--logs-dir", type=Path, default=None)
    result.add_argument("--pops", default=None)
    result.add_argument("--chroms", default="1-22")
    result.add_argument("--n-sims", type=int, default=None)
    result.add_argument("--demography-epochs", type=int, default=10_000)
    result.add_argument("--base-seed", type=int, default=42)
    result.add_argument("--samples-per-population", type=int, default=0)
    result.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Local simulation/compact workers (default: local=4, Slurm=min(32, CPUs))",
    )
    result.add_argument("--max-pending", type=int, default=None)
    result.add_argument(
        "--decode-workers",
        type=int,
        default=None,
        help="Concurrent Gamma-SMC subprocesses (default: local=4, Slurm=min(32, CPUs))",
    )
    result.add_argument("--decode-threads", type=int, default=1)
    result.add_argument("--p-values", default="0.01,0.05")
    result.add_argument("--window-size", type=int, default=10_000)
    result.add_argument("--threshold-years", type=float, default=4_500.0)
    result.add_argument("--generation-time", type=float, default=25.0)
    result.add_argument("--theta", type=float, default=0.00075)
    result.add_argument("--rho-over-theta", type=float, default=0.8)
    result.add_argument("--mutation-rate", type=float, default=1.29e-8)
    result.add_argument("--billing-project", default=None)
    result.add_argument(
        "--deep-completeness-check",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Default: enabled for test, quick ZIP/sidecar checks for full",
    )
    result.add_argument("--keep-gamma-profiles", action="store_true")

    slurm = result.add_argument_group("Slurm execution")
    slurm.add_argument("--partition", default=DEFAULT_SLURM_PARTITION)
    slurm.add_argument("--cpus", type=int, default=DEFAULT_SLURM_CPUS)
    slurm.add_argument("--mem", default=DEFAULT_SLURM_MEMORY)
    slurm.add_argument("--tmp", default=DEFAULT_SLURM_TMP)
    slurm.add_argument("--time", default=None)
    slurm.add_argument("--account", default=None)
    slurm.add_argument("--job-name", default=None)
    slurm.add_argument("--sbatch", default="sbatch")
    slurm.add_argument("--squeue", default="squeue")
    slurm.add_argument(
        "--slurm-extra",
        action="append",
        default=[],
        help="Repeat for extra sbatch options, e.g. --slurm-extra=--constraint=genoa",
    )
    slurm.add_argument("--force-submit", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    return result


def resolve(args: argparse.Namespace) -> argparse.Namespace:
    args.pops = args.pops or ("EUR" if args.profile == "test" else ",".join(DEFAULT_POPS))
    populations = [part.strip().upper() for part in args.pops.split(",") if part.strip()]
    unknown = sorted(set(populations) - set(DEFAULT_POPS))
    if unknown or not populations or len(populations) != len(set(populations)):
        raise ValueError(f"invalid or duplicate populations: {unknown or populations}")
    args.pops = ",".join(populations)
    parse_chroms(args.chroms)

    args.n_sims = (
        args.n_sims if args.n_sims is not None else (100 if args.profile == "test" else 1_000)
    )
    args.sim_dir = args.sim_dir or (
        DEFAULT_TEST_SIM_DIR if args.profile == "test" else DEFAULT_SIM_DIR
    )
    args.sim_dir = args.sim_dir.expanduser().resolve()
    args.demography_cache = (
        args.demography_cache.expanduser().resolve()
        if args.demography_cache is not None
        else args.sim_dir / "demographies"
    )
    args.logs_dir = (
        args.logs_dir.expanduser().resolve()
        if args.logs_dir is not None
        else args.sim_dir / "logs"
    )
    args.map = (args.map or DEFAULT_MAP_PATH).expanduser().resolve()
    args.mask = (args.mask or DEFAULT_HARDMASK_PATH).expanduser().resolve()
    args.mvn_dir = (
        args.mvn_dir.expanduser().resolve()
        if args.mvn_dir is not None
        else DEFAULT_MVN_DIR
    )
    args.gamma_smc_repo = (
        args.gamma_smc_repo.expanduser().resolve()
        if args.gamma_smc_repo is not None
        else DEFAULT_GAMMA_SMC_REPO
    )

    default_parallelism = 4 if args.mode == "local" else min(32, args.cpus)
    args.workers = args.workers if args.workers is not None else default_parallelism
    args.decode_workers = (
        args.decode_workers if args.decode_workers is not None else default_parallelism
    )
    args.max_pending = args.max_pending if args.max_pending is not None else 2 * args.workers
    args.deep_completeness_check = (
        args.profile == "test"
        if args.deep_completeness_check is None
        else args.deep_completeness_check
    )
    args.time = args.time or ("2-00:00:00" if args.profile == "test" else "14-00:00:00")
    args.job_name = args.job_name or f"phase2-{args.profile}"

    positive = {
        "n_sims": args.n_sims,
        "demography_epochs": args.demography_epochs,
        "workers": args.workers,
        "max_pending": args.max_pending,
        "decode_workers": args.decode_workers,
        "decode_threads": args.decode_threads,
        "cpus": args.cpus,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid or args.base_seed < 0 or args.samples_per_population < 0:
        raise ValueError(f"invalid nonpositive arguments or seed: {invalid}")
    if args.mode == "slurm" and args.workers > args.cpus:
        raise ValueError("--workers cannot exceed the Slurm --cpus request")
    if args.mode == "slurm" and args.decode_workers * args.decode_threads > args.cpus:
        raise ValueError("--decode-workers * --decode-threads cannot exceed Slurm --cpus")
    return args


def validate_inputs(args: argparse.Namespace) -> None:
    required = {
        "map": args.map,
        "hardmask": args.mask,
        "MVN directory": args.mvn_dir,
    }
    for label, path in required.items():
        exists = path.is_dir() if label == "MVN directory" else path.is_file()
        if not exists:
            raise FileNotFoundError(f"{label} does not exist: {path}")
    with h5py.File(args.map, "r") as handle:
        schema = str(handle.attrs.get("schema", ""))
        complete = bool(handle.attrs.get("complete", False))
        expected_mask_sha256 = str(handle.attrs.get("hardmask_sha256", ""))
    if schema != SCHEMA or not complete:
        raise ValueError(
            f"map is not a complete {SCHEMA} artifact: {args.map} "
            f"(schema={schema!r}, complete={complete})"
        )
    observed_mask_sha256 = sha256_file(args.mask)
    if observed_mask_sha256 != expected_mask_sha256:
        raise ValueError(
            "hardmask SHA256 differs from map contract: "
            f"{observed_mask_sha256} != {expected_mask_sha256}"
        )
    needs_gamma = args.phase in ("all", "cutoffs") and args.cutoff_mode in (
        "gamma-smc",
        "both",
    )
    if needs_gamma:
        required_gamma = (
            args.gamma_smc_repo / "scripts" / "aou.sh",
            args.gamma_smc_repo / "bin" / "gamma_smc",
        )
        missing = [
            str(path)
            for path in required_gamma
            if not path.is_file() or not os.access(path, os.X_OK)
        ]
        if missing:
            raise FileNotFoundError(
                "Gamma-SMC is not bootstrapped; missing " + ", ".join(missing)
            )


def runner_command(args: argparse.Namespace) -> list[str]:
    repository = Path(__file__).resolve().parent
    command = [
        sys.executable,
        "-u",
        str(repository / "run_full_simulation.py"),
        "--phase",
        args.phase,
        "--cutoff-mode",
        args.cutoff_mode,
        "--map",
        str(args.map),
        "--mask",
        str(args.mask),
        "--mvn-dir",
        str(args.mvn_dir),
        "--demography-cache",
        str(args.demography_cache),
        "--sim-dir",
        str(args.sim_dir),
        "--logs-dir",
        str(args.logs_dir),
        "--pops",
        args.pops,
        "--chroms",
        args.chroms,
        "--n-sims",
        str(args.n_sims),
        "--demography-epochs",
        str(args.demography_epochs),
        "--base-seed",
        str(args.base_seed),
        "--samples-per-population",
        str(args.samples_per_population),
        "--workers",
        str(args.workers),
        "--max-pending",
        str(args.max_pending),
        "--decode-workers",
        str(args.decode_workers),
        "--decode-threads",
        str(args.decode_threads),
        "--p-values",
        args.p_values,
        "--window-size",
        str(args.window_size),
        "--threshold-years",
        str(args.threshold_years),
        "--generation-time",
        str(args.generation_time),
        "--gamma-smc-repo",
        str(args.gamma_smc_repo),
        "--theta",
        str(args.theta),
        "--rho-over-theta",
        str(args.rho_over_theta),
        "--mutation-rate",
        str(args.mutation_rate),
    ]
    if args.billing_project:
        command.extend(["--billing-project", args.billing_project])
    if args.deep_completeness_check:
        command.append("--deep-completeness-check")
    if args.keep_gamma_profiles:
        command.append("--keep-gamma-profiles")
    return command


def launch_contract(args: argparse.Namespace, command: list[str]) -> dict[str, object]:
    return {
        "schema": LAUNCH_SCHEMA,
        "profile": args.profile,
        "mode": args.mode,
        "resolved": {
            "phase": args.phase,
            "cutoff_mode": args.cutoff_mode,
            "populations": args.pops.split(","),
            "chromosomes": parse_chroms(args.chroms),
            "n_simulations": args.n_sims,
            "demography_epochs": args.demography_epochs,
            "base_seed": args.base_seed,
            "samples_per_population": args.samples_per_population,
            "workers": args.workers,
            "max_pending": args.max_pending,
            "decode_workers": args.decode_workers,
            "decode_threads": args.decode_threads,
            "p_values": args.p_values,
            "window_size": args.window_size,
            "threshold_years": args.threshold_years,
            "generation_time": args.generation_time,
            "theta": args.theta,
            "rho_over_theta": args.rho_over_theta,
            "mutation_rate": args.mutation_rate,
            "deep_completeness_check": args.deep_completeness_check,
        },
        "paths": {
            "map": str(args.map),
            "mask": str(args.mask),
            "mvn_dir": str(args.mvn_dir),
            "gamma_smc_repo": str(args.gamma_smc_repo),
            "sim_dir": str(args.sim_dir),
            "demography_cache": str(args.demography_cache),
            "logs_dir": str(args.logs_dir),
        },
        "slurm": {
            "partition": args.partition,
            "cpus": args.cpus,
            "memory": args.mem,
            "temporary_storage": args.tmp,
            "walltime": args.time,
            "account": args.account,
            "job_name": args.job_name,
            "extra": args.slurm_extra,
        }
        if args.mode == "slurm"
        else None,
        "command": command,
    }


def contract_key(contract: dict[str, object]) -> str:
    payload = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def slurm_script(command: list[str]) -> str:
    repository = Path(__file__).resolve().parent
    return "\n".join(
        (
            "#!/bin/bash -l",
            "set -euo pipefail",
            "export OMP_NUM_THREADS=1",
            "export OPENBLAS_NUM_THREADS=1",
            "export MKL_NUM_THREADS=1",
            "export NUMEXPR_NUM_THREADS=1",
            "export MALLOC_ARENA_MAX=2",
            'export TMPDIR="${SLURM_TMPDIR:-/tmp}"',
            f"cd {shlex.quote(str(repository))}",
            f"exec {shlex.join(command)}",
            "",
        )
    )


def sbatch_command(args: argparse.Namespace, script: Path, key: str) -> list[str]:
    output = args.logs_dir / f"slurm.{args.profile}.{key[:12]}.%j.out"
    error = args.logs_dir / f"slurm.{args.profile}.{key[:12]}.%j.err"
    command = [
        args.sbatch,
        "--parsable",
        f"--partition={args.partition}",
        "--nodes=1",
        "--ntasks=1",
        f"--cpus-per-task={args.cpus}",
        f"--mem={args.mem}",
        f"--tmp={args.tmp}",
        f"--time={args.time}",
        f"--job-name={args.job_name}",
        f"--output={output}",
        f"--error={error}",
        "--export=ALL",
    ]
    if args.account:
        command.append(f"--account={args.account}")
    command.extend(args.slurm_extra)
    command.append(str(script))
    return command


def _active_job(squeue: str, job_id: str) -> str | None:
    executable = shutil.which(squeue)
    if executable is None:
        return None
    completed = subprocess.run(
        [executable, "--noheader", "--jobs", job_id, "--format=%T"],
        text=True,
        capture_output=True,
        check=False,
    )
    state = completed.stdout.strip()
    return state or None


def submit_slurm(args: argparse.Namespace, command: list[str], contract: dict[str, object]) -> int:
    launch_dir = args.sim_dir / "launches"
    key = contract_key(contract)
    contract_path = launch_dir / f"{args.profile}.{key}.json"
    script_path = launch_dir / f"{args.profile}.{key}.sbatch"
    submission_path = launch_dir / f"{args.profile}.{key}.submission.json"
    submission_command = sbatch_command(args, script_path, key)
    print(f"[launch] contract={contract_path}")
    print(f"[launch] script={script_path}")
    print(f"[launch] submit={shlex.join(submission_command)}")
    if args.dry_run:
        return 0

    args.logs_dir.mkdir(parents=True, exist_ok=True)
    launch_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(contract_path, contract)
    _atomic_text(script_path, slurm_script(command))

    if submission_path.is_file() and not args.force_submit:
        try:
            previous = json.loads(submission_path.read_text(encoding="utf-8"))
            job_id = str(previous["job_id"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            job_id = ""
        state = _active_job(args.squeue, job_id) if job_id else None
        if state:
            print(f"[launch] existing Slurm job {job_id} is {state}; not resubmitting")
            return 0

    executable = shutil.which(args.sbatch)
    if executable is None:
        raise FileNotFoundError(f"cannot find Slurm submit command: {args.sbatch}")
    submission_command[0] = executable
    completed = subprocess.run(
        submission_command,
        text=True,
        capture_output=True,
        check=True,
    )
    job_id = completed.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise RuntimeError(f"could not parse sbatch job id: {completed.stdout!r}")
    _atomic_json(
        submission_path,
        {
            "schema": "simulatephase2.slurm-submission/v1",
            "contract_sha256": key,
            "job_id": job_id,
            "submitted_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "command": submission_command,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        },
    )
    print(f"[launch] submitted Slurm job {job_id} to partition {args.partition}")
    return 0


def run_local(args: argparse.Namespace, command: list[str], contract: dict[str, object]) -> int:
    launch_dir = args.sim_dir / "launches"
    key = contract_key(contract)
    contract_path = launch_dir / f"{args.profile}.{key}.json"
    print(f"[launch] contract={contract_path}")
    print(f"[launch] local={shlex.join(command)}")
    if args.dry_run:
        return 0
    _atomic_json(contract_path, contract)
    environment = os.environ.copy()
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment.setdefault(name, "1")
    environment.setdefault("MALLOC_ARENA_MAX", "2")
    completed = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parent,
        env=environment,
        check=False,
    )
    return int(completed.returncode)


def main(argv: list[str] | None = None) -> int:
    try:
        args = resolve(parser().parse_args(argv))
        if not args.dry_run:
            validate_inputs(args)
        command = runner_command(args)
        contract = launch_contract(args, command)
        print(
            f"[launch] profile={args.profile} mode={args.mode} pops={args.pops} "
            f"n_sims={args.n_sims:,} workers={args.workers} "
            f"decode={args.decode_workers}x{args.decode_threads} sim_dir={args.sim_dir}"
        )
        if args.mode == "slurm":
            return submit_slurm(args, command, contract)
        return run_local(args, command, contract)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        raise SystemExit(f"simulation launch failed: {error}") from error


if __name__ == "__main__":
    raise SystemExit(main())
