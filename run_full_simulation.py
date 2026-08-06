#!/usr/bin/env python3
"""Run restartable demography, simulation, completeness, and cutoff phases."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from phase2_map import (
    DEFAULT_GAMMA_SMC_REPO,
    DEFAULT_HARDMASK_PATH,
    DEFAULT_MAP_PATH,
    DEFAULT_MVN_DIR,
    DEFAULT_POPS,
)
from run_sim import DEFAULT_DEMOGRAPHY_CACHE, DEFAULT_SIM_DIR


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--phase",
        choices=("all", "demography", "simulate", "check", "cutoffs"),
        default="all",
    )
    result.add_argument(
        "--cutoff-mode", choices=("compact", "gamma-smc", "both"), default="compact"
    )
    result.add_argument("--map", type=Path, default=DEFAULT_MAP_PATH)
    result.add_argument("--mask", type=Path, default=DEFAULT_HARDMASK_PATH)
    result.add_argument("--mvn-dir", type=Path, default=DEFAULT_MVN_DIR)
    result.add_argument("--demography-cache", type=Path, default=DEFAULT_DEMOGRAPHY_CACHE)
    result.add_argument("--sim-dir", type=Path, default=DEFAULT_SIM_DIR)
    result.add_argument("--pops", default=",".join(DEFAULT_POPS))
    result.add_argument("--chroms", default="1-22")
    result.add_argument("--n-sims", type=int, default=1_000)
    result.add_argument("--demography-epochs", type=int, default=10_000)
    result.add_argument("--base-seed", type=int, default=42)
    result.add_argument("--samples-per-population", type=int, default=0)
    result.add_argument("--workers", type=int, default=4)
    result.add_argument("--max-pending", type=int, default=0)
    result.add_argument("--p-values", default="0.01,0.05")
    result.add_argument("--window-size", type=int, default=10_000)
    result.add_argument("--threshold-years", type=float, default=4_500.0)
    result.add_argument("--generation-time", type=float, default=25.0)
    result.add_argument("--gamma-smc-repo", type=Path, default=DEFAULT_GAMMA_SMC_REPO)
    result.add_argument("--decode-workers", type=int, default=4)
    result.add_argument("--decode-threads", type=int, default=1)
    result.add_argument("--theta", type=float, default=0.00075)
    result.add_argument("--rho-over-theta", type=float, default=0.8)
    result.add_argument("--mutation-rate", type=float, default=1.29e-8)
    result.add_argument("--billing-project", default=None)
    result.add_argument("--logs-dir", type=Path, default=None)
    result.add_argument("--deep-completeness-check", action="store_true")
    result.add_argument("--keep-gamma-profiles", action="store_true")
    return result


def phase_commands(args: argparse.Namespace, executable: str) -> list[tuple[str, list[str]]]:
    script_dir = Path(__file__).resolve().parent
    common = [
        "--pops",
        args.pops,
        "--n-sims",
        str(args.n_sims),
    ]
    demography = [
        executable,
        "-u",
        str(script_dir / "prepare_demographies.py"),
        "--mvn-dir",
        str(args.mvn_dir),
        "--demography-cache",
        str(args.demography_cache),
        *common,
        "--demography-epochs",
        str(args.demography_epochs),
        "--base-seed",
        str(args.base_seed),
    ]
    simulate = [
        executable,
        "-u",
        str(script_dir / "run_sim.py"),
        "--map",
        str(args.map),
        "--mvn-dir",
        str(args.mvn_dir),
        "--demography-cache",
        str(args.demography_cache),
        "--sim-dir",
        str(args.sim_dir),
        *common,
        "--chroms",
        args.chroms,
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
    ]
    if args.mask:
        simulate.extend(["--mask", str(args.mask)])
    if args.billing_project:
        simulate.extend(["--billing-project", args.billing_project])
    completeness = [
        executable,
        "-u",
        str(script_dir / "check_sim_completeness.py"),
        "--sim-dir",
        str(args.sim_dir),
        *common,
        "--chroms",
        args.chroms,
    ]
    if args.deep_completeness_check:
        completeness.append("--deep")
    compact = [
        executable,
        "-u",
        str(script_dir / "generate_cutoffs.py"),
        "--sim-dir",
        str(args.sim_dir),
        *common,
        "--chroms",
        args.chroms,
        "--window-size",
        str(args.window_size),
        "--threshold-years",
        str(args.threshold_years),
        "--generation-time",
        str(args.generation_time),
        "--p-values",
        args.p_values,
        "--workers",
        str(args.workers),
    ]
    gamma = [
        executable,
        "-u",
        str(script_dir / "generate_cutoff_gamma_smc.py"),
        "--sim-dir",
        str(args.sim_dir),
        *common,
        "--chroms",
        args.chroms,
        "--p-values",
        args.p_values,
        "--gamma-smc-repo",
        str(args.gamma_smc_repo),
        "--threshold-years",
        str(args.threshold_years),
        "--generation-time",
        str(args.generation_time),
        "--stride",
        str(args.window_size),
        "--decode-workers",
        str(args.decode_workers),
        "--decode-threads",
        str(args.decode_threads),
        "--theta",
        str(args.theta),
        "--rho-over-theta",
        str(args.rho_over_theta),
        "--mutation-rate",
        str(args.mutation_rate),
    ]
    if args.mask:
        if str(args.mask).startswith("gs://"):
            raise ValueError(
                "Gamma-SMC cutoff mode requires --mask to be a localized BED/BED.gz path"
            )
        gamma.extend(["--hardmask", str(args.mask)])
    if args.keep_gamma_profiles:
        gamma.append("--keep-profiles")

    selected: list[tuple[str, list[str]]] = []
    if args.phase in ("all", "demography"):
        selected.append(("phase1_demography", demography))
    if args.phase in ("all", "simulate"):
        selected.extend([("phase2_simulate", simulate), ("phase2_check", completeness)])
    elif args.phase == "check":
        selected.append(("phase2_check", completeness))
    if args.phase in ("all", "cutoffs"):
        if args.phase == "cutoffs":
            selected.append(("phase2_check", completeness))
        if args.cutoff_mode in ("compact", "both"):
            selected.append(("phase3_compact_cutoffs", compact))
        if args.cutoff_mode in ("gamma-smc", "both"):
            selected.append(("phase3_gamma_smc_cutoffs", gamma))
    return selected


def run_logged(name: str, command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "phase": name,
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "cwd": os.getcwd(),
        "command": command,
        "python": sys.version,
    }
    print(f"[{name}] {shlex.join(command)}", flush=True)
    with log.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(header, sort_keys=True) + "\n")
        handle.flush()
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line)
        returncode = process.wait()
        handle.write(
            json.dumps(
                {
                    "phase": name,
                    "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "returncode": returncode,
                },
                sort_keys=True,
            )
            + "\n"
        )
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    positive = (
        args.n_sims,
        args.demography_epochs,
        args.workers,
        args.window_size,
        args.decode_workers,
        args.decode_threads,
    )
    if any(value <= 0 for value in positive) or args.base_seed < 0:
        raise SystemExit("counts/epochs/workers must be positive and seed nonnegative")
    args.map = args.map.expanduser().resolve()
    args.mask = args.mask.expanduser().resolve() if args.mask is not None else None
    args.mvn_dir = args.mvn_dir.expanduser().resolve()
    args.demography_cache = args.demography_cache.expanduser().resolve()
    args.sim_dir = args.sim_dir.expanduser().resolve()
    args.gamma_smc_repo = args.gamma_smc_repo.expanduser().resolve()
    logs_dir = (
        args.logs_dir.expanduser().resolve()
        if args.logs_dir is not None
        else args.sim_dir / "logs"
    )
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    try:
        commands = phase_commands(args, sys.executable)
        for name, command in commands:
            run_logged(name, command, logs_dir / f"{stamp}.{name}.log")
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"full simulation runner failed: {error}") from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
