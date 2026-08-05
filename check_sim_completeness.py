#!/usr/bin/env python3
"""Quickly verify every expected simulation sidecar and TSZip central directory."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import tszip

from phase2_map import DEFAULT_POPS, parse_chroms
from run_sim import (
    DEFAULT_SIM_DIR,
    SIM_ROOT_CONTRACT_NAME,
    SIM_ROOT_CONTRACT_SCHEMA,
    atomic_text,
)
from simulation_outputs import completed_units

COMPLETENESS_SCHEMA = "simulatephase2.sim-completeness/v1"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--sim-dir", type=Path, default=DEFAULT_SIM_DIR)
    result.add_argument("--pops", default=",".join(DEFAULT_POPS))
    result.add_argument("--chroms", default="1-22")
    result.add_argument("--n-sims", type=int, default=1_000)
    result.add_argument(
        "--deep",
        action="store_true",
        help="Also fully decompress each TSZip (slow; central-directory checks are the default)",
    )
    result.add_argument("--output", type=Path, default=None)
    return result


def _load_contract(sim_dir: Path, populations: list[str]) -> dict[str, object]:
    path = sim_dir / SIM_ROOT_CONTRACT_NAME
    try:
        contract = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise ValueError(f"cannot read simulation contract {path}: {error}") from error
    entries = contract.get("populations") if isinstance(contract, dict) else None
    if (
        not isinstance(contract, dict)
        or contract.get("schema") != SIM_ROOT_CONTRACT_SCHEMA
        or not isinstance(entries, dict)
        or any(population not in entries for population in populations)
    ):
        raise ValueError(f"invalid or incomplete simulation contract: {path}")
    return contract


def check(
    *,
    sim_dir: Path,
    populations: list[str],
    chromosomes: list[str],
    n_sims: int,
    deep: bool,
) -> dict[str, object]:
    started = time.monotonic()
    contract = _load_contract(sim_dir, populations)
    groups: dict[str, object] = {}
    for population in populations:
        for chromosome in chromosomes:
            paths, digest = completed_units(
                sim_dir,
                population,
                chromosome,
                n_sims,
                check_archive=True,
            )
            if deep:
                for path in paths:
                    tszip.decompress(str(path))
            groups[f"{population}/{chromosome}"] = {
                "complete": True,
                "n_simulations": n_sims,
                "source_manifest_sha256": digest,
            }
            print(
                f"[completeness] {population} {chromosome}: {n_sims:,} complete",
                flush=True,
            )
    return {
        "schema": COMPLETENESS_SCHEMA,
        "complete": True,
        "sim_dir": str(sim_dir),
        "simulation_contract_schema": contract["schema"],
        "populations": populations,
        "chromosomes": chromosomes,
        "n_simulations_per_population_chromosome": n_sims,
        "n_units": len(populations) * len(chromosomes) * n_sims,
        "validation": "full_tsz_decompression" if deep else "sidecar_size_and_tsz_zip_footer",
        "groups": groups,
        "seconds": time.monotonic() - started,
    }


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.n_sims <= 0:
        raise SystemExit("--n-sims must be positive")
    try:
        chromosomes = parse_chroms(args.chroms)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    populations = [part.strip().upper() for part in args.pops.split(",") if part.strip()]
    unknown = [population for population in populations if population not in DEFAULT_POPS]
    if unknown or not populations or len(populations) != len(set(populations)):
        raise SystemExit(f"invalid or duplicate populations: {unknown or populations}")
    sim_dir = args.sim_dir.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else sim_dir / "simulation_completeness.json"
    )
    try:
        report = check(
            sim_dir=sim_dir,
            populations=populations,
            chromosomes=chromosomes,
            n_sims=args.n_sims,
            deep=args.deep,
        )
    except Exception as error:
        raise SystemExit(f"simulation completeness check failed: {error}") from error
    atomic_text(output, json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"[completeness] wrote {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
