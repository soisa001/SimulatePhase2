#!/usr/bin/env python
"""Materialize deterministic per-population demographic draws from PHLASH MVNs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from phase2_map import DEFAULT_POPS
from run_sim import (
    DEFAULT_DEMOGRAPHY_CACHE,
    DEFAULT_DEMOGRAPHY_EPOCHS,
    prepare_demography_cache,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--mvn-dir", type=Path, default=Path("mvn"))
    result.add_argument("--demography-cache", type=Path, default=DEFAULT_DEMOGRAPHY_CACHE)
    result.add_argument("--pops", default=",".join(DEFAULT_POPS))
    result.add_argument("--n-sims", type=int, default=1_000)
    result.add_argument("--demography-epochs", type=int, default=DEFAULT_DEMOGRAPHY_EPOCHS)
    result.add_argument("--base-seed", type=int, default=42)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.n_sims <= 0 or args.demography_epochs <= 0 or args.base_seed < 0:
        raise SystemExit("n-sims and demography-epochs must be positive; base-seed must be >= 0")
    populations = [part.strip().upper() for part in args.pops.split(",") if part.strip()]
    unknown = [population for population in populations if population not in DEFAULT_POPS]
    if unknown or len(populations) != len(set(populations)):
        raise SystemExit(f"invalid or duplicate populations: {unknown or populations}")
    cache = prepare_demography_cache(
        args.demography_cache.expanduser().resolve(),
        args.mvn_dir.expanduser().resolve(),
        populations,
        n_sims=args.n_sims,
        epochs=args.demography_epochs,
        seed=args.base_seed,
    )
    print(
        json.dumps(
            {
                "mvn_dir": str(args.mvn_dir.expanduser().resolve()),
                "demography_cache": str(args.demography_cache.expanduser().resolve()),
                "populations": cache,
                "n_sims": args.n_sims,
                "requested_epochs": args.demography_epochs,
                "base_seed": args.base_seed,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
