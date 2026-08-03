#!/usr/bin/env python3
"""Backward-compatible entry point; use :mod:`run_sim` for new runs."""

from run_sim import main

if __name__ == "__main__":
    raise SystemExit(main())
