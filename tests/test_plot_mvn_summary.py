from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

import plot_mvn_summary
from phase2_map import DEFAULT_POPS


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_summarize_orders_pointwise_interval() -> None:
    draws = np.arange(1.0, 21.0).reshape(4, 5)
    summary = plot_mvn_summary.summarize(draws)
    assert np.all(summary["low"] <= summary["median"])
    assert np.all(summary["median"] <= summary["high"])


def test_checked_in_plot_manifest_matches_outputs() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "mvn" / "plots" / "plot_manifest.json").read_text())
    assert manifest["schema"] == plot_mvn_summary.REPORT_SCHEMA
    assert manifest["complete"] is True
    assert manifest["n_draws_per_population"] == 1_000
    assert manifest["axis_limits"] == {
        "diploid_ne": [1_000.0, 80_000.0],
        "generations": [100.0, 40_000.0],
    }
    assert set(manifest["populations"]) == set(DEFAULT_POPS)
    plots = [manifest["combined_plots"]]
    plots.extend(value["plots"] for value in manifest["populations"].values())
    for formats in plots:
        for value in formats.values():
            path = root / value["path"]
            assert path.stat().st_size > 1_000
            assert _sha256(path) == value["sha256"]
