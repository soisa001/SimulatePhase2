from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

import evaluate_mvn_error
from phase2_map import DEFAULT_POPS


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_compare_curves_is_zero_for_identical_distributions() -> None:
    curves = np.arange(1.0, 41.0).reshape(8, 5)
    times = np.geomspace(100.0, 40_000.0, curves.shape[1])
    arrays, metrics = evaluate_mvn_error.compare_curves(curves, curves, times)
    assert np.allclose(arrays["relative_error_percent"], 0.0)
    assert metrics["median_mean_absolute_error_percent"] == 0.0
    assert metrics["central_quantile_log_rmse_mean"] == 0.0


def test_compare_curves_reports_a_known_scale_error() -> None:
    reference = np.arange(1.0, 41.0).reshape(8, 5)
    times = np.geomspace(100.0, 40_000.0, reference.shape[1])
    arrays, metrics = evaluate_mvn_error.compare_curves(reference, 2.0 * reference, times)
    assert np.allclose(arrays["relative_error_percent"], 100.0)
    assert np.isclose(metrics["central_quantile_log_rmse_mean"], np.log(2.0))


def test_checked_in_mvn_error_report_matches_outputs() -> None:
    root = Path(__file__).resolve().parents[1]
    report = json.loads((root / "mvn" / "plots" / "mvn_error_report.json").read_text())
    assert report["schema"] == evaluate_mvn_error.REPORT_SCHEMA
    assert report["complete"] is True
    assert report["base_seed"] == 42
    assert report["n_draws_per_population"] == 1_000
    assert report["source_pkl_manifest"]["count"] == 600
    assert set(report["populations"]) == set(DEFAULT_POPS)
    for value in report["plots"].values():
        path = root / value["path"]
        assert path.stat().st_size > 1_000
        assert _sha256(path) == value["sha256"]
