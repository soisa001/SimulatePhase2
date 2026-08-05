from __future__ import annotations

import json
from pathlib import Path

import validate_mvn_artifacts


def test_checked_in_mvn_set_passes_production_validation(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    report = tmp_path / "validation.json"
    assert (
        validate_mvn_artifacts.main(
            ["--mvn-dir", str(root / "mvn"), "--report", str(report)]
        )
        == 0
    )
    result = json.loads(report.read_text(encoding="utf-8"))
    assert result["complete"] is True
    assert set(result["populations"]) == {"AFR", "EUR", "AMR", "SAS", "MID", "EAS"}
    assert all(value["complete"] for value in result["populations"].values())
