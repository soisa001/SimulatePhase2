from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

import plot_gamma_smc_sanity


def write_fixture(root: Path, *, mismatch_cutoff: bool = False) -> None:
    positions = np.asarray([0, 10_000, 20_000, 30_000], dtype=np.int64)
    all_values = []
    for simulation in range(10):
        values = np.asarray(
            [0.01 * simulation, 0.001 + 0.01 * simulation, 0.02, 0.03],
            dtype=np.float32,
        )
        all_values.append(values)
        profile = root / "profiles" / "afr" / f"sim_{simulation:05d}" / "chr1.npz"
        profile.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "chromosome": "chr1",
            "simulation": simulation,
            "decode_seconds": float(10 + simulation),
        }
        np.savez_compressed(
            profile,
            position_0based=positions,
            mean_p_tmrca_lt_threshold=values,
            n_pairs=np.asarray(100_000, dtype=np.int64),
            metadata_json=np.asarray(json.dumps(metadata)),
        )

    cutoff_path = root / "cutoffs" / "afr.gamma_smc_cutoffs.10kb.h5"
    cutoff_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(cutoff_path, "w") as handle:
        handle.attrs["population"] = "AFR"
        handle.create_dataset("p_value", data=[0.1])
        group = handle.create_group("chr1")
        group.attrs.update({"complete": True, "n_simulations": 10, "n_pairs": 100_000})
        group.create_dataset("position_0based", data=positions)
        group.create_dataset("end", data=positions + 10_000)
        cutoff = np.stack(all_values).max(axis=0)
        if mismatch_cutoff:
            cutoff[0] += np.float32(0.01)
        group.create_dataset("cutoff", data=cutoff[None, :])


def test_chromosome_analysis_writes_validated_tables_and_both_figure_formats(
    tmp_path: Path,
) -> None:
    sanity = tmp_path / "sanity"
    output = tmp_path / "analysis"
    write_fixture(sanity)
    output.mkdir()
    for name in ("profiles_and_cutoff.png", "profiles_and_cutoff.pdf"):
        (output / name).write_bytes(b"stale legacy plot")
    assert (
        plot_gamma_smc_sanity.main(
            [
                "--sanity-dir",
                str(sanity),
                "--output-dir",
                str(output),
                "--population",
                "AFR",
                "--chromosome",
                "1",
                "--n-sims",
                "10",
                "--dpi",
                "72",
            ]
        )
        == 0
    )
    for name in (
        "null_profiles.png",
        "null_profiles.pdf",
        "across_simulation_summary.png",
        "across_simulation_summary.pdf",
        "profile_heatmap.png",
        "profile_heatmap.pdf",
        "profile_distributions.png",
        "profile_distributions.pdf",
        "per_simulation_summary.tsv",
        "position_summary.tsv.gz",
        "analysis.json",
        "checksums.sha256",
    ):
        assert (output / name).is_file()
    for name in ("profiles_and_cutoff.png", "profiles_and_cutoff.pdf"):
        assert not (output / name).exists()
    manifest = json.loads((output / "analysis.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == "simulatephase2.gamma-smc-sanity-chromosome-analysis/v2"
    assert manifest["validation"]["p_le_0.1_cutoff_equals_pointwise_profile_maximum"]
    assert manifest["n_pairs_per_simulation"] == 100_000


def test_chromosome_analysis_rejects_a_cutoff_not_derived_from_profiles(tmp_path: Path) -> None:
    sanity = tmp_path / "sanity"
    write_fixture(sanity, mismatch_cutoff=True)
    profiles = plot_gamma_smc_sanity.load_profiles(
        sanity, population="AFR", chromosome="chr1", n_sims=10
    )
    with pytest.raises(ValueError, match="pointwise maximum"):
        plot_gamma_smc_sanity.load_cutoff(
            sanity,
            population="AFR",
            chromosome="chr1",
            profiles=profiles,
        )
