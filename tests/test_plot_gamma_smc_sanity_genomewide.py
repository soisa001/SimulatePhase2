from __future__ import annotations

import gzip
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

import plot_gamma_smc_sanity_genomewide
from run_sim import sha256_file


def write_fixture(root: Path, chromosomes: tuple[str, ...]) -> None:
    positions = np.asarray([0, 10_000, 20_000, 30_000], dtype=np.int64)
    values_by_chromosome: dict[str, list[np.ndarray]] = {}
    for chromosome_index, chromosome in enumerate(chromosomes):
        all_values = []
        for simulation in range(10):
            values = np.asarray(
                [
                    0.001 * chromosome_index + 0.005 * simulation,
                    0.002 + 0.001 * chromosome_index + 0.005 * simulation,
                    0.02,
                    0.03,
                ],
                dtype=np.float32,
            )
            all_values.append(values)
            profile = root / "profiles" / "afr" / f"sim_{simulation:05d}" / f"{chromosome}.npz"
            profile.parent.mkdir(parents=True, exist_ok=True)
            metadata = {
                "chromosome": chromosome,
                "simulation": simulation,
                "decode_seconds": float(10 + chromosome_index + simulation),
            }
            np.savez_compressed(
                profile,
                position_0based=positions,
                mean_p_tmrca_lt_threshold=values,
                n_pairs=np.asarray(100_000, dtype=np.int64),
                metadata_json=np.asarray(json.dumps(metadata)),
            )
        values_by_chromosome[chromosome] = all_values

    cutoff_path = root / "cutoffs" / "afr.gamma_smc_cutoffs.10kb.h5"
    cutoff_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(cutoff_path, "w") as handle:
        handle.attrs["population"] = "AFR"
        handle.create_dataset("p_value", data=[0.1])
        for chromosome in chromosomes:
            group = handle.create_group(chromosome)
            group.attrs.update({"complete": True, "n_simulations": 10, "n_pairs": 100_000})
            group.create_dataset("position_0based", data=positions)
            group.create_dataset("end", data=positions + 10_000)
            cutoff = np.stack(values_by_chromosome[chromosome]).max(axis=0)
            group.create_dataset("cutoff", data=cutoff[None, :])


def test_genomewide_analysis_writes_validated_tables_and_figures(tmp_path: Path) -> None:
    sanity = tmp_path / "sanity"
    output = tmp_path / "analysis"
    chromosomes = ("chr1", "chr2", "chr3")
    write_fixture(sanity, chromosomes)

    assert (
        plot_gamma_smc_sanity_genomewide.main(
            [
                "--sanity-dir",
                str(sanity),
                "--output-dir",
                str(output),
                "--population",
                "AFR",
                "--chroms",
                "1-3",
                "--n-sims",
                "10",
                "--dpi",
                "72",
            ]
        )
        == 0
    )
    expected = (
        "genomewide_null_profiles.png",
        "genomewide_null_profiles.pdf",
        "genomewide_across_simulation_summary.png",
        "genomewide_across_simulation_summary.pdf",
        "genomewide_profile_heatmap.png",
        "genomewide_profile_heatmap.pdf",
        "genomewide_profile_distributions.png",
        "genomewide_profile_distributions.pdf",
        "per_simulation_summary.tsv",
        "per_chromosome_summary.tsv",
        "position_summary.tsv.gz",
        "analysis.json",
        "checksums.sha256",
    )
    for name in expected:
        assert (output / name).is_file()

    manifest = json.loads((output / "analysis.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == "simulatephase2.gamma-smc-sanity-genomewide-analysis/v1"
    assert manifest["chromosomes"] == list(chromosomes)
    assert manifest["n_chromosomes"] == 3
    assert manifest["n_positions"] == 12
    assert manifest["n_pairs_per_simulation"] == 100_000
    assert manifest["validation"]["each_cutoff_equals_its_pointwise_profile_maximum"]

    with gzip.open(output / "position_summary.tsv.gz", "rt", encoding="utf-8") as source:
        rows = list(source)
    assert rows[0].startswith("chromosome\tposition_0based\tend\t")
    assert len(rows) == 13

    for line in (output / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        expected_sha, relative_path = line.split("  ", 1)
        assert sha256_file(output / relative_path) == expected_sha


def test_genomewide_analysis_fails_if_one_profile_is_missing(tmp_path: Path) -> None:
    sanity = tmp_path / "sanity"
    chromosomes = ("chr1", "chr2")
    write_fixture(sanity, chromosomes)
    (sanity / "profiles" / "afr" / "sim_00007" / "chr2.npz").unlink()

    with pytest.raises(SystemExit, match="missing Gamma-SMC restart profile"):
        plot_gamma_smc_sanity_genomewide.main(
            [
                "--sanity-dir",
                str(sanity),
                "--population",
                "AFR",
                "--chroms",
                "1-2",
            ]
        )
