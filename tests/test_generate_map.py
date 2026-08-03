from __future__ import annotations

import argparse
import gzip
import io
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

import generate_map
from phase2_map import SCHEMA, ChromosomeTarget, canonical_chrom, load_target, parse_chroms

AWK = shutil.which("awk")
if AWK is None:
    git_awk = Path("C:/Program Files/Git/usr/bin/awk.exe")
    AWK = str(git_awk) if git_awk.is_file() else None


def test_parse_chromosomes_and_window_geometry() -> None:
    assert canonical_chrom("CHR1") == "chr1"
    assert parse_chroms("1-2,chr7") == ["chr1", "chr2", "chr7"]
    mask = generate_map.merge_intervals(
        [(50, 150), (100, 300), (9_900, 10_100), (34_900, 40_000)], 35_000
    )
    np.testing.assert_array_equal(mask, [[50, 300], [9_900, 10_100], [34_900, 35_000]])
    starts, ends, callable_bp = generate_map.window_geometry(35_000, 10_000, mask)
    np.testing.assert_array_equal(starts, [0, 10_000, 20_000, 30_000])
    np.testing.assert_array_equal(ends, [10_000, 20_000, 30_000, 35_000])
    np.testing.assert_array_equal(callable_bp, [9_650, 9_900, 10_000, 4_900])


def test_target_rejects_nonuniform_interior_windows() -> None:
    with pytest.raises(ValueError, match="uniform fixed-width"):
        ChromosomeTarget(
            chromosome="chr1",
            population="AFR",
            length_bp=20_000,
            window_size=10_000,
            starts=np.array([0, 5_000, 10_000]),
            ends=np.array([5_000, 10_000, 20_000]),
            callable_bp=np.array([5_000, 5_000, 10_000]),
            theta=np.array([1, 1, 1]),
            mask_intervals=np.empty((0, 2), dtype=np.int64),
        ).validate()


def test_gzipped_mask_is_stream_merged(tmp_path: Path) -> None:
    path = tmp_path / "mask.bed.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write("1\t10\t20\nchr1\t20\t30\nchr1\t25\t40\nchr2\t5\t8\n")
    masks = generate_map.load_mask(path, ("chr1", "chr2"))
    np.testing.assert_array_equal(masks["chr1"], [[10, 40]])
    np.testing.assert_array_equal(masks["chr2"], [[5, 8]])


def test_ancestry_qc_and_relatedness_gating(tmp_path: Path) -> None:
    ancestry = tmp_path / "ancestry.tsv"
    ancestry.write_text(
        "s\tancestry_pred_other\n"
        "1\tAFR\n2\tEUR\n3\tancestry_pred_other\n4\tAMR\n"
        "6\tSAS\n7\tMID\n8\tEAS\n",
        encoding="utf-8",
    )
    flagged = tmp_path / "flagged.tsv"
    flagged.write_text("research_id\n2\n", encoding="utf-8")
    related = tmp_path / "related.tsv"
    related.write_text("research_id\n4\n", encoding="utf-8")
    populations = ("AFR", "EUR", "AMR", "SAS", "MID", "EAS")
    selected, stats = generate_map.build_gated_panel(
        list(map(str, range(1, 9))),
        ancestry,
        flagged,
        related,
        populations,
        ancestry_column="ancestry_pred_other",
        ancestry_id_column="auto",
    )
    assert selected == {
        "AFR": ["1"],
        "EUR": [],
        "AMR": [],
        "SAS": ["6"],
        "MID": ["7"],
        "EAS": ["8"],
    }
    assert stats["missing_ancestry"] == 1
    assert stats["other_or_unrequested"] == 1
    assert stats["flagged"] == 1
    assert stats["related"] == 1


def test_aou_tsv_header_wins_over_comma_arrays(tmp_path: Path) -> None:
    ancestry = tmp_path / "ancestry_preds.tsv"
    ancestry.write_text(
        "s\tprobabilities\tpca_features\tancestry_pred_other\n"
        "1000980\t[0.0,0.99,0.01]\t[0.1,0.2]\tamr\n"
        "1000981\t[0.0,0.01,0.99]\t[0.3,0.4]\teur\n",
        encoding="utf-8",
    )
    columns, rows = generate_map.read_delimited(ancestry)
    assert columns == ["s", "probabilities", "pca_features", "ancestry_pred_other"]
    assert rows[0]["probabilities"] == "[0.0,0.99,0.01]"
    selected, _ = generate_map.build_gated_panel(
        ["1000980", "1000981"],
        ancestry,
        None,
        None,
        ("AMR", "EUR"),
        ancestry_column="ancestry_pred_other",
        ancestry_id_column="auto",
    )
    assert selected == {"AMR": ["1000980"], "EUR": ["1000981"]}


def test_conflicting_ancestry_labels_are_fatal(tmp_path: Path) -> None:
    ancestry = tmp_path / "ancestry.tsv"
    ancestry.write_text("s\tancestry_pred_other\n1\tAFR\n1\tEUR\n", encoding="utf-8")
    with pytest.raises(ValueError, match="conflicting ancestry"):
        generate_map.build_gated_panel(
            ["1"],
            ancestry,
            None,
            None,
            ("AFR", "EUR"),
            ancestry_column="ancestry_pred_other",
            ancestry_id_column="auto",
        )


def test_conflicting_manifest_assignments_are_fatal(tmp_path: Path) -> None:
    manifest = tmp_path / "samples.tsv"
    manifest.write_text("sample_id\tpopulation\nlong-id\tAFR\nlong-id\tEUR\n", encoding="utf-8")
    with pytest.raises(ValueError, match="assigned to both"):
        generate_map.load_manifest(manifest, ("AFR", "EUR"))


def test_compact_unsigned_grows_without_overflow() -> None:
    assert generate_map.compact_unsigned(np.array([0, 10_000])).dtype == np.uint16
    values = generate_map.compact_unsigned(np.array([0, 70_000]))
    assert values.dtype == np.uint32
    assert int(values[-1]) == 70_000


def test_gcs_generation_changes_cache_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    generation = {"value": "101"}

    def fake_run(*args, **kwargs):
        return SimpleNamespace(
            stdout=('{"generation":"' + generation["value"] + '","size":"123","crc32c_hash":"abc"}')
        )

    monkeypatch.setattr(generate_map.subprocess, "run", fake_run)
    first = generate_map.source_identity(
        "gs://bucket/file.bcf", gcloud="gcloud", billing_project=None
    )
    generation["value"] = "102"
    second = generate_map.source_identity(
        "gs://bucket/file.bcf", gcloud="gcloud", billing_project=None
    )
    assert first["generation"] != second["generation"]
    assert generate_map.cache_destination("gs://bucket/file.bcf", Path("cache"), first) != (
        generate_map.cache_destination("gs://bucket/file.bcf", Path("cache"), second)
    )


def test_localize_downloads_described_gcs_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[list[str]] = []

    def fake_run(command, **kwargs):
        commands.append(command)
        Path(command[-1]).write_bytes(b"data")
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(generate_map.subprocess, "run", fake_run)
    identity = {
        "source": "gs://bucket/file.bcf",
        "kind": "gcs",
        "generation": "101",
        "size": "4",
    }
    localized, downloaded = generate_map.localize(
        "gs://bucket/file.bcf",
        tmp_path,
        gcloud="gcloud",
        billing_project=None,
        identity=identity,
    )
    assert downloaded
    assert localized.read_bytes() == b"data"
    assert commands[0][-2] == "gs://bucket/file.bcf#101"


def test_nonempty_mask_outside_contig_is_fatal() -> None:
    with pytest.raises(RuntimeError, match="none overlap its BCF length"):
        generate_map.clip_merged_mask(np.array([[30_000, 40_000]]), 25_000)


def test_checkpoint_hit_deletes_leftover_cloud_bcf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity = {
        "source": "gs://bucket/chr1.bcf",
        "kind": "gcs",
        "generation": "101",
        "size": "4",
    }
    cache_root = tmp_path / "cache"
    destination = generate_map.cache_destination(
        "gs://bucket/chr1.bcf", cache_root / "bcf", identity
    )
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"data")
    work_dir = tmp_path / "work"
    checkpoint = work_dir / "chromosomes" / "chr1.npz"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"already complete")
    monkeypatch.setattr(generate_map, "checkpoint_valid", lambda *args: True)

    def unexpected_localize(*args, **kwargs):
        pytest.fail("a checkpoint hit must not reopen or download the BCF")

    monkeypatch.setattr(generate_map, "localize", unexpected_localize)
    args = argparse.Namespace(
        bcf_template="gs://bucket/{chrom}.bcf",
        work_dir=work_dir,
        cache_dir=cache_root,
        gcloud="gcloud",
        billing_project=None,
        bcftools_version="bcftools 1.22",
        window_size=10_000,
        filters="PASS,.",
        min_call_rate=0.0,
        fresh=False,
        delete_localized=True,
    )
    result = generate_map.process_chromosome(
        "chr1",
        args=args,
        source_identity_value=identity,
        populations=("AFR",),
        samples={"AFR": ["A"]},
        sample_digest="samples",
        mask_by_chrom={"chr1": np.empty((0, 2), dtype=np.int64)},
        mask_digest="mask",
        samples_file=tmp_path / "samples.txt",
        groups_file=tmp_path / "groups.tsv",
    )
    assert result == checkpoint
    assert not destination.exists()


@pytest.mark.skipif(AWK is None, reason="awk is unavailable")
def test_awk_reducer_drops_repeated_positions() -> None:
    # POS0, REF, ALT, FILTER, then AC/AN pairs. Position 10000 has a SNP
    # plus an indel sibling and must contribute to neither population. The
    # filtered singleton at 15000 is also ineligible.
    data = (
        "0\tA\tG\tPASS\t1\t4\t0\t4\n"
        "9999\tC\tT\tPASS\t4\t4\t2\t4\n"
        "10000\tA\tC\tPASS\t1\t4\t1\t4\n"
        "10000\tA\tAT\tPASS\t2\t4\t2\t4\n"
        "15000\tG\tA\tq10\t1\t4\t1\t4\n"
        "19999\tG\tA\t.\t1\t4\t3\t4\n"
    )
    result = subprocess.run(
        [
            AWK,
            "-v",
            "W=10000",
            "-v",
            "NW=2",
            "-v",
            "NP=2",
            "-v",
            "FILTERS=PASS,.",
            "-v",
            "EXPECTED_AN=4,4",
            "-v",
            "MIN_AN=0,0",
            generate_map.awk_reducer(2),
        ],
        input=data,
        check=True,
        capture_output=True,
        text=True,
    )
    lines = result.stdout.splitlines()
    assert lines[0] == "#QC\t6\t5\t1\t3"
    assert lines[1] == "#LOWAN\t0\t0"
    matrix = np.loadtxt(io.StringIO("\n".join(lines[2:])), delimiter="\t", dtype=int)
    np.testing.assert_array_equal(matrix, [[0, 1, 1], [1, 1, 1]])


@pytest.mark.skipif(AWK is None, reason="awk is unavailable")
def test_awk_reducer_applies_per_population_minimum_an() -> None:
    data = "0\tA\tG\tPASS\t1\t4\t1\t2\n1\tC\tT\tPASS\t1\t2\t1\t4\n"
    result = subprocess.run(
        [
            AWK,
            "-v",
            "W=10000",
            "-v",
            "NW=1",
            "-v",
            "NP=2",
            "-v",
            "FILTERS=PASS,.",
            "-v",
            "EXPECTED_AN=4,4",
            "-v",
            "MIN_AN=4,4",
            generate_map.awk_reducer(2),
        ],
        input=data,
        check=True,
        capture_output=True,
        text=True,
    )
    lines = result.stdout.splitlines()
    assert lines[1] == "#LOWAN\t1\t1"
    assert lines[2] == "0\t1\t1"


def test_fully_masked_chromosome_skips_variant_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bcf = tmp_path / "chr1.bcf"
    bcf.write_bytes(b"placeholder")
    samples_file = tmp_path / "samples.txt"
    samples_file.write_text("A\nB\n", encoding="utf-8")
    groups_file = tmp_path / "groups.tsv"
    groups_file.write_text("A\tAFR\nB\tEUR\n", encoding="utf-8")
    monkeypatch.setattr(
        generate_map,
        "bcf_header",
        lambda path, executable: ({"chr1": 20_000}, ["A", "B"], "header"),
    )

    def unexpected_pipeline(**kwargs):
        pytest.fail("the variant pipeline must not run for a fully masked contig")

    monkeypatch.setattr(generate_map, "run_count_pipeline", unexpected_pipeline)
    args = argparse.Namespace(
        bcf_template=str(bcf),
        work_dir=tmp_path / "work",
        cache_dir=tmp_path / "cache",
        gcloud="gcloud",
        billing_project=None,
        bcftools_version="bcftools 1.22",
        window_size=10_000,
        filters="PASS,.",
        min_call_rate=0.95,
        bcftools="bcftools",
        awk="awk",
        threads=1,
        fresh=False,
        delete_localized=False,
    )
    checkpoint = generate_map.process_chromosome(
        "chr1",
        args=args,
        source_identity_value=generate_map.source_identity(
            bcf, gcloud="gcloud", billing_project=None
        ),
        populations=("AFR", "EUR"),
        samples={"AFR": ["A"], "EUR": ["B"]},
        sample_digest="samples",
        mask_by_chrom={"chr1": np.array([[0, 20_000]], dtype=np.int64)},
        mask_digest="mask",
        samples_file=samples_file,
        groups_file=groups_file,
    )
    with np.load(checkpoint, allow_pickle=False) as data:
        np.testing.assert_array_equal(data["callable_bp"], [0, 0])
        np.testing.assert_array_equal(data["theta"], np.zeros((2, 2), dtype=np.uint16))
        qc = json.loads(str(data["qc_json"]))
        assert qc["source_records_in_callable_intervals"] == 0
        assert qc["minimum_an"] == {"AFR": 2, "EUR": 2}
        assert qc["segregating_positions_excluded_low_an"] == {"AFR": 0, "EUR": 0}


def test_compact_hdf5_round_trip(tmp_path: Path) -> None:
    checkpoint = tmp_path / "chr1.npz"
    starts = np.array([0, 10_000, 20_000, 30_000], dtype=np.int32)
    ends = np.array([10_000, 20_000, 30_000, 35_000], dtype=np.int32)
    theta = np.array([[1, 2, 0, 1], [0, 3, 1, 0]], dtype=np.uint16)
    generate_map.write_checkpoint(
        checkpoint,
        checkpoint_key=np.asarray("key"),
        chromosome=np.asarray("chr1"),
        source=np.asarray("toy.bcf"),
        source_identity_json=np.asarray('{"kind":"local"}'),
        source_local_size=np.asarray(123, dtype=np.int64),
        header_sha256=np.asarray("header"),
        contig=np.asarray("chr1"),
        length_bp=np.asarray(35_000, dtype=np.int64),
        starts=starts,
        ends=ends,
        callable_bp=np.array([10_000, 10_000, 9_900, 5_000], dtype=np.uint16),
        mask_intervals=np.array([[20_100, 20_200]], dtype=np.int32),
        theta=theta,
        min_call_rate=np.asarray(0.5, dtype=np.float64),
        expected_an=np.array([4, 2], dtype=np.uint32),
        minimum_an=np.array([2, 1], dtype=np.uint32),
        qc_json=np.asarray(
            json.dumps(
                {
                    "biallelic_snv_records": 8,
                    "segregating_positions_excluded_low_an": {"AFR": 3, "EUR": 1},
                }
            )
        ),
    )
    args = argparse.Namespace(
        window_size=10_000,
        bcf_template="toy.{chrom}.bcf",
        filters="PASS,.",
        bcftools_version="bcftools 1.22",
        min_call_rate=0.5,
    )
    output = tmp_path / "map.h5"
    summary = generate_map.write_hdf5(
        output,
        {"chr1": checkpoint},
        args=args,
        populations=("AFR", "EUR"),
        samples={"AFR": ["1", "2"], "EUR": ["3"]},
        sample_digest="samples",
        mask_digest="mask",
        selection_stats={"retained": 3},
    )
    assert summary["theta_totals"] == {"AFR": 4, "EUR": 4}
    assert summary["callability_policy"]["minimum_an"] == {"AFR": 2, "EUR": 1}
    with h5py.File(output, "r") as handle:
        assert handle.attrs["schema"] == SCHEMA
        assert bool(handle.attrs["complete"])
        dataset = handle["chr1/afr/theta"]
        assert dataset.dtype == np.dtype("uint16")
        assert dataset.compression == "gzip"
        assert dataset.fletcher32
        assert dataset.attrs["minimum_an"] == 2
        assert dataset.attrs["segregating_positions_excluded_low_an"] == 3
        assert json.loads(handle["chr1"].attrs["qc_json"])[
            "segregating_positions_excluded_low_an"
        ] == {"AFR": 3, "EUR": 1}
    target = load_target(output, "1", "afr")
    np.testing.assert_array_equal(target.theta, theta[0])
    np.testing.assert_array_equal(target.mask_intervals, [[20_100, 20_200]])
