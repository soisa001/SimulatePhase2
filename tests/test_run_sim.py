from __future__ import annotations

import hashlib
import json
import weakref
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import msprime
import numpy as np
import pytest
import tskit
import tszip

import run_sim
from phase2_map import SCHEMA, watterson_a_n


def toy_ancestry(length: int = 35_000, samples: int = 10):
    demography = msprime.Demography()
    demography.add_population(name="afr", initial_size=200)
    return msprime.sim_ancestry(
        samples={"afr": samples},
        demography=demography,
        sequence_length=length,
        recombination_rate=1e-8,
        ploidy=2,
        random_seed=13,
    )


def geometry():
    starts = np.array([0, 10_000, 20_000, 30_000], dtype=np.int64)
    ends = np.array([10_000, 20_000, 30_000, 35_000], dtype=np.int64)
    mask = np.array([[100, 300], [9_900, 10_100], [32_000, 32_100]], dtype=np.int64)
    return starts, ends, mask


def toy_contract(
    target: np.ndarray,
    *,
    initial_rate: float,
    retry_rate: float,
    max_retries: int,
    samples: int = 10,
) -> dict[str, object]:
    config = {
        "map_sha256": "toy-map",
        "mask_sha256": "toy-mask",
        "demography_cache": {"AFR": {"key": "toy-demography"}},
        "base_seed": 42,
        "recombination_rate": 1e-8,
        "initial_rate": initial_rate,
        "retry_rate": retry_rate,
        "max_retries": max_retries,
    }
    _, contract = run_sim.unit_signature(
        config=config,
        population="AFR",
        simulation=0,
        chromosome="chr1",
        sample_count=samples,
        length_bp=35_000,
        window_size=10_000,
        callable_bp=34_500,
        source_sites=int(target.sum()),
        target_sites=int(target.sum()),
        map_sample_count=samples,
        map_a_n=watterson_a_n(samples),
        simulation_a_n=watterson_a_n(samples),
        target_scale=1.0,
    )
    return contract


def test_production_ancestry_seeds_are_unique() -> None:
    seeds = {
        run_sim.stable_seed(42, pop, simulation, f"chr{chromosome}", 1)
        for pop in run_sim.DEFAULT_POPS
        for simulation in range(1_000)
        for chromosome in range(1, 23)
    }
    assert len(seeds) == 6 * 1_000 * 22


def test_demography_coarsening_has_exact_requested_size() -> None:
    times = np.geomspace(100.0, 40_000.0, 1_000)
    indices = run_sim.coarsen_indices(times, 64)
    assert len(indices) == 64
    assert indices[0] == 0 and indices[-1] == 999
    assert np.all(np.diff(indices) > 0)


def test_mask_aware_rate_map() -> None:
    _, _, mask = geometry()
    rate_map = run_sim.mutation_rate_map(
        length_bp=35_000,
        window_size=10_000,
        active_windows=np.array([0, 1, 3]),
        mask=mask,
        rate=1e-5,
    )
    assert rate_map.sequence_length == 35_000
    assert rate_map.get_rate(50) == 1e-5
    assert rate_map.get_rate(150) == 0
    assert rate_map.get_rate(25_000) == 0
    assert rate_map.get_rate(32_050) == 0
    assert rate_map.get_rate(34_000) == 1e-5


def test_exact_calibration_with_forced_retry() -> None:
    starts, ends, mask = geometry()
    target = np.array([5, 8, 0, 3], dtype=np.int64)
    result, stats = run_sim.calibrate_mutations(
        toy_ancestry(),
        target=target,
        starts=starts,
        ends=ends,
        mask=mask,
        unit_contract=toy_contract(target, initial_rate=1e-8, retry_rate=1e-7, max_retries=8),
    )
    assert stats["retry_attempts"] > 1
    assert result.num_sites == int(target.sum())
    realized = result.segregating_sites(windows=np.r_[starts, ends[-1]], span_normalise=False)
    np.testing.assert_array_equal(realized, target)
    assert np.all(np.bincount(result.tables.mutations.site, minlength=result.num_sites) == 1)
    assert not np.any(run_sim.positions_masked(result.tables.sites.position, mask))


def test_calibration_dumps_only_mutation_free_ancestry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mutation overlays are released before ancestry is copied to mutable tables."""
    starts, ends, mask = geometry()
    ancestry = toy_ancestry()
    calls: list[tskit.TreeSequence] = []
    overlay_refs: list[weakref.ReferenceType[tskit.TreeSequence]] = []
    events: list[str] = []
    original_dump = tskit.TreeSequence.dump_tables
    original_mutations = msprime.sim_mutations

    def tracked_dump(tree_sequence: tskit.TreeSequence, *args, **kwargs):
        events.append("dump")
        assert all(reference() is None for reference in overlay_refs)
        calls.append(tree_sequence)
        return original_dump(tree_sequence, *args, **kwargs)

    def tracked_mutations(*args, **kwargs):
        assert all(reference() is None for reference in overlay_refs)
        events.append("mutate")
        overlay = original_mutations(*args, **kwargs)
        overlay_refs.append(weakref.ref(overlay))
        return overlay

    monkeypatch.setattr(tskit.TreeSequence, "dump_tables", tracked_dump)
    monkeypatch.setattr(msprime, "sim_mutations", tracked_mutations)
    target = np.array([5, 8, 0, 3], dtype=np.int64)
    result, stats = run_sim.calibrate_mutations(
        ancestry,
        target=target,
        starts=starts,
        ends=ends,
        mask=mask,
        unit_contract=toy_contract(target, initial_rate=1e-8, retry_rate=1e-7, max_retries=8),
    )
    assert stats["retry_attempts"] > 1
    assert result.num_sites == 16
    assert calls == [ancestry]
    assert events.count("mutate") == stats["retry_attempts"] + 1
    assert events[-1] == "dump"


def test_retry_exhaustion_is_fatal() -> None:
    starts, ends, mask = geometry()
    target = np.array([20, 20, 0, 10])
    with pytest.raises(run_sim.TargetDeficit):
        run_sim.calibrate_mutations(
            toy_ancestry(),
            target=target,
            starts=starts,
            ends=ends,
            mask=mask,
            unit_contract=toy_contract(target, initial_rate=1e-12, retry_rate=1e-12, max_retries=0),
        )


def test_recurrent_sites_are_not_eligible() -> None:
    ancestry = toy_ancestry(length=100, samples=2)
    tables = ancestry.dump_tables()
    site = tables.sites.add_row(position=10, ancestral_state="0")
    child = int(tables.edges.child[0])
    first = tables.mutations.add_row(site=site, node=child, derived_state="1")
    tables.mutations.add_row(site=site, node=child, derived_state="0", parent=first)
    tables.sort()
    recurrent = tables.tree_sequence()
    eligible, _ = run_sim.eligible_site_ids(recurrent, np.empty((0, 2), dtype=np.int64))
    assert len(eligible) == 0


def write_toy_map(
    path: Path,
    target: np.ndarray | None = None,
    *,
    mask_source: str = "NONE",
    mask_sha256: str | None = None,
) -> None:
    if target is None:
        target = np.array([2, 3, 0, 1], dtype=np.uint16)
    target = np.asarray(target, dtype=np.uint16)
    if mask_sha256 is None:
        mask_sha256 = hashlib.sha256(b"NO_MASK").hexdigest()
    with h5py.File(path, "w") as handle:
        handle.attrs.update(
            {
                "schema": SCHEMA,
                "complete": True,
                "window_size": 10_000,
                "total_windows": 4,
                "hardmask_source": mask_source,
                "hardmask_sha256": mask_sha256,
            }
        )
        handle.create_dataset("populations", data=np.array([b"AFR"]))
        handle.create_dataset("sample_counts", data=np.array([3], dtype=np.uint32))
        handle.create_dataset("watterson_a_n", data=np.array([watterson_a_n(3)]))
        samples = handle.create_group("samples")
        dataset = samples.create_dataset("AFR", data=np.array([b"1", b"2", b"3"]))
        dataset.attrs["count"] = 3
        chromosome = handle.create_group("chr1")
        chromosome.attrs["length_bp"] = 35_000
        chromosome.attrs["n_windows"] = 4
        chromosome.create_dataset("S", data=target.reshape(1, -1))


def test_map_snapshot_is_content_addressed_immutable_and_race_safe(tmp_path: Path) -> None:
    source = tmp_path / "source.h5"
    cache = tmp_path / "snapshots"
    write_toy_map(source)
    expected = source.read_bytes()
    expected_sha = run_sim.sha256_file(source)

    with ThreadPoolExecutor(max_workers=8) as executor:
        snapshots = list(executor.map(lambda _: run_sim.snapshot_map_h5(source, cache), range(16)))

    assert {snapshot.path for snapshot in snapshots} == {
        cache.resolve() / f"snv_theta_map.{expected_sha}.h5"
    }
    assert {snapshot.sha256 for snapshot in snapshots} == {expected_sha}
    assert {snapshot.size_bytes for snapshot in snapshots} == {len(expected)}
    assert snapshots[0].path.read_bytes() == expected
    assert len(list(cache.glob("snv_theta_map.*.h5"))) == 1
    assert not list(cache.glob(".map-snapshot.*"))

    with source.open("ab") as handle:
        handle.write(b"new-source-version")
    changed = run_sim.snapshot_map_h5(source, cache)
    assert changed.path != snapshots[0].path
    assert snapshots[0].path.read_bytes() == expected


def test_map_snapshot_reads_the_source_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = (tmp_path / "source.h5").resolve()
    write_toy_map(source)
    expected = source.read_bytes()
    source_opens = 0
    original_open = Path.open

    def tracked_open(path: Path, *args, **kwargs):
        nonlocal source_opens
        if path == source:
            source_opens += 1
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", tracked_open)
    snapshot = run_sim.snapshot_map_h5(source, tmp_path / "snapshots", chunk_size=257)
    assert source_opens == 1
    assert snapshot.sha256 == hashlib.sha256(expected).hexdigest()


def test_mask_snapshot_verifies_local_content(tmp_path: Path) -> None:
    mask = tmp_path / "mask.bed"
    cache = tmp_path / "mask-cache"
    mask.write_text("chr1\t10\t20\n", encoding="utf-8")
    digest = run_sim.sha256_file(mask)
    snapshot = run_sim.snapshot_mask(str(mask), cache, expected_sha256=digest)
    assert snapshot.path == cache.resolve() / f"hardmask.{digest}.bed"
    assert snapshot.path.read_bytes() == mask.read_bytes()
    assert snapshot.sha256 == digest
    with pytest.raises(ValueError, match="differs from the map contract"):
        run_sim.snapshot_mask(str(mask), cache, expected_sha256="0" * 64)


def test_runtime_target_density_and_sample_rescaling() -> None:
    raw_s = np.array([0, 1, 2, 10], dtype=np.int64)
    callable_bp = np.array([0, 100, 200, 1_000], dtype=np.int64)
    source_a_n = watterson_a_n(100)
    exact, density, scale = run_sim.runtime_target_from_raw_s(
        raw_s,
        callable_bp,
        map_a_n=source_a_n,
        simulation_a_n=source_a_n,
    )
    np.testing.assert_array_equal(exact, raw_s)
    np.testing.assert_allclose(density[1:], raw_s[1:] / (source_a_n * callable_bp[1:]))
    assert density[0] == 0.0
    assert scale == 1.0

    target_a_n = watterson_a_n(3)
    rescaled, _, scale = run_sim.runtime_target_from_raw_s(
        raw_s,
        callable_bp,
        map_a_n=source_a_n,
        simulation_a_n=target_a_n,
    )
    np.testing.assert_array_equal(
        rescaled, np.floor(raw_s * (target_a_n / source_a_n) + 0.5).astype(np.int64)
    )
    assert scale == pytest.approx(target_a_n / source_a_n)


def toy_sim_config(tmp_path: Path, target: np.ndarray) -> dict[str, object]:
    _, _, mask = geometry()
    mask_path = tmp_path / "mask.bed"
    mask_path.write_text(
        "".join(f"chr1\t{start}\t{end}\n" for start, end in mask), encoding="utf-8"
    )
    mask_sha256 = run_sim.sha256_file(mask_path)
    map_source = tmp_path / "map.h5"
    write_toy_map(
        map_source,
        target,
        mask_source=str(mask_path),
        mask_sha256=mask_sha256,
    )
    map_snapshot = run_sim.snapshot_map_h5(map_source, tmp_path / "map_snapshots")
    demography = tmp_path / "AFR.npz"
    np.savez_compressed(
        demography,
        cache_key=np.asarray("demo"),
        times=np.array([100.0, 1_000.0]),
        ne=np.array([[200.0, 200.0]], dtype=np.float32),
    )
    return {
        "map_path": str(map_snapshot.path),
        "map_sha256": map_snapshot.sha256,
        "mask_path": str(mask_path),
        "mask_source": str(mask_path),
        "mask_sha256": mask_sha256,
        "sim_dir": str(tmp_path / "sims"),
        "demography_cache": {"AFR": {"path": str(demography), "key": "demo"}},
        "map_sample_counts": {"AFR": 3},
        "map_watterson_a_n": {"AFR": watterson_a_n(3)},
        "sample_counts": {"AFR": 3},
        "simulation_watterson_a_n": {"AFR": watterson_a_n(3)},
        "base_seed": 42,
        "recombination_rate": 1e-8,
        "initial_rate": 1e-5,
        "retry_rate": 1e-4,
        "max_retries": 3,
        "verify_existing": True,
        "fresh": False,
    }


def root_guard_config(sim_dir: Path, populations: tuple[str, ...]) -> dict[str, object]:
    sample_counts = {
        population: 10 + run_sim.DEFAULT_POPS.index(population) for population in populations
    }
    return {
        "sim_dir": str(sim_dir),
        "map_sha256": "a" * 64,
        "mask_sha256": "b" * 64,
        "demography_cache": {
            population: {"key": f"demography-{population.lower()}"} for population in populations
        },
        "map_sample_counts": sample_counts.copy(),
        "map_watterson_a_n": {
            population: watterson_a_n(count) for population, count in sample_counts.items()
        },
        "sample_counts": sample_counts,
        "simulation_watterson_a_n": {
            population: watterson_a_n(count) for population, count in sample_counts.items()
        },
        "base_seed": 42,
        "recombination_rate": 1e-8,
        "initial_rate": 5e-8,
        "retry_rate": 1e-7,
        "max_retries": 8,
    }


def test_sim_root_contract_concurrently_extends_compatible_populations(tmp_path: Path) -> None:
    sim_dir = tmp_path / "sims"
    selections = [("AFR", "EUR"), ("EUR", "EAS"), ("AMR", "SAS", "MID")]
    configs = [root_guard_config(sim_dir, populations) for populations in selections]
    with ThreadPoolExecutor(max_workers=len(configs)) as executor:
        manifests = list(executor.map(run_sim.ensure_sim_root_contract, configs))

    assert set(manifests) == {sim_dir.resolve() / run_sim.SIM_ROOT_CONTRACT_NAME}
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert manifest["schema"] == run_sim.SIM_ROOT_CONTRACT_SCHEMA
    assert set(manifest["populations"]) == set(run_sim.DEFAULT_POPS)
    for population in run_sim.DEFAULT_POPS:
        assert manifest["populations"][population] == {
            "demography_key": f"demography-{population.lower()}",
            "diploid_samples": 10 + run_sim.DEFAULT_POPS.index(population),
            "map_diploid_samples": 10 + run_sim.DEFAULT_POPS.index(population),
            "map_watterson_a_n": pytest.approx(
                watterson_a_n(10 + run_sim.DEFAULT_POPS.index(population))
            ),
            "simulation_watterson_a_n": pytest.approx(
                watterson_a_n(10 + run_sim.DEFAULT_POPS.index(population))
            ),
            "S_scale": 1.0,
        }

    # A compatible subset reuses the union without dropping other populations.
    run_sim.ensure_sim_root_contract(root_guard_config(sim_dir, ("AFR", "EAS")))
    unchanged = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert unchanged == manifest


def test_sim_root_contract_rejects_global_and_population_mismatches(tmp_path: Path) -> None:
    sim_dir = tmp_path / "sims"
    original = root_guard_config(sim_dir, ("AFR",))
    manifest = run_sim.ensure_sim_root_contract(original)
    original_text = manifest.read_text(encoding="utf-8")

    global_mismatch = root_guard_config(sim_dir, ("AFR",))
    global_mismatch["base_seed"] = 43
    with pytest.raises(run_sim.SimRootContractMismatch, match="global contract mismatch"):
        run_sim.ensure_sim_root_contract(global_mismatch)

    sample_mismatch = root_guard_config(sim_dir, ("AFR",))
    sample_mismatch["sample_counts"]["AFR"] = 999
    with pytest.raises(run_sim.SimRootContractMismatch, match="population contract mismatch"):
        run_sim.ensure_sim_root_contract(sample_mismatch)

    demography_mismatch = root_guard_config(sim_dir, ("AFR",))
    demography_mismatch["demography_cache"]["AFR"]["key"] = "different"
    with pytest.raises(run_sim.SimRootContractMismatch, match="population contract mismatch"):
        run_sim.ensure_sim_root_contract(demography_mismatch)
    assert manifest.read_text(encoding="utf-8") == original_text


def test_sim_root_contract_refuses_to_adopt_unmanifested_outputs(tmp_path: Path) -> None:
    sim_dir = tmp_path / "sims"
    orphan = sim_dir / "afr/sim_00000/chr1.tsz"
    orphan.parent.mkdir(parents=True)
    orphan.write_bytes(b"legacy-output")
    with pytest.raises(run_sim.SimRootContractMismatch, match="output exists without"):
        run_sim.ensure_sim_root_contract(root_guard_config(sim_dir, ("AFR",)))
    assert not (sim_dir / run_sim.SIM_ROOT_CONTRACT_NAME).exists()


def assert_compact_unit_provenance(
    ts: tskit.TreeSequence, metadata: dict[str, object]
) -> dict[str, object]:
    assert ts.num_provenances == 1
    provenance = ts.provenance(0)
    assert provenance.timestamp == run_sim.DETERMINISTIC_PROVENANCE_TIMESTAMP
    assert len(provenance.record) < 4_096
    record = json.loads(provenance.record)
    tskit.validate_provenance(record)
    assert record["schema_version"] == "1.0.0"
    assert record["software"]["version"] == run_sim.ALGORITHM_VERSION
    parameters = record["parameters"]
    assert parameters["unit_signature"] == metadata["signature"]
    assert parameters["unit_contract"] == metadata["contract"]
    return parameters


def test_simulate_unit_publishes_only_exact_output(tmp_path: Path) -> None:
    config = toy_sim_config(tmp_path, np.array([2, 3, 0, 1]))
    first = run_sim.simulate_unit((config, "AFR", 0, "chr1"))
    assert first["status"] == "ok", first
    output = tmp_path / "sims/afr/sim_00000/chr1.tsz"
    sidecar = output.with_suffix(".tsz.json")
    assert output.is_file() and sidecar.is_file()
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    assert metadata["target_sites"] == metadata["realized_sites"] == 6
    ts = tszip.decompress(str(output))
    provenance = assert_compact_unit_provenance(ts, metadata)
    seeds = provenance["unit_contract"]["seeds"]
    assert seeds["ancestry"] == run_sim.stable_seed(42, "AFR", 0, "chr1", 1)
    assert seeds["mutation_initial"] == run_sim.stable_seed(42, "AFR", 0, "chr1", 2)
    assert seeds["thinning"] == run_sim.stable_seed(42, "AFR", 0, "chr1", 3)
    assert seeds["mutation_retries"] == [
        run_sim.stable_seed(42, "AFR", 0, "chr1", 4, attempt) for attempt in range(1, 4)
    ]
    starts, ends, mask = geometry()
    run_sim.validate_calibrated(
        ts,
        target=np.array([2, 3, 0, 1]),
        starts=starts,
        ends=ends,
        mask=mask,
        expected_haploids=6,
    )
    second = run_sim.simulate_unit((config, "AFR", 0, "chr1"))
    assert second["status"] == "skip"


def test_zero_theta_output_has_the_same_compact_provenance_contract(tmp_path: Path) -> None:
    config = toy_sim_config(tmp_path, np.zeros(4, dtype=np.int64))
    result = run_sim.simulate_unit((config, "AFR", 0, "chr1"))
    assert result["status"] == "ok", result
    output = tmp_path / "sims/afr/sim_00000/chr1.tsz"
    metadata = json.loads(output.with_suffix(".tsz.json").read_text(encoding="utf-8"))
    ts = tszip.decompress(str(output))
    parameters = assert_compact_unit_provenance(ts, metadata)
    assert ts.num_sites == 0
    assert parameters["outcome"] == {
        "initial_retained": 0,
        "retry_added": 0,
        "retry_attempts": 0,
        "realized_sites": 0,
    }


def test_fresh_rerun_is_fully_deterministic_after_decompression(tmp_path: Path) -> None:
    config = toy_sim_config(tmp_path, np.array([2, 3, 0, 1]))
    config["fresh"] = True
    first = run_sim.simulate_unit((config, "AFR", 0, "chr1"))
    assert first["status"] == "ok", first
    output = tmp_path / "sims/afr/sim_00000/chr1.tsz"
    first_ts = tszip.decompress(str(output))

    second = run_sim.simulate_unit((config, "AFR", 0, "chr1"))
    assert second["status"] == "ok", second
    second_ts = tszip.decompress(str(output))
    assert first_ts.equals(second_ts, ignore_provenance=False)
