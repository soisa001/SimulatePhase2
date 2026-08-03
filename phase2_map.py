"""Shared reader and validation helpers for SimulatePhase2 theta maps."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

SCHEMA = "simulatephase2.snv-theta-map/v1"
DEFAULT_POPS = ("AFR", "EUR", "AMR", "SAS", "MID", "EAS")


def canonical_chrom(value: str | int) -> str:
    value = str(value).strip()
    if value.lower().startswith("chr"):
        value = value[3:]
    return f"chr{value}"


def parse_chroms(spec: str) -> list[str]:
    """Parse ``1-3,7,chr10`` while preserving order and rejecting duplicates."""
    result: list[str] = []
    for token in (part.strip() for part in spec.split(",")):
        if not token:
            continue
        raw = token[3:] if token.lower().startswith("chr") else token
        if "-" in raw:
            left, right = raw.split("-", 1)
            start, stop = int(left), int(right)
            if stop < start:
                raise ValueError(f"invalid chromosome range: {token}")
            result.extend(f"chr{i}" for i in range(start, stop + 1))
        else:
            result.append(canonical_chrom(raw))
    if not result:
        raise ValueError("no chromosomes were selected")
    if len(set(result)) != len(result):
        raise ValueError(f"duplicate chromosomes in {spec!r}")
    return result


def callable_from_mask(
    starts: np.ndarray, ends: np.ndarray, mask_intervals: np.ndarray
) -> np.ndarray:
    """Compute exact callable bases per window without expanding long intervals."""
    starts = np.asarray(starts, dtype=np.int64)
    ends = np.asarray(ends, dtype=np.int64)
    mask = np.asarray(mask_intervals, dtype=np.int64).reshape(-1, 2)
    widths = ends - starts
    masked = np.zeros(len(starts), dtype=np.int64)
    if len(mask) == 0:
        return widths
    first = np.searchsorted(ends, mask[:, 0], side="right")
    last = np.searchsorted(starts, mask[:, 1], side="left") - 1
    same = first == last
    np.add.at(masked, first[same], mask[same, 1] - mask[same, 0])
    spanning = ~same
    if np.any(spanning):
        first_span = first[spanning]
        last_span = last[spanning]
        intervals = mask[spanning]
        np.add.at(masked, first_span, ends[first_span] - intervals[:, 0])
        np.add.at(masked, last_span, intervals[:, 1] - starts[last_span])
        difference = np.zeros(len(starts) + 1, dtype=np.int64)
        np.add.at(difference, first_span + 1, 1)
        np.add.at(difference, last_span, -1)
        masked += np.cumsum(difference[:-1]) * widths
    return widths - masked


@dataclass(frozen=True)
class ChromosomeTarget:
    chromosome: str
    population: str
    length_bp: int
    window_size: int
    starts: np.ndarray
    ends: np.ndarray
    callable_bp: np.ndarray
    theta: np.ndarray
    mask_intervals: np.ndarray

    def validate(self) -> None:
        n = len(self.theta)
        if self.window_size <= 0 or self.length_bp <= 0 or n == 0:
            raise ValueError("map geometry must be positive")
        for name, array in (
            ("starts", self.starts),
            ("ends", self.ends),
            ("callable_bp", self.callable_bp),
        ):
            if np.asarray(array).shape != (n,):
                raise ValueError(f"{name} does not match theta length")
        if self.starts[0] != 0 or self.ends[-1] != self.length_bp:
            raise ValueError("windows do not span the full chromosome")
        if not np.array_equal(self.starts[1:], self.ends[:-1]):
            raise ValueError("map windows are not contiguous")
        widths = self.ends - self.starts
        if np.any(widths <= 0) or np.any(widths > self.window_size):
            raise ValueError("invalid map window widths")
        expected_starts = np.arange(n, dtype=np.int64) * self.window_size
        expected_ends = np.minimum(expected_starts + self.window_size, self.length_bp)
        if not np.array_equal(self.starts, expected_starts) or not np.array_equal(
            self.ends, expected_ends
        ):
            raise ValueError("map windows are not a uniform fixed-width tiling")
        if np.any(self.theta < 0) or np.any(self.callable_bp < 0):
            raise ValueError("theta and callable bases must be nonnegative")
        if np.any(self.callable_bp > widths):
            raise ValueError("callable bases exceed window widths")
        if np.any(self.theta > self.callable_bp):
            raise ValueError("segregating-site target exceeds callable positions")
        mask = np.asarray(self.mask_intervals)
        if mask.size:
            if mask.ndim != 2 or mask.shape[1] != 2:
                raise ValueError("mask intervals must have shape (n, 2)")
            if np.any(mask[:, 0] < 0) or np.any(mask[:, 1] > self.length_bp):
                raise ValueError("mask interval lies outside the chromosome")
            if np.any(mask[:, 1] <= mask[:, 0]):
                raise ValueError("mask intervals must have positive span")
            if len(mask) > 1 and np.any(mask[1:, 0] < mask[:-1, 1]):
                raise ValueError("mask intervals must be sorted and nonoverlapping")
        expected_callable = callable_from_mask(self.starts, self.ends, mask)
        if not np.array_equal(self.callable_bp, expected_callable):
            raise ValueError("callable bases disagree with the embedded mask")


def populations(path: Path | str) -> list[str]:
    with h5py.File(path, "r") as handle:
        schema = str(handle.attrs.get("schema", ""))
        if schema == SCHEMA and not bool(handle.attrs.get("complete", False)):
            raise ValueError(f"map is not marked complete: {path}")
        if "populations" in handle:
            return [
                item.decode() if isinstance(item, bytes) else str(item)
                for item in handle["populations"][...]
            ]
        chrom = next((key for key in handle if key.lower().startswith("chr")), None)
        if chrom is None:
            return []
        return [key.upper() for key in handle[chrom] if isinstance(handle[chrom][key], h5py.Group)]


def _population_group(group: h5py.Group, population: str) -> h5py.Group:
    for candidate in (population, population.upper(), population.lower()):
        if candidate in group and isinstance(group[candidate], h5py.Group):
            return group[candidate]
    raise KeyError(f"population {population!r} is absent from {group.name}")


def load_target(
    path: Path | str,
    chromosome: str | int,
    population: str,
    *,
    legacy_mask: np.ndarray | None = None,
) -> ChromosomeTarget:
    """Load one population/chromosome target, including legacy map support."""
    chromosome = canonical_chrom(chromosome)
    population = population.upper()
    with h5py.File(path, "r") as handle:
        if chromosome not in handle:
            raise KeyError(f"{chromosome} is absent from {path}")
        group = handle[chromosome]
        pop_group = _population_group(group, population)
        theta = np.asarray(pop_group["theta"], dtype=np.int64)
        if "window_starts" in group and "window_ends" in group:
            starts = np.asarray(group["window_starts"], dtype=np.int64)
            ends = np.asarray(group["window_ends"], dtype=np.int64)
            length_bp = int(group.attrs.get("length_bp", ends[-1]))
            inferred_window = int(ends[0] - starts[0])
        else:
            window_size = int(handle.attrs["window_size"])
            length_bp = int(group.attrs["length_bp"])
            starts = np.arange(len(theta), dtype=np.int64) * window_size
            ends = np.minimum(starts + window_size, length_bp)
            inferred_window = window_size
        window_size = int(handle.attrs.get("window_size", inferred_window))
        if "callable_bp" in group:
            callable_bp = np.asarray(group["callable_bp"], dtype=np.int64)
        else:
            callable_bp = ends - starts
        if "mask_intervals" in group:
            mask = np.asarray(group["mask_intervals"], dtype=np.int64).reshape(-1, 2)
        elif legacy_mask is not None:
            mask = np.asarray(legacy_mask, dtype=np.int64).reshape(-1, 2)
        else:
            mask = np.empty((0, 2), dtype=np.int64)
    target = ChromosomeTarget(
        chromosome=chromosome,
        population=population,
        length_bp=length_bp,
        window_size=window_size,
        starts=starts,
        ends=ends,
        callable_bp=callable_bp,
        theta=theta,
        mask_intervals=mask,
    )
    target.validate()
    return target


def map_sample_counts(path: Path | str) -> dict[str, int]:
    with h5py.File(path, "r") as handle:
        if "samples" not in handle:
            return {}
        result = {}
        for pop, dataset in handle["samples"].items():
            count = int(dataset.attrs.get("count", len(dataset)))
            if count != len(dataset):
                raise ValueError(f"sample count attribute disagrees with {dataset.name}")
            result[pop.upper()] = count
        if "sample_counts" in handle:
            populations_in_file = [
                value.decode() if isinstance(value, bytes) else str(value)
                for value in handle["populations"][...]
            ]
            declared = np.asarray(handle["sample_counts"], dtype=np.int64)
            observed = np.asarray([result[pop.upper()] for pop in populations_in_file])
            if not np.array_equal(declared, observed):
                raise ValueError("root sample_counts disagrees with samples datasets")
        return result


def iter_targets(
    path: Path | str, chromosomes: Iterable[str], pops: Iterable[str]
) -> Iterable[ChromosomeTarget]:
    for population in pops:
        for chromosome in chromosomes:
            yield load_target(path, chromosome, population)
