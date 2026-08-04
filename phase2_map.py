"""Shared geometry, mask, and compact-map helpers for SimulatePhase2."""

from __future__ import annotations

import gzip
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

SCHEMA = "simulatephase2.snv-count-map/v2"
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


def merge_intervals(intervals: Iterable[tuple[int, int]], length_bp: int) -> np.ndarray:
    clipped = sorted(
        (max(0, int(start)), min(length_bp, int(end)))
        for start, end in intervals
        if int(end) > 0 and int(start) < length_bp and int(end) > int(start)
    )
    merged: list[list[int]] = []
    for start, end in clipped:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return np.asarray(merged, dtype=np.int64).reshape(-1, 2)


def load_mask(path: Path | None, chromosomes: Iterable[str]) -> dict[str, np.ndarray]:
    """Stream a BED/BED.gz once and merge intervals for requested chromosomes."""
    wanted = {canonical_chrom(chrom) for chrom in chromosomes}
    buckets: dict[str, list[list[int]]] = {chrom: [] for chrom in wanted}
    if path is None:
        return {chrom: np.empty((0, 2), dtype=np.int64) for chrom in wanted}
    handle = (
        gzip.open(path, "rt", encoding="utf-8")
        if path.suffix.lower() == ".gz"
        else path.open("r", encoding="utf-8")
    )
    with handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            fields = line.split()
            if len(fields) < 3:
                raise ValueError(f"{path}:{line_number}: expected at least three BED columns")
            chrom = canonical_chrom(fields[0])
            if chrom not in buckets:
                continue
            start, end = int(fields[1]), int(fields[2])
            if end <= start:
                raise ValueError(f"{path}:{line_number}: BED end must exceed start")
            bucket = buckets[chrom]
            if bucket and start < bucket[-1][0]:
                raise ValueError(f"{path}:{line_number}: BED must be position-sorted per contig")
            if bucket and start <= bucket[-1][1]:
                bucket[-1][1] = max(bucket[-1][1], end)
            else:
                bucket.append([start, end])
    return {
        chrom: np.asarray(intervals, dtype=np.int64).reshape(-1, 2)
        for chrom, intervals in buckets.items()
    }


def clip_merged_mask(mask: np.ndarray, length_bp: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=np.int64).reshape(-1, 2)
    if len(mask) == 0:
        return mask
    result = mask.copy()
    result[:, 0] = np.maximum(result[:, 0], 0)
    result[:, 1] = np.minimum(result[:, 1], length_bp)
    result = result[result[:, 1] > result[:, 0]]
    if len(result) == 0:
        raise ValueError("mask intervals for the contig do not overlap its map length")
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


def window_geometry(
    length_bp: int, window_size: int, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    starts = np.arange(0, length_bp, window_size, dtype=np.int64)
    ends = np.minimum(starts + window_size, length_bp)
    return starts, ends, callable_from_mask(starts, ends, mask)


def watterson_a_n(diploid_samples: int) -> float:
    """Return a_n for the expected 2N sampled haplotypes."""
    if diploid_samples <= 0:
        raise ValueError("diploid sample count must be positive")
    return math.fsum(1.0 / index for index in range(1, 2 * diploid_samples))


def populations(path: Path | str) -> list[str]:
    with h5py.File(path, "r") as handle:
        if str(handle.attrs.get("schema", "")) != SCHEMA:
            raise ValueError(f"map does not use schema {SCHEMA}: {path}")
        if not bool(handle.attrs.get("complete", False)):
            raise ValueError(f"map is not marked complete: {path}")
        return [
            item.decode() if isinstance(item, bytes) else str(item)
            for item in handle["populations"][...]
        ]


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

    @property
    def segregating_sites(self) -> np.ndarray:
        return self.theta

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
                raise ValueError(f"{name} does not match S length")
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
            raise ValueError("S and callable bases must be nonnegative")
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
        if not np.array_equal(self.callable_bp, callable_from_mask(self.starts, self.ends, mask)):
            raise ValueError("callable bases disagree with the supplied mask")


def load_target(
    path: Path | str,
    chromosome: str | int,
    population: str,
    *,
    mask_intervals: np.ndarray,
) -> ChromosomeTarget:
    """Load one compact matrix row and derive window geometry from the mask."""
    chromosome = canonical_chrom(chromosome)
    population = population.upper()
    with h5py.File(path, "r") as handle:
        if str(handle.attrs.get("schema", "")) != SCHEMA:
            raise ValueError(f"map does not use schema {SCHEMA}: {path}")
        if chromosome not in handle:
            raise KeyError(f"{chromosome} is absent from {path}")
        pop_order = [
            item.decode() if isinstance(item, bytes) else str(item)
            for item in handle["populations"][...]
        ]
        try:
            pop_index = [value.upper() for value in pop_order].index(population)
        except ValueError as error:
            raise KeyError(f"population {population!r} is absent from {path}") from error
        window_size = int(handle.attrs["window_size"])
        group = handle[chromosome]
        length_bp = int(group.attrs["length_bp"])
        n_windows = int(group.attrs["n_windows"])
        matrix = group["S"]
        if matrix.shape != (len(pop_order), n_windows):
            raise ValueError(f"invalid S matrix shape in {group.name}: {matrix.shape}")
        theta = np.asarray(matrix[pop_index], dtype=np.int64)
    mask = clip_merged_mask(mask_intervals, length_bp)
    starts, ends, callable_bp = window_geometry(length_bp, window_size, mask)
    if len(starts) != n_windows:
        raise ValueError(f"window count disagrees with chromosome length in {chromosome}")
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
    pop_order = populations(path)
    with h5py.File(path, "r") as handle:
        counts = np.asarray(handle["sample_counts"], dtype=np.int64)
        if counts.shape != (len(pop_order),) or np.any(counts <= 0):
            raise ValueError("invalid root sample_counts array")
        if "samples" in handle:
            for pop, count in zip(pop_order, counts, strict=True):
                dataset = handle["samples"][pop]
                if len(dataset) != int(count) or int(dataset.attrs.get("count", -1)) != int(count):
                    raise ValueError(f"sample count disagrees with {dataset.name}")
    return {pop.upper(): int(count) for pop, count in zip(pop_order, counts, strict=True)}


def map_watterson_a_n(path: Path | str) -> dict[str, float]:
    pop_order = populations(path)
    with h5py.File(path, "r") as handle:
        values = np.asarray(handle["watterson_a_n"], dtype=np.float64)
    if values.shape != (len(pop_order),) or np.any(~np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("invalid root watterson_a_n array")
    return {pop.upper(): float(value) for pop, value in zip(pop_order, values, strict=True)}


def iter_targets(
    path: Path | str,
    chromosomes: Iterable[str],
    pops: Iterable[str],
    *,
    mask_by_chrom: dict[str, np.ndarray],
) -> Iterable[ChromosomeTarget]:
    for population in pops:
        for chromosome in chromosomes:
            canonical = canonical_chrom(chromosome)
            yield load_target(
                path,
                canonical,
                population,
                mask_intervals=mask_by_chrom[canonical],
            )
