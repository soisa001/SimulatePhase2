#!/usr/bin/env python3
"""Build a compact per-population 10 kb SNV segregating-site target map.

The expensive BCF pass is streaming and occurs once per chromosome for all
populations.  ``bcftools +fill-tags`` computes population-specific AC/AN in C;
an awk reducer keeps memory proportional to the number of output windows, not
the number of variants or genotypes.  Per-chromosome checkpoints make the run
restartable and the final HDF5 file is written atomically.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

from phase2_map import DEFAULT_POPS, SCHEMA, callable_from_mask, canonical_chrom, parse_chroms

DEFAULT_BCF_TEMPLATE = (
    "gs://rw-long-reads-transfer-2026-06-17/v9/lrWGS/panel/panel/"
    "panel_bubble_split_vcf/aou_lr_phase2_v1.{chrom}.bubble.split.bcf"
)
DEFAULT_ANCESTRY = (
    "gs://vwb-aou-datasets-controlled/v9/wgs/short_read/snpindel/aux/ancestry/ancestry_preds.tsv"
)
DEFAULT_FLAGGED = (
    "gs://vwb-aou-datasets-controlled/v9/wgs/short_read/snpindel/aux/qc/flagged_samples.tsv"
)
DEFAULT_RELATED = (
    "gs://vwb-aou-datasets-controlled/v9/wgs/short_read/snpindel/aux/"
    "relatedness/relatedness_flagged_samples.tsv"
)
DEFAULT_HARDMASK = "gs://rw-migration-aou-rw-fa99430f/hardmask.hg38.v4.over99.bed"
DEFAULT_SAMPLES_PER_POPULATION = 224
DEFAULT_SAMPLE_SELECTION_SEED = 42
SAMPLE_SELECTION_ALGORITHM = "sha256-rank-v1"
COUNTER_VERSION = "group-source-coordinates-before-snv-filter/v4-call-rate"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


@contextmanager
def advisory_lock(path: Path):
    """Serialize launchers that share a work directory or final artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    try:
        if os.name == "nt":
            import msvcrt

            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _gcloud_prefix(executable: str, billing_project: str | None) -> list[str]:
    command = [executable]
    if billing_project:
        command.append(f"--billing-project={billing_project}")
    return command


def source_identity(
    source: str | Path, *, gcloud: str, billing_project: str | None
) -> dict[str, object]:
    """Return a stable cache/checkpoint identity without reading a whole BCF."""
    source = str(source)
    if not source.startswith("gs://"):
        path = Path(source).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        stat = path.stat()
        return {
            "source": str(path),
            "kind": "local",
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    command = _gcloud_prefix(gcloud, billing_project) + [
        "storage",
        "objects",
        "describe",
        source,
        "--format=json",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    metadata = json.loads(result.stdout)
    identity: dict[str, object] = {"source": source, "kind": "gcs"}
    for key in (
        "generation",
        "metageneration",
        "size",
        "crc32c",
        "crc32c_hash",
        "md5Hash",
        "md5_hash",
        "etag",
    ):
        if key in metadata and metadata[key] not in (None, ""):
            identity[key] = metadata[key]
    if "generation" not in identity or "size" not in identity:
        raise RuntimeError(f"gcloud did not return generation and size for {source}")
    return identity


def cache_destination(source: str | Path, cache_dir: Path, identity: dict[str, object]) -> Path:
    source = str(source)
    stem = Path(source).name
    key = sha256_bytes(json.dumps(identity, sort_keys=True).encode())[:16]
    return cache_dir / f"{key}.{stem}"


def localize(
    source: str | Path,
    cache_dir: Path,
    *,
    gcloud: str,
    billing_project: str | None,
    identity: dict[str, object] | None = None,
) -> tuple[Path, bool]:
    """Return a local path and whether this call downloaded it."""
    source = str(source)
    if not source.startswith("gs://"):
        path = Path(source).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path, False
    cache_dir.mkdir(parents=True, exist_ok=True)
    identity = identity or source_identity(source, gcloud=gcloud, billing_project=billing_project)
    destination = cache_destination(source, cache_dir, identity)
    expected_size = int(identity["size"])
    if destination.is_file() and destination.stat().st_size == expected_size:
        return destination, False
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.part")
    temporary.unlink(missing_ok=True)
    # Pin the read to the generation that was described above.  Copying the
    # live URI would allow an object replaced between describe and cp (even by
    # an equal-sized payload) to be cached under the old generation's key.
    generation = str(identity["generation"])
    generation_suffix = re.search(r"#(\d+)$", source)
    download_source = (
        source
        if generation_suffix is not None and generation_suffix.group(1) == generation
        else f"{source}#{generation}"
    )
    command = _gcloud_prefix(gcloud, billing_project) + [
        "storage",
        "cp",
        "--quiet",
        download_source,
        str(temporary),
    ]
    try:
        subprocess.run(command, check=True)
        if not temporary.is_file() or temporary.stat().st_size != expected_size:
            observed_size = temporary.stat().st_size if temporary.exists() else "missing"
            raise RuntimeError(
                f"download size differs from GCS metadata for {source}: "
                f"{observed_size} != {expected_size}"
            )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination, True


def render_bcf(template: str, chromosome: str) -> str:
    number = canonical_chrom(chromosome)[3:]
    if "{" not in template:
        return template
    return template.format(chrom=f"chr{number}", n=number)


@contextmanager
def stream_delimited(path: Path) -> Iterator[tuple[list[str], Iterator[list[str]]]]:
    """Yield a header and rows without retaining wide auxiliary-table fields."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(16_384)
        handle.seek(0)
        header = next((line for line in sample.splitlines() if line.strip()), "")
        if "\t" in header:
            dialect = csv.excel_tab
        elif "," in header:
            dialect = csv.excel
        else:
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters="\t,")
            except csv.Error:
                dialect = csv.excel_tab
        reader = csv.reader(handle, dialect=dialect)
        header_row = next((row for row in reader if any(value.strip() for value in row)), None)
        if not header_row:
            raise ValueError(f"{path} has no header")
        columns = [str(value).strip() for value in header_row]
        if len(columns) != len(set(columns)):
            raise ValueError(f"{path} has duplicate column names after trimming whitespace")
        rows = (row for row in reader if any(value.strip() for value in row))
        yield columns, rows


def row_value(row: list[str], index: int) -> str:
    return row[index].strip() if index < len(row) else ""


def read_delimited(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Compatibility reader; production gating uses ``stream_delimited``."""
    with stream_delimited(path) as (columns, rows):
        materialized = [
            {column: row_value(row, index) for index, column in enumerate(columns)} for row in rows
        ]
    return columns, materialized


def first_column(columns: Iterable[str], choices: Iterable[str], path: Path) -> str:
    for choice in choices:
        if choice in columns:
            return choice
    raise ValueError(f"{path} needs one of these columns: {', '.join(choices)}")


def read_ids(path: Path) -> list[str]:
    lines = [
        line.strip().split()[0]
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if lines and lines[0].lower() in {
        "sample_id",
        "research_id",
        "person_id",
        "sample",
        "sid",
        "s",
    }:
        lines = lines[1:]
    return list(dict.fromkeys(lines))


def load_manifest(path: Path, wanted: tuple[str, ...]) -> dict[str, list[str]]:
    result = {pop: [] for pop in wanted}
    assignments: dict[str, str] = {}
    with stream_delimited(path) as (columns, rows):
        sample_column = first_column(
            columns, ("sample_id", "research_id", "person_id", "sample", "sid", "s"), path
        )
        population_column = first_column(
            columns,
            ("population", "pop", "ancestry", "ancestry_pred", "ancestry_pred_other"),
            path,
        )
        sample_index = columns.index(sample_column)
        population_index = columns.index(population_column)
        for row in rows:
            sample = row_value(row, sample_index)
            pop = row_value(row, population_index).upper()
            if not sample or pop not in result:
                continue
            previous = assignments.get(sample)
            if previous is not None and previous != pop:
                raise ValueError(
                    f"{path}: sample {sample} is assigned to both {previous} and {pop}"
                )
            if previous is None:
                result[pop].append(sample)
                assignments[sample] = pop
    return result


def build_gated_panel(
    ordered_panel: list[str],
    ancestry_path: Path,
    flagged_path: Path | None,
    related_path: Path | None,
    wanted: tuple[str, ...],
    *,
    ancestry_column: str,
    ancestry_id_column: str,
) -> tuple[dict[str, list[str]], dict[str, int]]:
    labels: dict[str, str] = {}
    with stream_delimited(ancestry_path) as (columns, rows):
        choices = (
            ("s", "research_id", "sample_id", "person_id", "sample", "sid")
            if ancestry_id_column == "auto"
            else (ancestry_id_column,)
        )
        sample_column = first_column(columns, choices, ancestry_path)
        pop_column = first_column(columns, (ancestry_column,), ancestry_path)
        sample_index = columns.index(sample_column)
        pop_index = columns.index(pop_column)
        for row in rows:
            sample = row_value(row, sample_index)
            label = row_value(row, pop_index).upper()
            if label == "ANCESTRY_PRED_OTHER":
                label = "OTH"
            if not sample:
                continue
            if sample in labels and labels[sample] != label:
                raise ValueError(
                    f"{ancestry_path}: sample {sample} has conflicting ancestry labels "
                    f"{labels[sample]!r} and {label!r}"
                )
            labels.setdefault(sample, label)

    excluded_by_source: dict[str, set[str]] = {"flagged": set(), "related": set()}
    for name, path in (("flagged", flagged_path), ("related", related_path)):
        if path is None:
            continue
        with stream_delimited(path) as (exc_columns, exc_rows):
            exc_column = first_column(
                exc_columns,
                ("research_id", "sample_id", "person_id", "sample", "sid", "s"),
                path,
            )
            exc_index = exc_columns.index(exc_column)
            excluded_by_source[name] = {
                value for row in exc_rows if (value := row_value(row, exc_index))
            }
    excluded = excluded_by_source["flagged"] | excluded_by_source["related"]
    result = {
        pop: [
            sample
            for sample in ordered_panel
            if sample not in excluded and labels.get(sample) == pop
        ]
        for pop in wanted
    }
    stats = {
        "panel": len(ordered_panel),
        "ancestry_ids": len(labels),
        "missing_ancestry": sum(sample not in labels for sample in ordered_panel),
        "other_or_unrequested": sum(
            labels.get(sample) not in wanted for sample in ordered_panel if sample in labels
        ),
        "flagged": len(set(ordered_panel) & excluded_by_source["flagged"]),
        "related": len(set(ordered_panel) & excluded_by_source["related"]),
        "excluded_unique": len(set(ordered_panel) & excluded),
        "retained": sum(map(len, result.values())),
    }
    return result, stats


def select_population_samples(
    samples: dict[str, list[str]],
    samples_per_population: int,
    seed: int,
) -> tuple[dict[str, list[str]], dict[str, int]]:
    """Select a stable pseudo-random subset while preserving panel order.

    SHA256 ranking avoids Python- or NumPy-RNG version dependence. Population
    and sample ID are both part of the rank, so selecting a population alone
    gives the same IDs as selecting it alongside the other populations.
    ``samples_per_population=0`` retains every gated sample.
    """
    if samples_per_population < 0:
        raise ValueError("samples per population must be nonnegative")
    if seed < 0:
        raise ValueError("sample-selection seed must be nonnegative")

    eligible = {population: len(ids) for population, ids in samples.items()}
    if samples_per_population == 0:
        selected = {population: list(ids) for population, ids in samples.items()}
    else:
        insufficient = {
            population: count
            for population, count in eligible.items()
            if count < samples_per_population
        }
        if insufficient:
            detail = ", ".join(
                f"{population}={count:,}" for population, count in insufficient.items()
            )
            raise ValueError(f"fewer than {samples_per_population:,} QC-gated samples: {detail}")
        selected = {}
        for population, ids in samples.items():
            ranked = sorted(
                ids,
                key=lambda sample: hashlib.sha256(
                    (f"{SAMPLE_SELECTION_ALGORITHM}\0{seed}\0{population}\0{sample}").encode()
                ).digest(),
            )
            keep = set(ranked[:samples_per_population])
            selected[population] = [sample for sample in ids if sample in keep]

    retained = sum(map(len, selected.values()))
    stats = {
        "retained_after_qc": sum(eligible.values()),
        "removed_by_population_subsample": sum(eligible.values()) - retained,
        "samples_per_population": samples_per_population,
        "sample_selection_seed": seed,
        "retained": retained,
    }
    stats.update({f"eligible_{population}": count for population, count in eligible.items()})
    return selected, stats


CONTIG_RE = re.compile(r"^##contig=<(.+)>$")


def bcf_header(path: Path, bcftools: str) -> tuple[dict[str, int], list[str], str]:
    result = subprocess.run(
        [bcftools, "view", "--no-version", "-h", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    lengths: dict[str, int] = {}
    samples: list[str] = []
    for line in result.stdout.splitlines():
        match = CONTIG_RE.match(line)
        if match:
            fields = {}
            for item in match.group(1).split(","):
                key, _, value = item.partition("=")
                fields[key] = value.strip('"')
            if "ID" in fields and fields.get("length", "").isdigit():
                lengths[fields["ID"]] = int(fields["length"])
        elif line.startswith("#CHROM\t"):
            samples = line.split("\t")[9:]
    if not samples:
        raise RuntimeError(f"BCF header has no samples: {path}")
    return lengths, samples, sha256_bytes(result.stdout.encode())


def bcftools_version(executable: str) -> str:
    result = subprocess.run([executable, "--version"], check=True, capture_output=True, text=True)
    return result.stdout.splitlines()[0].strip()


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
        raise RuntimeError(
            "hard-mask intervals exist for the contig but none overlap its BCF length; "
            "check the assembly and contig coordinates"
        )
    return result


def window_geometry(
    length_bp: int, window_size: int, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    starts = np.arange(0, length_bp, window_size, dtype=np.int64)
    ends = np.minimum(starts + window_size, length_bp)
    return starts, ends, callable_from_mask(starts, ends, mask)


def write_normalized_mask(path: Path, contig: str, mask: np.ndarray) -> None:
    text = "".join(f"{contig}\t{int(start)}\t{int(end)}\n" for start, end in mask)
    atomic_text(path, text)


def callable_intervals(length_bp: int, mask: np.ndarray) -> np.ndarray:
    result: list[tuple[int, int]] = []
    cursor = 0
    for start, end in mask:
        if int(start) > cursor:
            result.append((cursor, int(start)))
        cursor = int(end)
    if cursor < length_bp:
        result.append((cursor, length_bp))
    return np.asarray(result, dtype=np.int64).reshape(-1, 2)


def awk_reducer(pop_count: int) -> str:
    # Records sharing a coordinate are excluded wholesale.  This removes both
    # duplicated records and split representations of multiallelic sites.
    return r"""
BEGIN {
  OFS="\t"
  nexpected=split(EXPECTED_AN, expected_an, ",")
  nminimum=split(MIN_AN, minimum_an, ",")
  if (nexpected != NP || nminimum != NP) {
    print "EXPECTED_AN and MIN_AN must have one value per population" > "/dev/stderr"
    exit 43
  }
  if (FILTERS != "") {
    nallowed=split(FILTERS, value, ",")
    for (i=1; i<=nallowed; i++) allowed[value[i]]=1
  }
}
function filter_ok( n,i,value) {
  if (FILTERS == "") return 1
  n=split(filt, value, ";")
  for (i=1; i<=n; i++) if (value[i] in allowed) return 1
  return 0
}
function capture( p) {
  ref=$2; alt=$3; filt=$4
  eligible=(ref ~ /^[ACGT]$/ && alt ~ /^[ACGT]$/ && filter_ok())
  for (p=1; p<=NP; p++) { ac[p]=$(2*p+3); an[p]=$(2*p+4) }
}
function emit( w,p) {
  if (!have) return
  npos++
  if (dup) { nduppos++; return }
  if (!eligible) return
  neligible++
  w=int(pos/W)
  if (w < 0 || w >= NW) return
  for (p=1; p<=NP; p++) {
    if (an[p] > 0 && ac[p] > 0 && ac[p] < an[p]) {
      if (an[p] >= minimum_an[p]) count[p,w]++
      else nlowan[p]++
    }
  }
}
{
  nrec++
  if (!have) { pos=$1; capture(); have=1; next }
  if ($1 < pos) { print "input positions are not sorted" > "/dev/stderr"; exit 42 }
  if ($1 == pos) { dup=1; next }
  emit(); pos=$1; dup=0; capture()
}
END {
  emit()
  print "#QC", nrec+0, npos+0, nduppos+0, neligible+0
  printf "#LOWAN"
  for (p=1; p<=NP; p++) printf "\t%d", nlowan[p]+0
  printf "\n"
  for (w=0; w<NW; w++) {
    printf "%d", w
    for (p=1; p<=NP; p++) printf "\t%d", count[p,w]+0
    printf "\n"
  }
}
""".strip()


def run_count_pipeline(
    *,
    bcf: Path,
    samples_file: Path,
    groups_file: Path,
    callable_bed: Path,
    populations: tuple[str, ...],
    sample_counts: dict[str, int],
    min_call_rate: float,
    length_bp: int,
    window_size: int,
    filters: str,
    bcftools: str,
    awk: str,
    threads: int,
    log_path: Path,
) -> tuple[np.ndarray, dict[str, object]]:
    n_windows = (length_bp + window_size - 1) // window_size
    expected_an = [2 * int(sample_counts[pop]) for pop in populations]
    minimum_an = [math.ceil(value * min_call_rate) for value in expected_an]
    view = [
        bcftools,
        "view",
        "--threads",
        str(threads),
        "-Ou",
        "-S",
        str(samples_file),
    ]
    # Keep every source record until the reducer has grouped by coordinate.
    # Otherwise a SNP sibling of an indel, multiallelic, or filtered record can
    # survive a bcftools pre-filter and be misclassified as a singleton SNV.
    # A positive callable-target BED restricts the expected contig and excludes
    # the exact embedded hard mask.
    # Avoid --targets-overlap here because it is unavailable in bcftools 1.13,
    # which is still installed on some AoU Workbench images.
    view.extend(["-T", str(callable_bed)])
    view.append(str(bcf))
    fill = [
        bcftools,
        "+fill-tags",
        "-Ou",
        "--",
        "-S",
        str(groups_file),
        "-t",
        "AC,AN",
    ]
    fields = ["%POS0", "%REF", "%ALT", "%FILTER"]
    for pop in populations:
        fields.extend((f"%AC_{pop}{{0}}", f"%AN_{pop}"))
    query = [bcftools, "query", "-f", "\t".join(fields) + "\n"]
    reduce_command = [
        awk,
        "-v",
        f"W={window_size}",
        "-v",
        f"NW={n_windows}",
        "-v",
        f"NP={len(populations)}",
        "-v",
        f"FILTERS={filters}",
        "-v",
        "EXPECTED_AN=" + ",".join(map(str, expected_an)),
        "-v",
        "MIN_AN=" + ",".join(map(str, minimum_an)),
        awk_reducer(len(populations)),
    ]

    log_path.parent.mkdir(parents=True, exist_ok=True)
    processes: list[subprocess.Popen] = []
    with log_path.open("wb") as log:
        try:
            first = subprocess.Popen(view, stdout=subprocess.PIPE, stderr=log)
            processes.append(first)
            second = subprocess.Popen(fill, stdin=first.stdout, stdout=subprocess.PIPE, stderr=log)
            first.stdout.close()
            processes.append(second)
            third = subprocess.Popen(query, stdin=second.stdout, stdout=subprocess.PIPE, stderr=log)
            second.stdout.close()
            processes.append(third)
            fourth = subprocess.Popen(
                reduce_command,
                stdin=third.stdout,
                stdout=subprocess.PIPE,
                stderr=log,
                text=True,
            )
            third.stdout.close()
            processes.append(fourth)
            output, _ = fourth.communicate()
            returncodes = [process.wait() for process in processes]
        except BaseException:
            for process in processes:
                process.terminate()
            for process in processes:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
            raise
    if any(returncodes):
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-8000:]
        raise RuntimeError(
            f"variant counting pipeline failed with codes {returncodes} for {bcf}\n{tail}"
        )
    lines = output.splitlines()
    if len(lines) < 2 or not lines[0].startswith("#QC\t") or not lines[1].startswith("#LOWAN\t"):
        raise RuntimeError(f"counter produced malformed output for {bcf}")
    _, records, positions, duplicate_positions, eligible_positions = lines[0].split("\t")
    low_an_fields = lines[1].split("\t")[1:]
    if len(low_an_fields) != len(populations):
        raise RuntimeError(f"counter produced malformed low-AN QC for {bcf}")
    low_an = [int(value) for value in low_an_fields]
    matrix = np.loadtxt(io.StringIO("\n".join(lines[2:])), delimiter="\t", dtype=np.int64)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    if matrix.shape != (n_windows, len(populations) + 1):
        raise RuntimeError(
            f"counter produced {matrix.shape}, expected {(n_windows, len(populations) + 1)}"
        )
    if not np.array_equal(matrix[:, 0], np.arange(n_windows)):
        raise RuntimeError("counter output windows are not consecutive")
    qc = {
        "source_records_in_callable_intervals": int(records),
        "distinct_source_positions": int(positions),
        "duplicate_or_multiallelic_positions_excluded": int(duplicate_positions),
        "eligible_singleton_biallelic_snv_positions": int(eligible_positions),
        "expected_diploid_an": dict(zip(populations, expected_an, strict=True)),
        "minimum_an": dict(zip(populations, minimum_an, strict=True)),
        "min_call_rate": float(min_call_rate),
        "segregating_positions_excluded_low_an": dict(zip(populations, low_an, strict=True)),
    }
    return matrix[:, 1:].T, qc


def checkpoint_valid(path: Path, key: str, population_count: int) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            required = {
                "checkpoint_key",
                "counter_version",
                "chromosome",
                "source",
                "source_identity_json",
                "source_local_size",
                "header_sha256",
                "contig",
                "length_bp",
                "starts",
                "ends",
                "callable_bp",
                "mask_intervals",
                "theta",
                "min_call_rate",
                "expected_an",
                "minimum_an",
                "qc_json",
            }
            if not required <= set(data.files):
                return False
            starts = np.asarray(data["starts"], dtype=np.int64)
            ends = np.asarray(data["ends"], dtype=np.int64)
            callable_bp = np.asarray(data["callable_bp"], dtype=np.int64)
            theta = np.asarray(data["theta"], dtype=np.int64)
            mask = np.asarray(data["mask_intervals"], dtype=np.int64)
            min_call_rate = float(data["min_call_rate"])
            expected_an = np.asarray(data["expected_an"], dtype=np.int64)
            minimum_an = np.asarray(data["minimum_an"], dtype=np.int64)
            length_bp = int(data["length_bp"])
            geometry_valid = bool(
                len(starts) > 0
                and starts[0] == 0
                and ends[-1] == length_bp
                and np.array_equal(starts[1:], ends[:-1])
                and np.all(ends > starts)
            )
            mask_valid = bool(
                mask.ndim == 2
                and mask.shape[1] == 2
                and np.all(mask[:, 0] >= 0)
                and np.all(mask[:, 1] <= length_bp)
                and np.all(mask[:, 1] > mask[:, 0])
                and (len(mask) < 2 or np.all(mask[1:, 0] >= mask[:-1, 1]))
            )
            return bool(
                str(data["checkpoint_key"]) == key
                and str(data["counter_version"]) == COUNTER_VERSION
                and starts.ndim == 1
                and ends.shape == starts.shape
                and callable_bp.shape == starts.shape
                and theta.shape == (population_count, len(starts))
                and 0.0 <= min_call_rate <= 1.0
                and expected_an.shape == (population_count,)
                and minimum_an.shape == (population_count,)
                and np.all(expected_an > 0)
                and np.array_equal(minimum_an, np.ceil(expected_an * min_call_rate))
                and geometry_valid
                and mask_valid
                and np.array_equal(callable_bp, callable_from_mask(starts, ends, mask))
                and np.all(theta >= 0)
                and np.all(theta <= callable_bp)
            )
    except Exception:
        return False


def write_checkpoint(path: Path, **arrays: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def sample_hash(samples: dict[str, list[str]]) -> str:
    payload = "".join(f"{sample}\t{pop}\n" for pop in sorted(samples) for sample in samples[pop])
    return sha256_bytes(payload.encode())


def compact_unsigned(values: np.ndarray) -> np.ndarray:
    """Use uint16 for 10 kb maps and grow safely for wider custom windows."""
    values = np.asarray(values)
    maximum = int(values.max(initial=0))
    dtype = np.uint16 if maximum <= np.iinfo(np.uint16).max else np.uint32
    if maximum > np.iinfo(np.uint32).max or np.any(values < 0):
        raise ValueError("map counts cannot be represented as an unsigned 32-bit integer")
    return values.astype(dtype)


def process_chromosome(
    chromosome: str,
    *,
    args: argparse.Namespace,
    source_identity_value: dict[str, object],
    populations: tuple[str, ...],
    samples: dict[str, list[str]],
    sample_digest: str,
    mask_by_chrom: dict[str, np.ndarray],
    mask_digest: str,
    samples_file: Path,
    groups_file: Path,
) -> Path:
    source = render_bcf(args.bcf_template, chromosome)
    identity = dict(source_identity_value)
    base_key = sha256_bytes(
        json.dumps(
            {
                "schema": SCHEMA,
                "counter_version": COUNTER_VERSION,
                "bcftools_version": args.bcftools_version,
                "chromosome": chromosome,
                "source_identity": identity,
                "window_size": args.window_size,
                "filters": args.filters,
                "min_call_rate": args.min_call_rate,
                "samples": sample_digest,
                "mask": mask_digest,
                "populations": populations,
            },
            sort_keys=True,
        ).encode()
    )
    checkpoint = args.work_dir / "chromosomes" / f"{chromosome}.npz"
    cache_dir = args.cache_dir / "bcf"
    cloud_source = source.startswith("gs://")
    destination = cache_destination(source, cache_dir, identity) if cloud_source else None
    if not args.fresh and checkpoint_valid(checkpoint, base_key, len(populations)):
        # A previous process may have been killed after publishing its atomic
        # checkpoint but before deleting the localized BCF.  A resumed
        # --delete-localized run should still enforce its disk-bound contract.
        if args.delete_localized and destination is not None:
            cache_lock = destination.with_name(destination.name + ".lock")
            with advisory_lock(cache_lock):
                destination.unlink(missing_ok=True)
        print(f"[{chromosome}] checkpoint present", flush=True)
        return checkpoint

    lease = (
        advisory_lock(destination.with_name(destination.name + ".lock"))
        if destination is not None
        else nullcontext()
    )
    lease.__enter__()
    bcf: Path | None = None
    try:
        bcf, _ = localize(
            source,
            cache_dir,
            gcloud=args.gcloud,
            billing_project=args.billing_project,
            identity=identity,
        )
        lengths, header_samples, header_digest = bcf_header(bcf, args.bcftools)
        wanted_set = {sample for ids in samples.values() for sample in ids}
        missing = sorted(wanted_set - set(header_samples))
        if missing:
            raise RuntimeError(
                f"{chromosome}: {len(missing)} selected samples are absent from the BCF; "
                f"first: {missing[:5]}"
            )
        actual_contig = chromosome if chromosome in lengths else None
        if actual_contig is None:
            bare = chromosome[3:]
            if bare in lengths:
                actual_contig = bare
            else:
                raise RuntimeError(
                    f"cannot resolve {chromosome} in BCF contigs {sorted(lengths)[:8]}"
                )
        length_bp = lengths[actual_contig]
        mask = clip_merged_mask(mask_by_chrom[chromosome], length_bp)
        starts, ends, callable_bp = window_geometry(length_bp, args.window_size, mask)
        normalized_mask = args.work_dir / "masks" / f"{chromosome}.masked.bed"
        write_normalized_mask(normalized_mask, actual_contig, mask)
        callable_bed = args.work_dir / "masks" / f"{chromosome}.callable.bed"
        write_normalized_mask(callable_bed, actual_contig, callable_intervals(length_bp, mask))
        print(
            f"[{chromosome}] counting {len(header_samples):,} BCF samples -> "
            f"{len(starts):,} windows ({callable_bp.sum():,} callable bp)",
            flush=True,
        )
        if int(callable_bp.sum()) == 0:
            # bcftools rejects an empty -T file.  A completely hard-masked
            # contig nevertheless has a well-defined all-zero target map.
            theta = np.zeros((len(populations), len(starts)), dtype=np.int64)
            qc = {
                "source_records_in_callable_intervals": 0,
                "distinct_source_positions": 0,
                "duplicate_or_multiallelic_positions_excluded": 0,
                "eligible_singleton_biallelic_snv_positions": 0,
                "expected_diploid_an": {pop: 2 * len(samples[pop]) for pop in populations},
                "minimum_an": {
                    pop: math.ceil(2 * len(samples[pop]) * args.min_call_rate)
                    for pop in populations
                },
                "min_call_rate": float(args.min_call_rate),
                "segregating_positions_excluded_low_an": {pop: 0 for pop in populations},
            }
            print(f"[{chromosome}] fully masked; variant pipeline skipped", flush=True)
        else:
            theta, qc = run_count_pipeline(
                bcf=bcf,
                samples_file=samples_file,
                groups_file=groups_file,
                callable_bed=callable_bed,
                populations=populations,
                sample_counts={pop: len(samples[pop]) for pop in populations},
                min_call_rate=args.min_call_rate,
                length_bp=length_bp,
                window_size=args.window_size,
                filters=args.filters,
                bcftools=args.bcftools,
                awk=args.awk,
                threads=args.threads,
                log_path=args.work_dir / "logs" / f"{chromosome}.log",
            )
        if np.any(theta < 0) or np.any(theta > callable_bp[None, :]):
            where = np.argwhere(theta > callable_bp[None, :])[:5].tolist()
            raise RuntimeError(f"{chromosome}: theta exceeds callable positions at {where}")
        write_checkpoint(
            checkpoint,
            checkpoint_key=np.asarray(base_key),
            chromosome=np.asarray(chromosome),
            source=np.asarray(source),
            source_identity_json=np.asarray(json.dumps(identity, sort_keys=True)),
            counter_version=np.asarray(COUNTER_VERSION),
            source_local_size=np.asarray(bcf.stat().st_size, dtype=np.int64),
            header_sha256=np.asarray(header_digest),
            contig=np.asarray(actual_contig),
            length_bp=np.asarray(length_bp, dtype=np.int64),
            starts=starts.astype(np.int32),
            ends=ends.astype(np.int32),
            callable_bp=compact_unsigned(callable_bp),
            mask_intervals=mask.astype(np.int32),
            theta=compact_unsigned(theta),
            min_call_rate=np.asarray(args.min_call_rate, dtype=np.float64),
            expected_an=np.asarray([2 * len(samples[pop]) for pop in populations], dtype=np.uint32),
            minimum_an=np.asarray(
                [math.ceil(2 * len(samples[pop]) * args.min_call_rate) for pop in populations],
                dtype=np.uint32,
            ),
            qc_json=np.asarray(json.dumps(qc, sort_keys=True)),
        )
        totals = ", ".join(
            f"{pop}={int(theta[index].sum()):,}" for index, pop in enumerate(populations)
        )
        print(f"[{chromosome}] done: {totals}", flush=True)
        return checkpoint
    finally:
        if args.delete_localized and cloud_source and bcf is not None:
            bcf.unlink(missing_ok=True)
        lease.__exit__(None, None, None)


def fixed_ascii(values: Iterable[str]) -> np.ndarray:
    encoded = [str(value).encode("utf-8") for value in values]
    width = max((len(value) for value in encoded), default=1)
    return np.asarray(encoded, dtype=f"S{width}")


def write_hdf5(
    output: Path,
    checkpoints: dict[str, Path],
    *,
    args: argparse.Namespace,
    populations: tuple[str, ...],
    samples: dict[str, list[str]],
    sample_digest: str,
    mask_digest: str,
    selection_stats: dict[str, int],
    input_provenance: dict[str, object] | None = None,
) -> dict[str, object]:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    totals = {pop: 0 for pop in populations}
    chromosome_stats: dict[str, dict[str, object]] = {}
    expected_an = {pop: 2 * len(samples[pop]) for pop in populations}
    minimum_an = {pop: math.ceil(expected_an[pop] * args.min_call_rate) for pop in populations}
    try:
        with h5py.File(temporary, "w") as handle:
            handle.attrs.update(
                {
                    "schema": SCHEMA,
                    "complete": False,
                    "assembly": "GRCh38",
                    "window_size": args.window_size,
                    "coordinate_system": "0-based half-open windows; BCF POS converted with %POS0",
                    "theta_definition": (
                        "number of distinct unmasked biallelic SNV positions with 0 < AC < AN "
                        "and AN >= ceil(2 * diploid sample count * min_call_rate) within the "
                        "named population"
                    ),
                    "min_call_rate": float(args.min_call_rate),
                    "callability_policy": (
                        "population AN must meet the recorded minimum_an; 0.0 preserves the "
                        "literal any-called segregating-site definition"
                    ),
                    "expected_diploid_an_json": json.dumps(expected_an, sort_keys=True),
                    "minimum_an_json": json.dumps(minimum_an, sort_keys=True),
                    "bcf_template": args.bcf_template,
                    "bcf_filters": args.filters,
                    "bcftools_version": args.bcftools_version,
                    "sample_manifest_sha256": sample_digest,
                    "hardmask_sha256": mask_digest,
                    "selection_json": json.dumps(selection_stats, sort_keys=True),
                    "input_provenance_json": json.dumps(input_provenance or {}, sort_keys=True),
                    "counter_version": COUNTER_VERSION,
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                }
            )
            handle.create_dataset("populations", data=fixed_ascii(populations))
            handle.create_dataset(
                "sample_counts",
                data=np.asarray([len(samples[pop]) for pop in populations], dtype=np.uint32),
            )
            sample_group = handle.create_group("samples")
            for pop in populations:
                values = fixed_ascii(samples[pop])
                dataset = sample_group.create_dataset(
                    pop,
                    data=values,
                    compression="gzip",
                    compression_opts=6,
                    shuffle=True,
                )
                dataset.attrs["count"] = len(values)
                dataset.attrs["sha256"] = sha256_bytes(b"\n".join(values.tolist()) + b"\n")

            for chromosome, checkpoint in checkpoints.items():
                with np.load(checkpoint, allow_pickle=False) as data:
                    group = handle.create_group(chromosome)
                    group.attrs.update(
                        {
                            "contig_in_bcf": str(data["contig"]),
                            "length_bp": int(data["length_bp"]),
                            "source_bcf": str(data["source"]),
                            "source_identity_json": str(data["source_identity_json"]),
                            "source_local_size": int(data["source_local_size"]),
                            "bcf_header_sha256": str(data["header_sha256"]),
                        }
                    )
                    common = dict(
                        compression="gzip",
                        compression_opts=6,
                        shuffle=True,
                        fletcher32=True,
                    )
                    group.create_dataset("window_starts", data=data["starts"], **common)
                    group.create_dataset("window_ends", data=data["ends"], **common)
                    group.create_dataset("callable_bp", data=data["callable_bp"], **common)
                    mask_intervals = np.asarray(data["mask_intervals"])
                    if len(mask_intervals):
                        group.create_dataset("mask_intervals", data=mask_intervals, **common)
                    else:
                        group.create_dataset("mask_intervals", data=mask_intervals)
                    theta = np.asarray(data["theta"])
                    checkpoint_expected = np.asarray(data["expected_an"], dtype=np.int64)
                    checkpoint_minimum = np.asarray(data["minimum_an"], dtype=np.int64)
                    declared_expected = np.asarray(
                        [expected_an[pop] for pop in populations], dtype=np.int64
                    )
                    declared_minimum = np.asarray(
                        [minimum_an[pop] for pop in populations], dtype=np.int64
                    )
                    if (
                        float(data["min_call_rate"]) != float(args.min_call_rate)
                        or not np.array_equal(checkpoint_expected, declared_expected)
                        or not np.array_equal(checkpoint_minimum, declared_minimum)
                    ):
                        raise ValueError(f"call-rate policy mismatch in {checkpoint}")
                    qc = json.loads(str(data["qc_json"]))
                    group.attrs["qc_json"] = json.dumps(qc, sort_keys=True)
                    for index, pop in enumerate(populations):
                        pop_group = group.create_group(pop.lower())
                        dataset = pop_group.create_dataset("theta", data=theta[index], **common)
                        dataset.attrs["sample_count"] = len(samples[pop])
                        dataset.attrs["expected_an"] = expected_an[pop]
                        dataset.attrs["minimum_an"] = minimum_an[pop]
                        dataset.attrs["min_call_rate"] = float(args.min_call_rate)
                        dataset.attrs["segregating_positions_excluded_low_an"] = int(
                            qc["segregating_positions_excluded_low_an"][pop]
                        )
                        dataset.attrs["units"] = "segregating SNV sites per window"
                        totals[pop] += int(theta[index].sum())
                    chromosome_stats[chromosome] = {
                        "length_bp": int(data["length_bp"]),
                        "windows": int(theta.shape[1]),
                        "callable_bp": int(np.asarray(data["callable_bp"], dtype=np.int64).sum()),
                        "source_identity": json.loads(str(data["source_identity_json"])),
                        "qc": qc,
                    }
            handle.attrs["complete"] = True
            handle.flush()
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    digest = sha256_file(output)
    summary: dict[str, object] = {
        "schema": SCHEMA,
        "artifact": str(output.resolve()),
        "sha256": digest,
        "size_bytes": output.stat().st_size,
        "window_size": args.window_size,
        "populations": list(populations),
        "sample_counts": {pop: len(samples[pop]) for pop in populations},
        "callability_policy": {
            "min_call_rate": float(args.min_call_rate),
            "expected_diploid_an": expected_an,
            "minimum_an": minimum_an,
            "segregating_definition": "0 < AC < AN and AN >= minimum_an",
        },
        "selection": selection_stats,
        "input_provenance": input_provenance or {},
        "theta_totals": totals,
        "chromosomes": chromosome_stats,
    }
    atomic_text(
        output.with_suffix(output.suffix + ".json"),
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    atomic_text(output.with_suffix(output.suffix + ".sha256"), f"{digest}  {output.name}\n")
    return summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--bcf-template", default=DEFAULT_BCF_TEMPLATE)
    result.add_argument("--chroms", default="1-22")
    result.add_argument("--output", type=Path, default=Path("data/snv_theta_map.10kb.h5"))
    result.add_argument("--work-dir", type=Path, default=Path("mutation_map_work"))
    result.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache/simulatephase2")
    result.add_argument("--window-size", type=int, default=10_000)
    result.add_argument("--population", dest="populations", action="append")
    result.add_argument("--sample-manifest", type=Path)
    result.add_argument(
        "--samples-per-population",
        "--samples-per-pop",
        type=int,
        default=DEFAULT_SAMPLES_PER_POPULATION,
        help=(
            "Deterministically select this many QC-gated diploids per population "
            f"[{DEFAULT_SAMPLES_PER_POPULATION}]; 0 retains the full gated panel. "
            "Ignored when --sample-manifest supplies the exact samples."
        ),
    )
    result.add_argument(
        "--sample-selection-seed",
        type=int,
        default=DEFAULT_SAMPLE_SELECTION_SEED,
        help=(
            "Seed for stable SHA256-ranked population subsampling "
            f"[{DEFAULT_SAMPLE_SELECTION_SEED}]"
        ),
    )
    result.add_argument(
        "--panel-samples", help="Optional ordered one-column panel; default is first BCF header"
    )
    result.add_argument("--ancestry-table", default=DEFAULT_ANCESTRY)
    result.add_argument("--ancestry-column", default="ancestry_pred_other")
    result.add_argument("--ancestry-id-column", default="auto")
    result.add_argument("--flagged-samples-table", default=DEFAULT_FLAGGED)
    result.add_argument("--relatedness-table", default=DEFAULT_RELATED)
    mask = result.add_mutually_exclusive_group()
    mask.add_argument(
        "--hardmask",
        default=DEFAULT_HARDMASK,
        help=f"BED of sites to exclude and embed [{DEFAULT_HARDMASK}]",
    )
    mask.add_argument(
        "--no-mask",
        action="store_true",
        help="Explicitly build an unmasked map (cannot be combined with --hardmask)",
    )
    result.add_argument(
        "--filters", default="PASS,.", help="BCF FILTER values to retain; empty keeps all"
    )
    result.add_argument(
        "--min-call-rate",
        type=float,
        default=0.0,
        help=(
            "Minimum called-allele fraction per population at a segregating site [0.0]; "
            "AN must be at least ceil(2 * diploid samples * rate)"
        ),
    )
    result.add_argument("--jobs", type=int, default=4, help="Chromosome pipelines in flight")
    result.add_argument(
        "--threads", type=int, default=2, help="bcftools threads per chromosome pipeline"
    )
    result.add_argument("--bcftools", default="bcftools")
    result.add_argument("--awk", default="awk")
    result.add_argument("--gcloud", default="gcloud")
    result.add_argument("--billing-project")
    result.add_argument("--delete-localized", action="store_true")
    result.add_argument("--fresh", action="store_true", help="Ignore per-chromosome checkpoints")
    result.add_argument("--upload", help="Optional gs:// destination for the HDF5 and sidecars")
    return result


def _run(args: argparse.Namespace) -> int:
    if args.window_size <= 0 or args.jobs <= 0 or args.threads <= 0:
        raise SystemExit("--window-size, --jobs, and --threads must be positive")
    if not 0.0 <= args.min_call_rate <= 1.0:
        raise SystemExit("--min-call-rate must be between 0 and 1")
    if args.samples_per_population < 0:
        raise SystemExit("--samples-per-population must be nonnegative")
    if args.sample_selection_seed < 0:
        raise SystemExit("--sample-selection-seed must be nonnegative")
    if args.no_mask:
        args.hardmask = None
    for executable in (args.bcftools, args.awk):
        if shutil.which(executable) is None:
            raise SystemExit(f"required executable not found: {executable}")
    cloud_sources = [args.bcf_template, args.hardmask or ""]
    if args.sample_manifest is None:
        cloud_sources.extend(
            (
                args.panel_samples or "",
                args.ancestry_table,
                args.flagged_samples_table or "",
                args.relatedness_table or "",
            )
        )
    if (
        any(str(source).startswith("gs://") for source in cloud_sources)
        and shutil.which(args.gcloud) is None
    ):
        raise SystemExit(f"required executable not found for gs:// input: {args.gcloud}")

    chromosomes = parse_chroms(args.chroms)
    if len(chromosomes) > 1 and "{" not in args.bcf_template:
        raise SystemExit(
            "--bcf-template must contain {chrom} or {n} when multiple chromosomes are requested"
        )
    populations = tuple(pop.upper() for pop in (args.populations or DEFAULT_POPS))
    unknown = sorted(set(populations) - set(DEFAULT_POPS))
    if unknown or len(set(populations)) != len(populations):
        raise SystemExit(f"invalid or duplicate populations: {unknown or populations}")
    args.work_dir = args.work_dir.expanduser().resolve()
    args.cache_dir = args.cache_dir.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.bcftools_version = bcftools_version(args.bcftools)
    print(f"{args.bcftools_version}; jobs={args.jobs}; threads/job={args.threads}", flush=True)

    input_provenance: dict[str, object] = {}
    # Describe each input BCF exactly once.  Workers download the recorded GCS
    # generation, so header discovery and counting operate on one deliberate
    # per-run snapshot even if a live object is replaced while the run is active.
    source_identities: dict[str, dict[str, object]] = {}
    for chromosome in chromosomes:
        source = render_bcf(args.bcf_template, chromosome)
        source_identities[chromosome] = source_identity(
            source, gcloud=args.gcloud, billing_project=args.billing_project
        )
    first_source = render_bcf(args.bcf_template, chromosomes[0])
    first_identity = source_identities[chromosomes[0]]
    first_cache = args.cache_dir / "bcf"
    first_destination = (
        cache_destination(first_source, first_cache, first_identity)
        if first_source.startswith("gs://")
        else None
    )
    first_lease = (
        advisory_lock(first_destination.with_name(first_destination.name + ".lock"))
        if first_destination is not None
        else nullcontext()
    )
    with first_lease:
        first_bcf, _ = localize(
            first_source,
            first_cache,
            gcloud=args.gcloud,
            billing_project=args.billing_project,
            identity=first_identity,
        )
        _, first_header_samples, _ = bcf_header(first_bcf, args.bcftools)
    input_provenance["first_bcf"] = first_identity

    aux_cache = args.cache_dir / "aux"

    def localize_aux(label: str, source: str | Path) -> Path:
        identity = source_identity(source, gcloud=args.gcloud, billing_project=args.billing_project)
        path, _ = localize(
            source,
            aux_cache,
            gcloud=args.gcloud,
            billing_project=args.billing_project,
            identity=identity,
        )
        input_provenance[label] = {
            "identity": identity,
            "sha256": sha256_file(path),
        }
        return path

    selection_stats: dict[str, int]
    if args.sample_manifest:
        manifest_path = args.sample_manifest.expanduser().resolve()
        samples = load_manifest(manifest_path, populations)
        input_provenance["sample_manifest"] = {
            "identity": source_identity(
                manifest_path, gcloud=args.gcloud, billing_project=args.billing_project
            ),
            "sha256": sha256_file(manifest_path),
        }
        selection_stats = {
            "manifest_rows_retained": sum(map(len, samples.values())),
            "manifest_is_trusted_as_pre_gated": 1,
            "retained": sum(map(len, samples.values())),
        }
        input_provenance["population_subsample"] = {
            "mode": "explicit sample manifest; no automatic subsampling"
        }
    else:
        if args.panel_samples:
            panel_path = localize_aux("panel_samples", args.panel_samples)
            ordered_panel = read_ids(panel_path)
        else:
            ordered_panel = first_header_samples
            input_provenance["panel_samples"] = {"mode": "ordered first BCF header"}
        ancestry_path = localize_aux("ancestry_table", args.ancestry_table)
        flagged_path = None
        if args.flagged_samples_table:
            flagged_path = localize_aux("flagged_samples_table", args.flagged_samples_table)
        related_path = None
        if args.relatedness_table:
            related_path = localize_aux("relatedness_table", args.relatedness_table)
        samples, selection_stats = build_gated_panel(
            ordered_panel,
            ancestry_path,
            flagged_path,
            related_path,
            populations,
            ancestry_column=args.ancestry_column,
            ancestry_id_column=args.ancestry_id_column,
        )
        try:
            samples, subset_stats = select_population_samples(
                samples,
                args.samples_per_population,
                args.sample_selection_seed,
            )
        except ValueError as error:
            raise SystemExit(str(error)) from error
        selection_stats.update(subset_stats)
        input_provenance["population_subsample"] = {
            "mode": (
                "all QC-gated samples"
                if args.samples_per_population == 0
                else "deterministic pseudo-random population subset"
            ),
            "algorithm": SAMPLE_SELECTION_ALGORITHM,
            "seed": args.sample_selection_seed,
            "samples_per_population": args.samples_per_population,
        }
    empty = [pop for pop in populations if not samples.get(pop)]
    if empty:
        raise SystemExit(f"no retained samples for populations: {empty}")
    all_samples = [sample for pop in populations for sample in samples[pop]]
    if len(all_samples) != len(set(all_samples)):
        raise SystemExit("a sample was assigned to more than one population")
    absent_first = sorted(set(all_samples) - set(first_header_samples))
    if absent_first:
        raise SystemExit(
            f"{len(absent_first)} selected samples are absent from the first BCF; "
            f"first: {absent_first[:5]}"
        )

    samples_file = args.work_dir / "samples.used.txt"
    groups_file = args.work_dir / "sample_groups.tsv"
    manifest_file = args.work_dir / "sample_manifest.tsv"
    atomic_text(samples_file, "\n".join(all_samples) + "\n")
    atomic_text(
        groups_file,
        "".join(f"{sample}\t{pop}\n" for pop in populations for sample in samples[pop]),
    )
    atomic_text(
        manifest_file,
        "sample_id\tpopulation\n"
        + "".join(f"{sample}\t{pop}\n" for pop in populations for sample in samples[pop]),
    )
    digest = sample_hash(samples)
    print("sample selection:", flush=True)
    for key, value in selection_stats.items():
        print(f"  {key}: {value:,}", flush=True)
    print(
        "  retained: " + ", ".join(f"{pop}={len(samples[pop]):,}" for pop in populations),
        flush=True,
    )

    mask_path = None
    if args.hardmask:
        mask_path = localize_aux("hardmask", args.hardmask)
    else:
        input_provenance["hardmask"] = {"mode": "none"}
    mask_digest = sha256_file(mask_path) if mask_path else sha256_bytes(b"NO_MASK")
    mask_by_chrom = load_mask(mask_path, chromosomes)
    if mask_path:
        empty_mask = [
            chromosome for chromosome in chromosomes if len(mask_by_chrom[chromosome]) == 0
        ]
        if empty_mask:
            raise RuntimeError(
                f"hard mask has no intervals for requested chromosomes: {empty_mask}; "
                "check contig naming or use --no-mask explicitly"
            )

    checkpoints: dict[str, Path] = {}
    try:
        with ThreadPoolExecutor(max_workers=min(args.jobs, len(chromosomes))) as executor:
            futures = {
                executor.submit(
                    process_chromosome,
                    chromosome,
                    args=args,
                    source_identity_value=source_identities[chromosome],
                    populations=populations,
                    samples=samples,
                    sample_digest=digest,
                    mask_by_chrom=mask_by_chrom,
                    mask_digest=mask_digest,
                    samples_file=samples_file,
                    groups_file=groups_file,
                ): chromosome
                for chromosome in chromosomes
            }
            for future in as_completed(futures):
                chromosome = futures[future]
                checkpoints[chromosome] = future.result()
    finally:
        # The first BCF is localized for panel discovery before checkpoint
        # inspection.  On a fully resumed run no chromosome worker otherwise
        # reaches its deletion block, so clean this copy under the same lease.
        if args.delete_localized and first_destination is not None:
            first_lock = first_destination.with_name(first_destination.name + ".lock")
            with advisory_lock(first_lock):
                first_destination.unlink(missing_ok=True)
    checkpoints = {chrom: checkpoints[chrom] for chrom in chromosomes}
    artifact_lock = args.output.with_suffix(args.output.suffix + ".lock")
    with advisory_lock(artifact_lock):
        summary = write_hdf5(
            args.output,
            checkpoints,
            args=args,
            populations=populations,
            samples=samples,
            sample_digest=digest,
            mask_digest=mask_digest,
            selection_stats=selection_stats,
            input_provenance=input_provenance,
        )
        print(
            f"wrote {args.output} ({summary['size_bytes'] / 1e6:.2f} MB)\n"
            f"sha256 {summary['sha256']}",
            flush=True,
        )
        if args.upload:
            if not args.upload.startswith("gs://"):
                raise SystemExit("--upload must be a gs:// destination")
            for artifact in (
                args.output,
                args.output.with_suffix(args.output.suffix + ".json"),
                args.output.with_suffix(args.output.suffix + ".sha256"),
            ):
                command = _gcloud_prefix(args.gcloud, args.billing_project) + [
                    "storage",
                    "cp",
                    "--quiet",
                    str(artifact),
                    args.upload.rstrip("/") + "/",
                ]
                subprocess.run(command, check=True)
            print(f"uploaded map and sidecars to {args.upload.rstrip('/')}/", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    args.work_dir = args.work_dir.expanduser().resolve()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    lock = args.work_dir / ".generate_map.lock"
    print(f"acquiring run lock: {lock}", flush=True)
    with advisory_lock(lock):
        return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
