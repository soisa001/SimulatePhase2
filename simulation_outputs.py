"""Fast validation helpers for published simulation tree-sequence units."""

from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CompletedUnit:
    path: Path
    sidecar: Path
    signature: str
    size_bytes: int
    metadata: dict[str, object]


def canonical_chromosome(value: object) -> str:
    text = str(value).strip().lower()
    if text.startswith("chr"):
        text = text[3:]
    if not text.isdigit() or not 1 <= int(text) <= 22:
        raise ValueError(f"invalid autosome: {value}")
    return f"chr{int(text)}"


def quick_tsz_archive(path: Path) -> bool:
    """Check the TSZip ZIP central directory without decompressing tree arrays.

    TSZip is a ZIP-backed Zarr hierarchy. Opening the central directory catches
    the common interrupted/truncated-write case in essentially constant I/O,
    while the completion sidecar's byte count catches later size changes.
    """
    try:
        if not path.is_file() or path.stat().st_size <= 22 or not zipfile.is_zipfile(path):
            return False
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            names = {member.filename for member in members}
        return bool(members) and ".zgroup" in names and ".zattrs" in names
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile):
        return False


def validate_completed_unit(
    path: Path,
    *,
    population: str,
    simulation: int,
    chromosome: str,
    check_archive: bool = True,
) -> CompletedUnit:
    """Validate a unit's publication marker and optionally its TSZip footer."""
    path = path.expanduser().resolve()
    sidecar = path.with_suffix(path.suffix + ".json")
    if not path.is_file() or not sidecar.is_file():
        raise FileNotFoundError(f"missing completed simulation unit: {path}")
    try:
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception as error:
        raise ValueError(f"cannot read completion sidecar {sidecar}: {error}") from error
    if not isinstance(metadata, dict):
        raise ValueError(f"completion sidecar is not a JSON object: {sidecar}")
    contract = metadata.get("contract")
    try:
        size_bytes = int(metadata.get("size_bytes", -1))
        target_sites = int(metadata.get("target_sites", -1))
        realized_sites = int(metadata.get("realized_sites", -2))
        requested_target_sites = int(metadata.get("requested_target_sites", target_sites))
        skipped_target_sites = int(
            metadata.get("skipped_target_sites", requested_target_sites - target_sites)
        )
        contract_simulation = int(contract.get("simulation", -1))  # type: ignore[union-attr]
        contract_chromosome = canonical_chromosome(  # type: ignore[union-attr]
            contract.get("chromosome", "")
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(f"invalid completion sidecar fields: {sidecar}") from error
    signature = metadata.get("signature")
    skipped_windows = metadata.get("skipped_windows", [])
    policy_metadata_valid = isinstance(skipped_windows, list)
    if policy_metadata_valid:
        try:
            window_ids = [int(record["window"]) for record in skipped_windows]
            skipped_from_records = sum(int(record["requested_sites"]) for record in skipped_windows)
            policy_metadata_valid = (
                len(window_ids) == len(set(window_ids))
                and all(window >= 0 for window in window_ids)
                and skipped_from_records == skipped_target_sites
            )
        except (KeyError, TypeError, ValueError):
            policy_metadata_valid = False
    valid = (
        metadata.get("status") == "complete"
        and isinstance(contract, dict)
        and str(contract.get("population", "")).upper() == population.upper()
        and contract_simulation == simulation
        and contract_chromosome == canonical_chromosome(chromosome)
        and isinstance(signature, str)
        and len(signature) == 64
        and all(character in "0123456789abcdef" for character in signature.lower())
        and size_bytes == path.stat().st_size
        and target_sites >= 0
        and requested_target_sites >= target_sites
        and skipped_target_sites == requested_target_sites - target_sites
        and realized_sites == target_sites
        and policy_metadata_valid
    )
    if not valid:
        raise ValueError(f"invalid or stale completion sidecar: {sidecar}")
    if check_archive and not quick_tsz_archive(path):
        raise ValueError(f"truncated or unreadable TSZip archive: {path}")
    return CompletedUnit(path, sidecar, signature, size_bytes, metadata)


def completed_units(
    sim_dir: Path,
    population: str,
    chromosome: str,
    n_sims: int,
    *,
    check_archive: bool = True,
) -> tuple[list[Path], str]:
    """Return ordered unit paths and a content-derived completion digest."""
    chromosome = canonical_chromosome(chromosome)
    paths: list[Path] = []
    digest = hashlib.sha256()
    for simulation in range(n_sims):
        path = (
            sim_dir.expanduser().resolve()
            / population.lower()
            / f"sim_{simulation:05d}"
            / f"{chromosome}.tsz"
        )
        unit = validate_completed_unit(
            path,
            population=population,
            simulation=simulation,
            chromosome=chromosome,
            check_archive=check_archive,
        )
        digest.update(f"{simulation}\t{unit.signature}\t{unit.size_bytes}\n".encode())
        paths.append(unit.path)
    return paths, digest.hexdigest()
