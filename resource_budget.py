"""CPU-affinity aware resource checks for local and Slurm workflows."""

from __future__ import annotations

import os
from collections.abc import Mapping


def available_cpu_count(environ: Mapping[str, str] | None = None) -> int:
    """Return the CPUs available to this task, respecting Slurm and affinity."""
    environment = os.environ if environ is None else environ
    candidates: list[int] = []
    slurm_cpus = environment.get("SLURM_CPUS_PER_TASK", "").strip()
    if slurm_cpus:
        try:
            parsed = int(slurm_cpus)
        except ValueError:
            parsed = 0
        if parsed > 0:
            candidates.append(parsed)

    get_affinity = getattr(os, "sched_getaffinity", None)
    if get_affinity is not None:
        try:
            affinity_count = len(get_affinity(0))
        except OSError:
            affinity_count = 0
        if affinity_count > 0:
            candidates.append(affinity_count)

    system_count = os.cpu_count() or 1
    candidates.append(system_count)
    return max(1, min(candidates))


def cpu_resource_plan(
    *,
    workers: int,
    threads_per_worker: int = 1,
    producer_slots_per_worker: int = 0,
    reserved_cpus: int = 0,
    available_cpus: int | None = None,
    allow_oversubscription: bool = False,
    label: str,
) -> dict[str, int | bool | str]:
    """Validate native worker demand and report the full pipeline upper bound.

    ``producer_slots_per_worker`` is informational rather than part of the hard
    limit. A streaming producer and its consumer can make progress on one CPU,
    but provisioning both slots generally gives better throughput.
    """
    if workers <= 0 or threads_per_worker <= 0:
        raise ValueError("workers and threads per worker must be positive")
    if producer_slots_per_worker < 0 or reserved_cpus < 0:
        raise ValueError("producer slots and reserved CPUs must be nonnegative")
    available = available_cpu_count() if available_cpus is None else int(available_cpus)
    if available <= 0:
        raise ValueError("available CPUs must be positive")
    usable = max(0, available - reserved_cpus)
    native_demand = workers * threads_per_worker
    producer_demand = workers * producer_slots_per_worker
    pipeline_upper_bound = reserved_cpus + native_demand + producer_demand
    oversubscribed = native_demand > usable
    if oversubscribed and not allow_oversubscription:
        raise ValueError(
            f"CPU oversubscription for {label}: {workers} workers x "
            f"{threads_per_worker} native threads = {native_demand} CPUs, but only "
            f"{usable} of {available} are unreserved (reserved={reserved_cpus}). "
            "Request more CPUs or lower the worker/thread counts. Use "
            "--allow-cpu-oversubscription only for an intentional diagnostic run."
        )
    return {
        "label": label,
        "available_cpus": available,
        "reserved_cpus": reserved_cpus,
        "usable_cpus": usable,
        "workers": workers,
        "threads_per_worker": threads_per_worker,
        "native_cpu_demand": native_demand,
        "producer_slots_per_worker": producer_slots_per_worker,
        "producer_cpu_demand": producer_demand,
        "pipeline_cpu_upper_bound": pipeline_upper_bound,
        "native_oversubscribed": oversubscribed,
        "pipeline_upper_bound_exceeds_available": pipeline_upper_bound > available,
        "oversubscription_allowed": allow_oversubscription,
    }
