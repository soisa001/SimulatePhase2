from __future__ import annotations

import pytest

import resource_budget


def test_slurm_cpu_limit_takes_precedence_over_system_count(monkeypatch) -> None:
    monkeypatch.setattr(resource_budget.os, "cpu_count", lambda: 64)
    monkeypatch.setattr(
        resource_budget.os, "sched_getaffinity", lambda _pid: set(range(32)), raising=False
    )
    assert resource_budget.available_cpu_count({"SLURM_CPUS_PER_TASK": "8"}) == 8


def test_decode_plan_accounts_for_reserved_cpus_and_streaming_producers() -> None:
    plan = resource_budget.cpu_resource_plan(
        workers=10,
        threads_per_worker=4,
        producer_slots_per_worker=1,
        reserved_cpus=45,
        available_cpus=100,
        label="test decode",
    )
    assert plan["native_cpu_demand"] == 40
    assert plan["producer_cpu_demand"] == 10
    assert plan["pipeline_cpu_upper_bound"] == 95
    assert plan["native_oversubscribed"] is False


def test_oversubscribed_native_workers_fail_closed() -> None:
    with pytest.raises(ValueError, match=r"10 workers x 10 native threads = 100 CPUs"):
        resource_budget.cpu_resource_plan(
            workers=10,
            threads_per_worker=10,
            available_cpus=1,
            label="test decode",
        )


def test_intentional_oversubscription_is_recorded() -> None:
    plan = resource_budget.cpu_resource_plan(
        workers=4,
        available_cpus=1,
        allow_oversubscription=True,
        label="diagnostic",
    )
    assert plan["native_oversubscribed"] is True
    assert plan["oversubscription_allowed"] is True
