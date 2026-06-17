"""
Regression tests for concurrency-limiter enforcement in workflow execution.

Covers:
- run_workflow_once (manual) rejects tasks when limiter is full
- Scheduled _run_workflow rejects tasks when limiter is full
- Rejected tasks written to DB as failed with correct reason
- No slot leaked or double-released after rejection
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.secuscan.ratelimit import ConcurrentTaskLimiter, concurrent_limiter
from backend.secuscan.workflows import WorkflowScheduler


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def run(coro):
    """Run a coroutine synchronously."""
    return asyncio.run(coro)


def fill_limiter():
    """Fill all slots in the global concurrent_limiter."""
    async def _fill():
        async with concurrent_limiter.lock:
            for i in range(concurrent_limiter.max_concurrent):
                concurrent_limiter.running_tasks.append(f"dummy-{i}")
    run(_fill())


def drain_limiter():
    """Release all dummy slots added by fill_limiter."""
    async def _drain():
        async with concurrent_limiter.lock:
            concurrent_limiter.running_tasks.clear()
    run(_drain())


def get_slot_count():
    """Return how many slots are currently available."""
    return run(concurrent_limiter.get_available_slots())


# ---------------------------------------------------------------------------
# Helpers to seed a workflow in DB via the API
# ---------------------------------------------------------------------------

def create_workflow(client, name="Test Workflow"):
    resp = client.post(
        "/api/v1/workflows",
        json={
            "name": name,
            "steps": [
                {
                    "plugin_id": "http_inspector",
                    "inputs": {"url": "http://127.0.0.1"},
                }
            ],
        },
    )
    assert resp.status_code == 200, f"Failed to create workflow: {resp.text}"
    return resp.json()["id"]


# ===========================================================================
# 1.  run_workflow_once — manual API trigger
# ===========================================================================

def test_run_workflow_once_does_not_schedule_when_limiter_full(test_client):
    """
    POST /api/v1/workflow/{id}/run must not schedule any tasks
    when all concurrency slots are taken.
    """
    fill_limiter()
    scheduled = []

    workflow_id = create_workflow(test_client)

    try:
        with patch("asyncio.create_task", side_effect=lambda c, **kw: scheduled.append(c)):
            resp = test_client.post(f"/api/v1/workflow/{workflow_id}/run")

        assert resp.status_code == 200
        data = resp.json()
        assert data["queued_tasks"] == [], (
            f"Expected no queued tasks, got: {data['queued_tasks']}"
        )
        assert len(scheduled) == 0, (
            "asyncio.create_task was called despite limiter being full"
        )
    finally:
        drain_limiter()


def test_run_workflow_once_marks_task_failed_in_db(test_client):
    """
    When the limiter is full, the rejected task must exist in the DB
    with status='failed' and error_message containing 'Concurrency limit reached'.
    """
    from backend.secuscan.database import get_db

    fill_limiter()
    workflow_id = create_workflow(test_client, name="DB Status Test")

    try:
        test_client.post(f"/api/v1/workflow/{workflow_id}/run")

        async def check():
            db = await get_db()
            return await db.fetchone(
                "SELECT status, error_message FROM tasks ORDER BY created_at DESC LIMIT 1"
            )

        row = run(check())
        assert row is not None, "No task record found in DB"
        assert row["status"] == "failed", (
            f"Expected status=failed, got: {row['status']}"
        )
        assert "Concurrency limit reached" in (row["error_message"] or ""), (
            f"Unexpected error_message: {row['error_message']}"
        )
    finally:
        drain_limiter()


def test_run_workflow_once_no_slot_leak(test_client):
    """
    Slot count must be identical before and after a rejection.
    No phantom acquire or double-release.
    """
    fill_limiter()
    workflow_id = create_workflow(test_client, name="Slot Leak Test")

    try:
        slots_before = get_slot_count()
        test_client.post(f"/api/v1/workflow/{workflow_id}/run")
        slots_after = get_slot_count()

        assert slots_after == slots_before, (
            f"Slot count changed: before={slots_before}, after={slots_after}"
        )
    finally:
        drain_limiter()


# ===========================================================================
# 2.  WorkflowScheduler._run_workflow — scheduled execution
# ===========================================================================

@pytest.mark.asyncio
async def test_scheduled_workflow_does_not_schedule_when_limiter_full():
    """
    _run_workflow must not call asyncio.create_task when limiter is full.
    """
    limiter = ConcurrentTaskLimiter(max_concurrent=2)
    async with limiter.lock:
        limiter.running_tasks.extend(["dummy-0", "dummy-1"])

    mock_executor = MagicMock()
    mock_executor.create_task = AsyncMock(return_value="task-sched-001")
    mock_executor.mark_task_failed = AsyncMock()

    scheduled = []
    steps = [{"plugin_id": "http_inspector", "inputs": {"url": "http://127.0.0.1"}}]

    with patch("backend.secuscan.workflows.executor", mock_executor), \
         patch("backend.secuscan.workflows.concurrent_limiter", limiter), \
         patch("asyncio.create_task", side_effect=lambda c, **kw: scheduled.append(c)):

        await WorkflowScheduler()._run_workflow("wf-001", steps)

    assert len(scheduled) == 0, (
        "asyncio.create_task was called despite limiter being full"
    )


@pytest.mark.asyncio
async def test_scheduled_workflow_calls_mark_task_failed_with_correct_args():
    """
    When the limiter is full, mark_task_failed must be called
    with the correct task_id and reason='Concurrency limit reached'.
    """
    limiter = ConcurrentTaskLimiter(max_concurrent=1)
    async with limiter.lock:
        limiter.running_tasks.append("dummy-0")

    fake_task_id = "task-sched-002"
    mock_executor = MagicMock()
    mock_executor.create_task = AsyncMock(return_value=fake_task_id)
    mock_executor.mark_task_failed = AsyncMock()

    steps = [{"plugin_id": "http_inspector", "inputs": {"url": "http://127.0.0.1"}}]

    with patch("backend.secuscan.workflows.executor", mock_executor), \
         patch("backend.secuscan.workflows.concurrent_limiter", limiter), \
         patch("asyncio.create_task"):

        await WorkflowScheduler()._run_workflow("wf-002", steps)

    mock_executor.mark_task_failed.assert_awaited_once_with(
        fake_task_id, reason="Concurrency limit reached"
    )


@pytest.mark.asyncio
async def test_scheduled_workflow_no_slot_leak():
    """
    Available slot count must be unchanged after a rejection.
    """
    limiter = ConcurrentTaskLimiter(max_concurrent=2)
    async with limiter.lock:
        limiter.running_tasks.extend(["dummy-0", "dummy-1"])

    slots_before = await limiter.get_available_slots()

    mock_executor = MagicMock()
    mock_executor.create_task = AsyncMock(return_value="task-leak")
    mock_executor.mark_task_failed = AsyncMock()

    steps = [{"plugin_id": "http_inspector", "inputs": {"url": "http://127.0.0.1"}}]

    with patch("backend.secuscan.workflows.executor", mock_executor), \
         patch("backend.secuscan.workflows.concurrent_limiter", limiter), \
         patch("asyncio.create_task"):

        await WorkflowScheduler()._run_workflow("wf-003", steps)

    slots_after = await limiter.get_available_slots()
    assert slots_after == slots_before, (
        f"Slot leaked: before={slots_before}, after={slots_after}"
    )


@pytest.mark.asyncio
async def test_scheduled_workflow_partial_run_when_limiter_fills_mid_workflow():
    """
    If limiter has 1 slot and workflow has 2 steps:
    - Step 1 should be scheduled (acquires the slot)
    - Step 2 should be marked failed (no slot left)
    """
    limiter = ConcurrentTaskLimiter(max_concurrent=1)

    task_ids = ["task-step-1", "task-step-2"]
    call_index = 0

    async def fake_create_task(plugin_id, inputs, preset=None, consent_granted=False):
        nonlocal call_index
        tid = task_ids[call_index]
        call_index += 1
        return tid

    mock_executor = MagicMock()
    mock_executor.create_task = AsyncMock(side_effect=fake_create_task)
    mock_executor.mark_task_failed = AsyncMock()

    scheduled = []
    steps = [
        {"plugin_id": "http_inspector", "inputs": {"url": "http://127.0.0.1"}},
        {"plugin_id": "http_inspector", "inputs": {"url": "http://127.0.0.2"}},
    ]

    with patch("backend.secuscan.workflows.executor", mock_executor), \
         patch("backend.secuscan.workflows.concurrent_limiter", limiter), \
         patch("asyncio.create_task", side_effect=lambda c, **kw: scheduled.append(c)):

        await WorkflowScheduler()._run_workflow("wf-partial", steps)

    # Step 1 acquired the only slot — must be scheduled
    assert len(scheduled) == 1, f"Expected 1 scheduled task, got {len(scheduled)}"

    # Step 2 had no slot — must be marked failed
    mock_executor.mark_task_failed.assert_awaited_once_with(
        "task-step-2", reason="Concurrency limit reached"
    )

    # The 1 slot is held by step-1, so 0 available
    assert await limiter.get_available_slots() == 0
