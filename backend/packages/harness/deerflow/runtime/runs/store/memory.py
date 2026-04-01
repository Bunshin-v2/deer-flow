"""In-memory RunStore. Used when database.backend=memory (default) and in tests.

Equivalent to the original RunManager._runs dict behavior.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from deerflow.runtime.runs.store.base import RunStore


class MemoryRunStore(RunStore):
    def __init__(self) -> None:
        self._runs: dict[str, dict[str, Any]] = {}

    async def put(
        self,
        run_id,
        *,
        thread_id,
        assistant_id=None,
        owner_id=None,
        status="pending",
        multitask_strategy="reject",
        metadata=None,
        kwargs=None,
        error=None,
        created_at=None,
    ):
        now = datetime.now(UTC).isoformat()
        self._runs[run_id] = {
            "run_id": run_id,
            "thread_id": thread_id,
            "assistant_id": assistant_id,
            "owner_id": owner_id,
            "status": status,
            "multitask_strategy": multitask_strategy,
            "metadata": metadata or {},
            "kwargs": kwargs or {},
            "error": error,
            "created_at": created_at or now,
            "updated_at": now,
        }

    async def get(self, run_id):
        return self._runs.get(run_id)

    async def list_by_thread(self, thread_id, *, owner_id=None, limit=100):
        results = [r for r in self._runs.values() if r["thread_id"] == thread_id and (owner_id is None or r.get("owner_id") == owner_id)]
        results.sort(key=lambda r: r["created_at"], reverse=True)
        return results[:limit]

    async def update_status(self, run_id, status, *, error=None):
        if run_id in self._runs:
            self._runs[run_id]["status"] = status
            if error is not None:
                self._runs[run_id]["error"] = error
            self._runs[run_id]["updated_at"] = datetime.now(UTC).isoformat()

    async def delete(self, run_id):
        self._runs.pop(run_id, None)

    async def list_pending(self, *, before=None):
        now = before or datetime.now(UTC).isoformat()
        results = [r for r in self._runs.values() if r["status"] == "pending" and r["created_at"] <= now]
        results.sort(key=lambda r: r["created_at"])
        return results
