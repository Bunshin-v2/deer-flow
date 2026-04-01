"""Abstract interface for run metadata storage.

RunManager depends on this interface. Implementations:
- MemoryRunStore: in-memory dict (development, tests)
- Future: RunRepository backed by SQLAlchemy ORM

All methods accept an optional owner_id for user isolation.
When owner_id is None, no user filtering is applied (single-user mode).
"""

from __future__ import annotations

import abc
from typing import Any


class RunStore(abc.ABC):
    @abc.abstractmethod
    async def put(
        self,
        run_id: str,
        *,
        thread_id: str,
        assistant_id: str | None = None,
        owner_id: str | None = None,
        status: str = "pending",
        multitask_strategy: str = "reject",
        metadata: dict[str, Any] | None = None,
        kwargs: dict[str, Any] | None = None,
        error: str | None = None,
        created_at: str | None = None,
    ) -> None: ...

    @abc.abstractmethod
    async def get(self, run_id: str) -> dict[str, Any] | None: ...

    @abc.abstractmethod
    async def list_by_thread(
        self,
        thread_id: str,
        *,
        owner_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]: ...

    @abc.abstractmethod
    async def update_status(
        self,
        run_id: str,
        status: str,
        *,
        error: str | None = None,
    ) -> None: ...

    @abc.abstractmethod
    async def delete(self, run_id: str) -> None: ...

    @abc.abstractmethod
    async def list_pending(self, *, before: str | None = None) -> list[dict[str, Any]]: ...
