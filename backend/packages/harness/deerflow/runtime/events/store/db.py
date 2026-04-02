"""SQLAlchemy-backed RunEventStore implementation.

Persists events to the ``run_events`` table. Trace content is truncated
at ``max_trace_content`` bytes to avoid bloating the database.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from deerflow.persistence.models.run_event import RunEventRow
from deerflow.runtime.events.store.base import RunEventStore


class DbRunEventStore(RunEventStore):
    def __init__(self, session_factory: async_sessionmaker[AsyncSession], *, max_trace_content: int = 10240):
        self._sf = session_factory
        self._max_trace_content = max_trace_content

    @staticmethod
    def _row_to_dict(row: RunEventRow) -> dict:
        d = row.to_dict()
        d["metadata"] = d.pop("event_metadata", {})
        val = d.get("created_at")
        if isinstance(val, datetime):
            d["created_at"] = val.isoformat()
        d.pop("id", None)
        return d

    def _truncate_trace(self, category: str, content: str, metadata: dict | None) -> tuple[str, dict]:
        if category == "trace" and len(content) > self._max_trace_content:
            content = content[: self._max_trace_content]
            metadata = {**(metadata or {}), "content_truncated": True}
        return content, metadata or {}

    async def put(self, *, thread_id, run_id, event_type, category, content="", metadata=None, created_at=None):
        content, metadata = self._truncate_trace(category, content, metadata)
        async with self._sf() as session:
            max_seq = await session.scalar(select(func.max(RunEventRow.seq)).where(RunEventRow.thread_id == thread_id))
            seq = (max_seq or 0) + 1
            row = RunEventRow(
                thread_id=thread_id,
                run_id=run_id,
                event_type=event_type,
                category=category,
                content=content,
                event_metadata=metadata,
                seq=seq,
                created_at=datetime.fromisoformat(created_at) if created_at else datetime.now(UTC),
            )
            session.add(row)
            await session.commit()
            await session.refresh(row)
            return self._row_to_dict(row)

    async def put_batch(self, events):
        if not events:
            return []
        async with self._sf() as session:
            # Get max seq for the thread (assume all events in batch belong to same thread)
            thread_id = events[0]["thread_id"]
            max_seq = await session.scalar(select(func.max(RunEventRow.seq)).where(RunEventRow.thread_id == thread_id))
            seq = max_seq or 0
            rows = []
            for e in events:
                seq += 1
                content = e.get("content", "")
                category = e.get("category", "trace")
                metadata = e.get("metadata")
                content, metadata = self._truncate_trace(category, content, metadata)
                row = RunEventRow(
                    thread_id=e["thread_id"],
                    run_id=e["run_id"],
                    event_type=e["event_type"],
                    category=category,
                    content=content,
                    event_metadata=metadata,
                    seq=seq,
                    created_at=datetime.fromisoformat(e["created_at"]) if e.get("created_at") else datetime.now(UTC),
                )
                session.add(row)
                rows.append(row)
            await session.commit()
            for row in rows:
                await session.refresh(row)
            return [self._row_to_dict(r) for r in rows]

    async def list_messages(self, thread_id, *, limit=50, before_seq=None, after_seq=None):
        stmt = select(RunEventRow).where(RunEventRow.thread_id == thread_id, RunEventRow.category == "message")
        if before_seq is not None:
            stmt = stmt.where(RunEventRow.seq < before_seq)
        if after_seq is not None:
            stmt = stmt.where(RunEventRow.seq > after_seq)

        if after_seq is not None:
            # Forward pagination: first `limit` records after cursor
            stmt = stmt.order_by(RunEventRow.seq.asc()).limit(limit)
            async with self._sf() as session:
                result = await session.execute(stmt)
                return [self._row_to_dict(r) for r in result.scalars()]
        else:
            # before_seq or default (latest): take last `limit` records, return ascending
            stmt = stmt.order_by(RunEventRow.seq.desc()).limit(limit)
            async with self._sf() as session:
                result = await session.execute(stmt)
                rows = list(result.scalars())
                return [self._row_to_dict(r) for r in reversed(rows)]

    async def list_events(self, thread_id, run_id, *, event_types=None, limit=500):
        stmt = select(RunEventRow).where(RunEventRow.thread_id == thread_id, RunEventRow.run_id == run_id)
        if event_types:
            stmt = stmt.where(RunEventRow.event_type.in_(event_types))
        stmt = stmt.order_by(RunEventRow.seq.asc()).limit(limit)
        async with self._sf() as session:
            result = await session.execute(stmt)
            return [self._row_to_dict(r) for r in result.scalars()]

    async def list_messages_by_run(self, thread_id, run_id):
        stmt = select(RunEventRow).where(RunEventRow.thread_id == thread_id, RunEventRow.run_id == run_id, RunEventRow.category == "message").order_by(RunEventRow.seq.asc())
        async with self._sf() as session:
            result = await session.execute(stmt)
            return [self._row_to_dict(r) for r in result.scalars()]

    async def count_messages(self, thread_id):
        stmt = select(func.count()).select_from(RunEventRow).where(RunEventRow.thread_id == thread_id, RunEventRow.category == "message")
        async with self._sf() as session:
            return await session.scalar(stmt) or 0

    async def delete_by_thread(self, thread_id):
        async with self._sf() as session:
            count_stmt = select(func.count()).select_from(RunEventRow).where(RunEventRow.thread_id == thread_id)
            count = await session.scalar(count_stmt) or 0
            if count > 0:
                await session.execute(delete(RunEventRow).where(RunEventRow.thread_id == thread_id))
                await session.commit()
            return count

    async def delete_by_run(self, thread_id, run_id):
        async with self._sf() as session:
            count_stmt = select(func.count()).select_from(RunEventRow).where(RunEventRow.thread_id == thread_id, RunEventRow.run_id == run_id)
            count = await session.scalar(count_stmt) or 0
            if count > 0:
                await session.execute(delete(RunEventRow).where(RunEventRow.thread_id == thread_id, RunEventRow.run_id == run_id))
                await session.commit()
            return count
