"""Centralized accessors for singleton objects stored on ``app.state``.

**Getters** (used by routers): raise 503 when a required dependency is
missing, except ``get_store`` which returns ``None``.

Initialization is handled directly in ``app.py`` via :class:`AsyncExitStack`.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from deerflow.runtime import RunManager, StreamBridge
from deerflow.runtime.events.store.base import RunEventStore
from deerflow.runtime.runs.store.base import RunStore


@asynccontextmanager
async def langgraph_runtime(app: FastAPI) -> AsyncGenerator[None, None]:
    """Bootstrap and tear down all LangGraph runtime singletons.

    Usage in ``app.py``::

        async with langgraph_runtime(app):
            yield
    """
    from deerflow.agents.checkpointer.async_provider import make_checkpointer
    from deerflow.config import get_app_config
    from deerflow.persistence.engine import close_engine, init_engine_from_config
    from deerflow.runtime import make_store, make_stream_bridge

    async with AsyncExitStack() as stack:
        app.state.stream_bridge = await stack.enter_async_context(make_stream_bridge())
        app.state.checkpointer = await stack.enter_async_context(make_checkpointer())
        app.state.store = await stack.enter_async_context(make_store())
        # Initialize persistence layer from unified database config
        config = get_app_config()
        await init_engine_from_config(config.database)

        # Initialize run store (RunRepository if DB available, else MemoryRunStore)
        app.state.run_store = _make_run_store()

        # Initialize run event store based on config
        app.state.run_event_store = _make_run_event_store(config)

        # RunManager with store backing for persistence
        app.state.run_manager = RunManager(store=app.state.run_store)

        try:
            yield
        finally:
            await close_engine()


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def _make_run_store() -> RunStore:
    """Create a RunStore: RunRepository if DB engine is available, else MemoryRunStore."""
    from deerflow.persistence.engine import get_session_factory

    sf = get_session_factory()
    if sf is not None:
        from deerflow.persistence.repositories.run_repo import RunRepository

        return RunRepository(sf)
    from deerflow.runtime.runs.store.memory import MemoryRunStore

    return MemoryRunStore()


def _make_run_event_store(config) -> RunEventStore:
    from deerflow.runtime.events.store import make_run_event_store

    run_events_config = getattr(config, "run_events", None)
    return make_run_event_store(run_events_config)


# ---------------------------------------------------------------------------
# Getters -- called by routers per-request
# ---------------------------------------------------------------------------


def get_stream_bridge(request: Request) -> StreamBridge:
    """Return the global :class:`StreamBridge`, or 503."""
    bridge = getattr(request.app.state, "stream_bridge", None)
    if bridge is None:
        raise HTTPException(status_code=503, detail="Stream bridge not available")
    return bridge


def get_run_manager(request: Request) -> RunManager:
    """Return the global :class:`RunManager`, or 503."""
    mgr = getattr(request.app.state, "run_manager", None)
    if mgr is None:
        raise HTTPException(status_code=503, detail="Run manager not available")
    return mgr


def get_checkpointer(request: Request):
    """Return the global checkpointer, or 503."""
    cp = getattr(request.app.state, "checkpointer", None)
    if cp is None:
        raise HTTPException(status_code=503, detail="Checkpointer not available")
    return cp


def get_store(request: Request):
    """Return the global store (may be ``None`` if not configured)."""
    return getattr(request.app.state, "store", None)


def get_run_event_store(request: Request) -> RunEventStore:
    """Return the RunEventStore, or 503 if not available."""
    store = getattr(request.app.state, "run_event_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="Run event store not available")
    return store


def get_run_store(request: Request) -> RunStore:
    """Return the RunStore, or 503 if not available."""
    store = getattr(request.app.state, "run_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="Run store not available")
    return store


async def get_current_user(request: Request) -> str | None:
    """Extract user identity from request.

    Phase 2: always returns None (no authentication).
    Phase 3: extract user_id from JWT / session / API key header.
    """
    return None
