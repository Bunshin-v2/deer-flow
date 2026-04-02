"""Phase 2-B integration tests.

End-to-end test: simulate a run's complete lifecycle, verify data
is correctly written to both RunStore and RunEventStore.
"""

import asyncio
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from deerflow.runtime.events.store.memory import MemoryRunEventStore
from deerflow.runtime.journal import RunJournal
from deerflow.runtime.runs.store.memory import MemoryRunStore


def _make_llm_response(content="Hello", usage=None):
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = []
    msg.response_metadata = {"model_name": "test-model"}
    msg.usage_metadata = usage

    gen = MagicMock()
    gen.message = msg

    response = MagicMock()
    response.generations = [[gen]]
    return response


class TestRunLifecycle:
    @pytest.mark.anyio
    async def test_full_run_lifecycle(self):
        """Simulate a complete run lifecycle with RunStore + RunEventStore."""
        run_store = MemoryRunStore()
        event_store = MemoryRunEventStore()

        # 1. Create run
        await run_store.put("r1", thread_id="t1", status="pending")

        # 2. Write human_message
        await event_store.put(
            thread_id="t1",
            run_id="r1",
            event_type="human_message",
            category="message",
            content="What is AI?",
        )

        # 3. Simulate RunJournal callback sequence
        on_complete_data = {}

        def on_complete(**data):
            on_complete_data.update(data)

        journal = RunJournal("r1", "t1", event_store, on_complete=on_complete, flush_threshold=100)
        journal.set_first_human_message("What is AI?")

        # chain_start (top-level)
        journal.on_chain_start({}, {"messages": ["What is AI?"]}, run_id=uuid4(), parent_run_id=None)

        # llm_start + llm_end
        llm_run_id = uuid4()
        journal.on_llm_start({"name": "gpt-4"}, ["prompt"], run_id=llm_run_id, tags=["lead_agent"])
        usage = {"input_tokens": 50, "output_tokens": 100, "total_tokens": 150}
        journal.on_llm_end(_make_llm_response("AI is artificial intelligence.", usage=usage), run_id=llm_run_id, tags=["lead_agent"])

        # chain_end (triggers on_complete + flush_sync which creates a task)
        journal.on_chain_end({}, run_id=uuid4(), parent_run_id=None)
        await journal.flush()
        # Let event loop process any pending flush tasks from _flush_sync
        await asyncio.sleep(0.05)

        # 4. Verify messages
        messages = await event_store.list_messages("t1")
        assert len(messages) == 2  # human + ai
        assert messages[0]["event_type"] == "human_message"
        assert messages[1]["event_type"] == "ai_message"
        assert messages[1]["content"] == "AI is artificial intelligence."

        # 5. Verify events
        events = await event_store.list_events("t1", "r1")
        event_types = {e["event_type"] for e in events}
        assert "run_start" in event_types
        assert "llm_start" in event_types
        assert "llm_end" in event_types
        assert "run_end" in event_types

        # 6. Verify on_complete data
        assert on_complete_data["total_tokens"] == 150
        assert on_complete_data["llm_call_count"] == 1
        assert on_complete_data["lead_agent_tokens"] == 150
        assert on_complete_data["message_count"] == 1
        assert on_complete_data["last_ai_message"] == "AI is artificial intelligence."
        assert on_complete_data["first_human_message"] == "What is AI?"

    @pytest.mark.anyio
    async def test_run_with_tool_calls(self):
        """Simulate a run that uses tools."""
        event_store = MemoryRunEventStore()
        journal = RunJournal("r1", "t1", event_store, flush_threshold=100)

        # tool_start + tool_end
        journal.on_tool_start({"name": "web_search"}, '{"query": "AI"}', run_id=uuid4())
        journal.on_tool_end("Search results...", run_id=uuid4(), name="web_search")
        await journal.flush()

        events = await event_store.list_events("t1", "r1")
        assert len(events) == 2
        assert events[0]["event_type"] == "tool_start"
        assert events[1]["event_type"] == "tool_end"

    @pytest.mark.anyio
    async def test_multi_run_thread(self):
        """Multiple runs on the same thread maintain unified seq ordering."""
        event_store = MemoryRunEventStore()

        # Run 1
        await event_store.put(thread_id="t1", run_id="r1", event_type="human_message", category="message", content="Q1")
        await event_store.put(thread_id="t1", run_id="r1", event_type="ai_message", category="message", content="A1")

        # Run 2
        await event_store.put(thread_id="t1", run_id="r2", event_type="human_message", category="message", content="Q2")
        await event_store.put(thread_id="t1", run_id="r2", event_type="ai_message", category="message", content="A2")

        messages = await event_store.list_messages("t1")
        assert len(messages) == 4
        assert [m["seq"] for m in messages] == [1, 2, 3, 4]
        assert messages[0]["run_id"] == "r1"
        assert messages[2]["run_id"] == "r2"

    @pytest.mark.anyio
    async def test_runmanager_with_store_backing(self):
        """RunManager persists to RunStore when one is provided."""
        from deerflow.runtime.runs.manager import RunManager

        run_store = MemoryRunStore()
        mgr = RunManager(store=run_store)

        record = await mgr.create("t1", assistant_id="lead_agent")
        # Verify persisted to store
        row = await run_store.get(record.run_id)
        assert row is not None
        assert row["thread_id"] == "t1"
        assert row["status"] == "pending"

        # Status update
        from deerflow.runtime.runs.schemas import RunStatus

        await mgr.set_status(record.run_id, RunStatus.running)
        row = await run_store.get(record.run_id)
        assert row["status"] == "running"
