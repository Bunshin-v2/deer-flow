"""Tests for RunJournal callback handler.

Uses MemoryRunEventStore as the backend for direct event inspection.
"""

import asyncio
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from deerflow.runtime.events.store.memory import MemoryRunEventStore
from deerflow.runtime.journal import RunJournal


@pytest.fixture
def journal_setup():
    store = MemoryRunEventStore()
    on_complete_data = {}

    def on_complete(**data):
        on_complete_data.update(data)

    j = RunJournal("r1", "t1", store, on_complete=on_complete, flush_threshold=100)
    return j, store, on_complete_data


def _make_llm_response(content="Hello", usage=None):
    """Create a mock LLM response with a message."""
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


class TestLlmCallbacks:
    @pytest.mark.anyio
    async def test_on_llm_end_produces_trace_event(self, journal_setup):
        j, store, _ = journal_setup
        run_id = uuid4()
        j.on_llm_start({}, [], run_id=run_id, tags=["lead_agent"])
        j.on_llm_end(_make_llm_response("Hi"), run_id=run_id, tags=["lead_agent"])
        await j.flush()
        events = await store.list_events("t1", "r1")
        trace_events = [e for e in events if e["event_type"] == "llm_end"]
        assert len(trace_events) == 1
        assert trace_events[0]["category"] == "trace"

    @pytest.mark.anyio
    async def test_on_llm_end_lead_agent_produces_ai_message(self, journal_setup):
        j, store, _ = journal_setup
        run_id = uuid4()
        j.on_llm_start({}, [], run_id=run_id, tags=["lead_agent"])
        j.on_llm_end(_make_llm_response("Answer"), run_id=run_id, tags=["lead_agent"])
        await j.flush()
        messages = await store.list_messages("t1")
        assert len(messages) == 1
        assert messages[0]["event_type"] == "ai_message"
        assert messages[0]["content"] == "Answer"

    @pytest.mark.anyio
    async def test_on_llm_end_subagent_no_ai_message(self, journal_setup):
        j, store, _ = journal_setup
        run_id = uuid4()
        j.on_llm_start({}, [], run_id=run_id, tags=["subagent:research"])
        j.on_llm_end(_make_llm_response("Sub answer"), run_id=run_id, tags=["subagent:research"])
        await j.flush()
        messages = await store.list_messages("t1")
        assert len(messages) == 0

    @pytest.mark.anyio
    async def test_token_accumulation(self, journal_setup):
        j, store, on_complete_data = journal_setup
        usage1 = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
        usage2 = {"input_tokens": 20, "output_tokens": 10, "total_tokens": 30}
        j.on_llm_start({}, [], run_id=uuid4(), tags=["lead_agent"])
        j.on_llm_end(_make_llm_response("A", usage=usage1), run_id=uuid4(), tags=["lead_agent"])
        j.on_llm_start({}, [], run_id=uuid4(), tags=["lead_agent"])
        j.on_llm_end(_make_llm_response("B", usage=usage2), run_id=uuid4(), tags=["lead_agent"])
        assert j._total_input_tokens == 30
        assert j._total_output_tokens == 15
        assert j._total_tokens == 45
        assert j._llm_call_count == 2

    @pytest.mark.anyio
    async def test_caller_token_classification(self, journal_setup):
        j, store, _ = journal_setup
        usage = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
        j.on_llm_start({}, [], run_id=uuid4(), tags=["lead_agent"])
        j.on_llm_end(_make_llm_response("A", usage=usage), run_id=uuid4(), tags=["lead_agent"])
        j.on_llm_start({}, [], run_id=uuid4(), tags=["subagent:research"])
        j.on_llm_end(_make_llm_response("B", usage=usage), run_id=uuid4(), tags=["subagent:research"])
        j.on_llm_start({}, [], run_id=uuid4(), tags=["middleware:summarization"])
        j.on_llm_end(_make_llm_response("C", usage=usage), run_id=uuid4(), tags=["middleware:summarization"])
        assert j._lead_agent_tokens == 15
        assert j._subagent_tokens == 15
        assert j._middleware_tokens == 15

    @pytest.mark.anyio
    async def test_usage_metadata_none_no_crash(self, journal_setup):
        j, store, _ = journal_setup
        j.on_llm_start({}, [], run_id=uuid4(), tags=["lead_agent"])
        j.on_llm_end(_make_llm_response("No usage", usage=None), run_id=uuid4(), tags=["lead_agent"])
        # Should not raise
        await j.flush()

    @pytest.mark.anyio
    async def test_latency_tracking(self, journal_setup):
        j, store, _ = journal_setup
        run_id = uuid4()
        j.on_llm_start({}, [], run_id=run_id, tags=["lead_agent"])
        j.on_llm_end(_make_llm_response("Fast"), run_id=run_id, tags=["lead_agent"])
        await j.flush()
        events = await store.list_events("t1", "r1")
        llm_end = [e for e in events if e["event_type"] == "llm_end"][0]
        assert "latency_ms" in llm_end["metadata"]
        assert llm_end["metadata"]["latency_ms"] is not None


class TestLifecycleCallbacks:
    @pytest.mark.anyio
    async def test_on_chain_end_triggers_on_complete(self, journal_setup):
        j, store, on_complete_data = journal_setup
        j.on_chain_start({}, {}, run_id=uuid4(), parent_run_id=None)
        j.on_chain_end({}, run_id=uuid4(), parent_run_id=None)
        assert "total_tokens" in on_complete_data
        assert "message_count" in on_complete_data

    @pytest.mark.anyio
    async def test_nested_chain_ignored(self, journal_setup):
        j, store, on_complete_data = journal_setup
        parent_id = uuid4()
        j.on_chain_start({}, {}, run_id=uuid4(), parent_run_id=parent_id)
        j.on_chain_end({}, run_id=uuid4(), parent_run_id=parent_id)
        await j.flush()
        events = await store.list_events("t1", "r1")
        lifecycle = [e for e in events if e["category"] == "lifecycle"]
        assert len(lifecycle) == 0


class TestToolCallbacks:
    @pytest.mark.anyio
    async def test_tool_start_end_produce_trace(self, journal_setup):
        j, store, _ = journal_setup
        j.on_tool_start({"name": "web_search"}, "query", run_id=uuid4())
        j.on_tool_end("results", run_id=uuid4(), name="web_search")
        await j.flush()
        events = await store.list_events("t1", "r1")
        types = {e["event_type"] for e in events}
        assert "tool_start" in types
        assert "tool_end" in types


class TestCustomEvents:
    @pytest.mark.anyio
    async def test_summarization_event(self, journal_setup):
        j, store, _ = journal_setup
        j.on_custom_event(
            "summarization",
            {"summary": "Context was summarized.", "replaced_count": 5, "replaced_message_ids": ["a", "b"]},
            run_id=uuid4(),
        )
        await j.flush()
        events = await store.list_events("t1", "r1")
        trace = [e for e in events if e["event_type"] == "summarization"]
        assert len(trace) == 1
        messages = await store.list_messages("t1")
        assert len(messages) == 1
        assert messages[0]["event_type"] == "summary"


class TestBufferFlush:
    @pytest.mark.anyio
    async def test_flush_threshold(self, journal_setup):
        j, store, _ = journal_setup
        j._flush_threshold = 3
        j.on_tool_start({"name": "a"}, "x", run_id=uuid4())
        j.on_tool_start({"name": "b"}, "x", run_id=uuid4())
        # Buffer has 2 events, not yet flushed
        assert len(j._buffer) == 2
        j.on_tool_start({"name": "c"}, "x", run_id=uuid4())
        # Buffer should have been flushed (threshold=3 triggers flush)
        # Give the async task a chance to complete
        await asyncio.sleep(0.1)
        events = await store.list_events("t1", "r1")
        assert len(events) >= 3


class TestIdentifyCaller:
    def test_lead_agent_tag(self, journal_setup):
        j, _, _ = journal_setup
        assert j._identify_caller({"tags": ["lead_agent"]}) == "lead_agent"

    def test_subagent_tag(self, journal_setup):
        j, _, _ = journal_setup
        assert j._identify_caller({"tags": ["subagent:research"]}) == "subagent:research"

    def test_middleware_tag(self, journal_setup):
        j, _, _ = journal_setup
        assert j._identify_caller({"tags": ["middleware:summarization"]}) == "middleware:summarization"

    def test_no_tags_returns_unknown(self, journal_setup):
        j, _, _ = journal_setup
        assert j._identify_caller({"tags": []}) == "unknown"
        assert j._identify_caller({}) == "unknown"


class TestPublicMethods:
    @pytest.mark.anyio
    async def test_set_first_human_message(self, journal_setup):
        j, _, _ = journal_setup
        j.set_first_human_message("Hello world")
        assert j._first_human_msg == "Hello world"

    @pytest.mark.anyio
    async def test_get_completion_data(self, journal_setup):
        j, _, _ = journal_setup
        j._total_tokens = 100
        j._msg_count = 5
        data = j.get_completion_data()
        assert data["total_tokens"] == 100
        assert data["message_count"] == 5
