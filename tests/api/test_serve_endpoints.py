"""API endpoint tests — FastAPI `synapsekit serve`.

Tests all HTTP endpoints exposed by build_app():
  GET  /health  → 200 {"status": "ok"}
  POST /query   → RAG mode
  POST /run     → Agent/Graph mode
  GET  /stream  → Graph SSE

Uses httpx.AsyncClient with the ASGI transport — no real server needed.
Requires: pip install fastapi httpx
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from synapsekit.cli.serve import _detect_type, build_app

try:
    from fastapi.testclient import TestClient

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

pytestmark = pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="fastapi/httpx not installed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeRAG:
    """Fake RAG class — class name used by _detect_type."""

    def __init__(self, answer: str = "Paris"):
        self.aquery = AsyncMock(return_value=answer)


class _FakeFunctionCallingAgent:
    def __init__(self, answer: str = "Agent done"):
        self.arun = AsyncMock(return_value=answer)


class _FakeCompiledGraph:
    def __init__(self, result: dict | None = None):
        self._result = result or {"text": "PROCESSED"}
        self.arun = AsyncMock(return_value=self._result)

    async def astream(self, state):
        yield {"node": "step1", "state": state}
        yield {"node": "step2", "state": self._result}


def _make_rag_mock(answer: str = "Paris") -> _FakeRAG:
    return _FakeRAG(answer)


def _make_agent_mock(answer: str = "Agent done") -> _FakeFunctionCallingAgent:
    return _FakeFunctionCallingAgent(answer)


def _make_graph_mock(result: dict | None = None) -> _FakeCompiledGraph:
    return _FakeCompiledGraph(result)

# ---------------------------------------------------------------------------
# _detect_type unit tests
# ---------------------------------------------------------------------------


def test_detect_type_rag():
    # _detect_type matches on class name "RAG" in MRO
    class RAG:  # name must be exactly "RAG"
        pass

    assert _detect_type(RAG()) == "rag"


def test_detect_type_graph():
    class CompiledGraph:  # name must be exactly "CompiledGraph"
        pass

    assert _detect_type(CompiledGraph()) == "graph"


def test_detect_type_agent():
    class FunctionCallingAgent:
        pass

    assert _detect_type(FunctionCallingAgent()) == "agent"


# ---------------------------------------------------------------------------
# /health endpoint
# ---------------------------------------------------------------------------


def test_health_endpoint_rag():
    rag = _make_rag_mock()
    app = build_app(rag, app_type="rag")
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_health_endpoint_agent():
    agent = _make_agent_mock()
    app = build_app(agent, app_type="agent")
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_health_endpoint_graph():
    graph = _make_graph_mock()
    app = build_app(graph, app_type="graph")
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# /query endpoint (RAG mode)
# ---------------------------------------------------------------------------


def test_query_returns_answer():
    rag = _make_rag_mock(answer="The capital is Paris.")
    app = build_app(rag, app_type="rag")
    client = TestClient(app)
    resp = client.post("/query", json={"query": "What is the capital of France?"})
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert "Paris" in data["answer"]


def test_query_accepts_question_key():
    """'question' key is a synonym for 'query'."""
    rag = _make_rag_mock(answer="answer")
    app = build_app(rag, app_type="rag")
    client = TestClient(app)
    resp = client.post("/query", json={"question": "Something?"})
    assert resp.status_code == 200
    assert "answer" in resp.json()


def test_query_missing_field_returns_400():
    rag = _make_rag_mock()
    app = build_app(rag, app_type="rag")
    client = TestClient(app)
    resp = client.post("/query", json={})
    assert resp.status_code == 400
    assert "error" in resp.json()


def test_query_empty_string_returns_400():
    rag = _make_rag_mock()
    app = build_app(rag, app_type="rag")
    client = TestClient(app)
    resp = client.post("/query", json={"query": ""})
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# /run endpoint (Agent mode)
# ---------------------------------------------------------------------------


def test_agent_run_returns_answer():
    agent = _make_agent_mock(answer="Task complete")
    app = build_app(agent, app_type="agent")
    client = TestClient(app)
    resp = client.post("/run", json={"prompt": "Do something"})
    assert resp.status_code == 200
    assert "answer" in resp.json()
    assert "Task complete" in resp.json()["answer"]


def test_agent_run_accepts_input_key():
    """'input' key is a synonym for 'prompt'."""
    agent = _make_agent_mock(answer="done")
    app = build_app(agent, app_type="agent")
    client = TestClient(app)
    resp = client.post("/run", json={"input": "run task"})
    assert resp.status_code == 200


def test_agent_run_missing_prompt_returns_400():
    agent = _make_agent_mock()
    app = build_app(agent, app_type="agent")
    client = TestClient(app)
    resp = client.post("/run", json={})
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# /run endpoint (Graph mode)
# ---------------------------------------------------------------------------


def test_graph_run_returns_result():
    graph = _make_graph_mock(result={"text": "DONE"})
    app = build_app(graph, app_type="graph")
    client = TestClient(app)
    resp = client.post("/run", json={"state": {"text": "hello"}})
    assert resp.status_code == 200
    assert "result" in resp.json()


# ---------------------------------------------------------------------------
# /stream endpoint (Graph mode SSE)
# ---------------------------------------------------------------------------


def test_graph_stream_returns_event_stream():
    graph = _make_graph_mock()
    app = build_app(graph, app_type="graph")
    client = TestClient(app)
    resp = client.get("/stream", params={"state": json.dumps({"text": "test"})})
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers.get("content-type", "")


def test_graph_stream_content_is_sse_format():
    graph = _make_graph_mock()
    app = build_app(graph, app_type="graph")
    client = TestClient(app)
    resp = client.get("/stream", params={"state": json.dumps({"text": "hi"})})
    content = resp.text
    assert "data:" in content


# ---------------------------------------------------------------------------
# build_app raises on missing FastAPI
# ---------------------------------------------------------------------------


def test_build_app_returns_fastapi_app():
    from fastapi import FastAPI

    rag = _make_rag_mock()
    app = build_app(rag, app_type="rag")
    assert isinstance(app, FastAPI)
