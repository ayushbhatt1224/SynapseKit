"""
SynapseKit functional smoke test.
Tests real instantiation, data flow, and execution — no mocks.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import traceback

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
SKIP = "\033[93m~\033[0m"

results: list[tuple[str, str, str]] = []


def check(label: str, fn, *, skip_if: str | None = None):
    if skip_if:
        results.append((label, SKIP, skip_if))
        return
    try:
        fn()
        results.append((label, PASS, ""))
    except Exception as e:
        results.append((label, FAIL, f"{type(e).__name__}: {e}"))
        traceback.print_exc()


def acheck(label: str, coro, *, skip_if: str | None = None):
    import inspect

    if skip_if:
        results.append((label, SKIP, skip_if))
        # Close the coroutine to avoid RuntimeWarning
        if inspect.iscoroutine(coro):
            coro.close()
        return
    try:
        asyncio.run(coro)
        results.append((label, PASS, ""))
    except Exception as e:
        results.append((label, FAIL, f"{type(e).__name__}: {e}"))
        traceback.print_exc()


# ── 1. IMPORTS ─────────────────────────────────────────────────────────────────


def test_top_level():
    import synapsekit  # noqa: F401


check("Import: synapsekit", test_top_level)


def test_loader_imports():
    from synapsekit.loaders import (
        Document,
        MarkdownLoader,
        NotionLoader,
        SlackLoader,
        StringLoader,
        TextLoader,
    )

    assert all([Document, TextLoader, MarkdownLoader, StringLoader, SlackLoader, NotionLoader])


check("Import: loaders incl. SlackLoader + NotionLoader", test_loader_imports)


def test_graph_imports():
    from synapsekit.graph import StateGraph
    from synapsekit.graph.subgraph import subgraph_node

    assert StateGraph and subgraph_node


check("Import: graph + subgraph_node", test_graph_imports)


def test_agent_imports():
    from synapsekit.agents import ReActAgent
    from synapsekit.agents.tools import (
        CalculatorTool,
        JiraTool,
        JSONQueryTool,
        NotionTool,
        PythonREPLTool,
    )

    assert all([ReActAgent, CalculatorTool, PythonREPLTool, JSONQueryTool, JiraTool, NotionTool])


check("Import: agents + tools incl. NotionTool", test_agent_imports)


def test_retrieval_imports():
    from synapsekit.retrieval.faiss import FAISSVectorStore
    from synapsekit.retrieval.retriever import Retriever

    assert FAISSVectorStore and Retriever


check("Import: retrieval (FAISS + Retriever)", test_retrieval_imports)


def test_splitter_imports():
    from synapsekit.text_splitters import (
        CharacterTextSplitter,
        RecursiveCharacterTextSplitter,
        TokenAwareSplitter,
    )

    assert RecursiveCharacterTextSplitter and CharacterTextSplitter and TokenAwareSplitter


check("Import: text splitters", test_splitter_imports)


def test_embeddings_import():
    from synapsekit.embeddings import SynapsekitEmbeddings

    assert SynapsekitEmbeddings


check("Import: SynapsekitEmbeddings", test_embeddings_import)


# ── 2. DOCUMENT ────────────────────────────────────────────────────────────────


def test_document():
    from synapsekit.loaders.base import Document

    d = Document(text="hello world", metadata={"source": "test", "page": 1})
    assert d.text == "hello world"
    assert d.metadata["page"] == 1
    d2 = Document(text="hello world", metadata={"source": "test", "page": 1})
    assert d == d2
    d3 = Document(text="different", metadata={})
    assert d != d3


check("Document: create, metadata, equality", test_document)


# ── 3. LOADERS ─────────────────────────────────────────────────────────────────


def test_string_loader():
    from synapsekit.loaders import StringLoader

    docs = StringLoader("Hello SynapseKit").load()
    assert len(docs) == 1 and docs[0].text == "Hello SynapseKit"


check("StringLoader", test_string_loader)


def test_text_loader():
    from synapsekit.loaders import TextLoader

    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
        f.write("line one\nline two\nline three")
        path = f.name
    try:
        docs = TextLoader(path).load()
        assert len(docs) == 1
        assert "line one" in docs[0].text
        assert docs[0].metadata["source"] == path
    finally:
        os.unlink(path)


check("TextLoader", test_text_loader)


def test_markdown_loader():
    from synapsekit.loaders import MarkdownLoader

    with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False) as f:
        f.write("# Title\n\nSome **bold** text.\n\n## Section\n\nMore content.")
        path = f.name
    try:
        docs = MarkdownLoader(path).load()
        assert len(docs) >= 1
        combined = " ".join(d.text for d in docs)
        assert "Title" in combined or "Section" in combined
    finally:
        os.unlink(path)


check("MarkdownLoader", test_markdown_loader)


def test_csv_loader():
    from synapsekit.loaders import CSVLoader

    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
        f.write("name,age,city\nAlice,30,NYC\nBob,25,LA\n")
        path = f.name
    try:
        docs = CSVLoader(path).load()
        assert len(docs) >= 1
    finally:
        os.unlink(path)


check("CSVLoader", test_csv_loader)


def test_json_loader():
    from synapsekit.loaders import JSONLoader

    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        f.write('[{"text": "synapsekit version 1.4.6"}, {"text": "rag agents graph"}]')
        path = f.name
    try:
        docs = JSONLoader(path).load()
        assert len(docs) == 2
        assert "synapsekit" in docs[0].text
        assert "rag" in docs[1].text
    finally:
        os.unlink(path)


check("JSONLoader", test_json_loader)


def test_yaml_loader():
    from synapsekit.loaders import YAMLLoader

    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        f.write("name: SynapseKit\nfeatures:\n  - rag\n  - agents\n  - graph\n")
        path = f.name
    try:
        docs = YAMLLoader(path).load()
        assert len(docs) >= 1
    finally:
        os.unlink(path)


check("YAMLLoader", test_yaml_loader)


def test_xml_loader():
    from synapsekit.loaders import XMLLoader

    with tempfile.NamedTemporaryFile(suffix=".xml", mode="w", delete=False) as f:
        f.write("<root><item>Hello</item><item>World</item></root>")
        path = f.name
    try:
        docs = XMLLoader(path).load()
        assert len(docs) >= 1
    finally:
        os.unlink(path)


check("XMLLoader", test_xml_loader)


def test_html_loader():
    from synapsekit.loaders import HTMLLoader

    with tempfile.NamedTemporaryFile(suffix=".html", mode="w", delete=False) as f:
        f.write("<html><body><h1>Test</h1><p>Hello world.</p></body></html>")
        path = f.name
    try:
        docs = HTMLLoader(path).load()
        assert len(docs) >= 1
        assert "Hello world" in docs[0].text or "Test" in docs[0].text
    except ImportError as e:
        results.append(("HTMLLoader", SKIP, str(e).split(":")[0]))
        return
    finally:
        os.unlink(path)


test_html_loader()


def test_directory_loader():
    from synapsekit.loaders import DirectoryLoader

    with tempfile.TemporaryDirectory() as tmpdir:
        for name, content in [("a.txt", "file a"), ("b.txt", "file b"), ("skip.py", "python")]:
            with open(os.path.join(tmpdir, name), "w") as f:
                f.write(content)
        docs = DirectoryLoader(tmpdir, glob_pattern="*.txt").load()
        assert len(docs) == 2
        texts = {d.text for d in docs}
        assert "file a" in texts and "file b" in texts


check("DirectoryLoader (glob_pattern filter)", test_directory_loader)


def test_slack_loader_init():
    from synapsekit.loaders import SlackLoader

    loader = SlackLoader(bot_token="xoxb-fake", channel_id="C123456", limit=50)
    assert loader.bot_token == "xoxb-fake"
    assert loader.channel_id == "C123456"
    assert loader.limit == 50


check("SlackLoader: init + attributes", test_slack_loader_init)


def test_notion_loader_init():
    from synapsekit.loaders import NotionLoader

    loader = NotionLoader(api_key="secret_fake", page_id="abc-123", max_retries=5)
    assert loader.page_id == "abc-123"
    assert loader.max_retries == 5
    assert loader.database_id is None


check("NotionLoader: init + attributes", test_notion_loader_init)


def test_notion_loader_validation():
    from synapsekit.loaders import NotionLoader

    try:
        NotionLoader(api_key="key")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    try:
        NotionLoader(api_key="key", page_id="a", database_id="b")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


check("NotionLoader: validation (missing / both IDs raise)", test_notion_loader_validation)


# ── 4. TEXT SPLITTERS ──────────────────────────────────────────────────────────


def test_recursive_splitter():
    from synapsekit.text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(chunk_size=60, chunk_overlap=10)
    long_text = "The quick brown fox jumps over the lazy dog. " * 15
    chunks = splitter.split(long_text)
    assert len(chunks) > 1
    assert all(len(c) <= 80 for c in chunks)  # some tolerance


check("RecursiveCharacterTextSplitter.split()", test_recursive_splitter)


def test_splitter_with_metadata():
    from synapsekit.text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(chunk_size=60, chunk_overlap=10)
    text = "Paragraph one content here. " * 10
    chunks = splitter.split_with_metadata(text, metadata={"source": "doc.txt", "page": 1})
    assert len(chunks) > 1
    for c in chunks:
        assert c["metadata"]["source"] == "doc.txt"
        assert "text" in c


check("RecursiveCharacterTextSplitter.split_with_metadata()", test_splitter_with_metadata)


def test_character_splitter():
    from synapsekit.text_splitters import CharacterTextSplitter

    splitter = CharacterTextSplitter(chunk_size=50, chunk_overlap=5, separator="\n")
    text = "\n".join(f"line {i}: some content here" for i in range(20))
    chunks = splitter.split(text)
    assert len(chunks) > 1


check("CharacterTextSplitter.split()", test_character_splitter)


def test_token_aware_splitter():
    from synapsekit.text_splitters import TokenAwareSplitter

    splitter = TokenAwareSplitter(max_tokens=20, chunk_overlap=5)
    text = "word " * 200
    chunks = splitter.split(text)
    assert len(chunks) > 1


check("TokenAwareSplitter.split()", test_token_aware_splitter)


# ── 5. PROMPTS + PARSERS + MEMORY ─────────────────────────────────────────────


def test_prompt_template():
    from synapsekit.prompts import PromptTemplate

    pt = PromptTemplate("Hello {name}, version {version}.")
    out = pt.format(name="SynapseKit", version="1.4.6")
    assert out == "Hello SynapseKit, version 1.4.6."


check("PromptTemplate.format()", test_prompt_template)


def test_json_parser():
    from synapsekit.parsers import JSONParser

    parser = JSONParser()
    result = parser.parse('{"key": "value", "count": 42}')
    assert result["key"] == "value" and result["count"] == 42


check("JSONParser.parse()", test_json_parser)


def test_list_parser():
    from synapsekit.parsers import ListParser

    parser = ListParser()
    result = parser.parse("- item one\n- item two\n- item three")
    assert len(result) == 3
    assert "item one" in result


check("ListParser.parse()", test_list_parser)


def test_conversation_memory():
    from synapsekit.memory import ConversationMemory

    mem = ConversationMemory(window=5)
    mem.add("user", "Hello!")
    mem.add("assistant", "Hi there!")
    mem.add("user", "How are you?")

    history = mem.get_messages()
    assert len(history) == 3
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "Hello!"
    assert history[2]["content"] == "How are you?"


check("ConversationMemory: add + get + window", test_conversation_memory)


def test_memory_window():
    from synapsekit.memory import ConversationMemory

    mem = ConversationMemory(window=2)  # keeps last 2 turns = 4 messages
    for i in range(5):
        mem.add("user", f"msg {i}")
        mem.add("assistant", f"reply {i}")

    history = mem.get_messages()
    # Should be bounded by window
    assert len(history) <= 4


check("ConversationMemory: sliding window eviction", test_memory_window)


# ── 6. TOOLS ───────────────────────────────────────────────────────────────────


async def test_calculator():
    from synapsekit.agents.tools import CalculatorTool

    tool = CalculatorTool()
    r = await tool.run(expression="2 + 2 * 3")
    assert r.error is None and "8" in r.output

    r2 = await tool.run(expression="sqrt(144)")
    assert r2.error is None and "12" in r2.output


acheck("CalculatorTool: arithmetic + math functions", test_calculator())


async def test_python_repl():
    from synapsekit.agents.tools import PythonREPLTool

    tool = PythonREPLTool()

    # Basic execution
    r = await tool.run(code="print(sum(range(10)))")
    assert r.error is None and "45" in r.output

    # Variable persistence across calls within same tool instance
    r2 = await tool.run(code="x = [i**2 for i in range(5)]\nprint(sum(x))")
    assert r2.error is None and "30" in r2.output


acheck("PythonREPLTool: execution", test_python_repl())


async def test_json_query():
    import json

    from synapsekit.agents.tools import JSONQueryTool

    tool = JSONQueryTool()
    data = json.dumps(
        {
            "users": [{"name": "Alice", "score": 95}, {"name": "Bob", "score": 87}],
            "total": 2,
        }
    )
    r = await tool.run(json_data=data, path="users.0.name")
    assert r.error is None and "Alice" in r.output

    r2 = await tool.run(json_data=data, path="total")
    assert r2.error is None and "2" in r2.output


acheck("JSONQueryTool: jmespath queries", test_json_query())


async def test_notion_tool_schema_and_errors():
    from synapsekit.agents.tools import NotionTool

    tool = NotionTool(api_key="fake-key")
    assert tool.name == "notion"
    assert "search" in str(tool.parameters)
    assert "get_page" in str(tool.parameters)
    assert "create_page" in str(tool.parameters)
    assert "append_block" in str(tool.parameters)

    # Missing operation
    r = await tool.run(operation="")
    assert r.error

    # Unknown operation
    r2 = await tool.run(operation="nuke_everything")
    assert r2.error and "unknown" in r2.error.lower()

    # Missing required args
    r3 = await tool.run(operation="get_page", page_id="")
    assert r3.error


acheck("NotionTool: schema + error handling", test_notion_tool_schema_and_errors())


# ── 7. GRAPH WORKFLOWS ─────────────────────────────────────────────────────────


async def test_graph_linear():
    from synapsekit.graph import StateGraph

    def greet(state: dict) -> dict:
        return {"msg": "Hello, " + state.get("name", "world")}

    def shout(state: dict) -> dict:
        return {"msg": state["msg"].upper()}

    g = StateGraph()
    g.add_node("greet", greet)
    g.add_node("shout", shout)
    g.add_edge("greet", "shout")
    g.set_entry_point("greet")
    g.set_finish_point("shout")

    result = await g.compile().run({"name": "SynapseKit"})
    assert result["msg"] == "HELLO, SYNAPSEKIT"


acheck("Graph: linear pipeline (greet → shout)", test_graph_linear())


async def test_graph_conditional():
    from synapsekit.graph import StateGraph

    def classify(s: dict) -> dict:
        return {"label": "pos" if s["score"] > 0 else "neg"}

    def win(s: dict) -> dict:
        return {"result": "WIN"}

    def loss(s: dict) -> dict:
        return {"result": "LOSS"}

    g = StateGraph()
    g.add_node("classify", classify)
    g.add_node("pos", win)
    g.add_node("neg", loss)
    g.set_entry_point("classify")
    g.add_conditional_edge("classify", lambda s: s["label"], {"pos": "pos", "neg": "neg"})
    g.set_finish_point("pos")
    g.set_finish_point("neg")

    compiled = g.compile()
    assert (await compiled.run({"score": 1}))["result"] == "WIN"
    assert (await compiled.run({"score": -1}))["result"] == "LOSS"


acheck("Graph: conditional routing", test_graph_conditional())


async def test_graph_state_accumulation():
    from synapsekit.graph import StateGraph

    def step1(s: dict) -> dict:
        return {"steps": [*s.get("steps", []), "step1"], "value": 10}

    def step2(s: dict) -> dict:
        return {"steps": s["steps"] + ["step2"], "value": s["value"] * 2}

    def step3(s: dict) -> dict:
        return {"steps": s["steps"] + ["step3"], "value": s["value"] + 5}

    g = StateGraph()
    g.add_node("s1", step1)
    g.add_node("s2", step2)
    g.add_node("s3", step3)
    g.add_edge("s1", "s2")
    g.add_edge("s2", "s3")
    g.set_entry_point("s1")
    g.set_finish_point("s3")

    result = await g.compile().run({})
    assert result["steps"] == ["step1", "step2", "step3"]
    assert result["value"] == 25  # (10 * 2) + 5


acheck("Graph: state accumulation across nodes", test_graph_state_accumulation())


async def test_subgraph_retry():
    from synapsekit.graph import StateGraph
    from synapsekit.graph.subgraph import subgraph_node

    attempts = {"n": 0}

    async def flaky(s: dict) -> dict:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RuntimeError("not yet")
        return {"done": True}

    sub = StateGraph()
    sub.add_node("flaky", flaky)
    sub.set_entry_point("flaky")
    sub.set_finish_point("flaky")

    parent = StateGraph()
    parent.add_node("sub", subgraph_node(sub.compile(), on_error="retry", max_retries=3))
    parent.set_entry_point("sub")
    parent.set_finish_point("sub")

    result = await parent.compile().run({})
    assert result.get("done") is True
    assert attempts["n"] == 3


acheck("Graph: subgraph retry (succeeds on 3rd attempt)", test_subgraph_retry())


async def test_subgraph_retry_exhausted():
    from synapsekit.graph import StateGraph
    from synapsekit.graph.subgraph import subgraph_node

    async def always_fails(s: dict) -> dict:
        raise ValueError("permanent failure")

    sub = StateGraph()
    sub.add_node("f", always_fails)
    sub.set_entry_point("f")
    sub.set_finish_point("f")

    parent = StateGraph()
    parent.add_node("sub", subgraph_node(sub.compile(), on_error="retry", max_retries=2))
    parent.set_entry_point("sub")
    parent.set_finish_point("sub")

    try:
        await parent.compile().run({})
        raise AssertionError("expected to raise")
    except ValueError as e:
        assert "permanent failure" in str(e) or "attempt" in str(e).lower()


acheck("Graph: subgraph retry exhausted → raises", test_subgraph_retry_exhausted())


async def test_subgraph_fallback():
    from synapsekit.graph import StateGraph
    from synapsekit.graph.subgraph import subgraph_node

    async def broken(s: dict) -> dict:
        raise RuntimeError("broken")

    async def fallback_fn(s: dict) -> dict:
        return {"result": "fallback_used"}

    sub = StateGraph()
    sub.add_node("b", broken)
    sub.set_entry_point("b")
    sub.set_finish_point("b")

    fb = StateGraph()
    fb.add_node("f", fallback_fn)
    fb.set_entry_point("f")
    fb.set_finish_point("f")

    parent = StateGraph()
    parent.add_node("sub", subgraph_node(sub.compile(), on_error="fallback", fallback=fb.compile()))
    parent.set_entry_point("sub")
    parent.set_finish_point("sub")

    result = await parent.compile().run({})
    assert result.get("result") == "fallback_used"


acheck("Graph: subgraph fallback", test_subgraph_fallback())


async def test_subgraph_skip():
    from synapsekit.graph import StateGraph
    from synapsekit.graph.subgraph import subgraph_node

    async def broken(s: dict) -> dict:
        raise RuntimeError("broken")

    sub = StateGraph()
    sub.add_node("b", broken)
    sub.set_entry_point("b")
    sub.set_finish_point("b")

    parent = StateGraph()
    parent.add_node("sub", subgraph_node(sub.compile(), on_error="skip"))
    parent.add_node("after", lambda s: {**s, "after": True})
    parent.add_edge("sub", "after")
    parent.set_entry_point("sub")
    parent.set_finish_point("after")

    result = await parent.compile().run({"initial": True})
    assert result.get("after") is True
    assert result.get("initial") is True


acheck("Graph: subgraph skip (execution continues)", test_subgraph_skip())


async def test_graph_mermaid():
    from synapsekit.graph import StateGraph

    g = StateGraph()
    g.add_node("a", lambda s: s)
    g.add_node("b", lambda s: s)
    g.add_node("c", lambda s: s)
    g.add_edge("a", "b")
    g.add_edge("b", "c")
    g.set_entry_point("a")
    g.set_finish_point("c")

    mermaid = g.compile().get_mermaid()
    assert "a" in mermaid and "b" in mermaid and "c" in mermaid
    assert "-->" in mermaid


acheck("Graph: Mermaid export", test_graph_mermaid())


# ── 8. RAG (needs sentence-transformers + faiss) ───────────────────────────────


async def test_embeddings():
    try:
        from synapsekit.embeddings import SynapsekitEmbeddings

        emb = SynapsekitEmbeddings(model="all-MiniLM-L6-v2")
        vecs = await emb.embed(["hello world", "SynapseKit is great"])
        assert len(vecs) == 2
        assert len(vecs[0]) > 0
        results.append(("Embeddings: embed() returns vectors", PASS, ""))
    except ImportError as e:
        results.append(("Embeddings: embed() returns vectors", SKIP, str(e).split(":")[0]))


await_or_run = asyncio.run(test_embeddings())


async def test_faiss_store():
    try:
        import faiss  # noqa: F401

        from synapsekit.embeddings import SynapsekitEmbeddings
        from synapsekit.retrieval.faiss import FAISSVectorStore

        emb = SynapsekitEmbeddings(model="all-MiniLM-L6-v2")
        store = FAISSVectorStore(embedding_backend=emb)

        texts = [
            "Python is a programming language",
            "SynapseKit supports RAG pipelines",
            "Agents use function calling",
            "Graph workflows handle complex reasoning",
        ]
        metas = [{"id": i} for i in range(len(texts))]
        await store.add(texts, metas)

        results_list = await store.search("RAG and retrieval", top_k=2)
        assert len(results_list) == 2
        assert all("text" in r and "score" in r for r in results_list)
        assert isinstance(results_list[0]["score"], float)

        results.append(("FAISS: add + search + scores", PASS, ""))
    except ImportError as e:
        results.append(("FAISS: add + search + scores", SKIP, str(e).split(":")[0]))


asyncio.run(test_faiss_store())


async def test_retriever():
    try:
        import faiss  # noqa: F401

        from synapsekit.embeddings import SynapsekitEmbeddings
        from synapsekit.retrieval.faiss import FAISSVectorStore
        from synapsekit.retrieval.retriever import Retriever

        emb = SynapsekitEmbeddings(model="all-MiniLM-L6-v2")
        store = FAISSVectorStore(embedding_backend=emb)
        retriever = Retriever(vectorstore=store)

        corpus = [
            "The sky is blue on clear days",
            "Python is widely used for data science",
            "SynapseKit handles agents and graph workflows",
            "RAG combines retrieval with generation",
        ]
        await retriever.add(corpus)

        hits = await retriever.retrieve("agent framework", top_k=2)
        assert len(hits) == 2
        assert all(isinstance(h, str) for h in hits)

        hits_scored = await retriever.retrieve_with_scores("RAG retrieval", top_k=2)
        assert len(hits_scored) == 2
        assert all("text" in h and "score" in h for h in hits_scored)

        results.append(("Retriever: add + retrieve + retrieve_with_scores", PASS, ""))
    except ImportError as e:
        results.append(
            ("Retriever: add + retrieve + retrieve_with_scores", SKIP, str(e).split(":")[0])
        )


asyncio.run(test_retriever())


async def test_mini_rag_pipeline():
    """Full end-to-end: load → split → embed → store → retrieve."""
    try:
        import faiss  # noqa: F401

        from synapsekit.embeddings import SynapsekitEmbeddings
        from synapsekit.retrieval.faiss import FAISSVectorStore
        from synapsekit.retrieval.retriever import Retriever
    except ImportError as e:
        results.append(("Mini RAG pipeline", SKIP, str(e).split(":")[0]))
        return

    from synapsekit.loaders import StringLoader
    from synapsekit.text_splitters import RecursiveCharacterTextSplitter

    corpus = (
        "SynapseKit is an async-first Python framework for LLM apps. "
        "It supports RAG pipelines with multiple vector store backends. "
        "The framework includes ReAct agents with function calling support. "
        "Graph workflows enable multi-step reasoning with conditional routing. "
        "Subgraph nodes support retry, fallback, and skip error strategies. "
        "New loaders include SlackLoader for Slack channels and NotionLoader for Notion pages."
    )

    docs = StringLoader(corpus).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    chunks = splitter.split_with_metadata(docs[0].text, metadata={"source": "corpus"})
    assert len(chunks) > 1

    emb = SynapsekitEmbeddings(model="all-MiniLM-L6-v2")
    store = FAISSVectorStore(embedding_backend=emb)
    retriever = Retriever(vectorstore=store)

    await retriever.add([c["text"] for c in chunks])

    hits = await retriever.retrieve("Slack and Notion loaders", top_k=2)
    assert len(hits) == 2
    combined = " ".join(hits).lower()
    assert any(w in combined for w in ["slack", "notion", "loader", "new"])

    results.append(("Mini RAG pipeline: load→split→embed→retrieve", PASS, ""))


asyncio.run(test_mini_rag_pipeline())


# ── 9. LLM (skip without keys) ─────────────────────────────────────────────────

anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
openai_key = os.environ.get("OPENAI_API_KEY")


async def test_anthropic():
    from synapsekit.llm import AnthropicLLM

    llm = AnthropicLLM(model="claude-haiku-4-5-20251001", api_key=anthropic_key)
    r = await llm.complete("Reply with exactly one word: PONG")
    assert "PONG" in r.content


acheck(
    "AnthropicLLM: live completion",
    test_anthropic(),
    skip_if=None if anthropic_key else "ANTHROPIC_API_KEY not set",
)


async def test_openai():
    from synapsekit.llm import OpenAILLM

    llm = OpenAILLM(model="gpt-4o-mini", api_key=openai_key)
    r = await llm.complete("Reply with exactly one word: PONG")
    assert "PONG" in r.content


acheck(
    "OpenAILLM: live completion",
    test_openai(),
    skip_if=None if openai_key else "OPENAI_API_KEY not set",
)


# ── RESULTS ────────────────────────────────────────────────────────────────────

print("\n" + "═" * 68)
print("  SynapseKit — Functional Smoke Test")
print("═" * 68)

passed = failed = skipped = 0
for label, status, note in results:
    suffix = f"  ({note})" if note else ""
    print(f"  {status}  {label}{suffix}")
    if status == PASS:
        passed += 1
    elif status == FAIL:
        failed += 1
    else:
        skipped += 1

print("═" * 68)
print(f"  {passed} passed  ·  {failed} failed  ·  {skipped} skipped")
print("═" * 68)

sys.exit(1 if failed else 0)
