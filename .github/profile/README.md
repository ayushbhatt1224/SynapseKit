<div align="center">

# SynapseKit

### Ship LLM apps faster.

**Production-grade LLM framework for Python.**
Async-native RAG, agents, and graph workflows. 2 dependencies. Zero magic.

<br/>

[![PyPI version](https://img.shields.io/pypi/v/synapsekit?color=0a7bbd&label=pypi&logo=pypi&logoColor=white)](https://pypi.org/project/synapsekit/)
[![Downloads](https://img.shields.io/pypi/dm/synapsekit?color=0a7bbd&logo=pypi&logoColor=white)](https://pypi.org/project/synapsekit/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-22c55e)](https://github.com/SynapseKit/SynapseKit/blob/main/LICENSE)
[![Tests](https://img.shields.io/badge/tests-1752%20passing-22c55e?logo=pytest&logoColor=white)](https://github.com/SynapseKit/SynapseKit)
[![GitHub Stars](https://img.shields.io/github/stars/SynapseKit/SynapseKit?style=social)](https://github.com/SynapseKit/SynapseKit)

<br/>

[Documentation](https://synapsekit.github.io/synapsekit-docs/) &bull; [Quickstart](https://synapsekit.github.io/synapsekit-docs/docs/getting-started/quickstart) &bull; [EvalCI](https://synapsekit.github.io/synapsekit-docs/docs/evalci/overview) &bull; [Roadmap](https://synapsekit.github.io/synapsekit-docs/docs/roadmap) &bull; [Contributing](https://github.com/SynapseKit/SynapseKit/blob/main/CONTRIBUTING.md)

</div>

<br/>

---

<br/>

<div align="center">

### Why SynapseKit?

</div>

<table>
<tr>
<td width="50%">

**The problem:** Existing LLM frameworks are heavy — 50+ dependencies, hidden chains, magic callbacks, YAML configs. Hard to debug, harder to ship.

**The fix:** SynapseKit gives you everything you need to build production LLM apps with just **2 core dependencies** and plain Python you can actually read.

</td>
<td width="50%">

```bash
pip install synapsekit[openai]
```

```python
from synapsekit import RAG

rag = RAG(model="gpt-4o-mini", api_key="sk-...")
rag.add("Your document text here")
print(rag.ask_sync("What is the main topic?"))
```

**3 lines. That's it.**

</td>
</tr>
</table>

<br/>

---

<br/>

<div align="center">

### What's inside

</div>

<table>
<tr>
<td align="center" width="33%">
<h4>RAG Pipelines</h4>
9 text splitters &bull; 33 loaders<br/>
20+ retrieval strategies &bull; conversation memory<br/>
streaming retrieval-augmented generation
</td>
<td align="center" width="33%">
<h4>Agents</h4>
ReAct &bull; native function calling<br/>
OpenAI, Anthropic, Gemini, Mistral + more<br/>
46 built-in tools &bull; fully extensible
</td>
<td align="center" width="33%">
<h4>Graph Workflows</h4>
parallel execution &bull; conditional routing<br/>
human-in-the-loop &bull; checkpointing<br/>
Mermaid export &bull; subgraphs &bull; SSE streaming
</td>
</tr>
<tr>
<td align="center">
<h4>30 LLM Providers</h4>
OpenAI &bull; Anthropic &bull; Gemini &bull; Groq<br/>
DeepSeek &bull; Mistral &bull; Ollama &bull; Bedrock<br/>
and 22 more — one interface, swap anytime
</td>
<td align="center">
<h4>9 Vector Stores</h4>
InMemory &bull; ChromaDB &bull; FAISS<br/>
Qdrant &bull; Pinecone &bull; Weaviate<br/>
PGVector &bull; Milvus &bull; LanceDB
</td>
<td align="center">
<h4>🧪 EvalCI</h4>
LLM quality gates on every PR<br/>
GitHub Action · zero infra · 2-min setup<br/>
<a href="https://github.com/marketplace/actions/evalci-by-synapsekit">GitHub Marketplace →</a>
</td>
</tr>
</table>

<br/>

---

<br/>

<div align="center">

### EvalCI — Stop shipping quality regressions

</div>

<table>
<tr>
<td width="50%">

EvalCI is a GitHub Action that runs your `@eval_case` suites on every pull request and blocks merge if quality drops below threshold. No infrastructure, no backend — just add 5 lines to your workflow.

```yaml
- uses: SynapseKit/evalci@v1
  with:
    path: tests/evals
    threshold: "0.80"
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

</td>
<td width="50%">

```python
# tests/evals/eval_rag.py
from synapsekit import eval_case

@eval_case(min_score=0.80, max_cost_usd=0.01)
async def eval_rag_relevancy():
    result = await pipeline.ask("What is SynapseKit?")
    score = await result.score_relevancy(reference=...)
    return {"score": score, "cost_usd": result.cost_usd}
```

[Get started →](https://synapsekit.github.io/synapsekit-docs/docs/evalci/overview)

</td>
</tr>
</table>

<br/>

---

<br/>

<div align="center">

### Growing fast

**1752 tests** &bull; **30 LLM providers** &bull; **46 tools** &bull; **33 loaders** &bull; **Contributors welcome** &bull; **Apache 2.0 Licensed**

We're building the most comprehensive async-native LLM framework in Python.
Whether you're a seasoned open-source contributor or looking for your first PR — jump in.

<br/>

[**Star the repo**](https://github.com/SynapseKit/SynapseKit) &bull; [**Browse good first issues**](https://github.com/SynapseKit/SynapseKit/issues?q=label%3A%22good+first+issue%22) &bull; [**Join the discussion**](https://github.com/SynapseKit/SynapseKit/discussions)

<br/>

</div>

---

<div align="center">

[Documentation](https://synapsekit.github.io/synapsekit-docs/) &bull; [EvalCI](https://synapsekit.github.io/synapsekit-docs/docs/evalci/overview) &bull; [PyPI](https://pypi.org/project/synapsekit/) &bull; [Changelog](https://github.com/SynapseKit/SynapseKit/blob/main/CHANGELOG.md) &bull; [Contributing Guide](https://github.com/SynapseKit/SynapseKit/blob/main/CONTRIBUTING.md)

</div>
