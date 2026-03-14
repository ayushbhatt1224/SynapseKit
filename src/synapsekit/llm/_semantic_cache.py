"""Semantic LLM Cache: similarity-based cache lookup using embeddings."""

from __future__ import annotations

from typing import Any

import numpy as np


class SemanticCache:
    """Cache LLM responses using semantic similarity instead of exact match.

    Uses embeddings to find semantically similar prompts and returns
    cached responses when similarity exceeds a threshold.

    Usage::

        from synapsekit.llm._semantic_cache import SemanticCache
        from synapsekit import SynapsekitEmbeddings

        embeddings = SynapsekitEmbeddings()
        cache = SemanticCache(embeddings=embeddings, threshold=0.92)

        # Store a response
        cache.put("What is Python?", "Python is a programming language.")

        # Later, a semantically similar query hits the cache
        result = cache.get("Tell me about Python")
        # result → "Python is a programming language."
    """

    def __init__(
        self,
        embeddings: Any,
        threshold: float = 0.92,
        maxsize: int = 256,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")
        if maxsize < 1:
            raise ValueError("maxsize must be >= 1")
        self._embeddings = embeddings
        self._threshold = threshold
        self._maxsize = maxsize
        self._entries: list[dict[str, Any]] = []
        self._vectors: list[np.ndarray] = []
        self.hits: int = 0
        self.misses: int = 0

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        dot = float(np.dot(a, b))
        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    async def get(self, prompt: str) -> str | None:
        """Look up a semantically similar prompt in the cache.

        Returns the cached response if similarity >= threshold, else None.
        """
        if not self._entries:
            self.misses += 1
            return None

        query_vec = await self._embeddings.embed(prompt)
        query_arr = np.array(query_vec, dtype=np.float32)

        best_score = -1.0
        best_idx = -1

        for i, vec in enumerate(self._vectors):
            score = self._cosine_similarity(query_arr, vec)
            if score > best_score:
                best_score = score
                best_idx = i

        if best_score >= self._threshold:
            self.hits += 1
            result: str = self._entries[best_idx]["response"]
            return result

        self.misses += 1
        return None

    async def put(self, prompt: str, response: str) -> None:
        """Store a prompt-response pair in the cache."""
        vec = await self._embeddings.embed(prompt)
        arr = np.array(vec, dtype=np.float32)

        self._entries.append({"prompt": prompt, "response": response})
        self._vectors.append(arr)

        # Evict oldest if over maxsize
        if len(self._entries) > self._maxsize:
            self._entries.pop(0)
            self._vectors.pop(0)

    def clear(self) -> None:
        """Clear all cached entries."""
        self._entries.clear()
        self._vectors.clear()

    def __len__(self) -> int:
        return len(self._entries)
