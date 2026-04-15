from __future__ import annotations

import asyncio
import os
import tempfile
from unittest.mock import patch

import pytest

from synapsekit.loaders import Document
from synapsekit.loaders.parquet import ParquetLoader

# ---------------------------------------------------------------------------
# Helpers — write real Parquet bytes using pyarrow (dev dependency)
# ---------------------------------------------------------------------------


def _write_parquet(rows: list[dict], path: str) -> None:
    """Write a list of dicts to a real Parquet file via pyarrow."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not rows:
        table = pa.table({})
    else:
        keys = list(rows[0].keys())
        columns = {k: [r.get(k) for r in rows] for k in keys}
        table = pa.table(columns)
    pq.write_table(table, path)


# ---------------------------------------------------------------------------
# Initialisation validation
# ---------------------------------------------------------------------------


def test_init_requires_path() -> None:
    with pytest.raises(ValueError, match="path must be provided"):
        ParquetLoader(path="")


def test_init_defaults() -> None:
    loader = ParquetLoader(path="/tmp/dummy.parquet")
    assert loader._text_fields is None
    assert loader._limit is None


# ---------------------------------------------------------------------------
# Missing dependency
# ---------------------------------------------------------------------------


def test_load_import_error_missing_pyarrow() -> None:
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        existing_path = f.name
    try:
        with patch.dict("sys.modules", {"pyarrow": None, "pyarrow.parquet": None}):
            loader = ParquetLoader(path=existing_path)
            with pytest.raises(ImportError, match="pyarrow required"):
                loader.load()
    finally:
        os.unlink(existing_path)


# ---------------------------------------------------------------------------
# Missing file
# ---------------------------------------------------------------------------


def test_load_raises_file_not_found() -> None:
    loader = ParquetLoader(path="/nonexistent/path/data.parquet")
    with pytest.raises(FileNotFoundError, match="Parquet file not found"):
        loader.load()


# ---------------------------------------------------------------------------
# Normal load — real Parquet files (pyarrow is a dev transitive dep)
# ---------------------------------------------------------------------------


def test_load_returns_documents() -> None:
    pytest.importorskip("pyarrow")

    rows = [
        {"title": "First Doc", "content": "Hello world"},
        {"title": "Second Doc", "content": "Parquet rocks"},
    ]

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        path = f.name

    try:
        _write_parquet(rows, path)
        loader = ParquetLoader(path=path)
        docs = loader.load()
    finally:
        os.unlink(path)

    assert len(docs) == 2
    assert all(isinstance(doc, Document) for doc in docs)
    assert "First Doc" in docs[0].text
    assert "Hello world" in docs[0].text


def test_load_metadata_correctness() -> None:
    pytest.importorskip("pyarrow")

    rows = [{"content": "Test row"}]

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        path = f.name

    try:
        _write_parquet(rows, path)
        loader = ParquetLoader(path=path)
        docs = loader.load()
    finally:
        os.unlink(path)

    assert len(docs) == 1
    assert docs[0].metadata["source"] == path
    assert docs[0].metadata["row"] == 0


def test_load_text_fields() -> None:
    pytest.importorskip("pyarrow")

    rows = [{"title": "My Title", "body": "My Body", "author": "Alice"}]

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        path = f.name

    try:
        _write_parquet(rows, path)
        loader = ParquetLoader(path=path, text_fields=["title", "body"])
        docs = loader.load()
    finally:
        os.unlink(path)

    assert len(docs) == 1
    assert docs[0].text == "My Title My Body"
    assert "Alice" not in docs[0].text


def test_load_respects_limit() -> None:
    pytest.importorskip("pyarrow")

    rows = [{"content": f"Row {i}"} for i in range(10)]

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        path = f.name

    try:
        _write_parquet(rows, path)
        loader = ParquetLoader(path=path, limit=3)
        docs = loader.load()
    finally:
        os.unlink(path)

    assert len(docs) == 3


def test_load_skips_empty_rows() -> None:
    pytest.importorskip("pyarrow")

    rows = [
        {"content": ""},
        {"content": "Valid content"},
        {"content": None},
    ]

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        path = f.name

    try:
        _write_parquet(rows, path)
        loader = ParquetLoader(path=path, text_fields=["content"])
        docs = loader.load()
    finally:
        os.unlink(path)

    assert len(docs) == 1
    assert docs[0].text == "Valid content"


def test_load_missing_text_field_is_skipped() -> None:
    pytest.importorskip("pyarrow")

    rows = [{"title": "Present", "body": None}]

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        path = f.name

    try:
        _write_parquet(rows, path)
        loader = ParquetLoader(path=path, text_fields=["title", "body"])
        docs = loader.load()
    finally:
        os.unlink(path)

    assert len(docs) == 1
    assert docs[0].text == "Present"
    assert "None" not in docs[0].text


def test_load_all_columns_when_no_text_fields() -> None:
    pytest.importorskip("pyarrow")

    rows = [{"a": "foo", "b": "bar", "c": "baz"}]

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        path = f.name

    try:
        _write_parquet(rows, path)
        loader = ParquetLoader(path=path)
        docs = loader.load()
    finally:
        os.unlink(path)

    assert len(docs) == 1
    assert "foo" in docs[0].text
    assert "bar" in docs[0].text
    assert "baz" in docs[0].text


def test_load_mixed_types() -> None:
    pytest.importorskip("pyarrow")

    rows = [{"name": "Alice", "score": 42, "active": True}]

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        path = f.name

    try:
        _write_parquet(rows, path)
        loader = ParquetLoader(path=path)
        docs = loader.load()
    finally:
        os.unlink(path)

    assert len(docs) == 1
    assert "Alice" in docs[0].text
    assert "42" in docs[0].text


# ---------------------------------------------------------------------------
# Async
# ---------------------------------------------------------------------------


def test_aload() -> None:
    pytest.importorskip("pyarrow")

    rows = [{"content": "Async parquet test"}]

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        path = f.name

    try:
        _write_parquet(rows, path)
        loader = ParquetLoader(path=path)
        docs = asyncio.run(loader.aload())
    finally:
        os.unlink(path)

    assert len(docs) == 1
    assert docs[0].text == "Async parquet test"
