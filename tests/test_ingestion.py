import json
import socket
from pathlib import Path

import pytest
import yaml

from app.ingestion import sync
from app.ingestion.markdown import discover, extract, sections


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Unexpected network request")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.delenv("DATABASE_URL", raising=False)


def write(root, text, name="sample.md"):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_headings_and_formatting(tmp_path):
    text = "Preamble\n\n# Main\n\n## Child\n```python\n# code\n    x = 1\n```\n\n    # indented code\n\nSibling\n-------\ntext\n"
    doc = extract(write(tmp_path, text), tmp_path)
    assert "".join(chunk["text"] for chunk in doc.chunks) == text
    assert [chunk["metadata"]["headings"] for chunk in doc.chunks] == [
        [],
        ["Main"],
        ["Main", "Child"],
        ["Main", "Sibling"],
    ]


def test_ids_and_bounds(tmp_path):
    text = (
        "# Ünicode\n"
        + "A long paragraph. " * 150
        + "\n## Repeat\nSame\n## Repeat\nSame\n"
    )
    a = tmp_path / "a"
    b = tmp_path / "b"
    first = extract(write(a, text, "nested/sample.md"), a)
    second = extract(write(b, text, "nested/sample.md"), b)
    assert first == second
    assert len({chunk["id"] for chunk in first.chunks}) == len(first.chunks)
    assert all(len(chunk["text"]) <= 600 for chunk in first.chunks)
    assert all(
        chunk["metadata"]["source"] == "nested/sample.md" for chunk in first.chunks
    )
    assert (
        extract(write(a, text + "edit", "nested/sample.md"), a).source_digest
        != first.source_digest
    )


def test_frontmatter_provenance(tmp_path):
    doc = extract(
        write(
            tmp_path,
            "---\nsource: original.pdf\nexpires_on: 2020-01-01\nvalid_from: 2019-01-01\nchunk_id: untrusted\n---\n# Certificate\nExpired\n",
        ),
        tmp_path,
    )
    metadata = doc.chunks[0]["metadata"]
    assert metadata["source"] == "sample.md"
    assert metadata["frontmatter"]["source"] == "original.pdf"
    assert metadata["frontmatter"]["expires_on"] == "2020-01-01"
    assert metadata["frontmatter"]["valid_from"] == "2019-01-01"
    assert metadata["chunk_id"] != "untrusted"


@pytest.mark.parametrize(
    "text",
    [
        "---\ninvalid",
        "---\n- list\n---\nbody",
        "---\na: [\n---\nbody",
        "---\n1: value\n---\n",
        "---\na: .nan\n---\n",
    ],
)
def test_bad_frontmatter(tmp_path, text):
    with pytest.raises((ValueError, yaml.YAMLError)):
        extract(write(tmp_path, text), tmp_path)


def test_empty_and_discovery(tmp_path):
    assert extract(write(tmp_path, ""), tmp_path).chunks == []
    write(tmp_path, "hello", "nested/B.MD")
    write(tmp_path, "ignored", "other.txt")
    (tmp_path / "link.md").symlink_to(tmp_path / "sample.md")
    (tmp_path / "linked").symlink_to(tmp_path / "nested", target_is_directory=True)
    assert [path.relative_to(tmp_path).as_posix() for path in discover(tmp_path)] == [
        "nested/B.MD",
        "sample.md",
    ]


def test_invalid_utf8(tmp_path):
    path = tmp_path / "invalid.md"
    path.write_bytes(b"\xff")
    with pytest.raises(UnicodeError):
        extract(path, tmp_path)


def test_cli_inspection_and_failure(tmp_path, capsys):
    write(tmp_path, "# Hello\nEvidence\n", "good.md")
    write(tmp_path, "---\ninvalid", "bad.md")
    assert sync.main(["--path", str(tmp_path), "--inspect"]) == 1
    out, err = capsys.readouterr()
    assert json.loads(out)["metadata"]["source"] == "good.md"
    assert json.loads(err.splitlines()[-1]) == {
        "indexed": 0,
        "skipped": 0,
        "failed": 1,
        "inspected": 1,
    }


def test_cli_index_failures_and_repeat(tmp_path, monkeypatch, capsys):
    write(tmp_path, "# Good\nEvidence", "good.md")
    write(tmp_path, "# Bad\nFailure", "bad.md")
    stored = {}

    class Embeddings:
        model = "fake"

        def embed_documents(self, texts):
            if "Failure" in texts[0]:
                raise RuntimeError("private text")
            return [[1.0, 2.0] for _ in texts]

    class Store:
        def __init__(self, *args):
            pass

        def unchanged(self, doc):
            return doc.document_id in stored

        def add(self, doc, vectors):
            stored[doc.document_id] = doc
            return True

        def close(self):
            pass

    monkeypatch.setenv("DATABASE_URL", "fake")
    monkeypatch.setattr(sync, "IngestionStore", Store)
    monkeypatch.setattr(sync, "create_embeddings", Embeddings)
    for expected in ({"indexed": 1, "skipped": 0}, {"indexed": 0, "skipped": 1}):
        assert sync.main(["--path", str(tmp_path)]) == 1
        _, err = capsys.readouterr()
        counts = json.loads(err.splitlines()[-1])
        assert all(counts[key] == value for key, value in expected.items())
        assert counts["failed"] == 1
        assert "private text" not in err
    assert len(stored) == 1


def test_missing_root_and_database(tmp_path, capsys):
    assert sync.main(["--path", str(tmp_path / "missing"), "--inspect"]) == 1
    assert sync.main(["--path", str(tmp_path)]) == 1
    assert "DATABASE_URL is required" in capsys.readouterr().err


def test_fence_lengths():
    body = "# Real\n````\n```\n# Still code\n````\n# Next\ntext\n"
    assert [headings for headings, _ in sections(body)] == [["Real"], ["Next"]]


def test_synthetic_examples():
    root = (
        Path("examples/documents")
        if Path("examples/documents").exists()
        else Path("/fixtures")
    )
    documents = [extract(path, root) for path in discover(root)]
    assert len(documents) == 3
    certification = next(doc for doc in documents if doc.source == "certification.md")
    assert all(
        chunk["metadata"]["frontmatter"]["expires_on"] == "2026-01-01"
        for chunk in certification.chunks
    )
