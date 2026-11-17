import json
import socket

import pytest

from app.retrieval import search


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Unexpected network")

    monkeypatch.setattr(socket.socket, "connect", forbidden)


@pytest.mark.parametrize(
    "query,k", [("", 5), ("  ", 5), ("q", 0), ("q", -1), ("q", True), ("q", 1.5)]
)
def test_invalid_arguments(query, k):
    with pytest.raises(ValueError):
        search.search(query, k)


def test_cli(monkeypatch, capsys):
    monkeypatch.setattr(search, "search", lambda query, k: [])
    assert search.main([" question ", "--top-k", "2"]) == 0
    assert json.loads(capsys.readouterr().out) == {"query": "question", "results": []}

    def fail(*args):
        raise RuntimeError("secret")

    monkeypatch.setattr(search, "search", fail)
    assert search.main(["question"]) == 1
    captured = capsys.readouterr()
    assert not captured.out
    assert "RuntimeError" in captured.err and "secret" not in captured.err


def test_similarity_cutoff_is_inclusive(monkeypatch, capsys):
    hits = [{"score": 0.8}, {"score": 0.5}, {"score": 0.49}]
    monkeypatch.setattr(search, "rank_candidates", lambda *args, **kwargs: hits)
    monkeypatch.setenv("MIN_SIMILARITY", "0.5")
    assert search.search("question") == hits[:2]
    assert search.main(["question"]) == 0
    assert json.loads(capsys.readouterr().out)["results"] == hits[:2]
    monkeypatch.setenv("MIN_SIMILARITY", "0.9")
    assert search.search("question") == []
