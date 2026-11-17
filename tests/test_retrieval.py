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
