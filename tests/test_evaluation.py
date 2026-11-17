from app.evaluation import run


def test_metrics_count_unique_evidence_and_skip_missing_questions(monkeypatch):
    def hit(heading):
        return {
            "source": "policy.md",
            "headings": [heading],
            "metadata": {"frontmatter": {}},
        }

    monkeypatch.setattr(
        run, "search", lambda *args, **kwargs: [hit("Other"), hit("A"), hit("A")]
    )
    report = run.run_questions(
        [
            {
                "query": "policy",
                "case": "contradictory",
                "expected": [["policy.md", ["A"]], ["policy.md", ["B"]]],
            },
            {"query": "missing", "case": "missing", "expected": []},
        ],
        "test",
    )
    assert report["recall_at_5"] == 0.5
    assert report["mrr_at_5"] == 0.5
    assert report["outcomes"][1]["recall"] is None
    assert report["passed"] is False
