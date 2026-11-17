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


def test_missing_evidence_is_a_release_gate(monkeypatch):
    hit = {
        "source": "policy.md",
        "headings": ["A"],
        "score": 0.75,
        "text": "Retain records for 90 days.",
        "metadata": {"frontmatter": {}},
    }
    questions = [
        {
            "query": "retention",
            "expected": [["policy.md", ["A"]]],
            "passages": [["policy.md", ["A"], "90 days"]],
        },
        {"query": "missing", "expected": []},
    ]
    monkeypatch.setattr(run, "search", lambda *args, **kwargs: [hit])
    report = run.run_questions(questions, "test")
    assert report["recall_at_5"] == report["mrr_at_5"] == 1
    assert not report["passed"]

    def candidates(query, *args, **kwargs):
        return [hit if query == "retention" else {**hit, "score": 0.5}]

    monkeypatch.setattr(run, "rank_candidates", candidates)
    report = run.run_questions(questions, "test", calibrate=True)
    assert report["passed"]
    assert 0.5 < report["min_similarity"] <= 0.75
