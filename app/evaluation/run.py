"""Benchmark retrieval against labeled Markdown in an isolated test schema."""

import argparse
import json
import math
import os
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from uuid import uuid4

import psycopg
from psycopg import sql
from psycopg.conninfo import make_conninfo

from app.ingestion.sync import run_sync
from app.retrieval.search import SearchSettings, rank_candidates, search

TOP_K = 5


def evidence_key(source, headings):
    return source, tuple(headings)


def run_questions(questions, database_url, *, calibrate=False):
    retrieved = []
    latencies = []
    retrieve = rank_candidates if calibrate else search
    for question in questions:
        start = perf_counter()
        results = retrieve(question["query"], TOP_K, database_url=database_url)
        latencies.append((perf_counter() - start) * 1000)
        retrieved.append(results)

    cutoff = SearchSettings().min_similarity
    if calibrate:
        negatives = [
            hit["score"]
            for question, results in zip(questions, retrieved, strict=True)
            if not question["expected"]
            for hit in results
        ]
        if not negatives:
            raise ValueError("Calibration requires ranked missing-evidence cases")
        cutoff = math.nextafter(max(negatives), math.inf)
        if cutoff > 1:
            raise ValueError("No cosine cutoff can reject all missing-evidence cases")
        retrieved = [
            [hit for hit in results if hit["score"] >= cutoff] for results in retrieved
        ]

    outcomes = []
    for question, results in zip(questions, retrieved, strict=True):
        expected = {evidence_key(*reference) for reference in question["expected"]}
        matches = set()
        ranks = []
        for rank, result in enumerate(results, 1):
            key = evidence_key(result["source"], result["headings"])
            frontmatter = result["metadata"]["frontmatter"]
            if (
                key in expected
                and all(
                    frontmatter.get(name) == value
                    for name, value in question.get("frontmatter", {}).items()
                )
                and all(
                    excerpt in result["text"]
                    for source, headings, excerpt in question.get("passages", [])
                    if evidence_key(source, headings) == key
                )
            ):
                matches.add(key)
                ranks.append(rank)
        outcomes.append(
            {
                "query": question["query"],
                "case": question.get("case", "ordinary"),
                "recall": len(matches) / len(expected) if expected else None,
                "reciprocal_rank": 1 / ranks[0] if ranks else (0 if expected else None),
                "returned": [
                    evidence_key(hit["source"], hit["headings"]) for hit in results
                ],
            }
        )

    labeled = [outcome for outcome in outcomes if outcome["recall"] is not None]
    recall = mean(outcome["recall"] for outcome in labeled)
    mrr = mean(outcome["reciprocal_rank"] for outcome in labeled)
    return {
        "recall_at_5": recall,
        "mrr_at_5": mrr,
        "median_search_ms": median(latencies),
        "p95_search_ms": sorted(latencies)[math.ceil(0.95 * len(latencies)) - 1],
        "min_similarity": cutoff,
        "passed": all(
            not outcome["returned"] for outcome in outcomes if outcome["recall"] is None
        )
        and recall >= 0.90
        and mrr >= 0.80
        and all(
            outcome["recall"] == 1
            for outcome in labeled
            if outcome["case"] in {"contradictory", "expired", "review"}
        ),
        "outcomes": outcomes,
    }


def evaluate(database_url, documents, questions, *, calibrate=False):
    schema = sql.Identifier("evaluation_" + uuid4().hex)
    with psycopg.connect(database_url, autocommit=True) as connection:
        connection.execute("CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public")
        connection.execute(sql.SQL("CREATE SCHEMA {}").format(schema))
        try:
            isolated_url = make_conninfo(
                database_url,
                options=f"-csearch_path={schema.as_string(connection)},public",
            )
            start = perf_counter()
            counts, errors = run_sync(documents, database_url=isolated_url)
            indexing_ms = (perf_counter() - start) * 1000
            if errors or not counts["indexed"]:
                raise ValueError("Evaluation corpus sync failed")
            report = run_questions(questions, isolated_url, calibrate=calibrate)
            report["indexing_ms"] = indexing_ms
            return report
        finally:
            connection.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(schema))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--calibrate", action="store_true")
    args = parser.parse_args(argv)
    url = os.environ.get("TEST_DATABASE_URL")
    if not url or url == os.environ.get("DATABASE_URL"):
        parser.error("TEST_DATABASE_URL must identify a dedicated test database")
    try:
        questions = json.loads(args.dataset.read_text())
        documents = args.dataset.with_name(args.dataset.stem + "_documents")
        split = "calibration" if args.calibrate else "held_out"
        questions = [q for q in questions if q.get("split", "held_out") == split]
        if not questions or not any(q["expected"] for q in questions):
            raise ValueError(f"{split} requires positive evidence cases")
        runs = [
            evaluate(url, documents, questions, calibrate=args.calibrate)
            for _ in range(1 if args.calibrate else 3)
        ]
        timings = {
            key: median(run[key] for run in runs)
            for key in ("p95_search_ms", "indexing_ms")
        }
        regressions = []
        if args.baseline:
            baseline = json.loads(args.baseline.read_text())["timings"]
            regressions = [
                key for key, value in timings.items() if value > 1.25 * baseline[key]
            ]
    except ValueError as error:
        parser.error(str(error))
    except Exception as error:  # noqa: BLE001 - avoid leaking backend connection details
        parser.error(type(error).__name__)

    passed = all(run["passed"] for run in runs) and not regressions
    print(
        json.dumps(
            {
                "runs": runs,
                "timings": timings,
                "regressions": regressions,
                "passed": passed,
            }
        )
    )
    return int(not passed)


if __name__ == "__main__":
    raise SystemExit(main())
