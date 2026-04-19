from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any

import ir_datasets
import yaml


def whitespace_len(text: str | None) -> int:
    if not text:
        return 0
    return len(text.split())


def get_doc_text(doc: Any) -> str:
    for field in ("text", "body", "contents"):
        if hasattr(doc, field):
            value = getattr(doc, field)
            if isinstance(value, str) and value.strip():
                return value
    parts: list[str] = []
    for field in ("title", "subtitle", "abstract"):
        if hasattr(doc, field):
            value = getattr(doc, field)
            if isinstance(value, str) and value.strip():
                parts.append(value.strip())
    return "\n\n".join(parts)


def get_query_text(query: Any) -> str:
    for field in ("text", "query", "title", "description"):
        if hasattr(query, field):
            value = getattr(query, field)
            if isinstance(value, str) and value.strip():
                return value
    return str(query)


def safe_count(handler: Any) -> int | None:
    if handler is None:
        return None
    if hasattr(handler, "count"):
        try:
            return int(handler.count())
        except Exception:
            return None
    return None


def summarize_lengths(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"sampled": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "sampled": len(values),
        "mean": round(mean(values), 2),
        "median": median(values),
        "min": min(values),
        "max": max(values),
    }


def profile_dataset(dataset_id: str, max_doc_samples: int, max_query_samples: int, preview_docs: int, preview_queries: int) -> dict[str, Any]:
    dataset = ir_datasets.load(dataset_id)
    docs_handler = dataset.docs_handler()
    queries_handler = dataset.queries_handler()
    qrels_handler = dataset.qrels_handler()

    doc_lengths: list[int] = []
    query_lengths: list[int] = []
    doc_preview: list[dict[str, Any]] = []
    query_preview: list[dict[str, Any]] = []
    qrels_per_query: Counter[str] = Counter()
    relevance_hist: Counter[int] = Counter()

    if docs_handler is not None:
        for idx, doc in enumerate(dataset.docs_iter()):
            text = get_doc_text(doc)
            doc_lengths.append(whitespace_len(text))
            if len(doc_preview) < preview_docs:
                doc_preview.append({
                    "doc_id": getattr(doc, "doc_id", None),
                    "title": getattr(doc, "title", None),
                    "text_preview": text[:300],
                })
            if idx + 1 >= max_doc_samples:
                break

    if queries_handler is not None:
        for idx, query in enumerate(dataset.queries_iter()):
            text = get_query_text(query)
            query_lengths.append(whitespace_len(text))
            if len(query_preview) < preview_queries:
                query_preview.append({
                    "query_id": getattr(query, "query_id", None),
                    "text": text,
                })
            if idx + 1 >= max_query_samples:
                break

    qrels_count = 0
    if qrels_handler is not None:
        for qrel in dataset.qrels_iter():
            qrels_count += 1
            qrels_per_query[str(qrel.query_id)] += 1
            relevance_hist[int(qrel.relevance)] += 1

    qrels_density_values = list(qrels_per_query.values())

    return {
        "dataset_id": dataset_id,
        "docs_count": safe_count(docs_handler),
        "queries_count": safe_count(queries_handler),
        "qrels_count": qrels_count,
        "doc_length_ws": summarize_lengths(doc_lengths),
        "query_length_ws": summarize_lengths(query_lengths),
        "qrels_per_query": summarize_lengths(qrels_density_values),
        "relevance_hist": dict(sorted(relevance_hist.items())),
        "doc_preview": doc_preview,
        "query_preview": query_preview,
        "topology_notes": [
            "docs, queries, qrels form the core retrieval topology",
            "qrels_per_query approximates supervision density",
            "doc/query length skews often predict baseline failure modes",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile IR datasets via ir_datasets")
    parser.add_argument("--config", default="configs/datasets.yaml")
    parser.add_argument("--profile", default="starter_small")
    parser.add_argument("--outdir", default="outputs")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    dataset_ids = config["profiles"][args.profile]
    sampling = config["sampling"]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    results = [
        profile_dataset(
            dataset_id=dataset_id,
            max_doc_samples=int(sampling["max_doc_samples"]),
            max_query_samples=int(sampling["max_query_samples"]),
            preview_docs=int(sampling["preview_docs"]),
            preview_queries=int(sampling["preview_queries"]),
        )
        for dataset_id in dataset_ids
    ]

    json_path = outdir / f"dataset_profile_{args.profile}.json"
    csv_path = outdir / f"dataset_profile_{args.profile}.csv"

    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset_id",
                "docs_count",
                "queries_count",
                "qrels_count",
                "doc_len_mean_ws",
                "doc_len_median_ws",
                "query_len_mean_ws",
                "query_len_median_ws",
                "qrels_per_query_mean",
                "qrels_per_query_median",
            ],
        )
        writer.writeheader()
        for row in results:
            writer.writerow({
                "dataset_id": row["dataset_id"],
                "docs_count": row["docs_count"],
                "queries_count": row["queries_count"],
                "qrels_count": row["qrels_count"],
                "doc_len_mean_ws": row["doc_length_ws"]["mean"],
                "doc_len_median_ws": row["doc_length_ws"]["median"],
                "query_len_mean_ws": row["query_length_ws"]["mean"],
                "query_len_median_ws": row["query_length_ws"]["median"],
                "qrels_per_query_mean": row["qrels_per_query"]["mean"],
                "qrels_per_query_median": row["qrels_per_query"]["median"],
            })

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
