from __future__ import annotations

import json
import math
import pathlib
from statistics import mean
from typing import Any

import pandas as pd
import yaml
from beir.datasets.data_loader import GenericDataLoader


def load_config(config_path: pathlib.Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def percentile(sorted_values: list[int], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    idx = (len(sorted_values) - 1) * q
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return float(sorted_values[lo])
    frac = idx - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def safe_text_len(title: str | None, text: str | None) -> int:
    return len((title or "").strip()) + len((text or "").strip())


def sample_dict_items(d: dict[str, Any], limit: int) -> list[tuple[str, Any]]:
    return list(d.items())[:limit]


def main() -> None:
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    config = load_config(repo_root / "configs" / "datasets.yaml")

    base_data_dir = repo_root / config["base_data_dir"]
    output_dir = repo_root / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    doc_sample_limit = int(config.get("sample_docs_per_dataset", 1000))
    query_sample_limit = int(config.get("sample_queries_per_dataset", 200))
    examples_limit = int(config.get("sample_examples_per_dataset", 5))

    rows: list[dict[str, Any]] = []
    examples: dict[str, Any] = {}

    for ds in config["datasets"]:
        if not ds.get("enabled", True):
            continue

        dataset_name = ds["name"]
        split = ds.get("split", "test")
        data_path = base_data_dir / dataset_name

        if not data_path.exists():
            raise FileNotFoundError(
                f"Dataset folder not found: {data_path}. Run download_beir_datasets.py first."
            )

        corpus, queries, qrels = GenericDataLoader(data_folder=str(data_path)).load(split=split)

        sampled_docs = sample_dict_items(corpus, doc_sample_limit)
        sampled_queries = sample_dict_items(queries, query_sample_limit)

        doc_lengths = []
        title_missing = 0
        text_missing = 0

        for _, doc in sampled_docs:
            title = doc.get("title", "")
            text = doc.get("text", "")
            if not (title or "").strip():
                title_missing += 1
            if not (text or "").strip():
                text_missing += 1
            doc_lengths.append(safe_text_len(title, text))

        query_lengths = [len((q or "").strip()) for _, q in sampled_queries]

        qrels_per_query = [len(v) for v in qrels.values()]
        total_qrels = sum(qrels_per_query)

        doc_lengths_sorted = sorted(doc_lengths)
        query_lengths_sorted = sorted(query_lengths)

        rows.append(
            {
                "dataset": dataset_name,
                "split": split,
                "docs": len(corpus),
                "queries": len(queries),
                "qrels_total": total_qrels,
                "avg_qrels_per_query": round(mean(qrels_per_query), 3) if qrels_per_query else 0.0,
                "queries_with_single_rel": sum(1 for x in qrels_per_query if x == 1),
                "queries_with_single_rel_share": round(
                    sum(1 for x in qrels_per_query if x == 1) / len(qrels_per_query), 4
                ) if qrels_per_query else 0.0,
                "sampled_docs": len(sampled_docs),
                "sampled_queries": len(sampled_queries),
                "doc_chars_avg": round(mean(doc_lengths), 1) if doc_lengths else 0.0,
                "doc_chars_p50": round(percentile(doc_lengths_sorted, 0.50), 1) if doc_lengths_sorted else 0.0,
                "doc_chars_p90": round(percentile(doc_lengths_sorted, 0.90), 1) if doc_lengths_sorted else 0.0,
                "query_chars_avg": round(mean(query_lengths), 1) if query_lengths else 0.0,
                "query_chars_p50": round(percentile(query_lengths_sorted, 0.50), 1) if query_lengths_sorted else 0.0,
                "query_chars_p90": round(percentile(query_lengths_sorted, 0.90), 1) if query_lengths_sorted else 0.0,
                "title_missing_rate": round(title_missing / len(sampled_docs), 4) if sampled_docs else 0.0,
                "text_missing_rate": round(text_missing / len(sampled_docs), 4) if sampled_docs else 0.0,
            }
        )

        examples[dataset_name] = {
            "documents": [
                {
                    "doc_id": doc_id,
                    "title": doc.get("title", ""),
                    "text_preview": (doc.get("text", "") or "")[:500],
                }
                for doc_id, doc in sampled_docs[:examples_limit]
            ],
            "queries": [
                {"query_id": query_id, "text": query}
                for query_id, query in sampled_queries[:examples_limit]
            ],
            "qrels": [
                {"query_id": query_id, "rels": rels}
                for query_id, rels in list(qrels.items())[:examples_limit]
            ],
        }

    df = pd.DataFrame(rows).sort_values(["docs", "queries"], ascending=[True, True])
    csv_path = output_dir / "beir_dataset_profile.csv"
    json_path = output_dir / "beir_dataset_profile.json"
    examples_path = output_dir / "beir_examples.json"

    df.to_csv(csv_path, index=False, encoding="utf-8")
    df.to_json(json_path, orient="records", force_ascii=False, indent=2)
    with examples_path.open("w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)

    print(f"Saved profile CSV: {csv_path}")
    print(f"Saved profile JSON: {json_path}")
    print(f"Saved examples JSON: {examples_path}")


if __name__ == "__main__":
    main()
