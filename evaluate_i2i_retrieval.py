#!/usr/bin/env python3
"""
Evaluate image-to-image retrieval with Precision@K and Recall@K.

This script is intended for external query images only:
- Query set: local query images (e.g., your 25 user images)
- Ground truth: qrels CSV (query image -> a single relevant_vector_id)

Examples:
    python evaluate_i2i_retrieval.py --query-dir ./queries --qrels-csv ./qrels.csv --topk 1 5 10
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.vector_db import VectorDB
from src.data.utils import read_image_bytes


@dataclass
class QueryEval:
    query_vector_id: str
    true_label: str
    retrieved_ids: list[str]
    recall_by_k: dict[int, float]
    precision_at_1: float
    reciprocal_rank: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate i2i retrieval precision/recall@k")
    parser.add_argument(
        "--topk",
        type=int,
        nargs="+",
        default=[1, 3, 5],
        help="One or multiple K values, e.g. --topk 1 3 5",
    )
    parser.add_argument(
        "--query-limit",
        type=int,
        default=25,
        help="Number of query images to evaluate; use <=0 for all",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Random seed for query sampling",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional output JSON path for detailed report",
    )
    parser.add_argument(
        "--query-dir",
        type=str,
        default="",
        help="Required for external mode. Directory containing query images.",
    )
    parser.add_argument(
        "--qrels-csv",
        type=str,
        default="",
        help=(
            "Required for external mode. CSV ground truth with columns: "
            "query_image, relevant_vector_id. Multiple rows allowed per query_image."
        ),
    )
    parser.add_argument(
        "--query-pattern",
        type=str,
        default="**/*",
        help="Glob pattern under query-dir in external mode (default: **/*)",
    )
    return parser.parse_args()


def _build_single_label_ground_truth(qrels_df: pd.DataFrame) -> dict[str, str]:
    grouped: dict[str, set[str]] = defaultdict(set)
    for _, row in qrels_df.iterrows():
        query_image = str(row["query_image"]).strip()
        relevant_vector_id = str(row["relevant_vector_id"]).strip()
        if not query_image or not relevant_vector_id:
            continue
        grouped[query_image].add(relevant_vector_id)

    ambiguous = {query: sorted(labels) for query, labels in grouped.items() if len(labels) > 1}
    if ambiguous:
        examples = list(ambiguous.items())[:3]
        raise ValueError(
            "Single-label evaluation requires exactly one relevant_vector_id per query_image. "
            f"Found multiple labels for {len(ambiguous)} queries, examples: {examples}"
        )

    return {query: next(iter(labels)) for query, labels in grouped.items() if labels}


def _metrics_for_query(
    retrieved_ids: list[str],
    true_label: str,
    topk_list: list[int],
) -> tuple[dict[int, float], float, float]:
    recall_by_k: dict[int, float] = {}

    for k in topk_list:
        topk_ids = retrieved_ids[:k]
        recall_by_k[k] = 1.0 if true_label in topk_ids else 0.0

    precision_at_1 = 1.0 if retrieved_ids and retrieved_ids[0] == true_label else 0.0

    reciprocal_rank = 0.0
    for rank, retrieved_id in enumerate(retrieved_ids, start=1):
        if retrieved_id == true_label:
            reciprocal_rank = 1.0 / float(rank)
            break

    return recall_by_k, precision_at_1, reciprocal_rank


def evaluate_i2i_external(
    vdb: VectorDB,
    query_dir: str,
    qrels_csv: str,
    query_pattern: str,
    topk_list: list[int],
    query_limit: int,
    sample_seed: int,
) -> dict[str, Any]:
    qdir = Path(query_dir)
    if not qdir.exists() or not qdir.is_dir():
        raise ValueError(f"query-dir is invalid: {query_dir}")

    qrels_path = Path(qrels_csv)
    if not qrels_path.exists():
        raise ValueError(f"qrels-csv not found: {qrels_csv}")

    qrels_df = pd.read_csv(qrels_path)
    required_cols = {"query_image", "relevant_vector_id"}
    if not required_cols.issubset(qrels_df.columns):
        raise ValueError(
            "qrels-csv must contain columns: query_image, relevant_vector_id"
        )

    ground_truth = _build_single_label_ground_truth(qrels_df)

    supported_suffixes = {".png", ".jpg", ".jpeg", ".webp"}
    query_paths = [
        path
        for path in sorted(qdir.glob(query_pattern))
        if path.is_file() and path.suffix.lower() in supported_suffixes
    ]
    if not query_paths:
        raise ValueError(f"No query images found in {query_dir} with pattern: {query_pattern}")

    if query_limit and query_limit > 0 and query_limit < len(query_paths):
        random.seed(sample_seed)
        query_paths = random.sample(query_paths, query_limit)

    max_k = max(topk_list)
    per_query: list[QueryEval] = []
    skipped_no_label = 0

    for qpath in query_paths:
        qname = qpath.relative_to(qdir).as_posix()
        true_label = ground_truth.get(qname, "")
        if not true_label:
            skipped_no_label += 1
            continue

        base64_img = read_image_bytes(qpath)
        query_embedding = vdb._agentic_ml_client.compute_image_embedding(
            base64_image=base64_img,
            return_tensor="np",
        ).tolist()

        raw_results = vdb._agentic_images_img_similarity_search(
            query_embedding=query_embedding,
            top_k=max_k,
        )
        retrieved_ids = [str(item.image.vector_id) for item in raw_results]

        recall_by_k, precision_at_1, reciprocal_rank = _metrics_for_query(
            retrieved_ids=retrieved_ids,
            true_label=true_label,
            topk_list=topk_list,
        )

        per_query.append(
            QueryEval(
                query_vector_id=qname,
                true_label=true_label,
                retrieved_ids=retrieved_ids,
                recall_by_k=recall_by_k,
                precision_at_1=precision_at_1,
                reciprocal_rank=reciprocal_rank,
            )
        )

    if not per_query:
        raise RuntimeError("No valid external queries to evaluate.")

    metrics: dict[str, float] = {}
    for k in topk_list:
        metrics[f"Recall@{k}"] = sum(q.recall_by_k[k] for q in per_query) / len(per_query)
    metrics["Precision@1"] = sum(q.precision_at_1 for q in per_query) / len(per_query)
    metrics["MRR"] = sum(q.reciprocal_rank for q in per_query) / len(per_query)

    return {
        "config": {
            "mode": "external",
            "query_dir": str(qdir),
            "query_pattern": query_pattern,
            "query_key": "relative_path_from_query_dir",
            "qrels_csv": str(qrels_path),
            "ground_truth_mode": "single_label",
            "topk": topk_list,
            "query_limit": query_limit,
            "sample_seed": sample_seed,
            "evaluated_queries": len(per_query),
            "skipped_no_label": skipped_no_label,
        },
        "metrics": metrics,
        "per_query": [
            {
                "query": q.query_vector_id,
                "true_label": q.true_label,
                "retrieved_ids": q.retrieved_ids,
                "recall_by_k": q.recall_by_k,
                "precision_at_1": q.precision_at_1,
                "reciprocal_rank": q.reciprocal_rank,
            }
            for q in per_query
        ],
    }


def main() -> None:
    args = parse_args()
    topk_list = sorted(set(args.topk))
    if any(k <= 0 for k in topk_list):
        raise ValueError("All K values must be > 0")

    vdb = VectorDB()
    if not args.query_dir or not args.qrels_csv:
        raise ValueError("This script requires --query-dir and --qrels-csv")

    report = evaluate_i2i_external(
        vdb=vdb,
        query_dir=args.query_dir,
        qrels_csv=args.qrels_csv,
        query_pattern=args.query_pattern,
        topk_list=topk_list,
        query_limit=args.query_limit,
        sample_seed=args.sample_seed,
    )

    print("===== I2I Retrieval Evaluation =====")
    print(json.dumps(report["config"], ensure_ascii=False, indent=2))

    print("\n[Single-Label Metrics]")
    for key, value in report["metrics"].items():
        print(f"{key}: {value:.4f}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaved detailed report to: {out_path}")


if __name__ == "__main__":
    main()
