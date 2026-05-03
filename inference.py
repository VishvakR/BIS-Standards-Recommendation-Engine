#!/usr/bin/env python3
"""
BIS Standards Recommendation Engine — Inference Entry Point

Usage:
    python inference.py --input dataset.json --output results.json
"""

import argparse
import json
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()


def load_engine():
    """Load and return the BIS engine. Import here so errors are caught."""
    try:
        from app.main import BISRecommendationEngine
        engine = BISRecommendationEngine()
        return engine
    except Exception as e:
        print(f"ERROR: Failed to load engine: {e}", file=sys.stderr)
        sys.exit(1)


def run_inference(input_path: str, output_path: str):
    print("Loading BIS Recommendation Engine...")
    engine = load_engine()
    print("Engine loaded. Starting inference...")

    try:
        with open(input_path, "r", encoding="utf-8") as f:
            queries = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in input: {e}", file=sys.stderr)
        sys.exit(1)

    results = []

    for i, item in enumerate(queries):
        query_id = item.get("id", f"unknown_{i}")
        query_text = item.get("query", "")

        if not query_text:
            result = {
                "id": query_id,
                "retrieved_standards": [],
                "latency_seconds": 0.0,
            }
            # Copy expected_standards from input if present (for eval)
            if "expected_standards" in item:
                result["expected_standards"] = item["expected_standards"]
            results.append(result)
            continue

        start_time = time.perf_counter()
        try:
            recs = engine.recommend(query_text, top_k=5)
            standard_ids = [rec.standard_id for rec in recs]
        except Exception as e:
            print(f"WARNING: Retrieval failed for id={query_id}: {e}")
            standard_ids = []

        latency = round(time.perf_counter() - start_time, 4)

        result = {
            "id": query_id,
            "retrieved_standards": standard_ids[:5],
            "latency_seconds": latency,
        }
        # Copy expected_standards from input if present (for eval)
        if "expected_standards" in item:
            result["expected_standards"] = item["expected_standards"]

        results.append(result)

        print(
            f"[{i + 1}/{len(queries)}] id={query_id} | "
            f"retrieved={len(standard_ids)} | latency={latency}s"
        )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nDone. {len(results)} queries → {output_path}")
    if results:
        avg = sum(r["latency_seconds"] for r in results) / len(results)
        print(f"Average latency: {avg:.3f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BIS Standards Recommendation Engine - Inference"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run_inference(args.input, args.output)
