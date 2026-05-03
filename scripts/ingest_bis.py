#!/usr/bin/env python3
"""
BIS SP 21 PDF Ingestion Pipeline
Usage: python scripts/ingest_bis.py --pdf data/bis_sp21.pdf
"""

import argparse
import json
import sys
import os
from pathlib import Path
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.ingestion.bis_pdf_ingestor import (
    extract_text_from_pdf,
    parse_standards,
    ingest_to_chromadb,
)



def main():
    parser = argparse.ArgumentParser(description="Ingest BIS SP 21 PDF into ChromaDB")
    parser.add_argument("--pdf", required=True, help="Path to BIS SP 21 PDF")
    parser.add_argument("--save-parsed", default="data/parsed_standards.json",
                        help="Save parsed standards as JSON for inspection")
    args = parser.parse_args()

    if not os.path.isfile(args.pdf):
        print(f"ERROR: PDF not found: {args.pdf}")
        sys.exit(1)

    # Step 1: Extract
    print(f"Step 1: Extracting text from {args.pdf}...")
    full_text = extract_text_from_pdf(args.pdf)
    print(f"  Extracted {len(full_text):,} characters")

    # Step 2: Parse
    print("Step 2: Parsing standard blocks...")
    standards = parse_standards(full_text)
    print(f"  Found {len(standards)} standards")

    cats = Counter(s["category"] for s in standards)
    for cat, count in sorted(cats.items()):
        print(f"    {cat}: {count}")

    # Save parsed JSON for inspection
    Path(args.save_parsed).parent.mkdir(parents=True, exist_ok=True)
    with open(args.save_parsed, "w") as f:
        json.dump(standards, f, indent=2, ensure_ascii=False)
    print(f"  Saved parsed standards to {args.save_parsed}")

    # Spot-check critical IDs
    key_ids = [
        "IS 269: 1989", "IS 383: 1970", "IS 8112: 1989",
        "IS 458: 2003", "IS 1489 (Part 1): 1991", "IS 2185 (Part 1): 1979",
    ]
    found_ids = {s["standard_id"] for s in standards}
    print("\n  Spot-check critical standards:")
    for kid in key_ids:
        status = "✓ FOUND" if kid in found_ids else "✗ MISSING"
        print(f"    {kid}: {status}")

    # Step 3: Ingest
    print("\nStep 3: Ingesting into ChromaDB...")
    total_chunks = ingest_to_chromadb(standards)
    print(f"\n✅ Done! {total_chunks} chunks indexed from {len(standards)} standards.")


if __name__ == "__main__":
    main()
