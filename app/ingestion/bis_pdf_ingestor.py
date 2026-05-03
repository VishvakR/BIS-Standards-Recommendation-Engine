"""
DATASET PDF Ingestion Pipeline.

Parses standards from dataset PDF and ingests them into ChromaDB.

PDF Structure:
  Each standard is preceded by "SUMMARY OF" on its own line.
  IS header: "IS XXXX : YYYY TITLE TEXT"
  Body with numbered sections, scope at "1. Scope —"
  Ends with "For detailed information, refer to IS XXXX..."
"""

import re
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path: str) -> str:
    """Use pdfplumber — the PDF is text-based (not scanned)."""
    import pdfplumber

    full_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text.append(text)
    return "\n".join(full_text)


def normalize_standard_id(raw_id: str) -> str:
    """
    Normalize raw PDF IDs to exact ground-truth format.

    "IS 383 : 1970"           → "IS 383: 1970"
    "IS 1489 (PART1) : 1991"  → "IS 1489 (Part 1): 1991"
    "IS 2185 (PART 1) : 1979" → "IS 2185 (Part 1): 1979"
    """
    pattern = r'IS\s+(\d+)\s*(?:\(PART\s*(\d+)\))?\s*:\s*(\d{4})'
    match = re.search(pattern, raw_id, re.IGNORECASE)
    if not match:
        return raw_id.strip()

    number = match.group(1)
    part = match.group(2)
    year = match.group(3)

    if part:
        return f"IS {number} (Part {part}): {year}"
    return f"IS {number}: {year}"


def extract_scope(body_text: str) -> str:
    """Extract scope — always begins after '1. Scope —'."""
    m = re.search(
        r'1\.\s+Scope\s*[—\-–]+\s*(.+?)(?=\n\s*\n|\n\s*2\.|\Z)',
        body_text, re.DOTALL | re.IGNORECASE,
    )
    if m:
        return re.sub(r'\s+', ' ', m.group(1).strip())[:500]
    return ""


def assign_category(standard_id: str, title: str, body: str) -> str:
    """Classify into Cement / Steel / Concrete / Aggregates / General."""
    text = (standard_id + " " + title + " " + body[:300]).upper()

    if any(k in text for k in [
        'CEMENT', 'PORTLAND', 'POZZOLANA', 'SLAG CEMENT',
        'MASONRY CEMENT', 'ALUMINA CEMENT', 'CLINKER',
        'SUPERSULPHATED', 'RAPID HARDENING', 'WHITE CEMENT',
        'HYDROPHOBIC', 'OIL WELL CEMENT',
    ]):
        return "Cement"

    if any(k in text for k in [
        'AGGREGATE', 'SAND', 'GRAVEL', 'CRUSHED STONE',
        'COARSE', 'FINE AGGREGATE', 'LIGHTWEIGHT AGGREGATE',
    ]):
        return "Aggregates"

    if any(k in text for k in [
        'REINFORCEMENT', 'STEEL BAR', 'STRUCTURAL STEEL',
        'MILD STEEL', 'HIGH YIELD', 'WIRE ROPE', 'STEEL WIRE',
        'WELDING', 'ELECTRODE', 'TMT', 'REBAR',
    ]):
        return "Steel"

    if any(k in text for k in [
        'CONCRETE', 'PRECAST', 'REINFORCED CONCRETE',
        'MASONRY UNIT', 'CONCRETE BLOCK', 'CONCRETE PIPE',
        'CONCRETE MIX', 'MORTAR', 'ADMIXTURE',
    ]):
        return "Concrete"

    return "General"


def parse_standards(full_text: str) -> List[Dict[str, str]]:
    """
    Split PDF text on "SUMMARY OF" markers and parse each standard block.
    Returns list of dicts: standard_id, title, category, scope, full_text.
    """
    blocks = re.split(r'\nSUMMARY OF\s*\n', full_text)

    standards = []
    seen_ids = set()

    for block in blocks[1:]:  # skip preamble
        lines = block.strip().split('\n')
        if not lines:
            continue

        # Find IS header line (first line starting with "IS ")
        header_idx = None
        for idx, line in enumerate(lines[:5]):
            if re.match(r'^IS\s+\d+', line.strip(), re.IGNORECASE):
                header_idx = idx
                break
        if header_idx is None:
            continue

        # Title may wrap to the next line
        header_line = lines[header_idx].strip()
        title_parts = [header_line]
        nxt = header_idx + 1
        if nxt < len(lines):
            nxt_line = lines[nxt].strip()
            if (nxt_line
                    and not nxt_line.startswith('(')
                    and not re.match(r'^\d+\.', nxt_line)
                    and not nxt_line.startswith('IS ')):
                title_parts.append(nxt_line)

        full_header = ' '.join(title_parts)

        # Extract IS number : year
        id_match = re.search(
            r'(IS\s+\d+(?:\s*\(PART\s*\d+\))?\s*:\s*\d{4})',
            full_header, re.IGNORECASE,
        )
        if not id_match:
            continue

        standard_id = normalize_standard_id(id_match.group(1))
        if standard_id in seen_ids:
            continue
        seen_ids.add(standard_id)

        # Title = everything after the IS id on the header line
        title_raw = full_header[id_match.end():].strip()
        title = re.sub(r'\s+', ' ', title_raw).title() or f"BIS Standard {standard_id}"

        # Body text — trim at "For detailed information"
        body_text = '\n'.join(lines[header_idx + 1:])
        cutoff = re.search(r'For detailed information.*?refer to', body_text, re.IGNORECASE)
        if cutoff:
            body_text = body_text[:cutoff.start()].strip()

        standards.append({
            "standard_id": standard_id,
            "title": title,
            "category": assign_category(standard_id, title, body_text),
            "scope": extract_scope(body_text),
            "full_text": body_text.strip(),
        })

    logger.info(f"Parsed {len(standards)} standards from PDF")
    return standards


def ingest_to_chromadb(
    standards: List[Dict[str, str]],
    collection_name: str = "bis_standards",
    persist_dir: str = "./storage/chromadb",
) -> int:
    """
    Chunk each standard and store in ChromaDB with metadata.
    Uses native chromadb client + SentenceTransformerEmbeddingFunction.
    Returns number of chunks indexed.
    """
    import chromadb
    from chromadb.utils import embedding_functions
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    client = chromadb.PersistentClient(path=persist_dir)

    # Clean slate
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="BAAI/bge-small-en-v1.5",
    )
    collection = client.create_collection(
        name=collection_name,
        embedding_function=emb_fn,
        metadata={"hnsw:space": "cosine"},
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512, chunk_overlap=64,
        separators=["\n\n", "\n", ". ", " "],
    )

    all_docs, all_ids, all_metas = [], [], []

    for std in standards:
        # Prepend metadata so even short chunks are contextualised
        rich_text = (
            f"Standard: {std['standard_id']}\n"
            f"Title: {std['title']}\n"
            f"Category: {std['category']}\n"
            f"Scope: {std['scope']}\n\n"
            f"{std['full_text']}"
        )
        chunks = splitter.split_text(rich_text)

        for ci, chunk in enumerate(chunks):
            doc_id = (
                std["standard_id"]
                .replace(" ", "_").replace(":", "_")
                .replace("(", "").replace(")", "")
                + f"_{ci}"
            )
            all_docs.append(chunk)
            all_ids.append(doc_id)
            all_metas.append({
                "standard_id": std["standard_id"],
                "title": std["title"],
                "category": std["category"],
                "scope": std["scope"][:200],
                "chunk_index": ci,
            })

    batch = 500
    for i in range(0, len(all_docs), batch):
        end = min(i + batch, len(all_docs))
        collection.add(
            documents=all_docs[i:end],
            ids=all_ids[i:end],
            metadatas=all_metas[i:end],
        )
        logger.info(f"  Batch {i // batch + 1}: {end}/{len(all_docs)} chunks")

    logger.info(f"Ingestion complete: {len(all_docs)} chunks from {len(standards)} standards")
    return len(all_docs)
