"""
LLM-powered rationale generator for BIS standard recommendations.

Takes a product description and retrieved BIS results, calls the LLM to
generate a concise rationale for why each standard applies.

Used by the API and Streamlit UI (NOT by inference.py — the auto-scorer
only checks retrieved_standards, not rationale).
"""

import json
import logging
import re
from typing import List, Optional

from app.llm.main import LanguageModel
from app.models.schemas import BISResult

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a BIS (Bureau of Indian Standards) compliance expert specializing \
in building materials. Your role is to explain why specific Indian Standards \
apply to a given product."""

_USER_PROMPT_TEMPLATE = """\
Product Description: {product_description}

For each of the following BIS standards retrieved as potentially relevant, \
write a concise rationale (2–3 sentences) explaining exactly why this \
standard applies to the described product. Be specific to the product's \
properties. If a standard is NOT relevant, say "Not applicable" instead.

Standards to evaluate:
{standards_block}

Respond ONLY as valid JSON in this format:
[
  {{
    "standard_id": "IS XXX",
    "rationale": "...",
    "is_applicable": true
  }},
  ...
]"""


def _build_standards_block(results: List[BISResult]) -> str:
    lines = []
    for r in results:
        lines.append(
            f"- {r.standard_id}: {r.title}\n"
            f"  Scope: {r.scope}\n"
            f"  Context: {r.context_chunk[:300]}"
        )
    return "\n".join(lines)


def _extract_json_from_response(text: str) -> Optional[list]:
    """Robustly extract a JSON array from LLM output."""
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting JSON from markdown code block
    match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding array brackets
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None


class RationaleGenerator:
    """Generate LLM-powered rationale for BIS standard recommendations."""

    def __init__(self, llm: LanguageModel):
        self.llm = llm

    def generate(
        self,
        product_description: str,
        results: List[BISResult],
    ) -> List[BISResult]:
        """
        Enrich BISResult list with LLM-generated rationale strings.

        Filters to only applicable standards. Returns at most 5, ordered
        by original relevance_score.

        If the LLM call fails, returns the original results unchanged
        (with empty rationale).
        """
        if not results:
            return results

        standards_block = _build_standards_block(results)
        prompt = (
            _SYSTEM_PROMPT + "\n\n" +
            _USER_PROMPT_TEMPLATE.format(
                product_description=product_description,
                standards_block=standards_block,
            )
        )

        try:
            response = self.llm.generate_response(prompt)
            parsed = _extract_json_from_response(response)
        except Exception as e:
            logger.warning(f"LLM rationale generation failed: {e}")
            return results[:5]

        if parsed is None:
            logger.warning("Could not parse JSON from LLM response.")
            return results[:5]

        # Build lookup from LLM output
        rationale_map = {}
        for item in parsed:
            sid = item.get("standard_id", "")
            rationale_map[sid] = {
                "rationale": item.get("rationale", ""),
                "is_applicable": item.get("is_applicable", True),
            }

        # Enrich results
        enriched = []
        for r in results:
            llm_data = rationale_map.get(r.standard_id, {})
            is_applicable = llm_data.get("is_applicable", True)

            if is_applicable:
                enriched.append(
                    BISResult(
                        standard_id=r.standard_id,
                        title=r.title,
                        category=r.category,
                        scope=r.scope,
                        relevance_score=r.relevance_score,
                        context_chunk=r.context_chunk,
                    )
                )
                # Attach rationale as an attribute (BISResult is a dataclass)
                enriched[-1].rationale = llm_data.get("rationale", "")

        # Cap at 5, ordered by relevance_score
        enriched.sort(key=lambda r: r.relevance_score, reverse=True)
        return enriched[:5]
