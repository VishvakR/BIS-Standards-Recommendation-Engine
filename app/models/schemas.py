"""
Pydantic models and dataclasses for the BIS Standards Recommendation Engine.
"""

from dataclasses import dataclass
from typing import List

from pydantic import BaseModel, Field


@dataclass
class BISResult:
    """
    Internal dataclass returned by the HybridBISRetriever.
    Represents a single matched BIS standard with its best chunk context.
    """
    standard_id: str          # e.g. "IS 269: 1989"
    title: str                # e.g. "Ordinary Portland Cement - Specification"
    category: str             # e.g. "Cement"
    scope: str                # Brief scope from the BIS document
    relevance_score: float    # 0.0–1.0, from cross-encoder reranker
    context_chunk: str        # Best matching chunk text


class StandardRecommendation(BaseModel):
    """A single BIS standard recommendation returned to the client."""
    standard_id: str = Field(
        ..., description="BIS standard identifier, e.g. 'IS 269: 1989'"
    )
    title: str = Field(
        ..., description="Full title of the BIS standard"
    )
    category: str = Field(
        ..., description="Building-material category: Cement, Steel, Concrete, or Aggregates"
    )
    scope: str = Field(
        ..., description="Brief 1-2 line scope from the BIS document"
    )
    rationale: str = Field(
        default="",
        description="LLM-generated explanation of why this standard applies"
    )
    relevance_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="Relevance score from the cross-encoder reranker"
    )


class RecommendationResponse(BaseModel):
    """API response for the /recommend endpoint."""
    query: str
    recommendations: List[StandardRecommendation] = Field(
        ..., max_length=5,
        description="Top BIS standard recommendations (max 5)"
    )
    latency_seconds: float = Field(
        ..., description="End-to-end processing time in seconds"
    )
