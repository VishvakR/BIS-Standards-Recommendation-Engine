import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.logger import Logger
from app.main import BISRecommendationEngine
from app.models.schemas import RecommendationResponse, StandardRecommendation

load_dotenv()
Logger.setup()
logger = logging.getLogger(__name__)

_bis_engine: Optional[BISRecommendationEngine] = None


def get_bis_engine() -> BISRecommendationEngine:
    """Return the cached BIS engine, creating it if necessary."""
    global _bis_engine
    if _bis_engine is None:
        logger.info("Initialising BISRecommendationEngine …")
        _bis_engine = BISRecommendationEngine()
        logger.info("BISRecommendationEngine ready.")
    return _bis_engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("BIS Standards API starting up …")
    yield
    logger.info("BIS Standards API shut down.")


app = FastAPI(
    title="BIS Standards Recommendation Engine API",
    description=(
        "RESTful API for BIS Standards Recommendation Engine. "
        "Ingest BIS SP 21 PDF, then query for relevant Indian Standards "
        "based on product descriptions."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════
# Request / Response models
# ═══════════════════════════════════════════════════════

class BISRecommendRequest(BaseModel):
    query: str = Field(
        ..., min_length=1,
        description="Product description to find applicable BIS standards for",
    )
    top_k: int = Field(
        5, ge=1, le=10,
        description="Number of standards to return",
    )
    generate_rationale: bool = Field(
        False,
        description="Whether to generate LLM rationale (slower)",
    )


class BISIngestRequest(BaseModel):
    pdf_path: str = Field(
        ..., description="Path to the BIS SP 21 PDF file",
    )


class BISIngestResponse(BaseModel):
    status: str
    chunks_indexed: int


# ═══════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════

@app.get("/health", tags=["Utility"])
async def health_check():
    return {"status": "ok", "service": "BIS Standards Recommendation Engine API"}


@app.post("/recommend", response_model=RecommendationResponse, tags=["BIS Standards"])
def recommend_endpoint(request: BISRecommendRequest):
    """Find applicable BIS standards for a product description."""
    try:
        logger.info(f"[/recommend] query={request.query!r} top_k={request.top_k}")
        engine = get_bis_engine()

        start = time.perf_counter()
        recs = engine.recommend(
            query=request.query,
            top_k=request.top_k,
            generate_rationale=request.generate_rationale,
        )
        latency = round(time.perf_counter() - start, 4)

        return RecommendationResponse(
            query=request.query,
            recommendations=recs,
            latency_seconds=latency,
        )
    except Exception as exc:
        logger.exception("[/recommend] Unexpected error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/ingest", response_model=BISIngestResponse, tags=["BIS Standards"])
def bis_ingest_endpoint(request: BISIngestRequest):
    """Ingest a BIS SP 21 PDF into the vector store."""
    try:
        logger.info(f"[/ingest] pdf_path={request.pdf_path}")
        from app.ingestion.bis_pdf_ingestor import BISPDFIngestor

        ingestor = BISPDFIngestor()
        num_chunks = ingestor.ingest(request.pdf_path)

        # Reset the BIS engine so it picks up new data
        global _bis_engine
        _bis_engine = None

        return BISIngestResponse(status="ok", chunks_indexed=num_chunks)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("[/ingest] Unexpected error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
