"""
BIS Standards Finder — Streamlit UI

Run with:  streamlit run ui/app.py
"""

import sys
import os
import time

# Ensure project root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ───────────────────────────────────────────────────────
# Page config
# ───────────────────────────────────────────────────────

st.set_page_config(
    page_title="BIS Standards Finder",
    page_icon="🏗️",
    layout="wide",
)

# ───────────────────────────────────────────────────────
# Custom CSS
# ───────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        max-width: 1000px;
    }

    /* Header */
    .header-title {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        color: white;
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 2rem;
    }
    .header-title h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
    }
    .header-title p {
        margin: 0.5rem 0 0;
        opacity: 0.8;
        font-size: 1rem;
    }

    /* Standard card */
    .std-card {
        background: #f8f9fa;
        border-left: 4px solid #0f3460;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }
    .std-card .std-id {
        font-size: 1.3rem;
        font-weight: 700;
        color: #0f3460;
    }
    .std-card .std-title {
        font-size: 0.95rem;
        color: #333;
        margin: 0.3rem 0 0.8rem;
    }

    /* Category badge */
    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        color: white;
        margin-left: 8px;
        vertical-align: middle;
    }
    .badge-cement     { background: #e74c3c; }
    .badge-steel      { background: #3498db; }
    .badge-concrete   { background: #2ecc71; }
    .badge-aggregates { background: #f39c12; }
    .badge-general    { background: #9b59b6; }

    /* Latency footer */
    .latency-footer {
        text-align: center;
        color: #888;
        font-size: 0.85rem;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────────────
# Header
# ───────────────────────────────────────────────────────

st.markdown("""
<div class="header-title">
    <h1>🏗️ BIS Standards Finder for Building Materials</h1>
    <p>Find applicable Bureau of Indian Standards for your products</p>
</div>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────────────
# Engine (cached)
# ───────────────────────────────────────────────────────

@st.cache_resource
def load_engine():
    """Load the BIS engine once and cache across reruns."""
    from app.main import BISRecommendationEngine
    return BISRecommendationEngine()


# ───────────────────────────────────────────────────────
# Input form
# ───────────────────────────────────────────────────────

col1, col2 = st.columns([3, 1])

with col1:
    query = st.text_area(
        "Describe your product",
        placeholder="e.g. High-strength concrete mix for bridge construction",
        height=120,
    )

with col2:
    category_filter = st.selectbox(
        "Filter by category",
        ["All", "Cement", "Steel", "Concrete", "Aggregates"],
    )
    top_k = st.slider("Results", min_value=1, max_value=10, value=5)

search_clicked = st.button("🔍 Find Applicable Standards", use_container_width=True)

# ───────────────────────────────────────────────────────
# Results
# ───────────────────────────────────────────────────────

if search_clicked and query.strip():
    with st.spinner("Searching BIS standards …"):
        try:
            engine = load_engine()
            start = time.perf_counter()
            recs = engine.recommend(query.strip(), top_k=top_k)
            latency = time.perf_counter() - start

            # Apply category filter
            if category_filter != "All":
                recs = [r for r in recs if r.category == category_filter]

            if not recs:
                st.warning("No matching standards found. Try a different description.")
            else:
                for rec in recs:
                    badge_cls = f"badge-{rec.category.lower()}"
                    score_pct = int(rec.relevance_score * 100)

                    st.markdown(f"""
                    <div class="std-card">
                        <span class="std-id">{rec.standard_id}</span>
                        <span class="badge {badge_cls}">{rec.category}</span>
                        <div class="std-title">{rec.title}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    if rec.scope:
                        st.caption(f"**Scope:** {rec.scope}")

                    if rec.rationale:
                        st.info(f"**Rationale:** {rec.rationale}")

                    st.progress(score_pct, text=f"Relevance: {rec.relevance_score:.2%}")
                    st.divider()

            st.markdown(
                f'<div class="latency-footer">⏱️ Query completed in {latency:.3f}s</div>',
                unsafe_allow_html=True,
            )

        except Exception as e:
            st.error(f"Error: {e}")

elif search_clicked:
    st.warning("Please enter a product description.")
