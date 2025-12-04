"""

RESPONSIBILITIES:
✓ Display LLM metadata (tokens, cost, model)
✓ Show context metadata (length, sentence count)
✓ Render KPI/RAG flags
✓ Format processing time

DISPLAY PATTERN:
- st.expander("Query Details") for collapsible metadata
- Formatted tables or metrics for key stats
- Cost tracking (per-query and cumulative)

"""


# frontend/metrics.py
"""
Metrics display components for finSight frontend.

Handles rendering of query metadata, cost tracking, and performance stats.

Usage:
    from frontend.metrics import display_query_metadata, display_sidebar_stats
    
    display_query_metadata(metadata)
    display_sidebar_stats()
"""

# frontend/metrics.py
"""
Metrics display components for finSight frontend.

Handles rendering of query metadata, cost tracking, and performance stats.

Usage:
    from metrics import display_query_metadata, display_sidebar_stats
    
    display_query_metadata(metadata)
    display_sidebar_stats()
"""

import streamlit as st
from typing import Dict, Any, Optional


def display_query_metadata(metadata: Dict[str, Any]) -> None:
    """
    Display query metadata in an expandable section.
    
    Shows LLM info (tokens, cost, model) and context info (KPI/RAG flags).
    
    Args:
        metadata: Metadata dict from backend response
    """
    with st.expander("[+] Query Details", expanded=False):
        # Create two columns for better layout
        col1, col2 = st.columns(2)
        
        # Left column: LLM metadata
        with col1:
            st.markdown("**[LLM] Information**")
            
            llm = metadata.get("llm", {})
            
            # Model
            model_id = llm.get("model_id", "Unknown")
            # Shorten model ID for display
            model_display = model_id.split(":")[-1] if ":" in model_id else model_id
            st.text(f"Model: {model_display}")
            
            # Tokens
            input_tokens = llm.get("input_tokens", 0)
            output_tokens = llm.get("output_tokens", 0)
            total_tokens = llm.get("total_tokens", 0)
            
            st.text(f"Input Tokens: {input_tokens:,}")
            st.text(f"Output Tokens: {output_tokens:,}")
            st.text(f"Total Tokens: {total_tokens:,}")
            
            # Cost
            cost = llm.get("cost", 0.0)
            st.text(f"Cost: ${cost:.4f}")
        
        # Right column: Context metadata
        with col2:
            st.markdown("**[Context] Information**")
            
            ctx = metadata.get("context", {})
            
            # KPI/RAG flags
            kpi_included = ctx.get("kpi_included", False)
            rag_included = ctx.get("rag_included", False)
            
            kpi_status = "[YES]" if kpi_included else "[NO]"
            rag_status = "[YES]" if rag_included else "[NO]"
            
            st.text(f"KPI Lookup: {kpi_status}")
            st.text(f"RAG Search: {rag_status}")
            
            # Context length
            context_length = ctx.get("context_length", 0)
            sentence_count = ctx.get("sentence_count", 0)
            
            st.text(f"Context Length: {context_length:,} chars")
            if sentence_count > 0:
                st.text(f"Sentences: {sentence_count}")
        
        # Processing time (full width at bottom)
        processing_time_ms = metadata.get("processing_time_ms")
        if processing_time_ms:
            st.markdown("---")
            st.text(f"[TIME] Processing: {processing_time_ms:,.0f}ms ({processing_time_ms/1000:.1f}s)")


def display_sidebar_stats() -> None:
    """
    Display cumulative statistics in sidebar.
    
    Shows total queries, total cost, and backend health status.
    Uses session_state for data.
    """
    st.sidebar.markdown("### [Statistics]")
    
    # Total queries
    total_queries = st.session_state.get("total_queries", 0)
    st.sidebar.metric("Total Queries", total_queries)
    
    # Total cost
    total_cost = st.session_state.get("total_cost", 0.0)
    st.sidebar.metric("Total Cost", f"${total_cost:.4f}")
    
    # Average cost per query
    if total_queries > 0:
        avg_cost = total_cost / total_queries
        st.sidebar.metric("Avg Cost/Query", f"${avg_cost:.4f}")
    
    # Backend health indicator
    st.sidebar.markdown("---")
    st.sidebar.markdown("### [System Status]")
    
    backend_healthy = st.session_state.get("backend_healthy")
    
    if backend_healthy is True:
        st.sidebar.success("[OK] Backend: Healthy")
    elif backend_healthy is False:
        st.sidebar.error("[ERROR] Backend: Offline")
    else:
        st.sidebar.warning("[UNKNOWN] Backend: Not Checked")


def display_error_message(error: str, error_type: str, stage: str) -> None:
    """
    Display error message in a formatted error box.
    
    Args:
        error: Error message text
        error_type: Error classification
        stage: Pipeline stage where error occurred
    """
    st.error(f"**Error:** {error}")
    
    with st.expander("[+] Error Details"):
        st.text(f"Type: {error_type}")
        st.text(f"Stage: {stage}")
        
        # Helpful hints based on error type
        if error_type == "ConnectionError":
            st.info("[TIP] Make sure the backend server is running on http://localhost:8000")
        elif error_type == "TimeoutError":
            st.info("[TIP] The query is taking longer than expected. Complex queries may take 15-20 seconds.")
        elif error_type == "ValidationError":
            st.info("[TIP] Questions must be between 10-500 characters.")


def format_cost(cost: float) -> str:
    """
    Format cost for display.
    
    Args:
        cost: Cost in USD
    
    Returns:
        str: Formatted cost string
    """
    if cost < 0.01:
        return f"${cost:.6f}"  # Show more precision for very small costs
    else:
        return f"${cost:.4f}"


def format_tokens(tokens: int) -> str:
    """
    Format token count for display.
    
    Args:
        tokens: Token count
    
    Returns:
        str: Formatted token string with commas
    """
    return f"{tokens:,}"