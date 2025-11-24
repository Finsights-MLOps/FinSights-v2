# frontend/app.py
"""
FinRAG - Financial Document Intelligence System

Main Streamlit application that provides a chat interface for querying
SEC 10-K filings using RAG (Retrieval-Augmented Generation).

Usage:
    streamlit run frontend/app.py --server.port 8501

Architecture:
    Browser → Streamlit (port 8501) → FastAPI (port 8000) → ML Pipeline
"""

import streamlit as st
from api_client import FinRAGClient
from state import (
    init_session_state,
    set_backend_health
)
from chat import (
    render_chat_history,
    handle_user_input,
    render_clear_button
)
from metrics import display_sidebar_stats


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="FinRAG Assistant",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================================
# INITIALIZATION
# ============================================================================

# Initialize session state
init_session_state()

# Initialize API client (cached so it's only created once)
@st.cache_resource
def get_api_client():
    """Get API client instance (singleton)."""
    return FinRAGClient(
        base_url="http://localhost:8000",
        timeout=120
    )

client = get_api_client()


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.title("🏦 FinRAG")
    st.markdown("**Financial Document Intelligence**")
    st.markdown("---")
    
    # Backend health check
    st.markdown("### 🔧 System Status")
    
    if st.button("🔄 Check Backend", use_container_width=True):
        with st.spinner("Checking backend..."):
            health = client.health_check()
            
            if health.get("status") == "healthy":
                set_backend_health(True)
                st.success("✅ Backend: Healthy")
            else:
                set_backend_health(False)
                st.error(f"❌ Backend: {health.get('error', 'Offline')}")
    
    # Display current status
    backend_healthy = st.session_state.get("backend_healthy")
    if backend_healthy is True:
        st.success("✅ Backend: Online")
    elif backend_healthy is False:
        st.error("❌ Backend: Offline")
    else:
        st.info("⚠️ Status: Unknown")
    
    st.markdown("---")
    
    # Statistics
    display_sidebar_stats()
    
    st.markdown("---")
    
    # Clear chat button
    render_clear_button()
    
    st.markdown("---")
    
    # About section - NATIVE STREAMLIT EXPANDER (one emoji in heading only)
    with st.expander("ℹ️ About"):
        st.markdown("""
        **FinRAG** - Financial Document Intelligence System
        
        Analyzes SEC 10-K filings using hybrid retrieval architecture
        
        **Core Capabilities:**
        - Data: Uses SEC Edgar 10-K filings sentence-level data
        - RAG Search: Semantic retrieval, uses AWS S3 vectors, AWS S3 storage, Bedrock for Claude
        - Entity/Metadata/KPI Extraction: Financial metrics from EdgarTools
        - LLM Synthesis: Powered by Claude, Embeddings by Cohere v4
        - Store: Parquet-based (99% cost savings vs. managed DBs)
        - Query Cost: ~$0.01 per query
        
        **Dataset Coverage:**
        - Companies: Currently only 21 companies across 2015-2020 years! Expanding soon
        - Document Types: 10-K annual filings
        
        **Project:** IE7374 MLOps Capstone | Northeastern University
        """)
    
    st.markdown("---")
    
    # Example questions - NATIVE STREAMLIT EXPANDER (one emoji in heading only)
    with st.expander("💡 Example Questions"):
        st.markdown("""
        **Complex Analysis:**
        
        *"Across its fiscal 2018-2020 10-K filings, how does Walmart Inc. explain the main drivers behind changes in its long-term debt and related cash flows from financing activities?"*
        
        *"In their 2020 Form 10-K risk-factor disclosures, how do Radian Group, Netflix and Mastercard each describe their exposure to data protection, information security and customer privacy risks?"*
        
        **Comparative Metrics:**
        
        *"How does MICROSOFT CORP describe the change in its Intelligent Cloud revenue in 2017, including both the direction and magnitude of the change?"*
        
        *"Show me Apple, Microsoft, Amazon, Alphabet, Google, and Tesla's financial performance from 2018 to 2022. I need their total sales, bottom line profits, operating cash flows, gross margins, total debt levels, shareholder equity, cost of goods sold, tax expenses, return on assets, and earnings per share. How did these companies explain their revenue growth and profitability trends in their MD&A sections, and what supply chain or competitive risks did they highlight in their Risk Factors?"*
                
        **Guidelines:**
        - 10-500 characters
        - Takes 10-15 seconds
        - Cost: ~$0.017/query
        """)



# ============================================================================
# MAIN CONTENT
# ============================================================================

# Header
st.title("💬 FinRAG Assistant")
st.markdown("Ask questions about SEC 10-K financial filings")
st.markdown("---")

# Check if backend is healthy (show warning if not checked yet)
if st.session_state.get("backend_healthy") is None:
    st.warning("⚠️ Backend status unknown. Click '🔄 Check Backend' in the sidebar to verify connection.")

# Render chat history
render_chat_history()

# Handle user input (chat input box at bottom)
handle_user_input(client)


# ============================================================================
# FOOTER (NATIVE STREAMLIT - NO HTML INJECTION)
# ============================================================================

st.markdown("---")

# Use native Streamlit columns for footer layout
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.caption("FinRAG v1.0 | IE7374 MLOps Capstone Project")
    st.caption("Built with Streamlit + FastAPI + AWS Bedrock")