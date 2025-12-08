# ModelPipeline\serving\frontend\app.py

"""
FinSight - Financial Document Intelligence System

---------------------------------------------------------------------------
cd ModelPipeline
.\\serving\\frontend\\venv_frontend\\Scripts\\Activate.ps1
streamlit run .\\serving\\frontend\\app.py
---------------------------------------------------------------------------
"""

import streamlit as st
from api_client import FinSightClient
from state import init_session_state, auto_check_backend_health
from config import BACKEND_URL, API_TIMEOUT

# Import styling
from components.styles import inject_global_css

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="FinSight Intelligence",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="collapsed"  # Hidden for home, shown for chatbot
)

# ============================================================================
# INJECT GLOBAL CSS
# ============================================================================

inject_global_css()

# ============================================================================
# INITIALIZATION
# ============================================================================

init_session_state()

# Initialize page navigation
if "page" not in st.session_state:
    st.session_state.page = "Home"

@st.cache_resource
def get_api_client():
    """Get API client instance (singleton)."""
    return FinSightClient(base_url=BACKEND_URL, timeout=API_TIMEOUT)

client = get_api_client()

# auto-check backend health on first load
auto_check_backend_health(client)


# ============================================================================
# NAVIGATION BAR
# ============================================================================

col_left, col_right = st.columns([1, 1])

with col_left:
    # Logo
    st.markdown("""
    <div class="nav-logo">
        <div class="logo-dot"></div>
        <div class="logo-text">Fin<span class="highlight">sights</span></div>
    </div>
    """, unsafe_allow_html=True)

with col_right:
    # Navigation buttons
    nav_col1, nav_col2 = st.columns(2)
    
    with nav_col1:
        if st.button("Home", key="nav_home", use_container_width=True):
            st.session_state.page = "Home"
            st.rerun()
    
    with nav_col2:
        if st.button("Chatbot", key="nav_chat", type="primary", use_container_width=True):
            st.session_state.page = "Chatbot"
            st.rerun()

st.divider()

# ============================================================================
# PAGE ROUTING
# ============================================================================

if st.session_state.page == "Home":
    from pages.home import render_home
    render_home()

elif st.session_state.page == "Chatbot":
    from chat import render_chat_history, handle_user_input
    from sidebar import render_sidebar
    
    # Render enhanced sidebar
    render_sidebar(client)
    
    # Clean main chat interface
    st.title("FinSight Assistant")
    st.markdown("Ask questions about SEC 10-K financial filings")
    
    st.divider()
    
    # Chat area (clean, no extra UI elements)
    render_chat_history()
    handle_user_input(client)
    
    # Footer below input bar
    st.divider()
    st.caption("FinSight v1.0 | IE7374 MLOps Capstone Project")
    st.caption("Built with Streamlit + FastAPI + AWS Bedrock")