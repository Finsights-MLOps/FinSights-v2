"""
Enhanced sidebar for FinSight Chatbot.

Sections (in order):
1. System Status
2. Statistics
3. Best Results Tips
4. Example Queries
5. Clear Chat
"""

import streamlit as st
from api_client import FinSightClient
from state import set_backend_health, clear_chat_history
from metrics import display_sidebar_stats


def render_sidebar(client: FinSightClient) -> None:
    """
    Render complete sidebar with all sections.
    
    Args:
        client: FinSightClient instance for health checks
    """
    
    with st.sidebar:
        # Logo
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1.5rem;">
            <div style="width: 10px; height: 10px; border-radius: 50%; 
                        background: linear-gradient(135deg, #22c55e, #0ea5e9);"></div>
            <div style="font-size: 1.2rem; font-weight: 700; color: #e2e8f0;">
                Fin<span style="color: #22c55e;">sights</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        # ====================================================================
        # 1. SYSTEM STATUS
        # ====================================================================
        st.markdown("### System Status")
        
        if st.button("Check Backend", use_container_width=True, type="secondary"):
            with st.spinner("Checking..."):
                health = client.health_check()
                
                # if health.get("status") == "healthy":
                #     set_backend_health(True)
                #     st.success("Backend Online")
                # else:
                #     set_backend_health(False)
                #     st.error(f"Backend Offline: {health.get('error', 'Unknown')}")
        
        # Display current status
        backend_healthy = st.session_state.get("backend_healthy")
        if backend_healthy is True:
            st.success("✓ Backend Online")
        elif backend_healthy is False:
            st.error(f"✗ Backend Offline: {health.get('error', 'Unknown')}")
        else:
            st.info("⚬ Status Unknown")
        
        st.divider()
        
        # ====================================================================
        # 2. STATISTICS
        # ====================================================================
        display_sidebar_stats()
        
        st.divider()
        
        # ====================================================================
        # 3. BEST RESULTS TIPS
        # ====================================================================
        st.markdown("### Best Results")
        
        st.markdown("""
        **Query Guidelines:**
        - Mention the company name
        - Specify the filing year
        - Ask about specific metrics or trends
        - Phrase questions clearly
        
        **System Behavior:**
        - Typical response: 10-15 seconds
        - Cost per query: ~$0.01-0.02
        - Answers cite source sections
        """)
        
        st.divider()
        
        # ====================================================================
        # 4. EXAMPLE QUERIES
        # ====================================================================
        st.markdown("### Example Queries")
        
        with st.expander("KPI Queries", expanded=False):
            st.markdown("""
            - "Show me Apple, Microsoft, Amazon, Alphabet, Google, and Tesla's financial performance from 2018 to 2022. I need their total sales, bottom line profits, operating cash flows, gross margins, total debt levels, shareholder equity, cost of goods sold, tax expenses, return on assets, and earnings per share."
            """)
        

        with st.expander("Complex Analysis", expanded=False):
            st.markdown("""
            - "Across its fiscal 2017-2020 10-K filings, how does Walmart Inc. explain the main drivers behind changes in its long-term debt and related cash flows from financing activities?"
            - "How does Tesla, Apple and MICROSOFT CORP describe the change in their cloud or AI revenues in 2017, including both the direction and magnitude of the change?"
            - "Talk to me about Exxon Mobil's risk data and business overview in 2022"
            - "How do Radian Group, Netflix and Mastercard each describe their exposure to data protection, information security and customer privacy risks?"
            """)
        
        st.divider()
        
        # ====================================================================
        # 5. CLEAR CHAT
        # ====================================================================
        if st.button("Clear Chat History", use_container_width=True, type="secondary"):
            clear_chat_history()
            st.rerun()
        
        st.divider()
        
        # Disclaimer
        st.caption("**Disclaimer:** FinSight is a current student-prototype idea, a research tool, product showcase. It is not an investment advice tool. Always verify information independently.")