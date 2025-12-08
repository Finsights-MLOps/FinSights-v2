# ModelPipeline/serving/frontend/sidebar.py
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
                
                # Update state based on health check
                if health.get("status") == "healthy":
                    set_backend_health(True)
                    st.session_state.backend_error = None
                else:
                    set_backend_health(False)
                    st.session_state.backend_error = health.get("error", "Unknown error")
                
                # Force rerun to display updated status below
                st.rerun()

        # Display current status
        backend_healthy = st.session_state.get("backend_healthy")
        if backend_healthy is True:
            st.success("✓ Backend Online")
        elif backend_healthy is False:
            error_msg = st.session_state.get("backend_error", "Unknown")
            st.error(f"✗ Backend Offline: {error_msg}")
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
            - "From 2016 to 2022, show me financial performance for Radian Group, Exxon Mobil, Netflix, Costco, and Mastercard. I need: Revenue, net income, operating cash flow, gross profit, return on assets, debt-to-assets ratio, and stockholders' equity for each company by year."
            """)

        with st.expander("Complex Analysis", expanded=False):
            st.markdown("""
            - "Across its fiscal 2017-2020 10-K filings, how does Walmart Inc. explain the main drivers behind changes in its long-term debt and related cash flows from financing activities?"
            - "How does Tesla, Apple and MICROSOFT CORP describe the change in their cloud or AI revenues in 2017, including both the direction and magnitude of the change?"
            - "Talk to me about Exxon Mobil's risk data and business overview in 2022"
            - "How do Radian Group, Netflix and Mastercard each describe their exposure to data protection, information security and customer privacy risks?"
            """)
        
        with st.expander("Financial Metrics (KPI)", expanded=False):
            st.markdown("""
            **High-Coverage Queries** (Recommended):
                Show revenue, net income, operating cash flow, and ROA for Apple, Microsoft from 2018-2022
                        
            **Multi-Company Comparison**:
                Compare total sales, bottom line profits, and shareholder equity for Netflix, Costco, Mastercard between 2016-2022        
                **Note**: EPS, tax expense, and some metrics are not available as of now. 
            """)

        with st.expander("Available Metrics", expanded=False):
            st.markdown("""
            ### Reference on triggers: Financial Metrics
            - The keywords/phrases on the right, trigger extraction of the standardized financial metric on the left.
                                    
            **Income Statement:**
            - **Revenue** → sales, total sales, revenues, top line, sales revenue
            - **Net Revenue** → net sales
            - **Gross Revenue** → gross sales
            - **Net Income** → profit, earnings, bottom line, net profit, net earnings, income
            - **Gross Profit** → gross income, gross margins, gross margin dollars
            
            **Cash Flow:**
            - **Operating Cash Flow** → OCF, CFO, cash from operations, operating activities, cash flow operations, cashflow
            - **Operating Cash Flow - Continuing Ops** → OCF continuing operations
            
            **Balance Sheet - Equity:**
            - **Stockholders' Equity** → shareholders equity, shareholder equity, total equity, equity, book value, net worth
            - **Equity Including NCI** → equity with noncontrolling interest
            - **Other Stockholders' Equity** → other equity
            - **Other Equity Shares**
            
            ---
            
            ### Financial Ratios (Derived)
            
            **Profitability:**
            - **ROA %** → return on assets, return on average assets
            - **ROE %** → return on equity, return on average equity
            - **Operating Margin %** → EBIT margin, op margin
            - **Net Profit Margin %** → net margin, profit margin, NPM
            
            **Leverage:**
            - **Debt-to-Assets** → debt to asset ratio, debt levels, total debt levels, total debt, leverage
            - **Debt-to-Equity** → debt to equity ratio, leverage ratio
            
            ---
            
            ### Balance Sheet - Liabilities
            
            **General Liabilities:**
            - **Other Liabilities** → liabilities
            - **Other Liabilities (Noncurrent)**
            - **Other Liabilities (Current)**
            - **Other Liabilities - Fair Value**
            - **Other Sundry Liabilities**
            - **Other Accrued Liabilities (Current)** → accrued liabilities
            
            **Specialized Liabilities:**
            - **Derivative Liabilities (Current)** → derivative liabilities
            - **Employee-Related Liabilities** → employee liabilities
            - **Pension & Benefit Liabilities** → pension liabilities, pension benefits
            - **Business Combination - Liabilities Assumed** → acquisition liabilities
            - **Current Liabilities - Disposal Group** → disposal liabilities
            
            ---
            
            ### Balance Sheet - Assets
            
            **Other Assets:**
            - **Other Assets (Noncurrent)** → other noncurrent assets
            - **Other Current Assets** → other assets current
            - **Other Assets - Fair Value**
            - **Miscellaneous Assets (Noncurrent)** → misc assets
            
            **Asset Categories:**
            - **Prepaid Expenses (Current)** → prepaid
            - **Intangible Assets (Gross)** → intangibles, intangible assets
            - **Securities Loaned** → loaned securities
            - **Separate Account Assets**
            
            **Asset Changes:**
            - **Impairment - Indefinite-Lived Intangibles** → intangible impairment
            - **Impairment - Long-Lived Assets** → asset impairment
            - **Change in Other Assets**
            
            ---
            
            ### Tax Metrics (Deferred Tax)
            
            **Deferred Tax Assets:**
            - **DTA - Net** → deferred tax assets net, net deferred tax assets
            - **DTA - NOL** → deferred tax NOL, NOL carryforwards
            - DTA - NOL Domestic
            - DTA - NOL Foreign
            - DTA - NOL State/Local
            - **DTA - Derivatives** → deferred tax assets derivatives
            - **DTA - PPE** → deferred tax assets property plant equipment
            - **DTA - Deferred Income**
            - **DTA - IPR&D** → deferred tax research development
            - **DTA - OCI Loss** → deferred tax other comprehensive income
            - **DTA - Unrealized Losses (AFS)**
            - **DTA - Capital Loss CF**
            - **DTA - Other**
            
            **Deferred Tax Liabilities:**
            - **DTL - Derivatives**
            - **DTL - PPE**
            - **DTL - Goodwill & Intangibles** → deferred tax goodwill
            - **DTL - Noncontrolled Affiliates**
            - **DTL - Undistributed Foreign Earnings** → deferred tax foreign earnings
            - **DTL - Unrealized Gains (Trading)**
            
            **Tax Valuation:**
            - **Deferred Tax Asset Valuation Allowance Change** → tax valuation allowance
            - **Tax Rate - Change in Valuation Allowance**
            
            ---
            
            ### Capital Expenditures & Investing
            
            **Capex:**
            - **Payments to Acquire Productive Assets** → capex, capital expenditures, capex productive
            - **Payments to Acquire Intangible Assets** → capex intangibles, intangible capex
            - **Net Payments/Proceeds - Productive Assets** → net capex
            
            **Proceeds:**
            - **Proceeds from Productive Asset Sales** → asset sale proceeds
            - **Proceeds from Other Asset Sales (Investing)**
            - **Proceeds from Sale of Businesses / Assets** → business sale proceeds
            
            **Segment:**
            - **Segment Additions to Long-Lived Assets** → segment capex, segment additions
            
            ---
            
            ### Specialized Metrics
            
            **Variable Interest Entities (VIE):**
                - **VIE Consolidated Assets**, **VIE Consolidated Liabilities**, **VIE Liabilities (No Recourse)**, **VIE Assets Pledged**, **VIE Non-consolidated Assets**
            
            **Other:**
            - **Financial Asset Transfers Derecognized** → asset transfers derecognized
            - **Share-based Payment Liabilities Paid** → stock compensation paid
            
            ---
            
            ### Net Income Variants (Specialized)
            
            **By Share Class:**
            - **Net Income Available to Common - Basic** → net income common basic
            - **Net Income Available to Common - Diluted** → net income diluted
            - **Net Income Attributable to Parent - Diluted** → parent net income
            
            **By Ownership:**
            - **Net Income - Noncontrolling Interest** → NCI income
            - **Net Income - Nonredeemable NCI**
            - **Net Income - Redeemable NCI**
            - **Net Income - Including Nonredeemable NCI**
            
            **LP Units:**
            - **Net Income per LP Unit (Basic)** → income per LP unit
            - **Net Income per LP Unit (Basic & Diluted)**
            
            **Unavailable**:
            - EPS/Earnings Per Share, Income Tax Expense, etc.
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