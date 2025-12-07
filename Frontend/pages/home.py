"""
FinSights Home Page - Product landing page with enhanced UI.

Uses custom components for icons, cards, and animations.
"""

import streamlit as st
from components.html_builders import icon, feature_card, metric_card, fade_in


def render_home():
    """Render complete homepage with hero, metrics, features, and examples."""
    
    # ========================================================================
    # HERO SECTION
    # ========================================================================
    
    st.markdown('<div class="hero">', unsafe_allow_html=True)
    
    col_left, col_right = st.columns([3, 2])
    
    with col_left:
        # Hero headline with animated gradient text
        st.markdown(
            fade_in("""
            <div class="hero-title">
                Markets shouldn't overwhelm — <span>filings should explain</span>.
            </div>
            """),
            unsafe_allow_html=True,
        )
        
        # Hero subtitle
        st.markdown(
            fade_in("""
            <div class="hero-subtitle">
                Finsights turns raw 10-K filings into a question and answer surface for your research.
                Ask questions, surface KPIs, and trace every answer back to the underlying text
                in seconds.
            </div>
            """, delay=1),
            unsafe_allow_html=True,
        )
        
        # CTA button
        cta = st.button("Try the 10-K chatbot →", key="hero_chat_button")
        if cta:
            st.session_state.page = "Chatbot"
            st.rerun()
        
        # Footnote disclaimer
        st.markdown(
            fade_in("""
            <div class="hero-footnote">
                No trading advice, just transparent, document-grounded answers.
            </div>
            """, delay=2),
            unsafe_allow_html=True,
        )
    
    with col_right:
        # Metrics grid with icons
        st.markdown(
            fade_in(f"""
            <div class="metrics-grid">
                {metric_card(
                    "Time saved / 10-K",
                    "Faster Research",
                    "From first question to key insights.",
                    "bolt"
                )}
                {metric_card(
                    "Section coverage",
                    "Core SEC Items",
                    "Risk Factors, Business Overview, MD&A & more.",
                    "layer-group"
                )}
                {metric_card(
                    "Answer traceability",
                    "100%",
                    "Every response cites filing sections.",
                    "check-double"
                )}
                {metric_card(
                    "Filings supported",
                    "10-K",
                    "10-Q & 8-K on the near-term roadmap.",
                    "file-contract"
                )}
            </div>
            """, delay=1),
            unsafe_allow_html=True,
        )
    
    st.markdown("</div>", unsafe_allow_html=True)  # Close hero
    
    st.markdown("---")
    
    # ========================================================================
    # FEATURES SECTION
    # ========================================================================
    
    st.markdown(
        fade_in("""
        <div class="section-heading">What Finsights unlocks</div>
        <div class="section-sub">
            A clean, question-and-answer layer on top of SEC filings — built for analysts, 
            PMs, and research teams who rely on 10-K disclosures.
        </div>
        """, delay=2),
        unsafe_allow_html=True,
    )
    
    # Feature cards data with icons
    features = [
        (
            "Context-aware filing Q&A",
            "Ask questions in plain language and get answers grounded in 10-K text, "
            "not a generic model memory.",
            "magnifying-glass"  # Search icon
        ),
        (
            "Section-level Insights",
            "Quickly access the most important parts of a filing, including Risk Factors, "
            "Business Overview, and MD&A, all parsed into clean sections.",
            "file-lines"  # Document icon
        ),
        (
            "Risk & narrative summarization",
            "Generate concise summaries of dense sections like Risk Factors or MD&A, "
            "tied back to their source paragraphs.",
            "list-check"  # Checklist icon
        ),
        (
            "Citation-first answers",
            "Every response includes filing-level references so you always know exactly "
            "where the information came from.",
            "quote-left"  # Quote icon
        ),
        (
            "Multi-company support",
            "Ask about any supported filing — Apple 2023, Google 2020, Microsoft 2022 — "
            "and get a section-grounded answer each time.",
            "building"  # Building icon
        ),
        (
            "Compliance-aware design",
            "Each answer provides clear document references, supporting internal review "
            "and transparent research documentation.",
            "shield-halved"  # Shield icon
        ),
    ]
    
    # Render feature cards in 3x2 grid with staggered fade-ins
    for row_idx, row in enumerate([features[:3], features[3:]]):
        cols = st.columns(3)
        for col_idx, (col, (title, body, icon_name)) in enumerate(zip(cols, row)):
            # Calculate staggered delay (3, 4, 5, 6, 7, 8)
            delay = 3 + (row_idx * 3) + col_idx
            col.markdown(
                fade_in(feature_card(title, body, icon_name), delay=min(delay, 6)),
                unsafe_allow_html=True,
            )
    
    st.markdown("---")
    
    # ========================================================================
    # EXAMPLE QUESTIONS SECTION
    # ========================================================================
    
    st.markdown(
        fade_in("""
        <div class="section-heading">Example questions</div>
        """, delay=5),
        unsafe_allow_html=True
    )
    
    # Example queries in two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(
            fade_in("""
            <div style="color: #cbd5e1; line-height: 1.8;">
                <div style="margin-bottom: 0.8rem;">
                    <strong style="color: #22c55e;">Quick Queries:</strong>
                </div>
                <div style="color: #94a3b8; font-size: 0.9rem;">
                    • "What is Google's Revenue for 2023?"<br>
                    • "What strategic priorities did Amazon outline for 2023?"<br>
                    • "Show me Apple's gross margins for 2020-2023"
                </div>
            </div>
            """, delay=5),
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            fade_in("""
            <div style="color: #cbd5e1; line-height: 1.8;">
                <div style="margin-bottom: 0.8rem;">
                    <strong style="color: #0ea5e9;">Complex Analysis:</strong>
                </div>
                <div style="color: #94a3b8; font-size: 0.9rem;">
                    • "What drove Apple's revenue and margin changes in 2023?"<br>
                    • "How did Microsoft's Cloud segment perform vs. last year?"<br>
                    • "Compare Netflix and Disney's subscription risks in 2022"
                </div>
            </div>
            """, delay=6),
            unsafe_allow_html=True
        )
    
    # ========================================================================
    # FOOTER SECTION
    # ========================================================================
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    st.markdown(
        fade_in("""
        <div style="text-align: center; color: #64748b; font-size: 0.85rem; padding: 2rem 0 1rem 0;">
            <div style="margin-bottom: 0.5rem;">
                FinSight v1.0 | IE7374 MLOps Capstone Project
            </div>
            <div>
                Built with Streamlit + FastAPI + AWS Bedrock
            </div>
        </div>
        """, delay=6),
        unsafe_allow_html=True
    )