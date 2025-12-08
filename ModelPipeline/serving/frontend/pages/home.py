# ModelPipeline\serving\frontend\pages\home.py
"""
FinSights Home Page - Polished UI with Streamlit Components.
"""

import streamlit as st


def render_home():
    """Render homepage with polished UI."""

    # =======================================================================
    # HERO
    # =======================================================================

    st.markdown(
        """
        <div class="hero-headline">
            10-Ks shouldn't be barriers. Cut through complexity—
            <span class="gradient-text">Find Clarity. Find Interpretations. Power your Financial Research.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        st.markdown(
            """
            Finsights turns raw 10-K filings into a question and answer surface for your research.
            Ask questions, surface KPIs, and trace every answer back to the underlying text in seconds.
            """
        )

        if st.button("Try the 10-K chatbot →", type="primary", use_container_width=False):
            st.session_state.page = "Chatbot"
            st.rerun()

        st.caption("No trading advice, just transparent, document-grounded answers.")

    with col_right:
        metric_col1, metric_col2 = st.columns(2)

        with metric_col1:
            st.markdown(
                '<i class="fa-solid fa-bolt feature-icon"></i>',
                unsafe_allow_html=True,
            )
            st.metric(
                label="TIME TO INSIGHT",
                value="Seconds",
                help="From first question to key insights",
            )

            st.markdown(
                '<i class="fa-solid fa-check-double feature-icon"></i>',
                unsafe_allow_html=True,
            )
            st.metric(
                label="WORKFLOW VALUE",
                value="Audit-ready",
                help="Every response cites filing sections",
            )

        with metric_col2:
            st.markdown(
                '<i class="fa-solid fa-layer-group feature-icon"></i>',
                unsafe_allow_html=True,
            )
            st.metric(
                label="FILINGS COVERED",
                value="Core 10-K",
                help="Risk Factors, Business Overview, MD&A? All sections covered!",
            )

            st.markdown(
                '<i class="fa-solid fa-file-contract feature-icon"></i>',
                unsafe_allow_html=True,
            )
            st.metric(
                label="ANSWER STYLE",
                value="Grounded",
                help="Answers and Exports have citations/references",
            )

    st.divider()


    # ========================================================================
    # FEATURES SECTION (icons INSIDE the cards, flex layout)
    # ========================================================================

    st.subheader("What Finsights unlocks")
    st.write(
        "A clean, question-and-answer layer on top of SEC filings — "
        "built for analysts, PMs, and research teams."
    )

    # Pure HTML: one wrapper + six cards, no Streamlit containers/columns
    st.markdown(
        """
        <div class="feature-grid">

          <div class="feature-card">
            <div class="feature-card-header">
              <h4>Context-aware Q&amp;A</h4>
              <i class="fa-solid fa-magnifying-glass feature-icon-inline"></i>
            </div>
            <p>
              Ask questions in plain language and get answers grounded in 10-K text,
              not generic model memory.
            </p>
          </div>

          <div class="feature-card">
            <div class="feature-card-header">
              <h4>Section-level insights</h4>
              <i class="fa-solid fa-file-lines feature-icon-inline"></i>
            </div>
            <p>
              Access Risk Factors, Business Overview, and MD&amp;A, all parsed into
              clean sections.
            </p>
          </div>

          <div class="feature-card">
            <div class="feature-card-header">
              <h4>Risk summarization</h4>
              <i class="fa-solid fa-list-check feature-icon-inline"></i>
            </div>
            <p>
              Generate concise summaries of dense sections tied back to source
              paragraphs.
            </p>
          </div>

          <div class="feature-card">
            <div class="feature-card-header">
              <h4>Citation-first answers</h4>
              <i class="fa-solid fa-quote-left feature-icon-inline"></i>
            </div>
            <p>
              Every response includes filing-level references so you know exactly
              where information came from.
            </p>
          </div>

          <div class="feature-card">
            <div class="feature-card-header">
              <h4>Multi-company support</h4>
              <i class="fa-solid fa-building feature-icon-inline"></i>
            </div>
            <p>
              Ask about any supported filing — Apple 2023, Google 2020,
              Microsoft 2022.
            </p>
          </div>

          <div class="feature-card">
            <div class="feature-card-header">
              <h4>Compliance-aware</h4>
              <i class="fa-solid fa-shield-halved feature-icon-inline"></i>
            </div>
            <p>
              Clear document references support internal review
              and transparent research.
            </p>
          </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # =======================================================================
    # EXAMPLE QUESTIONS (keep as Streamlit containers; they use default look)
    # =======================================================================

    st.subheader("Example questions")

    col_q1, col_q2 = st.columns(2, gap="large")

    with col_q1:
        st.markdown("**Quick Queries:**")
        with st.container(border=True):
            st.markdown(
                """
                - "What is Google's Revenue for 2023?"
                - "What strategic priorities did Amazon outline?"
                - "Show me Apple's gross margins for 2020-2023"
                """
            )

    with col_q2:
        st.markdown("**Complex Analysis:**")
        with st.container(border=True):
            st.markdown(
                """
                - "What drove Apple's revenue and margin changes in 2023?"
                - "How did Microsoft's Cloud segment perform vs. last year?"
                - "Compare Netflix and Disney's subscription risks in 2022"
                """
            )

    st.divider()

    st.caption("FinSight v1.0 | IE7374 MLOps Capstone Project")
    st.caption("Built with Streamlit + FastAPI + AWS Bedrock")
