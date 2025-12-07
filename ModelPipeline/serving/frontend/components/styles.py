import streamlit as st


def inject_global_css():
    """Inject global CSS for FinSight / FinSights UI."""

    # Font Awesome
    st.markdown(
        """
        <link rel="stylesheet"
              href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <style>
        /* =======================================================================
           BASE THEME & LAYOUT
           ======================================================================= */

        #MainMenu { visibility: hidden; }
        footer   { visibility: hidden; }
        header   { visibility: hidden; }

        [data-testid="stSidebar"],
        [data-testid="stSidebarNav"] {
            display: none !important;
        }

        .stApp {
            background-color: #020617 !important; /* slate-950 */
        }

        .main .block-container {
            max-width: 1200px;
            margin: 0 auto;
            padding-left: 2rem;
            padding-right: 2rem;
        }

        /* Text colours */
        .stMarkdown, .stText, p, span, div, li {
            color: #e5e7eb !important; /* slate-200 */
        }

        h1, h2, h3, h4, h5, h6 {
            color: #f8fafc !important; /* slate-50 */
        }

        /* =======================================================================
           HERO & GRADIENT TEXT
           ======================================================================= */

        .hero-headline {
            font-size: 2.8rem;
            line-height: 1.15;
            margin-bottom: 1.5rem;
            font-weight: 700;
        }

        .gradient-text {
            background: linear-gradient(
                90deg,
                #22c55e 0%,
                #0ea5e9 25%,
                #38bdf8 50%,
                #22c55e 75%,
                #0ea5e9 100%
            );
            background-size: 400% 400%;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            animation: gradientFlow 22s linear infinite;
            font-weight: 700;
            display: inline-block;
        }

        @keyframes gradientFlow {
            0%   { background-position:   0% 50%; }
            100% { background-position: 400% 50%; }
        }

        /* =======================================================================
           HERO METRICS (right-hand column)
           ======================================================================= */

        [data-testid="stMetric"] {
            background: rgba(10, 15, 35, 0.9) !important;
            padding: 1.3rem !important;
            border-radius: 24px !important;
            border: 1.5px solid rgba(59, 130, 246, 0.4) !important;
            box-shadow:
                0 8px 32px rgba(0, 0, 0, 0.7),
                0 0 24px rgba(59, 130, 246, 0.3) !important;
            transition: all 0.3s ease !important;
        }

        [data-testid="stMetric"]:hover {
            transform: translateY(-5px) scale(1.03) !important;
            box-shadow:
                0 16px 48px rgba(0, 0, 0, 0.8),
                0 0 32px rgba(34, 197, 94, 0.35) !important;
            border-color: rgba(34, 197, 94, 0.5) !important;
        }

        [data-testid="stMetricLabel"] {
            color: #94a3b8 !important;
            font-size: 0.75rem !important;
            text-transform: uppercase !important;
            letter-spacing: 0.08em !important;
            font-weight: 600 !important;
        }

        [data-testid="stMetricValue"] {
            color: #f8fafc !important;
            font-size: 1.5rem !important;
            font-weight: 700 !important;
        }

        /* =======================================================================
           BUTTONS
           ======================================================================= */

        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #22c55e, #0ea5e9) !important;
            color: #020617 !important;
            font-weight: 700 !important;
            border: none !important;
            padding: 0.65rem 2rem !important;
            border-radius: 12px !important;
            box-shadow: 0 8px 24px rgba(34, 197, 94, 0.35) !important;
            transition: all 0.3s ease !important;
            font-size: 1rem !important;
        }

        .stButton > button[kind="primary"]:hover {
            transform: translateY(-3px) scale(1.02) !important;
            box-shadow: 0 12px 32px rgba(34, 197, 94, 0.5) !important;
            filter: brightness(1.08) !important;
        }

        .stButton > button[kind="secondary"] {
            background: rgba(15, 23, 42, 0.6) !important;
            color: #e5e7eb !important;
            border: 1px solid rgba(148, 163, 184, 0.3) !important;
            border-radius: 10px !important;
            transition: all 0.3s ease !important;
        }

        .stButton > button[kind="secondary"]:hover {
            border-color: rgba(34, 197, 94, 0.6) !important;
            transform: translateY(-2px) !important;
        }

        /* =======================================================================
           NAVIGATION (logo)
           ======================================================================= */

        .nav-logo {
            display: flex;
            align-items: center;
            gap: 0.6rem;
            margin-bottom: 1rem;
        }

        .logo-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: linear-gradient(135deg, #22c55e, #0ea5e9);
            box-shadow: 0 0 0 4px rgba(34, 197, 94, 0.2);
        }

        .logo-text {
            font-size: 1.4rem;
            font-weight: 700;
            color: #e2e8f0;
            letter-spacing: 0.02em;
        }

        .logo-text .highlight {
            color: #22c55e;
        }

        /* =======================================================================
           ICONS (Font Awesome for metrics & cards)
           ======================================================================= */

        .feature-icon {
            font-size: 2.2rem;
            color: #22c55e;
            margin-bottom: 0.8rem;
            display: block;
            filter: drop-shadow(0 0 12px rgba(34, 197, 94, 0.5));
        }

        /* =======================================================================
        FEATURE CARDS – 6 cards in a responsive flexbox grid
        ======================================================================= */

        .feature-grid {
            display: flex;
            flex-wrap: wrap;
            gap: 2rem;
            margin-top: 2.5rem;
        }

        /* 3 per row on desktop, wrap as needed */
        .feature-card {
            flex: 1 1 calc(33.333% - 2rem);
            min-width: 260px;

            background: rgba(10, 15, 35, 0.9);
            border: 1.5px solid rgba(59, 130, 246, 0.5);
            border-radius: 40px;
            padding: 1.8rem 1.6rem;

            box-shadow:
                0 10px 40px rgba(0, 0, 0, 0.8),
                0 0 30px rgba(59, 130, 246, 0.4),
                0 0 60px rgba(59, 130, 246, 0.2),
                inset 0 1px 0 rgba(255, 255, 255, 0.05);
            transition: all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
        }

        .feature-card:hover {
            transform: translateY(-10px) scale(1.04);
            box-shadow:
                0 24px 60px rgba(0, 0, 0, 0.9),
                0 0 40px rgba(34, 197, 94, 0.5),
                0 0 80px rgba(34, 197, 94, 0.25),
                inset 0 1px 0 rgba(255, 255, 255, 0.1);
            border-color: rgba(34, 197, 94, 0.7);
        }

        /* Icon + heading row inside card */
        .feature-card-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.75rem;
            margin-bottom: 0.75rem;
        }

        .feature-card-header h4 {
            margin: 0;
            font-size: 1.05rem;
        }

        /* Inline icon inside card header */
        .feature-icon-inline {
            font-size: 1.6rem;
            color: #22c55e;
            flex-shrink: 0;
            filter: drop-shadow(0 0 10px rgba(34, 197, 94, 0.5));
        }

        /* Body text inside cards */
        .feature-card p {
            margin: 0;
            font-size: 0.95rem;
            line-height: 1.5;
        }

        /* Responsive tweaks: 2 per row on medium, 1 per row on small */
        @media (max-width: 960px) {
            .feature-card {
                flex: 1 1 calc(50% - 2rem);
            }
        }

        @media (max-width: 640px) {
            .feature-card {
                flex: 1 1 100%;
            }
        }


        /* =======================================================================
           DIVIDERS & CAPTIONS
           ======================================================================= */

        hr {
            border-color: rgba(148, 163, 184, 0.1) !important;
            margin: 3rem 0 !important;
        }

        .stCaption {
            color: #64748b !important;
            text-align: center !important;
        }
        
        /* =======================================================================
           CHATBOT PAGE STYLING - ChatGPT-style narrow interface
           ======================================================================= */
        
        /* Narrow chat container for chatbot page */
        body:has([data-testid="stChatInput"]) .main .block-container {
            max-width: 720px !important;
            padding-left: 1.5rem;
            padding-right: 1.5rem;
        }
        
        /* Chat message bubbles with distinct colors */
        [data-testid="stChatMessage"] {
            padding: 1rem 1.2rem !important;
            border-radius: 16px !important;
            margin-bottom: 1rem !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3) !important;
            border: 1px solid transparent !important;
        }
        
        /* User messages - green accent */
        [data-testid="stChatMessage"]:has([aria-label="user"]) {
            background: rgba(34, 197, 94, 0.08) !important;
            border-color: rgba(34, 197, 94, 0.3) !important;
        }
        
        /* Assistant messages - blue accent */
        [data-testid="stChatMessage"]:has([aria-label="assistant"]) {
            background: rgba(59, 130, 246, 0.08) !important;
            border-color: rgba(59, 130, 246, 0.3) !important;
        }
        
        /* Chat input bar */
        [data-testid="stChatInput"] {
            border-radius: 12px !important;
            border: 1px solid rgba(148, 163, 184, 0.3) !important;
            background: rgba(15, 23, 42, 0.6) !important;
        }
        
        [data-testid="stChatInput"]:focus-within {
            border-color: rgba(34, 197, 94, 0.5) !important;
            box-shadow: 0 0 0 3px rgba(34, 197, 94, 0.1) !important;
        }
        
        /* Sidebar visibility for chatbot */
        body:has([data-testid="stChatInput"]) [data-testid="stSidebar"] {
            display: block !important;
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background: rgba(15, 23, 42, 0.95) !important;
            border-right: 1px solid rgba(148, 163, 184, 0.1) !important;
        }
        
        /* Sidebar content */
        [data-testid="stSidebar"] .stMarkdown h3 {
            font-size: 1.1rem !important;
            margin-bottom: 0.8rem !important;
            margin-top: 1rem !important;
        }
        
        /* Sidebar expanders */
        [data-testid="stSidebar"] [data-testid="stExpander"] {
            background: rgba(10, 15, 35, 0.6) !important;
            border: 1px solid rgba(148, 163, 184, 0.2) !important;
            border-radius: 8px !important;
            margin-bottom: 0.5rem !important;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )