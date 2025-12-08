# ModelPipeline\serving\frontend\config.py
"""
Frontend configuration - Single source of truth for all settings.

Reads from environment variables with sensible defaults for local development.

Usage:
    from frontend.config import BACKEND_URL, API_TIMEOUT
    
    client = finSightClient(base_url=BACKEND_URL)
"""

import os
from typing import Optional


# ============================================================================
# BACKEND CONNECTION SETTINGS
# ============================================================================

BACKEND_URL: str = os.getenv("BACKEND_URL", "http://localhost:8000")
"""
Backend API base URL.

Sources (in priority order):
  1. BACKEND_URL environment variable (Sevalla cloud deployment)
  2. Default: http://localhost:8000 (local development)
"""

API_TIMEOUT: int = int(os.getenv("API_TIMEOUT", "120"))
"""
API request timeout in seconds.

Default: 120s (queries can take 10-20s for complex analysis)
"""


# ============================================================================
# DISPLAY SETTINGS
# ============================================================================

SHOW_DEBUG_INFO: bool = os.getenv("SHOW_DEBUG_INFO", "false").lower() == "true"
"""
Show debug information in UI (backend URL, environment, etc.)

Enable by setting: SHOW_DEBUG_INFO=true
"""


# ============================================================================
# QUERY DEFAULTS
# ============================================================================

DEFAULT_INCLUDE_KPI: bool = True
"""Default: Include KPI analysis in queries"""

DEFAULT_INCLUDE_RAG: bool = True
"""Default: Include semantic RAG in queries"""


# ============================================================================
# UI PREFERENCES
# ============================================================================

DEFAULT_SHOW_METADATA: bool = True
"""Default: Show query metadata (cost, tokens, model info)"""


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_config_summary() -> dict:
    """
    Get summary of current configuration.
    
    Useful for debugging or displaying in UI.
    
    Returns:
        dict: Current configuration values
    """
    return {
        "backend_url": BACKEND_URL,
        "api_timeout": API_TIMEOUT,
        "debug_mode": SHOW_DEBUG_INFO,
        "environment": "cloud" if os.getenv("BACKEND_URL") else "local"
    }


def print_config() -> None:
    """Print configuration to console (for debugging)."""
    config = get_config_summary()
    print("=" * 60)
    print("Frontend Configuration")
    print("=" * 60)
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 60)


# Auto-print config in debug mode
if SHOW_DEBUG_INFO:
    print_config()