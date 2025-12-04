"""
User types query in Streamlit
Streamlit POSTs to localhost:8000/query with {question, include_kpi, include_rag, model_key}
FastAPI validates, calls answer_query()
Returns {answer, metadata, request_id, processing_time_ms}
Streamlit displays result, updates chat history


RESPONSIBILITIES:
✓ Initialize st.session_state on first load
✓ Store chat history (list of Q&A pairs)
✓ Track query count, total cost
✓ Manage UI preferences (show_metadata, model_key)

STATE SCHEMA:
{
    "messages": [
        {"role": "user", "content": str, "timestamp": str},
        {"role": "assistant", "content": str, "metadata": dict}
    ],
    "total_queries": int,
    "total_cost": float,
    "backend_healthy": bool
}

"""



# frontend/state.py
"""
Session state management for finSight frontend.

Handles chat history, cost tracking, and UI preferences using Streamlit's
session_state. Provides clean interface for state initialization and updates.

Usage:
    import streamlit as st
    from frontend.state import init_session_state, add_user_message
    
    init_session_state()  # Call once at app start
    add_user_message("What was Apple's revenue?")
"""

import streamlit as st
from datetime import datetime
from typing import Dict, Any, Optional


def init_session_state():
    """
    Initialize session state on first app load.
    
    Creates all necessary state variables if they don't exist.
    Safe to call multiple times (only initializes once).
    """
    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Metrics tracking
    if "total_queries" not in st.session_state:
        st.session_state.total_queries = 0
    
    if "total_cost" not in st.session_state:
        st.session_state.total_cost = 0.0
    
    # Backend status
    if "backend_healthy" not in st.session_state:
        st.session_state.backend_healthy = None  # Unknown until checked
    
    # UI preferences
    if "show_metadata" not in st.session_state:
        st.session_state.show_metadata = True  # Show metadata by default
    
    if "model_key" not in st.session_state:
        st.session_state.model_key = None  # Use backend default


def add_user_message(content: str) -> None:
    """
    Add user message to chat history.
    
    Args:
        content: User's question text
    """
    message = {
        "role": "user",
        "content": content,
        "timestamp": datetime.utcnow().isoformat()
    }
    st.session_state.messages.append(message)


def add_assistant_message(
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
    error: bool = False
) -> None:
    """
    Add assistant message to chat history.
    
    Args:
        content: Assistant's response text (answer or error message)
        metadata: Optional metadata dict from backend
        error: Whether this is an error message
    """
    message = {
        "role": "assistant",
        "content": content,
        "timestamp": datetime.utcnow().isoformat(),
        "error": error
    }
    
    if metadata is not None:
        message["metadata"] = metadata
    
    st.session_state.messages.append(message)


def update_metrics(cost: float) -> None:
    """
    Update cumulative metrics after a query.
    
    Args:
        cost: Cost of the query in USD
    """
    st.session_state.total_queries += 1
    st.session_state.total_cost += cost


def get_last_message() -> Optional[Dict[str, Any]]:
    """
    Get the most recent message.
    
    Returns:
        dict: Last message or None if history is empty
    """
    if st.session_state.messages:
        return st.session_state.messages[-1]
    return None


def clear_chat_history() -> None:
    """
    Clear all chat history and reset metrics.
    
    Useful for "New Chat" button.
    """
    st.session_state.messages = []
    st.session_state.total_queries = 0
    st.session_state.total_cost = 0.0


def get_message_count() -> Dict[str, int]:
    """
    Get count of user and assistant messages.
    
    Returns:
        dict: {"user": int, "assistant": int, "total": int}
    """
    user_count = sum(1 for msg in st.session_state.messages if msg["role"] == "user")
    assistant_count = sum(1 for msg in st.session_state.messages if msg["role"] == "assistant")
    
    return {
        "user": user_count,
        "assistant": assistant_count,
        "total": len(st.session_state.messages)
    }


def set_backend_health(healthy: bool) -> None:
    """
    Update backend health status.
    
    Args:
        healthy: Whether backend is healthy
    """
    st.session_state.backend_healthy = healthy


def get_backend_health() -> Optional[bool]:
    """
    Get current backend health status.
    
    Returns:
        bool or None: True if healthy, False if unhealthy, None if unknown
    """
    return st.session_state.backend_healthy