# ModelPipeline\serving\frontend\chat.py
"""
RESPONSIBILITIES:
✓ Render chat message bubbles (user + assistant)
✓ Display loading spinner during query processing
✓ Show error messages with retry option
✓ Handle chat input submission

UI PATTERN:
- st.chat_message("user") for questions
- st.chat_message("assistant") for answers
- st.chat_input() for new questions

"""

# frontend/chat.py
"""
Chat interface components for FinSight frontend.

Handles chat message rendering, user input, and query submission.

Usage:
    from chat import render_chat_history, handle_user_input
    
    render_chat_history()
    handle_user_input()
"""

import streamlit as st
from typing import Dict, Any

# FIXED: Relative imports (no 'frontend.' prefix)
from api_client import FinSightClient
from state import (
    add_user_message,
    add_assistant_message,
    update_metrics
)
from metrics import (
    display_query_metadata,
    display_error_message
)


def render_chat_message(message: Dict[str, Any]) -> None:
    """
    Render a single chat message (user or assistant).
    
    Args:
        message: Message dict from session state
    """
    role = message["role"]
    content = message["content"]
    
    with st.chat_message(role):
        # Display message content
        if message.get("error", False):
            # Error message styling
            st.markdown(f"[WARNING] {content}")
        else:
            # CHANGED: Use st.write instead of st.markdown to avoid LaTeX parsing
            # st.write handles text more naturally and doesn't trigger LaTeX
            # st.write is still using regex. flaws. detects content and renders appropriately
            st.markdown(content)

        
        # Display metadata for assistant messages (if exists and not error)
        if role == "assistant" and not message.get("error", False):
            metadata = message.get("metadata")
            if metadata:
                display_query_metadata(metadata)




def render_chat_history() -> None:
    """
    Render all messages in chat history.
    
    Reads from st.session_state.messages and displays each message.
    """
    messages = st.session_state.get("messages", [])
    
    for message in messages:
        render_chat_message(message)


def handle_user_input(client: FinSightClient) -> None:
    """
    Handle user input from chat input box.
    
    Processes new user questions, calls backend API, and updates state.
    
    Args:
        client: FinSightClient instance for API calls
    """
    # Chat input at bottom of page
    prompt = st.chat_input("Ask a question about SEC 10-K filings...")
    
    if prompt:
        # Validate input length (client-side check)
        if len(prompt) < 10:
            st.error("[Hey!] Question must be at least 10 characters long.")
            return
        
        # Add user message to history
        add_user_message(prompt)
        
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Process query with loading indicator
        with st.chat_message("assistant"):
            with st.spinner("[Searching..!] Processing your question..."):
                # Get model_key from session state
                model_key = st.session_state.get("model_key")
                
                # Call backend
                result = client.query(
                    question=prompt,
                    include_kpi=True,
                    include_rag=True,
                    model_key=model_key
                )
            
            # Handle response
            if result.get("success"):
                # Success - display answer
                answer = result.get("answer", "")
                metadata = result.get("metadata", {})
                
                # markdown -- issues? lets attempt.
                # st.markdown(answer) / st.text, st.write
                st.markdown(answer)
                
                # Add to history
                add_assistant_message(
                    content=answer,
                    metadata=metadata,
                    error=False
                )
                
                # Update metrics
                cost = metadata.get("llm", {}).get("cost", 0.0)
                update_metrics(cost)
                
                # Display metadata
                display_query_metadata(metadata)
            
            else:
                # Error - display error message
                error_msg = result.get("error", "Unknown error occurred")
                error_type = result.get("error_type", "UnknownError")
                stage = result.get("stage", "unknown")
                
                display_error_message(error_msg, error_type, stage)
                
                # Add error to history
                add_assistant_message(
                    content=error_msg,
                    metadata=None,
                    error=True
                )
        
        # Rerun to update UI
        st.rerun()


def render_clear_button() -> None:
    """
    Render "Clear Chat" button in sidebar.
    
    Allows users to start a new conversation.
    """
    #: Relative import
    from state import clear_chat_history
    
    if st.sidebar.button("Clear Chat History", use_container_width=True):
        clear_chat_history()
        st.rerun()