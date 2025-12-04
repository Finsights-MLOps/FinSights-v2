# frontend/test_state.py
"""
Quick test for state.py
"""

import streamlit as st

from frontend.state import (
    init_session_state,
    add_user_message,
    add_assistant_message,
    update_metrics,
    get_message_count,
    clear_chat_history
)

# Mock st.session_state for testing
class MockSessionState:
    def __init__(self):
        self._state = {}
    
    def __setattr__(self, name, value):
        if name.startswith('_'):
            object.__setattr__(self, name, value)
        else:
            self._state[name] = value
    
    def __getattr__(self, name):
        if name.startswith('_'):
            return object.__getattribute__(self, name)
        return self._state.get(name)
    
    def __contains__(self, name):
        return name in self._state

st.session_state = MockSessionState()

def test_initialization():
    """Test state initialization."""
    print("\n=== Testing Initialization ===")
    init_session_state()
    
    assert st.session_state.messages == []
    assert st.session_state.total_queries == 0
    assert st.session_state.total_cost == 0.0
    print("✅ State initialized correctly")
    return True

def test_add_messages():
    """Test adding messages."""
    print("\n=== Testing Add Messages ===")
    
    add_user_message("Test question")
    add_assistant_message("Test answer", metadata={"cost": 0.01})
    
    assert len(st.session_state.messages) == 2
    assert st.session_state.messages[0]["role"] == "user"
    assert st.session_state.messages[1]["role"] == "assistant"
    print(f"✅ Added 2 messages, count: {len(st.session_state.messages)}")
    return True

def test_metrics():
    """Test metrics tracking."""
    print("\n=== Testing Metrics ===")
    
    update_metrics(0.01)
    update_metrics(0.02)
    
    assert st.session_state.total_queries == 2
    assert st.session_state.total_cost == 0.03
    print(f"✅ Queries: {st.session_state.total_queries}, Cost: ${st.session_state.total_cost:.4f}")
    return True

def test_clear():
    """Test clearing history."""
    print("\n=== Testing Clear History ===")
    
    clear_chat_history()
    
    assert st.session_state.messages == []
    assert st.session_state.total_queries == 0
    assert st.session_state.total_cost == 0.0
    print("✅ History cleared")
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("State Management Tests")
    print("=" * 60)
    
    results = {
        "Initialization": test_initialization(),
        "Add Messages": test_add_messages(),
        "Metrics Tracking": test_metrics(),
        "Clear History": test_clear()
    }
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")



"""

T1: Activate ML - Start BACKEND server.
# Start At FinSights !

deactivate
cd .\ModelPipeline\serving\
..\FinSight_ml_tg1\venv_ml_rag\Scripts\Activate.ps1
uvicorn backend.api_service:app --reload --host 0.0.0.0 --port 8000

deactivate
cd ModelPipeline
.\serving\frontend\venv_frontend\Scripts\Activate.ps1
cd serving
python -m frontend.test_state

"""