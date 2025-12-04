# frontend/test_api_client.py
"""
Quick test for api_client.py
Run with backend running in Terminal 1
"""

from frontend.api_client import finSightClient

def test_health_check():
    """Test backend health check."""
    print("\n=== Testing Health Check ===")
    client = finSightClient()
    health = client.health_check()
    print(f"Status: {health.get('status')}")
    print(f"Full response: {health}")
    return health.get('status') == 'healthy'


def test_query_success():
    """Test successful query."""
    print("\n=== Testing Successful Query ===")
    client = finSightClient()
    
    result = client.query(
        question="What was Apple's revenue in 2017?",
        include_kpi=True,
        include_rag=True
    )
    
    print(f"Success: {result.get('success')}")
    
    if result.get('success'):
        print(f"Query: {result.get('query')}")
        print(f"Answer (first 100 chars): {result.get('answer', '')[:100]}...")
        print(f"Model: {result['metadata']['llm']['model_id']}")
        print(f"Cost: ${result['metadata']['llm']['cost']:.4f}")
    else:
        print(f"Error: {result.get('error')}")
        print(f"Stage: {result.get('stage')}")
    
    return result.get('success')



def test_query_validation():
    """Test query with invalid input (too short)."""
    print("\n=== Testing Validation Error ===")
    client = finSightClient()
    
    result = client.query(question="Hi")  # Too short, should fail
    
    print(f"Success: {result.get('success')}")
    print(f"Error: {result.get('error')}")
    
    # ADD DEBUG OUTPUT
    print(f"\n[DEBUG] Full response:")
    print(f"  success: {result.get('success')}")
    print(f"  error: {result.get('error')}")
    print(f"  error_type: {result.get('error_type')}")
    print(f"  stage: {result.get('stage')}")
    
    # If it succeeded when it shouldn't have, show the answer
    if result.get('success'):
        print(f"\n[UNEXPECTED] Backend accepted short query!")
        print(f"  Query: {result.get('query')}")
        print(f"  Answer: {result.get('answer', '')[:100]}...")
    
    return not result.get('success')  # Should fail (success=False means PASS)


if __name__ == "__main__":
    print("=" * 60)
    print("API Client Tests")
    print("=" * 60)
    print("Make sure backend is running on http://localhost:8000")
    print("=" * 60)
    
    results = {
        "Health Check": test_health_check(),
        "Successful Query": test_query_success(),
        "Validation Error": test_query_validation()
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
..\finSight_ml_tg1\venv_ml_rag\Scripts\Activate.ps1
uvicorn backend.api_service:app --reload --host 0.0.0.0 --port 8000

deactivate
cd ModelPipeline
.\serving\frontend\venv_frontend\Scripts\Activate.ps1
cd serving
python -m frontend.test_api_client

"""