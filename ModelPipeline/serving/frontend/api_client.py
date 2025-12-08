# ModelPipeline\serving\frontend\api_client.py
"""
RESPONSIBILITIES:
✓ Make HTTP POST requests to backend (/query endpoint)
✓ Handle connection errors and timeouts
✓ Parse JSON responses into Python dicts
✓ Health check validation (/health endpoint)

INPUT:  User question string, optional parameters
OUTPUT: QueryResponse dict or ErrorResponse dict

-----------------------------------------------------------------------------------------------------------------

class FinSightClient:
    - __init__(base_url, timeout)
    - health_check() → dict
    - query(question, ...) → dict
    - _handle_response(response) → dict  # Helper

Backend URL: http://localhost:8000
Timeout: 120s (queries take ~10-15s)
Request format: {"question": str, "include_kpi": bool, "include_rag": bool, "model_key": Optional[str]}
Success response has: query, answer, context, metadata
Error response has: query, error, error_type, stage, timestamp

Important updates on the flow:
No BACKEND_URL env var set - finSightClient() called with no arguments, None, 
                             os.getenv("BACKEND_URL", "http://localhost:8000") returns default
CLOUD - os.getenv("BACKEND_URL", ...) returns "http://backend:8000" // (K8s service DNS).

-----------------------------------------------------------------------------------------------------------------

Usage:
    client = FinSightClient()
    result = client.query("What was Apple's revenue?")

"""

# ModelPipeline/serving/frontend/api_client.py


# frontend/api_client.py
"""
Backend API client for FinSight.

Handles all HTTP communication with the FastAPI backend.
Provides clean interface for health checks and query submission.

Usage:
    client = FinSightClient()
    result = client.query("What was Apple's revenue?")
"""

import requests
from typing import Dict, Optional, Any
from datetime import datetime
import os
from config import BACKEND_URL, API_TIMEOUT
## why os?: things are gonna get on cloud soon. we use os.environ. > patterns. 

class FinSightClient:
    """
    Client for communicating with FinSight FastAPI backend.
    
    Attributes:
        base_url: Backend API base URL (default: http://localhost:8000)
        timeout: Request timeout in seconds (default: 120)
    """
    
    def __init__(
        self,
        ## base_url: str = "http://localhost:8000", 
        ## hardcoded default - changed now. use None.
        base_url: Optional[str] = None,
        timeout: int = 120
    ):
        """
        Initialize API client.
        
        Args:
            base_url: Backend API URL
            timeout: Request timeout (queries can take 10-15s)
        """
        
        # Environment-aware backend URL
        # Local: Uses default localhost:8000
        # Cloud: Uses Sevalla's service DNS (http://backend:8000)
        if base_url is None:
            base_url = BACKEND_URL  # From config.py
        
        if timeout is None:
            timeout = API_TIMEOUT  # From config.py
        
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
    
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check if backend is running and healthy.
        
        Returns:
            dict: Health status response
                {
                    "status": "healthy",
                    "model_root_exists": bool,
                    "aws_configured": bool or None,
                    "timestamp": str
                }
        
        Raises:
            Never raises - returns error dict on failure
        """
        try:
            response = requests.get(
                f"{self.base_url}/health",
                timeout=5  # Quick timeout for health check
            )
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.Timeout:
            return {
                "status": "timeout",
                "error": "Backend health check timed out"
            }
        
        except requests.exceptions.ConnectionError:
            return {
                "status": "unreachable",
                "error": f"Cannot connect to backend at {self.base_url}"
            }
        
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    
    def query(
        self,
        question: str,
        include_kpi: bool = True,
        include_rag: bool = True,
        model_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Send query to backend for processing.
        
        Args:
            question: User's question (10-500 chars)
            include_kpi: Include structured KPI lookup
            include_rag: Include semantic RAG retrieval
            model_key: Optional model configuration key
        
        Returns:
            dict: Always returns a dict with 'success' field
            
            Success response (success=True):
            {
                "success": True,
                "query": str,
                "answer": str,
                "context": str,
                "metadata": {
                    "llm": {...},
                    "context": {...}
                },
                "exports": None
            }
            
            Error response (success=False):
            {
                "success": False,
                "query": str,
                "error": str,
                "error_type": str,
                "stage": str,
                "timestamp": str,
                "http_status": int (optional)
            }
        """
        # Build request payload
        payload = {
            "question": question,
            "include_kpi": include_kpi,
            "include_rag": include_rag,
        }
        
        # Add model_key only if provided
        if model_key is not None:
            payload["model_key"] = model_key
        
        try:
            # Send POST request
            response = requests.post(
                f"{self.base_url}/query",
                json=payload,
                timeout=self.timeout
            )
            
            # Handle response based on status code
            return self._handle_response(response, question)
        
        except requests.exceptions.Timeout:
            return self._error_response(
                query=question,
                error=f"Query timed out after {self.timeout} seconds",
                error_type="TimeoutError",
                stage="http_request"
            )
        
        except requests.exceptions.ConnectionError:
            return self._error_response(
                query=question,
                error=f"Cannot connect to backend at {self.base_url}",
                error_type="ConnectionError",
                stage="http_request"
            )
        
        except Exception as e:
            return self._error_response(
                query=question,
                error=f"Unexpected error: {str(e)}",
                error_type="UnexpectedError",
                stage="http_request"
            )
    
    
    # ========================================================================
    # PRIVATE HELPER METHODS
    # ========================================================================
    
    def _handle_response(
        self, 
        response: requests.Response, 
        question: str
    ) -> Dict[str, Any]:
        """
        Handle HTTP response from backend.
        
        Categorizes responses by status code and extracts appropriate data.
        
        Args:
            response: requests.Response object
            question: Original question string
        
        Returns:
            dict: Standardized response dict
        """
        status_code = response.status_code
        
        # SUCCESS: 2xx responses
        if 200 <= status_code < 300:
            return self._handle_success_response(response, question)
        
        # CLIENT ERROR: 4xx responses
        elif 400 <= status_code < 500:
            return self._handle_client_error(response, question, status_code)
        
        # SERVER ERROR: 5xx responses
        elif 500 <= status_code < 600:
            return self._handle_server_error(response, question, status_code)
        
        # UNEXPECTED: Other status codes
        else:
            return self._error_response(
                query=question,
                error=f"Unexpected HTTP status code: {status_code}",
                error_type="UnexpectedStatusCode",
                stage="http_response",
                http_status=status_code
            )
    
    
    def _handle_success_response(
        self, 
        response: requests.Response,
        question: str
    ) -> Dict[str, Any]:
        """
        Handle successful 2xx response.
        
        Args:
            response: requests.Response object
            question: Original question string
        
        Returns:
            dict: Success response or error if response contains error field
        """
        try:
            data = response.json()
        except ValueError:
            return self._error_response(
                query=question,
                error="Backend returned invalid JSON",
                error_type="InvalidJSON",
                stage="response_parsing"
            )
        
        # Check if backend returned error dict (even with 200 status)
        # This handles cases where orchestrator fails but FastAPI returns 200
        if "error" in data:
            return {
                "success": False,
                "query": data.get("query", question),
                "error": data.get("error"),
                "error_type": data.get("error_type", "BackendError"),
                "stage": data.get("stage", "unknown"),
                "timestamp": data.get("timestamp", datetime.utcnow().isoformat())
            }
        
        # True success - return data with success flag
        return {
            "success": True,
            **data
        }
    
    
    def _handle_client_error(
        self,
        response: requests.Response,
        question: str,
        status_code: int
    ) -> Dict[str, Any]:
        """
        Handle 4xx client errors.
        
        Common cases:
        - 400 Bad Request: Malformed request
        - 422 Unprocessable Entity: Validation error (Pydantic)
        - 404 Not Found: Endpoint doesn't exist
        
        Args:
            response: requests.Response object
            question: Original question string
            status_code: HTTP status code
        
        Returns:
            dict: Error response
        """
        try:
            data = response.json()
        except ValueError:
            return self._error_response(
                query=question,
                error=f"HTTP {status_code}: {response.text[:200]}",
                error_type="ClientError",
                stage="http_response",
                http_status=status_code
            )
        
        # Handle FastAPI validation errors (422)
        if status_code == 422 and "detail" in data:
            error_msg = self._format_validation_errors(data["detail"])
            return self._error_response(
                query=question,
                error=error_msg,
                error_type="ValidationError",
                stage="request_validation",
                http_status=422
            )
        
        # Handle other client errors with error field
        if "error" in data:
            return self._error_response(
                query=data.get("query", question),
                error=data["error"],
                error_type=data.get("error_type", "ClientError"),
                stage=data.get("stage", "http_response"),
                http_status=status_code
            )
        
        # Generic client error
        return self._error_response(
            query=question,
            error=f"HTTP {status_code}: {str(data)[:200]}",
            error_type="ClientError",
            stage="http_response",
            http_status=status_code
        )
    
    
    def _handle_server_error(
        self,
        response: requests.Response,
        question: str,
        status_code: int
    ) -> Dict[str, Any]:
        """
        Handle 5xx server errors.
        
        Common cases:
        - 500 Internal Server Error: Backend crash
        - 502 Bad Gateway: AWS/external service failure
        - 503 Service Unavailable: Backend overloaded
        
        Args:
            response: requests.Response object
            question: Original question string
            status_code: HTTP status code
        
        Returns:
            dict: Error response
        """
        try:
            data = response.json()
            error_msg = data.get("error", f"Server error: {status_code}")
        except ValueError:
            error_msg = f"Server error: {response.text[:200]}"
        
        return self._error_response(
            query=question,
            error=error_msg,
            error_type="ServerError",
            stage="backend_processing",
            http_status=status_code
        )
    
    
    def _format_validation_errors(self, details: list) -> str:
        """
        Format FastAPI validation errors into readable message.
        
        Args:
            details: List of Pydantic validation error dicts
        
        Returns:
            str: Human-readable error message
        """
        if not details:
            return "Validation error"
        
        # Extract first error (usually most relevant)
        first_error = details[0]
        field = " -> ".join(str(loc) for loc in first_error.get("loc", []))
        msg = first_error.get("msg", "Invalid input")
        
        if len(details) == 1:
            return f"Validation error in {field}: {msg}"
        else:
            return f"Validation error in {field}: {msg} (and {len(details)-1} more)"
    
    
    def _error_response(
        self,
        query: str,
        error: str,
        error_type: str,
        stage: str,
        http_status: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Create standardized error response.
        
        Args:
            query: Original question string
            error: Error message
            error_type: Error classification
            stage: Pipeline stage where error occurred
            http_status: Optional HTTP status code
        
        Returns:
            dict: Standardized error response
        """
        response = {
            "success": False,
            "query": query,
            "error": error,
            "error_type": error_type,
            "stage": stage,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if http_status is not None:
            response["http_status"] = http_status
        
        return response
    

