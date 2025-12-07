"""
Pydantic models for FinRAG API request/response validation.

These models define the contract between frontend and backend:
- What data the API expects (QueryRequest)
- What data the API returns (QueryResponse, ErrorResponse)
- Automatic validation (Pydantic rejects invalid requests)

Pydantic v2 reserves field names starting with model_ for internal use:
model_dump() - serialize model
model_fields - introspect fields
model_validate() - validate data
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


# ============================================================================
# REQUEST MODELS
# ============================================================================

class QueryRequest(BaseModel):
    """
    Request model for /query endpoint.
    
    Frontend sends this structure when user asks a question.
    
    Example:
        {
            "question": "What was Apple's revenue in 2023?",
            "include_kpi": true,
            "include_rag": true,
            "model_key": "production_balanced"
        }
    """
    model_config = ConfigDict(protected_namespaces=())
    
    question: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description="User's natural language question about 10-K filings",
        examples=["What was Microsoft's cloud revenue growth in 2023?"]
    )
    
    include_kpi: bool = Field(
        default=True,
        description="Whether to include structured KPI data (Supply Line 1)"
    )
    
    include_rag: bool = Field(
        default=True,
        description="Whether to include semantic RAG context (Supply Line 2)"
    )
    
    model_key: Optional[str] = Field(
        default=None,
        description="Model selection: 'development', 'production_balanced', 'production_premium'"
    )


# ============================================================================
# RESPONSE MODELS
# ============================================================================

class LLMMetadata(BaseModel):
    """Metadata about LLM invocation (tokens, cost, model)."""
    model_config = ConfigDict(protected_namespaces=())
    
    model_id: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost: float
    stop_reason: str


class ContextMetadata(BaseModel):
    """Metadata about context assembly (KPI, RAG, entities)."""
    kpi_included: bool
    rag_included: bool
    context_length: int
    kpi_entities: Optional[Dict[str, Any]] = None
    rag_entities: Optional[Dict[str, Any]] = None
    retrieval_stats: Optional[Dict[str, Any]] = None


class ResponseMetadata(BaseModel):
    """Complete metadata combining LLM and context info."""
    llm: LLMMetadata
    context: ContextMetadata
    timestamp: str
    processing_time_ms: Optional[float] = None


class QueryResponse(BaseModel):
    """
    Successful response from /query endpoint.
    
    This is what frontend receives when query succeeds.
    
    Example:
        {
            "query": "What was Apple's revenue?",
            "answer": "Apple's fiscal year 2023 revenue was...",
            "context": "=== [AAPL] APPLE INC | FY 2023...",
            "metadata": {...},
            "exports": {...}
        }
    """
    query: str
    answer: str
    context: str
    metadata: ResponseMetadata

    ## NOTE: Exports are internal backend concern, not exposed to API consumers
    # The frontend should not have a design where they access the disk usage or spillage. 
    # It could be a security risk for the hardware or cloud or disk.
        # exports: Optional[Dict[str, Any]] = Field(
        #     default=None,
        #     description="File paths for exported logs/context (optional)"
        # )


class ErrorResponse(BaseModel):
    """
    Error response when query fails.
    
    Example:
        {
            "query": "What was...",
            "error": "AWS Bedrock rate limit exceeded",
            "error_type": "ClientError",
            "stage": "llm_invocation",
            "timestamp": "2024-11-21T10:30:00Z"
        }
    """
    query: str
    error: str
    error_type: str
    stage: str
    timestamp: str
    answer: None = None
    context: None = None
    metadata: None = None


# ============================================================================
# HEALTH CHECK MODEL
# ============================================================================

class HealthResponse(BaseModel):
    """Response for /health endpoint."""
    model_config = ConfigDict(protected_namespaces=())
    
    status: str = Field(default="healthy")
    model_root_exists: bool
    aws_configured: Optional[bool] = None    
    timestamp: str