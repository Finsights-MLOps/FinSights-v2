# How to Query the RAG Model

This guide shows all the ways you can query the FinRAG model, apart from the frontend.

## Option 1: Simple Standalone Script (Recommended for Quick Testing)

Use the `query_rag.py` script for the easiest way to run queries:

### Basic Usage
```bash
cd ModelPipeline
python query_rag.py "What was NVIDIA's revenue in 2023?"
```

### Interactive Mode
```bash
python query_rag.py --interactive
```

This lets you enter multiple queries in a loop.

### With Options
```bash
# Use a specific model
python query_rag.py --query "Your question" --model development

# Disable KPI or RAG
python query_rag.py --query "Your question" --no-kpi
python query_rag.py --query "Your question" --no-rag

# Export response to JSON
python query_rag.py --query "Your question" --export-response
```

## Option 2: Use Existing CLI (Advanced with MLflow Tracking)

The synthesis pipeline has a full-featured CLI with MLflow tracking:

```bash
cd ModelPipeline
python -m finrag_ml_tg1.rag_modules_src.synthesis_pipeline.main --query "Your question"
```

### Options
```bash
# Use different model
python -m finrag_ml_tg1.rag_modules_src.synthesis_pipeline.main --model production_budget

# Disable MLflow tracking
python -m finrag_ml_tg1.rag_modules_src.synthesis_pipeline.main --no-tracking

# Export full response
python -m finrag_ml_tg1.rag_modules_src.synthesis_pipeline.main --export-response

# Custom MLflow experiment
python -m finrag_ml_tg1.rag_modules_src.synthesis_pipeline.main --experiment "MyExperiment"
```

## Option 3: Use FastAPI Backend (HTTP API)

Start the backend server and make HTTP requests:

### Start Server
```bash
cd ModelPipeline/serving
uvicorn backend.api_service:app --reload --host 0.0.0.0 --port 8000
```

### Query via HTTP
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What was NVIDIA'\''s revenue in 2023?",
    "include_kpi": true,
    "include_rag": true
  }'
```

### Or use Python requests
```python
import requests

response = requests.post(
    "http://localhost:8000/query",
    json={
        "question": "What was NVIDIA's revenue in 2023?",
        "include_kpi": True,
        "include_rag": True
    }
)

result = response.json()
print(result['answer'])
```

### View API Docs
Open http://localhost:8000/docs in your browser for interactive API documentation.

## Option 4: Use Directly in Your Python Code

Import the orchestrator function directly:

```python
import sys
from pathlib import Path

# Add ModelPipeline to path
MODEL_PIPELINE_ROOT = Path("/path/to/ModelPipeline")
if str(MODEL_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_PIPELINE_ROOT))

from finrag_ml_tg1.rag_modules_src.synthesis_pipeline.orchestrator import answer_query

# Find model root
model_root = MODEL_PIPELINE_ROOT  # or use Path(__file__).parent if script is in ModelPipeline

# Run query
result = answer_query(
    query="What was NVIDIA's revenue in 2023?",
    model_root=model_root,
    include_kpi=True,
    include_rag=True,
    model_key=None,  # Use default model
    export_context=False,
    export_response=False
)

# Check for errors
if result.get("error"):
    print(f"Error: {result['error']}")
else:
    print(result['answer'])
    print(f"Cost: ${result['metadata']['llm']['cost']:.4f}")
```

See `example_direct_query.py` for a complete example.

## Response Structure

All methods return the same response dictionary:

```python
{
    'query': str,                    # Your original query
    'answer': str,                   # LLM response (None if error)
    'context': str,                  # Full context sent to LLM (None if error)
    'metadata': {
        'llm': {
            'model_id': str,         # Model used
            'input_tokens': int,
            'output_tokens': int,
            'total_tokens': int,
            'cost': float,
            'stop_reason': str
        },
        'context': {
            'kpi_included': bool,
            'rag_included': bool,
            'context_length': int,
            'kpi_entities': dict,    # Extracted KPI entities
            'rag_entities': dict,    # Extracted RAG entities
            'retrieval_stats': dict  # Retrieval statistics
        },
        'timestamp': str,
        'processing_time_ms': float
    },
    'exports': {
        'log_file': str,             # Path to query log (always present)
        'context_file': str,         # Path to context file (if exported)
        'response_file': str         # Path to response file (if exported)
    },
    'error': str,                    # Error message (only if failed)
    'error_type': str,               # Error type (only if failed)
    'stage': str                     # Error stage (only if failed)
}
```

## Which Option Should I Use?

- **Quick testing**: Use `query_rag.py` (Option 1)
- **Experiments with tracking**: Use the CLI with MLflow (Option 2)
- **Integration with other services**: Use FastAPI backend (Option 3)
- **Custom Python scripts**: Import orchestrator directly (Option 4)

## Notes

- All queries are automatically logged to S3 (via QueryLogger)
- The model root path should point to the `ModelPipeline` directory
- AWS credentials must be configured for the RAG retrieval to work
- Processing time is typically 20-40 seconds depending on query complexity

