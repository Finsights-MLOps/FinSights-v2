#!/usr/bin/env python3
"""
Example: How to use the RAG model programmatically in your own Python scripts.

This shows how to import and use the answer_query function directly.
"""

import sys
from pathlib import Path

# Add ModelPipeline to Python path
MODEL_PIPELINE_ROOT = Path(__file__).parent
if str(MODEL_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_PIPELINE_ROOT))

from finrag_ml_tg1.rag_modules_src.synthesis_pipeline.orchestrator import answer_query


def find_model_root() -> Path:
    """Find ModelPipeline root directory."""
    current = Path(__file__).resolve().parent
    
    if current.name == "ModelPipeline":
        return current
    
    for parent in current.parents:
        if parent.name == "ModelPipeline":
            return parent
    
    return current


def main():
    """Example: Query the RAG model programmatically."""
    
    # Find the model root
    model_root = find_model_root()
    
    # Example query
    query = "What was NVIDIA's revenue in 2023?"
    
    print(f"Query: {query}\n")
    print("Processing...\n")
    
    # Call the orchestrator directly
    result = answer_query(
        query=query,
        model_root=model_root,
        include_kpi=True,      # Include KPI data
        include_rag=True,      # Include RAG context
        model_key=None,        # Use default model from config
        export_context=False,  # Don't save context file
        export_response=False  # Don't save response file
    )
    
    # Check for errors
    if result.get("error"):
        print(f"❌ Error: {result['error']}")
        print(f"   Stage: {result.get('stage', 'Unknown')}")
        return
    
    # Access the answer
    answer = result.get("answer", "")
    metadata = result.get("metadata", {})
    llm_meta = metadata.get("llm", {})
    
    # Print results
    print("Answer:")
    print("-" * 80)
    print(answer)
    print("-" * 80)
    print(f"\nCost: ${llm_meta.get('cost', 0):.4f}")
    print(f"Tokens: {llm_meta.get('total_tokens', 0):,}")
    
    # You can also access:
    # - result['query'] - Original query
    # - result['context'] - Full context sent to LLM
    # - result['metadata']['context']['kpi_entities'] - Extracted entities
    # - result['metadata']['context']['rag_entities'] - RAG entities
    # - result['metadata']['llm']['model_id'] - Model used
    # - result['exports']['log_file'] - Path to query log


if __name__ == "__main__":
    main()

