# ModelPipeline\finrag_ml_tg1\rag_modules_src\synthesis_pipeline\orchestrator.py

"""
QueryOrchestrator - Thin glue layer connecting supply lines to LLM.

Design Philosophy:
    - Minimal code - all heavy lifting in supply_lines, prompts, bedrock_client
    - Models for type safety - dicts for external flexibility
    - Logging built-in - every query tracked automatically
    - Clear data flow - each field's source is explicit

Responsibilities:
    1. Initialize components (config, supply lines, prompts, LLM, logger)
    2. Wire components together (supply → prompts → LLM)
    3. Track timing (optional processing_time_ms)
    4. Create typed responses (models for structure)
    5. Log everything (metadata, contexts, responses)
    6. Return dicts (external interoperability)

Does NOT:
    - Build prompts (done by PromptLoader)
    - Assemble context (done by supply_lines)
    - Extract entities (done by supply_lines)
    - Format anything (done by formatters)
"""

from pathlib import Path
from typing import Dict, Optional
import logging
import time

from finrag_ml_tg1.loaders.ml_config_loader import MLConfig
from finrag_ml_tg1.rag_modules_src.synthesis_pipeline.supply_lines import (
    init_rag_components,
    build_combined_context
)
from finrag_ml_tg1.rag_modules_src.prompts.prompt_loader import PromptLoader
from finrag_ml_tg1.rag_modules_src.synthesis_pipeline.bedrock_client import (
    create_bedrock_client_from_config
)
from finrag_ml_tg1.rag_modules_src.synthesis_pipeline.models import (
    create_success_response,
    create_error_response
)
from finrag_ml_tg1.rag_modules_src.synthesis_pipeline.query_logger import QueryLogger

logger = logging.getLogger(__name__)


def answer_query(
    query: str,
    model_root: Path,
    include_kpi: bool = True,
    include_rag: bool = True,
    model_key: Optional[str] = None,
    export_context: bool = True,
    export_response: bool = False
) -> Dict:
    """
    End-to-end query answering using FinRAG pipeline.
    
    This is the main entry point that wires together:
        1. Supply lines (KPI + RAG + assembly) → build_combined_context()
        2. Prompt system (YAML templates) → PromptLoader
        3. Bedrock client (LLM invocation) → BedrockClient
        4. Response models (type safety) → QueryResponse/ErrorResponse
        5. Query logger (persistence) → QueryLogger
    
    Data Flow:
        Query → Supply Lines → Prompts → Bedrock → Models → Logger → Dict
    
    Args:
        query: User's natural language question
        model_root: Path to ModelPipeline root directory
        include_kpi: Whether to include KPI data (Supply Line 1)
        include_rag: Whether to include RAG context (Supply Line 2)
        model_key: Optional model selection ('development', 'production_balanced', etc.)
                  Uses default from ml_config.yaml if None
        export_context: Whether to save assembled context to text file
        export_response: Whether to save full response to JSON file
    
    Returns:
        Dictionary (from QueryResponse.to_dict() or ErrorResponse.to_dict()):
        {
            'query': str,
            'answer': str,                   # None if error
            'context': str,                  # None if error
            'metadata': {
                'llm': {
                    'model_id': str,
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
                    'kpi_entities': Optional[Dict],
                    'rag_entities': Optional[Dict],
                    'retrieval_stats': Optional[Dict]
                },
                'timestamp': str,
                'processing_time_ms': Optional[float]
            },
            'exports': {
                'log_file': str,
                'context_file': Optional[str],
                'response_file': Optional[str]
            },
            'error': Optional[str],          # Only present if failed
            'error_type': Optional[str],     # Only present if failed
            'stage': Optional[str]           # Only present if failed
        }
    
    Example:
        >>> from pathlib import Path
        >>> result = answer_query(
        ...     query="What were NVIDIA's 2020 revenues?",
        ...     model_root=Path("/path/to/ModelPipeline")
        ... )
        >>> if result.get('error'):
        ...     print(f"Error: {result['error']}")
        ... else:
        ...     print(result['answer'])
        ...     print(f"Cost: ${result['metadata']['llm']['cost']:.4f}")
    
    Raises:
        Does NOT raise - all errors returned as ErrorResponse dicts
    """
    logger.info(f"answer_query called: '{query[:50]}...'")
    
    # Start timing (for processing_time_ms)
    start_time = time.time()
    
    # ========================================================================
    # INITIALIZATION - Create all components
    # ========================================================================
    
    try:
        # Config service (external service pattern)
        config = MLConfig()
        logger.debug("MLConfig loaded")
        
        # RAG components (entity adapter, embedder, retriever, assembler)
        rag_components = init_rag_components()
        logger.debug("RAG components initialized")
        
        # Prompt loader (YAML-based system + query templates)
        prompt_loader = PromptLoader()
        logger.debug("PromptLoader initialized")
        
        # Bedrock client (AWS API wrapper with cost tracking)
        llm_client = create_bedrock_client_from_config(config, model_key)
        logger.debug(f"BedrockClient initialized: {llm_client.model_id}")
        
        # Query logger (persistent logging)
        query_logger = QueryLogger()
        logger.debug("QueryLogger initialized (Always-S3 mode)")
        
    except Exception as e:
        logger.error(f"Initialization failed: {e}", exc_info=True)
        
        # Create error response (no metadata available yet)
        error_response = create_error_response(
            query=query,
            error=e,
            stage='initialization'
        )
        
        # Convert to dict and return (can't log if logger failed to init)
        return error_response.to_dict()
    
    # ========================================================================
    # CONTEXT BUILDING - Supply lines do all the heavy lifting
    # ========================================================================
    
    try:
        combined_context, context_metadata = build_combined_context(
            query=query,
            rag=rag_components,
            include_kpi=include_kpi,
            include_rag=include_rag
        )
        
        logger.info(
            f"Context built: {len(combined_context)} chars, "
            f"KPI={'yes' if context_metadata.get('kpi_entities') else 'no'}, "
            f"RAG={'yes' if context_metadata.get('rag_entities') else 'no'}"
        )
        
    except Exception as e:
        logger.error(f"Context building failed: {e}", exc_info=True)
        
        error_response = create_error_response(
            query=query,
            error=e,
            stage='context_building'
        )
        
        result = error_response.to_dict()
        
        # Log the error
        try:
            exports = query_logger.log_query(
                result=result,
                export_context=False,
                export_response=export_response
            )
            result['exports'] = exports
        except Exception as log_error:
            logger.error(f"Logging failed: {log_error}")
            ## creating dict; log_file as key for new dict, not access. not attribute.
            result['exports'] = {'log_file': None, 'logging_error': str(log_error)}
        
        return result  # Return the dict
    
    # ========================================================================
    # PROMPT FORMATTING - Wrap context in YAML templates
    # ========================================================================
    
    try:
        system_prompt = prompt_loader.load_system_prompt()
        user_prompt = prompt_loader.format_query_template(combined_context)
        
        logger.info(
            f"Prompts formatted: system={len(system_prompt)} chars, "
            f"user={len(user_prompt)} chars"
        )
        
    except Exception as e:
        logger.error(f"Prompt formatting failed: {e}", exc_info=True)
        
        error_response = create_error_response(
            query=query,
            error=e,
            stage='prompt_formatting'
        )
        
        result = error_response.to_dict()
        
        exports = query_logger.log_query(
            result=result,
            export_context=export_context,  # Save context for debugging
            export_response=export_response
        )
        result['exports'] = exports
        
        return result
    
    # ========================================================================
    # LLM INVOCATION - Call AWS Bedrock API
    # ========================================================================
    
    try:
        llm_response = llm_client.invoke(
            system=system_prompt,
            user=user_prompt
        )
        
        logger.info(
            f"LLM response received: {llm_response['usage']['output_tokens']} tokens, "
            f"cost=${llm_response['cost']:.4f}, "
            f"stop_reason={llm_response['stop_reason']}"
        )
        
    except Exception as e:
        logger.error(f"LLM invocation failed: {e}", exc_info=True)
        
        error_response = create_error_response(
            query=query,
            error=e,
            stage='llm_invocation'
        )
        
        result = error_response.to_dict()
        
        exports = query_logger.log_query(
            result=result,
            export_context=export_context,  # Save context to debug what was sent
            export_response=export_response
        )
        result['exports'] = exports
        
        return result
    
    # ========================================================================
    # RESPONSE PACKAGING - Create typed models and convert to dict
    # ========================================================================
    
    try:
        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Create typed response (models for structure)
        # Factory function handles all field population per responsibility matrix:
        #   - model_id: from llm_response (originally from MLConfig)
        #   - input_tokens: from llm_response (from AWS Bedrock)
        #   - output_tokens: from llm_response (from AWS Bedrock)
        #   - total_tokens: calculated by factory (input + output)
        #   - cost: from llm_response (calculated by BedrockClient using MLConfig rates)
        #   - stop_reason: from llm_response (from AWS Bedrock)
        #   - kpi_included: from function parameter
        #   - rag_included: from function parameter
        #   - context_length: calculated by factory (len(context))
        #   - kpi_entities: from context_metadata (from supply_lines → EntityAdapter)
        #   - rag_entities: from context_metadata (from supply_lines → EntityAdapter)
        #   - retrieval_stats: from context_metadata (from supply_lines → S3VectorsRetriever)
        #   - timestamp: calculated by factory (datetime.utcnow())
        #   - processing_time_ms: measured by orchestrator
        response = create_success_response(
            query=query,
            answer=llm_response['content'],
            context=combined_context,
            llm_response=llm_response,
            context_metadata=context_metadata,
            processing_time_ms=processing_time_ms
        )
        
        # Convert to dict (external interoperability)
        result = response.to_dict()
        
        logger.info("Response packaging complete")
        
    except Exception as e:
        logger.error(f"Response packaging failed: {e}", exc_info=True)
        
        error_response = create_error_response(
            query=query,
            error=e,
            stage='response_packaging'
        )
        
        result = error_response.to_dict()
        
        exports = query_logger.log_query(
            result=result,
            export_context=export_context,
            export_response=export_response
        )
        result['exports'] = exports
        
        return result
    
    # ========================================================================
    # LOGGING - Persist metadata, context, response
    # ========================================================================
    
    try:
        # Log to Parquet + export files
        exports = query_logger.log_query(
            result=result,
            export_context=export_context,
            export_response=export_response
        )
        
        ## -- exports is a dict returned by log_query(), which contains 'log_file'.
        # Add export paths to result
        result['exports'] = exports
        
        logger.info(
            f"Query logged: log={exports['log_file']}, "
            f"context={exports.get('context_file', 'not exported')}"
        )
        
    except Exception as e:
        # Logging failure should NOT crash query
        logger.error(f"Logging failed (non-fatal): {e}", exc_info=True)
        
        # Add minimal exports info // (creating dict key)
        result['exports'] = {
            'log_file': None,
            'context_file': None,
            'response_file': None,
            'logging_error': str(e)
        }
    
    # ========================================================================
    # RETURN - Success response with all metadata
    # ========================================================================
    
    logger.info("Query processing complete")
    return result


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def answer_query_batch(
    queries: list[str],
    model_root: Path,
    include_kpi: bool = True,
    include_rag: bool = True,
    model_key: Optional[str] = None,
    export_contexts: bool = True,
    export_responses: bool = True
) -> list[Dict]:
    """
    Process multiple queries in sequence.
    
    Useful for evaluation harness (P3 gold set).
    
    Args:
        queries: List of user questions
        model_root: Path to ModelPipeline root
        include_kpi: Include KPI data for all queries
        include_rag: Include RAG context for all queries
        model_key: Model selection (same for all queries)
        export_contexts: Export contexts (generates many files)
        export_responses: Export responses (generates many files)
    
    Returns:
        List of result dictionaries (one per query)
    
    Example:
        >>> gold_queries = [
        ...     "What were NVIDIA's 2020 revenues?",
        ...     "How did Microsoft describe AI risks in 2021?"
        ... ]
        >>> results = answer_query_batch(gold_queries, model_root)
        >>> for r in results:
        ...     print(f"Q: {r['query']}")
        ...     print(f"A: {r['answer']}")
        ...     print(f"Cost: ${r['metadata']['llm']['cost']:.4f}")
    """
    logger.info(f"Batch processing {len(queries)} queries")
    
    results = []
    for i, q in enumerate(queries, 1):
        logger.info(f"Processing batch query {i}/{len(queries)}")
        
        result = answer_query(
            query=q,
            model_root=model_root,
            include_kpi=include_kpi,
            include_rag=include_rag,
            model_key=model_key,
            export_context=export_contexts,
            export_response=export_responses
        )
        
        results.append(result)
        
        # Log progress
        if not result.get('error'):
            logger.info(f"  ✓ Success: ${result['metadata']['llm']['cost']:.4f}")
        else:
            logger.warning(f"  ✗ Failed: {result['error']}")
    
    logger.info(f"Batch complete: {len(results)} results")
    return results


def get_query_stats(model_root: Path) -> Dict:
    """
    Get statistics from query logs.
    
    Args:
        model_root: Path to ModelPipeline root
        
    Returns:
        Dictionary with cost and usage statistics
    
    Example:
        >>> stats = get_query_stats(model_root)
        >>> print(f"Total cost: ${stats['total_cost']:.2f}")
        >>> print(f"Queries: {stats['total_queries']}")
    """
    query_logger = QueryLogger()
    return query_logger.get_cost_summary()


def get_recent_queries(model_root: Path, n: int = 10):
    """
    Get recent query logs.
    
    Args:
        model_root: Path to ModelPipeline root
        n: Number of recent queries to retrieve
        
    Returns:
        Polars DataFrame with recent logs
    
    Example:
        >>> recent = get_recent_queries(model_root, n=5)
        >>> print(recent[['timestamp', 'query', 'cost']])
    """
    query_logger = QueryLogger()
    return query_logger.get_recent_logs(n=n)


# ============================================================================
# LEGACY COMPATIBILITY (Optional - can remove if not needed)
# ============================================================================

class QueryOrchestrator:
    """
    Legacy class wrapper for backward compatibility.
    
    New code should use answer_query() function directly.
    This exists only if old code expects a class-based interface.
    """
    
    def __init__(self, model_root: Path):
        """
        Initialize orchestrator with model root.
        
        Args:
            model_root: Path to ModelPipeline directory
        """
        self.model_root = model_root
        logger.info(f"QueryOrchestrator (legacy) initialized: {model_root}")
    
    def process_query(
        self,
        user_query: str,
        include_kpi: bool = True,
        include_rag: bool = True,
        model_key: Optional[str] = None
    ) -> Dict:
        """
        Process query using orchestrator's model_root.
        
        Args:
            user_query: Natural language question
            include_kpi: Include KPI data
            include_rag: Include RAG context
            model_key: Optional model selection
            
        Returns:
            Response dictionary (same as answer_query)
        """
        return answer_query(
            query=user_query,
            model_root=self.model_root,
            include_kpi=include_kpi,
            include_rag=include_rag,
            model_key=model_key
        )


def create_orchestrator(model_root: Path) -> QueryOrchestrator:
    """
    Factory function for creating orchestrator instance.
    
    Args:
        model_root: Path to ModelPipeline directory
        
    Returns:
        Initialized QueryOrchestrator (legacy interface)
    """
    return QueryOrchestrator(model_root)