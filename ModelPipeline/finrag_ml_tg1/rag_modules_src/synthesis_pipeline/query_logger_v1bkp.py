# # ModelPipeline\finrag_ml_tg1\rag_modules_src\synthesis_pipeline\query_logger.py

# """
# Query Logger - Persistent logging for all FinRAG queries.

# Logs minimal metadata to Parquet for analytics/monitoring.
# Optionally exports full contexts and responses.

# Design:
#     - Append-only Parquet log (never delete, only grow)
#     - Separate text exports for human review
#     - Automatic directory creation
#     - Thread-safe append operations
# """

# import polars as pl
# from pathlib import Path
# from datetime import datetime
# from typing import Dict, Optional
# import logging

# logger = logging.getLogger(__name__)


# class QueryLogger:
#     """
#     Persistent logger for FinRAG query metadata.
    
#     Logs every query (success or failure) to:
#         1. Parquet file - minimal metadata for analytics
#         2. Text file - full assembled context (optional)
#         3. JSON file - full response (optional)
#     """
    
#     def __init__(self, model_root: Optional[Path] = None):
#         """
#         Initialize logger with automatic ModelPipeline root resolution.
        
#         Follows project pattern:
#             1. Walk up to 'ModelPipeline' root
#             2. Define absolute paths from root
#             3. No .resolve() hacks, no __file__ traversal
        
#         Exports structure:
#             ModelPipeline/
#             └── finrag_ml_tg1/
#                 └── rag_modules_src/
#                     └── exports/              # 1 level higher than synthesis_pipeline
#                         ├── contexts/
#                         ├── responses/
#                         └── logs/
#                             └── query_logs.parquet
#         """
#         ## Find ModelPipeline root
#         if model_root is None:
#             model_root = self._find_model_root()  # Fallback

#         if model_root is None:
#             raise RuntimeError( "Cannot find 'ModelPipeline' root directory. Ensure you're running from within the project." )
        
#         # Define absolute paths from model_root (no path resolution hacks)
#         self.model_root = model_root
#         self.exports_dir = model_root / "finrag_ml_tg1" / "rag_modules_src" / "exports"
#         self.contexts_dir = self.exports_dir / "contexts"
#         self.responses_dir = self.exports_dir / "responses"
#         self.logs_dir = self.exports_dir / "logs"
        
#         # Ensure directories exist
#         self.contexts_dir.mkdir(parents=True, exist_ok=True)
#         self.responses_dir.mkdir(parents=True, exist_ok=True)
#         self.logs_dir.mkdir(parents=True, exist_ok=True)
        
#         # Parquet log file path
#         self.log_file = self.logs_dir / "query_logs.parquet"
        
#         logger.info(f"QueryLogger initialized")
#         logger.info(f"  Model root: {self.model_root}")
#         logger.info(f"  Exports: {self.exports_dir}")
#         logger.info(f"  Log file: {self.log_file}")

    
#     def log_query(
#         self,
#         result: Dict,
#         export_context: bool = True,
#         export_response: bool = False
#     ) -> Dict[str, Optional[str]]:
#         """
#         Log a query result (success or error).
        
#         Always logs metadata to Parquet.
#         Optionally exports context and full response.
        
#         Args:
#             result: Dictionary from answer_query() (QueryResponse or ErrorResponse)
#             export_context: Whether to save full context to text file
#             export_response: Whether to save full response to JSON file
            
#         Returns:
#             Dictionary with export file paths:
#             {
#                 'log_file': str,
#                 'context_file': Optional[str],
#                 'response_file': Optional[str]
#             }
#         """
#         timestamp = result.get('metadata', {}).get('timestamp') or datetime.utcnow().isoformat() + 'Z'
        
#         # Generate filename suffix (timestamp without special chars)
#         file_suffix = timestamp.replace(':', '').replace('-', '').replace('.', '')[:15]
        
#         # Export context if requested and available
#         context_file = None
#         if export_context and result.get('context'):
#             context_file = self._export_context(
#                 context=result['context'],
#                 suffix=file_suffix
#             )
        
#         # Export full response if requested
#         response_file = None
#         if export_response:
#             response_file = self._export_response(
#                 result=result,
#                 suffix=file_suffix
#             )
        
#         # Extract metadata for Parquet log
#         is_error = 'error' in result and result['error'] is not None
        
#         if is_error:
#             # Error case
#             log_entry = {
#                 'timestamp': timestamp,
#                 'query': result.get('query', ''),
#                 'model_id': None,
#                 'input_tokens': None,
#                 'output_tokens': None,
#                 'total_tokens': None,
#                 'cost': None,
#                 'context_length': None,
#                 'processing_time_ms': None,
#                 'error': result.get('error'),
#                 'error_type': result.get('error_type'),
#                 'stage': result.get('stage'),
#                 'context_file': context_file,
#                 'response_file': response_file
#             }
#         else:
#             # Success case
#             metadata = result.get('metadata', {})
#             llm_meta = metadata.get('llm', {})
#             context_meta = metadata.get('context', {})
            
#             log_entry = {
#                 'timestamp': timestamp,
#                 'query': result.get('query', ''),
#                 'model_id': llm_meta.get('model_id'),
#                 'input_tokens': llm_meta.get('input_tokens'),
#                 'output_tokens': llm_meta.get('output_tokens'),
#                 'total_tokens': llm_meta.get('total_tokens'),
#                 'cost': llm_meta.get('cost'),
#                 'context_length': context_meta.get('context_length'),
#                 'processing_time_ms': metadata.get('processing_time_ms'),
#                 'error': None,
#                 'error_type': None,
#                 'stage': None,
#                 'context_file': context_file,
#                 'response_file': response_file
#             }
        
#         # Append to Parquet log
#         self._append_to_log(log_entry)
        
#         logger.info(f"Query logged: {timestamp}, error={is_error}")
        
#         return {
#             'log_file': str(self.log_file),
#             'context_file': context_file,
#             'response_file': response_file
#         }
    
#     def _export_context(self, context: str, suffix: str) -> str:
#         """Export assembled context to text file."""
#         filename = f"context_{suffix}.txt"
#         filepath = self.contexts_dir / filename
        
#         filepath.write_text(context, encoding='utf-8')
        
#         logger.debug(f"Context exported: {filepath}")
#         return str(filepath)
    
#     def _export_response(self, result: Dict, suffix: str) -> str:
#         """Export full response to JSON file."""
#         import json
        
#         filename = f"response_{suffix}.json"
#         filepath = self.responses_dir / filename
        
#         filepath.write_text(
#             json.dumps(result, indent=2, ensure_ascii=False),
#             encoding='utf-8'
#         )
        
#         logger.debug(f"Response exported: {filepath}")
#         return str(filepath)
    


#     def _append_to_log(self, log_entry: Dict):
#         """
#         Append log entry to Parquet file.
        
#         Creates file if doesn't exist, appends if exists.
#         Uses explicit schema to handle None/String type mixing.
#         """
#         # Define explicit schema (handles None properly)
#         schema = {
#             'timestamp': pl.Utf8,
#             'query': pl.Utf8,
#             'model_id': pl.Utf8,
#             'input_tokens': pl.Int64,
#             'output_tokens': pl.Int64,
#             'total_tokens': pl.Int64,
#             'cost': pl.Float64,
#             'context_length': pl.Int64,
#             'processing_time_ms': pl.Float64,
#             'error': pl.Utf8,           # allow nulls
#             'error_type': pl.Utf8,      # 
#             'stage': pl.Utf8,           # 
#             'context_file': pl.Utf8,    # 
#             'response_file': pl.Utf8    # 
#         }
        
#         # Create DataFrame with explicit schema
#         new_df = pl.DataFrame(log_entry, schema=schema)
        
#         # Append to existing or create new
#         if self.log_file.exists():
#             # Read existing with same schema
#             existing_df = pl.read_parquet(self.log_file)
            
#             # Ensure schemas match (cast if needed)
#             new_df = new_df.cast(existing_df.schema)
            
#             # Append new row
#             combined_df = pl.concat([existing_df, new_df], how='vertical')
            
#             # Write back
#             combined_df.write_parquet(self.log_file)
#         else:
#             # Create new file with explicit schema
#             new_df.write_parquet(self.log_file)
        
#         logger.debug(f"Log entry appended to: {self.log_file}")
    


#     def get_recent_logs(self, n: int = 10) -> pl.DataFrame:
#         if not self.log_file.exists():
#             # Return empty DataFrame with explicit schema
#             return pl.DataFrame(schema={
#                 'timestamp': pl.Utf8,
#                 'query': pl.Utf8,
#                 'model_id': pl.Utf8,
#                 'input_tokens': pl.Int64,
#                 'output_tokens': pl.Int64,
#                 'total_tokens': pl.Int64,
#                 'cost': pl.Float64,
#                 'context_length': pl.Int64,
#                 'processing_time_ms': pl.Float64,
#                 'error': pl.Utf8,
#                 'error_type': pl.Utf8,
#                 'stage': pl.Utf8,
#                 'context_file': pl.Utf8,
#                 'response_file': pl.Utf8
#             })
        
#         df = pl.read_parquet(self.log_file)
#         return df.tail(n)
    


#     def get_cost_summary(self) -> Dict:
#         """
#         Calculate cost summary from logs.
        
#         Returns:
#             Dictionary with cost statistics:
#             {
#                 'total_queries': int,
#                 'successful_queries': int,
#                 'failed_queries': int,
#                 'total_cost': float,
#                 'total_tokens': int,
#                 'avg_cost_per_query': float
#             }
#         """
#         if not self.log_file.exists():
#             return {
#                 'total_queries': 0,
#                 'successful_queries': 0,
#                 'failed_queries': 0,
#                 'total_cost': 0.0,
#                 'total_tokens': 0,
#                 'avg_cost_per_query': 0.0
#             }
        
#         df = pl.read_parquet(self.log_file)
        
#         # Handle empty DataFrame
#         if len(df) == 0:
#             return {
#                 'total_queries': 0,
#                 'successful_queries': 0,
#                 'failed_queries': 0,
#                 'total_cost': 0.0,
#                 'total_tokens': 0,
#                 'avg_cost_per_query': 0.0
#             }
        
#         total_queries = len(df)
        
#         # : Handle nullable columns properly
#         successful = df.filter(pl.col('error').is_null())
#         failed = df.filter(pl.col('error').is_not_null())
        
#         # : Safe aggregation (handle empty successful)
#         if len(successful) > 0:
#             total_cost = successful['cost'].sum()
#             total_tokens = successful['total_tokens'].sum()
#             avg_cost = total_cost / len(successful)
#         else:
#             total_cost = 0.0
#             total_tokens = 0
#             avg_cost = 0.0
        
#         return {
#             'total_queries': total_queries,
#             'successful_queries': len(successful),
#             'failed_queries': len(failed),
#             'total_cost': float(total_cost) if total_cost is not None else 0.0,
#             'total_tokens': int(total_tokens) if total_tokens is not None else 0,
#             'avg_cost_per_query': float(avg_cost) if avg_cost is not None else 0.0
#         }
