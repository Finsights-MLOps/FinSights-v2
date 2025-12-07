# ModelPipeline\finrag_ml_tg1\rag_modules_src\synthesis_pipeline\query_logger.py

"""
Query Logger - Always-S3 persistence for FinRAG queries.

Design Philosophy:
    - Always writes to S3 (local dev AND Lambda)
    - S3 as single source of truth
    - Optional local sync for notebook analytics
    - No environment-specific if-else logic
    - All S3 paths from MLConfig (config-driven)

Logs every query to S3:
    1. Individual JSON log entries (one per query)
    2. Full assembled contexts (optional)
    3. Complete responses with metadata (optional)

    - public API (log_query(), get_recent_logs(), get_cost_summary())
Local Analytics:
    - Use sync_to_local() to download logs for Polars analysis
    - Smart sync: only downloads new/changed files
    - Automatic overwrite of stale local files
"""

import polars as pl
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple
import logging
import json
import io

logger = logging.getLogger(__name__)


class QueryLogger:
    """
    Always-S3 query logger with optional local sync.
    
    Design:
        - Single code path (no mode switching)
        - S3 persistence (works in local dev and Lambda)
        - Config-driven paths (MLConfig properties)
        - Optional local sync for analytics
    """
    
    def __init__(self):
        """
        Initialize QueryLogger with S3 persistence (environment-agnostic).
        
        Note:
            - No model_root parameter needed
            - MLConfig handles all path configuration
            - Works identically everywhere (local, Lambda, anywhere)
        """
        from finrag_ml_tg1.loaders.ml_config_loader import MLConfig
        
        config = MLConfig()
        
        self.config = config
        self.s3_client = config.get_s3_client()
        self.bucket = config.bucket
        
        # Single S3 Parquet file (not individual JSONs!)
        self.s3_log_key = "DATA_MERGE_ASSETS/LOGS/FINRAG/logs/query_logs.parquet"
        self.s3_log_uri = f"s3://{self.bucket}/{self.s3_log_key}"
        
        # Context/response prefixes (unchanged)
        self.s3_contexts_prefix = config.query_contexts_path
        self.s3_responses_prefix = config.query_responses_path
        
        # Local sync directory (optional, for analytics)
        if config.model_root:
            self.local_sync_dir = config.model_root / "finrag_ml_tg1" / "rag_modules_src" / "exports"
            self.local_log_file = self.local_sync_dir / "logs" / "query_logs.parquet"
        else:
            self.local_sync_dir = None
            self.local_log_file = None
        
        logger.info("QueryLogger initialized: Single Parquet file mode")
        logger.info(f"  S3 log file: {self.s3_log_uri}")
        logger.info(f"  Contexts: s3://{self.bucket}/{self.s3_contexts_prefix}")
        logger.info(f"  Responses: s3://{self.bucket}/{self.s3_responses_prefix}")
        if self.local_log_file:
            logger.info(f"  Local cache: {self.local_log_file}")
    
    
    def log_query(
        self,
        result: Dict,
        export_context: bool = True,
        export_response: bool = False
    ) -> Dict[str, Optional[str]]:
        """
        Log a query result to S3.
        
        Always writes to S3 (local dev and Lambda).
        
        Args:
            result: Dictionary from answer_query() (QueryResponse or ErrorResponse)
            export_context: Whether to save full context to S3
            export_response: Whether to save full response to S3
            
        Returns:
            Dictionary with S3 URIs:
            {
                'log_file': str (S3 URI),
                'context_file': Optional[str] (S3 URI),
                'response_file': Optional[str] (S3 URI)
            }
        """
        timestamp = result.get('metadata', {}).get('timestamp') or datetime.utcnow().isoformat() + 'Z'
        
        # Generate filename suffix (timestamp without special chars)
        file_suffix = timestamp.replace(':', '').replace('-', '').replace('.', '')[:15]
        
        # Export context if requested and available
        context_file = None
        if export_context and result.get('context'):
            context_file = self._export_context(
                context=result['context'],
                suffix=file_suffix
            )
        
        # Export full response if requested
        response_file = None
        if export_response:
            response_file = self._export_response(
                result=result,
                suffix=file_suffix
            )
        
        # Extract metadata for log entry
        is_error = 'error' in result and result['error'] is not None
        
        if is_error:
            # Error case
            log_entry = {
                'timestamp': timestamp,
                'query': result.get('query', ''),
                'model_id': None,
                'input_tokens': None,
                'output_tokens': None,
                'total_tokens': None,
                'cost': None,
                'context_length': None,
                'processing_time_ms': None,
                'error': result.get('error'),
                'error_type': result.get('error_type'),
                'stage': result.get('stage'),
                'context_file': context_file,
                'response_file': response_file
            }
        else:
            # Success case
            metadata = result.get('metadata', {})
            llm_meta = metadata.get('llm', {})
            context_meta = metadata.get('context', {})
            
            log_entry = {
                'timestamp': timestamp,
                'query': result.get('query', ''),
                'model_id': llm_meta.get('model_id'),
                'input_tokens': llm_meta.get('input_tokens'),
                'output_tokens': llm_meta.get('output_tokens'),
                'total_tokens': llm_meta.get('total_tokens'),
                'cost': llm_meta.get('cost'),
                'context_length': context_meta.get('context_length'),
                'processing_time_ms': metadata.get('processing_time_ms'),
                'error': None,
                'error_type': None,
                'stage': None,
                'context_file': context_file,
                'response_file': response_file
            }
        
        # Write log entry to S3
        log_s3_uri = self._append_to_log(log_entry)
        
        logger.info(f"Query logged to S3: {timestamp}, error={is_error}")
        
        return {
            'log_file': log_s3_uri,
            'context_file': context_file,
            'response_file': response_file
        }
    
    
    def _export_context(self, context: str, suffix: str) -> str:
        """
        Export assembled context to S3.
        
        Args:
            context: Assembled context string
            suffix: Timestamp suffix for filename
        
        Returns:
            S3 URI of uploaded context
        """
        filename = f"context_{suffix}.txt"
        s3_key = f"{self.s3_contexts_prefix}/{filename}"
        
        # Upload to S3
        self.s3_client.put_object(
            Bucket=self.bucket,
            Key=s3_key,
            Body=context.encode('utf-8'),
            ContentType='text/plain; charset=utf-8'
        )
        
        s3_uri = f"s3://{self.bucket}/{s3_key}"
        logger.debug(f"Context exported to S3: {s3_uri}")
        
        return s3_uri
    
    
    def _export_response(self, result: Dict, suffix: str) -> str:
        """
        Export full response to S3.
        
        Args:
            result: Full result dictionary
            suffix: Timestamp suffix for filename
        
        Returns:
            S3 URI of uploaded response
        """
        filename = f"response_{suffix}.json"
        s3_key = f"{self.s3_responses_prefix}/{filename}"
        
        # Convert to JSON
        response_json = json.dumps(result, indent=2, ensure_ascii=False)
        
        # Upload to S3
        self.s3_client.put_object(
            Bucket=self.bucket,
            Key=s3_key,
            Body=response_json.encode('utf-8'),
            ContentType='application/json; charset=utf-8'
        )
        
        s3_uri = f"s3://{self.bucket}/{s3_key}"
        logger.debug(f"Response exported to S3: {s3_uri}")
        
        return s3_uri
    
        
    def _append_to_log(self, log_entry: Dict) -> str:
        """
        Append log entry to single Parquet file in S3.
        
        Pattern: Download → Append → Re-upload
        Cost: ~$0.000023 per call (negligible for 200KB file)
        
        Args:
            log_entry: Dictionary with query metadata
        
        Returns:
            S3 URI of the single Parquet file
        """
        try:
            # Download existing Parquet from S3
            df_existing = pl.read_parquet(
                self.s3_log_uri,
                storage_options=self.config.get_storage_options()
            )
            logger.debug(f"Downloaded existing log: {len(df_existing)} rows")
            
        except Exception as e:
            # File doesn't exist yet - create new
            logger.info("No existing log file, creating new")
            df_existing = self._empty_log_dataframe()
        
        # Append new entry
        df_new = pl.DataFrame([log_entry])
        df_combined = pl.concat([df_existing, df_new])
        
        logger.debug(f"Appended row, total: {len(df_combined)}")
        
        # Re-upload to S3
        buffer = io.BytesIO()
        df_combined.write_parquet(buffer, compression='zstd')
        buffer.seek(0)
        
        self.s3_client.put_object(
            Bucket=self.bucket,
            Key=self.s3_log_key,
            Body=buffer.getvalue(),
            ContentType='application/octet-stream'
        )
        
        logger.info(f"Log updated in S3: {len(df_combined)} total rows")
        
        # Optionally update local cache
        if self.local_log_file:
            self.local_log_file.parent.mkdir(parents=True, exist_ok=True)
            df_combined.write_parquet(self.local_log_file)
            logger.debug(f"Local cache updated: {self.local_log_file}")
        
        return self.s3_log_uri


    def sync_to_local(self, local_dir: Optional[Path] = None) -> Tuple[int, int]:
        """
        Download single Parquet file from S3 to local.
        
        Much simpler than v1.3 - just one file to sync!
        
        Args:
            local_dir: Target directory (defaults to exports/)
        
        Returns:
            (1 if downloaded, 0 if skipped)
        
        Example:
            >>> logger = QueryLogger()
            >>> downloaded, skipped = logger.sync_to_local()
            >>> print(f"Downloaded {downloaded} file(s)")
        """
        if local_dir is None:
            if self.local_sync_dir is None:
                raise RuntimeError(
                    "No local_sync_dir configured. "
                    "This might be a Lambda environment with no model_root."
                )
            local_dir = self.local_sync_dir
        
        local_file = local_dir / "logs" / "query_logs.parquet"
        local_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Get S3 file metadata
            s3_metadata = self.s3_client.head_object(
                Bucket=self.bucket,
                Key=self.s3_log_key
            )
            s3_modified = s3_metadata['LastModified']
            
            # Check if local file is up-to-date
            if local_file.exists():
                local_modified = datetime.fromtimestamp(
                    local_file.stat().st_mtime,
                    tz=s3_modified.tzinfo
                )
                
                if local_modified >= s3_modified:
                    logger.info("✓ Local file is up-to-date, skipping download")
                    return (0, 1)  # skipped
            
            # Download file
            self.s3_client.download_file(
                self.bucket,
                self.s3_log_key,
                str(local_file)
            )
            
            logger.info(f"✓ Downloaded query_logs.parquet to {local_file}")
            return (1, 0)  # downloaded
            
        except Exception as e:
            if 'Not Found' in str(e) or '404' in str(e):
                logger.warning("No log file in S3 yet")
                return (0, 0)
            else:
                raise


    def get_recent_logs(self, n: int = 10, auto_sync: bool = True) -> pl.DataFrame:
        """
        Get recent query logs from single Parquet file.
        
        Much simpler than v1.3 - just read one file!
        
        Args:
            n: Number of recent queries to retrieve
            auto_sync: Whether to sync from S3 before reading
        
        Returns:
            Polars DataFrame with recent logs (sorted by timestamp desc)
        """
        if self.local_sync_dir is None:
            raise RuntimeError("No local_sync_dir configured (Lambda environment?)")
        
        # Sync from S3 if requested
        if auto_sync:
            self.sync_to_local()
        
        # Read local Parquet file
        local_file = self.local_sync_dir / "logs" / "query_logs.parquet"
        
        if not local_file.exists():
            logger.warning("No local log file found")
            return self._empty_log_dataframe()
        
        # Read entire file, then take most recent n
        df = pl.read_parquet(local_file)
        
        if len(df) == 0:
            return df
        
        # Sort by timestamp descending, take top n
        df_recent = df.sort("timestamp", descending=True).head(n)
        
        return df_recent



    def sync_exports_to_local(
        self, 
        local_exports_dir: Optional[Path] = None,
        max_files: int = 100
    ) -> Tuple[int, int]:
        """
        Download context and response exports from S3 to local directory.
        
        Syncs files from:
            s3://sentence-data-ingestion/DATA_MERGE_ASSETS/LOGS/FINRAG/contexts/ → local/exports/contexts/
            s3://sentence-data-ingestion/DATA_MERGE_ASSETS/LOGS/FINRAG/responses/ → local/exports/responses/
        
        Skips files that are already up-to-date locally (timestamp check).
        
        Args:
            local_exports_dir: Target directory (defaults to exports/ next to logs/)
            max_files: Maximum files to download per folder (prevents runaway downloads)
        
        Returns:
            Tuple of (downloaded_count, skipped_count)
        
        Example:
            >>> logger = QueryLogger()
            >>> downloaded, skipped = logger.sync_exports_to_local()
            >>> print(f"✓ Downloaded {downloaded} files, {skipped} up-to-date")
        """
        # Determine local export directory
        if local_exports_dir is None:
            if self.local_sync_dir is None:
                raise RuntimeError(
                    "No local_sync_dir configured. "
                    "Cannot determine export directory."
                )
            # Place exports/ next to logs/ directory
            local_exports_dir = self.local_sync_dir / "exports"
        
        # Create subdirectories
        contexts_dir = local_exports_dir / "contexts"
        responses_dir = local_exports_dir / "responses"
        contexts_dir.mkdir(parents=True, exist_ok=True)
        responses_dir.mkdir(parents=True, exist_ok=True)
        
        downloaded = 0
        skipped = 0
        
        # FIXED: Use correct S3 paths matching your bucket structure
        # Sync contexts folder
        logger.info("Syncing contexts from S3...")
        d, s = self._sync_s3_folder(
            s3_prefix="DATA_MERGE_ASSETS/LOGS/FINRAG/contexts/",  # ← FIXED
            local_dir=contexts_dir,
            max_files=max_files
        )
        downloaded += d
        skipped += s
        
        # Sync responses folder
        logger.info("Syncing responses from S3...")
        d, s = self._sync_s3_folder(
            s3_prefix="DATA_MERGE_ASSETS/LOGS/FINRAG/responses/",  # ← FIXED
            local_dir=responses_dir,
            max_files=max_files
        )
        downloaded += d
        skipped += s
        
        logger.info(f"✓ Export sync complete: {downloaded} downloaded, {skipped} skipped")
        return (downloaded, skipped)


    def _sync_s3_folder(
        self,
        s3_prefix: str,
        local_dir: Path,
        max_files: int
    ) -> Tuple[int, int]:
        """
        Helper: Download files from S3 prefix to local directory.
        
        Args:
            s3_prefix: S3 key prefix (e.g., "DATA_MERGE_ASSETS/LOGS/FINRAG/contexts/")
            local_dir: Local target directory
            max_files: Max files to download (safety limit)
        
        Returns:
            (downloaded_count, skipped_count)
        """
        downloaded = 0
        skipped = 0
        
        try:
            # List objects in S3 folder
            logger.debug(f"  Listing S3: s3://{self.bucket}/{s3_prefix}")
            
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket,
                Prefix=s3_prefix,
                MaxKeys=max_files
            )
            
            if 'Contents' not in response:
                logger.info(f"  No files found in s3://{self.bucket}/{s3_prefix}")
                return (0, 0)
            
            file_count = len([obj for obj in response['Contents'] if not obj['Key'].endswith('/')])
            logger.info(f"  Found {file_count} file(s) in S3")
            
            # Download each file
            for obj in response['Contents']:
                s3_key = obj['Key']
                
                # Skip folder markers (keys ending in /)
                if s3_key.endswith('/'):
                    continue
                
                # Extract filename (last part of key)
                filename = Path(s3_key).name
                local_file = local_dir / filename
                
                # Check if local file is up-to-date
                s3_modified = obj['LastModified']
                
                if local_file.exists():
                    local_modified = datetime.fromtimestamp(
                        local_file.stat().st_mtime,
                        tz=s3_modified.tzinfo
                    )
                    
                    if local_modified >= s3_modified:
                        skipped += 1
                        continue  # Skip this file
                
                # Download file
                self.s3_client.download_file(
                    self.bucket,
                    s3_key,
                    str(local_file)
                )
                
                downloaded += 1
                logger.debug(f"  ✓ Downloaded {filename}")
            
            logger.info(f"  Downloaded: {downloaded}, Skipped: {skipped}")
            return (downloaded, skipped)
            
        except Exception as e:
            logger.error(f"Error syncing folder {s3_prefix}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return (downloaded, skipped)


    ## =================================================================================
    ############### Cost Sum, Empty log df, Empty Cost Sum #############################


    def get_cost_summary(self, auto_sync: bool = True) -> Dict:
        """
        Calculate cost summary from single Parquet file.
        
        Much simpler than v1.3 - just read one file!
        
        Args:
            auto_sync: Whether to sync from S3 before calculating
        
        Returns:
            Dictionary with cost statistics
        """
        if self.local_sync_dir is None:
            raise RuntimeError("No local_sync_dir configured (Lambda environment?)")
        
        # Sync from S3 if requested
        if auto_sync:
            self.sync_to_local()
        
        # Read local Parquet file
        local_file = self.local_sync_dir / "logs" / "query_logs.parquet"
        
        if not local_file.exists():
            logger.warning("No local log file found")
            return self._empty_cost_summary()
        
        # Read entire file
        df = pl.read_parquet(local_file)
        
        if len(df) == 0:
            return self._empty_cost_summary()
        
        total_queries = len(df)
        
        # Filter successful vs failed
        successful = df.filter(pl.col('error').is_null())
        failed = df.filter(pl.col('error').is_not_null())
        
        # Safe aggregation
        if len(successful) > 0:
            total_cost = successful['cost'].sum()
            total_tokens = successful['total_tokens'].sum()
            avg_cost = total_cost / len(successful)
        else:
            total_cost = 0.0
            total_tokens = 0
            avg_cost = 0.0
        
        return {
            'total_queries': total_queries,
            'successful_queries': len(successful),
            'failed_queries': len(failed),
            'total_cost': float(total_cost) if total_cost is not None else 0.0,
            'total_tokens': int(total_tokens) if total_tokens is not None else 0,
            'avg_cost_per_query': float(avg_cost) if avg_cost is not None else 0.0
        }


    def _empty_log_dataframe(self) -> pl.DataFrame:
        """Return empty DataFrame with query log schema"""
        return pl.DataFrame(schema={
            'timestamp': pl.Utf8,
            'query': pl.Utf8,
            'model_id': pl.Utf8,
            'input_tokens': pl.Int64,
            'output_tokens': pl.Int64,
            'total_tokens': pl.Int64,
            'cost': pl.Float64,
            'context_length': pl.Int64,
            'processing_time_ms': pl.Float64,
            'error': pl.Utf8,
            'error_type': pl.Utf8,
            'stage': pl.Utf8,
            'context_file': pl.Utf8,
            'response_file': pl.Utf8
        })
    
    
    def _empty_cost_summary(self) -> Dict:
        """Return empty cost summary"""
        return {
            'total_queries': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'total_cost': 0.0,
            'total_tokens': 0,
            'avg_cost_per_query': 0.0
        }