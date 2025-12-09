"""
S3 Vectors Bulk Insertion Pipeline
===================================

Filtered batch insertion of Stage 3 embeddings into AWS S3 Vectors index
with intelligent retry logic, payload guards, and preflight validation.

Features:
- Parameterized filtering by CIK and year for incremental insertions
- Exponential backoff retry with jitter for throttling/capacity errors
- Automatic payload size guard (20 MiB AWS limit)
- Preflight index validation (dimension, metric, metadata config)
- Progress tracking with tqdm and comprehensive summary reporting

Usage:
    inserter = S3VectorsBulkInserter(
        provider="cohere_1024d",
        cik_filter=[1318605, 789019],    # Tesla, Microsoft only
        year_filter=[2021, 2022, 2023],  # Recent years only
        batch_size=500,
        max_retries=7
    )
    
    summary = inserter.run()
    # Returns: {'total_inserted': 12500, 'success_rate': 100.0, ...}

Author: Joel Markapudi
Course: IE7374 MLOps - Northeastern University
Project: FinRAG - Financial Document Intelligence System
"""

# ============================================================================
# STANDARD IMPORTS + PATH SETUP
# ============================================================================

import sys
from pathlib import Path
import time
import random
import json
from typing import Optional, List, Dict, Any

import polars as pl
import numpy as np
import boto3
from botocore.exceptions import ClientError
from tqdm import tqdm


def _find_model_pipeline_root() -> Path:
    """
    Walk up from file location to find ModelPipeline directory.
    
    Standard pattern used across all FinRAG modules for consistent
    path resolution in local dev, Docker, Lambda, and cloud environments.
    """
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if parent.name == "ModelPipeline":
            return parent
    raise RuntimeError(
        f"Cannot find ModelPipeline root directory.\n"
        f"  Searched from: {current}\n"
        f"  Expected 'ModelPipeline' directory in path tree"
    )


# Add ModelPipeline to path for absolute imports
MODEL_PIPELINE_ROOT = _find_model_pipeline_root()
sys.path.insert(0, str(MODEL_PIPELINE_ROOT))

from finrag_ml_tg1.loaders.ml_config_loader import MLConfig


# ============================================================================
# CONSTANTS
# ============================================================================

MAX_PAYLOAD_SIZE = 20 * 1024 * 1024  # 20 MiB (AWS hard limit for PutVectors)
DEFAULT_BATCH_SIZE = 500              # AWS maximum vectors per request
DEFAULT_MAX_RETRIES = 7               # Exponential backoff attempts


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def validate_embedding(embedding: List[float], expected_dim: int) -> List[float]:
    """
    Convert embedding to float32 and validate dimensions.
    
    Args:
        embedding: Raw embedding list from Stage 3 parquet
        expected_dim: Expected dimension count (e.g., 1024)
    
    Returns:
        Float32 list ready for S3 Vectors API
    
    Raises:
        ValueError: If dimension mismatch detected
    """
    arr = np.asarray(embedding, dtype=np.float32)
    if arr.shape[0] != expected_dim:
        raise ValueError(
            f"Embedding dimension mismatch: got {arr.shape[0]}, expected {expected_dim}"
        )
    return arr.tolist()


def estimate_payload_size(vectors_batch: List[Dict]) -> int:
    """
    Estimate JSON payload size to stay under AWS 20 MiB limit.
    
    AWS limit: PutVectors request ≤ 20 MiB total
    Typical batch: 500 vectors × ~4KB each = ~2 MiB (safe)
    
    Risk scenario: Large non-filterable metadata text fields could
    push payload over limit, requiring dynamic batch size reduction.
    
    Args:
        vectors_batch: List of S3 Vectors formatted vector dicts
    
    Returns:
        Estimated payload size in bytes
    """
    sample_payload = {"vectors": vectors_batch}
    estimated_bytes = len(json.dumps(sample_payload, default=str))
    return estimated_bytes


# ============================================================================
# MAIN CLASS
# ============================================================================

class S3VectorsBulkInserter:
    """
    Self-contained S3 Vectors bulk insertion pipeline with filtering.
    
    Handles complete workflow from Stage 3 loading through batch insertion
    with intelligent retry logic, progress tracking, and summary reporting.
    
    Filtering Strategy:
        Loads Stage 3 data with lazy Polars, applies cik_int and report_year
        filters before materialization. This enables incremental insertions:
        
        - Full insertion: cik_filter=None, year_filter=None
        - New years only: year_filter=[2006, 2007, 2008, 2009, 2010, 2011]
        - Specific companies: cik_filter=[1318605, 789019]
        - Validation subset: Both filters for small test set
    
    Retry Strategy:
        Exponential backoff with jitter for AWS throttling (429) and capacity
        errors (503). Client errors (4xx) fail fast except throttling.
    
    Attributes:
        provider: Embedding provider (e.g., "cohere_1024d")
        cik_filter: List of CIK integers to insert (None = all companies)
        year_filter: List of report years to insert (None = all years)
        batch_size: Vectors per PutVectors request (AWS max = 500)
        max_retries: Exponential backoff retry attempts
    """
    
    def __init__(
        self,
        provider: str,
        cik_filter: Optional[List[int]] = None,
        year_filter: Optional[List[int]] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_retries: int = DEFAULT_MAX_RETRIES
    ):
        """
        Initialize S3 Vectors bulk insertion pipeline.
        
        Args:
            provider: Embedding provider identifier matching Stage 3 cache
                     (e.g., "cohere_1024d", "titan_1024d")
            cik_filter: Optional list of CIK integers to insert. None = all.
                       Example: [1318605, 789019] for Tesla + Microsoft only
            year_filter: Optional list of report years to insert. None = all.
                        Example: [2021, 2022, 2023] for recent years only
            batch_size: Vectors per batch (AWS max = 500, don't exceed)
            max_retries: Exponential backoff retry attempts for throttling
        
        Raises:
            RuntimeError: If MLConfig initialization fails
            FileNotFoundError: If Stage 3 cache doesn't exist
        """
        self.provider = provider
        self.cik_filter = cik_filter
        self.year_filter = year_filter
        self.batch_size = min(batch_size, 500)  # AWS hard limit
        self.max_retries = max_retries
        
        # Load configuration
        self.config = MLConfig()
        
        # Extract S3 Vectors configuration from MLConfig
        self.vector_bucket = "finrag-embeddings-s3vectors"
        self.index_name = "finrag-sentence-fact-embed-1024d"
        self.dimensions = self.config.s3vectors_dimensions(provider)
        self.region = self.config.region
        
        # Initialize S3 Vectors client
        self.s3vectors_client = boto3.client(
            "s3vectors",
            region_name=self.region,
            aws_access_key_id=self.config.aws_access_key,
            aws_secret_access_key=self.config.aws_secret_key
        )
        
        # Tracking variables (set during run)
        self.start_time = None
        self.total_inserted = 0
        self.failed_batches = 0
        self.shrunk_batches = 0
    
    
    def _validate_index_configuration(self) -> tuple[str, set]:
        """
        Preflight check: Validate S3 Vectors index configuration.
        
        Validates:
        1. Dimension matches Stage 3 embeddings (e.g., 1024d)
        2. Distance metric appropriate for semantic search
        3. Non-filterable metadata keys declared correctly
        
        Returns:
            Tuple of (distance_metric, non_filterable_keys_set)
        
        Raises:
            RuntimeError: If index doesn't exist or API call fails
            ValueError: If dimension mismatch between index and embeddings
        """
        print(f"\n[Preflight Check - Index Validation]")
        
        try:
            response = self.s3vectors_client.get_index(
                vectorBucketName=self.vector_bucket,
                indexName=self.index_name
            )
        except ClientError as e:
            raise RuntimeError(f"Failed to get index: {e}")
        
        # Extract configuration (S3 Vectors uses flat structure, lowercase keys)
        index_data = response['index']
        actual_dim = index_data['dimension']
        distance_metric = index_data['distanceMetric']
        data_type = index_data['dataType']
        
        # Get non-filterable metadata keys
        meta_cfg = index_data.get('metadataConfiguration', {})
        nonfilterable_keys = set(meta_cfg.get('nonFilterableMetadataKeys', []))
        
        print(f"  Index: {self.index_name}")
        print(f"  ARN: {index_data['indexArn']}")
        print(f"  Created: {index_data['creationTime']}")
        print(f"  Dimension: {actual_dim}d")
        print(f"  Data Type: {data_type}")
        print(f"  Distance Metric: {distance_metric}")
        print(f"  Non-filterable Keys: {nonfilterable_keys or 'None'}")
        
        # Validate dimension (critical check)
        if actual_dim != self.dimensions:
            raise ValueError(
                f"Dimension mismatch!\n"
                f"  Index configured: {actual_dim}d\n"
                f"  Stage 3 embeddings: {self.dimensions}d\n"
                f"  → Cannot insert mismatched dimensions"
            )
        
        # Informational checks
        if data_type != 'float32':
            print(f"  ⚠️  Warning: Index expects '{data_type}', sending float32")
        
        if distance_metric not in ['cosine', 'euclidean', 'dotProduct']:
            print(f"  ⚠️  Warning: Unusual distance metric '{distance_metric}'")
        
        # Validate non-filterable metadata alignment
        expected_nonfilterable = {'sentenceID', 'embedding_id', 'section_sentence_count'}
        
        if nonfilterable_keys:
            if nonfilterable_keys == expected_nonfilterable:
                print(f"  ✓ Non-filterable keys match expected configuration")
            else:
                missing = expected_nonfilterable - nonfilterable_keys
                extra = nonfilterable_keys - expected_nonfilterable
                if missing:
                    print(f"  Note: Expected non-filterable keys missing: {missing}")
                if extra:
                    print(f"  Note: Additional non-filterable keys: {extra}")
        else:
            print(f"  Note: No non-filterable keys configured")
            print(f"     All metadata will be filterable (may increase costs)")
        
        print(f"  ✓ Preflight validation passed")
        
        return distance_metric, nonfilterable_keys
    
    
    def _load_and_filter_stage3(self) -> pl.DataFrame:
        """
        Load Stage 3 data with lazy Polars and apply filters.
        
        Strategy:
        1. Scan parquet lazily (builds query plan, no memory load)
        2. Apply cik_int and report_year filters (if specified)
        3. Materialize only filtered subset (collect)
        
        This approach enables memory-efficient incremental insertions
        where only the delta (e.g., new years) is materialized.
        
        Returns:
            Filtered Stage 3 DataFrame ready for insertion
        
        Raises:
            FileNotFoundError: If Stage 3 cache doesn't exist
            ValueError: If required columns missing from Stage 3
        """
        cache_path = self.config.get_s3vectors_cache_path(self.provider)
        
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Stage 3 data not found: {cache_path}\n"
                f"Run Notebook 03 (S3Vector_TableProvisioning) first to build Stage 3"
            )
        
        print(f"\n[Loading Stage 3 Data]")
        print(f"  Source: {cache_path.name}")
        
        # Lazy load Stage 3 (no memory consumption yet)
        df_stage3_lazy = pl.scan_parquet(cache_path)
        
        # Apply filters if specified
        if self.cik_filter is not None:
            print(f"  Filter: CIK in {len(self.cik_filter)} companies")
            df_stage3_lazy = df_stage3_lazy.filter(
                pl.col('cik_int').is_in(self.cik_filter)
            )
        else:
            print(f"  Filter: ALL companies")
        
        if self.year_filter is not None:
            print(f"  Filter: Years in {self.year_filter}")
            df_stage3_lazy = df_stage3_lazy.filter(
                pl.col('report_year').is_in(self.year_filter)
            )
        else:
            print(f"  Filter: ALL years")
        
        # Materialize filtered data (this is where memory is allocated)
        df_stage3 = df_stage3_lazy.collect()
        
        total_rows = len(df_stage3)
        print(f"  ✓ Filtered Stage 3: {total_rows:,} vectors")
        
        if total_rows == 0:
            print(f"\n  ⚠️  No vectors match filter criteria!")
            print(f"     Check that your cik_filter and year_filter values")
            print(f"     actually exist in Stage 3 (cik_int, report_year columns)")
            return df_stage3
        
        # Validate schema (critical columns for S3 Vectors)
        required_cols = [
            'sentenceID_numsurrogate',  # Key for S3 Vectors
            'embedding',                 # 1024-d vector
            'cik_int',                   # Filterable metadata
            'report_year',               # Filterable metadata
            'section_name',              # Filterable metadata
            'sic',                       # Filterable metadata
            'sentence_pos',              # Filterable metadata
            'sentenceID',                # Non-filterable metadata
            'embedding_id',              # Non-filterable metadata
            'section_sentence_count'     # Non-filterable metadata
        ]
        
        missing = [c for c in required_cols if c not in df_stage3.columns]
        if missing:
            raise ValueError(f"Missing required columns in Stage 3: {missing}")
        
        return df_stage3
    
    
    def _convert_batch_to_s3vectors_format(
        self, 
        batch_df: pl.DataFrame
    ) -> List[Dict[str, Any]]:
        """
        Convert Polars DataFrame batch to S3 Vectors PutVectors format.
        
        Transforms Stage 3 rows into AWS S3 Vectors API structure:
        {
            "key": "12345678",  # sentenceID_numsurrogate
            "data": {"float32": [0.1, 0.2, ..., 0.9]},  # 1024-d vector
            "metadata": {
                "cik_int": 1318605,           # Filterable
                "report_year": 2021,          # Filterable
                "section_name": "MD&A",       # Filterable
                "sic": "3711",                # Filterable
                "sentence_pos": 42,           # Filterable
                "sentenceID": "1318605_...",  # Non-filterable
                "embedding_id": "bedrock_...", # Non-filterable
                "section_sentence_count": 156  # Non-filterable
            }
        }
        
        Args:
            batch_df: Stage 3 DataFrame subset (≤500 rows)
        
        Returns:
            List of S3 Vectors formatted vector dicts
        
        Raises:
            ValueError: If embedding dimension validation fails
        """
        vectors = []
        
        for row in batch_df.iter_rows(named=True):
            # Validate and convert embedding to float32
            embedding_float32 = validate_embedding(row['embedding'], self.dimensions)
            
            vector_item = {
                "key": str(row['sentenceID_numsurrogate']),
                "data": {"float32": embedding_float32},
                "metadata": {
                    # Filterable metadata (5 fields)
                    # Used in query filters: cik_int=1318605, report_year=2021
                    "cik_int": int(row['cik_int']),
                    "report_year": int(row['report_year']),
                    "section_name": str(row['section_name']),
                    "sic": str(row['sic']),
                    "sentence_pos": int(row['sentence_pos']),
                    
                    # Non-filterable metadata (3 fields)
                    # Cannot be used in filters, stored for reference only
                    "sentenceID": str(row['sentenceID']),
                    "embedding_id": str(row['embedding_id']),
                    "section_sentence_count": int(row['section_sentence_count'])
                }
            }
            
            vectors.append(vector_item)
        
        return vectors
    
    
    def _shrink_batch_if_needed(
        self, 
        vectors_batch: List[Dict],
        max_size: int = MAX_PAYLOAD_SIZE
    ) -> tuple[List[Dict], bool]:
        """
        Dynamically reduce batch size if payload exceeds AWS 20 MiB limit.
        
        AWS enforces hard 20 MiB limit on PutVectors requests. Typical batch
        of 500 vectors × 4KB = ~2 MiB is safe. This guard handles edge cases
        where large non-filterable metadata text fields push payload over limit.
        
        Strategy: Binary search to find acceptable batch size
        
        Args:
            vectors_batch: S3 Vectors formatted vector list
            max_size: Maximum payload size in bytes (default: 20 MiB)
        
        Returns:
            Tuple of (potentially_smaller_batch, was_shrunk)
        """
        size = estimate_payload_size(vectors_batch)
        
        if size <= max_size:
            return vectors_batch, False
        
        # Payload too large, shrink batch
        print(f"  ⚠️  Payload too large ({size/1024/1024:.1f} MB), shrinking batch...")
        
        while size > max_size and len(vectors_batch) > 1:
            vectors_batch = vectors_batch[:len(vectors_batch) // 2]
            size = estimate_payload_size(vectors_batch)
        
        print(f"     → Reduced to {len(vectors_batch)} vectors ({size/1024/1024:.1f} MB)")
        return vectors_batch, True
    
    
    def _put_vectors_with_retry(
        self, 
        vectors_batch: List[Dict]
    ) -> tuple[int, bool]:
        """
        Insert vectors with exponential backoff retry for AWS throttling.
        
        Handles AWS S3 Vectors transient errors:
        - 429 TooManyRequestsException (throttling) → Retry with backoff
        - 503 ServiceUnavailableException (capacity) → Retry with backoff
        - 5xx server errors → Retry with backoff
        - 4xx client errors (except 429) → Fail fast (bad request)
        
        Backoff strategy:
        - Initial delay: 0.5s + random jitter (±0.25s)
        - Exponential doubling: 0.5s → 1s → 2s → 4s (capped)
        - Jitter prevents thundering herd on retry
        
        Args:
            vectors_batch: S3 Vectors formatted vector list
        
        Returns:
            Tuple of (vectors_inserted_count, was_batch_shrunk)
        
        Raises:
            ClientError: If non-retryable client error or max retries exceeded
            Exception: If unexpected error occurs
        """
        # Guard against oversized payloads
        vectors_batch, was_shrunk = self._shrink_batch_if_needed(vectors_batch)
        
        attempt = 0
        delay = 0.5  # Start with 500ms
        
        while attempt < self.max_retries:
            try:
                # Attempt insertion via AWS S3 Vectors API
                response = self.s3vectors_client.put_vectors(
                    vectorBucketName=self.vector_bucket,
                    indexName=self.index_name,
                    vectors=vectors_batch
                )
                
                # Success! Note: S3 Vectors upserts by key, so duplicate
                # insertions are idempotent (overwrites existing vector)
                return len(vectors_batch), was_shrunk
            
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', 'Unknown')
                http_status = e.response.get('ResponseMetadata', {}).get('HTTPStatusCode', 500)
                
                # Classify error type for retry decision
                is_throttle = error_code == 'TooManyRequestsException'
                is_capacity = error_code == 'ServiceUnavailableException'
                is_client_error = 400 <= http_status < 500
                is_server_error = http_status >= 500
                
                # Client errors (4xx) except throttle → Fail fast, don't retry
                if is_client_error and not is_throttle:
                    print(f"\n  ❌ Client error (non-retryable): {error_code}")
                    print(f"     Message: {e.response.get('Error', {}).get('Message', 'No details')}")
                    raise  # Don't waste retries on bad requests
                
                # Retryable errors: throttle, capacity, server errors
                if is_throttle or is_capacity or is_server_error:
                    attempt += 1
                    
                    if attempt >= self.max_retries:
                        print(f"\n  ❌ Max retries ({self.max_retries}) exceeded")
                        raise
                    
                    # Calculate backoff with jitter (prevents thundering herd)
                    jitter = random.random() * 0.25  # ±0.25s randomization
                    sleep_time = min(delay + jitter, 4.0)  # Cap at 4s
                    
                    retry_reason = (
                        "throttled" if is_throttle else 
                        ("capacity" if is_capacity else "server error")
                    )
                    print(f"  ⏳ Retry {attempt}/{self.max_retries} ({retry_reason}), "
                          f"waiting {sleep_time:.2f}s...")
                    
                    time.sleep(sleep_time)
                    delay *= 2  # Exponential backoff
                    continue
                
                # Unknown error type
                print(f"\n  ❌ Unknown error: {error_code}")
                raise
            
            except Exception as e:
                # Unexpected error (not AWS ClientError)
                print(f"\n  ❌ Unexpected error: {type(e).__name__}: {e}")
                raise
        
        # Should never reach here (loop exits via return or raise)
        raise RuntimeError("Retry logic error - exhausted retries without return")
    
    
    def run(self) -> Dict[str, Any]:
        """
        Execute complete filtered insertion pipeline.
        
        Workflow:
        1. Validate S3 Vectors index configuration (preflight)
        2. Load Stage 3 data with lazy Polars
        3. Apply cik_int + report_year filters (if specified)
        4. Batch convert to S3 Vectors format
        5. Insert with exponential backoff retry
        6. Track progress with tqdm
        7. Return comprehensive summary dict
        
        Returns:
            Summary dictionary with keys:
            {
                'provider': 'cohere_1024d',
                'cik_filter': [1318605, 789019] or None,
                'year_filter': [2021, 2022, 2023] or None,
                'total_in_stage3': 407048,
                'filtered_count': 12500,
                'total_inserted': 12500,
                'failed_batches': 0,
                'shrunk_batches': 0,
                'success_rate': 100.0,
                'duration_seconds': 324.7,
                'batches_processed': 25
            }
        
        Raises:
            RuntimeError: If index validation fails
            FileNotFoundError: If Stage 3 cache doesn't exist
            Exception: If insertion fails after max retries
        """
        self.start_time = time.time()
        
        print("=" * 70)
        print("S3 VECTORS BULK INSERTION PIPELINE")
        print("=" * 70)
        print(f"Provider: {self.provider}")
        print(f"Vector Bucket: {self.vector_bucket}")
        print(f"Index Name: {self.index_name}")
        print(f"Batch Size: {self.batch_size} vectors/request")
        
        # ====================================================================
        # STEP 1: PREFLIGHT VALIDATION
        # ====================================================================
        
        distance_metric, nonfilterable_keys = self._validate_index_configuration()
        
        # ====================================================================
        # STEP 2: LOAD AND FILTER STAGE 3
        # ====================================================================
        
        df_stage3 = self._load_and_filter_stage3()
        total_rows = len(df_stage3)
        
        if total_rows == 0:
            print(f"\n{'=' * 70}")
            print(f"⚠️  NO VECTORS TO INSERT (filter returned 0 rows)")
            print(f"{'=' * 70}")
            
            return {
                'provider': self.provider,
                'cik_filter': self.cik_filter,
                'year_filter': self.year_filter,
                'total_in_stage3': 0,
                'filtered_count': 0,
                'total_inserted': 0,
                'failed_batches': 0,
                'shrunk_batches': 0,
                'success_rate': 0.0,
                'duration_seconds': time.time() - self.start_time,
                'batches_processed': 0
            }
        
        # ====================================================================
        # STEP 3: BATCH INSERTION WITH RETRY
        # ====================================================================
        
        print(f"\n[Batch Insertion with Retry Logic]")
        num_batches = (total_rows + self.batch_size - 1) // self.batch_size
        print(f"  Total batches: {num_batches}")
        print(f"  Retry strategy: Exponential backoff (max {self.max_retries} attempts)")
        
        self.total_inserted = 0
        self.failed_batches = 0
        self.shrunk_batches = 0
        batch_num = 0
        
        # Progress tracking with tqdm
        with tqdm(total=total_rows, desc="Inserting", unit="vectors") as pbar:
            for i in range(0, total_rows, self.batch_size):
                batch_num += 1
                
                # Extract batch from DataFrame
                batch_df = df_stage3[i:i + self.batch_size]
                
                # Convert to S3 Vectors format
                vectors_batch = self._convert_batch_to_s3vectors_format(batch_df)
                
                try:
                    # Insert with retry logic
                    inserted, was_shrunk = self._put_vectors_with_retry(vectors_batch)
                    
                    self.total_inserted += inserted
                    if was_shrunk:
                        self.shrunk_batches += 1
                    
                    pbar.update(inserted)
                    
                except Exception as e:
                    print(f"\n  ❌ Batch {batch_num} failed permanently: {e}")
                    self.failed_batches += 1
                    # Continue to next batch (don't halt entire pipeline)
                    continue
        
        # ====================================================================
        # STEP 4: SUMMARY AND RETURN
        # ====================================================================
        
        duration = time.time() - self.start_time
        success_rate = (self.total_inserted / total_rows * 100) if total_rows > 0 else 0.0
        
        print(f"\n{'=' * 70}")
        print(f"✓ INSERTION COMPLETE")
        print(f"{'=' * 70}")
        print(f"  Vectors inserted: {self.total_inserted:,} / {total_rows:,}")
        print(f"  Success rate: {success_rate:.1f}%")
        print(f"  Duration: {duration:.1f}s")
        
        if self.failed_batches > 0:
            print(f"  ⚠️  Failed batches: {self.failed_batches}/{num_batches}")
        
        if self.shrunk_batches > 0:
            print(f"  Note: {self.shrunk_batches} batches auto-shrunk (payload > 20MB)")
        
        print(f"\n[Index Ready for Queries]")
        print(f"  Query example:")
        print(f"  response = s3vectors.query_vectors(")
        print(f"      vectorBucketName='{self.vector_bucket}',")
        print(f"      indexName='{self.index_name}',")
        print(f"      queryVector=[...{self.dimensions}d embedding...],")
        print(f"      topK=10,")
        print(f"      filter={{'cik_int': 1318605, 'report_year': 2021}})")
        print("=" * 70)
        
        # Return comprehensive summary for programmatic use
        return {
            'provider': self.provider,
            'cik_filter': self.cik_filter,
            'year_filter': self.year_filter,
            'total_in_stage3': total_rows,
            'filtered_count': total_rows,  # Same after filtering
            'total_inserted': self.total_inserted,
            'failed_batches': self.failed_batches,
            'shrunk_batches': self.shrunk_batches,
            'success_rate': success_rate,
            'duration_seconds': duration,
            'batches_processed': batch_num
        }


# ============================================================================
# EXAMPLE USAGE (for testing)
# ============================================================================

if __name__ == "__main__":
    """
    Example usage patterns for testing.
    Run this file directly to validate class functionality.
    """
    
    # Example 1: Full insertion (all companies, all years)
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Full Insertion")
    print("=" * 70)
    
    inserter_full = S3VectorsBulkInserter(
        provider="cohere_1024d",
        cik_filter=None,  # All companies
        year_filter=None   # All years
    )
    
    # Uncomment to run:
    # summary_full = inserter_full.run()
    # print(f"\nSummary: {summary_full}")
    
    
    # Example 2: Incremental - New years only
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Incremental Insertion (New Years)")
    print("=" * 70)
    
    inserter_incremental = S3VectorsBulkInserter(
        provider="cohere_1024d",
        cik_filter=None,
        year_filter=[2021, 2022, 2023, 2024, 2025, 2012, 2013, 2014]
    )
    
    # Uncomment to run:
    # summary_incremental = inserter_incremental.run()
    # print(f"\nSummary: {summary_incremental}")
    
    
    # Example 3: Validation subset (1 company, 1 year)
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Validation Subset")
    print("=" * 70)
    
    inserter_validation = S3VectorsBulkInserter(
        provider="cohere_1024d",
        cik_filter=[1318605],  # Tesla only
        year_filter=[2021]      # 2021 only
    )
    
    # Uncomment to run:
    # summary_validation = inserter_validation.run()
    # print(f"\nSummary: {summary_validation}")
    
    
    print("\n✓ Example usage patterns defined")
    print("  Uncomment 'summary = inserter.run()' lines to execute")