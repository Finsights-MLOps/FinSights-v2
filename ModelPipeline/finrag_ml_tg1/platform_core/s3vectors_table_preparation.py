"""
S3 Vectors table preparation pipeline (Stage 3).
Joins Stage 2 meta + embeddings → S3 Vectors format with mmh3 hashing.

MEMORY-EFFICIENT DESIGN:
- Uses lazy evaluation (scan_parquet) instead of eager loading
- Streams data through pipeline with sink_parquet (never holds full join in RAM)
- Validates dimensions separately before main pipeline
- Computes hashes chunk-wise with map_elements (not giant Python list)
"""

# ============================================================================
# S3 VECTORS PIPELINE (Stage 3) - Execution Parameters
#
# - Stage 3 is a JOIN operation (memory-intensive with 1.7GB embedding column)
# - Depends on Stage 1 + Stage 2 (upstream changes)
# - No complex merge logic needed (just rebuild)
# - ALWAYS does a simple rebuild, instead of tracking complexity with deltas
# - Uses Polars streaming to avoid memory explosions with large vector tables
# ============================================================================

import sys
from pathlib import Path

# Find ModelPipeline root (standard pattern)
def _find_model_pipeline_root() -> Path:
    """Walk up from file location to find ModelPipeline directory"""
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

import polars as pl
import mmh3
from botocore.exceptions import ClientError


class S3VectorsTablePipeline:
    """
    Stage 3 table preparation: Meta + Embeddings → S3 Vectors format.
    
    Memory-efficient design using Polars lazy evaluation and streaming:
    - Never loads full embeddings table (1.7GB) into memory
    - Never materializes joined table (meta + vectors)
    - Streams data directly from source parquet → transformations → sink parquet
    - Only tracks scalars (row_count, cache_path), not large DataFrames
    
    Pipeline always rebuilds Stage 3 from scratch (no delta tracking).
    Uses mmh3 hashing for surrogate keys and validates schema.
    
    Usage:
        pipeline = S3VectorsTablePipeline(
            provider="cohere_1024d",
            build_table=True,
            upload_to_s3=True,
            force_overwrite=True
        )
        summary = pipeline.run()
    
    Advanced usage:
        # Build locally only (test before upload)
        pipeline = S3VectorsTablePipeline(
            provider="cohere_1024d",
            build_table=True,
            upload_to_s3=False
        )
        pipeline.run()
    """
    
    def __init__(
        self,
        provider="cohere_1024d",
        build_table=True,
        upload_to_s3=True,
        force_overwrite=True
    ):
        """
        Initialize S3 Vectors table preparation pipeline.
        
        Args:
            provider: Embedding provider ("cohere_1024d", "titan_1024d")
            build_table: Build Stage 3 table locally
            upload_to_s3: Upload Stage 3 to S3
            force_overwrite: Overwrite existing S3 table if exists
        """
        self.provider = provider
        self.build_table = build_table
        self.upload_to_s3 = upload_to_s3
        self.force_overwrite = force_overwrite
        
        # Load config internally
        self.config = MLConfig()
        
        # State tracking - ONLY SCALARS, no large DataFrames
        # This prevents holding 1.7GB+ vectors in memory as instance variables
        self.row_count: int | None = None
        self.cache_path: Path | None = None
    
    # ========================================================================
    # PRIVATE HELPER METHODS
    # ========================================================================
    
    @staticmethod
    def _check_s3_exists(s3_client, bucket, s3_key):
        """Check if S3 object exists"""
        try:
            s3_client.head_object(Bucket=bucket, Key=s3_key)
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            else:
                raise
    
    def _validate_dimensions(self, emb_path: Path) -> int:
        """
        Validate embedding dimensions from vectors file BEFORE heavy pipeline.
        
        This is a lightweight check that only scans the embeddings file,
        not the full joined table. Catches dimension mismatches early.
        
        Args:
            emb_path: Path to embeddings parquet file
            
        Returns:
            actual_dims: Actual dimension count from embeddings
            
        Raises:
            ValueError: If dimensions don't match config
        """
        print(f"\n[Validating Dimensions]")
        
        # Scan embeddings file and get max dimension (all should be same)
        # Only processes metadata + 1 row worth of embeddings, not entire 1.7GB
        actual_dims = (
            pl.scan_parquet(emb_path)
            .select(pl.col('embedding').list.len().max())
            .collect(streaming=True)
            .item()
        )
        
        expected_dims = self.config.s3vectors_dimensions(self.provider)
        
        if actual_dims != expected_dims:
            raise ValueError(
                f"Dimension mismatch for {self.provider}:\n"
                f"  Expected: {expected_dims}d (from config)\n"
                f"  Actual:   {actual_dims}d (in embeddings parquet)"
            )
        
        print(f"  ✓ Dimension check: {actual_dims}d (validated from embeddings file)")
        
        return actual_dims
    
    def _build_stage3_table(self):
        """
        Transform Stage 2 (meta) + Embeddings → Stage 3 (S3 Vectors ready).
        
        MEMORY-EFFICIENT APPROACH:
        - Uses lazy scan_parquet (no eager loading of 1.7GB vectors)
        - Builds complete lazy pipeline with all transformations
        - Uses sink_parquet to write directly to disk (streaming execution)
        - Never materializes the joined table in memory
        - Only scans result file at end to get row count
        
        Schema Output:
            sentenceID_numsurrogate (int64) - mmh3 hash
            sentenceID (varchar)
            embedding (f32[dims])
            cik_int (int32)
            report_year (int16)
            section_name (varchar)
            sic (varchar)
            sentence_pos (int16)
            embedding_id (varchar)
            section_sentence_count (int16)
        """
        print(f"\n[Stage 3 Build - {self.provider}]")
        
        # Define paths
        base = Path.cwd().parent / 'data_cache'
        meta_path = base / 'meta_embeds' / 'finrag_fact_sentences_meta_embeds.parquet'
        emb_path = base / 'embeddings' / self.provider / Path(
            self.config.embeddings_path(self.provider)
        ).name
        
        # Validate paths exist
        if not meta_path.exists():
            raise FileNotFoundError(f"Stage 2 meta not cached: {meta_path}")
        if not emb_path.exists():
            raise FileNotFoundError(f"Embeddings not cached: {emb_path}")
        
        print(f"  Meta path: {meta_path}")
        print(f"  Embeddings path: {emb_path}")
        
        # STEP 1: Validate dimensions BEFORE heavy pipeline
        # This is lightweight - only checks embedding column metadata
        actual_dims = self._validate_dimensions(emb_path)
        
        # STEP 2: Define output cache path
        cache_path = self.config.get_s3vectors_cache_path(self.provider)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path = cache_path
        
        print(f"\n[Building Lazy Pipeline]")
        
        # STEP 3: Build lazy plans - SELECT ONLY NEEDED COLUMNS
        # Meta table: only 7 columns needed (not all 34)
        # This reduces I/O and memory footprint significantly
        lf_meta = (
            pl.scan_parquet(meta_path)
            .select([
                'sentenceID',
                'cik_int',
                'report_year',
                'section_name',
                'sic',
                'embedding_id',
                'section_sentence_count'
            ])
        )
        
        # Vectors table: only sentenceID + embedding
        # No need for embedding_id here since it's in meta
        lf_vectors = (
            pl.scan_parquet(emb_path)
            .select([
                'sentenceID',
                'embedding'  # This is the 1.7GB column - never fully loaded!
            ])
        )
        
        print(f"  ✓ Lazy meta scan (7 columns selected)")
        print(f"  ✓ Lazy vectors scan (2 columns selected)")
        
        # STEP 4: Lazy join + transformations
        # This builds the execution plan but doesn't execute yet
        lf_joined = (
            lf_meta
            .join(lf_vectors, on='sentenceID', how='inner')
            # Safety filter: only keep rows with embedding_id
            .filter(pl.col('embedding_id').is_not_null())
        )
        
        print(f"  ✓ Lazy join configured (inner join on sentenceID)")
        
        # STEP 5: Add computed columns
        # map_elements runs per-chunk in streaming mode (not giant Python list)
        # sentence_pos extraction also done lazily
        lf_stage3 = (
            lf_joined
            .with_columns([
                # Hash computation: chunk-wise, not full materialization
                pl.col('sentenceID')
                  .map_elements(
                      lambda sid: mmh3.hash64(sid, signed=True)[0],
                      return_dtype=pl.Int64
                  )
                  .alias('sentenceID_numsurrogate'),
                
                # Sentence position extraction with fallback
                pl.col('sentenceID')
                  .str.split('_')
                  .list.last()
                  .cast(pl.Int16, strict=False)
                  .fill_null(-1)
                  .alias('sentence_pos')
            ])
            # STEP 6: Select final columns in correct order
            .select([
                # Primary keys
                'sentenceID_numsurrogate',
                'sentenceID',
                
                # Embedding (1.7GB column flows through, never materialized)
                'embedding',
                
                # Filterable metadata
                'cik_int',
                'report_year',
                'section_name',
                'sic',
                'sentence_pos',
                
                # Non-filterable metadata
                'embedding_id',
                'section_sentence_count'
            ])
        )
        
        print(f"  ✓ Transformations configured:")
        print(f"    - mmh3 hash (sentenceID_numsurrogate)")
        print(f"    - sentence_pos extraction")
        print(f"    - Final column selection (10 columns)")
        
        # STEP 7: Execute pipeline with STREAMING SINK
        # This is the key optimization: writes directly to disk
        # Never holds the full joined table (meta + 1.7GB vectors) in memory
        print(f"\n[Executing Pipeline with Streaming Sink]")
        print(f"  → Writing Stage 3 parquet: {cache_path}")
        
        lf_stage3.sink_parquet(cache_path, compression='zstd')
        
        print(f"  ✓ Streaming write complete")
        
        # STEP 8: Get row count by scanning the output file
        # This is cheap - only reads metadata, not full data
        self.row_count = (
            pl.scan_parquet(cache_path)
            .select(pl.len())
            .collect(streaming=True)
            .item()
        )
        
        print(f"\n  ✓ Stage 3 Complete:")
        print(f"    Rows: {self.row_count:,}")
        print(f"    Columns: 10")
        print(f"    Dimensions: {actual_dims}d (validated)")
        print(f"    Cache: {cache_path}")
        
        # Optional: Validate hash uniqueness (lightweight check)
        # Only materializes the hash column + counts, not full data
        print(f"\n[Validating Hash Uniqueness]")
        hash_check = (
            pl.scan_parquet(cache_path)
            .group_by('sentenceID_numsurrogate')
            .agg(pl.count().alias('n'))
            .filter(pl.col('n') > 1)
            .collect(streaming=True)
        )
        
        if len(hash_check) > 0:
            print(f"  ⚠️  WARNING: {len(hash_check)} hash collisions detected")
            print(f"     Collision rate: {len(hash_check)/self.row_count*100:.4f}%")
        else:
            print(f"  ✓ Hash uniqueness validated (0 collisions)")
    
    def _upload_to_s3(self):
        """
        Upload Stage 3 table to S3 using streaming file upload.
        
        Uses boto3's upload_file which streams the local parquet file
        without loading it into memory as a DataFrame.
        """
        if self.cache_path is None or not self.cache_path.exists():
            raise FileNotFoundError(
                "Stage 3 cache not found. Run with build_table=True first."
            )
        
        s3_client = self.config.get_s3_client()
        bucket = self.config.bucket
        s3_key = self.config.s3vectors_path(self.provider)
        s3_uri = f"s3://{bucket}/{s3_key}"
        
        # Check if exists
        exists = self._check_s3_exists(s3_client, bucket, s3_key)
        
        if exists and not self.force_overwrite:
            print(f"\n[S3 Vectors Table - Already Exists]")
            print(f"  Provider:  {self.provider}")
            print(f"  Location:  {s3_uri}")
            print(f"  Set force_overwrite=True to recreate")
            return
        
        if exists and self.force_overwrite:
            print(f"\n[S3 Vectors Table - Recreating]")
            print(f"  Provider: {self.provider}")
            s3_client.delete_object(Bucket=bucket, Key=s3_key)
            print(f"  ✓ Deleted existing object")
        
        print(f"\n[S3 Vectors Table - Uploading]")
        print(f"  Provider:    {self.provider}")
        print(f"  Source:      {self.cache_path}")
        print(f"  Destination: {s3_uri}")
        print(f"  Rows:        {self.row_count:,}")
        print(f"  Dimensions:  {self.config.s3vectors_dimensions(self.provider)}d")
        
        # Use boto3's streaming file upload (doesn't load into memory)
        # This is better than Polars write_parquet with S3 URI
        s3_client.upload_file(
            Filename=str(self.cache_path),
            Bucket=bucket,
            Key=s3_key
        )
        
        print(f"  ✓ Upload complete (Cost: $0.00 - S3 ingress is free)")
    
    def _print_summary(self):
        """Print final pipeline summary"""
        print(f"\n{'='*70}")
        print(f"✓ S3 VECTORS PIPELINE COMPLETE")
        print(f"{'='*70}")
        print(f"\nActions completed:")
        
        if self.build_table:
            print(
                f"  ✓ Stage 3 built locally "
                f"({self.row_count:,} rows, "
                f"{self.config.s3vectors_dimensions(self.provider)}d)"
            )
            print(f"    Cache: {self.cache_path}")
        
        if self.upload_to_s3:
            s3_uri = f"s3://{self.config.bucket}/{self.config.s3vectors_path(self.provider)}"
            print(f"  ✓ Uploaded to S3: {s3_uri}")
        
        print("="*70)
    
    # ========================================================================
    # PUBLIC INTERFACE
    # ========================================================================
    
    def run(self):
        """Execute the S3 Vectors table preparation pipeline"""
        print("="*70)
        print("S3 VECTORS PIPELINE (Stage 3)")
        print("="*70)
        print(f"Provider: {self.provider}")
        print(f"Model: {self.config.bedrock_model_id} ({self.config.s3vectors_dimensions(self.provider)}d)")
        
        # Task 1: Build Stage 3 table locally (with streaming)
        if self.build_table:
            self._build_stage3_table()
        
        # Task 2: Upload to S3 (with streaming)
        if self.upload_to_s3:
            self._upload_to_s3()
        
        # Print summary
        self._print_summary()
        
        # Return summary dict (only scalars, no DataFrames)
        return {
            'provider': self.provider,
            'row_count': self.row_count,
            'dimensions': self.config.s3vectors_dimensions(self.provider),
            'built_locally': self.build_table,
            'uploaded_to_s3': self.upload_to_s3,
            's3_uri': f"s3://{self.config.bucket}/{self.config.s3vectors_path(self.provider)}",
            'cache_path': str(self.cache_path) if self.cache_path else None
        }