"""
Data preparation pipeline for FinRAG Stage 1→2 transformation and caching.
"""

"""
Data preparation pipeline for FinRAG Stage 1→2 transformation and caching.
"""

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

# Now use absolute imports from ModelPipeline root
from finrag_ml_tg1.loaders.ml_config_loader import MLConfig
import polars as pl
from botocore.exceptions import ClientError


# ====== Parmameters - Explained, for easy toggling ==========================================================
# # S3 Table Initialization (One-time setup)
# INIT_META_TABLE = False           # Create Stage 2 meta table (24→35 cols)
# INIT_VECTORS_TABLE = False        # Create empty vectors table

# FORCE_REINIT_META = False        # Delete and recreate meta table
# FORCE_REINIT_VECTORS = False     # Delete and recreate vectors table

# # Local Caching (Development optimization)
# CACHE_STAGE1_LOCALLY = True      # Download Stage 1 fact table
# FORCE_RECACHE_STAGE1 = True     # Re-download even if cached

# CACHE_STAGE2_LOCALLY = True       # Download Stage 2 (meta_embeds) table
# FORCE_RECACHE_STAGE2 = True      # Re-download Stage 2 even if cached

# CACHE_EMBEDS_LOCALLY = True       # Download embeddings fact table(s)
# FORCE_RECACHE_EMBEDS = True      # Re-download embeddings even if cached

# EMBEDS_PROVIDER = "cohere_1024d"            # "cohere_1024d" (None → try sensible default or all)
# ===============================================================================================================

class DataPreparationPipeline:
    """
    Self-contained pipeline for Stage 1→2 transformation and S3/local caching.
    
    Usage:
        pipeline = DataPreparationPipeline(
            cache_stage1_locally=True,
            force_recache_stage1=True,
            cache_stage2_locally=True,
            force_recache_stage2=True,
            cache_embeds_locally=True,
            force_recache_embeds=True,
            embeds_provider="cohere_1024d",
            init_meta_table=False,
            force_reinit_meta=False,
            init_vectors_table=False,
            force_reinit_vectors=False
        )
        pipeline.run()
    """
    
    def __init__(
        self,
        init_meta_table=False,
        init_vectors_table=False,
        force_reinit_meta=False,
        force_reinit_vectors=False,
        cache_stage1_locally=True,
        force_recache_stage1=True,
        cache_stage2_locally=True,
        force_recache_stage2=True,
        cache_embeds_locally=True,
        force_recache_embeds=True,
        embeds_provider="cohere_1024d"
    ):
        self.init_meta_table = init_meta_table
        self.init_vectors_table = init_vectors_table
        self.force_reinit_meta = force_reinit_meta
        self.force_reinit_vectors = force_reinit_vectors
        self.cache_stage1_locally = cache_stage1_locally
        self.force_recache_stage1 = force_recache_stage1
        self.cache_stage2_locally = cache_stage2_locally
        self.force_recache_stage2 = force_recache_stage2
        self.cache_embeds_locally = cache_embeds_locally
        self.force_recache_embeds = force_recache_embeds
        self.embeds_provider = embeds_provider
        
        # Load config internally
        self.config = MLConfig()
        
        # State tracking
        self.df_stage1 = None
        self.df_stage2_meta = None
        self.embeds_cached = {}
    
    # ========================================================================
    # PRIVATE HELPER METHODS
    # ========================================================================
    
    @staticmethod
    def _check_s3_exists(s3_client, bucket, s3_key):
        """Check if S3 object exists (no download, no cost)"""
        try:
            s3_client.head_object(Bucket=bucket, Key=s3_key)
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            else:
                raise
    
    @staticmethod
    def _egress_cost_usd(bytes_count: int) -> float:
        """Calculate rough S3 egress cost"""
        gb = bytes_count / (1024 * 1024 * 1024)
        return gb * 0.09
    
    def _download_parquet_from_s3_to_local(self, s3_key: str, local_path: Path) -> pl.DataFrame:
        """Read a parquet from S3 via Polars and cache locally (zstd)."""
        s3_uri = f"s3://{self.config.bucket}/{s3_key}"
        s3_client = self.config.get_s3_client()
        head = s3_client.head_object(Bucket=self.config.bucket, Key=s3_key)
        size_mb = head['ContentLength'] / 1024 / 1024
        cost = self._egress_cost_usd(head['ContentLength'])

        print(f"  Source: {s3_uri}")
        print(f"  Size: {size_mb:.1f} MB")
        df = pl.read_parquet(s3_uri, storage_options=self.config.get_storage_options())
        print(f"  Downloaded: {len(df):,} rows (Cost: ${cost:.4f} egress)")

        local_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(local_path, compression='zstd')
        print(f"  ✓ Cached to: {local_path}")
        return df
    
    def _cache_stage1_table(self):
        """Download and cache Stage 1 fact table (original 24 columns)"""
        cache_file = Path.cwd().parent / 'data_cache' / 'stage1_facts' / 'finrag_fact_sentences.parquet'
        
        # Check local cache first
        if not self.force_recache_stage1 and cache_file.exists():
            print(f"\n[Stage 1 Table - Using Cache]")
            print(f"  Location: {cache_file.name}")
            df = pl.read_parquet(cache_file)
            print(f"  Loaded: {len(df):,} rows × {len(df.columns)} columns (Cost: $0.00)")
            return df
        
        # Download from S3
        print(f"\n[Stage 1 Table - Downloading from S3]")
        s3_uri = f"s3://{self.config.bucket}/{self.config.input_sentences_path}"
        
        s3_client = self.config.get_s3_client()
        response = s3_client.head_object(Bucket=self.config.bucket, Key=self.config.input_sentences_path)
        file_size_mb = response['ContentLength'] / 1024 / 1024
        egress_cost = file_size_mb / 1024 * 0.09
        
        print(f"  Source: {s3_uri}")
        print(f"  Size: {file_size_mb:.1f} MB")
        
        df = pl.read_parquet(s3_uri, storage_options=self.config.get_storage_options())
        print(f"  Downloaded: {len(df):,} rows (Cost: ${egress_cost:.4f} egress)")
        
        # Cache for future use
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(cache_file, compression='zstd')
        print(f"  ✓ Cached to: {cache_file}")
        
        return df
    
    def _cache_stage2_meta_table(self):
        """Download and cache Stage-2 meta table (35 columns with ML metadata)"""
        meta_key = self.config.meta_embeds_path
        meta_filename = Path(meta_key).name
        cache_file = Path.cwd().parent / 'data_cache' / 'meta_embeds' / meta_filename

        if not self.force_recache_stage2 and cache_file.exists():
            print(f"\n[Stage 2 Meta Table - Using Cache]")
            print(f"  Location: {cache_file.name}")
            df = pl.read_parquet(cache_file)
            print(f"  Loaded: {len(df):,} rows × {len(df.columns)} columns (Cost: $0.00)")
            return df

        print(f"\n[Stage 2 Meta Table - Downloading from S3]")
        return self._download_parquet_from_s3_to_local(meta_key, cache_file)
    
    def _resolve_embed_providers(self, explicit: str | None):
        """Try to resolve provider list from config or explicit parameter"""
        if explicit:
            return [explicit]

        if hasattr(self.config, "embedding_providers") and isinstance(self.config.embedding_providers, (list, tuple)) and self.config.embedding_providers:
            return list(self.config.embedding_providers)

        if hasattr(self.config, "bedrock_default_model_key") and self.config.bedrock_default_model_key:
            return [self.config.bedrock_default_model_key]

        candidates = [k for k in ("cohere_768d", "cohere_1024d", "titan_1024d") if hasattr(self.config, "embeddings_path") and self.config.embeddings_path(provider=k)]
        return candidates or []
    
    def _cache_embeddings_tables(self):
        """Download and cache embeddings parquet for one or many providers"""
        providers = self._resolve_embed_providers(self.embeds_provider)
        results = {}
        
        if not providers:
            print("\n[Embeddings Fact - No providers resolved] Skipping.")
            return results

        for prov in providers:
            try:
                emb_key = self.config.embeddings_path(provider=prov)
            except TypeError:
                emb_key = getattr(self.config, "embeddings_path", None)
            
            if not emb_key:
                print(f"\n[Embeddings Fact - {prov}] Cannot resolve S3 key via config.embeddings_path(provider=...). Skipping.")
                continue

            emb_filename = Path(emb_key).name
            cache_file = Path.cwd().parent / 'data_cache' / 'embeddings' / prov / emb_filename

            if not self.force_recache_embeds and cache_file.exists():
                print(f"\n[Embeddings Fact - Using Cache] Provider: {prov}")
                print(f"  Location: {emb_filename}")
                df = pl.read_parquet(cache_file)
                print(f"  Loaded: {len(df):,} rows × {len(df.columns)} columns (Cost: $0.00)")
                results[prov] = df
                continue

            print(f"\n[Embeddings Fact - Downloading from S3] Provider: {prov}")
            df = self._download_parquet_from_s3_to_local(emb_key, cache_file)
            results[prov] = df

        return results
    
    def _add_ml_columns(self, df):
        """Transform Stage 1 (24 cols) → Stage 2 (35 cols) with ML metadata"""
        
        # Extract sequence for sorting
        df = df.with_columns([
            pl.col('sentenceID').str.split('_').list.slice(0, -1).list.join('_').alias('_doc_prefix'),
            pl.col('sentenceID').str.split('_').list.last().cast(pl.Int32).alias('_sequence_num')
        ])
        
        df = df.sort(['_doc_prefix', '_sequence_num'])
        
        # Neighbor pointers
        df = df.with_columns([
            pl.col('sentenceID').shift(1).over('_doc_prefix').alias('prev_sentenceID'),
            pl.col('sentenceID').shift(-1).over('_doc_prefix').alias('next_sentenceID')
        ])
        
        # Content metadata
        df = df.with_columns([
            pl.col('sentence').str.len_chars().alias('sentence_char_length'),
            ((pl.col('sentence').str.count_matches(' ') + 1) * 1.33).cast(pl.Int16).alias('sentence_token_count')
        ])
        
        # Section metadata
        section_counts = df.group_by(['docID', 'section_ID']).agg(pl.len().alias('section_sentence_count'))
        df = df.join(section_counts, on=['docID', 'section_ID'], how='left')
        
        # ML metadata (NULL initially)
        df = df.with_columns([
            pl.lit(None).cast(pl.Utf8).alias('embedding_id'),
            pl.lit(None).cast(pl.Utf8).alias('embedding_model'),
            pl.lit(None).cast(pl.Int16).alias('embedding_dims'),
            pl.lit(None).cast(pl.Datetime).alias('embedding_date'),
            pl.lit(None).cast(pl.Utf8).alias('embedding_ref')
        ])
        
        # Cleanup
        df = df.drop(['_doc_prefix', '_sequence_num'])
        
        print(f"  ✓ Transformed: 24 cols → {len(df.columns)} cols")
        
        return df
    
    def _initialize_meta_table(self):
        """Create Stage 2 meta table on S3 (35 columns)"""
        s3_client = self.config.get_s3_client()
        meta_s3_key = self.config.meta_embeds_path
        meta_exists = self._check_s3_exists(s3_client, self.config.bucket, meta_s3_key)
        
        # Handle existing table
        if meta_exists and not self.force_reinit_meta:
            print(f"\n[Stage 2 Meta Table - Already Exists]")
            print(f"  Location: s3://{self.config.bucket}/{meta_s3_key}")
            print(f"  Set force_reinit_meta=True to recreate")
            return
        
        elif meta_exists and self.force_reinit_meta:
            print(f"\n[Stage 2 Meta Table - Recreating]")
            s3_client.delete_object(Bucket=self.config.bucket, Key=meta_s3_key)
            print(f"  ✓ Deleted existing")
        
        # Create new table
        print(f"\n[Stage 2 Meta Table - Creating]")
        
        if self.df_stage1 is None:
            print(f"\n  Loading Stage 1 for transformation...")
            self.df_stage1 = self._cache_stage1_table()
        
        df_stage2 = self._add_ml_columns(self.df_stage1)
        
        meta_uri = f"s3://{self.config.bucket}/{meta_s3_key}"
        print(f"  Saving to: {meta_uri}")
        df_stage2.write_parquet(meta_uri, storage_options=self.config.get_storage_options(), compression='zstd')
        print(f"  ✓ Created: {len(df_stage2):,} rows × {len(df_stage2.columns)} cols (Cost: $0.00 ingress)")
    
    def _initialize_vectors_table(self):
        """Create empty vectors table on S3"""
        s3_client = self.config.get_s3_client()
        vectors_s3_key = self.config.embeddings_path(provider=None)
        vectors_exists = self._check_s3_exists(s3_client, self.config.bucket, vectors_s3_key)
        
        # Handle existing table
        if vectors_exists and not self.force_reinit_vectors:
            print(f"\n[Vectors Table - Already Exists]")
            print(f"  Location: s3://{self.config.bucket}/{vectors_s3_key}")
            print(f"  Provider: {self.config.bedrock_default_model_key} ({self.config.bedrock_dimensions}d)")
            print(f"  Set force_reinit_vectors=True to recreate")
            return
        
        elif vectors_exists and self.force_reinit_vectors:
            print(f"\n[Vectors Table - Recreating]")
            s3_client.delete_object(Bucket=self.config.bucket, Key=vectors_s3_key)
            print(f"  ✓ Deleted existing")
        
        # Create empty table
        print(f"\n[Vectors Table - Creating]")
        empty_vectors = pl.DataFrame({
            'sentenceID': pl.Series([], dtype=pl.Utf8),
            'embedding_id': pl.Series([], dtype=pl.Utf8),
            'embedding': pl.Series([], dtype=pl.List(pl.Float32))
        })
        
        vectors_uri = f"s3://{self.config.bucket}/{vectors_s3_key}"
        empty_vectors.write_parquet(vectors_uri, storage_options=self.config.get_storage_options(), compression='zstd')
        print(f"  ✓ Created: {vectors_uri}")
        print(f"  Schema: [sentenceID, embedding_id, embedding (Float32)]")
    
    def _print_summary(self):
        """Print final execution summary"""
        print(f"\n{'='*70}")
        print(f"✓ DATA PREPARATION COMPLETE")
        print(f"{'='*70}")
        print(f"\nTables initialized / cached:")
        
        if self.cache_stage1_locally:
            print(f"  ✓ Stage 1 cached locally")
        
        if self.cache_stage2_locally:
            print(f"  ✓ Stage 2 meta cached locally")
        
        if self.cache_embeds_locally:
            if self.embeds_cached:
                prows = ", ".join([f"{k}: {len(v)} rows" for k, v in self.embeds_cached.items()])
                print(f"  ✓ Embeddings cached locally → {{ {prows} }}")
            else:
                print(f"  ✓ Embeddings cache attempted (no providers resolved)")
        
        if self.init_meta_table:
            print(f"  ✓ Stage 2 meta table ready (S3)")
        
        if self.init_vectors_table:
            print(f"  ✓ Vectors table ready (S3)")
        
        print("="*70)
    
    # ========================================================================
    # PUBLIC INTERFACE
    # ========================================================================
    
    def run(self):
        """Execute the full data preparation pipeline"""
        print("="*70)
        print("DATA PREPARATION PIPELINE")
        print("="*70)
        print(f"Model: {self.config.bedrock_model_id} ({self.config.bedrock_dimensions}d)")
        
        # Task 1: Cache Stage 1 Table Locally
        if self.cache_stage1_locally:
            self.df_stage1 = self._cache_stage1_table()
        
        # Task 2: Cache Stage 2 Meta Table Locally
        if self.cache_stage2_locally:
            self.df_stage2_meta = self._cache_stage2_meta_table()
        
        # Task 3: Cache Embeddings Fact Table(s) Locally
        if self.cache_embeds_locally:
            self.embeds_cached = self._cache_embeddings_tables()
        
        # Task 4: Initialize Meta Table (Stage 2) on S3
        if self.init_meta_table:
            self._initialize_meta_table()
        
        # Task 5: Initialize Empty Vectors Table on S3
        if self.init_vectors_table:
            self._initialize_vectors_table()
        
        # Print summary
        self._print_summary()
        
        return {
            'stage1_cached': self.df_stage1 is not None,
            'stage2_meta_cached': self.df_stage2_meta is not None,
            'embeds_cached': list(self.embeds_cached.keys()),
            'meta_table_initialized': self.init_meta_table,
            'vectors_table_initialized': self.init_vectors_table
        }