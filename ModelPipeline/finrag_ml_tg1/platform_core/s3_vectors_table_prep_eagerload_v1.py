# # THIS IS AN EAGER LOADING SCRIPT WHICH NOW DOESNT WORK: 
# # Reason, study memory explosion technical research notes in the same project.

# # ============================================================================
# # OUTDATED: Maintained for reference only
# # ============================================================================

# # ============================================================================
# # OUTDATED: Maintained for reference only
# # ============================================================================

# # ============================================================================
# # OUTDATED: Maintained for reference only
# # ============================================================================

# # ============================================================================
# # OUTDATED: Maintained for reference only
# # ============================================================================



# # ============================================================================
# # S3 VECTORS PIPELINE (Stage 3) - Execution Parameters

# # - Stage 3 is a JOIN operation (cheap, ~1-2 min)
# # - Depends on Stage 1 + Stage 2 (upstream changes)
# # - No complex merge logic needed (just rebuild)
# # - NOTE: ALWAYS does a simple rebuild, instead of tracking complexity with deltas.
# # ============================================================================

# # Build Stage 3 locally
# BUILD_S3VECTORS_TABLE = True      # Create Stage 3 from Stage 2 + Embeddings

# # S3 Table Initialization
# UPLOAD_TO_S3  = True      # Upload Stage 3 to S3
# FORCE_OVERWRITE_S3  = True    # Overwrite existing S3 table

# # Local Caching -- Not needed with always rebuild concept. DELETED.
# # CACHE_S3VECTORS_LOCALLY = False   # Download from S3 and cache
# # FORCE_RECACHE_S3VECTORS = False   # Re-download even if cached

# # Provider Selection
# S3VECTORS_PROVIDER = "cohere_1024d"  # 'cohere_1024d' or 'titan_1024d'

# # ============================================================================
# # IMPORTS
# # ============================================================================
# import polars as pl
# import mmh3
# import sys
# from pathlib import Path
# sys.path.append(str(Path.cwd().parent / 'loaders'))

# from ml_config_loader import MLConfig
# import polars as pl
# import mmh3

# # Helper function (from DATA PREP)
# from botocore.exceptions import ClientError

# def check_s3_exists(s3_client, bucket, s3_key):
#     """Check if S3 object exists"""
#     try:
#         s3_client.head_object(Bucket=bucket, Key=s3_key)
#         return True
#     except ClientError as e:
#         if e.response['Error']['Code'] == '404':
#             return False
#         else:
#             raise

# def _egress_cost_usd(bytes_count: int) -> float:
#     """Estimate S3 egress cost"""
#     gb = bytes_count / (1024 * 1024 * 1024)
#     return gb * 0.09

# # ============================================================================

# # 1. Standard: ends with numeric sequence → extract it
# # 2. Malformed: no numeric end → default to -1 (flag for investigation)

# def extract_sentence_position(sentenceID_series):
#     """
#     Extract sentence position from sentenceID with fallback
    
#     Logic:
#     - Split by '_', take last segment
#     - If numeric → cast to int16
#     - If non-numeric → return -1 (sentinel value)
    
#     Sentinel value choice: -1 vs 99 vs NULL
#     - -1: Clear indicator of extraction failure, sorts first
#     - 99: Ambiguous (could be real position)
#     - NULL: Causes filtering issues in S3 Vectors
    
#     Recommendation: Use -1
#     """
#     return (
#         pl.col(sentenceID_series)
#         .str.split('_')
#         .list.last()
#         .cast(pl.Int16, strict=False)  # strict=False → NULL on cast failure
#         .fill_null(-1)                  # NULL → -1 sentinel
#     )



# def build_s3vectors_stage3(meta_df, vectors_df, provider, config):
#     """
#     Transform Stage 2 (meta) + Embeddings → Stage 3 (S3 Vectors ready)
    
#     Args:
#         meta_df: Stage 2 meta table (34 cols, 1M rows)
#         vectors_df: Embeddings table (3 cols, ~900K rows)
#         provider: 'cohere_1024d' or 'titan_1024d'
#         config: MLConfig instance
    
#     Returns:
#         DataFrame with 10 columns, ready for S3 Vectors ingestion
    
#     Schema:
#         sentenceID_numsurrogate (int64)
#         sentenceID (varchar)
#         embedding (f32[1024])
#         cik_int (int32)
#         report_year (int16)
#         section_name (varchar)
#         sic (varchar)
#         sentence_pos (int16)
#         embedding_id (varchar)
#         section_sentence_count (int16)
#     """
    
#     print(f"\n[Stage 3 Build - {provider}]")
#     print(f"  Input Meta: {len(meta_df):,} rows × {len(meta_df.columns)} cols")
#     print(f"  Input Vectors: {len(vectors_df):,} rows × {len(vectors_df.columns)} cols")
    
#     # STEP 1: Inner Join (only embedded sentences)
#     df_joined = (
#         meta_df
#         .join(vectors_df, on='sentenceID', how='inner')
#         .filter(pl.col('embedding_id').is_not_null())  # Safety filter
#     )
    
#     print(f"  After join: {len(df_joined):,} rows")
    
#     if len(df_joined) == 0:
#         raise ValueError(f"No embeddings found for provider {provider}. Check join keys.")
    
#     # STEP 2: Derive sentenceID_numsurrogate (mmh3 hash)
#     print(f"  Computing mmh3 hashes...")
    
#     # Convert sentenceID to hash using mmh3
#     sentence_ids = df_joined['sentenceID'].to_list()
#     hashes = [mmh3.hash64(sid, signed=True)[0] for sid in sentence_ids]
    
#     df_joined = df_joined.with_columns([
#         pl.Series('sentenceID_numsurrogate', hashes, dtype=pl.Int64)
#     ])
    
#     # Validate hash uniqueness
#     hash_counts = df_joined.group_by('sentenceID_numsurrogate').agg(pl.count().alias('n'))
#     collisions = hash_counts.filter(pl.col('n') > 1)
    
#     if len(collisions) > 0:
#         print(f"  ⚠️  WARNING: {len(collisions)} hash collisions detected")
#         print(f"     Collision rate: {len(collisions)/len(df_joined)*100:.4f}%")
#         # Note: At 1M scale, collisions are extremely rare with mmh3
#     else:
#         print(f"  ✓ Hash uniqueness validated (0 collisions)")
    
#     # STEP 3: Extract sentence_pos with fallback
#     df_joined = df_joined.with_columns([
#         pl.col('sentenceID')
#           .str.split('_')
#           .list.last()
#           .cast(pl.Int16, strict=False)
#           .fill_null(-1)
#           .alias('sentence_pos')
#     ])
    
#     # Diagnostic: Check how many failed extraction
#     failed_extractions = df_joined.filter(pl.col('sentence_pos') == -1).height
#     if failed_extractions > 0:
#         print(f"  ⚠️  {failed_extractions} sentences with position extraction failure (set to -1)")
    
#     # STEP 4: Select & Order Final Columns
#     df_stage3 = df_joined.select([
#         # Primary keys
#         'sentenceID_numsurrogate',
#         'sentenceID',
        
#         # Embedding
#         'embedding',
        
#         # Filterable metadata
#         'cik_int',
#         'report_year',
#         'section_name',
#         'sic',
#         'sentence_pos',
        
#         # Non-filterable metadata
#         'embedding_id',
#         'section_sentence_count'
#     ])
    
#     # STEP 5: Validate Schema
#     expected_dims = config.s3vectors_dimensions(provider)
#     actual_dims = df_stage3['embedding'].list.len()[0]
    
#     if actual_dims != expected_dims:
#         raise ValueError(
#             f"Dimension mismatch for {provider}:\n"
#             f"  Expected: {expected_dims}d (from config)\n"
#             f"  Actual: {actual_dims}d (in embedding column)"
#         )
    
#     print(f"\n  ✓ Stage 3 Complete:")
#     print(f"    Rows: {len(df_stage3):,}")
#     print(f"    Columns: {len(df_stage3.columns)}")
#     print(f"    Dimensions: {actual_dims}d (validated)")
    
#     return df_stage3



# def initialize_s3vectors_table(config, provider, df_stage3, force_reinit=False):
#     """
#     Upload Stage 3 table to S3 (provider-specific)
    
#     Args:
#         config: MLConfig instance
#         provider: 'cohere_1024d' or 'titan_1024d'
#         df_stage3: Transformed DataFrame
#         force_reinit: If True, overwrite existing S3 table
#     """
    
#     s3_client = config.get_s3_client()
#     s3_key = config.s3vectors_path(provider)
#     s3_uri = f"s3://{config.bucket}/{s3_key}"
    
#     # Check if exists
#     exists = check_s3_exists(s3_client, config.bucket, s3_key)
    
#     if exists and not force_reinit:
#         print(f"\n[S3 Vectors Table - Already Exists]")
#         print(f"  Provider: {provider}")
#         print(f"  Location: {s3_uri}")
#         print(f"  Set FORCE_OVERWRITE_S3 =True to recreate")
#         return
    
#     elif exists and force_reinit:
#         print(f"\n[S3 Vectors Table - Recreating]")
#         print(f"  Provider: {provider}")
#         s3_client.delete_object(Bucket=config.bucket, Key=s3_key)
#         print(f"  ✓ Deleted existing")
    
#     # Upload to S3
#     print(f"\n[S3 Vectors Table - Creating]")
#     print(f"  Provider: {provider}")
#     print(f"  Destination: {s3_uri}")
#     print(f"  Rows: {len(df_stage3):,}")
#     print(f"  Dimensions: {config.s3vectors_dimensions(provider)}d")
    
#     df_stage3.write_parquet(
#         s3_uri,
#         storage_options=config.get_storage_options(),
#         compression='zstd'
#     )
    
#     print(f"  ✓ Upload complete (Cost: $0.00 ingress)")
    
    
    
# # wont be called for now.
# def cache_s3vectors_table(config, provider, force_recache=False):
#     """
#     Download and cache Stage 3 table locally
    
#     Args:
#         config: MLConfig instance
#         provider: 'cohere_1024d' or 'titan_1024d'
#         force_recache: If True, re-download even if cached
    
#     Returns:
#         DataFrame loaded from cache or S3
#     """
    
#     cache_path = config.get_s3vectors_cache_path(provider)
    
#     # Check local cache first
#     if not force_recache and cache_path.exists():
#         print(f"\n[S3 Vectors Table - Using Cache]")
#         print(f"  Provider: {provider}")
#         print(f"  Location: {cache_path.name}")
#         df = pl.read_parquet(cache_path)
#         print(f"  Loaded: {len(df):,} rows × {len(df.columns)} columns (Cost: $0.00)")
#         return df
    
#     # Download from S3
#     print(f"\n[S3 Vectors Table - Downloading from S3]")
#     print(f"  Provider: {provider}")
    
#     s3_key = config.s3vectors_path(provider)
#     s3_uri = f"s3://{config.bucket}/{s3_key}"
    
#     s3_client = config.get_s3_client()
    
#     # Check if exists on S3
#     if not check_s3_exists(s3_client, config.bucket, s3_key):
#         raise FileNotFoundError(
#             f"S3 Vectors table not found for {provider}!\n"
#             f"  Expected: {s3_uri}\n"
#             f"  Run with UPLOAD_TO_S3 =True to create"
#         )
    
#     response = s3_client.head_object(Bucket=config.bucket, Key=s3_key)
#     file_size_mb = response['ContentLength'] / 1024 / 1024
#     egress_cost = _egress_cost_usd(response['ContentLength'])
    
#     print(f"  Source: {s3_uri}")
#     print(f"  Size: {file_size_mb:.1f} MB")
    
#     df = pl.read_parquet(s3_uri, storage_options=config.get_storage_options())
#     print(f"  Downloaded: {len(df):,} rows (Cost: ${egress_cost:.4f} egress)")
    
#     # Cache for future use
#     cache_path.parent.mkdir(parents=True, exist_ok=True)
#     df.write_parquet(cache_path, compression='zstd')
#     print(f"  ✓ Cached to: {cache_path}")
    
#     return df
# # ============================================================================
# # MAIN EXECUTION
# # ============================================================================

# config = MLConfig()

# print("="*70)
# print("S3 VECTORS PIPELINE (Stage 3)")
# print("="*70)
# print(f"Provider: {S3VECTORS_PROVIDER}")
# print(f"Model: {config.bedrock_model_id} ({config.s3vectors_dimensions(S3VECTORS_PROVIDER)}d)")

# df_stage3 = None

# # ============================================================================
# # TASK 1: Build Stage 3 Table Locally
# # ============================================================================

# if BUILD_S3VECTORS_TABLE:
#     # Load dependencies
#     print(f"\n[Loading Dependencies]")
    
#     # Stage 2 meta table
#     meta_cache = Path.cwd().parent / 'data_cache' / 'meta_embeds' / 'finrag_fact_sentences_meta_embeds.parquet'
#     if not meta_cache.exists():
#         raise FileNotFoundError(f"Stage 2 meta not cached: {meta_cache}")
#     meta_df = pl.read_parquet(meta_cache)
#     print(f"  ✓ Stage 2 Meta: {len(meta_df):,} rows")
    
#     # Embeddings table
#     emb_filename = Path(config.embeddings_path(S3VECTORS_PROVIDER)).name
#     emb_cache = Path.cwd().parent / 'data_cache' / 'embeddings' / S3VECTORS_PROVIDER / emb_filename
#     if not emb_cache.exists():
#         raise FileNotFoundError(f"Embeddings not cached: {emb_cache}")
#     vectors_df = pl.read_parquet(emb_cache)
#     print(f"  ✓ Embeddings: {len(vectors_df):,} rows")
    
#     # Build Stage 3
#     df_stage3 = build_s3vectors_stage3(meta_df, vectors_df, S3VECTORS_PROVIDER, config)
    
#     # Cache locally
#     cache_path = config.get_s3vectors_cache_path(S3VECTORS_PROVIDER)
#     cache_path.parent.mkdir(parents=True, exist_ok=True)
#     df_stage3.write_parquet(cache_path, compression='zstd')
#     print(f"\n  ✓ Cached locally: {cache_path}")

# # ============================================================================
# # TASK 2: Initialize S3 Table
# # ============================================================================

# if UPLOAD_TO_S3 :
#     if df_stage3 is None:
#         # Load from local cache
#         cache_path = config.get_s3vectors_cache_path(S3VECTORS_PROVIDER)
#         if not cache_path.exists():
#             raise FileNotFoundError(f"Stage 3 not built. Run with BUILD_S3VECTORS_TABLE=True first.")
#         df_stage3 = pl.read_parquet(cache_path)
    
#     initialize_s3vectors_table(config, S3VECTORS_PROVIDER, df_stage3, force_reinit=FORCE_OVERWRITE_S3 )

# # ============================================================================
# # TASK 3: Cache from S3
# # ============================================================================

# #  DELETED. DELETED. DELETED. DELETED.
# # if CACHE_S3VECTORS_LOCALLY:
# #     df_cached = cache_s3vectors_table(config, S3VECTORS_PROVIDER, force_recache=FORCE_RECACHE_S3VECTORS)

# # ============================================================================
# # SUMMARY
# # ============================================================================

# print(f"\n{'='*70}")
# print(f"✓ S3 VECTORS PIPELINE COMPLETE")
# print(f"{'='*70}")
# print(f"\nActions completed:")
# if BUILD_S3VECTORS_TABLE:
#     print(f"  ✓ Stage 3 built locally ({len(df_stage3):,} rows)")
# if UPLOAD_TO_S3 :
#     print(f"  ✓ Uploaded to S3")

# print("="*70)