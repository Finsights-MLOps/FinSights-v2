"""
Embedding generation pipeline for FinRAG.
Generates embeddings for filtered sentences and merges with existing data.

Main Pipeline Steps:
    Load meta table (with caching logic)
    Filter sentences (based on mode: "full" or "parameterized")
    Generate embeddings (batch processing with Bedrock API)
    Merge vectors table (ETL pattern)
    Update meta table (with embedding metadata)
    Save to S3 + local cache

Helper Functions:
    load_meta_table_with_cache() - loads from cache or S3
    filter_sentences() - applies filters based on config
    generate_embeddings_batch() - main embedding generation with batching
    _call_bedrock_api() - internal API caller
    merge_vectors_table() - ETL merge pattern
    update_meta_table() - ETL update pattern
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

from finrag_ml_tg1.loaders.ml_config_loader import MLConfig
import polars as pl
import json
from datetime import datetime
import time


class EmbeddingGenerationPipeline:
    """
    Self-contained pipeline for generating and storing sentence embeddings.
    
    All configuration (mode, filters, model, paths) via MLConfig.
    Batching parameters control API efficiency and can be customized if needed.
    
    Usage:
        pipeline = EmbeddingGenerationPipeline()
        summary = pipeline.run()
    
    Advanced usage with custom batching:
        pipeline = EmbeddingGenerationPipeline(
            max_tokens_per_sentence=1000,    # Filter extreme outliers
            max_texts_per_batch=96,          # Cohere API limit
            max_tokens_per_batch=128000,     # Cohere v4 capacity
            batch_log_interval=40            # Progress update frequency
        )
        summary = pipeline.run()
    """
    
    def __init__(
        self,
        max_tokens_per_sentence=1000,
        max_texts_per_batch=96,
        max_tokens_per_batch=128000,
        batch_log_interval=40
    ):
        """
        Initialize embedding generation pipeline.
        
        Args:
            max_tokens_per_sentence: Filter outliers above this token count
            max_texts_per_batch: Cohere API limit (96 texts per batch)
            max_tokens_per_batch: Cohere v4 capacity (128K tokens)
            batch_log_interval: Print progress every N batches
        """
        self.max_tokens_per_sentence = max_tokens_per_sentence
        self.max_texts_per_batch = max_texts_per_batch
        self.max_tokens_per_batch = max_tokens_per_batch
        self.batch_log_interval = batch_log_interval
        
        # Load config internally
        self.config = MLConfig()
        
        # Resolve S3 paths once at initialization
        self.vectors_s3_key = self.config.embeddings_path(provider=None)
        self.meta_s3_key = self.config.meta_embeds_path
        self.vectors_uri = f"s3://{self.config.bucket}/{self.vectors_s3_key}"
        self.meta_uri = f"s3://{self.config.bucket}/{self.meta_s3_key}"
        
        # State tracking
        self.df_meta = None
        self.df_filtered = None
        self.filtered_ids = []
        self.df_new_vectors = None
        self.embedding_id = None
        self.skipped_ids = []
        self.df_merged_vectors = None
        self.df_updated_meta = None
        self.total_tokens = 0
        self.embedding_cost = 0.0
    
    # ========================================================================
    # PRIVATE HELPER METHODS
    # ========================================================================
    
    def _load_meta_table_with_cache(self):
        """Load meta table (cache first, S3 fallback, error if missing)"""
        cache_file = Path.cwd().parent / 'data_cache' / 'meta_embeds' / 'finrag_fact_sentences_meta_embeds.parquet'
        
        if cache_file.exists():
            print(f"✓ Using cached meta table")
            df = pl.read_parquet(cache_file)
            print(f"  Loaded: {len(df):,} rows × {len(df.columns)} columns (Cost: $0.00)")
            return df
        
        # Try S3
        s3_client = self.config.get_s3_client()
        
        try:
            response = s3_client.head_object(Bucket=self.config.bucket, Key=self.meta_s3_key)
            file_size_bytes = response['ContentLength']
            file_size_mb = file_size_bytes / 1024 / 1024
            egress_cost = file_size_mb / 1024 * 0.09

            print(f"⬇️  Loading meta table from S3")
            df = pl.read_parquet(self.meta_uri, storage_options=self.config.get_storage_options())
            
            print(f"  Downloaded: {len(df):,} rows × {len(df.columns)} columns")
            print(f"  Cost: ${egress_cost:.4f} egress")
            
            # Cache for next time
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            df.write_parquet(cache_file)
            print(f"✓ Cached for future use")
            
            return df
            
        except:
            raise FileNotFoundError(
                f"Meta table not found!\n"
                f"  Cache: {cache_file}\n"
                f"  S3: {self.meta_uri}\n"
                f"  → Run data preparation pipeline first to create meta table"
            )
    
    def _filter_sentences(self):
        """Filter sentences with multi-value support for CIK and year"""
        mode = self.config.embedding_mode
        
        if mode == "full":
            print(f"\n[Filtering: FULL MODE]")
            df_filtered = self.df_meta
        
        elif mode == "parameterized":
            cik_filter = self.config.filter_cik
            year_filter = self.config.filter_year
            
            # Validate filters are not null
            if cik_filter is None or year_filter is None:
                raise ValueError(
                    "PARAMETERIZED mode requires filters!\n"
                    "  Set 'cik_int' and 'year' in ml_config.yaml\n"
                    "  Example: cik_int: [320193] or cik_int: [320193, 789019]\n"
                    "  Example: year: [2022] or year: [2020, 2021, 2022]"
                )
            
            # Ensure filters are lists
            cik_list = cik_filter if isinstance(cik_filter, list) else [cik_filter]
            year_list = year_filter if isinstance(year_filter, list) else [year_filter]
            
            print(f"\n[Filtering: PARAMETERIZED MODE]")
            print(f"  CIKs: {cik_list}")
            print(f"  Years: {year_list}")
            
            # Apply filters
            df_filtered = self.df_meta.filter(
                pl.col('cik_int').is_in(cik_list) &
                pl.col('report_year').is_in(year_list)
            )
            
            if len(df_filtered) == 0:
                raise ValueError(f"No sentences found for cik={cik_list}, year={year_list}")
            
            # Show selected companies
            companies = df_filtered.select(['cik_int', 'name']).unique().sort('cik_int')
            print(f"  Companies selected:")
            for row in companies.iter_rows(named=True):
                print(f"    - {row['name']} (CIK: {row['cik_int']})")
            
            print(f"  Total sentences: {len(df_filtered):,}")
        
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        filtered_ids = df_filtered['sentenceID'].to_list()
        
        return df_filtered, filtered_ids
    
    def _generate_embeddings_batch(self):
        """
        Generate embeddings with intelligent batching and outlier handling
        
        Returns: tuple of (df_vectors, embedding_id, skipped_ids)
        """
        bedrock = self.config.get_bedrock_client()
        model_id = self.config.bedrock_model_id
        input_type = self.config.bedrock_input_type
        dimensions = self.config.bedrock_dimensions
        
        print(f"\n[Generating Embeddings]")
        print(f"  Model: {model_id}")
        print(f"  Dimensions: {dimensions}")
        print(f"  Input sentences: {len(self.df_filtered):,}")
        
        # Pre-filter: Remove extreme outliers
        df_valid = self.df_filtered.filter(
            pl.col('sentence_token_count') <= self.max_tokens_per_sentence
        )
        
        df_skipped = self.df_filtered.filter(
            pl.col('sentence_token_count') > self.max_tokens_per_sentence
        )
        
        if len(df_skipped) > 0:
            skipped_pct = len(df_skipped) / len(self.df_filtered) * 100
            print(f"  ⚠️  Skipped {len(df_skipped):,} outliers (>{self.max_tokens_per_sentence} tokens, {skipped_pct:.2f}%)")
        
        print(f"  Embedding: {len(df_valid):,} sentences")
        
        # Prepare for batching
        all_embeddings = []
        all_sentence_ids = []
        sentences_data = df_valid.select(['sentenceID', 'sentence', 'sentence_token_count']).to_dicts()
        
        current_batch = []
        current_batch_tokens = 0
        batches_processed = 0
        total_tokens = 0
        t0 = time.perf_counter()
        batch_times = []

        # Token-aware batching
        for idx, row in enumerate(sentences_data):
            sent_id = row['sentenceID']
            sent_text = row['sentence']
            sent_tokens = row['sentence_token_count']
            
            # Check if batch is full
            would_exceed_tokens = (current_batch_tokens + sent_tokens) > self.max_tokens_per_batch
            would_exceed_size = len(current_batch) >= self.max_texts_per_batch
            
            if would_exceed_tokens or would_exceed_size:
                # Process current batch
                if current_batch:
                    b_start = time.perf_counter()
                    batch_embeddings = self._call_bedrock_api(
                        bedrock, model_id, current_batch, input_type, dimensions
                    )
                    batch_time = time.perf_counter() - b_start
                    batch_times.append(batch_time)

                    all_embeddings.extend(batch_embeddings)
                    batches_processed += 1
                    total_tokens += current_batch_tokens
                    
                    # Progress every n batches
                    if batches_processed % self.batch_log_interval == 0:
                        avg = sum(batch_times) / len(batch_times)
                        eta = avg * ((len(df_valid) / len(current_batch)) - batches_processed)
                        print(f"    Batch Number: {batches_processed} | Progress: {len(all_embeddings):,}/{len(df_valid):,} "
                              f"({len(all_embeddings)/len(df_valid)*100:.1f}%) "
                              f"| last {batch_time:.2f}s | avg/batch {avg:.2f}s | ETA {eta:.0f}s")
                
                # Reset batch
                current_batch = []
                current_batch_tokens = 0
            
            # Add to batch
            current_batch.append({'id': sent_id, 'text': sent_text})
            current_batch_tokens += sent_tokens
            all_sentence_ids.append(sent_id)
        
        # Final batch
        if current_batch:
            b_start = time.perf_counter()
            batch_embeddings = self._call_bedrock_api(
                bedrock, model_id, current_batch, input_type, dimensions
            )
            batch_time = time.perf_counter() - b_start
            batch_times.append(batch_time)

            all_embeddings.extend(batch_embeddings)
            batches_processed += 1
            total_tokens += current_batch_tokens

        elapsed = time.perf_counter() - t0
        print(f"  ✓ Completed: {len(all_embeddings):,} embeddings in {batches_processed} batches "
              f"| time {elapsed:.1f}s | avg/batch {elapsed/max(1,batches_processed):.2f}s")

        embedding_id = f"bedrock_cohere_v4_{dimensions}d_{datetime.now().strftime('%Y%m%d_%H%M')}"
        df_vectors = pl.DataFrame({
            'sentenceID': all_sentence_ids,
            'embedding_id': [embedding_id] * len(all_embeddings),
            'embedding': pl.Series(all_embeddings, dtype=pl.List(pl.Float32))
        })

        cost = total_tokens / 1000 * self.config.get_cost_per_1k()
        print(f"  Tokens: {total_tokens:,} | Cost: ${cost:.4f}")

        # Store for summary
        self.total_tokens = total_tokens
        self.embedding_cost = cost

        skipped_ids = df_skipped['sentenceID'].to_list() if len(df_skipped) > 0 else []
        return df_vectors, embedding_id, skipped_ids
    
    def _call_bedrock_api(self, bedrock, model_id, batch, input_type, dimensions):
        """Single Bedrock API call (internal helper)"""
        texts = [item['text'] for item in batch]
        
        body = json.dumps({
            "texts": texts,
            "input_type": input_type,
            "embedding_types": ["float"],
            "output_dimension": dimensions,
            "max_tokens": 128000,
            "truncate": "RIGHT"
        })
        
        response = bedrock.invoke_model(
            body=body,
            modelId=model_id,
            accept='*/*',
            contentType='application/json'
        )
        
        result = json.loads(response['body'].read())
        return result['embeddings']['float']
    
    def _merge_vectors_table(self):
        """Merge new vectors with existing (table exists from data prep)"""
        # Load existing (guaranteed to exist)
        existing_vectors = pl.read_parquet(
            self.vectors_uri,
            storage_options=self.config.get_storage_options()
        )
        
        print(f"\n[Merging Vectors]")
        print(f"  Existing: {len(existing_vectors):,} rows")
        print(f"  New: {len(self.df_new_vectors):,} rows")
        
        # Ensure column order matches before concat
        new_vectors = self.df_new_vectors.select(existing_vectors.columns)

        # ETL pattern: concat + unique
        merged = pl.concat([existing_vectors, new_vectors])
        merged = merged.unique(subset=['sentenceID'], keep='last')
        
        duplicates = len(existing_vectors) + len(new_vectors) - len(merged)
        print(f"  ✓ Merged: {len(merged):,} rows (replaced {duplicates:,})")
        
        return merged
    
    def _update_meta_table(self):
        """Update metadata columns for embedded sentences (ETL pattern)"""
        # Successfully embedded = filtered - skipped
        embedded_ids = [sid for sid in self.filtered_ids if sid not in self.skipped_ids]
        
        print(f"\n[Updating Meta Table]")
        print(f"  Total rows: {len(self.df_meta):,}")
        print(f"  Successfully embedded: {len(embedded_ids):,}")
        if self.skipped_ids:
            print(f"  Skipped (outliers): {len(self.skipped_ids):,}")
        
        model_info = {
            'embedding_id': self.embedding_id,
            'model': self.config.bedrock_model_id,
            'dimensions': self.config.bedrock_dimensions,
            'timestamp': datetime.now()
        }
        
        # Create updated rows for embedded sentences
        df_updated_rows = self.df_meta.filter(
            pl.col('sentenceID').is_in(embedded_ids)
        ).with_columns([
            pl.lit(model_info['embedding_id']).alias('embedding_id'),
            pl.lit(model_info['model']).alias('embedding_model'),
            pl.lit(model_info['dimensions']).cast(pl.Int16).alias('embedding_dims'),
            pl.lit(model_info['timestamp']).alias('embedding_date'),
            pl.lit(self.vectors_uri).alias('embedding_ref')
        ])
        
        # Rows not embedded (keep unchanged)
        df_unchanged_rows = self.df_meta.filter(
            ~pl.col('sentenceID').is_in(embedded_ids)
        )
        
        # ETL pattern: concat + unique
        merged_meta = pl.concat([df_unchanged_rows, df_updated_rows])
        merged_meta = merged_meta.unique(subset=['sentenceID'], keep='last')
        
        # Verify row count
        assert len(merged_meta) == len(self.df_meta), "Row count mismatch after meta update!"
        
        total_embedded = merged_meta.filter(pl.col('embedding_id').is_not_null()).shape[0]
        print(f"  ✓ Updated: {total_embedded:,} total rows now have embeddings")
        
        return merged_meta
    
    def _save_results(self):
        """Save results to S3 and local cache"""
        print(f"\n[Saving Results]")

        # Derive filenames/folders from resolved S3 keys
        vectors_filename = Path(self.vectors_s3_key).name
        vectors_provider = Path(self.vectors_s3_key).parent.name
        meta_filename = Path(self.meta_s3_key).name

        # Canonical local cache locations
        meta_cache_dir = Path.cwd().parent / 'data_cache' / 'meta_embeds'
        vectors_cache_dir = Path.cwd().parent / 'data_cache' / 'embeddings' / vectors_provider

        meta_cache_dir.mkdir(parents=True, exist_ok=True)
        vectors_cache_dir.mkdir(parents=True, exist_ok=True)

        meta_cache_path = meta_cache_dir / meta_filename
        vectors_cache_path = vectors_cache_dir / vectors_filename

        # Save vectors to S3
        print(f"  Vectors → S3: {self.vectors_uri}")
        self.df_merged_vectors.write_parquet(
            self.vectors_uri,
            storage_options=self.config.get_storage_options(),
            compression='zstd'
        )
        print(f"  ✓ S3 saved: {len(self.df_merged_vectors):,} rows (Cost: $0.00 - ingress)")

        # Save vectors to local cache
        print(f"  Vectors → Local: {vectors_cache_path}")
        self.df_merged_vectors.write_parquet(vectors_cache_path, compression='zstd')
        print(f"  ✓ Cached locally")

        # Save meta to S3
        print(f"  Meta → S3: {self.meta_uri}")
        self.df_updated_meta.write_parquet(
            self.meta_uri,
            storage_options=self.config.get_storage_options(),
            compression='zstd'
        )
        print(f"  ✓ S3 saved: {len(self.df_updated_meta):,} rows (Cost: $0.00 - ingress)")

        # Save meta to local cache
        print(f"  Meta → Local: {meta_cache_path}")
        self.df_updated_meta.write_parquet(meta_cache_path, compression='zstd')
        print(f"  ✓ Cached locally")

        print(f"\n  Local cache locations:")
        print(f"    - {vectors_cache_path}")
        print(f"    - {meta_cache_path}")
    
    def _print_summary(self):
        """Print final pipeline summary"""
        print(f"\n{'='*70}")
        print(f"✓ EMBEDDING PIPELINE COMPLETE")
        print(f"{'='*70}")
        print(f"  Mode: {self.config.embedding_mode}")
        print(f"  Sentences embedded: {len(self.df_new_vectors):,}")
        if self.skipped_ids:
            print(f"  Sentences skipped: {len(self.skipped_ids):,}")
        print(f"  Total vectors in storage: {len(self.df_merged_vectors):,}")
        print(f"  Total meta rows with embeddings: {self.df_updated_meta.filter(pl.col('embedding_id').is_not_null()).shape[0]:,}")
        print(f"  Embedding ID: {self.embedding_id}")
        print(f"  Total tokens: {self.total_tokens:,}")
        print(f"  Total cost: ${self.embedding_cost:.4f}")
        print("="*70)
    
    # ========================================================================
    # PUBLIC INTERFACE
    # ========================================================================
    
    def run(self):
        """Execute the full embedding generation pipeline"""
        print("="*70)
        print("EMBEDDING GENERATION PIPELINE")
        print("="*70)
        print(f"Mode: {self.config.embedding_mode}")
        print(f"Model: {self.config.bedrock_model_id} ({self.config.bedrock_dimensions}d)")
        print(f"\n[Resolved Paths]")
        print(f"  Vectors: {self.vectors_s3_key}")
        print(f"  Meta: {self.meta_s3_key}")
        
        # Step 1: Load meta table
        self.df_meta = self._load_meta_table_with_cache()
        
        # Step 2: Filter sentences
        self.df_filtered, self.filtered_ids = self._filter_sentences()
        
        # Step 3: Generate embeddings
        self.df_new_vectors, self.embedding_id, self.skipped_ids = self._generate_embeddings_batch()
        
        # Step 4: Merge vectors table
        self.df_merged_vectors = self._merge_vectors_table()
        
        # Step 5: Update meta table
        self.df_updated_meta = self._update_meta_table()
        
        # Step 6: Save results
        self._save_results()
        
        # Print summary
        self._print_summary()
        
        # Return summary dict
        return {
            'mode': self.config.embedding_mode,
            'embedding_id': self.embedding_id,
            'sentences_embedded': len(self.df_new_vectors),
            'sentences_skipped': len(self.skipped_ids),
            'total_vectors_stored': len(self.df_merged_vectors),
            'total_meta_embedded': self.df_updated_meta.filter(pl.col('embedding_id').is_not_null()).shape[0],
            'total_tokens': self.total_tokens,
            'total_cost': self.embedding_cost
        }