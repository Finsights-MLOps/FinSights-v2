"""
═══════════════════════════════════════════════════════════════════════════════
                    PLATFORM CORE INFRASTRUCTURE CONTRACT
                   (Embedding Generation & Vector Storage Setup)
═══════════════════════════════════════════════════════════════════════════════

MODULE PURPOSE:
    One-time infrastructure setup for sentence-level embedding generation,
    validation, and S3 Vectors index provisioning. These are MANUAL operations
    run sequentially by developers during system initialization or data updates.
    
    Jupyter notebooks are just for clean visual orchestration - the actual files
    are in platform_core/ as reusable Python modules. They are all perfectly 
    isolated, and modular. 

    This is NOT a 'real-time' pipeline 
    - these are heavy batch operations (30min - 2hrs each sometimes)
    just coordinated through Jupyter notebooks with checkpoint/caching between stages.

    For someone who's just *reusing* the final S3 Vectors index for RAG queries,
    they wouldnt need to re-run platform_core/ at all.

DESIGN PHILOSOPHY:
    • Manual orchestration via notebooks (not automated pipeline)
    • Centralized config (MLConfig) for path/credential coordination
    • Local caching for resumability (S3 authoritative, local mirrors)
    • Independent utilities (can be run/tested in isolation)
    • Heavy operations separated by checkpoints (not chained)

═══════════════════════════════════════════════════════════════════════════════


┌─────────────────────────────────────────────────────────────────────────────┐
│                         INFRASTRUCTURE ARCHITECTURE                         │
│                         (Manual Multi-Stage Setup)                          │
└─────────────────────────────────────────────────────────────────────────────┘

                        ╔═══════════════════════╗
                        ║   MLConfig Service    ║  Centralized Configuration
                        ║   (loaders/)          ║  • Paths, credentials, S3 URIs
                        ╚═══════════╤═══════════╝  • Dimensions, providers
                                    │              • Region, bucket names
                    ┌───────────────┼───────────────┐
                    ↓               ↓               ↓
        ┌───────────────┐  ┌──────────────┐  ┌──────────────┐
        │  Stage 1 Data │  │  AWS Bedrock │  │  S3 Vectors  │
        │  (S3 Bucket)  │  │  (Embeddings)│  │   (Index)    │
        └───────┬───────┘  └──────┬───────┘  └──────┬───────┘
                │                 │                  │
                └─────────────────┴──────────────────┘
                            EXTERNAL DEPENDENCIES


═══════════════════════════════════════════════════════════════════════════════
                            MANUAL ORCHESTRATION FLOW
                          (Notebooks = Human Decision Points)
═══════════════════════════════════════════════════════════════════════════════

┌──────────────────────────────────────────────────────────────────────────────┐
│ NOTEBOOK 01: Stage2_EmbeddingGen.ipynb                                       │
│ ────────────────────────────────────────────────────────────────────────────│
│ Human decisions:                                                              │
│   • Which companies/years to embed (YAML config)                             │
│   • FULL vs PARAMETERIZED mode                                               │
│   • Force recache vs use existing                                            │
│                                                                               │
│ Modules used:                                                                 │
│   ├─ data_preparation.py           (Cache Stage 1 → Stage 2 meta)           │
│   └─ embedding_generation.py       (Batch embed with Bedrock)               │
│                                                                               │
│ Runtime: ~50 minutes (21 companies, 5 years)                                 │
│ Output: Stage 2 Meta + Embeddings (local cache + S3)                         │
└──────────────────────────────────────────────────────────────────────────────┘
                    ┊ CHECKPOINT: Review embedding stats
                    ┊ Developer validates before proceeding
                    ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│ NOTEBOOK 02: EmbeddingAnalytics.ipynb                                        │
│ ────────────────────────────────────────────────────────────────────────────│
│ Human decisions:                                                              │
│   • Inspect execution history                                                │
│   • Validate vector-metadata parity                                          │
│   • Check for gaps or staleness                                              │
│                                                                               │
│ Modules used:                                                                 │
│   └─ embedding_audit.py            (Lazy Polars analytics)                  │
│                                                                               │
│ Runtime: ~30 seconds (analytics queries)                                     │
│ Output: Validation reports, gap analysis                                     │
└──────────────────────────────────────────────────────────────────────────────┘
                    ┊ CHECKPOINT: Decide if ready for Stage 3
                    ┊ Fix gaps if needed, re-run embeddings
                    ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│ NOTEBOOK 03: S3Vector_TableProvisioning.ipynb                                │
│ ────────────────────────────────────────────────────────────────────────────│
│ Human decisions:                                                              │
│   • Validate config paths                                                    │
│   • Build Stage 3 table (join meta + vectors)                                │
│   • Upload to S3 (force overwrite?)                                          │
│                                                                               │
│ Modules used:                                                                 │
│   ├─ stage3_config_validation.py   (Verify S3 paths)                        │
│   ├─ s3vectors_table_preparation.py (Build Stage 3 with lazy Polars)        │
│   └─ s3vectors_index_inspector.py  (Validate index schema)                  │
│                                                                               │
│ Runtime: ~2-5 minutes (table transformation)                                 │
│ Output: Stage 3 S3 Vectors table (with numeric surrogates)                   │
└──────────────────────────────────────────────────────────────────────────────┘
                    ┊ CHECKPOINT: Validate Stage 3 schema
                    ┊ Check dimension, row count, surrogates
                    ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│ NOTEBOOK 04: S3Vector_BulkIngestion.ipynb                                    │
│ ────────────────────────────────────────────────────────────────────────────│
│ Human decisions:                                                              │
│   • Full insertion vs incremental (CIK/year filters)                         │
│   • Batch size tuning (AWS max = 500)                                        │
│   • Monitor progress, handle throttling                                      │
│                                                                               │
│ Modules used:                                                                 │
│   └─ s3vectors_bulk_insertion.py   (Filtered insertion with retry)          │
│                                                                               │
│ Runtime: ~50 minutes (407K vectors) or ~5 minutes (filtered subset)          │
│ Output: Vectors inserted into S3 Vectors index (queryable)                   │
└──────────────────────────────────────────────────────────────────────────────┘
                    ┊ CHECKPOINT: System ready for queries
                    ┊ Move to RAG pipeline (rag_modules_src/)


═══════════════════════════════════════════════════════════════════════════════
                        CRITICAL DESIGN DISTINCTIONS
═══════════════════════════════════════════════════════════════════════════════

❌ NOT A DETERMINISTIC PIPELINE:
   Unlike rag_pipeline/ where data flows through modules in milliseconds,
   platform_core operations are:
   
   • HOURS-LONG: Embedding generation = 50 min, insertion = 50 min
   • IO-BOUND: S3 downloads, Bedrock API calls, S3 Vectors uploads
   • MEMORY-HEAVY: 407K vectors × 1024d = ~2GB processing
   • MANUALLY TRIGGERED: Developer runs notebooks one-by-one
   • CHECKPOINT-DRIVEN: Each stage outputs to cache/S3 before next

✅ ACTUAL DESIGN PATTERN:
   
   NOTEBOOKS (Jupyter)          MODULES (Python)             STATE (Cached)
   ═════════════════            ═══════════════════          ══════════════
   
   01_Stage2_Embed ────calls──> data_preparation.py ────→ Stage1.parquet
         ↓                       embedding_generation.py   Stage2_meta.parquet
         ↓ (human checkpoint)                              Stage2_vectors.parquet
         ↓
   02_EmbedAnalytics ──calls──> embedding_audit.py ─────→ Gap reports
         ↓                                                  (no new state)
         ↓ (human checkpoint)
         ↓
   03_S3VectorTable ───calls──> s3vectors_table_prep.py ─→ Stage3.parquet
         ↓                       stage3_config_val.py       (S3 uploaded)
         ↓ (human checkpoint)    s3vectors_inspector.py
         ↓
   04_BulkIngestion ───calls──> s3vectors_bulk_insert.py ─→ S3 Vectors Index
         ↓                                                   (AWS managed)
   ✓ READY FOR RAG QUERIES


═══════════════════════════════════════════════════════════════════════════════
                             MODULE CONTRACTS
═══════════════════════════════════════════════════════════════════════════════

┌──────────────────────────────────────────────────────────────────────────────┐
│ data_preparation.py                                                           │
│ ────────────────────────────────────────────────────────────────────────────│
│ Functions:                                                                    │
│   • cache_stage1_table(config, force_recache) → DataFrame                   │
│   • cache_stage2_meta_table(config, force_recache) → DataFrame              │
│   • cache_embeddings_tables(config, providers, force_recache) → dict        │
│   • initialize_meta_table(config, df_stage1, force_reinit) → None           │
│   • initialize_vectors_table(config, force_reinit) → None                   │
│                                                                               │
│ Purpose: Stage 1→2 transformation, S3↔local caching coordination             │
│ Pattern: Lazy download (check cache first), transform, upload if init        │
│ State: Creates/updates Stage 2 meta + empty vectors tables on S3             │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│ embedding_generation.py                                                       │
│ ────────────────────────────────────────────────────────────────────────────│
│ Class: EmbeddingPipeline                                                      │
│   __init__(config, mode, filters)                                            │
│   run() → summary_dict                                                        │
│                                                                               │
│ Purpose: Batch embedding generation with AWS Bedrock (Cohere v4)             │
│ Pattern: Filter → Batch (96 texts, 15K tokens) → Embed → Merge → Upload     │
│ State: Updates Stage 2 meta (embedding_id, dims, date, ref)                  │
│        Creates Stage 2 vectors (sentenceID, embedding_id, embedding)         │
│ Runtime: ~50 minutes for 407K sentences (21 companies, 8 years)              │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│ embedding_audit.py                                                            │
│ ────────────────────────────────────────────────────────────────────────────│
│ Functions:                                                                    │
│   • audit_embedding_execution_history(config, provider) → None              │
│   • show_missing_embeddings(config, provider, gap_threshold) → None         │
│   • show_completely_unembedded_company_years(config, provider) → None       │
│                                                                               │
│ Purpose: Lazy Polars analytics on Stage 2 meta + vectors                     │
│ Pattern: scan_parquet() → filter() → join() → group_by() → collect()        │
│ State: Read-only analytics (no modifications)                                │
│ Runtime: ~10-30 seconds (depends on filter scope)                            │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│ s3vectors_table_preparation.py                                                │
│ ────────────────────────────────────────────────────────────────────────────│
│ Class: S3VectorsTablePipeline                                                 │
│   __init__(provider, build_table, upload_to_s3, force_overwrite)            │
│   run() → summary_dict                                                        │
│                                                                               │
│ Purpose: Join Stage 2 meta + vectors → Stage 3 with numeric surrogates       │
│ Pattern: Lazy load → join → add sentenceID_numsurrogate (CRC32) → validate  │
│ State: Creates Stage 3 S3 Vectors table (filterable + non-filterable meta)   │
│ Runtime: ~2-5 minutes (depends on Stage 2 size)                              │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│ s3vectors_bulk_insertion.py                                                   │
│ ────────────────────────────────────────────────────────────────────────────│
│ Class: S3VectorsBulkInserter                                                  │
│   __init__(provider, cik_filter, year_filter, batch_size, max_retries)      │
│   run() → summary_dict                                                        │
│                                                                               │
│ Purpose: Filtered batch insertion to S3 Vectors with retry logic             │
│ Pattern: Lazy filter → batch (500) → convert → retry (exp backoff) → insert │
│ State: Populates S3 Vectors index (AWS managed, queryable)                   │
│ Runtime: ~50 minutes (407K vectors full), ~5 minutes (filtered subset)       │
│ Incremental: Use cik_filter/year_filter for new data only                    │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│ s3vectors_index_inspector.py                                                  │
│ ────────────────────────────────────────────────────────────────────────────│
│ Functions:                                                                    │
│   • inspect_s3vectors_index(bucket, index, verbose) → dict                  │
│   • get_index_status(bucket, index) → dict                                   │
│                                                                               │
│ Purpose: Query S3 Vectors index configuration and status                     │
│ Pattern: get_index() API call → parse response → display/return             │
│ State: Read-only inspection (no modifications)                               │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│ UTILITIES (Validation & Config)                                              │
│ ────────────────────────────────────────────────────────────────────────────│
│ • sentID_pattern_validation.py → validate_sentence_id_pattern(sid)          │
│ • stage3_config_validation.py  → validate_s3vectors_config()                │
│                                                                               │
│ Purpose: Lightweight validation utilities (no heavy operations)              │
│ Pattern: Input validation, path checking, regex matching                     │
└──────────────────────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════════
                          CACHING & RESUMABILITY STRATEGY
═══════════════════════════════════════════════════════════════════════════════

LOCAL CACHE (ModelPipeline/finrag_ml_tg1/data_cache/)
├── stage1/                              # Stage 1 raw data (from S3)
│   └── sec_filings_stage1.parquet      # 71.8M sentences, 1.53GB
├── meta_embeds/                         # Stage 2 metadata
│   └── finrag_fact_sentences_meta_embeds.parquet  # 469K rows, ~73MB
├── embeddings/                          # Stage 2 vectors
│   └── cohere_1024d/
│       └── finrag_embeddings_cohere_1024d.parquet  # 407K embeddings, ~700MB
└── stage3_s3vectors/                    # Stage 3 S3 Vectors format
    └── cohere_1024d/
        └── finrag_embeddings_s3vectors_cohere_1024d.parquet  # ~700MB

S3 AUTHORITATIVE (s3://finrag-data-lake/)
├── stage1_fact_tables/                  # Source of truth
├── stage2_meta_tables/                  # Embedding metadata
├── stage2_vector_tables/                # Embedding vectors
└── stage3_s3vectors_tables/             # Ready for S3 Vectors upload

STRATEGY:
- Local cache = Development speed (avoid repeated S3 downloads)
- S3 = Production authority (team collaboration, deployment source)
- force_recache flag = Bypass local cache, fresh S3 download
- Notebooks check cache first, only download if missing or forced


═══════════════════════════════════════════════════════════════════════════════
                        FAILURE RECOVERY & RESUMABILITY
═══════════════════════════════════════════════════════════════════════════════

SCENARIO 1: Embedding Generation Crashes Mid-Run
────────────────────────────────────────────────────────────────────────────────
Problem: Kernel crash after 30 minutes, 150K/407K vectors embedded

Solution:
  1. Last successful batch merged to Stage 2 vectors (parquet)
  2. Re-run with force_recache=False (loads existing progress)
  3. Pipeline detects existing embeddings, skips them
  4. Continues from ~150K onwards

Pattern: Session anchor tracking + merge-on-append strategy


SCENARIO 2: S3 Vectors Insertion Fails at Batch 200/408
────────────────────────────────────────────────────────────────────────────────
Problem: AWS throttling or network error after 25 minutes

Solution:
  1. S3 Vectors upserts by key (sentenceID_numsurrogate)
  2. Re-run same parameters (same cik_filter, year_filter)
  3. Already-inserted vectors overwritten (idempotent)
  4. New vectors inserted from batch 200 onwards

Pattern: Upsert semantics + exponential backoff retry


SCENARIO 3: Incremental Update (New Year Added)
────────────────────────────────────────────────────────────────────────────────
Problem: Embedded [2012-2025], now need to add [2006-2011]

Solution:
  1. Run Notebook 01 with year_filter=[2006-2011] (PARAMETERIZED mode)
  2. New embeddings appended to Stage 2 vectors
  3. Run Notebook 03 to rebuild Stage 3 (includes old + new data)
  4. Run Notebook 04 with year_filter=[2006-2011] (filtered insertion)
  5. S3 Vectors now contains [2006-2025] (old vectors unchanged)

Pattern: Filter-based incremental insertion


═══════════════════════════════════════════════════════════════════════════════
                        COST & PERFORMANCE CHARACTERISTICS
═══════════════════════════════════════════════════════════════════════════════

EMBEDDING GENERATION (Notebook 01):
- API Cost: ~$20-40 per full corpus (407K sentences × Bedrock pricing)
- Time: ~50 minutes (batch size 96, 4,250 API calls)
- Memory: ~2-4GB peak (Polars in-memory operations)
- Network: ~10MB/s sustained (Bedrock API throughput)

S3 VECTORS INSERTION (Notebook 04):
- API Cost: ~$0.01 (S3 Vectors PutVectors is cheap)
- Time: ~50 minutes (407K vectors, 69 vectors/sec)
- Memory: ~1-2GB (Stage 3 loaded, filtered, batched)
- Network: ~5MB/s sustained (S3 Vectors API)

TOTAL SETUP TIME (Cold Start):
- Notebook 01: ~50 minutes
- Notebook 02: ~30 seconds
- Notebook 03: ~3 minutes
- Notebook 04: ~50 minutes
- TOTAL: ~105 minutes (~1.75 hours)


═══════════════════════════════════════════════════════════════════════════════
                            INTEGRATION POINTS
═══════════════════════════════════════════════════════════════════════════════

UPSTREAM (Consumes):
├─ Stage 1 Data Pipeline (Edgar-Sentences-SDK)
│  └─ Produces: sec_filings_stage1.parquet (71.8M sentences)
│
└─ MLConfig (loaders/ml_config_loader.py)
   └─ Provides: Paths, credentials, S3 URIs, dimensions

DOWNSTREAM (Produces For):
├─ RAG Pipeline (rag_modules_src/)
│  └─ Consumes: S3 Vectors index (queryable via S3VectorsRetriever)
│
└─ Validation Suite (validation_notebooks/)
   └─ Consumes: Stage 2 meta + vectors (for test suite evaluation)

EXTERNAL DEPENDENCIES:
├─ AWS Bedrock (Embedding generation)
├─ AWS S3 (Storage for Stage 1/2/3)
└─ AWS S3 Vectors (Managed ANN index for retrieval)


═══════════════════════════════════════════════════════════════════════════════
                            OPERATIONAL GUIDELINES
═══════════════════════════════════════════════════════════════════════════════

WHEN TO RUN FULL SETUP:
- Initial system setup (first time)
- Re-embedding entire corpus with new model (e.g., Cohere v4 → v5)
- Data corruption detected (force_reinit=True)

WHEN TO RUN INCREMENTAL:
- Adding new fiscal years (parameterized filters)
- Adding new companies (parameterized filters)
- Fixing gaps from failed embeddings (show_missing_embeddings)

MONITORING CHECKLIST:
1. After Notebook 01: Check embedding count, cost estimate, gaps
2. After Notebook 02: Validate vector-metadata parity (should be 100%)
3. After Notebook 03: Check Stage 3 row count, dimension validation
4. After Notebook 04: Verify S3 Vectors index count matches Stage 3

DEBUGGING TIPS:
- Kernel crashes → Use lazy Polars (scan_parquet, collect at end)
- AWS throttling → Reduce batch_size or increase max_retries
- Dimension mismatches → Check config.bedrock_dimensions matches index
- Missing vectors → Run embedding_audit.py to find gaps


═══════════════════════════════════════════════════════════════════════════════
                                    SUMMARY
═══════════════════════════════════════════════════════════════════════════════

platform_core/ is NOT a real-time pipeline but a MANUAL INFRASTRUCTURE SETUP
toolkit for batch embedding generation and vector storage provisioning.

Key characteristics:
- Heavy operations (30min - 2hrs each)
- Manual orchestration via Jupyter notebooks
- Checkpoint-driven with caching between stages
- Config-driven coordination (MLConfig)
- Resumable on failures (merge-on-append, upsert semantics)
- Incremental-capable (filter-based updates)

After completion, the S3 Vectors index is ready for real-time queries via
rag_modules_src/rag_pipeline/s3vectors_retriever.py (~100-300ms per query).

═══════════════════════════════════════════════════════════════════════════════

Last Updated: December 2025
Status: Production-ready (407K vectors, 21 companies, 8 years)
Maintainer: Joel Markapudi | IE7374 MLOps | Northeastern University
Project: FinRAG - Financial Document Intelligence System

═══════════════════════════════════════════════════════════════════════════════
"""
