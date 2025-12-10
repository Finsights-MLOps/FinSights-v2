# ModelPipeline Module Explanation

This document provides a comprehensive explanation of every module within the ModelPipeline system and how they work together.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Core Architecture](#core-architecture)
3. [Module Breakdown](#module-breakdown)
   - [Orchestrator (Entry Point)](#orchestrator-entry-point)
   - [Entity Adapter](#entity-adapter)
   - [Metric Pipeline](#metric-pipeline)
   - [RAG Pipeline](#rag-pipeline)
   - [Synthesis Pipeline](#synthesis-pipeline)
   - [Platform Core](#platform-core)
   - [Loaders](#loaders)
   - [Utilities](#utilities)
   - [Serving Layer](#serving-layer)
4. [Data Flow](#data-flow)
5. [Key Design Patterns](#key-design-patterns)

---

## System Overview

**ModelPipeline** is a production-grade Financial Retrieval-Augmented Generation (FinRAG) system that processes SEC 10-K filings to answer complex financial queries. It combines:

- **Structured KPI extraction** from financial metrics
- **Semantic search** using vector embeddings (200K+ Cohere v4 vectors in S3 Vectors)
- **LLM synthesis** via AWS Bedrock (Claude models)
- **Hybrid retrieval** fusing structured and narrative data

The system is designed for both **local development** and **AWS Lambda deployment** with environment-agnostic data loading.

---

## Core Architecture

```
User Query
    ↓
┌─────────────────────────────────────────────────────────┐
│  Orchestrator (orchestrator.py)                        │
│  - Entry point: answer_query()                          │
│  - Coordinates all components                           │
│  - Handles errors and logging                           │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│  Supply Lines (supply_lines.py)                        │
│  ├─ Supply Line 1: KPI Extraction                      │
│  └─ Supply Line 2: RAG Context Retrieval                │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│  Prompt System → Bedrock Client → Response              │
└─────────────────────────────────────────────────────────┘
```

---

## Module Breakdown

### Orchestrator (Entry Point)

**Location**: `rag_modules_src/synthesis_pipeline/orchestrator.py`

**Purpose**: Main entry point that orchestrates the entire query-answering pipeline.

**Key Function**: `answer_query(query, model_root, ...)`

**How it works**:
1. **Initialization**: Creates all components (config, RAG components, prompt loader, Bedrock client, logger)
2. **Context Building**: Calls `build_combined_context()` which runs both supply lines
3. **Prompt Formatting**: Wraps context in YAML prompt templates
4. **LLM Invocation**: Calls AWS Bedrock API via BedrockClient
5. **Response Packaging**: Creates typed response models with metadata
6. **Logging**: Persists query, context, response, and costs to Parquet files

**Key Features**:
- Error handling at every stage (returns ErrorResponse dicts, never raises)
- Processing time tracking
- Cost tracking (tokens, pricing)
- Export capabilities (context files, response files)

**Dependencies**:
- `supply_lines.py` - Context assembly
- `prompt_loader.py` - Prompt templates
- `bedrock_client.py` - LLM API wrapper
- `query_logger.py` - Persistent logging

---

### Entity Adapter

**Location**: `rag_modules_src/entity_adapter/`

**Purpose**: Extracts structured entities from natural language queries.

**Key Components**:

1. **EntityAdapter** (`entity_adapter.py`)
   - Main orchestrator for entity extraction
   - Calls all individual extractors
   - Returns `EntityExtractionResult` with companies, years, metrics, sections, risk topics

2. **CompanyExtractor** (`company_extractor.py`)
   - Fuzzy matching against company dimension table
   - Handles tickers (NVDA, AAPL), names (NVIDIA, Apple), CIKs
   - Uses `CompanyUniverse` for lookup

3. **YearExtractor** (`year_extractor.py`)
   - Extracts years from queries (2019, 2020-2023, etc.)
   - Categorizes as past/current/future
   - Provides warnings for future years

4. **MetricAdapter** (`metric_adapter.py`)
   - Maps business terms to canonical metric IDs
   - Uses `metric_mapping_v2.py` for keyword matching
   - Returns canonical IDs like `income_stmt_Revenue`

5. **SectionExtractor** (`section_extractor.py`)
   - Extracts SEC section references (ITEM_1A, ITEM_7, etc.)
   - Identifies risk topics (liquidity_credit, regulatory, etc.)
   - Uses `SectionUniverse` for lookup

**How it works**:
```
Query: "What were NVIDIA's AI risks in 2017-2020?"
    ↓
EntityAdapter.extract()
    ↓
├─ CompanyExtractor → {CIK: 1045810, ticker: NVDA, name: NVIDIA}
├─ YearExtractor → {years: [2017, 2018, 2019, 2020]}
├─ MetricAdapter → {metrics: []}  (no metrics detected)
├─ SectionExtractor → {sections: [ITEM_1A], risk_topics: [regulatory]}
└─ Returns EntityExtractionResult
```

**Data Dependencies**:
- Company dimension table (via DataLoader)
- Section dimension table (via DataLoader)
- Metric mapping constants

---

### Metric Pipeline

**Location**: `rag_modules_src/metric_pipeline/`

**Purpose**: Extracts structured financial KPIs (revenue, net income, etc.) from queries and looks them up in fact tables.

**Key Components**:

1. **MetricPipeline** (`src/pipeline.py`)
   - Main orchestrator
   - Determines if query needs metric layer (`needs_metric_layer()`)
   - Coordinates FilterExtractor and MetricLookup

2. **FilterExtractor** (`src/filter_extractor.py`)
   - Extracts filters from query (companies, years, metrics)
   - Uses EntityAdapter internally for company/year extraction
   - Handles fuzzy company matching

3. **MetricLookup** (`src/metric_lookup.py`)
   - Queries KPI fact table (Parquet file)
   - Filters by CIK, year, metric name
   - Returns structured data with values, units, evidence

**How it works**:
```
Query: "What was NVIDIA's revenue in 2021?"
    ↓
MetricPipeline.process()
    ↓
├─ needs_metric_layer() → True (has "revenue", year, company)
├─ FilterExtractor.extract() → {cik: 1045810, year: 2021, metric: "revenue"}
├─ MetricLookup.lookup() → Queries KPI fact table
└─ Returns: {value: 26914.0, unit: "millions", evidence: "10-K filing"}
```

**Data Dependencies**:
- KPI fact table (Parquet) - loaded via DataLoader
- Company dimension table (for fuzzy matching)

**Output Format**:
- Structured table with company, year, metric, value, unit
- Formatted as compact analytical block for LLM context

---

### RAG Pipeline

**Location**: `rag_modules_src/rag_pipeline/`

**Purpose**: Retrieves relevant narrative context from SEC filings using semantic search.

**Key Components**:

1. **QueryEmbedderV2** (`utilities/query_embedder_v2.py`)
   - Converts query to 1024-d embedding using AWS Bedrock (Cohere v4)
   - Handles query validation and error cases
   - Cost tracking (~$0.0001 per query)

2. **MetadataFilterBuilder** (`metadata_filters.py`)
   - Builds S3 Vectors filter JSON from entities
   - Creates two filter sets:
     - **Filtered**: Strict constraints (exact years, sections)
     - **Global**: Relaxed temporal constraints (year >= 2015)

3. **VariantPipeline** (`variant_pipeline.py`)
   - Generates query variants using LLM (Claude Haiku)
   - Rephrases query 3 times for semantic diversity
   - Re-embeds each variant
   - Cost: ~$0.0006 per query (3 variants)

4. **S3VectorsRetriever** (`s3_retriever.py`)
   - Main retrieval engine using AWS S3 Vectors API
   - **Strategy**:
     - Base query: Filtered call (topK=30) + Global call (topK=15)
     - Variant queries: Filtered calls only (topK=15 each)
   - **Deduplication**: By (sentence_id, embedding_id), keeps best distance
   - **Proportional sampling**: 70% filtered, 30% global (before expansion)
   - Returns `RetrievalBundle` with filtered_hits, global_hits, union_hits

5. **SentenceExpander** (`sentence_expander.py`)
   - **Window Expansion**: For each hit, expands ±3 sentences for context
   - Queries Stage 2 meta table by position range
   - **Deduplication**: By (sentence_id, cik, year, section) after expansion
   - Marks `is_core_hit` (True = actual S3 hit, False = neighbor)
   - Returns list of `SentenceRecord` objects

6. **ContextAssembler** (`context_assembler.py`)
   - Sorts sentences by (company, year DESC, section, doc, pos)
   - Groups by (company, year, section) for headers
   - Formats as:
     ```
     === NVIDIA CORP | 2020 | ITEM_1A ===
     sentence1
     sentence2
     
     === NVIDIA CORP | 2019 | ITEM_1A ===
     sentence3
     ```
   - Returns formatted string for LLM

**How it works** (Complete Flow):
```
Query: "What were NVIDIA's AI risks in 2017-2020?"
    ↓
1. EntityAdapter.extract() → {cik: 1045810, years: [2017-2020], section: ITEM_1A}
    ↓
2. QueryEmbedderV2.embed_query() → 1024-d vector
    ↓
3. MetadataFilterBuilder.build_filters() → Filtered + Global JSON
    ↓
4. VariantPipeline.generate_variants() → 3 rephrased queries + embeddings
    ↓
5. S3VectorsRetriever.retrieve()
   ├─ Base query: Filtered (topK=30) + Global (topK=15)
   ├─ Variants: 3 × Filtered (topK=15 each)
   ├─ Deduplication: 75 hits → 34 unique
   └─ Proportional sampling: 30 hits (21 filtered + 9 global)
    ↓
6. SentenceExpander.expand_and_deduplicate()
   ├─ Window expansion: 30 hits → ~210 sentences (±3 window)
   └─ Deduplication: ~210 → ~140 unique sentences
    ↓
7. ContextAssembler.assemble() → Formatted string with headers
```

**Data Dependencies**:
- S3 Vectors index (AWS managed, 200K+ vectors)
- Stage 2 meta table (for window expansion) - loaded via DataLoader

**Performance**:
- Retrieval: ~100-300ms per query
- Cost: <$0.10 per 10K queries (S3 Vectors pricing)

---

### Synthesis Pipeline

**Location**: `rag_modules_src/synthesis_pipeline/`

**Purpose**: Generates final LLM responses and manages synthesis workflow.

**Key Components**:

1. **Orchestrator** (`orchestrator.py`)
   - Main entry point (covered above)

2. **Supply Lines** (`supply_lines.py`)
   - **`init_rag_components()`**: Factory that creates all RAG components
   - **`run_supply_line_1_kpi()`**: KPI extraction chain
   - **`run_supply_line_2_rag()`**: RAG retrieval chain
   - **`build_combined_context()`**: Combines KPI + RAG blocks

3. **BedrockClient** (`bedrock_client.py`)
   - Wraps AWS Bedrock API calls
   - Handles Claude models (Sonnet, Opus, Haiku)
   - Tracks token usage and costs
   - Error handling and retries

4. **PromptLoader** (`prompts/prompt_loader.py`)
   - Loads YAML prompt templates
   - Formats system prompts and user prompts
   - Injects context and query

5. **QueryLogger** (`query_logger.py`)
   - Logs all queries to Parquet files
   - Tracks: query, answer, context, metadata, costs, timestamps
   - Supports cost analytics and query history

6. **Models** (`models.py`)
   - Typed response models:
     - `QueryResponse`: Success response with answer, context, metadata
     - `ErrorResponse`: Error response with error details
   - Factory functions: `create_success_response()`, `create_error_response()`

**How it works**:
```
User Query
    ↓
build_combined_context()
    ├─ Supply Line 1 (KPI) → "KPI SNAPSHOT" block
    └─ Supply Line 2 (RAG) → "NARRATIVE CONTEXT" block
    ↓
Combined: [KPI Block] + [RAG Block] + [USER QUESTION]
    ↓
PromptLoader.format_query_template() → Full prompt
    ↓
BedrockClient.invoke() → LLM response
    ↓
create_success_response() → Typed response
    ↓
QueryLogger.log_query() → Persist to Parquet
    ↓
Return dict
```

---

### Platform Core

**Location**: `platform_core/`

**Purpose**: **One-time infrastructure setup** for embedding generation and vector storage. These are **manual, batch operations** (not real-time).

**Key Components**:

1. **Data Preparation** (`data_preparation.py`)
   - Caches Stage 1 → Stage 2 meta table transformation
   - Coordinates S3 ↔ local caching
   - Creates/updates Stage 2 meta + empty vectors tables

2. **Embedding Generation** (`embedding_generation.py`)
   - Batch embedding generation using AWS Bedrock (Cohere v4)
   - Processes 407K sentences in ~50 minutes
   - Batch size: 96 texts, 15K tokens
   - Updates Stage 2 meta (embedding_id, dims, date, ref)
   - Creates Stage 2 vectors (sentenceID, embedding_id, embedding)

3. **Embedding Audit** (`embedding_audit.py`)
   - Lazy Polars analytics on Stage 2 meta + vectors
   - Validates vector-metadata parity
   - Finds gaps and staleness
   - Read-only (no modifications)

4. **S3 Vectors Table Preparation** (`s3vectors_table_preparation.py`)
   - Joins Stage 2 meta + vectors → Stage 3 format
   - Adds numeric surrogates (CRC32) for S3 Vectors
   - Validates schema and dimensions
   - Runtime: ~2-5 minutes

5. **S3 Vectors Bulk Insertion** (`s3vectors_bulk_insertion.py`)
   - Filtered batch insertion to S3 Vectors index
   - Batch size: 500 (AWS max)
   - Retry logic with exponential backoff
   - Runtime: ~50 minutes (407K vectors) or ~5 minutes (filtered)

6. **S3 Vectors Index Inspector** (`s3vectors_index_inspector.py`)
   - Queries S3 Vectors index configuration
   - Validates index status and schema
   - Read-only inspection

**Orchestration Flow** (Manual, via Jupyter Notebooks):
```
Notebook 01: Stage2_EmbeddingGen.ipynb
  ├─ data_preparation.py → Cache Stage 1 → Stage 2 meta
  └─ embedding_generation.py → Generate 407K embeddings
  Runtime: ~50 minutes
  ↓ [CHECKPOINT]
  
Notebook 02: EmbeddingAnalytics.ipynb
  └─ embedding_audit.py → Validate parity, find gaps
  Runtime: ~30 seconds
  ↓ [CHECKPOINT]
  
Notebook 03: S3Vector_TableProvisioning.ipynb
  ├─ stage3_config_validation.py → Validate paths
  ├─ s3vectors_table_preparation.py → Build Stage 3 table
  └─ s3vectors_index_inspector.py → Validate schema
  Runtime: ~2-5 minutes
  ↓ [CHECKPOINT]
  
Notebook 04: S3Vector_BulkIngestion.ipynb
  └─ s3vectors_bulk_insertion.py → Insert to S3 Vectors index
  Runtime: ~50 minutes
  ↓
✓ System ready for RAG queries
```

**Key Characteristics**:
- **NOT real-time**: Heavy batch operations (30min - 2hrs each)
- **Manual orchestration**: Developer runs notebooks one-by-one
- **Checkpoint-driven**: Each stage outputs to cache/S3 before next
- **Resumable**: Merge-on-append, upsert semantics for failures
- **Incremental-capable**: Filter-based updates for new data

**Data Flow**:
- **Stage 1**: Raw sentences from SEC filings (71.8M sentences, 1.53GB)
- **Stage 2**: Meta table + embeddings (407K embeddings, ~700MB)
- **Stage 3**: S3 Vectors format (with numeric surrogates)
- **S3 Vectors Index**: AWS managed, queryable via API

---

### Loaders

**Location**: `loaders/`

**Purpose**: Environment-agnostic data loading (works in both local dev and AWS Lambda).

**Key Components**:

1. **DataLoaderStrategy** (`data_loader_strategy.py`)
   - Abstract base class defining interface
   - Methods:
     - `load_stage2_meta()` - Stage 2 metadata table
     - `load_dimension_companies()` - Company dimension
     - `load_dimension_sections()` - Section dimension
     - `get_sentences_by_ids()` - Fetch specific sentences
     - `load_kpi_fact_data()` - KPI fact table

2. **LocalCacheLoader** (`data_loader_strategy.py`)
   - **Environment**: Local development
   - **Strategy**: Loads from `data_cache/` directory
   - **Caching**: In-memory cache (loads once, reuses)
   - **Performance**: Fast (local filesystem)

3. **S3StreamingLoader** (`data_loader_strategy.py`)
   - **Environment**: AWS Lambda
   - **Strategy**: Streams from S3, caches in `/tmp`
   - **Caching**: `/tmp` cache (Lambda ephemeral storage)
   - **Performance**: Slower (S3 API calls), but Lambda-compatible

4. **DataLoaderFactory** (`data_loader_factory.py`)
   - Auto-detects environment
   - Checks `config.data_loading_mode` ('LOCAL_CACHE' or 'S3_STREAMING')
   - Returns appropriate loader implementation

5. **MLConfigLoader** (`ml_config_loader.py`)
   - Loads `ml_config.yaml` configuration
   - Provides paths, credentials, S3 URIs, dimensions
   - Auto-detects ModelPipeline root directory

**How it works**:
```
Component needs data
    ↓
create_data_loader(config)  # Factory
    ↓
├─ If LOCAL_CACHE → LocalCacheLoader
│  └─ Loads from data_cache/ (in-memory cache)
│
└─ If S3_STREAMING → S3StreamingLoader
   └─ Streams from S3, caches in /tmp
    ↓
Component uses DataLoaderStrategy interface
    ↓
Same code works in both environments!
```

**Benefits**:
- **Environment-agnostic**: Same code works locally and in Lambda
- **Dependency injection**: Components receive DataLoader, don't know implementation
- **Testable**: Can mock DataLoader for unit tests

---

### Utilities

**Location**: `rag_modules_src/utilities/`

**Purpose**: Shared helper functions and utilities.

**Key Components**:

1. **QueryEmbedderV2** (`query_embedder_v2.py`)
   - Converts queries to embeddings (covered in RAG Pipeline)

2. **Supply Line Formatters** (`supply_line_formatters.py`)
   - Formats KPI data as compact analytical blocks
   - Formats RAG context with headers
   - Combines blocks for LLM context

3. **Sentence Utils** (`sentence_utils.py`)
   - Helper functions for sentence manipulation
   - Position calculations, ID parsing

4. **Response Cleaner** (`response_cleaner.py`)
   - Cleans LLM responses
   - Removes artifacts, normalizes formatting

5. **Evaluation Metrics** (`evaluation_metrics.py`)
   - ROUGE-L, BERTScore, BLEURT, Cosine similarity
   - Used for answer quality evaluation

6. **Notebook Display** (`notebook_display.py`)
   - Pretty-printing utilities for Jupyter notebooks
   - Formatted tables, visualizations

---

### Serving Layer

**Location**: `serving/`

**Purpose**: Three-tier service architecture (presentation, application, business logic).

**Key Components**:

1. **Backend** (`serving/backend/`)
   - **FastAPI** application
   - **api_service.py**: RESTful HTTP endpoints
     - `POST /query` - Process query
     - `GET /health` - Health check
     - `GET /stats` - Query statistics
   - **config.py**: Backend configuration
   - **models.py**: Pydantic request/response models

2. **Frontend** (`serving/frontend/`)
   - **Streamlit** application
   - **app.py**: Main Streamlit app
   - **chat.py**: Chat interface component
   - **api_client.py**: HTTP client for backend API
   - **state.py**: Session state management
   - **metrics.py**: Metrics display

**How it works**:
```
User → Streamlit Frontend
    ↓ (HTTP)
FastAPI Backend
    ↓ (Python call)
Orchestrator.answer_query()
    ↓
[Full RAG pipeline]
    ↓
Response → Backend → Frontend → User
```

---

## Data Flow

### Complete End-to-End Flow

```
1. User Query: "What were NVIDIA's AI risks in 2017-2020?"
    ↓
2. Orchestrator.answer_query()
    ↓
3. Supply Lines:
   ├─ Supply Line 1 (KPI):
   │  ├─ EntityAdapter.extract() → {cik: 1045810, years: [2017-2020]}
   │  ├─ MetricPipeline.process() → No metrics (query is narrative)
   │  └─ Returns: Empty KPI block
   │
   └─ Supply Line 2 (RAG):
      ├─ EntityAdapter.extract() → {cik: 1045810, years: [2017-2020], section: ITEM_1A}
      ├─ QueryEmbedderV2.embed_query() → 1024-d vector
      ├─ MetadataFilterBuilder.build_filters() → Filtered + Global JSON
      ├─ VariantPipeline.generate_variants() → 3 variants + embeddings
      ├─ S3VectorsRetriever.retrieve() → 30 hits (deduplicated)
      ├─ SentenceExpander.expand_and_deduplicate() → ~140 unique sentences
      └─ ContextAssembler.assemble() → Formatted context string
    ↓
4. build_combined_context() → [RAG Block] + [USER QUESTION]
    ↓
5. PromptLoader.format_query_template() → Full prompt
    ↓
6. BedrockClient.invoke() → LLM response
    ↓
7. create_success_response() → Typed response with metadata
    ↓
8. QueryLogger.log_query() → Persist to Parquet
    ↓
9. Return dict to user
```

### Data Dependencies

**Upstream Data Sources**:
- **Stage 1**: SEC filings sentences (from DataPipeline)
- **S3 Vectors Index**: 200K+ embeddings (from Platform Core)
- **KPI Fact Table**: Financial metrics (from DataPipeline)
- **Dimension Tables**: Companies, sections (from DataPipeline)

**Data Loading**:
- All data loaded via `DataLoaderStrategy` interface
- Environment-agnostic (local cache or S3 streaming)

---

## Key Design Patterns

### 1. **Strategy Pattern** (Data Loading)
- `DataLoaderStrategy` abstract base
- `LocalCacheLoader` and `S3StreamingLoader` implementations
- Components receive interface, not implementation

### 2. **Factory Pattern** (Component Initialization)
- `init_rag_components()` creates all RAG components
- `create_data_loader()` auto-detects environment
- Centralized initialization logic

### 3. **Supply Line Pattern** (Context Assembly)
- **Supply Line 1**: KPI extraction chain
- **Supply Line 2**: RAG retrieval chain
- Combined in `build_combined_context()`

### 4. **Deduplication Strategy**
- **Stage 1**: Hit-level dedup (by sentence_id + embedding_id)
- **Stage 2**: Sentence-level dedup (after window expansion)
- Keeps best distance, aggregates provenance

### 5. **Error Handling**
- All errors returned as `ErrorResponse` dicts
- Never raises exceptions (graceful degradation)
- Error tracking with `stage` field

### 6. **Cost Tracking**
- Token usage tracked at every LLM call
- Cost calculated using config pricing
- Logged to Parquet for analytics

### 7. **Checkpoint-Driven** (Platform Core)
- Each notebook stage outputs to cache/S3
- Resumable on failures
- Manual orchestration (not automated)

---

## Summary

**ModelPipeline** is a sophisticated RAG system with:

- **8 major modules**: Orchestrator, Entity Adapter, Metric Pipeline, RAG Pipeline, Synthesis Pipeline, Platform Core, Loaders, Serving
- **Hybrid retrieval**: Structured KPI + semantic search
- **Environment-agnostic**: Works locally and in Lambda
- **Production-ready**: Error handling, logging, cost tracking
- **Scalable**: 200K+ vectors, efficient retrieval

The system processes financial queries by:
1. Extracting entities (companies, years, metrics, sections)
2. Retrieving structured KPIs (if applicable)
3. Retrieving narrative context (semantic search)
4. Synthesizing answer via LLM
5. Returning formatted response with citations

All modules work together through well-defined interfaces and contracts, making the system maintainable and extensible.

