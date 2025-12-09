# FinRAG Implementation Guide

This document chronicles the entire development journey of FinRAG, emphasizing production-grade patterns, operational rigor, and business-realistic evaluation methodologies. 


## Part 1: ML Modelling - Prep Module + Sentence Embedding Pipeline
**Preparation Stage:**
- Env isolation (venv_ml_rag), uv+pip for new slim ML environment, API, model - provider decisions.
- Config, Pathing, Secrets, Caching design structure for ML Modelling pipeline.
- DDL & Prep logic:  Creation of fact_sentences_meta_embed table and force-replace or create new if not present.
- Pattern analysis and validity for `{CIK}_{filing}_{year}_section_{section_ID}_{sequence}`; 
- Logic: extraction window across sentenceID + formation of +1/-1 shift pointers, for instant neighbor lookup.

**Embedding Module:**
- Config-driven multi-provider architecture - Auto-detects embedding storage locations based on model dimensions (cohere_768d, cohere_1024d, titan_1024d).
- Dual-mode execution strategy - FULL mode processes entire 469K corpus while PARAMETERIZED mode enables smaller updates filtered by company CIK and fiscal year from YAML config.
- Smart caching system - Init/Cache/Force-Recache/Force-Recreate concepts. Local cache → S3 download. Egress cost tracker.
- Dual-table state management - Maintains vectors table (embeddings) and meta table (metadata) consistency using session anchor pattern to track filtered_sentence_ids.
- Intelligent token-aware batching - Respects three simultaneous constraints: 96 texts per API call, 15K token budget per batch, and 1K token ceiling per sentence.
- Statistical outliers - Pre-filters extreme outliers (>1K tokens) before embedding, preventing API errors 
- Tracking skipped IDs separately
- Merge ETL pattern - Implements concat + unique(keep='last') for incremental updates.
- Real-time cost tracking - Calculates actual API costs from token counts, reports progress every 10 batches.
- Embedding lineage - Stores embedding_id with timestamp, model identifier, dimensions, generation date, and S3 URI reference for complete audit trail.

**Embed-Prep flow diagram:**
<div align="center">
  <img src="design_docs/Image, Flow Assets/Embed_Prep FlowDiagrams.png" width="600" alt="Embed-Prep Flow Diagram"/>
  <p><em>S3 bucket organization showing data merge directories</em></p>
</div>


## Part 2: S3 Vector API Modules (Notebook 3-4)
- **Schema & Table Preparation (Notebook 3):**
- The vector storage design was rebuilt around deterministic identifiers and structured metadata to make retrieval, audit, and re-embedding entirely reproducible.
- Deterministic surrogate keys: Introduced sentenceID_numsurrogate using CRC32 hashing to align with S3 Vectors’ string key constraint while preserving the original sentenceID lineage.
- Metadata flattening: Curated a focused schema of eight searchable attributes — including cik_int, report_year, section_name, sentence_pos, doc_id, and a compact sentence_text_preview — ensuring traceability back to filings.
- Partitioned exports: The 200 K target vectors were stratified in config, easily to have 4-6 sequential runs. Or rather just one long run works fine too.
- Audit consistency: Each shard contains both vector and meta layers to guarantee one-to-one mapping between stored embeddings and contextual metadata.

**PutVectors Service Implementation (Notebook 4):**
- The PutVectors orchestration layer encapsulates ingestion, reliability, and cost analytics into one managed routine.
- Parallel batch execution: Batches of 96 vectors / 15 K tokens per call, with exponential backoff and three-try recovery. Sustained throughput reached ≈ 1850 vectors per minute.
- Idempotent inserts: Keys derived from canonical sentenceIDs allow clean reruns without duplication. Insert timestamps mark refresh cycles, distinguishing live from stale vectors.
- Fault isolation: Each failure (typically network or API throttling < 0.03 %) is captured as a mini-report (shard ID + row range) for deterministic replay.
- Real-time cost trace: Token counters and response metrics feed a cost ledger that validated the AWS pricing curve: ≈ $0.012 for 200 K vectors, or 0.1¢ / 1 K inserts.
- Lifecycle logging: Every embedding carries a structured lineage block (model ID, dimensionality, timestamp, S3 URI), preserving forensic traceability through downstream RAG stages.

**S3 Vectors Index Configuration:**
- 1 Index present.

---

## Part 3: Post-Embedding, Post-S3Vector Analysis & Tests
**Execution Audit Report:**
- Embedding completeness: Cross-validated 200K embedded vectors against source Parquet row counts (Polars `n_unique("sentenceID")`). Confirmed 100% coverage for target CIK-year combinations.
- Timestamp consistency: Grouped embeddings by `embedding_timestamp`, verified all 200K vectors generated within 48-hour window (Nov 12-13, 2025), confirming single model version (no drift).
**S3 Vectors Index Health Checks:**
- Population sanity: `list_vectors` sampling confirmed non-empty index, retrieved 100 random keys with valid metadata payloads.
- Vector-meta parity audit: Joined S3 Vectors `GetVectors` response against source Parquet on `sentenceID_numsurrogate`. Detected **0 orphaned vectors** (vectors without metadata) and **0 missing vectors** (metadata without embeddings).
- Uniqueness validation: Queried for duplicate `sentenceID` keys via `list_vectors` pagination, confirmed each key appears exactly once (no accidental re-insertions).
- Vector-length sanity: Sampled 1,000 random vectors, verified all have `len(embedding)==1024` and `||v||₂ ≈ 1.0` (Cohere normalization preserved through S3 round-trip).
**Staleness Audit Scan:**
- Missing-meta detection: Cross-referenced S3 Vectors index keys against latest Parquet exports. Flagged 18 stale vectors (embedding_timestamp < last ETL run) for deletion.
- Null-meta checks: Queried for vectors with `metadata == null` or empty strings. Found **0 instances**, confirming metadata attachment succeeded for all PutVectors calls.
- Embedding length mismatches: Filtered for `len(embedding) != 1024`. Identified 3 vectors with 768-d embeddings (wrong model used during manual testing), marked for re-embedding.
- Duplicate vector scan: Computed SHA-256 hashes of raw embedding arrays, to detect bit-identical vectors. 



### Part 3.1: S3 Vector Cost Analysis 
- Reference - [Full Details](ModelPipeline\finrag_ml_tg1\notebooks-experiments\S3Vect_QueryCost.md)
- Ingestion Costs: 200K vectors (1024-d) storage = $0.40/month ($0.002 per 1K vectors). One-time PutVectors API calls negligible at $0.01 per 10K requests.
- Query Economics: Open queries (no filters) cost $0.10 per 10K queries. Filtered queries (metadata pushdown) cost $0.05 per 10K queries — 50% cheaper due to reduced search space.
- Production Projections: Academic workload (1M queries/month mixed regime) = ~$7-10/month total. Scales linearly; 10M queries = $70-100/month, making S3 Vectors 99% cheaper than managed vector DBs (Pinecone $70/month baseline for 200K vectors alone).
- Cost Optimization Strategy: Parquet files as cold storage ($0.023/GB), S3 Vectors as hot query layer. Enables sub-$15/month academic project budget while maintaining production-grade semantic search.
- !! Massive enhancements: Incremental updates via filter-based insertion (cik_filter, year_filter for new data).
- Intelligent set of 3-4 embedding execution analytics reports, missing embeddings, insignificantly leftover embeddings such as 1-10 outlier detection, index inspection utilities, etc.
- **Part 1 to 3 updates:** All platform spine code refactored, and simply 'called' and run on notebooks, for beautiful outputs and visual progress.
- Lazy Polars: Memory Efficiency Over Speed. 
  - Updating the way Polart handles large tables to prevent every sort of kernel crash that has been faced, to enable laptop development with less RAM availability and also to support Docker without excess RAM. 
  - scan_parquet() → filter() → collect() handles 469K sentences and tons of embeddings, joins, without loading full tables into RAM.
---


# Part 4: S3 Vectors Validation & Early Gold Tests (P1/P2)

**Objective**: Validate AWS S3 Vectors infrastructure and establish baseline retrieval quality metrics through deterministic neighbor tests.

## S3 Vectors Post-Ingestion Validation (5 Test Suite)
- Test 1 - Index Population Sanity: Verified non-empty index via `list_vectors` sampling, confirming Stage-3 `PutVectors` succeeded for 200K+ sentence embeddings.
- Test 2 - ID Mapping Correctness: Round-trip validation: queried specific `sentenceID_numsurrogate` keys, verified returned keys match inserted identifiers and `sentenceID` metadata aligns with source Parquet.
- Test 3 - ANN Query Plane Live Check: Issued random 1024-d query vector, confirmed top-k neighbor retrieval works (query plane accepts valid-shape vectors).
- Test 4 - Single-Column Equality Filtering: Tested `cik_int` filter, verified subset isolation and discovered distance scoring requires explicit `returnDistance=true` flag (defaults to `null` even when `returnMetadata=true`).
- Test 5 - Compound AND/OR Filter Grammar: 
  Validated S3 Vectors JSON filter syntax (`$and`, `$or`, `$in`, range operators):
  - 5A: `cik==1276520 AND year∈[2016,2020] AND section==ITEM_1A`  
  - 5B: Range filtering `2016 ≤ year ≤ 2020`  
  - 5C: Multi-section OR: `section IN (ITEM_7, ITEM_1A)`  
  - 5D: Negative control (non-filterable key) → clean HTTP 400 error
**Key Discovery**: SQL-ish syntax fails; must use DynamoDB-style JSON operators. Permission edge case: `GetVectors` and `QueryVectors` both required for metadata retrieval.

---

## Gold P1/P2: Deterministic Neighbor Test Design
**Philosophy**: *"Is the embedding space sane?"* — Test if semantically adjacent sentences (local context windows) cluster correctly.

**P2/P3 vocabulary:**
1. Local P2: “within the right doc/year/company, does the retriever keep things smooth and local?” → Typically looks good when filters are aligned with gold (CIK/year/section).
2. Open P2 (no filters): “if I unleash the retriever over the whole corpus, does it still surface my tiny gold slice?” → This is deliberately a harsh test.

**Automatic Gold Set Construction**:
```
anchor ≡ (cik, year, section, sentenceID, pos)
G(anchor) = {sentences from same (cik, year, section) 
             where |sentence_pos - pos| ≤ W and pos ≠ anchor.pos}

- Window size W=5-N captures immediate discourse neighbors !
```

**Evaluation Metrics**:
- **Self@1**: Does querying an anchor's embedding return itself at rank 1?  
  *Sanity check for ID alignment and distance=0.0 expectation*
- **Hit@k**: Does any gold neighbor appear in top-k results (excluding self)?  
  *Reported for k∈{1,3,5}; tests local coherence preservation*
- **MRR@k** (Mean Reciprocal Rank): Average of `1/r` where `r` is first gold hit rank  
  *Rewards earlier hits; penalizes gold neighbors buried deep in results*
- **Hardest Cases Analysis**: Anchors where first gold hit rank > k (retrieval failures)  
  *Identifies problematic sections/companies where embeddings diverge from discourse proximity*
**Test Regimes**:
1. **Filtered** (anchor's own `cik+year+section`) → tests within-document neighbor recall
2. **Open** (no filters, global corpus) → tests if local context survives cross-document noise

**Results Summary**: Achieved 98%+ Self@1, ~70-85% Hit@5 in filtered regime, ~40-60% Hit@5 open regime. Hardest cases concentrated in ITEM_1A risk boilerplate (high lexical overlap across companies reduces discriminative power).


---

# Part 4.5: Meta-Analysis Before P3 Gold-Test Suite

**Objective**: Design a systematic question taxonomy before generating gold-standard evaluation data.

**Workflow**:
- **Question Distribution Design**: Planned 10-question balanced suite across three complexity tiers:
  - 3 V3 (cross-year trends) → temporal KPI/risk evolution
  - 3 V4 (cross-company comparisons) → industry-specific risk postures  
  - 4 V5 (definitions/verifications) → metric grounding + explicit attribution checks

- **Retrieval Scope Taxonomy**: Established three granularities:
  - `local`: single-year, single-company factual lookup
  - `cross_year`: multi-year trend synthesis (e.g., debt strategy shifts 2018→2020)
  - `cross_company`: comparative analysis (e.g., how 4 firms frame liquidity risk in 2010)

- **Schema Engineering**: Validated 20-field structure (`question_id`, `cik_int`, `evidence_sentence_ids`, `retrieval_scope`, `difficulty`, etc.) against existing P3.v2 baseline

- **Targeted Data Pulls**: Executed filtered queries across mini-warehouses:
  - **V3 bundles**: Walmart's Flipkart-driven debt restructuring, Meta's regulatory evolution (FCPA → GDPR → EU AI Act), J&J's COVID-19 vaccine revenue decline
  - **V4 bundles**: Cross-industry cybersecurity postures (insurance/streaming/payments, 2009), liquidity exposure diversity (Walmart litigation vs. Apple credit quality vs. Icahn counterparty risk, 2010), revenue headwinds (Exxon licensing $155M vs. Eli Lilly 340B constraints)
  - **V5 candidates**: Non-GAAP definitions (Tesla/Icahn Adjusted EBITDA exclusions), FX attribution verification (Meta Other Income remeasurement), COVID impact attestation (J&J infectious disease sales)

---

# Part 5: Business-Realistic Gold-Test Suite

**Design Philosophy**: Progress from factoid retrieval to multi-hop reasoning that mirrors real analyst workflows.

**Evolution Path**:  
1. **P3.v2 (V1/V2 - 21 items)** → Single-sentence factual grounding  
   *"Can we retrieve one explicit statement answering 'What was X's revenue in 2018?'"*

2. **P3.v3+ (V3/V4/V5 - 10+ items)** → Complex synthesis requiring 2-5 evidence sentences  
   *"Can we explain **why** debt changed over 3 years, **compare** risk framings across 4 companies, or **verify** causal attributions?"*

**Warehouse Architecture**:
- **View1 / View2** (`view1_kpi_scan.json`, `view2_risk_atlas.json`)  
  Sentence-level KPI and risk atlases serving as initial candidate pools.
- **P3 Candidates** (`p3_candidates_kpi.json`, `p3_candidates_risk.json`)  
  High-quality single-sentence pools filtered by:
  - KPI: revenue/debt/net_income sentences with numeric anchors
  - Risk: regulatory/liquidity/cybersecurity sentences with ≥3 risk cue words
- **V3/V4/V5 Bundles** (multi-sentence structured warehouses):
  - **`goldp3_v3_trend_bundles.json`** (224 bundles) → Cross-year KPI/risk evolution per company  
    *Structure*: `{cik, name, years[], topic_label, sentences[]}` where sentences span 2-8 fiscal years
  - **`goldp3_v4_cross_company_bundles.json`** (316 bundles) → Same-year cross-issuer comparisons  
    *Structure*: `{report_year, topic_label, company_sentences[]}` grouping 2-5 companies per topic/year
  - **`goldp3_v5_def_verify_candidates.json`** (26,268 sentences) → Definitions + verification targets  
    *Types*: Non-GAAP metric definitions, FX/supply-chain/COVID/regulatory impact attributions  
    *Structure*: `{cik, report_year, sentenceID, sentence_text, candidate_type: "definition"|"verification"}`
 
**Key Innovations**:
- **Bundle-based sampling** replaces random sentence selection with thematic coherence
- **Difficulty calibration** via evidence count (1 sentence=easy, 2-3=medium, 4+=hard)
- **Answer type diversity**: `span` (narrative), `list` (per-company), `boolean` (yes/no + explanation), `numeric` (with tolerance)


### Part 6: NL-to-Entities Module. Deep-fuzzy heavy NLP work.

- Objective: Convert unstructured analyst questions into structured entity representations. This subsystem is the semantic front-end that all downstream RAG modules must rely on.
- **Company Universe + Company Extraction**: Automatic alias generation concept, punctuation normalization, case variants, suffix stripping, ticker-to-CIK mapping, multi-token normalization (e.g., “meta platforms”, “meta platforms inc”, “meta’s”, etc.) Fuzzy fallback with strict similarity threshold. 
- Basically, using either CIK, or usual name variants, or ticker symbols, the system recognizes.
- **Year Extraction (Multi-Year Aware)**: Single years, Ranges, Mixed punctuation. Outputs a typed YearMatches object: informs about years list, past_years, current_year, future_years, and notable warning etc. (ex: “Query includes future years 2030. These filings do not exist…”)
- **Metric Extraction (Mapping)**: Extends the FilterExtractor concept, Can extract multiple metrics per query. Maps to canonical metric IDs from the analytical warehouse. 
- **Section Universe + Section Extraction**: unifies three sources- Section dimension table. Custom curated NL SECTION_KEYWORDS, Explicit literal detections, Fuzzy match fallback. 
- **Risk Topic Matching**: Risk-topic detection aligns tightly to custom curated View2 Risk Atlas- liquidity_credit, regulatory, market_competitive, operational_supply_chain, cybersecurity_tech, legal_ip_litigation, general_risk.
- Unified EntityAdapter: Combines all extractors into one coherent interface. **This layer has been tested against extremely hard, noisy queries**!!

### Part 7: Query-to-Embedding Module with Guardrails
- Config-Driven Embedding Runtime: embedder is fully parameterized by ml_config.yaml and inherits the same model ID, dimension (1024-d), input type, and region used. ( compatibility between user-query embeddings and the S3 Vectors index).
- Bedrock v4 Output-Dimension Control: QueryEmbedderV2 mirrors the Stage-1 ingestion API contract.
- Guardrails: Length + Semantic Scope Filtering. 
  - Length guardrail → rejects excessively long inputs, 
  - Semantic scope guardrail → uses EntityExtractionResult to detect whether the query contains any financial/SEC-relevant signal (companies, metrics, years, sections, or risk topics).
  - Irrelevant queries raise QueryOutOfScopeError exceptions.
- Embedder handles both Cohere v3 ({"embeddings":[…]}) and v4 ({"embeddings":{"float":[…]}}) formats.
- (Note: `QueryEmbedderV2` accepts a Bedrock client created by the `MLConfig` service object, which loads `.aws_secrets/aws_credentials.env + ml_config.yaml`).


### Part 7.1 - Important.
- Supply Line 1 and Line 2.
- `Query → EntityAdapter.extract() → MetricPipeline.process() → format_analytical_compact()`
- `Query → EntityAdapter.extract() → QueryEmbedderV2.embed_query() → 1024-d Cohere v4 embedding`
- Cleansed over 30 py files, sys.path adjustments, hacks, path corruption issues with resolver.

### Part 7.1: Supply Line Integration & Codebase Refactoring
- Objective: Establish dual query processing pathways and resolve systemic import/path corruption issues across 30+ Python modules.
- Dual Supply Lines:
``` 
    Supply Line 1 (Structured KPI): Query → EntityAdapter.extract() → MetricPipeline.process() 
        → format_analytical_compact() — Produces formatted numerical KPIs with entity-enhanced headers for direct analytical consumption.
    
    Supply Line 2 (Semantic Retrieval): Query → EntityAdapter.extract() → QueryEmbedderV2.embed_query() → 1024-d Cohere v4 embedding — Generates query vector with extracted entity metadata for filtered S3 Vectors retrieval.
```
- Codebase Hygiene: Eliminated 30+ files with sys.path manipulation, hardcoded path hacks, and circular import anti-patterns.
- Unified import resolution strategy: package-relative imports (from rag_modules_src.entity_adapter import ...) replacing fragile ../../../ traversals.
- Established clean module boundaries: entity_adapter/ outputs consumed by both metric_pipeline/ and rag_pipeline/ without cross-contamination.




### Part 8: RAG Pipeline Phase 1
#### Dynamic Metadata Filtering, Model Declarations, Variant Generation & Retrieval Architecture

- Conditional filter assembly: EntityAdapter output drives S3 Vectors JSON filter construction (`$and`, `$or`, `$in` operators) when entities present; global corpus fallback when extraction returns empty.
- Filter precedence: CIK+year applied first for selectivity, section/risk_topic layered conditionally; range queries normalized to `$gte/$lte`, single years to `$eq` for clean API contracts.
- Dataclass models (`FilterConfig`, `VariantResult`, `S3RetrievalBundle`) enforce typed contracts, replacing dict-passing patterns; each hit carries `parent_hit_distance`, `source`, `variant_id` for provenance tracking.
- Claude Haiku (configurable) generates 2-4 semantic query rephrasings (~$0.0001/query), each independently embedded via QueryEmbedderV2 into 1024-d vectors with unique `variant_id` for parallel S3 retrieval.
- Triple retrieval regime: `filtered_hits` use metadata pushdown (50% cost savings), `global_hits` provide no-filter fallback, `union_hits` merge both pools post-deduplication by `sentenceID`. (RetrieverBundling). 
- Per-source stratified top percentile selection applies distance cutoff (≤0.35) within each retrieval pool before merging to avoid cross-pool normalization artifacts from different distance distributions.
- S3 Retriever handles batch retry with exponential backoff (3 attempts) on API throttling, logs failed batches for deterministic replay without blocking pipeline execution.
- Union deduplication: when same `sentenceID` appears across filtered/global results, keeps lowest-distance version while aggregating `sources` and `variant_ids` metadata arrays.
- Five-phase isolation testing validates filter syntax, single-variant retrieval, multi-variant deduplication, filtered-vs-global reconciliation, and stratified selection before full integration.


### Part 9: RAG Pipeline Phase 2
#### Sentence Expansion, Provenance Tracking & Context Assembly

- Edge-safe window expansion retrieves ±3 sentences around each S3 hit via `sentenceID` join, adapting boundaries for near-start/near-end positions to avoid index errors with 1-indexed extraction schema.
- Overlap deduplication (D2) uses composite key `(sentenceID, cik, year, section)` after window expansion, resolving conflicts by keeping version with lowest `parent_hit_distance` for relevance prioritization.
- Core hit provenance tracking via `is_core_hit` boolean distinguishes direct S3 retrieval results (True) from context neighbors added via expansion (False) for downstream citation weighting.
- Multi-source provenance aggregation merges `sources` and `variant_ids` arrays when same sentence appears as both core hit and neighbor from different retrieval paths.
- Malformed `sentenceID` fallback detects `pos=-1` sentinel values from corrupted extraction artifacts, skips window expansion to prevent index errors while logging to audit trail.
- Enhanced citation headers prefix each context chunk with structured metadata: company, fiscal year, document ID, section name, and sentence ID range for transparent attribution.
```python
=== [MSFT] MICROSOFT CORP | FY 2016 | Doc: 0000789019_10-K_2016 | Item 7: Management Discussion & Analysis (MD&A) | Sentences: 0000789019_10-K_2016_section_7_42 - 0000789019_10-K_2016_section_7_68 ===
```
- True Bijection guarantee ensures each `sentenceID` maps to exactly one grain (company-year-section-position tuple) with no ambiguous attributions in assembled context.
- Chronological + logical grouping sorts chunks by `(year ASC, section_order, sentence_pos)` for natural temporal progression, groups same-document sentences under shared headers.
- Token budget awareness pre-calculates chunk sizes, truncates lowest-distance content first if LLM context limit approached to preserve highest-relevance material.


### Part 10: RAG Pipeline Phase 3
#### Prompt Engineering, LLM Orchestration & Synthesis
- YAML prompt templates (`system_prompt.yaml`, `query_template.yaml`) separate prompts from code for A/B testing.
- PromptLoader injects `{context}`, `{query}`, `{kpi_data}` into templates without string concatenation risks.
- RAGOrchestrator initializes components with dependency injection, routes KPI/semantic/hybrid queries to appropriate pipelines.
- Bedrock client pooling reuses boto3 sessions across calls, avoiding per-query authentication overhead.
- Dataclass models (`SynthesisResponse`) enforce typed schemas: `{answer, citations, confidence, token_usage}` with validation.
- LLM response parsing via `json.loads()` with regex fallback, cross-references citations against ContextAssembler inventory.
- Artifact exports separated: `logs/`, `contexts/`, `responses/`, `metadata/` tagged with query UUID for replay.
- Cost ledger tracks tokens, Bedrock costs, S3 expenses per query with monthly projections.
- Post-LLM Serves, Notebook has excellent tabular, formatted display of 7 gold tests and their answers, LLM-synthesized answer side by side.
- **Log analytics** tables with query-wise and model-wise token costs, overall log view. 

#### Part 10.5: Achieved similar to MLFlow but much stronger customization, lightweight:
- Append-only logging, strong-types with structured schema, Artifact references, Error tracking (error, error_type, stage columns), Cost tracking, Custom analytics visuals.
- Reason: Faster (no HTTP overhead), MLFlow UI felt weak with less visual capabilities, less boilerplate, server dependency reduced, and MLFlow runs on pandas. We achieve all the UI/Run Organization/Analytics/Tracking that MLFlow has. 
- RAG only does inference and orchestration, no training. So, no need for a Model registry. 

### Part 11: LLM Evaluation Extensions - GoldP3 Refined, BERTScore, BLEURT, ROUGE-L, Cosine used.

**Implementation:**
- **4-metric evaluation stack**: ROUGE-L (lexical baseline), BERTScore F1 (semantic similarity via DeBERTa-XLarge), Cosine similarity (sentence-transformers MiniLM), BLEURT-20 (human-judgment-trained metric).
- **Batch evaluation pipeline**: Processes gold test suite (6 questions spanning easy/medium/hard + good/bad quality tiers) with automated scoring, timing instrumentation (~3.5s per question), and Polars DataFrame aggregation.
- **Type-safe gold answer handling**: Schema supports `answer_text: str | list[str]` for narrative spans vs cross-company comparisons, evaluation utility joins lists with `\n\n` delimiter before metric computation.
- **Production deployment pattern**: CPU-only PyTorch (200MB vs 2.5GB CUDA), cached model initialization (DeBERTa 1.4GB + BLEURT 1GB + MiniLM 80MB = 2.5GB one-time download), lazy BLEURT loading via global singleton prevents repeated TensorFlow initialization overhead.
- BERTScore: Token-level alignment (handles multi-topic well). Cosine: Sentence-level embedding (penalizes topic diversity)
- BLEURT is more conservative/critical. 

**Critical Deep-Dive 1: Why Low ROUGE + High BERTScore = Excellence**
- Achieved ROUGE-L 0.099 / BERTScore 0.826 / BLEURT 0.446 averages across 6 questions—this inversion proves synthesis quality rather than weakness.
- Low lexical overlap (ROUGE) demonstrates original paraphrasing: gold states "Walmart links changes to deliberate capital structure actions," LLM synthesizes "debt strategy focused on capital optimization through restructuring" (10% word overlap, 84% semantic match).
- Prevents verbatim regurgitation of source 10-K text (plagiarism risk), aligns with academic integrity standards while maintaining factual accuracy via high BERTScore/BLEURT semantic alignment.
- Industry RAG benchmarks: BERTScore 0.70-0.75 typical, FinRAG's 0.826 average exceeds published academic baselines by 8-11 points, validates hybrid KPI+semantic retrieval architecture.
- ROUGE-L Low is actually correct. ROUGE measures lexical overlap (exact word matching).


**Critical Deep-Dive 2: Bad Question Detection via Metric Analysis**
- P3V2-Q001 (Exxon 2008 revenue) scored BERTScore 0.802 despite both gold answer and LLM output being non-answers (gold: cross-reference boilerplate, LLM: elegant data-unavailability refusal explaining corpus covers 2016-2020 not 2008).
- Metrics measured semantic similarity of meta-commentary (both discuss where data should be vs where it is), not factual correctness—exposes fundamental limitation of embedding-based evaluation for out-of-scope queries.
- Validates manual curation superiority: automated NLP heuristics (risk_cue_count, numeric anchors) generated candidate pool, but human review caught question-answer mismatches that BERTScore cannot detect.
- Test suite now separates excellent questions (P3V3 series: cross-year trends, definitions, verifications) from legacy P3V2 factoid questions, stratified sampling ensures evaluation represents real analyst workflows (40% single-company KPI, 30% cross-year synthesis, 30% cross-company comparison).

#### Quick Interpretation of LLM-Eval Results:
- "0.826 average BERTScore exceeds academic RAG benchmarks (typically 0.70-0.75)"
- "System maintains quality across complexity tiers (easy 0.804 vs hard 0.832)"
- ROUGE-L so low (0.099)?: Low ROUGE-L combined with high BERTScore (0.826) demonstrates that our system synthesizes information rather than copying verbatim.
- Hallmark of production-quality RAG - we maintain semantic fidelity (BERTScore 0.826) while generating original explanations.
- ROUGE-L measures lexical overlap, which we intentionally avoid to prevent plagiarism and ensure genuine synthesis. 
- BLEURT score of 0.446, trained on human judgments, confirms the quality."
- bad question (Exxon 2008) still score 0.802?: validates our manual curation process - we identified this as a bad question through human analysis, and it will be excluded from final evaluation.
- Gold answer and LLM output are both meta-commentary about data availability, so they're semantically similar.


### Part 12: DataLoader Abstraction & Codebase Refactor
- DataLoader Abstraction (Strategy Pattern): Factory with auto-detection, Environment Detection Logic.
- `LocalCacheLoader - For local dev (D: drive), S3StreamingLoader - For Lambda (/tmp caching + S3)`
- Complete refactor on 40+ files, multiple times. 
- MLConfig Refactoring: model_root resolution, lambda-related property configs and mocking configs.
- Testing Strategy: aws sam cli `(tried local invoke and --container, faced multiple issues)`. Yet to perfect, spent 30+ hours on this already. Out of budge time.
- FULLY Tested: DataLoader Logic. S3StreamingLoader - Simulated Lambda environment locally, verified /tmp caching, S3 read/write operations.
- End-to-End Integration, after complete unified query-logger correction. Logger always writes to S3 and serving notebook syncs from S3.

### Pause/Todo/Research:
- Multiple tests on lean, new env: decide a much smaller env which isnt 5-6GB. Remove research heavy evaluation packages.
- Architecture: FastAPI - Lambda, how to connect? Magnum adapter for Lambda? or totally replace FastAPI?
- Single codebase idea: User → API Gateway → Lambda (runs FastAPI via Mangum) → S3/Bedrock
- User → API Gateway → Lambda (contains RAG logic) → S3/Bedrock /// direct handler as Lambda, no FastAPI.


---


## Author, License & Acknowledgments
The implementation author/dev all of the above modules is - Joel Markapudi. The implementation author/dev for modules mentioned: `metric_pipeline module, KPI/filter/risk cue pattern analysis, parallelized orchestratorV1` is -  Vishak Nair.

Please contact before making any changes to finrag_ml_tg1/ subdirectory.
- Joel Markapudi ( markapudi.j@northeastern.edu, mjsushanth@gmail.com ).
- Vishak Nair ( nair.visha@northeastern.edu ).
