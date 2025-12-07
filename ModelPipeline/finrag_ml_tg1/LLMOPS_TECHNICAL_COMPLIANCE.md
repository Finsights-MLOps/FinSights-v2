# LLMOPS TECHNICAL COMPLIANCE

This document details the technical compliance of the FinRAG ModelPipeline with respect to MLOps/LLMOPS best practices, at our scale, according to the instructor's guidelines documents and guidance from class.

### 1. Executive Summary
- Brief overview: LLM/RAG vs traditional ML work, model and validation concepts.
- RAG validation requires retrieval-specific metrics (Hit@k, MRR) distinct from traditional supervised ML validation (accuracy/F1).

### 2. RAG System Architecture 
- Infrastructure: AWS S3 (storage), S3 Vectors (retrieval), Bedrock (LLM inference).
- Modular design: `rag_modules_src/` contains isolated components with dataclass-based contracts, detailed in [ARCHITECTURE.md](ARCHITECTURE.md).
- Platform core notebooks handle one-time embedding setup, incremental merges, S3 Vector ingestion, Table updates, Re-cache strategies, and API-PutVector operations.
- Config-driven parameterization: 200+ settings via MLConfig service + YAML templates.

### 3. Data Pipeline Integration 
- Three-stage loading: Raw SEC filings → Stage 1 sentence extraction → Stage 2 metadata enrichment → Stage 3 embedding generation.
- Cloud-native storage: All artifacts versioned in S3 with deterministic paths, loaded via Polars/DuckDB.
- Smart caching: Local `.parquet` mirrors for development, **S3 authoritative source for production**.
- Evidence: Notebooks 01-04 demonstrate complete data loading, validation, and S3 synchronization.

### 4. Validation Framework & Infrastructure Testing
1. Three-tier validation: Infrastructure tests → Gold P1/P2 → Business P3 warehouses.
2. Retrieval metrics (Self@1, Hit@k, MRR, distance) replace accuracy/F1 scores.
3. Five-phase S3 Vectors test suite validates index population, ID mapping, filters.
4. Component isolation and chained-integration tests in `01_Isolation_Test_NBS/` validate modules.
5. Vector-metadata parity audits: 0 orphaned vectors, 0 missing embeddings, 100% bijection.

### 4.1. LLM-Eval Validation Extension (BERTScore, BLEURT, ROUGE-L, Cosine)
1. BERTScore (0.826) and Cosine (0.675) are strong, ROUGE-L (0.099) .
```
Average BERTScore F1: 0.826  ← EXCELLENT
Average BLEURT: 0.446        ← GOOD
Average ROUGE-L: 0.099       ← LOW (but expected, makes perfect sense for 'genuine synthesis')
Average Cosine: 0.675        ← GOOD
```
2. Full deepdive explanation at: [LLM-Evaluation Metrics Deepdive](IMPLEMENTATION_GUIDE.md#L284).
3. Roughly, 0.826 average BERTScore exceeds academic benchmarks, maintains quality across complexity tiers, maintain semantic fidelity (BERTScore 0.826) while generating original explanations, BLEURT score of 0.446, trained on human judgments, confirms the quality.


### 5. Sensitivity Analysis (Hyperparameter/Feature Sensitivity)
1. **Window size sensitivity (W=2, 3, 5 vs 7)**: Comparative anchor tests show Hit@5 stability (65.0%→60.0%, -5% expected variance from broader sampling)—validates ±3 sentence expansion as best default without over-tuning. Large windows cause immediate noise and dilution of core context.
2. **Regime sensitivity (filtered vs open hits)**: Filtered queries achieve 82% Hit@5 vs 61% open regime, but open regime captures diverse context.
   - Core insight here: **Global hit discovery**: Despite low aggregate Hit@k (0.5-1% of union), global hits provide irreplaceable cross-document diversity— we did manual analysis of 34-hit union bundles. Here, approximately global-only hits surfaced 12-23% novel contextual evidence absent from entity-filtered results. 
3. **Distance threshold calibration**: Median non-self distances cluster 0.50-0.60 across regimes, establishing empirical acceptance threshold — S3 Vectors intrinsic quality prevents low-confidence pollution.
4. **TopK sensitivity (20 vs 30)**: MRR@20 (0.292) vs MRR@30 (0.275) shows diminishing returns beyond k=20 for filtered regimes.
- **Evidence**: Notebooks - HLD has records, screenshots of W=3/W=5 experiments
  
### 6. Bias Detection, Mitigation, Slice-Based Bias
#### Keywords: (deterministic gold sets, multi-tier testing, slice-based bias detection) 
1. FinRAG implements bias detection across multiple architectural layers. More than traditional demographic slicing.
2. We address retrieval-specific bias patterns. Bias mitigation strategy operates at three levels: **embedding space calibration, retrieval architecture design, and evaluation framework construction.** 
   - Distance threshold analysis across 10+ anchor tests revealed S3 Vectors maintains consistently high-quality retrievals—all top-30 results exhibit cosine distances >0.50~0.60, no low-confidence artifacts.
   - No poor-quality matches disproportionately affect underrepresented document types.
3. Avoiding Generative/Conditioning bias problems: Dual-supply-line architecture (structured KPI extraction + semantic RAG) prevents conditioning bias where generative models over-rely on narrative context while ignoring numerical grounding. 
4. Query variant generation via LLM rephrasings (2-4 semantic variants per query) reduces lexical-overlap bias—ensuring retrieval success doesn't depend on exact keyword matches that favor templated language.
5. **Filter regime analysis** quantifies this effect: open (global) retrieval achieves 60% Self@1 compared to 96.7% in filtered regimes. (cross-document boilerplate dominates nearest-neighbor space without metadata constraints.)
6. **The Gold P3 warehouse construction**: Slice-based bias detection through multi-dim json warehouses. 
   - V1/V2 factoid vs V3/V4/V5 multi-hop views.
   - V3 trend bundles (224 bundles spanning 2-8 fiscal years per company) expose temporal bias in risk/KPI language evolution; 
   - V4 cross-company bundles (316 bundles grouping 2-5 firms per topic/year) surface sector-specific lexical patterns that could disadvantage smaller companies with less standardized disclosure language. 
   - Heuristic NLP filtering and manual curation: identifying bias-prone content slices and creating bias-aware evaluation data. 
   - Section-size stratification in Gold P2 revealed that retrieval performance degrades predictably.
7. Quantitative slice analysis: We have metrics across **section length buckets** (<10, 10-19, 20-39, 40+ sentences), **fiscal year buckets** (2006-2010 vs 2016-2020), and filter regimes (**filtered 82% Hit@5 vs open 61% Hit@5**) documented in Notebook 05 anchor test results.
8. Triple retrieval-path concept too mitigates bias: core-hits from filtered (entity-constrained), global (no-filter fallback), and non-core neighbor expansions (±3 window context) ensure no single retrieval strategy dominates.
9. (Update!) - Test suite now separates excellent questions (P3V3 series: cross-year trends, definitions, verifications) from legacy P3V2 factoid questions, stratified sampling ensures evaluation represents real analyst workflows (40% single-company KPI, 30% cross-year synthesis, 30% cross-company comparison).
 
### 7. Continuous Bias Detection Strategy
1. **Monthly Gold P1/P2 Anchor Regression Tests**: Re-run 60-anchor deterministic neighbor tests (W=5, filtered+open regimes) on production S3 Vectors index to detect embedding drift, distance distribution shifts, or metadata filter degradation—track Self@1, Hit@5, MRR@30 stability across monthly snapshots with ±5% tolerance thresholds.
2. **Quarterly Gold P3 End-to-End Pipeline Tests**: Execute full 31-question Gold P3 suite (V1/V2 factoid + V3/V4/V5 multi-hop) through complete RAG pipeline every 2-3 months or post-embedding refresh. 
   - Track Hit@5, answer accuracy, and citation faithfulness across question difficulty tiers, with potential MMR (Maximal Marginal Relevance). 
3. **Ad-Hoc Filter Regime Validation**: Trigger filtered-vs-open retrieval comparison tests whenever EntityAdapter logic changes or new metadata fields added. Or, periodically validate when new data flows in, new queries arise.
- Evidence: `Notebook 06, V3/V4 bundle construction, rag_modules_src`.. etc.

### 8. Model Selection: Flexible/Tuned in configs
- **No singular, scalar preset.**
- We have multiple potential parameters that keep getting tuned over the entire pipeline architecture; So it's not just one model selected. Our project does not hold true to the annual philosophy of training and checkpoints or a singular scalable model. Rather, it's an integration of multiple linear and complex components. 
- For example, we use `AWS Bedrock Models`, `S3 vectors` with custom index, `cohere embed v3/4, amazon titan embeddings`. We use `Claude Haiku` or `Sonnet` during variant generation phase and LLM synthesis phase.

### 9. Experiment tracking, Cost, Token, Result, Context, Artifact - tracking.
- Please read [Result Tracking Deepdive](IMPLEMENTATION_GUIDE.md#L279) Part 10.5 for full details. 
- We achieve Experiment tracking, Parameter logging, Metrics tracking, Artifact storage, Error logging, Analytics - custom logging with full control and zero bloat.  

### 10. Other Considerations from Guidelines document
- Training/ReTraining concepts: Not really applicable, this is a RAG system using served LLMs and vector stores.
- Artifact Registry: Translates for us as S3 Vectors - our place for production vector bucket, data tables, access.

### Key System Design Concepts matched (with best possible attempts):
   - Three-tier SOA matches AWS SageMaker, Google Vertex AI production inference patterns.
   - Pydantic contract: auto-generate API docs; 
   - HTTP boundary enables N frontends → 1 backend without code coupling.
   - Stateless backend: can scale horizontally.
   - FastAPI process independence: frontend crash never kills ML pipeline execution.
   - Frontend 60x lighter (50MB vs 3GB) - deploys to serverless, backend needs GPU.
   - Separate environments: frontend updates cost $0 compute; unified reruns 800 packages
   - Parquet-based tracking: $0.02/month vs MLFlow server $50/month - Savings. 
   - Test HTTP client independently; monolith requires mocking entire Streamlit runtime.
   - ( Keywords: Dependency Inversion Principles / Contract-Driven / Solid Patterns )

### Ground-Up Python Orchestration vs Framework Abstractions
- We realize RAG chains or Frameworks like LangChain are powerful: Ecosystem with pre-built connectors for 50+ vector databases and LLM providers.
- **FinSights** Uses Primitive Python Orchestration:
  - Every AWS API call is explicit—you see token counts, model invocations, and S3 operations
  - Debug why a query costs $0.50 vs $0.02 by tracing exact function calls
  - Orchestration modules are pure Python, no hidden framework layers. Architectural Control is high.
  - EntityAdapter, MetricPipeline, S3VectorsRetriever are pure functions with deterministic behavior
  - Stack traces point to YOUR code, not framework internals 500 lines deep!
  - We also thought, eliminate framework abstraction overhead for latency-sensitive serving
  - Direct AWS SDK calls avoid unnecessary serialization/deserialization layers
  - Evaluators (ROUGE-L, BERTScore, BLEURT-20) operate on raw model outputs.



### 11. Key Decisions for Design-Choices:
1. Monorepo Pattern / Services with Frontend - Backend - ML Code.
   - Frontend is the ONLY consumer of this specific backend API, monorepo structure.
   - Serving layer requires access to ModelPipeline/ as the sys.path root for orchestrator imports.
   - Serving/ outside: breaks structure matches the established import pattern across 40+ files in the codebase.
   - Prefer avoiding PYTHONPATH, sys.resolve[] based hacks. 
   - Using fixed root + walk up, absolute imports pattern for all code.
2. Atomic Deployment Unit:
   - Usually, the model + serving layer + UI constitute a single deployment artifact. Followed the similar pattern here.
   - Google's TensorFlow Serving: Model + API in same repo, Hugging Face Spaces: Model + Gradio UI in same directory.
   - Orchestrator expects model_root parameter - serving must be relative to it.
1. Technical Justification for tracking approach:
   - Embedded Instrumentation > Separate Tracking. 
   - no discrepancies between 'experiment code' and 'production code'.
   - MLFlow: persistent infrastructure ($50-100/month for managed hosting). For a system with sub-$0.017 query costs, adding $50/month overhead is economically unjustifiable.
   - no model training - it's inference serving algorithms and components.
   - synthesis_pipeline/ logs every query with metadata (tokens, cost, latency) - this IS the tracking, just following serverless patterns.
2. Environments in Specific Locations:
   - Each environment lives with its deployment unit.
   - finrag_ml_tg1/ - one backend environment for all ML code + FastAPI wrapper.
   - `BACKEND_ENV = "finrag_ml_tg1/venv_ml_rag"  # Must be here for imports to work`
   - `FRONTEND_ENV = "serving/frontend/venv_frontend"  # Must be here for serving isolation`
   - Update! `BACKEND_END` now becomes te minimal `venv_serving` within 0.3GB for 'ML APP' serving.
   - The proper `venv_ml_rag` can be used for heavy torch libs, debugging, analytics and dev work.
3. Why Multiple Environments:
   - Boundary Enforcement, Basic Deployment Independence, Dependency Conflict Avoidance, Build Time Efficiency.
   - Makes no sense for Web-Dev team UI/UX to install Data Engineering team's libraries. Same goes for ML research team.
   - Example:
     - `venv_ml_rag:      3GB`   - boto3, transformers, sentence-transformers, polars, bluert, torch etc.
     - `venv_frontend:    50MB`  - streamlit, requests, pandas (minimal)
     - !! updated. now, we have 3 environments. `venv_serving` within 0.3GB for 'ML APP' serving, as we dont want to burden the consumers to have all the libs which developers, researchers use!
4. Update! Logics around single-client instantiations and caching, making microservices architecture 'cleaner', making sure backend, frontend, base_urls have clear separations and configurability; such new refactors and work has been done recently.


### P.S.: About minimal tech debt:
- 95% of modules: adapters, rag modules, configs loader modules, data loaders/streamers, orchestrator, synthesis modules, etc. and lightweight logging: are perfectly modular.
- In one module, conceptually, 'Retrieval backend spines' (`platform_core_notebooks`), I recognize some of the notebook logics aren't fully modularized python scripts - that's my tech debt, but with my current budget hours and with every other modules' work, research volume, across many weeks and months, I couldn't handle finishing it. There are only 2-4 core logics to be modularized.
- For now, the notebooks are well-structured with clear exec flows, documented, produce high-quality visual outputs when executed. They have helpful embedding-audit execution history, they have analytic queries and tabular logs quickly after doing the rag's necessary, time-taking, cloud operations - PutVectors or Generate embeds operations. These are only dormant 'infra setup' notebooks, they are not iteratively run for every query.
- The scientific coding notebook approach is still very valuable here. They are very cleanly explained in markdown cells.
- These are also not run for every query, they are 'Preparation' infrastructure and not daily 'Serving' infrastructure. My philosophy is that, these are handled once a year, twice a year or on-demand.
- 'Automated gold tests' Anchors, and Gold-Suite with scores are also in notebook, with clean, visual appeal - coding. These havent been automated for 'every' run because these are expensive. They run 60+, or 100+ independent semantic searches depending on anchor sizes. Further, next sets of Gold P3 tests call LLM synthesis for final evaluation answer. They are expensive too.


#### Author's Retrospection:
- Building FinRAG ModelPipeline required 11-13K+ lines of code across data engineering, embedding infrastructure, retrieval architecture, and validation frameworks. And countless adhoc analysis queries.
- I hope this work exceeds traditional requirements or expectations; especially through these aspects of deterministic gold set generation, curation, multi-dimensional bias detection, business-realistic evaluation frameworks, and— achievements that emerged from months of experimentation, cloud-research, ML system-design study, and algorithmic refinement.

#### Compliance Write-up Author
Joel Markapudi - ( markapudi.j@northeastern.edu, mjsushanth@gmail.com )
- Please contact me for any questions.
---
