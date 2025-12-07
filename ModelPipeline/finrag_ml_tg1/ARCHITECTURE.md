# FinRAG System Architecture

This document provides structural overviews of the codebase organization and data flow patterns. Later in the code files, please do pay attention to multiple `_contract.py` files which have excellent architectural flow diagrams, Data-Entity flows, Data-Responsibility understanding and more. 

Example: `ModelPipeline\finrag_ml_tg1\rag_modules_src\synthesis_pipeline\models.py`, `ModelPipeline\finrag_ml_tg1\rag_modules_src\rag_pipeline\models.py` etc.

## Project Directory Structure
### Parent Directory Structure Overview:

```
📦 finrag_ml_tg1/
 ┣ 📂 __pycache__/                              # Python bytecode cache (auto-generated)
 ┣ 📂 .aws_config/                              # AWS service configuration files
 ┣ 📂 .aws_secrets/                             # AWS credentials and sensitive keys (gitignored)
 ┣ 📂 data_cache/                               # Local data storage for intermediate processing results
 ┣ 📂 environments/                             # Conda/Python environment specifications and dependencies
 ┣ 📂 loaders/                                  # Data loading utilities and ETL ingestion modules
 ┣ 📂 platform_core_notebooks/                  # Core development notebooks for data lifecycle management
 ┣ 📂 rag_modules_src/                          # Production RAG pipeline components (query-time execution)
 ┣ 📂 venv_ml_rag/                              # Python virtual environment (local development)
 ┣ 📜 __init__.py                               # Python package initialization
 ┣ 📜 .python-version                           # Python version specification for project
 ┣ 📜 ML_Modelling_README.md                    # Main project documentation and architecture overview
 ┗ 📜 S3Vect_QueryCost.md                       # S3 Vector store cost analysis and projections

📦 finrag_ml_tg2/                               # [Secondary project workspace or experimental branch]

📦 serving/                               		# serving - backend FastAPI + pydantic, streamlit etc.
 ┣ 📂 frontend/                                # Frontend Streamlit application for user interaction
 ┣ 📂 backend/                                 # Backend FastAPI service for model inference and API endpoints


📜 .dvcignore                                   # DVC ignore patterns for data version control
📜 .gitignore                                   # Git ignore patterns for version control
```

### Embedding-Infra and Spines Overview:

```

📦 platform_core_notebooks/
 ┣ 📜 01_Stage2_EmbeddingGen.ipynb              # Stage 2 meta table creation + embedding generation pipeline
 ┣ 📜 02_EmbeddingAnalytics.ipynb               # Vector-metadata parity, staleness checks, integration audits
 ┣ 📜 03_S3Vector_TableProvisioning.ipynb       # S3 Vector store schema setup and initialization
 ┣ 📜 04_S3Vector_BulkIngestion.ipynb           # Mass vector insertion pipeline (200K+ vectors)
	
	# these are 'dormant', infra_setup_notebooks which are performed in low frequency, expected: twice a year.

 📦 validation_notebooks/
 ┣ 📜 05_GoldP1P2_TestSuite.ipynb               # Validation framework, anchor design, Gold P1/P2 methodology
 ┣ 📜 06_GoldP3_HeuristicEng_Curation.ipynb     # Query taxonomy, warehouse design, NLP-heuristic curation for Gold P3
 ┣ 📜 07_S3_CostProjections.ipynb               # Query cost modeling and operational expense analysis
 ┣ 📜 08_RAGArch_DesignNotes.ipynb              # RAG architecture decisions, technical rationale, design patterns
 ┣ 📜 09_RAG_Comp_ITests_01.ipynb               # Component-level tests for entity adapter and early integration
	
	# can be deal with as 'iterative development artifacts', for dev tests.
```

### RAG Modules Source Code Overview:
```
📦 rag_modules_src/
 ┣ 📂 01_Isolation_Test_NBS/                    # Isolated unit tests and component validation notebooks
 ┣ 📂 constants/                                # Project-wide constants, configurations, and static definitions
 ┣ 📂 entity_adapter/                           # Entity extraction and structured KPI data transformation logic
 ┣ 📂 exports/                                  # Output formatting, result serialization, and data export utilities
 ┣ 📂 metric_pipeline/                          # KPI extraction pipeline for structured financial metrics
 ┣ 📂 prompts/                                  # LLM prompt templates and instruction engineering modules
 ┣ 📂 rag_pipeline/                             # Core RAG orchestration: retrieval, reranking, context assembly
 ┣ 📂 synthesis_pipeline/                       # LLM response generation and answer synthesis logic
 ┣ 📂 test_outputs/                             # Test results, validation artifacts, and debugging outputs
 ┣ 📂 utilities/                                # Shared helper functions, logging, error handling, common tools
 ┗ 📜 __init__.py                               # Python package initialization for rag_modules_src
```

### Summary of Entity-Chaining and Flows:
**(Semantic Search + Context Assembly)**
```
User Raw Query:
	→ EntityAdapter / Extraction 
	→ QueryEmbedderV2 
	→ MetadataFilterBuilder 
	→ VariantPipeline (LLM rephrasings + re-embeds)
	→ S3VectorsRetriever (filtered + global regimes, plus variants)
	→ Post Retrieve - dedupe + per-source stratified top percentile selection.
	→ SentenceExpander (edge/window expansion + d2 overlap-dedup)
	→ Core Hit + Non-Core Neighbour Provenance Tracking, Provenance Aggregation
	→ ContextAssembler (sort + headered, chronological + logical grouping - based assembly)
	→ [ Returnable ] 
```

**Metric Extraction Pipeline**
```
User Raw Query:
	→ EntityAdapter / Extraction 
	→ Metric processor - extractor (Extended Vishak's and made V2 fix versions/reused logics.)
	→ Entity-Meta, Header Enhanced Util (analytical data formatting/assembly)
	→ [ SupplyLine / Wiring - Returnable. ]
```

**End-to-End Orchestration (Query → Answer Generation)**
```
User Raw Query
	→ RAGOrchestrator (initialize pipeline components)
	→ RAG Retrieval Pipeline (semantic context assembly)
	→ Metric Extraction Pipeline (structured KPI extraction)
	→ Context Merge (combine narrative + structured data)
	→ PromptLoader (load template + inject assembled context)
	→ BedrockClient (LLM inference with prompt)
	→ SynthesisPipeline (format response + add citations)
	→ [ Final Answer Returned to User ]
```


### Data Dependencies:
- Stage 2 Meta Table: finrag_fact_sentences_meta_embeds.parquet (~73MB)
- Stage 3 S3 Vectors Table: finrag_s3vectors_cohere_1024d.parquet (~360MB)
- Dimension Tables: Companies, Sections (small)
- Metric Data: downloaded_data.json (KPI lookup)

## Next Steps
→ See [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) for detailed technical implementation across 10 major development phases.


### Dual Environment Approach: (Potential)

- ENVIRONMENT 1: venv_ml_rag (FULL - Keep for Analytics)
```
  Location: finrag_ml_tg1/venv_ml_rag/
  Uses: environments/requirements.txt
  Size: ~2-3GB
  Purpose: Notebooks, eval metrics, torch, transformers
  When: Running platform_core_notebooks/, validation work
```

- ENVIRONMENT 2: venv_serving (MINIMAL - New for Deployment)
```
  Location: finrag_ml_tg1/venv_serving/  (NEW!)
  Uses: environments/requirements-sevalla.txt
  Size: ~500MB
  Purpose: Backend + Frontend serving ONLY
  When: Running start_finrag, testing deployment, Sevalla cloud
```


### Architecture Author:
Author: Joel Markapudi. ( markapudi.j@northeastern.edu, mjsushanth@gmail.com )