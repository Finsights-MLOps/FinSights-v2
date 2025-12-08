# FinRAG: Financial Retrieval-Augmented Generation System

---

## Overview
FinRAG is out attempt at a production-grade financial intelligence platform that processes SEC 10-K filings through a RAG architecture. The system combines traditional data engineering, communication with financial and sentence fact tables with modern semantic search to have an LLM-synthesized answer. We find it particularly suited for complex, narrative and factoid financial queries.

**Core Innovation**: Hybrid retrieval system fusing structured KPI extraction - communication, multi-context assembly, narrative semantic search with meaningful variant-diversity queries, window-expansion on hop context, and excellent sets of adapters or extractors - that grab complex multi-year, multi-company, multi-KPI, multi-section financial patterns through natural language.

---

## Philosophy & Design Principles
- Choose best cloud production over Local-Prototypes: **AWS infrastructure (S3 storage, S3 Vectors, Bedrock, Lambda)**. 
- Cost-conscious architecture: Cost-tracking analytical queries at multiple modules, and Cost-logging at every LLM call!
- Proper LLM validation: Sets of Automated Gold-Validation with Self@, Hit@, MMR@, Median distance and other concepts. Complete heuristic-NLP based approach for curation of a **Business-realistic Gold evaluation suite** (P3 Gold Standards)
- Validation at every layer: 5-stage S3 Vectors tests, staleness audits, parity checks, Chained-Integration test notebooks, Isolation tests.
- Real world data over toy datasets. And, tons of algorithmic strong features!

## (Some) Technical Highlights
- **Embedding Infrastructure**: 200K+ Cohere v4 1024-d vectors in S3 Vectors, <$0.10/10K queries
- **Hybrid Retrieval**: Semantic search (query variants + metadata filters) + structured KPI extraction
- **Context Engineering**: Edge-safe window expansion (±3 sentences), provenance tracking, chronological assembly
- **Evaluation Framework**: 31-question Gold P3 suite spanning factoid → multi-hop reasoning, Self, Hit@k, MRR, distance metrics; then, LLM-Eval (BERTScore, BLEURT, ROUGE-L, Cosine). 
- **NLP Front-End**: Fuzzy company resolution, multi-year parsing, risk topic classification, metric canonicalization

---

## System Capabilities

**Query Examples**:
- *"How did Meta's regulatory risk profile evolve from FCPA (2012) to GDPR (2018) to EU AI Act (2020)?"*  
  → Cross-year trend synthesis and citation headers

---

## Repository Navigation

### Start easily here: 
1. **[Quick Start with Docker! (RECOMMENDED)](finrag_docker_loc_tg1/LOC_DOCKER_README.md)** → Local Docker-based setup for easy launch of backend + frontend.
2. **[Quick Start with Command/Ps1 Scripts](SETUP_README.md)** → Get up and running quickly
  - Do setup with just click-install, and Run batch/sh files - launches streamlit.

### **Documentation**
- **[ARCHITECTURE.md](finrag_ml_tg1/ARCHITECTURE.md)** → Directory structure + pipeline flow diagrams
- **[IMPLEMENTATION_GUIDE.md](finrag_ml_tg1/IMPLEMENTATION_GUIDE.md)** → Detailed technical implementation (Parts 1-10)
- **[S3Vect_QueryCost.md](finrag_ml_tg1/S3Vect_QueryCost.md)** → Cost analysis and optimization strategies
- **[LLMOPS_TECHNICAL_COMPLIANCE.md](finrag_ml_tg1/LLMOPS_TECHNICAL_COMPLIANCE.md)** → LLM engineering standards, Test suites, LLM usage compliance, and best practices

### **Code Organization**
- **`platform_core_notebooks/`** → Data lifecycle management (embedding generation, S3 provisioning, gold test curation)
- **`rag_modules_src/`** → Production RAG components (entity extraction, retrieval, synthesis)

### **Key Artifacts**
- **Config**: `ml_config.yaml`, `.aws_secrets/` (credentials), `environments/` (specs)
- **Outputs**: `rag_modules_src\exports` (proper final exports), `data_cache/` (intermediate results), `test_outputs/` (some validation artifacts). 

---

## Quick Start
1. Our current quick demo is by viewing the exports, and the final-serve notebook: `ModelPipeline\finrag_ml_tg1\rag_modules_src\01_Isolation_Test_NBS\08_ITest_Start_To_LLM_Serve.ipynb`
2. We plan on clean integration with a front-end soon.


## Quick Screenshots/Demo:
1. As a short example, we have the following screenshots or attachments below. One would display a custom rendering of DOM object inside JupyterServe so that you have a very well formatted, pretty display. We currently have a notebook where we serve display tables of a couple of gold test queries. 
2. Secondly, we have two further screenshots: that would be query wise token cost analytics table and model wise token cost analytics table. 
3. If vector data embedding, or sentence-data are missing from dataset, we setup proper guardrails to acknowledge missing data and not hallucinate! 


**Image 1**: Gold Test Query Serve Example -
<p align="center">
  <img src="finrag_ml_tg1\demo_images_export\G1PQ002_JCell.png" width="700" alt="Gold Test Query Serve Example">
</p>
<p align="center"><em>Gold Test Query Serve Example</em></p>

**Image 3**: Query Wise Token Cost Analytics -
<p align="center">
  <img src="finrag_ml_tg1\demo_images_export\QueryWise_TokenCostAnalytics.png" width="700" alt="Query Wise Token Cost Analytics">
</p>
<p align="center"><em>Query Wise Token Cost Analytics</em></p>

**Image 4**: Model Wise Token Cost Analytics -
<p align="center">
  <img src="finrag_ml_tg1\demo_images_export\ModelWise_TokenCostAnalytics.png" width="700" alt="Model Wise Token Cost Analytics">
</p>
<p align="center"><em>Model Wise Token Cost Analytics</em></p>
---

## Contact 
**Author**: Joel Markapudi  
**Institution**: Northeastern University | CS AI Masters | IE7374 MLOps 

