"""
run_supply_line_1_kpi(...) - KPI chain
run_supply_line_2_rag(...) - retrieval chain

Design three lean helpers:
    Supply Line 1 → KPI block.
    Supply Line 2 → RAG context block.
    A small “conductor” that calls both and concatenates the two strings.
"""

# ModelPipeline/finrag_ml_tg1/synthesis_pipeline/supply_lines.py

from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Tuple

from finrag_ml_tg1.rag_modules_src.entity_adapter.entity_adapter import EntityAdapter
from finrag_ml_tg1.rag_modules_src.metric_pipeline.src.pipeline import MetricPipeline
from finrag_ml_tg1.rag_modules_src.utilities.supply_line_formatters import (
    format_analytical_compact,
)
from finrag_ml_tg1.rag_modules_src.utilities.query_embedder_v2 import ( QueryEmbedderV2, EmbeddingRuntimeConfig )

from finrag_ml_tg1.rag_modules_src.rag_pipeline.metadata_filters import MetadataFilterBuilder
from finrag_ml_tg1.rag_modules_src.rag_pipeline.variant_pipeline import VariantPipeline
from finrag_ml_tg1.rag_modules_src.rag_pipeline.s3_retriever import (
    S3VectorsRetriever,
)
from finrag_ml_tg1.rag_modules_src.rag_pipeline.sentence_expander import (
    SentenceExpander,
)
from finrag_ml_tg1.rag_modules_src.rag_pipeline.context_assembler import (
    ContextAssembler,
)

from finrag_ml_tg1.loaders.ml_config_loader import MLConfig

# ──────────────────────────────────────────────────────────────────────────────
# Bundle of RAG components so callers don’t have to pass 7 args ertime orz.
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RAGComponents:
    adapter: EntityAdapter
    metric_pipeline: MetricPipeline
    embedder: QueryEmbedderV2
    filter_builder: MetadataFilterBuilder
    variant_pipeline: VariantPipeline
    retriever: S3VectorsRetriever
    expander: SentenceExpander
    assembler: ContextAssembler



def init_rag_components(model_root: Path) -> RAGComponents:
    """
    Convenience factory to build all core RAG components from the standard config.
    centralises the initialization that you previously did in the isolation
    notebooks (MLConfig, Bedrock client, dimensions, metric JSON, etc.).
    """
    # Global config & Bedrock client
    config = MLConfig()
    bedrock_client = config.get_bedrock_client()

    # Dimension paths
    dim_companies = model_root / "finrag_ml_tg1/data_cache/dimensions/finrag_dim_companies_21.parquet"
    dim_sections = model_root / "finrag_ml_tg1/data_cache/dimensions/finrag_dim_sec_sections.parquet"

    # Metric JSON path -- JSON outdated, use parquet.
    # metric_json = model_root / "finrag_ml_tg1/rag_modules_src/metric_pipeline/data/downloaded_data.json"
    METRIC_DATA_FACT = model_root / "finrag_ml_tg1/rag_modules_src/metric_pipeline/data/KPI_FACT_DATA_EDGAR.parquet" 

    # 1) Entity adapter
    adapter = EntityAdapter(company_dim_path=dim_companies, section_dim_path=dim_sections)

    # 2) Metric pipeline
    metric_pipeline = MetricPipeline(
        data_path=str(METRIC_DATA_FACT),
        company_dim_path=str(dim_companies),
    )

    # 3) Query embedder
    embedding_cfg = config.cfg["embedding"]
    runtime_cfg = EmbeddingRuntimeConfig.from_ml_config(embedding_cfg)
    embedder = QueryEmbedderV2(runtime_cfg, boto_client=bedrock_client)

    # 4) Filter builder
    filter_builder = MetadataFilterBuilder(config)

    # 5) Variant pipeline
    variant_pipeline = VariantPipeline(config, adapter, embedder, bedrock_client)

    # 6) S3 retriever
    retrieval_cfg = config.get_retrieval_config()
    retriever = S3VectorsRetriever(
        retrieval_config=retrieval_cfg,
        aws_access_key_id=config.aws_access_key,
        aws_secret_access_key=config.aws_secret_key,
        region=config.region,
        variant_pipeline=variant_pipeline,
    )

    # 7) Sentence expander
    expander = SentenceExpander(config)

    # 8) Context assembler
    assembler = ContextAssembler(config)

    return RAGComponents(
        adapter=adapter,
        metric_pipeline=metric_pipeline,
        embedder=embedder,
        filter_builder=filter_builder,
        variant_pipeline=variant_pipeline,
        retriever=retriever,
        expander=expander,
        assembler=assembler,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Supply Line 1: KPI side
# ──────────────────────────────────────────────────────────────────────────────

def run_supply_line_1_kpi(
    query: str,
    rag: RAGComponents,
) -> Tuple[str, Any, Dict[str, Any]]:
    """
    Supply Line 1 wiring.

    Query → EntityAdapter → MetricPipeline → KPI formatted block.

    Returns:
        kpi_block:    formatted KPI string (may be empty if no data)
        entities:     EntityExtractionResult from EntityAdapter
        metric_result: raw dict from MetricPipeline.process()
    """
    # 1) Extract entities once (even though MetricPipeline has its own logic)
    entities = rag.adapter.extract(query)

    # 2) Run metric pipeline
    metric_result = rag.metric_pipeline.process(query)

    # 3) Build a small entity_meta summary for the formatter
    entity_meta = {
        "companies": list(entities.companies.tickers), "years": list(entities.years.years), 
        "sections": list(entities.sections), }

    kpi_block = format_analytical_compact(metric_result, entity_meta=entity_meta)

    return kpi_block, entities, metric_result


## thank god notebooks exist and isolation-integration tests make these easier.

# ──────────────────────────────────────────────────────────────────────────────
# Supply Line 2: Retrieval / narrative context side
# ──────────────────────────────────────────────────────────────────────────────

def run_supply_line_2_rag(
    query: str,
    rag: RAGComponents,
) -> Tuple[str, Any, Any, List[Any], str]:
    """
    Supply Line 2 wiring.

    Query
      → EntityAdapter
      → QueryEmbedderV2
      → MetadataFilterBuilder
      → VariantPipeline (inside retriever)
      → S3VectorsRetriever
      → SentenceExpander
      → ContextAssembler

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
        → [ ... !! ]

    Returns:
        context_block:  full assembled context string with metadata header
        entities:       EntityExtractionResult
        bundle:         RetrievalBundle from S3VectorsRetriever
        unique_sents:   list of expanded unique sentence records
        context_str:    raw context text (without the header wrapper)
    """
    # Step 1: Entity extraction
    entities = rag.adapter.extract(query)

    # Step 2: Query embedding
    base_embedding = rag.embedder.embed_query(query, entities)

    # Step 3: Metadata filters
    filtered_filters = rag.filter_builder.build_filters(entities)
    global_filters = rag.filter_builder.build_global_filters(entities)

    # Steps 4–5: S3 retrieval (with variants internal to retriever)
    bundle = rag.retriever.retrieve(
        base_embedding=base_embedding,
        base_query=query,
        filtered_filters=filtered_filters,
        global_filters=global_filters,
    )

    # Steps 6–7: Sentence expansion + dedup
    unique_sents = rag.expander.expand_and_deduplicate(bundle.union_hits)

    # Step 10: Context assembly
    context_str = rag.assembler.assemble(unique_sents)

    # Very lean header: just tell the LLM what this block is
    header_lines = [
        "══════════════════════════════════════════════════════════════════════",
        "NARRATIVE CONTEXT - SEC FILINGS",
        "══════════════════════════════════════════════════════════════════════",
        "",
    ]

    context_block = "\n".join(header_lines) + context_str + "\n"

    return context_block, entities, bundle, unique_sents, context_str



# ──────────────────────────────────────────────────────────────────────────────
# Combined assembly: KPI + RAG
# ──────────────────────────────────────────────────────────────────────────────


def build_combined_context(
    query: str,
    rag: RAGComponents,
    include_kpi: bool = True,
    include_rag: bool = True,
) -> Tuple[str, Dict[str, Any]]:
    """
    High-level helper: run both supply lines and append their outputs.
    
    Final format:
        [KPI SNAPSHOT]
        
        [NARRATIVE CONTEXT]
        
        ══════════════════════════════════════════════════════════════════════
        USER QUESTION
        ══════════════════════════════════════════════════════════════════════
        
        {user_query}

    Returns:
        combined: big string (KPI + RAG + Query at end)
        meta:     minimal dict with intermediate pieces
    """
    meta: Dict[str, Any] = {
        "kpi_block": "",
        "rag_block": "",
        "kpi_entities": None,
        "rag_entities": None,
        "retrieval_bundle": None,
    }

    pieces: List[str] = []

    # KPI side
    if include_kpi:
        kpi_block, kpi_entities, _ = run_supply_line_1_kpi(query, rag)
        meta["kpi_block"] = kpi_block
        meta["kpi_entities"] = kpi_entities
        if kpi_block:
            pieces.append(kpi_block)

    # RAG side
    if include_rag:
        rag_block, rag_entities, rag_bundle, _, _ = run_supply_line_2_rag(query, rag)
        meta["rag_block"] = rag_block
        meta["rag_entities"] = rag_entities
        meta["retrieval_bundle"] = rag_bundle
        if rag_block:
            if pieces:
                pieces.append("")  # blank line between KPI and RAG
            pieces.append(rag_block)

    # ------------------------------------------------------------------
    # Add query footer at the end
    # ------------------------------------------------------------------
    if pieces:  # Only add footer if we have content
        pieces.append("")  # blank line before footer
        pieces.extend([
            "══════════════════════════════════════════════════════════════════════",
            "USER QUESTION",
            "══════════════════════════════════════════════════════════════════════",
            "",
            query
        ])

    combined = "\n".join(pieces)
    return combined, meta


