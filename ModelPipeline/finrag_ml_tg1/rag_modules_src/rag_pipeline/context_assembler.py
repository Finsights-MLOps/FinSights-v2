"""
HIGH-LEVEL RESPONSIBILITY:
Purpose: Transform List[SentenceRecord] → LLM-ready formatted string

Sort sentences by logical reading order
Insert headers when context changes (company/year/section)
-------------------------------------------------------------------------------
# Step 1: Optional filtering (top K by score) x no need topK
# Step 2: Sort by functional order
# Step 3: Format with headers
# Step 4: Return string
--------------------------------------------------------------------------------

Sort Order: Company → Year ASC → DocID -> Section → Sentence Pos

"""



"""
Context assembler module - Formats deduplicated sentences into LLM-ready context.

Converts List[SentenceRecord] → formatted string with minimal headers.

Architecture:
    ~140 unique SentenceRecords
      ↓
    Sort by (company, year ASC, section, doc, pos)
      ↓
    Group by (company, year, section) for headers
      ↓
    Format with double-newline spacing
      ↓
    Single formatted string
      ↓
    Ready for BedrockClient / LLM prompt

Design Decisions:
- Sort order: Company → Year ASC → Section → Doc → Position
  • Year ascending: Shows chronological progression (2017 → 2020)
  • Company grouping: All NVIDIA together, all MSFT together
  • Section grouping: All ITEM_1A together, all ITEM_7 together
  
- Minimal headers: Only when (company, year, section) changes
  • Format: === COMPANY NAME | YEAR | SECTION ===
  • No scores, no metadata, clean provenance only
  
- Double-newline spacing: Improves LLM parsing and readability
  • Between sentences: `\n\n` (paragraph-like)
  • Between groups: Extra blank line before header
  
- No topK limiting: Already done in S3VectorsRetriever proportional topK
  
- No deduplication: Already done in SentenceExpander

Grain Validation:
- (cik_int, report_year, docID, section_name, sentence_pos) = Unique sentence
- Functionally equivalent to sentenceID
- Validated: 469,252 rows, 0 duplicates, perfect bijection

"""


from typing import List
import logging
from pathlib import Path

import polars as pl
from finrag_ml_tg1.loaders.ml_config_loader import MLConfig
from finrag_ml_tg1.rag_modules_src.rag_pipeline.models import SentenceRecord
from finrag_ml_tg1.loaders.data_loader_strategy import DataLoaderStrategy


logger = logging.getLogger(__name__)


class ContextAssembler:
    """
    Formats deduplicated sentences into LLM-ready context string.
    
    Single responsibility: Sort + group + format.
    No deduplication, no limiting, no scoring - pure formatting.
    
    The assembly process:
    1. Sort sentences by logical reading order (company, year, section, position)
    2. Detect group boundaries (when company/year/section changes)
    3. Insert headers at boundaries
    4. Format with double-newline spacing
    5. Return single string

    ## v2. Enhanced citation headers     
    Output format:
        === [AAPL] APPLE INC | FY 2018 | Doc: aapl_2018_10k | Item 1A: Risk Factors | Sentences: sent_abc123_xyz - sent_def456_uvw ===
        
        sentence1
        
        sentence2
    
    Usage:
        assembler = ContextAssembler(config)
        context_str = assembler.assemble(unique_sentences)
        # Pass context_str to BedrockClient for LLM synthesis

    Dont do this way:
    X All class-level path resolution code (_current, _model_root, loop)
    X DIM_COMPANIES_PATH class constant
    X DIM_SECTIONS_PATH class constant
    """
    
    def __init__(self, data_loader: DataLoaderStrategy, config: MLConfig):
        """
        Initialize context assembler with DataLoader injection.
        
        Args:
            data_loader: DataLoader instance (LocalCacheLoader or S3StreamingLoader)
            config: MLConfig instance (currently unused, but available for future config)
        
        Note: Loads dimension tables via data_loader for ticker and section name lookups.
        """
        self.data_loader = data_loader
        self.config = config
        
        logger.info("ContextAssembler initializing...")
        
        # Load dimension tables via DataLoader
        self.dim_companies = self.data_loader.load_dimension_companies()
        self.dim_sections = self.data_loader.load_dimension_sections()
        
        logger.info(
            f"  ✓ Loaded {len(self.dim_companies)} companies\n"
            f"  ✓ Loaded {len(self.dim_sections)} sections"
        )
        logger.info("ContextAssembler initialized")




    def assemble(self, sentences: List[SentenceRecord]) -> str:
        """
        Format sentences into LLM-ready context string.
        
        Process:
        1. Sort by (company, year ASC, section, doc, pos)
        2. Insert headers when (company, year, section) changes
        3. Add sentences with double-newline spacing
        4. Return formatted string
        
        Args:
            sentences: Deduplicated SentenceRecords from SentenceExpander
        
        Returns:
            Formatted context string with headers and clean text.
            Empty string if input is empty.
        
        Example:
            >>> assembler = ContextAssembler(config)
            >>> context = assembler.assemble(sentences)
            >>> print(context[:200])
            === [NVDA] NVIDIA CORP | FY 2018 | Doc: nvda_2018_10k | Item 1A: Risk Factors | Sentences: sent_123 - sent_145 ===
            
            We face risks related to supply chain constraints...
            
            Competition in the data center market has intensified...
        """
        if not sentences:
            logger.warning("Empty sentences list, returning empty context")
            return ""
        
        logger.info(
            f"═══════════════════════════════════════════════════════════════\n"
            f"Context Assembly: {len(sentences)} sentences\n"
            f"═══════════════════════════════════════════════════════════════"
        )
        
        # ════════════════════════════════════════════════════════════════════
        # STEP 1: Sort by functional order
        # ════════════════════════════════════════════════════════════════════
        logger.info("→ Sorting sentences by (company, year ASC, section, doc, pos)...")
        
        sorted_sentences = self._sort_sentences(sentences)
        
        # Log grouping stats
        companies = {s.company_name for s in sorted_sentences}
        years = {s.report_year for s in sorted_sentences}
        sections = {s.section_name for s in sorted_sentences}
        
        logger.info(
            f"  ✓ Sorted {len(sorted_sentences)} sentences\n"
            f"    Companies: {len(companies)}\n"
            f"    Years: {sorted(years)}\n"
            f"    Sections: {len(sections)}"
        )
        
        # ════════════════════════════════════════════════════════════════════
        # STEP 2: Format with headers
        # ════════════════════════════════════════════════════════════════════
        logger.info("→ Formatting with headers...")
        
        formatted_context = self._format_with_headers(sorted_sentences)
        
        # Stats
        char_count = len(formatted_context)
        estimated_tokens = char_count // 4
        header_count = formatted_context.count("===")
        
        logger.info(
            f"  ✓ Assembly complete\n"
            f"    Characters: {char_count:,}\n"
            f"    Estimated tokens: {estimated_tokens:,}\n"
            f"    Headers inserted: {header_count}"
        )
        
        logger.info(
            f"═══════════════════════════════════════════════════════════════\n"
            f"✓ Context ready for LLM ({estimated_tokens:,} tokens)\n"
            f"═══════════════════════════════════════════════════════════════"
        )
        
        return formatted_context
    
    
    def _sort_sentences(self, sentences: List[SentenceRecord]) -> List[SentenceRecord]:
        """
        Sort sentences by logical reading order.
        
        Sort key: (company_name, report_year ASC, section_name, doc_id, sentence_pos)
        
        This produces chronological progression within each company:
        - NVIDIA 2017 ITEM_1A
        - NVIDIA 2018 ITEM_1A
        - NVIDIA 2019 ITEM_1A
        - NVIDIA 2020 ITEM_1A
        - NVIDIA 2017 ITEM_7
        - etc.
        
        Args:
            sentences: Unsorted SentenceRecords
        
        Returns:
            Sorted SentenceRecords
        """
        return sorted(sentences, key=lambda s: (
            s.company_name,
            s.report_year,
            s.section_name,
            s.doc_id,
            s.sentence_pos
        ))
    
    
    def _format_with_headers(self, sentences: List[SentenceRecord]) -> str:
        """
        Format sorted sentences with headers.
        
        Inserts header when (company, year, doc_id, section) changes.
        Uses double-newline spacing for clarity.
        
        Format:
            === [TICKER] COMPANY | FY YEAR | Doc: doc_id | Section Display | Sentences: first_id - last_id ===
            
            sentence1 text
            
            sentence2 text
        
        Args:
            sentences: Sorted SentenceRecords
        
        Returns:
            Formatted string with headers and double-newline spacing
        """
        context_parts = []
        current_group = None
        group_sentences = []
        
        for sent in sentences:
            # Group key includes doc_id to handle multiple filings same year
            group_key = (sent.company_name, sent.report_year, sent.doc_id, sent.section_name)
            
            # ════════════════════════════════════════════════════════════════
            # Insert header when context changes
            # ════════════════════════════════════════════════════════════════
            if group_key != current_group:
                # Finalize previous group
                if group_sentences:
                    if context_parts:
                        context_parts.append("")  # Extra blank line between groups
                    
                    # Build and add header
                    header = self._build_header(group_sentences)
                    context_parts.append(header)
                    context_parts.append("")  # Blank line after header
                    
                    # Add sentences from group
                    for gs in group_sentences:
                        context_parts.append(gs.text)
                        context_parts.append("")  # Double newline
                
                # Start new group
                current_group = group_key
                group_sentences = [sent]
            else:
                group_sentences.append(sent)
        
        # Finalize last group
        if group_sentences:
            if context_parts:
                context_parts.append("")
            
            header = self._build_header(group_sentences)
            context_parts.append(header)
            context_parts.append("")
            
            for gs in group_sentences:
                context_parts.append(gs.text)
                context_parts.append("")
        
        return "\n".join(context_parts)
    
        
    def _build_header(self, group_sentences: List[SentenceRecord]) -> str:
        """
        Build enhanced citation header for group of sentences.
        
        Format: === [TICKER] COMPANY | FY YEAR | Doc: doc_id | Section Display | Sentences: first_id - last_id ===
        
        Args:
            group_sentences: Sentences in this group
        
        Returns:
            Formatted header string
        """
        first = group_sentences[0]
        
        # Get first/last sentence IDs by position
        sorted_by_pos = sorted(group_sentences, key=lambda s: s.sentence_pos)
        first_sent_id = sorted_by_pos[0].sentence_id
        last_sent_id = sorted_by_pos[-1].sentence_id
        
        # Lookup ticker
        ticker_result = self.dim_companies.filter(pl.col('cik_int') == first.cik_int)
        if len(ticker_result) > 0:
            ticker = ticker_result['ticker'][0]
        else:
            ticker = first.company_name.split()[0][:4].upper()  # Fallback
        
        # Lookup section display
        section_result = self.dim_sections.filter(pl.col('sec_item_canonical') == first.section_name)
        if len(section_result) > 0:
            section_display = section_result['section_name'][0]  # Correct field: "Item 1: Business"
        else:
            section_display = first.section_name  # Fallback: use as-is
        
        # Build header
        header = (
            f"=== [{ticker}] {first.company_name} | "
            f"FY {first.report_year} | "
            f"Doc: {first.doc_id} | "
            f"{section_display} | "
            f"Sentences: {first_sent_id} - {last_sent_id} ==="
        )
        
        return header
    

"""
## legacy version;

    # Find ModelPipeline root - proper pattern. If you use resolve, pls quit. 
    _current = Path.cwd()
    _model_root = None
    for _parent in [_current] + list(_current.parents):
        if _parent.name == "ModelPipeline":
            _model_root = _parent
            break
    
    if _model_root is None:
        raise RuntimeError("Cannot find 'ModelPipeline' root in path tree")
    
    # Dimension table paths
    DIM_COMPANIES_PATH = _model_root / "finrag_ml_tg1" / "data_cache" / "dimensions" / "finrag_dim_companies_21.parquet"
    DIM_SECTIONS_PATH = _model_root / "finrag_ml_tg1" / "data_cache" / "dimensions" / "finrag_dim_sec_sections.parquet"

    
    def __init__(self, config: MLConfig):
        logger.info("ContextAssembler initializing...")
        
        # Load dimension tables
        if not self.DIM_COMPANIES_PATH.exists():
            raise FileNotFoundError(f"Companies dimension table not found: {self.DIM_COMPANIES_PATH}")
        
        if not self.DIM_SECTIONS_PATH.exists():
            raise FileNotFoundError(f"Sections dimension table not found: {self.DIM_SECTIONS_PATH}")
        
        self.dim_companies = pl.read_parquet(self.DIM_COMPANIES_PATH)
        self.dim_sections = pl.read_parquet(self.DIM_SECTIONS_PATH)
        
        logger.info(
            f"  ✓ Loaded {len(self.dim_companies)} companies\n"
            f"  ✓ Loaded {len(self.dim_sections)} sections"
        )
        logger.info("ContextAssembler initialized")
    

"""