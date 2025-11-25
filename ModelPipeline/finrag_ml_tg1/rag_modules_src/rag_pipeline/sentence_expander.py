"""
Sentence expansion module - Converts S3 hits to deduplicated sentence records.

Two-stage pipeline:
1. Window Expansion: Each S3Hit → SentenceRecords (core + ±N neighbors)
2. Sentence Deduplication: Merge overlapping windows (keep best evidence)

Does NOT create blocks or group sentences - that's Assembly's responsibility.
Just expands windows and deduplicates at sentence level.

-------------------------------------------------------------------------------
Architecture:
    30 S3Hits
      ↓
    Expand windows (±3 sentences each)
      ↓
    ~210 SentenceRecords (with duplicates from overlapping windows)
      ↓
    Deduplicate by sentenceID (keep best parent_hit_distance)
      ↓
    ~140 unique SentenceRecords
      ↓
    [Pass to ContextAssembler for sorting/formatting]

-------------------------------------------------------------------------------
Design Decisions:
- Position-based windows (sentence_pos ± N) - fast, reliable for curated 2015-2020 data
- No embedding_id filter for neighbors - we want TEXT regardless of embedding status
- Edge cases auto-handled by max/min boundary clamping
- Populate prev/next/section_count fields for safety (even if not strictly needed)
- is_core_hit marker distinguishes S3 retrieval hits from context neighbors
- Sentence-level dedup is FINAL dedup (no Stage-2 dedup module needed)

Critical Schema:
- Stage 2 Meta Table: finrag_fact_sentences_meta_embeds.parquet
  • sentenceID (str) - join key
  • sentence (str) - THE TEXT
  • sentence_pos (int) - for range queries
  • prev_sentenceID, next_sentenceID (str) - navigation (populated for safety)
  • section_sentence_count (int) - boundary info
  • cik_int, report_year, section_name, name, docID - metadata
------------------------------------------------------------------------------

# Note: Assumes standard paths - adjust if your setup differs
from pathlib import Path
model_root = Path(__file__).resolve().parents[3]  # Up to ModelPipeline/

dim_companies = model_root / "finrag_ml_tg1/data_cache/dimensions/finrag_dim_companies_21.parquet"
dim_sections = model_root / "finrag_ml_tg1/data_cache/dimensions/finfrag_dim_sec_sections.parquet"
---------------------------------------------------------------------------------
"""

from typing import List
from pathlib import Path
from collections import defaultdict
import logging

import polars as pl

from finrag_ml_tg1.loaders.ml_config_loader import MLConfig
from finrag_ml_tg1.rag_modules_src.rag_pipeline.models import S3Hit, SentenceRecord
from finrag_ml_tg1.loaders.data_loader_strategy import DataLoaderStrategy


logger = logging.getLogger(__name__)


class SentenceExpander:
    """
    Expands S3 sentence hits to windowed sentence records with deduplication.
    
    Responsibilities:
    - Window expansion: ±N sentences around each S3 hit
    - Boundary handling: First/last sentence edge cases
    - Sentence deduplication: Keep best evidence when windows overlap
    - Provenance tracking: is_core_hit marker + sources aggregation
    
    Does NOT handle:
    - Block creation (that's ContextAssembler)
    - Text formatting (that's ContextAssembler)
    - Grouping logic (that's ContextAssembler)
    
    Usage:
        expander = SentenceExpander(config)
        sentences = expander.expand_and_deduplicate(bundle.union_hits)
        # Returns: List[SentenceRecord] (deduplicated, ready for assembly)
    """
    

    def __init__(self, data_loader: DataLoaderStrategy, config: MLConfig):
        """
        Initialize sentence expander with DataLoader injection.
        
        Args:
            data_loader: DataLoader instance (LocalCacheLoader or S3StreamingLoader)
            config: MLConfig instance (for window_size + meta table path)
        
        Note:
            Stage 2 Meta table is loaded via data_loader.load_stage2_meta()
            This works in both local dev (filesystem) and Lambda (S3 streaming)
        """
        self.data_loader = data_loader
        self.config = config
        
        # ════════════════════════════════════════════════════════════════════
        # Load Stage 2 Meta Table via DataLoader
        # ════════════════════════════════════════════════════════════════════
        logger.info("Loading Stage 2 meta table via DataLoader...")
        
        meta_df = self.data_loader.load_stage2_meta()
        
        logger.info(f"✓ Loaded Stage 2 meta table: {len(meta_df):,} rows")
        
        # ════════════════════════════════════════════════════════════════════
        # CRITICAL FIX: Extract sentence_pos from sentenceID
        # ════════════════════════════════════════════════════════════════════
        from finrag_ml_tg1.rag_modules_src.utilities.sentence_utils import extract_sentence_position
        
        logger.info("Extracting sentence_pos from sentenceID...")
        
        self.meta_df = extract_sentence_position(meta_df, sentenceid_col='sentenceID')
        
        # Validate extraction
        failed_count = self.meta_df.filter(pl.col('sentence_pos') == -1).height
        
        if failed_count > 0:
            logger.warning(
                f"  ⚠ {failed_count} sentences have pos=-1 (malformed sentenceID)\n"
                f"    These will get random neighbor selection if they're core hits"
            )
        
        logger.info(
            f"✓ Loaded {len(self.meta_df):,} sentences with extracted positions\n"
            f"  Valid positions: {len(self.meta_df) - failed_count:,}\n"
            f"  Failed extraction: {failed_count}"
        )
        
        # ════════════════════════════════════════════════════════════════════
        # Cache config parameters
        # ════════════════════════════════════════════════════════════════════
        retrieval_cfg = config.get_retrieval_config()
        self.window_size = retrieval_cfg.get('window_size', 3)
        
        logger.info(
            f"SentenceExpander initialized: window_size=±{self.window_size} sentences"
        )





    def expand_and_deduplicate(self, hits: List[S3Hit]) -> List[SentenceRecord]:
        """
        Complete expansion + deduplication pipeline.
        
        Two-stage process:
        1. Expand each hit to sentence records (core + neighbors)
        2. Deduplicate at sentence level (keep best evidence)
        
        This is the main entry point - returns deduplicated sentences ready
        for assembly into LLM context.
        
        Args:
            hits: S3 retrieval hits (from bundle.union_hits after proportional topK)
        
        Returns:
            List of unique SentenceRecords (one per sentence, deduplicated)
        
        Example:
            >>> expander = SentenceExpander(config)
            >>> sentences = expander.expand_and_deduplicate(bundle.union_hits)
            >>> len(sentences)
            142
            >>> sentences[0].is_core_hit
            True
        """
        if not hits:
            logger.warning("Empty hits list, returning empty sentences")
            return []
        
        logger.info(
            f"═══════════════════════════════════════════════════════════════\n"
            f"Sentence Expansion Pipeline: {len(hits)} S3 hits\n"
            f"═══════════════════════════════════════════════════════════════"
        )
        
        # ════════════════════════════════════════════════════════════════════
        # STEP 1: Window Expansion
        # ════════════════════════════════════════════════════════════════════
        logger.info("→ Step 1: Expanding windows (core + neighbors)...")
        sentence_records = self._expand_windows(hits)
        logger.info(f"  ✓ Created {len(sentence_records)} sentence records (with duplicates)")
        
        # ════════════════════════════════════════════════════════════════════
        # STEP 2: Sentence-Level Deduplication (FINAL DEDUP)
        # ════════════════════════════════════════════════════════════════════
        logger.info("→ Step 2: Deduplicating at sentence level (final dedup)...")
        unique_sentences = self._deduplicate_sentences(sentence_records)
        logger.info(f"  ✓ Deduplicated to {len(unique_sentences)} unique sentences")
        
        logger.info(
            f"═══════════════════════════════════════════════════════════════\n"
            f"✓ Expansion complete: {len(hits)} hits → {len(unique_sentences)} sentences\n"
            f"═══════════════════════════════════════════════════════════════"
        )
        
        return unique_sentences
    


    def _expand_windows(self, hits: List[S3Hit]) -> List[SentenceRecord]:
        """
        Expand each S3Hit to sentence records (core + ±N neighbors).
        
        For each hit:
        1. Calculate window boundaries using sentence_pos ± window_size
        2. Clamp to section bounds (handles first/last sentence edge cases)
        3. Query Stage 2 meta for sentences in position range
        4. Create SentenceRecord for each sentence
        5. Mark is_core_hit for the actual S3 hit vs neighbors
        6. Populate navigation fields (prev/next/count) for safety
        
        Edge cases automatically handled:
        - pos=-1 (malformed sentenceID): Get core + 2 random neighbors
        - First sentence (pos=0 or low): window starts at 0 (no negative positions)
        - Last sentence (pos=max): window ends at max (no overflow)
        - Tiny sections: window doesn't exceed available sentences
        - Single sentence sections: window = just that sentence
        - Empty query results: logged and skipped (graceful degradation)
        
        Args:
            hits: S3 retrieval hits (already deduplicated at hit level)
        
        Returns:
            List of SentenceRecords (may contain duplicates from overlapping windows)
        """
        all_records = []
        
        for hit_idx, hit in enumerate(hits, start=1):
            # ════════════════════════════════════════════════════════════════
            # EDGE CASE: Malformed sentenceID (pos=-1)
            # ════════════════════════════════════════════════════════════════
            if hit.sentence_pos == -1:
                logger.warning(
                    f"    Hit {hit_idx}/{len(hits)}: Malformed sentenceID (pos=-1)\n"
                    f"      sentenceID={hit.sentence_id}\n"
                    f"      Fallback: Getting core + 2 random neighbors from section"
                )
                
                # Get core sentence first
                core_sentence = self.meta_df.filter(
                    pl.col('sentenceID') == hit.sentence_id
                )
                
                if len(core_sentence) == 0:
                    logger.error(
                        f"      ✗ Core sentence not found in meta table: {hit.sentence_id}\n"
                        f"        Skipping this hit."
                    )
                    continue
                
                # Get 2 random neighbors from same section (exclude core)
                neighbors = self.meta_df.filter(
                    (pl.col('cik_int') == hit.cik_int) &
                    (pl.col('report_year') == hit.report_year) &
                    (pl.col('section_name') == hit.section_name) &
                    (pl.col('sentenceID') != hit.sentence_id)
                )
                
                if len(neighbors) >= 2:
                    random_neighbors = neighbors.sample(n=2, seed=42)
                elif len(neighbors) > 0:
                    random_neighbors = neighbors  # Take whatever is available
                else:
                    random_neighbors = pl.DataFrame()  # No neighbors available
                
                # Combine core + neighbors
                if len(random_neighbors) > 0:
                    window_sentences = pl.concat([core_sentence, random_neighbors])
                else:
                    window_sentences = core_sentence  # Just core, no neighbors
                
                logger.debug(
                    f"      Fallback result: {len(window_sentences)} sentences "
                    f"(1 core + {len(window_sentences)-1} neighbors)"
                )
                
                # Create records for fallback window
                for row in window_sentences.iter_rows(named=True):
                    is_core = (row['sentenceID'] == hit.sentence_id)
                    
                    prev_id = row.get('prev_sentenceID')
                    next_id = row.get('next_sentenceID')
                    section_count = row.get('section_sentence_count')
                    
                    if prev_id is not None and isinstance(prev_id, str) and prev_id.strip() == "":
                        prev_id = None
                    if next_id is not None and isinstance(next_id, str) and next_id.strip() == "":
                        next_id = None
                    
                    record = SentenceRecord(
                        sentence_id=row['sentenceID'],
                        sentence_pos=row.get('sentence_pos', -1),  # May still be -1
                        cik_int=row['cik_int'],
                        report_year=row['report_year'],
                        section_name=row['section_name'],
                        doc_id=row['docID'],
                        company_name=row['name'],
                        text=row['sentence'],
                        is_core_hit=is_core,
                        parent_hit_distance=hit.distance,
                        sources=hit.sources.copy(),
                        variant_ids=hit.variant_ids.copy(),
                        prev_sentence_id=prev_id,
                        next_sentence_id=next_id,
                        section_sentence_count=section_count
                    )
                    
                    all_records.append(record)
                
                continue  # Skip normal window expansion, move to next hit
            
            # ════════════════════════════════════════════════════════════════
            # NORMAL CASE: Valid sentence_pos
            # ════════════════════════════════════════════════════════════════
            # STEP 1.1: Calculate window boundaries (edge-case safe)
            # ════════════════════════════════════════════════════════════════
            
            ## rewriting: v1
            # window_start = max(0, hit.sentence_pos - self.window_size)
            # window_end = min( hit.section_sentence_count - 1, hit.sentence_pos + self.window_size )
            
            ## v2
            window_start = max(1, hit.sentence_pos - self.window_size)
            window_end = min(hit.section_sentence_count, hit.sentence_pos + self.window_size)

            # Detect and log edge cases
            is_near_start = (hit.sentence_pos < self.window_size)
            is_near_end = (hit.sentence_pos > hit.section_sentence_count - self.window_size - 1)
            is_tiny_section = (hit.section_sentence_count <= 2 * self.window_size + 1)
            
            if is_near_start or is_near_end or is_tiny_section:
                logger.debug(
                    f"    Hit {hit_idx}/{len(hits)}: Edge case\n"
                    f"      pos={hit.sentence_pos}/{hit.section_sentence_count-1}, "
                    f"window=[{window_start},{window_end}], "
                    f"near_start={is_near_start}, near_end={is_near_end}, "
                    f"tiny={is_tiny_section}"
                )
            
            # ════════════════════════════════════════════════════════════════
            # STEP 1.2: Query Stage 2 Meta for window sentences
            # ════════════════════════════════════════════════════════════════
            # CRITICAL: NO embedding_id filter
            # We want TEXT regardless of whether neighbors were embedded
            window_sentences = self.meta_df.filter(
                (pl.col('cik_int') == hit.cik_int) &
                (pl.col('report_year') == hit.report_year) &
                (pl.col('section_name') == hit.section_name) &
                (pl.col('sentence_pos') >= window_start) &
                (pl.col('sentence_pos') <= window_end)
            ).unique(
                subset=['sentenceID'], 
                keep='first'  # If multiple embeddings, take first (text is same)
            ).sort('sentence_pos')
            
            # ════════════════════════════════════════════════════════════════
            # EDGE CASE: Empty window (defensive - shouldn't happen)
            # ════════════════════════════════════════════════════════════════
            if len(window_sentences) == 0:
                logger.warning(
                    f"    ✗ Empty window for hit {hit_idx}:\n"
                    f"      sentenceID={hit.sentence_id}, pos={hit.sentence_pos}\n"
                    f"      cik={hit.cik_int}, year={hit.report_year}, "
                    f"section={hit.section_name}\n"
                    f"      window=[{window_start}, {window_end}]\n"
                    f"      Skipping this hit."
                )
                continue
            
            # Validate window size
            expected_size = window_end - window_start + 1
            actual_size = len(window_sentences)
            
            if actual_size < expected_size:
                logger.debug(
                    f"    Hit {hit_idx}: Incomplete window "
                    f"({actual_size}/{expected_size} sentences)"
                )
            
            # ════════════════════════════════════════════════════════════════
            # STEP 1.3: Create SentenceRecord for each sentence in window
            # ════════════════════════════════════════════════════════════════
            for row in window_sentences.iter_rows(named=True):
                # Determine if this sentence is the core S3 hit
                is_core = (row['sentenceID'] == hit.sentence_id)
                
                # Extract navigation fields (populate for safety, even if optional)
                prev_id = row.get('prev_sentenceID')
                next_id = row.get('next_sentenceID')
                section_count = row.get('section_sentence_count')
                
                # Handle None/empty string as None
                if prev_id is not None and isinstance(prev_id, str) and prev_id.strip() == "":
                    prev_id = None
                if next_id is not None and isinstance(next_id, str) and next_id.strip() == "":
                    next_id = None
                
                # Create sentence record
                record = SentenceRecord(
                    # Identity & Position
                    sentence_id=row['sentenceID'],
                    sentence_pos=row['sentence_pos'],
                    
                    # Context Metadata
                    cik_int=row['cik_int'],
                    report_year=row['report_year'],
                    section_name=row['section_name'],
                    doc_id=row['docID'],
                    company_name=row['name'],
                    
                    # Content
                    text=row['sentence'],
                    
                    # Provenance (inherited from parent S3Hit)
                    is_core_hit=is_core,
                    parent_hit_distance=hit.distance,
                    sources=hit.sources.copy(),  # Deep copy to avoid mutation
                    variant_ids=hit.variant_ids.copy(),
                    
                    # Navigation (for safety/future use)
                    prev_sentence_id=prev_id,
                    next_sentence_id=next_id,
                    section_sentence_count=section_count
                )
                
                all_records.append(record)
            
            # ════════════════════════════════════════════════════════════════
            # VALIDATION: Core hit must be in its own window
            # ════════════════════════════════════════════════════════════════
            core_found = any(
                row['sentenceID'] == hit.sentence_id 
                for row in window_sentences.iter_rows(named=True)
            )
            
            if not core_found:
                logger.error(
                    f"    ✗ CRITICAL: Core hit not found in window!\n"
                    f"      Hit: {hit.sentence_id} (pos={hit.sentence_pos})\n"
                    f"      Window: [{window_start}, {window_end}]\n"
                    f"      This indicates data integrity issue."
                )
        
        # ════════════════════════════════════════════════════════════════════
        # Stats: Core hits vs neighbors
        # ════════════════════════════════════════════════════════════════════
        core_count = sum(1 for r in all_records if r.is_core_hit)
        neighbor_count = len(all_records) - core_count
        
        logger.info(
            f"  Window expansion stats:\n"
            f"    Total records: {len(all_records)}\n"
            f"    Core hits: {core_count}\n"
            f"    Neighbors: {neighbor_count}\n"
            f"    Avg window size: {len(all_records)/len(hits):.1f} sentences/hit"
        )
        
        return all_records


    


    
    def _deduplicate_sentences(self, records: List[SentenceRecord]) -> List[SentenceRecord]:
        """
        Deduplicate sentence records by sentenceID (FINAL deduplication).
        
        When multiple windows include the same sentence, keep the version from
        the hit with the best (lowest) distance. This ensures we use the
        highest-quality evidence for each sentence.
        
        Deduplication key: (sentence_id, cik_int, report_year, section_name)
        Keep strategy: Record with lowest parent_hit_distance
        Aggregation: sources, variant_ids, is_core_hit (OR operation)
        
        This is the FINAL deduplication in the pipeline - no Stage-2 dedup needed.
        After this, we have one record per unique sentence with best evidence.
        
        Edge cases:
        - Empty input: Returns empty list
        - Single version: No aggregation needed, returns as-is
        - Multiple versions: Keeps best, aggregates provenance
        - is_core_hit conflict: TRUE if ANY version was core
        
        Args:
            records: SentenceRecords from window expansion (may have duplicates)
        
        Returns:
            Deduplicated SentenceRecords (one per unique sentence)
        """
        if not records:
            logger.debug("  Empty records, nothing to deduplicate")
            return []
        
        # ════════════════════════════════════════════════════════════════════
        # STEP 2.1: Group by composite key
        # ════════════════════════════════════════════════════════════════════
        groups = defaultdict(list)
        
        for rec in records:
            # Composite key ensures same sentence from different docs = different keys
            key = (rec.sentence_id, rec.cik_int, rec.report_year, rec.section_name)
            groups[key].append(rec)
        
        logger.debug(
            f"  Sentence grouping: {len(records)} records → {len(groups)} unique sentences"
        )
        
        # ════════════════════════════════════════════════════════════════════
        # STEP 2.2: Keep best version per sentence
        # ════════════════════════════════════════════════════════════════════
        deduped = []
        multi_version_count = 0
        
        for key, group_recs in groups.items():
            # Edge case: Single version (no aggregation needed)
            if len(group_recs) == 1:
                deduped.append(group_recs[0])
                continue
            
            # Multiple versions exist - need aggregation
            multi_version_count += 1
            
            # Sort by parent_hit_distance (best first)
            group_recs.sort(key=lambda r: r.parent_hit_distance)
            
            # Keep the best (first after sorting)
            best = group_recs[0]
            
            # Aggregate provenance from ALL versions
            all_sources = set()
            all_variant_ids = set()
            any_core_hit = False
            
            for rec in group_recs:
                all_sources.update(rec.sources)
                all_variant_ids.update(rec.variant_ids)
                if rec.is_core_hit:
                    any_core_hit = True
            
            # Update best record with aggregated metadata
            best.sources = all_sources
            best.variant_ids = all_variant_ids
            best.is_core_hit = any_core_hit  # TRUE if ANY version was core
            
            logger.debug(
                f"    Merged {len(group_recs)} versions of {best.sentence_id}: "
                f"kept distance={best.parent_hit_distance:.4f}, "
                f"is_core={best.is_core_hit}"
            )
            
            deduped.append(best)
        
        # ════════════════════════════════════════════════════════════════════
        # Stats: Deduplication effectiveness
        # ════════════════════════════════════════════════════════════════════
        core_count = sum(1 for r in deduped if r.is_core_hit)
        neighbor_count = len(deduped) - core_count
        
        logger.info(
            f"  Deduplication complete:\n"
            f"    {len(records)} records → {len(deduped)} unique sentences\n"
            f"    Sentences with multiple versions: {multi_version_count}\n"
            f"    Final composition:\n"
            f"      Core hits: {core_count}\n"
            f"      Neighbors: {neighbor_count}"
        )
        
        return deduped
    




"""
## ===========================================================================================================
## SentenceExpander without DataLoader injection (legacy version) 

    def __init__(self, config: MLConfig):
        Initialize sentence expander and load Stage 2 meta table with sentence_pos.
        
        Args:
            config: MLConfig instance (for window_size + meta table path)
        
        Raises:
            FileNotFoundError: If Stage 2 meta table doesn't exist
            RuntimeError: If ModelPipeline root cannot be found
        # ════════════════════════════════════════════════════════════════════
        # Resolve paths
        # ════════════════════════════════════════════════════════════════════
        current_file = Path(__file__).resolve()
        model_root = None
        
        for parent in current_file.parents:
            if parent.name == "ModelPipeline":
                model_root = parent
                break
        
        if model_root is None:
            raise RuntimeError(
                f"Cannot find 'ModelPipeline' root in path tree.\n"
                f"Current file: {current_file}"
            )
        
        self.meta_path = (
            model_root / 
            "finrag_ml_tg1/data_cache/meta_embeds/finrag_fact_sentences_meta_embeds.parquet"
        )
        
        if not self.meta_path.exists():
            raise FileNotFoundError(
                f"Stage 2 meta table not found: {self.meta_path}\n"
                f"Expected location: {self.meta_path.parent}"
            )
        
        # ════════════════════════════════════════════════════════════════════
        # Load Stage 2 Meta Table
        # ════════════════════════════════════════════════════════════════════
        logger.info(f"Loading Stage 2 meta table: {self.meta_path}")
        
        meta_df = pl.read_parquet(self.meta_path)
        
        # ════════════════════════════════════════════════════════════════════
        # CRITICAL FIX: Extract sentence_pos from sentenceID
        # ════════════════════════════════════════════════════════════════════
        from finrag_ml_tg1.rag_modules_src.utilities.sentence_utils import extract_sentence_position
        
        logger.info("Extracting sentence_pos from sentenceID...")
        
        self.meta_df = extract_sentence_position(meta_df, sentenceid_col='sentenceID')
        
        # Validate extraction
        failed_count = self.meta_df.filter(pl.col('sentence_pos') == -1).height
        
        if failed_count > 0:
            logger.warning(
                f"  ⚠ {failed_count} sentences have pos=-1 (malformed sentenceID)\n"
                f"    These will get random neighbor selection if they're core hits"
            )
        
        logger.info(
            f"✓ Loaded {len(self.meta_df):,} sentences with extracted positions\n"
            f"  Valid positions: {len(self.meta_df) - failed_count:,}\n"
            f"  Failed extraction: {failed_count}"
        )
        
        # ════════════════════════════════════════════════════════════════════
        # Cache config parameters
        # ════════════════════════════════════════════════════════════════════
        retrieval_cfg = config.get_retrieval_config()
        self.window_size = retrieval_cfg.get('window_size', 3)
        
        logger.info(
            f"SentenceExpander initialized: window_size=±{self.window_size} sentences"
        )
"""