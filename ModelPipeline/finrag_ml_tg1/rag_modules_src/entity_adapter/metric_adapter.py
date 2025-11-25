# rag_modules_src/entity_adapter/metric_adapter.py

# Reuse the *existing* metric pipeline logic
# But point it at your richer v2 mapping constants

from __future__ import annotations

import logging
from typing import List, Optional
from pathlib import Path

from finrag_ml_tg1.rag_modules_src.metric_pipeline.src.filter_extractor import (
    FilterExtractor,
)
from finrag_ml_tg1.rag_modules_src.constants.metric_mapping_v2 import METRIC_MAPPINGS
from finrag_ml_tg1.loaders.data_loader_strategy import DataLoaderStrategy

from .models import MetricMatches

logger = logging.getLogger(__name__)



class FilterExtractorV2(FilterExtractor):
    """
    Extension of the original FilterExtractor that swaps in the
    v2 metric mappings.

    We do *not* reimplement _extract_metrics; we only override the
    metric_map that that method consults.

    This means:
    - All bugfixes / improvements in the original FilterExtractor
      (tokenization, fuzzy matching, etc.) are inherited automatically.
    - Only the underlying NL->canonical metric dictionary changes,
      coming from metric_mapping_v2.METRIC_MAPPINGS.
    """

    def __init__(self) -> None:
        super().__init__()
        # Original class does: self.metric_map = METRIC_MAPPINGS from config.metric_mappings
        # We overwrite it to use the richer v2 mapping instead.
        self.metric_map = METRIC_MAPPINGS
        logger.info(
            "FilterExtractorV2 initialized with "
            f"{len(self.metric_map)} metric mapping entries from metric_mapping_v2"
        )


class MetricAdapter:
    """
    Adapter for extracting metric entities from queries using v2 mappings.
    
    This wraps FilterExtractor but provides an interface consistent with
    the rest of the entity_adapter package.
    """
    
    def __init__(self, data_loader: Optional[DataLoaderStrategy] = None, company_dim_path: Optional[str | Path] = None):
        """
        Initialize MetricAdapter.
        
        Args:
            data_loader: Optional DataLoader instance (new pattern, preferred)
            company_dim_path: Optional path to company dimension parquet file (legacy)
                            If both are None, FilterExtractor uses its default path.
        
        Note:
            Passes data_loader to FilterExtractor for Lambda-compatible initialization.
        """
        # Initialize FilterExtractor with DataLoader support
        self.extractor = FilterExtractor(
            company_dim_path=company_dim_path,
            data_loader=data_loader
        )
        
        # Override with v2 metric mappings
        self.extractor.metric_map = METRIC_MAPPINGS
        
        logger.info("MetricAdapter initialized with v2 metric mappings")
        

    def extract(self, query: str) -> MetricMatches:
        """
        Extract metrics from a query.
        
        Args:
            query: Natural language query
            
        Returns:
            MetricMatches with list of canonical metric IDs
        """
        # Use FilterExtractor's metric extraction logic
        metrics = self.extractor._extract_metrics(query)
        return MetricMatches(metrics=metrics)
    

    