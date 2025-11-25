"""
Main metric pipeline orchestration

ModelPipeline\finrag_ml_tg1\rag_modules_src\metric_pipeline\src\pipeline.py

"""

# uses src.* and config.*.
# Relative within src for FilterExtractor and MetricLookup ? use relative .
# Absolute- use finrag_ml_tg1.rag_modules_src 

from __future__ import annotations

from typing import Dict, Optional
import re
import logging
logger = logging.getLogger(__name__)

from .filter_extractor import FilterExtractor
from .metric_lookup import MetricLookup

# # ModelPipeline\finrag_ml_tg1\rag_modules_src\constants\metric_mapping_v2.py
from finrag_ml_tg1.rag_modules_src.constants.metric_mapping_v2 import (
    METRIC_KEYWORDS,
    QUANTITATIVE_INDICATORS,
)
from finrag_ml_tg1.rag_modules_src.entity_adapter.string_utils import simple_fuzzy_match
from finrag_ml_tg1.loaders.data_loader_strategy import DataLoaderStrategy



class MetricPipeline:
    """Orchestrate the full metric extraction and lookup pipeline"""
    
    def __init__(self, data_loader: DataLoaderStrategy):
        """
        Initialize metric pipeline with DataLoader injection.
        
        Args:
            data_loader: DataLoader instance (LocalCacheLoader or S3StreamingLoader)
        
        Note:
            Loads KPI fact data via data_loader.load_kpi_fact_data()
            Works in both local dev (filesystem) and Lambda (S3 streaming).
        """
        logger.info("MetricPipeline initializing with DataLoader...")
        
        # ════════════════════════════════════════════════════════════════════
        # Initialize FilterExtractor with DataLoader
        # ════════════════════════════════════════════════════════════════════
        self.extractor = FilterExtractor(data_loader=data_loader)
        
        # ════════════════════════════════════════════════════════════════════
        # Load KPI Fact Data via DataLoader
        # ════════════════════════════════════════════════════════════════════
        logger.info("Loading KPI fact data via DataLoader...")
        kpi_df_pl = data_loader.load_kpi_fact_data()
        
        # Convert Polars → Pandas (MetricLookup uses pandas)
        import pandas as pd
        kpi_df_pd = kpi_df_pl.to_pandas()
        
        logger.info(f"  ✓ Loaded {len(kpi_df_pd):,} KPI records")
        
        # ════════════════════════════════════════════════════════════════════
        # Initialize MetricLookup with DataFrame
        # ════════════════════════════════════════════════════════════════════
        self.lookup = MetricLookup.from_dataframe(kpi_df_pd)
        
        logger.info("MetricPipeline initialized successfully")


    def needs_metric_layer(self, query: str) -> bool:
        """
        Determine if query requires metric lookup with FULL FUZZY MATCHING
        Handles any typo in metric keywords automatically
        
        Args:
            query: User's query string
            
        Returns:
            True if metric layer should be activated
        """
        query_lower = query.lower()
        
        # Check for quantitative indicators (exact match is fine)
        has_quantitative = any(ind in query_lower for ind in QUANTITATIVE_INDICATORS)
        
        # Check for year
        has_year = bool(re.search(r'\b(19|20)\d{2}\b', query))
        
        # Check for ticker/company name (use the extractor to be consistent)
        tickers = self.extractor._extract_tickers(query)
        has_ticker = len(tickers) > 0
        
        # Check for metric keywords with fuzzy matching
        has_metric = False
        
        # First try exact match (fast)
        if any(keyword in query_lower for keyword in METRIC_KEYWORDS):
            has_metric = True
        else:
            # Fuzzy match each word against metric keywords
            words = query_lower.split()
            
            for word in words:
                # Skip very short words and common words
                if len(word) < 4:
                    continue
                if word in ['the', 'and', 'what', 'how', 'was', 'is', 'are', 'were', 'from', 'with', 'that', 'this']:
                    continue
                if word.isdigit():
                    continue
                
                best_match, score = simple_fuzzy_match(
                    word,
                    METRIC_KEYWORDS,
                    threshold=0.70  # 70% similarity
                )
                
                if best_match and score >= 70:
                    has_metric = True
                    break
        
        # Trigger if: (has_metric AND (year OR ticker)) OR (quantitative AND ticker)
        return (has_metric and (has_year or has_ticker)) or \
               (has_quantitative and has_ticker)
    
    
    def process(self, query: str) -> Dict[str, any]:
        """
        Main processing pipeline - NOW SUPPORTS MULTI-COMPANY, MULTI-YEAR
        
        Args:
            query: User's natural language query
            
        Returns:
            Dictionary with results or error information
        """
        # Step 1: Check if metric layer needed
        if not self.needs_metric_layer(query):
            return {
                'success': False,
                'reason': 'Query does not require metric lookup',
                'query': query
            }
        
        # Step 2: Extract filters
        filters = self.extractor.extract(query)
        
        # Step 3: Validate filters
        if not self.extractor.is_valid(filters):
            return {
                'success': False,
                'reason': 'Could not extract all required filters (tickers, years, metrics)',
                'extracted_filters': filters,
                'query': query
            }
        
        # Step 4: Lookup data for ALL combinations using optimized batch query
        results = self.lookup.query_batch_optimized(
            tickers=filters['tickers'],
            years=filters['years'],
            metrics=filters['metrics']
        )
        
        # Step 5: Get statistics
        stats = self.lookup.get_batch_statistics(
            tickers=filters['tickers'],
            years=filters['years'],
            metrics=filters['metrics']
        )
        
        # Step 6: Handle results
        if not results:
            return {
                'success': False,
                'reason': 'No data found for any of the requested combinations',
                'filters': filters,
                'stats': stats,
                'query': query
            }
        
        # Filter out None results and results with no data
        valid_results = [r for r in results if r and r.get('found', False)]
        
        if not valid_results:
            return {
                'success': False,
                'reason': 'Data found but all values are missing/NaN',
                'filters': filters,
                'stats': stats,
                'query': query
            }
        
        # Success!
        return {
            'success': True,
            'data': valid_results,
            'filters': filters,
            'stats': stats,
            'query': query,
            'count': len(valid_results)
        }
    
    def format_response(self, pipeline_result: Dict[str, any]) -> str:
        """
        Format pipeline result into human-readable response
        NOW SUPPORTS MULTI-COMPANY, MULTI-YEAR OUTPUT
        
        Args:
            pipeline_result: Output from process() method
            
        Returns:
            Formatted string response
        """
        if not pipeline_result['success']:
            msg = f"❌ {pipeline_result['reason']}\n"
            msg += f"Query: '{pipeline_result['query']}'\n"
            
            if 'extracted_filters' in pipeline_result:
                filters = pipeline_result['extracted_filters']
                msg += f"\nExtracted filters:\n"
                msg += f"  - Tickers: {filters.get('tickers', 'NOT FOUND')}\n"
                msg += f"  - Years: {filters.get('years', 'NOT FOUND')}\n"
                msg += f"  - Metrics: {filters.get('metrics', 'NOT FOUND')}\n"
            
            if 'stats' in pipeline_result:
                stats = pipeline_result['stats']
                msg += f"\n📊 Coverage:\n"
                msg += f"  - Total combinations requested: {stats['total_combinations']}\n"
                msg += f"  - Found with values: {stats['found_with_values']}\n"
                msg += f"  - Coverage: {stats['coverage_pct']:.1f}%\n"
            
            return msg
        
        # Success case - handle multiple companies/years/metrics
        data_list = pipeline_result['data']
        count = pipeline_result['count']
        stats = pipeline_result.get('stats', {})
        
        msg = f"✓ Found {count} data points!\n"
        msg += f"📊 Coverage: {stats.get('coverage_pct', 0):.1f}%\n\n"
        
        # Group by company, then year
        from collections import defaultdict
        by_company = defaultdict(lambda: defaultdict(list))
        
        for data in data_list:
            ticker = data['ticker']
            year = data['year']
            by_company[ticker][year].append(data)
        
        # Format output
        for ticker in sorted(by_company.keys()):
            msg += f"{'='*60}\n"
            msg += f"📈 {ticker}\n"
            msg += f"{'='*60}\n"
            
            for year in sorted(by_company[ticker].keys()):
                msg += f"\n  Year {year}:\n"
                
                for data in by_company[ticker][year]:
                    value = data['value']

                    ## metric_name = data['metric'].replace('_', ' ')
                    metric_name = data['metric_label']

                    # Format value based on magnitude
                    if abs(value) >= 1_000_000_000:
                        formatted_value = f"${value/1_000_000_000:.2f}B"
                    elif abs(value) >= 1_000_000:
                        formatted_value = f"${value/1_000_000:.2f}M"
                    elif value < 0:
                        formatted_value = f"-${abs(value):,.2f}"
                    else:
                        formatted_value = f"${value:,.2f}"
                    
                    msg += f"    • {metric_name}: {formatted_value}\n"
            
            msg += "\n"
        
        return msg
    

"""
## Legacy INIT for reference. 

# LEGACY --- reference.
# def __init__(self, data_path: str, company_dim_path: Optional[str] = None):
#     self.extractor = FilterExtractor(company_dim_path=company_dim_path)
#     self.lookup = MetricLookup(data_path)    
"""
