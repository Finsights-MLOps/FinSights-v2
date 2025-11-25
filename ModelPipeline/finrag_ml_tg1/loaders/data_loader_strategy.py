"""
Data Loader Strategy - Environment-agnostic data access

Design: Strategy pattern with two implementations:
  - LocalCacheLoader: For development (loads from data_cache/)
  - S3StreamingLoader: For Lambda (streams from S3, caches in /tmp)

Usage:
    from loaders.data_loader_factory import create_data_loader
    
    config = MLConfig()
    loader = create_data_loader(config)  # Auto-detects environment
    
    # Same interface, different backend
    df = loader.load_stage2_meta()
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional
import polars as pl
import logging
import tempfile
import os

logger = logging.getLogger(__name__)


class DataLoaderStrategy(ABC):
    """Abstract base for all data loading strategies"""
    
    @abstractmethod
    def load_stage2_meta(self) -> pl.DataFrame:
        """
        Load Stage 2 metadata table (73MB)
        Contains: sentenceID, sentence, sentence_pos, prev/next IDs, etc.
        """
        pass
    
    @abstractmethod
    def load_dimension_companies(self) -> pl.DataFrame:
        """Load companies dimension table (small, <1MB)"""
        pass
    
    @abstractmethod
    def load_dimension_sections(self) -> pl.DataFrame:
        """Load SEC sections dimension table (tiny, <100KB)"""
        pass
    
    @abstractmethod
    def get_sentences_by_ids(self, sentence_ids: List[str]) -> pl.DataFrame:
        """
        Fetch specific sentences by sentenceID.
        Optimized per strategy (in-memory filter vs S3 query).
        """
        pass

    @abstractmethod
    def load_kpi_fact_data(self) -> pl.DataFrame:
        """
        Load KPI fact data (company financial metrics from EDGAR)
        Contains: CIK, period, fiscal year, metric names, values, units, evidence
        Source: Structured extraction from 10-K filings
        """
        pass



class LocalCacheLoader(DataLoaderStrategy):
    """
    Local development loader - uses data_cache/ directory.
    Loads tables into memory once, reuses across calls.
    """
    
    def __init__(self, model_root: Path, config):
        self.model_root = model_root
        self.config = config
        
        # In-memory cache
        self._stage2_df: Optional[pl.DataFrame] = None
        self._dim_companies: Optional[pl.DataFrame] = None
        self._dim_sections: Optional[pl.DataFrame] = None
        self._kpi_fact_df: Optional[pl.DataFrame] = None
        
        logger.info(f"LocalCacheLoader initialized: {model_root}")
    
    def load_stage2_meta(self) -> pl.DataFrame:
        if self._stage2_df is None:
            cache_path = (
                self.model_root / 
                "finrag_ml_tg1/data_cache/meta_embeds/finrag_fact_sentences_meta_embeds.parquet"
            )
            logger.info(f"Loading Stage 2 Meta from: {cache_path}")
            self._stage2_df = pl.read_parquet(cache_path)
            logger.info(f"  ✓ Loaded {len(self._stage2_df):,} rows")
        return self._stage2_df
    
    def load_dimension_companies(self) -> pl.DataFrame:
        if self._dim_companies is None:
            cache_path = (
                self.model_root / 
                "finrag_ml_tg1/data_cache/dimensions/finrag_dim_companies_21.parquet"
            )
            self._dim_companies = pl.read_parquet(cache_path)
        return self._dim_companies
    
    def load_dimension_sections(self) -> pl.DataFrame:
        if self._dim_sections is None:
            cache_path = (
                self.model_root / 
                "finrag_ml_tg1/data_cache/dimensions/finrag_dim_sec_sections.parquet"
            )
            self._dim_sections = pl.read_parquet(cache_path)
        return self._dim_sections
    
    def get_sentences_by_ids(self, sentence_ids: List[str]) -> pl.DataFrame:
        """In-memory filter - fast for local dev"""
        df = self.load_stage2_meta()
        return df.filter(pl.col('sentenceID').is_in(sentence_ids))


    def load_kpi_fact_data(self) -> pl.DataFrame:
        """Load KPI fact data from local cache (metrics_fact/)"""
        if self._kpi_fact_df is None:
            kpi_path = (
                self.model_root / 
                "finrag_ml_tg1/data_cache/metrics_fact/KPI_FACT_DATA_EDGAR.parquet"
            )
            logger.info(f"Loading KPI Fact Data from: {kpi_path}")
            self._kpi_fact_df = pl.read_parquet(kpi_path)
            logger.info(f"  ✓ Loaded {len(self._kpi_fact_df):,} rows")
        return self._kpi_fact_df


class S3StreamingLoader(DataLoaderStrategy):
    """
    Lambda-optimized loader - streams from S3 with /tmp caching.
    """
    
    def __init__(self, config):
        self.config = config
        self.s3_client = config.get_s3_client()
        
        # Determine cache directory based on environment
        if os.getenv('AWS_LAMBDA_FUNCTION_NAME'):
            # Real Lambda: Use /tmp
            self._tmp_cache = Path('/tmp/finrag_cache')
        else:
            # Local testing (Windows/Mac/Linux): Use system temp
            self._tmp_cache = Path(tempfile.gettempdir()) / 'finrag_cache'
            print(f"[DEBUG: S3StreamingLoader] Using temp cache: {self._tmp_cache}")
            print(f"[DEBUG: S3StreamingLoader] This is if-else only for local testing, or mocking. In real Lambda: Uses /tmp/finrag_cache (standard Lambda location), In local testing: Uses C:\\Users\\...\\Temp\\finrag_cache")

        
        # Create cache directory
        self._tmp_cache.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache (for current invocation)
        self._stage2_df: Optional[pl.DataFrame] = None
        self._dim_companies: Optional[pl.DataFrame] = None
        self._dim_sections: Optional[pl.DataFrame] = None
        self._kpi_fact_df: Optional[pl.DataFrame] = None
        
        logger.info(f"S3StreamingLoader initialized (cache: {self._tmp_cache})")


    
    def load_stage2_meta(self) -> pl.DataFrame:
        if self._stage2_df is not None:
            return self._stage2_df
        
        # Check /tmp cache (warm invocations)
        cache_file = self._tmp_cache / "stage2_meta.parquet"
        if cache_file.exists():
            logger.info(f"Loading Stage 2 Meta from /tmp cache")
            self._stage2_df = pl.read_parquet(cache_file)
            logger.info(f"  ✓ Loaded {len(self._stage2_df):,} rows from cache")
            return self._stage2_df
        
        # Download from S3 (cold start)
        s3_uri = self.config.s3_uri(self.config.meta_embeds_path)
        logger.info(f"Downloading Stage 2 Meta from S3: {s3_uri}")
        
        df = pl.read_parquet(
            s3_uri,
            storage_options=self.config.get_storage_options()
        )
        
        # Cache in /tmp for next invocation
        df.write_parquet(cache_file, compression='zstd')
        logger.info(f"  ✓ Downloaded {len(df):,} rows, cached in /tmp")
        
        self._stage2_df = df
        return df
    
    def load_dimension_companies(self) -> pl.DataFrame:
        if self._dim_companies is None:
            # Small file - just stream, no caching needed
            
            # s3_key = "ML_EMBED_ASSETS/DIMENSIONS/finrag_dim_companies_21.parquet"
            s3_key = self.config.dimension_companies_path  
            s3_uri = f"s3://{self.config.bucket}/{s3_key}"
            self._dim_companies = pl.read_parquet(
                s3_uri,
                storage_options=self.config.get_storage_options()
            )
        return self._dim_companies
    
    def load_dimension_sections(self) -> pl.DataFrame:
        if self._dim_sections is None:
            
            # s3_key = "ML_EMBED_ASSETS/DIMENSIONS/finrag_dim_sec_sections.parquet"
            s3_key = self.config.dimension_sections_path
            s3_uri = f"s3://{self.config.bucket}/{s3_key}"
            self._dim_sections = pl.read_parquet(
                s3_uri,
                storage_options=self.config.get_storage_options()
            )
        return self._dim_sections
    
    def get_sentences_by_ids(self, sentence_ids: List[str]) -> pl.DataFrame:
        """
        For Lambda: Load full Stage 2 table (73MB acceptable).
        Future optimization: DuckDB S3 pushdown filter.
        """
        df = self.load_stage2_meta()
        return df.filter(pl.col('sentenceID').is_in(sentence_ids))
    

    def load_kpi_fact_data(self) -> pl.DataFrame:
        """
        Load KPI fact data from S3 with /tmp caching.
        """
        if self._kpi_fact_df is not None:
            return self._kpi_fact_df
        
        # Check /tmp cache (warm invocations)
        cache_file = self._tmp_cache / "kpi_fact_data.parquet"
        if cache_file.exists():
            logger.info(f"Loading KPI Fact Data from /tmp cache")
            self._kpi_fact_df = pl.read_parquet(cache_file)
            logger.info(f"  ✓ Loaded {len(self._kpi_fact_df):,} rows from cache")
            return self._kpi_fact_df
        
        # Download from S3 (cold start) - NOW CONFIG-DRIVEN
        s3_key = self.config.kpi_fact_data_path  # From config!
        s3_uri = f"s3://{self.config.bucket}/{s3_key}"
        logger.info(f"Downloading KPI Fact Data from S3: {s3_uri}")
        
        df = pl.read_parquet(
            s3_uri,
            storage_options=self.config.get_storage_options()
        )
        
        # Cache in /tmp for next invocation
        df.write_parquet(cache_file, compression='zstd')
        logger.info(f"  ✓ Downloaded {len(df):,} rows, cached in /tmp")
        
        self._kpi_fact_df = df
        return df