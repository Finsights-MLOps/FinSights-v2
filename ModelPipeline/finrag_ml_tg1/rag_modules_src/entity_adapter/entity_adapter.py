
"""
ModelPipeline\finrag_ml_tg1\rag_modules_src\entity_adapter\entity_adapter.py

Entity adapter: top-level NL → {company, year, metric, section, risk_topic} extraction.

## Usage example: !

from rag_modules_src.entity_adapter.entity_adapter import EntityAdapter

adapter = EntityAdapter()
queries = [ ... ]

for q in queries:
    res = adapter.extract(q)
    EntityAdapter.debug_print(res)

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging

from .company_universe import CompanyUniverse
from .company_extractor import CompanyExtractor
from .year_extractor import YearExtractor
from .metric_adapter import MetricAdapter
from .section_universe import SectionUniverse
from .section_extractor import SectionExtractor
from .models import CompanyMatches, YearMatches, MetricMatches

from finrag_ml_tg1.loaders.data_loader_strategy import DataLoaderStrategy


logger = logging.getLogger(__name__)



@dataclass
class EntityExtractionResult:
    """
    Unified view of everything we can extract from a user query.

    This is what the RAG orchestrator / metric pipeline should see:

    - companies:   all company-level signals (CIKs, tickers, names)
    - years:       all year signals (past/current/future + warning)
    - metrics:     canonical metric IDs (income_stmt_Revenue, etc.)
    - sections:    all section candidates in canonical SEC form (ITEM_7, ITEM_1A, …)
    - primary_section: single best guess for routing / default filter
    - risk_topics: high-level semantic risk buckets (liquidity_credit, regulatory, …)
    """

    query: str

    companies: CompanyMatches
    years: YearMatches
    metrics: MetricMatches

    sections: List[str]
    primary_section: Optional[str]

    risk_topics: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """
        Lightweight dict view for logging / JSON / passing to other layers.
        Keeps structure but avoids dataclass objects.

        NOTE: does not attempt to be a full schema for external APIs,
        just a convenient, explicit snapshot.
        """
        return {
            "query": self.query,
            "companies": {
                "ciks_int": self.companies.ciks_int,
                "ciks_str": self.companies.ciks_str,
                "tickers": self.companies.tickers,
                "names": self.companies.names,
            },
            "years": {
                "years": self.years.years,
                "past_years": self.years.past_years,
                "current_years": self.years.current_years,
                "future_years": self.years.future_years,
                "warning": self.years.warning,
            },
            "metrics": {
                "metrics": self.metrics.metrics,
            },
            "sections": {
                "items": self.sections,
                "primary": self.primary_section,
            },
            "risk_topics": self.risk_topics,
        }


class EntityAdapter:
    """
    Top-level NL → {company, year, metric, section, risk_topic} adapter.

    Responsibilities:
      - Hide all the individual extractors behind a single .extract() call.
      - Own the default dimension paths (companies, sections).
      - Return a clean, typed EntityExtractionResult that downstream code
        can consume without knowing about fuzzy matching, alias maps, etc.
    """

    def __init__(
        self,
        data_loader: DataLoaderStrategy,
        *,
        # Allow dependency injection for advanced/testing use
        company_universe: CompanyUniverse | None = None,
        section_universe: SectionUniverse | None = None,
        company_extractor: CompanyExtractor | None = None,
        year_extractor: YearExtractor | None = None,
        metric_adapter: MetricAdapter | None = None,
        section_extractor: SectionExtractor | None = None,
    ) -> None:
        """
        Initialize EntityAdapter with DataLoader injection.
        
        Args:
            data_loader: DataLoader instance (LocalCacheLoader or S3StreamingLoader)
            company_universe: Optional pre-built CompanyUniverse (for testing)
            section_universe: Optional pre-built SectionUniverse (for testing)
            company_extractor: Optional pre-built CompanyExtractor (for testing)
            year_extractor: Optional pre-built YearExtractor (for testing)
            metric_adapter: Optional pre-built MetricAdapter (for testing)
            section_extractor: Optional pre-built SectionExtractor (for testing)
        
        Note:
            Dimension tables are loaded via DataLoader.
            Works in both local dev (filesystem) and Lambda (S3 streaming).
        """
        self.data_loader = data_loader
        
        logger.info("EntityAdapter initializing with DataLoader...")
        
        # ════════════════════════════════════════════════════════════════════
        # Load dimension tables via DataLoader
        # ════════════════════════════════════════════════════════════════════
        dim_companies_pl = self.data_loader.load_dimension_companies()
        dim_sections_pl = self.data_loader.load_dimension_sections()
        
        logger.info(
            f"  ✓ Loaded {len(dim_companies_pl)} companies\n"
            f"  ✓ Loaded {len(dim_sections_pl)} sections"
        )
        
        # Convert Polars → Pandas (CompanyUniverse/SectionUniverse use pandas)
        dim_companies_pd = dim_companies_pl.to_pandas()
        dim_sections_pd = dim_sections_pl.to_pandas()

        # ════════════════════════════════════════════════════════════════════
        # Initialize Company Universe & Extractor
        # ════════════════════════════════════════════════════════════════════
        if company_universe is None:
            company_universe = CompanyUniverse.from_dataframe(dim_companies_pd)
        if company_extractor is None:
            company_extractor = CompanyExtractor(company_universe)

        # ════════════════════════════════════════════════════════════════════
        # Initialize Section Universe & Extractor
        # ════════════════════════════════════════════════════════════════════
        if section_universe is None:
            section_universe = SectionUniverse.from_dataframe(dim_sections_pd)
        if section_extractor is None:
            section_extractor = SectionExtractor(section_universe)

        # ════════════════════════════════════════════════════════════════════
        # Initialize Year & Metric Adapters (no data loading needed)
        # ════════════════════════════════════════════════════════════════════
        if year_extractor is None:
            year_extractor = YearExtractor()
        if metric_adapter is None:
            metric_adapter = MetricAdapter(data_loader=data_loader)

        # ════════════════════════════════════════════════════════════════════
        # Store all initialized components
        # ════════════════════════════════════════════════════════════════════
        self.company_universe = company_universe
        self.section_universe = section_universe

        self.company_extractor = company_extractor
        self.year_extractor = year_extractor
        self.metric_adapter = metric_adapter
        self.section_extractor = section_extractor

        logger.info("EntityAdapter initialized successfully")




    # -------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------
    def extract(self, query: str) -> EntityExtractionResult:
        """
        Run the full NL → entity extraction stack for a single user query.

        This is the primary call that the RAG orchestrator / metric pipeline
        should use.

        Steps:
          1) Companies (CIKs, tickers, names)
          2) Years (past/current/future + warning)
          3) Metrics (canonical metric IDs via v2 mappings)
          4) Sections (sec_item_canonical like 'ITEM_7', 'ITEM_1A', ...)
          5) Risk topics (liquidity_credit, regulatory, etc.)
        """

        logger.info("EntityAdapter.extract: starting for query=%r", query)

        companies: CompanyMatches = self.company_extractor.extract(query)
        years: YearMatches = self.year_extractor.extract(query)
        metrics: MetricMatches = self.metric_adapter.extract(query)

        section_matches = self.section_extractor.extract_sections(query)
        risk_matches = self.section_extractor.extract_risk_topics(query)

        sections: List[str]
        primary_section: Optional[str]
        risk_topics: List[str]

        if section_matches is None:
            sections = []
            primary_section = None
        else:
            # normalize to a plain list[str] for the result
            sections = list(section_matches.items)
            primary_section = section_matches.primary

        if risk_matches is None:
            risk_topics = []
        else:
            risk_topics = list(risk_matches.topics)

        result = EntityExtractionResult(
            query=query,
            companies=companies,
            years=years,
            metrics=metrics,
            sections=sections,
            primary_section=primary_section,
            risk_topics=risk_topics,
        )

        logger.info(
            "EntityAdapter.extract: done. companies=%d, years=%d, metrics=%d, "
            "sections=%d, risk_topics=%d",
            len(companies.ciks_int),
            len(years.years),
            len(metrics.metrics),
            len(sections),
            len(risk_topics),
        )

        return result

    # -------------------------------------------------------------
    # Convenience for notebooks / dev mode
    # -------------------------------------------------------------
    @staticmethod
    def debug_print(result: EntityExtractionResult) -> None:
        """
        Pretty-print an EntityExtractionResult in the style
        you've been using in notebooks.
        """
        print("Query:", result.query)
        print("  CIKs int:     ", result.companies.ciks_int)
        print("  CIKs str:     ", result.companies.ciks_str)
        print("  Tickers:      ", result.companies.tickers)
        print("  Names:        ", result.companies.names)
        print("  Years:        ", result.years.years)
        print("  Metrics:      ", result.metrics.metrics)
        print("  Sections:     ", result.sections)
        print("  Primary sec:  ", result.primary_section)
        print("  Risk topics:  ", result.risk_topics)
        print("  Year warning: ", result.years.warning)
        print()



"""
Usage Example: Once more. From testfile.

from pathlib import Path
from rag_modules_src.entity_adapter.entity_adapter import EntityAdapter

adapter = EntityAdapter()
print("project_root:", adapter.project_root)
print("company_dim:", adapter.company_universe.dim_path)
print("sections_dim:", adapter.section_universe.dim_path)

queries = [
    "What was Nvidia's, Apple's and Amazon's revenue and net income in 2021, 2022, and 2023?",
    "Between Nvidia's gaming business, Apple's services, Teslsa, and MSFT, "
    "compare revenue, net income, operating cash flow, total assets and gross "
    "profit margin for 2019, 2020, 2023 and 2026.",
    
    "In Item 1A and Item 7, what liquidity and market risks did Tesla highlight in 2020?",
]

for q in queries:
    res = adapter.extract(q)
    EntityAdapter.debug_print(res)
"""


"""
## =======================================================================
LEGACY VERSION W NO DATALODER
## =======================================================================



    def __init__(
        self,
        company_dim_path: str | Path,
        section_dim_path: str | Path,
        *,
        # Allow dependency injection for advanced/testing use
        company_universe: CompanyUniverse | None = None,
        section_universe: SectionUniverse | None = None,
        company_extractor: CompanyExtractor | None = None,
        year_extractor: YearExtractor | None = None,
        metric_adapter: MetricAdapter | None = None,
        section_extractor: SectionExtractor | None = None,
    ) -> None:

        Initialize EntityAdapter with explicit dimension file paths.
        
        Args:
            company_dim_path: Full path to company dimension parquet file
            section_dim_path: Full path to section dimension parquet file
            company_universe: Optional pre-built CompanyUniverse (for testing)
            section_universe: Optional pre-built SectionUniverse (for testing)
            company_extractor: Optional pre-built CompanyExtractor (for testing)
            year_extractor: Optional pre-built YearExtractor (for testing)
            metric_adapter: Optional pre-built MetricAdapter (for testing)
            section_extractor: Optional pre-built SectionExtractor (for testing)
        
        Raises:
            FileNotFoundError: If dimension files don't exist

        # Convert to Path and validate existence
        company_dim_path = Path(company_dim_path)
        section_dim_path = Path(section_dim_path)
        
        if not company_dim_path.exists():
            raise FileNotFoundError(f"Company dimension file not found: {company_dim_path}")
        if not section_dim_path.exists():
            raise FileNotFoundError(f"Section dimension file not found: {section_dim_path}")
        
        logger.info(f"EntityAdapter using company_dim: {company_dim_path}")
        logger.info(f"EntityAdapter using section_dim: {section_dim_path}")

        # Company universe / extractor
        if company_universe is None:
            company_universe = CompanyUniverse(dim_path=company_dim_path)
        if company_extractor is None:
            company_extractor = CompanyExtractor(company_universe)

        # Section universe / extractor
        if section_universe is None:
            section_universe = SectionUniverse(dim_path=section_dim_path)
        if section_extractor is None:
            section_extractor = SectionExtractor(section_universe)

        # Year + metric adapters
        if year_extractor is None:
            year_extractor = YearExtractor()
        if metric_adapter is None:
            metric_adapter = MetricAdapter(company_dim_path=company_dim_path)

        self.company_universe = company_universe
        self.section_universe = section_universe

        self.company_extractor = company_extractor
        self.year_extractor = year_extractor
        self.metric_adapter = metric_adapter
        self.section_extractor = section_extractor

        logger.info("EntityAdapter initialized successfully")

## =======================================================================
## =======================================================================
"""