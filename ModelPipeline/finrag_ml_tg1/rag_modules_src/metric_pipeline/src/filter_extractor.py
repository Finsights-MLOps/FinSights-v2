"""
Extract structured filters (ticker, year, metric) from natural language queries
"""

# All sys hacks commented- remove. Import the v2 mappings
# from config.metric_mappings import METRIC_MAPPINGS
# import sys
# sys.path.append(str(Path(__file__).resolve().parents[2]))  # Add rag_modules_src to path

# ModelPipeline\finrag_ml_tg1\rag_modules_src\metric_pipeline\src\filter_extractor.py

from __future__ import annotations

import re
from typing import Optional, Dict, List
from pathlib import Path 
# old dont use: // # from finrag_ml_tg1.rag_modules_src.metric_pipeline.config.metric_mappings import ( METRIC_MAPPINGS, ) 

# ModelPipeline\finrag_ml_tg1\rag_modules_src\constants\metric_mapping_v2.py
from finrag_ml_tg1.rag_modules_src.constants.metric_mapping_v2 import ( METRIC_MAPPINGS, )
from finrag_ml_tg1.rag_modules_src.entity_adapter.company_universe import ( CompanyUniverse, ) 
from finrag_ml_tg1.rag_modules_src.entity_adapter.company_extractor import ( CompanyExtractor, ) 
from finrag_ml_tg1.rag_modules_src.entity_adapter.string_utils import simple_fuzzy_match


# ============================================================================
# MODULE-LEVEL PATH CONSTANT
# ============================================================================
# This path is relative to ModelPipeline root
DEFAULT_COMPANY_DIM_PATH = Path("finrag_ml_tg1/data_cache/dimensions/finrag_dim_companies_21.parquet")


class FilterExtractor:
    """Extract ticker, year, and metric from user queries"""
    
    def __init__(self, company_dim_path: Optional[str | Path] = None):
        """
        Initialize with company universe.
        
        Args:
            company_dim_path: Path to company dimension parquet file.
                            If None, uses DEFAULT_COMPANY_DIM_PATH constant.
                            Can be absolute or relative to current working directory.
        
        Raises:
            FileNotFoundError: If company dimension file doesn't exist
        """
        # Use default if not provided
        if company_dim_path is None:
            company_dim_path = DEFAULT_COMPANY_DIM_PATH
        
        # Convert to Path object
        company_dim_path = Path(company_dim_path)
        
        # Make absolute if relative (assumes cwd is ModelPipeline or contains it)
        if not company_dim_path.is_absolute():
            company_dim_path = company_dim_path.resolve()
        
        # Validate existence
        if not company_dim_path.exists():
            raise FileNotFoundError(
                f"Company dimension file not found: {company_dim_path}\n"
                f"Expected location: {DEFAULT_COMPANY_DIM_PATH}\n"
                f"Make sure ModelPipeline/ is your working directory or on sys.path"
            )
        
        # Initialize company universe and extractor
        self.company_universe = CompanyUniverse(dim_path=company_dim_path)
        self.company_extractor = CompanyExtractor(self.company_universe)
        
        # Keep metric map
        self.metric_map = METRIC_MAPPINGS
        
        print(f"✓ FilterExtractor initialized with {len(self.company_universe.tickers)} companies")
        print(f"  Using: {company_dim_path.name}")
    
    
    def extract(self, query: str) -> Dict[str, any]:
        """
        Main extraction method
        
        Args:
            query: User's natural language query
            
        Returns:
            Dictionary with tickers (list), years (list), metrics (list), and confidence
        """
        filters = {
            'tickers': self._extract_tickers(query),
            'years': self._extract_years(query),
            'metrics': self._extract_metrics(query),
            'query': query
        }
        
        # Calculate confidence based on how many filters were found
        has_tickers = len(filters['tickers']) > 0
        has_years = len(filters['years']) > 0
        has_metrics = len(filters['metrics']) > 0
        
        found_filters = sum([has_tickers, has_years, has_metrics])
        filters['confidence'] = found_filters / 3.0
        
        return filters



    def _extract_tickers(self, query: str) -> List[str]:
        """
        Extract ALL ticker symbols using CompanyExtractor
        
        This leverages the full company universe with:
        - Ticker matching (NVDA, AAPL)
        - CIK matching
        - Company name matching with fuzzy logic (NVIDIA -> NVDA)
        - Alias matching (apple -> AAPL, microsoft -> MSFT)
        
        Returns:
            List of ticker symbols (uppercase)
        """
        company_matches = self.company_extractor.extract(query)
        
        # Return the tickers list (already uppercase and deduplicated)
        return company_matches.tickers
    
    def _extract_years(self, query: str) -> List[int]:
        """
        Extract ALL years from query, including ranges
        
        Examples:
            "revenue in 2024" -> [2024]
            "2021 to 2023 earnings" -> [2021, 2022, 2023]
            "compare 2020, 2021, and 2022" -> [2020, 2021, 2022]
            "from 2015-2020" -> [2015, 2016, 2017, 2018, 2019, 2020]
        
        Returns:
            List of years (sorted, can be empty)
        """
        years_set = set()
        
        # Pattern for year ranges: "2015 to 2020", "2015-2020", "2015 - 2020"
        range_pattern = r'\b((19|20)\d{2})\s*(?:to|-|–|—)\s*((19|20)\d{2})\b'
        
        for match in re.finditer(range_pattern, query, re.IGNORECASE):
            start_year = int(match.group(1))
            end_year = int(match.group(3))
            
            # Ensure correct order
            if start_year > end_year:
                start_year, end_year = end_year, start_year
            
            # Expand range
            if 1950 <= start_year <= 2030 and 1950 <= end_year <= 2030:
                for year in range(start_year, end_year + 1):
                    years_set.add(year)
        
        # Pattern for individual years
        year_pattern = r'\b((19|20)\d{2})\b'
        
        for match in re.finditer(year_pattern, query):
            year = int(match.group(1))
            if 1950 <= year <= 2030:
                years_set.add(year)
        
        return sorted(list(years_set))
    
    def _extract_metrics(self, query: str) -> List[str]:
        """
        Map natural language to exact metric names with AUTOMATIC FUZZY MATCHING
        Handles ANY typo automatically without manual mappings
        
        Examples:
            "what is revenue" -> ["income_stmt_Revenue"]
            "prifit in 2024" -> ["income_stmt_Net Income"] (any typo!)
            "revenu and assts" -> ["income_stmt_Revenue", "balance_sheet_Total Assets"]
        """
        query_lower = query.lower()
        found_metrics = []
        
        # Sort by length to match longer phrases first
        sorted_metrics = sorted(self.metric_map.items(), 
                               key=lambda x: len(x[0]), 
                               reverse=True)
        
        # Track which parts of the query we've already matched
        matched_spans = []
        matched_words = set()  # Track words already matched
        
        # STEP 1: Try exact matching first (fast)
        for keyword, metric_name in sorted_metrics:
            pattern = r'\b' + re.escape(keyword) + r'\b'
            for match in re.finditer(pattern, query_lower):
                span = match.span()
                
                overlaps = any(
                    (span[0] < end and span[1] > start)
                    for start, end in matched_spans
                )
                
                if not overlaps:
                    if metric_name not in found_metrics:
                        found_metrics.append(metric_name)
                        matched_spans.append(span)
                        # Track the words that were matched
                        matched_words.update(keyword.split())
        
        # STEP 2: Fuzzy match on remaining unmatched words
        words = query_lower.split()
        
        for word in words:
            # Skip if already matched, too short, common word, or number
            if word in matched_words:
                continue
            if len(word) < 4:
                continue
            if word in ['the', 'and', 'what', 'how', 'was', 'is', 'are', 'were', 'from', 'with']:
                continue
            if word.isdigit():
                continue
            
            # Try fuzzy matching against all metric keywords
            best_match, score = simple_fuzzy_match(
                word,
                list(self.metric_map.keys()),
                threshold=0.70  # 70% similarity threshold
            )
            
            if best_match and score >= 70:
                metric_name = self.metric_map[best_match]
                if metric_name not in found_metrics:
                    found_metrics.append(metric_name)
                    matched_words.add(word)  # Mark as matched
        
        return found_metrics
    
    def is_valid(self, filters: Dict[str, any]) -> bool:
        """
        Check if extracted filters are sufficient for lookup
        """
        return all([
            len(filters.get('tickers', [])) > 0,
            len(filters.get('years', [])) > 0,
            len(filters.get('metrics', [])) > 0
        ])


# LEGACY FUNCTION - kept for backward compatibility with older code
def simple_fuzzy_match_legacy(word: str, choices: list, threshold: float = 0.8) -> tuple:
    """
    Simple fuzzy matching using Levenshtein distance (no external libs needed)
    Returns: (best_match, similarity_score)
    
    NOTE: This is the LEGACY version. New code should import from entity_adapter.string_utils
    """
    def levenshtein_distance(s1: str, s2: str) -> int:
        """Calculate edit distance between two strings"""
        if len(s1) < len(s2):
            return levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def similarity(s1: str, s2: str) -> float:
        """Calculate similarity ratio (0-1)"""
        distance = levenshtein_distance(s1.lower(), s2.lower())
        max_len = max(len(s1), len(s2))
        return 1 - (distance / max_len) if max_len > 0 else 0
    
    best_match = None
    best_score = 0
    
    for choice in choices:
        score = similarity(word, choice)
        if score > best_score:
            best_score = score
            best_match = choice
    
    if best_score >= threshold:
        return (best_match, best_score * 100)
    return (None, 0)



"""

import sys
from pathlib import Path

# cwd is ModelPipeline or inside it
current = Path.cwd()
if current.name != "ModelPipeline":
    for parent in [current] + list(current.parents):
        if parent.name == "ModelPipeline":
            model_root = parent
            break
    else:
        raise RuntimeError("Cannot find ModelPipeline root")
else:
    model_root = current

if str(model_root) not in sys.path:
    sys.path.insert(0, str(model_root))

print("✓ sys.path root:", model_root)

"""