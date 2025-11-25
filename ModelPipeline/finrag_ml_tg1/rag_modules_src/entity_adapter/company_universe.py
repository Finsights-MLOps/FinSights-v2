# ModelPipeline\finrag_ml_tg1\rag_modules_src\entity_adapter\company_universe.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Set, Tuple, List
import re
import logging

import pandas as pd

from .models import CompanyInfo

logger = logging.getLogger(__name__)


class CompanyUniverse:
    """
    Loads and indexes the company dimension table.

    Responsibilities:
    - Read the parquet (or JSON/CSV later) with company metadata.
    - Normalize and index by CIK (int), ticker, and company name.
    - Build a dynamic alias map (e.g. "apple" -> Apple Inc.) from the dim.
    - Provide fast membership / lookup for the extractor layer.

    This class is intentionally read-only after initialization.
    """

    def __init__(
        self,
        dim_path: str | Path,
        *,
        company_id_col: str = "company_id",
        cik_int_col: str = "cik_int",
        cik_str_col: str = "cik",
        ticker_col: str = "ticker",
        name_col: str = "company_name",
    ) -> None:
        """
        Parameters
        ----------
        dim_path:
            Absolute or relative path to the company dimension file.
            Currently expected to be a parquet file
            (e.g. ModelPipeline/finrag_ml_tg1/data_cache/dimensions/finrag_dim_companies_21.parquet).

        company_id_col, cik_int_col, cik_str_col, ticker_col, name_col:
            Column names in the dim table. Adjust if your schema changes.
        """
        self.dim_path = Path(dim_path)

        self.company_id_col = company_id_col
        self.cik_int_col = cik_int_col
        self.cik_str_col = cik_str_col
        self.ticker_col = ticker_col
        self.name_col = name_col

        if not self.dim_path.exists():
            raise FileNotFoundError(f"Company dimension file not found: {self.dim_path}")

        logger.info(f"Loading company dim from: {self.dim_path}")
        df = self._load_dim()
        logger.info(f"Loaded dim with {len(df)} rows and columns: {list(df.columns)}")
        self._validate_columns(df)

        # Primary indexes
        self._records: Dict[int, CompanyInfo] = {}          # keyed by cik_int
        self._by_ticker: Dict[str, CompanyInfo] = {}        # ticker -> info
        self._by_name_norm: Dict[str, CompanyInfo] = {}     # normalized name -> info

        # Alias index: alias token (e.g. "apple") -> list[CompanyInfo]
        self._by_alias: Dict[str, List[CompanyInfo]] = {}

        self._build_indexes(df)

        logger.info(
            "CompanyUniverse initialized: "
            f"{len(self._records)} companies, "
            f"{len(self._by_ticker)} tickers, "
            f"{len(self._by_alias)} alias tokens"
        )
        logger.debug(f"Alias tokens: {sorted(self._by_alias.keys())}")


    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        *,
        company_id_col: str = "company_id",
        cik_int_col: str = "cik_int",
        cik_str_col: str = "cik",
        ticker_col: str = "ticker",
        name_col: str = "company_name",
    ) -> CompanyUniverse:
        """
        Create CompanyUniverse from an already-loaded DataFrame.
        
        Used when DataLoader has already loaded the dimension table
        (e.g., from S3 in Lambda or local cache).
        
        Args:
            df: Pandas DataFrame with company dimension data
            company_id_col: Column name for company_id
            cik_int_col: Column name for integer CIK
            cik_str_col: Column name for string CIK
            ticker_col: Column name for ticker
            name_col: Column name for company name
        
        Returns:
            Initialized CompanyUniverse instance
        """
        # Create instance without calling __init__ (bypass path validation)
        instance = cls.__new__(cls)
        
        # Set column mappings
        instance.company_id_col = company_id_col
        instance.cik_int_col = cik_int_col
        instance.cik_str_col = cik_str_col
        instance.ticker_col = ticker_col
        instance.name_col = name_col
        
        # No file path (loaded from DataFrame)
        instance.dim_path = None
        
        # Validate columns exist
        logger.info(f"Building CompanyUniverse from DataFrame: {len(df)} rows")
        instance._validate_columns(df)
        
        # Initialize index structures
        instance._records = {}
        instance._by_ticker = {}
        instance._by_name_norm = {}
        instance._by_alias = {}
        
        # Build indexes from DataFrame
        instance._build_indexes(df)
        
        logger.info(
            "CompanyUniverse initialized from DataFrame: "
            f"{len(instance._records)} companies, "
            f"{len(instance._by_ticker)} tickers, "
            f"{len(instance._by_alias)} alias tokens"
        )
        
        return instance


    # ------------------------------------------------------------------ #
    # Public accessors
    # ------------------------------------------------------------------ #

    @property
    def ciks_int(self) -> Set[int]:
        """Set of all known integer CIKs."""
        return set(self._records.keys())

    @property
    def ciks(self) -> Set[int]:
        """Alias for ciks_int, for backward compatibility."""
        return self.ciks_int

    @property
    def ciks_str(self) -> Set[str]:
        """Set of all known zero-padded CIK strings."""
        return {rec.cik_str for rec in self._records.values()}

    @property
    def tickers(self) -> Set[str]:
        """Set of all known non-null uppercased tickers."""
        return set(self._by_ticker.keys())

    @property
    def names(self) -> Set[str]:
        """Set of all canonical company names."""
        return {rec.name for rec in self._records.values()}

    @property
    def alias_tokens(self) -> Set[str]:
        """
        Set of all alias tokens (lowercased, alphanumeric), e.g. "apple",
        "nvidia", "microsoft".
        """
        return set(self._by_alias.keys())

    # ---- lookups ------------------------------------------------------ #

    def get_by_cik_int(self, cik_int: int) -> Optional[CompanyInfo]:
        """Lookup by integer CIK."""
        return self._records.get(cik_int)

    def get_by_cik(self, cik_int: int) -> Optional[CompanyInfo]:
        """
        Alias for get_by_cik_int to keep extractor code simple.
        Expects an integer (cik_int), not the zero-padded string.
        """
        return self.get_by_cik_int(cik_int)

    def get_by_cik_str(self, cik_str: str) -> Optional[CompanyInfo]:
        """Lookup by zero-padded CIK string, if needed."""
        for info in self._records.values():
            if info.cik_str == cik_str:
                return info
        return None

    def get_by_ticker(self, ticker: str) -> Optional[CompanyInfo]:
        return self._by_ticker.get(ticker.upper())

    def get_by_name(self, name: str) -> Optional[CompanyInfo]:
        """
        Lookup by exact canonical name (case-insensitive).
        For free-text queries prefer using normalized name matching
        or alias matching in the extractor rather than calling this directly.
        """
        norm = self._normalize_name(name)
        return self._by_name_norm.get(norm)

    def get_by_alias_exact(self, alias_token: str) -> List[CompanyInfo]:
        """
        Lookup one or more companies by alias token (e.g. "apple").
        """
        return list(self._by_alias.get(alias_token, []))

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _load_dim(self) -> pd.DataFrame:
        """
        Load the dimension file into a DataFrame.

        If later you want to support JSON/CSV, extend this method
        based on file suffix.
        """
        suffix = self.dim_path.suffix.lower()

        if suffix == ".parquet":
            df = pd.read_parquet(self.dim_path)
        elif suffix in {".json", ".ndjson"}:
            df = pd.read_json(self.dim_path)
        elif suffix in {".csv"}:
            df = pd.read_csv(self.dim_path)
        else:
            raise ValueError(
                f"Unsupported company dim file format: {self.dim_path.name}"
            )

        return df

    def _validate_columns(self, df: pd.DataFrame) -> None:
        required = [
            self.company_id_col,
            self.cik_int_col,
            self.cik_str_col,
            self.ticker_col,
            self.name_col,
        ]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(
                f"Company dim file {self.dim_path} is missing required columns: {missing}. "
                f"Available columns: {list(df.columns)}"
            )

    def _build_indexes(self, df: pd.DataFrame) -> None:
        """
        Build internal maps from the raw DataFrame.

        - Ensure cik_int is an int.
        - Normalize tickers to upper-case strings.
        - Normalize names through a simple canonicalization function.
        - Build alias tokens like "apple", "nvidia", "microsoft".
        """
        df = df.copy()

        df[self.cik_int_col] = df[self.cik_int_col].astype("Int64")
        df = df[df[self.cik_int_col].notna()]

        df[self.company_id_col] = df[self.company_id_col].astype("Int64")
        df = df[df[self.company_id_col].notna()]

        df[self.ticker_col] = (
            df[self.ticker_col]
            .astype("string")
            .str.strip()
            .str.upper()
        )

        df[self.name_col] = df[self.name_col].astype("string").str.strip()
        df[self.cik_str_col] = df[self.cik_str_col].astype("string").str.strip()

        logger.info(f"Building indexes from dim with {len(df)} valid rows")

        for _, row in df.iterrows():
            cik_int_val = int(row[self.cik_int_col])
            company_id_val = int(row[self.company_id_col])

            name_val = str(row[self.name_col]) if pd.notna(row[self.name_col]) else ""
            if not name_val:
                logger.debug(f"Skipping row with empty name, cik_int={cik_int_val}")
                continue

            cik_str_val = str(row[self.cik_str_col]) if pd.notna(row[self.cik_str_col]) else ""
            ticker_val = (
                str(row[self.ticker_col]) if pd.notna(row[self.ticker_col]) else None
            )

            info = CompanyInfo(
                company_id=company_id_val,
                cik_int=cik_int_val,
                cik_str=cik_str_val,
                ticker=ticker_val if ticker_val else None,
                name=name_val,
            )

            self._records[cik_int_val] = info

            if info.ticker:
                self._by_ticker[info.ticker] = info

            norm_name = self._normalize_name(info.name)
            self._by_name_norm[norm_name] = info

            alias = self._compute_alias(info.name)
            logger.debug(
                f"Row name='{info.name}' -> norm='{norm_name}' -> alias='{alias}'"
            )
            if alias:
                self._by_alias.setdefault(alias, []).append(info)

    @staticmethod
    def _normalize_name(name: str) -> str:
        """
        Normalize company names for matching:
        - lowercase
        - strip leading/trailing whitespace
        - collapse internal whitespace
        """
        name = name.strip().lower()
        parts = name.split()
        return " ".join(parts)

    @staticmethod
    def _compute_alias(name: str) -> Optional[str]:
        """
        Compute a conservative alias token from a company name.

        Strategy:
        - normalize the name
        - take the first token
        - strip non-alphanumeric characters
        - lowercase
        - ignore very short or generic aliases
        """
        norm = CompanyUniverse._normalize_name(name)
        if not norm:
            return None

        first = norm.split()[0]
        alias = re.sub(r"[^a-z0-9]+", "", first.lower())

        if not alias or len(alias) < 2:
            return None

        generic = {"inc", "corp", "corporation", "company", "group", "co"}
        if alias in generic:
            return None

        return alias

    # ------------------------------------------------------------------ #
    # Utilities used by the extractor (optional but handy)
    # ------------------------------------------------------------------ #

    def iter_records(self) -> Iterable[CompanyInfo]:
        """Iterate over all companies in the universe."""
        return self._records.values()

    def find_by_normalized_substring(self, query_norm: str) -> Tuple[CompanyInfo, ...]:
        """
        Very simple name-based search: for a normalized query string, return
        all companies whose normalized full name appears as a substring.

        This is a fallback; the main path for user-friendly names is alias matching.
        """
        matches: List[CompanyInfo] = []
        for norm_name, info in self._by_name_norm.items():
            if norm_name and norm_name in query_norm:
                matches.append(info)
        return tuple(matches)
