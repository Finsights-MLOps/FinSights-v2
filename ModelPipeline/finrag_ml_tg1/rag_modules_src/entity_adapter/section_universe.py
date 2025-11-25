# ModelPipeline\finrag_ml_tg1\rag_modules_src\entity_adapter\section_universe.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import logging
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SectionInfo:
    """
    Canonical representation of a single SEC 10-K section row.

    This wraps the finrag_dim_sec_sections dimension table and exposes the
    fields that are most relevant for retrieval and UI / debugging.
    """
    sec_item_canonical: str        # e.g. "ITEM_7", "ITEM_1A"
    section_code: str              # e.g. "PART_II_ITEM_7"
    section_name: str              # human-readable
    section_description: str
    section_category: str
    part_number: int
    priority: str                  # e.g. "P0", "P1", "P2", "P3"
    has_sub_items: bool


class SectionUniverse:
    """
    Universe of all sections known to FinRAG, backed by
    finrag_dim_sec_sections.parquet.

    This is the single source of truth for which sec_item_canonical values
    are valid for filtering in S3 Vectors.
    """

    def __init__(self, dim_path: Path) -> None:
        self.dim_path = Path(dim_path)
        self._by_canonical: Dict[str, SectionInfo] = {}
        self._load()


    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> SectionUniverse:
        """
        Create SectionUniverse from an already-loaded DataFrame.
        
        Used when DataLoader has already loaded the dimension table
        (e.g., from S3 in Lambda or local cache).
        
        Args:
            df: Pandas DataFrame with section dimension data
        
        Returns:
            Initialized SectionUniverse instance
        """
        # Create instance without calling __init__
        instance = cls.__new__(cls)
        
        # No file path (loaded from DataFrame)
        instance.dim_path = None
        
        # Initialize index structure
        instance._by_canonical = {}
        
        # Validate required columns
        required_cols = {
            "sec_item_canonical",
            "section_code",
            "section_name",
            "section_description",
            "section_category",
            "part_number",
            "priority",
            "has_sub_items",
        }
        missing = required_cols.difference(df.columns)
        if missing:
            raise ValueError(
                f"Section dimension DataFrame missing required columns: {sorted(missing)}"
            )
        
        # Build indexes from DataFrame
        logger.info(f"Building SectionUniverse from DataFrame: {len(df)} rows")
        
        by_canonical: Dict[str, SectionInfo] = {}
        for _, row in df.iterrows():
            canonical = str(row["sec_item_canonical"]).strip()
            if not canonical:
                continue

            info = SectionInfo(
                sec_item_canonical=canonical,
                section_code=str(row["section_code"]),
                section_name=str(row["section_name"]),
                section_description=str(row["section_description"]),
                section_category=str(row["section_category"]),
                part_number=int(row["part_number"]),
                priority=str(row["priority"]),
                has_sub_items=bool(row["has_sub_items"]),
            )
            by_canonical[canonical] = info

        instance._by_canonical = by_canonical
        
        logger.info(
            f"SectionUniverse initialized from DataFrame: {len(instance._by_canonical)} sections"
        )
        
        return instance


    # ------------------------------------------------------------------
    # Internal loading
    # ------------------------------------------------------------------
    def _load(self) -> None:
        if not self.dim_path.exists():
            raise FileNotFoundError(f"Section dimension not found: {self.dim_path}")

        df = pd.read_parquet(self.dim_path)

        required_cols = {
            "sec_item_canonical",
            "section_code",
            "section_name",
            "section_description",
            "section_category",
            "part_number",
            "priority",
            "has_sub_items",
        }
        missing = required_cols.difference(df.columns)
        if missing:
            raise ValueError(
                f"Section dimension missing required columns: {sorted(missing)}"
            )

        by_canonical: Dict[str, SectionInfo] = {}
        for _, row in df.iterrows():
            canonical = str(row["sec_item_canonical"]).strip()
            if not canonical:
                continue

            info = SectionInfo(
                sec_item_canonical=canonical,
                section_code=str(row["section_code"]),
                section_name=str(row["section_name"]),
                section_description=str(row["section_description"]),
                section_category=str(row["section_category"]),
                part_number=int(row["part_number"]),
                priority=str(row["priority"]),
                has_sub_items=bool(row["has_sub_items"]),
            )
            by_canonical[canonical] = info

        self._by_canonical = by_canonical
        logger.info(
            "SectionUniverse loaded %d sections from %s",
            len(self._by_canonical),
            self.dim_path,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get(self, sec_item_canonical: str) -> Optional[SectionInfo]:
        return self._by_canonical.get(sec_item_canonical)

    def has(self, sec_item_canonical: str) -> bool:
        return sec_item_canonical in self._by_canonical

    @property
    def all_canonical(self) -> List[str]:
        return sorted(self._by_canonical.keys())

    def filter_existing(self, items: Iterable[str]) -> List[str]:
        """
        Keep only valid sec_item_canonical values from the given iterable.
        """
        result = []
        for s in items:
            if s in self._by_canonical:
                result.append(s)
        # preserve input order; caller can sort if desired
        return result
