"""
Query the metrics data using extracted filters

ModelPipeline\finrag_ml_tg1\rag_modules_src\metric_pipeline\src\metric_lookup.py

"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List


class MetricLookup:
    """Handle querying the metrics dataframe"""
    
    def __init__(self, data_path: str):
        """
        Initialize with path to metrics data
        
        Args:
            data_path: Path to JSON or Parquet file with metrics data
        """
        self.data_path = Path(data_path)
        self.df = None
        self._load_data()
        
    def _load_data(self):
        """Load and prepare the metrics dataframe - auto-detects JSON or Parquet"""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        # Auto-detect format FIRST
        suffix = self.data_path.suffix.lower()
        
        # THEN load based on format
        if suffix == '.json':
            self.df = pd.read_json(self.data_path)
        elif suffix == '.parquet':
            self.df = pd.read_parquet(self.data_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
        
        if self.df['year'].dtype != 'int64':
            self.df['year'] = self.df['year'].astype(int)
        
        print(
            f"✓ Loaded {len(self.df):,} records from {self.data_path.name} "
            f"({self.df['ticker'].nunique()} tickers, "
            f"{self.df['year'].min()}-{self.df['year'].max()})"  # Remove :.0f since it's int now
        )
        


    def query(self, ticker: str, year: int, metric: str) -> Optional[Dict[str, any]]:
        """
        Query for a specific metric value
        
        Args:
            ticker: Stock ticker symbol (e.g., "NVDA")
            year: Year (e.g., 2024)
            metric: Exact metric name (e.g., "income_stmt_Revenue")
            
        Returns:
            Dictionary with query results or None if not found
        """
        # Filter dataframe
        result = self.df[
            (self.df['ticker'] == ticker) &
            (self.df['year'] == year) &
            (self.df['metric_label'] == metric)
        ]
        
        if result.empty:
            return None
        
        value = result.iloc[0]['value']
        
        # Check for NaN values
        if pd.isna(value):
            return {
                'ticker': ticker,
                'year': year,
                'metric_label': metric,
                'value': None,
                'found': False,
                'reason': 'Value is NaN/missing in dataset'
            }
        
        return {
            'ticker': ticker,
            'year': year,
            'metric_label': metric,
            'value': value,
            'found': True
        }
    
    def get_available_years(self, ticker: str, metric: str) -> list:
        """Get all years where data is available for a ticker/metric combo"""
        results = self.df[
            (self.df['ticker'] == ticker) &
            (self.df['metric_label'] == metric) &
            (self.df['value'].notna())
        ]
        return sorted(results['year'].unique().tolist())
    
    def get_available_metrics(self, ticker: str, year: int) -> list:
        """Get all available metrics for a ticker/year combo"""
        results = self.df[
            (self.df['ticker'] == ticker) &
            (self.df['year'] == year) &
            (self.df['value'].notna())
        ]
        return sorted(results['metric_label'].unique().tolist())
    
    def query_multiple(self, ticker: str, year: int, metrics: List[str]) -> List[Dict[str, any]]:
        """
        Query for multiple metrics at once (LEGACY - single ticker/year)
        
        Args:
            ticker: Stock ticker symbol (e.g., "NVDA")
            year: Year (e.g., 2024)
            metrics: List of exact metric names
            
        Returns:
            List of dictionaries with query results
        """
        results = []
        
        for metric in metrics:
            result = self.query(ticker, year, metric)
            if result:
                results.append(result)
        
        return results
    
    # NEW METHODS FOR MULTI-COMPANY, MULTI-YEAR SUPPORT
    
    def query_batch(
        self, 
        tickers: List[str], 
        years: List[int], 
        metrics: List[str]
    ) -> List[Dict[str, any]]:
        """
        Query for multiple tickers × years × metrics (Cartesian product)
        
        Args:
            tickers: List of stock ticker symbols (e.g., ["NVDA", "AAPL", "MSFT"])
            years: List of years (e.g., [2021, 2022, 2023])
            metrics: List of exact metric names
            
        Returns:
            List of dictionaries with query results for all combinations
            
        Example:
            tickers = ["NVDA", "AAPL"]
            years = [2021, 2022]
            metrics = ["income_stmt_Revenue", "income_stmt_Net Income"]
            
            Returns 8 results (2 tickers × 2 years × 2 metrics)
        """
        results = []
        
        # Cartesian product: iterate through all combinations
        for ticker in tickers:
            for year in years:
                for metric in metrics:
                    result = self.query(ticker, year, metric)
                    if result:
                        results.append(result)
        
        return results
    
    def query_batch_optimized(
        self, 
        tickers: List[str], 
        years: List[int], 
        metrics: List[str]
    ) -> List[Dict[str, any]]:
        """
        Optimized batch query using pandas vectorized filtering
        Much faster than query_batch() for large queries
        
        Args:
            tickers: List of stock ticker symbols
            years: List of years
            metrics: List of exact metric names
            
        Returns:
            List of dictionaries with query results
        """
        if not tickers or not years or not metrics:
            return []
        
        # Single vectorized filter
        mask = (
            self.df['ticker'].isin(tickers) &
            self.df['year'].isin(years) &
            self.df['metric_label'].isin(metrics)
        )
        
        filtered_df = self.df[mask].copy()
        
        # Convert to list of dicts
        results = []
        for _, row in filtered_df.iterrows():
            value = row['value']
            
            if pd.isna(value):
                result = {
                    'ticker': row['ticker'],
                    'year': int(row['year']),
                    'metric_label': row['metric_label'],
                    'value': None,
                    'found': False,
                    'reason': 'Value is NaN/missing in dataset'
                }
            else:
                result = {
                    'ticker': row['ticker'],
                    'year': int(row['year']),
                    'metric_label': row['metric_label'],
                    'value': float(value),
                    'found': True
                }
            
            results.append(result)
        
        return results
    
    def get_batch_statistics(
        self, 
        tickers: List[str], 
        years: List[int], 
        metrics: List[str]
    ) -> Dict[str, any]:
        """
        Get statistics about data availability for a batch query
        Useful for understanding coverage before running expensive queries
        
        Returns:
            Dictionary with coverage statistics
        """
        total_combinations = len(tickers) * len(years) * len(metrics)
        
        results = self.query_batch_optimized(tickers, years, metrics)
        
        found_count = sum(1 for r in results if r.get('found', False))
        missing_count = len(results) - found_count
        not_in_db_count = total_combinations - len(results)
        
        return {
            'total_combinations': total_combinations,
            'results_in_db': len(results),
            'found_with_values': found_count,
            'found_but_nan': missing_count,
            'not_in_db': not_in_db_count,
            'coverage_pct': (found_count / total_combinations * 100) if total_combinations > 0 else 0
        }