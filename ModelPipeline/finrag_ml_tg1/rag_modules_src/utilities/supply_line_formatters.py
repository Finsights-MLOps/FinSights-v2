# ModelPipeline/finrag_ml_tg1/rag_modules_src/utilities/supply_line_formatters.py


from typing import Dict, Any, Optional
from collections import defaultdict


"""
Formatting utilities for Supply Line 1 - KPI fact table output.

Used to convert MetricPipeline results into compact, human-readable text
suitable for notebook display or LLM context.
"""

from typing import Dict, Any
from collections import defaultdict


def format_value_compact(value: float, metric_label: str = None) -> str:
    """
    Format financial values with intelligent type detection.
    
    Args:
        value: Numeric value
        metric_label: Optional metric name for type detection
    
    Returns:
        Formatted string: "$394.3B", "17%", "0.54", etc.
    
    Examples:
        >>> format_value_compact(394328000000, "Revenue")
        '$394.3B'
        >>> format_value_compact(17, "ROA % (Avg Assets)")
        '17.0%'
        >>> format_value_compact(0.545, "Debt-to-Assets")
        '0.55'
    """
    # ===================================================================
    # TYPE DETECTION: Percentage, Ratio, or Currency
    # ===================================================================
    
    if metric_label:
        metric_lower = metric_label.lower()
        
        # 1. PERCENTAGE METRICS (contains % symbol or "percent")
        if '%' in metric_label or 'percent' in metric_lower:
            return f"{value:.1f}%"
        
        # 2. RATIO METRICS (contains specific keywords)
        ratio_keywords = [
            '-to-',           # Debt-to-Assets, Debt-to-Equity
            'ratio',          # Any ratio
            'margin %',       # Already handled above but safe fallback
            'per unit',       # Per LP Unit metrics
            'per share',      # EPS-type metrics
        ]
        
        if any(keyword in metric_lower for keyword in ratio_keywords):
            # Ratios typically displayed as decimals (2-4 decimal places)
            if abs(value) < 10:
                return f"{value:.2f}"  # 0.54, 1.23
            else:
                return f"{value:.1f}"  # 12.3
    
    # ===================================================================
    # CURRENCY FORMATTING (default)
    # ===================================================================
    
    # Special case: Small absolute values without metric label context
    # Might be ratios/multipliers - show as decimal
    if metric_label is None and abs(value) < 100:
        if abs(value) < 1:
            return f"{value:.3f}"  # 0.545
        elif abs(value) < 10:
            return f"{value:.2f}"  # 5.43
        else:
            return f"{value:.1f}"  # 54.3
    
    # Standard currency formatting
    if abs(value) >= 1_000_000_000:
        return f"${value/1_000_000_000:.1f}B"
    elif abs(value) >= 1_000_000:
        return f"${value/1_000_000:.1f}M"
    elif abs(value) >= 1_000:
        return f"${value/1_000:.0f}K"
    else:
        return f"${value:.0f}"




def format_analytical_compact(
    raw_result: Dict[str, Any],
    entity_meta: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Format MetricPipeline result into a rich, LLM-friendly KPI block.

    Args:
        raw_result:
            Output from MetricPipeline.process().
            Expected keys on success:
              - 'success': bool
              - 'query': str (optional)
              - 'filters': {'tickers': [...], 'years': [...], 'metrics': [...]}
              - 'data': list of rows with keys:
                    'ticker', 'year', 'metric_label', 'value', 'found'
              - 'stats': optional {'found_with_values', 'total_combinations'}
        entity_meta:
            Optional dictionary with entity-level scope, e.g.:
              {
                "companies": ["MSFT", "NVDA"],
                "years": [2016, 2017, 2018, 2019, 2020],
                "sections": ["ITEM_7", "ITEM_1A"],
              }
            If provided, these are shown as "(entities)" lines in the header.

    Returns:
        Multi-line string containing:
          - Header with query + scope
          - Per-company, per-year KPI lines
        Empty string if success is False or no data rows.
    """
    # ------------------------------------------------------------------
    # Handle failure / empty cases
    # ------------------------------------------------------------------
    if not raw_result.get("success"):
        return ""

    data = raw_result.get("data", [])
    if not data:
        return ""

    # ------------------------------------------------------------------
    # Extract filters and stats
    # ------------------------------------------------------------------
    filters = raw_result.get("filters", {}) or {}
    tickers = filters.get("tickers", []) or []
    years = sorted(filters.get("years", []) or [])
    metric_labels = filters.get("metrics", []) or []

    stats = raw_result.get("stats", {}) or {}
    combos_found = stats.get("found_with_values")
    combos_total = stats.get("total_combinations")

    # Optional entity metadata from EntityAdapter
    entity_meta = entity_meta or {}
    ent_companies = entity_meta.get("companies") or []
    ent_years = entity_meta.get("years") or []
    ent_sections = entity_meta.get("sections") or []

    # ------------------------------------------------------------------
    # Group data: ticker → year → {metric_label: value}
    # ------------------------------------------------------------------
    grouped = defaultdict(lambda: defaultdict(dict))

    for item in data:
        if not item.get("found"):
            continue
        ticker = item.get("ticker")
        year = item.get("year")
        metric_label = item.get("metric_label") or item.get("metric")  # ✅ FIXED
        value = item.get("value")

        if ticker is None or year is None or metric_label is None or value is None:
            continue

        grouped[ticker][year][metric_label] = value

    if not grouped:
        return ""

    # ------------------------------------------------------------------
    # Build header
    # ------------------------------------------------------------------
    header_lines = [
        "══════════════════════════════════════════════════════════════════════",
        "KPI SNAPSHOT - METRIC PIPELINE OUTPUT",
        "══════════════════════════════════════════════════════════════════════",
        "",
    ]

    header_lines.append("Scope:")
    
    # Entity-side view (optional)
    if ent_companies or ent_years or ent_sections:
        if ent_companies:
            header_lines.append(
                "  Companies (entities): " + ", ".join(ent_companies)
            )
        if ent_years:
            header_lines.append(
                "  Years (entities):     " + ", ".join(str(y) for y in ent_years)
            )
        if ent_sections:
            header_lines.append(
                "  Sections (entities):  " + ", ".join(ent_sections)
            )

    # Metric filter view (always shown)
    header_lines.append(
        "  Companies (metrics):  " + (", ".join(tickers) or "(none)")
    )
    header_lines.append(
        "  Years (metrics):      " + (", ".join(str(y) for y in years) or "(none)")
    )
    header_lines.append(
        "  Metrics:              "
        + (", ".join(metric_labels) or "(none)")  # ✅ FIXED - already clean labels
    )

    if combos_found is not None and combos_total is not None:
        header_lines.append(
            f"  Coverage:             {combos_found}/{combos_total} metric combinations with values"
        )

    header_lines.append("")
    header_lines.append("DETAILS BY COMPANY AND YEAR")
    header_lines.append("")

    # ------------------------------------------------------------------
    # Build body
    # ------------------------------------------------------------------
    body_lines = []

    for ticker in sorted(grouped.keys()):
        body_lines.append(f"{ticker}:")
        years_for_t = sorted(grouped[ticker].keys())

        for year in years_for_t:
            metric_map = grouped[ticker][year]

            # Order metrics according to filter order, then any extras
            ordered_metrics = list(metric_labels)
            for m_label in metric_map.keys():
                if m_label not in ordered_metrics:
                    ordered_metrics.append(m_label)

            parts = []
            for m_label in ordered_metrics:
                if m_label not in metric_map:
                    continue
                value_str = format_value_compact(metric_map[m_label], m_label)  
                parts.append(f"{m_label}={value_str}")  # ✅ FIXED - use label directly

            if parts:
                body_lines.append(f"  {year}: " + ", ".join(parts))

        body_lines.append("")  # blank line between companies

    return "\n".join(header_lines + body_lines)




# ===============================================================================
# # ModelPipeline/finrag_ml_tg1/rag_modules_src/utilities/supply_line_formatters.py

# """
# Formatting utilities for Supply Line 1 - KPI fact table output.

# Used to convert MetricPipeline results into compact, human-readable text
# suitable for notebook display or LLM context.
# """

# from typing import Dict, Any
# from collections import defaultdict


# def format_value_compact(value: float) -> str:
#     """
#     Format financial values compactly using B/M/K suffixes.
    
#     Args:
#         value: Numeric value (e.g., revenue in dollars)
    
#     Returns:
#         Compact string like "$394.3B", "$1.5M", "$250K"
    
#     Examples:
#         >>> format_value_compact(394328000000)
#         '$394.3B'
#         >>> format_value_compact(1500000)
#         '$1.5M'
#     """
#     if abs(value) >= 1_000_000_000:
#         return f"${value/1_000_000_000:.1f}B"
#     elif abs(value) >= 1_000_000:
#         return f"${value/1_000_000:.1f}M"
#     elif abs(value) >= 1_000:
#         return f"${value/1_000:.0f}K"
#     else:
#         return f"${value:.0f}"


# def format_analytical_compact(raw_result: Dict[str, Any]) -> str:
#     """
#     Format MetricPipeline result into compact multi-line text.
    
#     Converts fact table query results into format like:
#         MSFT 2021: Revenue=$168.1B, TotalAssets=$333.8B
#         MSFT 2022: Revenue=$198.3B, TotalAssets=$364.8B
#         NVDA 2021: Revenue=$26.9B, TotalAssets=$28.8B
    
#     Args:
#         raw_result: Output from MetricPipeline.process()
#                    Expected keys: 'success', 'data', 'filters'
    
#     Returns:
#         Multi-line string with compact KPI data, or empty string if no data
#     """
#     # Handle failure cases
#     if not raw_result.get('success'):
#         return ""
    
#     data = raw_result.get('data', [])
#     if not data:
#         return ""
    
#     # Group by ticker → year → metrics
#     grouped = defaultdict(lambda: defaultdict(dict))
    
#     for item in data:
#         if item.get('found'):
#             ticker = item['ticker']
#             year = item['year']
#             metric = item['metric']
#             value = item['value']
            
#             # Shorten metric names for readability
#             metric_short = (
#                 metric.replace('income_stmt_', '')
#                       .replace('balance_sheet_', '')
#                       .replace('cash_flow_', '')
#                       .replace('_', '')
#             )
#             grouped[ticker][year][metric_short] = value
    
#     # Build output lines
#     lines = []
#     for ticker in sorted(grouped.keys()):
#         for year in sorted(grouped[ticker].keys()):
#             metrics = grouped[ticker][year]
#             metrics_str = ', '.join([
#                 f"{k}={format_value_compact(v)}" 
#                 for k, v in metrics.items()
#             ])
#             lines.append(f"{ticker} {year}: {metrics_str}")
    
#     return '\n'.join(lines)
# ===============================================================================