

import json
import boto3
from datetime import datetime
import os
import re
import time
from typing import List, Dict
import numpy as np
import pandas as pd
from edgar import set_identity, Company, MultiFinancials

try:
    from .gaap_aliases import GAAP_ALIASES
except ImportError:
    from gaap_aliases import GAAP_ALIASES

from typing import Set, Dict, Any, Optional

import pandas as pd
import smtplib
from email.message import EmailMessage
# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------
EDGAR_IDENTITY = os.getenv("EDGAR_IDENTITY", "your-email@example.com")
set_identity(EDGAR_IDENTITY)

START_YEAR = 2000
END_YEAR   = 2025
FORMS_10K  = {"10-K", "10-K/A"}

# GAAP concepts to query (keys from your alias dict)
CONCEPTS: List[str] = list(GAAP_ALIASES.keys())

EXPECTED_CIKS = [
    "0001065280",
    "0001326801",
    "0001045810",
    "0000104169",
    "0000814585",
    "0001318605",
    "0001018724",
    "0001141391",
    "0000890926",
    "0000059478",
    "0000789019",
    "0001403161",
    "0000320193",
    "0001341439",
    "0001652044",
    "0001276520",
    "0000909832",
    "0001273813",
    "0000200406",
    "0000034088",
    "0000813762",
]


# Aliases for MultiFinancials statement labels
ALIASES = {
    # Income statement
    "Revenue": [
        "Revenue", "Revenues", "Sales Revenue, Net", "SalesRevenueNet", "SalesRevenueServicesNet","Product Revenue", "Contract Revenue",    "Sales and other operating revenue",
    "Total revenues and other income",
    "Sales and Operating Revenue",
    "Sales and Operating Revenues"
    ],
    "OperatingIncome": [
        "Operating Income", "Operating Income (Loss)", "OperatingIncomeLoss","Income Before Tax from Continuing Operations", 
        "Income Before Tax"
    ],
    "NetIncome": [
        "Net Income", "Net Income (Loss)", "NetIncomeLoss", "Net Income from Continuing Operations"
    ],
    # Balance sheet
    "TotalAssets": [
        "Total Assets", "Assets"
    ],
    "TotalLiabilities": [
        "Total Liabilities", "Liabilities", "Total Liabilities and Stockholders' Equity"
    ],
    "Equity": [
        "Stockholders’ Equity", "Stockholders' Equity", "Total equity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        "StockholdersEquity","Total Stockholders' Equity",
        "Stockholders' Equity before Treasury Stock"
    ],
    "CurrentAssets": [
        "Current Assets", "Assets, Current", "AssetsCurrent","Total Current Assets"
    ],
    "CurrentLiabilities": [
        "Current Liabilities", "Liabilities, Current", "LiabilitiesCurrent","Total Current Liabilities"
    ],
    "Inventory": [
        "Inventory, Net", "InventoryNet", "Inventory","Crude oil, products and merchandise",
    "Materials and supplies"
    ],
    # Cash flow
    "CFO": ["Net Cash from Operating Activities",
        "Net Cash Provided by (Used in) Operating Activities",
        "Net Cash Provided by Operating Activities",
        "NetCashProvidedByUsedInOperatingActivities","Net Cash from Operating Activities"
    ],
    "CapEx": [
        "Payments to Acquire Property, Plant and Equipment",
        "Purchases of property and equipment",
        "Capital Expenditures",
        "PaymentsToAcquirePropertyPlantAndEquipment","Payments for Property, Plant and Equipment",
        "Purchases of property, equipment and technology",
        "Purchases of property, equipment, technology and intangible assets",
    ],
}


DERIVED_METRIC_METADATA = {
    "Current Ratio": {"canonical_key": "current_ratio", "unit": "ratio"},
    "Quick Ratio": {"canonical_key": "quick_ratio", "unit": "ratio"},
    "Debt-to-Assets": {"canonical_key": "debt_to_assets", "unit": "ratio"},
    "Debt-to-Equity": {"canonical_key": "debt_to_equity", "unit": "ratio"},
    "Free Cash Flow": {"canonical_key": "free_cash_flow", "unit": "USD"},
    "Operating Cash Flow Ratio": {"canonical_key": "operating_cf_ratio", "unit": "ratio"},
    "Operating Margin %": {"canonical_key": "operating_margin", "unit": "percent"},
    "Net Profit Margin %": {"canonical_key": "net_profit_margin", "unit": "percent"},
    "ROA % (Avg Assets)": {"canonical_key": "roa", "unit": "percent"},
    "ROE % (Avg Equity)": {"canonical_key": "roe", "unit": "percent"},
}

EXPECTED_DERIVED_LABELS = list(DERIVED_METRIC_METADATA.keys())

# 1) Your full expected derived metrics (labels)
EXPECTED_DERIVED_LABELS = {
    "Net Profit Margin %",
    "Operating Margin %",
    "ROA % (Avg Assets)",
    "ROE % (Avg Equity)",
    "Current Ratio",
    "Quick Ratio",
    "Debt-to-Equity",
    "Debt-to-Assets",
    "Free Cash Flow",
    "Operating Cash Flow Ratio",
}

# 2) Metrics that are NOT applicable per CIK (example)
#    You can extend this dict as you decide which companies are financial/insurance/etc.
NOT_APPLICABLE_BY_CIK = {
    # --------------------------------------------------------------
    # 0000890926 — Radian Group Financial / Insurance profile (no liquidity ratios, no FCF)
    # --------------------------------------------------------------
    "0000890926": {
        "Current Ratio",
        "Quick Ratio",
        "Free Cash Flow",
        "Operating Cash Flow Ratio",
        "ROA % (Avg Assets)",      # equity/asset behavior abnormal — insurer-like
        "ROE % (Avg Equity)",
        "Net Profit Margin %",    # optional — revenue breakouts missing in some yrs
    },

    # --------------------------------------------------------------
    # 0000813762 — Icahn Enterprise
    # Missing CA/CL/Inventory structurally in many years → skip liquidity ratios
    # --------------------------------------------------------------
    "0000813762": {
        "Current Ratio",
        "Quick Ratio",
        "Free Cash Flow",         # CapEx missing in many years
        "Operating Cash Flow Ratio",
    },

    # --------------------------------------------------------------
    # 0001141391 — Mastercard (MA)
    # No inventory → Quick Ratio always invalid
    # --------------------------------------------------------------
    "0001141391": {
        "Quick Ratio",
        "Net Profit Margin %",
        "ROA % (Avg Assets)",
        "ROE % (Avg Equity)",
        "Quick Ratio",
        "Free Cash Flow",             # since CapEx not findable as GAAP
        "Operating Cash Flow Ratio",
    },
    
    # --------------------------------------------------------------
    # 0001018724 — Amazon (AMZN)
    # TotalLiabilities missing due to reporting structure (lease liabilities split)
    # Debt-based metrics are invalid
    # --------------------------------------------------------------
    "0001018724": {
        "Debt-to-Equity",
        "Debt-to-Assets",
    },

    # --------------------------------------------------------------
    # 0000059478 — Eli Lilly (LLY)
    # No liability field present → debt-based metrics not computable
    # --------------------------------------------------------------
    "0000059478": {
        "Debt-to-Equity",
        "Debt-to-Assets",
    },
        # Assured Guaranty Ltd — financial guaranty insurer
    "0001273813": {
        "Current Ratio",
        "Quick Ratio",
        "Free Cash Flow",
        "Operating Cash Flow Ratio",
    },
    "0001276520": {
        "Current Ratio",
        "Quick Ratio",
        "Free Cash Flow",
        "Operating Cash Flow Ratio",
        "Operating Margin %",     # insurer – not reported as operating income
        "ROA % (Avg Assets)",     # optional – can be kept if assets exist
        "ROE % (Avg Equity)"      # optional – but safe to exclude also
    },
      "0000059478": {
        "Debt-to-Equity",
        "Debt-to-Assets",
    }
}



# ---------------------------------------------------
# HELPERS
# ---------------------------------------------------

def _cols_to_year_index(cols):
    yrs = pd.to_datetime(cols, errors="coerce").year
    yrs = [
        y if not pd.isna(y)
        else (int(re.search(r"(20\d{2})", str(c)).group(1)) if re.search(r"(20\d{2})", str(c)) else np.nan)
        for c, y in zip(cols, yrs)
    ]
    return pd.Index(yrs, dtype="float64")

def _clean_numeric_series(s: pd.Series) -> pd.Series:
    # convert to string and normalize common SEC formats
    s = (
        s.astype(str)
         .str.replace(",", "", regex=False)                 # 1,234 -> 1234
         .str.replace(r"\((.*)\)", r"-\1", regex=True)      # (123) -> -123
         .str.replace("—", "", regex=False)                 # em dash -> ""
         .str.replace("–", "", regex=False)                 # en dash -> ""
         .str.replace("$", "", regex=False)                 # remove $
         .str.strip()
    )
    return pd.to_numeric(s, errors="coerce")

def _pad_cik(cik) -> str:
    s = "".join(ch for ch in str(cik) if ch.isdigit())
    return s.zfill(10)

def _raw_cik(cik) -> str:
    s = "".join(ch for ch in str(cik) if ch.isdigit())
    return s.lstrip("0") or "0"   # "0000200406" -> "200406"

def strip_namespace(metric: str) -> Optional[str]:
    if not isinstance(metric, str):
        return None
    if ":" in metric:
        return metric.split(":", 1)[1]
    return metric


def normalize_metric_key(raw_metric: str) -> Optional[str]:
    """
    raw_metric: us-gaap:NetIncomeLoss → canonical_key via GAAP_ALIASES
    """
    code = strip_namespace(raw_metric)
    if not code:
        return None
    info = GAAP_ALIASES.get(code)
    return info["canonical_key"] if info else None


def normalize_metric_label(raw_metric: str) -> str:
    """
    raw_metric: us-gaap:NetIncomeLoss → "Net Income" or fallback
    """
    code = strip_namespace(raw_metric)
    if not code:
        return ""
    info = GAAP_ALIASES.get(code)
    if info:
        return info["human_label"]

    # Fallback: split CamelCase
    s = re.sub(r"(?<!^)(?=[A-Z])", " ", code)
    return " ".join(s.split())


def _avg_series(s: pd.Series) -> pd.Series:
    return (s + s.shift(1)) / 2


def _sdiv(a: Optional[pd.Series], b: Optional[pd.Series]) -> pd.Series:
    if a is None or b is None:
        return pd.Series(dtype="float64")
    out = a.astype("float64") / b.astype("float64")
    return out.replace([np.inf, -np.inf], np.nan)


def _normalize_stmt_df(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return df
    if "label" not in df.columns:
        if "Label" in df.columns:
            df = df.rename(columns={"Label": "label"})
        elif df.index.name == "label":
            df = df.reset_index()
    return df


def _row_to_year_series(stmt_df: Optional[pd.DataFrame], label_aliases: List[str]) -> pd.Series:
    if stmt_df is None or stmt_df.empty:
        return pd.Series(dtype="float64")

    labels_norm = stmt_df["label"].astype(str).str.strip().str.lower()

    for alias in label_aliases:
        alias_norm = alias.strip().lower()
        row = stmt_df.loc[labels_norm == alias_norm]

        if not row.empty:
            s = row.iloc[0].drop(labels=["label"])

            years = _cols_to_year_index(s.index)
            s.index = years

            s = _clean_numeric_series(s)
            s = s[s.index.notna()]

            # keep last per year
            s = s.groupby(level=0).last()
            return s

    return pd.Series(dtype="float64")


def _sum_rows_to_year_series(stmt_df: Optional[pd.DataFrame], label_aliases: List[str]) -> pd.Series:
    if stmt_df is None or stmt_df.empty:
        return pd.Series(dtype="float64")

    labels_norm = stmt_df["label"].astype(str).str.strip().str.lower()
    pieces = []

    for alias in label_aliases:
        alias_norm = alias.strip().lower()
        row = stmt_df.loc[labels_norm == alias_norm]

        if not row.empty:
            s = row.iloc[0].drop(labels=["label"])
            years = _cols_to_year_index(s.index)
            s.index = years
            s = _clean_numeric_series(s)
            s = s[s.index.notna()]
            s = s.groupby(level=0).last()
            pieces.append(s)

    if not pieces:
        return pd.Series(dtype="float64")

    # sum across rows per year
    out = pd.concat(pieces, axis=1).sum(axis=1, min_count=1)
    return out


def _get_stmt_df(mf: MultiFinancials, candidates: List[str]) -> Optional[pd.DataFrame]:
    for name in candidates:
        if hasattr(mf, name):
            stmt = getattr(mf, name)()
            if hasattr(stmt, "to_dataframe"):
                # No include_dimensions arg to avoid the error you saw
                return stmt.to_dataframe()
    return None


def compute_operating_income(inc_df):
    """
    Try to extract Operating Income. If missing, fallback to:

    Operating Income ≈ Gross Profit - SG&A - R&D
    """

    # 1) preferred exact aliases
    op = _row_to_year_series(inc_df, [
        "Operating Income",
        "Operating Income (Loss)",
        "OperatingIncomeLoss",
    ])
    if not op.empty:
        return op

    # 2) fallback: use "Income Before Tax..." when available
    op = _row_to_year_series(inc_df, [
        "Income Before Tax from Continuing Operations",
        "Income Before Tax",
        "Earnings before provision for taxes on income",
    ])
    if not op.empty:
        return op

    # 3) last fallback: manually reconstruct from line items
    gross_profit = _row_to_year_series(inc_df, ["Gross Profit"])
    sga = _row_to_year_series(inc_df, ["Selling, General and Administrative Expense"])
    rnd = _row_to_year_series(inc_df, ["Research and development expense",
                                       "Research and Development Expense"])

    # operating income ≈ gross profit - sga - rnd
    manual = pd.concat([gross_profit, sga, rnd], axis=1)
    manual.columns = ["gp", "sga", "rnd"]
    manual = manual.dropna()
    if not manual.empty:
        return manual["gp"] - manual["sga"] - manual["rnd"]

    return pd.Series(dtype="float64")

# ---------------------------------------------------
# 1) GAAP FACTS via EntityFacts.query()
# ---------------------------------------------------
def get_total_liabilities_series(bs_df: pd.DataFrame) -> pd.Series:
    """
    Prefer a total-liabilities row if available.
    Otherwise sum current + noncurrent liabilities.
    Safe across companies (won't double count).
    """

    # 1) Try direct totals first
    totals_priority = [
        "Total Liabilities",
        "Liabilities",
        "Total liabilities",  # just in case casing varies
    ]
    liab_total = _row_to_year_series(bs_df, totals_priority)
    if not liab_total.empty and liab_total.notna().any():
        return liab_total

    # 2) Else sum split totals
    split_aliases = [
        "Total Current Liabilities",
        "Total Non Current Liabilities",
        "Total Noncurrent Liabilities",
        "Total current liabilities",
        "Total non current liabilities",
        "Total noncurrent liabilities",
    ]
    liab_split = _sum_rows_to_year_series(bs_df, split_aliases)
    return liab_split

def fetch_10k_facts_for_analytical_layer(cik: str) -> pd.DataFrame:
    """
    Uses EntityFacts.query() to build a GAAP facts table:

    cik, ticker, year,
    metric_gaap, metric_code, metric_key, metric_label,
    metric_type="gaap",
    value, unit, form, filed_date, accession_no
    """
    cik10 = _pad_cik(cik)
    cik_raw = _raw_cik(cik)
    co = Company(cik_raw)
    facts = co.facts
    ticker = co.get_ticker()

    rows = []

    for concept in CONCEPTS:
        for year in range(START_YEAR, END_YEAR + 1):
            q = (
                facts.query()
                     .by_concept(concept)           # e.g. "NetIncomeLoss"
                     .by_form_type(FORMS_10K)       # 10-K / 10-K/A
                     .by_fiscal_year(year)
                     .sort_by("filing_date", ascending=True)
            )

            dfp = q.to_dataframe(
                "concept", "numeric_value", "unit",
                "fiscal_year", "fiscal_period",
                "filing_date", "form_type", "accession"
            )
            if dfp is None or dfp.empty:
                continue

            # ✅ keep ONLY the last-filed metric for that year & concept
            dfp = dfp.sort_values(["fiscal_year", "filing_date"]).tail(1)

            for _, r in dfp.iterrows():
                val = r.get("numeric_value", r.get("value"))
                if val is None:
                    continue

                metric_code = r["concept"]           # ex: "NetIncomeLoss"
                metric_gaap = metric_code
                metric_key  = normalize_metric_key(metric_gaap)
                metric_lbl  = normalize_metric_label(metric_gaap)

                rows.append({
                    "cik": cik10,
                    "ticker": ticker,
                    "year": int(r["fiscal_year"]),
                    "metric_gaap": metric_gaap,
                    "metric_code": metric_code,
                    "metric_key": metric_key,
                    "metric_label": metric_lbl,
                    "metric_type": "gaap",
                    "value": float(val),
                    "unit": r.get("unit"),
                    "form": r.get("form_type"),
                    "filed_date": str(r.get("filing_date")),
                    "accession_no": r.get("accession"),
                })

    if not rows:
        return pd.DataFrame(columns=[
            "cik", "ticker", "year",
            "metric_gaap", "metric_code", "metric_key", "metric_label",
            "metric_type",
            "value", "unit", "form", "filed_date", "accession_no",
        ])

    return pd.DataFrame(rows)


# ---------------------------------------------------
# 2) DERIVED KPIs via MultiFinancials
# ---------------------------------------------------

def compute_core_kpis_for_company(cik: str, n_years: int = 8) -> pd.DataFrame:
    """
    Use MultiFinancials to compute 10 derived KPIs.
    Matches GAAP schema but:

      metric_gaap, metric_code, form, filed_date, accession_no are None
      metric_type = "derived"
    """
    cik10 = _pad_cik(cik)
    cik_raw = _raw_cik(cik)

    co = Company(cik_raw)
    ticker = co.get_ticker()

    filings = co.get_filings(form="10-K").head(n_years)
    if filings.empty:
        return pd.DataFrame(columns=[
            "cik", "ticker", "year",
            "metric_gaap", "metric_code", "metric_key", "metric_label",
            "metric_type",
            "value", "unit", "form", "filed_date", "accession_no",
        ])

    mf = MultiFinancials.extract(filings)

    inc = _normalize_stmt_df(_get_stmt_df(mf, ["income_statement", "income"]))
    bs  = _normalize_stmt_df(_get_stmt_df(mf, ["balance_sheet", "balance"]))
    cf  = _normalize_stmt_df(_get_stmt_df(mf, ["cashflow_statement", "cash_flow", "cashflow"]))

    rev = _sum_rows_to_year_series(inc, ALIASES["Revenue"])
    opinc = compute_operating_income(inc)
    net   = _row_to_year_series(inc, ALIASES["NetIncome"])

    assets = _row_to_year_series(bs,  ALIASES["TotalAssets"])
    eq     = _row_to_year_series(bs,  ALIASES["Equity"])
    liab   = get_total_liabilities_series(bs)
    if (liab is None or liab.empty) and bs is not None and not bs.empty:
        tlse = _row_to_year_series(bs, ["Total Liabilities and Stockholders' Equity"])
        if tlse is not None and not tlse.empty and eq is not None and not eq.empty:
            liab = tlse - eq
        else:
            liab = pd.Series(dtype="float64")
    ca     = _row_to_year_series(bs,  ALIASES["CurrentAssets"])
    cl     = _row_to_year_series(bs,  ALIASES["CurrentLiabilities"])
    inv    = _row_to_year_series(bs,  ALIASES["Inventory"])

    cfo   = _row_to_year_series(cf,  ALIASES["CFO"])
    capex = _row_to_year_series(cf,  ALIASES["CapEx"]).abs()  # positive spend

    avg_assets = _avg_series(assets)
    avg_equity = _avg_series(eq)

    idx = rev.index.union(assets.index).union(eq.index).union(cl.index).union(cfo.index)
    metrics = pd.DataFrame(index=idx)

    metrics["Net Profit Margin %"]       = _sdiv(net, rev) * 100
    metrics["Operating Margin %"]        = _sdiv(opinc, rev) * 100
    metrics["ROA % (Avg Assets)"]        = _sdiv(net, avg_assets) * 100
    metrics["ROE % (Avg Equity)"]        = _sdiv(net, avg_equity) * 100
    metrics["Current Ratio"]             = _sdiv(ca, cl)
    metrics["Quick Ratio"]               = _sdiv(ca - inv, cl)
    metrics["Debt-to-Equity"]            = _sdiv(liab, eq)
    metrics["Debt-to-Assets"]            = _sdiv(liab, assets)
    metrics["Free Cash Flow"]            = cfo - capex
    metrics["Operating Cash Flow Ratio"] = _sdiv(cfo, cl)

    long_kpi = (
        metrics.reset_index()
               .rename(columns={"index": "year"})
               .melt(id_vars=["year"], var_name="metric_label", value_name="value")
               .dropna(subset=["value"], how="all")
               .sort_values(["year", "metric_label"])
               .reset_index(drop=True)
    )

    # Attach metadata
    long_kpi["cik"]    = cik10
    long_kpi["ticker"] = ticker

    # Map to canonical_key + unit from DERIVED_METRIC_METADATA
    long_kpi["metric_key"]  = long_kpi["metric_label"].apply(
        lambda lbl: DERIVED_METRIC_METADATA.get(lbl, {}).get("canonical_key")
    )
    long_kpi["unit"]        = long_kpi["metric_label"].apply(
        lambda lbl: DERIVED_METRIC_METADATA.get(lbl, {}).get("unit")
    )

    long_kpi["metric_gaap"]   = None
    long_kpi["metric_code"]   = None
    long_kpi["metric_type"]   = "derived"
    long_kpi["form"]          = None
    long_kpi["filed_date"]    = None
    long_kpi["accession_no"]  = None

    cols = [
        "cik", "ticker", "year",
        "metric_gaap", "metric_code", "metric_key", "metric_label",
        "metric_type",
        "value", "unit", "form", "filed_date", "accession_no",
    ]
    return long_kpi[cols]


# ---------------------------------------------------
# 3) DRIVER – BUILD ONE PARQUET WITH BOTH
# ---------------------------------------------------
def build_analytical_layer(
    ciks: List[str],
    out_parquet: str,
    between: float = 1.5,
    n_years_derived: int = 2,
):
    """
    For each CIK:
      - fetch GAAP facts via EntityFacts.query() (controlled by START_YEAR / END_YEAR)
      - compute derived KPIs via MultiFinancials (last n_years_derived 10-Ks)
    Then concatenate and save to one parquet.
    """
    frames = []

    for i, cik in enumerate(ciks, start=1):
        print(f"[{i}/{len(ciks)}] CIK {cik}")
        try:
            df_gaap = fetch_10k_facts_for_analytical_layer(cik)
            df_kpis = compute_core_kpis_for_company(cik, n_years=n_years_derived)
            if not df_gaap.empty:
                frames.append(df_gaap)
            if not df_kpis.empty:
                frames.append(df_kpis)
        except Exception as e:
            print(f"  ✖ error for {cik}: {e}")
        time.sleep(between)

    if not frames:
        raise RuntimeError("No data collected for any CIKs.")

    final_df = pd.concat(frames, ignore_index=True)
    final_df.to_parquet(out_parquet, index=False)
    print(f"✅ Wrote {len(final_df):,} rows → {os.path.abspath(out_parquet)}")

def diagnose_derived_coverage_from_df(
    df: pd.DataFrame,
    cik: str,
    not_applicable_by_cik: Optional[dict] = None,
    verbose: bool = True,
):
    """
    df: full analytical parquet loaded as a DataFrame
    cik: zero-padded cik string, e.g. "0000059478"
    not_applicable_by_cik:
        {
          "0000059478": {"Debt-to-Assets", "Debt-to-Equity"},
          ...
        }
    """
    if not_applicable_by_cik is None:
        not_applicable_by_cik = {}

    df_cik = df[df["cik"] == cik].copy()
    if df_cik.empty:
        if verbose:
            print(f"No rows found for CIK {cik}")
        return {}

    # Make sure year is clean (int)
    df_cik["year"] = df_cik["year"].astype(int)

    # What derived metrics do we actually have per year?
    metrics_by_year = (
        df_cik[df_cik["metric_type"] == "derived"]
        .groupby("year")["metric_label"]
        .apply(set)
        .to_dict()
    )

    cik_exclusions = not_applicable_by_cik.get(cik, set())
    # allow either set or dict[year -> set(...)]
    cik_exclusions_is_dict = isinstance(cik_exclusions, dict)

    results = {}

    if verbose:
        print("============ DERIVED COVERAGE DIAG ============")

    for year, have in sorted(metrics_by_year.items()):
        expected = set(EXPECTED_DERIVED_LABELS)

        if cik_exclusions_is_dict:
            # year-specific exclusions: NOT_APPLICABLE_BY_CIK["0000..."] = {year: {...}}
            not_applicable = set(cik_exclusions.get(year, set()))
        else:
            # same exclusions for all years of that CIK
            not_applicable = set(cik_exclusions)

        effective_expected = expected - not_applicable
        missing = sorted(effective_expected - have)

        results[year] = {
            "present": sorted(have),
            "not_applicable": sorted(not_applicable),
            "missing": missing,
        }

        if verbose:
            print(f"\nYear {year}")
            print(f"  present        ({len(have)}): {sorted(have)}")
            print(f"  not_applicable ({len(not_applicable)}): {sorted(not_applicable)}")
            print(f"  missing        ({len(missing)}): {missing}")

    return results



def upload_results_to_s3(
    final_parquet_path: str,
    summary: dict,
    dag_id: str = "local_notebook_test",
    task_id: str = "merge_step",
    run_id: str = None,
):
    """
    Upload final parquet + run metadata JSON to S3.
    No large data is pushed to XCom, only optional S3 URIs.
    """

    # -----------------------------
    # S3 CONFIG
    # -----------------------------
    bucket = "sentence-data-ingestion"
    base_prefix = "DATA_MERGE_ASSETS/FINRAG_FACT_METRICS"
    
    run_id = run_id or datetime.utcnow().strftime("%Y%m%dT%H%M%S")

    # Parquet destination
    parquet_key = f"{base_prefix}/analytical_layer_metrics_final.parquet"

    # JSON metadata destination
    json_key = (
        f"{base_prefix}/run_metadata/"
        f"{dag_id}/{task_id}/{run_id.replace(':','_')}.json"
    )

    # -----------------------------
    # JSON-safe serialization
    # -----------------------------
    def _json_default(o):
        try:
            import numpy as np
            if isinstance(o, np.integer): return int(o)
            if isinstance(o, np.floating): return float(o)
            if o is np.nan: return None
        except Exception:
            pass
        try:
            import pandas as pd
            if o is pd.NA: return None
        except Exception:
            pass
        return str(o)

    payload = json.dumps(
        summary,
        ensure_ascii=False,
        default=_json_default
    ).encode("utf-8")

    # -----------------------------
    # Upload to S3
    # -----------------------------
    s3 = boto3.client("s3")

    # 1) Upload parquet
    with open(final_parquet_path, "rb") as f:
        s3.put_object(
            Bucket=bucket,
            Key=parquet_key,
            Body=f,
            ContentType="application/octet-stream",
        )

    # 2) Upload metadata JSON
    s3.put_object(
        Bucket=bucket,
        Key=json_key,
        Body=payload,
        ContentType="application/json",
    )

    print("Uploaded:")
    print("  Parquet →", f"s3://{bucket}/{parquet_key}")
    print("  Metadata →", f"s3://{bucket}/{json_key}")

    return {
        "parquet_s3_uri": f"s3://{bucket}/{parquet_key}",
        "metadata_s3_uri": f"s3://{bucket}/{json_key}",
    }

def generate_coverage_report_csv(df, expected_ciks, out_csv_path):
    """
    Build a CSV report summarizing:
      - missing CIKs (not present in df)
      - missing derived metrics per (cik, year), after applying NOT_APPLICABLE_BY_CIK

    Columns:
      issue_type: "missing_cik" or "missing_metric"
      cik: CIK string
      year: int or empty for missing_cik
      missing_metrics: semicolon-separated list of labels or empty
    """
    rows = []

    present_ciks = set(df["cik"].unique())
    missing_ciks = sorted(set(expected_ciks) - present_ciks)

    # 1) Rows for missing CIKs
    for cik in missing_ciks:
        rows.append(
            {
                "issue_type": "missing_cik",
                "cik": cik,
                "year": None,
                "missing_metrics": None,
            }
        )

    # 2) Rows for missing derived metrics for each present CIK
    for cik in expected_ciks:
        if cik not in present_ciks:
            continue  # already logged as missing_cik

        missing_by_year = diagnose_derived_coverage_from_df(
            df, cik, verbose=False
        )

        for year, missing_list in missing_by_year.items():
            if not missing_list:
                continue  # full coverage for this year

            rows.append(
                {
                    "issue_type": "missing_metric",
                    "cik": cik,
                    "year": int(year),
                    "missing_metrics": "; ".join(missing_list),
                }
            )

    report_df = pd.DataFrame(rows)
    report_df.to_csv(out_csv_path, index=False)

    print(f"Coverage report written to {out_csv_path}")
    return report_df


# ------------------------------------------------------------------------------
# Helper: coverage computation
# ------------------------------------------------------------------------------

def compute_total_missing_derived(df: pd.DataFrame, years: set[int]) -> int:
    """
    Total number of missing derived metrics across EXPECTED_CIKS
    for the given set of years. Lower = better coverage.
    """
    total_missing = 0

    df = df.copy()
    df["year"] = df["year"].astype(int)

    for cik in EXPECTED_CIKS:
        df_cik = df[
            (df["cik"] == cik)
            & (df["metric_type"] == "derived")
            & (df["year"].isin(years))
        ]

        # If we have nothing at all for this CIK in those years → treat as "all-applicable missing"
        if df_cik.empty:
            excluded   = NOT_APPLICABLE_BY_CIK.get(cik, set())
            applicable = EXPECTED_DERIVED_LABELS - excluded
            total_missing += len(applicable) * len(years)
            continue

        missing_by_year = diagnose_derived_coverage_from_df(df, cik, verbose=False)

        for year, missing_list in missing_by_year.items():
            y = int(year)
            if y in years:
                total_missing += len(missing_list)

    return total_missing

# ------------------------------------------------------------------------------
# Helper: S3 upload
# ------------------------------------------------------------------------------

def upload_results_to_s3(
    final_parquet_path: str,
    metadata_path: str,
    coverage_csv_path: str,
    dag_id: str,
    task_id: str,
    run_id: str,
):
    bucket = os.getenv("FINRAG_S3_BUCKET", "sentence-data-ingestion")
    base_prefix = "DATA_MERGE_ASSETS/FINRAG_FACT_METRICS"

    # Keep the same key as your earlier logs (analytical_layer_metrics_final.parquet)
    parquet_key  = f"{base_prefix}/analytical_layer_metrics_final.parquet"
    metadata_key = f"{base_prefix}/run_metadata/{dag_id}/{task_id}/{run_id}.json"
    coverage_key = f"{base_prefix}/coverage_reports/{dag_id}/{task_id}/{run_id}.csv"

    s3 = boto3.client("s3")

    # Upload parquet
    with open(final_parquet_path, "rb") as f:
        s3.put_object(
            Bucket=bucket,
            Key=parquet_key,
            Body=f,
            ContentType="application/octet-stream",
        )

    # Upload metadata JSON
    with open(metadata_path, "rb") as f:
        s3.put_object(
            Bucket=bucket,
            Key=metadata_key,
            Body=f,
            ContentType="application/json",
        )

    # Upload coverage CSV
    with open(coverage_csv_path, "rb") as f:
        s3.put_object(
            Bucket=bucket,
            Key=coverage_key,
            Body=f,
            ContentType="text/csv",
        )

    print("Uploaded:")
    print("  Parquet  →", f"s3://{bucket}/{parquet_key}")
    print("  Metadata →", f"s3://{bucket}/{metadata_key}")
    print("  Coverage →", f"s3://{bucket}/{coverage_key}")

    return {
        "parquet_s3_uri": f"s3://{bucket}/{parquet_key}",
        "metadata_s3_uri": f"s3://{bucket}/{metadata_key}",
        "coverage_s3_uri": f"s3://{bucket}/{coverage_key}",
    }

# ------------------------------------------------------------------------------
# Helper: email
# ------------------------------------------------------------------------------

def send_coverage_email(summary: dict, coverage_csv_path: str):
    smtp_host = os.getenv("SMTP_HOST")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER")
    smtp_password = os.getenv("SMTP_PASSWORD")

    email_from = os.getenv("ALERT_EMAIL_FROM", smtp_user)
    email_to   = os.getenv("ALERT_EMAIL_TO", email_from)  # default to self

    if not all([smtp_host, smtp_port, smtp_user, smtp_password, email_from, email_to]):
        raise RuntimeError("SMTP / email environment variables are not fully configured.")

    merged       = summary.get("merged")
    status_str   = "MERGED" if merged else "SKIPPED"
    start_year   = summary.get("start_year")
    end_year     = summary.get("end_year")
    missing_prev = summary.get("missing_prev")
    missing_new  = summary.get("missing_new")
    reason       = summary.get("reason")

    subject = f"[FINRAG ANALYTICAL LAYER] {status_str} for {start_year}-{end_year}"

    body_lines = [
        f"Analytical layer run for years {start_year}-{end_year}.",
        "",
        f"Merge status: {status_str}",
        f"Reason: {reason}",
        "",
        f"Previous missing derived (last2yrs): {missing_prev}",
        f"New missing derived      (last2yrs): {missing_new}",
        "",
        f"Rows in new run: {summary.get('rows_new')}",
        f"Rows in previous final: {summary.get('rows_prev')}",
        "",
        f"Run timestamp (UTC): {summary.get('run_timestamp_utc')}",
        "",
        "Coverage details by (cik, year) are attached as CSV.",
    ]
    body = "\n".join(body_lines)

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = email_from
    msg["To"] = email_to
    msg.set_content(body)

    with open(coverage_csv_path, "rb") as f:
        data = f.read()

    msg.add_attachment(
        data,
        maintype="text",
        subtype="csv",
        filename=os.path.basename(coverage_csv_path),
    )

    with smtplib.SMTP(smtp_host, smtp_port) as server:
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.send_message(msg)

    print(f"📧 Coverage email sent to: {email_to}")

def send_success_email(run_timestamp: str):
    smtp_host = os.getenv("SMTP_HOST")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER")
    smtp_password = os.getenv("SMTP_PASSWORD")

    email_from = os.getenv("ALERT_EMAIL_FROM", smtp_user)
    email_to   = os.getenv("ALERT_EMAIL_TO", email_from)

    if not all([smtp_host, smtp_port, smtp_user, smtp_password, email_from, email_to]):
        raise RuntimeError("SMTP environment variables missing for success email.")

    subject = "[FINRAG ANALYTICAL LAYER] SUCCESS"
    body = (
        "FINRAG Analytical Layer DAG completed successfully.\n\n"
        f"Run timestamp (UTC): {run_timestamp}\n\n"
        "All tasks completed without errors."
    )

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = email_from
    msg["To"] = email_to
    msg.set_content(body)

    with smtplib.SMTP(smtp_host, smtp_port) as server:
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.send_message(msg)

    print(f"📧 Success email sent to: {email_to}")

def send_failure_email(error_msg: str, run_timestamp: str):
    smtp_host = os.getenv("SMTP_HOST")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER")
    smtp_password = os.getenv("SMTP_PASSWORD")

    email_from = os.getenv("ALERT_EMAIL_FROM", smtp_user)
    email_to   = os.getenv("ALERT_EMAIL_TO", email_from)

    if not all([smtp_host, smtp_port, smtp_user, smtp_password, email_from, email_to]):
        raise RuntimeError("SMTP environment variables missing for failure email.")

    subject = "[FINRAG ANALYTICAL LAYER] FAILURE ❌"
    body = (
        "FINRAG Analytical Layer DAG failed.\n\n"
        f"Error: {error_msg}\n\n"
        f"Run timestamp (UTC): {run_timestamp}\n\n"
        "Please check Airflow logs for details."
    )

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = email_from
    msg["To"] = email_to
    msg.set_content(body)

    with smtplib.SMTP(smtp_host, smtp_port) as server:
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.send_message(msg)

    print(f"📧 FAILURE email sent to: {email_to}")


def run_analytical_layer_pipeline(
    base_dir: str,
    polite_delay: float,
    run_date: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    Runs the full analytical layer pipeline for the last 2 years:

      1. Set START_YEAR / END_YEAR to (end_year-1, end_year)
      2. Build NEW parquet for those 2 years
      3. Compare derived coverage vs existing final parquet
      4. If NEW has strictly better coverage → merge & overwrite final parquet
      5. Write:
         - analytical_layer_metrics_last2yrs.parquet
         - analytical_layer_metrics_final.parquet
         - analytical_layer_run_metadata.json
         - analytical_layer_coverage_last2yrs.csv

    Returns:
        {
          "summary": { ... },
          "new_parquet_path": str,
          "final_parquet_path": str,
          "coverage_csv_path": str,
          "metadata_json_path": str,
          "last2_years": [start_year, end_year],
        }
    """
    run_date   = run_date or datetime.today()
    end_year   = run_date.year
    start_year = end_year - 1
    last2_years = {start_year, end_year}

    # Paths under the given base_dir
    new_parquet_path   = os.path.join(base_dir, "analytical_layer_metrics_last2yrs.parquet")
    final_parquet_path = os.path.join(base_dir, "analytical_layer_metrics_final.parquet")
    coverage_csv_path  = os.path.join(base_dir, "analytical_layer_coverage_last2yrs.csv")
    metadata_json_path = os.path.join(base_dir, "analytical_layer_run_metadata.json")

    os.makedirs(base_dir, exist_ok=True)

    # Override global year range used by fetch_10k_facts_for_analytical_layer
    global START_YEAR, END_YEAR
    START_YEAR = start_year
    END_YEAR   = end_year

    print(f"[analytical_layer] Run date: {run_date}")
    print(f"[analytical_layer] Building NEW analytical layer for years {sorted(last2_years)}")
    print("[analytical_layer] Output:", new_parquet_path)

    # Build new 2-year layer
    build_analytical_layer(
        ciks=EXPECTED_CIKS,
        out_parquet=new_parquet_path,
        between=polite_delay,
        n_years_derived=2,
    )

    df_new = pd.read_parquet(new_parquet_path)
    df_new["year"] = df_new["year"].astype(int)
    print("[analytical_layer] New data shape:", df_new.shape)

    missing_new_test = compute_total_missing_derived(df_new, last2_years)
    print("[analytical_layer] Total missing derived (NEW, test subset, last2yrs):", missing_new_test)

    summary: Dict[str, Any] = {
        "start_year": start_year,
        "end_year": end_year,
        "merged": False,
        "reason": "",
        "rows_new": int(len(df_new)),
        "rows_prev": 0,
        "missing_prev": None,
        "missing_new": None,
        "run_timestamp_utc": datetime.utcnow().isoformat(),
    }

    # Merge / bootstrap logic
    if not os.path.exists(final_parquet_path):
        print("[analytical_layer] No previous FINAL_PARQUET_PATH found, bootstrapping with NEW.")
        df_new.to_parquet(final_parquet_path, index=False)
        summary["merged"]      = True
        summary["reason"]      = "Initialized final parquet with new 2-year layer (no previous data)."
        summary["missing_new"] = int(compute_total_missing_derived(df_new, last2_years))
    else:
        df_prev = pd.read_parquet(final_parquet_path)
        df_prev["year"] = df_prev["year"].astype(int)
        summary["rows_prev"] = int(len(df_prev))

        # 1) Schema check
        if set(df_prev.columns) != set(df_new.columns):
            raise ValueError(
                "Schema mismatch!\n"
                f"prev columns: {sorted(df_prev.columns)}\n"
                f"new columns:  {sorted(df_new.columns)}"
            )

        # 2) Coverage comparison on last 2 years
        missing_prev = compute_total_missing_derived(df_prev, last2_years)
        missing_new  = compute_total_missing_derived(df_new, last2_years)

        summary["missing_prev"] = int(missing_prev)
        summary["missing_new"]  = int(missing_new)

        print(f"[analytical_layer] Previous missing (last2yrs): {missing_prev}")
        print(f"[analytical_layer] New missing      (last2yrs): {missing_new}")

        if missing_new < missing_prev:
            df_prev_keep = df_prev[~df_prev["year"].isin(last2_years)]
            df_merged = pd.concat([df_prev_keep, df_new], ignore_index=True)

            df_merged.to_parquet(final_parquet_path, index=False)

            summary["merged"] = True
            summary["reason"] = "Merged: new 2-year layer has strictly better coverage."
            print("[analytical_layer] ✅ Merged. New final shape:", df_merged.shape)
        else:
            summary["merged"] = False
            summary["reason"] = "Skipped merge: new coverage is equal or worse than previous data."
            print("[analytical_layer] ⚠️ Merge skipped; final parquet unchanged.")

    # Save metadata locally
    with open(metadata_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Generate coverage report (last 2 years from final parquet)
    df_final = pd.read_parquet(final_parquet_path)
    df_final["year"] = df_final["year"].astype(int)
    df_last2 = df_final[df_final["year"].isin(last2_years)].copy()

    report_df = generate_coverage_report_csv(
        df=df_last2,
        expected_ciks=EXPECTED_CIKS,
        out_csv_path=coverage_csv_path,
    )

    print("[analytical_layer] Coverage report written to:", coverage_csv_path)
    print(report_df.head())

    return {
        "summary": summary,
        "new_parquet_path": new_parquet_path,
        "final_parquet_path": final_parquet_path,
        "coverage_csv_path": coverage_csv_path,
        "metadata_json_path": metadata_json_path,
        "last2_years": sorted(last2_years),
    }
