"""
Embedding audit and coverage analysis for FinRAG pipeline.
Memory-efficient implementation using Polars lazy evaluation.
"""

import polars as pl
from pathlib import Path

pl.Config.set_tbl_rows(50)
pl.Config.set_tbl_cols(20)
pl.Config.set_fmt_str_lengths(500)
pl.Config.set_tbl_width_chars(2000)




import polars as pl
from pathlib import Path

def audit_embedding_execution_history():
    """
    Summarize embedding executions without ever joining the large list column
    onto the meta table.

    Returns
    -------
    pl.DataFrame
        One row per (embedded flag, embedding_id, embedding_date) with counts,
        company/year coverage, and dimension checks.
    """
    base_dir = Path.cwd().parent / "data_cache"
    meta_path = base_dir / "meta_embeds" / "finrag_fact_sentences_meta_embeds.parquet"
    emb_path  = base_dir / "embeddings"  / "cohere_1024d" / "finrag_embeddings_cohere_1024d.parquet"

    print("[Scanning tables with lazy evaluation…]")

    # ---- META SIDE: coverage + scope (no list column involved) ----
    lf_meta = (
        pl.scan_parquet(meta_path)
        .select([
            "sentenceID",
            "embedding_id",
            "embedding_date",
            "cik_int",
            "name",
            "report_year",
        ])
        .with_columns(
            pl.when(pl.col("embedding_id").is_not_null())
              .then(pl.lit("Yes"))
              .otherwise(pl.lit("No"))
              .alias("embedded")
        )
    )

    lf_meta_summary = (
        lf_meta
        .group_by(["embedded", "embedding_id", "embedding_date"])
        .agg([
            pl.len().alias("sentences"),
            pl.col("cik_int").n_unique().alias("distinct_companies"),
            pl.col("report_year").n_unique().alias("distinct_years"),
            pl.col("name").unique().sort().str.join(", ").alias("companies"),
            pl.col("report_year").unique().sort().cast(pl.Utf8).str.join(", ").alias("years"),
        ])
    )

    # ---- EMBEDDING SIDE: dimension checks only, one row per embedding_id ----
    lf_dim_summary = (
        pl.scan_parquet(emb_path)
        .select([
            "embedding_id",
            # list.len() only used here, never dragged through a big join
            pl.col("embedding").list.len().alias("dim"),
        ])
        .group_by("embedding_id")
        .agg([
            pl.col("dim").n_unique().alias("distinct_dims"),
            pl.col("dim").min().alias("min_dim"),
            pl.col("dim").max().alias("max_dim"),
        ])
    )

    # ---- Join only summaries (very small) ----
    audit_summary = (
        lf_meta_summary
        .join(lf_dim_summary, on="embedding_id", how="left")
        .sort("embedding_date", descending=True, nulls_last=True)
    )

    print("[Executing optimized summary plan…]")
    df_result = audit_summary.collect(streaming=True)

    print(f"[Complete - Generated summary: {df_result.shape[0]} rows]")
    print("\n" + "=" * 70)
    print("EMBEDDING AUDIT - Execution History")
    print("=" * 70)

    return df_result





def show_missing_embeddings(min_gap_threshold: int = 20):
    """
    Analyze gaps *within* the already-embedded scope using only the meta table.

    We treat a sentence as embedded iff `embedding_id` is non-null in meta.
    """

    base_dir = Path.cwd().parent / "data_cache"
    meta_path = base_dir / "meta_embeds" / "finrag_fact_sentences_meta_embeds.parquet"

    print("[Scanning meta table with lazy evaluation…]")

    lf_meta = (
        pl.scan_parquet(meta_path)
        .select(["cik_int", "name", "report_year", "embedding_id"])
        .with_columns(
            pl.when(pl.col("embedding_id").is_not_null())
              .then(pl.lit(1))
              .otherwise(pl.lit(0))
              .alias("embedded_flag")
        )
    )

    # Total + embedded per company-year
    lf_counts = (
        lf_meta
        .group_by(["cik_int", "name", "report_year"])
        .agg([
            pl.len().alias("total_sentences"),
            pl.col("embedded_flag").sum().alias("embedded_count"),
        ])
        .with_columns(
            (pl.col("total_sentences") - pl.col("embedded_count")).alias("missing_count")
        )
    )

    # Scope: company-years where at least one sentence is embedded
    lf_scope = lf_counts.filter(pl.col("embedded_count") > 0)

    print("[Executing: gap analysis in embedded scope…]")
    df_scope = lf_scope.with_columns(
        (pl.col("missing_count") / pl.col("total_sentences") * 100).alias("missing_pct")
    ).collect(streaming=True)

    total_missing = int(df_scope["missing_count"].sum())

    meaningful_gaps = df_scope.filter(pl.col("missing_count") >= min_gap_threshold)
    trivial_gaps    = df_scope.filter(pl.col("missing_count") <  min_gap_threshold)

    print("=" * 70)
    print("INTELLIGENT MISSING EMBEDDINGS ANALYSIS")
    print("=" * 70)
    print(f"\n[Your Embedding Scope]")
    print(f"  Companies with embeddings: {df_scope['cik_int'].n_unique()}")
    print(f"  Years covered: {sorted(df_scope['report_year'].unique().to_list())}")
    print(f"\n[Gap Analysis]")
    print(f"  Total missing in your scope: {total_missing:,} sentences")
    print(f"  Trivial gaps (< {min_gap_threshold} sentences): "
          f"{trivial_gaps['missing_count'].sum() if len(trivial_gaps) else 0:,}")
    print(f"  Meaningful gaps: "
          f"{meaningful_gaps['missing_count'].sum() if len(meaningful_gaps) else 0:,}")

    if len(trivial_gaps) > 0:
        print("\n  [Trivial gaps by company-year]")
        for row in trivial_gaps.sort("missing_count", descending=True).iter_rows(named=True):
            print(f"    - {row['name']} ({row['cik_int']}) {row['report_year']}: "
                  f"{row['missing_count']} sentences ({row['missing_pct']:.1f}%)")

    if len(meaningful_gaps) == 0:
        print("\n" + "=" * 70)
        print("✅ NO MEANINGFUL GAPS FOUND")
        print("=" * 70)
        return {
            "missing_ciks": [],
            "missing_years": [],
            "meaningful_gap_count": 0,
            "trivial_gap_count": total_missing,
        }

    print("\n" + "=" * 70)
    print("MEANINGFUL GAPS - Require Attention")
    print("=" * 70)

    for row in meaningful_gaps.sort(["cik_int", "report_year"]).iter_rows(named=True):
        print(f"  {row['name']} ({row['cik_int']}) - {row['report_year']}: "
              f"{row['missing_count']:,} missing / {row['total_sentences']:,} "
              f"({row['missing_pct']:.1f}%)")

    gap_ciks  = meaningful_gaps["cik_int"].unique().sort().to_list()
    gap_years = meaningful_gaps["report_year"].unique().sort().to_list()

    print("\n" + "=" * 70)
    print("CONFIG FORMAT - Re-run Embeddings for Gaps")
    print("=" * 70)
    print("\nfilters:")
    print(f"  cik_int: {gap_ciks}")
    print(f"  year: {gap_years}")
    print("  sections: null")
    print("\n" + "=" * 70)
    print("Batch Suggestions (5 companies per batch):")
    print("=" * 70)
    for i in range(0, len(gap_ciks), 5):
        print(f"  cik_int: {gap_ciks[i:i+5]}")
    print("\n" + "=" * 70)

    return {
        "missing_ciks": gap_ciks,
        "missing_years": gap_years,
        "meaningful_gap_count": int(meaningful_gaps["missing_count"].sum()),
        "trivial_gap_count": int(trivial_gaps["missing_count"].sum()),
    }





def show_completely_unembedded_company_years():
    """
    Show company-year combinations with ZERO embeddings.
    Simple query: if a company-year has 0 embedded sentences, show it.
    
    Uses lazy evaluation for memory-efficient aggregation.
    
    Returns:
        dict: Contains zero_ciks, zero_years, and count
    """
    # Define paths
    cache_dir = Path.cwd().parent / 'data_cache' / 'meta_embeds'
    
    print("[Scanning meta table with lazy evaluation...]")
    
    # LAZY SCAN and build query
    coverage = (
        pl.scan_parquet(cache_dir / 'finrag_fact_sentences_meta_embeds.parquet')
        # Group by company and year, count embeddings
        .group_by(['cik_int', 'name', 'report_year'])
        .agg([
            pl.len().alias('total_sentences'),
            pl.col('embedding_id').is_not_null().sum().alias('embedded_count')
        ])
        # Only show where ZERO embeddings exist
        .filter(pl.col('embedded_count') == 0)
        .sort(['cik_int', 'report_year'])
    )
    
    # EXECUTE
    print("[Executing: Finding unembedded company-years...]")
    df_coverage = coverage.collect()
    
    print("="*70)
    print("COMPANY-YEARS WITH ZERO EMBEDDINGS")
    print("="*70)
    print(f"\nTotal company-year combinations: {len(df_coverage)}")
    
    if len(df_coverage) == 0:
        print("\n✅ All company-years have at least some embeddings!")
        print("="*70)
        return {
            'zero_ciks': [],
            'zero_years': [],
            'company_year_count': 0
        }
    
    # Group by company for cleaner display (post-collection, small result)
    print(f"\n[By Company]")
    for company_data in df_coverage.group_by('cik_int'):
        cik = company_data[0]
        company_df = company_data[1].sort('report_year')
        company_name = company_df['name'][0]
        
        # Build years and counts inline
        years_display = ", ".join([
            f"{row['report_year']} ({row['total_sentences']:,})" 
            for row in company_df.iter_rows(named=True)
        ])
        
        print(f"\n  {company_name} (CIK: {cik})")
        print(f"    Years (sentences): {years_display}")
    
    # Get unique CIKs and years for config format
    zero_ciks = df_coverage['cik_int'].unique().sort().to_list()
    zero_years = df_coverage['report_year'].unique().sort().to_list()
    
    print(f"\n{'='*70}")
    print("CONFIG FORMAT")
    print("="*70)
    print(f"\nfilters:")
    print(f"  cik_int: {zero_ciks}")
    print(f"  year: {zero_years}")
    print(f"  sections: null")
    
    print(f"\n{'='*70}")
    
    return {
        'zero_ciks': zero_ciks,
        'zero_years': zero_years,
        'company_year_count': len(df_coverage)
    }