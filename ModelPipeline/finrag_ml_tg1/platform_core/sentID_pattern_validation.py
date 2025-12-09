"""
Data validation utilities for FinRAG pipeline.
"""

import polars as pl
from pathlib import Path


def validate_sentenceid_pattern():
    """
    Validate sentenceID pattern reliability for the fact sentences table.
    Loads data from cache and performs full validation analysis.
    
    Expected pattern: {CIK}_{filing}_{year}_section_{section_ID}_{sequence}
    Example: 0000320193_10-K_2016_section_1_42
    
    Returns:
        dict: Validation statistics including:
            - total_rows
            - full_pattern_valid_count
            - full_pattern_valid_pct
            - numeric_suffix_valid_count
            - numeric_suffix_valid_pct
            - recommendation
    """
    # Load from local cache
    cache_file = Path.cwd().parent / 'data_cache' / 'stage1_facts' / 'finrag_fact_sentences.parquet'
    print(f"Loading: {cache_file}")
    
    df = pl.read_parquet(cache_file)
    print(f"✓ Loaded: {len(df):,} rows\n")
    
    # Define expected pattern
    pattern = r'^(\d{10})_(10-[KQ]|8-K)_(\d{4})_section_(\w+)_(\d+)$'

    # Validate pattern
    df_validated = df.with_columns([
        pl.col('sentenceID').str.contains(pattern).alias('_matches_full_pattern'),
        pl.col('sentenceID').str.extract(pattern, 1).alias('_cik_part'),
        pl.col('sentenceID').str.extract(pattern, 2).alias('_filing_part'),
        pl.col('sentenceID').str.extract(pattern, 3).alias('_year_part'),
        pl.col('sentenceID').str.extract(pattern, 4).alias('_section_part'),
        pl.col('sentenceID').str.extract(pattern, 5).alias('_sequence_part'),
        pl.col('sentenceID').str.split('_').list.last()
            .cast(pl.Int32, strict=False)
            .is_not_null()
            .alias('_has_numeric_suffix')
    ])

    # Calculate statistics
    total_rows = len(df_validated)
    full_pattern_valid = df_validated.filter(pl.col('_matches_full_pattern')).shape[0]
    numeric_suffix_valid = df_validated.filter(pl.col('_has_numeric_suffix')).shape[0]

    print("="*70)
    print("SENTENCEID PATTERN VALIDATION")
    print("="*70)
    print(f"\nTotal rows: {total_rows:,}")
    print(f"\n[Full Pattern: CIK_filing_year_section_sectionID_sequence]")
    print(f"  Valid: {full_pattern_valid:,} ({full_pattern_valid/total_rows*100:.2f}%)")
    print(f"  Invalid: {total_rows - full_pattern_valid:,} ({(1-full_pattern_valid/total_rows)*100:.2f}%)")

    print(f"\n[Numeric Suffix Only: ends with number]")
    print(f"  Valid: {numeric_suffix_valid:,} ({numeric_suffix_valid/total_rows*100:.2f}%)")
    print(f"  Invalid: {total_rows - numeric_suffix_valid:,} ({(1-numeric_suffix_valid/total_rows)*100:.2f}%)")

    # Show invalid examples
    invalid_full = df_validated.filter(~pl.col('_matches_full_pattern'))
    if len(invalid_full) > 0:
        print(f"\nExamples of INVALID sentenceIDs (full pattern):")
        for row in invalid_full.select('sentenceID').head(10).iter_rows():
            print(f"  - {row[0]}")

    # Show valid examples with parsed components
    valid_samples = df_validated.filter(pl.col('_matches_full_pattern')).head(5)
    print(f"\n✓ Examples of VALID sentenceIDs (parsed):")
    for row in valid_samples.select(['sentenceID', '_cik_part', '_year_part', '_section_part', '_sequence_part']).iter_rows():
        print(f"  {row[0]}")
        print(f"    → CIK: {row[1]}, Year: {row[2]}, Section: {row[3]}, Seq: {row[4]}")

    # Component-level validation
    print(f"\n[Component Validation]")
    print(f"  CIK extracted: {df_validated.filter(pl.col('_cik_part').is_not_null()).shape[0]:,} rows")
    print(f"  Filing extracted: {df_validated.filter(pl.col('_filing_part').is_not_null()).shape[0]:,} rows")
    print(f"  Year extracted: {df_validated.filter(pl.col('_year_part').is_not_null()).shape[0]:,} rows")
    print(f"  Section extracted: {df_validated.filter(pl.col('_section_part').is_not_null()).shape[0]:,} rows")
    print(f"  Sequence extracted: {df_validated.filter(pl.col('_sequence_part').is_not_null()).shape[0]:,} rows")

    # Final recommendation
    print(f"\n{'='*70}")
    if full_pattern_valid / total_rows >= 0.95:
        print("✅ RECOMMENDATION: Pattern highly reliable (≥95%)")
        print("   → Safe to use shift() for prev/next_sentenceID")
        print("   → Sequence numbers are trustworthy for ordering")
        recommendation = "reliable"
    elif numeric_suffix_valid / total_rows >= 0.95:
        print("⚠️ RECOMMENDATION: Full pattern has issues, but numeric suffix reliable")
        print("   → Can use shift() but validate sorting carefully")
        recommendation = "numeric_suffix_only"
    else:
        print("❌ RECOMMENDATION: Pattern unreliable (<95%)")
        print("   → Skip prev/next_sentenceID columns")
        print("   → Use runtime neighbor lookups instead")
        recommendation = "unreliable"
    print("="*70)

    # Show sentenceIDs from most recent year
    latest_year = df['report_year'].max()
    sample = df.filter(pl.col('report_year') == latest_year).select('sentenceID').head(10)

    print(f"\nLatest year: {latest_year}\nSample sentenceIDs:")
    for sid in sample['sentenceID']:
        print(f"  {sid}")

    return {
        'total_rows': total_rows,
        'full_pattern_valid_count': full_pattern_valid,
        'full_pattern_valid_pct': full_pattern_valid / total_rows * 100,
        'numeric_suffix_valid_count': numeric_suffix_valid,
        'numeric_suffix_valid_pct': numeric_suffix_valid / total_rows * 100,
        'recommendation': recommendation
    }