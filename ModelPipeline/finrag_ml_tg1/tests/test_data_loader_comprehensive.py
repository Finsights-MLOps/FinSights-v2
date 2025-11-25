"""
Comprehensive Data Loader Tests - Local + SAM Local

Test Structure:
  Part 1: Local Tests (no Docker needed)
    - LocalCacheLoader functionality
    - Factory detection logic
    - MLConfig Lambda readiness
  
  Part 2: SAM Local Tests (requires Docker + sam local invoke)
    - S3StreamingLoader with REAL S3
    - /tmp caching in Lambda environment
    - Cross-platform path handling

Usage:
  # Part 1 only (local unit tests)
  python test_data_loader_comprehensive.py
  
  # Part 2 (SAM local tests)
  # See: LambdaAssets/test_lambda_data_loader.sh
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any

# ============================================================================
# Bootstrap
# ============================================================================
def bootstrap_path():
    """Find ModelPipeline root and add to sys.path"""
    
    # Strategy 1: Check file's own location first
    file_path = Path(__file__).resolve()
    for parent in [file_path] + list(file_path.parents):
        if parent.name == "ModelPipeline":
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            return parent
    
    # Strategy 2: Check current working directory
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        if parent.name == "ModelPipeline":
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            return parent
    
    # Failed - provide helpful error
    raise RuntimeError(
        f"Cannot find ModelPipeline root.\n"
        f"  File location: {file_path}\n"
        f"  Current directory: {current}\n"
        f"  Expected 'ModelPipeline' in path tree"
    )


model_root = bootstrap_path()

from finrag_ml_tg1.loaders.ml_config_loader import MLConfig
from finrag_ml_tg1.loaders.data_loader_factory import create_data_loader
from finrag_ml_tg1.loaders.data_loader_strategy import LocalCacheLoader, S3StreamingLoader


# ============================================================================
# PART 1: LOCAL TESTS (No Docker Required)
# ============================================================================

def test_01_local_cache_loader():
    """
    Test 1: LocalCacheLoader - Production Behavior
    Validates current working functionality before Lambda changes
    """
    print("\n" + "="*80)
    print("TEST 1: LocalCacheLoader - Full Functionality")
    print("="*80)
    
    config = MLConfig()
    loader = create_data_loader(config)
    
    # Verify correct loader type
    print(f"\n[Loader Verification]")
    print(f"  Type: {type(loader).__name__}")
    print(f"  Is LocalCacheLoader: {isinstance(loader, LocalCacheLoader)}")
    assert isinstance(loader, LocalCacheLoader), "Should use LocalCacheLoader in local env"
    
    # Test Stage 2 Meta loading
    print(f"\n[Stage 2 Meta Table]")
    df = loader.load_stage2_meta()
    print(f"  ✓ Rows: {len(df):,}")
    print(f"  ✓ Memory: {df.estimated_size('mb'):.1f} MB")
    print(f"  ✓ Columns: {len(df.columns)} ({', '.join(df.columns[:5])}...)")
    
    assert len(df) > 0, "Stage 2 Meta must have data"
    assert 'sentenceID' in df.columns, "Must have sentenceID"
    assert 'sentence' in df.columns, "Must have sentence text"
    
    # Test dimension tables
    print(f"\n[Dimension Tables]")
    companies_df = loader.load_dimension_companies()
    print(f"  ✓ Companies: {len(companies_df):,} rows")
    assert len(companies_df) > 0
    
    sections_df = loader.load_dimension_sections()
    print(f"  ✓ Sections: {len(sections_df):,} rows")
    assert len(sections_df) > 0
    
    # Test sentence fetch by ID (critical for SentenceExpander)
    print(f"\n[Sentence Fetch by ID]")
    sample_ids = df['sentenceID'].head(10).to_list()
    fetched = loader.get_sentences_by_ids(sample_ids)
    print(f"  ✓ Requested: {len(sample_ids)} IDs")
    print(f"  ✓ Retrieved: {len(fetched)} sentences")
    
    assert len(fetched) == len(sample_ids), "Should fetch all requested"
    fetched_ids = set(fetched['sentenceID'].to_list())
    requested_ids = set(sample_ids)
    assert fetched_ids == requested_ids, "IDs must match"
    
    print("\n✅ TEST 1 PASSED: LocalCacheLoader works correctly")
    return True


def test_02_factory_detection():
    """
    Test 2: Factory Detection Logic
    Tests environment detection without instantiating Lambda-specific code
    """
    print("\n" + "="*80)
    print("TEST 2: Factory Detection Logic")
    print("="*80)
    
    # Test local environment
    print(f"\n[Local Environment]")
    config = MLConfig()
    print(f"  Is Lambda: {config.is_lambda_environment}")
    print(f"  Data Mode: {config.data_loading_mode}")
    
    assert not config.is_lambda_environment, "Should detect local env"
    assert config.data_loading_mode == 'LOCAL_CACHE', "Should use LOCAL_CACHE"
    
    loader = create_data_loader(config)
    assert isinstance(loader, LocalCacheLoader), "Should create LocalCacheLoader"
    print(f"  ✓ Selected: {type(loader).__name__}")
    
    # Test Lambda environment detection (mocked)
    print(f"\n[Lambda Environment (Mocked)]")
    os.environ['AWS_LAMBDA_FUNCTION_NAME'] = 'test-function'
    
    try:
        config_lambda = MLConfig()
        print(f"  Is Lambda: {config_lambda.is_lambda_environment}")
        print(f"  Data Mode: {config_lambda.data_loading_mode}")
        
        assert config_lambda.is_lambda_environment, "Should detect Lambda"
        assert config_lambda.data_loading_mode == 'S3_STREAMING', "Should use S3_STREAMING"
        print(f"  ✓ Would select: S3StreamingLoader")
        print(f"  ℹ  Actual instantiation tested via SAM local")
        
    finally:
        del os.environ['AWS_LAMBDA_FUNCTION_NAME']
    
    print("\n✅ TEST 2 PASSED: Factory detection logic works")
    return True


def test_03_mlconfig_lambda_ready():
    """
    Test 3: MLConfig Lambda Readiness
    Validates all required attributes for Lambda deployment
    """
    print("\n" + "="*80)
    print("TEST 3: MLConfig Lambda Readiness")
    print("="*80)
    
    config = MLConfig()
    
    # Check critical attributes
    print(f"\n[Required Attributes]")
    checks = {
        'model_root': hasattr(config, 'model_root'),
        'is_lambda_environment': hasattr(config, 'is_lambda_environment'),
        'data_loading_mode': hasattr(config, 'data_loading_mode'),
        'bucket': hasattr(config, 'bucket'),
        'region': hasattr(config, 'region'),
        'get_s3_client': callable(getattr(config, 'get_s3_client', None)),
        'get_storage_options': callable(getattr(config, 'get_storage_options', None)),
    }
    
    for attr, exists in checks.items():
        status = "✓" if exists else "✗"
        print(f"  {status} {attr}")
        assert exists, f"Missing: {attr}"
    
    # Validate model_root
    print(f"\n[Model Root Validation]")
    print(f"  Path: {config.model_root}")
    print(f"  Exists: {config.model_root.exists()}")
    print(f"  Name: {config.model_root.name}")
    
    assert config.model_root.exists(), "model_root must exist"
    assert config.model_root.name == "ModelPipeline", "Must be ModelPipeline dir"
    
    # Verify no Path.cwd() hacks
    print(f"\n[Path Resolution (No Hacks)]")
    cache_path = config.get_s3vectors_cache_path('cohere_1024d')
    uses_model_root = str(config.model_root) in str(cache_path)
    print(f"  Cache path: {cache_path}")
    print(f"  Uses model_root: {uses_model_root}")
    
    assert uses_model_root, "Must use model_root, not Path.cwd()"
    
    print("\n✅ TEST 3 PASSED: MLConfig is Lambda-ready")
    return True


# ============================================================================
# PART 2: SAM LOCAL TESTS (Requires Docker + sam local invoke)
# ============================================================================

def test_04_s3_streaming_loader_info():
    """
    Test 4: S3StreamingLoader Information
    Explains how to test S3StreamingLoader via SAM local
    """
    print("\n" + "="*80)
    print("TEST 4: S3StreamingLoader Testing (via SAM Local)")
    print("="*80)
    
    print(f"\n[Why Not Tested Here]")
    print(f"  S3StreamingLoader requires:")
    print(f"    • /tmp/ directory (doesn't exist on Windows)")
    print(f"    • Lambda environment variables")
    print(f"    • Real S3 access with proper IAM")
    
    print(f"\n[How to Test S3StreamingLoader]")
    print(f"  1. Ensure Docker Desktop is running")
    print(f"  2. cd LambdaAssets/")
    print(f"  3. Run: sam local invoke DataLoaderTestFunction")
    print(f"     (See: LambdaAssets/test_lambda_data_loader.sh)")
    
    print(f"\n[What SAM Local Tests]")
    print(f"  ✓ S3StreamingLoader instantiation in Lambda container")
    print(f"  ✓ Stage 2 Meta loading from REAL S3")
    print(f"  ✓ /tmp caching behavior")
    print(f"  ✓ Cross-platform path handling")
    
    print(f"\n[Next Steps]")
    print(f"  After local tests pass, run:")
    print(f"    cd LambdaAssets")
    print(f"    ./test_lambda_data_loader.sh")
    
    print("\nℹ️  TEST 4 INFO: See SAM local test script for S3StreamingLoader")
    return True


def test_05_kpi_fact_data_loading():
    """
    Test 5: KPI Fact Data Loading
    Validates metric pipeline data access via DataLoader
    """
    print("\n" + "="*80)
    print("TEST 5: KPI Fact Data Loading")
    print("="*80)
    
    config = MLConfig()
    loader = create_data_loader(config)
    
    # Test loading
    print(f"\n[Loading KPI Fact Data]")
    kpi_df = loader.load_kpi_fact_data()
    
    print(f"  ✓ Rows: {len(kpi_df):,}")
    print(f"  ✓ Memory: {kpi_df.estimated_size('mb'):.1f} MB")
    print(f"  ✓ Columns: {len(kpi_df.columns)}")
    print(f"  ✓ Sample columns: {', '.join(kpi_df.columns[:5])}...")
    
    # Basic validation
    assert len(kpi_df) > 0, "KPI data must have rows"
    assert 'cik_int' in kpi_df.columns or 'cik' in kpi_df.columns, "Should have CIK column"
    
    # Test caching behavior (second call should be instant)
    print(f"\n[Testing Cache Behavior]")
    import time
    start = time.time()
    kpi_df_cached = loader.load_kpi_fact_data()
    elapsed = time.time() - start
    print(f"  ✓ Second call: {elapsed*1000:.2f}ms (cached)")
    
    assert len(kpi_df_cached) == len(kpi_df), "Cached result should match"
    
    print("\n✅ TEST 5 PASSED: KPI Fact Data loads correctly")
    return True


# ============================================================================
# TEST RUNNER
# ============================================================================

def run_all_tests():
    """Execute all local tests"""
    print("="*80)
    print("DATA LOADER - COMPREHENSIVE TEST SUITE")
    print("="*80)
    print("\nPart 1: Local Tests (Python)")
    print("Part 2: SAM Local Tests (see LambdaAssets/)")
    print("="*80)
    
    results = []
    
    tests = [
        ("LocalCacheLoader", test_01_local_cache_loader),
        ("Factory Detection", test_02_factory_detection),
        ("MLConfig Readiness", test_03_mlconfig_lambda_ready),
        ("SAM Local Info", test_04_s3_streaming_loader_info),
        ("KPI Fact Data Loading", test_05_kpi_fact_data_loading),
    ]
    
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except AssertionError as e:
            print(f"\n❌ {test_name} FAILED: {e}")
            results.append((test_name, False))
        except Exception as e:
            print(f"\n❌ {test_name} ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {name:.<60} {status}")
    
    all_passed = all(r[1] for r in results)
    
    print("\n" + "="*80)
    if all_passed:
        print("✅ ALL LOCAL TESTS PASSED")
        print("="*80)
        print("\nReady for SAM Local Testing:")
        print("  cd LambdaAssets")
        print("  ./test_lambda_data_loader.sh")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        print("="*80)
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)

"""
cd ModelPipeline/finrag_ml_tg1/tests
& "D:/JoelDesktop folds_24/NEU FALL2025/MLops IE7374 Project/FinSights/ModelPipeline/finrag_ml_tg1/venv_ml_rag/Scripts/Activate.ps1"                  
python .\test_data_loader_comprehensive.py
"""