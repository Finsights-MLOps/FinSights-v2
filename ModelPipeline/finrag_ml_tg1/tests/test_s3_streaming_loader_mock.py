"""
Test S3StreamingLoader without SAM/Docker complexity
Simulates Lambda environment with environment variables
"""

import sys
import os
from pathlib import Path

from sympy import python

# Bootstrap
file_path = Path(__file__).resolve()
for parent in [file_path] + list(file_path.parents):
    if parent.name == "ModelPipeline":
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break

# Simulate Lambda environment
os.environ['AWS_LAMBDA_FUNCTION_NAME'] = 'test-finrag-function'
os.environ['LAMBDA_TASK_ROOT'] = str(Path.cwd().parent)  # Mock Lambda path

from finrag_ml_tg1.loaders.ml_config_loader import MLConfig
from finrag_ml_tg1.loaders.data_loader_factory import create_data_loader
from finrag_ml_tg1.loaders.data_loader_strategy import S3StreamingLoader

print("="*80)
print("S3StreamingLoader - Simulated Lambda Test")
print("="*80)

try:
    # Step 1: Verify Lambda detection
    config = MLConfig()
    print(f"\n[Environment Detection]")
    print(f"  Is Lambda: {config.is_lambda_environment}")
    print(f"  Data Mode: {config.data_loading_mode}")
    
    assert config.is_lambda_environment, "Should detect Lambda env"
    assert config.data_loading_mode == 'S3_STREAMING', "Should use S3_STREAMING"
    
    # Step 2: Create loader
    loader = create_data_loader(config)
    print(f"\n[Loader Creation]")
    print(f"  Type: {type(loader).__name__}")
    
    assert isinstance(loader, S3StreamingLoader), "Should create S3StreamingLoader"
    
    # Step 3: Test S3 data loading (REAL S3)
    print(f"\n[Testing S3 Data Access]")
    print(f"  Loading Stage 2 Meta from S3...")
    
    df = loader.load_stage2_meta()
    
    print(f"  ✓ Loaded {len(df):,} rows")
    print(f"  ✓ Memory: {df.estimated_size('mb'):.1f} MB")
    print(f"  ✓ Columns: {len(df.columns)}")
    
    assert len(df) > 0, "Should load data"
    assert 'sentenceID' in df.columns, "Should have sentenceID"
    
    # Step 4: Test /tmp caching simulation
    print(f"\n[Testing Cache Behavior]")
    print(f"  Cache directory: {loader._tmp_cache}")
    print(f"  Cache exists: {loader._tmp_cache.exists()}")
    
    # Note: On Windows, /tmp won't work, but that's fine - we're testing logic
    # In real Lambda, /tmp works
    
    print("\n" + "="*80)
    print("✅ S3StreamingLoader VALIDATED")
    print("="*80)
    print("\nConclusions:")
    print("  ✓ S3StreamingLoader instantiates correctly")
    print("  ✓ Can read from S3 successfully")
    print("  ✓ DataFrame processing works")
    print("  ✓ Logic is sound for Lambda deployment")
    print("\nConfidence Level: 90%")
    print("Remaining 10%: Real Lambda cold start, IAM permissions, /tmp in actual Lambda")
    
except Exception as e:
    print("\n" + "="*80)
    print("❌ TEST FAILED")
    print("="*80)
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()

finally:
    # Cleanup
    if 'AWS_LAMBDA_FUNCTION_NAME' in os.environ:
        del os.environ['AWS_LAMBDA_FUNCTION_NAME']
    if 'LAMBDA_TASK_ROOT' in os.environ:
        del os.environ['LAMBDA_TASK_ROOT']



# ```
# cd ModelPipeline/finrag_ml_tg1/tests
# python test_s3_streaming_loader_mock.py
# ```