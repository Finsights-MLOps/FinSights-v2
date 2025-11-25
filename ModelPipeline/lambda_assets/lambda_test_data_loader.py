"""
Lambda handler for testing DataLoaderStrategy in Lambda environment

This tests ONLY the data loader abstraction, not the full orchestrator.
Tests S3StreamingLoader's ability to:
  - Load Stage 2 Meta from S3
  - Cache in /tmp
  - Load dimension tables
  - Fetch sentences by ID
"""

import json
import sys
import os
from pathlib import Path
import traceback

print("="*70)
print("LAMBDA DATA LOADER TEST - STARTING")
print("="*70)

# Lambda environment paths
LAMBDA_TASK_ROOT = Path(os.getenv('LAMBDA_TASK_ROOT', '/var/task'))
MODEL_PIPELINE_ROOT = LAMBDA_TASK_ROOT / 'ModelPipeline'

print(f"\nLambda Task Root: {LAMBDA_TASK_ROOT}")
print(f"Model Pipeline Root: {MODEL_PIPELINE_ROOT}")
print(f"Model Pipeline Exists: {MODEL_PIPELINE_ROOT.exists()}")

# Add to sys.path
if str(MODEL_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_PIPELINE_ROOT))
    print(f"✓ Added to sys.path: {MODEL_PIPELINE_ROOT}")

# Import FinRAG components
try:
    from finrag_ml_tg1.loaders.ml_config_loader import MLConfig
    from finrag_ml_tg1.loaders.data_loader_factory import create_data_loader
    print("✓ Imports successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    traceback.print_exc()
    raise


def lambda_handler(event, context):
    """
    Test S3StreamingLoader in actual Lambda environment
    
    Args:
        event: Test configuration (which operations to run)
        context: Lambda context (memory, request_id, etc.)
    
    Returns:
        Test results as JSON
    """
    
    print("\n" + "="*70)
    print("LAMBDA HANDLER INVOKED")
    print("="*70)
    
    print(f"\nContext:")
    print(f"  Request ID: {context.request_id}")
    print(f"  Memory Limit: {context.memory_limit_in_mb} MB")
    print(f"  Time Remaining: {context.get_remaining_time_in_millis()} ms")
    
    print(f"\nEvent: {json.dumps(event, indent=2)}")
    
    try:
        # Step 1: Create MLConfig
        print("\n[Step 1: Initialize MLConfig]")
        config = MLConfig()
        print(f"  Environment: {'Lambda' if config.is_lambda_environment else 'Local'}")
        print(f"  Data Mode: {config.data_loading_mode}")
        print(f"  Model Root: {config.model_root}")
        
        # Step 2: Create Data Loader
        print("\n[Step 2: Create Data Loader]")
        loader = create_data_loader(config)
        loader_type = type(loader).__name__
        print(f"  Loader Type: {loader_type}")
        
        if loader_type != 'S3StreamingLoader':
            return {
                'statusCode': 500,
                'body': json.dumps({
                    'error': f'Expected S3StreamingLoader, got {loader_type}',
                    'environment': 'lambda'
                })
            }
        
        # Step 3: Run test operations
        results = {}
        operations = event.get('operations', [])
        
        if 'load_stage2_meta' in operations:
            print("\n[Step 3a: Load Stage 2 Meta from S3]")
            df = loader.load_stage2_meta()
            print(f"  ✓ Loaded {len(df):,} rows")
            print(f"  ✓ Memory: {df.estimated_size('mb'):.1f} MB")
            print(f"  ✓ Columns: {len(df.columns)}")
            
            results['stage2_meta'] = {
                'rows': len(df),
                'columns': len(df.columns),
                'memory_mb': round(df.estimated_size('mb'), 1)
            }
        
        if 'load_dimension_companies' in operations:
            print("\n[Step 3b: Load Companies Dimension]")
            df = loader.load_dimension_companies()
            print(f"  ✓ Loaded {len(df):,} companies")
            results['companies'] = {'rows': len(df)}
        
        if 'load_dimension_sections' in operations:
            print("\n[Step 3c: Load Sections Dimension]")
            df = loader.load_dimension_sections()
            print(f"  ✓ Loaded {len(df):,} sections")
            results['sections'] = {'rows': len(df)}
        
        if 'fetch_sentences_by_ids' in operations:
            print("\n[Step 3d: Fetch Sentences by ID]")
            # Get Stage 2 Meta first
            stage2_df = loader.load_stage2_meta()
            sample_ids = stage2_df['sentenceID'].head(10).to_list()
            
            fetched = loader.get_sentences_by_ids(sample_ids)
            print(f"  ✓ Requested {len(sample_ids)} IDs")
            print(f"  ✓ Retrieved {len(fetched)} sentences")
            
            results['sentence_fetch'] = {
                'requested': len(sample_ids),
                'retrieved': len(fetched)
            }
        
        # Step 4: Return success
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED IN LAMBDA")
        print("="*70)
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'test': 'data_loader',
                'environment': 'lambda',
                'loader_type': loader_type,
                'data_mode': config.data_loading_mode,
                'results': results,
                'request_id': context.request_id
            }, indent=2)
        }
        
    except Exception as e:
        print("\n" + "="*70)
        print("❌ TEST FAILED")
        print("="*70)
        print(f"\nError: {e}")
        traceback.print_exc()
        
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'error_type': type(e).__name__,
                'traceback': traceback.format_exc()
            })
        }