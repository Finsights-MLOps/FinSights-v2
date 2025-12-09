"""
Configuration validation utilities for FinRAG S3 Vectors setup.
"""

import sys
from pathlib import Path

# Find ModelPipeline root (standard pattern)
def _find_model_pipeline_root() -> Path:
    """Walk up from file location to find ModelPipeline directory"""
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if parent.name == "ModelPipeline":
            return parent
    raise RuntimeError(
        f"Cannot find ModelPipeline root directory.\n"
        f"  Searched from: {current}\n"
        f"  Expected 'ModelPipeline' directory in path tree"
    )

# Add ModelPipeline to path for absolute imports
MODEL_PIPELINE_ROOT = _find_model_pipeline_root()
sys.path.insert(0, str(MODEL_PIPELINE_ROOT))

from finrag_ml_tg1.loaders.ml_config_loader import MLConfig


def validate_s3vectors_config():
    """
    Validate S3 Vectors configuration paths and settings.
    
    Checks:
    - Base S3 path resolution
    - Provider-specific paths and dimensions
    - Auto-detection with None provider
    - Local cache path setup
    
    Returns:
        dict: Validation results with all paths
    """
    config = MLConfig()
    
    print("="*70)
    print("CONFIG VALIDATION - S3 VECTORS PATHS")
    print("="*70)
    
    # Test 1: Base path
    print(f"\n✓ Base path: {config.s3vectors_base_path}")
    
    results = {
        'base_path': config.s3vectors_base_path,
        'providers': {}
    }
    
    # Test 2: Provider-specific paths
    for provider in config.s3vectors_providers:
        s3_path = config.s3vectors_path(provider)
        dims = config.s3vectors_dimensions(provider)
        
        print(f"\n✓ Provider: {provider}")
        print(f"  S3 Path: {s3_path}")
        print(f"  Dimensions: {dims}d")
        
        results['providers'][provider] = {
            's3_path': s3_path,
            'dimensions': dims
        }
    
    # Test 3: Auto-detection (None provider)
    print(f"\n{'='*70}")
    print("AUTO-DETECTION TEST (provider=None)")
    print("="*70)
    print(f"  Default model: {config.bedrock_default_model_key}")
    print(f"  Auto-detected path: {config.s3vectors_path(None)}")
    print(f"  Auto-detected dims: {config.s3vectors_dimensions(None)}d")
    
    results['auto_detection'] = {
        'default_model': config.bedrock_default_model_key,
        'path': config.s3vectors_path(None),
        'dimensions': config.s3vectors_dimensions(None)
    }
    
    # Test 4: Local cache paths
    print(f"\n{'='*70}")
    print("LOCAL CACHE PATH VALIDATION")
    print("="*70)
    
    for provider in config.s3vectors_providers:
        cache_path = config.get_s3vectors_cache_path(provider)
        exists = cache_path.exists()
        
        print(f"\n✓ Provider: {provider}")
        print(f"  Cache: {cache_path}")
        print(f"  Exists: {exists}")
        
        results['providers'][provider]['cache_path'] = str(cache_path)
        results['providers'][provider]['cache_exists'] = exists
    
    print(f"\n{'='*70}")
    print("✓ Config Validation Complete")
    print("="*70)
    
    return results