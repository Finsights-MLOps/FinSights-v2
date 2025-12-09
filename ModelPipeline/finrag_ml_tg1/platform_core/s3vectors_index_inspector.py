"""
S3 Vectors index inspector utility.
Debug helper to inspect get_index response structure and validate configuration.
"""

import boto3
import json
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


def inspect_s3vectors_index(
    vector_bucket="finrag-embeddings-s3vectors",
    index_name="finrag-sentence-fact-embed-1024d",
    verbose=True
):
    """
    Inspect S3 Vectors index configuration and response structure.
    
    Useful for debugging index setup, validating configuration,
    and understanding AWS S3 Vectors API response format.
    
    Args:
        vector_bucket: S3 Vectors bucket name
        index_name: Index name to inspect
        verbose: Print detailed response structure
    
    Returns:
        dict: Full get_index response
    
    Raises:
        Exception: If get_index fails (index doesn't exist, auth issues, etc.)
    """
    config = MLConfig()
    
    # Initialize S3 Vectors client
    s3vectors = boto3.client(
        "s3vectors",
        region_name=config.region,
        aws_access_key_id=config.aws_access_key,
        aws_secret_access_key=config.aws_secret_key
    )
    
    if verbose:
        print("="*70)
        print("S3 VECTORS INDEX INSPECTOR")
        print("="*70)
        print(f"\nVector Bucket: {vector_bucket}")
        print(f"Index Name:    {index_name}")
        print(f"Region:        {config.region}")
    
    try:
        response = s3vectors.get_index(
            vectorBucketName=vector_bucket,
            indexName=index_name
        )
        
        if verbose:
            print("\n[Full Response]")
            print(json.dumps(response, indent=2, default=str))
            
            print("\n[Top-level Keys]")
            print(f"Available keys: {list(response.keys())}")
            
            # Try common paths
            print("\n[Common Paths]")
            
            if 'Index' in response:
                print("✓ Found: response['Index']")
                print(f"  Keys: {list(response['Index'].keys())}")
            elif 'index' in response:
                print("✓ Found: response['index']")
                print(f"  Keys: {list(response['index'].keys())}")
            else:
                print("✗ No 'Index' or 'index' key found")
            
            if 'Configuration' in response:
                print("✓ Found: response['Configuration']")
            
            if 'ResponseMetadata' in response:
                print("✓ Found: response['ResponseMetadata'] (AWS metadata)")
            
            print("\n" + "="*70)
        
        return response
        
    except Exception as e:
        if verbose:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            print("\n" + "="*70)
        raise



def get_index_status(
    vector_bucket="finrag-embeddings-s3vectors",
    index_name="finrag-sentence-fact-embed-1024d"
):
    """
    Check if S3 Vectors index exists and return basic info.
    
    Note: S3 Vectors get_index doesn't return a 'status' field.
    If the index exists, it returns configuration. If not, it throws exception.
    
    Args:
        vector_bucket: S3 Vectors bucket name
        index_name: Index name to check
    
    Returns:
        dict: Index info with 'exists', 'dimension', 'metric', 'created' keys
              Or {'exists': False} if not found
    """
    try:
        response = inspect_s3vectors_index(
            vector_bucket=vector_bucket,
            index_name=index_name,
            verbose=False
        )
        
        # S3 Vectors uses lowercase 'index' key
        if 'index' in response:
            idx = response['index']
            return {
                'exists': True,
                'dimension': idx.get('dimension'),
                'metric': idx.get('distanceMetric'),
                'created': idx.get('creationTime'),
                'arn': idx.get('indexArn'),
                'non_filterable_keys': idx.get('metadataConfiguration', {}).get('nonFilterableMetadataKeys', [])
            }
        
        # Fallback if structure changes
        return {'exists': True, 'structure_unknown': True}
        
    except Exception:
        return {'exists': False}
    
    