"""
ML Config Loader - Standalone Configuration for Embedding Pipeline
Loads AWS credentials and ML-specific settings from YAML configs

python loaders/ml_config_loader.py

"""

import os
import yaml
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any

class MLConfig:
    """
    Standalone ML configuration loader
    Manages AWS credentials, S3 paths, and embedding settings
    """
    
    def __init__(self, config_path=None):
        """Load YAML configuration and credentials"""
        
        # Step 1: Find and store ModelPipeline root (foundation for everything)
        self.model_root = self._find_model_pipeline_root()
        
        # Step 2: Load ML path config (now using model_root!)
        if config_path is None:
            config_path = self.model_root / 'finrag_ml_tg1' / '.aws_config' / 'ml_config.yaml'
        
        # Step 3: Load YAML
        with open(config_path, encoding='utf-8') as f:
            self.cfg = yaml.safe_load(f)
        
        # Step 4: Load credentials (AWS + ML APIs)
        self._load_aws_credentials()
        self._load_ml_credentials()

        

    
    # ========== Credential Loading ==========
    
    def _load_aws_credentials(self):
        """Load AWS credentials from .aws_secrets/aws_credentials.env"""
        aws_creds_path = self.model_root / 'finrag_ml_tg1' / '.aws_secrets' / 'aws_credentials.env'
        
        if aws_creds_path.exists():
            load_dotenv(aws_creds_path, override=True)
            
            if os.getenv('AWS_ACCESS_KEY_ID') and os.getenv('AWS_SECRET_ACCESS_KEY'):
                print(f"[DEBUG] ✓ AWS credentials loaded from {aws_creds_path.name}")
                self._aws_creds_source = str(aws_creds_path.name)
            else:
                raise ValueError(f"AWS credentials file exists but is empty: {aws_creds_path}")
        else:
            raise FileNotFoundError(
                f"AWS credentials not found!\n"
                f"  Expected: {aws_creds_path.absolute()}\n"
                f"  Create file with:\n"
                f"    AWS_ACCESS_KEY_ID=AKIA...\n"
                f"    AWS_SECRET_ACCESS_KEY=..."
            )
    
    def _load_ml_credentials(self):
        """Load ML API keys from .aws_secrets/aws_credentials.env (same file)"""
        # Already loaded in _load_aws_credentials
        pass
    
    # ========== AWS Configuration ==========
    
    @property
    def bucket(self):
        """S3 bucket name"""
        return self.cfg['s3']['bucket_name']
    
    @property
    def region(self):
        """AWS region"""
        return os.getenv('AWS_DEFAULT_REGION', self.cfg['s3']['region'])
    
    @property
    def aws_access_key(self):
        """AWS access key from environment"""
        return os.getenv('AWS_ACCESS_KEY_ID')
    
    @property
    def aws_secret_key(self):
        """AWS secret key from environment"""
        return os.getenv('AWS_SECRET_ACCESS_KEY')
    
    # ========== Data Paths - Input (ETL Layer) ==========
    
    @property
    def input_sentences_path(self):
        """Stage 1: Original fact table from ETL (24 columns)"""
        d = self.cfg['data_etl']['sentence_fact']
        return f"{d['path']}/{d['filename']}"
    
    # ========== Data Paths - Output (ML Layer) ==========
    
    @property
    def meta_embeds_path(self):
        """Stage 2: Enhanced fact table with ML metadata (35 columns)"""
        d = self.cfg['data_ml']['meta_embeds']
        return f"{d['path']}/{d['filename']}"
    
    

    ## if provider is None: block needs to BUILD the provider string before trying to use it as a dict key.
    def embeddings_path(self, provider=None):
        """
        Embedding storage path for specific provider
        Auto-detects provider based on current model config if not specified
        """
        if provider is None:
            # Dynamic detection based on default model
            if self.embedding_provider == 'bedrock':
                model_key = self.bedrock_default_model_key
                dims = self.bedrock_dimensions
                
                # Map model to provider key
                if 'cohere' in model_key.lower():
                    provider = f"cohere_{dims}d"
                elif 'titan' in model_key.lower():
                    provider = f"titan_{dims}d"
                else:
                    # Fallback
                    provider = f"cohere_{dims}d"
            else:
                # Direct API provider fallback
                provider = 'cohere_768d'
        
        d = self.cfg['data_ml']['embeddings'][provider]
        return f"{d['path']}/{d['filename']}"


    def embeddings_metadata_path(self, provider='cohere_768d'):
        """Embedding metadata JSON path"""
        d = self.cfg['data_ml']['embeddings'][provider]
        return f"{d['path']}/{d['metadata_file']}"
    
    # ========== Embedding Configuration - Bedrock ==========
    
    @property
    def embedding_provider(self):
        """Default embedding provider (bedrock, cohere, openai)"""
        return self.cfg['embedding']['default_provider']
    
    @property
    def bedrock_region(self):
        """Bedrock region"""
        return self.cfg['embedding']['bedrock']['region']
    
    @property
    def bedrock_default_model_key(self):
        """Default Bedrock model key (cohere_embed_v3, titan_v2, etc.)"""
        return self.cfg['embedding']['bedrock']['default_model']
    
    @property
    def bedrock_model_id(self):
        """Get Bedrock model ID for default model"""
        model_key = self.bedrock_default_model_key
        return self.cfg['embedding']['bedrock']['models'][model_key]['model_id']
    
    @property
    def bedrock_dimensions(self):
        """Get embedding dimensions for default Bedrock model"""
        model_key = self.bedrock_default_model_key
        return self.cfg['embedding']['bedrock']['models'][model_key]['dimensions']
    
    @property
    def bedrock_batch_size(self):
        """Get batch size for default Bedrock model"""
        model_key = self.bedrock_default_model_key
        return self.cfg['embedding']['bedrock']['models'][model_key]['batch_size']
    
    @property
    def bedrock_input_type(self):
        """Get input_type for default Bedrock model"""
        model_key = self.bedrock_default_model_key
        return self.cfg['embedding']['bedrock']['models'][model_key]['input_type']
    
    def get_bedrock_model_config(self, model_key=None):
        """Get full config for a specific Bedrock model"""
        if model_key is None:
            model_key = self.bedrock_default_model_key
        return self.cfg['embedding']['bedrock']['models'][model_key]
    
    # ========== Legacy Properties (for backward compatibility) ==========
    
    @property
    def embedding_model(self):
        """Model ID for current provider"""
        if self.embedding_provider == 'bedrock':
            return self.bedrock_model_id
        else:
            provider = self.embedding_provider
            return self.cfg['embedding'][provider]['model']
    
    @property
    def embedding_dimensions(self):
        """Vector dimensions for current provider"""
        if self.embedding_provider == 'bedrock':
            return self.bedrock_dimensions
        else:
            provider = self.embedding_provider
            return self.cfg['embedding'][provider]['dimensions']
    
    @property
    def embedding_batch_size(self):
        """Batch size for current provider"""
        if self.embedding_provider == 'bedrock':
            return self.bedrock_batch_size
        else:
            provider = self.embedding_provider
            return self.cfg['embedding'][provider]['batch_size']
    
    @property
    def cohere_api_key(self):
        """Cohere API key from environment (for direct API access)"""
        return os.getenv('COHERE_API_KEY')
    
    @property
    def openai_api_key(self):
        """OpenAI API key from environment"""
        return os.getenv('OPENAI_API_KEY')
    
    # ========== Filtering Configuration ==========
    
    @property
    def min_char_length(self):
        """Minimum sentence character length"""
        return self.cfg['embedding']['filtering']['min_char_length']
    
    @property
    def max_char_length(self):
        """Maximum sentence character length"""
        return self.cfg['embedding']['filtering']['max_char_length']
    
    @property
    def max_token_count(self):
        """Maximum tokens per sentence"""
        return self.cfg['embedding']['filtering']['max_token_count']
    
    @property
    def exclude_sections(self):
        """List of sections to exclude from embedding"""
        return self.cfg['embedding']['filtering']['exclude_sections']
    
    # ========== Retrieval Configuration ==========
    
    @property
    def retrieval_top_k(self):
        """Number of top results to retrieve"""
        return self.cfg['retrieval']['top_k']
    
    @property
    def context_window(self):
        """Number of sentences to retrieve around target (±N)"""
        return self.cfg['retrieval']['context_window']
    
    @property
    def priority_sections(self):
        """High-priority sections for retrieval"""
        return self.cfg['retrieval']['priority_sections']
    
    @property
    def recent_years_threshold(self):
        """Filter to filings after this year"""
        return self.cfg['retrieval']['recent_years_threshold']
    
    # ========== Cost Tracking ==========
    
    @property
    def embedding_budget(self):
        """Total embedding budget in USD"""
        return self.cfg['costs']['embedding_budget_usd']
    
    @property
    def alert_threshold(self):
        """Alert when budget usage exceeds this percentage"""
        return self.cfg['costs']['alert_threshold_pct']
    
    def get_cost_per_1k(self, provider='cohere_768d'):
        """Get cost per 1K tokens for provider"""
        return self.cfg['costs']['rates'].get(provider, 0.0001)
    
    # ========== AWS Client Helper Methods ==========
    
    def s3_uri(self, key):
        """Convert S3 key to full URI"""
        return f"s3://{self.bucket}/{key}"
    
    def get_s3_client(self):
        """Create boto3 S3 client with loaded credentials"""
        import boto3
        return boto3.client(
            's3',
            aws_access_key_id=self.aws_access_key,
            aws_secret_access_key=self.aws_secret_key,
            region_name=self.region
        )
    
    def get_bedrock_client(self):
        """Create Bedrock runtime client for embeddings"""
        import boto3
        return boto3.client(
            service_name='bedrock-runtime',
            region_name=self.bedrock_region,
            aws_access_key_id=self.aws_access_key,
            aws_secret_access_key=self.aws_secret_key
        )
    
    def get_storage_options(self):
        """Get Polars/PyArrow storage options for S3 access"""
        return {
            'aws_access_key_id': self.aws_access_key,
            'aws_secret_access_key': self.aws_secret_key,
            'aws_region': self.region
        }
    
    def download_from_s3(self, s3_key, local_path):
        """Download file from S3"""
        s3_client = self.get_s3_client()
        s3_client.download_file(self.bucket, s3_key, local_path)
        print(f"✓ Downloaded: s3://{self.bucket}/{s3_key} → {local_path}")
    
    def upload_to_s3(self, local_path, s3_key):
        """Upload file to S3"""
        s3_client = self.get_s3_client()
        s3_client.upload_file(local_path, self.bucket, s3_key)
        print(f"✓ Uploaded: {local_path} → s3://{self.bucket}/{s3_key}")


    @property
    def embedding_mode(self):
        return self.cfg['embedding_execution']['mode']

    @property
    def filter_cik(self):
        return self.cfg['embedding_execution']['filters']['cik_int']

    @property
    def filter_year(self):
        return self.cfg['embedding_execution']['filters']['year']
    
    @property
    def filter_sections(self):
        return self.cfg['embedding_execution']['filters']['sections']
    


# ============================================================================
    # S3 VECTORS STAGING PATHS (Stage 3)
    # ============================================================================

    @property
    def s3vectors_base_path(self):
        """Base path for S3 Vectors staging area"""
        return self.cfg['data_ml']['s3_vectors_staging']['base_path']

    def s3vectors_path(self, provider=None):
        """
        Get S3 Vectors staging path for specific provider or default
        
        Args:
            provider: Model provider key (e.g., 'cohere_1024d', 'titan_1024d')
                     If None, auto-detects from bedrock_default_model_key
        
        Returns:
            Full S3 key path: ML_EMBED_ASSETS/S3_VECTORS_STAGING/{provider}/{filename}
        """
        if provider is None:
            # Auto-detect provider (matches embeddings_path logic)
            model_key = self.bedrock_default_model_key
            dims = self.bedrock_dimensions
            
            if 'cohere' in model_key.lower():
                provider = f"cohere_{dims}d"
            elif 'titan' in model_key.lower():
                provider = f"titan_{dims}d"
            else:
                provider = f"cohere_{dims}d"
        
        provider_config = self.cfg['data_ml']['s3_vectors_staging'].get(provider)
        if not provider_config:
            available = [k for k in self.cfg['data_ml']['s3_vectors_staging'].keys() if k != 'base_path']
            raise ValueError(f"Unknown S3 Vectors provider: {provider}. Available: {available}")
        
        return f"{provider_config['path']}/{provider_config['filename']}"

    def s3vectors_dimensions(self, provider=None):
        """Get vector dimensions for specific provider"""
        if provider is None:
            model_key = self.bedrock_default_model_key
            dims = self.bedrock_dimensions
            
            if 'cohere' in model_key.lower():
                provider = f"cohere_{dims}d"
            elif 'titan' in model_key.lower():
                provider = f"titan_{dims}d"
            else:
                provider = f"cohere_{dims}d"
        
        provider_config = self.cfg['data_ml']['s3_vectors_staging'].get(provider)
        if not provider_config:
            raise ValueError(f"Unknown S3 Vectors provider: {provider}")
        
        return provider_config.get('dimensions', self.bedrock_dimensions)

    @property
    def s3vectors_providers(self):
        """List all configured S3 Vectors providers"""
        return [k for k in self.cfg['data_ml']['s3_vectors_staging'].keys() if k != 'base_path']

    def get_s3vectors_cache_path(self, provider):
        """
        Get local cache path for S3 Vectors staging table
        
        Args:
            provider: Model provider (e.g., 'cohere_1024d')
        
        Returns:
            Path: ModelPipeline/finrag_ml_tg1/data_cache/stage3_s3vectors/{provider}/{filename}.parquet
        """
        s3_key = self.s3vectors_path(provider)
        filename = Path(s3_key).name
        
        # Use model_root - works everywhere
        cache_path = self.model_root / 'finrag_ml_tg1' / 'data_cache' / 'stage3_s3vectors' / provider / filename
        return cache_path

    # ========= Semantic Variants & Retrieval Config ==========

    def get_variant_config(self) -> Dict[str, Any]:
        """
        Get semantic variants configuration from YAML.
        
        Returns:
            Dict with keys: enabled, model_id, max_tokens, temperature, count, prompt_template
        
        Raises:
            KeyError: If 'semantic_variants' section missing from config
        """
        if "semantic_variants" not in self.cfg:
            raise KeyError("'semantic_variants' section not found in ml_config.yaml")
        
        return self.cfg["semantic_variants"]


    def get_retrieval_config(self) -> Dict[str, Any]:
        """
        Get retrieval pipeline configuration from YAML.
        
        Returns:
            Dict with keys: top_k_filtered, top_k_global, enable_global, 
                        enable_variants, recent_year_threshold, min_similarity,
                        vector_bucket, index_name, dimensions
        
        Raises:
            KeyError: If 'retrieval' section missing from config
        """
        if "retrieval" not in self.cfg:
            raise KeyError("'retrieval' section not found in ml_config.yaml")
        
        return self.cfg["retrieval"]



    # ========= Serving Models Configuration ==========

    def get_serving_models_config(self) -> Dict[str, Any]:
        """
        Get serving models configuration for LLM synthesis.
        
        Returns:
            Dict with keys: development, production_balanced, production_budget, 
                        openai_compatible, default_serving_model
        
        Raises:
            KeyError: If 'serving_models' section missing from config
        """
        if "serving_models" not in self.cfg:
            raise KeyError("'serving_models' section not found")
        
        serving = self.cfg["serving_models"]
        
        # print(f"\n[DEBUG] Keys in serving_models: {list(serving.keys())}")
        # print(f"[DEBUG] 'default_serving_model' in serving_models? {'default_serving_model' in serving}")
        
        return serving

    def get_serving_model(self, model_key: str = None) -> Dict[str, Any]:
        """
        Get specific serving model configuration.
        
        Args:
            model_key: Model key (e.g., 'development', 'production_balanced')
                    If None, uses default_serving_model from config
        
        Returns:
            Dict with model configuration (model_id, max_tokens, etc.)
        
        Raises:
            KeyError: If model_key not found in serving_models config
        """
        serving_config = self.get_serving_models_config()
        
        if model_key is None:
            model_key = serving_config.get('default_serving_model', 'production_balanced')
        
        if model_key not in serving_config:
            available = [k for k in serving_config.keys() if k != 'default_serving_model']
            raise KeyError(
                f"Model '{model_key}' not found in serving_models config.\n"
                f"Available models: {available}"
            )
        
        return serving_config[model_key]


    def get_default_serving_model(self) -> Dict[str, Any]:
            """
            Get default serving model configuration.
            
            Returns:
                Dict with model configuration for default model
                
            Raises:
                KeyError: If default_serving_model not configured or points to invalid model
            """
            serving_config = self.get_serving_models_config()
            
            # No hardcoded fallback - trust YAML config
            if 'default_serving_model' not in serving_config:
                raise KeyError(
                    "'default_serving_model' key missing in ml_config.yaml.\n"
                    "Add this key inside the serving_models section:\n"
                    "  serving_models:\n"
                    "    development: {...}\n"
                    "    default_serving_model: 'development'  # At this level"
                )
            
            default_key = serving_config['default_serving_model']
            
            # Validate that the default key points to an actual model
            if default_key not in serving_config:
                available = [k for k in serving_config.keys() if k != 'default_serving_model']
                raise KeyError(
                    f"Default model '{default_key}' not found in serving_models.\n"
                    f"Available models: {available}\n"
                    f"Update default_serving_model in ml_config.yaml"
                )
            
            return serving_config[default_key]



    # ============================================================================
    # DIMENSION TABLE PATHS
    # ============================================================================

    @property
    def dimension_companies_path(self):
        """
        Get S3 key for companies dimension table
        
        Returns:
            str: S3 key like 'DATA_MERGE_ASSETS/DIMENSION_TABLES/finrag_dim_companies_21.parquet'
        """
        dim_config = self.cfg['data_ml'].get('dimensions', {})
        
        # Fallback if not in YAML (use known path)
        if not dim_config:
            return "DATA_MERGE_ASSETS/DIMENSION_TABLES/finrag_dim_companies_21.parquet"
        
        path = dim_config.get('path', 'DATA_MERGE_ASSETS/DIMENSION_TABLES')
        filename = dim_config.get('companies', {}).get('filename', 'finrag_dim_companies_21.parquet')
        
        return f"{path}/{filename}"

    @property
    def dimension_sections_path(self):
        """
        Get S3 key for sections dimension table
        
        Returns:
            str: S3 key like 'DATA_MERGE_ASSETS/DIMENSION_TABLES/finrag_dim_sec_sections.parquet'
        """
        dim_config = self.cfg['data_ml'].get('dimensions', {})
        
        # Fallback if not in YAML
        if not dim_config:
            return "DATA_MERGE_ASSETS/DIMENSION_TABLES/finrag_dim_sec_sections.parquet"
        
        path = dim_config.get('path', 'DATA_MERGE_ASSETS/DIMENSION_TABLES')
        filename = dim_config.get('sections', {}).get('filename', 'finrag_dim_sec_sections.parquet')
        
        return f"{path}/{filename}"


    @property
    def kpi_fact_data_path(self):
        """
        Get S3 key for KPI fact data table
        
        Returns:
            str: S3 key like 'DATA_MERGE_ASSETS/FINRAG_FACT_METRICS/KPI_FACT_DATA_EDGAR.parquet'
        """
        kpi_config = self.cfg['data_ml'].get('kpi_facts', {})
        
        # Fallback if not in YAML (use known path)
        if not kpi_config:
            return "DATA_MERGE_ASSETS/FINRAG_FACT_METRICS/KPI_FACT_DATA_EDGAR.parquet"
        
        path = kpi_config.get('path', 'DATA_MERGE_ASSETS/FINRAG_FACT_METRICS')
        filename = kpi_config.get('filename', 'KPI_FACT_DATA_EDGAR.parquet')
        
        return f"{path}/{filename}"


    # ============================================================================
    # QUERY LOGGING PATHS (QueryLogger S3 exports)
    # ============================================================================

    @property
    def query_logs_path(self):
        """
        Get S3 prefix for query logs
        
        Returns:
            str: S3 prefix like 'DATA_MERGE_ASSETS/LOGS/FINRAG/logs'
        """
        logging_config = self.cfg['data_ml'].get('query_logging', {})
        
        # Fallback if not in YAML
        if not logging_config:
            return "DATA_MERGE_ASSETS/LOGS/FINRAG/logs"
        
        return logging_config.get('logs', {}).get('path', 'DATA_MERGE_ASSETS/LOGS/FINRAG/logs')

    @property
    def query_contexts_path(self):
        """
        Get S3 prefix for query contexts
        
        Returns:
            str: S3 prefix like 'DATA_MERGE_ASSETS/LOGS/FINRAG/contexts'
        """
        logging_config = self.cfg['data_ml'].get('query_logging', {})
        
        if not logging_config:
            return "DATA_MERGE_ASSETS/LOGS/FINRAG/contexts"
        
        return logging_config.get('contexts', {}).get('path', 'DATA_MERGE_ASSETS/LOGS/FINRAG/contexts')

    @property
    def query_responses_path(self):
        """
        Get S3 prefix for query responses
        
        Returns:
            str: S3 prefix like 'DATA_MERGE_ASSETS/LOGS/FINRAG/responses'
        """
        logging_config = self.cfg['data_ml'].get('query_logging', {})
        
        if not logging_config:
            return "DATA_MERGE_ASSETS/LOGS/FINRAG/responses"
        
        return logging_config.get('responses', {}).get('path', 'DATA_MERGE_ASSETS/LOGS/FINRAG/responses')


    ## ========= New Methods for Lambda Environment Support ==========

    def _find_model_pipeline_root(self) -> Path:
        """
        Find ModelPipeline root directory.
        Supports both local development and Lambda environments.
        
        Search order:
        1. Lambda environment (/var/task/ModelPipeline)
        2. File's own path tree (where this .py file lives)
        3. Current working directory tree
        """
        # Lambda detection
        if os.getenv('AWS_LAMBDA_FUNCTION_NAME'):
            lambda_task_root = Path(os.getenv('LAMBDA_TASK_ROOT', '/var/task'))
            model_pipeline = lambda_task_root / "ModelPipeline"
            if model_pipeline.exists():
                print(f"[DEBUG] ✓ Lambda environment detected: {model_pipeline}")
                return model_pipeline
            else:
                print(f"[WARNING] Lambda env detected but ModelPipeline not found at {model_pipeline}")
        
        # Strategy 1: Check file's own location first
        # If this file is at: ModelPipeline/finrag_ml_tg1/loaders/ml_config_loader.py
        # Then walk up from file location
        file_path = Path(__file__).resolve()
        for parent in [file_path] + list(file_path.parents):
            if parent.name == "ModelPipeline":
                print(f"[DEBUG] ✓ Found ModelPipeline via file path: {parent}")
                return parent
        
        # Strategy 2: Check current working directory (for notebook usage)
        current = Path.cwd()
        for parent in [current] + list(current.parents):
            if parent.name == "ModelPipeline":
                print(f"[DEBUG] ✓ Found ModelPipeline via cwd: {parent}")
                return parent
        
        # Strategy 3: Failed - provide helpful error
        raise RuntimeError(
            f"Cannot find ModelPipeline root directory.\n"
            f"  Searched in file path: {file_path}\n"
            f"  Searched in cwd: {current}\n"
            f"  Expected 'ModelPipeline' directory in path tree or Lambda /var/task/ModelPipeline"
        )

    # Add new properties (after __init__):
    @property
    def is_lambda_environment(self) -> bool:
        """Detect if running in AWS Lambda"""
        return bool(os.getenv('AWS_LAMBDA_FUNCTION_NAME'))

    @property
    def data_loading_mode(self) -> str:
        """
        Determine data loading strategy.
        Returns: 'LOCAL_CACHE' or 'S3_STREAMING'
        """
        return 'S3_STREAMING' if self.is_lambda_environment else 'LOCAL_CACHE'

# ============================================================================
# TEST / DEMO
# ============================================================================


if __name__ == "__main__":
    try:
        config = MLConfig()
        
        print("=" * 80)
        print("ML PIPELINE CONFIGURATION - COMPREHENSIVE TEST")
        print("=" * 80)
        
        # ====================================================================
        # AWS CONFIGURATION
        # ====================================================================
        print(f"\n[AWS Configuration]")
        print(f"  Credentials Source: {config._aws_creds_source}")
        print(f"  Bucket: {config.bucket}")
        print(f"  Region: {config.region}")
        print(f"  Access Key: {config.aws_access_key[:12]}..." if config.aws_access_key else "  ERROR: Missing")
        print(f"  Secret Key: {'*' * 40}... (hidden)")
        
        # ====================================================================
        # DATA PATHS
        # ====================================================================
        print(f"\n[Data Paths - Input (ETL)]")
        print(f"  Stage 1: {config.s3_uri(config.input_sentences_path)}")
        
        print(f"\n[Data Paths - Output (ML)]")
        print(f"  Stage 2 Meta: {config.s3_uri(config.meta_embeds_path)}")
        print(f"  Embeddings: {config.s3_uri(config.embeddings_path())}")
        
        # ====================================================================
        # EMBEDDING CONFIGURATION
        # ====================================================================
        print(f"\n[Embedding Config - Bedrock]")
        print(f"  Provider: {config.embedding_provider}")
        print(f"  Default Model Key: {config.bedrock_default_model_key}")
        print(f"  Model ID: {config.bedrock_model_id}")
        print(f"  Dimensions: {config.bedrock_dimensions}")
        print(f"  Batch Size: {config.bedrock_batch_size}")
        print(f"  Input Type: {config.bedrock_input_type}")
        print(f"  Region: {config.bedrock_region}")
        
        print(f"\n[Filtering]")
        print(f"  Char Length: {config.min_char_length} - {config.max_char_length}")
        print(f"  Max Tokens: {config.max_token_count}")
        print(f"  Exclude Sections: {', '.join(config.exclude_sections)}")
        
        # ====================================================================
        # RETRIEVAL CONFIGURATION
        # ====================================================================
        print(f"\n[Retrieval Config]")
        retrieval = config.get_retrieval_config()
        print(f"  Top K Filtered: {retrieval['top_k_filtered']}")
        print(f"  Top K Global: {retrieval['top_k_global']}")
        print(f"  Enable Global: {retrieval['enable_global']}")
        print(f"  Enable Variants: {retrieval['enable_variants']}")
        print(f"  Window Size: {retrieval['window_size']}")
        print(f"  Index Name: {retrieval['index_name']}")
        print(f"  Vector Bucket: {retrieval['vector_bucket']}")
        
        # ====================================================================
        # SERVING MODELS CONFIGURATION (CRITICAL SECTION)
        # ====================================================================
        print(f"\n[Serving Models Config]")
        print("-" * 80)
        
        serving_config = config.get_serving_models_config()
        
        # Check if default_serving_model is nested correctly
        print(f"\nKeys in serving_models section:")
        for key in serving_config.keys():
            if key == 'default_serving_model':
                print(f"  {key}: '{serving_config[key]}' <- DEFAULT KEY")
            else:
                model = serving_config[key]
                print(f"  {key}: {model.get('display_name', 'N/A')}")
        
        print(f"\nDefault Key Location Check:")
        if 'default_serving_model' in serving_config:
            print(f"  STATUS: INSIDE serving_models (CORRECT)")
            print(f"  Value: '{serving_config['default_serving_model']}'")
        else:
            print(f"  STATUS: MISSING or OUTSIDE serving_models (ERROR)")
            print(f"  Check YAML indentation!")
        
        # Test default model retrieval
        print(f"\nDefault Model Retrieval Test:")
        try:
            default_model = config.get_default_serving_model()
            print(f"  Default Key: {serving_config.get('default_serving_model', 'NOT FOUND')}")
            print(f"  Model ID: {default_model['model_id']}")
            print(f"  Display Name: {default_model['display_name']}")
            print(f"  Max Tokens: {default_model['max_tokens']}")
            print(f"  Input Cost: ${default_model['cost_per_1k_input']}/1K tokens")
            print(f"  Output Cost: ${default_model['cost_per_1k_output']}/1K tokens")
            print(f"  STATUS: SUCCESS")
        except Exception as e:
            print(f"  STATUS: FAILED")
            print(f"  Error: {e}")
        
        # List all available models
        print(f"\nAll Available Serving Models:")
        print("-" * 80)
        model_keys = [k for k in serving_config.keys() if k != 'default_serving_model']
        for i, key in enumerate(model_keys, 1):
            model = serving_config[key]
            is_default = (key == serving_config.get('default_serving_model'))
            marker = " [DEFAULT]" if is_default else ""
            
            print(f"\n  {i}. {key}{marker}")
            print(f"     Model ID: {model['model_id']}")
            print(f"     Display: {model['display_name']}")
            print(f"     Cost: ${model['cost_per_1k_input']}/1K in, ${model['cost_per_1k_output']}/1K out")
            print(f"     Context: {model['context_window']:,} tokens")
            print(f"     Use Case: {model.get('use_case', 'N/A')}")
            
            # Check for CRIS prefix
            if model['model_id'].startswith('us.') or model['model_id'].startswith('eu.'):
                print(f"     CRIS: YES (cross-region inference)")
            else:
                print(f"     CRIS: NO (single region)")
        
        # ====================================================================
        # SEMANTIC VARIANTS CONFIGURATION
        # ====================================================================
        print(f"\n[Semantic Variants Config]")
        variants = config.get_variant_config()
        print(f"  Enabled: {variants['enabled']}")
        print(f"  Model ID: {variants['model_id']}")
        print(f"  Count: {variants['count']} variants per query")
        print(f"  Max Tokens: {variants['max_tokens']}")
        
        # ====================================================================
        # COST TRACKING
        # ====================================================================
        print(f"\n[Cost Tracking]")
        print(f"  Embedding Budget: ${config.embedding_budget:.2f}")
        print(f"  Alert Threshold: {config.alert_threshold}%")
        print(f"  Cohere 768d Rate: ${config.get_cost_per_1k('cohere_768d'):.5f}/1K tokens")
        
        # ====================================================================
        # EMBEDDING EXECUTION PARAMETERS
        # ====================================================================
        print(f"\n[Embedding Execution]")
        print(f"  Mode: {config.embedding_mode}")
        print(f"  Filter CIKs: {config.filter_cik}")
        print(f"  Filter Years: {config.filter_year}")
        print(f"  Filter Sections: {config.filter_sections}")
        
        print(f"\n[Embedding Execution]")
        print(f"  Mode: {config.embedding_mode}")
        print(f"  Filter CIK: {config.filter_cik}")
        print(f"  Filter Year: {config.filter_year}")
        print(f"  Filter Sections: {config.filter_sections}")
        
        # ====================================================================
        # NEW: LAMBDA REFACTORING VALIDATION
        # ====================================================================
        print(f"\n" + "=" * 80)
        print("LAMBDA REFACTORING - NEW FEATURES")
        print("=" * 80)
        
        print(f"\n[Model Root Resolution]")
        print(f"  Model Root: {config.model_root}")
        print(f"  Root Type: {type(config.model_root).__name__}")
        print(f"  Root Exists: {config.model_root.exists()}")
        print(f"  Config Path: {config.model_root / 'finrag_ml_tg1' / '.aws_config' / 'ml_config.yaml'}")
        
        print(f"\n[Environment Detection]")
        print(f"  Is Lambda: {config.is_lambda_environment}")
        print(f"  Data Loading Mode: {config.data_loading_mode}")
        print(f"  Expected Mode: {'S3_STREAMING' if config.is_lambda_environment else 'LOCAL_CACHE'}")
        
        print(f"\n[Path Resolution Tests]")
        # Test all major path methods using model_root
        print(f"  ✓ AWS Creds Path: {config.model_root / 'finrag_ml_tg1' / '.aws_secrets' / 'aws_credentials.env'}")
        print(f"  ✓ YAML Config Path: {config.model_root / 'finrag_ml_tg1' / '.aws_config' / 'ml_config.yaml'}")
        
        # Test cache path methods
        try:
            s3vec_cache = config.get_s3vectors_cache_path('cohere_1024d')
            print(f"  ✓ S3 Vectors Cache: {s3vec_cache}")
            print(f"    Uses model_root: {str(config.model_root) in str(s3vec_cache)}")
        except Exception as e:
            print(f"  ✗ S3 Vectors Cache Error: {e}")
        
        print(f"\n[Data Loader Integration Ready]")
        print(f"  Config has model_root: {hasattr(config, 'model_root')}")
        print(f"  Config has is_lambda_environment: {hasattr(config, 'is_lambda_environment')}")
        print(f"  Config has data_loading_mode: {hasattr(config, 'data_loading_mode')}")
        print(f"  ✓ Ready for create_data_loader(config)")
        
        # ====================================================================
        # FINAL VALIDATION
        # ====================================================================
        print(f"\n" + "=" * 80)
        print("✓ CONFIGURATION LOADED SUCCESSFULLY - LAMBDA-READY!")
        print("=" * 80)
        
        print(f"\n[Quick Stats]")
        print(f"  Total config sections: {len(config.cfg)}")
        print(f"  S3 bucket: {config.bucket}")
        print(f"  Bedrock model: {config.bedrock_model_id}")
        print(f"  Default serving model: {config.get_default_serving_model()['display_name']}")
        print(f"  Environment: {'Lambda' if config.is_lambda_environment else 'Local Development'}")

    except Exception as e:
        print("\n" + "=" * 80)
        print("CONFIGURATION ERROR")
        print("=" * 80)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        print("\n" + "=" * 80)