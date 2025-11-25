### Container Image Deployment 

GitHub Repo (FinSights/)
    ↓ [GitHub Actions builds Docker image]
    ↓
Lambda receives: ECR container image
    ↓
Runs as: Standard container with exact directory structure
    /app/
    ├── ModelPipeline/
    ├── DataPipeline/
    ├── Frontend/
    └── lambda_handler.py

- Directory Structure WILL Be Preserved

```
Lambda, this WILL work:
/var/task/
├── FinSights/                    # repo root
│   ├── ModelPipeline/           # ✅ Present
│   │   ├── finrag_ml_tg1/
│   │   └── loaders/
│   ├── DataPipeline/            # ✅ Present
│   ├── Frontend/                # ✅ Present (if you include it)
│   └── lambda_function.py       # New handler
```

- `/opt/python` Lambda Layer pattern they showed is **optional optimization**, not required structure.

### **Lambda Layers (opt)**
- Used for **shared dependencies** across multiple Lambda functions
- Example: If you had 5 different Lambdas all using `boto3`, you'd create one layer with `boto3`
- **Your case**: Single Lambda function = No benefit from layers
- **Skip layers entirely** - just deploy your whole repo

###  /tmp Directory 
- Lambda's `/tmp` is limited to 512MB
- ephemeral **per container instance**, not per invocation
- Warm containers keep `/tmp` contents between invocations (for ~15 minutes idle)
- Use `/tmp` as a cache for downloaded Parquet files


### Validation Checklist:
- GitHub Actions workflow syntax valid
- AWS credentials configured as secrets
- SAM build succeeds in CI environment
- Deployment completes without errors
- API Gateway URL returned

### Config and loader debug:
- YAML Structure (ml_config.yaml)
  - All S3 paths (bucket, base_path, filenames) - STAYS SAME
  - Bedrock config (model_id, dimensions, batch_size, temperature) - STAYS SAME
  - Cost per token, region, all AWS settings - STAYS SAME
  - Retrieval config (top_k, similarity_threshold) - STAYS SAME
  - Serving models config (Haiku, Sonnet, Nova) - STAYS SAME

- MLConfig Properties (95% unchanged)
  - bucket, region, aws_access_key, aws_secret_key - STAYS SAME
  - bedrock_model_id, bedrock_dimensions, bedrock_batch_size - STAYS SAME
  - input_sentences_path(), meta_embeds_path(), embeddings_path() - STAYS SAME
  - get_storage_options(), get_s3_client(), get_bedrock_client() - STAYS SAME
  - All S3 URI generation methods - STAYS SAME

### Loader Behavior
- "streaming" in the sense of "not using local cache files", but it does load into RAM.
- LocalCacheLoader (Development), S3StreamingLoader (Lambda),  /tmp Caching Strategy
- Cold start: 2-3s to download from S3 / Warm start: 0.2s to read from /tmp. 
- Increments: 1 MB steps // Cost: Scales linearly (~$0.0000166667 per GB-second)
  - 2 GB (2048 MB): Good starting point, handles your current data
  - 3 GB (3072 MB): +50% headroom if Stage 2 grows
  - 4 GB (4096 MB): Comfortable for future scaling
  - 10 GB: Maximum, for truly huge datasets

### LambdaAssets Directory Structure
1. LambdaAssets/ directory structure
2. template.yaml (SAM configuration)
3. lambda_test_data_loader.py (test handler)
4. test_events/test_data_loader.json (test input)
5. test_lambda_data_loader.sh (test script)


### SAM Testing Flow
``` 
-- run: sam local invoke
    ↓
SAM reads: template.yaml (what function to run?)
    ↓
SAM reads: test_events/test_data_loader.json (what input data?)
    ↓
SAM launches: Docker container with Lambda environment
    ↓
SAM executes: lambda_test_data_loader.py (your test code)
    ↓
Returns: Results to your terminal
```

1. template.yaml - The Blueprint
- What it is: SAM's configuration file (like a recipe)
1. test_events/test_data_loader.json - The Test Input
- What it is: Mock event data (simulates what Lambda receives)
- Why JSON? Lambda functions receive JSON input. This simulates API Gateway or direct invocations.
- Lambda handler receives this as the event parameter: lambda_handler(event, context):
1. lambda_test_data_loader.py - The Test Handler
- What it is: A special Lambda function just for testing
- Why separate from real handler?
- Real handler (lambda_function.py): Runs answer_query() for production
- Test handler (lambda_test_data_loader.py): Tests just the data loader in isolation
1. test_lambda_data_loader.sh - The Runner Script
- What it is: Bash script that automates the test. Instead of typing multiple commands: sam build, sam local invoke.. 

**The 95% test**
```
cd lambda_assets
sam build --use-container
sam local invoke DataLoaderTestFunction --event test_events/test_data_loader.json
```

```
cd lambda_assets
bash test_lambda_data_loader.sh
```

**The lighter test**
```
cd ModelPipeline/finrag_ml_tg1
& "venv_ml_rag\Scripts\Activate.ps1"

cd..
cd..
cd ModelPipeline/lambda_assets

sam build
ls .aws-sam/build/DataLoaderTestFunction/ -R | Select-String "data_cache"
```

### Cleanup:

```
cd ModelPipeline/lambda_assets
rm -r .aws-sam  
```


### Sam Testing Notes:
1. Method 1: sam build
```
sam build  # Builds on YOUR Windows machine
sam local invoke  # Runs in Docker, but with Windows-built package
```
3. DOES NOT test: Binary compatibility, Exact Lambda environment etc.
4. Method 2: sam build --use-container
```
sam build --use-container  # Builds INSIDE Amazon Linux Docker container
sam local invoke  # Runs in Docker with Linux-built package
```
5. Amazon Linux 2 specifics, Exact same build environment as AWS Lambda, Dependency resolution.
6. DOES NOT test: AWS infrastructure integration - Still local Docker, not real AWS
7. 95% accurate to real Lambda. 
8. Method 3: Deploy to Real AWS Lambda
```
sam deploy --guided  # Actually deploys to AWS
aws lambda invoke  # Invokes REAL Lambda function

~$0.0001 per invocation (Lambda compute)
~$0.0004 per S3 GET request
```

### Strat 1:

- For Lambda Execution: S3StreamingLoader loads heavy data from S3.
```python   
    # This reads from S3, not local cache
   df = pl.read_parquet("s3://bucket/ML_EMBED_ASSETS/...", ...)
```

- Dimension tables are available locally:
```python   
   # Small files, can be in package
   companies = pl.read_parquet(model_root / "data_cache/dimensions/finrag_dim_companies_21.parquet")
```

- Gold test sets available for validation:
```python   
    # Useful for testing in Lambda
   gold_sets = load_json("data_cache/qa_manual_exports/goldp3_analysis/...")
```

- Folder structure preserved for path references:
```python
    # This path exists, even if empty
   cache_dir = model_root / "data_cache" / "embeddings" / "cohere_1024d"
   cache_dir.exists()  # True (directory exists)
```

- Code → SAM packages code + deps → Uploads to AWS → Runs in Lambda container
- Lambda cares about: What dependencies are in the deployment package, That those dependencies work on Amazon Linux.
- two venvs (venv_ml_rag, venv_frontend) are irrelevant to Lambda.
- "listed in requirements.txt that SAM reads"


### Better test understanding.
1. This is NOT a hack - it's proper cross-platform code:
- Production Lambda:
- python os.getenv('AWS_LAMBDA_FUNCTION_NAME') = 'finrag-query'
- → Uses Path('/tmp/finrag_cache')  # Real Lambda /tmp
2. Local Testing:
- python os.getenv('AWS_LAMBDA_FUNCTION_NAME') = None
- → Uses Path(tempfile.gettempdir()) / 'finrag_cache'  # Windows temp
3. SAM Local Testing:
- python os.getenv('AWS_LAMBDA_FUNCTION_NAME') = 'test-function'
- → Uses Path('/tmp/finrag_cache')  # Docker container /tmp
