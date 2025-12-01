import os
import sys
from datetime import datetime
import analytical_layer as al

def main():
    print("Starting GitHub Actions Metrics Pipeline...")
    
    # 1. Setup Environment
    base_dir = os.getenv("ANALYTICAL_LAYER_BASE_DIR", "metrics_output")
    os.makedirs(base_dir, exist_ok=True)
    
    polite_delay = float(os.getenv("EDGAR_POLITE_DELAY", "1.5"))
    
    # Ensure Identity is set
    edgar_identity = os.getenv("EDGAR_IDENTITY")
    if not edgar_identity:
        print("WARNING: EDGAR_IDENTITY environment variable is not set. Using default.")
        edgar_identity = "Default User <default@example.com>"
    
    al.set_identity(edgar_identity)

    # 2. Run Pipeline
    try:
        print(f"Running pipeline with Base Dir: {base_dir}")
        result = al.run_analytical_layer_pipeline(
            base_dir=base_dir,
            polite_delay=polite_delay,
            run_date=datetime.today()
        )
        print("Pipeline execution successful.")
    except Exception as e:
        print(f"Pipeline execution failed: {e}")
        sys.exit(1)

    # 3. Upload to S3
    # Check if AWS credentials are present (implicit check via boto3 or explicit env var check)
    if os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"):
        try:
            print("Uploading results to S3...")
            # Generate dummy IDs for GH Actions context
            dag_id = "gh_actions_metrics"
            task_id = "deploy_step"
            run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
            
            s3_info = al.upload_results_to_s3(
                final_parquet_path=result["final_parquet_path"],
                metadata_path=result["metadata_json_path"],
                coverage_csv_path=result["coverage_csv_path"],
                dag_id=dag_id,
                task_id=task_id,
                run_id=run_id,
            )
            print("S3 Upload successful:", s3_info)
        except Exception as e:
            print(f"S3 Upload failed: {e}")
            sys.exit(1)
    else:
        print(f"Skipping S3 upload: AWS_ACCESS_KEY_ID present: {bool(os.getenv('AWS_ACCESS_KEY_ID'))}")
        print(f"Skipping S3 upload: AWS_SECRET_ACCESS_KEY present: {bool(os.getenv('AWS_SECRET_ACCESS_KEY'))}")
        # END DEBUG
        print("Skipping S3 upload: AWS credentials not found.")
        
if __name__ == "__main__":
    main()
