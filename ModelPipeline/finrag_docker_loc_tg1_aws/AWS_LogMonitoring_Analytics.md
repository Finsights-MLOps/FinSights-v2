## AWS Log Monitoring & Analytics on SageMaker Studio
### Done on: AWS SageMaker Studio. 

- Due to the natural restrictions of organization accounts or domain accounts, since we are working on clear organization accounts and with respective keys, the link will only work for certain domain ID and collect ID where the developers belong to this particular domain or project that we are working on.
- For obvious security reasons, the bucket access or SageMaker Notebook Project ID access are not public. 

**Link:** https://dzd-5znn554opul4h3.sagemaker.us-east-1.on.aws/projects/b8r202203aqcvb/notebooks/FinSights_Log_Analytics.ipynb

**Features:** 
- 4 executive visualizations (cost trends, usage patterns)
- Detailed query analytics tables

**Execution Schedule**:
- (in progress) enabling and working on a Lambda biweekly; schedule using EventBridge Cron triggering Lambda.
- Lambda calls SageMaker API to execute notebook, Result gets saved to S3 or emailed.

**What we monitor**:
- The system monitors: **Query & Token Analysis Plots + Tables, Overall query history, Model Usage Distribution Analysis, Efficiency, Volume/Reliability, LLM-Cost Analytics** in our plots and tables.
- At a raw level, we track input, output, total tokens, cost, context length, processing times in ms, errors and types, and the JSON bodies: **context-assembled** files for queries, and the complete **response** body from LLMs.
- We have them all streamed directly to S3 buckets as the centralized location. It has logs/, contexts/, responses/ folders.


**More Details**:
PROJECT: FinSights_Log_Analytics
═══════════════════════════════════════════════════════════════

- IDENTIFIERS:
  - Domain ID:        dzd-5znn554opul4h3
  - Project ID:       b8r202203aqcvb
  - Notebook ID:      aq6jx4r7wql7on
  - AWS Region:       us-east-1
  - Account ID:       729472661729

- URLS:
  - Domain:           https://dzd-5znn554opul4h3.sagemaker.us-east-1.on.aws/
  - Project:          https://dzd-5znn554opul4h3.sagemaker.us-east-1.on.aws/projects/b8r202203aqcvb
  - Notebook:         https://dzd-5znn554opul4h3.sagemaker.us-east-1.on.aws/projects/b8r202203aqcvb/notebooks/notebook/aq6jx4r7wql7on

- STORAGE:
  - S3 Bucket:        sentence-data-ingestion
  - Project Storage:  s3://sentence-data-ingestion/ML_NOTEBOOK_ASSETS/
  - Log Data:         s3://sentence-data-ingestion/DATA_MERGE_ASSETS/LOGS/FINRAG/logs/query_logs.parquet

- IAM:
  - Execution Role:   AmazonSageMakerUserIAMExecutionRole

- COMPUTE:
  - Instance Type:    sc.t3.medium (2 vCPU, 4GB RAM)
  - Idle Timeout:     60 minutes
  - Cost:             ~$0.05/hour when running
