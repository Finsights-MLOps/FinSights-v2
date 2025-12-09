## AWS Log Monitoring & Analytics on SageMaker Studio

### Analytics, Logs & Drift Monitoring:
- We monitor logs, usage patterns, and cost trends using AWS SageMaker Studio notebooks (1).
- The system monitors: **Query & Token Analysis Plots + Tables, Overall query history, Model Usage Distribution Analysis, Efficiency, Volume/Reliability, LLM-Cost Analytics** in our plots and tables.
- At a raw level, we track 
  - input, output, total tokens, 
  - cost, context length, processing times in ms, 
  - errors and types, 
  - the complete JSON bodies: **context-assembled** files for queries, and full **response** body from LLMs.
- We have these fundamental **logging results all streamed directly to S3 buckets** and saved, as the centralized location. It has **logs/, contexts/, responses/** folders. 
- Not only are the logs stored as compressed parquet files, but the JSON bodies have rich metadata, and full answers.
- Apart from that, we have **another logging adoption** (2) readily available and integrated **MLFlow** work, which tracks experiments. Developers who use the backend code can easily access the main, and example commands on how to set arguments and log experiments are provided.
- Based on the ease, and cost feasibility/concerns of Managed MLFlow on AWS, both approaches can be adopted.


----

### Done on: AWS SageMaker Studio. 

- Due to the natural restrictions of organization accounts or domain accounts, since we are working on clear organization accounts and with respective keys, the link will only work for certain domain ID and collect ID where the developers belong to this particular domain or project that we are working on.
- For security reasons, this current SageMaker Project access is not public. AWS ensures that, it requires a proper AWS IAM role and assigned group roles for user, then a custom policy role and inline policy changes. Sagemaker especially enforces principal membership where, even the IAM users have to switch role first and only then they can access the SageMaker domain and project.
- Alongside the exact Sagemaker link, we provide the **exact notebook** file in the repo for reference.

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

### Retraining Concerns:
- As we dont have a training-loop or trained checkpoint, and this is a system of orchestrated algorithms between cloud services, assembling, vector querying, data engineering, and LLM inference serving  - Retraining is not applicable here. 
- We've achieved many other counterparts of 'retraining' and 'parameter tuning' instead: 
  - Automated gold tests, Multi Layered Evaluation Suite at Gold P1, P2 phases, and then a Business scale Gold P3 phase with proper Bluert, BERTScore, ROUGE, and custom metrics.
  - Versioned prompt templates, a config-based modular pipeline which can swap out S3 paths, base data files' paths, serving LLMs from bedrock, retrain embedding models, and re-generate S3 Vectors as needed.



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
