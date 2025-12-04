## KPI Metrics - CI/CD Update - Working Successfully



# What is done: 

We are building and running Docker Image and container respectively on Github Actions ~ replacing the purpose of airflow



# Why: 

Any time we change anything inside the company metrics extraction pipeline, we will get the desired metrics parquet downloaded into S3 by just changing the Github repo source code( DataPipeline -> src_metrics/* ).



# Workflow file name: src_metrics - 
Link: https://github.com/Finsights-MLOps/FinSights/blob/main/.github/workflows/src_metrics.yml