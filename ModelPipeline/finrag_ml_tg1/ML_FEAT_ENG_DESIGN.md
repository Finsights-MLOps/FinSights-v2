## Architectural case studies: 
### Why ML Feature Engineering ≠ (or not within subset of) Data Engineering

**Proofs**: This doc is meant for layers of proof from industry standards, production systems, and framework architectures. The main proof that this document wants to simply provide is that the design behind why ML feature engineering or ML feature infrastructure is separate from data pipelines is explained clearly. A case study with TensorFlow or any other separation patterns are learned and displayed here. Some of the core reasons are that feature definitions, feature specific creations, image or text embeddings, feature computations alongside the models and checkpoints are something that are heavily coupled with model pipeline and model logic. It doesn't make sense to carry over that ML heavy resource into a data engineering layer or a data ingestion layer.

- TensorFlow explicitly separates data engineering from ML feature engineering.
- tf.keras.layers.TextVectorization doesn't just "process text" - it learns vocabulary from the training data and saves it with the model. preprocessing logic is embedded in the model artifact.
- Similarity with mine: model-specific features (1024d Cohere vectors) that the RAG pipeline depends on.
- Data engineering / input pipelines live in tf.data + TFX ExampleGen/data-validation world.
- Feature engineering / embeddings live either as Keras preprocessing layers or TF Transform graphs.
- A feature store (Uber Michelangelo, Feast, Vertex AI Feature Store) is explicitly described as an ML-specific data system that runs feature pipelines, stores feature values, and serves them consistently for training and online inference.
- Link: `https://feast.dev/blog/what-is-a-feature-store/`
- `https://www.tensorflow.org/tfx/guide/transform`
  - "The Transform TFX pipeline component performs feature engineering on tf.Examples emitted from an ExampleGen component… and emits both a SavedModel as well as statistics on pre- and post-transform data."
  - we ingest data with ExampleGen; Transform is explicitly “the feature engineering place” --> emits a SavedModel.
- `https://cloud.google.com/blog/topics/developers-practitioners/model-training-cicd-system-part-i`
- From the link: "..the most important files for understanding TFX pipeline are listed"
  - features and feature_preprocessing exist in model/ not in data/
- Any model dependant features: embedding model, tokenizer, or configuration, must be versioned and owned together with the ML pipeline, not with generic ETL/table schemas.

## TensorFlow Architectural Patterns
```
┌─────────────────────────────────────────────────────────┐
│  DATA ENGINEERING LAYER (External to TF)               │
├─────────────────────────────────────────────────────────┤
│  • Apache Spark ETL (cleaning, joining, aggregating)   │
│  • SQL warehouses (BigQuery, Redshift)                 │
│  • Data validation (Great Expectations, dbt)           │
│  • Schema evolution, data quality checks               │
│                                                         │
│  Output: Clean tabular data (Parquet, CSV, JDBC)       │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  ML FEATURE ENGINEERING (Inside TF Model Directory)    │
├─────────────────────────────────────────────────────────┤
│  tensorflow/python/data/                               │
│  ├── tf.data.Dataset (batching, shuffling, prefetch)  │
│  ├── Feature columns (embedding, bucketization)       │
│  ├── Preprocessing layers (normalization, hashing)    │
│                                                         │
│  tensorflow/python/keras/layers/                       │
│  ├── TextVectorization (tokenization → integers)      │
│  ├── Embedding (integers → dense vectors)             │
│  ├── StringLookup, IntegerLookup (vocabulary)         │
│                                                         │
│  tensorflow_transform/ (TFT)                           │
│  ├── analyzers.vocabulary() - builds vocab from data  │
│  ├── tft.compute_and_apply_vocabulary()               │
│  ├── Analyzers run ONCE during training               │
│                                                         │
│  Output: Model-ready tensors, saved with model         │
└─────────────────────────────────────────────────────────┘
```

---

## Google's Vertex AI Pipeline Structure
```
vertex_ai_project/
├── data_ingestion/           # Data engineering (separate repo)
│   └── bigquery_etl/
│       └── raw_logs_to_tables.sql
│
├── model_pipeline/           # ML feature engineering (model repo)
│   ├── feature_engineering/
│   │   ├── text_embeddings.py      # ← Computes BERT embeddings
│   │   └── feature_store_sync.py   # ← Syncs to online serving
│   ├── training/
│   │   └── train_model.py
│   └── serving/
│       └── prediction_service.py   # ← Uses feature store at inference
```

### Why They Separate:
- **Data Engineering changes rarely**: SQL logic for extracting user events is stable
- **Feature Engineering changes frequently**: Embedding models, aggregation windows evolve with model experiments


### **Netflix Metaflow Pattern**

- Netflix open-sourced Metaflow specifically to solve this problem. 
```
┌─────────────────────────────────────────────────────────┐
│  DATA WAREHOUSE (Separate Infrastructure)              │
├─────────────────────────────────────────────────────────┤
│  • Hive/Presto: User viewing logs → clean tables       │
│  • Managed by Data Platform team                       │
│  • SLA: 99.9% availability, schema validation          │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  FEATURE ENGINEERING (Metaflow DAGs in Model Repo)     │
├─────────────────────────────────────────────────────────┤
│  metaflow_project/                                     │
│  ├── feature_flow.py                                   │
│  │   ├── @step: load_raw_data (from warehouse)        │
│  │   ├── @step: compute_embeddings (TensorFlow)       │
│  │   ├── @step: build_feature_index (FAISS)           │
│  │   └── @step: validate_features (QA checks)         │
│  │                                                     │
│  ├── training_flow.py                                 │
│  │   └── @step: train_model (uses feature artifacts)  │
│  │                                                     │
│  └── artifacts/                                        │
│      ├── embeddings.npy      # ← Model-specific       │
│      ├── faiss_index.bin     # ← Model-specific       │
│      └── feature_metadata    # ← Model-specific       │
└─────────────────────────────────────────────────────────┘
```


### **Uber's Feature Store Architecture**

- From Uber's engineering blog on Michelangelo:
```
┌─────────────────────────────────────────────────────────┐
│  DATA LAKE (Managed by Data Platform)                  │
├─────────────────────────────────────────────────────────┤
│  • Kafka → Hadoop: Trip events, driver locations       │
│  • Spark jobs: Aggregations, deduplication             │
│  • Output: Parquet files in HDFS                       │
│                                                         │
│  Ownership: Data Engineering team                      │
│  Consumers: All teams (analytics, reporting, ML)       │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  FEATURE STORE (Owned by ML Platform)                  │
├─────────────────────────────────────────────────────────┤
│  Michelangelo Feature Store:                           │
│  ├── Feature definitions (Python DSL)                  │
│  ├── Batch feature computation (Spark jobs)            │
│  ├── Online feature serving (Cassandra)                │
│  └── Feature lineage (tied to model versions)          │
│                                                         │
│  Example: "driver_last_30_trips_embedding"             │
│  ├── Computed by ML team's feature pipeline            │
│  ├── Uses trip data from data lake (input)             │
│  ├── Applies neural network (Word2Vec on trip seqs)    │
│  ├── Stored in feature store (online/offline)          │
│  └── Versioned with model training runs                │
└─────────────────────────────────────────────────────────┘
```

- "Feature extraction code is one of the most expensive components to maintain. It sits at the boundary between data pipelines and model code, and must be versioned with the model to ensure training-serving consistency."




### Author:
- Joel Markapudi. ( markapudi.j@northeastern.edu, mjsushanth@gmail.com )