### Model Optimization:
- For model optimization, we have refactored a lot of the ML retrieval 'spine' work, where prior data engineering heavy-lifting was not 'lazy'. We utilize lazy loading, scan and sink patterns, meta-only compute patterns, accurate memory math (Temporary arrays for joins, hash map - calculations.)
- For example, we calculated the exact memory spikes for 300MB, 750MB and 1.7GB dataset versions of compressed vectors; and tracked them across operations of `meta_df` and `vectors_df`.
- We refactor all **eagerly materialize** codes on big tables, replacing it with `scan_parquet` and `sink_parquet` patterns, or `collect` only when necessary.
- In some of our infra coding, we use `lf.fetch(N)` pattern, or `collect(streaming=True)`. 

---

**Rough memory math**: We noticed the following and calculated things, explicitly. 
- Embeddings table: 407,048 rows × 1024 floats × 4 bytes ≈ 1.67 GB of raw vector data.

**On top of that**
- Polars overhead (validity bitmaps, offsets, column metadata).
- Other columns (sentenceID, embedding_id, etc.).
- A second full copy of those vectors in the joined DataFrame.
- A Python list of hashes in hashes = [...] in current code.
- It explodes massively: exactly what the error code 3221225477 usually means: a native fault or OOM in a C/Rust library (here: Polars’ join or allocation).
- key pattern: eagerly materialize both big tables in memory, or join materializes a third DataFrame with the large list column, len() or list on full large GB tables, etc.
  
**Soon realized**:
- Lazy scan_parquet alone doesn’t help if, at the end, we still call .collect() on a query that conceptually is “meta + full vector list joined and held in RAM”.


**For polars, this is a good strategy to check**:
```python
plan = (
    pl.scan_parquet(meta_path)
      .join(pl.scan_parquet(emb_path), on="sentenceID", how="inner")
      .select(...)
)

print(plan.explain())    # or plan.describe_optimized_plan()
```

**What we now do instead:**
- Stream meta and embeddings from disk, Join on sentenceID, Add a hash and sentence_pos column.. etc. Continue respective operations. ( This is with respect to one of the logics in `s3vectors_table_preparation.py` )

