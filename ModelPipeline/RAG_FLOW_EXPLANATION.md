# RAG System Flow: What Gets Sent to the LLM?

## Quick Answer

**YES** - The LLM receives **only the retrieved hit-related sentences**, NOT the entire S3 vector embeddings database.

The S3 vector embeddings are used **only for retrieval/search** to find relevant sentences. Only the top-K most relevant sentences (after expansion and deduplication) are sent to the LLM as context.

---

## Detailed Flow

### Step 1: Query Processing & Variant Generation

```
User Query: "What was NVIDIA's revenue in 2023?"
    ↓
Entity Extraction (companies, years, sections)
    ↓
Query Embedding (1024-d vector)
    ↓
[Optional] Variant Generation (semantic rephrasings)
    - Variant 1: "NVIDIA's 2023 revenue"
    - Variant 2: "How much revenue did NVIDIA generate in 2023?"
    - Variant 3: "NVIDIA 2023 financial revenue"
```

### Step 2: S3 Vector Retrieval

The retriever searches the S3 Vectors database (which contains embeddings for **all sentences** in the SEC filings):

```
Base Query → S3 Vectors Search
    ├─ Filtered Search (strict filters: company, year, section)
    │   └─ Returns: ~30 top-K hits (sentence-level)
    │
    └─ Global Search (relaxed filters: company, year >= threshold)
        └─ Returns: ~20 top-K hits (sentence-level)

[If variants enabled]
Variant 1 → Filtered Search → ~15 hits
Variant 2 → Filtered Search → ~15 hits
Variant 3 → Filtered Search → ~15 hits
```

**Key Point**: The S3 Vectors database contains embeddings for **millions of sentences**, but we only retrieve the **top-K most similar** (typically 30-50 hits total after deduplication).

### Step 3: Deduplication

```
Raw Hits: ~80-100 hits (from base + variants)
    ↓
Deduplicate by (sentence_id, embedding_id)
    ↓
Union Hits: ~30-50 unique sentence hits
```

### Step 4: Sentence Expansion (Window Context)

Each retrieved hit is expanded to include neighboring sentences for context:

```
30 S3Hits (core hits)
    ↓
Window Expansion (±3 sentences around each hit)
    ↓
~210 SentenceRecords (with overlapping windows)
    ↓
Deduplicate by sentenceID (keep best evidence)
    ↓
~140 unique SentenceRecords
```

**Example**:
- Core hit: Sentence #45 about "revenue growth"
- Expanded: Sentences #42, #43, #44, #45, #46, #47, #48
- This provides context around the hit

### Step 5: Context Assembly

The unique sentences are sorted and formatted:

```
~140 unique SentenceRecords
    ↓
Sort by: (company, year ASC, section, doc, position)
    ↓
Format with headers:
    === [NVDA] NVIDIA CORP | FY 2023 | Doc: nvda_2023_10k | Item 7: MD&A ===
    
    Sentence text 1
    
    Sentence text 2
    
    ...
```

### Step 6: Final Context Sent to LLM

The assembled context string is combined with KPI data and sent to the LLM:

```
┌─────────────────────────────────────────────────┐
│ KPI SNAPSHOT                                    │
│ (Structured financial metrics)                  │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│ NARRATIVE CONTEXT - SEC FILINGS                │
│                                                 │
│ === [NVDA] NVIDIA CORP | FY 2023 | ... ===     │
│                                                 │
│ [~140 sentences from retrieved hits]          │
│                                                 │
│ === [MSFT] MICROSOFT CORP | FY 2023 | ... ===  │
│                                                 │
│ [More sentences...]                             │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│ USER QUESTION                                   │
│                                                 │
│ What was NVIDIA's revenue in 2023?             │
└─────────────────────────────────────────────────┘
```

---

## Key Points

### ✅ What the LLM Receives

1. **Only retrieved sentences** (~140 sentences after expansion)
2. **Sorted and formatted** with provenance headers
3. **Combined with KPI data** (structured metrics)
4. **User's original question** at the end

### ❌ What the LLM Does NOT Receive

1. **NOT the entire S3 vector database** (millions of sentences)
2. **NOT all embeddings** (only the top-K most similar)
3. **NOT raw vector data** (only the sentence text)

### 🔍 How Retrieval Works

- **S3 Vectors** = Search engine (like Google)
- **Embeddings** = Index for semantic search
- **Retrieval** = Find top-K most similar sentences
- **Expansion** = Add neighboring sentences for context
- **Assembly** = Format for LLM consumption

---

## Example: Query Flow

**Query**: "What were NVIDIA's revenue trends from 2018 to 2020?"

1. **Retrieval**: Searches S3 Vectors → finds ~40 sentence hits about NVIDIA revenue 2018-2020
2. **Expansion**: Each hit expands to ±3 neighbors → ~280 sentences
3. **Deduplication**: Overlapping windows merged → ~150 unique sentences
4. **Assembly**: Sorted by year, formatted with headers
5. **LLM Context**: ~150 sentences sent to LLM (NOT millions)

**Result**: LLM generates answer based on these ~150 relevant sentences, not the entire knowledge base.

---

## Configuration Parameters

You can control what gets retrieved:

- `top_k_filtered`: Max hits from filtered search (default: ~30)
- `top_k_global`: Max hits from global search (default: ~20)
- `top_k_filtered_variants`: Max hits per variant (default: ~15)
- `window_size`: Sentences around each hit (default: ±3)
- `enable_variants`: Use query variants (default: false)
- `enable_global`: Use global search (default: true)

These parameters control the **size of context** sent to the LLM, not the size of the searchable database.

---

## Summary

- **S3 Vector Embeddings**: Used for **retrieval/search** only
- **LLM Context**: Contains **only the top-K retrieved sentences** (after expansion)
- **Not included**: The entire knowledge base or all embeddings
- **Result**: Efficient, focused context that's relevant to the query

The RAG system is like a librarian: it searches the entire library (S3 vectors) but only brings you the most relevant books (retrieved sentences) to answer your question.

