# 📘 Research Paper: Architectural Evolution of an Enterprise Healthcare RAG Pipeline

## Abstract
This paper presents the architectural evolution of an Enterprise Healthcare Retrieval-Augmented Generation (RAG) system, tracking its development across three iterative phases. The primary objective was to build a highly accurate, modular system capable of ingesting, vectorizing, and querying 12 complex corporate healthcare insurance policies without semantic interference. We document the progression from a naive stateless architecture (Version 1) to a stateful hybrid model (Version 2), culminating in a sophisticated, intent-routed hierarchical system utilizing Vision-Driven Contextual Prepending (Version 3). By detailing the limitations encountered at each phase and the corresponding engineering solutions, this paper serves as a comprehensive technical guide for designing contaminant-free vector retrieval systems.

---

## 1. Introduction
Retrieval-Augmented Generation (RAG) fundamentally relies on the quality and contextual integrity of its underlying vector store. In the domain of enterprise healthcare insurance, policies contain highly overlapping terminology (e.g., "deductibles", "maternity limits", "co-pays") belonging to distinctly competing entities (e.g., HDFC ERGO vs. SBI General Insurance). Injecting these policies into a standard vector namespace inherently risks "context contamination," wherein a Large Language Model (LLM) synthesizes a hallucinated response using mixed constraints from multiple competitors.

To solve this, the pipeline was engineered through three distinct iterations, evolving the data ingestion strategy, the database schema, and the conversational retrieval logic.

---

## 2. Version 1: The Basic Stateless RAG Architecture

### 2.1 Ingestion Strategy
The initial phase (`v1_basic_stateless_rag/ingest_policies.py`) relied on the `unstructured` library for document partitioning. The database architecture utilized PostgreSQL with `pgvector`, structured as a simple parent-child relationship:
- **`production_documents` (Parent):** Stored merely the `file_name` and the `total_blocks`.
- **`document_elements` (Child):** Stored the raw `content` payload and the 768-D `embedding` generated via `gemini-embedding-2`.

### 2.2 Retrieval Engine
The retrieval system (`chat_V1.py`) implemented a Hybrid Search, executing both Semantic Vector search (using the `<=>` Cosine Distance operator) and Full-Text Keyword Search (using PostgreSQL's `to_tsvector` and `plainto_tsquery`). The rankings from both methods were merged using Reciprocal Rank Fusion (RRF) via Common Table Expressions (CTEs).

### 2.3 Core Deficiencies Identified
1. **Amnesia and Semantic Interference:** The `unstructured` text splitting algorithm stripped away macro-document identity. A child chunk containing the text *"The maternity limit is 50,000 INR"* lacked attribution to its parent policy (e.g., HDFC Optima). When a user queried *"What is the HDFC maternity limit?"*, the vector search blindly returned limits from multiple companies, polluting the LLM context.
2. **Statelessness:** The chatbot interface operated without memory. If a user asked *"What is the limit?"* following a previous question about *"Bajaj Allianz"*, the search executed blindly on the word "limit", failing entirely to retrieve company-specific data.

---

## 3. Version 2: Stateful Hybrid RRF RAG

### 3.1 Ingestion Enhancements
To combat the loss of spatial awareness, Version 2 (`v2_stateful_hybrid_rrf_rag/fast_vision_based_ingest.py`) transitioned to a Vision-based extraction pipeline. Instead of relying on blind NLP text splitters, the system rasterized entire PDF pages (at 150 DPI) and passed them directly to `gemini-2.5-flash`. The model extracted text and formatted complex tables natively into Markdown. 

Crucially, a rudimentary positional marker was injected into the content string:
```text
[PAGE 4 START]
Extracted content...
[PAGE 4 END]
```

### 3.2 Retrieval Enhancements
To resolve the statelessness defect, `chat_V2.py` introduced **Conversational Query Condensation**. A global `CHAT_HISTORY` array tracked recent interactions. Before querying the PostgreSQL database, the user's raw input was intercepted and rewritten by `gemini-2.5-flash` to include missing context (e.g., rewriting *"What is the limit?"* to *"What is the room rent limit for Bajaj Allianz?"*). The Hybrid RRF search then executed using this optimized, standalone query.

### 3.3 Core Deficiencies Identified
While Version 2 successfully achieved conversational statefulness and preserved table layouts via Markdown, the underlying vector embeddings still lacked deep macro-context. The `[PAGE X START]` marker provided spatial coordinates, but it did not resolve the semantic interference issue. To the `gemini-embedding-2` model, the structural vectors of HDFC rules and SBI rules remained indistinguishable without the company name explicitly bound to the textual chunk.

---

## 4. Version 3: Hierarchical Summary RAG (The Production Architecture)

Version 3 represents the finalized, production-grade architecture. It completely overhauls the chunking philosophy and introduces modular control flows for retrieval.

### 4.1 Ingestion Strategy: Vision-Driven Contextual Prepending (VDCP)
The final ingestion engine (`v3_hierarchical_summary_rag/Contextual_Pretending_Chunking/super_ingest_failsafe.py`) abandons blind text splitting in favor of Contextual Prepending.

1. **Macro-Extraction:** The engine isolates the PDF Cover Page and prompts the Vision model to infer the `clean_title` and synthesize a 1-sentence `global_summary`.
2. **Micro-Extraction:** For every subsequent page, the Vision model extracts the narrative text, Markdown tables, and dynamically generates a localized 1-sentence summary of the page's exact contents.
3. **The Stitch (Deterministic Prepending):** Before vectorization, the spatial identity is mathematically bound to the semantic content by prepending the macro and micro context directly onto the raw text:
   > `[CONTEXT: (Page X) This page details annual aggregate deductible rules for HDFC ERGO Optima Secure.] [RAW TEXT: The aggregate deductible shall apply...]`

This forces the resulting 768-dimensional vector to gravitationally pull toward the specific company's semantic cluster, eliminating cross-document contamination. 

*Engineering Note:* To prevent API network timeouts caused by transferring massive Base64 images, the pipeline was optimized to pass `PIL.Image` objects directly into the `google.genai` SDK's synchronous `contents` payload (`contents=[page_image, prompt]`), allowing the SDK to natively manage the binary transfer in-memory and bypass WSL IPv6 async freezing.

### 4.2 Advanced Modular Retrieval
The final chatbot architecture (`main_chat.py`) breaks away from the monolithic execution of V1/V2, implementing a sophisticated multi-stage router.

- **Stage 1: Intent & Metadata Routing:** The `analyze_and_route_query` module intercepts the user's input. It classifies the intent (e.g., "single_search", "multi_compare", "chitchat") and isolates the target company entities required for the SQL `WHERE document_id = X` filter.
- **Stage 2: Control Layer Interception:** If the intent is classified as "chitchat" (e.g., "Hello"), the query bypasses the PostgreSQL database entirely and generates a rapid response, saving expensive vector compute.
- **Stage 3: Query Decomposition & Isolated Execution:** Complex questions are broken down into targeted sub-queries. The Hybrid Search executes purely on the isolated sub-graphs defined by the metadata router. This prevents "chunk starvation," where a broad query fails to retrieve granular clauses because they were pushed out of the top-K limit.
- **Stage 4: Response Synthesis:** The `generate_response` module fuses the balanced, deduplicated context blocks to generate the final, grounded AI response.

---

## 5. Conclusion
The evolution from Version 1 to Version 3 demonstrates that standard NLP chunking algorithms and flat vector namespaces are insufficient for enterprise RAG applications containing highly overlapping, contradictory policies. By shifting to Vision-Driven Contextual Prepending (VDCP) and enforcing strict Relational Metadata Isolation via PostgreSQL, the pipeline successfully guarantees zero-contamination retrieval. Coupled with the stateful query condensation and intent-routing modules, the resulting architecture acts as a highly resilient, enterprise-grade Healthcare Co-Pilot.
