# V3 Decoupled Modular RAG Architecture

Welcome to the **V3 Hierarchical Summary RAG** implementation folder. This capability stage moves away from monolithic scripts toward a **decoupled, modular architecture**. This allows independent optimization of each RAG stage (routing, retrieving, generating) and prepares the codebase for advanced post-retrieval routing and reranking.

---

## 📂 Directory Structure

Here is how the pipeline stages are mapped to specific files:

```
v3_hierarchical_summary_rag/
├── README.md                  # This documentation file
├── chat_V3.py                 # Primary entry point (bootstraps paths & runs chat)
├── main_chat.py               # Orchestration controller (coordinates RAG pipeline flow)
├── migration.py               # DB Schema Migration script (adds titles & global summaries)
│
├── config/
│   └── config.py              # Configuration manager (DB connections & Google GenAI Client)
│
├── routing/
│   └── router.py              # Pre-Retrieval stage (Intent routing & Query decomposition)
│
├── retrieval/
│   └── retriever.py           # Retrieval stage (768-D vector embedding & RRF Hybrid SQL search)
│
└── generation/
    └── generator.py           # Post-Retrieval stage (System prompts & citation generation)
```

---

## 🧭 Where to Start?

To trace how a user's question flows through the system, read the files in this order:

1.  **[chat_V3.py](file:///home/bhanu/prod-rag-pipeline/v3_hierarchical_summary_rag/chat_V3.py)**: The script you execute. It adds the local directories to the python path (`sys.path`) and calls the main loop.
2.  **[main_chat.py](file:///home/bhanu/prod-rag-pipeline/v3_hierarchical_summary_rag/main_chat.py)**: The main orchestrator. You'll see the 4 core stages of a production RAG:
    *   *Stage 1 (Routing)*: Checks intent (e.g. comparison queries vs chitchat).
    *   *Stage 2 (Interception)*: Bypasses the database if the query is just pleasantries ("chitchat").
    *   *Stage 3 (Data Gathering)*: Decomposes comparison queries into separate sub-queries and searches them in parallel.
    *   *Stage 4 (Synthesis)*: Fuses context blocks together and requests a response.
3.  **[routing/router.py](file:///home/bhanu/prod-rag-pipeline/v3_hierarchical_summary_rag/routing/router.py)**: The intent router. Uses `gemini-3.1-flash-lite` to classify queries and extract target entities/metadata.
4.  **[retrieval/retriever.py](file:///home/bhanu/prod-rag-pipeline/v3_hierarchical_summary_rag/retrieval/retriever.py)**: The retrieval engine. It queries PostgreSQL using a Reciprocal Rank Fusion (RRF) algorithm to combine semantic vector matches and keyword indexes.
5.  **[generation/generator.py](file:///home/bhanu/prod-rag-pipeline/v3_hierarchical_summary_rag/generation/generator.py)**: The response generator. It sets up strict negative instructions and prompts `gemini-3-flash` to generate the grounded response with citations.
6.  **[config/config.py](file:///home/bhanu/prod-rag-pipeline/v3_hierarchical_summary_rag/config/config.py)**: Contains connection configurations and environment initialization.

---

## 🚀 Running the Modules

### 1. Database Migration
If you need to migrate your database tables to include the updated schema fields (`clean_title` and `global_summary`), run the migration script:
```bash
./venv/bin/python v3_hierarchical_summary_rag/migration.py
```

### 2. Launching the Interactive Chat
To start talking to the RAG co-pilot:
```bash
./venv/bin/python v3_hierarchical_summary_rag/chat_V3.py
```
