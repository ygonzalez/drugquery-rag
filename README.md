# DrugQuery RAG System

Production-grade Retrieval-Augmented Generation (RAG) system for FDA drug label Q&A.

## 🎯 What This Demonstrates

- **RAG Architecture**: Hybrid search with Weaviate (semantic + keyword)
- **Production Evaluation**: Comprehensive metrics pipeline with RAGAS
- **Guardrails**: Hallucination detection and medical advice boundaries
- **Source Citations**: Every answer includes verifiable FDA sources
- **Observability**: Full tracing with LangSmith

## 🏗️ Project Status

| Phase | Status | Description |
|-------|--------|-------------|
| 0. Setup | ✅ | Project structure, dependencies |
| 1. Data Ingestion | ✅ | Download, parse, chunk FDA data |
| 2. Vector Store | 🔲 | Weaviate schema and indexing |
| 3. Retrieval | 🔲 | Query expansion, hybrid search, reranking |
| 4. Generation | 🔲 | RAG chain with citations, guardrails |
| 5. Evaluation | 🔲 | Test set, retrieval/generation metrics |
| 6. API/Frontend | 🔲 | FastAPI backend, Streamlit UI |
| 7. Documentation | 🔲 | README, blog post |

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager
- Docker (for local Weaviate)

### Setup

```bash
# Clone the repository
git clone https://github.com/ygonzalez/drugquery-rag.git
cd drugquery-rag

# Create virtual environment and install dependencies
uv sync

# Copy environment template
cp .env.example .env
# Edit .env with your API keys

# Start Weaviate
docker compose up -d

# Verify installation
uv run drugquery --version
```

### Development

```bash
# Run tests
uv run pytest

# Lint and format
uv run ruff check .
uv run ruff format .

# Type check
uv run mypy src

# Start API server (after Phase 6)
uv run drugquery serve --reload

# Run Streamlit frontend
uv run streamlit run streamlit_app/app.py
```

## 📁 Project Structure

```
drugquery-rag/
├── src/drugquery/          # Main package
│   ├── ingestion/          # Data download, parsing, chunking
│   ├── vectorstore/        # Weaviate operations
│   ├── retrieval/          # Query expansion, search, reranking
│   ├── generation/         # RAG chain, citations, guardrails
│   ├── evaluation/         # Test sets, metrics
│   └── api/                # FastAPI backend
├── data/                   # Data files (not committed)
├── notebooks/              # Exploration and analysis
├── scripts/                # Utility scripts
├── streamlit_app/          # Frontend
└── tests/                  # Test suite
```

## 📊 Evaluation Results

*Coming after Phase 5*

| Metric | Score |
|--------|-------|
| MRR | - |
| Recall@5 | - |
| Faithfulness | - |
| Answer Relevancy | - |

## 🔗 Links

- [Live Demo](https://drugquery.streamlit.app) *(coming soon)*
- [Blog Post: Beyond Basic RAG](https://ygonzalez.github.io/blog/beyond-basic-rag) *(coming soon)*
- [FDA DailyMed](https://dailymed.nlm.nih.gov/)

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## ⚠️ Disclaimer

This tool provides information from FDA drug labels for **educational purposes only**. 
It is NOT medical advice. Always consult a healthcare provider for medical decisions.
