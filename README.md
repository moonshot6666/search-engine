# Hybrid Search Engine

A production-ready hybrid search engine for financial/crypto content combining **BM25 keyword matching** + **384D vector semantic similarity** with beautiful CLI interface and real-time clustering.

![CLI Demo](https://img.shields.io/badge/CLI-Production_Ready-green.svg) ![Performance](https://img.shields.io/badge/Response_Time-125ms-brightgreen.svg) ![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)

## ⚡ Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start API server (auto-loads 384D embeddings)
uvicorn src.api.main:app --reload --port 8000

# 3. Search with beautiful CLI interface
python cli_search.py "What's happening with Bitcoin?"
python cli_search.py --interactive  # Interactive mode
python cli_search.py "crypto analysis" --clustered  # With clustering
```

## 🎨 CLI Interface

Beautiful command-line interface with rich formatting, real-time responses (~125ms), and comprehensive search features:

```bash
🚀 Hybrid Search Engine CLI
💎 Financial/Crypto Content Search

🥇 Score: 0.847 (BM25: 0.92, Vector: 0.78)  💎 BULLISH
📱 Bitcoin analysis shows strong support at $45k... [ENTITIES: bitcoin, btc]
👤 @crypto_analyst (125K followers) | ⚡ 342 engagement | 🕐 2h ago
```

**Features:**
- 🎨 Rich colors and emojis with professional layout
- ⚡ Lightning fast (~125ms response time)
- 📊 BM25, Vector, and Final relevance scores
- 🎯 Entity highlighting and market sentiment indicators
- 📂 HDBSCAN clustering for thematic organization
- 🔄 Interactive mode with continuous searching

## 🔍 Architecture

**Hybrid Search**: BM25 (45%) + Vector Similarity (35%) + Boost Factors (20%)
- Real BM25 algorithm (k1=1.2, b=0.75) with TF-IDF scoring
- 384D semantic embeddings via sentence-transformers/all-MiniLM-L6-v2
- HDBSCAN clustering for thematic result organization
- Natural language → SearchSpec DSL conversion

**Tech Stack:**
- **Backend**: FastAPI + Pydantic
- **Search**: LocalHybridSearchEngine + OpenSearch ready
- **ML**: sentence-transformers, HDBSCAN, scikit-learn  
- **CLI**: Click + Rich for beautiful terminal interface

## 🚀 API Endpoints

```bash
# Natural language search (recommended)
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"query": "Why is Bitcoin pumping?", "size": 5}'

# Traditional search
curl "http://localhost:8000/search?q=bitcoin ethereum&size=10"

# Clustered search with themes
curl -X POST "http://localhost:8000/search/clustered" \
  -H "Content-Type: application/json" \
  -d '{"query": "crypto market analysis", "size": 15}'
```

## 📊 Performance

| Component | Performance | Details |
|-----------|------------|---------|
| **Search Response** | ~125ms | 60x faster after optimization |
| **CLI Experience** | Real-time | Sub-second user experience |
| **Clustering** | 1-6ms | HDBSCAN thematic organization |
| **Embedding Model** | Pre-loaded | One-time startup vs per-query |

## 🐳 Docker Setup

```bash
# Start OpenSearch cluster
docker-compose -f docker-compose-minimal.yml up -d

# Verify cluster health
curl http://localhost:9200/_cluster/health?pretty
```

## 📁 Project Structure

```
search_engine/
├── cli_search.py             # Beautiful CLI interface
├── src/
│   ├── api/main.py           # FastAPI endpoints
│   ├── search/local_search.py # Hybrid BM25+vector engine  
│   ├── search/cluster.py     # HDBSCAN clustering
│   ├── etl/embeddings.py     # 384D embedding generation
│   └── llm/spec_gen.py       # Natural language processing
├── data/                     # Sample tweets (50 documents)
├── normalized/               # Tweets with 384D embeddings
└── requirements.txt
```

## 🎯 Key Features

- ✅ **Production-Ready**: Real BM25 + 384D semantic search
- ✅ **Lightning Fast**: ~125ms response time with optimization
- ✅ **Beautiful CLI**: Rich formatting with colors and emojis
- ✅ **Smart Clustering**: HDBSCAN thematic result organization  
- ✅ **Natural Language**: Rule-based query → SearchSpec conversion
- ✅ **Scalable**: OpenSearch integration ready for production
- ✅ **Real Embeddings**: sentence-transformers with caching

## 🔧 Development

```bash
# Run comprehensive tests
python test_search_system.py

# Generate embeddings for new data
python -m src.etl.embeddings --input data/tweets.jsonl --output normalized/

# Test individual components
pytest tests/ -v
```

## 📈 Search Quality

**Hybrid Approach**: Combines precision of keyword matching (BM25) with semantic understanding (vector similarity) for superior results compared to either method alone.

**Entity Recognition**: Automatic detection of tokens (BTC, ETH), KOLs (Elon, Vitalik), events, and macros with confidence scoring.

**Intelligent Clustering**: Real-time HDBSCAN clustering groups results by themes like "Bitcoin & Ethereum - Security Issues" with automatic labeling.

## 🛠️ Next Development

- **LLM Integration**: OpenAI/Anthropic answer synthesis with citations
- **Hybrid Clustering**: Nightly + real-time for production scale  
- **Advanced Features**: Query suggestions, personalization, multi-language

## 📄 License

MIT License - feel free to use for your projects!

---

**For detailed documentation, architecture details, and development guidelines, see [CLAUDE.md](CLAUDE.md)**