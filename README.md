# Investment-Aware Hybrid Search Engine 💎

A production-ready **investment advisory search engine** for financial/crypto content combining **BM25 keyword matching** + **384D vector semantic similarity** + **intelligent investment guidance**. Features beautiful CLI interface, real-time clustering, and balanced investment perspectives.

**🚀 Key Innovation**: Eliminates confirmation bias in crypto investment research by providing balanced bullish + bearish perspectives for informed decision-making.

![CLI Demo](https://img.shields.io/badge/CLI-Investment_Advisory-brightgreen.svg) ![Performance](https://img.shields.io/badge/Response_Time-10_125ms-brightgreen.svg) ![Python](https://img.shields.io/badge/Python-3.10+-blue.svg) ![Investment](https://img.shields.io/badge/Investment-Balanced_Analysis-gold.svg)

## ⚡ Quick Start

```bash
# 1. Setup development environment
./scripts/setup_dev.sh

# 2. Start API server (auto-loads 384D embeddings)  
./scripts/start_server.sh

# 3. Search with investment-aware CLI interface
./scripts/search.sh "should I buy Bitcoin?"        # Investment advisory with balanced analysis
./scripts/search.sh "What's happening with Bitcoin?" # Traditional analysis  
./scripts/search.sh --interactive                  # Interactive investment guidance mode
./scripts/search.sh "crypto analysis" --clustered  # With thematic clustering
```

## 🎨 Investment-Aware CLI Interface

Beautiful command-line interface with rich formatting, real-time responses (~10-125ms), and **intelligent investment advisory capabilities**:

```bash
🚀 Hybrid Search Engine CLI
💎 Financial/Crypto Content Search

🥇 Score: 0.900 (BM25: 0.00, Vector: 1.00)  💎 BULLISH
📱 Magic Eden Solana NFT volume surpassing OpenSea [ENTITIES: Solana]
👤 @nft_metrics (31K followers) | ⚡ 445 engagement | 🕐 588d ago

🥈 Score: 0.882 (BM25: 0.00, Vector: 0.95)  🔻 BEARISH  
📱 Solana network congestion again. When will they fix scaling? [ENTITIES: Solana]
👤 @sol_critic (8.5K followers) | ⚡ 109 engagement | 🕐 591d ago
```

**Investment Advisory Features:**
- 💡 **Balanced Perspectives**: "should I buy SOL" returns BOTH bullish opportunities AND risk factors
- 🧠 **Entity-Aware**: Automatically filters by query subject (SOL queries → SOL content only)
- 🎯 **Investment Intent**: Detects buy/sell/timing queries and adjusts scoring accordingly
- 📊 **Educational Focus**: Prioritizes analytical content over speculation and hype

**Technical Features:**
- 🎨 Rich colors and emojis with professional layout
- ⚡ Lightning fast (~10-125ms response time)
- 📊 BM25, Vector, and Final relevance scores with precision
- 🎯 Entity highlighting and market sentiment indicators (💎 BULLISH, 🔻 BEARISH)
- 📂 HDBSCAN clustering for thematic organization
- 🔄 Interactive mode with continuous investment guidance

## 🔍 Investment-Aware Architecture

**Investment-Enhanced Hybrid Search**: BM25 (45%) + Vector Similarity (35%) + Investment Intelligence (20%)
- Real BM25 algorithm (k1=1.2, b=0.75) with TF-IDF scoring
- 384D semantic embeddings via sentence-transformers/all-MiniLM-L6-v2
- **Investment-aware entity-sentiment scoring** for balanced perspectives
- HDBSCAN clustering for thematic result organization
- Natural language → Investment intent detection → SearchSpec conversion

**Investment Intelligence:**
- **Entity Relevance**: +0.30 boost for matching entities, -0.50 penalty for off-topic tokens
- **Sentiment Balance**: Investment queries boost BOTH bullish AND bearish content
- **Educational Priority**: +0.25 boost for analysis, -0.20 penalty for speculation
- **Query Intent Detection**: Buy/Sell/Timing/Analysis queries get tailored scoring

**Tech Stack:**
- **Backend**: FastAPI + Pydantic
- **Search**: LocalHybridSearchEngine with investment-aware scoring
- **ML**: sentence-transformers, HDBSCAN, scikit-learn, TF-IDF
- **Investment Logic**: Entity detection, sentiment analysis, balanced scoring algorithms
- **CLI**: Click + Rich for beautiful terminal interface

## 🚀 API Endpoints

```bash
# Investment advisory search (recommended) - provides balanced perspectives
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"query": "should I buy Bitcoin?", "size": 5}'

# Traditional analysis search  
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"query": "Why is Bitcoin pumping?", "size": 5}'

# Simple keyword search
curl "http://localhost:8000/search?q=bitcoin ethereum&size=10"

# Clustered search with thematic organization
curl -X POST "http://localhost:8000/search/clustered" \
  -H "Content-Type: application/json" \
  -d '{"query": "crypto market analysis", "size": 15}'
```

## 📊 Performance

| Component | Performance | Details |
|-----------|------------|---------|
| **Search Response** | ~10-125ms | Optimized for real-time investment guidance |
| **CLI Experience** | Real-time | Sub-second user experience with balanced results |
| **Investment Analysis** | ~10ms | Entity-sentiment scoring with balanced perspectives |
| **Clustering** | 1-6ms | HDBSCAN thematic organization |
| **Embedding Model** | Pre-loaded | One-time startup (3s) vs per-query loading |

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
├── scripts/                  # Executable scripts and utilities
│   ├── cli_search.py         # Beautiful CLI interface
│   ├── setup_dev.sh          # Development environment setup
│   ├── start_server.sh       # API server startup
│   ├── search.sh             # CLI search wrapper
│   └── run_tests.sh          # Test runner
├── src/                      # Core application code
│   ├── api/main.py           # FastAPI endpoints
│   ├── search/local_search.py # Hybrid BM25+vector engine  
│   ├── search/cluster.py     # HDBSCAN clustering
│   ├── etl/embeddings.py     # 384D embedding generation
│   └── llm/spec_gen.py       # Natural language processing
├── tests/                    # Test suites
│   ├── integration/          # End-to-end system tests
│   └── unit/                 # Unit tests
├── data/                     # Sample tweets (50 documents)
├── normalized/               # Tweets with 384D embeddings
├── config/                   # Configuration files
└── requirements.txt
```

## 🎯 Key Features

- 💎 **Investment Advisory**: Balanced buy/sell/timing guidance eliminating confirmation bias
- 🧠 **Entity-Aware Filtering**: Automatic subject filtering (BTC queries → BTC content only)
- 📊 **Sentiment Intelligence**: Investment queries return both bullish AND bearish perspectives  
- ⚡ **Lightning Fast**: ~10-125ms response time optimized for real-time investment guidance
- ✅ **Production-Ready**: Real BM25 + 384D semantic search with investment-aware scoring
- 🎨 **Beautiful CLI**: Rich formatting with colors, emojis, and investment indicators
- 📂 **Smart Clustering**: HDBSCAN thematic result organization with confidence scoring
- 🔍 **Natural Language**: Advanced query → Investment intent detection → SearchSpec conversion
- 📈 **Educational Focus**: Prioritizes analysis and research over speculation and hype
- 🛠️ **Extensible**: Ready for OpenSearch integration and production scaling

## 🔧 Development

```bash
# Setup development environment
./scripts/setup_dev.sh

# Run comprehensive tests
./scripts/run_tests.sh

# Generate embeddings for new data
python -m src.etl.embeddings --input data/tweets.jsonl --output normalized/

# Test individual components
pytest tests/ -v
```

## 📈 Investment-Aware Search Quality

**Investment-Enhanced Hybrid Approach**: Combines precision of keyword matching (BM25) with semantic understanding (vector similarity) and **investment intelligence** for superior decision-making support.

**Advanced Entity-Sentiment Analysis**: 
- Automatic detection of tokens (BTC, ETH), KOLs, events, and macros with confidence scoring
- **Entity filtering**: "SOL queries" only return Solana-related content (eliminates irrelevant results)
- **Sentiment balancing**: Investment queries boost both bullish opportunities AND risk factors

**Investment Advisory Intelligence**:
- **Buy advice queries**: Return growth potential + risk assessment + market timing
- **Sell advice queries**: Prioritize exit signals + profit-taking strategies + market context  
- **Analysis queries**: Provide balanced perspectives + educational content + fundamental analysis
- **Speculation filtering**: Promotes research-based content over hype and speculation

**Intelligent Clustering**: Real-time HDBSCAN clustering groups results by themes like "Solana - Performance Issues vs NFT Growth" with automatic labeling and confidence scoring.

## 🛠️ Next Development

- **Advanced Investment Features**: Portfolio analysis, risk scoring, market timing indicators
- **LLM Integration**: OpenAI/Anthropic answer synthesis with citations and investment disclaimers
- **Personalization**: User investment preferences, risk tolerance, portfolio tracking
- **Advanced Analytics**: Sentiment trend analysis, social volume indicators, whale activity detection
- **Multi-Asset Support**: Expand beyond crypto to stocks, commodities, forex with investment intelligence

## 📄 License

MIT License - feel free to use for your projects!

---

**For detailed documentation, architecture details, and development guidelines, see [CLAUDE.md](CLAUDE.md)**