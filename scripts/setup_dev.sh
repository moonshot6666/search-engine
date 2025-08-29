#!/bin/bash

# Development environment setup for Hybrid Search Engine
# This script should be run from the project root directory

echo "🛠️  Hybrid Search Engine - Development Setup"
echo "============================================="

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    echo "   Expected to find: requirements.txt"
    exit 1
fi

echo ""
echo "🐍 Setting up Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "ℹ️  Virtual environment already exists"
fi

source venv/bin/activate
echo "✅ Virtual environment activated"

echo ""
echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
echo "✅ Dependencies installed"

echo ""
echo "🐳 Checking Docker setup..."
if command -v docker &> /dev/null; then
    echo "✅ Docker is available"
    if docker ps &> /dev/null; then
        echo "✅ Docker daemon is running"
    else
        echo "⚠️  Docker daemon not running - start Docker to use OpenSearch"
    fi
else
    echo "⚠️  Docker not found - install Docker to use OpenSearch"
fi

echo ""
echo "📊 Checking data files..."
if [ -f "data/expanded_sample_tweets.jsonl" ]; then
    tweet_count=$(wc -l < "data/expanded_sample_tweets.jsonl")
    echo "✅ Sample data: $tweet_count tweets"
else
    echo "⚠️  Sample data not found: data/expanded_sample_tweets.jsonl"
fi

if [ -f "normalized/tweets_with_embeddings.jsonl" ]; then
    echo "✅ Embeddings data available"
else
    echo "⚠️  Embeddings not found - run embedding generation if needed"
fi

echo ""
echo "🎯 Setup complete! Next steps:"
echo "   1. Start API server:     ./scripts/start_server.sh"
echo "   2. Run tests:            ./scripts/run_tests.sh"  
echo "   3. Search queries:       ./scripts/search.sh \"your query\""
echo "   4. Interactive search:   ./scripts/search.sh --interactive"
echo ""
echo "📖 Documentation: README.md"
echo "🔧 Configuration: CLAUDE.md"