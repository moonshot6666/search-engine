#!/bin/bash

# Hybrid Search Engine - Interactive Demo Script
# Showcases the key features of the investment-aware search system

echo "🚀 Hybrid Search Engine - Interactive Demo"
echo "==========================================="
echo ""

# Check if API server is running
if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "⚠️  API server not detected. Starting server..."
    echo "   Run './scripts/start_server.sh' in another terminal"
    echo "   Then re-run this demo script"
    exit 1
fi

echo "✅ API server is running at http://localhost:8000"
echo ""

echo "🎯 Demo 1: Investment Advisory Queries"
echo "--------------------------------------"
echo "Query: 'should I buy sol' (balanced perspectives)"
echo ""
./scripts/search.sh "should I buy sol" --size 3
echo ""

echo "🎯 Demo 2: Compound Entity + Sentiment Queries"  
echo "----------------------------------------------"
echo "Query: 'why is sol bullish' (entity + sentiment matching)"
echo ""
./scripts/search.sh "why is sol bullish" --size 3
echo ""

echo "🎯 Demo 3: News Queries with Entity Priority"
echo "--------------------------------------------"
echo "Query: 'show news for sol' (news + entity filtering)"
echo ""
./scripts/search.sh "show news for sol" --size 3
echo ""

echo "🎯 Demo 4: Multi-Entity Relationship Queries"
echo "--------------------------------------------" 
echo "Query: 'what is elon musk saying about btc' (KOL + token)"
echo ""
./scripts/search.sh "what is elon musk saying about btc" --size 3
echo ""

echo "🎯 Demo 5: Clustered Results"
echo "---------------------------"
echo "Query: 'crypto market analysis' with thematic clustering"
echo ""
./scripts/search.sh "crypto market analysis" --clustered --size 8
echo ""

echo "✅ Demo Complete!"
echo ""
echo "🔧 Try these commands yourself:"
echo "   ./scripts/search.sh --interactive    # Interactive mode"
echo "   ./scripts/search.sh --help           # All options"
echo "   ./scripts/run_tests.sh               # Run all tests"
echo ""
echo "📖 Documentation: README.md and CLAUDE.md"