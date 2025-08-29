#!/bin/bash

# Start the Hybrid Search Engine API server
# This script should be run from the project root directory

echo "🚀 Starting Hybrid Search Engine API server..."
echo "📍 Project root: $(pwd)"

# Check if we're in the right directory
if [ ! -f "src/api/main.py" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    echo "   Expected to find: src/api/main.py"
    exit 1
fi

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "🔧 Activating virtual environment..."
    source venv/bin/activate
fi

# Check if dependencies are installed
if ! python -c "import fastapi" 2>/dev/null; then
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
fi

echo "🔄 Starting API server on http://localhost:8000"
echo "📊 API endpoints available:"
echo "   • POST /ask - Natural language queries"
echo "   • GET  /search - Traditional search"
echo "   • POST /search/clustered - Clustered results"
echo ""
echo "🛑 Press Ctrl+C to stop the server"
echo ""

uvicorn src.api.main:app --reload --port 8000