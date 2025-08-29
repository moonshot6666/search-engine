#!/bin/bash

# Test runner for Hybrid Search Engine
# This script should be run from the project root directory

echo "🧪 Hybrid Search Engine - Test Runner"
echo "====================================="

# Check if we're in the right directory
if [ ! -f "tests/integration/test_search_system.py" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    echo "   Expected to find: tests/integration/test_search_system.py"
    exit 1
fi

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "🔧 Activating virtual environment..."
    source venv/bin/activate
fi

echo ""
echo "🔍 Running integration tests..."
echo "-------------------------------------"

# Run comprehensive system test
python tests/integration/test_search_system.py

echo ""
echo "🔍 Running unit tests..."
echo "-------------------------------------"

# Run unit tests if they exist
if [ -d "tests/unit" ] && [ "$(ls -A tests/unit/*.py 2>/dev/null)" ]; then
    pytest tests/unit/ -v
else
    echo "ℹ️  No unit tests found in tests/unit/"
fi

# Run API tests
if [ -f "tests/test_api.py" ]; then
    echo ""
    echo "🔍 Running API tests..."
    echo "-------------------------------------"
    pytest tests/test_api.py -v
fi

echo ""
echo "✅ Test run completed!"