#!/bin/bash

# CLI Search Interface wrapper script
# This script should be run from the project root directory

# Check if we're in the right directory
if [ ! -f "scripts/cli_search.py" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    echo "   Expected to find: scripts/cli_search.py"
    exit 1
fi

# Check if virtual environment exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Pass all arguments to the CLI search script
python scripts/cli_search.py "$@"