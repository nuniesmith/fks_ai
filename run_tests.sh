#!/bin/bash
# Test runner for AI service

set -e

echo "Running AI service tests..."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Install dependencies if needed
if [ ! -d ".pytest_cache" ]; then
    echo "Installing test dependencies..."
    pip install -r requirements.txt
fi

# Run tests
echo "Running unit tests..."
pytest tests/unit/test_bots/ -v --tb=short

echo "Tests completed!"

