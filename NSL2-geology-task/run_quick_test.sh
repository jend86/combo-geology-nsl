#!/bin/bash
# Quick test run (assumes containers are already built)

echo "Quick Feature Hypothesis Test"
echo "============================="

# Check if .env has been updated
if grep -q "your-openrouter-api-key-here" .env; then
    echo "ERROR: Please update .env with your actual OpenRouter API key"
    exit 1
fi

# Just run the episode
echo "Running single episode..."
uv run python scripts/run_episode.py config/config-feature-hypothesis-australia.toml

echo "Test completed!"
