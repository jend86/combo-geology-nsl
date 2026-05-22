#!/bin/bash
# Test run with FIXED original workflow (BIC evaluation added to translate phase)

echo "🔧 Testing FIXED Feature Hypothesis Workflow"
echo "============================================="

# Check if .env has been updated
if grep -q "your-openrouter-api-key-here" .env; then
    echo "ERROR: Please update .env with your actual OpenRouter API key"
    exit 1
fi

echo "📋 Using FIXED original task with BIC evaluation in translate phase"
echo "🎯 Expected: translate phase now requires create_feature_layer call"
echo "🧪 Expected flow: spatial_ops → create_feature_layer → rewrite"
echo ""

# Run single episode with fixed workflow
echo "🚀 Running single episode with fixed workflow..."
uv run python scripts/run_episode.py config/config-feature-hypothesis-aiq.toml

echo ""
echo "📊 Checking results..."

# Check if data was persisted
if [ -f "./data/feature-hypothesis/training/training_pairs.pkl" ]; then
    echo "✅ Training data saved successfully"
else
    echo "❌ Training data not saved"
fi

if [ -f "./data/feature-hypothesis/knowledge/coe_fairbairn/experiments.jsonl" ]; then
    echo "✅ Knowledge graph data saved successfully"  
else
    echo "❌ Knowledge graph data not saved"
fi

echo ""
echo "🎯 Fixed Workflow Test completed!"
