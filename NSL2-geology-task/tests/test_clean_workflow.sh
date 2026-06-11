#!/bin/bash
# Test the clean workflow with create_feature_layer terminator

echo "🚀 Testing Clean BIC Workflow"
echo "============================="

# Check if .env has been updated
if grep -q "your-openrouter-api-key-here" .env; then
    echo "ERROR: Please update .env with your actual OpenRouter API key"
    exit 1
fi

echo "🎯 CLEAN WORKFLOW APPROACH:"
echo "  • Agent does spatial operations (spatial_add_point, spatial_add_line)"
echo "  • Agent MUST call create_feature_layer(name='layer_name') to complete translate phase" 
echo "  • This triggers BIC evaluation and saves results to phase records"
echo "  • Rewrite phase gets complete experiment data including BIC results"
echo "  • Agent writes training pair with full context of effectiveness"
echo ""

# Run single episode
echo "🚀 Running episode with clean workflow..."
uv run python scripts/run_episode.py config/config-feature-hypothesis-australia.toml

echo ""
echo "📊 Checking results..."

# Check if data was persisted
if [ -f "./data/feature-hypothesis/training/training_pairs.pkl" ]; then
    echo "✅ Training data saved successfully"
    python -c "
import pickle
with open('./data/feature-hypothesis/training/training_pairs.pkl', 'rb') as f:
    data = pickle.load(f)
print(f'📝 Training pairs in database: {len(data)}')
if data:
    latest = data[-1]
    print(f'📊 Latest BIC delta: {latest.get(\"bic_delta\", \"N/A\")}')
    print(f'🎯 Latest admitted: {latest.get(\"admitted\", \"N/A\")}')
"
else
    echo "❌ Training data not saved"
fi

if [ -f "./data/feature-hypothesis/knowledge/coe_fairbairn/experiments.jsonl" ]; then
    echo "✅ Knowledge graph data saved successfully"
    lines=$(wc -l < "./data/feature-hypothesis/knowledge/coe_fairbairn/experiments.jsonl")
    echo "📚 Knowledge entries: $lines"
else
    echo "❌ Knowledge graph data not saved"
fi

echo ""
echo "🎯 Clean Workflow Test completed!"
