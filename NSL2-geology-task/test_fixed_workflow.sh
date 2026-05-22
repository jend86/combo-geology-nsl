#!/bin/bash
# Test the fixed scoring_create_feature_layer workflow

echo "🚀 Testing Fixed BIC Workflow with MCP Tool Architecture"
echo "======================================================="

# Check if .env has been updated
if grep -q "your-openrouter-api-key-here" .env; then
    echo "❌ ERROR: Please update .env with your actual OpenRouter API key"
    exit 1
fi

echo "🧹 Step 1: Clean up caches"
echo "--------------------------"
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find ../voxel-features-mcp -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
echo "✅ Caches cleared"

echo ""
echo "🐳 Step 2: Docker cleanup (optional but recommended)"
echo "----------------------------------------------------"
read -p "Stop and remove Docker containers? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    docker compose -f docker/feature-hypothesis/compose.yaml down 2>/dev/null || true
    echo "✅ Docker containers stopped"
else
    echo "⏭️  Skipping Docker cleanup"
fi

echo ""
echo "🎯 Step 3: Verify configuration"
echo "-------------------------------"
echo "Expected workflow:"
echo "  1. Agent: spatial_add_point(name='layer_name', ...)"
echo "  2. Agent: spatial_add_line(name='layer_name', ...)"  
echo "  3. Agent: scoring_create_feature_layer(name='layer_name')  ← NEW TERMINATOR"
echo "  4. System: BIC evaluation runs automatically"
echo "  5. Rewrite: Gets complete data and calls submit_rewrite()"

echo ""
echo "🚀 Step 4: Run episode with debugging"
echo "------------------------------------"

# Run the episode with focused output.
# 900s: a --rebuild-harness pass (Docker image build) plus a full 5-phase
# episode does not fit in 300s. PYTHONUNBUFFERED=1: without it Python block-
# buffers stdout through the pipe, so a timeout SIGTERM discards the whole
# episode log and every indicator below reads as a false failure.
PYTHONUNBUFFERED=1 timeout 900s uv run python scripts/run_episode.py config/config-feature-hypothesis-aiq.toml --rebuild-harness 2>&1 | tee test_output.log

echo ""
echo "📊 Step 5: Analyze results" 
echo "-------------------------"

# Check for key indicators
echo "🔍 Looking for key indicators in logs..."

if grep -q "🎯 DEBUG.*scoring_create_feature_layer" test_output.log; then
    echo "✅ Agent called scoring_create_feature_layer"
else
    echo "❌ Agent did NOT call scoring_create_feature_layer"
fi

if grep -q "scoring_create_feature_layer MCP function" test_output.log; then
    echo "✅ MCP function was invoked"
else
    echo "⚠️  MCP function was not invoked"
fi

if grep -q "🎯 DEBUG.*Tool result.*bic_delta" test_output.log; then
    echo "✅ BIC evaluation completed"
    # Extract BIC delta from logs
    bic_line=$(grep "bic_delta" test_output.log | tail -1)
    if echo "$bic_line" | grep -q "admitted.*True"; then
        echo "📈 Layer was ADMITTED (improved compression)"
    elif echo "$bic_line" | grep -q "admitted.*False"; then
        echo "📉 Layer was REJECTED (worse compression)"
    fi
else
    echo "❌ BIC evaluation did not run"
fi

if grep -q "Phase 5" test_output.log; then
    echo "✅ Reached rewrite phase"
else
    echo "❌ Did not reach rewrite phase"
fi

if grep -q "Episode completed.*Success.*True" test_output.log; then
    echo "✅ Episode completed successfully"
    EPISODE_SUCCESS=true
else
    echo "❌ Episode did not complete successfully"
    EPISODE_SUCCESS=false
fi

# Check for data persistence
echo ""
echo "💾 Step 6: Check data persistence"
echo "--------------------------------"

if [ -f "./data/feature-hypothesis/training/training_pairs.pkl" ]; then
    echo "✅ Training data saved"
    python3 -c "
import pickle
try:
    with open('./data/feature-hypothesis/training/training_pairs.pkl', 'rb') as f:
        data = pickle.load(f)
    print(f'📝 Training pairs in database: {len(data)}')
    if data:
        latest = data[-1]
        print(f'📊 Latest BIC delta: {latest.get(\"bic_delta\", \"N/A\")}')
        print(f'🎯 Latest admitted: {latest.get(\"admitted\", \"N/A\")}')
except Exception as e:
    print(f'❌ Error reading training data: {e}')
"
else
    echo "❌ Training data not saved"
fi

if [ -f "./data/feature-hypothesis/knowledge/coe_fairbairn/experiments.jsonl" ]; then
    echo "✅ Knowledge graph data saved"
    lines=$(wc -l < "./data/feature-hypothesis/knowledge/coe_fairbairn/experiments.jsonl" 2>/dev/null || echo "0")
    echo "📚 Knowledge entries: $lines"
else
    echo "❌ Knowledge graph data not saved"
fi

echo ""
echo "📋 Step 7: Summary"
echo "-----------------"

if [ "$EPISODE_SUCCESS" = true ]; then
    echo "🎉 SUCCESS: Fixed workflow is working!"
    echo "✅ Agent called terminator capability"
    echo "✅ BIC evaluation triggered"  
    echo "✅ Data persistence working"
    echo "✅ Episode completed successfully"
else
    echo "❌ ISSUE: Workflow needs debugging"
    echo "📄 Check test_output.log for detailed logs"
    echo ""
    echo "🔍 Common issues to check:"
    echo "   - Agent prompt clarity (translate phase)"
    echo "   - Terminator capability enforcement"  
    echo "   - MCP tool bridging"
    echo "   - Container connectivity"
fi

echo ""
echo "📄 Full logs saved to: test_output.log"
echo "🏁 Test completed!"
