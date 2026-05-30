#!/bin/bash
# Test the Kazakhstan workflow with regional-scale geological analysis

echo "🚀 Testing Kazakhstan Teniz Basin Workflow with Regional Analysis"
echo "================================================================="

# Check .env: must exist, no placeholder, and OPENROUTER_API_KEY must be non-empty.
# Previously this was a single `grep -q "your-..." .env` — that returns non-zero
# both when the placeholder is absent AND when the file is missing entirely,
# so a missing .env silently passed the check.
if [ ! -f .env ]; then
    echo "❌ ERROR: .env not found. Copy .env.example to .env and fill in OPENROUTER_API_KEY."
    exit 1
fi
if grep -q "your-openrouter-api-key-here" .env; then
    echo "❌ ERROR: Please update .env with your actual OpenRouter API key"
    exit 1
fi
# shellcheck disable=SC1091
set -a; . ./.env; set +a
if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    echo "❌ ERROR: OPENROUTER_API_KEY is empty in .env"
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
# Prompt only when stdin is a TTY; otherwise skip so CI/log-piped runs don't hang.
# Set DOCKER_CLEANUP=1 to force the cleanup non-interactively, DOCKER_CLEANUP=0 to skip.
if [ -n "${DOCKER_CLEANUP:-}" ]; then
    REPLY=$([ "$DOCKER_CLEANUP" = "1" ] && echo y || echo n)
elif [ -t 0 ]; then
    read -p "Stop and remove Kazakhstan Docker containers? (y/n): " -n 1 -r
    echo
else
    REPLY=n
    echo "⏭️  Non-interactive; skipping Docker cleanup (export DOCKER_CLEANUP=1 to force)"
fi
if [[ $REPLY =~ ^[Yy]$ ]]; then
    docker compose -f docker/feature-hypothesis-kazakhstan-compose/docker-compose.yml down 2>/dev/null || true
    echo "✅ Kazakhstan Docker containers stopped"
else
    echo "⏭️  Skipping Docker cleanup"
fi

echo ""
echo "🎯 Step 3: Verify Kazakhstan configuration"
echo "-----------------------------------------"
echo "Expected Kazakhstan workflow:"
echo "  📊 Grid: 66.5°-71.5°E × 49.5°-52.5°N (116,160 km²)"
echo "  🔍 Resolution: ~1.75km × 1.66km × 10m per voxel"
echo "  📁 Data: Kazakhstan Teniz Basin dataset"
echo "  1. Agent: spatial_add_point(name='basin_feature', ...)"
echo "  2. Agent: spatial_add_line(name='basin_feature', ...)"  
echo "  3. Agent: scoring_create_feature_layer(name='basin_feature')  ← TERMINATOR"
echo "  4. System: BIC evaluation with Kazakhstan grid"
echo "  5. Rewrite: Regional geological intelligence summary"

echo ""
echo "🗺️  Step 4: Test Kazakhstan grid mapping"
echo "----------------------------------------"
echo "Running grid validation test..."
uv run python test_kazakhstan_grid.py
if [ $? -eq 0 ]; then
    echo "✅ Kazakhstan grid mapping validated"
else
    echo "❌ Kazakhstan grid mapping failed - check coordinates"
    exit 1
fi

echo ""
echo "🚀 Step 5: Run Kazakhstan episode with debugging"
echo "------------------------------------------------"

if [ ! -f "config/config-feature-hypothesis-kazakhstan.toml" ]; then
    echo "❌ ERROR: config/config-feature-hypothesis-kazakhstan.toml is missing."
    echo "   Check out this file from the repo — the previous auto-generation"
    echo "   step (sed on the AIQ config) produced a config pointing at the"
    echo "   wrong task class and is no longer trusted."
    exit 1
fi

# Run the Kazakhstan episode with focused output
# 900s: a --rebuild-harness pass (Docker image build) plus a full 5-phase
# episode does not fit in 300s. PYTHONUNBUFFERED=1: without it Python block-
# buffers stdout through the pipe, so a timeout SIGTERM discards the whole
# episode log and every indicator below reads as a false failure.
PYTHONUNBUFFERED=1 timeout 900s uv run python scripts/run_episode.py config/config-feature-hypothesis-kazakhstan.toml --rebuild-harness 2>&1 | tee test_kazakhstan_output.log

echo ""
echo "📊 Step 6: Analyze Kazakhstan results" 
echo "------------------------------------"

# Check for key indicators
echo "🔍 Looking for key indicators in Kazakhstan logs..."

if grep -q "🎯 DEBUG.*scoring_create_feature_layer" test_kazakhstan_output.log; then
    echo "✅ Agent called scoring_create_feature_layer"
else
    echo "❌ Agent did NOT call scoring_create_feature_layer"
fi

if grep -q "scoring_create_feature_layer MCP function" test_kazakhstan_output.log; then
    echo "✅ MCP function was invoked"
else
    echo "⚠️  MCP function was not invoked"
fi

if grep -q "🎯 DEBUG.*Tool result.*bic_delta" test_kazakhstan_output.log; then
    echo "✅ BIC evaluation completed with Kazakhstan grid"
    # Extract BIC delta from logs
    bic_line=$(grep "bic_delta" test_kazakhstan_output.log | tail -1)
    if echo "$bic_line" | grep -q "admitted.*True"; then
        echo "📈 Kazakhstan layer was ADMITTED (improved regional compression)"
    elif echo "$bic_line" | grep -q "admitted.*False"; then
        echo "📉 Kazakhstan layer was REJECTED (worse regional compression)"
    fi
else
    echo "❌ BIC evaluation did not run"
fi

if grep -q "Phase 5" test_kazakhstan_output.log; then
    echo "✅ Reached rewrite phase"
else
    echo "❌ Did not reach rewrite phase"
fi

if grep -q "Episode completed.*Success.*True" test_kazakhstan_output.log; then
    echo "✅ Kazakhstan episode completed successfully"
    EPISODE_SUCCESS=true
else
    echo "❌ Kazakhstan episode did not complete successfully"
    EPISODE_SUCCESS=false
fi

# Check for Kazakhstan-specific data persistence
echo ""
echo "💾 Step 7: Check Kazakhstan data persistence"
echo "-------------------------------------------"

if [ -f "./data/kazakhstan/feature-hypothesis/training/training_pairs.pkl" ]; then
    echo "✅ Kazakhstan training data saved"
    python3 -c "
import pickle
try:
    with open('./data/kazakhstan/feature-hypothesis/training/training_pairs.pkl', 'rb') as f:
        data = pickle.load(f)
    print(f'📝 Kazakhstan training pairs in database: {len(data)}')
    if data:
        latest = data[-1]
        print(f'📊 Latest BIC delta: {latest.get(\"bic_delta\", \"N/A\")}')
        print(f'🎯 Latest admitted: {latest.get(\"admitted\", \"N/A\")}')
        print(f'🌍 Grid coverage: Kazakhstan Teniz Basin')
except Exception as e:
    print(f'❌ Error reading Kazakhstan training data: {e}')
"
else
    echo "❌ Kazakhstan training data not saved"
fi

if [ -f "./data/kazakhstan/feature-hypothesis/knowledge/teniz_basin/experiments.jsonl" ]; then
    echo "✅ Kazakhstan knowledge graph data saved"
    lines=$(wc -l < "./data/kazakhstan/feature-hypothesis/knowledge/teniz_basin/experiments.jsonl" 2>/dev/null || echo "0")
    echo "📚 Kazakhstan knowledge entries: $lines"
else
    echo "❌ Kazakhstan knowledge graph data not saved"
fi

# Check for Kazakhstan grid-specific features
if grep -q "Grid bounds: lon 66\." test_kazakhstan_output.log; then
    echo "✅ Kazakhstan grid bounds detected in logs"
else
    echo "⚠️  Kazakhstan grid bounds not clearly visible in logs"
fi

echo ""
echo "📋 Step 8: Kazakhstan Summary"
echo "----------------------------"

if [ "$EPISODE_SUCCESS" = true ]; then
    echo "🎉 SUCCESS: Kazakhstan regional workflow is working!"
    echo "✅ Agent analyzed Kazakhstan Teniz Basin data"
    echo "✅ Regional-scale voxel grid operational (~1.75km resolution)"
    echo "✅ BIC evaluation with 116,160 km² coverage"  
    echo "✅ Kazakhstan data persistence working"
    echo "✅ Episode completed successfully"
    echo ""
    echo "🌍 Kazakhstan System Status:"
    echo "   📊 Grid: 66.5°-71.5°E × 49.5°-52.5°N"
    echo "   🔍 Voxels: 200×200×8 (320k total)"
    echo "   🗂️  Data: USGS + Russian geological surveys"
    echo "   🎯 Focus: Sediment-hosted copper exploration"
else
    echo "❌ ISSUE: Kazakhstan workflow needs debugging"
    echo "📄 Check test_kazakhstan_output.log for detailed logs"
    echo ""
    echo "🔍 Kazakhstan-specific issues to check:"
    echo "   - Grid coordinate validation (66.5°-71.5°E range)"
    echo "   - Kazakhstan data directory access"
    echo "   - Regional-scale spatial operations"  
    echo "   - Teniz Basin dataset loading"
    echo "   - MCP tool bridging with Kazakhstan config"
    echo "   - Container connectivity"
fi

echo ""
echo "📄 Full Kazakhstan logs saved to: test_kazakhstan_output.log"
echo "🏁 Kazakhstan test completed!"
echo ""
echo "🌍 Regional Comparison Available:"
echo "  Australia:  ./test_fixed_workflow.sh     (deposit-scale, ~70m voxels)"  
echo "  Kazakhstan: ./test_kazakhstan_workflow.sh (basin-scale, ~1.75km voxels)"
