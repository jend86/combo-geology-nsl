#!/bin/bash
echo "🔍 Quick Phase Transition Check"
echo "==============================="

echo "🚀 Running episode with short timeout..."
timeout 60s uv run python scripts/run_episode.py config/config-feature-hypothesis-australia.toml --rebuild-harness 2>&1 | grep -E "Phase [0-9]:|🎯 DEBUG.*scoring|scoring_create_feature_layer|submit_code|translate|rewrite" | tail -20

echo ""
echo "📊 Summary:"
echo "----------"
last_output=$(timeout 60s uv run python scripts/run_episode.py config/config-feature-hypothesis-australia.toml --rebuild-harness 2>&1)

if echo "$last_output" | grep -q "Phase 3.*Code"; then
    echo "✅ Reached Phase 3 (Code)"
else
    echo "❌ Did not reach Phase 3"
fi

if echo "$last_output" | grep -q "Phase 4.*Translate"; then
    echo "✅ Reached Phase 4 (Translate)"
else
    echo "❌ Did not reach Phase 4 (Translate)"  
fi

if echo "$last_output" | grep -q "scoring_create_feature_layer"; then
    echo "✅ Agent called scoring_create_feature_layer"
else
    echo "❌ Agent did NOT call scoring_create_feature_layer"
fi

if echo "$last_output" | grep -q "🎯 DEBUG.*scoring"; then
    echo "✅ Scoring capability executed"
else
    echo "❌ Scoring capability was not executed"
fi
