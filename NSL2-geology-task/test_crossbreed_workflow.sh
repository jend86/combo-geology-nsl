#!/bin/bash
# test_crossbreed_workflow.sh
#
# Shortish end-to-end check that BOTH workflow kinds run and succeed:
#   - survey     rounds  (bootstrap — no parent experiments yet)
#   - crossbreed round   (combines two admitted parent experiments)
#
# How crossbreeding triggers — see tasks/feature_hypothesis.py :: populate():
#   workflow_kind = "crossbreed"  iff
#       variation.crossbreed_enabled            (default True)
#       AND n_features >= variation.min_features (default 0 — always true)
#       AND >=2 admitted experiments (bic_delta < 0) are recorded in
#           data/feature-hypothesis/knowledge/coe_fairbairn/experiments.jsonl
#
# Only an *admitted* experiment (bic_delta < 0) is appended to the knowledge
# graph, so it takes >=2 admitted survey episodes before crossbreeding triggers.
# With real data not every episode's feature layer is admitted, so the script
# runs survey episodes one at a time until the graph holds >=2 entries, then the
# next episode auto-switches to "crossbreed". MAX_EPISODES is a generous cap.

set -u
cd "$(dirname "$0")"

CONFIG="config/config-feature-hypothesis-australia.toml"
KG="./data/feature-hypothesis/knowledge/coe_fairbairn/experiments.jsonl"
MAX_EPISODES=8
PER_EPISODE_TIMEOUT=900s

echo "🧬 Crossbreed Workflow Test (survey → crossbreed)"
echo "================================================="
echo "Runs survey episodes until the knowledge graph holds >=2 admitted"
echo "experiments, then a crossbreed episode. Verifies both kinds work."
echo ""

# --- Fresh slate -----------------------------------------------------------
# Episode counts must be deterministic: a leftover knowledge graph with >=2
# entries would make episode 1 crossbreed and we'd never observe a survey run.
echo "🧹 Step 1: Clear data/, code/, and __pycache__ for a deterministic run"
echo "---------------------------------------------------------------------"
rm -rf data/feature-hypothesis code/feature-hypothesis
find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
echo "✅ Cleared"
echo ""

# --- Run episodes until both kinds are observed ----------------------------
echo "🚀 Step 2: Run episodes (survey rounds seed the graph, then crossbreed)"
echo "----------------------------------------------------------------------"

survey_ok=0
crossbreed_seen=0
crossbreed_ok=0
crossbreed_log=""

for i in $(seq 1 "$MAX_EPISODES"); do
    log="/tmp/crossbreed_ep_${i}.log"
    echo ""
    echo "▶️  Episode $i — running (log: $log) ..."

    PYTHONUNBUFFERED=1 timeout "$PER_EPISODE_TIMEOUT" \
        uv run python scripts/run_episode.py "$CONFIG" > "$log" 2>&1
    rc=$?

    # Mode: the crossbreed workflow's entry-phase prompt is the only place the
    # string "Crossbreed Mode" appears (Phase 2: Hypothesise (Crossbreed Mode)).
    if grep -q "Crossbreed Mode" "$log"; then
        mode="crossbreed"
    else
        mode="survey"
    fi

    if grep -q "Episode completed - Success: True" "$log"; then
        success="True"
    else
        success="False"
    fi

    kg_count=0
    [ -f "$KG" ] && kg_count=$(grep -c . "$KG" 2>/dev/null || echo 0)

    echo "   → mode=$mode  success=$success  exit_code=$rc  knowledge_graph_entries=$kg_count"

    if [ "$mode" = "survey" ] && [ "$success" = "True" ]; then
        survey_ok=1
    fi
    if [ "$mode" = "crossbreed" ]; then
        crossbreed_seen=1
        crossbreed_log="$log"
        [ "$success" = "True" ] && crossbreed_ok=1
    fi

    # Done once we've confirmed a good survey episode and have run a crossbreed.
    if [ "$survey_ok" = 1 ] && [ "$crossbreed_seen" = 1 ]; then
        echo ""
        echo "✅ Observed both a survey and a crossbreed episode — stopping."
        break
    fi
done

# --- Inspect the crossbreed episode ----------------------------------------
echo ""
echo "🔍 Step 3: Inspect the crossbreed episode"
echo "-----------------------------------------"
if [ -n "$crossbreed_log" ]; then
    parents=$(grep -m1 "Parent experiments:" "$crossbreed_log" | sed 's/.*Parent experiments:/Parent experiments:/')
    [ -n "$parents" ] && echo "✅ $parents" || echo "⚠️  No 'Parent experiments:' line found"

    grep -q "Crossbreed Mode" "$crossbreed_log" \
        && echo "✅ Ran the crossbreed hypothesise phase" \
        || echo "❌ Crossbreed phase prompt not found"

    grep -q "🎯 DEBUG.*scoring_create_feature_layer" "$crossbreed_log" \
        && echo "✅ Called scoring_create_feature_layer (BIC terminator)" \
        || echo "❌ scoring_create_feature_layer not called"

    bic=$(grep -oE "'bic_delta': [-0-9.]+" "$crossbreed_log" | tail -1)
    [ -n "$bic" ] && echo "✅ BIC evaluation ran ($bic)" || echo "❌ BIC evaluation did not run"

    grep -q "Episode completed - Success: True" "$crossbreed_log" \
        && echo "✅ Crossbreed episode completed successfully" \
        || echo "❌ Crossbreed episode did not complete successfully"
else
    echo "❌ No crossbreed episode was reached within $MAX_EPISODES episodes."
fi

# --- Summary ---------------------------------------------------------------
echo ""
echo "📋 Step 4: Summary"
echo "------------------"
[ "$survey_ok" = 1 ]     && echo "✅ Survey round ran and succeeded" \
                         || echo "❌ Survey round did not succeed"
[ "$crossbreed_ok" = 1 ] && echo "✅ Crossbreed round ran and succeeded" \
                         || echo "❌ Crossbreed round did not succeed"
echo ""

if [ "$survey_ok" = 1 ] && [ "$crossbreed_ok" = 1 ]; then
    echo "🎉 SUCCESS: both survey and crossbreed workflows work."
    exit 0
else
    echo "❌ ISSUE: see /tmp/crossbreed_ep_*.log for details."
    exit 1
fi
