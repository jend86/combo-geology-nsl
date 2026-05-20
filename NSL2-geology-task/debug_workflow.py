#!/usr/bin/env python3
"""Debug workflow step configuration to verify terminator capabilities."""

import sys
sys.path.append('.')

from tasks.feature_hypothesis import FeatureHypothesisTask
from tasks.feature_hypothesis import FeatureHypothesisVariation

# Create task with config and get workflow  
from pathlib import Path

task_config_path = Path("/home/jen/Desktop/geonsl/NSL2-geology-task/tasks/feature_hypothesis")
task = FeatureHypothesisTask(task_config_path)
variations = task.list_variations()
variation = variations[0]  # Get first variation

if isinstance(variation, FeatureHypothesisVariation):
    workflow = task.get_workflow(variation, crossbreed=False)
    
    print("🔍 WORKFLOW DEBUG")
    print("="*50)
    
    for step in workflow.steps:
        print(f"\n📝 Step: {step.name}")
        print(f"   Capabilities: {step.capabilities}")
        print(f"   🎯 Terminators: {step.terminator_capabilities}")
        print(f"   ➡️  Next steps: {step.next_steps}")
        
        if step.name == "translate":
            print(f"\n🔧 TRANSLATE STEP ANALYSIS:")
            print(f"   - Has create_feature_layer: {'create_feature_layer' in step.capabilities}")
            print(f"   - Terminator is create_feature_layer: {step.terminator_capabilities == ('create_feature_layer',)}")
            print(f"   - Next step is rewrite: {step.next_steps == ('rewrite',)}")
            
            # Check if prompt mentions terminator requirement
            prompt_mentions_mandatory = any(word in step.prompt.lower() for word in ['mandatory', 'required', 'must call'])
            print(f"   - Prompt mentions requirement: {prompt_mentions_mandatory}")
else:
    print("❌ Could not get FeatureHypothesisVariation")
