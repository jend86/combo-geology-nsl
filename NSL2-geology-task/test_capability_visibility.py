#!/usr/bin/env python3
"""Test if scoring_create_feature_layer capability is visible."""

import sys
import os
sys.path.append('.')

try:
    from tasks.feature_hypothesis import FeatureHypothesisTask, FeatureHypothesisVariation
    from pathlib import Path
    
    # Create task with minimal config
    task_config = {
        "dataset_dir": "/home/jen/Desktop/geonsl/NSL2-geology-task/data/coe_fairbairn"
    }
    task = FeatureHypothesisTask(task_config)
    
    # Get capabilities
    variations = task.list_variations()
    capabilities = task.list_capabilities(variations[0], {})
    
    print("🔍 ALL TASK CAPABILITIES:")
    for cap in capabilities:
        print(f"  - {cap.name}: {cap.description}")
    
    print(f"\n🎯 Total capabilities: {len(capabilities)}")
    
    # Check specifically for our capability
    scoring_caps = [cap for cap in capabilities if 'scoring' in cap.name]
    if scoring_caps:
        print("\n✅ SCORING CAPABILITIES FOUND:")
        for cap in scoring_caps:
            print(f"  - {cap.name}: {cap.description}")
    else:
        print("\n❌ NO SCORING CAPABILITIES FOUND")
    
    # Check workflow step directly from task
    variation = variations[0]
    workflow_def = task._survey_workflow(variation, {})
    
    for step in workflow_def.steps:
        if step.name == "translate":
            print(f"\n📝 TRANSLATE STEP CAPABILITIES:")
            for cap_name in step.capabilities:
                print(f"  - {cap_name}")
            print(f"\n🎯 TERMINATORS: {step.terminator_capabilities}")
            print(f"🔄 NEXT STEPS: {step.next_steps}")
            
            # Check if our capability is in the list
            if "scoring_create_feature_layer" in step.capabilities:
                print("✅ scoring_create_feature_layer is in translate capabilities")
            else:
                print("❌ scoring_create_feature_layer NOT in translate capabilities")
                
            if "scoring_create_feature_layer" in step.terminator_capabilities:
                print("✅ scoring_create_feature_layer is a terminator")
            else:
                print("❌ scoring_create_feature_layer NOT a terminator")
            break
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
