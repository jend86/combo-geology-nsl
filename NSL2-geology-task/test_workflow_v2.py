#!/usr/bin/env python3
"""
Test script to validate the V2 workflow loads correctly with BIC evaluation step.
Forces module reload to bypass any caching issues.
"""

import sys
import importlib

def force_reload_modules():
    """Force reload of task modules to bypass Python caching."""
    modules_to_reload = [
        'tasks.feature_hypothesis',
        'tasks.feature_hypothesis_v2'
    ]
    
    for module_name in modules_to_reload:
        if module_name in sys.modules:
            print(f"🔄 Reloading {module_name}")
            importlib.reload(sys.modules[module_name])
        else:
            print(f"📦 Loading {module_name} for first time")

def test_original_workflow():
    """Test the original workflow to confirm it's missing BIC evaluation."""
    print("\n🔍 Testing Original Workflow")
    print("=" * 40)
    
    try:
        from tasks.feature_hypothesis import FeatureHypothesisTask, FeatureHypothesisVariation
        
        task = FeatureHypothesisTask()
        variation = FeatureHypothesisVariation(
            name='coe_fairbairn', 
            description='test', 
            dataset_dir='/test', 
            store_dir='/test', 
            kg_dir='/test'
        )
        workflow = task.episode_workflow(variation, {'workflow_kind': 'survey'})
        
        step_names = [step.name for step in workflow.steps]
        print(f"📋 Original workflow steps: {step_names}")
        
        # Check translate step
        translate_step = next((s for s in workflow.steps if s.name == 'translate'), None)
        if translate_step:
            print(f"🔧 Translate capabilities: {translate_step.capabilities}")
            print(f"🎯 Translate terminators: {translate_step.terminator_capabilities}")
            print(f"➡️  Translate next steps: {translate_step.next_steps}")
            
            has_bic_eval = 'create_feature_layer' in translate_step.capabilities
            print(f"🧪 Has BIC evaluation: {has_bic_eval}")
        
        # Check for evaluation step
        eval_step = next((s for s in workflow.steps if s.name == 'evaluate_spatial_layer'), None)
        print(f"⚡ Has evaluation step: {eval_step is not None}")
        
    except Exception as e:
        print(f"❌ Error testing original workflow: {e}")
        import traceback
        traceback.print_exc()

def test_v2_workflow():
    """Test the V2 workflow to confirm BIC evaluation is present."""
    print("\n🔍 Testing V2 Workflow")
    print("=" * 40)
    
    try:
        from tasks.feature_hypothesis_v2 import FeatureHypothesisTaskV2, FeatureHypothesisVariation
        
        task = FeatureHypothesisTaskV2()
        variation = FeatureHypothesisVariation(
            name='coe_fairbairn', 
            description='test', 
            dataset_dir='/test', 
            store_dir='/test', 
            kg_dir='/test'
        )
        workflow = task.episode_workflow(variation, {'workflow_kind': 'survey'})
        
        step_names = [step.name for step in workflow.steps]
        print(f"📋 V2 workflow steps: {step_names}")
        
        # Check translate step
        translate_step = next((s for s in workflow.steps if s.name == 'translate'), None)
        if translate_step:
            print(f"🔧 Translate capabilities: {translate_step.capabilities}")
            print(f"🎯 Translate terminators: {translate_step.terminator_capabilities}")
            print(f"➡️  Translate next steps: {translate_step.next_steps}")
            
            has_bic_eval = 'create_feature_layer' in translate_step.capabilities
            print(f"🧪 Translate has BIC evaluation: {has_bic_eval}")
        
        # Check for evaluation step
        eval_step = next((s for s in workflow.steps if s.name == 'evaluate_spatial_layer'), None)
        if eval_step:
            print(f"⚡ Has evaluation step: {eval_step is not None}")
            print(f"🔧 Evaluation capabilities: {eval_step.capabilities}")
            print(f"🎯 Evaluation terminators: {eval_step.terminator_capabilities}")
            print(f"➡️  Evaluation next steps: {eval_step.next_steps}")
            
            has_create_layer = 'create_feature_layer' in eval_step.capabilities
            print(f"🧪 Evaluation has create_feature_layer: {has_create_layer}")
        else:
            print(f"❌ No evaluation step found!")
        
        # Validate workflow flow
        if translate_step and eval_step:
            correct_flow = (
                translate_step.next_steps == ("evaluate_spatial_layer",) and
                eval_step.next_steps == ("rewrite",)
            )
            print(f"✅ Correct workflow flow: {correct_flow}")
        
        print(f"🎯 Expected flow: translate → evaluate_spatial_layer → rewrite")
        
    except Exception as e:
        print(f"❌ Error testing V2 workflow: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main test function."""
    print("🧪 Feature Hypothesis Workflow V2 Test")
    print("=" * 50)
    
    # Force reload modules
    force_reload_modules()
    
    # Test both workflows
    test_original_workflow()
    test_v2_workflow()
    
    print(f"\n🎯 Summary:")
    print(f"- Original workflow: Should be missing BIC evaluation step")
    print(f"- V2 workflow: Should have proper evaluate_spatial_layer step")
    print(f"- V2 flow: translate → evaluate_spatial_layer → rewrite")

if __name__ == "__main__":
    main()
