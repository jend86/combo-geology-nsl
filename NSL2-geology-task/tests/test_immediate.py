#!/usr/bin/env python3
"""Test immediately if create_feature_layer works when called directly."""

# Simulate what would happen if agent actually called create_feature_layer
# by calling task.execute_capability directly

import sys
import os
sys.path.append('.')

# Add the real task import
from tasks.feature_hypothesis import FeatureHypothesisTask, FeatureHypothesisVariation
from src.task.types import CapabilityInvocation, CapabilityExecutionContext
from pathlib import Path

try:
    # Create minimal task context
    task_config = {
        "dataset_dir": "/home/jen/Desktop/geonsl/NSL2-geology-task/data/coe_fairbairn"
    }
    task = FeatureHypothesisTask(task_config)
    
    # Create fake context
    ctx = CapabilityExecutionContext(
        episode_context={
            "episode_id": "test",
            "store_dir": "/home/jen/Desktop/geonsl/NSL2-geology-task/data/feature-hypothesis/store/coe_fairbairn",
            "phase_records": {
                "translate": {
                    "feature_layer_name": "test_layer"
                }
            }
        }
    )
    
    # Create capability invocation
    invocation = CapabilityInvocation(
        name="create_feature_layer",
        input={"name": "test_layer"}
    )
    
    print("🧪 Testing create_feature_layer capability directly...")
    print(f"   Invocation: {invocation}")
    
    # Try to call it (this will fail due to no containers, but should show if capability exists)
    try:
        result = task.execute_capability(invocation, [], task.list_variations()[0], ctx)
        print(f"✅ Capability executed successfully: {result}")
    except Exception as e:
        if "containers" in str(e).lower() or "container" in str(e).lower():
            print(f"✅ Capability found but failed due to containers (expected): {e}")
        else:
            print(f"❌ Capability execution failed: {e}")
            
except Exception as e:
    print(f"❌ Setup failed: {e}")
    import traceback
    traceback.print_exc()
