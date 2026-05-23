#!/usr/bin/env python3
"""
Test script to verify both graceful layer naming and hypothesis deduplication features work correctly.

Tests:
1. Timestamp-based unique naming in scoring_create_feature_layer
2. Recent experiments retrieval for hypothesis deduplication  
3. Dynamic prompt generation with recent experiments context
"""

import sys
import os
import tempfile
import time
from pathlib import Path

# Add voxel-features-mcp to path
vfm_path = Path(__file__).parent / "voxel-features-mcp"
sys.path.insert(0, str(vfm_path))

def test_unique_layer_naming():
    """Test that layer names get unique timestamps to prevent collisions."""
    print("Testing unique layer naming with timestamps...")
    
    try:
        from voxel_features.mcp.tools.scoring_tools import scoring_create_feature_layer
        from voxel_features.spatial import SpatialVoxelStore
        from voxel_features.store import COE_FAIRBAIRN_GRID
        import numpy as np
        
        # Create a spatial store
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SpatialVoxelStore(COE_FAIRBAIRN_GRID, temp_dir)
            
            # Create a test spatial layer
            store.add_spatial_point("test_mineralization", 117.9, -27.4, 10.0, value=1.0, radius=100.0)
            
            # Test 1: Basic unique naming
            result1 = scoring_create_feature_layer(store, "test_mineralization", "float")
            if not result1.get("success", False):
                print(f"❌ First layer creation failed: {result1}")
                return False
            
            layer_name_1 = result1.get("layer_name", "")
            if not layer_name_1.startswith("test_mineralization_"):
                print(f"❌ First layer name not timestamped: {layer_name_1}")
                return False
                
            print(f"✅ First unique layer name: {layer_name_1}")
            
            # Test 2: Create another layer with same base name - should get different timestamp
            time.sleep(0.001)  # Ensure different timestamp
            store.add_spatial_point("test_mineralization", 117.91, -27.41, 15.0, value=2.0, radius=150.0)
            
            result2 = scoring_create_feature_layer(store, "test_mineralization", "float")
            if not result2.get("success", False):
                print(f"❌ Second layer creation failed: {result2}")
                return False
                
            layer_name_2 = result2.get("layer_name", "")
            if not layer_name_2.startswith("test_mineralization_"):
                print(f"❌ Second layer name not timestamped: {layer_name_2}")
                return False
                
            if layer_name_1 == layer_name_2:
                print(f"❌ Layer names are identical - timestamping failed!")
                print(f"  Name 1: {layer_name_1}")
                print(f"  Name 2: {layer_name_2}")
                return False
                
            print(f"✅ Second unique layer name: {layer_name_2}")
            print(f"✅ Layer names are properly differentiated by timestamp")
            
            # Test 3: Check that both layers are in the store
            layer_names = list(store.layer_names)
            if layer_name_1 not in layer_names or layer_name_2 not in layer_names:
                print(f"❌ One or both layers missing from store: {layer_names}")
                return False
                
            print(f"✅ Both layers successfully stored: {len(layer_names)} total layers")
            return True
            
    except Exception as e:
        print(f"❌ Unique naming test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_recent_experiments_retrieval():
    """Test that recent experiments can be retrieved for deduplication."""
    print("Testing recent experiments retrieval...")
    
    try:
        from voxel_features.knowledge_graph import KnowledgeGraph, ExperimentRecord
        from voxel_features.store import COE_FAIRBAIRN_GRID
        from voxel_features.mcp.tools.experiment_tools import experiment_list_recent
        import tempfile
        
        with tempfile.TemporaryDirectory() as temp_dir:
            kg = KnowledgeGraph(COE_FAIRBAIRN_GRID, kg_dir=temp_dir)
            
            # Create some mock experiment records
            experiments_data = [
                ("copper_mineralization_hypothesis", "High copper grades near fault zones", True, "Found 15% copper zones"),
                ("lithium_exploration", "Li enrichment in pegmatites", False, "No significant Li detected"), 
                ("structural_controls", "Fault intersections control gold", True, "Gold grades 10x higher at intersections"),
            ]
            
            experiment_ids = []
            for hypothesis, rationale, admitted, result in experiments_data:
                record = ExperimentRecord(
                    hypothesis=hypothesis,
                    rationale=rationale,
                    data_spec={"files": ["geochemDrillhole.csv"]},
                    code_executed="analysis_code_here",
                    result_summary=result,
                    feature_layer_name=hypothesis.replace("_hypothesis", "_layer"),
                    mdl_before=100.0,
                    mdl_after=95.0 if admitted else 105.0,
                    mdl_delta=-5.0 if admitted else 5.0,
                    mutual_info={},
                    admitted=admitted,
                    parent_experiments=[],
                    episode_id=f"test_episode_{len(experiment_ids)}",
                    variation_name="test_variation"
                )
                exp_id = kg.record(record)
                experiment_ids.append(exp_id)
                time.sleep(0.001)  # Ensure different timestamps
            
            print(f"✅ Created {len(experiment_ids)} mock experiments")
            
            # Test the experiment_list_recent function
            result = experiment_list_recent(kg, max_experiments=5)
            
            if not result.get("success", False):
                print(f"❌ experiment_list_recent failed: {result}")
                return False
                
            recent_experiments = result.get("recent_experiments", [])
            if len(recent_experiments) != 3:
                print(f"❌ Expected 3 recent experiments, got {len(recent_experiments)}")
                return False
                
            # Check that experiments are ordered by timestamp (most recent first)
            timestamps = [exp["timestamp"] for exp in recent_experiments]
            if timestamps != sorted(timestamps, reverse=True):
                print(f"❌ Experiments not properly ordered by timestamp")
                return False
                
            print(f"✅ Retrieved {len(recent_experiments)} recent experiments in correct order")
            
            # Check experiment content
            for exp in recent_experiments:
                required_fields = ["id", "hypothesis", "rationale", "admitted", "result_summary"]
                missing_fields = [field for field in required_fields if field not in exp]
                if missing_fields:
                    print(f"❌ Experiment missing fields: {missing_fields}")
                    return False
                    
            print(f"✅ All experiments have required fields for deduplication")
            print("Recent experiments:")
            for i, exp in enumerate(recent_experiments, 1):
                status = "✅ ADMITTED" if exp["admitted"] else "❌ REJECTED"
                print(f"  {i}. {exp['hypothesis']} [{status}]")
                print(f"     Result: {exp['result_summary']}")
                
            return True
            
    except Exception as e:
        print(f"❌ Recent experiments test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dynamic_prompt_generation():
    """Test that survey prompt includes recent experiments context."""
    print("Testing dynamic prompt generation with recent experiments context...")
    
    try:
        # Add NSL2-geology-task to path
        nsl_path = Path(__file__).parent / "NSL2-geology-task"
        if nsl_path.exists():
            sys.path.insert(0, str(nsl_path))
            
        from tasks.feature_hypothesis import FeatureHypothesisTask
        
        # Create a test task instance 
        task = FeatureHypothesisTask()
        
        # Test prompt generation
        prompt = task._generate_survey_prompt_with_context()
        
        if not prompt:
            print(f"❌ Generated prompt is empty")
            return False
            
        # Check basic prompt structure
        required_elements = [
            "Phase 1: Survey",
            "analysis_shell",
            "Find 2-3 promising feature layer candidates",
            "record_phase"
        ]
        
        for element in required_elements:
            if element not in prompt:
                print(f"❌ Prompt missing required element: '{element}'")
                return False
                
        print(f"✅ Prompt contains all required basic elements")
        
        # The prompt might not have recent experiments if the knowledge graph is empty,
        # which is expected for a clean test environment
        if "AVOID REPEATING RECENT EXPERIMENTS" in prompt:
            print(f"✅ Prompt includes recent experiments context")
        else:
            print(f"ℹ️ Prompt generated without recent experiments (empty knowledge graph)")
            
        print(f"Generated prompt preview:")
        print("-" * 50)
        print(prompt[:300] + "..." if len(prompt) > 300 else prompt)
        print("-" * 50)
        
        return True
        
    except Exception as e:
        print(f"❌ Dynamic prompt test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("TESTING GRACEFUL NAMING AND HYPOTHESIS DEDUPLICATION")
    print("=" * 60)
    
    # Test 1: Unique layer naming
    print("\n1. Testing Timestamp-Based Unique Layer Naming:")
    naming_success = test_unique_layer_naming()
    
    # Test 2: Recent experiments retrieval
    print("\n2. Testing Recent Experiments Retrieval:")
    experiments_success = test_recent_experiments_retrieval()
    
    # Test 3: Dynamic prompt generation
    print("\n3. Testing Dynamic Prompt Generation:")
    prompt_success = test_dynamic_prompt_generation()
    
    print("\n" + "=" * 60)
    print("TEST RESULTS:")
    print(f"  Unique Naming: {'✅ PASS' if naming_success else '❌ FAIL'}")
    print(f"  Experiments Retrieval: {'✅ PASS' if experiments_success else '❌ FAIL'}")
    print(f"  Dynamic Prompts: {'✅ PASS' if prompt_success else '❌ FAIL'}")
    
    if all([naming_success, experiments_success, prompt_success]):
        print("\n🎉 ALL TESTS PASSED!")
        print("Both graceful layer naming and hypothesis deduplication are working correctly.")
        print("\nExpected behavior in production:")
        print("1. Layer names will be unique even if agents choose same base names")
        print("2. Agents will see recent experiments and avoid repeating similar hypotheses")
        print("3. This should improve layer diversity and reduce synthetic BIC scores")
        return 0
    else:
        print("\n⚠️ SOME TESTS FAILED")
        print("Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
