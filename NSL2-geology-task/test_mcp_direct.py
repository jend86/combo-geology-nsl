#!/usr/bin/env python3
"""
Direct test of MCP integration without full episode workflow
"""

import sys
from pathlib import Path

# Add paths
sys.path.append(str(Path(__file__).parent.parent / "voxel-features-mcp"))

def test_mcp_tool():
    """Test the MCP tool directly"""
    print("🧪 Direct MCP Tool Test")
    print("======================")
    
    try:
        # Import MCP tool directly
        from voxel_features.spatial import SpatialVoxelStore
        from voxel_features.store import COE_FAIRBAIRN_GRID
        from voxel_features.mcp.tools.scoring_tools import scoring_create_feature_layer
        print("✅ Imports successful")
        
        # Create a test store
        store_dir = "./data/feature-hypothesis/store/coe_fairbairn"
        store = SpatialVoxelStore(store_dir, COE_FAIRBAIRN_GRID)
        print(f"✅ Store created: {store.grid.shape}")
        
        # Add a test layer
        from voxel_features.mcp.tools.spatial_tools import spatial_add_point
        result = spatial_add_point(
            store=store,
            name="test_layer",
            longitude=117.9,
            latitude=-27.35,
            depth_m=30,
            value=1.0,
            radius_m=100
        )
        print(f"✅ Test layer created: {result}")
        
        # Test our scoring function
        scoring_result = scoring_create_feature_layer(
            store=store,
            name="test_layer",
            dtype="float"
        )
        print(f"✅ Scoring function worked: {scoring_result.get('success', False)}")
        print(f"📊 BIC Delta: {scoring_result.get('bic_delta', 'N/A')}")
        print(f"🎯 Admitted: {scoring_result.get('admitted', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_task_capability():
    """Test that task knows about the capability"""
    print("\n🔧 Task Capability Test")
    print("======================")
    
    try:
        from tasks.feature_hypothesis import FeatureHypothesisTask
        task = FeatureHypothesisTask()
        capabilities = task.list_capabilities()
        
        scoring_caps = [cap for cap in capabilities if 'scoring' in cap.name]
        if scoring_caps:
            print(f"✅ Found scoring capabilities: {[cap.name for cap in scoring_caps]}")
            return True
        else:
            print("❌ No scoring capabilities found")
            return False
            
    except Exception as e:
        print(f"❌ Error testing task: {e}")
        return False

if __name__ == "__main__":
    mcp_ok = test_mcp_tool()
    task_ok = test_task_capability()
    
    print(f"\n🏁 Summary:")
    print(f"MCP Tool: {'✅' if mcp_ok else '❌'}")
    print(f"Task Capability: {'✅' if task_ok else '❌'}")
    
    if mcp_ok and task_ok:
        print("🎉 MCP Integration is working!")
        print("The issue is likely in episode workflow progression, not our integration.")
    else:
        print("🔧 There are integration issues to fix.")
