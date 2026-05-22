#!/usr/bin/env python3
"""Test script for the new async execution MCP tools."""

import asyncio
import json
import sys
import os

# Add the voxel-features-mcp to the path
sys.path.insert(0, '/home/jen/Desktop/GeoNSL_monorepo-pre-kazakh-complete/voxel-features-mcp')

from voxel_features.mcp.tools.execution_tools import (
    execution_submit, execution_status, execution_results, execution_reset_session
)


async def test_execution_workflow():
    """Test the complete async execution workflow."""
    print("🧪 Testing async execution workflow...")
    
    # Test 1: Submit execution
    print("\n1. Testing execution_submit...")
    result = execution_submit(
        code="""
import pandas as pd
import numpy as np

# Create test data
data = pd.DataFrame({
    'x': np.random.randn(100),
    'y': np.random.randn(100),
    'z': np.random.randn(100)
})

# Analyze data
correlation = data.corr()
summary = data.describe()

print("Analysis completed!")
print(f"Data shape: {data.shape}")
print(f"Correlation matrix:\\n{correlation}")
""",
        timeout_s=30,
        max_attempts=3
    )
    
    print(f"Submit result: {json.dumps(result, indent=2)}")
    
    if not result.get('success', False):
        print("❌ Submit failed!")
        return False
    
    execution_id = result['execution_id']
    print(f"✅ Execution submitted with ID: {execution_id}")
    
    # Test 2: Monitor execution
    print(f"\n2. Testing execution_status for {execution_id}...")
    max_polls = 10
    polls = 0
    
    while polls < max_polls:
        status_result = execution_status(execution_id)
        print(f"Status poll {polls + 1}: {json.dumps(status_result, indent=2)}")
        
        if not status_result.get('success', False):
            print("❌ Status check failed!")
            return False
            
        status = status_result.get('status', '')
        if status in ['completed', 'failed', 'timeout']:
            print(f"✅ Execution finished with status: {status}")
            break
            
        polls += 1
        await asyncio.sleep(1)
    
    if polls >= max_polls:
        print("❌ Execution didn't complete in time!")
        return False
    
    # Test 3: Get results
    print(f"\n3. Testing execution_results for {execution_id}...")
    results_result = execution_results(execution_id)
    print(f"Results: {json.dumps(results_result, indent=2)}")
    
    if not results_result.get('success', False):
        print("❌ Results retrieval failed!")
        return False
    
    artifacts_count = results_result.get('artifacts_count', 0)
    print(f"✅ Retrieved results with {artifacts_count} artifacts")
    
    # Test 4: Budget exhaustion
    print(f"\n4. Testing budget exhaustion...")
    session_id = "test_session"
    
    # Reset session first
    reset_result = execution_reset_session(session_id)
    print(f"Reset result: {json.dumps(reset_result, indent=2)}")
    
    # Submit 3 executions to exhaust budget
    for i in range(4):  # Try 4, should fail on the 4th
        submit_result = execution_submit(
            code=f"print('Attempt {i + 1}')",
            session_id=session_id,
            max_attempts=3
        )
        
        print(f"Attempt {i + 1}: {json.dumps(submit_result, indent=2)}")
        
        if i < 3:
            if not submit_result.get('success', False):
                print(f"❌ Attempt {i + 1} should have succeeded!")
                return False
        else:
            if submit_result.get('success', False):
                print("❌ 4th attempt should have failed due to budget exhaustion!")
                return False
            else:
                print("✅ Budget exhaustion working correctly")
    
    print("\n🎉 All tests passed!")
    return True


def test_sync_execution():
    """Test execution tools synchronously."""
    print("🧪 Testing sync execution...")
    
    result = execution_submit(
        code="print('Hello from sync test!')",
        timeout_s=10
    )
    
    print(f"Sync result: {json.dumps(result, indent=2)}")
    
    if result.get('success', False):
        print("✅ Sync execution submitted successfully")
        return True
    else:
        print("❌ Sync execution failed")
        return False


async def main():
    """Run all tests."""
    print("🚀 Starting execution tools tests...")
    
    # Set up environment
    os.environ['VFM_ARTIFACT_DIR'] = '/tmp/voxel-features/artifacts'
    
    # Test sync first
    sync_success = test_sync_execution()
    
    if not sync_success:
        print("❌ Sync tests failed, skipping async tests")
        return
    
    # Test async workflow
    async_success = await test_execution_workflow()
    
    if async_success:
        print("🎉 All execution tool tests passed!")
    else:
        print("❌ Some tests failed")


if __name__ == "__main__":
    asyncio.run(main())
