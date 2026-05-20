#!/usr/bin/env python3
"""Test scoring_create_feature_layer capability directly."""

import sys
import os
sys.path.append('/home/jen/Desktop/geonsl/voxel-features-mcp')

try:
    from voxel_features.mcp.tools.scoring_tools import scoring_create_feature_layer
    print("✅ Successfully imported scoring_create_feature_layer")
    
    # Test the function signature
    import inspect
    sig = inspect.signature(scoring_create_feature_layer)
    print(f"📝 Function signature: {sig}")
    
    # Try calling with minimal args (will fail but should show it's callable)
    try:
        result = scoring_create_feature_layer(None, "test_layer")
        print(f"⚠️  Unexpected success: {result}")
    except Exception as e:
        if "NoneType" in str(e) or "store" in str(e).lower():
            print("✅ Function callable (failed as expected with None store)")
        else:
            print(f"❌ Unexpected error: {e}")

except ImportError as e:
    print(f"❌ Import failed: {e}")
except Exception as e:
    print(f"❌ Other error: {e}")
