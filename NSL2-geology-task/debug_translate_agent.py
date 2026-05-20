#!/usr/bin/env python3
"""Debug what's happening with the translate agent."""

import subprocess
import re
import sys

# Run episode and capture all output
cmd = ["uv", "run", "python", "scripts/run_episode.py", "config/config-feature-hypothesis-aiq.toml"]

try:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    output = result.stdout + result.stderr
    
    print("🔍 TRANSLATE AGENT DEBUG")
    print("="*60)
    
    # Look for translate phase
    translate_section = ""
    lines = output.split('\n')
    in_translate = False
    
    for i, line in enumerate(lines):
        if "Phase 4: Translate" in line:
            in_translate = True
            translate_section = ""
        elif in_translate:
            translate_section += line + "\n"
            if "Phase 5:" in line or "Episode completed" in line:
                break
    
    if translate_section:
        print("📝 TRANSLATE PHASE ACTIVITY:")
        print("-" * 40)
        
        # Look for capability calls
        capability_calls = []
        for line in translate_section.split('\n'):
            if any(cap in line for cap in ['spatial_add', 'create_feature_layer', 'get_experiment_summary']):
                capability_calls.append(line.strip())
        
        if capability_calls:
            print("🔧 CAPABILITY CALLS MADE:")
            for call in capability_calls:
                print(f"  {call}")
        else:
            print("❌ NO CAPABILITY CALLS FOUND")
        
        # Check if create_feature_layer was mentioned at all
        if "create_feature_layer" in translate_section:
            print("✅ create_feature_layer mentioned in logs")
        else:
            print("❌ create_feature_layer NOT mentioned anywhere")
            
        # Check if termination happened
        if "Episode completed" in output:
            print("⚠️  Episode completed - phase may have ended prematurely")
        else:
            print("🔄 Episode still running or timed out")
            
    else:
        print("❌ Could not find translate phase in output")
        
    # Show last few lines for context
    print("\n📋 LAST 10 LINES OF OUTPUT:")
    print("-" * 40)
    for line in lines[-10:]:
        print(line)
        
except subprocess.TimeoutExpired:
    print("⏰ Test timed out - episode took too long")
except Exception as e:
    print(f"❌ Error running test: {e}")
