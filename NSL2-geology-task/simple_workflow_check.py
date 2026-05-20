#!/usr/bin/env python3
"""Simple check of workflow step configuration."""

import sys
sys.path.append('.')

# Just read the workflow definition directly from the Python file
import re

with open("tasks/feature_hypothesis.py", "r") as f:
    content = f.read()

# Extract translate step configuration
translate_start = content.find('WorkflowStep(\n                    name="translate"')
if translate_start == -1:
    print("❌ Could not find translate step")
    sys.exit(1)

# Find the terminator_capabilities line
terminator_line = ""
lines = content[translate_start:].split('\n')
for line in lines:
    if 'terminator_capabilities' in line:
        terminator_line = line.strip()
        break

print("🔍 TRANSLATE STEP TERMINATOR CHECK")
print("="*50)
print(f"Found terminator line: {terminator_line}")

if "create_feature_layer" in terminator_line:
    print("✅ create_feature_layer is set as terminator")
else:
    print("❌ create_feature_layer is NOT set as terminator") 

# Also check if spatial operations are still terminators
if "spatial_add_point" in terminator_line or "spatial_add_line" in terminator_line:
    print("⚠️  WARNING: Spatial operations are still terminators!")
else:
    print("✅ Spatial operations are not terminators")

print(f"\n🎯 Current terminator configuration: {terminator_line}")

# Check capabilities list too
caps_line = ""
for line in lines:
    if 'capabilities=(' in line:
        caps_section_start = lines.index(line)
        for j in range(caps_section_start, min(caps_section_start + 10, len(lines))):
            if 'create_feature_layer' in lines[j]:
                caps_line = lines[j].strip()
                break
        break

if caps_line:
    print(f"✅ create_feature_layer in capabilities: {caps_line}")
else:
    print("❌ create_feature_layer NOT in capabilities")
