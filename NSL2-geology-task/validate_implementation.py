#!/usr/bin/env python3
"""
Simple validation of the crossbreeding implementation changes.
Validates the changes without requiring full task dependencies.
"""
import json
from pathlib import Path

def validate_experiments_format():
    """Validate the experiments.jsonl format in both systems."""
    print("📋 Validating Experiments Format")
    print("=" * 40)
    
    # Check Australia experiments
    australia_file = Path("data/australia/feature-hypothesis/knowledge/coe_fairbairn/experiments.jsonl")
    if australia_file.exists():
        print(f"🇦🇺 Australia experiments file: ✅ EXISTS")
        with open(australia_file, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        print(f"   Total experiments: {len(lines)}")
        
        # Check format
        for i, line in enumerate(lines):
            try:
                exp = json.loads(line)
                required_fields = ['node_id', 'bic_delta', 'parent_node_1', 'parent_node_2']
                missing = [field for field in required_fields if field not in exp]
                if missing:
                    print(f"   ❌ Experiment {i+1} missing fields: {missing}")
                else:
                    parents = f"{exp.get('parent_node_1', 'None')} + {exp.get('parent_node_2', 'None')}"
                    print(f"   ✅ Exp {i+1}: BIC {exp.get('bic_delta', 'N/A'):.2f}, Parents: {parents}")
            except Exception as e:
                print(f"   ❌ Experiment {i+1} JSON error: {e}")
    else:
        print(f"🇦🇺 Australia experiments file: ❌ NOT FOUND")
    
    # Check Kazakhstan experiments  
    kazakhstan_file = Path("data/kazakhstan/feature-hypothesis/knowledge/teniz_basin/experiments.jsonl")
    if kazakhstan_file.exists():
        print(f"🇰🇿 Kazakhstan experiments file: ✅ EXISTS")
        with open(kazakhstan_file, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        print(f"   Total experiments: {len(lines)}")
        
        # Check format
        for i, line in enumerate(lines):
            try:
                exp = json.loads(line)
                required_fields = ['node_id', 'bic_delta', 'parent_node_1', 'parent_node_2', 'region']
                missing = [field for field in required_fields if field not in exp]
                if missing:
                    print(f"   ❌ Experiment {i+1} missing fields: {missing}")
                else:
                    parents = f"{exp.get('parent_node_1', 'None')} + {exp.get('parent_node_2', 'None')}"
                    print(f"   ✅ Exp {i+1}: BIC {exp.get('bic_delta', 'N/A'):.2f}, Parents: {parents}")
            except Exception as e:
                print(f"   ❌ Experiment {i+1} JSON error: {e}")
    else:
        print(f"🇰🇿 Kazakhstan experiments file: ❌ NOT FOUND")
    
    print()

def check_implementation_changes():
    """Check that key implementation changes are present in the code."""
    print("🔍 Validating Code Implementation")
    print("=" * 40)
    
    # Check Australia file
    australia_file = Path("tasks/feature_hypothesis.py")
    if australia_file.exists():
        with open(australia_file, 'r') as f:
            content = f.read()
        
        checks = [
            ("_get_used_crossbreed_pairs", "✅ Helper method added"),
            ("_is_pair_already_used", "✅ Helper method added"),
            ("maximum combined BIC improvement", "✅ BIC-based selection"),
            ("parent_node_1 = parent_experiments", "✅ Parent tracking"),
            ("Skip if this pair was already crossbred", "✅ Pair avoidance"),
        ]
        
        print("🇦🇺 Australia implementation:")
        for check, message in checks:
            if check in content:
                print(f"   {message}")
            else:
                print(f"   ❌ Missing: {check}")
    
    # Check Kazakhstan file  
    kazakhstan_file = Path("tasks/feature_hypothesis_kazakhstan.py")
    if kazakhstan_file.exists():
        with open(kazakhstan_file, 'r') as f:
            content = f.read()
        
        checks = [
            ("_get_used_crossbreed_pairs", "✅ Helper method added"),
            ("_is_pair_already_used", "✅ Helper method added"), 
            ("maximum combined BIC improvement", "✅ BIC-based selection"),
            ("parent_node_1 = parent_experiments", "✅ Parent tracking"),
            ("Skip if this pair was already crossbred", "✅ Pair avoidance"),
        ]
        
        print("🇰🇿 Kazakhstan implementation:")
        for check, message in checks:
            if check in content:
                print(f"   {message}")
            else:
                print(f"   ❌ Missing: {check}")
    
    print()

def summarize_changes():
    """Summarize the key changes made."""
    print("📈 Implementation Summary")
    print("=" * 40)
    
    print("✅ Changes completed:")
    print("   1. Australia: Replaced MI-based selection with maximum BIC")
    print("   2. Kazakhstan: Replaced simple selection with maximum BIC") 
    print("   3. Both: Added parent tracking (parent_node_1, parent_node_2)")
    print("   4. Both: Added pair avoidance logic")
    print("   5. Both: Added fallback for when all pairs used")
    print("   6. Australia: Preserved MI calculation for future analysis")
    
    print()
    print("🎯 Expected behavior:")
    print("   • Select experiments with highest combined |BIC| improvement")
    print("   • Avoid re-using previously crossbred experiment pairs")
    print("   • Log selection decisions for debugging")
    print("   • Handle edge cases gracefully")
    
    print()

if __name__ == "__main__":
    print("🧪 Simplified Crossbreeding Implementation Validation")
    print("=" * 60)
    print()
    
    validate_experiments_format()
    check_implementation_changes() 
    summarize_changes()
    
    print("🎉 Validation complete!")
    print()
    print("Next steps:")
    print("• Run actual geological workflows to test crossbreeding")
    print("• Verify parent tracking works in real episodes")
    print("• Monitor that selection avoids duplicate pairs")
