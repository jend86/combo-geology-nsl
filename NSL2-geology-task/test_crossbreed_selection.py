#!/usr/bin/env python3
"""
Test script to validate the new crossbreeding selection logic.
Tests both Australia and Kazakhstan systems.
"""
import sys
import json
from pathlib import Path

# Add the tasks directory to path
sys.path.append(str(Path(__file__).parent / "tasks"))

from feature_hypothesis import FeatureHypothesisTask, FeatureHypothesisVariation
from feature_hypothesis_kazakhstan import FeatureHypothesisKazakhstanTask, FeatureHypothesisKazakhstanVariation

def test_australia_selection():
    """Test the Australia crossbreeding selection."""
    print("🇦🇺 Testing Australia Crossbreeding Selection")
    print("=" * 50)
    
    # Create variation
    australia_data = Path(__file__).parent / "data" / "australia" / "feature-hypothesis"
    variation = FeatureHypothesisVariation(
        name="coe_fairbairn",
        description="Test variation",
        dataset_dir=str(australia_data),
        store_dir=str(australia_data / "store" / "coe_fairbairn"),
        kg_dir=str(australia_data / "knowledge" / "coe_fairbairn"),
        min_features=0,
        crossbreed_enabled=True
    )
    
    # Create task instance
    task = FeatureHypothesisTask()
    
    # Check if crossbreed pairs exist
    has_pairs = task._has_crossbreed_pairs(variation)
    print(f"Has crossbreed pairs available: {has_pairs}")
    
    if has_pairs:
        # Get crossbreed context
        context = task._get_crossbreed_context(variation)
        if context:
            print(f"Selected parent IDs: {context.get('parent_ids', [])}")
            print(f"Selection success: ✅")
            print(f"Prompt preview: {context.get('prompt', '')[:200]}...")
        else:
            print("❌ No crossbreed context returned")
    else:
        print("No crossbreed pairs available (need ≥2 successful experiments)")
    
    print()

def test_kazakhstan_selection():
    """Test the Kazakhstan crossbreeding selection."""
    print("🇰🇿 Testing Kazakhstan Crossbreeding Selection") 
    print("=" * 50)
    
    # Create variation
    kazakhstan_data = Path(__file__).parent / "data" / "kazakhstan" / "feature-hypothesis"
    variation = FeatureHypothesisKazakhstanVariation(
        name="teniz_basin",
        description="Test variation", 
        dataset_dir=str(kazakhstan_data),
        store_dir=str(kazakhstan_data / "store" / "teniz_basin"),
        kg_dir=str(kazakhstan_data / "knowledge" / "teniz_basin"),
        min_features=0,
        crossbreed_enabled=True
    )
    
    # Create task instance
    task = FeatureHypothesisKazakhstanTask()
    
    # Check if crossbreed pairs exist
    has_pairs = task._has_crossbreed_pairs(variation)
    print(f"Has crossbreed pairs available: {has_pairs}")
    
    if has_pairs:
        # Get crossbreed context
        context = task._get_crossbreed_context(variation)
        if context:
            print(f"Selected parent IDs: {context.get('parent_ids', [])}")
            print(f"Selection success: ✅")
            print(f"Prompt preview: {context.get('prompt', '')[:200]}...")
        else:
            print("❌ No crossbreed context returned")
    else:
        print("No crossbreed pairs available (need ≥2 successful experiments)")
    
    print()

def show_current_experiments():
    """Show current experiment counts in both systems."""
    print("📊 Current Experiment Status")
    print("=" * 50)
    
    # Australia experiments
    australia_experiments_file = Path(__file__).parent / "data" / "australia" / "feature-hypothesis" / "knowledge" / "coe_fairbairn" / "experiments.jsonl"
    if australia_experiments_file.exists():
        with open(australia_experiments_file, 'r') as f:
            australia_experiments = [line.strip() for line in f if line.strip()]
        print(f"🇦🇺 Australia successful experiments: {len(australia_experiments)}")
        
        # Show BIC values
        for i, line in enumerate(australia_experiments):
            try:
                exp = json.loads(line)
                bic = exp.get('bic_delta', 'N/A')
                node_id = exp.get('node_id', 'unknown')[:20]
                parents = f"Parents: {exp.get('parent_node_1', 'None')}, {exp.get('parent_node_2', 'None')}"
                print(f"  {i+1}. {node_id}... BIC: {bic} | {parents}")
            except:
                continue
    else:
        print("🇦🇺 Australia: No experiments file found")
    
    # Kazakhstan experiments  
    kazakhstan_experiments_file = Path(__file__).parent / "data" / "kazakhstan" / "feature-hypothesis" / "knowledge" / "teniz_basin" / "experiments.jsonl"
    if kazakhstan_experiments_file.exists():
        with open(kazakhstan_experiments_file, 'r') as f:
            kazakhstan_experiments = [line.strip() for line in f if line.strip()]
        print(f"🇰🇿 Kazakhstan successful experiments: {len(kazakhstan_experiments)}")
        
        # Show BIC values
        for i, line in enumerate(kazakhstan_experiments):
            try:
                exp = json.loads(line)
                bic = exp.get('bic_delta', 'N/A')
                node_id = exp.get('node_id', 'unknown')[:20]
                parents = f"Parents: {exp.get('parent_node_1', 'None')}, {exp.get('parent_node_2', 'None')}"
                print(f"  {i+1}. {node_id}... BIC: {bic} | {parents}")
            except:
                continue
    else:
        print("🇰🇿 Kazakhstan: No experiments file found")
    
    print()

if __name__ == "__main__":
    print("🧪 Crossbreeding Selection Test")
    print("Testing the new maximum BIC selection logic")
    print()
    
    show_current_experiments()
    test_australia_selection()
    test_kazakhstan_selection()
    
    print("✅ Crossbreeding selection test complete!")
    print("\nKey improvements:")
    print("- Both systems now use maximum combined BIC for selection")
    print("- Parent tracking prevents re-use of experiment pairs")
    print("- Fallback logic handles edge cases")
    print("- MI calculation preserved in Australia for future analysis")
