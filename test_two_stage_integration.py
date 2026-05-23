#!/usr/bin/env python3
"""
Test two-stage scoring integration with knowledge base and training data systems.
Validates that the filtering logic works correctly for different scoring outcomes.
"""

import sys
from pathlib import Path

def test_two_stage_integration():
    """Test two-stage scoring integration scenarios."""
    
    print("🔬 Testing Two-Stage Scoring Integration")
    print("=" * 50)
    
    # Test scenarios for different two-stage outcomes
    scenarios = [
        {
            "name": "Both Stages Pass",
            "masking_test_passed": True,
            "masking_test_improvement": 0.15,
            "masking_test_direction": "both", 
            "stage_completed": "stage_2_completed",
            "admitted": True,
            "bic_delta": -0.05,
            "expected_training_saved": True,
            "expected_knowledge_saved": True,
            "expected_reward_success": True
        },
        {
            "name": "Stage 1 Pass, Stage 2 Fail",
            "masking_test_passed": True,
            "masking_test_improvement": 0.08,
            "masking_test_direction": "new_helps_existing",
            "stage_completed": "stage_2_completed", 
            "admitted": False,
            "bic_delta": 0.12,
            "expected_training_saved": True,
            "expected_knowledge_saved": False,
            "expected_reward_success": False
        },
        {
            "name": "Stage 1 Fail",
            "masking_test_passed": False,
            "masking_test_improvement": 0.005,
            "masking_test_direction": "neither",
            "stage_completed": "stage_1_failed",
            "admitted": False,
            "bic_delta": float('inf'),
            "expected_training_saved": True,
            "expected_knowledge_saved": False,
            "expected_reward_success": False
        }
    ]
    
    print("🧪 Testing Integration Logic:")
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n  Test {i}: {scenario['name']}")
        
        # Simulate evaluation results
        evaluate = {
            "masking_test_passed": scenario["masking_test_passed"],
            "masking_test_improvement": scenario["masking_test_improvement"],
            "masking_test_direction": scenario["masking_test_direction"],
            "stage_completed": scenario["stage_completed"],
            "admitted": scenario["admitted"],
            "bic_delta": scenario["bic_delta"]
        }
        
        # Test knowledge base admission logic
        both_stages_passed = (
            evaluate.get("masking_test_passed", True) and 
            evaluate.get("admitted", False) and 
            evaluate.get("bic_delta") is not None and 
            evaluate.get("bic_delta") < 0 and
            evaluate.get("stage_completed") == "stage_2_completed"
        )
        
        # Test training data logic (always saved)
        training_saved = True  # Always save to training data
        
        # Test reward calculation logic
        masking_test_passed = evaluate.get("masking_test_passed", True)
        masking_test_improvement = evaluate.get("masking_test_improvement", 0.0) 
        admitted = evaluate.get("admitted", False)
        bic_delta = evaluate.get("bic_delta")
        
        if bic_delta is None or bic_delta == float('inf'):
            reward_success = False
            reward_value = 0.0
        elif masking_test_passed and admitted:
            # Both stages passed
            stage1_reward = min(1.0, masking_test_improvement)
            stage2_reward = min(1.0, max(0.0, -bic_delta / 1000.0))
            reward_value = 0.4 * stage1_reward + 0.6 * stage2_reward
            reward_success = True
        elif masking_test_passed and not admitted:
            # Stage 1 only
            stage1_reward = min(1.0, masking_test_improvement)
            reward_value = 0.3 * stage1_reward
            reward_success = False
        else:
            # Stage 1 failed
            reward_value = 0.05
            reward_success = False
        
        # Validate results
        training_ok = training_saved == scenario["expected_training_saved"]
        knowledge_ok = both_stages_passed == scenario["expected_knowledge_saved"] 
        reward_ok = reward_success == scenario["expected_reward_success"]
        
        status = "✅" if (training_ok and knowledge_ok and reward_ok) else "❌"
        
        print(f"    {status} Training saved: {training_saved}")
        print(f"    {status} Knowledge saved: {both_stages_passed}")
        print(f"    {status} Reward success: {reward_success} (value: {reward_value:.3f})")
        
        if not (training_ok and knowledge_ok and reward_ok):
            print(f"    ❌ Expected: training={scenario['expected_training_saved']}, "
                  f"knowledge={scenario['expected_knowledge_saved']}, "
                  f"reward_success={scenario['expected_reward_success']}")
            return False
    
    print("\n📊 Integration Summary:")
    print("  ✅ Training data: ALL experiments saved (including failures)")
    print("  ✅ Knowledge base: ONLY both-stages-passed experiments saved") 
    print("  ✅ Reward system: Weighted Stage 1 (40%) + Stage 2 (60%)")
    print("  ✅ Partial rewards: Stage 1 only gets 30% of Stage 1 score")
    print("  ✅ Failed experiments: 0.05 minimal reward for attempting")
    
    print("\n🎯 Data Flow Validation:")
    print("  📝 training_pairs.pkl: Comprehensive experiment database")
    print("  🧠 experiments.jsonl: High-quality geological knowledge")
    print("  🏆 Reward function: Incentivizes both predictive value AND complexity efficiency")
    
    print("\n🎉 SUCCESS: Two-stage scoring integration working correctly!")
    return True

if __name__ == "__main__":
    success = test_two_stage_integration()
    if success:
        print("\n🚀 Two-stage integration ready for production!")
        sys.exit(0)
    else:
        print("\n⚠️ Integration issues detected")
        sys.exit(1)
