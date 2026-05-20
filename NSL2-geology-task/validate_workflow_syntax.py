#!/usr/bin/env python3
"""
Simple syntax validation for V2 workflow without importing full task.
Checks if the workflow structure is correct without Docker dependencies.
"""

import ast
import sys

def validate_workflow_syntax():
    """Validate the V2 workflow file syntax and structure."""
    print("🔍 Validating V2 Workflow Syntax")
    print("=" * 40)
    
    try:
        # Read and parse the V2 file
        with open('/home/jen/Desktop/geonsl/NSL2-geology-task/tasks/feature_hypothesis_v2.py', 'r') as f:
            source_code = f.read()
        
        # Parse the AST
        tree = ast.parse(source_code)
        print("✅ Python syntax is valid")
        
        # Look for the episode_workflow method
        workflow_method = None
        for node in ast.walk(tree):
            if (isinstance(node, ast.FunctionDef) and 
                node.name == 'episode_workflow'):
                workflow_method = node
                break
        
        if workflow_method:
            print("✅ Found episode_workflow method")
            
            # Extract workflow steps from the source
            steps_info = extract_workflow_steps(source_code)
            if steps_info:
                print("✅ Found workflow steps:")
                for step_name, details in steps_info.items():
                    print(f"   📋 {step_name}")
                    if details['capabilities']:
                        print(f"      🔧 Capabilities: {', '.join(details['capabilities'])}")
                    if details['terminators']:
                        print(f"      🎯 Terminators: {', '.join(details['terminators'])}")
                    if details['next_steps']:
                        print(f"      ➡️  Next: {', '.join(details['next_steps'])}")
                
                # Validate the BIC evaluation flow
                validate_bic_flow(steps_info)
            else:
                print("❌ Could not extract workflow steps")
        else:
            print("❌ episode_workflow method not found")
            
    except Exception as e:
        print(f"❌ Error validating syntax: {e}")
        import traceback
        traceback.print_exc()

def extract_workflow_steps(source_code):
    """Extract workflow step information from source code."""
    steps = {}
    
    # Look for WorkflowStep definitions
    lines = source_code.split('\n')
    current_step = None
    in_step = False
    
    for line in lines:
        line = line.strip()
        
        # Start of a workflow step
        if 'WorkflowStep(' in line:
            in_step = True
            continue
            
        # Step name
        if in_step and line.startswith('name="'):
            current_step = line.split('"')[1]
            steps[current_step] = {
                'capabilities': [],
                'terminators': [],
                'next_steps': []
            }
            continue
            
        # Capabilities
        if in_step and current_step and 'capabilities=(' in line:
            # Look for the capabilities list in the next few lines
            cap_start = True
            continue
            
        if in_step and current_step and line.startswith('"') and line.endswith('",'):
            # This is likely a capability
            capability = line.strip('"",')
            if capability and '(' not in capability:  # Skip comments
                steps[current_step]['capabilities'].append(capability)
            continue
            
        # Terminator capabilities
        if in_step and current_step and 'terminator_capabilities=(' in line:
            # Extract terminators from the line
            if '"' in line:
                # Single line terminators
                terms = line.split('(')[1].split(')')[0]
                for term in terms.split(','):
                    term = term.strip().strip('"')
                    if term:
                        steps[current_step]['terminators'].append(term)
            continue
            
        # Next steps
        if in_step and current_step and 'next_steps=(' in line:
            if '"' in line:
                nexts = line.split('(')[1].split(')')[0]
                for next_step in nexts.split(','):
                    next_step = next_step.strip().strip('"')
                    if next_step:
                        steps[current_step]['next_steps'].append(next_step)
            continue
            
        # End of step
        if in_step and line == '),':
            in_step = False
            current_step = None
            continue
    
    return steps

def validate_bic_flow(steps_info):
    """Validate the BIC evaluation flow is correct."""
    print(f"\n🧪 Validating BIC Evaluation Flow")
    print("=" * 40)
    
    # Check if evaluate_spatial_layer step exists
    if 'evaluate_spatial_layer' in steps_info:
        print("✅ Found evaluate_spatial_layer step")
        
        eval_step = steps_info['evaluate_spatial_layer']
        
        # Check capabilities
        if 'create_feature_layer' in eval_step['capabilities']:
            print("✅ evaluate_spatial_layer has create_feature_layer capability")
        else:
            print("❌ evaluate_spatial_layer missing create_feature_layer capability")
            
        # Check terminators
        if 'create_feature_layer' in eval_step['terminators']:
            print("✅ evaluate_spatial_layer terminates on create_feature_layer")
        else:
            print("❌ evaluate_spatial_layer doesn't terminate on create_feature_layer")
            
        # Check next steps
        if 'rewrite' in eval_step['next_steps']:
            print("✅ evaluate_spatial_layer goes to rewrite")
        else:
            print("❌ evaluate_spatial_layer doesn't go to rewrite")
            
    else:
        print("❌ evaluate_spatial_layer step not found")
        
    # Check translate step flow
    if 'translate' in steps_info:
        translate_step = steps_info['translate']
        if 'evaluate_spatial_layer' in translate_step['next_steps']:
            print("✅ translate step goes to evaluate_spatial_layer")
        else:
            print("❌ translate step doesn't go to evaluate_spatial_layer")
            print(f"   Current next steps: {translate_step['next_steps']}")
    else:
        print("❌ translate step not found")
        
    # Validate complete flow
    expected_flow = ['survey', 'hypothesise', 'code', 'translate', 'evaluate_spatial_layer', 'rewrite']
    actual_steps = list(steps_info.keys())
    
    print(f"\n📋 Expected flow: {' → '.join(expected_flow)}")
    print(f"📋 Actual steps: {actual_steps}")
    
    has_all_steps = all(step in actual_steps for step in expected_flow)
    print(f"✅ All required steps present: {has_all_steps}")

def main():
    """Main validation function."""
    print("🧪 V2 Workflow Syntax Validation")
    print("=" * 50)
    
    validate_workflow_syntax()

if __name__ == "__main__":
    main()
