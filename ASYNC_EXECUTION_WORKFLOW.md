# Async Code Execution Workflow

## Overview

The geological AI training system now uses asynchronous code execution with budget control to prevent agent rabbit holes and provide better timeout handling. This replaces the previous synchronous `submit_code` capability.

## Key Benefits

- **Agent timeout awareness**: Agents can detect and respond to timeouts
- **Budget control**: Prevents endless retry loops (3 attempts max)
- **Better error feedback**: Distinguish between timeouts, execution errors, and missing artifacts
- **Artifact validation**: Ensures coding phase produces useable outputs before proceeding

## MCP Tools

### execution.submit
Submit code for async execution with budget tracking.

**Parameters:**
- `code` (required): Python code to execute
- `timeout_s` (optional): Execution timeout in seconds (default 300)
- `session_id` (optional): Session ID for budget tracking
- `max_attempts` (optional): Maximum attempts for session (default 3)
- `artifact_root` (optional/internal): Host directory for this episode's artifacts. Kazakhstan runs pass a repo-local `train_data/artifacts/<run_id>/<episode_id>` root automatically.

**Returns:**
```json
{
  "success": true,
  "execution_id": "exec_abc123",
  "attempts_used": "1/3",
  "status": "pending"
}
```

### execution.status
Check execution status and progress.

**Parameters:**
- `execution_id` (required): Execution ID to check

**Returns:**
```json
{
  "success": true,
  "execution_id": "exec_abc123",
  "status": "completed",
  "progress": "Execution completed successfully with 2 artifacts",
  "runtime_s": 45.2,
  "timeout_s": 300,
  "artifacts_count": 2
}
```

### execution.results
Get final results and artifacts from completed execution.

**Parameters:**
- `execution_id` (required): Execution ID to get results for

**Returns:**
```json
{
  "success": true,
  "execution_id": "exec_abc123",
  "status": "completed",
  "stdout": "Analysis output...",
  "stderr": "",
  "artifact_directory": "NSL2-geology-task/data/kazakhstan/feature-hypothesis/train_data/artifacts/<run_id>/<episode_id>/exec_abc123",
  "artifact_files": ["data.csv", "results.pkl"],
  "artifacts_count": 2,
  "execution_success": true,
  "runtime_s": 45.2
}
```

### execution.cancel
Cancel a running execution.

### execution.reset_session
Reset execution budget for a session (admin tool).

## Agent Workflow Pattern

### 1. Coding Agent Experience

```
# Old way (removed)
submit_code(code="...", expected_output="...")

# New way
execution_submit(code="...", timeout_s=600)
# -> {"execution_id": "exec_123", "attempts_used": "1/3"}

execution_status(execution_id="exec_123")  
# -> {"status": "running", "progress": "Loading data..."}

execution_status(execution_id="exec_123")
# -> {"status": "completed", "artifacts_count": 3}

execution_results(execution_id="exec_123")
# -> {"artifacts": [...], "success": true}

execution_finalize(execution_id="exec_123", success=True, summary="Created correlation analysis")
```

### 2. Budget Control

- Each agent session gets **3 execution attempts**
- Status checks are **unlimited** (agents can monitor freely)
- When budget exhausted: `execution.submit` returns error
- Step fails → workflow restarts with new hypothesis
- Budget resets for new episodes

### 3. Retry Strategy

Agents are instructed to:
- **Analyze errors**: Different approaches for timeouts vs execution failures
- **Focus on artifacts**: If no outputs, next attempt should emphasize data creation  
- **Strategic budgeting**: Use attempts wisely, don't waste on trivial retries

## Implementation Details

### Phase Records Integration

Execution results are stored in `phase_records["code"]` with same format as old `submit_code`:

```python
{
    "code_executed": "original_python_code", 
    "result_summary": "stdout_output",
    "artifact_directory": "NSL2-geology-task/data/kazakhstan/feature-hypothesis/train_data/artifacts/<run_id>/<episode_id>/exec_123",
    "artifact_files": ["file1.csv", "file2.pkl"],
    "success": True,
    "execution_id": "exec_123",
    "summary": "Brief description"
}
```

### Artifact Validation

- `execution_finalize` validates that artifacts were created if execution succeeded
- Returns error if `success=True` but `artifacts_count=0`
- Prevents translate agent confusion from missing artifacts
- Artifacts should be rooted in run-owned repo-local directories, not shared global temp directories. `VFM_ARTIFACT_DIR` remains a manual override for standalone tool runs.

### Container Integration

- Execution tools run via voxel-features-mcp container
- Container execution writes artifacts to `/workspace/out`, then copies them back to the host `artifact_root` with Docker archive extraction.

## Migration Status

- ✅ **Phase 1**: MCP tools implemented and tested
- ✅ **Phase 2**: Workflow updated to use new capabilities  
- ✅ **Phase 3**: Phase records integration complete
- ✅ **Phase 4**: Real container execution integration
- ⏳ **Phase 5**: Agent testing and validation

## Configuration

Default settings in feature_hypothesis.py workflow:

```python
# Code step capabilities
capabilities=(
    "phase_get",
    "execution_submit",
    "execution_status", 
    "execution_results",
    "execution_cancel",
    "execution_finalize",
),
terminator_capabilities=("execution_finalize",),
```

## Error Handling

### Budget Exhaustion
```json
{
  "success": false,
  "error": "Execution budget exhausted (3/3 attempts used)",
  "attempts_used": "3/3"
}
```

### Missing Artifacts
```json
{
  "success": false, 
  "error": "Execution reported success but no artifacts were created"
}
```

### Timeout Detection
```json
{
  "status": "timeout",
  "progress": "Execution timed out after 300s"
}
```

## Future Enhancements

- **Real container integration**: Replace simulation with actual Docker execution
- **Progress streaming**: Real-time execution progress updates
- **Partial results**: Capture intermediate artifacts during long-running executions
- **Dynamic budgets**: Adjust attempt limits based on hypothesis complexity
- **Execution queuing**: Handle multiple concurrent executions
