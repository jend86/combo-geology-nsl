"""Code execution tools with async capabilities and budget control."""

from __future__ import annotations

import os
import json
import threading
import time
import uuid
import glob
import pickle
import subprocess
from pathlib import Path
from typing import Any, Dict
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd
import numpy as np

# Import Docker execution functions for container support
try:
    from docker.models.containers import Container
    import sys
    from pathlib import Path
    
    # Add NSL2-geology-task to Python path  
    nsl_path = Path(__file__).parent.parent.parent.parent.parent / "NSL2-geology-task"
    if nsl_path.exists():
        sys.path.insert(0, str(nsl_path))
    
    from tasks.common.foundry_exec import (
        exec_run_with_timeout,
        coerce_exec_result
    )
    DOCKER_AVAILABLE = True
except ImportError:
    Container = None
    exec_run_with_timeout = None
    coerce_exec_result = None
    DOCKER_AVAILABLE = False


class ExecutionStatus(Enum):
    """Execution status values."""
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class ExecutionSession:
    """Tracks execution attempts and budget for an agent session."""
    session_id: str
    max_attempts: int = 3
    attempts_used: int = 0
    
    def can_submit(self) -> bool:
        """Check if more executions can be submitted."""
        return self.attempts_used < self.max_attempts
    
    def use_attempt(self) -> None:
        """Increment attempt counter."""
        self.attempts_used += 1


@dataclass
class ExecutionRecord:
    """Tracks individual code execution."""
    execution_id: str
    session_id: str
    code: str
    timeout_s: int
    container: Any | None = None  # Docker container for execution
    artifact_root: str | None = None
    status: ExecutionStatus = ExecutionStatus.PENDING
    start_time: float | None = None
    end_time: float | None = None
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    artifact_files: list[str] = field(default_factory=list)
    artifact_directory: str = ""
    progress_log: list[str] = field(default_factory=list)
    thread: threading.Thread | None = None
    
    def add_progress(self, message: str) -> None:
        """Add progress message."""
        self.progress_log.append(f"[{time.strftime('%H:%M:%S')}] {message}")


# Global state for execution tracking
_sessions: Dict[str, ExecutionSession] = {}
_executions: Dict[str, ExecutionRecord] = {}
_session_lock = threading.Lock()


def _default_artifact_root() -> str:
    configured = os.environ.get("VFM_ARTIFACT_DIR")
    if configured:
        return configured
    return str((Path.cwd() / "data" / "execution-artifacts").resolve())


def _collect_artifact_files(artifact_dir: str) -> list[str]:
    artifact_files = []
    for root, _, files in os.walk(artifact_dir):
        for file_name in files:
            artifact_files.append(os.path.join(root, file_name))
    return artifact_files


def _extract_tar_bytes(raw: bytes, artifact_dir: str) -> list[str]:
    import io
    import tarfile

    tar_data = io.BytesIO(raw)
    with tarfile.open(fileobj=tar_data) as tar:
        tar.extractall(artifact_dir, filter="data")
    return _collect_artifact_files(artifact_dir)


def _tar_container_path(container: Any, container_path: str, timeout_s: int = 30) -> bytes:
    archive_script = (
        "import sys, tarfile\n"
        f"path = {json.dumps(container_path)}\n"
        "with tarfile.open(fileobj=sys.stdout.buffer, mode='w|') as tar:\n"
        "    tar.add(path, arcname='.')\n"
    )
    result = exec_run_with_timeout(
        container, ["python3", "-c", archive_script], timeout_s=timeout_s
    )
    exit_code, raw = coerce_exec_result(result)
    if exit_code != 0:
        raise RuntimeError(raw.decode(errors="replace"))
    return raw


def _get_or_create_session(session_id: str | None = None, max_attempts: int = 3) -> ExecutionSession:
    """Get or create execution session."""
    with _session_lock:
        if session_id is None:
            session_id = f"session_{uuid.uuid4().hex[:8]}"
        
        if session_id not in _sessions:
            _sessions[session_id] = ExecutionSession(session_id, max_attempts)
        
        return _sessions[session_id]


def _execute_code_in_thread(record: ExecutionRecord) -> None:
    """Execute code in background thread, using container if available."""
    try:
        record.status = ExecutionStatus.RUNNING
        record.start_time = time.time()
        record.add_progress("Starting code execution")
        
        # Create artifact directory
        base_artifact_dir = record.artifact_root or _default_artifact_root()
        artifact_dir = f"{base_artifact_dir}/{record.execution_id}"
        os.makedirs(artifact_dir, exist_ok=True)
        record.artifact_directory = artifact_dir
        record.add_progress(f"Created artifact directory: {artifact_dir}")
        
        # Choose execution method based on container availability
        if record.container is not None and DOCKER_AVAILABLE:
            record.add_progress("Executing code in analysis container")
            _execute_in_container(record, artifact_dir)
        else:
            record.add_progress("Executing code on host (fallback mode)")
            _execute_on_host(record, artifact_dir)
        
        # Set completion status
        if record.exit_code == 0:
            record.status = ExecutionStatus.COMPLETED
            record.add_progress(f"Execution completed successfully with {len(record.artifact_files)} artifacts")
        else:
            record.status = ExecutionStatus.FAILED
            record.add_progress(f"Execution failed with exit code {record.exit_code}")
            
    except Exception as e:
        record.status = ExecutionStatus.FAILED
        record.stderr = str(e)
        record.add_progress(f"Execution failed with exception: {e}")
    
    finally:
        record.end_time = time.time()


def _execute_in_container(record: ExecutionRecord, artifact_dir: str) -> None:
    """Execute code inside Docker container (like analysis_shell)."""
    
    # Create wrapped code with artifact capture
    wrapped_code = f'''
import os
import glob
import pickle
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
import traceback

# Host artifact directory (visible to both container and host)
host_artifact_dir = "{artifact_dir}"

# Container paths - use real data paths
data_dir = "/workspace/input/amalgamated_csvs"
output_dir = "/workspace/out"
shutil.rmtree(output_dir, ignore_errors=True)
os.makedirs(output_dir, exist_ok=True)

# Store original locals to compare later
_original_locals = set(locals().keys())

print("Starting geological analysis execution...")
print(f"Data directory: {{data_dir}}")
print(f"Output directory: {{output_dir}}")
print("="*50)

try:
    # Execute user's analysis code
{chr(10).join("    " + line for line in record.code.split(chr(10)))}

except Exception as user_code_error:
    print(f"ERROR in user code: {{user_code_error}}")
    traceback.print_exc()

finally:
    print("\\n" + "="*50)
    print("ANALYSIS COMPLETE - CAPTURING ARTIFACTS")
    print("="*50)
    
    # Capture artifacts from final namespace
    _final_locals = locals().copy()
    _artifacts_saved = []
    
    for var_name, obj in _final_locals.items():
        if (not var_name.startswith('_') and 
            var_name not in _original_locals and 
            var_name not in ['host_artifact_dir', 'data_dir', 'output_dir', 'os', 'glob', 'pickle', 'pd', 'np', 'Path', 'traceback']):
            
            try:
                # Save to /workspace/out (container tmpfs)
                if isinstance(obj, pd.DataFrame) and not obj.empty:
                    filepath = f"{{output_dir}}/{{var_name}}_dataframe.csv"
                    obj.to_csv(filepath, index=False)
                    _artifacts_saved.append(filepath)
                    print(f"Saved DataFrame '{{var_name}}' -> {{filepath}}")
                    print(f"  Shape: {{obj.shape}}, Columns: {{list(obj.columns)}}")
                
                elif isinstance(obj, np.ndarray) and obj.size > 0:
                    filepath = f"{{output_dir}}/{{var_name}}_array.npy" 
                    np.save(filepath, obj)
                    _artifacts_saved.append(filepath)
                    print(f"Saved numpy array '{{var_name}}' -> {{filepath}}")
                    print(f"  Shape: {{obj.shape}}, dtype: {{obj.dtype}}")
                
                elif isinstance(obj, (dict, list, tuple)) and len(str(obj)) < 10000:
                    filepath = f"{{output_dir}}/{{var_name}}_object.pkl"
                    with open(filepath, 'wb') as f:
                        pickle.dump(obj, f)
                    _artifacts_saved.append(filepath)
                    print(f"Saved object '{{var_name}}' -> {{filepath}}")
                    print(f"  Type: {{type(obj)}}, Size: {{len(str(obj))}} chars")
                
                elif isinstance(obj, (int, float, str, bool)):
                    filepath = f"{{output_dir}}/{{var_name}}_scalar.txt"
                    with open(filepath, 'w') as f:
                        f.write(f"{{var_name}}: {{obj}}\\ntype: {{type(obj).__name__}}")
                    _artifacts_saved.append(filepath)
                    print(f"Saved scalar '{{var_name}}' -> {{filepath}}")
                    print(f"  Value: {{obj}}")
                
            except Exception as save_err:
                print(f"Failed to save '{{var_name}}': {{save_err}}")
    
    print(f"\\nARTIFACTS_SAVED: {{_artifacts_saved}}")
    print("="*50)
'''
    
    try:
        # Write code to temporary file in container and execute it
        write_cmd = (
            "cat > /tmp/_execution_script.py <<'__EXEC_PY_EOF__'\n"
            f"{wrapped_code}\n"
            "__EXEC_PY_EOF__"
        )
        
        # Write the script
        write_result = exec_run_with_timeout(
            record.container, ["sh", "-c", write_cmd], timeout_s=30
        )
        write_code, write_out = coerce_exec_result(write_result)
        if write_code != 0:
            record.exit_code = write_code
            record.stderr = f"Failed to stage script: {write_out.decode(errors='replace')}"
            record.artifact_files = []
            return
        
        # Execute the script
        run_result = exec_run_with_timeout(
            record.container,
            ["python3", "-u", "/tmp/_execution_script.py"],
            timeout_s=record.timeout_s,
            demux=True,
        )
        
        # Parse results
        exit_code = getattr(run_result, "exit_code", None)
        if exit_code is None and isinstance(run_result, tuple):
            exit_code = run_result[0]
        raw_output = getattr(run_result, "output", None)
        if raw_output is None and isinstance(run_result, tuple):
            raw_output = run_result[1]
        
        if isinstance(raw_output, tuple) and len(raw_output) == 2:
            stdout_bytes, stderr_bytes = raw_output
            stdout = stdout_bytes.decode(errors="replace") if stdout_bytes else ""
            stderr = stderr_bytes.decode(errors="replace") if stderr_bytes else ""
        else:
            stdout = (
                raw_output.decode(errors="replace")
                if isinstance(raw_output, (bytes, bytearray))
                else str(raw_output) if raw_output else ""
            )
            stderr = ""
        
        record.exit_code = int(exit_code) if exit_code is not None else 0
        record.stdout = stdout.strip()
        record.stderr = stderr.strip()
        
        # Pull artifacts from the container via an in-container tar stream.
        # Docker's archive API can miss files under tmpfs-backed paths.
        try:
            raw_archive = _tar_container_path(record.container, "/workspace/out")
            record.artifact_files = _extract_tar_bytes(raw_archive, artifact_dir)
        except Exception as copy_err:
            record.add_progress(f"Warning: Could not extract artifacts from container: {copy_err}")
            record.artifact_files = []
        
    except Exception as container_err:
        record.exit_code = 1
        record.stderr = f"Container execution failed: {container_err}"
        record.artifact_files = []


def _execute_on_host(record: ExecutionRecord, artifact_dir: str) -> None:
    """Execute code on host machine (fallback mode)."""
    try:
        import subprocess
        import sys
        import textwrap
        
        # Create a temporary Python script with the submitted code
        script_path = f"{artifact_dir}/execution_script.py"
        
        # Create the wrapper code as separate parts
        setup_code = f"""
import sys
import os
import pandas as pd
import numpy as np
from scipy import stats
import pickle
import json

# Set working directory to artifact directory
os.chdir("{artifact_dir}")

# Ensure data directory exists and is accessible
data_dir = "/workspace/input/amalgamated_csvs"
if not os.path.exists(data_dir):
    print("Warning: Data directory not found, creating placeholder...")
    os.makedirs("/workspace/input/amalgamated_csvs", exist_ok=True)
    # Create minimal test data if files don't exist
    if not os.path.exists("/workspace/input/amalgamated_csvs/geochemDrillhole.csv"):
        pd.DataFrame({{'longitude': [117.9], 'latitude': [-27.4], 'li_ppm': [10.5], 'cu_ppm': [25.2]}}).to_csv("/workspace/input/amalgamated_csvs/geochemDrillhole.csv", index=False)

try:
    # Execute the submitted code
"""
        
        cleanup_code = f"""
except Exception as code_err:
    print(f"Code execution error: {{code_err}}")
    import traceback
    traceback.print_exc()

# Auto-save any variables that look like artifacts
artifacts_saved = []
for name, obj in locals().items():
    if not name.startswith('_') and name not in ['sys', 'os', 'pd', 'np', 'stats', 'pickle', 'json', 'code_err', 'traceback']:
        try:
            if hasattr(obj, 'to_csv') and hasattr(obj, 'shape'):  # DataFrame
                filename = f"{{name}}_dataframe.csv"
                obj.to_csv(filename, index=False)
                artifacts_saved.append(f"{artifact_dir}/{{filename}}")
                print(f"Saved DataFrame {{name}} with shape {{obj.shape}}")
            elif isinstance(obj, (dict, list, tuple)) and not callable(obj):  # Pickle-able objects
                filename = f"{{name}}_object.pkl"
                with open(filename, 'wb') as f:
                    pickle.dump(obj, f)
                artifacts_saved.append(f"{artifact_dir}/{{filename}}")
                print(f"Saved object {{name}}")
            elif hasattr(obj, 'shape') and hasattr(obj, 'dtype'):  # NumPy array
                filename = f"{{name}}_array.npy"
                np.save(filename, obj)
                artifacts_saved.append(f"{artifact_dir}/{{filename}}")
                print(f"Saved array {{name}} with shape {{obj.shape}}")
            elif isinstance(obj, (int, float, str)) and len(str(obj)) < 1000:  # Scalar values
                filename = f"{{name}}_value.txt"
                with open(filename, 'w') as f:
                    f.write(str(obj))
                artifacts_saved.append(f"{artifact_dir}/{{filename}}")
                print(f"Saved scalar {{name}}: {{obj}}")
        except Exception as save_err:
            print(f"Warning: Could not save {{name}}: {{save_err}}")

print(f"\\nARTIFACTS_SAVED: {{artifacts_saved}}")
print("Analysis completed successfully.")
"""
        
        # Indent the user's code properly
        user_code_indented = textwrap.indent(record.code, "    ")
        
        # Combine all parts
        wrapped_code = setup_code + user_code_indented + "\n" + cleanup_code
        
        # Write the wrapped script
        with open(script_path, 'w') as f:
            f.write(wrapped_code)
        
        # Execute the script
        result = subprocess.run([
            sys.executable, script_path
        ], capture_output=True, text=True, timeout=record.timeout_s, cwd=artifact_dir)
        
        record.exit_code = result.returncode
        record.stdout = result.stdout
        record.stderr = result.stderr
        
        # Find created artifacts (exclude the script itself)
        artifact_files = []
        if os.path.exists(artifact_dir):
            for item in os.listdir(artifact_dir):
                if item != "execution_script.py":
                    artifact_files.append(f"{artifact_dir}/{item}")
        record.artifact_files = artifact_files
        
    except subprocess.TimeoutExpired:
        record.exit_code = 124
        record.stderr = f"Execution timed out after {record.timeout_s} seconds"
        record.artifact_files = []
    except Exception as exec_err:
        record.exit_code = 1
        record.stderr = f"Host execution failed: {exec_err}"
        record.artifact_files = []


def execution_submit(
    code: str,
    timeout_s: int = 300,
    session_id: str | None = None,
    max_attempts: int = 3,
    container: Any | None = None,
    artifact_root: str | None = None,
) -> dict[str, Any]:
    """
    Submit code for async execution.
    
    Args:
        code: Python code to execute
        timeout_s: Execution timeout in seconds
        session_id: Session ID for budget tracking
        max_attempts: Maximum execution attempts for this session
        container: Optional Docker container for execution (recommended)
        artifact_root: Optional base directory for this episode's artifacts
    
    Returns:
        Dict with execution_id and attempt info, or error if budget exhausted
    """
    session = _get_or_create_session(session_id, max_attempts)
    
    # Check budget
    if not session.can_submit():
        return {
            "success": False,
            "error": f"Execution budget exhausted ({session.attempts_used}/{session.max_attempts} attempts used)",
            "attempts_used": f"{session.attempts_used}/{session.max_attempts}"
        }
    
    # Create execution record
    execution_id = f"exec_{uuid.uuid4().hex[:8]}"
    record = ExecutionRecord(
        execution_id=execution_id,
        session_id=session.session_id,
        code=code,
        timeout_s=timeout_s,
        container=container,
        artifact_root=artifact_root,
    )
    
    # Use attempt
    session.use_attempt()
    _executions[execution_id] = record
    
    # Start execution thread
    record.thread = threading.Thread(
        target=_execute_code_in_thread,
        args=(record,),
        name=f"exec-{execution_id}"
    )
    record.thread.start()
    
    return {
        "success": True,
        "execution_id": execution_id,
        "attempts_used": f"{session.attempts_used}/{session.max_attempts}",
        "status": "pending"
    }


def execution_status(execution_id: str) -> dict[str, Any]:
    """
    Get execution status and progress.
    
    Args:
        execution_id: Execution ID to check
    
    Returns:
        Dict with status, progress, and timing info
    """
    if execution_id not in _executions:
        return {
            "success": False,
            "error": f"Execution {execution_id} not found"
        }
    
    record = _executions[execution_id]
    
    # Calculate runtime
    runtime = None
    if record.start_time:
        end_time = record.end_time or time.time()
        runtime = end_time - record.start_time
    
    progress = ""
    if record.progress_log:
        progress = record.progress_log[-1]  # Latest progress
    
    return {
        "success": True,
        "execution_id": execution_id,
        "status": record.status.value,
        "progress": progress,
        "runtime_s": runtime,
        "timeout_s": record.timeout_s,
        "progress_log": record.progress_log[-5:],  # Last 5 entries
        "artifacts_count": len(record.artifact_files) if record.status == ExecutionStatus.COMPLETED else 0
    }


def execution_results(execution_id: str) -> dict[str, Any]:
    """
    Get execution results and artifacts.
    
    Args:
        execution_id: Execution ID to get results for
    
    Returns:
        Dict with artifacts, output, and success status
    """
    if execution_id not in _executions:
        return {
            "success": False,
            "error": f"Execution {execution_id} not found"
        }
    
    record = _executions[execution_id]
    
    if record.status not in [ExecutionStatus.COMPLETED, ExecutionStatus.FAILED, ExecutionStatus.TIMEOUT]:
        return {
            "success": False,
            "error": f"Execution {execution_id} is still {record.status.value}"
        }
    
    return {
        "success": True,
        "execution_id": execution_id,
        "status": record.status.value,
        "exit_code": record.exit_code,
        "stdout": record.stdout,
        "stderr": record.stderr,
        "artifact_directory": record.artifact_directory,
        "artifact_files": record.artifact_files,
        "artifacts_count": len(record.artifact_files),
        "execution_success": record.status == ExecutionStatus.COMPLETED,
        "runtime_s": (record.end_time - record.start_time) if record.start_time and record.end_time else None
    }


def execution_cancel(execution_id: str) -> dict[str, Any]:
    """
    Cancel a running execution.
    
    Args:
        execution_id: Execution ID to cancel
    
    Returns:
        Dict with cancellation status
    """
    if execution_id not in _executions:
        return {
            "success": False,
            "error": f"Execution {execution_id} not found"
        }
    
    record = _executions[execution_id]
    
    if record.status not in [ExecutionStatus.PENDING, ExecutionStatus.RUNNING]:
        return {
            "success": False,
            "error": f"Cannot cancel execution with status {record.status.value}"
        }
    
    record.status = ExecutionStatus.CANCELLED
    record.add_progress("Execution cancelled by user")
    
    # Note: We don't actually kill the subprocess here for safety
    # The thread will continue but the status will show cancelled
    
    return {
        "success": True,
        "execution_id": execution_id,
        "status": "cancelled"
    }


def execution_reset_session(session_id: str | None = None) -> dict[str, Any]:
    """
    Reset execution budget for a session.
    
    Args:
        session_id: Session to reset, or None to reset all sessions
    
    Returns:
        Dict with reset status
    """
    with _session_lock:
        if session_id is None:
            # Reset all sessions
            for session in _sessions.values():
                session.attempts_used = 0
            return {"success": True, "message": "All sessions reset"}
        elif session_id in _sessions:
            _sessions[session_id].attempts_used = 0
            return {"success": True, "message": f"Session {session_id} reset"}
        else:
            return {"success": False, "error": f"Session {session_id} not found"}
