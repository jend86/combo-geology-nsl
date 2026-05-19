import ast
import subprocess
import sys

import pytest

from src.harness.orchestrator_modes import _wrap_shell_as_python as wrap_shell_as_python
from src.parsing.code_extraction import CodeBlock, WhenNoMatch, extract_code_block


@pytest.mark.parametrize(
    ("text", "accepted_langs", "fallback", "expected"),
    [
        (
            "```python\nprint(1)\n```",
            ["python"],
            WhenNoMatch.NONE,
            CodeBlock("python", "print(1)"),
        ),
        ("```bash\nls\n```", ["python"], WhenNoMatch.NONE, None),
        (
            "```bash\nls\n```",
            ["python", "bash"],
            WhenNoMatch.NONE,
            CodeBlock("bash", "ls"),
        ),
        (
            "```shell\necho hi\n```",
            ["python", "shell"],
            WhenNoMatch.NONE,
            CodeBlock("shell", "echo hi"),
        ),
        (
            "```sh\ncat f\n```",
            ["python", "sh"],
            WhenNoMatch.NONE,
            CodeBlock("sh", "cat f"),
        ),
        ("```\nprint(1)\n```", ["python"], WhenNoMatch.NONE, None),
        (
            "```\nprint(1)\n```",
            ["python", ""],
            WhenNoMatch.NONE,
            CodeBlock("", "print(1)"),
        ),
        ("```ruby\nputs 1\n```", ["python"], WhenNoMatch.NONE, None),
        (
            '```json\n{"k": 1}\n```\n```python\nprint(1)\n```',
            ["python"],
            WhenNoMatch.NONE,
            CodeBlock("python", "print(1)"),
        ),
        ("No fences at all", ["python"], WhenNoMatch.NONE, None),
        (
            "No fences at all",
            ["python"],
            WhenNoMatch.RETURN_RAW,
            CodeBlock("", "No fences at all"),
        ),
        (
            "No fences at all",
            ["python"],
            WhenNoMatch.PREFLIGHT,
            CodeBlock("", "No fences at all"),
        ),
        ("```json\n{}\n```", ["python"], WhenNoMatch.PREFLIGHT, None),
        ("```python\n\n```", ["python"], WhenNoMatch.NONE, None),
    ],
)
def test_extract_code_block_matrix(
    text: str,
    accepted_langs: list[str],
    fallback: WhenNoMatch,
    expected: CodeBlock | None,
) -> None:
    assert extract_code_block(text, accepted_langs, fallback=fallback) == expected


def test_wrap_shell_as_python_produces_parseable_python() -> None:
    wrapped = wrap_shell_as_python("printf 'hello'\n")
    ast.parse(wrapped)


def test_wrap_shell_as_python_captures_stdout_and_stderr() -> None:
    wrapped = wrap_shell_as_python("printf 'out\\n'; printf 'warn\\n' >&2")

    result = subprocess.run(
        [sys.executable, "-c", wrapped],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout == "out\n"
    assert result.stderr == "warn\n"


def test_wrap_shell_as_python_propagates_non_zero_exit_code() -> None:
    wrapped = wrap_shell_as_python("printf 'boom\\n' >&2; exit 7")

    result = subprocess.run(
        [sys.executable, "-c", wrapped],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 7
    assert result.stdout == ""
    assert result.stderr == "boom\n"
