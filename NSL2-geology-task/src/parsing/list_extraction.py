from __future__ import annotations

import ast
import json
import re
from collections.abc import Callable, Sequence
from enum import Enum

from result import Err, Ok, Result


DEFAULT_STRATEGY_KEYS: tuple[str, ...] = (
    "strategies",
    "strats",
    "strategy",
    "Strategies",
    "Strats",
)


class ListStrategy(Enum):
    JSON_KEYED = "json_keyed"
    JSON_BRACED_KEYED = "json_braced_keyed"
    CLAUDE_JSON_OR_MARKDOWN = "claude_json_or_markdown"
    DREAM_AST = "dream_ast"


def extract_list(
    text: str,
    strategy: ListStrategy = ListStrategy.JSON_KEYED,
    expected_keys: Sequence[str] = DEFAULT_STRATEGY_KEYS,
) -> Result[list[str], str]:
    handler = _STRATEGY_HANDLERS[strategy]
    return handler(text, expected_keys)


def _extract_json_keyed(
    text: str,
    expected_keys: Sequence[str],
) -> Result[list[str], str]:
    try:
        return Ok(_parse_keyed_json(_strip_code_fences(text), expected_keys))
    except Exception as exc:
        return Err(
            f"extract_list(JSON_KEYED): Failed to parse keyed JSON, error: {exc}"
        )


def _extract_json_braced_keyed(
    text: str,
    expected_keys: Sequence[str],
) -> Result[list[str], str]:
    try:
        return Ok(_parse_keyed_json(_extract_first_braced_json(text), expected_keys))
    except Exception as json_exc:
        markdown_result = _extract_backtick_markdown_list(text)
        match markdown_result:
            case Ok(_):
                return markdown_result
            case Err(markdown_error):
                return Err(
                    "extract_list(JSON_BRACED_KEYED): Failed JSON parse and markdown fallback, "
                    f"json_error: {json_exc}, markdown_error: {markdown_error}"
                )


def _extract_claude_json_or_markdown(
    text: str,
    expected_keys: Sequence[str],
) -> Result[list[str], str]:
    try:
        return Ok(_parse_keyed_json(_extract_first_braced_json(text), expected_keys))
    except Exception as json_exc:
        markdown_items = _extract_markdown_items(text)
        if markdown_items:
            return Ok(markdown_items)
        return Err(
            "extract_list(CLAUDE_JSON_OR_MARKDOWN): Failed JSON parse and markdown fallback, "
            f"json_error: {json_exc}"
        )


def _extract_dream_ast(
    text: str,
    _expected_keys: Sequence[str],
) -> Result[list[str], str]:
    try:
        start = text.index("[")
        end = text.rindex("]") + 1
        parsed = ast.literal_eval(text[start:end])
        return Ok(_validate_string_list(parsed))
    except Exception as exc:
        return Err(
            f"extract_list(DREAM_AST): Failed to parse Python list, error: {exc}"
        )


def _parse_keyed_json(text: str, expected_keys: Sequence[str]) -> list[str]:
    parsed = json.loads(text)
    for key in expected_keys:
        if key in parsed:
            return _validate_string_list(parsed[key])
    raise ValueError(f"No matching strategy keys found in parsed JSON: {expected_keys}")


def _strip_code_fences(text: str) -> str:
    return text.replace("```json", "").replace("```", "").strip()


def _extract_first_braced_json(text: str) -> str:
    json_start = text.find("{")
    if json_start == -1:
        raise ValueError("No JSON object found")

    brace_count = 0
    in_string = False
    escape = False
    for index in range(json_start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            brace_count += 1
        elif char == "}":
            brace_count -= 1
            if brace_count == 0:
                return text[json_start : index + 1]

    raise ValueError("Unmatched braces in JSON")


def _extract_backtick_markdown_list(text: str) -> Result[list[str], str]:
    match = re.search(r"```(.*?)```", text, re.DOTALL)
    if match is None:
        return Err("No fenced markdown list found")

    items = _extract_markdown_items(match.group(1))
    if items:
        return Ok(items)

    return Err("No markdown list items found inside fenced block")


def _extract_markdown_items(text: str) -> list[str]:
    items: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("- ") or line.startswith("* "):
            items.append(line[2:].strip())
            continue

        numbered_match = re.match(r"^\d+\.\s*(.+)$", line)
        if numbered_match is not None:
            items.append(numbered_match.group(1).strip())

    return [item for item in items if item]


def _validate_string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        raise TypeError("Parsed value is not a list")
    if not all(isinstance(item, str) for item in value):
        raise TypeError("All parsed list items must be strings")
    return list(value)


_STRATEGY_HANDLERS: dict[
    ListStrategy,
    Callable[[str, Sequence[str]], Result[list[str], str]],
] = {
    ListStrategy.JSON_KEYED: _extract_json_keyed,
    ListStrategy.JSON_BRACED_KEYED: _extract_json_braced_keyed,
    ListStrategy.CLAUDE_JSON_OR_MARKDOWN: _extract_claude_json_or_markdown,
    ListStrategy.DREAM_AST: _extract_dream_ast,
}
