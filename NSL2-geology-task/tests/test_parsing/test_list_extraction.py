import pytest

from src.parsing.list_extraction import ListStrategy, extract_list


def test_extract_list_json_keyed_strategy() -> None:
    result = extract_list(
        '```json\n{"strategies": ["alpha", "beta"]}\n```',
        strategy=ListStrategy.JSON_KEYED,
    )

    assert result.unwrap() == ["alpha", "beta"]


def test_extract_list_json_braced_keyed_strategy() -> None:
    result = extract_list(
        'Here you go:\n\n```json\n{"strategy": ["alpha"]}\n```\n\nThanks.',
        strategy=ListStrategy.JSON_BRACED_KEYED,
    )

    assert result.unwrap() == ["alpha"]


def test_extract_list_claude_json_or_markdown_strategy() -> None:
    result = extract_list(
        "Strategies:\n- clean logs\n- remove cache",
        strategy=ListStrategy.CLAUDE_JSON_OR_MARKDOWN,
    )

    assert result.unwrap() == ["clean logs", "remove cache"]


def test_extract_list_dream_ast_strategy() -> None:
    result = extract_list(
        "Some prose before ['alpha', 'beta'] and some prose after",
        strategy=ListStrategy.DREAM_AST,
    )

    assert result.unwrap() == ["alpha", "beta"]


def test_extract_list_json_braced_keyed_falls_back_to_markdown_list() -> None:
    result = extract_list(
        "```\n- alpha\n- beta\n```",
        strategy=ListStrategy.JSON_BRACED_KEYED,
    )

    assert result.unwrap() == ["alpha", "beta"]


def test_extract_list_json_keyed_rejects_markdown_list() -> None:
    result = extract_list(
        "- alpha\n- beta",
        strategy=ListStrategy.JSON_KEYED,
    )

    with pytest.raises(Exception):
        result.unwrap()


def test_backtick_markdown_list_extracts_only_first_fenced_block() -> None:
    # Two fenced blocks: only first block's items should be returned.
    text = "```\n- alpha\n- beta\n```\nmiddle prose\n```\n- gamma\n- delta\n```"
    result = extract_list(text, strategy=ListStrategy.JSON_BRACED_KEYED)

    assert result.unwrap() == ["alpha", "beta"]


def test_json_braced_with_braces_inside_string_values() -> None:
    # Braces inside JSON string values must not confuse brace counting.
    text = '```json\n{"strategies": ["item with { brace", "also } here"]}\n```'
    result = extract_list(text, strategy=ListStrategy.JSON_BRACED_KEYED)

    assert result.unwrap() == ["item with { brace", "also } here"]
