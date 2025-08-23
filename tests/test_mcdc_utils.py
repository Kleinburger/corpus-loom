import pytest
from ollama_client_plus.utils import extract_json_str

def test_extract_json_with_fence_and_valid():
    text = "```json\n{\"a\": 1, \"b\": [2,3]}\n```"
    assert extract_json_str(text) == '{"a": 1, "b": [2, 3]}' or extract_json_str(text) == '{"a": 1, "b": [2,3]}'

def test_extract_json_no_fence_valid_array():
    text = "Here: [1, 2, 3] trailing"
    assert extract_json_str(text) == "[1, 2, 3]" or extract_json_str(text) == "[1,2,3]"

def test_extract_json_mismatch_braces_returns_none():
    # exercises branch: mismatched closing bracket
    assert extract_json_str("{]") is None

def test_extract_json_no_start_brace_returns_none():
    # no JSON-looking content
    assert extract_json_str("no json here") is None

def test_extract_json_candidate_invalid_returns_none():
    # invalid json between braces
    assert extract_json_str("{ invalid }") is None
