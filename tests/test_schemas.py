"""Tests for agent schema system."""

import pytest
from pydantic import BaseModel

from agent.errors import SchemaValidationError
from agent.schemas import Schema, extract_json, repair_json, schema_to_prompt


class Person(BaseModel):
    name: str
    age: int


class Address(BaseModel):
    street: str
    city: str
    zip_code: str


# ── Schema ───────────────────────────────────────────────────────


class TestSchema:
    def test_pydantic_schema(self):
        s = Schema(Person)
        assert s._is_pydantic is True

    def test_dict_schema(self):
        s = Schema({"type": "object", "properties": {"x": {"type": "string"}}})
        assert s._is_pydantic is False

    def test_json_schema_pydantic(self):
        s = Schema(Person)
        js = s.json_schema
        assert "properties" in js
        assert "name" in js["properties"]
        assert "age" in js["properties"]

    def test_json_schema_dict(self):
        raw = {"type": "object", "properties": {"x": {"type": "string"}}}
        s = Schema(raw)
        assert s.json_schema is raw

    def test_validate_pydantic_dict(self):
        s = Schema(Person)
        result = s.validate({"name": "Alice", "age": 30})
        assert isinstance(result, Person)
        assert result.name == "Alice"
        assert result.age == 30

    def test_validate_pydantic_json_string(self):
        s = Schema(Person)
        result = s.validate('{"name": "Bob", "age": 25}')
        assert result.name == "Bob"

    def test_validate_pydantic_invalid(self):
        s = Schema(Person)
        with pytest.raises(SchemaValidationError):
            s.validate({"name": "Alice"})  # Missing age

    def test_validate_pydantic_wrong_type(self):
        s = Schema(Person)
        with pytest.raises(SchemaValidationError):
            s.validate({"name": "Alice", "age": "not a number"})

    def test_validate_dict_schema_passthrough(self):
        s = Schema({"type": "object"})
        data = {"anything": "goes"}
        assert s.validate(data) is data

    def test_parse_json_plain(self):
        s = Schema(Person)
        result = s.parse_json('{"name": "Charlie", "age": 35}')
        assert result.name == "Charlie"

    def test_parse_json_code_block(self):
        s = Schema(Person)
        text = '```json\n{"name": "Diana", "age": 28}\n```'
        result = s.parse_json(text)
        assert result.name == "Diana"

    def test_parse_json_code_block_no_lang(self):
        s = Schema(Person)
        text = '```\n{"name": "Eve", "age": 22}\n```'
        result = s.parse_json(text)
        assert result.name == "Eve"

    def test_parse_json_invalid(self):
        s = Schema(Person)
        with pytest.raises(SchemaValidationError):
            s.parse_json("not json at all")

    def test_strict_default(self):
        s = Schema(Person)
        assert s.strict is True

    def test_repair_attempts_default(self):
        s = Schema(Person)
        assert s.repair_attempts == 1


# ── schema_to_prompt ─────────────────────────────────────────────


class TestSchemaToPrompt:
    def test_contains_schema(self):
        s = Schema(Person)
        prompt = schema_to_prompt(s)
        assert "JSON" in prompt
        assert "name" in prompt
        assert "age" in prompt

    def test_contains_instruction(self):
        s = Schema(Person)
        prompt = schema_to_prompt(s)
        assert "IMPORTANT" in prompt

    def test_dict_schema(self):
        s = Schema({"type": "object", "properties": {"x": {"type": "string"}}})
        prompt = schema_to_prompt(s)
        assert "JSON" in prompt


# ── extract_json ─────────────────────────────────────────────────


class TestExtractJson:
    def test_plain_json(self):
        result = extract_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_json_with_whitespace(self):
        result = extract_json('  {"key": "value"}  ')
        assert result == {"key": "value"}

    def test_json_in_code_block(self):
        text = 'Here is the result:\n```json\n{"key": "value"}\n```\nDone.'
        result = extract_json(text)
        assert result == {"key": "value"}

    def test_json_in_code_block_no_lang(self):
        text = '```\n{"key": "value"}\n```'
        result = extract_json(text)
        assert result == {"key": "value"}

    def test_json_embedded_in_text(self):
        text = 'The result is {"key": "value"} as expected.'
        result = extract_json(text)
        assert result == {"key": "value"}

    def test_nested_json(self):
        result = extract_json('{"a": {"b": [1, 2, 3]}}')
        assert result == {"a": {"b": [1, 2, 3]}}

    def test_no_json_raises(self):
        with pytest.raises(ValueError, match="Could not extract JSON"):
            extract_json("no json here at all")

    def test_array_json(self):
        result = extract_json("[1, 2, 3]")
        assert result == [1, 2, 3]

    def test_multiple_code_blocks_first_valid(self):
        text = '```\nnot json\n```\n```json\n{"ok": true}\n```'
        result = extract_json(text)
        assert result == {"ok": True}


# ── repair_json ──────────────────────────────────────────────────


class TestRepairJson:
    def test_trailing_comma_object(self):
        text = '{"a": 1, "b": 2,}'
        repaired = repair_json(text, ValueError("test"))
        import json

        result = json.loads(repaired)
        assert result == {"a": 1, "b": 2}

    def test_trailing_comma_array(self):
        text = "[1, 2, 3,]"
        repaired = repair_json(text, ValueError("test"))
        import json

        result = json.loads(repaired)
        assert result == [1, 2, 3]

    def test_missing_closing_brace(self):
        text = '{"a": 1'
        repaired = repair_json(text, ValueError("test"))
        assert repaired.count("{") == repaired.count("}")

    def test_missing_closing_bracket(self):
        text = "[1, 2, 3"
        repaired = repair_json(text, ValueError("test"))
        assert repaired.count("[") == repaired.count("]")

    def test_multiple_missing_braces(self):
        text = '{"a": {"b": 1'
        repaired = repair_json(text, ValueError("test"))
        assert repaired.count("{") == repaired.count("}")

    def test_balanced_unchanged(self):
        text = '{"a": 1}'
        repaired = repair_json(text, ValueError("test"))
        import json

        assert json.loads(repaired) == {"a": 1}

    def test_combined_issues(self):
        text = '{"a": [1, 2,'
        repaired = repair_json(text, ValueError("test"))
        assert repaired.count("[") == repaired.count("]")
        assert repaired.count("{") == repaired.count("}")
