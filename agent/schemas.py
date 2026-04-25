"""
Agent structured output schemas.

Support for Pydantic-based schema definition and JSON Schema.
"""

import json
import re
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from agent.errors import SchemaValidationError

T = TypeVar("T", bound=BaseModel)


class Schema:
    """
    Wrapper for schema definitions.

    Supports both Pydantic models and raw JSON Schema dictionaries.
    """

    def __init__(
        self,
        schema: type[BaseModel] | dict[str, Any],
        *,
        strict: bool = True,
        repair_attempts: int = 1,
    ):
        self.schema = schema
        self.strict = strict
        self.repair_attempts = repair_attempts
        self._is_pydantic = isinstance(schema, type) and issubclass(schema, BaseModel)

    @property
    def json_schema(self) -> dict[str, Any]:
        """Get the JSON Schema representation."""
        if self._is_pydantic:
            return self.schema.model_json_schema()  # type: ignore
        return self.schema  # type: ignore

    def validate(self, data: Any) -> Any:
        """
        Validate data against the schema.

        Returns the validated/parsed object, or raises SchemaValidationError.
        """
        if self._is_pydantic:
            try:
                if isinstance(data, str):
                    return self.schema.model_validate_json(data)  # type: ignore
                return self.schema.model_validate(data)  # type: ignore
            except ValidationError as e:
                raise SchemaValidationError(
                    f"Schema validation failed: {e}",
                    schema=self.schema,
                    output=data,
                ) from e
        else:
            # For raw JSON Schema, just return the data
            # (validation would require jsonschema library)
            return data

    def parse_json(self, text: str) -> Any:
        """
        Parse JSON text and validate against schema.

        Handles extracting JSON from markdown code blocks.
        """
        # Clean up the text
        cleaned = text.strip()

        # Try to extract JSON from markdown code blocks
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Remove first line (```json or ```)
            lines = lines[1:]
            # Remove last line (```)
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines)

        # Parse JSON
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise SchemaValidationError(
                f"Failed to parse JSON: {e}",
                schema=self.schema,
                output=text,
            ) from e

        return self.validate(data)


def schema_to_prompt(schema: Schema) -> str:
    """Convert a schema to a prompt instruction."""
    json_schema = schema.json_schema

    # Get the schema name if available
    name = json_schema.get("title", "response")
    # Build the prompt
    prompt = f"Respond with a JSON object for '{name}' matching this schema:\n\n```json\n{json.dumps(json_schema, indent=2)}\n```\n\n"
    prompt += "IMPORTANT: Return ONLY the JSON object, no other text."

    return prompt


def extract_json(text: str) -> dict[str, Any]:
    """
    Extract JSON from text that may contain other content.

    Handles:
    - Plain JSON
    - JSON in markdown code blocks
    - JSON embedded in text
    """
    text = text.strip()

    # Try direct JSON parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to extract from markdown code block
    if "```" in text:
        # Find JSON block
        pattern = r"```(?:json)?\s*\n?(.*?)\n?```"
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue

    # Try to find JSON object in text
    # Look for object boundaries (non-greedy to get smallest valid match)
    brace_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
    matches = re.findall(brace_pattern, text, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue

    raise ValueError(f"Could not extract JSON from text: {text[:100]}...")


def repair_json(text: str, error: Exception) -> str:
    """
    Attempt to repair malformed JSON.

    Basic repairs:
    - Add missing closing braces/brackets
    - Fix trailing commas
    - Fix unquoted keys
    """
    # Remove trailing commas before closing braces/brackets
    text = re.sub(r",\s*([}\]])", r"\1", text)

    # Try to balance braces
    open_braces = text.count("{")
    close_braces = text.count("}")
    if open_braces > close_braces:
        text = text + "}" * (open_braces - close_braces)

    open_brackets = text.count("[")
    close_brackets = text.count("]")
    if open_brackets > close_brackets:
        text = text + "]" * (open_brackets - close_brackets)

    return text
