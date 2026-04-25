"""
Structured output handling.

Manages schema-based output generation and validation.
"""

import contextlib
from typing import Any

from pydantic import BaseModel

from agent.errors import SchemaValidationError
from agent.schemas import Schema, extract_json, repair_json, schema_to_prompt


class StructuredOutputHandler:
    """
    Handles structured output generation and validation.

    Supports multiple strategies:
    1. Native schema-enforced generation (when provider supports it)
    2. Prompt-based JSON generation with validation
    3. Repair attempts for malformed output
    """

    def __init__(
        self,
        schema: type[BaseModel] | dict[str, Any],
        strict: bool = True,
        repair_attempts: int = 1,
    ):
        self.schema = Schema(schema, strict=strict, repair_attempts=repair_attempts)
        self.strict = strict
        self.repair_attempts = repair_attempts

    def get_json_schema(self) -> dict[str, Any]:
        """Get the JSON schema for provider configuration."""
        return self.schema.json_schema

    def get_system_prompt_addition(self) -> str:
        """Get additional system prompt for schema instruction."""
        return schema_to_prompt(self.schema)

    def parse_response(self, text: str) -> Any:
        """
        Parse and validate the response text.

        Args:
            text: Raw response text from LLM

        Returns:
            Validated/parsed output

        Raises:
            SchemaValidationError: If validation fails after all attempts
        """
        last_error: Exception | None = None

        for attempt in range(self.repair_attempts + 1):
            try:
                # Try to extract and parse JSON
                data = extract_json(text)
                return self.schema.validate(data)
            except (ValueError, SchemaValidationError) as e:
                last_error = e

                # Try to repair if we have attempts left
                if attempt < self.repair_attempts:
                    with contextlib.suppress(Exception):
                        repaired = repair_json(text, e)
                        if repaired == text:
                            # Repair didn't change anything, no point retrying
                            break
                        text = repaired

        # All attempts failed
        raise SchemaValidationError(
            f"Failed to parse structured output after {self.repair_attempts + 1} attempts: {last_error}",
            schema=self.schema.schema,
            output=text,
        )

    def validate_native_output(self, data: Any) -> Any:
        """
        Validate output from native schema-enforced generation.

        Args:
            data: Parsed JSON data from provider

        Returns:
            Validated/parsed output

        Raises:
            SchemaValidationError: If validation fails
        """
        return self.schema.validate(data)


def prepare_structured_request(
    schema: type[BaseModel] | dict[str, Any] | None,
    system: str | None,
    supports_native_schema: bool,
) -> tuple[str | None, dict[str, Any] | None]:
    """
    Prepare request modifications for structured output.

    Args:
        schema: The output schema
        system: Current system prompt
        supports_native_schema: Whether provider supports native schema

    Returns:
        Tuple of (modified system prompt, json_schema if native supported)
    """
    if schema is None:
        return system, None

    handler = StructuredOutputHandler(schema)

    if supports_native_schema:
        # Use native schema support
        return system, handler.get_json_schema()
    else:
        # Add schema instructions to system prompt
        schema_prompt = handler.get_system_prompt_addition()
        modified_system = f"{system}\n\n{schema_prompt}" if system else schema_prompt
        return modified_system, None
