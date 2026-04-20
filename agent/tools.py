"""
Agent tool system.

Provides the @tool decorator and tool-related types for registering
Python functions as tools that can be called by LLMs.
"""

import asyncio
import inspect
import json
from collections.abc import Callable
from typing import Any, get_type_hints

from pydantic import BaseModel

from agent.types.tools import ToolCall, ToolResult, ToolSpec


class Tool:
    """A registered tool with its specification and callable function."""

    def __init__(
        self,
        spec: ToolSpec,
        function: Callable[..., Any],
        is_async: bool = False,
        timeout: float | None = None,
        max_retries: int = 0,
    ):
        self.spec = spec
        self.function = function
        self.is_async = is_async
        self.timeout = timeout
        self.max_retries = max_retries

    @property
    def name(self) -> str:
        """Get tool name."""
        return self.spec.name

    async def execute(self, arguments: dict[str, Any]) -> str:
        """Execute the tool with given arguments."""
        try:
            if self.is_async:
                result = await self.function(**arguments)
            else:
                # Run sync function in executor
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, lambda: self.function(**arguments))

            # Convert result to string
            if isinstance(result, str):
                return result
            elif isinstance(result, BaseModel):
                return result.model_dump_json()
            else:
                return json.dumps(result, default=str)
        except Exception as e:
            return f"Error: {e}"

    def execute_sync(self, arguments: dict[str, Any]) -> str:
        """Execute the tool synchronously."""
        try:
            if self.is_async:
                # Run async function in new event loop
                return asyncio.run(self.function(**arguments))
            else:
                result = self.function(**arguments)

            # Convert result to string
            if isinstance(result, str):
                return result
            elif isinstance(result, BaseModel):
                return result.model_dump_json()
            else:
                return json.dumps(result, default=str)
        except Exception as e:
            return f"Error: {e}"


def _get_json_schema_type(python_type: type) -> dict[str, Any]:
    """Convert Python type to JSON Schema type."""
    origin = getattr(python_type, "__origin__", None)

    # Handle None
    if python_type is type(None):
        return {"type": "null"}

    # Handle basic types
    type_map = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
        bytes: {"type": "string"},
    }

    if python_type in type_map:
        return type_map[python_type]

    # Handle list/List
    if origin is list:
        args = getattr(python_type, "__args__", (Any,))
        item_type = args[0] if args else Any
        return {
            "type": "array",
            "items": _get_json_schema_type(item_type) if item_type is not Any else {},
        }

    # Handle dict/Dict
    if origin is dict:
        return {"type": "object"}

    # Handle Optional (Union with None)
    if origin is type(None | str):  # Union type
        args = getattr(python_type, "__args__", ())
        non_none_types = [a for a in args if a is not type(None)]
        if len(non_none_types) == 1:
            schema = _get_json_schema_type(non_none_types[0])
            # JSON Schema doesn't have a standard optional, just allow null
            return schema
        return {}

    # Handle Pydantic models
    if hasattr(python_type, "model_json_schema"):
        return python_type.model_json_schema()

    # Handle Literal
    if origin is type(None):  # Literal
        args = getattr(python_type, "__args__", ())
        return {"enum": list(args)}

    # Default to object
    return {"type": "object"}


def _extract_schema_from_function(func: Callable[..., Any]) -> dict[str, Any]:
    """Extract JSON Schema from function signature."""
    sig = inspect.signature(func)
    hints = get_type_hints(func)

    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue

        # Get type hint
        python_type = hints.get(name, Any)

        # Get schema for this type
        prop_schema = _get_json_schema_type(python_type)

        # Add description from docstring if available
        properties[name] = prop_schema

        # Check if required (no default value)
        if param.default is inspect.Parameter.empty:
            required.append(name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def _extract_description(func: Callable[..., Any]) -> str:
    """Extract description from function docstring."""
    doc = inspect.getdoc(func)
    if not doc:
        return f"Execute the {func.__name__} function."

    # Get first line/paragraph of docstring
    lines = doc.strip().split("\n\n")
    return lines[0].strip()


def tool(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    timeout: float | None = None,
    max_retries: int = 0,
) -> Tool | Callable[[Callable[..., Any]], Tool]:
    """
    Decorator to register a function as a tool.

    Usage:
        @tool
        def search(query: str) -> str:
            '''Search for information.'''
            return f"Results for: {query}"

        @tool(name="custom_name", timeout=30.0)
        def fetch_data(url: str) -> str:
            '''Fetch data from a URL.'''
            ...
    """

    def decorator(f: Callable[..., Any]) -> Tool:
        tool_name = name or f.__name__
        tool_description = description or _extract_description(f)
        parameters = _extract_schema_from_function(f)
        is_async = asyncio.iscoroutinefunction(f)

        spec = ToolSpec(
            name=tool_name,
            description=tool_description,
            parameters=parameters,
            function=f,
            is_async=is_async,
        )

        return Tool(
            spec=spec,
            function=f,
            is_async=is_async,
            timeout=timeout,
            max_retries=max_retries,
        )

    if func is not None:
        return decorator(func)
    return decorator


class ToolRegistry:
    """Registry for managing tools."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def get_all(self) -> list[Tool]:
        """List all registered tools."""
        return list(self._tools.values())

    def specs(self) -> list[ToolSpec]:
        """Get specs for all registered tools."""
        return [t.spec for t in self._tools.values()]

    def clear(self) -> None:
        """Clear all registered tools."""
        self._tools.clear()

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)


# Re-export types for backwards compatibility
__all__ = ["tool", "Tool", "ToolSpec", "ToolCall", "ToolResult", "ToolRegistry"]
