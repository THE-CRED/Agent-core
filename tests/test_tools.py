"""Tests for the tool system."""


from agent import Tool, ToolCall, ToolSpec, tool
from agent.tools import ToolRegistry, _extract_schema_from_function


class TestToolDecorator:
    """Test the @tool decorator."""

    def test_basic_tool_creation(self):
        """Test creating a basic tool."""
        @tool
        def greet(name: str) -> str:
            """Say hello to someone."""
            return f"Hello, {name}!"

        assert isinstance(greet, Tool)
        assert greet.name == "greet"
        assert "Say hello" in greet.spec.description

    def test_tool_with_custom_name(self):
        """Test tool with custom name."""
        @tool(name="custom_greeting")
        def greet(name: str) -> str:
            """Greet a person."""
            return f"Hi, {name}!"

        assert greet.name == "custom_greeting"

    def test_tool_with_custom_description(self):
        """Test tool with custom description."""
        @tool(description="A custom description")
        def my_func() -> str:
            """Original docstring."""
            return "result"

        assert my_func.spec.description == "A custom description"

    def test_tool_schema_extraction(self):
        """Test that schema is extracted from type hints."""
        @tool
        def search(query: str, limit: int = 10) -> str:
            """Search for items."""
            return "results"

        schema = search.spec.parameters
        assert schema["type"] == "object"
        assert "query" in schema["properties"]
        assert "limit" in schema["properties"]
        assert "query" in schema["required"]
        assert "limit" not in schema["required"]  # Has default

    def test_tool_execution_sync(self):
        """Test synchronous tool execution."""
        @tool
        def add(a: int, b: int) -> str:
            """Add two numbers."""
            return str(a + b)

        result = add.execute_sync({"a": 2, "b": 3})
        assert result == "5"

    def test_async_tool(self):
        """Test async tool detection."""
        @tool
        async def async_fetch(url: str) -> str:
            """Fetch a URL."""
            return f"Fetched: {url}"

        assert async_fetch.is_async is True


class TestSchemaExtraction:
    """Test schema extraction from functions."""

    def test_basic_types(self):
        """Test extraction of basic types."""
        def func(s: str, i: int, f: float, b: bool) -> str:
            return ""

        schema = _extract_schema_from_function(func)
        props = schema["properties"]

        assert props["s"]["type"] == "string"
        assert props["i"]["type"] == "integer"
        assert props["f"]["type"] == "number"
        assert props["b"]["type"] == "boolean"

    def test_list_type(self):
        """Test extraction of list types."""
        def func(items: list[str]) -> str:
            return ""

        schema = _extract_schema_from_function(func)
        props = schema["properties"]

        assert props["items"]["type"] == "array"
        assert props["items"]["items"]["type"] == "string"

    def test_required_vs_optional(self):
        """Test required vs optional parameters."""
        def func(required: str, optional: str = "default") -> str:
            return ""

        schema = _extract_schema_from_function(func)

        assert "required" in schema["required"]
        assert "optional" not in schema["required"]


class TestToolSpec:
    """Test ToolSpec class."""

    def test_to_openai_schema(self):
        """Test conversion to OpenAI format."""
        spec = ToolSpec(
            name="search",
            description="Search for items",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        )

        openai_schema = spec.to_openai_schema()

        assert openai_schema["type"] == "function"
        assert openai_schema["function"]["name"] == "search"
        assert openai_schema["function"]["description"] == "Search for items"

    def test_to_anthropic_schema(self):
        """Test conversion to Anthropic format."""
        spec = ToolSpec(
            name="search",
            description="Search for items",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        )

        anthropic_schema = spec.to_anthropic_schema()

        assert anthropic_schema["name"] == "search"
        assert anthropic_schema["description"] == "Search for items"
        assert "input_schema" in anthropic_schema


class TestToolRegistry:
    """Test ToolRegistry class."""

    def test_register_and_get(self):
        """Test registering and retrieving tools."""
        registry = ToolRegistry()

        @tool
        def my_tool() -> str:
            """A tool."""
            return "result"

        registry.register(my_tool)

        retrieved = registry.get("my_tool")
        assert retrieved is my_tool

    def test_list_tools(self):
        """Test listing registered tools."""
        registry = ToolRegistry()

        @tool
        def tool_a() -> str:
            return "a"

        @tool
        def tool_b() -> str:
            return "b"

        registry.register(tool_a)
        registry.register(tool_b)

        tools = registry.get_all()
        assert len(tools) == 2

    def test_get_unknown_tool(self):
        """Test getting unknown tool returns None."""
        registry = ToolRegistry()
        assert registry.get("unknown") is None

    def test_clear_registry(self):
        """Test clearing the registry."""
        registry = ToolRegistry()

        @tool
        def my_tool() -> str:
            return "result"

        registry.register(my_tool)
        assert len(registry) == 1

        registry.clear()
        assert len(registry) == 0


class TestToolCall:
    """Test ToolCall class."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        call = ToolCall(
            id="call_123",
            name="search",
            arguments={"query": "test"},
        )

        data = call.to_dict()

        assert data["id"] == "call_123"
        assert data["name"] == "search"
        assert data["arguments"] == {"query": "test"}

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "id": "call_456",
            "name": "fetch",
            "arguments": {"url": "http://example.com"},
        }

        call = ToolCall.from_dict(data)

        assert call.id == "call_456"
        assert call.name == "fetch"
        assert call.arguments == {"url": "http://example.com"}
