# Tool System

Agent's tool system allows you to register Python functions that LLMs can call. This enables agents to interact with external systems, perform calculations, and access real-time data.

## Basic Usage

### The @tool Decorator

```python
from agent import Agent, tool

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    # Your implementation
    return f"Weather in {city}: 72F, Sunny"

agent = Agent(
    provider="openai",
    model="gpt-4o",
    tools=[get_weather],
)

response = agent.run("What's the weather in Tokyo?")
# The agent will call get_weather("Tokyo") and use the result
```

### Multiple Tools

```python
@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Search results for: {query}"

@tool
def read_file(path: str) -> str:
    """Read contents of a file."""
    with open(path) as f:
        return f.read()

@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file."""
    with open(path, 'w') as f:
        f.write(content)
    return f"Wrote {len(content)} chars to {path}"

agent = Agent(
    provider="anthropic",
    model="claude-sonnet",
    tools=[search_web, read_file, write_file],
)
```

## Tool Configuration

### Custom Name and Description

```python
@tool(
    name="web_search",
    description="Search the internet for current information about any topic.",
)
def search(query: str) -> str:
    return f"Results for: {query}"
```

### Timeout and Retries

```python
@tool(timeout=30.0, max_retries=3)
def slow_api_call(endpoint: str) -> str:
    """Call an external API that might be slow."""
    import requests
    response = requests.get(endpoint, timeout=25)
    return response.text
```

## Type Annotations

Agent automatically generates JSON schemas from type annotations:

### Basic Types

```python
@tool
def process_data(
    text: str,           # -> {"type": "string"}
    count: int,          # -> {"type": "integer"}
    ratio: float,        # -> {"type": "number"}
    enabled: bool,       # -> {"type": "boolean"}
) -> str:
    """Process various data types."""
    return f"Processed: {text}, {count}, {ratio}, {enabled}"
```

### Complex Types

```python
from typing import Optional

@tool
def search_items(
    query: str,
    tags: list[str],           # -> {"type": "array", "items": {"type": "string"}}
    limit: int = 10,           # Optional with default (not in "required")
    category: str | None = None,  # Optional parameter
) -> str:
    """Search for items with filters."""
    return f"Found items matching {query}"
```

### Pydantic Models

```python
from pydantic import BaseModel

class SearchParams(BaseModel):
    query: str
    max_results: int = 10
    include_metadata: bool = False

@tool
def advanced_search(params: SearchParams) -> str:
    """Perform an advanced search with structured parameters."""
    return f"Searching for: {params.query}"
```

## Async Tools

```python
import asyncio

@tool
async def fetch_url(url: str) -> str:
    """Fetch content from a URL asynchronously."""
    import aiohttp
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()

# Works with both sync and async agent methods
response = agent.run("Fetch https://example.com")  # Runs async tool in executor
response = await agent.run_async("Fetch https://example.com")  # Native async
```

## Tool Results

### Accessing Tool Calls

```python
response = agent.run("What's 2+2 and what's the weather?")

if response.has_tool_calls:
    for tool_call in response.tool_calls:
        print(f"Tool: {tool_call.name}")
        print(f"Arguments: {tool_call.arguments}")
        print(f"ID: {tool_call.id}")
```

### Tool Loop

Agent automatically handles the tool calling loop:

1. Agent sends request to LLM
2. LLM returns tool calls
3. Agent executes tools
4. Agent sends results back to LLM
5. Repeat until LLM returns final response

Configure the tool loop:

```python
from agent.types import ToolLoopConfig

agent = Agent(
    provider="openai",
    model="gpt-4o",
    tools=[...],
)

# Access via runtime (advanced)
agent._runtime.tool_loop.config.max_iterations = 5
agent._runtime.tool_loop.config.parallel_tool_execution = True
```

## Error Handling

### Tool Errors

```python
@tool
def risky_operation(data: str) -> str:
    """Perform an operation that might fail."""
    if not data:
        raise ValueError("Data cannot be empty")
    return f"Processed: {data}"

# By default, tool errors are returned to the LLM as error messages
# The LLM can then decide how to handle the error
```

### Stop on Error

```python
from agent.types import ToolLoopConfig

config = ToolLoopConfig(stop_on_error=True)
# Now tool errors will raise ToolExecutionError instead of continuing
```

## Manual Tool Registration

For more control, create Tool objects directly:

```python
from agent import Tool
from agent.types import ToolSpec

def my_function(x: int, y: int) -> int:
    return x + y

spec = ToolSpec(
    name="add_numbers",
    description="Add two numbers together",
    parameters={
        "type": "object",
        "properties": {
            "x": {"type": "integer", "description": "First number"},
            "y": {"type": "integer", "description": "Second number"},
        },
        "required": ["x", "y"],
    },
)

tool = Tool(spec=spec, function=my_function)
agent = Agent(provider="openai", model="gpt-4o", tools=[tool])
```

## Tool Registry

Manage tools programmatically:

```python
from agent.tools import ToolRegistry

registry = ToolRegistry()

@tool
def tool_a(x: str) -> str:
    return x

@tool
def tool_b(y: int) -> str:
    return str(y)

registry.register(tool_a)
registry.register(tool_b)

print(registry.get("tool_a"))  # Get by name
print(registry.get_all())      # List all tools
print(registry.specs())        # Get all ToolSpecs
```

## Best Practices

### 1. Clear Descriptions

```python
# Good - tells the LLM when to use this tool
@tool
def search_documentation(query: str) -> str:
    """Search the project documentation for API references, 
    tutorials, and examples. Use this when the user asks 
    about how to use specific features."""
    ...

# Bad - vague description
@tool
def search(q: str) -> str:
    """Search for stuff."""
    ...
```

### 2. Specific Return Values

```python
# Good - structured, parseable output
@tool
def get_user_info(user_id: str) -> str:
    user = fetch_user(user_id)
    return json.dumps({
        "name": user.name,
        "email": user.email,
        "created_at": user.created_at.isoformat(),
    })

# Bad - ambiguous output
@tool
def get_user(id: str) -> str:
    return str(fetch_user(id))
```

### 3. Error Messages

```python
@tool
def database_query(sql: str) -> str:
    """Execute a read-only SQL query."""
    try:
        results = db.execute(sql)
        return json.dumps(results)
    except Exception as e:
        return f"Error executing query: {e}. Please check the SQL syntax."
```

### 4. Idempotent Operations

```python
@tool
def update_setting(key: str, value: str) -> str:
    """Update a configuration setting. Safe to call multiple times."""
    config[key] = value
    return f"Set {key} = {value}"
```

## Next Steps

- [Structured Outputs](structured-outputs.md) - Type-safe LLM responses
- [Sessions](sessions.md) - Multi-turn conversations with tools
- [Middleware](middleware.md) - Hook into tool execution
