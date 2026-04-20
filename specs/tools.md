# Tools

Tools give your agent the ability to take actions and access external data. This guide covers creating, configuring, and using tools.

## Overview

Tools are Python functions that the LLM can call to perform actions. Agent handles:
- Schema extraction from type hints
- Provider-specific format conversion
- Tool execution and result handling
- Multi-turn tool loops

## Quick Start

```python
from agent import Agent, tool

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    # Your implementation
    return f"Weather in {city}: 72F, sunny"

agent = Agent(
    provider="openai",
    model="gpt-4o",
    tools=[get_weather],
)

response = agent.run("What's the weather in Tokyo?")
# Agent will call get_weather("Tokyo") and use the result
```

## Creating Tools

### Basic Tool

Use the `@tool` decorator on any function:

```python
@tool
def search(query: str) -> str:
    """Search for information online."""
    # Implementation
    return f"Results for: {query}"
```

Key points:
- **Docstring** becomes the tool description (used by the LLM)
- **Type hints** are converted to JSON Schema
- **Return value** should be a string (or will be JSON-serialized)

### Custom Name and Description

```python
@tool(name="web_search", description="Search the web for current information")
def search(query: str) -> str:
    """This docstring is ignored when description is provided."""
    return f"Results for: {query}"
```

### Multiple Parameters

```python
@tool
def book_flight(
    origin: str,
    destination: str,
    date: str,
    passengers: int = 1,
) -> str:
    """
    Book a flight between cities.
    
    Args:
        origin: Departure city
        destination: Arrival city
        date: Travel date (YYYY-MM-DD)
        passengers: Number of passengers
    """
    return f"Booked flight from {origin} to {destination} on {date} for {passengers}"
```

### Optional Parameters

Parameters with default values are optional:

```python
@tool
def search(
    query: str,                    # Required
    limit: int = 10,              # Optional
    sort: str = "relevance",      # Optional
) -> str:
    """Search with optional filters."""
    return f"Found results for '{query}' (limit={limit}, sort={sort})"
```

### Complex Types

```python
from typing import List, Dict

@tool
def analyze_data(
    values: list[float],
    options: dict[str, str] | None = None,
) -> str:
    """Analyze a list of values."""
    avg = sum(values) / len(values) if values else 0
    return f"Average: {avg}"
```

### Pydantic Models

```python
from pydantic import BaseModel

class SearchParams(BaseModel):
    query: str
    filters: list[str]
    max_results: int = 10

@tool
def advanced_search(params: SearchParams) -> str:
    """Search with structured parameters."""
    return f"Searching: {params.query} with {len(params.filters)} filters"
```

## Async Tools

Async functions are fully supported:

```python
import aiohttp

@tool
async def fetch_url(url: str) -> str:
    """Fetch content from a URL."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()
```

Use with async agent methods:

```python
response = await agent.run_async("Fetch example.com")
```

## Tool Configuration

### Timeout

```python
@tool(timeout=30.0)
def slow_operation(data: str) -> str:
    """Operation that might be slow."""
    # ...
```

### Max Retries

```python
@tool(max_retries=3)
def flaky_api(query: str) -> str:
    """API that sometimes fails."""
    # ...
```

## Using Tools with Agents

### Single Tool

```python
agent = Agent(
    provider="openai",
    model="gpt-4o",
    tools=[my_tool],
)
```

### Multiple Tools

```python
agent = Agent(
    provider="anthropic",
    model="claude-sonnet",
    tools=[search, calculate, fetch_data, send_email],
)
```

### Tool Selection

The LLM decides which tools to use based on:
1. Tool names
2. Tool descriptions
3. Parameter schemas
4. User's request

Write clear descriptions to help the LLM choose correctly:

```python
@tool
def search_products(query: str) -> str:
    """Search the product catalog for items. Use for product lookups."""
    # ...

@tool
def search_orders(order_id: str) -> str:
    """Look up a specific order by ID. Use for order status checks."""
    # ...
```

## Tool Loops

When an agent has tools, it may call multiple tools in sequence:

```python
@tool
def get_user(user_id: str) -> str:
    """Get user details by ID."""
    return '{"name": "Alice", "email": "alice@example.com"}'

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email."""
    return f"Email sent to {to}"

agent = Agent(provider="openai", model="gpt-4o", tools=[get_user, send_email])

# The agent might:
# 1. Call get_user("123") to get details
# 2. Call send_email(...) with the user's email
response = agent.run("Send a welcome email to user 123")
```

### Loop Limits

Tool loops are bounded for safety:

```python
from agent.execution.tool_loop import ToolLoopConfig

config = ToolLoopConfig(
    max_iterations=10,              # Max tool loop rounds
    max_tool_calls_per_iteration=20, # Max calls per round
    timeout_per_tool=30.0,          # Timeout per tool call
    parallel_tool_execution=True,   # Run independent calls in parallel
    stop_on_error=False,            # Continue if a tool fails
)
```

## Error Handling

### Tool Errors

If a tool raises an exception, the error message is sent back to the LLM:

```python
@tool
def risky_operation(data: str) -> str:
    """Perform a risky operation."""
    if not data:
        raise ValueError("Data cannot be empty")
    return "Success"

# LLM receives: "Error: Data cannot be empty"
# and can decide how to proceed
```

### Graceful Failures

Handle errors gracefully for better UX:

```python
@tool
def fetch_data(url: str) -> str:
    """Fetch data from URL."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        return f"Failed to fetch {url}: {e}"
```

## Security Considerations

### Input Validation

Always validate tool inputs:

```python
@tool
def read_file(path: str) -> str:
    """Read a file from the allowed directory."""
    import os
    
    # Validate path
    allowed_dir = "/safe/directory"
    real_path = os.path.realpath(path)
    
    if not real_path.startswith(allowed_dir):
        raise ValueError("Access denied: path outside allowed directory")
    
    with open(real_path) as f:
        return f.read()
```

### Dangerous Operations

Be careful with tools that:
- Execute code
- Access filesystems
- Make network requests
- Modify data

```python
# DANGEROUS - Don't do this
@tool
def run_code(code: str) -> str:
    """Execute Python code."""
    return str(eval(code))  # Never do this!

# SAFER - Limited operations
@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    import ast
    import operator
    
    # Only allow safe math operations
    allowed_ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
    }
    
    # Safe evaluation logic...
```

### Rate Limiting

Implement rate limiting for expensive operations:

```python
from functools import wraps
import time

def rate_limit(calls_per_minute: int):
    """Decorator to rate limit tool calls."""
    interval = 60.0 / calls_per_minute
    last_call = [0.0]
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_call[0]
            if elapsed < interval:
                time.sleep(interval - elapsed)
            last_call[0] = time.time()
            return func(*args, **kwargs)
        return wrapper
    return decorator

@tool
@rate_limit(calls_per_minute=10)
def expensive_api_call(query: str) -> str:
    """Call an expensive external API."""
    # ...
```

## Advanced Patterns

### Tool Registry

Manage tools programmatically:

```python
from agent.tools import ToolRegistry

registry = ToolRegistry()
registry.register(search_tool)
registry.register(calculate_tool)

# Use with agent
agent = Agent(
    provider="openai",
    model="gpt-4o",
    tools=registry.get_all(),
)
```

### Dynamic Tools

Create tools dynamically:

```python
from agent.tools import Tool, ToolSpec

def create_db_tool(table_name: str) -> Tool:
    """Create a tool for querying a specific table."""
    
    def query_table(query: str) -> str:
        # Query the specific table
        return f"Results from {table_name}: ..."
    
    spec = ToolSpec(
        name=f"query_{table_name}",
        description=f"Query the {table_name} table",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "SQL WHERE clause"}
            },
            "required": ["query"],
        },
    )
    
    return Tool(spec=spec, function=query_table)

# Create tools for each table
tools = [create_db_tool(t) for t in ["users", "orders", "products"]]
```

### Context-Aware Tools

Pass context to tools using closures:

```python
def create_user_tools(user_id: str, permissions: list[str]):
    """Create tools scoped to a specific user."""
    
    @tool
    def get_my_data() -> str:
        """Get the current user's data."""
        return fetch_user_data(user_id)
    
    @tool
    def update_my_profile(name: str, email: str) -> str:
        """Update the current user's profile."""
        if "write" not in permissions:
            raise PermissionError("No write access")
        return update_profile(user_id, name, email)
    
    return [get_my_data, update_my_profile]

# Create user-specific agent
tools = create_user_tools(current_user.id, current_user.permissions)
agent = Agent(provider="openai", model="gpt-4o", tools=tools)
```

## Testing Tools

Use the fake provider for testing:

```python
from agent.testing import create_test_agent, FakeResponse
from agent.tools import ToolCall

def test_tool_usage():
    agent, provider = create_test_agent()
    
    # Simulate LLM requesting a tool call
    provider.set_responses([
        FakeResponse.with_tool_call(
            name="get_weather",
            arguments={"city": "Tokyo"},
        ),
        FakeResponse(text="The weather in Tokyo is 72F and sunny."),
    ])
    
    response = agent.run("What's the weather?")
    
    # Verify tool was called
    requests = provider.get_requests()
    assert len(requests) == 2  # Initial + after tool result
```
