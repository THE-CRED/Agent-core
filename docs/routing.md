# Routing & Fallback

AgentRouter enables intelligent routing across multiple agents with automatic failover, load balancing, and cost optimization.

## Basic Usage

```python
from agent import Agent, AgentRouter

router = AgentRouter(
    agents=[
        Agent(provider="anthropic", model="claude-sonnet"),
        Agent(provider="openai", model="gpt-4o"),
    ],
    strategy="fallback",
)

# Automatically falls back if first provider fails
response = router.run("Hello, world!")
print(f"Response from: {response.provider}")
```

## Routing Strategies

### Fallback (Default)

Try each agent in order until one succeeds:

```python
router = AgentRouter(
    agents=[
        Agent(provider="anthropic", model="claude-sonnet"),  # Primary
        Agent(provider="openai", model="gpt-4o"),            # Backup 1
        Agent(provider="gemini", model="gemini-pro"),        # Backup 2
    ],
    strategy="fallback",
)

# Tries anthropic first, then openai, then gemini
response = router.run("Complex query")
```

### Round Robin

Distribute load across agents:

```python
router = AgentRouter(
    agents=[
        Agent(provider="openai", model="gpt-4o"),
        Agent(provider="anthropic", model="claude-sonnet"),
    ],
    strategy="round_robin",
)

# First request goes to openai
router.run("Request 1")

# Second request goes to anthropic
router.run("Request 2")

# Third request goes back to openai
router.run("Request 3")
```

### Fastest

Race all agents, use the first response:

```python
router = AgentRouter(
    agents=[
        Agent(provider="openai", model="gpt-4o"),
        Agent(provider="anthropic", model="claude-sonnet"),
        Agent(provider="gemini", model="gemini-flash"),
    ],
    strategy="fastest",
)

# All agents race, fastest response wins
response = router.run("Quick question")
print(f"Fastest: {response.provider} ({response.latency_ms}ms)")
```

### Cheapest

Route to the most cost-effective available agent:

```python
router = AgentRouter(
    agents=[
        Agent(provider="openai", model="gpt-4o"),      # $2.50 input
        Agent(provider="openai", model="gpt-4o-mini"), # $0.15 input
        Agent(provider="anthropic", model="claude-haiku"),  # $0.25 input
    ],
    strategy="cheapest",
)

# Routes to gpt-4o-mini first (cheapest)
response = router.run("Simple question")
```

### Capability-Based

Route based on required features:

```python
router = AgentRouter(
    agents=[
        Agent(provider="openai", model="gpt-4o", tools=[my_tool]),
        Agent(provider="anthropic", model="claude-sonnet"),
        Agent(provider="deepseek", model="deepseek-chat"),
    ],
    strategy="capability",
)

# Routes to agents that support tools
response = router.run("Use the tool", tools=[my_tool])
```

### Custom Router

Implement your own routing logic:

```python
from agent.types import RouteResult

def smart_router(request, agents):
    """Route based on input content."""
    text = request.input or ""
    
    if "code" in text.lower():
        # Prefer Claude for coding
        for agent in agents:
            if agent.provider == "anthropic":
                return RouteResult(agent=agent, reason="coding task")
    
    if len(text) > 5000:
        # Use Gemini for long context
        for agent in agents:
            if agent.provider == "gemini":
                return RouteResult(agent=agent, reason="long context")
    
    # Default to first agent
    return RouteResult(agent=agents[0], reason="default")

router = AgentRouter(
    agents=[...],
    strategy="custom",
    custom_router=smart_router,
)
```

## Async Support

```python
# Async run
response = await router.run_async("Hello")

# Async streaming
async for event in await router.stream_async("Write a story"):
    if event.type == "text_delta":
        print(event.text, end="")
```

## Streaming with Routing

```python
# Fallback on connection errors
for event in router.stream("Write a poem"):
    if event.type == "text_delta":
        print(event.text, end="")
```

## Structured Output with Routing

```python
from pydantic import BaseModel

class Analysis(BaseModel):
    sentiment: str
    keywords: list[str]

response = router.json(
    "Analyze: Great product!",
    schema=Analysis,
)
print(response.output.sentiment)
```

## Error Handling

### Routing Errors

```python
from agent.errors import RoutingError

try:
    response = router.run("Hello")
except RoutingError as e:
    print(f"All agents failed: {e.message}")
    for i, error in enumerate(e.errors):
        print(f"  Agent {i}: {error}")
```

### Partial Failures

```python
# The router handles individual failures gracefully
router = AgentRouter(
    agents=[
        Agent(provider="invalid", model="x"),  # Will fail
        Agent(provider="openai", model="gpt-4o"),  # Will succeed
    ],
    strategy="fallback",
)

# Still works - falls back to openai
response = router.run("Hello")
```

## Configuration

```python
router = AgentRouter(
    agents=[...],
    strategy="fallback",
)

# Run with options
response = router.run(
    "Hello",
    temperature=0.7,
    max_tokens=500,
    system="Be helpful.",
)
```

## Use Cases

### High Availability

```python
# Production setup with multiple fallbacks
router = AgentRouter(
    agents=[
        # Primary region
        Agent(provider="openai", model="gpt-4o", base_url="https://us-east.api.openai.com"),
        # Backup region
        Agent(provider="openai", model="gpt-4o", base_url="https://us-west.api.openai.com"),
        # Different provider
        Agent(provider="anthropic", model="claude-sonnet"),
    ],
    strategy="fallback",
)
```

### Cost Optimization

```python
# Try cheap first, expensive as backup
router = AgentRouter(
    agents=[
        Agent(provider="openai", model="gpt-4o-mini"),     # Cheapest
        Agent(provider="anthropic", model="claude-haiku"), # Cheap
        Agent(provider="openai", model="gpt-4o"),          # Expensive
        Agent(provider="anthropic", model="claude-opus"),  # Most expensive
    ],
    strategy="cheapest",
)
```

### Load Balancing

```python
# Distribute across multiple API keys
router = AgentRouter(
    agents=[
        Agent(provider="openai", model="gpt-4o", api_key="key1"),
        Agent(provider="openai", model="gpt-4o", api_key="key2"),
        Agent(provider="openai", model="gpt-4o", api_key="key3"),
    ],
    strategy="round_robin",
)
```

### Speed Optimization

```python
# Race for fastest response
router = AgentRouter(
    agents=[
        Agent(provider="groq", model="llama-3.1-70b"),    # Fast
        Agent(provider="anthropic", model="claude-haiku"), # Fast
        Agent(provider="openai", model="gpt-4o-mini"),    # Fast
    ],
    strategy="fastest",
)
```

## Monitoring

Track routing decisions:

```python
from agent import Middleware

class RoutingLogger(Middleware):
    def after(self, request, response):
        print(f"Routed to: {response.provider}/{response.model}")
        print(f"Latency: {response.latency_ms}ms")
        print(f"Cost: ${response.cost_estimate:.4f}")
        return response

# Apply to all agents
agents = [
    Agent(provider="openai", model="gpt-4o", middleware=[RoutingLogger()]),
    Agent(provider="anthropic", model="claude-sonnet", middleware=[RoutingLogger()]),
]

router = AgentRouter(agents=agents, strategy="fallback")
```

## Best Practices

### 1. Order Agents by Preference

```python
# Most preferred first for fallback
router = AgentRouter(
    agents=[
        Agent(provider="anthropic", model="claude-sonnet"),  # Preferred
        Agent(provider="openai", model="gpt-4o"),            # Good backup
        Agent(provider="gemini", model="gemini-pro"),        # Last resort
    ],
    strategy="fallback",
)
```

### 2. Match Capabilities

```python
# Don't mix agents with different capabilities
router = AgentRouter(
    agents=[
        Agent(provider="openai", model="gpt-4o", tools=tools),
        Agent(provider="anthropic", model="claude-sonnet", tools=tools),
        # Include tools for all agents when using tools
    ],
    strategy="fallback",
)
```

### 3. Handle All Failures

```python
try:
    response = router.run(prompt)
except RoutingError:
    # Have a final fallback
    response = simple_agent.run(prompt)
```

## Next Steps

- [Middleware](middleware.md) - Add logging and tracing to routers
- [Error Handling](error-handling.md) - Handle routing errors
- [Providers](providers.md) - Configure individual providers
