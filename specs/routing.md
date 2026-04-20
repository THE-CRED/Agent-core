# Routing & Fallback

The AgentRouter enables multi-agent orchestration with automatic failover, load balancing, and custom routing strategies.

## Overview

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
response = router.run("Hello!")
```

## Strategies

### Fallback (Default)

Try each agent in order until one succeeds:

```python
router = AgentRouter(
    agents=[primary_agent, backup_agent],
    strategy="fallback",
)

# If primary fails (rate limit, error, timeout), tries backup
response = router.run("Hello")
```

Use cases:
- High availability
- Cost optimization (cheap primary, expensive backup)
- Provider redundancy

### Round Robin

Distribute requests evenly across agents:

```python
router = AgentRouter(
    agents=[agent1, agent2, agent3],
    strategy="round_robin",
)

# Request 1 -> agent1
# Request 2 -> agent2
# Request 3 -> agent3
# Request 4 -> agent1 (cycles back)
```

Use cases:
- Load balancing
- Rate limit distribution
- A/B testing

### Fastest

Race agents, use first response:

```python
router = AgentRouter(
    agents=[agent1, agent2],
    strategy="fastest",
)

# Both agents run in parallel, first response wins
response = router.run("Hello")
```

Use cases:
- Latency optimization
- Best-effort speed

**Note:** This uses more API calls (all agents are called).

### Cheapest

Route to the cheapest available agent:

```python
router = AgentRouter(
    agents=[
        Agent(provider="openai", model="gpt-4o"),        # $2.50/1M input
        Agent(provider="anthropic", model="claude-haiku"), # $0.25/1M input
        Agent(provider="openai", model="gpt-4o-mini"),   # $0.15/1M input
    ],
    strategy="cheapest",
)

# Routes to gpt-4o-mini (cheapest), falls back if it fails
response = router.run("Hello")
```

Use cases:
- Cost optimization
- Budget-conscious applications

### Capability-Based

Route based on required capabilities:

```python
router = AgentRouter(
    agents=[
        Agent(provider="openai", model="gpt-4o", tools=[my_tool]),
        Agent(provider="deepseek", model="deepseek-chat"),  # No vision
    ],
    strategy="capability",
)

# Only routes to agents that support required features
# If request has tools -> only tool-capable agents
# If request has images -> only vision-capable agents
```

Use cases:
- Feature-specific routing
- Multi-model architectures

### Custom

Implement your own routing logic:

```python
from agent.router import RouteResult
from agent.messages import AgentRequest

def my_router(request: AgentRequest, agents: list[Agent]) -> RouteResult:
    """Route based on input content."""
    input_text = request.input or ""
    
    # Route code questions to Claude
    if "code" in input_text.lower() or "function" in input_text.lower():
        claude = next(a for a in agents if a.provider == "anthropic")
        return RouteResult(agent=claude, reason="Code question")
    
    # Route math to GPT
    if any(word in input_text.lower() for word in ["calculate", "math", "number"]):
        gpt = next(a for a in agents if a.provider == "openai")
        return RouteResult(agent=gpt, reason="Math question")
    
    # Default
    return RouteResult(agent=agents[0], reason="Default")

router = AgentRouter(
    agents=[
        Agent(provider="openai", model="gpt-4o"),
        Agent(provider="anthropic", model="claude-sonnet"),
    ],
    strategy="custom",
    custom_router=my_router,
)
```

## Router Methods

All Agent methods are available on the router:

### run() / run_async()

```python
response = router.run("Hello")
response = await router.run_async("Hello")
```

### stream() / stream_async()

```python
for event in router.stream("Write a story"):
    if event.type == "text_delta":
        print(event.text, end="")

async for event in await router.stream_async("Write a story"):
    if event.type == "text_delta":
        print(event.text, end="")
```

### json()

```python
from pydantic import BaseModel

class Summary(BaseModel):
    title: str
    points: list[str]

response = router.json("Summarize this", schema=Summary)
```

## Error Handling

### RoutingError

Raised when all agents fail:

```python
from agent import RoutingError

try:
    response = router.run("Hello")
except RoutingError as e:
    print(f"All agents failed: {e.message}")
    
    # Access individual errors
    for error in e.errors:
        print(f"  - {error}")
```

### Partial Failures

With fallback, partial failures are transparent:

```python
router = AgentRouter(
    agents=[agent1, agent2, agent3],
    strategy="fallback",
)

# If agent1 fails, agent2 is tried
# If agent2 fails, agent3 is tried
# Only raises RoutingError if ALL fail
response = router.run("Hello")

# Check which agent was used
print(response.provider, response.model)
```

## Configuration

### Agent Order Matters

For fallback strategy, order determines priority:

```python
router = AgentRouter(
    agents=[
        Agent(provider="anthropic", model="claude-sonnet"),  # First choice
        Agent(provider="openai", model="gpt-4o"),            # Second choice
        Agent(provider="openai", model="gpt-4o-mini"),       # Last resort
    ],
    strategy="fallback",
)
```

### Per-Agent Configuration

Each agent has its own settings:

```python
router = AgentRouter(
    agents=[
        Agent(
            provider="anthropic",
            model="claude-sonnet",
            timeout=30.0,
            max_retries=3,
        ),
        Agent(
            provider="openai",
            model="gpt-4o",
            timeout=60.0,  # Different timeout
            max_retries=1,
        ),
    ],
)
```

## Patterns

### Production High-Availability

```python
def create_production_router():
    """Create a production-ready router with fallback tiers."""
    return AgentRouter(
        agents=[
            # Tier 1: Premium (best quality)
            Agent(provider="anthropic", model="claude-sonnet"),
            
            # Tier 2: Standard (good fallback)
            Agent(provider="openai", model="gpt-4o"),
            
            # Tier 3: Budget (last resort)
            Agent(provider="openai", model="gpt-4o-mini"),
        ],
        strategy="fallback",
    )

router = create_production_router()
```

### Cost-Optimized with Quality Fallback

```python
router = AgentRouter(
    agents=[
        # Try cheap first
        Agent(provider="deepseek", model="deepseek-chat"),
        Agent(provider="openai", model="gpt-4o-mini"),
        
        # Quality fallback if cheap fails
        Agent(provider="openai", model="gpt-4o"),
    ],
    strategy="fallback",
)
```

### Task-Specific Routing

```python
def task_router(request: AgentRequest, agents: list[Agent]) -> RouteResult:
    input_text = request.input or ""
    
    # Complex reasoning -> Claude
    if any(word in input_text.lower() for word in ["analyze", "reason", "explain"]):
        return RouteResult(
            agent=next(a for a in agents if a.provider == "anthropic"),
            reason="Complex reasoning task",
        )
    
    # Code generation -> GPT-4
    if any(word in input_text.lower() for word in ["code", "function", "implement"]):
        return RouteResult(
            agent=next(a for a in agents if a.model == "gpt-4o"),
            reason="Code generation task",
        )
    
    # Simple tasks -> cheapest
    return RouteResult(
        agent=next(a for a in agents if "mini" in a.model or "haiku" in a.model),
        reason="Simple task",
    )

router = AgentRouter(
    agents=[
        Agent(provider="anthropic", model="claude-sonnet"),
        Agent(provider="openai", model="gpt-4o"),
        Agent(provider="openai", model="gpt-4o-mini"),
    ],
    strategy="custom",
    custom_router=task_router,
)
```

### Multi-Provider Load Balancing

```python
# Distribute load across providers
router = AgentRouter(
    agents=[
        Agent(provider="openai", model="gpt-4o"),
        Agent(provider="anthropic", model="claude-sonnet"),
        Agent(provider="gemini", model="gemini-1.5-pro"),
    ],
    strategy="round_robin",
)
```

### Latency-Sensitive Application

```python
# Race for fastest response
router = AgentRouter(
    agents=[
        Agent(provider="anthropic", model="claude-haiku"),
        Agent(provider="openai", model="gpt-4o-mini"),
        Agent(provider="gemini", model="gemini-1.5-flash"),
    ],
    strategy="fastest",
)
```

## Monitoring

### Track Provider Usage

```python
from collections import Counter

usage_counter = Counter()

# After each request
response = router.run("Hello")
usage_counter[response.provider] += 1

# Check distribution
print(dict(usage_counter))
# {'anthropic': 45, 'openai': 32, 'gemini': 23}
```

### Track Failures

```python
failures = []

try:
    response = router.run("Hello")
except RoutingError as e:
    failures.append({
        "time": datetime.now(),
        "errors": [str(err) for err in e.errors],
    })
```

### Cost Tracking

```python
total_cost = 0.0

response = router.run("Hello")
if response.cost_estimate:
    total_cost += response.cost_estimate
    print(f"This request: ${response.cost_estimate:.6f}")
    print(f"Total so far: ${total_cost:.6f}")
```

## Best Practices

### 1. Test Fallback Behavior

```python
from agent.testing import create_test_agent, FakeResponse
from agent.errors import ProviderError

def test_fallback():
    agent1, provider1 = create_test_agent()
    agent2, provider2 = create_test_agent()
    
    # First agent fails
    provider1.set_response(FakeResponse.with_error(
        ProviderError("Service unavailable")
    ))
    # Second agent succeeds
    provider2.set_response(FakeResponse(text="Success"))
    
    router = AgentRouter(agents=[agent1, agent2])
    response = router.run("Test")
    
    assert response.text == "Success"
```

### 2. Log Routing Decisions

```python
import logging

def logged_router(request, agents):
    result = my_routing_logic(request, agents)
    logging.info(f"Routed to {result.agent.provider}: {result.reason}")
    return result
```

### 3. Handle Degraded Mode

```python
try:
    response = router.run("Hello")
except RoutingError:
    # All providers down - degrade gracefully
    return {
        "error": "Service temporarily unavailable",
        "retry_after": 60,
    }
```

### 4. Use Appropriate Timeouts

```python
# Faster timeout for fallback
router = AgentRouter(
    agents=[
        Agent(provider="anthropic", model="claude-sonnet", timeout=15.0),
        Agent(provider="openai", model="gpt-4o", timeout=15.0),
    ],
    strategy="fallback",
)
# Total max wait: 30s (15 + 15) instead of 240s (120 + 120)
```

### 5. Consider Stream Fallback Limitations

Streaming fallback only triggers on connection errors, not mid-stream failures:

```python
# For critical streaming use cases, consider:
# 1. Use reliable primary provider
# 2. Implement application-level retry
# 3. Use non-streaming with chunked display
```
