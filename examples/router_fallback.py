"""
Router and fallback examples.

Demonstrates multi-agent routing strategies.
"""

from agent import Agent, AgentRouter

# ============================================================================
# Basic Fallback
# ============================================================================


def basic_fallback():
    """Automatic failover between providers."""
    # Primary: Anthropic, Fallback: OpenAI
    router = AgentRouter(
        agents=[
            Agent(provider="anthropic", model="claude-sonnet"),
            Agent(provider="openai", model="gpt-4o"),
        ],
        strategy="fallback",
    )

    # If Anthropic fails (rate limit, error), automatically tries OpenAI
    response = router.run("Explain quantum computing in one sentence")
    print(f"Response from {response.provider}: {response.text}")


# ============================================================================
# Round Robin Load Balancing
# ============================================================================


def round_robin():
    """Distribute load across providers."""
    router = AgentRouter(
        agents=[
            Agent(provider="openai", model="gpt-4o"),
            Agent(provider="anthropic", model="claude-sonnet"),
        ],
        strategy="round_robin",
    )

    # Requests alternate between providers
    for i in range(4):
        response = router.run(f"Say '{i}'")
        print(f"Request {i}: {response.provider} said {response.text}")


# ============================================================================
# Cheapest Provider
# ============================================================================


def cheapest_first():
    """Route to cheapest available provider."""
    router = AgentRouter(
        agents=[
            Agent(provider="openai", model="gpt-4o"),  # More expensive
            Agent(provider="openai", model="gpt-4o-mini"),  # Cheaper
            Agent(provider="anthropic", model="claude-haiku"),  # Cheapest
        ],
        strategy="cheapest",
    )

    response = router.run("What is 2+2?")
    print(f"Routed to {response.provider}/{response.model}: {response.text}")


# ============================================================================
# Streaming with Fallback
# ============================================================================


def streaming_fallback():
    """Stream with automatic failover."""
    router = AgentRouter(
        agents=[
            Agent(provider="anthropic", model="claude-sonnet"),
            Agent(provider="openai", model="gpt-4o"),
        ],
        strategy="fallback",
    )

    print("Streaming: ", end="", flush=True)
    for event in router.stream("Write a haiku about code"):
        if event.type == "text_delta" and event.text:
            print(event.text, end="", flush=True)
    print()


# ============================================================================
# Structured Output with Fallback
# ============================================================================


def json_fallback():
    """Structured output with fallback."""
    from pydantic import BaseModel

    class Summary(BaseModel):
        title: str
        points: list[str]

    router = AgentRouter(
        agents=[
            Agent(provider="anthropic", model="claude-sonnet"),
            Agent(provider="openai", model="gpt-4o"),
        ],
        strategy="fallback",
    )

    response = router.json(
        "Summarize the benefits of exercise",
        schema=Summary,
    )

    summary: Summary = response.output
    print(f"Title: {summary.title}")
    print(f"Points: {summary.points}")


# ============================================================================
# Custom Routing Strategy
# ============================================================================


def custom_routing():
    """Custom routing logic."""
    from agent.messages import AgentRequest
    from agent.router import RouteResult

    def my_router(request: AgentRequest, agents: list[Agent]) -> RouteResult:
        """Route based on input content."""
        input_text = request.input or ""

        # Use Claude for creative tasks
        if any(word in input_text.lower() for word in ["creative", "story", "poem"]):
            return RouteResult(
                agent=next(a for a in agents if a.provider == "anthropic"),
                reason="Creative task -> Claude",
            )

        # Use GPT for analytical tasks
        if any(word in input_text.lower() for word in ["analyze", "calculate", "data"]):
            return RouteResult(
                agent=next(a for a in agents if a.provider == "openai"),
                reason="Analytical task -> GPT",
            )

        # Default to first agent
        return RouteResult(agent=agents[0], reason="Default")

    router = AgentRouter(
        agents=[
            Agent(provider="openai", model="gpt-4o"),
            Agent(provider="anthropic", model="claude-sonnet"),
        ],
        strategy="custom",
        custom_router=my_router,
    )

    # Test routing
    r1 = router.run("Write a creative story about a robot")
    print(f"Creative request -> {r1.provider}")

    r2 = router.run("Analyze this data: [1, 2, 3, 4, 5]")
    print(f"Analytical request -> {r2.provider}")


# ============================================================================
# Production Pattern
# ============================================================================


def production_pattern():
    """
    Production-ready pattern with multiple fallback tiers.
    """
    # Tier 1: Premium models
    # Tier 2: Standard models
    # Tier 3: Budget models
    router = AgentRouter(
        agents=[
            # Tier 1 - Best quality
            Agent(provider="anthropic", model="claude-sonnet"),
            # Tier 2 - Good quality
            Agent(provider="openai", model="gpt-4o"),
            # Tier 3 - Budget fallback
            Agent(provider="openai", model="gpt-4o-mini"),
        ],
        strategy="fallback",
    )

    try:
        response = router.run(
            "Explain machine learning",
            temperature=0.7,
        )
        print(f"Success via {response.provider}/{response.model}")
        print(f"Response: {(response.text or '')[:100]}...")
        if response.cost_estimate:
            print(f"Estimated cost: ${response.cost_estimate:.6f}")
    except Exception as e:
        print(f"All providers failed: {e}")


if __name__ == "__main__":
    # Uncomment to run:
    # basic_fallback()
    # round_robin()
    # cheapest_first()
    # streaming_fallback()
    # json_fallback()
    # custom_routing()
    # production_pattern()

    print("Uncomment an example function to run it!")
