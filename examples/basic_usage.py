"""
Basic Agent usage examples.

This file demonstrates the core features of the Agent library.
"""

from agent import Agent

# ============================================================================
# Basic Text Generation
# ============================================================================

def basic_generation():
    """Simple text generation with default settings."""
    agent = Agent(
        provider="openai",
        model="gpt-4o",
    )

    response = agent.run("What is the capital of France?")
    print(f"Response: {response.text}")
    print(f"Tokens used: {response.usage.total_tokens if response.usage else 'N/A'}")


# ============================================================================
# With System Prompt
# ============================================================================

def with_system_prompt():
    """Using a system prompt to set behavior."""
    agent = Agent(
        provider="anthropic",
        model="claude-sonnet",
        default_system="You are a helpful coding assistant. Be concise.",
    )

    response = agent.run("Explain what a Python decorator is")
    print(response.text)


# ============================================================================
# Streaming Response
# ============================================================================

def streaming_response():
    """Stream responses for real-time output."""
    agent = Agent(
        provider="openai",
        model="gpt-4o",
    )

    print("Streaming: ", end="", flush=True)
    for event in agent.stream("Write a haiku about programming"):
        if event.type == "text_delta" and event.text:
            print(event.text, end="", flush=True)
    print()  # Newline at end


# ============================================================================
# Provider Switching
# ============================================================================

def provider_switching():
    """Same code, different providers."""
    providers = [
        ("openai", "gpt-4o"),
        ("anthropic", "claude-sonnet"),
    ]

    prompt = "What is 2+2? Answer with just the number."

    for provider, model in providers:
        try:
            agent = Agent(provider=provider, model=model)
            response = agent.run(prompt)
            print(f"{provider}/{model}: {response.text}")
        except Exception as e:
            print(f"{provider}/{model}: Error - {e}")


# ============================================================================
# Temperature Control
# ============================================================================

def temperature_control():
    """Control randomness with temperature."""
    agent = Agent(
        provider="openai",
        model="gpt-4o",
    )

    prompt = "Generate a creative name for a coffee shop"

    # Low temperature - more deterministic
    response_low = agent.run(prompt, temperature=0.1)
    print(f"Low temp (0.1): {response_low.text}")

    # High temperature - more creative
    response_high = agent.run(prompt, temperature=0.9)
    print(f"High temp (0.9): {response_high.text}")


# ============================================================================
# Using with_config for Variations
# ============================================================================

def config_variations():
    """Create agent variations with with_config."""
    base_agent = Agent(
        provider="openai",
        model="gpt-4o",
        temperature=0.7,
    )

    # Create a more creative variant
    creative_agent = base_agent.with_config(temperature=1.0)

    # Create a more precise variant
    precise_agent = base_agent.with_config(temperature=0.1)

    prompt = "Describe the moon in one sentence"

    print(f"Base: {base_agent.run(prompt).text}")
    print(f"Creative: {creative_agent.run(prompt).text}")
    print(f"Precise: {precise_agent.run(prompt).text}")


if __name__ == "__main__":
    # Run whichever example you want to test
    # Uncomment the one you want to run:

    # basic_generation()
    # with_system_prompt()
    # streaming_response()
    # provider_switching()
    # temperature_control()
    # config_variations()

    print("Uncomment an example function to run it!")
