"""
Session examples.

Demonstrates multi-turn conversations with memory.
"""

from agent import Agent

# ============================================================================
# Basic Session
# ============================================================================


def basic_session():
    """Simple multi-turn conversation."""
    agent = Agent(
        provider="openai",
        model="gpt-4o",
    )

    session = agent.session()

    # First turn - introduce information
    r1 = session.run("My name is Alice and I'm a software engineer")
    print(f"Assistant: {r1.text}")

    # Second turn - the model remembers
    r2 = session.run("What's my name and profession?")
    print(f"Assistant: {r2.text}")

    # Check history
    print(f"\nSession has {len(session.messages)} messages")


# ============================================================================
# Session with System Prompt
# ============================================================================


def session_with_system():
    """Session with a custom system prompt."""
    agent = Agent(
        provider="anthropic",
        model="claude-sonnet",
    )

    session = agent.session(
        system="You are a friendly cooking assistant. Give concise recipe suggestions."
    )

    r1 = session.run("I have chicken, garlic, and lemon")
    print(f"Assistant: {r1.text}")

    r2 = session.run("What about a side dish?")
    print(f"Assistant: {r2.text}")


# ============================================================================
# Forking Sessions
# ============================================================================


def forking_sessions():
    """Create session branches."""
    agent = Agent(
        provider="openai",
        model="gpt-4o",
    )

    # Start main session
    main = agent.session()
    main.run("We're planning a trip to Japan")
    main.run("I'm interested in traditional culture")

    # Fork to explore different options
    tokyo_branch = main.fork()
    kyoto_branch = main.fork()

    # Each branch continues independently
    tokyo_response = tokyo_branch.run("What should I see in Tokyo?")
    print(f"Tokyo branch: {tokyo_response.text[:100]}...")

    kyoto_response = kyoto_branch.run("What should I see in Kyoto?")
    print(f"Kyoto branch: {kyoto_response.text[:100]}...")

    # Main session is unchanged
    print(f"\nMain session messages: {len(main.messages)}")
    print(f"Tokyo branch messages: {len(tokyo_branch.messages)}")
    print(f"Kyoto branch messages: {len(kyoto_branch.messages)}")


# ============================================================================
# Session Serialization
# ============================================================================


def session_serialization():
    """Save and restore sessions."""
    agent = Agent(
        provider="openai",
        model="gpt-4o",
    )

    # Create and populate session
    session = agent.session(system="You are a math tutor")
    session.run("Let's work on algebra")
    session.run("What's the quadratic formula?")

    # Serialize to dict (could be saved to JSON file/database)
    session_data = session.to_dict()
    print(f"Saved session with {len(session_data['messages'])} messages")

    # Later: restore session
    from agent.session import Session

    restored = Session.from_dict(session_data, agent)

    # Continue the conversation
    response = restored.run("Can you show me an example?")
    print(f"Restored and continued: {response.text[:100]}...")


# ============================================================================
# Streaming in Sessions
# ============================================================================


def streaming_session():
    """Stream responses in a session."""
    agent = Agent(
        provider="openai",
        model="gpt-4o",
    )

    session = agent.session()

    # First message
    print("User: Tell me a joke")
    print("Assistant: ", end="", flush=True)
    for event in session.stream("Tell me a joke"):
        if event.type == "text_delta" and event.text:
            print(event.text, end="", flush=True)
    print("\n")

    # Follow-up (session remembers)
    print("User: Explain why it's funny")
    print("Assistant: ", end="", flush=True)
    for event in session.stream("Explain why it's funny"):
        if event.type == "text_delta" and event.text:
            print(event.text, end="", flush=True)
    print()


# ============================================================================
# Interactive Chat Loop
# ============================================================================


def interactive_chat():
    """Simple interactive chat loop."""
    agent = Agent(
        provider="openai",
        model="gpt-4o",
    )

    session = agent.session(system="You are a helpful assistant. Be concise.")

    print("Chat started. Type 'quit' to exit, 'clear' to clear history.")
    print("-" * 50)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print("Goodbye!")
            break

        if user_input.lower() == "clear":
            session.clear()
            print("[History cleared]")
            continue

        if user_input.lower() == "history":
            print(f"[{len(session.messages)} messages in history]")
            continue

        # Stream the response
        print("Assistant: ", end="", flush=True)
        for event in session.stream(user_input):
            if event.type == "text_delta" and event.text:
                print(event.text, end="", flush=True)
        print()


if __name__ == "__main__":
    # Uncomment to run:
    # basic_session()
    # session_with_system()
    # forking_sessions()
    # session_serialization()
    # streaming_session()
    # interactive_chat()

    print("Uncomment an example function to run it!")
