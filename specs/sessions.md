# Sessions

Sessions provide multi-turn conversation management with automatic history tracking. Instead of manually managing message arrays, use sessions for natural conversational interactions.

## Overview

```python
from agent import Agent

agent = Agent(provider="openai", model="gpt-4o")
session = agent.session()

session.run("My name is Alice")
session.run("I'm a software engineer")

response = session.run("What do you know about me?")
# Agent remembers: Alice, software engineer
```

## Creating Sessions

### Basic Session

```python
session = agent.session()
```

### With Custom ID

```python
session = agent.session(session_id="user-123-conversation-456")
```

### With System Prompt

```python
session = agent.session(
    system="You are a helpful cooking assistant. Be concise and practical."
)
```

### With Initial Messages

```python
from agent.session import Session
from agent.messages import Message

session = Session(
    agent=agent,
    messages=[
        Message.user("Hello"),
        Message.assistant("Hi! How can I help?"),
    ],
)
```

## Session Methods

### run() / run_async()

Send a message and get a response. History is automatically updated.

```python
response = session.run("What's the weather?")
print(response.text)

# Async
response = await session.run_async("What's the weather?")
```

### stream() / stream_async()

Stream responses in real-time:

```python
for event in session.stream("Tell me a story"):
    if event.type == "text_delta":
        print(event.text, end="", flush=True)

# History updated after stream is consumed
```

### json()

Get structured output in session context:

```python
from pydantic import BaseModel

class Summary(BaseModel):
    points: list[str]
    
response = session.json(
    "Summarize our conversation",
    schema=Summary,
)
```

### history()

Get the message history (returns a copy):

```python
messages = session.history()
for msg in messages:
    print(f"{msg.role}: {msg.text[:50]}...")
```

### clear()

Clear the message history:

```python
session.clear()
# History is now empty, but system prompt is preserved
```

### fork()

Create a new session with copied history:

```python
# Original conversation
session.run("Let's plan a trip to Japan")
session.run("I like traditional culture")

# Fork to explore different options
tokyo_session = session.fork()
kyoto_session = session.fork()

# Each fork continues independently
tokyo_session.run("What to do in Tokyo?")
kyoto_session.run("What to do in Kyoto?")

# Original session is unchanged
```

### add_message()

Manually add a message:

```python
from agent.messages import Message

session.add_message(Message.user("Context info"))
session.add_message(Message.assistant("Acknowledged"))
```

## Session Properties

```python
session.session_id  # str: Unique identifier
session.system      # str | None: System prompt
session.messages    # list[Message]: Current history (read-only)
```

## Serialization

### Save Session

```python
# To dictionary (for JSON storage)
data = session.to_dict()

# Save to file
import json
with open("session.json", "w") as f:
    json.dump(data, f)
```

### Restore Session

```python
from agent.session import Session

# Load from file
with open("session.json") as f:
    data = json.load(f)

# Restore
session = Session.from_dict(data, agent)

# Continue conversation
response = session.run("Where were we?")
```

## Session Stores

For persistent storage beyond JSON files:

### In-Memory Store

```python
from agent.stores import InMemoryStore

store = InMemoryStore()

# Save
store.save(session.session_id, session.to_dict())

# Load
data = store.load(session.session_id)
if data:
    session = Session.from_dict(data, agent)
```

### SQLite Store

```python
from agent.stores import SQLiteStore

store = SQLiteStore("sessions.db")

# Save
store.save(session.session_id, session.to_dict())

# List all sessions
session_ids = store.list_sessions()

# Load specific session
data = store.load("session-123")

# Delete
store.delete("session-123")
```

## Patterns

### Interactive Chat Loop

```python
def chat_loop():
    agent = Agent(provider="openai", model="gpt-4o")
    session = agent.session(system="You are a helpful assistant.")
    
    print("Chat started. Type 'quit' to exit.")
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == "quit":
            break
        
        if user_input.lower() == "clear":
            session.clear()
            print("[History cleared]")
            continue
        
        print("Assistant: ", end="", flush=True)
        for event in session.stream(user_input):
            if event.type == "text_delta":
                print(event.text, end="", flush=True)
        print()
```

### Context-Aware Assistance

```python
def coding_assistant(codebase_context: str):
    agent = Agent(
        provider="anthropic",
        model="claude-sonnet",
        tools=[search_code, read_file, run_tests],
    )
    
    session = agent.session(
        system=f"""You are a coding assistant for this project.
        
Project context:
{codebase_context}

Help the user understand and modify the code."""
    )
    
    return session
```

### Branching Conversations

```python
# Main conversation
main = agent.session()
main.run("I need to decide between Python and JavaScript")
main.run("I'm building a web app")

# Explore Python path
python_branch = main.fork()
python_branch.run("Tell me more about using Python")
python_recommendation = python_branch.run("What frameworks do you recommend?")

# Explore JavaScript path
js_branch = main.fork()
js_branch.run("Tell me more about using JavaScript")
js_recommendation = js_branch.run("What frameworks do you recommend?")

# Compare
print("Python path:", python_recommendation.text[:100])
print("JS path:", js_recommendation.text[:100])
```

### Session with Tools

```python
@tool
def get_user_data(user_id: str) -> str:
    """Get data for a user."""
    return '{"name": "Alice", "plan": "premium"}'

agent = Agent(
    provider="openai",
    model="gpt-4o",
    tools=[get_user_data],
)

session = agent.session()
session.run("Look up user 123")  # Agent calls get_user_data
session.run("What plan are they on?")  # Agent remembers the result
```

### Persistent Sessions (Web App)

```python
from flask import Flask, request, session as flask_session
from agent.stores import SQLiteStore

app = Flask(__name__)
store = SQLiteStore("chat_sessions.db")

@app.route("/chat", methods=["POST"])
def chat():
    user_id = flask_session["user_id"]
    message = request.json["message"]
    
    # Load or create session
    session_data = store.load(user_id)
    if session_data:
        session = Session.from_dict(session_data, agent)
    else:
        session = agent.session()
    
    # Process message
    response = session.run(message)
    
    # Save session
    store.save(user_id, session.to_dict())
    
    return {"response": response.text}
```

## Best Practices

### 1. Use Meaningful Session IDs

```python
# Good - includes user context
session = agent.session(session_id=f"user-{user_id}-{conversation_type}")

# Bad - random or unclear
session = agent.session(session_id="abc123")
```

### 2. Set Clear System Prompts

```python
session = agent.session(
    system="""You are a customer support agent for TechCorp.
    
Guidelines:
- Be helpful and professional
- Only answer questions about our products
- Escalate complex issues to human support

Products: Widget Pro, Widget Lite, Widget Enterprise"""
)
```

### 3. Handle Long Conversations

Sessions can grow large. Consider:

```python
# Check history length
if len(session.messages) > 50:
    # Summarize and start fresh
    summary = session.run("Summarize our conversation so far")
    
    new_session = agent.session(
        system=f"{original_system}\n\nPrevious conversation summary:\n{summary.text}"
    )
    session = new_session
```

### 4. Fork for Exploration

```python
# Before making important decisions
checkpoint = session.fork()

# Try something
session.run("Let's try approach A")
result = session.run("Did that work?")

if "failed" in result.text.lower():
    # Go back to checkpoint
    session = checkpoint
    session.run("Let's try approach B instead")
```

### 5. Test Session Behavior

```python
from agent.testing import create_test_agent, FakeResponse

def test_session_memory():
    agent, provider = create_test_agent()
    provider.set_responses([
        FakeResponse(text="Nice to meet you, Alice!"),
        FakeResponse(text="Your name is Alice."),
    ])
    
    session = agent.session()
    session.run("My name is Alice")
    response = session.run("What's my name?")
    
    # Verify history was passed
    request = provider.get_last_request()
    assert len(request.messages) == 2
    assert "Alice" in request.messages[0].content
```

## Troubleshooting

### Agent Forgets Context

1. Check if history is being passed correctly
2. Verify session isn't being recreated each call
3. Check for history limits in provider

### Slow Response Times

Long history increases latency and cost:

```python
# Monitor history size
print(f"Messages in history: {len(session.messages)}")

# Estimate tokens
total_chars = sum(len(m.text) for m in session.messages)
approx_tokens = total_chars // 4
print(f"Approximate tokens: {approx_tokens}")
```

### Serialization Errors

Ensure all message content is serializable:

```python
# Good
session.add_message(Message.user("Hello"))

# Bad - binary data in message
session.add_message(Message.user(binary_image_data))  # Won't serialize
```
