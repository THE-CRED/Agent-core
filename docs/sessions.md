# Sessions

Sessions enable multi-turn conversations with automatic history management. The LLM maintains context across messages without manual message handling.

## Basic Usage

```python
from agent import Agent

agent = Agent(provider="openai", model="gpt-4o")

# Create a session
session = agent.session()

# Multi-turn conversation
session.run("My name is Alice")
session.run("I'm a software engineer")
response = session.run("What do you know about me?")

print(response.text)
# "Based on our conversation, I know that your name is Alice 
#  and you work as a software engineer."
```

## Session Configuration

### Custom Session ID

```python
# Track sessions by ID for persistence
session = agent.session(session_id="user-123-conv-456")
print(session.session_id)  # "user-123-conv-456"
```

### System Prompt

```python
session = agent.session(
    system="You are a helpful coding assistant. Always provide code examples."
)

response = session.run("How do I read a file in Python?")
# Response will include code examples
```

### Pre-populated History

```python
from agent import Message

history = [
    Message.user("What's the capital of France?"),
    Message.assistant("The capital of France is Paris."),
]

session = agent.session(messages=history)
response = session.run("What about Germany?")
# Session continues from existing history
```

## Session Methods

### run() - Synchronous

```python
response = session.run(
    "Explain this concept",
    temperature=0.7,
    max_tokens=500,
)
```

### run_async() - Asynchronous

```python
response = await session.run_async("Explain async/await")
```

### stream() - Streaming

```python
for event in session.stream("Write a story"):
    if event.type == "text_delta":
        print(event.text, end="", flush=True)
# History is updated after stream completes
```

### stream_async() - Async Streaming

```python
async for event in await session.stream_async("Write a poem"):
    if event.type == "text_delta":
        print(event.text, end="", flush=True)
```

### json() - Structured Output

```python
from pydantic import BaseModel

class Summary(BaseModel):
    key_points: list[str]
    conclusion: str

response = session.json(
    "Summarize our conversation",
    schema=Summary
)
print(response.output.key_points)
```

## History Management

### Access History

```python
# Get all messages
messages = session.messages  # Returns a copy
print(f"Message count: {len(messages)}")

for msg in messages:
    print(f"[{msg.role}]: {msg.text[:50]}...")
```

### Clear History

```python
session.clear()
print(len(session.messages))  # 0
```

### Add Messages Manually

```python
session.add_message(Message.user("Custom user message"))
session.add_message(Message.assistant("Custom assistant response"))
```

## Session Forking

Create a branch from the current conversation:

```python
# Original session
session = agent.session()
session.run("We're planning a trip to Japan")
session.run("I want to visit Tokyo and Kyoto")

# Fork for alternative planning
alt_session = session.fork(session_id="alternative-plan")
alt_session.run("Actually, let's focus on Osaka instead")

# Original session is unchanged
response = session.run("What cities are we visiting?")
# Still knows about Tokyo and Kyoto

# Forked session has diverged
response = alt_session.run("What cities are we visiting?")
# Knows about Osaka
```

## Serialization

### Save Session State

```python
# Serialize to dictionary
data = session.to_dict()

# Save to file
import json
with open("session.json", "w") as f:
    json.dump(data, f)

# Or save to database
db.save_session(session.session_id, data)
```

### Restore Session State

```python
# Load from file
with open("session.json") as f:
    data = json.load(f)

# Restore session
restored = Session.from_dict(data, agent)

# Continue conversation
response = restored.run("Where were we?")
```

## Session Stores

Use pluggable storage backends for persistence:

### Memory Store (Default)

```python
from agent.stores import MemoryStore

store = MemoryStore()
store.save("session-1", session.to_dict())
data = store.load("session-1")
```

### SQLite Store

```python
from agent.stores import SQLiteStore

store = SQLiteStore("sessions.db")
store.save(session.session_id, session.to_dict())

# Later...
data = store.load(session.session_id)
if data:
    session = Session.from_dict(data, agent)
```

### Custom Store

```python
from agent.stores import SessionStore

class RedisStore(SessionStore):
    def __init__(self, redis_client):
        self.redis = redis_client
    
    def save(self, session_id: str, data: dict) -> None:
        self.redis.set(f"session:{session_id}", json.dumps(data))
    
    def load(self, session_id: str) -> dict | None:
        data = self.redis.get(f"session:{session_id}")
        return json.loads(data) if data else None
    
    def delete(self, session_id: str) -> bool:
        return self.redis.delete(f"session:{session_id}") > 0
    
    def list_sessions(self) -> list[str]:
        keys = self.redis.keys("session:*")
        return [k.replace("session:", "") for k in keys]
```

## Sessions with Tools

Tools work seamlessly with sessions:

```python
from agent import Agent, tool

@tool
def get_user_orders(user_id: str) -> str:
    """Get orders for a user."""
    return json.dumps([
        {"id": "1", "item": "Laptop", "status": "shipped"},
        {"id": "2", "item": "Mouse", "status": "delivered"},
    ])

agent = Agent(
    provider="openai",
    model="gpt-4o",
    tools=[get_user_orders],
)

session = agent.session(
    system="You are a customer service assistant."
)

session.run("I'm user-123, what are my recent orders?")
# Agent calls get_user_orders("user-123")

response = session.run("When will order 1 arrive?")
# Agent remembers previous context and order details
```

## Best Practices

### 1. Use System Prompts

```python
session = agent.session(
    system="""You are a technical support agent for Acme Software.
    - Be helpful and patient
    - Ask clarifying questions when needed
    - Provide step-by-step solutions
    - Escalate complex issues to human support"""
)
```

### 2. Manage History Length

```python
# For very long conversations, consider summarization
if len(session.messages) > 20:
    # Get summary
    summary_response = session.json(
        "Summarize our conversation so far in 3 bullet points",
        schema=ConversationSummary
    )
    
    # Create new session with summary
    new_session = agent.session(
        system=f"Previous conversation summary: {summary_response.output}"
    )
```

### 3. Handle Errors Gracefully

```python
from agent.errors import AgentError

try:
    response = session.run(user_input)
except AgentError as e:
    # Don't add failed interaction to history
    print(f"Error: {e}. Please try again.")
```

### 4. Use Session IDs for Tracking

```python
import uuid

def create_support_session(user_id: str) -> Session:
    session_id = f"{user_id}-{uuid.uuid4().hex[:8]}"
    return agent.session(
        session_id=session_id,
        system="You are a support agent."
    )
```

## Next Steps

- [Streaming](streaming.md) - Real-time response streaming
- [Tools](tools.md) - Add capabilities to sessions
- [Routing](routing.md) - Session-aware routing
