# Streaming

Agent provides a normalized streaming interface that works consistently across all providers. Stream responses for real-time output and better user experience.

## Basic Streaming

### Synchronous

```python
from agent import Agent

agent = Agent(provider="anthropic", model="claude-sonnet")

for event in agent.stream("Write a short story about a robot"):
    if event.type == "text_delta":
        print(event.text, end="", flush=True)
print()  # Final newline
```

### Asynchronous

```python
import asyncio

async def main():
    agent = Agent(provider="openai", model="gpt-4o")
    
    async for event in await agent.stream_async("Explain quantum physics"):
        if event.type == "text_delta":
            print(event.text, end="", flush=True)
    print()

asyncio.run(main())
```

## Stream Events

Agent normalizes events across all providers:

### Event Types

```python
from agent import StreamEvent

for event in agent.stream("Hello"):
    match event.type:
        case "message_start":
            print("Response started")
        
        case "text_delta":
            print(event.text, end="")
        
        case "tool_call_start":
            print(f"Calling tool: {event.tool_call.name}")
        
        case "tool_call_delta":
            print(f"Tool args: {event.tool_call_delta}")
        
        case "usage":
            print(f"Tokens: {event.usage.total_tokens}")
        
        case "message_end":
            print("Response complete")
        
        case "error":
            print(f"Error: {event.error}")
```

### Event Properties

```python
event.type          # StreamEventType
event.text          # str | None - for text_delta
event.tool_call     # ToolCall | None - for tool_call_start
event.tool_call_delta  # dict | None - for tool_call_delta
event.usage         # Usage | None - for usage/message_end
event.error         # str | None - for error
event.raw           # Any - provider's raw event
```

## StreamResponse

The stream response accumulates data as you iterate:

```python
stream = agent.stream("Write a poem")

# Iterate through events
for event in stream:
    if event.type == "text_delta":
        print(event.text, end="")

# After iteration, access accumulated data
print(f"\nFull text: {stream.text}")
print(f"Tool calls: {stream.tool_calls}")
print(f"Usage: {stream.usage}")
```

### Collect All Events

```python
# Consume stream and get final state
stream = agent.stream("Hello").collect()
print(stream.text)  # Full response text
```

## Streaming with Tools

Tool calls are streamed as events:

```python
from agent import Agent, tool

@tool
def calculate(expression: str) -> str:
    return str(eval(expression))

agent = Agent(
    provider="openai",
    model="gpt-4o",
    tools=[calculate],
)

for event in agent.stream("What's 25 * 4 + 10?"):
    match event.type:
        case "text_delta":
            print(event.text, end="")
        case "tool_call_start":
            print(f"\n[Calling {event.tool_call.name}...]")
        case "message_end":
            print("\n[Done]")
```

## Session Streaming

Stream within multi-turn conversations:

```python
session = agent.session()

session.run("My name is Alice")

# Stream response, history updated after completion
for event in session.stream("Tell me a story about someone with my name"):
    if event.type == "text_delta":
        print(event.text, end="")

# History now includes the streamed response
print(f"\nMessages: {len(session.messages)}")
```

## Router Streaming

Stream with fallback support:

```python
from agent import Agent, AgentRouter

router = AgentRouter(
    agents=[
        Agent(provider="anthropic", model="claude-sonnet"),
        Agent(provider="openai", model="gpt-4o"),
    ],
    strategy="fallback",
)

# Falls back if first provider fails to connect
for event in router.stream("Write a haiku"):
    if event.type == "text_delta":
        print(event.text, end="")
```

## Web Streaming (FastAPI)

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from agent import Agent

app = FastAPI()
agent = Agent(provider="openai", model="gpt-4o")

@app.post("/chat")
async def chat(prompt: str):
    async def generate():
        async for event in await agent.stream_async(prompt):
            if event.type == "text_delta":
                yield f"data: {event.text}\n\n"
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )
```

## WebSocket Streaming

```python
from fastapi import FastAPI, WebSocket
from agent import Agent

app = FastAPI()
agent = Agent(provider="anthropic", model="claude-sonnet")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    while True:
        prompt = await websocket.receive_text()
        
        async for event in await agent.stream_async(prompt):
            if event.type == "text_delta":
                await websocket.send_text(event.text)
            elif event.type == "message_end":
                await websocket.send_text("[END]")
```

## Progress Indicators

Show progress during streaming:

```python
import sys

stream = agent.stream("Write a long analysis")
char_count = 0

for event in stream:
    if event.type == "text_delta":
        char_count += len(event.text)
        sys.stdout.write(event.text)
        sys.stdout.flush()
    elif event.type == "message_end":
        print(f"\n\n[{char_count} characters, {stream.usage.total_tokens} tokens]")
```

## Error Handling

```python
from agent.errors import AgentError

try:
    for event in agent.stream("Hello"):
        if event.type == "error":
            print(f"Stream error: {event.error}")
            break
        if event.type == "text_delta":
            print(event.text, end="")
except AgentError as e:
    print(f"Connection error: {e}")
```

## Configuration

```python
response = agent.stream(
    "Write something",
    temperature=0.9,
    max_tokens=500,
    system="Be creative and verbose.",
)
```

## Best Practices

### 1. Always Handle Events

```python
# Good - handle relevant events
for event in agent.stream(prompt):
    if event.type == "text_delta":
        process_text(event.text)
    elif event.type == "error":
        handle_error(event.error)

# Bad - assume only text
for event in agent.stream(prompt):
    print(event.text)  # May be None for non-text events
```

### 2. Use Flush for Real-time Output

```python
# Good - immediate output
print(event.text, end="", flush=True)

# Bad - may buffer
print(event.text, end="")
```

### 3. Clean Up Resources

```python
# Async context manager for cleanup
async with agent.stream_async(prompt) as stream:
    async for event in stream:
        process(event)
```

### 4. Handle Connection Failures

```python
import time
from agent.errors import ProviderError

def stream_with_retry(agent, prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            for event in agent.stream(prompt):
                yield event
            return
        except ProviderError:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise
```

## Next Steps

- [Sessions](sessions.md) - Streaming in conversations
- [Tools](tools.md) - Streaming tool calls
- [Routing](routing.md) - Streaming with fallback
