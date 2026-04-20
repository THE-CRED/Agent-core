"""
Session management for multi-turn conversations.
"""

import uuid
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from agent.messages import AgentRequest, Message
from agent.response import AgentResponse
from agent.stream import AsyncStreamResponse, StreamResponse

if TYPE_CHECKING:
    from agent.agent import Agent


class Session:
    """
    A session for multi-turn conversation.

    Sessions manage message history automatically, allowing natural
    conversational interactions without manual message management.

    Example:
        ```python
        agent = Agent(provider="openai", model="gpt-4o")
        session = agent.session()

        session.run("My name is Alice")
        response = session.run("What's my name?")
        print(response.text)  # "Your name is Alice"
        ```
    """

    def __init__(
        self,
        agent: "Agent",
        session_id: str | None = None,
        system: str | None = None,
        messages: list[Message] | None = None,
    ):
        """
        Initialize a session.

        Args:
            agent: The agent to use for this session
            session_id: Optional session identifier
            system: System prompt for this session
            messages: Initial message history
        """
        self._agent = agent
        self._session_id = session_id or str(uuid.uuid4())
        self._system = system
        self._messages: list[Message] = messages or []

    @property
    def session_id(self) -> str:
        """Get the session ID."""
        return self._session_id

    @property
    def system(self) -> str | None:
        """Get the system prompt."""
        return self._system

    @property
    def messages(self) -> list[Message]:
        """Get the message history (read-only copy)."""
        return list(self._messages)

    def run(
        self,
        input: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentResponse:
        """
        Send a message and get a response.

        The input and response are automatically added to history.

        Args:
            input: User message
            temperature: Sampling temperature (overrides default)
            max_tokens: Max tokens (overrides default)
            metadata: Request metadata

        Returns:
            AgentResponse with the model's response
        """
        # Build request with history
        request = AgentRequest(
            input=input,
            messages=self._messages.copy(),
            system=self._system,
            temperature=temperature or self._agent.config.temperature,
            max_tokens=max_tokens or self._agent.config.max_tokens,
            metadata=metadata or {},
            session_id=self._session_id,
        )

        # Execute
        response = self._agent._runtime.run(request)

        # Update history
        self._messages.append(Message.user(input))
        if response.has_tool_calls:
            # Add assistant message with tool calls
            self._messages.append(
                Message.assistant(
                    content=response.text or "",
                    tool_calls=[tc.to_dict() for tc in response.tool_calls],
                )
            )
        else:
            self._messages.append(Message.assistant(content=response.text or ""))

        return response

    async def run_async(
        self,
        input: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentResponse:
        """
        Send a message and get a response asynchronously.

        Args:
            input: User message
            temperature: Sampling temperature (overrides default)
            max_tokens: Max tokens (overrides default)
            metadata: Request metadata

        Returns:
            AgentResponse with the model's response
        """
        request = AgentRequest(
            input=input,
            messages=self._messages.copy(),
            system=self._system,
            temperature=temperature or self._agent.config.temperature,
            max_tokens=max_tokens or self._agent.config.max_tokens,
            metadata=metadata or {},
            session_id=self._session_id,
        )

        response = await self._agent._runtime.run_async(request)

        self._messages.append(Message.user(input))
        if response.has_tool_calls:
            self._messages.append(
                Message.assistant(
                    content=response.text or "",
                    tool_calls=[tc.to_dict() for tc in response.tool_calls],
                )
            )
        else:
            self._messages.append(Message.assistant(content=response.text or ""))

        return response

    def stream(
        self,
        input: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> StreamResponse:
        """
        Send a message and stream the response.

        Note: History is updated after stream is consumed.

        Args:
            input: User message
            temperature: Sampling temperature
            max_tokens: Max tokens
            metadata: Request metadata

        Returns:
            StreamResponse iterator
        """
        request = AgentRequest(
            input=input,
            messages=self._messages.copy(),
            system=self._system,
            temperature=temperature or self._agent.config.temperature,
            max_tokens=max_tokens or self._agent.config.max_tokens,
            metadata=metadata or {},
            session_id=self._session_id,
        )

        stream = self._agent._runtime.stream(request)

        # Wrap to capture final state
        return _SessionStreamResponse(
            stream=stream,
            session=self,
            user_input=input,
        )

    async def stream_async(
        self,
        input: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AsyncStreamResponse:
        """
        Send a message and stream the response asynchronously.

        Args:
            input: User message
            temperature: Sampling temperature
            max_tokens: Max tokens
            metadata: Request metadata

        Returns:
            AsyncStreamResponse iterator
        """
        request = AgentRequest(
            input=input,
            messages=self._messages.copy(),
            system=self._system,
            temperature=temperature or self._agent.config.temperature,
            max_tokens=max_tokens or self._agent.config.max_tokens,
            metadata=metadata or {},
            session_id=self._session_id,
        )

        stream = await self._agent._runtime.stream_async(request)

        return _AsyncSessionStreamResponse(
            stream=stream,
            session=self,
            user_input=input,
        )

    def json(
        self,
        input: str,
        *,
        schema: type[BaseModel] | dict[str, Any],
        temperature: float | None = None,
        max_tokens: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentResponse:
        """
        Send a message expecting structured JSON output.

        Args:
            input: User message
            schema: Pydantic model or JSON schema
            temperature: Sampling temperature
            max_tokens: Max tokens
            metadata: Request metadata

        Returns:
            AgentResponse with parsed output
        """
        request = AgentRequest(
            input=input,
            messages=self._messages.copy(),
            system=self._system,
            temperature=temperature or self._agent.config.temperature,
            max_tokens=max_tokens or self._agent.config.max_tokens,
            metadata=metadata or {},
            session_id=self._session_id,
        )

        response = self._agent._runtime.run(request, schema=schema)

        self._messages.append(Message.user(input))
        self._messages.append(Message.assistant(content=response.text or ""))

        return response

    def history(self) -> list[Message]:
        """Get the full message history."""
        return list(self._messages)

    def clear(self) -> None:
        """Clear the message history."""
        self._messages = []

    def fork(self, session_id: str | None = None) -> "Session":
        """
        Create a new session with a copy of the current history.

        Args:
            session_id: Optional ID for the new session

        Returns:
            New Session instance
        """
        return Session(
            agent=self._agent,
            session_id=session_id,
            system=self._system,
            messages=self._messages.copy(),
        )

    def add_message(self, message: Message) -> None:
        """
        Manually add a message to history.

        Args:
            message: Message to add
        """
        self._messages.append(message)

    def to_dict(self) -> dict[str, Any]:
        """Serialize session state to a dictionary."""
        return {
            "session_id": self._session_id,
            "system": self._system,
            "messages": [
                {
                    "role": m.role,
                    "content": m.content if isinstance(m.content, str) else [
                        {"type": p.type, "text": p.text, "image_url": p.image_url}
                        for p in m.content
                    ],
                    "name": m.name,
                    "tool_call_id": m.tool_call_id,
                    "tool_calls": m.tool_calls,
                }
                for m in self._messages
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], agent: "Agent") -> "Session":
        """
        Deserialize session state from a dictionary.

        Args:
            data: Serialized session data
            agent: Agent to use for this session

        Returns:
            Session instance
        """
        messages = []
        for m in data.get("messages", []):
            msg = Message(
                role=m["role"],
                content=m["content"],
                name=m.get("name"),
                tool_call_id=m.get("tool_call_id"),
                tool_calls=m.get("tool_calls"),
            )
            messages.append(msg)

        return cls(
            agent=agent,
            session_id=data.get("session_id"),
            system=data.get("system"),
            messages=messages,
        )

    def __repr__(self) -> str:
        return f"Session(id={self._session_id!r}, messages={len(self._messages)})"


class _SessionStreamResponse(StreamResponse):
    """Stream response wrapper that updates session history after consumption."""

    def __init__(
        self,
        stream: StreamResponse,
        session: Session,
        user_input: str,
    ):
        super().__init__(
            _events=stream._events,
            provider=stream.provider,
            model=stream.model,
        )
        self._session = session
        self._user_input = user_input
        self._finalized = False

    def __iter__(self):
        yield from super().__iter__()

        # Update session history after stream is consumed
        if not self._finalized:
            self._finalized = True
            self._session._messages.append(Message.user(self._user_input))
            if self.tool_calls:
                self._session._messages.append(
                    Message.assistant(
                        content=self.text,
                        tool_calls=[tc.to_dict() for tc in self.tool_calls],
                    )
                )
            else:
                self._session._messages.append(Message.assistant(content=self.text))


class _AsyncSessionStreamResponse(AsyncStreamResponse):
    """Async stream response wrapper that updates session history after consumption."""

    def __init__(
        self,
        stream: AsyncStreamResponse,
        session: Session,
        user_input: str,
    ):
        super().__init__(
            events=stream._events,
            provider=stream.provider,
            model=stream.model,
        )
        self._session = session
        self._user_input = user_input
        self._finalized = False

    async def __anext__(self):
        try:
            return await super().__anext__()
        except StopAsyncIteration:
            # Update session history after stream is consumed
            if not self._finalized:
                self._finalized = True
                self._session._messages.append(Message.user(self._user_input))
                if self.tool_calls:
                    self._session._messages.append(
                        Message.assistant(
                            content=self.text,
                            tool_calls=[tc.to_dict() for tc in self.tool_calls],
                        )
                    )
                else:
                    self._session._messages.append(Message.assistant(content=self.text))
            raise
