"""Tests for the Session class."""


from agent.session import Session
from agent.testing import FakeResponse, create_test_agent


class TestSession:
    """Test Session class functionality."""

    def test_create_session(self):
        """Test session creation."""
        agent, _ = create_test_agent()
        session = agent.session()

        assert isinstance(session, Session)
        assert session.session_id is not None
        assert len(session.messages) == 0

    def test_session_with_custom_id(self):
        """Test session with custom ID."""
        agent, _ = create_test_agent()
        session = agent.session(session_id="my-session")

        assert session.session_id == "my-session"

    def test_session_with_system_prompt(self):
        """Test session with system prompt."""
        agent, _ = create_test_agent()
        session = agent.session(system="You are helpful")

        assert session.system == "You are helpful"

    def test_session_run_adds_to_history(self):
        """Test that run adds messages to history."""
        agent, provider = create_test_agent()
        provider.set_response(FakeResponse(text="Hello!"))

        session = agent.session()
        session.run("Hi there")

        assert len(session.messages) == 2  # User + Assistant
        assert session.messages[0].role == "user"
        assert session.messages[0].content == "Hi there"
        assert session.messages[1].role == "assistant"
        assert session.messages[1].content == "Hello!"

    def test_session_maintains_history(self):
        """Test that session maintains conversation history."""
        agent, provider = create_test_agent()
        provider.set_responses([
            FakeResponse(text="Nice to meet you!"),
            FakeResponse(text="Your name is Alice."),
        ])

        session = agent.session()
        session.run("My name is Alice")
        session.run("What's my name?")

        # Verify history is passed to second request
        request = provider.get_last_request()
        assert len(request.messages) == 2  # First exchange

    def test_session_clear(self):
        """Test clearing session history."""
        agent, provider = create_test_agent()
        provider.set_response(FakeResponse(text="Hi"))

        session = agent.session()
        session.run("Hello")
        assert len(session.messages) == 2

        session.clear()
        assert len(session.messages) == 0

    def test_session_fork(self):
        """Test forking a session."""
        agent, provider = create_test_agent()
        provider.set_response(FakeResponse(text="Hi"))

        session = agent.session()
        session.run("Hello")

        forked = session.fork()

        assert forked.session_id != session.session_id
        assert len(forked.messages) == len(session.messages)
        assert forked.system == session.system

    def test_session_fork_is_independent(self):
        """Test that forked session is independent."""
        agent, provider = create_test_agent()
        provider.set_responses([
            FakeResponse(text="First"),
            FakeResponse(text="Second"),
        ])

        session = agent.session()
        session.run("One")

        forked = session.fork()
        forked.run("Two")

        # Original should still have 2 messages
        assert len(session.messages) == 2
        # Forked should have 4 messages
        assert len(forked.messages) == 4

    def test_session_history(self):
        """Test history() returns copy."""
        agent, provider = create_test_agent()
        provider.set_response(FakeResponse(text="Hi"))

        session = agent.session()
        session.run("Hello")

        history = session.history()
        history.clear()  # Modify the copy

        # Original should be unchanged
        assert len(session.messages) == 2

    def test_session_to_dict(self):
        """Test session serialization."""
        agent, provider = create_test_agent()
        provider.set_response(FakeResponse(text="Hello"))

        session = agent.session(system="Be helpful")
        session.run("Hi")

        data = session.to_dict()

        assert data["session_id"] == session.session_id
        assert data["system"] == "Be helpful"
        assert len(data["messages"]) == 2

    def test_session_from_dict(self):
        """Test session deserialization."""
        agent, _ = create_test_agent()

        data = {
            "session_id": "restored-session",
            "system": "Restored system",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ],
        }

        session = Session.from_dict(data, agent)

        assert session.session_id == "restored-session"
        assert session.system == "Restored system"
        assert len(session.messages) == 2


class TestSessionStreaming:
    """Test Session streaming functionality."""

    def test_session_stream_updates_history(self):
        """Test that streaming updates history after consumption."""
        agent, provider = create_test_agent()
        provider.set_response(FakeResponse(text="Streamed response"))

        session = agent.session()
        stream = session.stream("Hello")

        # History empty before consuming stream
        # (implementation may vary)

        # Consume stream
        for _ in stream:
            pass

        # History should be updated
        assert len(session.messages) == 2
