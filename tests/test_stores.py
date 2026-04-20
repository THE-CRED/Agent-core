"""Tests for agent session stores."""

import time

import pytest

from agent.stores.base import SessionStore
from agent.stores.memory import InMemoryStore
from agent.stores.sqlite import SQLiteStore


# ── SessionStore ABC ─────────────────────────────────────────────

class TestSessionStoreABC:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            SessionStore()

    def test_abstract_methods(self):
        assert {"save", "load", "delete", "list_sessions"} <= SessionStore.__abstractmethods__

    def test_exists_is_concrete(self):
        assert "exists" not in SessionStore.__abstractmethods__

    def test_partial_subclass_fails(self):
        class Partial(SessionStore):
            def save(self, session_id, data): ...
            def load(self, session_id): ...
            def delete(self, session_id): ...
        with pytest.raises(TypeError):
            Partial()


# ── Shared store tests (parametrized) ────────────────────────────

@pytest.fixture
def memory_store():
    return InMemoryStore()


@pytest.fixture
def sqlite_store(tmp_path):
    return SQLiteStore(db_path=tmp_path / "test.db")


class TestSharedStoreBehaviour:
    @pytest.fixture(params=["memory", "sqlite"])
    def store(self, request, memory_store, sqlite_store):
        return memory_store if request.param == "memory" else sqlite_store

    def test_save_and_load(self, store):
        store.save("s1", {"key": "value"})
        assert store.load("s1") == {"key": "value"}

    def test_save_overwrites(self, store):
        store.save("s1", {"v": 1})
        store.save("s1", {"v": 2})
        assert store.load("s1") == {"v": 2}

    def test_save_multiple(self, store):
        store.save("a", {"x": 1})
        store.save("b", {"x": 2})
        assert store.load("a") == {"x": 1}
        assert store.load("b") == {"x": 2}

    def test_save_complex_data(self, store):
        data = {"nested": {"a": [1, 2, 3]}, "flag": True, "count": 42, "nil": None}
        store.save("complex", data)
        assert store.load("complex") == data

    def test_load_nonexistent(self, store):
        assert store.load("ghost") is None

    def test_delete_existing(self, store):
        store.save("s1", {"k": "v"})
        assert store.delete("s1") is True
        assert store.load("s1") is None

    def test_delete_nonexistent(self, store):
        assert store.delete("ghost") is False

    def test_list_sessions_empty(self, store):
        assert store.list_sessions() == []

    def test_list_sessions(self, store):
        store.save("a", {})
        store.save("b", {})
        assert set(store.list_sessions()) == {"a", "b"}

    def test_list_after_delete(self, store):
        store.save("a", {})
        store.save("b", {})
        store.delete("a")
        assert store.list_sessions() == ["b"]

    def test_exists_true(self, store):
        store.save("s1", {})
        assert store.exists("s1") is True

    def test_exists_false(self, store):
        assert store.exists("nope") is False

    def test_exists_after_delete(self, store):
        store.save("s1", {})
        store.delete("s1")
        assert store.exists("s1") is False

    def test_clear(self, store):
        store.save("a", {})
        store.save("b", {})
        store.clear()
        assert store.list_sessions() == []

    def test_clear_empty(self, store):
        store.clear()
        assert store.list_sessions() == []


# ── InMemoryStore-specific ───────────────────────────────────────

class TestInMemoryStore:
    def test_len_empty(self):
        assert len(InMemoryStore()) == 0

    def test_len_after_saves(self):
        s = InMemoryStore()
        s.save("a", {})
        s.save("b", {})
        assert len(s) == 2

    def test_len_after_delete(self):
        s = InMemoryStore()
        s.save("a", {})
        s.save("b", {})
        s.delete("a")
        assert len(s) == 1

    def test_len_after_clear(self):
        s = InMemoryStore()
        s.save("a", {})
        s.clear()
        assert len(s) == 0

    def test_same_id_no_duplicate(self):
        s = InMemoryStore()
        s.save("a", {"v": 1})
        s.save("a", {"v": 2})
        assert len(s) == 1

    def test_is_subclass(self):
        assert issubclass(InMemoryStore, SessionStore)

    def test_stores_reference(self):
        s = InMemoryStore()
        original = {"key": "val"}
        s.save("s1", original)
        assert s.load("s1") is original


# ── SQLiteStore-specific ─────────────────────────────────────────

class TestSQLiteStore:
    def test_is_subclass(self):
        assert issubclass(SQLiteStore, SessionStore)

    def test_creates_db_file(self, tmp_path):
        db = tmp_path / "new.db"
        SQLiteStore(db_path=db)
        assert db.exists()

    def test_default_db_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        s = SQLiteStore()
        s.save("s1", {"a": 1})
        assert (tmp_path / "agent_sessions.db").exists()

    def test_upsert(self, tmp_path):
        s = SQLiteStore(db_path=tmp_path / "upsert.db")
        s.save("s1", {"v": 1})
        s.save("s1", {"v": 2})
        assert s.load("s1") == {"v": 2}
        assert s.list_sessions() == ["s1"]

    def test_list_ordered_by_updated(self, tmp_path):
        s = SQLiteStore(db_path=tmp_path / "order.db")
        s.save("first", {})
        time.sleep(1.1)  # SQLite CURRENT_TIMESTAMP has 1-second resolution
        s.save("second", {})
        time.sleep(1.1)
        s.save("third", {})
        assert s.list_sessions() == ["third", "second", "first"]

    def test_resave_updates_order(self, tmp_path):
        s = SQLiteStore(db_path=tmp_path / "reorder.db")
        s.save("old", {})
        time.sleep(1.1)
        s.save("new", {})
        time.sleep(1.1)
        s.save("old", {"updated": True})
        assert s.list_sessions()[0] == "old"

    def test_vacuum(self, tmp_path):
        s = SQLiteStore(db_path=tmp_path / "v.db")
        s.save("s1", {})
        s.delete("s1")
        s.vacuum()  # should not raise

    def test_vacuum_empty(self, tmp_path):
        s = SQLiteStore(db_path=tmp_path / "ev.db")
        s.vacuum()

    def test_data_persists_across_instances(self, tmp_path):
        db = tmp_path / "persist.db"
        SQLiteStore(db_path=db).save("s1", {"persisted": True})
        assert SQLiteStore(db_path=db).load("s1") == {"persisted": True}
