"""Caching middleware for LLM responses."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from typing import Any, TypeVar

from catena.models import Message, Response, Usage
from catena.tools import Tool

T = TypeVar("T")


def _jsonable(value: Any) -> Any:
    """Stand-in for a request argument that json cannot serialize itself."""
    if isinstance(value, Tool):
        # The full definition, not just the name: two schemas can share a name.
        return value.to_openai()
    if isinstance(value, type) and hasattr(value, "model_json_schema"):
        return value.model_json_schema()  # a Pydantic model class
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")  # a Pydantic model instance
    return repr(value)


def _hash_messages(messages: list[Message], model: str, **kwargs: Any) -> str:
    """Create a cache key from the messages and every request argument.

    Anything that can change the answer (system prompt, tools, schema,
    sampling options) is part of the key; arguments left as None are not.
    """
    data: dict[str, Any] = {
        "messages": [m.to_dict() for m in messages],
        "model": model,
        "kwargs": {k: v for k, v in kwargs.items() if v is not None},
    }
    content = json.dumps(data, sort_keys=True, default=_jsonable)
    return hashlib.sha256(content.encode()).hexdigest()[:32]


def _free_copy(response: Response) -> Response:
    """A copy of a stored response at zero cost: a cache hit is free."""
    return Response(
        content=response.content,
        model=response.model,
        usage=Usage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cost_usd=0.0,
        ),
        metadata=response.metadata.copy(),
        tool_calls=list(response.tool_calls),
    )


@dataclass
class CacheEntry:
    """A cached response with metadata."""

    response: Response
    created_at: float
    expires_at: float | None = None
    hits: int = 0


class Cache(ABC):
    """Abstract cache interface."""

    _stats: dict[str, int]

    @abstractmethod
    async def get(self, key: str) -> Response | None:
        """Retrieve a cached response."""
        ...

    @abstractmethod
    async def set(self, key: str, response: Response, ttl: float | None = None) -> None:
        """Store a response for `ttl` seconds (None: the cache's default)."""
        ...

    @abstractmethod
    async def clear(self) -> None:
        """Clear all cached entries."""
        ...

    @property
    def stats(self) -> dict[str, int]:
        """Hits and misses since the cache was created or cleared."""
        return self._stats.copy()

    @property
    def hit_rate(self) -> float:
        """Share of lookups served from the cache."""
        total = self._stats["hits"] + self._stats["misses"]
        return self._stats["hits"] / total if total > 0 else 0.0

    def wrap(
        self,
        func: Callable[..., Any],
        ttl: float | None = None,
    ) -> Callable[..., Any]:
        """Wrap a provider's complete method with caching.

        `ttl` is how long these responses stay cached, in seconds
        (None: the cache's default).
        """

        @wraps(func)
        async def wrapper(
            messages: list[Message],
            *,
            model: str | None = None,
            **kwargs: Any,
        ) -> Response:
            # Two providers can share a cache: an unnamed model means each one's own default.
            owner = getattr(func, "__self__", None)
            model_name = str(model or getattr(owner, "default_model", "default"))
            key = _hash_messages(messages, model_name, **kwargs)

            cached = await self.get(key)
            if cached is not None:
                # Mark as cached in metadata
                cached.metadata["cached"] = True
                return cached

            response: Response = await func(messages, model=model, **kwargs)
            await self.set(key, response, ttl)
            return response

        return wrapper


class MemoryCache(Cache):
    """In-memory cache that evicts the least recently used entry when full.

    `default_ttl` is how long an entry lives, in seconds; None keeps entries
    until they are evicted.

    Example:
        ```python
        from catena.middleware import MemoryCache

        cache = MemoryCache(max_size=1000, default_ttl=3600)

        # Wrap a provider
        provider.complete = cache.wrap(provider.complete)
        ```
    """

    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: float | None = 3600,
    ) -> None:
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: dict[str, CacheEntry] = {}
        self._stats = {"hits": 0, "misses": 0}

    async def get(self, key: str) -> Response | None:
        entry = self._cache.get(key)
        if entry is None:
            self._stats["misses"] += 1
            return None

        if entry.expires_at is not None and time.time() >= entry.expires_at:
            del self._cache[key]
            self._stats["misses"] += 1
            return None

        entry.hits += 1
        self._stats["hits"] += 1
        # Dicts keep insertion order: re-inserting marks this entry most recently used.
        self._cache[key] = self._cache.pop(key)
        return _free_copy(entry.response)

    async def set(self, key: str, response: Response, ttl: float | None = None) -> None:
        if ttl is None:
            ttl = self.default_ttl
        now = time.time()

        self._cache.pop(key, None)
        while len(self._cache) >= self.max_size > 0:
            del self._cache[next(iter(self._cache))]  # least recently used

        if self.max_size > 0:
            self._cache[key] = CacheEntry(
                response=response,
                created_at=now,
                expires_at=None if ttl is None else now + ttl,
            )

    async def clear(self) -> None:
        self._cache.clear()
        self._stats = {"hits": 0, "misses": 0}


class SQLiteCache(Cache):
    """A cache kept in a SQLite file, so it survives restarts.

    Useful while developing: rerunning a script replays earlier answers
    instantly and for free. `default_ttl` is how long an entry lives, in
    seconds; None (the default) keeps entries until `clear()`.

    Example:
        ```python
        from catena.middleware import SQLiteCache

        provider = providers.openai().cached(SQLiteCache(".catena-cache.db"))
        ```
    """

    def __init__(
        self,
        path: str | os.PathLike[str] = ".catena-cache.db",
        default_ttl: float | None = None,
    ) -> None:
        self.path = os.fspath(path)
        self.default_ttl = default_ttl
        self._stats = {"hits": 0, "misses": 0}
        # Autocommit: every write is on disk when set() returns.
        self._db = sqlite3.connect(self.path, isolation_level=None)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS responses ("
            "key TEXT PRIMARY KEY, response TEXT NOT NULL, expires_at REAL)"
        )

    async def get(self, key: str) -> Response | None:
        row = self._db.execute(
            "SELECT response, expires_at FROM responses WHERE key = ?", (key,)
        ).fetchone()
        if row is not None and row[1] is not None and time.time() >= row[1]:
            self._db.execute("DELETE FROM responses WHERE key = ?", (key,))
            row = None
        if row is None:
            self._stats["misses"] += 1
            return None

        self._stats["hits"] += 1
        return _free_copy(Response.from_dict(json.loads(row[0])))

    async def set(self, key: str, response: Response, ttl: float | None = None) -> None:
        if ttl is None:
            ttl = self.default_ttl
        self._db.execute(
            "INSERT OR REPLACE INTO responses (key, response, expires_at) VALUES (?, ?, ?)",
            (
                key,
                json.dumps(response.to_dict(), default=str),
                None if ttl is None else time.time() + ttl,
            ),
        )

    async def clear(self) -> None:
        self._db.execute("DELETE FROM responses")
        self._stats = {"hits": 0, "misses": 0}

    def close(self) -> None:
        """Close the database file."""
        self._db.close()
