"""Caching middleware for LLM responses."""

from __future__ import annotations

import hashlib
import json
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from typing import Any, TypeVar

from catena.models import Message, Response, Usage

T = TypeVar("T")


def _hash_messages(messages: list[Message], model: str, **kwargs: Any) -> str:
    """Create a cache key from messages and parameters."""
    data: dict[str, Any] = {
        "messages": [m.to_dict() for m in messages],
        "model": model,
        **{k: v for k, v in kwargs.items() if k in ("temperature", "max_tokens", "tool_choice")},
    }
    tools = kwargs.get("tools")
    if tools:
        # Full definitions, not just names: two schemas can share a class name.
        data["tools"] = sorted((t.to_openai() for t in tools), key=lambda d: d["function"]["name"])
    content = json.dumps(data, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class CacheEntry:
    """A cached response with metadata."""

    response: Response
    created_at: float
    hits: int = 0


class Cache(ABC):
    """Abstract cache interface."""

    @abstractmethod
    async def get(self, key: str) -> Response | None:
        """Retrieve a cached response."""
        ...

    @abstractmethod
    async def set(self, key: str, response: Response, ttl: float | None = None) -> None:
        """Store a response in the cache."""
        ...

    @abstractmethod
    async def clear(self) -> None:
        """Clear all cached entries."""
        ...

    def wrap(
        self,
        func: Callable[..., Any],
        ttl: float | None = None,
    ) -> Callable[..., Any]:
        """Wrap a provider's complete method with caching."""

        @wraps(func)
        async def wrapper(
            messages: list[Message],
            *,
            model: str | None = None,
            **kwargs: Any,
        ) -> Response:
            model_name = model or "default"
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
    """In-memory LRU cache with TTL support.

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

        # Check TTL
        if self.default_ttl and (time.time() - entry.created_at) > self.default_ttl:
            del self._cache[key]
            self._stats["misses"] += 1
            return None

        entry.hits += 1
        self._stats["hits"] += 1

        # Return a copy with zero cost (cached responses are free)
        return Response(
            content=entry.response.content,
            model=entry.response.model,
            usage=Usage(
                input_tokens=entry.response.usage.input_tokens,
                output_tokens=entry.response.usage.output_tokens,
                cost_usd=0.0,  # Cached = free
            ),
            metadata=entry.response.metadata.copy(),
            tool_calls=list(entry.response.tool_calls),
        )

    async def set(self, key: str, response: Response, ttl: float | None = None) -> None:
        # Evict oldest entries if at capacity
        if len(self._cache) >= self.max_size:
            oldest_key = min(self._cache, key=lambda k: self._cache[k].created_at)
            del self._cache[oldest_key]

        self._cache[key] = CacheEntry(
            response=response,
            created_at=time.time(),
        )

    async def clear(self) -> None:
        self._cache.clear()
        self._stats = {"hits": 0, "misses": 0}

    @property
    def stats(self) -> dict[str, int]:
        """Get cache statistics."""
        return self._stats.copy()

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self._stats["hits"] + self._stats["misses"]
        return self._stats["hits"] / total if total > 0 else 0.0
