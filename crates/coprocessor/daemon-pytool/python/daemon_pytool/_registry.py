"""The tool registry and the ``@tool`` authoring decorator.

Mirrors the *contract* (not the class hierarchy) of Hermes' ``tools/registry.py``: a tool is a
declarative registration of ``{name, description, schema, handler, concurrency, untrusted}`` on a
process-global registry, populated at import time. A handler is a plain callable
``(args: dict[, ctx: ToolContext]) -> str | dict | list | ToolResult`` and may be sync or async.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


SDK_VERSION = "0.1.0"


@dataclass
class ToolContext:
    """The per-call context handed to a handler that opts into a second parameter."""

    session_id: str = ""
    call_id: str = ""
    deadline_ms: int = 0


@dataclass
class Detail:
    """A structured result detail (the GUI renders ``body`` per ``kind``)."""

    kind: str
    body: Any


@dataclass
class ToolResult:
    """A rich tool result. Handlers may also return a bare ``str`` (content) or a ``dict``/``list``
    (JSON-encoded into the content); both are normalised to this shape by the runtime."""

    content: str
    ok: bool = True
    detail: Optional[Detail] = None
    # ``None`` => inherit the tool's manifest default; ``True``/``False`` => per-call override.
    untrusted: Optional[bool] = None


@dataclass
class ToolSpec:
    """A registered tool: its manifest metadata plus the handler and how to call it."""

    name: str
    description: str
    schema: dict
    concurrency: str  # "parallel" | "exclusive"
    untrusted: bool
    handler: Callable[..., Any]
    is_async: bool
    wants_ctx: bool

    def manifest(self) -> dict:
        """The wire ``ToolManifest`` (schema rendered to a JSON string for the daemon)."""
        return {
            "name": self.name,
            "description": self.description,
            "schema": json.dumps(self.schema),
            "concurrency": self.concurrency,
            "untrusted": self.untrusted,
        }


class Registry:
    """A name -> :class:`ToolSpec` map. The process-global instance is :data:`registry`."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec, *, override: bool = False) -> None:
        if spec.name in self._tools and not override:
            raise ValueError(
                f"tool {spec.name!r} already registered (pass override=True to replace)"
            )
        self._tools[spec.name] = spec

    def get(self, name: str) -> Optional[ToolSpec]:
        return self._tools.get(name)

    def manifests(self) -> list[dict]:
        return [spec.manifest() for spec in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools)


# The process-global registry the ``@tool`` decorator populates and the runtime serves from.
registry = Registry()


def tool(
    name: Optional[str] = None,
    *,
    description: str = "",
    schema: Optional[dict] = None,
    concurrency: str = "exclusive",
    untrusted: bool = False,
    register_to: Optional[Registry] = None,
):
    """Register the decorated function as a tool.

    ``name`` defaults to the function name; ``description`` to its docstring; ``schema`` to a
    permissive ``{"type": "object"}``. Set ``concurrency="parallel"`` for a side-effect-free tool
    and ``untrusted=True`` for one returning external/untrusted content (fenced by the daemon).
    """

    if concurrency not in ("parallel", "exclusive"):
        raise ValueError("concurrency must be 'parallel' or 'exclusive'")

    target = register_to or registry

    def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        params = inspect.signature(fn).parameters
        spec = ToolSpec(
            name=name or fn.__name__,
            description=description or (inspect.getdoc(fn) or "").strip(),
            schema=schema if schema is not None else {"type": "object"},
            concurrency=concurrency,
            untrusted=untrusted,
            handler=fn,
            is_async=inspect.iscoroutinefunction(fn),
            wants_ctx=len(params) >= 2,
        )
        target.register(spec)
        return fn

    return decorate
