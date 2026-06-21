"""``daemon_pytool`` — the daemon-native Python tool SDK + out-of-process worker.

Author a tool by decorating a function with :func:`tool`; the daemon discovers and calls it over a
length-framed JSON stdio protocol. The worker is launched by the daemon as ``python -m daemon_pytool
[--tools-dir DIR ...]`` (see :mod:`daemon_pytool.__main__`).

Example::

    from daemon_pytool import tool, ToolResult

    @tool("greet", description="Greet someone by name.",
          schema={"type": "object", "properties": {"name": {"type": "string"}},
                  "required": ["name"]})
    def greet(args):
        return f"Hello, {args['name']}!"
"""

from ._registry import (
    Detail,
    Registry,
    SDK_VERSION,
    ToolContext,
    ToolResult,
    ToolSpec,
    registry,
    tool,
)
from ._runtime import PROTOCOL_VERSION, serve

# Register the built-in tools on import so a bare worker is never empty.
from . import builtin_tools as _builtin_tools  # noqa: E402,F401

__all__ = [
    "tool",
    "registry",
    "Registry",
    "ToolSpec",
    "ToolResult",
    "ToolContext",
    "Detail",
    "serve",
    "SDK_VERSION",
    "PROTOCOL_VERSION",
]
