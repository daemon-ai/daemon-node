"""Built-in tools shipped with the worker so a zero-config worker still exposes something callable.

``py_echo`` is the demo-gate tool: it round-trips text and attaches a structured detail, exercising
the full content + ``ToolDetail`` path end-to-end.
"""

from __future__ import annotations

from ._registry import Detail, ToolResult, tool

_ECHO_SCHEMA = {
    "type": "object",
    "properties": {"text": {"type": "string", "description": "The text to echo back."}},
    "required": ["text"],
}


@tool(
    "py_echo",
    description="Echo the provided text back to the caller (built-in demo tool).",
    schema=_ECHO_SCHEMA,
    concurrency="parallel",
)
def py_echo(args: dict) -> ToolResult:
    text = str(args.get("text", ""))
    return ToolResult(content=text, detail=Detail(kind="py_echo", body={"echoed": text}))
