# `daemon_pytool` — Python tool SDK + worker

The daemon-native Python tool SDK and out-of-process worker. The daemon (`daemon-pytool-client`)
spawns this worker, discovers its tools, and registers a proxy `Tool` for each — so a Python
function becomes a first-class engine tool. The wire protocol is length-framed JSON over stdio
(`daemon_pytool::protocol`); the SDK and worker are **stdlib-only**.

## Author a tool

```python
from daemon_pytool import tool, ToolResult, Detail

@tool(
    "word_count",
    description="Count the words in a string.",
    schema={"type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"]},
    concurrency="parallel",   # side-effect-free: safe to run in a parallel batch
)
def word_count(args):                 # sync handler (runs in a thread pool)
    n = len(str(args["text"]).split())
    return ToolResult(content=str(n), detail=Detail(kind="word_count", body={"words": n}))

@tool("fetch_status", untrusted=True) # external/untrusted output is fenced by the daemon
async def fetch_status(args, ctx):    # async handler (runs on the event loop); ctx is optional
    return f"session={ctx.session_id}"
```

A handler may return a `str` (content), a `dict`/`list` (JSON-encoded into content), or a
`ToolResult` (content + `ok` + optional structured `Detail` + per-call `untrusted` override).

## Run the worker

```bash
python -m daemon_pytool [--tools-dir DIR ...]
```

Each `--tools-dir` is scanned for top-level `*.py` files (those not starting with `_`), which are
imported so their `@tool` registrations run. The built-in `py_echo` tool is always available.

## How the daemon launches it

The daemon's `[python]` config sets the interpreter, the tools dir, and (via `PYTHONPATH`) where
this package lives; it then spawns `python -m daemon_pytool --tools-dir <dir>` and talks the
protocol over the child's stdio. See the daemon's `build_python_tools`.
