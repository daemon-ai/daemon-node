"""The worker runtime: the length-framed JSON stdio loop that speaks ``daemon_pytool::protocol``.

The daemon spawns this process and exchanges length-framed JSON frames over stdio (the daemon owns
the ``u32``-LE length prefix). The runtime emits ``ready``, answers ``list_tools`` from the
registry, dispatches ``call_tool`` to handlers (sync handlers run in a thread pool, async handlers
on the loop) with concurrent in-flight calls, and exits on ``shutdown``. Stdlib-only: ``asyncio`` +
``json`` + ``struct`` + ``threading``.
"""

from __future__ import annotations

import asyncio
import json
import struct
import sys
import threading
from typing import Any, Optional

from ._registry import Detail, Registry, SDK_VERSION, ToolContext, ToolResult, registry as _registry

PROTOCOL_VERSION = 1
WORKER_NAME = "daemon_pytool"


def _read_exact(stream, n: int) -> Optional[bytes]:
    """Read exactly ``n`` bytes, or ``None`` on EOF / short read."""
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def _reader_thread(loop: asyncio.AbstractEventLoop, queue: "asyncio.Queue[Optional[bytes]]") -> None:
    """Blocking stdin reader: push each frame body (or ``None`` on EOF) onto the asyncio queue."""
    stdin = sys.stdin.buffer
    while True:
        header = _read_exact(stdin, 4)
        if header is None:
            loop.call_soon_threadsafe(queue.put_nowait, None)
            return
        (length,) = struct.unpack("<I", header)
        body = _read_exact(stdin, length) if length else b""
        if body is None:
            loop.call_soon_threadsafe(queue.put_nowait, None)
            return
        loop.call_soon_threadsafe(queue.put_nowait, body)


class _Writer:
    """Serializes outbound frames so concurrent tasks never interleave a frame on stdout."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._out = sys.stdout.buffer

    async def send(self, event: dict) -> None:
        data = json.dumps(event).encode("utf-8")
        async with self._lock:
            self._out.write(struct.pack("<I", len(data)))
            self._out.write(data)
            self._out.flush()


def _error_event(request_id, cls: str, message: str) -> dict:
    return {"event": "error", "request_id": request_id, "class": cls, "message": message}


def _normalise_result(value: Any, spec) -> dict:
    """Coerce a handler return value into a ``result`` event payload (sans request/call ids)."""
    detail = None
    untrusted = spec.untrusted
    if isinstance(value, ToolResult):
        content = value.content
        ok = value.ok
        if value.untrusted is not None:
            untrusted = value.untrusted
        if value.detail is not None:
            d = value.detail
            detail = d if isinstance(d, Detail) else Detail(kind=d["kind"], body=d["body"])
    elif isinstance(value, str):
        content, ok = value, True
    else:
        content, ok = json.dumps(value), True
    event: dict = {"event": "result", "ok": ok, "content": content, "untrusted": untrusted}
    if detail is not None:
        event["detail"] = {"kind": detail.kind, "body": detail.body}
    return event


async def _invoke(spec, args: dict, ctx: ToolContext) -> Any:
    """Call a handler: async handlers on the loop, sync handlers in the default executor."""
    if spec.is_async:
        return await (spec.handler(args, ctx) if spec.wants_ctx else spec.handler(args))
    loop = asyncio.get_running_loop()
    if spec.wants_ctx:
        return await loop.run_in_executor(None, spec.handler, args, ctx)
    return await loop.run_in_executor(None, spec.handler, args)


async def _handle_call(reg: Registry, cmd: dict, writer: _Writer, inflight: dict) -> None:
    request_id = cmd.get("request_id")
    call_id = cmd.get("call_id", "")
    name = cmd.get("name", "")
    try:
        spec = reg.get(name)
        if spec is None:
            await writer.send(_error_event(request_id, "unsupported", f"unknown tool {name!r}"))
            return
        try:
            args = json.loads(cmd.get("args") or "{}")
            if not isinstance(args, dict):
                raise ValueError("arguments must be a JSON object")
        except Exception as exc:  # noqa: BLE001 - report decode failures as bad requests
            await writer.send(_error_event(request_id, "bad_request", f"invalid args: {exc}"))
            return
        ctx = ToolContext(
            session_id=cmd.get("session_id", ""),
            call_id=call_id,
            deadline_ms=int(cmd.get("deadline_ms", 0) or 0),
        )
        try:
            value = await _invoke(spec, args, ctx)
        except asyncio.CancelledError:
            await writer.send(
                {
                    "event": "result",
                    "request_id": request_id,
                    "call_id": call_id,
                    "ok": False,
                    "content": "tool call cancelled",
                    "untrusted": False,
                }
            )
            return
        except Exception as exc:  # noqa: BLE001 - a handler error is a failed (not fatal) result
            await writer.send(
                {
                    "event": "result",
                    "request_id": request_id,
                    "call_id": call_id,
                    "ok": False,
                    "content": f"{type(exc).__name__}: {exc}",
                    "untrusted": False,
                }
            )
            return
        event = _normalise_result(value, spec)
        event["request_id"] = request_id
        event["call_id"] = call_id
        await writer.send(event)
    finally:
        inflight.pop(call_id, None)


async def serve(reg: Optional[Registry] = None) -> None:
    """Run the worker stdio loop until ``shutdown`` / EOF."""
    reg = reg or _registry
    loop = asyncio.get_running_loop()
    queue: "asyncio.Queue[Optional[bytes]]" = asyncio.Queue()
    writer = _Writer()

    threading.Thread(target=_reader_thread, args=(loop, queue), daemon=True).start()

    await writer.send(
        {
            "event": "ready",
            "worker": WORKER_NAME,
            "sdk_version": SDK_VERSION,
            "protocol_version": PROTOCOL_VERSION,
        }
    )

    inflight: dict[str, asyncio.Task] = {}
    while True:
        body = await queue.get()
        if body is None:
            break  # EOF: the daemon closed the cut.
        try:
            cmd = json.loads(body)
        except Exception:  # noqa: BLE001 - skip undecodable frames
            continue
        op = cmd.get("op")
        if op == "call_tool":
            task = asyncio.create_task(_handle_call(reg, cmd, writer, inflight))
            cid = cmd.get("call_id", "")
            if cid:
                inflight[cid] = task
        elif op == "list_tools":
            await writer.send(
                {"event": "tools", "request_id": cmd.get("request_id"), "tools": reg.manifests()}
            )
        elif op == "ping":
            await writer.send({"event": "pong", "request_id": cmd.get("request_id")})
        elif op == "cancel":
            task = inflight.get(cmd.get("call_id", ""))
            if task is not None:
                task.cancel()
        elif op == "initialize":
            continue
        elif op == "shutdown":
            break
        else:
            await writer.send(_error_event(cmd.get("request_id"), "bad_request", f"unknown op {op!r}"))

    # Best-effort: cancel any stragglers so the loop can exit promptly.
    for task in list(inflight.values()):
        task.cancel()
