"""Entry point: ``python -m daemon_pytool [--tools-dir DIR ...]``.

Loads user tool modules from each ``--tools-dir`` (every top-level ``.py`` file is imported so its
``@tool`` registrations run), then serves the stdio worker loop. Diagnostics go to stderr; stdout is
the cut transport.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path

from ._runtime import serve
from ._registry import registry


def _load_tools_dir(path: Path) -> None:
    if not path.is_dir():
        print(f"daemon_pytool: tools-dir {path} is not a directory; skipping", file=sys.stderr)
        return
    for file in sorted(path.glob("*.py")):
        if file.name.startswith("_"):
            continue
        mod_name = f"daemon_pytool_user_{file.stem}"
        spec = importlib.util.spec_from_file_location(mod_name, file)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        try:
            sys.modules[mod_name] = module
            spec.loader.exec_module(module)
        except Exception as exc:  # noqa: BLE001 - one bad tool file must not sink the worker
            print(f"daemon_pytool: failed to load {file}: {exc}", file=sys.stderr)


def _parse_tools_dirs(argv: list[str]) -> list[Path]:
    dirs: list[Path] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--tools-dir":
            i += 1
            if i < len(argv):
                dirs.append(Path(argv[i]))
        elif arg.startswith("--tools-dir="):
            dirs.append(Path(arg.split("=", 1)[1]))
        i += 1
    return dirs


def main() -> None:
    for path in _parse_tools_dirs(sys.argv[1:]):
        _load_tools_dir(path)
    print(
        f"daemon_pytool: serving {len(registry.names())} tool(s): {', '.join(registry.names())}",
        file=sys.stderr,
    )
    try:
        asyncio.run(serve(registry))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
