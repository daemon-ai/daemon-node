#!/usr/bin/env python3
# Mirror entity drift gate — the entity half of `codec-drift` (spec 09 §3.6).
#
# Extends the vendored-codec drift check to the second emitter's artifacts:
#   1. byte-identical regeneration of the 4 artifacts vs the vendored copies (no hand edits),
#   2. provenance completeness + CDDL grounding (regeneration fails otherwise — see entity_codegen),
#   3. no client_local field in a mirror table (enforced at generation time),
#   4. mapper signature match: every declared mapper (entities_map_gen.h) has a definition in the
#      human-owned skeleton (entities_map.cpp) with the SAME signature, and each referenced decoded
#      type still exists in the vendored codec types header.
#
# Pure Python stdlib. Exit 0 = in sync, non-zero = drift (with a diff / reason on stderr).

from __future__ import annotations

import argparse
import difflib
import re
import sys
from pathlib import Path

import entity_codegen as ec


_DECL_RE = re.compile(r"\]\]\s*(\w+)\s+(map_\w+)\(const ::(\w+)&\s*in\);")
_DEF_RE = re.compile(r"^(\w+)\s+(map_\w+)\(const ::(\w+)&\s*in\)\s*\{", re.MULTILINE)


def _parse_decls(text: str) -> dict[str, tuple[str, str]]:
    """map_fn -> (return_type, ctype) from the generated declarations header."""
    out: dict[str, tuple[str, str]] = {}
    for ret, fn, ctype in _DECL_RE.findall(text):
        out[fn] = (ret, ctype)
    return out


def _parse_defs(text: str) -> dict[str, tuple[str, str]]:
    """map_fn -> (return_type, ctype) from the human-owned skeleton .cpp."""
    out: dict[str, tuple[str, str]] = {}
    for ret, fn, ctype in _DEF_RE.findall(text):
        out[fn] = (ret, ctype)
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Mirror entity artifact drift gate.")
    ap.add_argument("--cddl", required=True, type=Path)
    ap.add_argument("--map", required=True, type=Path)
    ap.add_argument("--generated-dir", required=True, type=Path,
                    help="the vendored src/core/mirror/generated dir")
    ap.add_argument("--map-cpp", required=True, type=Path,
                    help="the human-owned entities_map.cpp skeleton")
    ap.add_argument("--types-header", required=True, type=Path,
                    help="the vendored daemon_api_client_types.h")
    args = ap.parse_args(argv)

    fail = 0

    # (2)+(3) regeneration re-validates provenance completeness, grounding, and the
    # no-client_local-in-mirror-table rule; a MapError here is a gate failure.
    try:
        artifacts = ec.generate(args.cddl, args.map)
    except ec.MapError as exc:
        print(f"DRIFT: entity map invalid: {exc}", file=sys.stderr)
        return 1

    # (1) byte-identical regeneration.
    for name, expected in artifacts.items():
        vendored_path = args.generated_dir / name
        if not vendored_path.exists():
            print(f"DRIFT: vendored artifact missing: {vendored_path}", file=sys.stderr)
            fail = 1
            continue
        actual = vendored_path.read_text()
        if actual != expected:
            print(f"DRIFT: vendored {name} differs from generated:", file=sys.stderr)
            diff = difflib.unified_diff(
                actual.splitlines(keepends=True),
                expected.splitlines(keepends=True),
                fromfile=f"vendored/{name}", tofile=f"generated/{name}",
            )
            sys.stderr.writelines(diff)
            fail = 1

    # (4) mapper signature match.
    decls = _parse_decls(artifacts["entities_map_gen.h"])
    if not args.map_cpp.exists():
        print(f"DRIFT: mapper skeleton missing: {args.map_cpp}", file=sys.stderr)
        return 1
    defs = _parse_defs(args.map_cpp.read_text())
    types_src = args.types_header.read_text()

    for fn, (ret, ctype) in sorted(decls.items()):
        if fn not in defs:
            print(f"DRIFT: mapper '{fn}' declared but not defined in {args.map_cpp.name}",
                  file=sys.stderr)
            fail = 1
            continue
        if defs[fn] != (ret, ctype):
            print(f"DRIFT: mapper '{fn}' signature mismatch: "
                  f"declared ({ret}, ::{ctype}) vs defined ({defs[fn][0]}, ::{defs[fn][1]})",
                  file=sys.stderr)
            fail = 1
        if not re.search(rf"^struct {re.escape(ctype)} \{{", types_src, re.MULTILINE):
            print(f"DRIFT: mapper '{fn}' references decoded type '::{ctype}' "
                  f"absent from {args.types_header.name} (codec changed — revisit the mapper)",
                  file=sys.stderr)
            fail = 1

    if fail:
        print("mirror entity artifacts are STALE; run: nix run .#update-codec", file=sys.stderr)
        return 1
    print("mirror entity artifacts + mapper signatures match the generated output")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
