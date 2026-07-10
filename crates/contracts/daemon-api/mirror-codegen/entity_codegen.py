#!/usr/bin/env python3
# Mirror entity codegen — the SECOND emitter on the update-codec pipeline (spec 09 §3.6, ADR-004).
#
# Inputs (pinned, human-owned): the authoritative daemon-api.cddl + the daemon-app entity map
# (src/core/mirror/entity-map.toml). Outputs (deterministic, vendored into daemon-app under
# src/core/mirror/generated/):
#
#   entities_gen.h            — Q_GADGET structs, typed key structs (std::hash + ==, canonical
#                               \x1f key serialization), the EntityKind enum.
#   entities_provenance_gen.h — constexpr field provenance table (for gates + doc rendering).
#   entities_map_gen.h        — mapper declarations `Entity map_x(const <decoded>&)`.
#   mirror_schema_gen.sql     — mirror DDL derived from the map (§4.5).
#
# plus a ONE-TIME skeleton (never regenerated over): entities_map.cpp (human-owned bodies).
#
# This emitter is pure Python stdlib (tomllib + a small CDDL member index): no third-party deps,
# so the drift gate is deterministic and reproducible. It deliberately does NOT reuse zcbor's
# internal CddlParser — that API chokes on the full contract (e.g. the bare `null` arm) and is not
# a stable public interface; a purpose-built member index is enough to GROUND provenance paths
# (§3.3) and keeps the pipeline hermetic.
#
# Usage:
#   entity_codegen.py --cddl <daemon-api.cddl> --map <entity-map.toml> --out <dir>
#   entity_codegen.py --cddl ... --map ... --emit-skeleton <entities_map.cpp>   # one-time only
#
# The generated artifacts carry no timestamps/versions, so re-running yields byte-identical output.

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
    raise SystemExit("entity_codegen.py requires Python 3.11+ (tomllib)") from exc


# --------------------------------------------------------------------------------------------
# CDDL member index — the provenance grounding oracle.
# --------------------------------------------------------------------------------------------

_RULE_START = re.compile(r"^([A-Za-z][A-Za-z0-9_-]*)\s*=(?!=)")
_MEMBER_KEY = re.compile(r'"([^"]+)"\s*:')


def _strip_comment(line: str) -> str:
    """Drop a trailing ``;`` comment, honoring double-quoted strings."""
    out = []
    in_str = False
    for ch in line:
        if ch == '"':
            in_str = not in_str
        if ch == ";" and not in_str:
            break
        out.append(ch)
    return "".join(out)


class CddlIndex:
    """Rule names + the set of quoted map-member keys reachable in each rule's definition.

    The member set is an over-approximation (it includes keys of nested/arm maps), which is safe
    for grounding: provenance deliberately anchors on a specific rule, and a wire path fails
    grounding only when its rule is unknown or its member truly does not occur in that rule.
    """

    def __init__(self, text: str) -> None:
        self.members: dict[str, set[str]] = {}
        self._parse(text)

    def _parse(self, text: str) -> None:
        current: str | None = None
        buf: list[str] = []

        def flush() -> None:
            if current is None:
                return
            body = "\n".join(buf)
            self.members[current] = set(_MEMBER_KEY.findall(body))

        for raw in text.splitlines():
            line = _strip_comment(raw)
            m = _RULE_START.match(line)
            if m:
                flush()
                current = m.group(1)
                buf = [line[m.end():]]
            elif current is not None:
                buf.append(line)
        flush()

    def has_rule(self, rule: str) -> bool:
        return rule in self.members

    def has_member(self, rule: str, member: str) -> bool:
        return member in self.members.get(rule, set())


# --------------------------------------------------------------------------------------------
# Provenance grammar: <rule>[.<member>][#<derivation>]  |  client_local
# --------------------------------------------------------------------------------------------

def ground_wire(index: CddlIndex, wire: str) -> str | None:
    """Return None if the wire path grounds; else a human-readable reason."""
    path = wire
    if "#" in path:
        path = path.split("#", 1)[0]
    if "." in path:
        rule, member = path.split(".", 1)
    else:
        rule, member = path, None
    if not index.has_rule(rule):
        return f"unknown CDDL rule '{rule}'"
    if member is not None and not index.has_member(rule, member):
        return f"CDDL rule '{rule}' has no member '{member}'"
    return None


# --------------------------------------------------------------------------------------------
# Type mapping.
# --------------------------------------------------------------------------------------------

_CPP_TYPES = {
    "QString": ("QString", "TEXT"),
    "QByteArray": ("QByteArray", "BLOB"),
    "int": ("int", "INTEGER"),
    "qint64": ("qint64", "INTEGER"),
    "quint64": ("quint64", "INTEGER"),
    "uint": ("uint", "INTEGER"),
    "bool": ("bool", "INTEGER"),
    "double": ("double", "REAL"),
}


def cpp_type(t: str) -> str:
    if t not in _CPP_TYPES:
        raise MapError(f"unknown field type '{t}'")
    return _CPP_TYPES[t][0]


def sql_type(t: str) -> str:
    if t not in _CPP_TYPES:
        raise MapError(f"unknown field type '{t}'")
    return _CPP_TYPES[t][1]


# --------------------------------------------------------------------------------------------
# Map model + validation.
# --------------------------------------------------------------------------------------------

class MapError(Exception):
    pass


CLASSES = {"M", "W", "L"}

# C++ (incl. C++20) reserved words + Qt macro keywords that cannot be struct member identifiers.
_CPP_RESERVED = {
    "alignas", "alignof", "and", "and_eq", "asm", "auto", "bitand", "bitor", "bool", "break",
    "case", "catch", "char", "char8_t", "char16_t", "char32_t", "class", "compl", "concept",
    "const", "consteval", "constexpr", "constinit", "const_cast", "continue", "co_await",
    "co_return", "co_yield", "decltype", "default", "delete", "do", "double", "dynamic_cast",
    "else", "enum", "explicit", "export", "extern", "false", "float", "for", "friend", "goto",
    "if", "inline", "int", "long", "mutable", "namespace", "new", "noexcept", "not", "not_eq",
    "nullptr", "operator", "or", "or_eq", "private", "protected", "public", "register",
    "reinterpret_cast", "requires", "return", "short", "signed", "sizeof", "static",
    "static_assert", "static_cast", "struct", "switch", "template", "this", "thread_local",
    "throw", "true", "try", "typedef", "typeid", "typename", "union", "unsigned", "using",
    "virtual", "void", "volatile", "wchar_t", "while", "xor", "xor_eq",
    # Qt moc keywords + our generated static member.
    "signals", "slots", "emit", "entity_kind",
}
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def check_identifier(where: str, name: str) -> None:
    if not _IDENT_RE.match(name):
        raise MapError(f"{where}: '{name}' is not a valid C++ identifier")
    if name in _CPP_RESERVED:
        raise MapError(f"{where}: '{name}' is a reserved C++/Qt keyword; rename it")


def snake(name: str) -> str:
    s = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    return s.lower()


class Field:
    __slots__ = ("name", "type", "wire", "client_local")

    def __init__(self, name: str, type_: str, wire: str | None, client_local: bool) -> None:
        self.name = name
        self.type = type_
        self.wire = wire
        self.client_local = client_local


class Entity:
    def __init__(self, name: str, spec: dict) -> None:
        self.name = name
        check_identifier(f"[entity.{name}]", name)
        self.cls = spec.get("class")
        if self.cls not in CLASSES:
            raise MapError(f"[entity.{name}] class must be one of {sorted(CLASSES)}, got {self.cls!r}")
        self.key = list(spec.get("key", []))
        self.scope = list(spec.get("scope", []))
        self.window_key = spec.get("window_key")
        self.policy = spec.get("policy")
        self.table = spec.get("table")
        self.singleton = bool(spec.get("singleton", False))
        self.invalidators = list(spec.get("invalidators", []))
        self.wire_read = list(spec.get("wire_read", []))
        self.source = spec.get("source")  # optional explicit primary DTO rule
        self.source_ctype = spec.get("source_ctype")  # optional explicit decoded C struct

        raw_fields = spec.get("fields", {})
        if not isinstance(raw_fields, dict) or not raw_fields:
            raise MapError(f"[entity.{name}] has no fields")
        self.fields: list[Field] = []
        for fname, fspec in raw_fields.items():
            check_identifier(f"[entity.{name}.fields.{fname}]", fname)
            if not isinstance(fspec, dict):
                raise MapError(f"[entity.{name}.fields.{fname}] must be a table")
            ftype = fspec.get("type")
            if ftype is None:
                raise MapError(f"[entity.{name}.fields.{fname}] missing 'type'")
            wire = fspec.get("wire")
            client_local = bool(fspec.get("client_local", False))
            # Provenance completeness (§3.3): exactly one of wire | client_local.
            if wire is None and not client_local:
                raise MapError(
                    f"[entity.{name}.fields.{fname}] has no provenance "
                    f"(needs 'wire' or 'client_local' = true)"
                )
            if wire is not None and client_local:
                raise MapError(
                    f"[entity.{name}.fields.{fname}] has both 'wire' and 'client_local'"
                )
            # Guardrail (§3.3/§14.11): a client_local field must live in sidecar, never a mirror table.
            if client_local:
                raise MapError(
                    f"[entity.{name}.fields.{fname}] is client_local but sits in a mirror table; "
                    f"move it under [entity.{name}.sidecar.*]"
                )
            self.fields.append(Field(fname, ftype, wire, client_local=False))

        self.sidecar: list[Field] = []
        for sname, sspec in spec.get("sidecar", {}).items():
            check_identifier(f"[entity.{name}.sidecar.{sname}]", sname)
            if not isinstance(sspec, dict) or "type" not in sspec:
                raise MapError(f"[entity.{name}.sidecar.{sname}] must be a table with 'type'")
            self.sidecar.append(Field(sname, sspec["type"], None, client_local=True))

        self._validate_shape()
        self.source = self.source or self._infer_source()

    def _field(self, fname: str) -> Field | None:
        for f in self.fields:
            if f.name == fname:
                return f
        return None

    def _validate_shape(self) -> None:
        if self.cls == "W":
            if not self.scope:
                raise MapError(f"[entity.{self.name}] class W requires 'scope'")
            if not self.window_key:
                raise MapError(f"[entity.{self.name}] class W requires 'window_key'")
            if not self.policy:
                raise MapError(f"[entity.{self.name}] class W requires 'policy' (§4.6 policy row)")
            for comp in self.scope:
                if self._field(comp) is None:
                    raise MapError(f"[entity.{self.name}] scope component '{comp}' is not a field")
            if self._field(self.window_key) is None:
                raise MapError(f"[entity.{self.name}] window_key '{self.window_key}' is not a field")
        else:
            if not self.key:
                raise MapError(f"[entity.{self.name}] class {self.cls} requires 'key'")
            for comp in self.key:
                if self._field(comp) is None:
                    raise MapError(f"[entity.{self.name}] key component '{comp}' is not a field")
        if self.cls == "M" and not self.table:
            raise MapError(f"[entity.{self.name}] class M requires 'table'")
        if self.cls == "W" and not self.table:
            raise MapError(f"[entity.{self.name}] class W requires 'table'")

    def _infer_source(self) -> str:
        """Modal DTO rule across field wire-paths (excluding request-/response- wrappers)."""
        counts: dict[str, int] = {}
        order: list[str] = []
        for f in self.fields:
            if f.wire is None:
                continue
            rule = f.wire.split("#", 1)[0].split(".", 1)[0]
            if rule.startswith("request-") or rule.startswith("response-"):
                continue
            if rule not in counts:
                counts[rule] = 0
                order.append(rule)
            counts[rule] += 1
        if not counts:
            # Fall back to the first field's rule even if a wrapper.
            return self.fields[0].wire.split("#", 1)[0].split(".", 1)[0]
        best = max(order, key=lambda r: (counts[r], -order.index(r)))
        return best

    def ctype(self) -> str:
        if self.source_ctype:
            return self.source_ctype
        return self.source.replace("-", "_")

    def key_components(self) -> list[str]:
        return self.scope if self.cls == "W" else self.key

    def key_struct(self) -> str:
        return self.name + ("Scope" if self.cls == "W" else "Key")

    def ordered_fields(self) -> list[Field]:
        """Key/scope components first (key order), then the rest in declaration order."""
        kc = self.key_components()
        head = [self._field(c) for c in kc]
        tail = [f for f in self.fields if f.name not in kc]
        return head + tail


def load_map(path: Path) -> list[Entity]:
    data = tomllib.loads(path.read_text())
    raw = data.get("entity", {})
    if not raw:
        raise MapError("entity map has no [entity.*] tables")
    entities = [Entity(name, spec) for name, spec in raw.items()]
    entities.sort(key=lambda e: e.name)  # deterministic ordering regardless of file order
    return entities


def validate_provenance(index: CddlIndex, entities: list[Entity]) -> None:
    errors: list[str] = []
    for e in entities:
        for f in e.fields:
            reason = ground_wire(index, f.wire)
            if reason is not None:
                errors.append(f"[entity.{e.name}.fields.{f.name}] wire '{f.wire}': {reason}")
    if errors:
        raise MapError("provenance grounding failed:\n  " + "\n  ".join(errors))


# --------------------------------------------------------------------------------------------
# Emitters.
# --------------------------------------------------------------------------------------------

BANNER = (
    "// @generated by mirror-codegen "
    "(daemon-node/crates/contracts/daemon-api/mirror-codegen/entity_codegen.py)\n"
    "// Source of truth: daemon-api.cddl + src/core/mirror/entity-map.toml. DO NOT EDIT.\n"
    "// Regenerate via `just update-codec` (superproject); guarded by `just codec-drift`.\n"
)

SQL_BANNER = (
    "-- @generated by mirror-codegen "
    "(daemon-node/crates/contracts/daemon-api/mirror-codegen/entity_codegen.py)\n"
    "-- Source of truth: daemon-api.cddl + src/core/mirror/entity-map.toml. DO NOT EDIT.\n"
    "-- Regenerate via `just update-codec` (superproject); guarded by `just codec-drift`.\n"
)


def _key_serialize_expr(entity: Entity) -> str:
    parts = []
    for comp in entity.key_components():
        f = entity._field(comp)
        expr = _to_qstring(f)
        parts.append(expr)
    if not parts:
        return 'QStringLiteral("")'
    joined = ' + QChar(0x1f) + '.join(parts)
    return joined


def _to_qstring(field: Field) -> str:
    t = field.type
    name = field.name
    if t == "QString":
        return name
    if t == "QByteArray":
        return f"QString::fromUtf8({name}.toHex())"
    if t == "bool":
        return f"QString::number(static_cast<int>({name}))"
    return f"QString::number({name})"


def emit_entities_gen(entities: list[Entity]) -> str:
    out: list[str] = []
    out.append("#pragma once")
    out.append(BANNER.rstrip("\n"))
    out.append("")
    out.append("#include <QByteArray>")
    out.append("#include <QChar>")
    out.append("#include <QHash>")
    out.append("#include <QMetaType>")
    out.append("#include <QString>")
    out.append("#include <cstddef>")
    out.append("#include <functional>")
    out.append("")
    out.append("namespace mirror {")
    out.append("")
    # EntityKind enum (alphabetical -> stable indices).
    out.append("enum class EntityKind : quint16 {")
    for i, e in enumerate(entities):
        out.append(f"    {e.name} = {i},")
    out.append("};")
    out.append("")
    out.append(f"inline constexpr std::size_t kEntityKindCount = {len(entities)};")
    out.append("")
    out.append("[[nodiscard]] inline const char* entityKindName(EntityKind kind) noexcept {")
    out.append("    switch (kind) {")
    for e in entities:
        out.append(f'    case EntityKind::{e.name}: return "{e.name}";')
    out.append("    }")
    out.append("    return \"\";")
    out.append("}")
    out.append("")

    for e in entities:
        ks = e.key_struct()
        comps = [e._field(c) for c in e.key_components()]
        # Key/scope struct.
        out.append(f"// {e.cls}: {e.name} "
                   + (f"(scope {'␟'.join(e.scope)}, window {e.window_key})" if e.cls == "W"
                      else f"(key {'␟'.join(e.key)})"))
        out.append(f"struct {ks} {{")
        for c in comps:
            out.append(f"    {cpp_type(c.type)} {c.name};")
        out.append("")
        # equality
        eq_terms = " && ".join(f"lhs.{c.name} == rhs.{c.name}" for c in comps) or "true"
        out.append(f"    friend bool operator==(const {ks}& lhs, const {ks}& rhs) noexcept {{")
        out.append(f"        return {eq_terms};")
        out.append("    }")
        out.append(f"    friend bool operator!=(const {ks}& lhs, const {ks}& rhs) noexcept {{")
        out.append("        return !(lhs == rhs);")
        out.append("    }")
        # canonical serialize
        out.append("    [[nodiscard]] QString serialize() const {")
        out.append(f"        return {_key_serialize_expr(e)};")
        out.append("    }")
        out.append("};")
        out.append("")

        # Entity struct (Q_GADGET).
        out.append(f"struct {e.name} {{")
        out.append("    Q_GADGET")
        for f in e.ordered_fields():
            out.append(f"    Q_PROPERTY({cpp_type(f.type)} {f.name} MEMBER {f.name})")
        out.append("")
        out.append("public:")
        for f in e.ordered_fields():
            default = _default_init(f.type)
            out.append(f"    {cpp_type(f.type)} {f.name}{default};")
        out.append("")
        # key() accessor
        key_init = ", ".join(f"{c.name}" for c in comps)
        out.append(f"    [[nodiscard]] {ks} {'scope' if e.cls == 'W' else 'key'}() const {{")
        out.append(f"        return {ks}{{{key_init}}};")
        out.append("    }")
        out.append(f"    static constexpr EntityKind entity_kind = EntityKind::{e.name};")
        out.append("};")
        out.append("")

    out.append("}  // namespace mirror")
    out.append("")
    # std::hash specializations for key/scope structs.
    out.append("template <>")
    for e in entities:
        ks = e.key_struct()
        out.append(f"struct std::hash<mirror::{ks}> {{")
        out.append(f"    std::size_t operator()(const mirror::{ks}& k) const noexcept {{")
        out.append("        return std::hash<QString>{}(k.serialize());")
        out.append("    }")
        out.append("};")
        out.append("template <>")
    # remove the trailing dangling "template <>"
    out.pop()
    out.append("")
    for e in entities:
        out.append(f"Q_DECLARE_METATYPE(mirror::{e.name})")
    out.append("")
    return "\n".join(out) + "\n"


def _default_init(t: str) -> str:
    if t in ("QString", "QByteArray"):
        return ""
    if t == "bool":
        return " = false"
    if t == "double":
        return " = 0.0"
    return " = 0"


def emit_provenance_gen(entities: list[Entity]) -> str:
    out: list[str] = []
    out.append("#pragma once")
    out.append(BANNER.rstrip("\n"))
    out.append("")
    out.append('#include "entities_gen.h"')
    out.append("")
    out.append("#include <cstddef>")
    out.append("")
    out.append("namespace mirror {")
    out.append("")
    out.append("// One row per generated field + sidecar declaration. `wire` is the provenance path")
    out.append("// (a CDDL rule.member or a named #derivation) or nullptr for client_local.")
    out.append("struct FieldProvenance {")
    out.append("    EntityKind entity;")
    out.append("    const char* entity_name;")
    out.append("    const char* field;")
    out.append("    const char* wire;")
    out.append("    bool client_local;")
    out.append("};")
    out.append("")
    out.append("inline constexpr FieldProvenance kProvenanceTable[] = {")
    for e in entities:
        for f in e.fields:
            wire = f'"{f.wire}"'
            out.append(f'    {{ EntityKind::{e.name}, "{e.name}", "{f.name}", {wire}, false }},')
        for f in e.sidecar:
            out.append(f'    {{ EntityKind::{e.name}, "{e.name}", "{f.name}", nullptr, true }},')
    out.append("};")
    out.append("")
    out.append("inline constexpr std::size_t kProvenanceTableSize =")
    out.append("    sizeof(kProvenanceTable) / sizeof(kProvenanceTable[0]);")
    out.append("")
    out.append("}  // namespace mirror")
    out.append("")
    return "\n".join(out) + "\n"


def emit_map_gen(entities: list[Entity]) -> str:
    out: list[str] = []
    out.append("#pragma once")
    out.append(BANNER.rstrip("\n"))
    out.append("")
    out.append("// Mapper DECLARATIONS (bodies are human-owned in entities_map.cpp — the drift gate")
    out.append("// checks these signatures against the vendored codec types, never the bodies).")
    out.append("")
    out.append('#include "entities_gen.h"')
    out.append("")
    out.append('extern "C" {')
    out.append('#include "daemon_api_client_types.h"')
    out.append("}")
    out.append("")
    out.append("namespace mirror {")
    out.append("")
    for e in entities:
        fn = "map_" + snake(e.name)
        out.append(f"// {e.name} <- {e.source} (decoded ::{e.ctype()})")
        out.append(f"[[nodiscard]] {e.name} {fn}(const ::{e.ctype()}& in);")
    out.append("")
    out.append("}  // namespace mirror")
    out.append("")
    return "\n".join(out) + "\n"


def emit_schema_sql(entities: list[Entity]) -> str:
    out: list[str] = []
    out.append(SQL_BANNER.rstrip("\n"))
    out.append("")
    out.append("-- Mirror cache schema (mirror-<id>.db, disposable). Derived from the entity map")
    out.append("-- per spec §4.5. Class-L entities are in-memory only and intentionally have no table.")
    out.append("")
    out.append("CREATE TABLE mirror_meta(k TEXT PRIMARY KEY, v TEXT);  -- schema_version, journal_head")
    out.append("")

    m_entities = [e for e in entities if e.cls == "M"]
    w_entities = [e for e in entities if e.cls == "W"]

    for e in m_entities:
        cols = []
        kc = e.key_components()
        for c in kc:
            f = e._field(c)
            cols.append(f"  {f.name} {sql_type(f.type)} NOT NULL")
        for f in e.fields:
            if f.name in kc:
                continue
            cols.append(f"  {f.name} {sql_type(f.type)}")
        out.append(f"CREATE TABLE {e.table}(")
        out.append("  key TEXT PRIMARY KEY,  -- canonical composite key (§3.1)")
        out.append(",\n".join(cols) + ",")
        out.append("  last_rev INTEGER NOT NULL, fetched_at_ms INTEGER NOT NULL);")
        out.append("")

    for e in w_entities:
        wk = e._field(e.window_key)
        out.append(f"-- W {e.name}: per-scope window (policy '{e.policy}', §4.6); item = canonical CBOR blob")
        out.append(f"CREATE TABLE {e.table}(")
        out.append("  scope TEXT NOT NULL,")
        out.append(f"  {e.window_key} {sql_type(wk.type)} NOT NULL,")
        out.append("  payload BLOB NOT NULL,")
        out.append("  origin_op TEXT, last_rev INTEGER NOT NULL,")
        out.append(f"  PRIMARY KEY(scope, {e.window_key}));")
        out.append("")

    # Fixed bookkeeping tables (§4.5).
    out.append("CREATE TABLE window_meta(")
    out.append("  kind TEXT NOT NULL, scope TEXT NOT NULL,")
    out.append("  item_count INTEGER, byte_count INTEGER,")
    out.append("  oldest_cursor INTEGER, newest_cursor INTEGER,")
    out.append("  contiguous_to_head INTEGER NOT NULL,")
    out.append("  PRIMARY KEY(kind, scope));")
    out.append("")
    out.append("CREATE TABLE mirror_journal(rev INTEGER PRIMARY KEY, kind TEXT, key TEXT,")
    out.append("  op INTEGER, origin INTEGER, origin_op TEXT, at_ms INTEGER);")
    out.append("")
    out.append("CREATE TABLE journal_watermarks(consumer TEXT PRIMARY KEY, rev INTEGER NOT NULL);")
    out.append("")
    out.append("CREATE TABLE sync_cursors(name TEXT PRIMARY KEY, cursor INTEGER, epoch INTEGER);")
    out.append("")
    out.append("CREATE TABLE node_revs(collection TEXT PRIMARY KEY, rev INTEGER,")
    out.append("  fetched_at_ms INTEGER, state INTEGER, last_error TEXT);")
    out.append("")
    return "\n".join(out) + "\n"


def emit_skeleton_cpp(entities: list[Entity]) -> str:
    out: list[str] = []
    out.append("// ONE-TIME skeleton — bodies are human-owned; regeneration NEVER overwrites this file.")
    out.append("// The drift gate checks that every declared mapper (entities_map_gen.h) has a matching")
    out.append("// definition here with the current codec signature — never the body contents.")
    out.append("//")
    out.append("// Each body maps a decoded wire DTO to the canonical entity per the provenance in")
    out.append("// src/core/mirror/entity-map.toml (merging extra reads / scope context as needed).")
    out.append("")
    out.append('#include "entities_map_gen.h"')
    out.append("")
    out.append("namespace mirror {")
    out.append("")
    for e in entities:
        fn = "map_" + snake(e.name)
        out.append(f"{e.name} {fn}(const ::{e.ctype()}& in) {{")
        out.append(f"    {e.name} out;")
        out.append("    (void)in;")
        out.append(f"    // TODO(mirror-map): populate {e.name} from wire per entity-map.toml provenance.")
        out.append("    return out;")
        out.append("}")
        out.append("")
    out.append("}  // namespace mirror")
    out.append("")
    return "\n".join(out) + "\n"


ARTIFACTS = {
    "entities_gen.h": emit_entities_gen,
    "entities_provenance_gen.h": emit_provenance_gen,
    "entities_map_gen.h": emit_map_gen,
    "mirror_schema_gen.sql": emit_schema_sql,
}


def generate(cddl: Path, mapfile: Path) -> dict[str, str]:
    index = CddlIndex(cddl.read_text())
    entities = load_map(mapfile)
    validate_provenance(index, entities)
    return {name: fn(entities) for name, fn in ARTIFACTS.items()}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Mirror entity codegen (second update-codec emitter).")
    ap.add_argument("--cddl", required=True, type=Path)
    ap.add_argument("--map", required=True, type=Path)
    ap.add_argument("--out", type=Path, help="write the byte-identical artifacts into this dir")
    ap.add_argument("--emit-skeleton", type=Path,
                    help="write the ONE-TIME mapper skeleton .cpp (refuses to overwrite)")
    ap.add_argument("--force-skeleton", action="store_true",
                    help="allow --emit-skeleton to overwrite an existing file")
    args = ap.parse_args(argv)

    try:
        index = CddlIndex(args.cddl.read_text())
        entities = load_map(args.map)
        validate_provenance(index, entities)
    except MapError as exc:
        print(f"entity_codegen: {exc}", file=sys.stderr)
        return 2

    if args.emit_skeleton is not None:
        if args.emit_skeleton.exists() and not args.force_skeleton:
            print(f"entity_codegen: refusing to overwrite existing skeleton {args.emit_skeleton}",
                  file=sys.stderr)
            return 3
        args.emit_skeleton.parent.mkdir(parents=True, exist_ok=True)
        args.emit_skeleton.write_text(emit_skeleton_cpp(entities))
        print(f"wrote skeleton {args.emit_skeleton}")
        return 0

    if args.out is None:
        print("entity_codegen: --out or --emit-skeleton is required", file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    for name, fn in ARTIFACTS.items():
        (args.out / name).write_text(fn(entities))
    print(f"wrote {len(ARTIFACTS)} artifacts into {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
