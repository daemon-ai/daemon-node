#!/usr/bin/env python3
# TDD suite for the mirror entity emitter + drift gate (spec 09 §3.6, §12 "Entity codegen" row).
#
# Run: python3 -m unittest discover -s <this dir> -p 'test_*.py'
# The real daemon-app entity map is validated when MIRROR_ENTITY_MAP points at it (this repo
# ships only the CDDL; the map lives in daemon-app).

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODEGEN_DIR = HERE.parent
FIXTURES = HERE / "fixtures"
CDDL = CODEGEN_DIR.parent / "daemon-api.cddl"
TYPES_HEADER_ENV = "MIRROR_CODEC_TYPES_HEADER"
MAP_ENV = "MIRROR_ENTITY_MAP"

sys.path.insert(0, str(CODEGEN_DIR))
import entity_codegen as ec  # noqa: E402
import entity_drift as ed  # noqa: E402


# The complete §3.1 census (encoded here so a dropped/renamed row fails the census gate).
CENSUS_KINDS = {
    "TransportAccount", "Adapter", "Conversation", "ConversationMember", "ChatMessage",
    "Contact", "Person", "PersonEndpoint", "Session", "FleetUnit", "Approval", "Checkpoint",
    "Profile", "Credential", "InstalledModel", "ModelDownload", "QuantizeJob",
    "ProviderDescriptor", "CustomProvider", "RoutePin", "Room", "Notification", "SavedPresence",
    "ToolInfo", "AgentEntry", "CronJob", "CommandSpec", "RememberedFingerprint", "FsEntry",
    "TranscriptBlock", "CapsReport", "GatewayStatus", "AccessUser", "RoleInfo",
}


class GoldenRegenTests(unittest.TestCase):
    def test_generate_is_byte_identical_across_runs(self):
        a = ec.generate(CDDL, FIXTURES / "mini_map.toml")
        b = ec.generate(CDDL, FIXTURES / "mini_map.toml")
        self.assertEqual(set(a), {"entities_gen.h", "entities_provenance_gen.h",
                                  "entities_map_gen.h", "mirror_schema_gen.sql"})
        for name in a:
            self.assertEqual(a[name], b[name], f"{name} is not deterministic")

    def test_no_timestamps_or_versions(self):
        for text in ec.generate(CDDL, FIXTURES / "mini_map.toml").values():
            low = text.lower()
            self.assertNotIn("2026", text)
            self.assertNotIn("generated on", low)


class ProvenanceGateTests(unittest.TestCase):
    def test_missing_provenance_fails(self):
        with self.assertRaises(ec.MapError) as ctx:
            ec.generate(CDDL, FIXTURES / "missing_provenance_map.toml")
        self.assertIn("provenance", str(ctx.exception).lower())

    def test_client_local_in_mirror_table_fails(self):
        with self.assertRaises(ec.MapError) as ctx:
            ec.generate(CDDL, FIXTURES / "client_local_in_table_map.toml")
        self.assertIn("client_local", str(ctx.exception))

    def test_ungrounded_wire_path_fails(self):
        with self.assertRaises(ec.MapError) as ctx:
            ec.generate(CDDL, FIXTURES / "bad_wire_map.toml")
        self.assertIn("no member", str(ctx.exception))


class CddlIndexTests(unittest.TestCase):
    def setUp(self):
        self.idx = ec.CddlIndex(CDDL.read_text())

    def test_known_rule_and_member(self):
        self.assertIsNone(ec.ground_wire(self.idx, "conversation-info.parent"))
        self.assertIsNone(ec.ground_wire(self.idx, "conversation-info.members#count"))
        self.assertIsNone(ec.ground_wire(self.idx, "transcript-block#tag"))

    def test_unknown_rule(self):
        self.assertIsNotNone(ec.ground_wire(self.idx, "not-a-rule.x"))

    def test_unknown_member(self):
        self.assertIsNotNone(ec.ground_wire(self.idx, "conversation-info.not_a_member"))


class SecondShapeDriftTests(unittest.TestCase):
    """A hand-edited generated region (a second entity shape) must fail the drift gate."""

    def _stage(self, tmp: Path):
        artifacts = ec.generate(CDDL, FIXTURES / "mini_map.toml")
        gen = tmp / "generated"
        gen.mkdir()
        for name, text in artifacts.items():
            (gen / name).write_text(text)
        # a matching one-time skeleton
        entities = ec.load_map(FIXTURES / "mini_map.toml")
        map_cpp = tmp / "entities_map.cpp"
        map_cpp.write_text(ec.emit_skeleton_cpp(entities))
        # a stub codec types header carrying the referenced decoded structs
        ctypes = sorted({e.ctype() for e in entities})
        header = tmp / "daemon_api_client_types.h"
        header.write_text("".join(f"struct {c} {{ int x; }};\n" for c in ctypes))
        return gen, map_cpp, header

    def _run_drift(self, gen, map_cpp, header):
        return ed.main([
            "--cddl", str(CDDL), "--map", str(FIXTURES / "mini_map.toml"),
            "--generated-dir", str(gen), "--map-cpp", str(map_cpp),
            "--types-header", str(header),
        ])

    def test_clean_tree_passes(self):
        with tempfile.TemporaryDirectory() as d:
            gen, map_cpp, header = self._stage(Path(d))
            self.assertEqual(self._run_drift(gen, map_cpp, header), 0)

    def test_hand_edit_detected(self):
        with tempfile.TemporaryDirectory() as d:
            gen, map_cpp, header = self._stage(Path(d))
            victim = gen / "entities_gen.h"
            victim.write_text(victim.read_text().replace(
                "struct Room {", "struct Room { int smuggled_field = 0;"))
            self.assertNotEqual(self._run_drift(gen, map_cpp, header), 0)

    def test_missing_mapper_definition_detected(self):
        with tempfile.TemporaryDirectory() as d:
            gen, map_cpp, header = self._stage(Path(d))
            # drop one mapper definition from the skeleton
            text = map_cpp.read_text()
            text = text.replace("Room map_room(const ::room_info& in) {",
                                "// removed Room map_room(const ::room_info& in) {")
            map_cpp.write_text(text)
            self.assertNotEqual(self._run_drift(gen, map_cpp, header), 0)

    def test_codec_type_removed_detected(self):
        with tempfile.TemporaryDirectory() as d:
            gen, map_cpp, header = self._stage(Path(d))
            # a codec type the mapper references disappears -> signature no longer matches
            header.write_text(header.read_text().replace("struct room_info {", "struct gone {"))
            self.assertNotEqual(self._run_drift(gen, map_cpp, header), 0)


class RealMapTests(unittest.TestCase):
    """Validates the actual daemon-app entity map when MIRROR_ENTITY_MAP is set."""

    def _map(self):
        p = os.environ.get(MAP_ENV)
        if not p:
            self.skipTest(f"{MAP_ENV} not set (daemon-app entity map path)")
        path = Path(p)
        if not path.exists():
            self.skipTest(f"{MAP_ENV}={p} does not exist")
        return path

    def test_real_map_grounds(self):
        # generate() runs full provenance grounding; a failure raises MapError.
        artifacts = ec.generate(CDDL, self._map())
        self.assertIn("entities_gen.h", artifacts)

    def test_census_complete(self):
        entities = ec.load_map(self._map())
        names = {e.name for e in entities}
        missing = CENSUS_KINDS - names
        self.assertEqual(missing, set(), f"census rows missing from the map: {sorted(missing)}")

    def test_real_map_deterministic(self):
        a = ec.generate(CDDL, self._map())
        b = ec.generate(CDDL, self._map())
        for name in a:
            self.assertEqual(a[name], b[name])


if __name__ == "__main__":
    unittest.main()
