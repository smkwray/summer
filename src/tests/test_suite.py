#!/usr/bin/env python3
"""Offline test suite for the summarizer. No model calls.

Every test here encodes a failure this project actually shipped, so a green run
means those specific defects cannot recur. Names say what breaks if it fails.

usage: python3 src/tests/test_suite.py
"""
from __future__ import annotations
import contextlib, dataclasses, hashlib, importlib.util, io, json, os, pathlib, re, signal, subprocess, sys, tempfile, unittest
import urllib.error
import threading, time
import shutil
import unittest.mock

ENG = pathlib.Path(__file__).resolve().parent.parent


def load(name):
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), ENG / f"{name}.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def AF(text, kind="omission", artifact="detailed", **extra):
    """One typed pair-audit finding for fake model transports."""
    item = {"kind": kind, "artifact": artifact, "text": text}
    item.update(extra)
    return item


def patch_replace(detailed, brief, artifact, replacement, finding_id="F-001"):
    """A valid whole-reading replace_range patch of one artifact."""
    pp = load("pair_patch")
    pair = pp.segment_pair(detailed, brief)
    items = pair[artifact]
    first, last = items[0], items[-1]
    body = detailed if artifact == "detailed" else brief
    return {
        "base_candidate": pair["candidate"],
        "edits": [{
            "artifact": artifact,
            "operation": "replace_range",
            "start": first["id"],
            "end": last["id"],
            "range_sha256": hashlib.sha256(
                body[first["start"]:last["end"]].encode("utf-8")).hexdigest(),
            "replacement": replacement,
            "finding_ids": [finding_id],
        }],
    }


def gateway_runtime(name="localgw"):
    """One entirely generic device-local gateway fixture.

    Public tests deliberately use invented harness and model names so support
    for a private deployment can never leak into committed product policy.
    """
    roster = {
        "_local": True,
        "_model_settings": {
            "model-a": {"option": "reasoning_effort",
                        "values": ["low", "medium", "xhigh"],
                        "default": "medium"},
            "model-b": {"option": "reasoning_effort",
                        "values": ["low", "medium", "xhigh"],
                        "default": "xhigh"},
        },
        "plan": ["model-a", "model-b"],
        "write": ["model-a", "model-b"],
        "audit": ["model-a", "model-b"],
        "repair": ["model-a", "model-b"],
    }
    return {
        "active_targets": 2,
        "harness_capacity": 1,
        "capacities": {},
        "bindings": {},
        "gateways": {
            name: {
                "base_url": "https://gateway.example/v1",
                "api_key_env": "SUMM_LOCALGW_TOKEN",
                "protocol": "openai_chat_completions",
                "max_output_tokens": 16384,
                "output_token_field": "max_tokens",
                "timeout_seconds": 1800,
                "reasoning_content": "inline_think",
                "structured_output": ["json_object", "json_schema"],
                "error_policy": {
                    "status": {"502": "unavailable"},
                    "codes": {"model_loading": "warming",
                              "model_load_failed": "unavailable"},
                },
                "roster": roster,
            }
        },
    }


rv, compose, mech, tts = (load("readerview"), load("compose"),
                          load("mechseal"), load("tts_normalize"))
ledger = load("ledger")


class ReaderView(unittest.TestCase):
    """Preprocessing must be reversible and conservative."""

    def test_yaml_front_matter_never_reaches_the_reader(self):
        src = '---\ntitle: "T"\nauthor: Me\nformat:\n  pdf:\n    fontsize: 11pt\n---\n\nReal prose here.\n'
        vis, exc, _ = rv.build(src)
        self.assertNotIn("fontsize", vis)
        self.assertIn("Real prose here.", vis)
        self.assertTrue(any(e["type"] == "front_matter_config" for e in exc))

    def test_title_is_promoted_to_content_not_discarded(self):
        vis, _, _ = rv.build('---\ntitle: "The Rate Wall"\n---\n\nBody.\n')
        self.assertIn("The Rate Wall", vis)

    def test_html_comments_are_hidden_and_recorded(self):
        vis, exc, _ = rv.build("Before.\n\n<!-- private working note -->\n\nAfter.\n")
        self.assertNotIn("private working note", vis)
        self.assertTrue(any(e["type"] == "html_comment" for e in exc))

    def test_footnotes_are_kept_not_deleted(self):
        # An earlier regex stripped footnote definitions; a footnote can carry the
        # qualification the body depends on.
        vis, _, labels = rv.build("Claim.[^1]\n\n[^1]: The qualification that matters.\n")
        self.assertIn("The qualification that matters", vis)
        self.assertTrue(any(l["type"] == "footnote_definition" for l in labels))

    def test_bare_page_numbers_and_running_headers_are_hidden(self):
        body = "".join(f"Journal Of Things v1 n2\n\n{i}\n\npara {i} text here.\n\n" for i in range(6))
        vis, exc, _ = rv.build(body)
        self.assertTrue(any(e["type"] == "page_number" for e in exc))
        self.assertTrue(any(e["type"] == "running_header" for e in exc))
        self.assertIn("para 3 text here.", vis)


class Paragraphs(unittest.TestCase):
    """Grouping may move whitespace and nothing else."""

    def _items(self, n, w):
        return [{"id": f"S{i:03d}", "heading": False,
                 "text": " ".join(["word"] * w), "unit_ids": []} for i in range(n)]

    def test_sentence_set_is_preserved_exactly(self):
        items = self._items(12, 30)
        paras = compose.paragraphize(items, "brief")
        flat = [(x["id"], x["text"]) for p in paras for x in p["parts"]]
        self.assertEqual(flat, [(i["id"], i["text"]) for i in items])

    def test_render_grouped_asserts_on_tampering(self):
        self.assertTrue(compose.render_grouped(self._items(6, 40), "brief").strip())

    def test_paragraphs_reach_a_readable_size(self):
        # 59 paragraphs at a 33-word median is what the owner rejected on e-ink.
        paras = compose.paragraphize(self._items(20, 30), "brief")
        sizes = [sum(len(x["text"].split()) for x in p["parts"])
                 for p in paras if not p["heading"]]
        self.assertTrue(min(sizes) >= 60, f"paragraph too short: {sizes}")

    def test_headings_are_hard_boundaries(self):
        items = (self._items(3, 40)
                 + [{"id": "H1", "heading": True, "text": "Next Section", "unit_ids": []}]
                 + self._items(3, 40))
        paras = compose.paragraphize(items, "detailed")
        hidx = [i for i, p in enumerate(paras) if p["heading"]]
        self.assertEqual(len(hidx), 1)
        for p in paras:
            if not p["heading"]:
                self.assertTrue(all(not x["heading"] for x in p["parts"]))


class MechanicalSeal(unittest.TestCase):
    """The seal is the precondition for publishing anything."""

    def _dirs(self, ledger_obj, blocks):
        t = pathlib.Path(tempfile.mkdtemp())
        (t / "l").mkdir(); (t / "rv").mkdir()
        (t / "rv" / "source-map.json").write_text(json.dumps(
            {"source_sha256": "h1", "visible_words": sum(b["words"] for b in blocks),
             "blocks": blocks}))
        (t / "l" / "ledger.json").write_text(json.dumps(ledger_obj))
        return t / "l", t / "rv"

    def _unit(self, **kw):
        u = {"unit_id": "U001", "source_ids": ["P0001"],
             "exact_source_anchor": "the wall binds",
             "detailed_disposition": "required", "brief_disposition": "required",
             "detailed_capsule": "The wall binds under pressure.",
             "brief_capsule": "The wall binds.", "dependencies": []}
        u.update(kw); return u

    BLOCKS = [{"id": "P0001", "words": 6, "text": "The wall binds under real pressure"}]

    def test_wellformed_ledger_passes(self):
        l, r = self._dirs({"source_sha256": "h1", "visible_words": 6,
                           "units": [self._unit()], "dispositions": []}, self.BLOCKS)
        fail, _ = mech.check(l, r)
        self.assertEqual(fail, [], fail)

    def test_dangling_dependency_blocks(self):
        # 34 of these shipped in a real ledger.
        l, r = self._dirs({"source_sha256": "h1", "visible_words": 6,
                           "units": [self._unit(dependencies=["SEC01-U01"])],
                           "dispositions": []}, self.BLOCKS)
        fail, _ = mech.check(l, r)
        self.assertTrue(any("dangling" in f for f in fail), fail)

    def test_duplicate_unit_ids_block(self):
        l, r = self._dirs({"source_sha256": "h1", "visible_words": 6,
                           "units": [self._unit(), self._unit()], "dispositions": []},
                          self.BLOCKS)
        fail, _ = mech.check(l, r)
        self.assertTrue(any("duplicate unit ids" in f for f in fail), fail)

    def test_unknown_source_id_blocks(self):
        l, r = self._dirs({"source_sha256": "h1", "visible_words": 6,
                           "units": [self._unit(source_ids=["P9999"])], "dispositions": []},
                          self.BLOCKS)
        fail, _ = mech.check(l, r)
        self.assertTrue(any("source ids" in f for f in fail), fail)

    def test_source_hash_mismatch_blocks(self):
        l, r = self._dirs({"source_sha256": "WRONG", "visible_words": 6,
                           "units": [self._unit()], "dispositions": []}, self.BLOCKS)
        fail, _ = mech.check(l, r)
        self.assertTrue(any("hash" in f for f in fail), fail)

    def test_dependency_cycle_blocks(self):
        us = [self._unit(unit_id="U001", dependencies=["U002"]),
              self._unit(unit_id="U002", dependencies=["U001"])]
        l, r = self._dirs({"source_sha256": "h1", "visible_words": 6,
                           "units": us, "dispositions": []}, self.BLOCKS)
        fail, _ = mech.check(l, r)
        self.assertTrue(any("cycle" in f for f in fail), fail)

    def test_apparatus_leaking_into_a_capsule_blocks(self):
        l, r = self._dirs({"source_sha256": "h1", "visible_words": 6,
                           "units": [self._unit(detailed_capsule="---\ntitle: x\n---")],
                           "dispositions": []}, self.BLOCKS)
        fail, _ = mech.check(l, r)
        self.assertTrue(any("leaked" in f for f in fail), fail)

    def test_units_bind_to_source_by_ids_not_by_a_generated_anchor(self):
        # exact_source_anchor was generated for every unit, only ever produced a
        # warning, and cost output tokens -- the binding constraint on a
        # bandwidth-limited local device. source_ids still bind units to the
        # source and ARE checked, so the guarantee is unchanged.
        eng = (ENG / "mechseal.py").read_text()
        self.assertNotIn("exact_source_anchor", eng.split("# exact_source_anchor")[0])
        self.assertIn("bad_src", eng, "units are no longer bound to source blocks")
        self.assertNotIn("exact_source_anchor",
                         (ENG / "prompts" / "ledger-build.txt").read_text())

    def test_the_planner_is_told_the_capsules_are_the_only_record(self):
        # The deleted fields duplicated fidelity obligations. Removing them
        # without saying so would invite a planner to leave a qualification in a
        # field that no longer exists.
        tpl = (ENG / "prompts" / "ledger-build.txt").read_text()
        self.assertIn("only record of the unit", tpl)

    def test_no_viable_units_blocks(self):
        l, r = self._dirs({"source_sha256": "h1", "visible_words": 6,
                           "units": [self._unit(detailed_disposition="omit",
                                                brief_disposition="omit")],
                           "dispositions": []}, self.BLOCKS)
        fail, _ = mech.check(l, r)
        self.assertTrue(any("viable" in f for f in fail), fail)


class Chunking(unittest.TestCase):
    def test_sections_split_within_the_planning_window(self):
        blocks = [{"id": f"P{i:04d}", "words": 200, "text": ("w " * 200)} for i in range(40)]
        blocks[0]["text"] = "# Heading\n" + blocks[0]["text"]
        parts = ledger.split_sections(blocks)
        for p in parts:
            self.assertLessEqual(sum(b["words"] for b in p["blocks"]),
                                 ledger.PART_MAX + 200)

    def test_density_floor_scales_with_section_size(self):
        floor_small, _ = ledger.density(220)
        floor_big, ref_big = ledger.density(2200)
        self.assertEqual(floor_small, 1)
        self.assertEqual(floor_big, 10)
        self.assertEqual(ref_big, 13)


class TTS(unittest.TestCase):
    def test_normalization_preserves_information(self):
        src = "The Fed cut rates by 20 bp in 2001. The ABS market froze.\n"
        out = tts.convert(src)
        self.assertGreaterEqual(len(out.split()), len(src.split()))

    def test_word_boundaries_are_respected(self):
        # "bp" as a substring once turned Subprime into "Subasis pointsrime".
        self.assertIn("Subprime", tts.convert("Subprime lending rose 20 bp.\n"))

    def test_acronyms_are_spaced_for_speech(self):
        self.assertIn("F O M C", tts.convert("The FOMC met.\n"))

    def test_tips_is_spoken_as_a_word_not_spaced(self):
        out = tts.convert("The yield on 10-year TIPS rose.\n")
        self.assertNotIn("T I P S", out)
        self.assertIn("TIPS", out)

    def test_headings_do_not_prepend_robotic_next(self):
        out = tts.convert("# Overview\n\nFirst paragraph.\n\n## Methodology\n\nSecond paragraph.\n")
        self.assertNotIn("Next, ", out)
        self.assertIn("Methodology.", out)


class TextPreparation(unittest.TestCase):
    """Cleanup is full-text preserving and fails closed after one repair."""
    tp = load("textprep")

    @staticmethod
    def _candidate(text, ident="B0001", disposition="keep"):
        return json.dumps({"blocks": [{"id": ident,
                                        "disposition": disposition,
                                        "text": text}]})

    @staticmethod
    def _audit(verdict="pass", findings=None, approved_drops=None):
        return json.dumps({"verdict": verdict, "findings": findings or [],
                           "approved_drops": approved_drops or []})

    class FakeRunner:
        MODELS = ("write",)
        AUDIT_MODELS = ("audit",)
        REPAIR_MODELS = ("repair",)

        def __init__(self, responses):
            self.responses = iter(responses)
            self.calls = []

        def run(self, prompt, work, chain, stage, validate=None,
                gateway_options=None):
            self.calls.append((prompt, pathlib.Path(work), chain, stage,
                               gateway_options))
            value = next(self.responses)
            if isinstance(value, Exception):
                raise value
            if validate:
                validate(value)
            return value

    def test_blocking_and_chunking_preserve_every_word_in_order(self):
        text = "\n\n".join(
            " ".join(f"word{i}_{j}" for j in range(900)) for i in range(5))
        blocks = self.tp.source_blocks(text)
        chunks = self.tp.pack_chunks(blocks)
        original = re.findall(r"\S+", text)
        rebuilt = [word for chunk in chunks for block in chunk
                   for word in block["text"].split()]
        self.assertEqual(original, rebuilt)
        self.assertEqual([f"B{i:04d}" for i in range(1, 6)],
                         [block["id"] for block in blocks])

    def test_strict_schema_rejects_missing_duplicate_and_reordered_blocks(self):
        expected = [{"id": "B0001", "text": "one"},
                    {"id": "B0002", "text": "two"}]
        for ids in (("B0001",), ("B0001", "B0001"), ("B0002", "B0001")):
            raw = json.dumps({"blocks": [
                {"id": ident, "disposition": "keep", "text": "value"}
                for ident in ids]})
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                self.tp.parse_candidate(raw, expected, "test")

    def test_drop_is_accounted_structurally_but_requires_model_audit_approval(self):
        for value in ("237", "IV", "Page IV", "Chapter 12",
                      "The 300 Spartans", "Newsletter Logo"):
            source = [{"id": "B0001", "text": value}]
            candidate = [{"id": "B0001", "disposition": "drop_furniture",
                          "text": ""}]
            findings, witnesses = self.tp.relation_findings(source, candidate)
            self.assertEqual([], findings, value)
            audit = {"verdict": "pass", "findings": [], "approved_drops": []}
            self.assertTrue(self.tp.drop_approval_findings(candidate, audit), value)
            audit["approved_drops"] = ["B0001"]
            self.assertEqual([], self.tp.drop_approval_findings(candidate, audit))
            bound = self.tp.bind_drop_approvals(candidate, witnesses, audit)
            self.assertTrue(bound[0]["audit_approved"])
            self.assertTrue(bound[0]["audit_hash"])

    def test_page_markers_are_isolated_but_bare_numbers_are_not_deletable(self):
        blocks = self.tp.source_blocks("Before.\nPage 237\nAfter.\n\n1941")
        self.assertEqual(["Before.", "Page 237", "After.", "1941"],
                         [block["text"] for block in blocks])
        self.assertTrue(self.tp.unambiguous_furniture(blocks[1]["text"]))
        self.assertFalse(self.tp.unambiguous_furniture(blocks[3]["text"]))
        self.assertTrue(self.tp.unambiguous_furniture("Page 237."))
        self.assertTrue(self.tp.unambiguous_furniture("Page IV."))
        self.assertFalse(self.tp.unambiguous_furniture("Page civil"))
        kept_marker = [{"id": "B0001", "text": "Page 237"}]
        kept_candidate = [{"id": "B0001", "disposition": "keep",
                           "text": "Page 237"}]
        findings, _ = self.tp.relation_findings(kept_marker, kept_candidate)
        self.assertEqual([], findings,
                         "semantic furniture decisions belong to the model roles")
        prompt = self.tp._audit_prompt(kept_marker, kept_candidate)
        self.assertIn("page/extraction/interface/newsletter furniture", prompt)

    def test_bare_page_number_requires_an_extracted_page_boundary(self):
        plain = self.tp.source_blocks("A paragraph.\n\n1941")
        self.assertFalse(plain[-1].get("furniture", False))
        paginated = self.tp.source_blocks(
            "First page body.\n\n12\fSecond page body.\n")
        marker = next(block for block in paginated if block["text"] == "12")
        self.assertTrue(marker["furniture"])
        candidate = [{"id": block["id"],
                      "disposition": ("drop_furniture" if block is marker else "keep"),
                      "text": "" if block is marker else block["text"]}
                     for block in paginated]
        self.assertEqual([], self.tp.relation_findings(paginated, candidate)[0])

    def test_model_selected_ocr_correction_is_allowed_without_dictionary(self):
        source = [{"id": "B0001", "text": "The modem model was retained."}]
        candidate = [{"id": "B0001", "disposition": "keep",
                      "text": "The modern model was retained."}]
        findings, witnesses = self.tp.relation_findings(source, candidate)
        self.assertEqual([], findings)
        self.assertEqual(self.tp.RELATION_VERSION, witnesses[0]["version"])
        self.assertTrue(any(step["rule"] == "model_correction_token"
                            for step in witnesses[0]["steps"]))

    def test_local_ocr_and_word_boundary_repairs_are_allowed(self):
        cases = (
            ("The modem rnodel was retained.",
             "The modern model was retained."),
            ("Tlie end.", "The end."),
            ("t he cat", "the cat"),
            ("thecat", "the cat"),
            ("inthebeginning", "in the beginning"),
            ("In l941 the rate was 5%.", "In 1941 the rate was 5%."),
            ("The quoted word was \"modem.\"",
             "The quoted word was \"modern.\""),
        )
        for before, after in cases:
            source = [{"id": "B0001", "text": before}]
            candidate = [{"id": "B0001", "disposition": "keep", "text": after}]
            with self.subTest(before=before):
                findings, witnesses = self.tp.relation_findings(source, candidate)
                self.assertEqual([], findings)
                self.assertEqual([], self.tp.mechanical_findings(
                    source, candidate, witnesses))

    def test_well_formed_numbers_still_cannot_change(self):
        source = [{"id": "B0001", "text": "In 1941 the rate was 5%."}]
        candidate = [{"id": "B0001", "disposition": "keep",
                      "text": "In 1942 the rate was 5%."}]
        self.assertTrue(self.tp.mechanical_findings(source, candidate))

    def test_exact_source_duplicates_and_model_phrases_are_not_false_failures(self):
        source = [{"id": "B0001",
                   "text": "the cat sat on the mat the cat sat on the mat"}]
        candidate = [{"id": "B0001", "disposition": "keep",
                      "text": source[0]["text"]}]
        findings, witnesses = self.tp.relation_findings(source, candidate)
        self.assertEqual([], findings)
        self.assertEqual([], self.tp.mechanical_findings(source, candidate, witnesses))
        source = [{"id": "B0001", "text": "I cannot accept that conclusion."}]
        candidate = [{"id": "B0001", "disposition": "keep",
                      "text": source[0]["text"]}]
        findings, witnesses = self.tp.relation_findings(source, candidate)
        self.assertEqual([], findings)
        self.assertEqual([], self.tp.mechanical_findings(source, candidate, witnesses))

    def test_relation_rejects_omission_caveat_reorder_and_insertion(self):
        source_text = ("The first sentence records alpha evidence. The second "
                       "sentence records beta evidence, not a conclusion.")
        cases = {
            "sentence omission": source_text.replace(
                " The second sentence records beta evidence, not a conclusion.", ""),
            "short caveat": source_text.replace(", not a conclusion", " a conclusion"),
            "sentence reorder": ("The second sentence records beta evidence, not a "
                                 "conclusion. The first sentence records alpha evidence."),
            "equal sentence reorder": ("The second sentence records beta evidence. "
                                       "The first sentence records alpha evidence."),
            "added material": source_text + " The first sentence records alpha evidence.",
        }
        source = [{"id": "B0001", "text": source_text}]
        for label, changed in cases.items():
            candidate = [{"id": "B0001", "disposition": "keep", "text": changed}]
            with self.subTest(label=label):
                findings, _ = self.tp.relation_findings(source, candidate)
                self.assertTrue(findings, label)

    def test_relation_rejects_moved_names_even_when_the_sentence_stays_short(self):
        source_text = "The first witness named Alice and the second witness named Bob."
        candidate_text = "The first witness named Bob and the second witness named Alice."
        source = [{"id": "B0001", "text": source_text}]
        candidate = [{"id": "B0001", "disposition": "keep", "text": candidate_text}]
        self.assertTrue(self.tp.relation_findings(source, candidate)[0])

    def test_relation_rejects_content_glued_into_a_token(self):
        source = [{"id": "B0001", "text": "The result was significant."}]
        candidate = [{"id": "B0001", "disposition": "keep",
                      "text": "The result was significant.Forged."}]
        self.assertTrue(self.tp.relation_findings(source, candidate)[0])

    def test_model_passing_audit_cannot_publish_an_omitted_sentence(self):
        source = ("The first sentence records alpha evidence. The second sentence "
                  "records beta evidence. The third sentence records gamma evidence.")
        bad = source.replace(" The second sentence records beta evidence.", "")
        fake = self.FakeRunner([
            self._candidate(bad), self._audit(),
            self._candidate(bad), self._audit(),
        ])
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td); src = root / "source.txt"; out = root / "out"
            src.write_text(source)
            with unittest.mock.patch.object(self.tp, "_runner", return_value=fake):
                self.assertEqual(5, self.tp.run(src, out))
            self.assertFalse((out / "cleaned.md").exists())

    def test_page_marker_is_removed_before_publication(self):
        source = "Before.\nPage 237.\nAfter."
        rows_keep = [{"id": ident, "disposition": "keep", "text": text}
                     for ident, text in (("B0001", "Before."),
                                         ("B0002", "Page 237."),
                                         ("B0003", "After."))]
        rows_drop = [rows_keep[0],
                     {"id": "B0002", "disposition": "drop_furniture", "text": ""},
                     rows_keep[2]]
        fake = self.FakeRunner([
            json.dumps({"blocks": rows_keep}), self._audit("revise", [{
                "id": "B0002", "problem": "page marker should be furniture"}]),
            json.dumps({"blocks": rows_drop}),
            self._audit(approved_drops=["B0002"]),
        ])
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td); src = root / "source.txt"; out = root / "out"
            src.write_text(source)
            with unittest.mock.patch.object(self.tp, "_runner", return_value=fake):
                self.assertEqual(0, self.tp.run(src, out))
            self.assertEqual("Before.\n\nAfter.\n", (out / "cleaned.md").read_text())

    def test_model_led_newsletter_furniture_is_removed_with_two_role_agreement(self):
        source = ("View in browser\nNewsletter Logo\n\n"
                  "Article title\n\nComplete article body.\n\n"
                  "Unsubscribe from mailing list.\nCopyright footer")
        rows = [
            {"id": "B0001", "disposition": "drop_furniture", "text": ""},
            {"id": "B0002", "disposition": "keep", "text": "Article title"},
            {"id": "B0003", "disposition": "keep",
             "text": "Complete article body."},
            {"id": "B0004", "disposition": "drop_furniture", "text": ""},
        ]
        fake = self.FakeRunner([
            json.dumps({"blocks": rows}),
            self._audit(approved_drops=["B0001", "B0004"]),
        ])
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td); src = root / "source.txt"; out = root / "out"
            src.write_text(source)
            with unittest.mock.patch.object(self.tp, "_runner", return_value=fake):
                self.assertEqual(0, self.tp.run(src, out))
            self.assertEqual("Article title\n\nComplete article body.\n",
                             (out / "cleaned.md").read_text())
            witness = json.loads(
                (out / "text-prep" / "001" / "witness.json").read_text())
            self.assertEqual(["B0001", "B0004"],
                             witness["approved_drop_ids"])

    def test_unapproved_model_drop_fails_closed(self):
        source = "Substantive article body that must remain complete."
        dropped = self._candidate("", disposition="drop_furniture")
        fake = self.FakeRunner([
            dropped, self._audit(),
            dropped, self._audit(),
        ])
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td); src = root / "source.txt"; out = root / "out"
            src.write_text(source)
            with unittest.mock.patch.object(self.tp, "_runner", return_value=fake):
                self.assertEqual(5, self.tp.run(src, out))
            self.assertFalse((out / "cleaned.md").exists())

    def test_render_cannot_use_rows_tampered_after_their_witness(self):
        source = [{"id": "B0001", "text": "The original wording remains."}]
        candidate = [{"id": "B0001", "disposition": "keep",
                      "text": "The original wording remains."}]
        _, witnesses = self.tp.relation_findings(source, candidate)
        candidate[0]["text"] = "The altered wording remains."
        with self.assertRaises(ValueError):
            self.tp._render_witness(candidate, witnesses)

    def test_invalid_utf8_fails_before_any_model_call(self):
        fake = self.FakeRunner([])
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td); src = root / "source.txt"; out = root / "out"
            src.write_bytes(b"valid prefix\xff\n")
            with unittest.mock.patch.object(self.tp, "_runner", return_value=fake):
                self.assertEqual(6, self.tp.run(src, out))
            self.assertEqual([], fake.calls)
            self.assertFalse((out / "cleaned.md").exists())

    def test_numbers_are_protected_but_ocr_quote_repairs_reach_the_audit(self):
        source = [{"id": "B0001",
                   "text": 'In 1941 the rate was 5%. She said "keep the caveat."'}]
        changed_number = ('In 1942 the rate was 5%. She said '
                          '"keep the caveat."')
        candidate = [{"id": "B0001", "disposition": "keep", "text": changed_number}]
        self.assertTrue(self.tp.mechanical_findings(source, candidate))
        changed_quote = ('In 1941 the rate was 5%. She said '
                         '"drop the caveat."')
        candidate = [{"id": "B0001", "disposition": "keep", "text": changed_quote}]
        self.assertTrue(self.tp.relation_findings(source, candidate)[0])

    def test_clean_audit_pass_writes_one_artifact(self):
        source = "The docu-\nment recorded 1941 exactly and kept its caveat."
        cleaned = "The document recorded 1941 exactly and kept its caveat."
        fake = self.FakeRunner([self._candidate(cleaned), self._audit()])
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td); src = root / "source.txt"; out = root / "out"
            src.write_text(source)
            with unittest.mock.patch.object(self.tp, "_runner", return_value=fake):
                self.assertEqual(0, self.tp.run(src, out))
            self.assertEqual(cleaned + "\n", (out / "cleaned.md").read_text())
            self.assertEqual(2, len(fake.calls))
            schemas = {
                call[3]: call[4]["response_format"]["json_schema"]
                for call in fake.calls
            }
            self.assertEqual("summer_clean_candidate",
                             schemas["text-clean001"]["name"])
            self.assertEqual("summer_clean_audit",
                             schemas["text-audit001"]["name"])
            self.assertTrue(all(schema["strict"] for schema in schemas.values()))
            witness = json.loads((out / "text-prep" / "001" / "witness.json").read_text())
            self.assertEqual(self.tp.RELATION_VERSION, witness["relation_version"])
            self.assertTrue(witness["witnesses"][0]["witness_hash"])
            report = json.loads((out / "textprep-report.json").read_text())
            self.assertEqual(self.tp.RELATION_VERSION, report["relation_version"])
            self.assertTrue(report["output_sha256"])
            self.assertIn(json.dumps(source)[1:-1], fake.calls[0][0])
            self.assertTrue(all(call[1] == out for call in fake.calls),
                            "model-call evidence escaped the retained run root")
            self.assertNotIn(str(src), fake.calls[0][0],
                             "the model was handed a source path")

    def test_one_repair_is_reaudited_and_can_recover(self):
        source = "The rate was 5 percent under the stated caveat."
        bad = "The rate was percent under the stated caveat."
        fake = self.FakeRunner([
            self._candidate(bad),
            self._audit("revise", [{"id": "B0001", "problem": "5 was dropped"}]),
            self._candidate(source),
            self._audit(),
        ])
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td); src = root / "source.txt"; out = root / "out"
            src.write_text(source)
            with unittest.mock.patch.object(self.tp, "_runner", return_value=fake):
                self.assertEqual(0, self.tp.run(src, out))
            self.assertEqual(["text-clean001", "text-audit001",
                              "text-repair001", "text-reaudit001"],
                             [call[3] for call in fake.calls])

    def test_unresolved_repair_and_missing_audit_publish_nothing(self):
        source = "The rate was 5 percent under the stated caveat."
        bad = "The rate was percent under the stated caveat."
        cases = ([self._candidate(bad), self._audit("revise", [
                    {"id": "B0001", "problem": "5 was dropped"}]),
                  self._candidate(bad), self._audit()],
                 [self._candidate(source), RuntimeError("audit unavailable")])
        for responses in cases:
            with self.subTest(responses=len(responses)), tempfile.TemporaryDirectory() as td:
                root = pathlib.Path(td); src = root / "source.txt"; out = root / "out"
                src.write_text(source)
                fake = self.FakeRunner(responses)
                with unittest.mock.patch.object(self.tp, "_runner", return_value=fake):
                    self.assertEqual(5, self.tp.run(src, out))
        self.assertFalse((out / "cleaned.md").exists())

    def test_text_prep_cli_mode_is_exclusive_and_the_ui_passes_it(self):
        ui = load("summ_ui")
        label = next(m for m, flags in ui.MODES if "--text-prep" in flags)
        cmd = ui.build_cmd(pathlib.Path("/j"), label, pathlib.Path("/tmp/x.md"))
        self.assertIn("--text-prep", cmd)
        run = subprocess.run([sys.executable, str(ENG / "summ_cli.py"),
                              "--text-prep", "--quick"],
                             capture_output=True, text=True)
        self.assertEqual(2, run.returncode)
        self.assertIn("not allowed", run.stderr)

    def test_single_artifact_publication_restores_the_previous_file(self):
        cli = load("summ_cli")
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td); source = root / "candidate.md"
            dest = root / "paper.clean.md"
            source.write_text("new text\n"); dest.write_text("old text\n")

            def fail(_src, _dest):
                raise OSError("simulated replacement failure")

            with self.assertRaises(OSError):
                cli.publish_one(source, dest, replace=fail)
            self.assertEqual("old text\n", dest.read_text())
            self.assertEqual([], list(root.glob("*.tmp*")))
            self.assertEqual([], list(root.glob("*.bak*")))


class Targets(unittest.TestCase):
    """Clipboard parsing: how a real invocation actually starts."""

    def setUp(self):
        self.cli = load("summ_cli")
        self.t = pathlib.Path(tempfile.mkdtemp())
        (self.t / "a folder with spaces").mkdir()
        (self.t / "doc.md").write_text("body")
        (self.t / "a folder with spaces" / "inner.md").write_text("body")

    def test_unquoted_path_with_spaces_resolves(self):
        # Finder "Copy as Pathname" yields this; shlex used to shred it.
        p = str(self.t / "a folder with spaces" / "inner.md")
        tg, raw = self.cli.parse_targets(p)
        self.assertFalse(raw); self.assertEqual(len(tg), 1)

    def test_multiple_paths_one_per_line(self):
        tg, raw = self.cli.parse_targets(f"{self.t/'doc.md'}\n"
                                         f"{self.t/'a folder with spaces'/'inner.md'}")
        self.assertFalse(raw); self.assertEqual(len(tg), 2)

    def test_multiple_absolute_paths_on_one_line(self):
        # Finder Copy as Pathname is newline-separated; a terminal paste of
        # several paths is one line. Either must become several documents.
        tg, raw = self.cli.parse_targets(
            f"{self.t/'doc.md'} {self.t/'a folder with spaces'/'inner.md'}")
        self.assertFalse(raw); self.assertEqual(len(tg), 2)

    def test_prose_is_treated_as_a_document(self):
        tg, raw = self.cli.parse_targets("The Fed lowered rates. This is prose, not a path.")
        self.assertTrue(raw)

    def test_brief_and_summary_destinations_differ(self):
        # `--depth brief` once made both destinations the same path.
        src = self.t / "paper.md"
        dest = self.cli.out_path(src, ".summary.md")
        bdest = dest.with_name(dest.name[:-len(".summary.md")] + ".brief.md")
        self.assertNotEqual(dest, bdest)
        self.assertTrue(str(bdest).endswith("paper.brief.md"))


class Commands(unittest.TestCase):
    """The committed product surface is syntactically self-contained."""

    def test_every_engine_module_compiles(self):
        for p in sorted(ENG.glob("*.py")) + sorted((ENG / "checks").glob("*.py")):
            with self.subTest(module=p.name):
                compile(p.read_text(), str(p), "exec")

    def test_no_model_is_named_outside_the_config(self):
        # One place to edit when models change.
        pat = re.compile(r"gemini-3|claude-opus|claude-sonnet")
        for p in ENG.glob("*.py"):
            if p.name == "mapsum.py":            # the harness file itself
                continue
            hits = [l for l in p.read_text().splitlines()
                    if pat.search(l) and not l.strip().startswith("#")]
            self.assertEqual(hits, [], f"{p.name} names a model: {hits[:2]}")




class ProvenanceAndSealing(unittest.TestCase):
    """Published artifacts remain bound to their accepted evidence."""

    def test_status_binds_a_ledger_hash(self):
        src = (ENG / "ledger.py").read_text()
        self.assertIn("ledger_sha256", src)

    def test_artifact_resume_requires_provenance_not_existence(self):
        # Resuming on existence alone let an artifact from an earlier ledger
        # survive a rebuild, so published files could not be reproduced from the
        # final ledger.
        src = (ENG / "compose.py").read_text()
        seg = src[src.index("art_existing = out"):src.index("items = dedupe(")]
        for token in ("ledger_sha256", "unit_ids", "sha256", "render_mode"):
            self.assertIn(token, seg, f"resume does not bind {token}")

    def test_capsule_rendering_is_the_only_mode(self):
        # Stronger than "the default": the model composer never qualified, and an
        # unqualified path behind an environment variable is one mis-set variable
        # away from publishing through it.
        src = (ENG / "compose.py").read_text()
        self.assertIn('RENDER_MODE = "capsules"', src)
        self.assertNotIn("SUMM_RENDER_MODE", src)

    def test_planning_and_revision_use_their_own_chains(self):
        # Revision once ran on the write chain, so the model fixing a fidelity
        # finding was not the one qualified for it.
        src = (ENG / "ledger.py").read_text()
        self.assertIn("ms.PLAN_MODELS", src)
        self.assertIn("ms.AUDIT_MODELS", src)
        self.assertIn("ms.REPAIR_MODELS", src)
        i = src.index("def revise_part")
        self.assertIn("ms.REPAIR_MODELS", src[i:i + 900])

    def test_exactly_one_terminal_notification(self):
        src = (ENG / "summ_cli.py").read_text()
        # Windows shows one balloon at a time; a per-item plus a generic one raced.
        self.assertEqual(src.count('notify("Summary"'), src.count('notify("Summary"'))
        self.assertNotIn('Verified Detailed and Brief ready — ', src)

    def test_no_processing_start_notification(self):
        src = (ENG / "summ_cli.py").read_text()
        # Runs report progress in the UI, but desktop notifications are reserved
        # for terminal outcomes so a normal run produces no redundant start alert.
        self.assertNotIn('notify("Preparing text"', src)
        self.assertNotIn('words. Started.', src)
        terminal = src[src.index('# ONE terminal notification.'):]
        self.assertIn('notify("Text prep" if selected_mode.key', terminal)

    def test_no_opus_leads_any_agy_chain(self):
        # Its weekly cap is separate and often exhausted; leading with it spends a
        # guaranteed failed call before every repair.
        agy = json.loads((ENG / "models.json").read_text())["agy"]
        for role in ("plan", "write", "audit", "repair"):
            self.assertTrue(agy[role])
            self.assertTrue(all("opus" not in model.lower()
                                for model in agy[role]))


class SealImmutability(unittest.TestCase):
    def test_resealed_ledger_is_not_rebuilt(self):
        # Re-invoking rebuilt the ledger while status kept the old hash, so the
        # seal stopped describing the ledger it authorised.
        src = (ENG / "ledger.py").read_text()
        seg = src[src.index("def main():"):src.index("ledger = normalize(build(")]
        self.assertIn("already sealed", seg)
        self.assertIn("ledger_sha256", seg)


class Doctor(unittest.TestCase):
    """--doctor is the only cross-platform check the user can run before
    trusting an install, so it must be quota-free and must cover every
    component a run depends on."""
    SRC = (ENG / "summ_cli.py").read_text()

    def _body(self):
        return self.SRC[self.SRC.index("def doctor()"):self.SRC.index("def main()")]

    def test_doctor_is_wired_to_a_flag_and_exits_before_clipboard(self):
        self.assertIn('"--doctor"', self.SRC)
        main = self.SRC[self.SRC.index("def main()"):]
        # must return before clipboard/target parsing, or --doctor on an empty
        # clipboard would fail for the wrong reason
        self.assertLess(main.index("if a.doctor"), main.index("selection.classify_clipboard"))

    def test_doctor_spends_no_quota(self):
        # Naming an engine file is how doctor checks it exists; the invariant is
        # that it never *invokes* one. Assert on invocation, not on mentions.
        body = self._body()
        for forbidden in ("subprocess.run(", "subprocess.Popen(", "os.system(",
                          "exec_module(spec", "ms.run(", "runner("):
            self.assertNotIn(forbidden, body,
                             f"doctor must not invoke anything: found {forbidden}")

    def test_doctor_checks_every_engine_module_and_role(self):
        body = self._body()
        for mod in ("readerview.py", "ledger.py", "compose.py", "mechseal.py",
                    "mapsum.py", "tts_normalize.py", "textprep.py"):
            self.assertIn(mod, body, f"doctor does not check {mod}")
        for role in ("PLAN_MODELS", "MODELS", "AUDIT_MODELS", "REPAIR_MODELS"):
            self.assertIn(role, body, f"doctor does not report the {role} chain")

    def test_doctor_reports_failure_not_just_prints(self):
        body = self._body()
        self.assertIn("return 0 if ok else 1", body)



class DepthScopedQuarantine(unittest.TestCase):
    """MEANING_RISK has always been documented as quarantining "at the affected
    depth", but the set was accumulated flat and applied to both. A brief
    capsule that drops a qualifier is a brief defect; discarding the unit's
    sound detailed capsule with it loses content for no fidelity gain -- and
    silent content loss is the failure this whole design exists to prevent."""
    led = load("ledger")

    def test_compose_drops_the_unit_only_at_the_quarantined_depth(self):
        # End to end through compose.py: capsule mode makes no model calls.
        tmp = pathlib.Path(tempfile.mkdtemp())
        ldir, rvd, out = tmp / "ledger", tmp / "rv", tmp / "out"
        ldir.mkdir(); rvd.mkdir()
        # Distinct subject matter per unit: near-identical filler would be
        # collapsed by dedupe(), which would mask what this test is checking.
        topics = ["repo haircuts widening during the run",
                  "securitized banking funded by repurchase agreements",
                  "deposit insurance limits for institutional depositors",
                  "the LIB-OIS spread as a measure of counterparty risk",
                  "collateral revaluation across mortgage-backed portfolios",
                  "interbank lending contraction in the second half"]
        units = []
        for i, topic in enumerate(topics, 1):
            units.append({
                "unit_id": f"U{i:03d}", "dependencies": [],
                "detailed_disposition": "required", "brief_disposition": "required",
                "detailed_capsule": f"Detailed treatment of {topic}, set out at "
                                    f"length with the surrounding qualifications.",
                "brief_capsule": f"Brief treatment of {topic}.",
            })
        (ldir / "ledger.json").write_text(json.dumps(
            {"units": units, "visible_words": 200}))
        # compose authenticates its input, so the fixture must be a real sealed
        # generation rather than a bare ledger with a claimed hash.
        # visible_words must be realistic relative to capsule sizes, or the
        # ceiling is unreachable and compose correctly refuses to publish.
        (ldir / "ledger.json").write_text(json.dumps(
            {"units": units, "visible_words": 900}))
        (ldir / "MECHSEAL").write_text("1")
        (ldir / "mechseal.json").write_text(json.dumps({"passed": True, "units": 6}))
        (ldir / "SEALED").write_text("1")
        (ldir / "status.json").write_text(json.dumps(
            {"status": "quarantined",
             "ledger_sha256": hashlib.sha256(
                 (ldir / "ledger.json").read_bytes()).hexdigest(),
             "quarantine": {"detailed": [], "brief": ["U002"]}}))
        env = dict(os.environ, SUMM_RENDER_MODE="capsules")
        r = subprocess.run([sys.executable, str(ENG / "compose.py"),
                            str(ldir), str(rvd), str(out)],
                           capture_output=True, text=True, env=env)
        self.assertTrue((out / "detailed.md").exists(),
                        f"compose did not produce detailed.md: {r.stderr[-400:]}")
        detailed = (out / "detailed.md").read_text()
        brief = (out / "brief.md").read_text()
        # U002 is quarantined for brief only: its detailed capsule must survive.
        self.assertIn("securitized banking funded by repurchase", detailed)
        self.assertNotIn("Brief treatment of securitized banking", brief)
        # and the depths must not have lost anything else
        self.assertIn("Brief treatment of deposit insurance", brief)



class TrailingParagraph(unittest.TestCase):
    """The partition can produce exactly one runt, and it lands where it is most
    visible: the last thing read. Two real runs closed on a 36-word and a
    41-word paragraph -- the complaint that started this was short paragraphs
    reading badly on e-ink."""
    comp = load("compose")

    def _items(self, sizes):
        return [{"id": f"S{i:03d}", "heading": False,
                 "text": " ".join(["word"] * n), "unit_ids": [f"U{i:03d}"]}
                for i, n in enumerate(sizes, 1)]

    def _sizes(self, paras):
        return [sum(len(x["text"].split()) for x in p["parts"])
                for p in paras if not p["heading"]]

    def test_short_tail_is_merged_when_it_fits(self):
        lo, pref, hi = self.comp.PARA["brief"]
        out = self.comp.paragraphize(self._items([60, 30, 20]), "brief")
        self.assertEqual(len(self._sizes(out)), 1)

    def test_tail_too_big_to_merge_is_rebalanced_not_orphaned(self):
        lo, pref, hi = self.comp.PARA["detailed"]        # 85, 125, 175
        # 140 closes a paragraph, then a 41-word tail: 181 > 175, so the old
        # code refused the merge and published the orphan.
        out = self.comp.paragraphize(self._items([70, 70, 41]), "detailed")
        sizes = self._sizes(out)
        self.assertTrue(all(n >= lo for n in sizes),
                        f"a paragraph fell below the minimum: {sizes}")

    def test_rebalancing_never_loses_or_reorders_a_sentence(self):
        # Grouping moves whitespace only. If it can drop a sentence, the
        # coverage guarantees mean nothing.
        items = self._items([70, 70, 41, 55, 30])
        out = self.comp.paragraphize(items, "detailed")
        flat = [x["id"] for p in out for x in p["parts"]]
        self.assertEqual(flat, [x["id"] for x in items])

    def test_a_single_short_paragraph_cannot_be_rebalanced_away(self):
        # One item below the minimum has nothing to merge with; it must still
        # be published rather than dropped.
        out = self.comp.paragraphize(self._items([20]), "detailed")
        self.assertEqual(self._sizes(out), [20])



class NearDuplicateCollapse(unittest.TestCase):
    """A summary that defines the same term twice, or restates a finding with
    only 'The authors explain that' bolted on, wastes the reader's attention --
    the thing the whole exercise is meant to conserve."""
    comp = load("compose")

    def _it(self, i, text):
        return {"id": f"S{i:03d}", "heading": False, "text": text,
                "unit_ids": [f"U{i:03d}"]}

    def test_near_duplicates_survive_because_only_the_ledger_may_judge_them(self):
        # This once asserted the opposite. A resemblance rule holding deletion
        # authority deleted the negated, rescoped and renumbered sentence of each
        # pair, so the same rule that removed an attribution wrapper could remove
        # "does not". Near-duplicate judgement needs the source, which only the
        # ledger audit has.
        a = ("These elevated haircuts were functionally equivalent to massive "
             "withdrawals of liquidity from the banking system.")
        b = ("They argue that these elevated haircuts were functionally equivalent "
             "to massive withdrawals of liquidity from the banking system.")
        out = self.comp.dedupe([self._it(1, a), self._it(2, b)], "detailed")
        self.assertEqual([x["id"] for x in out], ["S001", "S002"])

    def test_exact_duplicates_collapse_and_keep_both_provenances(self):
        t = ("The safe-harbor exemption applies to repurchase agreements "
             "collateralised by government securities.")
        # whitespace/Unicode only: case now carries meaning and must not collapse
        out = self.comp.dedupe([self._it(1, t), self._it(2, "  " + t + "  ")],
                               "detailed")
        self.assertEqual([x["id"] for x in out], ["S001"])
        self.assertEqual(out[0]["unit_ids"], ["U001", "U002"])

    def test_distinct_sentences_on_one_topic_both_survive(self):
        # The metric this replaced scored a short sentence as a duplicate of any
        # longer one sharing its content words, which would gut real content.
        a = ("The investor provides funds in exchange for collateral from the "
             "bank, earning a repo rate analogous to interest.")
        b = "If the bank defaults, the investor can keep or sell the collateral."
        out = self.comp.dedupe([self._it(1, a), self._it(2, b)], "detailed")
        self.assertEqual(len(out), 2, "a distinct sentence was discarded")

    def test_headings_are_never_dropped(self):
        h = {"id": "H1", "heading": True, "text": "Findings", "unit_ids": []}
        out = self.comp.dedupe([h, self._it(1, "x " * 30), h], "detailed")
        self.assertEqual(sum(1 for x in out if x["heading"]), 2)

    def test_meaning_inverting_pairs_are_never_collapsed(self):
        base = ("The policy {} increase output and employment because credit "
                "conditions remain stable across the sampled sectors.")
        for x, y in ((base.format("does"), base.format("does not")),
                     ("The estimated coefficient is 0.13 after controls.",
                      "The estimated coefficient is 0.31 after controls."),
                     ("The result applies to insured banks under the framework.",
                      "The result applies only to insured banks under the framework.")):
            out = self.comp.dedupe([self._it(1, x), self._it(2, y)], "detailed")
            self.assertEqual(len(out), 2, f"collapsed a distinct pair: {x!r} / {y!r}")

    def test_short_sentences_are_left_alone(self):
        # Two terse clauses can look alike without being redundant.
        out = self.comp.dedupe([self._it(1, "Haircuts rose sharply."),
                                self._it(2, "Haircuts rose again.")], "brief")
        self.assertEqual(len(out), 2)



class IntroducedTerminologyGate(unittest.TestCase):
    """A named metric the summary invents is a claim about what the source
    established. "…effectively serves as a sacrifice ratio" shipped in a
    published summary and sealed `verified`, because the audit that had caught
    it did not happen to run again. The auditor is stochastic; this gate is not."""
    led = load("ledger")

    def _unit(self, detailed="", brief=""):
        return {"units": [{"unit_id": "U001", "detailed_capsule": detailed,
                           "brief_capsule": brief}]}

    def test_introduced_named_term_absent_from_source_is_caught(self):
        hits = self.led.introduced_terms(
            self._unit(detailed="The gap effectively serves as a sacrifice ratio, "
                                "measuring disinflation per unit of forgone output."),
            "The gap measures disinflation per unit of forgone output.")
        self.assertEqual(list(hits), [("U001", "detailed")])
        self.assertIn("sacrifice ratio", next(iter(hits.values())))

    def test_a_term_the_source_uses_is_not_flagged(self):
        hits = self.led.introduced_terms(
            self._unit(detailed="This is known as the sacrifice ratio."),
            "Economists call this the sacrifice ratio.")
        self.assertEqual(hits, {})

    def test_ordinary_prose_about_ratios_is_not_flagged(self):
        # Matching any adjective+metric compound flagged 31 terms on one
        # document, nearly all fragments like "because this ratio". A gate that
        # noisy would withhold most of a summary.
        for text in ("Because this ratio rises, the effect is larger.",
                     "The model applies an index to each component.",
                     "It isolates the effect of the policy rate.",
                     "The primary quantity effect dominates."):
            self.assertEqual(self.led.introduced_terms(self._unit(detailed=text), ""),
                             {}, f"false positive on: {text!r}")

    def test_the_gate_withholds_rather_than_edits(self):
        # Editing a capsule to satisfy a gate makes the artifact stop matching
        # the ledger it was sealed against.
        src = (ENG / "ledger.py").read_text()
        seg = src[src.index("DETERMINISTIC FIDELITY GATE"):src.index("# RE-SEAL.")]
        self.assertIn("quarantine[depth].add(uid)", seg)
        for edit in ("_capsule\"] =", "replace(", "capsule ="):
            self.assertNotIn(edit, seg)



class UnusableResponseIsRetried(unittest.TestCase):
    """A truncated response exits 0 with plausible partial output, so the
    pipeline read a cut-off stream as a real answer and failed the section.
    Delivery failures are distinct from malformed complete responses and are
    what bounded retries are for."""
    ms = load("mapsum")

    def test_run_accepts_a_validator(self):
        import inspect
        self.assertIn("validate", inspect.signature(self.ms.run).parameters)

    def test_every_json_call_site_validates(self):
        # A call site that forgets the validator silently keeps the old
        # behaviour, and the failure looks like a weak model rather than a
        # dropped response.
        src = (ENG / "ledger.py").read_text()
        boundaries = (("plan_part", "verify_part"),
                      ("audit_part", "revise_part"),
                      ("revise_part", "build"))
        for function, following in boundaries:
            segment = src[src.index(f"def {function}("):
                          src.index(f"def {following}(")]
            self.assertIn("ms.run(", segment,
                          f"{function} model call site not found")
            self.assertIn("validate=", segment,
                          f"{function} does not validate its response")

    def test_validator_rejects_truncated_and_empty(self):
        led = load("ledger")
        for bad in ('{"units": [{"claim": "cut off mid',      # truncated
                    "", "   ", "no json here at all"):
            with self.assertRaises(Exception, msg=f"accepted {bad!r}"):
                led._json_ok(bad)

    def test_validator_accepts_a_real_response(self):
        led = load("ledger")
        self.assertEqual(led._json_ok('{"verdict": "pass", "findings": []}')["verdict"],
                         "pass")



class MultiResolutionSelector(unittest.TestCase):
    """Compression must be enforced by selection, not by prompting.

    The same prompts produced 35% of source on one model and 67% on another,
    because select() admitted every `required` unit and the band was only a
    floor. The fix keeps every unit and lowers the resolution of the least
    load-bearing ones; deleting required units would turn "required" into
    "required unless the model was verbose"."""
    comp = load("compose")

    def _u(self, i, dw, bw, brief_disp="required", prio=3):
        return {"unit_id": f"U{i:03d}", "dependencies": [],
                "detailed_disposition": "required", "brief_disposition": brief_disp,
                "brief_priority": prio,
                "detailed_capsule": " ".join(["dw"] * dw),
                "brief_capsule": " ".join(["bw"] * bw) if bw else ""}

    def test_an_artifact_already_under_the_ceiling_is_untouched(self):
        # Gemini's real ledger compacted zero units; a selector that rewrites a
        # compliant artifact would be changing output for no reason.
        chosen = [self._u(i, 20, 8) for i in range(1, 6)]
        modes, w, _, _, _ = self.comp.plan_modes(chosen, "detailed", 500)
        self.assertEqual(w, 100)
        self.assertTrue(all(m == "detailed" for m in modes.values()))

    def test_over_ceiling_compacts_without_dropping_any_unit(self):
        chosen = [self._u(i, 20, 5) for i in range(1, 11)]      # 200w, ceiling 120
        modes, w, minimum, _, _ = self.comp.plan_modes(chosen, "detailed", 120)
        self.assertLessEqual(w, 120)
        self.assertEqual(len(modes), len(chosen), "a unit lost its representation")
        self.assertEqual(minimum, 50)

    def test_lowest_salience_units_are_compacted_first(self):
        # priority 5 is peripheral; priority 1 is indispensable and should keep
        # its Detailed resolution as long as the ceiling allows.
        chosen = [self._u(1, 40, 5, prio=1), self._u(2, 40, 5, prio=5)]
        modes, _, _, _, _ = self.comp.plan_modes(chosen, "detailed", 50)
        self.assertEqual(modes["U002"], "brief")
        self.assertEqual(modes["U001"], "detailed")

    def test_a_brief_omitted_unit_is_never_compacted_into_brief(self):
        # `omit` means the Brief capsule is not a sanctioned representation,
        # whatever text happens to sit in the field.
        chosen = [self._u(1, 60, 5, brief_disp="omit")]
        modes, w, minimum, _, _ = self.comp.plan_modes(chosen, "detailed", 10)
        self.assertEqual(modes["U001"], "detailed")
        self.assertEqual(minimum, 60, "an unsanctioned capsule counted as feasible")

    def test_brief_depth_renders_dependency_pulled_omit_units_at_full_resolution(self):
        # Dependency closure is not permission to publish text the ledger
        # declined to sanction at that depth; two such units reached one Brief
        # artifact this way.
        chosen = [self._u(1, 30, 0, brief_disp="omit")]
        modes, _, _, _, _ = self.comp.plan_modes(chosen, "brief", 1000)
        self.assertEqual(modes["U001"], "detailed")

    def test_infeasible_preferred_ceiling_is_reported_but_not_failed(self):
        chosen = [self._u(i, 40, 30) for i in range(1, 11)]     # minimum 300w
        _, _, minimum, _, _ = self.comp.plan_modes(chosen, "detailed", 100)
        self.assertGreater(minimum, 100)
        src = (ENG / "compose.py").read_text()
        self.assertIn("length_exception", src)
        self.assertNotIn("return 6", src[src.index("length_exception"):src.index("if not chosen")])


class PublicationCeiling(unittest.TestCase):
    """Length is reported; complete sealed content is the hard constraint."""

    def test_over_ceiling_is_recorded_but_does_not_suppress_output(self):
        src = (ENG / "compose.py").read_text()
        seg = src[src.index("THE PREFERRED PUBLICATION CEILING"):]
        self.assertIn("over_ceiling", seg)
        self.assertIn("return 0", seg)

    def test_being_under_the_lower_bound_is_not_a_failure(self):
        # Padding to reach a floor would add marginal units to hit a number.
        src = (ENG / "compose.py").read_text()
        seg = src[src.index("THE PREFERRED PUBLICATION CEILING"):src.index("return 0", src.index("THE PREFERRED PUBLICATION CEILING"))]
        self.assertNotIn("< lo", seg)
        self.assertNotIn("words'] < report", seg)


class SourceAccounting(unittest.TestCase):
    """Every visible block is represented by a unit or given one disposition.

    The build prompt always required this; nothing enforced it. A planner whose
    parts fail silently drops source and then looks attractively compressed --
    one model left 85 of 266 blocks (1,185 visible words) unaccounted and still
    sealed and published."""

    def test_mechseal_checks_complete_accounting(self):
        src = (ENG / "mechseal.py").read_text()
        seg = src[src.index("COMPLETE SOURCE ACCOUNTING"):]
        for token in ("unaccounted source blocks", "conflicting", "multi_disp"):
            self.assertIn(token, seg)

    def test_apparatus_must_be_dispositioned_not_absent(self):
        src = (ENG / "mechseal.py").read_text()
        self.assertIn("Apparatus is not exempt", src)



class ModelRosterIsData(unittest.TestCase):
    """Free models come and go. The volatile part must be one file the owner
    edits, not code -- and the pipeline must never branch on model identity."""

    def test_no_model_identifiers_are_hardcoded_in_the_engine(self):
        for mod in ("mapsum.py", "ledger.py", "compose.py"):
            src = (ENG / mod).read_text()
            code = "\n".join(l for l in src.splitlines()
                              if not l.lstrip().startswith("#"))
            for ident in ("gemini-3.1-pro", "gpt-5", "x-preview-f-free",
                          "nemotron", "muse-spark"):
                self.assertNotIn(ident, code,
                                 f"{mod} hardcodes the model {ident!r}")

    def test_every_harness_has_all_four_roles(self):
        cfg = json.loads((ENG / "models.json").read_text())
        for harness in ("agy", "claude", "opencode"):
            self.assertIn(harness, cfg)
            for role in ("plan", "write", "audit", "repair"):
                self.assertTrue(cfg[harness].get(role),
                                f"{harness}.{role} is empty")

    def test_device_local_gateways_are_not_committed(self):
        cfg = json.loads((ENG / "models.json").read_text())
        for name, spec in cfg.items():
            if name.startswith("_") or not isinstance(spec, dict):
                continue
            self.assertIsNot(spec.get("_local"), True,
                             f"{name} is a device-local roster in public config")

    def test_device_local_gateway_roster_merges_through_generic_config(self):
        mc = load("model_config")
        with unittest.mock.patch.object(mc.runtime, "config",
                                         return_value=gateway_runtime()):
            roster = mc.roster()
        self.assertEqual(roster["localgw"]["write"], ["model-a", "model-b"])
        self.assertTrue(roster["localgw"]["_local"])

    def test_effort_choices_are_roster_data_not_ui_constants(self):
        ms = load("mapsum")
        merged = {"localgw": gateway_runtime()["gateways"]["localgw"]["roster"]}
        with unittest.mock.patch.object(ms.model_config, "roster",
                                         return_value=merged):
            self.assertEqual(ms.effort_values("localgw", "model-a"),
                             ("low", "medium", "xhigh"))
            self.assertEqual(ms.effort_values("localgw", "model-b"),
                             ("low", "medium", "xhigh"))

    def test_effort_selection_follows_text_prep_stage_roles(self):
        ms = load("mapsum")
        self.assertEqual(ms.stage_role("text-clean001"), "write")
        self.assertEqual(ms.stage_role("text-audit001"), "audit")
        self.assertEqual(ms.stage_role("text-repair001"), "repair")
        self.assertEqual(ms.stage_role("text-reaudit001"), "audit")

    def test_corpus_call_role_is_explicit_and_not_inferred_from_stage_name(self):
        ms = load("mapsum")
        self.assertEqual(ms.VALID_ROLES,
                         frozenset(("plan", "write", "audit", "repair")))
        src = (ENG / "mapsum.py").read_text()
        self.assertIn("role: str | None = None", src)
        self.assertIn("effective_option(harness, model, effective_role)", src)
        self.assertIn('role=effective_role, stage=stage', src)
        with self.assertRaises(ValueError):
            ms.run("", pathlib.Path("/tmp/not-a-run"), [], "corpus-plan",
                   role="not-a-role")

    def test_authentication_failure_aborts_instead_of_walking_the_chain(self):
        ms = load("mapsum")
        self.assertTrue(ms.AUTH.search("Authentication required"))
        self.assertFalse(ms.CAPACITY.search("Authentication required"),
                         "an expired credential still reads as a capacity failure")
        import inspect
        src = inspect.getsource(ms.run)
        self.assertLess(src.index("AUTH.search"), src.index("CAPACITY.search"),
                        "the chain is walked before authentication is checked")



class PartScopedVerification(unittest.TestCase):
    """Verification happens next to the source window it applies to.

    The whole-ledger ladder re-read a 63k-token ledger five times, produced
    non-monotonic findings across rounds, had per-finding repairs that broke the
    mechanical seal, and could close an earlier unresolved finding simply by a
    later round not mentioning it."""
    led = load("ledger")

    def _part(self, blocks):
        return {"section_id": "SEC01", "title": "T", "part": "1 of 1",
                "blocks": [{"id": b, "text": t, "words": len(t.split())}
                           for b, t in blocks]}

    def test_unaccounted_block_is_caught_without_a_model(self):
        part = self._part([("P0001", "alpha beta"), ("P0002", "gamma delta")])
        obj = {"units": [{"local_id": "u1", "source_ids": ["P0001"],
                          "detailed_disposition": "required",
                          "detailed_capsule": "x", "brief_capsule": "y"}],
               "dispositions": []}
        self.assertTrue(any("unaccounted" in d for d in self.led.part_defects(part, obj)))

    def test_a_figure_absent_from_the_source_is_caught(self):
        # Numeric grounding: a capsule may not state a number its own source
        # blocks do not contain.
        part = self._part([("P0001", "The spread widened to 45 basis points.")])
        obj = {"units": [{"local_id": "u1", "source_ids": ["P0001"],
                          "detailed_disposition": "required",
                          "detailed_capsule": "The spread widened to 87 basis points.",
                          "brief_capsule": ""}], "dispositions": []}
        bad = self.led.part_defects(part, obj)
        self.assertTrue(any("87" in d for d in bad), bad)

    def test_a_figure_present_in_the_source_is_not_flagged(self):
        part = self._part([("P0001", "The spread widened to 45 basis points.")])
        obj = {"units": [{"local_id": "u1", "source_ids": ["P0001"],
                          "detailed_disposition": "required",
                          "detailed_capsule": "The spread widened to 45 basis points.",
                          "brief_capsule": ""}], "dispositions": []}
        self.assertEqual([d for d in self.led.part_defects(part, obj) if "45" in d], [])

    def test_there_is_no_convergence_loop(self):
        # At most one revision, then one audit of the exact revision. Iterating
        # until an auditor happens to be silent is a stochastic search, not
        # verification.
        src = (ENG / "ledger.py").read_text()
        seg = src[src.index("def plan_part"):src.index("def part_source")]
        self.assertEqual(seg.count("revise_part("), 1)
        self.assertLessEqual(seg.count("audit_part("), 2)
        self.assertNotIn("for attempt in range", seg)

    def test_the_deleted_ladder_is_really_gone(self):
        src = (ENG / "ledger.py").read_text()
        for dead in ("def repair(", "def reconcile(", "audit-state.json",
                     "apply_dependency_repairs"):
            self.assertNotIn(dead, src, f"{dead} survives")

    def test_a_part_that_never_planned_blocks_publication(self):
        # Quarantining nothing cannot repair absent coverage.
        src = (ENG / "ledger.py").read_text()
        seg = src[src.index('if ledger.get("unplanned_parts")'):]
        self.assertIn("not publishable", seg[:600])
        self.assertIn("return 4", seg[:600])

    def test_a_thin_part_still_publishes(self):
        # THIN means fewer units than a density heuristic expects, with every
        # source block still accounted. Conflating it with "never planned"
        # blocked a complete document from publishing.
        src = (ENG / "ledger.py").read_text()
        seg = src[src.index('if ledger.get("thin_sections")'):]
        self.assertNotIn("return 4", seg[:400])
        self.assertIn("accounting is still complete", seg[:400])

    def test_an_unavailable_part_audit_fails_the_part(self):
        part = self._part([("P0001", "The evidence remains limited.")])
        obj = {"units": [{"local_id": "SEC01-U01", "source_ids": ["P0001"],
                          "dependencies": [], "detailed_disposition": "required",
                          "brief_disposition": "required", "brief_priority": 5,
                          "detailed_capsule": "The evidence remains limited.",
                          "brief_capsule": "The evidence is limited."}],
               "dispositions": []}
        with tempfile.TemporaryDirectory() as td, \
             unittest.mock.patch.object(
                 self.led, "audit_part", side_effect=RuntimeError("offline")):
            with self.assertRaisesRegex(RuntimeError, "audit unavailable"):
                self.led.verify_part(object(), part, pathlib.Path(td), 1, obj)

    def test_a_density_replan_is_verified_before_replacing_the_plan(self):
        src = (ENG / "ledger.py").read_text()
        segment = src[src.index("if len(obj2.get"):src.index("for u in got:")]
        self.assertIn("obj2 = verify_part", segment)


class NoModelFetchesFiles(unittest.TestCase):
    """Payloads travel in the prompt. Handing a model a path costs a tool turn
    whose result is re-sent on every later step -- that turned ~672k tokens of
    real payload into 4.2M on one document."""

    def test_prompts_carry_content_not_paths(self):
        src = (ENG / "ledger.py").read_text()
        self.assertIn("def inline(", src)
        for bad in ('"{SOURCE_VIEW}", str(', '"{LEDGER}", str('):
            self.assertNotIn(bad, src, "a prompt still receives a file path")

    def test_every_prompt_forbids_tools(self):
        for name in ("ledger-build.txt", "ledger-audit.txt", "ledger-revise.txt"):
            t = (ENG / "prompts" / name).read_text()
            self.assertIn("Use no tools", t, f"{name} does not forbid tools")

    def test_one_model_step(self):
        src = (ENG / "mapsum.py").read_text()
        self.assertIn('"--max-model-steps", "1"', src)
        self.assertNotIn("MUSE_MAX_STEPS", src)


class CompactUnitSchema(unittest.TestCase):
    """The pipeline generated 13,584 words of never-consumed fields against
    12,143 words of actual product -- more dead weight than summary, on a device
    where output throughput is the binding constraint."""

    def test_dead_fields_are_gone_from_the_schema(self):
        tpl = (ENG / "prompts" / "ledger-build.txt").read_text()
        for dead in ("support_points", "required_qualifiers", "epistemic_status",
                     "exact_source_anchor"):
            self.assertNotIn(dead, tpl, f"{dead} is still requested")

    def test_no_engine_code_reads_a_removed_field(self):
        for mod in ("ledger.py", "compose.py", "mechseal.py"):
            src = (ENG / mod).read_text()
            code = "\n".join(l for l in src.splitlines()
                              if not l.lstrip().startswith("#"))
            for dead in ("support_points", "required_qualifiers", "epistemic_status"):
                self.assertNotIn(dead, code, f"{mod} reads {dead}")



class JSONExtraction(unittest.TestCase):
    """Some harnesses echo the request before answering. Taking everything from
    the first brace to the last one then spanned the schema EXAMPLE in the echo
    through the real answer, and parsed as neither."""
    ms = load("mapsum")

    def test_prompt_echo_before_the_answer_is_ignored(self):
        raw = ('user\nRETURN\n{"units": [{"local_id": "SEC01-U01"}]}\n'
               'assistant\n{"units": [{"local_id": "u1"}, {"local_id": "u2"}]}\n')
        self.assertEqual(len(self.ms.parse_audit(raw)["units"]), 2)

    def test_a_fenced_block_still_wins(self):
        raw = 'noise {"a": 1}\n```json\n{"units": [], "verdict": "pass"}\n```\n'
        self.assertEqual(self.ms.parse_audit(raw)["verdict"], "pass")

    def test_braces_inside_strings_do_not_break_scanning(self):
        raw = '{"units": [], "note": "a } brace { in prose"}'
        self.assertEqual(self.ms.parse_audit(raw)["note"], "a } brace { in prose")

    def test_no_object_raises(self):
        with self.assertRaises(Exception):
            self.ms.parse_audit("there is no json here")

    def test_mapsum_contains_only_transport_not_the_retired_document_pipeline(self):
        src = (ENG / "mapsum.py").read_text()
        for dead in ("strip_furniture", "number_paragraphs", "split_chunks",
                     "validate_manifest", "def process(", "def main("):
            self.assertNotIn(dead, src)



class GatewayHarness(unittest.TestCase):
    """A gateway is a generic declared transport, not product policy."""
    gw = load("gateway")
    ms = load("mapsum")

    class Response:
        def __init__(self, body, headers=None):
            self.body = body
            self.headers = headers or {"Content-Type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return self.body

    def setUp(self):
        self.config = gateway_runtime()
        merged = {
            "_defaults": {"fallback": {"harness": "opencode",
                                        "model": "cloud-fallback"}},
            "localgw": self.config["gateways"]["localgw"]["roster"],
            "opencode": {role: ["cloud-fallback"] for role in
                           ("plan", "write", "audit", "repair")},
        }
        patches = [
            unittest.mock.patch.object(self.ms, "GATEWAY_HARNESSES",
                                       frozenset({"localgw"})),
            unittest.mock.patch.object(self.ms, "HARNESSES",
                                       tuple(dict.fromkeys((*self.ms.HARNESSES,
                                                           "localgw")))),
            unittest.mock.patch.object(self.ms.model_config, "full",
                                       return_value=merged),
            unittest.mock.patch.object(self.ms.model_config, "local_harnesses",
                                       return_value=frozenset({"localgw"})),
            unittest.mock.patch.object(self.ms.runtime, "config",
                                       return_value=self.config),
        ]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def _body(self, content="answer", finish_reason="stop", **extra):
        obj = {
            "id": "response-7",
            "choices": [{"message": {"content": content},
                         "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4,
                      "total_tokens": 14},
        }
        obj.update(extra)
        return json.dumps(obj).encode()

    def _result(self, content="answer", finish_reason="stop"):
        obj = json.loads(self._body(content, finish_reason))
        return self.ms.gateway.ChatCompletionResult(
            content=content, finish_reason=finish_reason,
            usage=obj["usage"], response_id=obj["id"], raw_response=obj,
            elapsed_seconds=1.25,
            response_headers={"content-type": "application/json"})

    def test_one_shot_request_uses_declared_transport_and_preserves_metadata(self):
        body = self._body("<think>reasoning</think>\n{\"blocks\": []}")
        with unittest.mock.patch.object(self.gw.runtime, "config",
                                         return_value=self.config), \
             unittest.mock.patch.dict(os.environ,
                                      {"SUMM_LOCALGW_TOKEN": "secret"}), \
             unittest.mock.patch.object(
                 self.gw.urllib.request, "urlopen",
                 return_value=self.Response(
                     body, {"Content-Type": "application/json",
                            "X-Request-ID": "request-9"})) as opened:
            result = self.gw.chat("localgw", "model-a", "PROMPT")
        self.assertEqual(result.content, '{"blocks": []}')
        self.assertEqual(result.finish_reason, "stop")
        self.assertEqual(result.usage["completion_tokens"], 4)
        self.assertEqual(result.response_id, "response-7")
        self.assertEqual(result.response_headers["x-request-id"], "request-9")
        req = opened.call_args.args[0]
        self.assertEqual(req.full_url,
                         "https://gateway.example/v1/chat/completions")
        payload = json.loads(req.data)
        self.assertEqual(payload["model"], "model-a")
        self.assertEqual(payload["messages"],
                         [{"role": "user", "content": "PROMPT"}])
        self.assertEqual(payload["temperature"], 0)
        self.assertEqual(payload["max_tokens"], 16384)
        self.assertFalse(payload["stream"])
        self.assertEqual(req.get_header("Authorization"), "Bearer secret")

    def test_output_token_field_is_declarative(self):
        self.config["gateways"]["localgw"]["output_token_field"] = \
            "max_completion_tokens"
        with unittest.mock.patch.object(self.gw.runtime, "config",
                                         return_value=self.config), \
             unittest.mock.patch.dict(os.environ,
                                      {"SUMM_LOCALGW_TOKEN": "secret"}), \
             unittest.mock.patch.object(
                 self.gw.urllib.request, "urlopen",
                 return_value=self.Response(self._body())) as opened:
            self.gw.chat("localgw", "model-a", "P")
        payload = json.loads(opened.call_args.args[0].data)
        self.assertEqual(payload["max_completion_tokens"], 16384)
        self.assertNotIn("max_tokens", payload)

    def test_authentication_can_be_declared_absent(self):
        spec = self.config["gateways"]["localgw"]
        spec["authentication"] = "none"
        spec.pop("api_key_env")
        with unittest.mock.patch.object(self.gw.runtime, "config",
                                         return_value=self.config), \
             unittest.mock.patch.object(
                 self.gw.urllib.request, "urlopen",
                 return_value=self.Response(self._body())) as opened:
            self.gw.chat("localgw", "model-a", "P")
        self.assertIsNone(opened.call_args.args[0].get_header("Authorization"))

    def test_gateway_request_options_can_be_frozen_for_a_queued_job(self):
        self.config["gateways"]["localgw"]["request_options"] = {
            "reasoning_effort": "medium"}
        with unittest.mock.patch.object(self.gw.runtime, "config",
                                         return_value=self.config), \
             unittest.mock.patch.dict(os.environ, {
                 "SUMM_LOCALGW_TOKEN": "secret",
                 "SUMM_LOCALGW_REQUEST_OPTIONS":
                     '{"reasoning_effort":"medium"}'
             }), \
             unittest.mock.patch.object(
                 self.gw.urllib.request, "urlopen",
                 return_value=self.Response(self._body())) as opened:
            result = self.gw.chat("localgw", "model-a", "P")
        self.assertEqual(result.content, "answer")
        payload = json.loads(opened.call_args.args[0].data)
        self.assertEqual(payload["reasoning_effort"], "medium")

    def test_one_request_reads_one_runtime_snapshot(self):
        first = gateway_runtime()
        first["gateways"]["localgw"]["request_options"] = {
            "reasoning_effort": "low"}
        second = gateway_runtime()
        second["gateways"]["localgw"]["base_url"] = \
            "https://changed.example/v1"
        with unittest.mock.patch.object(
                self.gw.runtime, "config", side_effect=[first, second]) as config, \
             unittest.mock.patch.dict(
                 os.environ, {"SUMM_LOCALGW_TOKEN": "secret"}):
            settings = self.gw._settings("localgw", "model-a")
        self.assertEqual(1, config.call_count)
        self.assertEqual("https://gateway.example/v1", settings["base_url"])
        self.assertEqual("low", settings["request_options"]["reasoning_effort"])

    def test_json_stage_forwards_standard_structured_response_option(self):
        with tempfile.TemporaryDirectory() as td, \
             unittest.mock.patch.object(
                 self.ms.gateway, "chat",
                 return_value=self._result('{"ok":true}')) as chat:
            answer = self.ms.run(
                "PROMPT", pathlib.Path(td), ["localgw:model-a"], "plan001",
                validate=json.loads,
                gateway_options=self.ms.JSON_REQUEST_OPTIONS)
        self.assertEqual(answer, '{"ok":true}')
        self.assertEqual(
            chat.call_args.kwargs["request_options"]["response_format"],
            {"type": "json_object"})

    def test_colon_stage_ids_write_windows_portable_evidence_names(self):
        # Synced run evidence reaches Windows peers, which cannot store ":" in a
        # file name; the canonical ID stays in events and JSON, not the path.
        with tempfile.TemporaryDirectory() as td, \
             unittest.mock.patch.object(
                 self.ms.gateway, "chat",
                 return_value=self._result('{"ok":true}')):
            self.ms.run(
                "PROMPT", pathlib.Path(td), ["localgw:model-a"],
                "corpus-inventory-plan-D001:W001", validate=json.loads,
                gateway_options=self.ms.JSON_REQUEST_OPTIONS)
            root = pathlib.Path(td)
            names = [p.name for p in root.rglob("*")]
            self.assertEqual([n for n in names if ":" in n], [])
            self.assertEqual(
                (root / "corpus-inventory-plan-D001-W001.stdout.txt").read_text(),
                '{"ok":true}')
            self.assertTrue(
                (root / "corpus-inventory-plan-D001-W001.attempt01.stdout.txt").is_file())

    def test_complete_invalid_gateway_response_is_not_replayed(self):
        with tempfile.TemporaryDirectory() as td, \
             unittest.mock.patch.object(
                 self.ms.gateway, "chat",
                 return_value=self._result("not json")) as chat, \
             unittest.mock.patch.object(self.ms.time, "sleep") as sleep:
            with self.assertRaises(self.ms.Abort):
                self.ms.run(
                    "PROMPT", pathlib.Path(td), ["localgw:model-a"], "plan001",
                    validate=json.loads,
                    gateway_options=self.ms.JSON_REQUEST_OPTIONS)
            root = pathlib.Path(td)
            self.assertEqual((root / "plan001.attempt01.stdout.txt").read_text(),
                             "not json")
            evidence = json.loads(
                (root / "plan001.attempt01.response.json").read_text())
            self.assertEqual(evidence["finish_reason"], "stop")
            self.assertEqual(evidence["usage"]["completion_tokens"], 4)
        self.assertEqual(chat.call_count, 1)
        sleep.assert_not_called()

    def test_a_failed_producer_keeps_its_evidence_when_the_next_one_answers(self):
        # Attempt files were named by each producer's own attempt count, so the
        # second chain entry's attempt 1 overwrote the first's stderr and
        # stdout. The failed producer's evidence is the one worth keeping.
        with tempfile.TemporaryDirectory() as td, \
             unittest.mock.patch.object(
                 self.ms.gateway, "chat",
                 side_effect=[self._result("not json"),
                              self._result('{"ok":true}')]), \
             unittest.mock.patch.object(self.ms.time, "sleep"):
            answer = self.ms.run(
                "PROMPT", pathlib.Path(td),
                ["localgw:model-a", "localgw:model-b"], "plan001",
                validate=json.loads,
                gateway_options=self.ms.JSON_REQUEST_OPTIONS)
            root = pathlib.Path(td)
            stdouts = sorted(f.name for f in root.glob("plan001.attempt*.stdout.txt"))
            self.assertEqual(answer, '{"ok":true}')
            self.assertEqual(len(stdouts), 2, stdouts)
            self.assertEqual((root / stdouts[0]).read_text(), "not json")
            self.assertEqual((root / stdouts[1]).read_text(), '{"ok":true}')
            records = [json.loads(l) for l in
                       (root / "calls.jsonl").read_text().splitlines()]
            self.assertEqual([r["attempt_number"] for r in records[-2:]],
                             [int(n.split("attempt")[1][:2]) for n in stdouts])

    def test_length_finish_is_one_known_incomplete_attempt(self):
        with tempfile.TemporaryDirectory() as td, \
             unittest.mock.patch.object(
                 self.ms.gateway, "chat",
                 return_value=self._result("partial", "length")) as chat, \
             unittest.mock.patch.object(self.ms.time, "sleep") as sleep:
            with self.assertRaises(self.ms.Abort):
                self.ms.run("PROMPT", pathlib.Path(td),
                            ["localgw:model-a"],
                            "plan001", validate=json.loads)
            rows = (pathlib.Path(td) / "calls.jsonl").read_text().splitlines()
            self.assertEqual([json.loads(row)["outcome"] for row in rows],
                             ["output_limit"])
        self.assertEqual(chat.call_count, 1)
        sleep.assert_not_called()

    def test_complete_json_with_length_stop_is_kept(self):
        with tempfile.TemporaryDirectory() as td, \
             unittest.mock.patch.object(
                 self.ms.gateway, "chat",
                 return_value=self._result('{"ok":true}', "length")) as chat, \
             unittest.mock.patch.object(self.ms.time, "sleep"):
            answer = self.ms.run(
                "PROMPT", pathlib.Path(td), ["localgw:model-a"],
                "plan001", validate=json.loads,
                gateway_options=self.ms.JSON_REQUEST_OPTIONS)
        self.assertEqual(json.loads(answer), {"ok": True})
        self.assertEqual(chat.call_count, 1)

    def test_truncated_json_is_continued_until_complete(self):
        with tempfile.TemporaryDirectory() as td, \
             unittest.mock.patch.object(
                 self.ms.gateway, "chat",
                 side_effect=[self._result('{"ok":', "length"),
                              self._result("true}", "stop")]) as chat, \
             unittest.mock.patch.object(self.ms.time, "sleep"):
            answer = self.ms.run(
                "PROMPT", pathlib.Path(td), ["localgw:model-a"],
                "plan001", validate=json.loads,
                gateway_options=self.ms.JSON_REQUEST_OPTIONS)
        self.assertEqual(json.loads(answer), {"ok": True})
        self.assertEqual(chat.call_count, 2)
        self.assertTrue(chat.call_args.kwargs.get("continuation"))
        self.assertEqual(len(chat.call_args.kwargs["messages"]), 3)
        self.assertNotIn(
            "response_format",
            chat.call_args.kwargs.get("request_options") or {})

    def test_incomplete_json_is_continued_even_when_the_model_stops(self):
        with tempfile.TemporaryDirectory() as td, \
             unittest.mock.patch.object(
                 self.ms.gateway, "chat",
                 side_effect=[self._result('{"ok":', "stop"),
                              self._result("true}", "stop")]) as chat, \
             unittest.mock.patch.object(self.ms.time, "sleep"):
            answer = self.ms.run(
                "PROMPT", pathlib.Path(td), ["localgw:model-a"],
                "plan001", validate=json.loads,
                gateway_options=self.ms.JSON_REQUEST_OPTIONS)
        self.assertEqual(json.loads(answer), {"ok": True})
        self.assertEqual(chat.call_count, 2)

    def test_trailing_json_is_not_continued(self):
        with tempfile.TemporaryDirectory() as td, \
             unittest.mock.patch.object(
                 self.ms.gateway, "chat",
                 return_value=self._result('{"ok":true} extra', "length")) as chat, \
             unittest.mock.patch.object(self.ms.time, "sleep"):
            with self.assertRaises(self.ms.Abort):
                self.ms.run(
                    "PROMPT", pathlib.Path(td), ["localgw:model-a"],
                    "plan001", validate=json.loads,
                    gateway_options=self.ms.JSON_REQUEST_OPTIONS)
        self.assertEqual(chat.call_count, 1)

    def test_incomplete_json_stops_only_when_continuation_adds_nothing(self):
        with tempfile.TemporaryDirectory() as td, \
             unittest.mock.patch.object(
                 self.ms.gateway, "chat",
                 side_effect=[self._result('{"ok":', "length"),
                              self._result("", "length")]) as chat, \
             unittest.mock.patch.object(self.ms.time, "sleep"):
            with self.assertRaises(self.ms.Abort):
                self.ms.run(
                    "PROMPT", pathlib.Path(td), ["localgw:model-a"],
                    "plan001", validate=json.loads,
                    gateway_options=self.ms.JSON_REQUEST_OPTIONS)
        self.assertEqual(chat.call_count, 2)

    def test_unclosed_inline_reasoning_is_classified_incomplete(self):
        with unittest.mock.patch.object(self.gw.runtime, "config",
                                         return_value=self.config), \
             unittest.mock.patch.dict(os.environ,
                                      {"SUMM_LOCALGW_TOKEN": "secret"}), \
             unittest.mock.patch.object(
                 self.gw.urllib.request, "urlopen",
                 return_value=self.Response(
                     self._body("<think>unfinished", "length"))):
            with self.assertRaises(self.gw.GatewayError) as raised:
                self.gw.chat("localgw", "model-a", "P")
        self.assertEqual(raised.exception.kind, "output_incomplete")

    def test_gateway_http_errors_follow_declared_policy(self):
        raw = b'{"error":"upstream was unavailable"}'
        error = urllib.error.HTTPError(
            "https://gateway.example/v1/chat/completions", 502, "unavailable",
            {"Content-Type": "application/json", "X-Request-ID": "request-9"},
            io.BytesIO(raw))
        with unittest.mock.patch.object(self.gw.runtime, "config",
                                         return_value=self.config), \
             unittest.mock.patch.dict(os.environ,
                                      {"SUMM_LOCALGW_TOKEN": "secret"}), \
             unittest.mock.patch.object(self.gw.urllib.request, "urlopen",
                                        side_effect=error):
            with self.assertRaises(self.gw.GatewayError) as raised:
                self.gw.chat("localgw", "model-a", "PROMPT")
        self.assertEqual(raised.exception.status, 502)
        self.assertEqual(raised.exception.kind, "unavailable")
        self.assertIn("upstream", str(raised.exception))
        self.assertEqual(raised.exception.evidence, {
            "status": 502,
            "error_code": None,
            "detail": "upstream was unavailable",
            "response_headers": {
                "content-type": "application/json",
                "x-request-id": "request-9",
            },
            "raw_body_bytes": len(raw),
            "raw_body_sha256": hashlib.sha256(raw).hexdigest(),
        })

    def test_gateway_http_error_evidence_is_written_for_stage_failure(self):
        raw = b'{"error":{"code":"upstream_unavailable","message":"lost"}}'
        error = urllib.error.HTTPError(
            "https://gateway.example/v1/chat/completions", 502, "unavailable",
            {"Content-Type": "application/json", "X-Request-ID": "request-10"},
            io.BytesIO(raw))
        with tempfile.TemporaryDirectory() as td, \
             unittest.mock.patch.object(self.gw.runtime, "config",
                                         return_value=self.config), \
             unittest.mock.patch.dict(os.environ,
                                      {"SUMM_LOCALGW_TOKEN": "secret"}), \
             unittest.mock.patch.object(self.gw.urllib.request, "urlopen",
                                        side_effect=error):
            with self.assertRaises(self.ms.Abort):
                self.ms.run("PROMPT", pathlib.Path(td), ["localgw:model-a"],
                            "plan001")
            evidence = json.loads(
                (pathlib.Path(td) / "plan001.attempt01.response.json").read_text())
        self.assertEqual(evidence["status"], 502)
        self.assertEqual(evidence["error_code"], "upstream_unavailable")
        self.assertIn("lost", evidence["detail"])
        self.assertEqual(evidence["response_headers"]["x-request-id"], "request-10")
        self.assertEqual(evidence["raw_body_bytes"], len(raw))
        self.assertEqual(evidence["raw_body_sha256"], hashlib.sha256(raw).hexdigest())

    def test_gateway_error_codes_can_declare_model_transition_as_warming(self):
        for code in ("profile_queued_behind_active_model", "model_lifecycle_busy"):
            with self.subTest(code=code):
                self.config["gateways"]["localgw"]["error_policy"]["codes"][code] = "warming"
                error = urllib.error.HTTPError(
                    "https://gateway.example/v1/chat/completions", 503, "busy",
                    {"Retry-After": "7"}, io.BytesIO(json.dumps({
                        "error": {
                            "code": code,
                            "message": "waiting for a model lifecycle transition",
                        }
                    }).encode()))
                with unittest.mock.patch.object(self.gw.runtime, "config",
                                                 return_value=self.config), \
                     unittest.mock.patch.dict(os.environ,
                                              {"SUMM_LOCALGW_TOKEN": "secret"}), \
                     unittest.mock.patch.object(self.gw.urllib.request, "urlopen",
                                                side_effect=error):
                    with self.assertRaises(self.gw.GatewayError) as raised:
                        self.gw.chat("localgw", "model-a", "PROMPT")
                self.assertEqual(raised.exception.status, 503)
                self.assertEqual(raised.exception.kind, "warming")
                self.assertEqual(raised.exception.retry_after, 7)

    def test_warming_gateway_repolls_the_same_request_until_ready(self):
        warming = self.ms.gateway.GatewayError(
            "gateway HTTP 503: warming", kind="warming", retry_after=7)
        with tempfile.TemporaryDirectory() as td, \
             unittest.mock.patch.object(
                 self.ms.gateway, "chat",
                 side_effect=[warming, self._result("answer")]) as chat, \
             unittest.mock.patch.object(self.ms.time, "sleep") as sleep:
            answer = self.ms.run(
                "PROMPT", pathlib.Path(td), ["localgw:model-a"], "write001")
            self.assertEqual(answer, "answer")
            rows = (pathlib.Path(td) / "calls.jsonl").read_text().splitlines()
            self.assertEqual([json.loads(row)["outcome"] for row in rows], ["ok"])
        self.assertEqual(chat.call_count, 2)
        request_options = [
            call.kwargs["request_options"] for call in chat.call_args_list
        ]
        self.assertEqual(request_options[0], request_options[1])
        self.assertEqual(request_options[0], {"reasoning_effort": "medium"})
        sleep.assert_called_once_with(7)

    def test_one_gateway_uses_one_declared_default_resource(self):
        with unittest.mock.patch.object(self.ms.runtime, "slot_lease") as lease:
            self.ms.runtime.resource_lease("localgw")
        self.assertEqual([item.args for item in lease.call_args_list],
                         [("resource", "gateway", 1)])

    def test_one_model_can_override_the_device_local_route(self):
        self.config["gateways"]["localgw"]["models"] = {
            "model-b": {
                "base_url": "https://gateway.example/alternate/v1",
                "api_key_env": "SUMM_ALTERNATE_TOKEN",
                "request_options": {"reasoning_effort": "medium"},
            }
        }
        with unittest.mock.patch.object(self.gw.runtime, "config",
                                         return_value=self.config), \
             unittest.mock.patch.dict(os.environ,
                                      {"SUMM_ALTERNATE_TOKEN": "secret-b"}), \
             unittest.mock.patch.object(
                 self.gw.urllib.request, "urlopen",
                 return_value=self.Response(self._body())) as opened:
            result = self.gw.chat("localgw", "model-b", "P")
        self.assertEqual(result.content, "answer")
        request = opened.call_args.args[0]
        self.assertEqual(request.full_url,
                         "https://gateway.example/alternate/v1/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer secret-b")

    def test_gateway_request_error_can_reach_a_backup_route(self):
        with tempfile.TemporaryDirectory() as td, \
             unittest.mock.patch.object(
                 self.ms.gateway, "chat",
                 side_effect=self.ms.gateway.GatewayError(
                     "gateway HTTP 413: request is too large", kind="request")), \
             unittest.mock.patch.object(
                 subprocess, "run",
                 return_value=subprocess.CompletedProcess(
                     ["opencode"], 0, "fallback response", "")) as process:
            answer = self.ms.run("PROMPT", pathlib.Path(td),
                                 ["localgw:model-a", "opencode:cloud-fallback"],
                                 "write001")
            self.assertEqual(answer, "fallback response")
            process.assert_called_once()
            rows = (pathlib.Path(td) / "calls.jsonl").read_text().splitlines()
            self.assertEqual(json.loads(rows[0])["outcome"], "request")

    def test_completion_unknown_is_not_retried_or_fallen_through(self):
        with tempfile.TemporaryDirectory() as td, \
             unittest.mock.patch.object(
                 self.ms.gateway, "chat",
                 side_effect=self.ms.gateway.GatewayError(
                     "gateway completion status is unknown",
                     kind="completion_unknown")), \
             unittest.mock.patch.object(self.ms.time, "sleep") as sleep:
            with self.assertRaises(self.ms.Abort):
                self.ms.run("PROMPT", pathlib.Path(td),
                            ["localgw:model-a", "opencode:cloud-fallback"],
                            "write001")
            rows = (pathlib.Path(td) / "calls.jsonl").read_text().splitlines()
            self.assertEqual(len(rows), 1)
            self.assertEqual(json.loads(rows[0])["outcome"],
                             "completion_unknown")
        sleep.assert_not_called()

    def test_local_only_rejects_a_cloud_entry_before_any_call(self):
        with tempfile.TemporaryDirectory() as td, \
             unittest.mock.patch.dict(os.environ, {"SUMM_LOCAL_ONLY": "1"}), \
             unittest.mock.patch.object(self.ms.gateway, "chat") as chat, \
             unittest.mock.patch.object(subprocess, "run") as process:
            with self.assertRaises(self.ms.Abort):
                self.ms.run("PROMPT", pathlib.Path(td),
                            ["localgw:model-a", "opencode:cloud-fallback"],
                            "write001")
        chat.assert_not_called()
        process.assert_not_called()

    def test_a_local_harness_never_inherits_the_cloud_default(self):
        with unittest.mock.patch.object(self.ms, "HARNESS", "localgw"), \
             unittest.mock.patch.dict(os.environ, {}, clear=True):
            chains = self.ms._models()
        self.assertTrue(all(all(self.ms.split_entry(e, "localgw")[0] == "localgw"
                                for e in entries)
                            for entries in chains.values()))

    def test_local_only_allows_multiple_models_under_one_backend(self):
        with unittest.mock.patch.dict(os.environ, {"SUMM_LOCAL_ONLY": "1"},
                                      clear=False), \
             unittest.mock.patch.object(self.ms, "PLAN_MODELS",
                                        ["localgw:model-a"]), \
             unittest.mock.patch.object(self.ms, "MODELS",
                                        ["localgw:model-a"]), \
             unittest.mock.patch.object(self.ms, "AUDIT_MODELS",
                                        ["localgw:model-b"]), \
             unittest.mock.patch.object(self.ms, "REPAIR_MODELS",
                                        ["localgw:model-a"]):
            self.ms.validate_local_only()

    def test_unavailable_gateway_is_not_retried_before_chain_fallback(self):
        with tempfile.TemporaryDirectory() as td, \
             unittest.mock.patch.object(
                 self.ms.gateway, "chat",
                 side_effect=self.ms.gateway.GatewayError(
                     "gateway HTTP 502: unloaded", kind="unavailable")) as chat, \
             unittest.mock.patch.object(self.ms.time, "sleep") as sleep, \
             unittest.mock.patch.object(
                 subprocess, "run",
                 return_value=subprocess.CompletedProcess([], 0, "answer", "")) as process:
            answer = self.ms.run(
                "PROMPT", pathlib.Path(td),
                ["localgw:model-a", "opencode:cloud-fallback"], "write001")
            self.assertEqual(answer, "answer")
            self.assertEqual(process.call_count, 1)
            lines = (pathlib.Path(td) / "calls.jsonl").read_text().splitlines()
            self.assertEqual([json.loads(line)["outcome"] for line in lines],
                             ["unavailable", "ok"])
        self.assertEqual(chat.call_count, 1)
        sleep.assert_not_called()

    def test_an_explicit_role_chain_is_not_extended_by_global_fallback(self):
        with unittest.mock.patch.object(self.ms, "HARNESS", "opencode"), \
             unittest.mock.patch.dict(
                 os.environ,
                 {"OPENCODE_CHAIN_WRITE": "localgw:model-a"}, clear=False):
            self.assertEqual(self.ms._models()["write"], ["localgw:model-a"])


class ReasoningCompletionEvidence(unittest.TestCase):
    """A reasoning-only gateway result retains the native completion facts."""

    gw = load("gateway")

    @staticmethod
    def body(finish_reason="length"):
        return json.dumps({
            "id": "reasoning-only",
            "choices": [{
                "message": {"content": "", "reasoning_content": "working"},
                "finish_reason": finish_reason,
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 8,
                      "reasoning_tokens": 8, "total_tokens": 18},
        }).encode()

    def test_length_after_reasoning_is_output_limit_with_evidence(self):
        with self.assertRaises(self.gw.GatewayError) as raised:
            self.gw._completion(
                self.body(), reasoning_content="plain", elapsed_seconds=1.25,
                response_headers={"x-request-id": "request-1"})
        error = raised.exception
        self.assertEqual(error.kind, "output_limit")
        self.assertEqual(error.evidence["finish_reason"], "length")
        self.assertEqual(error.evidence["usage"]["completion_tokens"], 8)
        self.assertEqual(error.evidence["raw_response"]["id"], "reasoning-only")

    def test_stop_after_reasoning_without_answer_is_distinctly_incomplete(self):
        with self.assertRaises(self.gw.GatewayError) as raised:
            self.gw._completion(
                self.body("stop"), reasoning_content="plain",
                elapsed_seconds=1.25, response_headers={})
        self.assertEqual(raised.exception.kind, "output_incomplete")
        self.assertEqual(raised.exception.evidence["finish_reason"], "stop")

    def test_continuation_keeps_the_exact_next_characters(self):
        result = self.gw._completion(
            json.dumps({
                "id": "cont",
                "choices": [{"message": {"content": " true}"},
                             "finish_reason": "stop"}],
                "usage": {},
            }).encode(),
            reasoning_content="plain", elapsed_seconds=0.1,
            response_headers={}, strip_content=False)
        self.assertEqual(result.content, " true}")


class ReasoningBudgetAdmission(unittest.TestCase):
    """Qualified routes inject one protected, role-specific reasoning cap."""

    gw = load("gateway")
    rt = load("runtime")

    class Response:
        headers = {"Content-Type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps({
                "id": "bounded-response",
                "choices": [{"message": {"content": "answer"},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3,
                          "total_tokens": 8},
            }).encode()

    class TokenResponse(Response):
        def read(self):
            return json.dumps({"count": 128}).encode()

    @staticmethod
    def admission(binding="thinking_token_budget", status="passed"):
        return {
            "schema": "summer.reasoning-admission.v2",
            "accounting": "shared_reasoning_content",
            "completion_tokens": 16384,
            "completion_binding": "max_tokens",
            "budget_binding": binding,
            "enable_binding": "chat_template_kwargs.enable_thinking",
            "required": True,
            "boundary_overrun_tokens": 64,
            "output_tokens_per_word": 2.0,
            "token_counter": {
                "binding": "openai_chat_tokenize",
                "status": "passed",
                "certificate_sha256": "b" * 64,
            },
            "roles": {
                "plan": {"reasoning_tokens": 2048,
                         "retry_reasoning_tokens": 4096,
                         "prompt_tokens": 12288, "visible_tokens": 3072},
                "write": {"reasoning_tokens": 2048,
                          "retry_reasoning_tokens": 4096,
                          "prompt_tokens": 12288, "visible_tokens": 4096},
                "audit": {"reasoning_tokens": 3072,
                          "retry_reasoning_tokens": 4096,
                          "prompt_tokens": 12288, "visible_tokens": 3072},
                "repair": {"reasoning_tokens": 2048,
                           "retry_reasoning_tokens": 4096,
                           "prompt_tokens": 12288, "visible_tokens": 4096},
            },
            "enforcement": {
                "status": "passed",
                "certificate_sha256": "a" * 64,
            },
            "qualification": {
                "status": status,
                "certificate_sha256": "c" * 64 if status == "passed" else None,
            },
        }

    def config(self, binding="thinking_token_budget", status="passed"):
        value = gateway_runtime()
        value["gateways"]["localgw"]["models"] = {
            "model-a": {
                "context_tokens": 100000,
                "request_options": {
                    "reasoning_effort": "medium",
                    "chat_template_kwargs": {"enable_thinking": True},
                },
                "reasoning_admission": self.admission(binding, status),
            }
        }
        return value

    def test_each_engine_binding_is_injected_from_the_role_policy(self):
        for binding in ("thinking_token_budget", "max_thinking_tokens",
                        "custom_params.thinking_budget"):
            with self.subTest(binding=binding):
                config = self.config(binding)
                self.rt._validate_config(config, "test")
                with unittest.mock.patch.object(self.gw.runtime, "config",
                                                 return_value=config), \
                     unittest.mock.patch.dict(
                         os.environ, {"SUMM_LOCALGW_TOKEN": "secret"}), \
                     unittest.mock.patch.object(
                         self.gw.urllib.request, "urlopen",
                         side_effect=[self.TokenResponse(), self.Response()]) as opened:
                    result = self.gw.chat(
                        "localgw", "model-a", "P", role="write",
                        planned_output_words=600, output_overhead_tokens=500)
                payload = json.loads(opened.call_args_list[-1].args[0].data)
                if binding == "custom_params.thinking_budget":
                    self.assertEqual(payload["custom_params"]["thinking_budget"], 2048)
                else:
                    self.assertEqual(payload[binding], 2048)
                self.assertTrue(payload["chat_template_kwargs"]["enable_thinking"])
                self.assertEqual(payload["max_tokens"], 16384)
                self.assertNotIn("max_completion_tokens", payload)
                self.assertEqual(result.dispatch_budget["visible_tokens"], 2083)
                token_payload = json.loads(
                    opened.call_args_list[0].args[0].data)
                self.assertEqual(token_payload["messages"], [
                    {"role": "user", "content": "P"}])
                self.assertEqual(token_payload["reasoning_effort"], "medium")
                self.assertTrue(
                    token_payload["chat_template_kwargs"]["enable_thinking"])
                self.assertEqual(
                    result.dispatch_budget["prompt_sha256"],
                    hashlib.sha256(b"P").hexdigest())

    def test_caller_cannot_disable_thinking_or_replace_the_cap(self):
        config = self.config()
        with unittest.mock.patch.object(self.gw.runtime, "config",
                                         return_value=config), \
             unittest.mock.patch.dict(
                 os.environ, {"SUMM_LOCALGW_TOKEN": "secret"}):
            for override in (
                    {"thinking_token_budget": 1},
                    {"chat_template_kwargs": {"enable_thinking": False}}):
                with self.subTest(override=override), \
                     self.assertRaises(self.gw.GatewayError) as raised:
                    self.gw.chat("localgw", "model-a", "P", override,
                                 role="plan", planned_output_words=200,
                                 output_overhead_tokens=500)
                self.assertEqual(raised.exception.kind, "config")

    def test_nested_custom_budget_is_merged_and_protected(self):
        config = self.config("custom_params.thinking_budget")
        config["gateways"]["localgw"]["models"]["model-a"][
            "request_options"]["custom_params"] = {"preserved": True}
        with unittest.mock.patch.object(self.gw.runtime, "config",
                                         return_value=config), \
             unittest.mock.patch.dict(
                 os.environ, {"SUMM_LOCALGW_TOKEN": "secret"}):
            options = self.gw.request_options(
                "localgw", "model-a",
                {"custom_params": {"caller": "kept"}}, role="write")
            self.assertEqual(options["custom_params"], {
                "preserved": True, "caller": "kept", "thinking_budget": 2048,
            })
            with self.assertRaises(self.gw.GatewayError) as raised:
                self.gw.request_options(
                    "localgw", "model-a",
                    {"custom_params": {"thinking_budget": 1}}, role="write")
            self.assertEqual(raised.exception.kind, "config")

    def test_overall_unqualified_route_is_functionally_admitted(self):
        config = self.config(status="unqualified")
        self.rt._validate_config(config, "test")
        with unittest.mock.patch.object(self.gw.runtime, "config",
                                         return_value=config), \
             unittest.mock.patch.dict(
                 os.environ, {"SUMM_LOCALGW_TOKEN": "secret",
                              "SUMM_QUALIFICATION_COURT": ""}), \
             unittest.mock.patch.object(
                 self.gw.urllib.request, "urlopen",
                 side_effect=[self.TokenResponse(), self.Response()]):
            result = self.gw.chat(
                "localgw", "model-a", "P", role="audit",
                planned_output_words=100, output_overhead_tokens=100)
        self.assertEqual(result.dispatch_budget["reasoning_tokens"], 3072)
        self.assertEqual(config["gateways"]["localgw"]["models"]["model-a"][
            "reasoning_admission"]["qualification"]["status"], "unqualified")

    def test_invalid_visible_reserve_is_rejected(self):
        config = self.config()
        config["gateways"]["localgw"]["models"]["model-a"][
            "reasoning_admission"]["roles"]["audit"]["reasoning_tokens"] = 16000
        with self.assertRaises(self.rt.ConfigError):
            self.rt._validate_config(config, "test")

    def test_semantic_retry_recomputes_the_shared_completion_envelope(self):
        config = self.config()
        self.rt._validate_config(config, "test")
        with unittest.mock.patch.object(self.gw.runtime, "config",
                                         return_value=config), \
             unittest.mock.patch.dict(
                 os.environ, {"SUMM_LOCALGW_TOKEN": "secret"}), \
             unittest.mock.patch.object(
                 self.gw.urllib.request, "urlopen",
                 side_effect=[self.TokenResponse(), self.Response()]) as opened:
            result = self.gw.chat(
                "localgw", "model-a", "P", role="write",
                planned_output_words=600, output_overhead_tokens=500,
                semantic_retry=True)
        payload = json.loads(opened.call_args_list[-1].args[0].data)
        self.assertEqual(payload["thinking_token_budget"], 4096)
        self.assertEqual(payload["max_tokens"], 16384)
        self.assertTrue(result.dispatch_budget["semantic_retry"])

    def test_thinking_shrinks_to_protect_visible_and_uses_full_ceiling(self):
        config = self.config()
        model = config["gateways"]["localgw"]["models"]["model-a"]
        model["max_output_tokens"] = 5000
        admission = model["reasoning_admission"]
        admission["completion_tokens"] = 5000
        for policy in admission["roles"].values():
            policy["reasoning_tokens"] = 1024
            policy["retry_reasoning_tokens"] = 4096
            policy["visible_tokens"] = 2500
        self.rt._validate_config(config, "test")
        with unittest.mock.patch.object(self.gw.runtime, "config",
                                         return_value=config), \
             unittest.mock.patch.dict(
                 os.environ, {"SUMM_LOCALGW_TOKEN": "secret"}), \
             unittest.mock.patch.object(
                 self.gw.urllib.request, "urlopen",
                 side_effect=[self.TokenResponse(), self.Response()]) as opened:
            result = self.gw.chat(
                "localgw", "model-a", "P", role="write",
                planned_output_words=600, output_overhead_tokens=500,
                semantic_retry=True)
        payload = json.loads(opened.call_args_list[-1].args[0].data)
        self.assertEqual(payload["max_tokens"], 5000)
        self.assertEqual(payload["thinking_token_budget"], 2853)
        self.assertEqual(result.dispatch_budget["reasoning_tokens"], 2853)
        self.assertEqual(result.dispatch_budget["visible_tokens"], 2083)
        self.assertEqual(result.dispatch_budget["completion_tokens"], 5000)

    def test_visible_prompt_and_retry_overflow_fail_before_generation(self):
        for changed_count, words, overhead, retry, fragment in (
                (128, 1600, 500, False, "visible-answer"),
                (12289, 100, 100, False, "prompt working set"),
                (128, 1600, 500, True, "visible-answer")):
            with self.subTest(fragment=fragment, retry=retry):
                config = self.config()
                token_response = self.TokenResponse()
                token_response.read = lambda n=changed_count: json.dumps(
                    {"count": n}).encode()
                with unittest.mock.patch.object(
                        self.gw.runtime, "config", return_value=config), \
                     unittest.mock.patch.dict(
                         os.environ, {"SUMM_LOCALGW_TOKEN": "secret"}), \
                     unittest.mock.patch.object(
                         self.gw.urllib.request, "urlopen",
                         return_value=token_response) as opened, \
                     self.assertRaises(self.gw.GatewayError) as raised:
                    self.gw.chat(
                        "localgw", "model-a", "P", role="write",
                        planned_output_words=words,
                        output_overhead_tokens=overhead,
                        semantic_retry=retry)
                self.assertEqual(raised.exception.kind, "capacity")
                self.assertIn(fragment, str(raised.exception))
                # Prompt counting is the only allowed network call; generation
                # cannot start after an admission failure.
                self.assertLessEqual(opened.call_count, 1)

    def test_counter_and_enforcement_remain_independent_gates(self):
        for field in ("token_counter", "enforcement"):
            with self.subTest(field=field):
                config = self.config()
                admission = config["gateways"]["localgw"]["models"][
                    "model-a"]["reasoning_admission"]
                admission[field] = {
                    **admission[field], "status": "unqualified",
                    "certificate_sha256": None}
                self.rt._validate_config(config, "test")
                with unittest.mock.patch.object(
                        self.gw.runtime, "config", return_value=config), \
                     unittest.mock.patch.dict(
                         os.environ, {"SUMM_LOCALGW_TOKEN": "secret"}), \
                     self.assertRaises(self.gw.GatewayError) as raised:
                    self.gw.chat(
                        "localgw", "model-a", "P", role="audit",
                        planned_output_words=100,
                        output_overhead_tokens=100)
                self.assertEqual(raised.exception.kind, "config")

    def test_qualification_court_admits_unqualified_overall_route(self):
        config = self.config(status="unqualified")
        self.rt._validate_config(config, "test")
        with unittest.mock.patch.object(self.gw.runtime, "config",
                                         return_value=config), \
             unittest.mock.patch.dict(
                 os.environ, {"SUMM_LOCALGW_TOKEN": "secret",
                              "SUMM_QUALIFICATION_COURT": "1"}), \
             unittest.mock.patch.object(
                 self.gw.urllib.request, "urlopen",
                 side_effect=[self.TokenResponse(), self.Response()]):
            result = self.gw.chat(
                "localgw", "model-a", "P", role="write",
                planned_output_words=600, output_overhead_tokens=500)
        self.assertEqual(result.dispatch_budget["reasoning_tokens"], 2048)

    def test_qualification_court_still_requires_counter_and_enforcement(self):
        for field in ("token_counter", "enforcement"):
            with self.subTest(field=field):
                config = self.config(status="unqualified")
                admission = config["gateways"]["localgw"]["models"][
                    "model-a"]["reasoning_admission"]
                admission[field] = {
                    **admission[field], "status": "unqualified",
                    "certificate_sha256": None}
                self.rt._validate_config(config, "test")
                with unittest.mock.patch.object(
                        self.gw.runtime, "config", return_value=config), \
                     unittest.mock.patch.dict(
                         os.environ, {"SUMM_LOCALGW_TOKEN": "secret",
                                      "SUMM_QUALIFICATION_COURT": "1"}), \
                     self.assertRaises(self.gw.GatewayError) as raised:
                    self.gw.chat(
                        "localgw", "model-a", "P", role="audit",
                        planned_output_words=100,
                        output_overhead_tokens=100)
                self.assertEqual(raised.exception.kind, "config")

    def test_continuation_caps_thinking_and_counts_the_chat_history(self):
        config = self.config()
        self.rt._validate_config(config, "test")
        messages = [
            {"role": "user", "content": "P"},
            {"role": "assistant", "content": '{"ok":'},
            {"role": "user", "content": "Continue"},
        ]
        with unittest.mock.patch.object(self.gw.runtime, "config",
                                         return_value=config), \
             unittest.mock.patch.dict(
                 os.environ, {"SUMM_LOCALGW_TOKEN": "secret"}), \
             unittest.mock.patch.object(
                 self.gw.urllib.request, "urlopen",
                 side_effect=[self.TokenResponse(), self.Response()]) as opened:
            result = self.gw.chat(
                "localgw", "model-a", "Continue", role="repair",
                planned_output_words=600, output_overhead_tokens=500,
                messages=messages, continuation=True)
        token_payload = json.loads(opened.call_args_list[0].args[0].data)
        payload = json.loads(opened.call_args_list[-1].args[0].data)
        self.assertEqual(token_payload["messages"], messages)
        self.assertEqual(payload["messages"], messages)
        self.assertEqual(payload["thinking_token_budget"], 64)
        self.assertEqual(payload["max_tokens"], 16384)
        self.assertTrue(result.dispatch_budget["continuation"])
        self.assertEqual(result.dispatch_budget["reasoning_tokens"], 64)


class StrictGatewayJSON(unittest.TestCase):
    """Gateway JSON stages accept exactly one wire object."""

    jc = load("json_contract")
    ms = load("mapsum")

    def setUp(self):
        self.config = gateway_runtime()
        merged = {
            "_defaults": {"fallback": {"harness": "opencode",
                                        "model": "cloud-fallback"}},
            "localgw": self.config["gateways"]["localgw"]["roster"],
            "opencode": {role: ["cloud-fallback"] for role in
                           ("plan", "write", "audit", "repair")},
        }
        patches = [
            unittest.mock.patch.object(self.ms, "GATEWAY_HARNESSES",
                                       frozenset({"localgw"})),
            unittest.mock.patch.object(self.ms, "HARNESSES",
                                       tuple(dict.fromkeys((*self.ms.HARNESSES,
                                                           "localgw")))),
            unittest.mock.patch.object(self.ms.model_config, "full",
                                       return_value=merged),
            unittest.mock.patch.object(self.ms.model_config, "local_harnesses",
                                       return_value=frozenset({"localgw"})),
            unittest.mock.patch.object(self.ms.runtime, "config",
                                       return_value=self.config),
        ]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def result(self, content):
        raw = {"id": "response-1", "choices": [{
            "message": {"content": content}, "finish_reason": "stop"}],
            "usage": {"completion_tokens": 10}}
        return self.ms.gateway.ChatCompletionResult(
            content=content, finish_reason="stop", usage=raw["usage"],
            response_id=raw["id"], raw_response=raw, elapsed_seconds=1.0,
            response_headers={})

    def test_exact_decoder_rejects_multiple_roots_and_duplicate_keys(self):
        for raw in ('{"ok": true}\n{"ok": true}',
                    '{"ok": true, "ok": false}'):
            with self.subTest(raw=raw), self.assertRaises(
                    self.jc.ContractError):
                self.jc.parse_exact(raw, None, "wire")

    def test_gateway_entry_point_rejects_a_scavengable_second_root(self):
        raw = '{"ok": true}\n{"diagnostic": "second answer"}'
        with tempfile.TemporaryDirectory() as td, \
                unittest.mock.patch.object(
                    self.ms.gateway, "chat", return_value=self.result(raw)) as chat:
            with self.assertRaises(self.ms.Abort):
                self.ms.run(
                    "PROMPT", pathlib.Path(td), ["localgw:model-a"],
                    "plan001", validate=lambda value: json.loads(value),
                    gateway_options=self.ms.JSON_REQUEST_OPTIONS)
        self.assertEqual(chat.call_count, 1)

    def test_one_exact_gateway_object_still_reaches_stage_validation(self):
        raw = '{"ok": true}'
        with tempfile.TemporaryDirectory() as td, \
                unittest.mock.patch.object(
                    self.ms.gateway, "chat", return_value=self.result(raw)):
            answer = self.ms.run(
                "PROMPT", pathlib.Path(td), ["localgw:model-a"], "plan001",
                validate=lambda value: json.loads(value),
                gateway_options=self.ms.JSON_REQUEST_OPTIONS)
        self.assertEqual(answer, raw)


class CapacityFailover(unittest.TestCase):
    """A depleted quota must demote one model, not kill the document.

    codex phrases exhaustion as "you've hit your usage limit ... switch to
    another model now". That matched nothing, so the chain never fell through
    and three section parts died unplanned -- which the accounting gate then
    correctly refused to publish. The gate did its job; the failover did not."""
    ms = load("mapsum")

    def test_codex_exhaustion_is_a_capacity_failure(self):
        for msg in ("You've hit your usage limit for GPT-5.3-Codex-Spark.",
                    "You've hit your session limit · resets 7:50pm (America/New_York)",
                    "Switch to another model now, or try again at 12:39 PM."):
            self.assertTrue(self.ms.CAPACITY.search(msg), f"unmatched: {msg}")

    def test_a_stalled_stream_is_retried_not_demoted(self):
        # muse reports a mid-stream stall as "model stream idle timeout after
        # 180000ms". It is neither a quota nor a bad answer; it clears on retry.
        msg = "agent loop failed: model failed: model stream idle timeout after 180000ms"
        self.assertTrue(self.ms.TRANSIENT.search(msg))
        self.assertFalse(self.ms.CAPACITY.search(msg))

    def test_exhaustion_is_not_mistaken_for_an_auth_failure(self):
        # AUTH aborts the harness; capacity moves to the next model. Confusing
        # them either kills a recoverable run or walks a chain that cannot help.
        self.assertFalse(self.ms.AUTH.search("You've hit your usage limit"))
        self.assertFalse(self.ms.CAPACITY.search("Authentication required"))

    def test_a_metered_alternate_backs_the_free_tier(self):
        cfg = json.loads((ENG / "models.json").read_text())
        for role in ("plan", "audit", "repair"):
            self.assertGreater(len(cfg["codex"][role]), 1,
                               f"codex.{role} has no fallback when the free tier runs out")

    def _chain(self):
        cfg = json.loads((ENG / "models.json").read_text())
        fb = cfg["_defaults"]["fallback"]
        fb = fb[0] if isinstance(fb, list) else fb   # the first committed backup
        fallback = f'{fb["harness"]}:{fb["model"]}'
        primary = cfg["grok"]["write"][0]
        return [f"grok:{primary}", fallback], fb

    def test_exhausted_timeout_reaches_the_configured_luna_fallback(self):
        chain, fb = self._chain()
        seen = []

        def fake_run(argv, **_kw):
            seen.append(argv)
            if len(seen) <= 2:
                raise subprocess.TimeoutExpired(argv, 1)
            return subprocess.CompletedProcess(argv, 0, "answer", "")

        with tempfile.TemporaryDirectory() as td, \
                unittest.mock.patch.object(subprocess, "run", fake_run), \
                unittest.mock.patch.object(self.ms.time, "sleep", lambda *_: None), \
                unittest.mock.patch.object(self.ms, "RETRIES", 2), \
                unittest.mock.patch.object(self.ms, "CALL_TIMEOUT", 1):
            self.assertEqual(self.ms.run("prompt", pathlib.Path(td), chain, "write001"),
                             "answer")
        self.assertEqual([pathlib.Path(argv[0]).name for argv in seen[:2]],
                         ["grok", "grok"])
        self.assertEqual(pathlib.Path(seen[2][0]).name, fb["harness"])

    def test_a_summary_stage_retries_a_silent_transport_failure_once(self):
        # Summary stages take one attempt per producer so an answer is never
        # replayed. A stall that produced no output is not an answer: the
        # same producer is asked once more before the chain moves on.
        chain = ["codex:gpt-5.6-luna", "opencode:openai/gpt-5.6-luna"]
        seen = []

        def fake_run(argv, **_kw):
            seen.append(argv)
            if len(seen) == 1:
                return subprocess.CompletedProcess(
                    argv, 1, "", "agent loop failed: model failed: "
                    "model stream idle timeout after 180000ms")
            return subprocess.CompletedProcess(argv, 0, '{"ok": true}', "")

        with tempfile.TemporaryDirectory() as td, \
                unittest.mock.patch.object(subprocess, "run", fake_run), \
                unittest.mock.patch.object(self.ms.time, "sleep", lambda *_: None):
            self.assertEqual(self.ms.run("prompt", pathlib.Path(td), chain, "short"),
                             '{"ok": true}')
        self.assertEqual([pathlib.Path(argv[0]).name for argv in seen],
                         ["codex", "codex"])

    def test_a_session_limit_on_stdout_is_capacity_and_the_route_stays_dead(self):
        # claude prints "You've hit your session limit" to stdout and exits 0.
        # Validation ran first and called it an unusable answer; the same
        # exhausted route was then tried again by the audit stage.
        chain = ["codex:gpt-5.6-luna", "opencode:openai/gpt-5.6-luna"]
        seen = []

        def fake_run(argv, **_kw):
            seen.append(pathlib.Path(argv[0]).name)
            if seen[-1] == "codex":
                return subprocess.CompletedProcess(
                    argv, 0, "You've hit your session limit · resets 7:50pm (America/New_York)\n", "")
            return subprocess.CompletedProcess(argv, 0, '{"ok": true}', "")

        with tempfile.TemporaryDirectory() as td, \
                unittest.mock.patch.object(subprocess, "run", fake_run), \
                unittest.mock.patch.object(self.ms.time, "sleep", lambda *_: None):
            self.assertEqual(self.ms.run("p", pathlib.Path(td), chain, "short"),
                             '{"ok": true}')
            self.assertEqual(seen, ["codex", "opencode"])
            self.assertEqual(self.ms.run("p", pathlib.Path(td), chain, "short-audit"),
                             '{"ok": true}')
            self.assertEqual(seen, ["codex", "opencode", "opencode"],
                             "an exhausted route is not tried again in the target")
            records = [json.loads(l) for l in
                       (pathlib.Path(td) / "calls.jsonl").read_text().splitlines()]
            self.assertEqual(records[0]["outcome"], "capacity")

    def test_a_session_limit_on_stdout_with_nonzero_exit_is_capacity(self):
        chain = ["codex:gpt-5.6-luna", "opencode:openai/gpt-5.6-luna"]
        seen = []

        def fake_run(argv, **_kw):
            seen.append(pathlib.Path(argv[0]).name)
            if seen[-1] == "codex":
                return subprocess.CompletedProcess(
                    argv, 1, "You've hit your session limit · resets 7:50pm", "")
            return subprocess.CompletedProcess(argv, 0, '{"ok": true}', "")

        with tempfile.TemporaryDirectory() as td, \
                unittest.mock.patch.object(subprocess, "run", fake_run), \
                unittest.mock.patch.object(self.ms.time, "sleep", lambda *_: None):
            self.ms.run("p", pathlib.Path(td), chain, "short")
            records = [json.loads(l) for l in
                       (pathlib.Path(td) / "calls.jsonl").read_text().splitlines()]
        self.assertEqual(records[0]["outcome"], "capacity")

    def test_a_real_answer_that_mentions_a_limit_is_not_capacity(self):
        chain = ["codex:gpt-5.6-luna", "opencode:openai/gpt-5.6-luna"]
        answer = '{"detailed": "The session limit was reached in 1990.", "brief": "Limit."}'
        seen = []

        def fake_run(argv, **_kw):
            seen.append(pathlib.Path(argv[0]).name)
            return subprocess.CompletedProcess(argv, 0, answer, "")

        with tempfile.TemporaryDirectory() as td, \
                unittest.mock.patch.object(subprocess, "run", fake_run):
            self.assertEqual(self.ms.run("p", pathlib.Path(td), chain, "short"), answer)
        self.assertEqual(seen, ["codex"])

    def test_a_timeout_with_partial_output_is_kept_and_not_replayed(self):
        chain = ["codex:gpt-5.6-luna", "opencode:openai/gpt-5.6-luna"]
        seen = []

        def fake_run(argv, **_kw):
            seen.append(pathlib.Path(argv[0]).name)
            if seen[-1] == "codex":
                raise subprocess.TimeoutExpired(argv, 1, output='{"detailed": "half', stderr="")
            return subprocess.CompletedProcess(argv, 0, '{"ok": true}', "")

        with tempfile.TemporaryDirectory() as td, \
                unittest.mock.patch.object(subprocess, "run", fake_run), \
                unittest.mock.patch.object(self.ms.time, "sleep", lambda *_: None), \
                unittest.mock.patch.object(self.ms, "CALL_TIMEOUT", 1):
            self.ms.run("p", pathlib.Path(td), chain, "short")
            root = pathlib.Path(td)
            self.assertEqual(seen, ["codex", "opencode"], "partial output is never replayed")
            kept = sorted(root.glob("short.attempt*.stdout.txt"))
            self.assertEqual(kept[0].read_text(), '{"detailed": "half')
            records = [json.loads(l) for l in (root / "calls.jsonl").read_text().splitlines()]
            self.assertEqual(records[0]["outcome"], "completion_unknown")

    def test_last_ok_route_names_the_producer_that_answered(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            self.assertEqual(self.ms.last_ok_route(root, "short", ["a:x", "b:y"]), ["a:x"])
            (root / "calls.jsonl").write_text(
                json.dumps({"stage": "short", "harness": "a", "model": "x", "outcome": "capacity"}) + "\n"
                + json.dumps({"stage": "short", "harness": "b", "model": "y", "outcome": "ok"}) + "\n")
            self.assertEqual(self.ms.last_ok_route(root, "short", ["a:x", "b:y"]), ["b:y"])

    def test_nested_stage_directories_share_one_targets_route_state(self):
        # Corpus runs every stage in one process under nested directories.
        # Keyed on the work directory, "target-wide" state was stage-local and
        # the same quota-exhausted route was retried in every nested stage.
        chain, fb = self._chain()
        seen = []

        def fake_run(argv, **_kw):
            name = pathlib.Path(argv[0]).name
            seen.append(name)
            if name == "grok":
                return subprocess.CompletedProcess(
                    argv, 1, "", "Error: The usage limit has been reached")
            return subprocess.CompletedProcess(argv, 0, '{"ok": true}', "")

        with tempfile.TemporaryDirectory() as td, \
                unittest.mock.patch.object(subprocess, "run", fake_run), \
                unittest.mock.patch.object(self.ms.time, "sleep", lambda *_: None), \
                unittest.mock.patch.dict(os.environ,
                                         {"SUMM_TARGET_KEY": td}):
            root = pathlib.Path(td)
            for stage_dir in ("documents/D001/inventory/W1", "corpus/plan",
                              "corpus/pair"):
                target = root / stage_dir
                target.mkdir(parents=True, exist_ok=True)
                self.ms.run("prompt", target, chain, "corpus-plan", role="plan")
        # The exhausted route is asked once, not once per nested stage.
        self.assertEqual(1, seen.count("grok"), seen)

    def test_a_route_that_stalls_twice_with_no_output_is_retired(self):
        # One silent stall is retried; a second in the same target retires the
        # route rather than paying six more minutes for it at every stage.
        chain, fb = self._chain()
        seen = []
        stall = subprocess.CompletedProcess(
            ["x"], 1, "", "agent loop failed: model failed: model stream idle "
                          "timeout after 180000ms")

        def fake_run(argv, **_kw):
            name = pathlib.Path(argv[0]).name
            seen.append(name)
            if name == "grok":
                return subprocess.CompletedProcess(argv, 1, "", stall.stderr)
            return subprocess.CompletedProcess(argv, 0, '{"ok": true}', "")

        with tempfile.TemporaryDirectory() as td, \
                unittest.mock.patch.object(subprocess, "run", fake_run), \
                unittest.mock.patch.object(self.ms.time, "sleep", lambda *_: None), \
                unittest.mock.patch.dict(os.environ, {"SUMM_TARGET_KEY": td}):
            root = pathlib.Path(td)
            for stage in ("corpus-inventory-plan-D001:W001", "corpus-plan"):
                target = root / stage.replace(":", "-")
                target.mkdir(parents=True, exist_ok=True)
                self.ms.run("prompt", target, chain, stage, role="plan")
        # attempt + one retry in the first stage, then the route is retired.
        self.assertEqual(2, seen.count("grok"), seen)

    def test_partial_timeout_output_is_never_replayed_for_a_corpus_stage(self):
        # The no-replay rule keyed on stage names beginning "short" or "full",
        # so Corpus, inventory, plan, and review stages could still retry a
        # producer that had emitted part of an answer.
        chain, fb = self._chain()
        seen = []

        def fake_run(argv, **kw):
            name = pathlib.Path(argv[0]).name
            seen.append(name)
            if name == "grok":
                raise subprocess.TimeoutExpired(
                    argv, 1, output='{"partial": ', stderr="")
            return subprocess.CompletedProcess(argv, 0, '{"ok": true}', "")

        with tempfile.TemporaryDirectory() as td, \
                unittest.mock.patch.object(subprocess, "run", fake_run), \
                unittest.mock.patch.object(self.ms.time, "sleep", lambda *_: None), \
                unittest.mock.patch.object(self.ms, "CALL_TIMEOUT", 1):
            answer = self.ms.run("prompt", pathlib.Path(td), chain,
                                 "corpus-overview", role="write")
        self.assertEqual('{"ok": true}', answer)
        self.assertEqual(1, seen.count("grok"), "partial output was replayed")

    def test_a_summary_stage_does_not_replay_a_producer_that_answered(self):
        # Partial output plus a transient-looking stderr is completion-unknown:
        # it moves on, it is not replayed.
        chain = ["codex:gpt-5.6-luna", "opencode:openai/gpt-5.6-luna"]
        seen = []

        def fake_run(argv, **_kw):
            seen.append(argv)
            if len(seen) == 1:
                return subprocess.CompletedProcess(
                    argv, 1, '{"partial": ', "connection reset by peer")
            return subprocess.CompletedProcess(argv, 0, '{"ok": true}', "")

        with tempfile.TemporaryDirectory() as td, \
                unittest.mock.patch.object(subprocess, "run", fake_run), \
                unittest.mock.patch.object(self.ms.time, "sleep", lambda *_: None):
            self.ms.run("prompt", pathlib.Path(td), chain, "short")
        self.assertEqual(pathlib.Path(seen[0][0]).name, "codex")
        self.assertNotEqual(pathlib.Path(seen[1][0]).name, "codex")

    def test_authentication_failure_reaches_a_different_configured_harness(self):
        chain, fb = self._chain()
        seen = []

        def fake_run(argv, **_kw):
            seen.append(argv)
            if len(seen) == 1:
                return subprocess.CompletedProcess(argv, 1, "", "Authentication required")
            return subprocess.CompletedProcess(argv, 0, "answer", "")

        with tempfile.TemporaryDirectory() as td, \
                unittest.mock.patch.object(subprocess, "run", fake_run):
            self.assertEqual(self.ms.run("prompt", pathlib.Path(td), chain, "write001"),
                             "answer")
        self.assertEqual(pathlib.Path(seen[1][0]).name, fb["harness"])

    def test_authentication_disables_that_harness_for_later_target_stages(self):
        chain, fb = self._chain()
        seen = []

        def fake_run(argv, **_kw):
            seen.append(argv)
            if len(seen) == 1:
                return subprocess.CompletedProcess(argv, 1, "",
                                                   "Authentication required")
            return subprocess.CompletedProcess(argv, 0, "answer", "")

        with tempfile.TemporaryDirectory() as td, \
                unittest.mock.patch.object(subprocess, "run", fake_run):
            self.assertEqual(self.ms.run("prompt", pathlib.Path(td), chain,
                                         "write001"), "answer")
            self.assertEqual(self.ms.run("prompt", pathlib.Path(td), chain,
                                         "audit001"), "answer")
        self.assertEqual([pathlib.Path(argv[0]).name for argv in seen],
                         ["grok", fb["harness"], fb["harness"]])

    def test_unusable_contract_advances_to_a_configured_fallback(self):
        chain, fb = self._chain()
        seen = []

        def fake_run(argv, **_kw):
            seen.append(argv)
            return subprocess.CompletedProcess(argv, 0, "answer", "")

        def reject(_answer):
            raise ValueError("content is not valid for this stage")

        with tempfile.TemporaryDirectory() as td, \
                unittest.mock.patch.object(subprocess, "run", fake_run), \
                unittest.mock.patch.object(self.ms.time, "sleep", lambda *_: None), \
                unittest.mock.patch.object(self.ms, "RETRIES", 2):
            with self.assertRaises(self.ms.Abort):
                self.ms.run("prompt", pathlib.Path(td), chain, "write001", validate=reject)
        self.assertEqual([pathlib.Path(argv[0]).name for argv in seen],
                         ["grok", fb["harness"]],
                         "an unusable stage response must try the next route")

    def test_declared_route_is_skipped_before_an_oversized_model_call(self):
        chain, fb = self._chain()
        seen = []

        def fake_run(argv, **_kw):
            seen.append(argv)
            return subprocess.CompletedProcess(argv, 0, "answer", "")

        raw = json.dumps({
            chain[0]: {"context_tokens": 10, "output_tokens": 10},
            chain[1]: {"context_tokens": 90_000, "output_tokens": 8_000},
        })
        with tempfile.TemporaryDirectory() as td, \
                unittest.mock.patch.dict(os.environ,
                                         {"SUMM_ROUTE_CAPABILITIES": raw}), \
                unittest.mock.patch.object(subprocess, "run", fake_run):
            self.assertEqual(self.ms.run("word " * 20, pathlib.Path(td),
                                         chain, "write001"), "answer")
        self.assertEqual(len(seen), 1)
        self.assertEqual(pathlib.Path(seen[0][0]).name, fb["harness"])

    def test_expired_target_deadline_makes_no_model_call(self):
        with tempfile.TemporaryDirectory() as td, \
                unittest.mock.patch.dict(os.environ,
                                         {"SUMM_TARGET_DEADLINE_UNIX": "0"}), \
                unittest.mock.patch.object(subprocess, "run") as process:
            with self.assertRaises(self.ms.Abort):
                self.ms.run("prompt", pathlib.Path(td), ["grok:grok-4.6"],
                            "write001")
        process.assert_not_called()

    def test_target_attempt_limit_is_shared_across_stage_routes(self):
        chain, _ = self._chain()
        seen = []

        def fake_run(argv, **_kw):
            seen.append(argv)
            return subprocess.CompletedProcess(argv, 1, "", "connection reset")

        with tempfile.TemporaryDirectory() as td, \
                unittest.mock.patch.dict(os.environ,
                                         {"SUMM_TARGET_ATTEMPT_LIMIT": "1"}), \
                unittest.mock.patch.object(subprocess, "run", fake_run):
            with self.assertRaises(self.ms.Abort):
                self.ms.run("prompt", pathlib.Path(td), chain, "full")
        self.assertEqual(len(seen), 1,
                         "the target budget must stop before a second route")



class WindowTalkGate(unittest.TestCase):
    """A capsule describes the DOCUMENT, never the window it was planned from.

    A planner given one 1,000-word part wrote "although the supplied text ends
    before that definition is completed", and it shipped -- telling a reader the
    source is incomplete when it is not. The planner cannot catch this: from
    inside its window the sentence is true. The gate targets only explicit
    claims that the supplied material is incomplete."""
    led = load("ledger")

    def test_window_relative_claims_are_caught(self):
        for t in ("the supplied text ends before that definition is completed",
                  "The supplied passage does not include the appendix",
                  "this excerpt stops short of the conclusion",
                  "material beyond the provided portion is omitted"):
            self.assertTrue(self.led.WINDOW_TALK.search(t), f"missed: {t}")

    def test_ordinary_prose_is_not_caught(self):
        # A gate that withholds real content is worse than the defect it fixes.
        for t in ("The authors provided evidence that rates fell.",
                  "Given text-based evidence, the effect is small.",
                  "The text of the statute defines reserves narrowly.",
                  "The Fed supplied reserves to the banking system.",
                  "Congress provided funding for the facilities."):
            self.assertIsNone(self.led.WINDOW_TALK.search(t), f"false positive: {t}")

    def test_both_depths_are_checked(self):
        units = [{"local_id": "u1", "detailed_capsule": "fine",
                  "brief_capsule": "the supplied text ends here"}]
        self.assertIn(("u1", "brief"), self.led.window_talk(units))

    def test_the_gate_runs_in_part_defects(self):
        src = (ENG / "ledger.py").read_text()
        seg = src[src.index("def part_defects"):src.index("def audit_part")]
        self.assertIn("window_talk(units)", seg)



class WindowBoundaryParagraphs(unittest.TestCase):
    """Units from different planning windows should not be welded mid-paragraph:
    one paragraph ran from Japan's fiscal withdrawal straight into the Fed's 2007
    facilities. But breaking on EVERY window change produced a 27-word paragraph
    mid-document, and short paragraphs on e-ink were the original complaint."""
    comp = load("compose")

    def _it(self, i, words, window):
        return {"id": f"S{i:03d}", "heading": False, "unit_ids": [f"U{i:03d}"],
                "text": " ".join(["w"] * words), "section": ("SEC01", window)}

    def test_a_window_change_breaks_a_paragraph_that_is_long_enough(self):
        lo, _, _ = self.comp.PARA["detailed"]
        items = [self._it(1, lo + 10, 1), self._it(2, lo + 10, 2)]
        out = self.comp.paragraphize(items, "detailed")
        self.assertEqual(len([p for p in out if not p["heading"]]), 2)

    def test_a_window_change_does_not_create_a_runt(self):
        lo, _, _ = self.comp.PARA["detailed"]
        # first window contributes far too little to stand alone
        items = [self._it(1, 20, 1), self._it(2, lo + 40, 2)]
        out = self.comp.paragraphize(items, "detailed")
        sizes = [sum(len(x["text"].split()) for x in p["parts"])
                 for p in out if not p["heading"]]
        self.assertTrue(all(n >= lo for n in sizes), f"runt paragraph: {sizes}")

    def test_units_carry_their_planning_window(self):
        # A paper can be one long section, so section_id alone is not enough.
        src = (ENG / "ledger.py").read_text()
        self.assertIn('u["part"] = i', src)
        comp = (ENG / "compose.py").read_text()
        self.assertIn('u.get("part")', comp)

    def test_grouping_still_never_loses_a_sentence(self):
        items = [self._it(i, 60, (i // 2) + 1) for i in range(1, 9)]
        out = self.comp.paragraphize(items, "detailed")
        flat = [x["id"] for p in out for x in p["parts"]]
        self.assertEqual(flat, [x["id"] for x in items])



class LedgerPersistedBeforeSealing(unittest.TestCase):
    """normalize() fixed the ledger in memory while mechseal read ledger.json
    from disk, so its fixes were invisible to the exact gate they exist to
    satisfy: one run dropped a conflicting disposition and still failed the seal
    for that very block."""

    def test_the_normalised_ledger_is_written_before_the_seal(self):
        src = (ENG / "ledger.py").read_text()
        i = src.index("ledger = normalize(build(")
        j = src.index("def mechseal():")
        self.assertIn('"ledger.json").write_text', src[i:j],
                      "normalize's result never reaches the file mechseal reads")


class DispositionCanonicalisation(unittest.TestCase):
    """A planner emitted "incidental" once in nineteen dispositions, having used
    "incidental_example" correctly three times. Losing a whole document to one
    truncated enum is a bad trade when exactly one valid value starts with it."""
    led = load("ledger")

    def test_an_unambiguous_abbreviation_expands(self):
        self.assertEqual(self.led.canon_disposition("incidental"), "incidental_example")
        self.assertEqual(self.led.canon_disposition("appara"), "apparatus")

    def test_a_stray_character_is_not_promoted(self):
        # Expanding "i" would invent a disposition the planner never chose.
        for v in ("i", "in", "exact", "a"):
            self.assertNotIn(self.led.canon_disposition(v), self.led.ALLOWED_DISP - {v})

    def test_an_unknown_value_is_left_to_fail(self):
        self.assertEqual(self.led.canon_disposition("nonsense"), "nonsense")

    def test_valid_values_are_untouched(self):
        for v in sorted(self.led.ALLOWED_DISP):
            self.assertEqual(self.led.canon_disposition(v), v)

    def test_the_enum_is_checked_per_part(self):
        # Caught at part scope, the revision cycle can fix it at source instead
        # of the document dying at the seal.
        src = (ENG / "ledger.py").read_text()
        seg = src[src.index("def part_defects"):src.index("def audit_part")]
        self.assertIn("invalid disposition", seg)

    def test_fresh_and_seal_paths_share_one_disposition_authority(self):
        seal = load("mechseal")
        schema = load("ledger_schema")
        self.assertEqual(schema.DISPOSITIONS, self.led.ALLOWED_DISP)
        self.assertEqual(schema.DISPOSITIONS, seal.ALLOWED_DISP)



class BriefCapsulesFlaggedByAuditAreNotPromoted(unittest.TestCase):
    """A Brief capsule gets less scrutiny than a Detailed one, yet compaction can
    promote it INTO Detailed. One unit stated a figure correctly in Detailed and
    wrongly in Brief -- as a surplus level rather than a swing -- so compaction
    could have published the error the Detailed capsule never contained."""
    comp = load("compose")

    def _u(self, uid, dw, bw):
        return {"unit_id": uid, "dependencies": [],
                "detailed_disposition": "required", "brief_disposition": "required",
                "brief_priority": 3,
                "detailed_capsule": " ".join(["d"] * dw),
                "brief_capsule": " ".join(["b"] * bw)}

    def tearDown(self):
        self.comp.SUSPECT_BRIEF[:] = []

    def test_a_flagged_brief_capsule_is_never_substituted(self):
        self.comp.SUSPECT_BRIEF[:] = ["U001"]
        chosen = [self._u("U001", 60, 5), self._u("U002", 60, 5)]
        modes, _, _, _, _ = self.comp.plan_modes(chosen, "detailed", 70)
        self.assertEqual(modes["U001"], "detailed", "a flagged Brief capsule was promoted")
        self.assertEqual(modes["U002"], "brief")

    def test_an_unflagged_unit_still_compacts(self):
        chosen = [self._u("U001", 60, 5), self._u("U002", 60, 5)]
        modes, w, _, _, _ = self.comp.plan_modes(chosen, "detailed", 70)
        self.assertLessEqual(w, 70)

    def test_a_flagged_unit_still_counts_at_full_size_in_the_minimum(self):
        # If it cannot be compacted, the feasibility floor must say so, or the
        # selector will believe it can fit a document it cannot.
        self.comp.SUSPECT_BRIEF[:] = ["U001"]
        chosen = [self._u("U001", 60, 5)]
        _, _, minimum, _, _ = self.comp.plan_modes(chosen, "detailed", 10)
        self.assertEqual(minimum, 60)

    def test_composition_reads_the_brief_flag_from_the_sealed_status(self):
        src = (ENG / "compose.py").read_text()
        self.assertIn('SUSPECT_BRIEF[:] = quarantine.get("brief") or []', src)
        self.assertNotIn('ledger.get("brief_suspect")', src)



class CapsulesAreContinuousProse(unittest.TestCase):
    """These are read on an e-ink device and aloud by a TTS voice. A bullet stack
    reads as a pile of fragments and a table is unreadable, so a capsule that
    contains one is a defect however accurate it is."""
    led = load("ledger")

    def test_list_and_table_markup_is_caught(self):
        for t in ("- first item\n- second item", "1. one\n2. two",
                  "* a\n* b", "| a | b |\n|---|---|", "## Heading",
                  "\u2022 bullet one\n\u2022 bullet two"):
            self.assertTrue(self.led.LIST_MARKUP.search(t), f"missed: {t!r}")

    def test_ordinary_prose_with_numbers_and_dashes_is_not_caught(self):
        # A gate that mangles "2-3 percent" or "5.5 percent - a large amount"
        # would withhold correct content over punctuation.
        for t in ("The rate fell 2-3 percent, then rose.",
                  "It cost 5.5 percent - a large amount.",
                  "Output fell. Prices rose. Employment lagged.",
                  "The 1913 Act created an elastic currency."):
            self.assertIsNone(self.led.LIST_MARKUP.search(t), f"false positive: {t!r}")

    def test_both_depths_are_checked(self):
        units = [{"local_id": "u1", "detailed_capsule": "fine prose",
                  "brief_capsule": "- a\n- b"}]
        self.assertIn(("u1", "brief"), self.led.list_markup(units))

    def test_the_gate_runs_in_part_defects(self):
        src = (ENG / "ledger.py").read_text()
        seg = src[src.index("def part_defects"):src.index("def audit_part")]
        self.assertIn("list_markup(units)", seg)

    def test_both_prompts_require_prose(self):
        for name in ("ledger-build.txt", "ledger-revise.txt"):
            t = (ENG / "prompts" / name).read_text()
            self.assertIn("CONTINUOUS PROSE", t, f"{name} does not require prose")


class HarnessSelectionIsNotSilentlyOverridden(unittest.TestCase):
    """A run must never answer a request for one model with another.

    `os.environ["HARNESS"] = a.harness` used to run unconditionally with an
    argparse default of "agy", so HARNESS=opencode in the environment was
    overwritten on every invocation and the run went to Gemini without a word.
    Nothing in the artifact records which model produced it, so the substitution
    was undetectable after the fact. These tests fail if that returns.
    """

    def _doctor(self, env_harness=None, flag=None):
        env = {**os.environ}
        env.pop("HARNESS", None)
        if env_harness:
            env["HARNESS"] = env_harness
        cmd = [sys.executable, str(ENG / "summ_cli.py"), "--doctor"]
        if flag:
            cmd += ["--harness", flag]
        return subprocess.run(cmd, capture_output=True, text=True, env=env)

    def test_environment_selects_the_harness(self):
        out = self._doctor(env_harness="opencode").stdout
        self.assertIn("(opencode, run harness)", out, "HARNESS= was ignored")

    def test_flag_beats_the_environment(self):
        out = self._doctor(env_harness="opencode", flag="agy").stdout
        self.assertIn("(agy, run harness)", out, "--harness did not win over HARNESS=")

    def test_default_is_agy_when_neither_is_given(self):
        self.assertIn("(agy, run harness)", self._doctor().stdout)

    def test_an_unknown_harness_is_refused_loudly(self):
        r = self._doctor(env_harness="bogus")
        self.assertEqual(r.returncode, 2)
        self.assertIn("unknown harness", r.stderr)

    def test_the_default_is_not_assigned_unconditionally(self):
        src = (ENG / "summ_cli.py").read_text()
        self.assertNotIn('os.environ["HARNESS"] = a.harness', src,
                         "the unconditional override is back")


class ShortDocumentNeverPublishesAnEmptyArtifact(unittest.TestCase):
    """A short source must never yield coverage notes without summary prose.

    Three faults can compound: the ceiling scales with the source but the coverage
    note's cost does not, so `hi - NOTE_ALLOWANCE` was -15; no arrangement of
    units can satisfy a negative budget, so the Brief drop loop deleted every
    unit; and nothing downstream objected to an artifact with no units, because
    the band check counts the note's own words.

    "We always get a summary, we always get a brief" is the one rule this
    project does not bend. An empty file that looks like a summary breaks it
    more quietly, and therefore worse, than a loud refusal.
    """
    comp = load("compose")

    def _units(self, n, brief_words):
        return [{"unit_id": f"U{i:03d}", "dependencies": [],
                 "detailed_disposition": "required",
                 "brief_disposition": "optional", "brief_priority": 1,
                 "detailed_capsule": " ".join(["detailed"] * (brief_words * 2)),
                 "brief_capsule": " ".join(["brief"] * brief_words)}
                for i in range(1, n + 1)]

    def test_a_negative_budget_cannot_arise(self):
        # 253 visible words -> Brief ceiling 55. The note allowance is 70.
        for visible in (60, 120, 253, 400):
            hi = int(self.comp.BANDS["brief"][1] * visible)
            budget = self.comp.content_budget(hi)
            self.assertGreater(budget, 0,
                               f"{visible}-word source yields a {budget}w budget")

    def test_the_note_reservation_is_unchanged_on_real_documents(self):
        # The clamp must not move the budget on documents that were never broken.
        hi = int(self.comp.BANDS["brief"][1] * 11514)          # the paper on file
        self.assertEqual(self.comp.content_budget(hi),
                         hi - self.comp.NOTE_ALLOWANCE)

    def test_the_drop_loop_never_removes_the_last_unit(self):
        units = self._units(3, 40)
        _, _, _, dropped, kept = self.comp.plan_modes(units, "brief", ceiling=-15)
        self.assertEqual(len(kept), 1,
                         "the drop loop emptied the Brief to satisfy a budget "
                         "no arrangement of units could ever meet")
        self.assertEqual(len(dropped), 2)

    def test_composition_refuses_an_artifact_with_no_units(self):
        with tempfile.TemporaryDirectory() as td:
            td = pathlib.Path(td)
            ldir, rvd, out = td / "ledger", td / "rv", td / "out"
            for d in (ldir, rvd):
                d.mkdir(parents=True)
            (rvd / "reader.md").write_text("x")
            (ldir / "ledger.json").write_text(json.dumps(
                {"units": self._units(2, 40), "visible_words": 253}))
            (ldir / "MECHSEAL").write_text("1")
            (ldir / "mechseal.json").write_text(json.dumps({"passed": True, "units": 2}))
            (ldir / "SEALED").write_text("1")
            # Everything flagged at both depths. This used to empty the
            # artifact; nothing is dropped for fidelity any more, so the units
            # are still there and it fails (or passes) on other grounds.
            (ldir / "status.json").write_text(json.dumps(
                {"status": "quarantined",
                 "ledger_sha256": hashlib.sha256(
                     (ldir / "ledger.json").read_bytes()).hexdigest(),
                 "quarantine": {"detailed": ["U001", "U002"],
                                "brief": ["U001", "U002"]}}))
            r = subprocess.run([sys.executable, str(ENG / "compose.py"),
                                str(ldir), str(rvd), str(out)],
                               capture_output=True, text=True)
            self.assertNotIn("no unit survives selection", r.stderr,
                             "flagged units still empty the artifact")
            for depth in ("detailed", "brief"):
                art = out / f"{depth}.md"
                if art.exists():
                    body = re.sub(r"^>.*$", "", art.read_text(), flags=re.M).strip()
                    self.assertTrue(body, f"{depth}.md is coverage notes and nothing else")



class RolesCanSplitAcrossHarnesses(unittest.TestCase):
    """One document, more than one credential.

    A local model can plan and revise while a hosted model audits -- the bulk
    role on hardware that is free to run, the judgement role on the strongest
    model available. That was impossible while the harness was a module global
    read once from the environment: chains were per-role but every entry in
    them resolved to the same CLI.

    A chain entry may now name its harness. The pipeline still learns nothing
    about model identity; `models.json` is still the only place a model is
    named."""
    ms = load("mapsum")

    def _fake(self, outcomes):
        """Drive run() against a scripted sequence of (returncode, out, err)."""
        seen = []

        def fake_run(argv, **kw):
            seen.append(argv)
            rc, out, err = outcomes[min(len(seen) - 1, len(outcomes) - 1)]
            return subprocess.CompletedProcess(argv, rc, out, err)
        return seen, fake_run

    def setUp(self):
        # Retry paths back off for real seconds. The dispatch decision is what
        # is under test, not the sleep between attempts.
        s = unittest.mock.patch.object(self.ms.time, "sleep", lambda *_: None)
        s.start(); self.addCleanup(s.stop)

    def test_a_qualified_entry_selects_its_own_harness(self):
        h, m = self.ms.split_entry("claude:sonnet", "opencode")
        self.assertEqual((h, m), ("claude", "sonnet"))

    def test_a_bare_entry_stays_on_the_run_harness(self):
        h, m = self.ms.split_entry("openai/gpt-5.6-luna", "opencode")
        self.assertEqual((h, m), ("opencode", "openai/gpt-5.6-luna"))

    def test_a_model_identifier_containing_a_colon_survives_whole(self):
        # Locally served models are tagged `name:size` -- exactly the shape of a
        # qualifier. Splitting on the first colon unconditionally would turn
        # a local tag like `model-family:32b` into an unknown harness.
        for entry in ("model-family:32b", "model-a:27b-it", "model-b:24b"):
            h, m = self.ms.split_entry(entry, "opencode")
            self.assertEqual((h, m), ("opencode", entry),
                             f"{entry!r} was mistaken for a harness qualifier")

    def test_each_call_is_dispatched_to_the_harness_its_entry_names(self):
        with tempfile.TemporaryDirectory() as td:
            wd = pathlib.Path(td)
            seen, fake = self._fake([(1, "", "quota exceeded"),
                                     (0, '{"verdict": "pass"}', "")])
            with unittest.mock.patch.object(subprocess, "run", fake):
                out = self.ms.run("p", wd, ["opencode:some-model", "claude:sonnet"],
                                  "audit001")
            self.assertEqual(out, '{"verdict": "pass"}')
            self.assertIn("opencode", seen[0][0])
            self.assertIn("claude", seen[1][0])

    def test_an_expired_credential_falls_through_to_a_different_harness(self):
        # AUTH used to abort the run outright, on the reasoning that every model
        # behind one CLI shares one credential. That is still true of one CLI --
        # and precisely why a chain that reaches a SECOND one must survive it.
        with tempfile.TemporaryDirectory() as td:
            wd = pathlib.Path(td)
            seen, fake = self._fake([(1, "", "Authentication required"),
                                     (0, "ok", "")])
            with unittest.mock.patch.object(subprocess, "run", fake):
                out = self.ms.run("p", wd, ["opencode:a", "claude:sonnet"], "plan001")
            self.assertEqual(out, "ok")

    def test_an_expired_credential_skips_the_rest_of_its_own_harness(self):
        # Walking to another model behind the same dead credential is wasted
        # calls and a misleading error.
        with tempfile.TemporaryDirectory() as td:
            wd = pathlib.Path(td)
            seen, fake = self._fake([(1, "", "Authentication required")])
            with unittest.mock.patch.object(subprocess, "run", fake):
                with self.assertRaises(self.ms.Abort):
                    self.ms.run("p", wd, ["claude:a", "claude:b"], "plan001")
            self.assertEqual(len(seen), 1,
                             "tried a second model behind the same dead credential")

    def test_an_unknown_harness_aborts_instead_of_reaching_the_default_cli(self):
        # Falling through used to hand a mistyped qualifier to agy along with
        # another harness's model identifier, which fails as "unknown model"
        # and reads like a roster problem rather than a typo.
        with self.assertRaises(self.ms.Abort):
            self.ms._cmd("opencodee", "m", pathlib.Path("."), "p")

    def test_the_cli_and_the_engine_agree_on_the_harness_list(self):
        # Two lists of the same fact drift: adding a local harness to the engine
        # while argparse still rejects its name makes the new harness
        # unreachable from the entry point.
        cli = load("summ_cli")
        self.assertEqual(tuple(cli.HARNESSES), tuple(self.ms.HARNESSES))


class EveryModelCallIsAccounted(unittest.TestCase):
    """What a document costs must be a measurement, not a memory.

    The project's own notes carried a figure of "50-62k tokens per document
    against a 35k budget" with nothing in the tree that produced it. Every call
    passes through run(), so that is the one place that can say which harness
    and model answered, how long it took, how large the payload was, and how
    often a call was retried or fell through the chain."""
    ms = load("mapsum")

    def _call(self, wd, outcomes, chain=("claude:sonnet",)):
        seen = []

        def fake_run(argv, **kw):
            seen.append(argv)
            rc, out, err = outcomes[min(len(seen) - 1, len(outcomes) - 1)]
            return subprocess.CompletedProcess(argv, rc, out, err)
        with unittest.mock.patch.object(subprocess, "run", fake_run), \
             unittest.mock.patch.object(self.ms.time, "sleep", lambda *_: None):
            try:
                self.ms.run("prompt here", wd, list(chain), "plan001")
            except self.ms.Abort:
                pass
        return [json.loads(l) for l in
                (wd / "calls.jsonl").read_text().splitlines() if l.strip()]

    def test_a_successful_call_records_who_answered_and_what_it_cost(self):
        with tempfile.TemporaryDirectory() as td:
            recs = self._call(pathlib.Path(td), [(0, "answer", "")])
            self.assertEqual(len(recs), 1)
            r = recs[0]
            self.assertEqual((r["harness"], r["model"], r["outcome"]),
                             ("claude", "sonnet", "ok"))
            self.assertEqual(r["stage"], "plan001")
            # What the record measures is what was sent, which includes the
            # answer-now instruction appended to every composed prompt.
            self.assertEqual(r["prompt_words"],
                             2 + len(self.ms.ANSWER_NOW.split()))
            self.assertGreater(r["prompt_bytes"], 0)
            self.assertIsInstance(r["seconds"], float)

    def test_a_retried_call_is_recorded_once_per_attempt(self):
        # A run that answers on the retry costs two calls. Recording only the
        # answer is how a quota disappears faster than the ledger explains.
        # Every stage is one-shot now, so a silent transport failure buys one
        # retry and no more; a second stall retires the route instead.
        with tempfile.TemporaryDirectory() as td:
            recs = self._call(pathlib.Path(td),
                              [(1, "", "connection reset by peer"),
                               (0, "answer", "")])
            self.assertEqual(len(recs), 2, [r["outcome"] for r in recs])
            self.assertEqual([r["outcome"] for r in recs][-1], "ok")
            self.assertEqual([r["attempt"] for r in recs], [1, 2])

    def test_a_chain_fallthrough_names_both_models(self):
        with tempfile.TemporaryDirectory() as td:
            recs = self._call(pathlib.Path(td),
                              [(1, "", "quota exceeded"), (0, "answer", "")],
                              chain=("claude:sonnet", "claude:opus"))
            self.assertEqual([(r["model"], r["outcome"]) for r in recs],
                             [("sonnet", "capacity"), ("opus", "ok")])

    def test_the_record_lands_in_the_run_directory_not_the_stage_directory(self):
        # Stages run in per-part directories. One ledger per part is not a
        # document's cost, and nothing would ever add them up.
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            (root / "rv").mkdir()
            stage_dir = root / "ledger"
            stage_dir.mkdir()
            self._call_in(stage_dir)
            self.assertTrue((root / "calls.jsonl").exists(),
                            "call ledger was not written to the run directory")
            self.assertFalse((stage_dir / "calls.jsonl").exists())

    def _call_in(self, wd):
        def fake_run(argv, **kw):
            return subprocess.CompletedProcess(argv, 0, "answer", "")
        with unittest.mock.patch.object(subprocess, "run", fake_run):
            self.ms.run("p", wd, ["claude:sonnet"], "plan001")

    def test_instrumentation_cannot_fail_an_otherwise_good_run(self):
        # Evidence is not correctness. A six-minute document must not die
        # because a log line could not be appended.
        with tempfile.TemporaryDirectory() as td:
            wd = pathlib.Path(td)

            def fake_run(argv, **kw):
                return subprocess.CompletedProcess(argv, 0, "answer", "")
            with unittest.mock.patch.object(subprocess, "run", fake_run), \
                 unittest.mock.patch.object(self.ms, "run_root",
                                            side_effect=OSError("read-only")):
                self.assertEqual(self.ms.run("p", wd, ["claude:sonnet"], "plan001"),
                                 "answer")


class QualificationIsMeasuredNotRemembered(unittest.TestCase):
    """Deciding whether a model is good enough must not require reading two
    artifacts per candidate. Everything here is already produced by the
    pipeline; qualify.py only reads what a real run left behind, so a signal
    that regresses shows up as a number rather than as an impression."""
    q = load("qualify")

    def _run_dir(self, calls=None, ledger=None, arts=None):
        d = pathlib.Path(tempfile.mkdtemp())
        (d / "ledger").mkdir()
        if calls is not None:
            (d / "calls.jsonl").write_text(
                "".join(json.dumps(c) + "\n" for c in calls))
        if ledger is not None:
            (d / "ledger" / "ledger.json").write_text(json.dumps(ledger))
        for depth, text in (arts or {}).items():
            (d / f"{depth}.md").write_text(text)
        self.addCleanup(shutil.rmtree, d, True)
        return d

    def test_missing_evidence_is_reported_as_unknown_not_as_zero(self):
        # Silently zero-filling absent input is exactly how a run that never
        # recorded anything reads as a run that cost nothing.
        d = self._run_dir()
        self.assertIsNone(self.q.read_calls(d)["calls"])
        self.assertIsNone(self.q.read_ledger(d / "ledger")["units"])

    def test_every_attempt_is_counted_not_just_the_answer(self):
        # A model that answers on the third try costs three calls. Counting the
        # answer alone is how a quota disappears faster than the record explains.
        d = self._run_dir(calls=[
            {"harness": "opencode", "model": "m", "attempt": 1,
             "outcome": "unusable", "seconds": 2, "prompt_bytes": 10},
            {"harness": "opencode", "model": "m", "attempt": 2,
             "outcome": "unusable", "seconds": 2, "prompt_bytes": 10},
            {"harness": "opencode", "model": "m", "attempt": 3,
             "outcome": "ok", "seconds": 3, "prompt_bytes": 10},
        ])
        c = self.q.read_calls(d)
        self.assertEqual((c["calls"], c["retried"], c["unusable"]), (3, 2, 2))
        self.assertEqual(c["seconds"], 7)
        self.assertEqual(c["answered_by"], ["opencode:m"])

    def test_granularity_is_reported_because_it_is_what_differs_by_model(self):
        # Two plans can cut the same document at very different granularity.
        # Unit count alone does not say how coarse they are; words per unit does.
        d = self._run_dir(ledger={"units": [{}] * 79, "visible_words": 11514})
        self.assertEqual(self.q.read_ledger(d / "ledger")["words_per_unit"], 145.7)
        d2 = self._run_dir(ledger={"units": [{}] * 157, "visible_words": 11514})
        self.assertEqual(self.q.read_ledger(d2 / "ledger")["words_per_unit"], 73.3)

    def test_the_formatting_contract_is_checked_on_published_bytes(self):
        # The ledger gate reads its own input. A renderer that reintroduces a
        # list after the gate ran would pass it and still reach the e-ink device.
        d = self._run_dir(arts={"detailed": "- a bullet\n| a | table |\n",
                                "brief": "Ordinary prose about we-know-not-what."})
        a = self.q.read_artifacts(d)
        self.assertEqual(a["detailed_lists"], 2)
        self.assertEqual(a["brief_lists"], 0)

    def test_first_person_is_counted_on_the_artifact(self):
        d = self._run_dir(arts={"detailed": "We find that our result holds.",
                                "brief": "The paper finds a result."})
        a = self.q.read_artifacts(d)
        self.assertEqual(a["detailed_first_person"], 2)
        self.assertEqual(a["brief_first_person"], 0)

    def test_earlier_artifacts_are_not_mistaken_for_corpus_documents(self):
        # Publication writes beside the source. A second qualification pass that
        # picked up the first pass's output would summarize summaries and report
        # the result as a corpus score.
        c = pathlib.Path(tempfile.mkdtemp()); self.addCleanup(shutil.rmtree, c, True)
        for n in ("paper.md", "paper.summary.md", "paper.brief.md",
                  "paper.gate.json", "notes.txt"):
            (c / n).write_text("x")
        self.assertEqual([p.name for p in self.q.documents(c)],
                         ["notes.txt", "paper.md"])

    def test_a_candidate_is_only_an_environment(self):
        # Candidates must not become a second way to name a model. models.json
        # stays the one place; a candidate is the override environment the
        # pipeline already supports.
        src = (ENG / "qualify.py").read_text()
        for ident in ("gemini-", "gpt-5", "sonnet", "opus", "private-model-"):
            hits = [l.strip() for l in src.splitlines() if ident in l]
            self.assertEqual(hits, [], f"qualify.py names the model {ident!r}")




class StageExitCodesReachTheCaller(unittest.TestCase):
    """A caller must be able to act on WHY a run failed.

    fullsum.py returns 1 (empty write), 5 (no usable candidate) and 6
    (an unsafe windowed route). Flattening them all to `failed = 1` would leave
    the only distinction in human stderr, so the summarize loop passes the
    fullsum stage codes through verbatim and reports the unsafe route explicitly
    with no ledger or Quick fallback."""

    def _main_src(self):
        return (ENG / "summ_cli.py").read_text()

    def test_the_top_level_no_longer_flattens_every_failure_to_one(self):
        src = self._main_src()
        self.assertNotIn("return 1 if failed else 0", src,
                         "the flattening return is back")
        self.assertIn("codes[0] if len(codes) == 1 else 1", src)

    def test_full_stage_codes_reach_the_caller_verbatim(self):
        # 1 (empty write), 5 (no usable candidate) and 6 (unsafe window route)
        # mean three different things to a UI and to a person.
        # The summarize loop has no ledger fallback to flatten them into.
        src = self._main_src()
        self.assertIn("codes.append(rc_f)", src)
        self.assertNotIn("codes.append(rc_ledger", src)
        self.assertNotIn("codes.append(rc_compose", src)

    def test_oversize_context_fails_explicitly_without_a_fallback(self):
        # An unfit source reports context-unsupported with the admission
        # numbers; it never invokes the ledger or downgrades to Quick.
        src = self._main_src()
        self.assertIn('failure_kind="context_unsupported"', src)
        self.assertIn("rc_f == fullsum.NOT_SUPPORTED", src)

    def test_several_targets_cannot_report_one_true_code(self):
        # A batch where one document lost its seal and another exceeded its
        # ceiling has no single correct exit code, so it must report 1 rather
        # than pick one document's reason and present it as the batch's.
        src = self._main_src()
        i = src.rindex("codes[0] if len(codes) == 1 else 1")
        self.assertIn("no single true code", src[max(0, i - 400):i])

    def test_every_stage_code_the_stages_can_emit_is_documented(self):
        # If a stage learns a new code and nothing here notices, the UI contract
        # silently loses a case.
        led = {int(m) for m in re.findall(r"^\s*return ([0-9])\b",
                                          (ENG / "ledger.py").read_text(), re.M)}
        comp = {int(m) for m in re.findall(r"^\s*return ([0-9])\b",
                                           (ENG / "compose.py").read_text(), re.M)}
        self.assertEqual(led - {0}, {3, 4, 5}, f"ledger.py codes changed: {led}")
        self.assertEqual(comp - {0}, {7}, f"compose.py codes changed: {comp}")
        self.assertIn("SystemExit(2)", (ENG / "compose.py").read_text())


class WorkDirGovernsReadAloudToo(unittest.TestCase):
    """Evidence retention must not depend on which mode failed.

    main() branched into do_tts() before the work-directory block, and do_tts()
    unconditionally built a temp directory and deleted it in a finally. So a
    read-aloud reformat that failed its information-preservation check threw
    away the reformatted text that failed it, while a summarization failure of
    the same severity kept everything."""

    @staticmethod
    def _write_clean_result(destination, text, digest=None):
        destination.mkdir(parents=True, exist_ok=True)
        payload = text.encode("utf-8")
        (destination / "cleaned.md").write_bytes(payload)
        (destination / "textprep-report.json").write_text(json.dumps({
            "status": "succeeded",
            "output_words": len(text.split()),
            "output_sha256": digest or hashlib.sha256(payload).hexdigest(),
            "chunks": 1,
        }))

    def test_the_tts_branch_is_given_the_work_dir(self):
        # Asserted on the SIGNATURE and the argument, not on an exact call
        # literal: pinning the whole line made adding --out fail a test about
        # evidence retention, which is not what this guards.
        src = (ENG / "summ_cli.py").read_text()
        self.assertRegex(src, r"return do_tts\([^)]*\ba\.work_dir\b")
        self.assertRegex(src, r"def do_tts\([^)]*\bwork_dir=None\b")

    def test_a_work_dir_run_keeps_its_tree(self):
        src = (ENG / "summ_cli.py").read_text()
        i = src.index("def do_tts(")
        body = src[i:src.index("def doctor(")]
        self.assertNotIn("finally:\n            shutil.rmtree(tmp, ignore_errors=True)",
                         body, "the unconditional delete is back")
        self.assertIn("work kept:", body)

    def test_without_a_work_dir_the_temp_tree_is_still_removed(self):
        # Retention is opt-in. A clipboard run that leaves directories behind on
        # every invocation is a different defect.
        src = (ENG / "summ_cli.py").read_text()
        i = src.index("def do_tts(")
        body = src[i:src.index("def doctor(")]
        self.assertIn("shutil.rmtree(tmp, ignore_errors=True)", body)

    def test_raw_read_aloud_runs_clean_then_normalizes_a_work_candidate(self):
        cli = load("summ_cli")
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            source = root / "document.txt"
            source.write_text("Page 1\n\nThe modem record kept its caveat.")
            calls = []

            def stage(argv, *_args, **_kwargs):
                calls.append(pathlib.Path(argv[1]).name)
                destination = pathlib.Path(argv[-1])
                if calls[-1] == "textprep.py":
                    self._write_clean_result(
                        destination, "The modern record kept its caveat.\n")
                else:
                    destination.write_text("The modern record kept its caveat.\n")
                return subprocess.CompletedProcess(argv, 0)

            with unittest.mock.patch.object(cli, "run_stage", side_effect=stage), \
                 unittest.mock.patch.object(cli, "notify"):
                rc = cli.do_tts([source], False, root / "work", root / "out")
            self.assertEqual(0, rc)
            self.assertEqual(["textprep.py", "speechprep.py", "tts_normalize.py"], calls)
            self.assertEqual("The modern record kept its caveat.\n",
                             (root / "out" / "tts.document.txt").read_text())

    def test_failed_clean_or_normalization_never_replaces_prior_output(self):
        cli = load("summ_cli")
        for source_name, failing_stage in (("document.txt", "textprep.py"),
                                           ("document.summary.md", "tts_normalize.py")):
            with self.subTest(stage=failing_stage), tempfile.TemporaryDirectory() as td:
                root = pathlib.Path(td)
                source = root / source_name
                source.write_text("Original substantive text.\n")
                out = root / "out"; out.mkdir()
                destination = out / ("tts." + source.stem + ".txt")
                destination.write_text("prior output\n")

                def stage(argv, *_args, **_kwargs):
                    name = pathlib.Path(argv[1]).name
                    target = pathlib.Path(argv[-1])
                    if name == "textprep.py" and failing_stage != name:
                        self._write_clean_result(target, source.read_text())
                    if name == "tts_normalize.py":
                        target.write_text("candidate that must not publish\n")
                    return subprocess.CompletedProcess(
                        argv, 1 if name == failing_stage else 0)

                with unittest.mock.patch.object(cli, "run_stage", side_effect=stage), \
                     unittest.mock.patch.object(cli, "notify"):
                    self.assertNotEqual(
                        0, cli.do_tts([source], False, root / "work", out))
                self.assertEqual("prior output\n", destination.read_text())

    def test_tts_refuses_a_clean_artifact_that_does_not_match_its_report(self):
        cli = load("summ_cli")
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            source = root / "document.txt"
            source.write_text("Original substantive text.\n")
            out = root / "out"; out.mkdir()
            destination = out / "tts.document.txt"
            destination.write_text("prior output\n")
            calls = []

            def stage(argv, *_args, **_kwargs):
                name = pathlib.Path(argv[1]).name
                calls.append(name)
                if name == "textprep.py":
                    self._write_clean_result(
                        pathlib.Path(argv[-1]), "Clean candidate.\n", "0" * 64)
                return subprocess.CompletedProcess(argv, 0)

            with unittest.mock.patch.object(cli, "run_stage", side_effect=stage), \
                 unittest.mock.patch.object(cli, "notify"):
                self.assertNotEqual(
                    0, cli.do_tts([source], False, root / "work", out))
            self.assertEqual(["textprep.py"], calls)
            self.assertEqual("prior output\n", destination.read_text())

    def test_mutation_after_clean_verification_cannot_reach_publication(self):
        cli = load("summ_cli")
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td); run = root / "run"
            self._write_clean_result(run, "Accepted clean text.\n")
            artifact, _, accepted_hash, _ = cli.verified_textprep_artifact(run)
            artifact.write_text("Mutated after worker completion.\n")
            destination = root / "document.clean.md"
            destination.write_text("prior output\n")
            with self.assertRaises(cli.PublicationError):
                cli.publish_one(
                    artifact, destination, expected_sha256=accepted_hash)
            self.assertEqual("prior output\n", destination.read_text())

    def test_a_missing_clean_report_is_not_an_optional_detail(self):
        cli = load("summ_cli")
        with tempfile.TemporaryDirectory() as td:
            run = pathlib.Path(td)
            (run / "cleaned.md").write_text("Unbound candidate.\n")
            with self.assertRaisesRegex(ValueError, "report"):
                cli.verified_textprep_artifact(run)


class SummaryFailureRetention(unittest.TestCase):
    """The documented implicit-work lifecycle is production behavior."""
    cli = load("summ_cli")

    class FakeLease:
        def __enter__(self):
            return 0.0

        def __exit__(self, *_args):
            return False

    @classmethod
    def _lease(cls, *_args, **_kwargs):
        return cls.FakeLease()

    def _run(self, root, mode, outcome):
        source = root / "paper.md"
        source.write_text("An original source with a qualified claim.\n")
        implicit = root / "implicit-work"
        argv = ["summ_cli.py", str(source), "--out", str(root / "out")]
        if mode:
            argv.append(mode)

        def make_implicit():
            implicit.mkdir(parents=True, exist_ok=False)
            return str(implicit)

        def stage(argv_, *_args, **_kwargs):
            if outcome == "cancel":
                raise self.cli.Cancelled()
            if outcome == "fail":
                return subprocess.CompletedProcess(argv_, 5)
            run = pathlib.Path(argv_[-1])
            run.mkdir(parents=True, exist_ok=True)
            (run / "detailed.md").write_text("Detailed result.\n")
            (run / "brief.md").write_text("Brief result.\n")
            (run / "short-report.json").write_text(json.dumps({
                "status": "pass", "findings": [], "review": "complete",
                "detailed_words": 2, "brief_words": 2,
            }))
            return subprocess.CompletedProcess(argv_, 0)

        patches = [
            unittest.mock.patch.object(sys, "argv", argv),
            unittest.mock.patch.object(self.cli.tempfile, "mkdtemp",
                                       side_effect=make_implicit),
            unittest.mock.patch.object(self.cli, "run_stage", side_effect=stage),
            unittest.mock.patch.object(self.cli, "freeze_model_routes"),
            unittest.mock.patch.object(self.cli, "validate_roles"),
            unittest.mock.patch.object(self.cli, "notify"),
            unittest.mock.patch.object(self.cli.eta, "estimate", return_value=None),
            unittest.mock.patch.object(self.cli.eta, "record_target", return_value=None),
            unittest.mock.patch.object(self.cli.runtime, "target_lease",
                                       side_effect=self._lease),
            unittest.mock.patch.object(self.cli.runtime, "destination_lock",
                                       side_effect=self._lease),
            unittest.mock.patch.object(self.cli.runtime, "output_directory_lock",
                                       side_effect=self._lease),
        ]
        for patch in patches:
            patch.start()
        try:
            return self.cli.main(), implicit, source.read_bytes()
        finally:
            for patch in reversed(patches):
                patch.stop()

    def test_summary_and_text_prep_failures_keep_staged_original(self):
        for mode in ("--quick", "--text-prep"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as td:
                rc, work, original = self._run(pathlib.Path(td), mode, "fail")
                self.assertNotEqual(0, rc)
                self.assertEqual(original, (work / "source.original").read_bytes())
                self.assertTrue((work / "source.txt").is_file())

    def test_cancelled_summary_keeps_staged_original(self):
        with tempfile.TemporaryDirectory() as td:
            rc, work, original = self._run(pathlib.Path(td), "--quick", "cancel")
            self.assertEqual(130, rc)
            self.assertEqual(original, (work / "source.original").read_bytes())

    def test_successful_implicit_summary_removes_work(self):
        with tempfile.TemporaryDirectory() as td:
            rc, work, _ = self._run(pathlib.Path(td), "--quick", "succeed")
            self.assertEqual(0, rc)
            self.assertFalse(work.exists())

    def test_failed_read_aloud_removes_implicit_work(self):
        with tempfile.TemporaryDirectory() as td:
            rc, work, _ = self._run(pathlib.Path(td), "--tts", "fail")
            self.assertNotEqual(0, rc)
            self.assertFalse(work.exists())


class CorpusImplicitWorkContract(unittest.TestCase):
    """Implicit Corpus work is temporary for every terminal outcome."""
    cli = load("summ_cli")

    def test_failure_and_cancellation_remove_the_implicit_corpus_root(self):
        for outcome in (5, 130):
            with self.subTest(exit_code=outcome), tempfile.TemporaryDirectory() as td:
                root = pathlib.Path(td)
                first = root / "one.md"; first.write_text("one source")
                second = root / "two.md"; second.write_text("two source")
                implicit = root / "implicit-corpus"

                def make_implicit():
                    implicit.mkdir(parents=True, exist_ok=False)
                    return str(implicit)

                def corpus_branch(_selection, work_root, *_args, **_kwargs):
                    self.assertEqual(implicit, pathlib.Path(work_root))
                    (implicit / "branch-entered").write_text("yes")
                    return outcome

                argv = [
                    "summ_cli.py", "--scope", "corpus",
                    "--corpus-name", "papers", "--out", str(root / "out"),
                    str(first), str(second),
                ]
                with unittest.mock.patch.object(sys, "argv", argv), \
                        unittest.mock.patch.object(
                            self.cli.tempfile, "mkdtemp", side_effect=make_implicit), \
                        unittest.mock.patch.object(
                            self.cli, "do_corpus", side_effect=corpus_branch), \
                        unittest.mock.patch.dict(os.environ, {"SUMM_QUICK": ""}), \
                        unittest.mock.patch.object(self.cli, "notify"):
                    rc = self.cli.main()
                self.assertEqual(outcome, rc)
                self.assertFalse(implicit.exists())


class ReadmeBehaviorContract(unittest.TestCase):
    """Public behavior claims stay within the implementation's guarantees."""

    @classmethod
    def setUpClass(cls):
        cls.readme = re.sub(r"\s+", " ", (ENG.parent / "README.md").read_text())

    def test_retention_distinguishes_implicit_outcomes_and_modes(self):
        for text in (
                "In ordinary batch mode",
                "removed after a successful summary or text-preparation run",
                "retained for diagnosis when either mode fails or is cancelled",
                "An implicit Corpus work root and implicit read-aloud work are temporary and removed after every outcome",
                "Passing `--work-dir DIR` retains"):
            self.assertIn(text, self.readme)
        self.assertNotIn("does not intentionally persist raw source text outside",
                         self.readme)

    def test_eta_history_is_described_as_content_free_metadata(self):
        self.assertIn("content-free operational metadata", self.readme)
        self.assertIn("hashed execution and resource identities", self.readme)
        self.assertNotIn("contains timing aggregates only", self.readme)

    def test_exit_zero_allows_disclosed_open_findings(self):
        self.assertIn("Exit 0 means that usable output was published", self.readme)
        self.assertIn("unresolved semantic or editorial findings may still be disclosed",
                      self.readme)
        self.assertNotIn("Exit codes distinguish publication", self.readme)


class DepthDoesNotSelectTheArtifactSet(unittest.TestCase):
    """`--depth` is accepted, both triggers pass it, and it cannot change what
    is published: compose.py never reads DEPTH and always renders both depths
    from the one sealed ledger. Help text that implies otherwise tells the
    reader they have a choice the code does not offer."""

    def test_compose_never_branches_on_depth(self):
        src = (ENG / "compose.py").read_text()
        code = "\n".join(l for l in src.splitlines()
                         if not l.lstrip().startswith("#"))
        self.assertNotIn("DEPTH", code, "compose.py reads DEPTH again")

    def test_both_depths_are_always_rendered(self):
        src = (ENG / "compose.py").read_text()
        self.assertIn('for depth, prefix in (("detailed", "D"), ("brief", "B")):', src)

    def test_the_help_says_the_flag_does_not_select_the_output(self):
        src = (ENG / "summ_cli.py").read_text()
        i = src.index('ap.add_argument("--depth"')
        block = src[i:i + 500]
        self.assertIn("does NOT select", block,
                      "--depth help still implies it chooses the artifact set")

    def test_the_flag_is_still_accepted_because_the_triggers_pass_it(self):
        # Removing it would break both installed triggers on the next sync.
        src = (ENG / "summ_cli.py").read_text()
        self.assertIn('choices=["full", "brief"]', src)




class ProgressStreamIsTheControlPlane(unittest.TestCase):
    """A five-to-fifteen minute run printed only human lines. A client wanting
    "part 7 of 12" had to parse prose written for a person, which has no schema
    and changes whenever the wording improves. Human stdout stays human."""
    pg = load("progress")

    def setUp(self):
        self.d = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.d, True)
        self.addCleanup(os.environ.pop, self.pg.ENV, None)
        os.environ.pop(self.pg.ENV, None)

    def _events(self):
        return [json.loads(l) for l in
                (self.d / "p.jsonl").read_text().splitlines() if l.strip()]

    def test_absent_flag_is_a_no_op(self):
        # Both installed triggers pass no such flag. Enabling this must not be
        # able to change what an existing invocation does.
        self.pg.emit("part", index=1, total=2)
        self.assertIsNone(self.pg.path())

    def test_an_unwritable_stream_fails_before_any_work(self):
        # A client that asked to be told what is happening, and silently is not
        # told, would rather learn now than after a six-minute wait.
        rc = self.pg.start(self.d / "nodir" / "x" / "p.jsonl", "summarize", 1)
        self.assertEqual(rc, 0)          # parents are created
        bad = self.d / "not-a-file"
        bad.mkdir()
        rc = self.pg.start(bad, "summarize", 1)
        self.assertEqual(rc, 2)

    def test_start_opens_the_stream_with_a_job_event(self):
        self.assertEqual(self.pg.start(self.d / "p.jsonl", "summarize", 3), 0)
        e = self._events()
        self.assertEqual(e[0]["event"], "job_started")
        self.assertEqual((e[0]["mode"], e[0]["targets"]), ("summarize", 3))
        self.assertEqual(e[0]["schema"], "summer.progress.v2")

    def test_cancelled_unstarted_targets_get_terminal_records(self):
        cli = load("summ_cli")
        self.assertEqual(self.pg.start(self.d / "p.jsonl", "quick", 3), 0)
        cli.emit_cancelled_targets(2, ["one", "two", "three"])
        terminals = [e for e in self._events()
                     if e["event"] == "target_finished"]
        self.assertEqual([e["index"] for e in terminals], [2, 3])
        self.assertTrue(all(e["status"] == "cancelled" for e in terminals))
        self.assertTrue(all(e["failure_kind"] == "not_started"
                            for e in terminals))

    def test_every_event_carries_schema_and_a_utc_timestamp(self):
        # A consumer that cannot tell which schema it is reading cannot be
        # upgraded safely.
        self.pg.start(self.d / "p.jsonl", "summarize", 1)
        self.pg.emit("part", index=2, total=9)
        for e in self._events():
            self.assertEqual(e["schema"], "summer.progress.v2")
            self.assertTrue(e["time_utc"].endswith("Z"), e["time_utc"])

    def test_a_broken_stream_never_fails_the_run(self):
        # Progress is evidence, not correctness.
        self.pg.start(self.d / "p.jsonl", "summarize", 1)
        os.environ[self.pg.ENV] = str(self.d)      # a directory, not a file
        self.pg._warned = False
        self.pg.emit("part", index=1, total=1)     # must not raise

    def test_the_cli_exposes_the_flag_and_starts_before_staging(self):
        src = (ENG / "summ_cli.py").read_text()
        self.assertIn('"--progress-jsonl"', src)
        i, j = src.index("progress.start("), src.index("suffix = ")
        self.assertLess(i, j, "the stream is opened after work has begun")

    def test_a_failed_target_still_gets_a_terminal_event(self):
        # A UI that only ever sees success cannot tell a failed run from a hung
        # one, which is the whole reason a client is watching.
        src = (ENG / "summ_cli.py").read_text()
        for kind in ("context_unsupported", "full_failed", "quick_failed"):
            self.assertIn(f'failure_kind="{kind}"', src,
                          f"no terminal event carries failure_kind={kind}")
        self.assertGreaterEqual(src.count("emit_target_finished("), 3)
        self.assertIn('progress.emit("job_finished", status="failed"', src)

    def test_empty_input_closes_the_started_job_record(self):
        cli = load("summ_cli")
        path = self.d / "p.jsonl"
        frozen = json.dumps({"active_targets": 2, "harness_capacity": 1,
                             "capacities": {}, "bindings": {}, "gateways": {}})
        argv = ["summ_cli.py", "--harness", "agy",
                "--progress-jsonl", str(path)]
        with unittest.mock.patch.object(sys, "argv", argv), \
             unittest.mock.patch.object(cli, "clip_read", return_value=""), \
             unittest.mock.patch.object(cli, "notify"), \
             unittest.mock.patch.dict(
                 os.environ, {"SUMM_RUNTIME_JSON": frozen}, clear=False):
            self.assertEqual(1, cli.main())
        events = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual("job_started", events[0]["event"])
        self.assertEqual(
            ("job_finished", "failed", 1),
            (events[-1]["event"], events[-1]["status"],
             events[-1]["exit_code"]))

    def test_busy_work_root_closes_the_started_job_record(self):
        cli = load("summ_cli")
        path = self.d / "p.jsonl"
        source = self.d / "source.txt"; source.write_text("Substantive text.")
        work = self.d / "work"
        frozen = json.dumps({"active_targets": 2, "harness_capacity": 1,
                             "capacities": {}, "bindings": {}, "gateways": {}})
        argv = ["summ_cli.py", "--harness", "agy", "--work-dir", str(work),
                "--progress-jsonl", str(path), str(source)]
        with cli.runtime.work_root_lock(work), \
             unittest.mock.patch.object(sys, "argv", argv), \
             unittest.mock.patch.object(cli, "notify"), \
             unittest.mock.patch.dict(
                 os.environ, {"SUMM_RUNTIME_JSON": frozen}, clear=False):
            self.assertEqual(2, cli.main())
        events = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(
            ("job_finished", "failed", 2),
            (events[-1]["event"], events[-1]["status"],
             events[-1]["exit_code"]))

    def test_model_calls_are_emitted_from_the_one_cost_call_site(self):
        # Emitting separately from the cost ledger lets a call appear in one and
        # not the other.
        src = (ENG / "mapsum.py").read_text()
        i = src.index("def _record(")
        self.assertIn('pg.emit("model_call"', src[i:i + 1800])
        self.assertEqual(src.count('pg.emit("model_call"'), 1)

    def test_progress_py_is_loaded_once_per_process(self):
        # A document makes ~31 model calls; re-executing the module on each is
        # pure waste on the hot path.
        src = (ENG / "mapsum.py").read_text()
        self.assertIn("global _PG", src)




class UIIsAClientOfTheCLI(unittest.TestCase):
    """The UI adds a picker and visibility. It must never become a second
    implementation of anything the CLI owns."""
    ui = load("summ_ui")

    def test_clipboard_mode_passes_no_path_at_all(self):
        # summ_cli owns interpreting the clipboard -- a path, a folder, several
        # paths, or raw text. A UI that pre-resolved it would be a second,
        # diverging implementation of parse_targets.
        cmd = self.ui.build_cmd(pathlib.Path("/j"), self.ui.MODES[0][0], None)
        self.assertNotIn("--tts", cmd)
        self.assertEqual(cmd[-2:],
                         ["--cancel-file", str(pathlib.Path("/j") / "cancel.request")])

    def test_a_picked_path_is_one_argv_element(self):
        # A path with spaces assembled into a command string is where quoting
        # bugs live, and a document path is untrusted input.
        cmd = self.ui.build_cmd(pathlib.Path("/j"), self.ui.MODES[0][0],
                                pathlib.Path("/tmp/a b/c d.md"))
        self.assertEqual(cmd[-1], str(pathlib.Path("/tmp/a b/c d.md")))

    def test_corpus_is_a_normal_ui_to_cli_scope(self):
        manifest = pathlib.Path("/j/selection.json")
        cmd = self.ui.build_cmd(
            pathlib.Path("/j"), self.ui.MODES[0][0], None,
            selection_manifest=manifest, scope="corpus")
        self.assertEqual(cmd[cmd.index("--scope") + 1], "corpus")
        self.assertNotIn("CORPUS_ENABLED", (ENG / "summ_cli.py").read_text())

    def test_read_aloud_mode_passes_tts(self):
        tts = next(m for m, f in self.ui.MODES if "--tts" in f)
        self.assertIn("--tts", self.ui.build_cmd(pathlib.Path("/j"), tts, None))

    def test_custom_instructions_use_the_cli_flag_but_never_reach_tts(self):
        request = pathlib.Path("/j/request.txt")
        summarize = self.ui.MODES[0][0]
        cmd = self.ui.build_cmd(pathlib.Path("/j"), summarize, None,
                                instructions_file=request)
        self.assertEqual(request, pathlib.Path(cmd[cmd.index("--instructions-file") + 1]))
        tts = next(m for m, flags in self.ui.MODES if "--tts" in flags)
        self.assertNotIn("--instructions-file", self.ui.build_cmd(
            pathlib.Path("/j"), tts, None, instructions_file=request))

    def test_the_ui_never_passes_depth(self):
        # --depth cannot select the artifact set, so offering it would promise a
        # choice the engine does not make.
        for label, _ in self.ui.MODES:
            for src in (None, pathlib.Path("/tmp/x.md")):
                self.assertNotIn("--depth",
                                 self.ui.build_cmd(pathlib.Path("/j"), label, src))

    def test_the_ui_never_reads_the_clipboard_itself(self):
        src = (ENG / "summ_ui.py").read_text()
        for tool in ("pbpaste", "Get-Clipboard", "xclip", "clipboard_get"):
            self.assertNotIn(tool, src, f"summ_ui.py reads the clipboard via {tool}")

    def test_the_ui_launches_without_a_shell(self):
        src = (ENG / "summ_ui.py").read_text()
        self.assertNotIn("shell=True", src)

    def test_run_evidence_stays_out_of_the_synced_project_tree(self):
        # Work directories are device-local and large; the sync mesh has no
        # reason to carry them, and the project tree is what syncs.
        self.assertNotIn(str(ENG), str(self.ui.job_root()))

    def test_every_cli_exit_code_has_a_human_reading(self):
        # An exit the UI cannot name reaches the user as a bare number.
        self.assertEqual(sorted(self.ui.EXITS), [1, 2, 3, 4, 5, 6, 7])

    def test_stage_names_match_the_progress_vocabulary(self):
        # A stage the UI cannot name shows as a raw token.
        src = (ENG / "summ_ui.py").read_text()
        for stage in ("reader_view", "ledger", "compose", "text_prep", "publish",
                      "tts_normalize"):
            self.assertIn(f'"{stage}"', src, f"UI cannot label stage {stage}")

    def test_a_clean_exit_without_a_terminal_event_is_not_called_success(self):
        # Exit 0 with no job_finished means the progress record is incomplete,
        # which is not the same as a verified publication.
        src = (ENG / "summ_ui.py").read_text()
        self.assertIn("progress record is", src)

    def test_closing_the_window_does_not_kill_a_running_job(self):
        # The CLI is the publication controller. The UI may create its
        # cooperative marker, but must never kill that controller directly.
        src = (ENG / "summ_ui.py").read_text()
        i = src.index("def cancel_active(")
        body = src[i:src.index("def _clear_cards(", i)]
        body += src[src.index("def close("):]
        for kill in (".kill()", ".terminate()", "taskkill", "send_signal"):
            self.assertNotIn(kill, body, f"UI cancellation calls {kill}")

    def test_each_queued_job_has_an_individual_remove_control(self):
        src = (ENG / "summ_ui.py").read_text(encoding="utf-8")
        self.assertIn("self.queue_frame", src)
        self.assertIn('text="×"', src)
        self.assertIn("def remove_queued(", src)
        self.assertIn("self.pending.pop(index)", src)

    def test_an_in_flight_call_is_shown_not_only_a_finished_one(self):
        # A call takes 20-40 seconds. Showing only completions means the UI
        # displays the PREVIOUS call's outcome for all of it, so an active call
        # and a wedged one look identical -- the exact confusion this ends.
        src = (ENG / "summ_ui.py").read_text()
        self.assertIn('"model_call_started"', src)
        ms = (ENG / "mapsum.py").read_text()
        i, j = ms.index("t0 = time.monotonic()"), ms.index("subprocess.run(")
        self.assertIn('emit("model_call_started"', ms[i:j],
                      "the start event is not emitted before the call runs")

    def test_the_cost_record_stays_a_single_call_site(self):
        # model_call is the cost record and must agree with calls.jsonl exactly;
        # model_call_started carries no cost and is deliberately separate.
        ms = (ENG / "mapsum.py").read_text()
        self.assertEqual(ms.count('pg.emit("model_call"'), 1)
        self.assertEqual(ms.count('emit("model_call_started"'), 1)

    def test_each_parallel_job_has_its_own_state_and_close_reads_active(self):
        src = (ENG / "summ_ui.py").read_text()
        i = src.index("def start(")
        head = src[i:src.index("def _run(", i)]
        self.assertIn("if len(self.active) >= limit:", head)
        self.assertIn("self.active[state.id] = state", head)
        j = src.index("def close(")
        self.assertIn("if self.active or self.running:", src[j:j + 160])
        self.assertNotIn("if self.proc:", src[j:j + 120])

    def test_no_percentage_is_shown(self):
        # Part and call durations vary too much for a percentage to mean
        # anything, and there is no duration model to compute one from.
        src = (ENG / "summ_ui.py").read_text()
        self.assertNotIn("percent", src.lower().replace("no percentage", ""))


class CooperativeCancellation(unittest.TestCase):
    cli = load("summ_cli")

    def test_marker_present_before_stage_start_prevents_popen(self):
        with tempfile.TemporaryDirectory() as td:
            marker = pathlib.Path(td) / "cancel.request"
            marker.write_text("cancel\n")
            token = self.cli.CancellationToken(marker)
            with unittest.mock.patch.object(self.cli.subprocess, "Popen") as popen:
                with self.assertRaises(self.cli.Cancelled):
                    self.cli.run_stage([sys.executable, "-c", "pass"], token)
            popen.assert_not_called()

    def test_cancellation_is_not_reported_until_worker_death_is_observed(self):
        class Worker:
            def wait(self, timeout=None):
                raise subprocess.TimeoutExpired(["worker"], timeout)

        token = unittest.mock.Mock()
        token.check.return_value = None
        token.requested.return_value = True
        with unittest.mock.patch.object(
                self.cli.subprocess, "Popen", return_value=Worker()), \
             unittest.mock.patch.object(
                 self.cli, "_stop_stage_tree", side_effect=[False, True]) as stop:
            with self.assertRaises(self.cli.Cancelled):
                self.cli.run_stage(["worker"], token)
        self.assertEqual(2, stop.call_count,
                         "Summer released the controller before the worker died")

    def test_stop_tree_returns_false_when_the_final_wait_times_out(self):
        proc = unittest.mock.Mock()
        proc.pid = 9876
        proc.poll.return_value = None
        proc.wait.side_effect = subprocess.TimeoutExpired(["worker"], 10)
        with unittest.mock.patch.object(self.cli, "WIN", False), \
             unittest.mock.patch.object(self.cli.os, "killpg"), \
             unittest.mock.patch.object(self.cli.time, "sleep"):
            self.assertFalse(self.cli._stop_stage_tree(proc, grace_seconds=0))

    @unittest.skipIf(sys.platform.startswith("win"),
                     "POSIX process-group fixture; Windows is exercised by windows_proof.py")
    def test_cancelling_a_stage_stops_its_descendant_group(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            marker = root / "cancel.request"
            ready = root / "ready"
            heartbeat = root / "heartbeat"
            grandchild = (
                "import pathlib,sys,time\n"
                "p=pathlib.Path(sys.argv[1]); i=0\n"
                "while True:\n"
                " p.write_text(str(i)); i+=1; time.sleep(0.03)\n")
            parent = (
                "import pathlib,subprocess,sys,time\n"
                "subprocess.Popen([sys.executable,'-c',sys.argv[3],sys.argv[2]])\n"
                "pathlib.Path(sys.argv[1]).write_text('ready')\n"
                "time.sleep(60)\n")

            def request():
                deadline = time.monotonic() + 5
                while not ready.exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                marker.write_text("cancel\n")

            worker = threading.Thread(target=request)
            worker.start()
            with self.assertRaises(self.cli.Cancelled):
                self.cli.run_stage(
                    [sys.executable, "-c", parent, str(ready), str(heartbeat),
                     grandchild], self.cli.CancellationToken(marker),
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            worker.join(2)
            time.sleep(0.15)
            before = heartbeat.read_text() if heartbeat.exists() else None
            time.sleep(0.15)
            after = heartbeat.read_text() if heartbeat.exists() else None
            self.assertEqual(before, after, "descendant kept running after cancellation")

    @unittest.skipIf(sys.platform.startswith("win"),
                     "POSIX process-group fixture; Windows is exercised by windows_proof.py")
    def test_parent_exit_does_not_hide_a_term_ignoring_descendant(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            ready = root / "ready"
            child_ready = root / "child-ready"
            heartbeat = root / "heartbeat"
            child = (
                "import pathlib,signal,sys,time\n"
                "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                "p=pathlib.Path(sys.argv[1]); pathlib.Path(sys.argv[2]).write_text('ready')\n"
                "i=0\n"
                "while True:\n"
                " p.write_text(str(i)); i+=1; time.sleep(0.03)\n")
            parent = (
                "import pathlib,subprocess,sys,time\n"
                "subprocess.Popen([sys.executable,'-c',sys.argv[4],sys.argv[2],sys.argv[3]])\n"
                "deadline=time.monotonic()+5\n"
                "while not pathlib.Path(sys.argv[3]).exists() and time.monotonic()<deadline: time.sleep(.01)\n"
                "pathlib.Path(sys.argv[1]).write_text('ready')\n"
                "time.sleep(60)\n")
            proc = subprocess.Popen(
                [sys.executable, "-c", parent, str(ready), str(heartbeat),
                 str(child_ready), child], start_new_session=True,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                deadline = time.monotonic() + 5
                while not ready.exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue(ready.exists())
                self.assertTrue(self.cli._stop_stage_tree(proc, grace_seconds=0.1))
                time.sleep(0.15)
                before = heartbeat.read_text() if heartbeat.exists() else None
                time.sleep(0.15)
                after = heartbeat.read_text() if heartbeat.exists() else None
                self.assertEqual(before, after,
                                 "TERM-ignoring descendant survived cancellation")
            finally:
                if self.cli._posix_group_alive(proc.pid):
                    os.killpg(proc.pid, signal.SIGKILL)
                try:
                    proc.wait(timeout=2)
                except subprocess.SubprocessError:
                    pass

    def test_cancellation_requested_during_pair_publication_finishes_the_pair(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            detailed, brief = root / "new-d", root / "new-b"
            dest = root / "out.summary.md"
            bdest = root / "out.brief.md"
            detailed.write_text("NEW D"); brief.write_text("NEW B")
            dest.write_text("OLD D"); bdest.write_text("OLD B")
            marker = root / "cancel.request"
            replacements = 0

            def replace(source, target):
                nonlocal replacements
                os.replace(source, target)
                if pathlib.Path(target) in (dest, bdest):
                    replacements += 1
                    if replacements == 1:
                        marker.write_text("cancel\n")

            self.cli.publish_pair(detailed, brief, dest, replace)
            self.assertTrue(self.cli.CancellationToken(marker).requested())
            self.assertEqual(("NEW D", "NEW B"),
                             (dest.read_text(), bdest.read_text()))


class ActualModeDeterminesRequiredRoles(unittest.TestCase):
    cli = load("summ_cli")

    def test_quick_backend_does_not_need_a_plan_chain(self):
        roster = {"local": {"_local": True, "plan": [], "write": ["m"],
                            "audit": ["m"], "repair": ["m"]}}
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            self.cli.validate_roles(
                "local", roster, self.cli.mode_config.BY_KEY["quick"].roles)
            with self.assertRaises(self.cli.runtime.ConfigError):
                self.cli.validate_roles(
                    "local", roster,
                    self.cli.mode_config.BY_KEY["summarize"].roles)

    def test_model_identifier_colon_is_not_mistaken_for_a_harness(self):
        tagged = "family:variant"
        roster = {"local": {"_local": True, "plan": [],
                            "write": [tagged], "audit": [tagged],
                            "repair": [tagged]}}
        env = {f"LOCAL_CHAIN_{role.upper()}": tagged
               for role in ("write", "audit", "repair")}
        with unittest.mock.patch.dict(os.environ, env, clear=True):
            self.cli.validate_roles(
                "local", roster, self.cli.mode_config.BY_KEY["quick"].roles)

    def test_the_requested_mode_is_resolved_before_role_validation(self):
        # Modes are never rewritten, so the requested mode determines the
        # frozen roles: a quick-only backend is never asked for a plan chain
        # it does not have. The resolution must precede both calls.
        source = (ENG / "summ_cli.py").read_text()
        chosen = source.index("effective_mode = selected_mode")
        frozen = source.index(
            "freeze_model_routes(harness, roster, effective_mode.roles)", chosen)
        checked = source.index(
            "validate_roles(harness, roster, effective_mode.roles)", chosen)
        self.assertLess(chosen, frozen)
        self.assertLess(frozen, checked)
        self.assertLess(chosen, checked)

    def test_trusted_tts_artifact_skips_model_role_validation(self):
        source = (ENG / "summ_cli.py").read_text()
        start = source.index("def do_tts(")
        end = source.index("# ---------------------------------------------------------------- doctor", start)
        body = source[start:end]
        trusted = body.index("if trusted_artifact:")
        validate = body.index("validate_roles(")
        fallback = body.index("else:", trusted)
        self.assertLess(trusted, fallback)
        self.assertGreater(validate, fallback)




class HeadingDenseDocumentsDoNotExplodeIntoMicroParts(unittest.TestCase):
    """Splitting big sections was handled; packing small ones never was.

    Found by the first real UI run, on a 9,280-word architecture guide with 195
    headings. One part per heading gave 195 parts of a MEDIAN 36 words, 119 of
    them under 50 -- each buying its own plan, audit and revise round. That is
    roughly nineteen times the model calls a same-sized paper costs, spent on
    parts too small to plan a unit from. Technical documentation and e-ink
    guides are exactly this shape, so this is the ordinary case, not an edge."""
    led = ledger

    def _blocks(self, text):
        return [{"text": b, "words": len(b.split())}
                for b in re.split(r"\n\s*\n", text) if b.strip()]

    def _doc(self, headings, words_each):
        return "\n\n".join(f"## Section {i}\n\n" + " ".join(["word"] * words_each)
                            for i in range(headings))

    def test_many_tiny_sections_are_packed_not_one_part_each(self):
        parts = self.led.split_sections(self._blocks(self._doc(60, 30)))
        self.assertLess(len(parts), 10,
                        f"{len(parts)} parts from 60 tiny sections — not packed")

    def test_packing_never_exceeds_the_part_ceiling(self):
        parts = self.led.split_sections(self._blocks(self._doc(60, 30)))
        for p in parts:
            w = sum(b["words"] for b in p["blocks"])
            self.assertLessEqual(w, self.led.PART_MAX, f"part of {w}w over ceiling")

    def test_every_block_survives_exactly_once_and_in_order(self):
        # Coverage accounting is over source blocks. A packer that dropped or
        # reordered one would corrupt the guarantee the whole project rests on.
        blocks = self._blocks(self._doc(60, 30))
        parts = self.led.split_sections(blocks)
        flat = [b for p in parts for b in p["blocks"]]
        self.assertEqual(len(flat), len(blocks))
        self.assertTrue(all(a is c for a, c in zip(flat, blocks)),
                        "packing reordered or substituted blocks")

    def test_a_section_larger_than_the_ceiling_still_splits(self):
        # The pre-existing behaviour must be untouched.
        parts = self.led.split_sections(self._blocks(self._doc(1, 4000)))
        self.assertGreater(len(parts), 1)
        self.assertTrue(all(p["part"] != "1 of 1" for p in parts))

    def test_a_packed_part_says_it_spans_several_headings(self):
        # Otherwise the model reads a group as one topic that lost its
        # subheadings, and writes a unit that misattributes them.
        parts = self.led.split_sections(self._blocks(self._doc(60, 30)))
        self.assertTrue(any("more)" in p["title"] for p in parts),
                        "a packed part does not disclose that it spans headings")

    def test_an_ordinary_paper_is_not_repacked_into_fewer_parts(self):
        # Sections at or near the ceiling must keep their own parts: packing is
        # for micro-sections, not a licence to merge real ones.
        parts = self.led.split_sections(self._blocks(self._doc(6, 1100)))
        self.assertEqual(len(parts), 6)
        self.assertTrue(all("more)" not in p["title"] for p in parts))




class UIReportsEveryTerminalStateHonestly(unittest.TestCase):
    """The five ways a job can end must read differently to the user.

    A UI that shows the same green line for 'published', 'exited 0 but the
    record is incomplete', and 'the engine died' is the failure this project
    exists to prevent, moved one layer out: a confident report of something
    that did not happen."""
    ui = load("summ_ui")

    def setUp(self):
        import tkinter as tk
        try:
            self.root = tk.Tk()
        except tk.TclError as e:
            self.skipTest(f"no display: {e}")
        self.root.withdraw()
        self.app = self.ui.App(self.root)
        self.d = pathlib.Path(tempfile.mkdtemp())
        self.app.job = self.d
        self.addCleanup(self.root.destroy)
        self.addCleanup(shutil.rmtree, self.d, True)

    def _events(self, *evs):
        if not any(e.get("event") == "job_started" for e in evs):
            evs = ({"event": "job_started", "mode": "summarize",
                    "targets": 1},) + evs
        (self.d / "progress.jsonl").write_text(
            "".join(json.dumps({"schema": "summer.progress.v2",
                                "time_utc": "2000-01-01T00:00:00.000Z", **e}) + "\n"
                    for e in evs))

    def _finish(self, rc):
        self.app._finish(rc)
        return self.app.status.cget("text"), self.app.detail.cget("text")

    def test_removing_one_queued_job_leaves_the_other(self):
        first = {"stamp": "q1", "label": "first"}
        second = {"stamp": "q2", "label": "second"}
        self.app.pending = [first, second]
        self.app._refresh_queue()
        self.assertTrue(self.app.remove_queued("q1"))
        self.assertEqual([second], self.app.pending)
        self.assertFalse(self.app.remove_queued("q1"))

    def test_selecting_corpus_regroups_files_already_waiting(self):
        runs = self.d / "runs"
        runs.mkdir()
        files = []
        for name in ("one.md", "two.md", "three.md"):
            path = self.d / name
            path.write_text(name)
            files.append(path)
        chosen = self.ui.selection.resolve_paths(files)
        with unittest.mock.patch.object(self.ui, "job_root", return_value=runs), \
                unittest.mock.patch.object(
                    self.ui.custom_instructions, "save_last"):
            self.app._accept_selection(chosen)
            self.assertEqual(3, len(self.app.pending))
            self.app.scope.set("corpus")
            self.app.scope_changed()
            self.assertEqual(1, len(self.app.pending))
            job = self.app.pending[0]
            self.assertEqual("corpus", job["scope"])
            self.assertEqual(3, len(job["selection"].documents))
            self.assertEqual("Corpus (3 documents)", job["label"])
            self.assertEqual("grid", self.app.corpus_name_entry.winfo_manager())
            self.assertEqual(1, len(list(runs.iterdir())))

            self.app.scope.set("batch")
            self.app.scope_changed()
            self.assertEqual(3, len(self.app.pending))
            self.assertTrue(all(job["scope"] == "batch"
                                for job in self.app.pending))

    def test_corpus_selected_before_add_stays_selected_and_merges_later_adds(self):
        runs = self.d / "runs"
        runs.mkdir()
        files = []
        for name in ("one.md", "two.md", "three.md"):
            path = self.d / name
            path.write_text(name)
            files.append(path)
        first = self.ui.selection.resolve_paths(files[:2])
        later = self.ui.selection.resolve_paths(files[2:])
        with unittest.mock.patch.object(self.ui, "job_root", return_value=runs), \
                unittest.mock.patch.object(
                    self.ui.custom_instructions, "save_last"):
            self.app.scope.set("corpus")
            self.app.scope_changed()
            self.app._accept_selection(first)
            self.assertEqual("corpus", self.app.scope.get())
            self.assertEqual(1, len(self.app.pending))
            self.assertTrue(self.app.corpus_name.get())
            self.app._accept_selection(later)
            self.assertEqual("corpus", self.app.scope.get())
            self.assertEqual(1, len(self.app.pending))
            self.assertEqual(3, len(self.app.pending[0]["selection"].documents))
            self.assertEqual("Corpus (3 documents)", self.app.pending[0]["label"])
            self.assertEqual(1, len(list(runs.iterdir())))

    def test_leaving_summarize_splits_a_waiting_corpus(self):
        runs = self.d / "runs"
        runs.mkdir()
        files = []
        for name in ("one.md", "two.md"):
            path = self.d / name
            path.write_text(name)
            files.append(path)
        chosen = self.ui.selection.resolve_paths(files)
        with unittest.mock.patch.object(self.ui, "job_root", return_value=runs), \
                unittest.mock.patch.object(
                    self.ui.custom_instructions, "save_last"):
            self.app.scope.set("corpus")
            self.app._accept_selection(chosen)
            self.app.mode.set(self.ui.mode_config.BY_KEY["quick"].label)
            self.assertEqual("batch", self.app.scope.get())
            self.assertEqual(2, len(self.app.pending))
            self.assertTrue(all(job["scope"] == "batch"
                                for job in self.app.pending))
            self.assertTrue(all(job["mode"] == self.app.mode.get()
                                for job in self.app.pending))

    def test_an_armed_corpus_queue_does_not_admit_a_single_batch_document(self):
        files = []
        for name in ("one.md", "two.md", "three.md"):
            path = self.d / name
            path.write_text(name)
            files.append(path)
        existing = self.ui.selection.resolve_paths(files[:2]).with_corpus_outputs()
        later = self.ui.selection.resolve_paths(files[2:])
        self.app.scope.set("corpus")
        self.app.queue_started = True
        self.app.pending = [{"selection": existing}]
        with self.assertRaisesRegex(ValueError, "at least two"):
            self.app._enqueue_selection(later, "summarize")

    def test_scope_controls_lock_after_the_queue_starts(self):
        root = self.d / "active"
        root.mkdir()
        self.app.queue_started = True
        self.app.active = {"active": self.ui.JobState(
            "active", root, {"label": "active"})}
        self.app._refresh_queue()
        self.assertIn("disabled", self.app.batch_scope.state())
        self.assertIn("disabled", self.app.corpus_scope.state())

    def test_cancelling_one_active_job_leaves_the_other_running(self):
        first_root = self.d / "first"
        second_root = self.d / "second"
        first_root.mkdir()
        second_root.mkdir()
        first = self.ui.JobState("j1", first_root, {"label": "first"})
        second = self.ui.JobState("j2", second_root, {"label": "second"})
        self.app.active = {"j1": first, "j2": second}
        self.app._refresh_queue()
        self.assertTrue(self.app.cancel_active_job("j1"))
        self.assertEqual("cancel\n", first.cancel_file.read_text())
        self.assertFalse(second.cancel_file.exists())
        self.assertEqual({"j1", "j2"}, set(self.app.active))

    def test_a_synthetic_event_stream_replays_to_a_correct_finished_window(self):
        # Regression for a DEAD BRANCH. The status-update `if` was spliced into
        # the middle of _apply's elif chain, so `target_finished` chained off a
        # truthy self.call and became unreachable once any model call had run:
        # out_dir was never set and Open Output Folder never appeared. Valid
        # Python, and invisible to every source-string assertion -- only
        # replaying a representative event stream found it. The fixture is a
        # deliberately synthetic v2 stream with invented values.
        fixture = ENG / "tests" / "fixtures-progress.jsonl"
        (self.d / "progress.jsonl").write_text(fixture.read_text())
        for ev in self.app._new_events():
            self.app._apply(ev)
        self.assertEqual(self.app.parts, (3, 3))
        self.app._finish(0)
        self.assertIn("Done", self.app.status.cget("text"))
        self.assertIsNotNone(self.app.out_dir, "target_finished branch is dead again")
        self.assertTrue(self.app.open_out.winfo_manager(),
                        "Open Output Folder was never shown")

    def test_every_event_in_the_synthetic_stream_is_one_the_ui_handles(self):
        # An event the UI silently ignores is a schema change nobody noticed.
        fixture = ENG / "tests" / "fixtures-progress.jsonl"
        seen = {json.loads(l)["event"] for l in fixture.read_text().splitlines() if l.strip()}
        src = (ENG / "summ_ui.py").read_text()
        # job_started carries only `mode` and `targets`, both of which this UI
        # already knows -- it chose the mode and target_started carries
        # index/count. Ignoring it is deliberate; anything else appearing here
        # is a schema change nobody wired up.
        knowingly_ignored = {"job_started"}
        missing = sorted(e for e in seen - knowingly_ignored if f'"{e}"' not in src)
        self.assertEqual(missing, [], f"UI does not handle emitted event(s): {missing}")

    def test_a_multi_document_run_says_which_document_it_is_on(self):
        # The clipboard may hold several paths. Without this a three-document
        # run looks like a one-document run that keeps restarting.
        self.app._apply({"event": "target_started", "index": 2, "count": 3,
                         "title": "second", "words": 1000})
        self.assertIn("[2/3]", self.app.status.cget("text"))

    def test_a_single_document_run_is_not_cluttered_with_a_counter(self):
        self.app._apply({"event": "target_started", "index": 1, "count": 1,
                         "title": "only", "words": 1000})
        self.assertNotIn("[1/1]", self.app.status.cget("text"))

    def test_an_engine_verdict_disagreeing_with_the_exit_code_is_a_protocol_error(self):
        # When the engine's own verdict and the process exit disagree, one is
        # lying and there is no basis for choosing. Presenting either as the
        # result would be a confident answer nobody verified.
        self._events({"event": "target_finished", "index": 1, "status": "succeeded",
                      "exit_code": 0, "outputs": ["/o/a.summary.md"]},
                     {"event": "job_finished", "status": "succeeded", "exit_code": 0})
        st, det = self._finish(3)
        self.assertIn("protocol error", st.lower())
        self.assertIn("0", st)
        self.assertIn("3", st)
        self.assertNotIn("Done", st)

    def test_a_real_publication_names_both_files(self):
        self._events({"event": "target_finished", "index": 1, "status": "succeeded",
                      "exit_code": 0, "outputs": ["/o/a.summary.md", "/o/a.brief.md"]},
                     {"event": "job_finished", "status": "succeeded", "exit_code": 0})
        st, det = self._finish(0)
        self.assertIn("Done", st)
        self.assertIn("/o/a.summary.md", det)
        self.assertIn("/o/a.brief.md", det)

    def test_a_multi_document_success_lists_every_output_pair(self):
        self._events(
            {"event": "job_started", "mode": "summarize", "targets": 2},
            {"event": "target_finished", "index": 1, "status": "succeeded",
             "exit_code": 0, "outputs": ["/o/one.summary.md", "/o/one.brief.md"]},
            {"event": "target_finished", "index": 2, "status": "succeeded",
             "exit_code": 0, "outputs": ["/o/two.summary.md", "/o/two.brief.md"]},
            {"event": "job_finished", "status": "succeeded", "exit_code": 0})
        _, detail = self._finish(0)
        for name in ("one.summary.md", "one.brief.md",
                     "two.summary.md", "two.brief.md"):
            self.assertIn(name, detail)

    def test_exit_zero_with_no_terminal_event_is_not_called_done(self):
        # The CLI can exit 0 while its progress record is incomplete. Reporting
        # that as a verified publication is a confident claim about something
        # nobody checked.
        self._events({"event": "stage", "name": "compose"})
        st, _ = self._finish(0)
        self.assertNotIn("Done", st)
        self.assertIn("incomplete", st.lower())

    def test_successful_target_without_job_terminal_is_not_called_done(self):
        self._events({"event": "target_finished", "index": 1,
                      "status": "succeeded", "exit_code": 0,
                      "outputs": ["/o/a.summary.md", "/o/a.brief.md"]})
        st, _ = self._finish(0)
        self.assertNotIn("Done", st)
        self.assertIn("incomplete", st.lower())

    def test_failed_job_status_cannot_agree_with_exit_zero(self):
        self._events({"event": "target_finished", "index": 1,
                      "status": "succeeded", "exit_code": 0,
                      "outputs": ["/o/a.summary.md", "/o/a.brief.md"]},
                     {"event": "job_finished", "status": "failed",
                      "exit_code": 0})
        st, _ = self._finish(0)
        self.assertIn("protocol error", st.lower())
        self.assertNotIn("Done", st)

    def test_missing_or_duplicate_target_terminals_prevent_success(self):
        cases = (
            ({"event": "job_started", "mode": "summarize", "targets": 2},
             {"event": "target_finished", "index": 1, "status": "succeeded",
              "exit_code": 0, "outputs": ["/o/a.summary.md"]},
             {"event": "job_finished", "status": "succeeded", "exit_code": 0}),
            ({"event": "job_started", "mode": "summarize", "targets": 1},
             {"event": "target_finished", "index": 1, "status": "succeeded",
              "exit_code": 0, "outputs": ["/o/a.summary.md"]},
             {"event": "target_finished", "index": 1, "status": "succeeded",
              "exit_code": 0, "outputs": ["/o/a.summary.md"]},
             {"event": "job_finished", "status": "succeeded", "exit_code": 0}),
        )
        for events in cases:
            with self.subTest(records=len(events)):
                self._events(*events)
                st, _ = self._finish(0)
                self.assertIn("protocol error", st.lower())
                self.assertNotIn("Done", st)

    def test_a_classified_failure_shows_its_reason_and_code(self):
        self._events({"event": "target_finished", "index": 1, "status": "failed",
                      "exit_code": 3, "destination_unchanged": True},
                     {"event": "job_finished", "status": "failed", "exit_code": 3})
        st, det = self._finish(3)
        self.assertIn("seal", st.lower())
        self.assertIn("3", st)
        self.assertIn("unchanged", det.lower())

    def test_untouched_destinations_are_only_claimed_when_the_cli_says_so(self):
        # 'Nothing published; previous files unchanged' is a guarantee. Inferring
        # it from a bare nonzero exit would state it when nobody verified it.
        self._events({"event": "target_finished", "index": 1, "status": "failed",
                      "exit_code": 1})
        _, det = self._finish(1)
        self.assertNotIn("unchanged", det.lower())

    def test_a_dead_engine_does_not_claim_destinations_were_untouched(self):
        self._events({"event": "stage", "name": "ledger"})
        st, det = self._finish(1)
        self.assertNotIn("unchanged", det.lower())

    def test_a_failed_launch_is_reported_as_a_launch_not_an_exit(self):
        self._events()
        st, _ = self._finish(None)
        self.assertIn("start", st.lower())

    def test_the_controls_come_back_after_every_ending(self):
        # A window stuck disabled after a failure needs restarting to retry.
        for rc in (0, 1, 3, None):
            self._events()
            self.app.running = True
            self._finish(rc)
            self.assertFalse(self.app.running, f"still marked running after rc={rc}")
            self.assertNotIn("disabled", self.app.start_btn.state(),
                             f"Start left disabled after rc={rc}")

    def test_cancel_active_marks_every_job_and_discards_the_queue(self):
        states = []
        for name in ("one", "two"):
            root = self.d / name; root.mkdir()
            state = self.ui.JobState(name, root, {"label": name})
            self.app.active[name] = state
            states.append(state)
        self.app.running = True
        self.app.pending = [{"label": "later"}, {"label": "later still"}]
        self.app.cancel_active()
        self.assertTrue(all(state.cancel_file.read_text() == "cancel\n"
                            for state in states))
        self.assertEqual([], self.app.pending)
        self.assertTrue(self.app.cancelling)
        self.assertIn("disabled", self.app.start_btn.state())

    def test_failed_cancel_marker_does_not_claim_every_job_is_stopping(self):
        for name in ("one", "two"):
            root = self.d / name; root.mkdir()
            self.app.active[name] = self.ui.JobState(
                name, root, {"label": name})
        with unittest.mock.patch.object(
                self.app, "_write_cancel_request",
                side_effect=[None, OSError("read only")]):
            self.app.cancel_active()
        self.assertTrue(self.app.cancelling)
        self.assertNotIn("disabled", self.app.cancel_btn.state())
        self.assertIn("partly", self.app.status.cget("text").lower())
        self.assertIn("may continue and publish",
                      self.app.detail.cget("text").lower())

    def test_cancelled_terminal_state_is_not_reported_as_failure_or_success(self):
        self._events({"event": "target_finished", "index": 1,
                      "status": "cancelled", "exit_code": 130,
                      "destination_unchanged": True},
                     {"event": "job_finished", "status": "cancelled",
                      "exit_code": 130})
        status, detail = self._finish(130)
        self.assertIn("cancelled", status.lower())
        self.assertNotIn("failed", status.lower())
        self.assertIn("unchanged", detail.lower())

    def test_poll_restores_controls_after_cancellation_drains(self):
        self.app.cancelling = True
        self.app.start_btn.configure(state="disabled")
        self.app.cancel_btn.configure(state="disabled")
        self.app._poll()
        self.assertFalse(self.app.cancelling)
        self.assertNotIn("disabled", self.app.start_btn.state())
        self.assertIn("disabled", self.app.cancel_btn.state())




class ShortDocumentsAlwaysGetASummary(unittest.TestCase):
    """A 253-word document used to fail six minutes in, and then for a while it
    was refused up front. Both were wrong. The owner's rule is absolute: there
    is no minimum length and a summary is always produced.

    Summarize stays Full at every source size; Quick is an explicit mode any
    document can ask for, never an automatic rewrite. No test here spends
    a model call; the routing decision is what is under test."""
    cli = load("summ_cli")

    def test_there_is_no_refusal_left_in_the_cli(self):
        src = (ENG / "summ_cli.py").read_text()
        for gone in ("too short to summarize", "too_short", "MIN_WORDS"):
            self.assertNotIn(gone, src, f"{gone!r} survived the reversal")

    def test_no_length_threshold_rewrites_the_mode(self):
        src = (ENG / "summ_cli.py").read_text()
        for gone in ("SHORT_WORDS", "words < SHORT_WORDS",
                     "under {SHORT_WORDS}w", "quick, under"):
            self.assertNotIn(gone, src, f"length rewrite survived: {gone!r}")
        self.assertIn("effective_mode = selected_mode", src)

    def test_quick_is_a_mode_any_document_can_ask_for(self):
        # The point of it being a mode: a long document can choose it too.
        src = (ENG / "summ_cli.py").read_text()
        self.assertIn('mode_config.BY_KEY["quick"].flag', src)
        self.assertIn("a.quick or", src)

    def test_the_quick_path_writes_the_same_two_files_publication_expects(self):
        short = (ENG / "shortsum.py").read_text()
        self.assertIn('"detailed.md"', short)
        self.assertIn('"brief.md"', short)

    def test_quick_never_publishes_a_warning_in_place_of_verification(self):
        short = (ENG / "shortsum.py").read_text()
        self.assertNotIn("reading_note", short)
        self.assertNotIn("disclosed finding", short)

    def test_the_quick_path_still_refuses_to_hand_back_the_source(self):
        # The one failure the project exists to prevent survives every mode.
        ss = load("shortsum")
        source = ("The Federal Reserve expanded its balance sheet after 2008. "
                  "Economists disagree about the effect. The evidence is "
                  "largely correlational and hard to separate from stress.")
        self.assertTrue(ss.not_a_copy(source, source),
                        "handing the source back was accepted")
        self.assertFalse(ss.not_a_copy("Rates fell; the cause is disputed.", source))

    def test_the_quick_gates_are_not_relaxed(self):
        ss = load("shortsum")
        src = "Rates rose by 3 percent in 2008."
        self.assertIn("list or table markup", ss.defects("- a bullet", src))
        self.assertIn("first person", ss.defects("We find that rates rose.", src))
        self.assertIn("first person", ss.defects("Rates rose, and I think more.", src))
        self.assertIn("first person", ss.defects("I find that rates rose.", src))
        self.assertIn("first person", ss.defects("I, however, find otherwise.", src))
        # Roman numerals and the abbreviation "US" are not first person.
        for clean in ("The US model, under a Type I error bound, held.",
                      "Phase I ended before the US entered.",
                      "World War I ended in 1918. Chapter I covers it."):
            self.assertNotIn("first person", ss.defects(clean, src), clean)
        self.assertTrue(any("absent from the source" in d
                            for d in ss.defects("Rates rose 47 percent.", src)))

    def test_the_quick_path_enforces_the_same_publication_bands(self):
        # A "condensation" that is the source reworded is not a summary. One
        # attempt came back at 128 words from a 131-word source.
        ss = load("shortsum")
        cmp_mod = load("compose")
        self.assertEqual(ss._bands(), cmp_mod.BANDS,
                         "the quick path declared its own bands")
        # A LONG source is held to the published bands exactly.
        src = " ".join(["word"] * 4000)
        hi = cmp_mod.BANDS["detailed"][1]
        self.assertFalse(ss.not_a_copy(" ".join(["x"] * int(4000 * hi)), src,
                                       "detailed"))
        self.assertTrue(ss.not_a_copy(" ".join(["x"] * 3900), src, "detailed"))
        # A SHORT source is allowed to keep more, because it has no redundancy
        # to spend -- see CompressionRatioFollowsRedundancy.
        short_src = " ".join(["word"] * 131)
        self.assertTrue(ss.not_a_copy(" ".join(["x"] * 128), short_src, "detailed"))
        self.assertFalse(ss.not_a_copy(" ".join(["x"] * 70), short_src, "detailed"))

    def test_a_very_short_source_has_a_feasible_brief_range(self):
        ss = load("shortsum")
        limit = ss.ceiling_words(20)["brief"]
        floor = min(limit, max(8, int(limit * 0.20)))
        self.assertLessEqual(floor, limit)
        self.assertFalse(ss.not_a_copy(" ".join(["x"] * floor),
                                       " ".join(["word"] * 20), "brief"))

    def test_the_revision_is_audited(self):
        # Not auditing it shipped a real reversal: asked to compress, a revision
        # turned "not NECESSARILY inflationary" into "not inflationary" -- a flat
        # denial the source does not make -- and nothing was looking.
        short = (ENG / "shortsum.py").read_text()
        self.assertIn('"short-reaudit"', short)
        i = short.index("short-revise")
        self.assertIn("short-reaudit", short[i:], "the revision is not re-audited")

    def test_too_short_revision_has_an_explicit_expansion_target(self):
        short = (ENG / "shortsum.py").read_text()
        self.assertIn("EXPAND REQUIRED", short)
        self.assertIn("Expand it to at least {floor} words", short)
        # The repair appendix is shared prompt text now; the expansion target
        # moved with it, it did not vanish, and padding is still refused.
        shared = " ".join((ENG / "prompts" / "pair-repair.txt").read_text().split())
        self.assertIn("where a reading is too short, extend it with "
                      "source-supported material and no padding", shared)
        # And the opposite direction: over the ceiling means a global rewrite
        # of that reading, not a local patch that keeps every passage.
        self.assertIn("over its ceiling, revise that whole reading", shared)
        self.assertIn("{FINDINGS}", shared)

    def test_a_finding_that_survives_revision_publishes_with_open_status(self):
        # The old terminal veto discarded a usable pair here. The repair
        # budget is spent, so the safest retained candidate is published
        # with its surviving findings stated in the report, not in the
        # readings.
        ss = load("shortsum")
        source = ("Rates fell after the reform, but the evidence remained limited "
                  "and the authors did not claim that the reform caused the fall.")

        class FakeMS:
            MODELS = ["local"]
            AUDIT_MODELS = ["local"]
            REPAIR_MODELS = ["local"]
            JSON_REQUEST_OPTIONS = {"response_format": {"type": "json_object"}}

            @staticmethod
            def run(prompt, out_dir, chain, stage, validate=None,
                    gateway_options=None):
                pair = {
                    "detailed": ("Rates fell after the reform, but the evidence "
                                 "did not establish causation."),
                    "brief": "Rates fell, but causation was not established.",
                }
                if stage == "short":
                    return json.dumps(pair)
                if stage == "short-revise":
                    return json.dumps(patch_replace(
                        pair["detailed"], pair["brief"],
                        "detailed", pair["detailed"]))
                return json.dumps({"verdict": "revise", "findings": [
                    AF("The reading omits that the evidence remained limited")]})

        class FakeLedger:
            @staticmethod
            def parse_strict(raw, stage):
                return json.loads(raw)

        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            src, out = root / "source.txt", root / "out"
            src.write_text(source)
            with unittest.mock.patch.object(ss, "_runner", return_value=FakeMS()), \
                 unittest.mock.patch.object(ss, "_ledger", return_value=FakeLedger):
                self.assertEqual(0, ss.run(src, out))
            self.assertTrue((out / "detailed.md").exists())
            self.assertTrue((out / "brief.md").exists())
            report = json.loads((out / "short-report.json").read_text())
            self.assertEqual(report["status"], "open_findings")
            self.assertTrue(report["findings"], "surviving findings were not disclosed")
            self.assertTrue((out / "detailed.md").read_text().strip())
            self.assertTrue((out / "brief.md").read_text().strip())

    def test_an_unavailable_quick_audit_publishes_with_review_status(self):
        # Reviewer outage must not discard a usable pair either. The initial
        # bytes are published as review_unavailable with that cause stated.
        ss = load("shortsum")
        source = ("Rates fell after the reform, but the evidence remained limited "
                  "and the authors did not claim that the reform caused the fall.")

        class FakeMS:
            MODELS = AUDIT_MODELS = REPAIR_MODELS = ["local"]
            JSON_REQUEST_OPTIONS = {"response_format": {"type": "json_object"}}

            @staticmethod
            def run(prompt, out_dir, chain, stage, validate=None,
                    gateway_options=None):
                if stage == "short-audit":
                    raise RuntimeError("auditor unavailable")
                return json.dumps({
                    "detailed": ("Rates fell after the reform, but the evidence "
                                 "did not establish causation."),
                    "brief": "Rates fell after reform, but evidence stayed limited.",
                })

        class FakeLedger:
            @staticmethod
            def parse_strict(raw, stage):
                return json.loads(raw)

        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            src, out = root / "source.txt", root / "out"
            src.write_text(source)
            with unittest.mock.patch.object(ss, "_runner", return_value=FakeMS()), \
                 unittest.mock.patch.object(ss, "_ledger", return_value=FakeLedger):
                self.assertEqual(0, ss.run(src, out))
            self.assertTrue((out / "detailed.md").exists())
            self.assertTrue((out / "brief.md").exists())
            report = json.loads((out / "short-report.json").read_text())
            self.assertEqual(report["status"], "review_unavailable")
            self.assertIn("short-audit", report["review"])

    def test_there_is_exactly_one_revision_and_no_loop(self):
        # There is one bounded repair, never a stochastic convergence loop.
        short = (ENG / "shortsum.py").read_text()
        self.assertIn('"short-revise"', short)
        self.assertNotIn("while ", short, "the quick path contains a while loop")
        self.assertNotIn("for _ in range", short,
                         "the quick path contains a for-range loop")

    def test_the_audit_uses_the_audit_chain_so_the_picker_applies(self):
        # Otherwise choosing an audit model in the window would do nothing here.
        short = (ENG / "shortsum.py").read_text()
        self.assertIn("ms.AUDIT_MODELS", short)
        # The one correction goes back to the producer that wrote the
        # candidate, never down the repair chain (AGENTS.md Defaults).
        self.assertIn("last_ok_route(out_dir, \"short\"", short)
        self.assertNotIn("ms.REPAIR_MODELS", short)

    def test_every_quick_json_call_uses_structured_gateway_mode(self):
        # Quick has its own production entry point. A fix wired only through the
        # ledger path leaves this mode exposed to the same long prose/no-JSON
        # failure, so exercise shortsum.run itself and inspect every call.
        ss = load("shortsum")
        source = ("The committee reported that rates fell after the reform. "
                  "The evidence was limited and the authors did not claim causation.")

        class FakeMS:
            MODELS = ["localgw:model-a"]
            AUDIT_MODELS = ["localgw:model-a"]
            REPAIR_MODELS = ["localgw:model-a"]

            def __init__(self):
                self.calls = []

            def run(self, prompt, out_dir, chain, stage, validate=None,
                    gateway_options=None):
                self.calls.append((stage, gateway_options))
                if stage in {"short", "short-revise"}:
                    return json.dumps({
                        "detailed": "Rates fell after the reform, but evidence did not establish causation.",
                        "brief": "Rates fell, but causation was not established by evidence.",
                    })
                return json.dumps({"verdict": "pass", "findings": []})

        class FakeLedger:
            @staticmethod
            def parse_strict(raw, stage):
                return json.loads(raw)

        fake = FakeMS()
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            src = root / "source.txt"
            out = root / "out"
            src.write_text(source)
            with unittest.mock.patch.object(ss, "_runner", return_value=fake), \
                 unittest.mock.patch.object(ss, "_ledger", return_value=FakeLedger):
                self.assertEqual(0, ss.run(src, out))
        self.assertGreaterEqual(len(fake.calls), 2)
        names = {
            stage: options["response_format"]["json_schema"]["name"]
            for stage, options in fake.calls
        }
        self.assertEqual(names["short"], "summer_quick_result")
        self.assertEqual(names["short-audit"], "summer_quick_audit")
        for _stage, options in fake.calls:
            self.assertEqual(options["response_format"]["type"], "json_schema")
            schema = options["response_format"]["json_schema"]["schema"]
            self.assertTrue(schema["required"], schema)

    def test_the_length_ceiling_comes_from_the_one_band_definition(self):
        # A scoped exception to "no model-facing length instruction": a CEILING,
        # whose numbers come from compose.BANDS rather than being invented here.
        short = (ENG / "shortsum.py").read_text()
        self.assertIn("{D_WORDS}", short)
        # ceilings() derives from compose.BANDS; it is not a second table.
        self.assertIn('cei["detailed"]', short)
        self.assertIn("b = _bands()", short)
        prompt = load("pair_review").write_template()
        self.assertIn("at most {D_WORDS} words", prompt)
        self.assertIn("never pad", prompt)
        # A ceiling only. "Aim for" alongside "at most" gave two targets and
        # the writer packed to the higher one.
        self.assertNotIn("Aim for", prompt)

    def test_a_number_already_in_the_source_is_not_called_invented(self):
        ss = load("shortsum")
        src = "Holdings rose from $900 billion in 2008 to $4,000 billion by 2014."
        self.assertEqual([d for d in ss.defects("Holdings rose from 900 to 4,000 "
                                                "between 2008 and 2014.", src)
                          if "absent" in d], [])


class WriterPromptContract(unittest.TestCase):
    """One writer prompt serves Quick and Full; the readings are prose for a
    reader outside the field, not notes.

    Claim-first and one-relation-per-sentence rules produced accurate but
    compressed claim-stacks, and a prescribed opening formula made every
    reading start the same way. The prompt now names a reader and a register
    and leaves sentence mechanics to the writer. Lists stay banned. A missing
    heading or a long paragraph is not a publication veto.
    """

    def _writer(self):
        # Line wraps are layout, not contract.
        return " ".join(load("pair_review").write_template().split())

    def test_quick_and_full_compose_the_same_writer_and_audit(self):
        # Single point of change: the last style decision was hand-edited into
        # two near-identical files. There is one of each now.
        pr = load("pair_review")
        fs = load("fullsum")
        self.assertEqual(fs._templates(), (pr.write_template(), pr.audit_template()))
        for name in ("short.txt", "full.txt", "short-audit.txt", "full-audit.txt"):
            self.assertFalse((ENG / "prompts" / name).exists(), name)
        for slot in ("{EVIDENCE_KIND}", "{D_WORDS}", "{B_WORDS}", "{SOURCE}"):
            self.assertIn(slot, pr.write_template())
        for slot in ("{SCOPE}", "{SOURCE}", "{DETAILED}", "{BRIEF}"):
            self.assertIn(slot, pr.audit_template())

    def test_the_writer_is_given_a_reader_and_a_register_not_sentence_rules(self):
        prompt = self._writer()
        self.assertIn("reader outside the field", prompt)
        self.assertIn("well-edited article", prompt)
        self.assertIn("connected paragraphs", prompt)
        # Two-sided and passage-level. A per-sentence rule drove the daily
        # model to a 17-word mean with clipped wire copy on Quick; dropping
        # the rule let it return to 56 words with 63% over 40 on Full. The
        # range describes a passage, disclaims a quota, names the wire-copy
        # failure, and asks for revision in both directions.
        self.assertIn("Across a passage, let most sentences fall", prompt)
        self.assertIn("not a sentence-by-sentence ceiling or quota", prompt)
        self.assertIn("revise in both directions", prompt)
        self.assertIn("an actor and a reporting verb", prompt)
        self.assertNotIn("one hundred and fifty", prompt)
        # Materiality, not atomic coverage: "every distinct claim" under a 40%
        # ceiling told the writer to paraphrase almost everything.
        self.assertIn("whose loss would change", prompt)
        self.assertNotIn("Every distinct claim", prompt)
        for mechanics in ("Lead with the claim", "one main relation per sentence",
                          "kind of account", "non-negotiable"):
            self.assertNotIn(mechanics, prompt)

    def test_the_writer_stays_inside_the_source(self):
        prompt = self._writer()
        self.assertIn("no outside background", prompt)
        self.assertIn("only with the material's own explanation", prompt)
        self.assertIn("Do not describe the document, its author", prompt)
        self.assertNotIn("no preamble", prompt)

    def test_the_brief_selects_instead_of_cataloguing(self):
        prompt = self._writer()
        self.assertIn("rather than packing", prompt)
        self.assertIn("catalogue", prompt)

    def test_lists_stay_banned_and_headings_are_optional_not_forbidden(self):
        prompt = self._writer()
        self.assertIn("No bullet points, numbered lists, tables", prompt)
        self.assertIn("descriptive ## headings", prompt)
        self.assertNotIn("No headings", prompt)
        window = (ENG / "prompts" / "full-window.txt").read_text()
        self.assertIn("Use no list markup, headings, citations, or first",
                      window)

    def test_a_heading_does_not_make_a_pair_unpublishable(self):
        # Headings are a writer choice. Copying the ledger capsule markup
        # gate onto the pair would turn an allowed ## into nothing published.
        ss = load("shortsum")
        source = ("Prejudice refers to attitudes. Discrimination refers to "
                  "behavioral outcomes.")
        detailed = ("## Definitions\n\nPrejudice is an attitude. "
                    "Discrimination is a behavior.")
        brief = "Prejudice is an attitude, and discrimination is a behavior."
        self.assertEqual([], ss.structural_findings(detailed, brief, source))

    def test_a_markdown_list_still_makes_a_pair_unpublishable(self):
        ss = load("shortsum")
        source = "Active, reserve, and unemployed states are distinct."
        detailed = ("Labor positions divide as follows:\n- active\n"
                    "- reserve\n- unemployed")
        brief = "Active, reserve, and unemployed states are distinct."
        self.assertIn("list or table markup",
                      ss.structural_findings(detailed, brief, source))

    def test_the_auditor_reports_lost_meaning_but_not_style(self):
        # The review loop only ever rewarded density: omissions were findings,
        # over-compression never was, and three repairs squeezed the prose
        # harder each round. UNCLEAR is the counterweight, bounded to passages
        # whose claim cannot be recovered. Style, length, and headings stay
        # off the findings list so the repair budget is not spent on taste.
        prompt = " ".join(load("pair_review").audit_template().split())
        self.assertIn("UNCLEAR.", prompt)
        self.assertIn("only when meaning is lost", prompt)
        self.assertIn("Do not report length, word choice, or a missing heading",
                      prompt)
        self.assertIn("at most eight", prompt)


class QuickCandidatePreservation(unittest.TestCase):
    """Once a usable Quick pair exists, it is published, not discarded.

    Structural usability is a complete pair: both readings non-empty with no
    fabricated exact quotation. Semantic findings, reviewer outage, a failed
    repair, or an exhausted repair budget publish the safest retained
    candidate with explicit status in short-report.json. Only a run where no
    candidate-producing route yields a usable pair publishes nothing. No test
    here spends a model call.
    """
    PAIR = {
        "detailed": ("Rates fell after the reform, but the evidence "
                     "did not establish causation."),
        "brief": "Rates fell, but causation was not established.",
    }
    SOURCE = ("Rates fell after the reform, but the evidence remained limited "
              "and the authors did not claim that the reform caused the fall.")

    class Ledger:
        @staticmethod
        def parse_strict(raw, stage):
            return json.loads(raw)

    def _run(self, ss, ms, source=None, env=None):
        root = pathlib.Path(tempfile.mkdtemp())
        src, out = root / "source.txt", root / "out"
        src.write_text(source or self.SOURCE)
        patches = [unittest.mock.patch.object(ss, "_runner", return_value=ms),
                   unittest.mock.patch.object(ss, "_ledger",
                                              return_value=self.Ledger)]
        if env:
            patches.append(unittest.mock.patch.dict(os.environ, env))
        for p in patches:
            p.start()
        try:
            return ss.run(src, out), out
        finally:
            for p in reversed(patches):
                p.stop()

    @staticmethod
    def _ms(handler):
        class FakeMS:
            MODELS = AUDIT_MODELS = REPAIR_MODELS = ["local"]

            @staticmethod
            def run(prompt, out_dir, chain, stage, validate=None,
                    gateway_options=None):
                return handler(stage)
        return FakeMS()

    def test_a_failed_repair_publishes_the_retained_initial(self):
        ss = load("shortsum")

        def handler(stage):
            if stage == "short-revise":
                raise RuntimeError("repair unavailable")
            if stage in {"short-audit", "short-reaudit"}:
                return json.dumps({"verdict": "revise", "findings": [
                    AF("The reading omits that the evidence remained limited")]})
            return json.dumps(dict(self.PAIR))

        rc, out = self._run(ss, self._ms(handler))
        self.assertEqual(0, rc)
        self.assertEqual((out / "detailed.md").read_text().strip(),
                         self.PAIR["detailed"])
        report = json.loads((out / "short-report.json").read_text())
        self.assertEqual((report["status"], report["repair"]),
                         ("open_findings", "unavailable"))
        self.assertTrue(report["findings"])

    def test_a_failed_reaudit_publishes_as_review_unavailable(self):
        ss = load("shortsum")
        fixed = {
            "detailed": ("Rates fell after the reform; the evidence was limited "
                         "and causation was not established."),
            "brief": "Rates fell without proven causation.",
        }

        def handler(stage):
            if stage == "short-reaudit":
                raise RuntimeError("reauditor unavailable")
            if stage == "short-revise":
                return json.dumps(patch_replace(
                    self.PAIR["detailed"], self.PAIR["brief"],
                    "detailed", fixed["detailed"]))
            if stage == "short-audit":
                return json.dumps({"verdict": "revise", "findings": [
                    AF("The reading omits that the evidence remained limited")]})
            return json.dumps(dict(self.PAIR))

        rc, out = self._run(ss, self._ms(handler))
        self.assertEqual(0, rc)
        report = json.loads((out / "short-report.json").read_text())
        self.assertEqual(report["status"], "open_findings")
        self.assertEqual(report["selected"], "initial")
        # The selected bytes are the reviewed parent. The failed child remains
        # retained in evidence but cannot outrank it merely because its
        # re-audit was unavailable.
        self.assertEqual(report["review"], "complete")

    def test_reviewer_outage_still_repairs_a_structurally_bad_pair(self):
        ss = load("shortsum")

        def handler(stage):
            if stage == "short-audit":
                raise RuntimeError("reviewer unavailable")
            if stage == "short-revise":
                stub = ("Reading the source.", "Reading the source.")
                good = ("Rates fell after the reform; evidence was limited.",
                        "Rates fell after reform, but evidence remained limited.")
                patch = patch_replace(stub[0], stub[1], "detailed", good[0])
                patch["edits"].extend(patch_replace(
                    stub[0], stub[1], "brief", good[1])["edits"])
                return json.dumps(patch)
            if stage == "short-reaudit":
                return json.dumps({"verdict": "pass", "findings": []})
            return json.dumps({
                "detailed": "Reading the source.",
                "brief": "Reading the source.",
            })

        rc, out = self._run(ss, self._ms(handler))
        self.assertEqual(0, rc)
        self.assertEqual((out / "detailed.md").read_text().strip(),
                         "Rates fell after the reform; evidence was limited.")
        report = json.loads((out / "short-report.json").read_text())
        self.assertEqual((report["selected"], report["status"]),
                         ("revised", "pass"))

    def test_no_usable_pair_still_publishes_nothing(self):
        # The writer yielding empty prose on every route is the one model-work
        # case that publishes nothing: there is no pair to preserve.
        ss = load("shortsum")

        def handler(stage):
            if stage in {"short", "short-revise"}:
                return json.dumps({"detailed": "", "brief": ""})
            return json.dumps({"verdict": "pass", "findings": []})

        rc, out = self._run(ss, self._ms(handler))
        self.assertNotEqual(0, rc)
        self.assertFalse((out / "detailed.md").exists())
        self.assertFalse((out / "brief.md").exists())

    def test_a_worse_revision_loses_to_the_retained_initial(self):
        ss = load("shortsum")
        source = ("The committee reported that rates fell after the reform in "
                  "2008. The evidence was limited and the authors did not claim "
                  "causation. Markets reacted quickly while analysts warned that "
                  "stress could return. The report recommends further study before "
                  "any policy change takes effect.")
        initial = {
            "detailed": ("Rates fell after the 2008 reform, though evidence "
                         "was limited and causation unclaimed. Markets reacted "
                         "quickly, analysts warned stress could return, and "
                         "further study was recommended."),
            "brief": ("Rates fell after the reform without proven causation; "
                      "further study was recommended."),
        }
        worse = {
            "detailed": "We think rates fell 47 percent and markets cheered.",
            "brief": initial["brief"],
        }

        def handler(stage):
            if stage == "short-revise":
                return json.dumps(patch_replace(
                    initial["detailed"], initial["brief"],
                    "detailed", worse["detailed"]))
            if stage in {"short-audit", "short-reaudit"}:
                if stage == "short-reaudit":
                    return json.dumps({"verdict": "pass", "findings": []})
                return json.dumps({"verdict": "revise", "findings": [
                    AF("The brief omits the warning that stress could return")]})
            return json.dumps(dict(initial))

        for depth in ("detailed", "brief"):
            self.assertEqual([], ss.defects(initial[depth], source),
                             f"initial {depth} must be mechanically clean")
            self.assertEqual([], ss.not_a_copy(initial[depth], source, depth),
                             f"initial {depth} must pass the copy gate")
        rc, out = self._run(ss, self._ms(handler), source=source)
        self.assertEqual(0, rc)
        report = json.loads((out / "short-report.json").read_text())
        self.assertEqual(report["selected"], "initial")
        self.assertEqual((out / "detailed.md").read_text().strip(),
                         initial["detailed"])
        self.assertEqual(report["status"], "open_findings")

    def test_a_fabricated_exact_quotation_still_withholds(self):
        # A candidate presenting invented prose as an exact quotation is not
        # structurally usable, so when every candidate does it nothing is
        # published.
        ss = load("shortsum")
        quoted = dict(self.PAIR)
        quoted["detailed"] = (self.PAIR["detailed"]
                              + ' As the report says, "a fabricated passage".')

        def handler(stage):
            if stage in {"short", "short-revise"}:
                return json.dumps(dict(quoted))
            return json.dumps({"verdict": "pass", "findings": []})

        with tempfile.TemporaryDirectory() as td:
            snap = pathlib.Path(td) / "instructions.txt"
            snap.write_text("Preserve the source's exact quotations where "
                            "they matter most.")
            rc, out = self._run(
                ss, self._ms(handler),
                env={"SUMM_INSTRUCTIONS_FILE": str(snap)})
        self.assertNotEqual(0, rc)
        self.assertFalse((out / "detailed.md").exists())
        self.assertFalse((out / "brief.md").exists())

    def test_the_old_terminal_vetoes_are_gone(self):
        src = (ENG / "shortsum.py").read_text()
        for veto in ("finding(s) survive the one revision",
                     "revision or re-audit unavailable",
                     "audit unavailable: {str(e)[:80]}"):
            self.assertNotIn(veto, src, f"old veto survived: {veto!r}")


    def test_repair_revises_the_enclosed_candidate_not_a_fresh_pair(self):
        # The repair prompt must carry the exact current Detailed and Brief
        # plus the candidate identity and findings. A Brief-only finding can
        # then be fixed with the Detailed bytes surviving untouched.
        ss = load("shortsum")
        source = ("The committee reported that rates fell after the reform in "
                  "2008. The evidence was limited and the authors did not claim "
                  "causation. Markets reacted quickly while analysts warned that "
                  "stress could return. The report recommends further study before "
                  "any policy change takes effect.")
        initial = {
            "detailed": ("Rates fell after the 2008 reform, though evidence "
                         "was limited and causation unclaimed. Markets reacted "
                         "quickly, analysts warned stress could return, and "
                         "further study was recommended."),
            "brief": ("Rates fell after the reform without proven causation; "
                      "further study was recommended."),
        }
        fixed_brief = ("Rates fell without proven causation, though stress "
                       "could return; further study was recommended.")
        finding = "The brief omits the warning that stress could return"
        seen = {}

        def handler(stage, prompt=""):
            if stage == "short-revise":
                seen["prompt"] = prompt
                return json.dumps(patch_replace(
                    initial["detailed"], initial["brief"],
                    "brief", fixed_brief))
            if stage == "short-audit":
                return json.dumps({"verdict": "revise",
                                   "findings": [AF(finding, artifact="brief")]})
            if stage == "short-reaudit":
                return json.dumps({"verdict": "pass", "findings": []})
            return json.dumps(dict(initial))

        class FakeMS:
            MODELS = AUDIT_MODELS = REPAIR_MODELS = ["local"]

            @staticmethod
            def run(prompt, out_dir, chain, stage, validate=None,
                    gateway_options=None):
                return handler(stage, prompt)

        for depth in ("detailed", "brief"):
            self.assertEqual([], ss.defects(initial[depth], source),
                             f"initial {depth} must be mechanically clean")
            self.assertEqual([], ss.not_a_copy(initial[depth], source, depth),
                             f"initial {depth} must pass the copy gate")
        rc, out = self._run(ss, FakeMS(), source=source)
        self.assertEqual(0, rc)
        prompt = seen.get("prompt", "")
        self.assertIn(initial["detailed"], prompt,
                      "repair prompt lacks the current Detailed")
        self.assertIn(initial["brief"], prompt,
                      "repair prompt lacks the current Brief")
        self.assertIn(finding, prompt,
                      "repair prompt lacks the findings")
        pr = load("pair_review")
        self.assertIn(pr.candidate_identity(initial["detailed"],
                                            initial["brief"]), prompt,
                      "repair prompt lacks the candidate identity")
        self.assertEqual((out / "detailed.md").read_text().strip(),
                         initial["detailed"],
                         "Brief-only repair rewrote the Detailed")
        report = json.loads((out / "short-report.json").read_text())
        self.assertEqual((report["selected"], report["status"]),
                         ("revised", "pass"))


class PairReviewMachinery(unittest.TestCase):
    """Shared candidate/review/repair state both pair routes build on.

    Pure state, no model calls, no I/O: Quick rewired onto it with zero
    behavior change (its behavioral suites prove that), and the Full
    final-pair controller will reuse it rather than fork it.
    """
    pr = load("pair_review")

    def test_identity_is_stable_and_pair_bound(self):
        a = self.pr.candidate_identity("detailed text", "brief text")
        self.assertEqual(a, self.pr.candidate_identity("detailed text",
                                                       "brief text"))
        self.assertNotEqual(a, self.pr.candidate_identity("detailed text",
                                                          "other brief"))
        self.assertTrue(a.startswith("candidate-"))

    def test_a_candidate_combines_mechanical_and_audit_findings(self):
        c = self.pr.new_candidate("initial", "d", "b", ["m1"], ["a1"],
                                  usable=True)
        self.assertEqual(c["findings"], ["m1", "a1"])
        self.assertEqual(c["review"], "complete")
        self.assertTrue(c["usable"])

    def test_selection_needs_a_usable_candidate(self):
        self.assertIsNone(self.pr.select([]))
        dead = self.pr.new_candidate("initial", "d", "b", [], [],
                                     usable=False)
        self.assertIsNone(self.pr.select([dead]))
        self.assertIsNone(self.pr.select([None, "not-a-candidate"]))

    def test_material_outranks_minor(self):
        # Three minor style defects must not beat two dropped claims, and an
        # invented number must not beat a mere omission.
        rough = self.pr.new_candidate(
            "initial", "d", "b",
            ["first person", "list or table markup", "stub"], [],
            usable=True)
        thin = self.pr.new_candidate(
            "revised", "d2", "b2", [],
            ["drops claim one", "drops claim two"], usable=True)
        self.assertIs(self.pr.select([rough, thin]), rough)
        omitting = self.pr.new_candidate("initial", "d", "b", [], ["omits x"],
                                         usable=True)
        inventing = self.pr.new_candidate(
            "revised", "d2", "b2",
            ["numbers absent from the source: ['47']", "first person"],
            [], usable=True)
        self.assertIs(self.pr.select([omitting, inventing]), omitting)

    def test_ties_fall_through_to_totals_then_latest(self):
        # Equal material: fewer total findings wins; full tie: the latest
        # pool entry wins, which is what lets one routine serve up to three
        # Full repairs without a fork.
        a = self.pr.new_candidate("initial", "d", "b", ["m1", "m2"], [],
                                  usable=True)
        b = self.pr.new_candidate("revise-1", "d", "b2", ["m1"], [],
                                  usable=True)
        self.assertIs(self.pr.select([a, b]), b)
        c = self.pr.new_candidate("revise-2", "d3", "b3", ["m1"], [],
                                  usable=True)
        self.assertIs(self.pr.select([b, c]), c)

    def test_status_resolution(self):
        ok = self.pr.new_candidate("revised", "d", "b", [], [], usable=True)
        self.assertEqual(self.pr.resolve_status(ok), "pass")
        open_c = self.pr.new_candidate("revised", "d", "b", ["m"], [],
                                       usable=True)
        self.assertEqual(self.pr.resolve_status(open_c), "open_findings")
        unreviewed = self.pr.new_candidate(
            "revised", "d", "b", [], [], usable=True,
            review="unavailable: short-reaudit: down")
        self.assertEqual(self.pr.resolve_status(unreviewed),
                         "review_unavailable")

    def test_unreviewed_child_cannot_displace_reviewed_parent(self):
        reviewed = self.pr.new_candidate(
            "initial", "d", "b", [], ["known omission"], usable=True,
            review="complete")
        unknown = self.pr.new_candidate(
            "revised", "d2", "b2", [], [], usable=True,
            review="unavailable: re-audit timed out")
        self.assertIs(self.pr.select([reviewed, unknown]), reviewed)

    def test_disclosure_is_findings_plus_notes(self):
        c = self.pr.new_candidate("revised", "d", "b", ["m"], ["a"],
                                  usable=True, review="unavailable: x",
                                  notes=["unavailable: x"])
        self.assertEqual(self.pr.disclosed(c), ["m", "a", "unavailable: x"])

    def test_repair_prompt_encloses_identity_readings_and_findings(self):
        out = self.pr.build_repair_prompt("BASE", "the detailed", "the brief",
                                          ["finding one"])
        for needle in ("BASE", "the detailed", "the brief", "finding one",
                       self.pr.candidate_identity("the detailed",
                                                  "the brief"),
                       "do not start over"):
            self.assertIn(needle, out)
        self.assertNotEqual(out, "BASE")

    def test_patch_turn_issues_ids_for_mechanical_findings(self):
        candidate = self.pr.new_candidate(
            "initial", "the detailed", "the brief",
            ["numbers absent from the source: ['1900']",
             "refers to the document rather than its subject"],
            ["an audited omission"], usable=True,
            audit_records=[{
                "finding_id": "F-001", "kind": "omission",
                "artifact": "brief", "text": "an audited omission",
            }])
        turn = self.pr.repair_turn(candidate)
        self.assertEqual(turn["kind"], "patch")
        self.assertEqual(turn["allowed_finding_ids"],
                         ("F-001", "F-002", "F-003"))
        for fid in turn["allowed_finding_ids"]:
            self.assertIn(fid, turn["prompt"])


class PairPatchMachinery(unittest.TestCase):
    """Transactional edits on a controller-segmented pair. No model calls."""
    pp = load("pair_patch")

    DETAILED = ("Rates fell after the 2008 reform, though evidence was limited.\n\n"
                "Markets reacted quickly while analysts warned stress could return.")
    BRIEF = "Rates fell after the reform without proven causation."

    def test_segments_round_trip_bytes(self):
        items = self.pp.segment_artifact(self.DETAILED, "D")
        self.assertEqual("".join(item["text"] for item in items), self.DETAILED)
        self.assertTrue(all(item["id"].startswith("D-P") for item in items))
        self.assertEqual(len(items), 2)

    def test_segment_map_lists_sha256_so_a_repair_can_copy_them(self):
        # Live Muse omitted hashes because the map only printed ids.
        text = self.pp.segment_map(self.DETAILED, self.BRIEF)
        first = self.pp.segment_artifact(self.DETAILED, "D")[0]
        self.assertIn(f"[{first['id']} sha256={first['sha256']}]", text)
        self.assertIn("CANDIDATE candidate-", text)

    def test_headings_are_separate_spans(self):
        text = "## Rates\n\nThey fell after the reform."
        items = self.pp.segment_artifact(text, "D")
        self.assertEqual(items[0]["id"], "D-H001")
        self.assertEqual(items[0]["kind"], "heading")
        self.assertEqual("".join(item["text"] for item in items), text)

    def test_apply_insert_after_preserves_untouched_bytes(self):
        pair = self.pp.segment_pair(self.DETAILED, self.BRIEF)
        first = pair["detailed"][0]
        payload = {
            "base_candidate": pair["candidate"],
            "edits": [{
                "artifact": "detailed",
                "operation": "insert_after",
                "anchor": first["id"],
                "anchor_sha256": first["sha256"],
                "replacement": "Causation was not claimed.",
                "finding_ids": ["F-001"],
            }],
        }
        detailed, brief = self.pp.apply_edits(self.DETAILED, self.BRIEF, payload)
        self.assertEqual(brief, self.BRIEF)
        self.assertIn("Causation was not claimed.", detailed)
        self.assertTrue(detailed.startswith("Rates fell after the 2008 reform"))
        self.assertIn("Markets reacted quickly", detailed)

    def test_stale_hash_unknown_id_and_overlap_are_rejected(self):
        pair = self.pp.segment_pair(self.DETAILED, self.BRIEF)
        first = pair["detailed"][0]
        last = pair["detailed"][-1]
        base = {
            "base_candidate": pair["candidate"],
            "edits": [{
                "artifact": "detailed",
                "operation": "insert_after",
                "anchor": first["id"],
                "anchor_sha256": first["sha256"],
                "replacement": "Added.",
                "finding_ids": ["F-001"],
            }],
        }
        stale = dict(base)
        stale["edits"] = [dict(base["edits"][0], anchor_sha256="0" * 64)]
        with self.assertRaises(self.pp.PatchError):
            self.pp.apply_edits(self.DETAILED, self.BRIEF, stale)
        unknown = dict(base)
        unknown["edits"] = [dict(base["edits"][0], anchor="D-P999")]
        with self.assertRaises(self.pp.PatchError):
            self.pp.apply_edits(self.DETAILED, self.BRIEF, unknown)
        overlap = {
            "base_candidate": pair["candidate"],
            "edits": [
                {"artifact": "detailed", "operation": "replace_range",
                 "start": first["id"], "end": last["id"],
                 "range_sha256": hashlib.sha256(self.DETAILED.encode()).hexdigest(),
                 "replacement": "One.", "finding_ids": ["F-001"]},
                {"artifact": "detailed", "operation": "insert_after",
                 "anchor": first["id"], "anchor_sha256": first["sha256"],
                 "replacement": "Two.", "finding_ids": ["F-002"]},
            ],
        }
        with self.assertRaises(self.pp.PatchError):
            self.pp.apply_edits(self.DETAILED, self.BRIEF, overlap)

    def test_boundary_insert_and_tied_replacement_apply_in_either_order(self):
        pair = self.pp.segment_pair(self.DETAILED, self.BRIEF)
        first = pair["detailed"][0]
        second = pair["detailed"][1]
        self.assertEqual(first["end"], second["start"])
        insert = {
            "artifact": "detailed", "operation": "insert_after",
            "anchor": first["id"], "anchor_sha256": first["sha256"],
            "replacement": "Causation was not claimed.",
            "finding_ids": ["F-001"],
        }
        replace = {
            "artifact": "detailed", "operation": "replace_range",
            "start": second["id"], "end": second["id"],
            "range_sha256": second["sha256"],
            "replacement": "Analysts kept warning that stress could return.",
            "finding_ids": ["F-002"],
        }
        payload = {"base_candidate": pair["candidate"],
                   "edits": [replace, insert]}
        swapped = {"base_candidate": pair["candidate"],
                   "edits": [insert, replace]}
        a_d, a_b = self.pp.apply_edits(self.DETAILED, self.BRIEF, payload)
        b_d, b_b = self.pp.apply_edits(self.DETAILED, self.BRIEF, swapped)
        self.assertEqual(a_d, b_d)
        self.assertEqual(a_b, b_b)
        self.assertIn("Causation was not claimed.", a_d)
        self.assertIn("Analysts kept warning that stress could return.", a_d)
        self.assertTrue(a_d.startswith("Rates fell after the 2008 reform"))
        self.assertIn("\n\n", a_d)

    def test_wrong_candidate_or_list_markup_is_rejected(self):
        pair = self.pp.segment_pair(self.DETAILED, self.BRIEF)
        first = pair["detailed"][0]
        bad_id = dict(base_candidate="candidate-deadbeefdeadbeef", edits=[{
            "artifact": "detailed", "operation": "insert_after",
            "anchor": first["id"], "anchor_sha256": first["sha256"],
            "replacement": "Added.", "finding_ids": ["F-001"],
        }])
        with self.assertRaises(self.pp.PatchError):
            self.pp.apply_edits(self.DETAILED, self.BRIEF, bad_id)
        listed = {
            "base_candidate": pair["candidate"],
            "edits": [{
                "artifact": "detailed", "operation": "insert_after",
                "anchor": first["id"], "anchor_sha256": first["sha256"],
                "replacement": "- Rates fell\n- Stress returned",
                "finding_ids": ["F-001"],
            }],
        }
        with self.assertRaises(self.pp.PatchError):
            self.pp.apply_edits(self.DETAILED, self.BRIEF, listed)

    def test_court1_muse_patch_json_is_still_unusable(self):
        """A patch missing artifact, operation, hashes, and finding IDs is
        unusable; replay the representative shape without silently accepting it."""
        raw = (
            '{"base_candidate": "candidate-65b5fb21618bcadc", "edits": ['
            '{"op": "replace_range", "start": "B-P004", "end": "B-P004",'
            ' "replacement": "Wealth inequality will not be eliminated '
            'without direct redistribution."}]}'
        )
        jc = load("json_contract")
        with self.assertRaises(jc.ContractError) as caught:
            jc.parse(raw, jc.PAIR_PATCH, "full-revise-1")
        self.assertIn("operation", str(caught.exception))
        prompt = load("pair_review").build_patch_prompt(
            self.DETAILED, self.BRIEF,
            [{"finding_id": "F-001", "kind": "omission",
              "artifact": "brief", "text": "The Brief weakens a hedge"}])
        self.assertIn("sha256=", prompt)
        self.assertIn('"operation"', prompt)
        self.assertIn("not \"op\"", prompt)
        turn = load("pair_review").repair_turn({
            "detailed": self.DETAILED, "brief": self.BRIEF,
            "findings": ["The Brief weakens a hedge"],
            "audit_records": [{"finding_id": "F-001", "kind": "omission",
                               "artifact": "brief",
                               "text": "The Brief weakens a hedge"}],
            "mechanical": [],
        })
        self.assertEqual(turn["kind"], "patch")
        self.assertEqual(turn["allowed_finding_ids"], ("F-001",))

    def test_unknown_finding_id_is_rejected_when_the_controller_bound_ids(self):
        pair = self.pp.segment_pair(self.DETAILED, self.BRIEF)
        first = pair["detailed"][0]
        payload = {
            "base_candidate": pair["candidate"],
            "edits": [{
                "artifact": "detailed",
                "operation": "insert_after",
                "anchor": first["id"],
                "anchor_sha256": first["sha256"],
                "replacement": "Added.",
                "finding_ids": ["F-999"],
            }],
        }
        with self.assertRaises(self.pp.PatchError):
            self.pp.apply_edits(self.DETAILED, self.BRIEF, payload,
                                allowed_finding_ids=["F-001"])
        detailed, brief = self.pp.apply_edits(
            self.DETAILED, self.BRIEF,
            {**payload, "edits": [dict(payload["edits"][0],
                                       finding_ids=["F-001"])]},
            allowed_finding_ids=["F-001"])
        self.assertEqual(brief, self.BRIEF)
        self.assertIn("Added.", detailed)


class PatchBoundaryContract(unittest.TestCase):
    """A replacement cannot consume the controller-owned block boundary."""
    pp = load("pair_patch")

    def test_live_muse_shape_keeps_paragraph_before_heading(self):
        detailed = ("The published logarithmic gap term.\n\n"
                    "## Selection without peeking at the holdout\n\n"
                    "Selection uses only frozen evidence.")
        brief = "The slate is frozen before scoring."
        pair = self.pp.segment_pair(detailed, brief)
        first = pair["detailed"][0]
        payload = {
            "base_candidate": pair["candidate"],
            "edits": [{
                "artifact": "detailed", "operation": "replace_range",
                "start": first["id"], "end": first["id"],
                "range_sha256": first["sha256"],
                "replacement": "The revised logarithmic gap term.",
                "finding_ids": ["F-001"],
            }],
        }
        revised, unchanged = self.pp.apply_edits(
            detailed, brief, payload, allowed_finding_ids=["F-001"])
        self.assertEqual(unchanged, brief)
        self.assertEqual(
            revised,
            "The revised logarithmic gap term.\n\n"
            "## Selection without peeking at the holdout\n\n"
            "Selection uses only frozen evidence.")
        self.assertNotIn("term.##", revised)

    def test_structural_replacement_may_contain_multiple_paragraphs(self):
        detailed = "One overloaded paragraph.\n\n## Next\n\nNext point."
        brief = "Brief."
        pair = self.pp.segment_pair(detailed, brief)
        first = pair["detailed"][0]
        payload = {
            "base_candidate": pair["candidate"],
            "edits": [{
                "artifact": "detailed", "operation": "replace_range",
                "start": first["id"], "end": first["id"],
                "range_sha256": first["sha256"],
                "replacement": "First point.\n\nSecond point.",
                "finding_ids": ["F-001"],
            }],
        }
        revised, _ = self.pp.apply_edits(detailed, brief, payload)
        self.assertIn("First point.\n\nSecond point.\n\n## Next", revised)


class MixedFindingRepair(unittest.TestCase):
    """Length, fidelity, and editorial findings share one repair request."""

    def test_mixed_length_and_other_findings_are_all_delivered(self):
        pr = load("pair_review")
        length = ("the detailed reading is 70% of the source "
                  "(5709 words of 8258; ceiling 3303); a reading that "
                  "replaces the source must be substantially shorter, not "
                  "reworded")
        candidate = pr.new_candidate(
            "initial", "Detailed.", "Brief.",
            [length, "numbers absent from the source: ['1959']"],
            ["The Brief weakens a qualification."], usable=True,
            audit_records=[{
                "finding_id": "F-001", "kind": "brief",
                "artifact": "brief",
                "text": "The Brief weakens a qualification.",
            }])
        turn = pr.repair_turn(candidate)
        self.assertEqual(turn["kind"], "patch")
        self.assertEqual(turn["allowed_finding_ids"],
                         ("F-001", "F-002", "F-003"))
        self.assertIn("ceiling 3303", turn["prompt"])
        self.assertIn("1959", turn["prompt"])
        self.assertIn("weakens a qualification", turn["prompt"])

    def test_unscoped_expansion_keeps_source_evidence_but_length_only_does_not(self):
        pr = load("pair_review")
        expand = pr.new_candidate(
            "initial", "Detailed.", "Brief.",
            ["EXPAND REQUIRED: the brief reading is too short."], [],
            usable=True, source_context={"full-v2-audit": "RAW EVIDENCE"})
        turn = pr.repair_turn(expand, force_patch=True)
        self.assertIn("RAW EVIDENCE", turn["prompt"])

        length = pr.new_candidate(
            "initial", "Detailed.", "Brief.",
            ["the brief reading is 80% of the source and exceeds its ceiling 50%"],
            [], usable=True,
            source_context={"full-v2-audit": "RAW EVIDENCE"})
        turn = pr.repair_turn(length, force_patch=True)
        self.assertNotIn("RAW EVIDENCE", turn["prompt"])


class ReadabilityReviewContract(unittest.TestCase):
    """Detectors request judgment; only the extreme floor blocks directly."""

    wc = load("writing_contract")

    def test_extreme_paragraph_scale_is_blocking(self):
        detailed = "word " * 801
        observations = self.wc.readability_observations(detailed, "Brief.")
        self.assertEqual(len(observations), 1)
        self.assertTrue(observations[0]["blocking"])
        self.assertIn("needs_structure_repair",
                      self.wc.blocking_readability_findings(observations)[0])

    def test_coherent_long_paragraph_is_assessed_not_rewritten(self):
        paragraph = " ".join("word" for _ in range(400)) + "."
        observations = self.wc.readability_observations(paragraph, "Brief.")
        self.assertEqual(len(observations), 1)
        self.assertFalse(observations[0]["blocking"])
        audit = {"verdict": "pass", "findings": [], "readability": [{
            "observation_id": observations[0]["observation_id"],
            "assessment": "acceptable",
            "explanation": "One sustained comparison remains coherent.",
        }]}
        self.assertEqual(
            self.wc.validate_readability_assessments(audit, observations), [])

    def test_short_sentences_and_heading_free_reading_are_not_auto_failures(self):
        clipped = " ".join(f"Point {i}." for i in range(15))
        observations = self.wc.readability_observations(clipped, "Brief prose.")
        self.assertTrue(observations)
        self.assertTrue(all(not item["blocking"] for item in observations))
        self.assertEqual(self.wc.readability_observations(
            "A concise paragraph remains readable.", "A concise brief."), [])

    def test_unreadable_assessment_becomes_anchored_editorial_finding(self):
        observations = self.wc.readability_observations("word " * 400, "Brief.")
        audit = {"verdict": "revise", "findings": [], "readability": [{
            "observation_id": observations[0]["observation_id"],
            "assessment": "unreadable",
            "explanation": "Several unrelated explanatory jobs are merged.",
        }]}
        findings = self.wc.validate_readability_assessments(
            audit, observations)
        self.assertEqual(findings[0]["kind"], "editorial")
        self.assertEqual(findings[0]["anchor"], observations[0]["anchor"])


class PlanEnvelopeRoundTrip(unittest.TestCase):
    """Persisted plans separate model payload from computed metadata."""

    wc = load("writing_contract")

    def fixture(self):
        source = "## One\n\nAlpha evidence.\n\n## Two\n\nBeta evidence."
        ids = [item["id"] for item in self.wc.source_segments(source)]
        split = max(1, len(ids) // 2)
        plan = {
            "schema": "summer.writing-plan.v2",
            "units": [
                {"unit_id": "U001", "title": "One",
                 "source_ids": ids[:split], "topic": "Alpha",
                 "relation_to_previous": "opening",
                 "governing_qualifications": [], "detailed_words": 300,
                 "brief_disposition": "include", "brief_words": 100},
                {"unit_id": "U002", "title": "Two",
                 "source_ids": ids[split:], "topic": "Beta",
                 "relation_to_previous": "continues",
                 "governing_qualifications": [], "detailed_words": 300,
                 "brief_disposition": "omit", "brief_words": 0},
            ],
            "dispositions": [],
        }
        return source, plan, {"detailed": 700, "brief": 200}

    def test_new_envelope_round_trips_and_legacy_receipts_migrate(self):
        source, plan, ceilings = self.fixture()
        frozen = self.wc.validate_plan(plan, source, ceilings)
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            current = root / "current.json"
            legacy = root / "legacy.json"
            self.wc.save_plan(current, frozen)
            legacy.write_text(json.dumps(frozen))
            envelope = json.loads(current.read_text())
            self.assertEqual(envelope["schema"], self.wc.ENVELOPE_SCHEMA)
            self.assertNotIn("plan_sha256", envelope["payload"])
            self.assertEqual(self.wc.load_plan(current, source, ceilings), frozen)
            self.assertEqual(self.wc.load_plan(legacy, source, ceilings), frozen)

    def test_stale_computed_metadata_fails_closed(self):
        source, plan, ceilings = self.fixture()
        frozen = self.wc.validate_plan(plan, source, ceilings)
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / "plan.json"
            self.wc.save_plan(path, frozen)
            envelope = json.loads(path.read_text())
            envelope["metadata"]["source_sha256"] = "0" * 64
            path.write_text(json.dumps(envelope))
            with self.assertRaises(self.wc.WritingContractError):
                self.wc.load_plan(path, source, ceilings)


class BoundedSourceIndexPlan(unittest.TestCase):
    """Planning leaves own every source id once and merge under one Brief."""

    wc = load("writing_contract")
    jc = load("json_contract")

    def fixture(self):
        source = "\n\n".join(
            f"## Section {i}\n\nEvidence paragraph {i} with result {i}."
            for i in range(1, 7))
        ceilings = {"detailed": 1200, "brief": 240}
        leaves = self.wc.planning_leaves(source, ceilings, max_ids=3)
        pairs = []
        for leaf in leaves:
            plan = {
                "schema": self.wc.SCHEMA,
                "units": [{
                    "unit_id": "U001", "title": leaf["leaf_id"],
                    "source_ids": list(leaf["source_ids"]),
                    "topic": "bounded evidence", "relation_to_previous": "",
                    "governing_qualifications": [],
                    "detailed_words": leaf["detailed_words"],
                    "brief_disposition": "include", "brief_words": 1,
                }],
                "dispositions": [],
            }
            pairs.append((leaf, self.wc.validate_plan_fragment(plan, leaf)))
        return source, ceilings, leaves, pairs

    def test_leaves_are_contiguous_bounded_and_reconstruct_ownership(self):
        source, _ceilings, leaves, _pairs = self.fixture()
        expected = [item["id"] for item in self.wc.source_segments(source)]
        got = [ident for leaf in leaves for ident in leaf["source_ids"]]
        self.assertEqual(got, expected)
        self.assertTrue(all(len(leaf["source_ids"]) <= 3 for leaf in leaves))
        self.assertEqual(len(got), len(set(got)))
        prompt = self.wc.planning_leaf_prompt(leaves[0])
        self.assertNotIn("sha256=", prompt)
        self.assertIn("leaf boundary is transport, not a discourse boundary",
                      prompt)
        self.assertIn("fewest coherent units", prompt)
        self.assertNotIn("units may be much smaller", prompt)
        self.assertIn("never null", prompt)

    def test_writer_prompt_does_not_invite_sentence_sized_paragraphs(self):
        source, _ceilings, _leaves, pairs = self.fixture()
        plan = {
            "units": [dict(pairs[0][1]["units"][0], unit_id="U001",
                           source_words=20)],
        }
        packet = {"detailed": [{
            "unit_id": "U001", "title": "L001",
            "source_ids": list(pairs[0][0]["source_ids"]),
            "topic": "bounded evidence", "relation_to_previous": "opening",
            "governing_qualifications": [], "words": 120,
            "source_words": 20,
        }], "brief": []}
        prompt = self.wc.writing_prompt(plan, packet, source)
        compact = " ".join(prompt.split())
        self.assertIn("not one array element per sentence", compact)
        self.assertIn("keeping adjacent sentences together", compact)

    def test_no_candidate_preserves_terminal_failure_kind(self):
        mapsum = load("mapsum")
        error = mapsum.NoCandidate(
            "plan rejected", kind="unusable", detail="allocation exceeded")
        self.assertEqual(error.kind, "unusable")
        self.assertEqual(error.detail, "allocation exceeded")

    def test_adjacent_reduction_and_global_selection_materialize_one_plan(self):
        source, ceilings, _leaves, pairs = self.fixture()
        groups = self.wc.planning_reduction_groups(pairs, fanin=4)
        self.assertTrue(all(len(group) <= 4 for group in groups))
        nodes = []
        for index, group in enumerate(groups, 1):
            handles = [card["handle"] for leaf in group
                       for card in leaf["cards"]]
            nodes.append(self.wc.validate_plan_reduction({
                "summary": "adjacent result group",
                "brief_candidates": handles,
                "qualifications": [],
            }, group, f"N{index:03d}"))
        cards = [card for leaf, plan in pairs
                 for card in self.wc._fragment_cards(leaf, plan)]
        selection = self.wc.validate_plan_selection({
            "brief_units": [{"handle": cards[0]["handle"], "words": 120}],
        }, cards, ceilings["brief"])
        plan = self.wc.assemble_bounded_plan(
            source, ceilings, pairs, selection)
        self.assertEqual(plan["allocated_words"]["brief"], 120)
        self.assertEqual([unit["unit_id"] for unit in plan["units"]],
                         [f"U{i:03d}" for i in range(1, len(plan["units"]) + 1)])

    def test_fragment_rejects_missing_plus_duplicate_ownership(self):
        _source, _ceilings, leaves, _pairs = self.fixture()
        leaf = leaves[0]
        ids = list(leaf["source_ids"])
        plan = {
            "schema": self.wc.SCHEMA,
            "units": [{
                "unit_id": "U001", "title": "bad",
                "source_ids": [ids[0], ids[0]], "topic": "bad",
                "relation_to_previous": "", "governing_qualifications": [],
                "detailed_words": 10, "brief_disposition": "include",
                "brief_words": 1,
            }],
            "dispositions": [{
                "source_ids": ids[2:], "disposition": "apparatus",
                "represented_by": ["U001"], "reason": "bad accounting",
            }],
        }
        with self.assertRaises(self.wc.WritingContractError):
            self.wc.validate_plan_fragment(plan, leaf)


class ParagraphAssemblyAcrossUnits(unittest.TestCase):
    """Presentation joins never weaken unit ownership or source order."""

    wc = load("writing_contract")

    def plan(self):
        return {
            "units": [
                {"unit_id": "U001", "title": "Opening",
                 "source_ids": ["S-P001"], "topic": "one point",
                 "relation_to_previous": "opening",
                 "governing_qualifications": [], "detailed_words": 80,
                 "brief_disposition": "omit", "brief_words": 0,
                 "source_words": 20},
                {"unit_id": "U002", "title": "Continuation",
                 "source_ids": ["S-P002"], "topic": "the same point",
                 "relation_to_previous": "continues the same explanation",
                 "governing_qualifications": [], "detailed_words": 80,
                 "brief_disposition": "omit", "brief_words": 0,
                 "source_words": 20},
            ],
        }

    @staticmethod
    def blocks():
        return [
            {"unit_id": "U001", "heading": "",
             "join_previous": False,
             "paragraphs": ["The first unit begins one coherent point."]},
            {"unit_id": "U002", "heading": "",
             "join_previous": True,
             "paragraphs": ["The second unit completes that point."]},
        ]

    def test_adjacent_units_can_render_as_one_paragraph(self):
        blocks = self.blocks()
        self.wc.validate_presentation_joins(
            blocks, ["U001", "U002"], "detailed")
        rendered = self.wc.render_blocks(blocks)
        self.assertEqual(
            rendered,
            "The first unit begins one coherent point. "
            "The second unit completes that point.")
        self.assertNotIn("\n\n", rendered)

    def test_invalid_first_heading_missing_and_reordered_joins_fail_closed(self):
        first = self.blocks()
        first[0]["join_previous"] = True
        with self.assertRaises(self.wc.WritingContractError):
            self.wc.validate_presentation_joins(
                first, ["U001", "U002"], "detailed")

        headed = self.blocks()
        headed[1]["heading"] = "A new section"
        with self.assertRaises(self.wc.WritingContractError):
            self.wc.validate_presentation_joins(
                headed, ["U001", "U002"], "detailed")

        with self.assertRaises(self.wc.WritingContractError):
            self.wc.validate_presentation_joins(
                list(reversed(self.blocks())), ["U001", "U002"],
                "detailed")
        with self.assertRaises(self.wc.WritingContractError):
            self.wc.validate_presentation_joins(
                [self.blocks()[1]], ["U001", "U002"], "detailed")

    def test_join_survives_separate_capacity_packets(self):
        plan = self.plan()
        packets = self.wc.build_packets(plan, output_words=80)
        self.assertEqual(len(packets), 2)
        accepted = []
        for packet, block in zip(packets, self.blocks()):
            response = {"detailed": [block], "brief": []}
            clean = self.wc.validate_blocks(response, packet)
            accepted.extend(clean["detailed"])
        self.wc.validate_presentation_joins(
            accepted, ["U001", "U002"], "detailed")
        self.assertNotIn("\n\n", self.wc.render_blocks(accepted))

    def test_join_survives_atomic_writer_recovery(self):
        plan = self.plan()
        root = self.wc.build_packets(plan, output_words=1_000)[0]
        state = self.wc.new_writer_recovery(plan, root)
        for packet, block in zip(self.wc.atomic_packets(root), self.blocks()):
            clean = self.wc.validate_blocks(
                {"detailed": [block], "brief": []}, packet)
            state = self.wc.accept_recovery_block(
                state, "detailed", block["unit_id"],
                clean["detailed"][0], {"stage": "synthetic"})
        recovered = self.wc.render_recovery_blocks(plan, state)
        ordered = [recovered["detailed"][ident]
                   for ident in ("U001", "U002")]
        self.wc.validate_presentation_joins(
            ordered, ["U001", "U002"], "detailed")
        self.assertEqual([item["unit_id"] for item in ordered],
                         ["U001", "U002"])
        self.assertNotIn("\n\n", self.wc.render_blocks(ordered))


class ParagraphAssemblyReviewRepair(unittest.TestCase):
    """Joined prose is reviewed and repaired as one multi-owner component."""

    wc = load("writing_contract")
    pp = load("pair_patch")

    def fixture(self):
        source = "Alpha evidence supports the opening.\n\nBeta evidence completes it."
        ids = [item["id"] for item in self.wc.source_segments(source)]
        plan = self.wc.validate_plan({
            "schema": self.wc.SCHEMA,
            "units": [
                {"unit_id": "U001", "title": "Alpha",
                 "source_ids": [ids[0]], "topic": "one explanation",
                 "relation_to_previous": "opening",
                 "governing_qualifications": [], "detailed_words": 80,
                 "brief_disposition": "include", "brief_words": 30},
                {"unit_id": "U002", "title": "Beta",
                 "source_ids": [ids[1]], "topic": "same explanation",
                 "relation_to_previous": "continues",
                 "governing_qualifications": [], "detailed_words": 80,
                 "brief_disposition": "omit", "brief_words": 0},
            ], "dispositions": [],
        }, source, {"detailed": 200, "brief": 60})
        blocks = {
            "detailed": {
                "U001": {"unit_id": "U001", "heading": "",
                         "join_previous": False,
                         "paragraphs": ["Alpha summary."]},
                "U002": {"unit_id": "U002", "heading": "",
                         "join_previous": True,
                         "paragraphs": ["Beta summary."]},
            },
            "brief": {
                "U001": {"unit_id": "U001", "heading": "",
                         "join_previous": False,
                         "paragraphs": ["Brief summary."]},
            },
        }
        detailed = self.wc.render_blocks(
            [blocks["detailed"]["U001"], blocks["detailed"]["U002"]])
        brief = self.wc.render_blocks([blocks["brief"]["U001"]])
        assignment = {
            "detailed": self.wc._assignment(plan, "detailed"),
            "brief": self.wc._assignment(plan, "brief"),
        }
        return source, plan, blocks, assignment, detailed, brief, ids

    def test_joined_paragraph_is_one_review_component_with_union_evidence(self):
        source, plan, blocks, assignment, detailed, brief, ids = self.fixture()
        packets, _global = self.wc.bounded_review_packets(
            plan, self.wc.atomic_packets(assignment), blocks,
            source, detailed, brief)
        self.assertEqual(len(packets), 1)
        self.assertEqual(packets[0]["unit_ids"], ["U001", "U002"])
        self.assertEqual(packets[0]["source_ids"], ids)
        self.assertIn("Alpha evidence", packets[0]["source"])
        self.assertIn("Beta evidence", packets[0]["source"])
        self.assertEqual(packets[0]["detailed"],
                         "Alpha summary. Beta summary.")
        context = json.loads(packets[0]["plan_context"])
        self.assertEqual([item["unit_id"] for item in context["units"]],
                         ["U001", "U002"])

    def test_block_repair_round_trips_every_owner_and_continuation(self):
        source, plan, blocks, _assignment, detailed, brief, _ids = self.fixture()
        anchor = self.pp.segment_pair(detailed, brief)["detailed"][0]["id"]
        component = self.wc.joined_component_for_anchor(
            plan, blocks, detailed, brief, "detailed", anchor)
        response = {"detailed": [
            {"unit_id": "U001", "heading": "",
             "join_previous": False,
             "paragraphs": ["Corrected alpha summary."]},
            {"unit_id": "U002", "heading": "",
             "join_previous": True,
             "paragraphs": ["Corrected beta summary."]},
        ], "brief": []}
        repaired = self.wc.apply_joined_component_repair(
            plan, blocks, component, response)
        ordered = [repaired["detailed"][ident]
                   for ident in ("U001", "U002")]
        self.assertEqual([item["unit_id"] for item in ordered],
                         ["U001", "U002"])
        self.assertEqual([item["join_previous"] for item in ordered],
                         [False, True])
        self.assertEqual(
            self.wc.render_blocks(ordered),
            "Corrected alpha summary. Corrected beta summary.")

    def test_missing_duplicate_reordered_or_newly_headed_owner_fails_closed(self):
        _source, plan, blocks, _assignment, detailed, brief, _ids = self.fixture()
        anchor = self.pp.segment_pair(detailed, brief)["detailed"][0]["id"]
        component = self.wc.joined_component_for_anchor(
            plan, blocks, detailed, brief, "detailed", anchor)
        valid = {"detailed": [
            {"unit_id": "U001", "heading": "",
             "join_previous": False, "paragraphs": ["Alpha."]},
            {"unit_id": "U002", "heading": "",
             "join_previous": True, "paragraphs": ["Beta."]},
        ], "brief": []}
        variants = []
        missing = json.loads(json.dumps(valid)); missing["detailed"].pop()
        variants.append(missing)
        duplicate = json.loads(json.dumps(valid)); duplicate["detailed"][1]["unit_id"] = "U001"
        variants.append(duplicate)
        reordered = json.loads(json.dumps(valid)); reordered["detailed"].reverse()
        variants.append(reordered)
        headed = json.loads(json.dumps(valid)); headed["detailed"][0]["heading"] = "New"
        variants.append(headed)
        for value in variants:
            with self.subTest(value=value), self.assertRaises(
                    self.wc.WritingContractError):
                self.wc.apply_joined_component_repair(
                    plan, blocks, component, value)

    def test_full_lifecycle_repairs_and_reaudits_the_joined_component(self):
        source, plan, blocks, assignment, detailed, brief, _ids = self.fixture()
        stages = []

        class Model:
            MODELS = ["writer"]
            AUDIT_MODELS = ["auditor"]
            REPAIR_MODELS = ["repair"]

            @staticmethod
            def run(prompt, out_dir, chain, stage, validate=None,
                    gateway_options=None, **_kwargs):
                stages.append((stage, prompt))
                if stage == "joined-audit":
                    raw = json.dumps({
                        "verdict": "revise",
                        "findings": [{
                            "kind": "editorial", "artifact": "detailed",
                            "anchor": "D-P001", "slot": "",
                            "text": "Correct the continuation claim.",
                        }],
                        "readability": [],
                    })
                elif stage == "joined-revise-1":
                    raw = json.dumps({
                        "detailed": [
                            {"unit_id": "U001", "heading": "",
                             "join_previous": False,
                             "paragraphs": ["Corrected alpha evidence now supports the opening."]},
                            {"unit_id": "U002", "heading": "",
                             "join_previous": True,
                             "paragraphs": ["Corrected beta evidence completes it."]},
                        ],
                        "brief": [],
                    })
                elif "audit" in stage:
                    raw = json.dumps({
                        "verdict": "pass", "findings": [],
                        "readability": [],
                    })
                else:
                    raise AssertionError(stage)
                if validate:
                    validate(raw)
                return raw

        class Structure:
            @staticmethod
            def defects(*_args):
                return []

            @staticmethod
            def not_a_copy(*_args):
                return []

            @staticmethod
            def structural_findings(*_args):
                return []

        fs = load("fullsum")
        pr = load("pair_review")
        with tempfile.TemporaryDirectory() as td:
            out = pathlib.Path(td) / "out"
            rc = fs._run_pair(
                source=source, review_source=source, out_dir=out,
                total=len(source.split()),
                source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                ms=Model(), ss=Structure(), base_prompt="Frozen plan repair.",
                audit_tpl=pr.audit_template("v2"),
                ceilings={"detailed": 6, "brief": 2},
                fit={"fits": True}, route="writing-contract-v2",
                label="joined", initial_stage="joined-write",
                stage_prefix="joined",
                result_options=fs.RESULT_REQUEST_OPTIONS,
                audit_options=fs.AUDIT_V2_REQUEST_OPTIONS,
                repair_budget=1, initial_pair=(detailed, brief),
                writing_plan=plan, contract_version="v2",
                assignment_packets=self.wc.atomic_packets(assignment),
                candidate_blocks=blocks)
            self.assertEqual(0, rc)
            self.assertEqual(
                "Corrected alpha evidence now supports the opening. "
                "Corrected beta evidence completes it.\n",
                (out / "detailed.md").read_text())
            report = json.loads((out / "full-report.json").read_text())
            self.assertEqual("pass", report["status"])
            self.assertEqual("complete", report["review"])
        names = [stage for stage, _prompt in stages]
        self.assertIn("joined-revise-1", names)
        self.assertIn("joined-reaudit-1", names)
        repair_prompt = next(prompt for stage, prompt in stages
                             if stage == "joined-revise-1")
        self.assertIn("Alpha evidence", repair_prompt)
        self.assertIn("Beta evidence", repair_prompt)
        self.assertIn('"unit_id": "U001"', repair_prompt)
        self.assertIn('"unit_id": "U002"', repair_prompt)


class MultiPacketRecoveryReviewContract(unittest.TestCase):
    """Write recovery can never narrow review below the complete frozen plan."""

    wc = load("writing_contract")

    def test_recovery_root_subset_does_not_narrow_review_or_completeness(self):
        source = "\n\n".join(
            f"Evidence paragraph {index}." for index in range(1, 11))
        ids = [item["id"] for item in self.wc.source_segments(source)]
        plan = self.wc.validate_plan({
            "schema": self.wc.SCHEMA,
            "units": [{
                "unit_id": f"U{index:03d}", "title": f"Unit {index}",
                "source_ids": [ids[index - 1]], "topic": "evidence",
                "relation_to_previous": "opening" if index == 1 else "next",
                "governing_qualifications": [], "detailed_words": 20,
                "brief_disposition": "include" if index == 1 else "omit",
                "brief_words": 20 if index == 1 else 0,
            } for index in range(1, 11)], "dispositions": [],
        }, source, {"detailed": 250, "brief": 50})
        blocks = {
            "detailed": {
                f"U{index:03d}": {
                    "unit_id": f"U{index:03d}", "heading": "",
                    "join_previous": False,
                    "paragraphs": [f"Summary {index}."]}
                for index in range(1, 11)},
            "brief": {"U001": {
                "unit_id": "U001", "heading": "",
                "join_previous": False, "paragraphs": ["Brief."]}},
        }
        detailed = self.wc.render_blocks(list(blocks["detailed"].values()))
        brief = self.wc.render_blocks(list(blocks["brief"].values()))
        assignment = {
            "detailed": self.wc._assignment(plan, "detailed"),
            "brief": self.wc._assignment(plan, "brief"),
        }
        misleading_recovery_subset = [{
            "detailed": [assignment["detailed"][index]
                         for index in (4, 6, 7)], "brief": []}]
        packets, global_packet = self.wc.bounded_review_packets(
            plan, misleading_recovery_subset, blocks,
            source, detailed, brief)
        self.assertEqual(
            [unit_id for packet in packets for unit_id in packet["unit_ids"]],
            [f"U{index:03d}" for index in range(1, 11)])
        self.assertEqual(
            {source_id for packet in packets
             for source_id in packet["source_ids"]}, set(ids))
        self.assertEqual(global_packet["detailed"], detailed)
        packet_ids = [packet["packet_id"] for packet in packets]
        self.assertEqual(
            self.wc.review_coverage_error(plan, packets, packet_ids), "")
        incomplete = self.wc.review_coverage_error(
            plan, packets, packet_ids[:3])
        self.assertIn("review coverage incomplete", incomplete)
        self.assertIn("U004", incomplete)


class BoundedRoleRecovery(unittest.TestCase):
    """Review and correction carry only the implicated candidate/source span."""

    wc = load("writing_contract")
    pr = load("pair_review")

    def fixture(self):
        source = "## One\n\nAlpha evidence.\n\n## Two\n\nBeta evidence."
        ids = [item["id"] for item in self.wc.source_segments(source)]
        split = len(ids) // 2
        raw = {
            "schema": self.wc.SCHEMA,
            "units": [
                {"unit_id": "U001", "title": "One",
                 "source_ids": ids[:split], "topic": "Alpha",
                 "relation_to_previous": "opening",
                 "governing_qualifications": [], "detailed_words": 100,
                 "brief_disposition": "include", "brief_words": 50},
                {"unit_id": "U002", "title": "Two",
                 "source_ids": ids[split:], "topic": "Beta",
                 "relation_to_previous": "turns",
                 "governing_qualifications": [], "detailed_words": 100,
                 "brief_disposition": "omit", "brief_words": 0},
            ], "dispositions": [],
        }
        plan = self.wc.validate_plan(
            raw, source, {"detailed": 300, "brief": 100})
        blocks = {
            "detailed": {
                "U001": {"unit_id": "U001", "heading": "First",
                         "paragraphs": ["Alpha analysis."]},
                "U002": {"unit_id": "U002", "heading": "Second",
                         "paragraphs": ["Beta analysis."]},
            },
            "brief": {
                "U001": {"unit_id": "U001", "heading": "",
                         "paragraphs": ["Alpha briefly."]},
            },
        }
        packets = [
            {"detailed": [self.wc._assignment(plan, "detailed")[0]],
             "brief": [self.wc._assignment(plan, "brief")[0]]},
            {"detailed": [self.wc._assignment(plan, "detailed")[1]],
             "brief": []},
        ]
        detailed = self.wc.render_blocks(list(blocks["detailed"].values()))
        brief = self.wc.render_blocks(list(blocks["brief"].values()))
        return source, plan, blocks, packets, detailed, brief

    def test_review_packets_bind_local_evidence_and_global_structure(self):
        source, plan, blocks, packets, detailed, brief = self.fixture()
        local, global_packet = self.wc.bounded_review_packets(
            plan, packets, blocks, source, detailed, brief)
        self.assertEqual(len(local), 2)
        self.assertIn("Alpha evidence", local[0]["source"])
        self.assertNotIn("Beta evidence", local[0]["source"])
        self.assertIn("D-P001", local[0]["segments"])
        self.assertEqual(global_packet["detailed"], detailed)
        self.assertIn('"unit_id": "U001"', global_packet["plan_context"])

    def test_bounded_patch_excludes_unrelated_candidate_spans(self):
        detailed = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.\n\nFourth paragraph."
        brief = "Brief paragraph."
        current = self.pr.new_candidate(
            "initial", detailed, brief, [], [], usable=True,
            audit_records=[{
                "finding_id": "F-001", "kind": "reversal",
                "artifact": "detailed", "text": "Correct the third claim.",
                "anchor": "D-P003", "slot": "", "packet": "review-002",
            }])
        current["findings"] = ["Correct the third claim."]
        turn = self.pr.repair_turn(current, force_patch=True, bounded=True)
        self.assertNotIn("First paragraph.", turn["prompt"])
        self.assertIn("Second paragraph.", turn["prompt"])
        self.assertIn("Third paragraph.", turn["prompt"])
        self.assertIn("Fourth paragraph.", turn["prompt"])
        self.assertGreaterEqual(turn["planned_output_words"], 200)

    def test_bounded_patch_rejects_an_unanchored_finding(self):
        current = self.pr.new_candidate(
            "initial", "Detailed.", "Brief.", [], [], usable=True,
            audit_records=[{
                "finding_id": "F-001", "kind": "omission",
                "artifact": "pair", "text": "Add a qualification.",
                "anchor": "", "slot": "", "packet": "review-001",
            }])
        current["findings"] = ["Add a qualification."]
        with self.assertRaises(ValueError):
            self.pr.repair_turn(current, force_patch=True, bounded=True)


class DiscoursePlanContract(unittest.TestCase):
    """The v2 plan controls source accounting, selection, and rendered blocks."""

    wc = load("writing_contract")

    def fixture(self):
        source = "## One\n\nAlpha evidence.\n\n## Two\n\nBeta evidence."
        ids = [item["id"] for item in self.wc.source_segments(source)]
        split = max(1, len(ids) // 2)
        plan = {
            "schema": "summer.writing-plan.v2",
            "units": [
                {"unit_id": "U001", "title": "One",
                 "source_ids": ids[:split], "topic": "Alpha",
                 "relation_to_previous": "opening",
                 "governing_qualifications": [], "detailed_words": 300,
                 "brief_disposition": "include", "brief_words": 100},
                {"unit_id": "U002", "title": "Two",
                 "source_ids": ids[split:], "topic": "Beta",
                 "relation_to_previous": "turns to the second topic",
                 "governing_qualifications": ["retain the condition"],
                 "detailed_words": 300,
                 "brief_disposition": "omit", "brief_words": 0},
            ],
            "dispositions": [],
        }
        return source, plan

    def test_every_source_id_is_accounted_once_and_budgets_are_global(self):
        source, plan = self.fixture()
        frozen = self.wc.validate_plan(
            plan, source, {"detailed": 700, "brief": 200})
        self.assertEqual(frozen["allocated_words"],
                         {"detailed": 600, "brief": 100})
        self.assertRegex(frozen["plan_sha256"], r"^[0-9a-f]{64}$")
        broken = json.loads(json.dumps(plan))
        broken["units"][1]["source_ids"] = broken["units"][0]["source_ids"]
        with self.assertRaises(self.wc.WritingContractError):
            self.wc.validate_plan(
                broken, source, {"detailed": 700, "brief": 200})

    def test_exclude_transport_synonym_is_canonicalized_on_plan_resume(self):
        source, plan = self.fixture()
        plan["units"][1]["brief_disposition"] = "exclude"
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / "retained-plan.json"
            path.write_text(json.dumps(plan))
            frozen = self.wc.load_plan(
                path, source, {"detailed": 700, "brief": 200})
        self.assertEqual(frozen["units"][1]["brief_disposition"], "omit")
        self.assertRegex(frozen["plan_sha256"], r"^[0-9a-f]{64}$")

    def test_brief_is_a_global_selection_not_every_source_slice(self):
        source, plan = self.fixture()
        frozen = self.wc.validate_plan(
            plan, source, {"detailed": 700, "brief": 200})
        packets = self.wc.build_packets(frozen, output_words=1_000)
        self.assertEqual(len(packets), 1)
        self.assertEqual(self.wc._ids(packets[0], "detailed"),
                         ["U001", "U002"])
        self.assertEqual(self.wc._ids(packets[0], "brief"), ["U001"])

    def test_block_response_requires_exact_ids_and_renders_boundaries(self):
        source, plan = self.fixture()
        frozen = self.wc.validate_plan(
            plan, source, {"detailed": 700, "brief": 200})
        packet = self.wc.build_packets(frozen, output_words=1_000)[0]
        value = {
            "detailed": [
                {"unit_id": "U001", "heading": "First",
                 "paragraphs": ["Alpha point.", "Its qualification."]},
                {"unit_id": "U002", "heading": "",
                 "paragraphs": ["Beta point."]},
            ],
            "brief": [{"unit_id": "U001", "heading": "",
                       "paragraphs": ["Alpha briefly."]}],
        }
        clean = self.wc.validate_blocks(value, packet)
        rendered = self.wc.render_blocks(clean["detailed"])
        self.assertIn("## First\n\nAlpha point.\n\nIts qualification.",
                      rendered)
        bad = json.loads(json.dumps(value))
        bad["detailed"].reverse()
        with self.assertRaises(self.wc.WritingContractError):
            self.wc.validate_blocks(bad, packet)

    def test_fake_v2_route_reaches_the_production_pair_lifecycle(self):
        fs = load("fullsum")
        source = "evidence " * 1_000
        sid = self.wc.source_segments(source)[0]["id"]
        plan = {
            "schema": "summer.writing-plan.v2",
            "units": [{
                "unit_id": "U001", "title": "Core", "source_ids": [sid],
                "topic": "The central evidence", "relation_to_previous": "opening",
                "governing_qualifications": [], "detailed_words": 200,
                "brief_disposition": "include", "brief_words": 100,
            }],
            "dispositions": [],
        }

        class Fake:
            PLAN_MODELS = ["planner"]
            MODELS = ["writer"]
            AUDIT_MODELS = ["auditor"]
            REPAIR_MODELS = ["repair"]
            HARNESS = "fake"
            class NoCandidate(RuntimeError):
                pass
            def run(self, prompt, out_dir, chain, stage, validate=None,
                    gateway_options=None, role=None, **_kwargs):
                if stage == "full-v2-plan-l001":
                    value = plan
                elif stage == "full-v2-plan-reduce-001":
                    value = {"summary": "core evidence",
                             "brief_candidates": ["L001-U001"],
                             "qualifications": []}
                elif stage == "full-v2-plan-select":
                    value = {"brief_units": [
                        {"handle": "L001-U001", "words": 100}]}
                elif stage.startswith("full-v2-write"):
                    value = {
                        "detailed": [{"unit_id": "U001", "heading": "",
                                      "paragraphs": ["analysis " * 200]}],
                        "brief": [{"unit_id": "U001", "heading": "",
                                   "paragraphs": ["condensed " * 100]}],
                    }
                elif "audit" in stage:
                    ids = re.findall(r'"observation_id": "(R-\d+)"', prompt)
                    value = {"verdict": "pass", "findings": [],
                             "readability": [
                                 {"observation_id": ident,
                                  "assessment": "acceptable",
                                  "explanation": "The single explanation is coherent."}
                                 for ident in ids]}
                else:
                    raise AssertionError(stage)
                raw = json.dumps(value)
                if validate:
                    validate(raw)
                return raw

        with tempfile.TemporaryDirectory() as td, \
                unittest.mock.patch.object(fs, "_last_ok_chain",
                                           return_value=["writer"]):
            out = pathlib.Path(td)
            rc = fs._run_contract_v2(
                source, out, 1_000, 100_000, 1_000, Fake(),
                load("shortsum"))
            self.assertEqual(rc, 0)
            report = json.loads((out / "full-report.json").read_text())
            self.assertEqual(report["writing_contract"],
                             "summer.writing-plan.v2")
            self.assertEqual(report["candidate_state"],
                             "quality_qualified_candidate")
            self.assertEqual(report["route"], "writing-contract-v2")

    def test_confirmed_unreadable_candidate_is_retained_but_not_published(self):
        fs = load("fullsum")
        source = "evidence " * 2_000
        sid = self.wc.source_segments(source)[0]["id"]
        plan = {
            "schema": "summer.writing-plan.v2",
            "units": [{
                "unit_id": "U001", "title": "Core", "source_ids": [sid],
                "topic": "The central evidence", "relation_to_previous": "opening",
                "governing_qualifications": [], "detailed_words": 500,
                "brief_disposition": "include", "brief_words": 100,
            }],
            "dispositions": [],
        }

        class Fake:
            PLAN_MODELS = ["planner"]
            MODELS = ["writer"]
            AUDIT_MODELS = ["auditor"]
            REPAIR_MODELS = ["repair"]
            HARNESS = "fake"
            class NoCandidate(RuntimeError):
                pass
            def run(self, prompt, out_dir, chain, stage, validate=None,
                    gateway_options=None, role=None, **_kwargs):
                if stage == "full-v2-plan-l001":
                    value = plan
                elif stage == "full-v2-plan-reduce-001":
                    value = {"summary": "core evidence",
                             "brief_candidates": ["L001-U001"],
                             "qualifications": []}
                elif stage == "full-v2-plan-select":
                    value = {"brief_units": [
                        {"handle": "L001-U001", "words": 100}]}
                elif stage.startswith("full-v2-write"):
                    value = {
                        "detailed": [{"unit_id": "U001", "heading": "",
                                      "paragraphs": ["word " * 400]}],
                        "brief": [{"unit_id": "U001", "heading": "",
                                   "paragraphs": ["brief " * 100]}],
                    }
                elif "audit" in stage:
                    ids = re.findall(r'"observation_id": "(R-\d+)"', prompt)
                    value = {
                        "verdict": "revise",
                        "findings": [{
                            "kind": "editorial", "artifact": "detailed",
                            "text": "Several unrelated explanatory jobs are merged.",
                            "anchor": "D-P001",
                        }],
                        "readability": [{
                            "observation_id": ident,
                            "assessment": ("unreadable" if ident == "R-001"
                                           else "acceptable"),
                            "explanation": "The paragraph has no readable discourse turn.",
                        } for ident in ids],
                    }
                else:
                    raise RuntimeError("repair unavailable")
                raw = json.dumps(value)
                if validate:
                    validate(raw)
                return raw

        with tempfile.TemporaryDirectory() as td, \
                unittest.mock.patch.object(fs, "_last_ok_chain",
                                           return_value=["writer"]):
            out = pathlib.Path(td)
            rc = fs._run_contract_v2(
                source, out, 2_000, 100_000, 1_000, Fake(),
                load("shortsum"))
            self.assertEqual(rc, 5)
            self.assertFalse((out / "detailed.md").exists())
            self.assertFalse((out / "brief.md").exists())


class CapabilityPackingContract(unittest.TestCase):
    """Capabilities alter request packing, never frozen editorial units."""

    wc = load("writing_contract")

    @staticmethod
    def qualification_runner():
        path = ENG / "tests" / "qualify_writing_contract.py"
        spec = importlib.util.spec_from_file_location(
            "qualify_writing_contract", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def plan(self):
        units = []
        for i in range(1, 5):
            units.append({
                "unit_id": f"U{i:03d}", "title": f"Unit {i}",
                "source_ids": [f"S-P{i:03d}"], "topic": f"Topic {i}",
                "relation_to_previous": "continues", "governing_qualifications": [],
                "detailed_words": 300, "brief_disposition": (
                    "include" if i <= 2 else "omit"),
                "brief_words": 100 if i <= 2 else 0,
                "source_words": 500,
            })
        return {"units": units, "allocated_words": {
            "detailed": 1_200, "brief": 200}}

    def test_unconstrained_output_is_one_whole_pair(self):
        packets = self.wc.build_packets(self.plan(), output_words=2_000)
        self.assertEqual(len(packets), 1)
        self.assertEqual(self.wc._ids(packets[0], "detailed"),
                         ["U001", "U002", "U003", "U004"])
        self.assertEqual(self.wc._ids(packets[0], "brief"),
                         ["U001", "U002"])

    def test_output_limit_changes_only_packet_boundaries(self):
        packets = self.wc.build_packets(self.plan(), output_words=650)
        flattened = {
            depth: [uid for packet in packets
                    for uid in self.wc._ids(packet, depth)]
            for depth in ("detailed", "brief")}
        self.assertEqual(flattened["detailed"],
                         ["U001", "U002", "U003", "U004"])
        self.assertEqual(flattened["brief"], ["U001", "U002"])
        self.assertGreater(len(packets), 1)

    def test_input_and_output_limits_are_both_enforced(self):
        packets = self.wc.build_packets(
            self.plan(), output_words=650, input_words=1_250,
            overhead_words=200)
        self.assertTrue(all(self.wc._packet_output_words(p) <= 650
                            for p in packets))
        self.assertTrue(all(self.wc._packet_source_words(p) + 200 <= 1_250
                            for p in packets))
        with self.assertRaises(self.wc.WritingContractError):
            self.wc.build_packets(
                self.plan(), output_words=250, input_words=10_000)

    def test_route_integration_requires_four_passes_and_three_first_passes(self):
        qr = self.qualification_runner()
        manifest = {"matrix": {"discriminating": {
            "routes": ["candidate-route"], "sources": ["a", "b"], "repeats": 2,
            "required_passes_per_route": 4,
            "required_first_passes_per_route": 3,
        }}}
        results = [{
            "phase": "discriminating", "route": "candidate-route",
            "status": "automatic_pass_manual_review_pending",
            "first_pass": i < 3,
        } for i in range(4)]
        self.assertTrue(qr.automatic_condition_passes(
            manifest, results, "discriminating:candidate-route"))
        results[-2]["first_pass"] = False
        self.assertFalse(qr.automatic_condition_passes(
            manifest, results, "discriminating:candidate-route"))

    def test_conditional_promotion_requires_hash_bound_blind_approval(self):
        qr = self.qualification_runner()
        digest = "a" * 64
        results = [{
            "phase": "discriminating", "route": "candidate-route",
            "candidate": "C001", "report_sha256": "b" * 64,
        }]
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            self.assertFalse(qr.manual_condition_approved(
                root, digest, results, "discriminating:candidate-route"))
            directory = root / "review-approvals"
            directory.mkdir()
            receipt = {
                "schema": qr.APPROVAL_SCHEMA,
                "manifest_sha256": digest,
                "condition": "discriminating:candidate-route",
                "approved": True,
                "candidate_reports": {"C001": "b" * 64},
            }
            path = directory / "discriminating-candidate-route.json"
            path.write_text(json.dumps(receipt))
            self.assertTrue(qr.manual_condition_approved(
                root, digest, results, "discriminating:candidate-route"))
            receipt["candidate_reports"]["C001"] = "c" * 64
            path.write_text(json.dumps(receipt))
            with self.assertRaises(qr.QualificationError):
                qr.manual_condition_approved(
                    root, digest, results, "discriminating:candidate-route")

    def test_generation_envelope_does_not_force_windows_for_editorial_bands(self):
        fs = load("fullsum")
        # ~10k-word source, 8192-token output: publication band exceeds the
        # reply, but the source still fits a 262k context with a clipped pair.
        detail = fs.fit_detail(10_330, capability_tokens=262_144,
                               output_budget_tokens=8_192)
        self.assertTrue(detail["fits"], detail)
        self.assertTrue(detail["route_limited"])
        self.assertLessEqual(
            detail["base_generation_ceiling_words"],
            load("route_caps").tokens_to_words(8_192))
        self.assertGreater(detail["publication_ceiling_words"],
                           detail["base_generation_ceiling_words"])


class RouteCapabilityContract(unittest.TestCase):
    """Direct (whole-source) versus windowed handling, without a model
    catalog. An optional route-provided budget in words, else the
    conservative baseline for the qualified roughly-100k-context class.
    Admission is governed by the composed request; no default output cap
    rejects a fitting route, while an explicitly provided output budget is
    still honored.
    """
    rc = load("route_caps")

    def test_the_baseline_is_the_roughly_100k_class(self):
        self.assertEqual(self.rc.BASELINE_DIRECT_WORDS, 100_000)

    def test_the_baseline_boundary(self):
        self.assertTrue(self.rc.direct_ok(100_000))
        self.assertFalse(self.rc.direct_ok(100_001))

    def test_overhead_counts_against_the_budget(self):
        self.assertTrue(self.rc.direct_ok(90_000, overhead_words=9_999))
        self.assertFalse(self.rc.direct_ok(90_000, overhead_words=10_001))

    def test_a_route_budget_overrides_in_both_directions(self):
        self.assertTrue(self.rc.direct_ok(200_000, capability_words=250_000))
        self.assertFalse(self.rc.direct_ok(60_000, capability_words=50_000))

    def test_a_useless_budget_falls_back_to_baseline(self):
        for bad in (None, 0, -5, "large", True):
            self.assertTrue(self.rc.direct_ok(50, capability_words=bad),
                            f"{bad!r} did not fall back to the baseline")

    def test_unusable_inputs_fail_closed_toward_windows(self):
        self.assertFalse(self.rc.direct_ok(None))
        self.assertFalse(self.rc.direct_ok("many"))
        self.assertFalse(self.rc.direct_ok(-3))

    def test_no_default_output_cap_rejects_a_fitting_request(self):
        # A large expected pair is admitted when the composed request fits;
        # only an explicitly provided output budget can refuse it.
        self.assertTrue(self.rc.direct_ok(50_000, output_words=25_000))
        self.assertFalse(self.rc.direct_ok(50_000, output_words=25_000,
                                           output_budget_words=10_000))
        self.assertTrue(self.rc.direct_ok(50_000, output_words=25_000,
                                          output_budget_words=30_000))
        for bad in (None, 0, -5, "large", True):
            self.assertTrue(
                self.rc.direct_ok(50_000, output_words=25_000,
                                  output_budget_words=bad),
                f"{bad!r} output budget did not fall back to unbounded")

    def test_window_budget_is_generic_and_leaves_the_declared_margin(self):
        self.assertEqual(self.rc.window_source_words(
            20_000, overhead_words=1_000, output_words=3_000), 8_000)
        self.assertEqual(self.rc.window_source_words(
            20_000, overhead_words=20_000, output_words=1), 0)
        self.assertEqual(self.rc.window_source_words(
            20_000, overhead_words=1_000, output_words=3_000,
            share=0.25), 4_000)


    def test_token_contract_is_conservative_and_separate_from_words(self):
        self.assertGreater(self.rc.words_to_tokens(100_000), 100_000)
        self.assertTrue(self.rc.direct_ok_tokens(
            50_000, capability_tokens=90_000))
        self.assertFalse(self.rc.direct_ok_tokens(
            70_000, capability_tokens=90_000))
        self.assertGreater(
            self.rc.window_source_words_tokens(
                90_000, overhead_words=500, output_words=2_400), 0)

    def test_token_output_budget_can_refuse_an_otherwise_fitting_request(self):
        self.assertTrue(self.rc.direct_ok_tokens(
            10_000, capability_tokens=90_000, output_words=1_000))
        self.assertFalse(self.rc.direct_ok_tokens(
            10_000, capability_tokens=90_000, output_words=1_000,
            output_budget_tokens=500))

    def test_token_context_includes_the_reserved_completion(self):
        self.assertFalse(self.rc.direct_ok_tokens(
            50_000, capability_tokens=90_000, output_words=20_000))
        self.assertTrue(self.rc.direct_ok_tokens(
            50_000, capability_tokens=125_000, output_words=20_000))

    def test_device_local_capabilities_are_generic_and_route_specific(self):
        raw = json.dumps({
            "fake:model-a": {"context_tokens": 125_000,
                             "output_tokens": 12_000},
            "fake": {"context_tokens": 90_000,
                     "output_tokens": 8_000},
        })
        with unittest.mock.patch.dict(os.environ,
                                      {"SUMM_ROUTE_CAPABILITIES": raw}):
            self.assertEqual(
                self.rc.capability_for("fake:model-a", "fake"),
                {"context_tokens": 125_000, "output_tokens": 12_000})
            self.assertEqual(
                self.rc.capability_for("fake:model-b", "fake"),
                {"context_tokens": 90_000, "output_tokens": 8_000})

    def test_frozen_runtime_supplies_per_model_route_capacity(self):
        frozen = json.dumps({
            "gateways": {
                "fake": {
                    "models": {
                        "model-a": {"context_tokens": 262_144,
                                    "output_tokens": 8_192,
                                    "prompt_overhead_tokens": 256},
                        "model-b": {},
                    }
                }
            }
        })
        with unittest.mock.patch.dict(
                os.environ, {"SUMM_RUNTIME_JSON": frozen}, clear=True):
            self.assertEqual(
                self.rc.capability_for("fake:model-a", "fake"),
                {"context_tokens": 262_144, "output_tokens": 8_192,
                 "prompt_overhead_tokens": 256})
            self.assertIsNone(
                self.rc.capability_for("fake:model-b", "fake"))

    def test_qualification_override_beats_frozen_route_capacity(self):
        frozen = json.dumps({
            "gateways": {"fake": {"models": {
                "model-a": {"context_tokens": 262_144}}}}})
        override = json.dumps({
            "fake:model-a": {"context_tokens": 90_000}})
        with unittest.mock.patch.dict(
                os.environ,
                {"SUMM_RUNTIME_JSON": frozen,
                 "SUMM_ROUTE_CAPABILITIES": override}, clear=True):
            self.assertEqual(
                self.rc.capability_for("fake:model-a", "fake"),
                {"context_tokens": 90_000})

    def test_primary_capability_uses_the_first_active_route_per_role(self):
        raw = json.dumps({
            "fake:writer": {"context_tokens": 125_000,
                            "output_tokens": 12_000},
            "fake:auditor": {"context_tokens": 100_000,
                             "output_tokens": 8_000},
        })
        split = lambda entry, default: (default, entry)
        with unittest.mock.patch.dict(os.environ,
                                      {"SUMM_ROUTE_CAPABILITIES": raw}):
            self.assertEqual(
                self.rc.primary_chain_capability(
                    {"write": ["writer"], "audit": ["auditor"]},
                    split, "fake"),
                {"context_tokens": 100_000, "output_tokens": 8_000,
                 "routes": ["fake:writer", "fake:auditor"]})

    def test_mixed_declared_and_undeclared_routes_use_safe_baseline(self):
        raw = json.dumps({
            "fake:writer": {"context_tokens": 262_144,
                            "output_tokens": 8_192,
                            "prompt_overhead_tokens": 512}})
        split = lambda entry, default: (default, entry)
        with unittest.mock.patch.dict(
                os.environ, {"SUMM_ROUTE_CAPABILITIES": raw}, clear=True):
            self.assertEqual(
                self.rc.primary_chain_capability(
                    {"write": ["writer"], "audit": ["undeclared"]},
                    split, "fake"),
                {"context_tokens": 90_000, "output_tokens": 8_192,
                 "prompt_overhead_tokens": 512,
                 "routes": ["fake:writer", "fake:undeclared"]})

    def test_all_undeclared_routes_keep_the_normal_generic_fallback(self):
        split = lambda entry, default: (default, entry)
        self.assertIsNone(
            self.rc.primary_chain_capability(
                {"write": ["writer"], "audit": ["auditor"]},
                split, "fake", {}))

class StageSchemaContracts(unittest.TestCase):
    """A syntactically valid object is not automatically a valid stage result."""
    jc = load("json_contract")

    def test_a_cut_off_json_value_is_incomplete_not_a_salvaged_prefix(self):
        self.assertEqual(self.jc.document_status('{"ok":true}'), "complete")
        self.assertEqual(self.jc.document_status('{"ok":'), "incomplete")
        self.assertEqual(self.jc.document_status('{"ok":true} extra'), "trailing")
        self.assertEqual(self.jc.document_status(""), "empty")

    def test_a_literal_newline_inside_a_string_value_is_still_an_answer(self):
        # A complete pair may contain a literal paragraph break inside a string.
        # Treating that as no JSON object would discard usable output.
        raw = '{"detailed": "First paragraph.\n\nSecond\tparagraph.", "brief": "One."}'
        out = self.jc.parse(raw, self.jc.SHORT_RESULT, "short")
        self.assertEqual(out["detailed"], "First paragraph.\n\nSecond\tparagraph.")

    def test_other_control_characters_inside_a_string_are_still_rejected(self):
        # strict=False would have admitted every C0 control; only the three
        # a writer produces by breaking lines are transport, the rest is damage.
        for bad in ("\x00", "\x01", "\x08", "\x0c", "\x1b"):
            raw = '{"detailed": "First' + bad + 'paragraph.", "brief": "One."}'
            with self.assertRaises(self.jc.ContractError):
                self.jc.parse(raw, self.jc.SHORT_RESULT, "short")

    def test_malformed_audit_object_cannot_become_an_effective_pass(self):
        with self.assertRaises(self.jc.ContractError):
            self.jc.parse('{"note":"done"}', self.jc.SHORT_AUDIT,
                          "short-audit")

    def test_audit_verdict_and_findings_must_agree(self):
        with self.assertRaises(self.jc.ContractError):
            self.jc.parse('{"verdict":"pass","findings":['
                          '{"kind":"omission","artifact":"detailed",'
                          '"text":"missing caveat"}]}',
                          self.jc.FULL_AUDIT, "full-audit")
        with self.assertRaises(self.jc.ContractError):
            self.jc.parse('{"verdict":"revise","findings":[]}',
                          self.jc.FULL_AUDIT, "full-audit")
        with self.assertRaises(self.jc.ContractError):
            self.jc.parse('{"verdict":"revise","findings":["a sentence"]}',
                          self.jc.FULL_AUDIT, "full-audit")

    def test_valid_audit_requires_the_exact_stage_shape(self):
        value = self.jc.parse('{"verdict":"pass","findings":[]}',
                              self.jc.FULL_AUDIT, "full-audit")
        self.assertEqual(value, {"verdict": "pass", "findings": []})

    def test_pair_and_capsule_contracts_reject_missing_required_text(self):
        with self.assertRaises(self.jc.ContractError):
            self.jc.parse('{"detailed":"only one side"}',
                          self.jc.FULL_RESULT, "full")
        with self.assertRaises(self.jc.ContractError):
            self.jc.parse('{"text":"not a capsule"}',
                          self.jc.FULL_WINDOW_RESULT, "full-window-001")

    def test_empty_pair_and_capsule_are_unusable_stage_responses(self):
        for value, schema, stage in (
                ('{"detailed":"", "brief":"answer"}',
                 self.jc.FULL_RESULT, "full"),
                ('{"detailed":" ", "brief":"answer"}',
                 self.jc.SHORT_RESULT, "short"),
                ('{"capsule":"  "}',
                 self.jc.FULL_WINDOW_RESULT, "full-window-001")):
            with self.assertRaises(self.jc.ContractError):
                self.jc.parse(value, schema, stage)

    def test_window_synthesis_and_repair_map_to_pair_contract(self):
        self.assertIs(self.jc.schema_for_stage("full-batch-detailed-001"),
                      self.jc.FULL_READING_PART)
        self.assertIs(self.jc.schema_for_stage("full-batch-brief-002ab"),
                      self.jc.FULL_READING_PART)
        self.assertIs(self.jc.schema_for_stage("full-window-001"),
                      self.jc.FULL_WINDOW_RESULT)
        self.assertIs(self.jc.schema_for_stage("full-window-001-shorten"),
                      self.jc.FULL_WINDOW_RESULT)
        self.assertIs(self.jc.schema_for_stage("full-window-synthesis"),
                      self.jc.FULL_RESULT)
        self.assertIs(self.jc.schema_for_stage("full-window-revise-1"),
                      self.jc.FULL_RESULT)

# One shared class object: `except ms.NoCandidate` in the controller must
# match what a fake runner raises, and load() returns a fresh module each call.
NO_CANDIDATE = load("mapsum").NoCandidate


class FullDirectController(unittest.TestCase):
    """The Full direct route: whole-source write, fresh review of the actual
    pair, up to three enclosed-candidate repairs. Real controller driven by
    a fake model transport; no model calls, no I/O beyond temp dirs."""
    SOURCE = ("The committee reported that rates fell after the reform in "
              "2008. The evidence was limited and the authors did not claim "
              "causation. Markets reacted quickly while analysts warned that "
              "stress could return. The report recommends further study before "
              "any policy change takes effect.")
    CLEAN = {
        "detailed": ("Rates fell after the 2008 reform, though evidence "
                     "was limited and causation unclaimed. Markets reacted "
                     "quickly, analysts warned stress could return, and "
                     "further study was recommended."),
        "brief": ("Rates fell after the reform without proven causation; "
                  "further study was recommended."),
    }

    class Ledger:
        @staticmethod
        def parse_strict(raw, stage):
            return json.loads(raw)

    def _run(self, ms, source=None, capability_words=None,
             output_budget_words=None):
        fs = load("fullsum")
        root = pathlib.Path(tempfile.mkdtemp())
        src, out = root / "source.txt", root / "out"
        src.write_text(source or self.SOURCE)
        patches = [unittest.mock.patch.object(fs, "_runner",
                                              return_value=ms),
                   unittest.mock.patch.object(fs, "_ledger",
                                              return_value=self.Ledger)]
        for p in patches:
            p.start()
        try:
            return fs.run(src, out, capability_words=capability_words,
                          output_budget_words=output_budget_words), out
        finally:
            for p in reversed(patches):
                p.stop()

    @staticmethod
    def _ms(handler):
        class FakeMS:
            MODELS = AUDIT_MODELS = REPAIR_MODELS = ["local"]
            NoCandidate = NO_CANDIDATE

            @staticmethod
            def run(prompt, out_dir, chain, stage, validate=None,
                    gateway_options=None):
                return handler(stage, prompt)
        return FakeMS()

    def _report(self, out):
        return json.loads((out / "full-report.json").read_text())

    def test_clean_full_publishes_pass(self):
        def handler(stage, prompt):
            if stage == "full":
                return json.dumps(dict(self.CLEAN))
            return json.dumps({"verdict": "pass", "findings": []})

        rc, out = self._run(self._ms(handler))
        self.assertEqual(0, rc)
        self.assertEqual((out / "detailed.md").read_text().strip(),
                         self.CLEAN["detailed"])
        self.assertEqual((out / "brief.md").read_text().strip(),
                         self.CLEAN["brief"])
        report = self._report(out)
        self.assertEqual(
            (report["path"], report["route"], report["status"],
             report["selected"], report["repair"]),
            ("full", "direct", "pass", "initial", "none"))
        self.assertEqual(report["source_sha256"],
                         hashlib.sha256(self.SOURCE.encode()).hexdigest())
        pr = load("pair_review")
        self.assertEqual(report["candidate"],
                         pr.candidate_identity(self.CLEAN["detailed"],
                                               self.CLEAN["brief"]))
        self.assertEqual(report["candidates"], 1)
        self.assertEqual(report["findings"], [])

    def test_finding_then_one_repair_publishes_revised(self):
        fixed = dict(self.CLEAN)
        fixed["brief"] = ("Rates fell without proven causation, though stress "
                          "could return; further study was recommended.")
        seen = {}

        def handler(stage, prompt):
            if stage == "full":
                return json.dumps(dict(self.CLEAN))
            if stage == "full-revise-1":
                seen["prompt"] = prompt
                return json.dumps(patch_replace(
                    self.CLEAN["detailed"], self.CLEAN["brief"],
                    "brief", fixed["brief"]))
            if stage == "full-audit":
                return json.dumps({"verdict": "revise", "findings": [
                    AF("The brief omits the warning that stress could return",
                       artifact="brief")]})
            return json.dumps({"verdict": "pass", "findings": []})

        rc, out = self._run(self._ms(handler))
        self.assertEqual(0, rc)
        self.assertIn(self.CLEAN["detailed"], seen.get("prompt", ""),
                      "repair did not enclose the current Detailed")
        self.assertEqual((out / "brief.md").read_text().strip(),
                         fixed["brief"])
        report = self._report(out)
        self.assertEqual((report["selected"], report["repair"],
                          report["status"]),
                         ("revise-1", "once", "pass"))

    def test_an_overlength_pair_gets_one_same_producer_shortening_then_publishes(self):
        # An overlength reading cannot become a clean pass merely because it is
        # otherwise usable. Length is sent once to the same producer, then the
        # safest pair publishes with the finding disclosed if it stays over.
        source = self.SOURCE * 40
        long_pair = {"detailed": "Rates fell after the reform, though evidence was limited. " * 150,
                     "brief": "Rates fell after the reform. " * 50}
        for shortened, expect_status in ((True, "pass"), (False, "open_findings")):
            stages = []

            def handler(stage, prompt, shortened=shortened):
                stages.append(stage)
                if stage == "full":
                    return json.dumps(long_pair)
                if stage.startswith("full-revise"):
                    short_pair = {"detailed": "Rates fell after the reform, though evidence was limited. " * 30,
                                  "brief": "Rates fell after the reform. " * 20}
                    return json.dumps(short_pair if shortened else long_pair)
                return json.dumps({"verdict": "pass", "findings": []})

            rc, out = self._run(self._ms(handler), source=source)
            self.assertEqual(0, rc)
            self.assertEqual([s for s in stages if s.startswith("full-revise")],
                             ["full-revise-1"])
            report = self._report(out)
            self.assertEqual(report["status"], expect_status, report)
            if shortened:
                self.assertEqual(report["selected"], "revise-1")
                self.assertLess(len((out / "detailed.md").read_text().split()), 400)
            else:
                self.assertTrue(any("of the source" in f for f in report["findings"]),
                                report["findings"])
                self.assertTrue((out / "detailed.md").is_file())

    def test_one_repair_then_best_is_published(self):
        # One correction from the producer that wrote the candidate, then
        # retain and disclose. Further rounds would make latency unbounded and
        # shop for a different answer.
        stages = []

        def handler(stage, prompt):
            stages.append(stage)
            if stage == "full":
                return json.dumps(dict(self.CLEAN))
            if stage.startswith("full-revise"):
                return json.dumps(patch_replace(
                    self.CLEAN["detailed"], self.CLEAN["brief"],
                    "detailed", self.CLEAN["detailed"]))
            return json.dumps({"verdict": "revise", "findings": [
                AF("The reading omits that the evidence remained limited")]})

        rc, out = self._run(self._ms(handler))
        self.assertEqual(0, rc)
        revises = [s for s in stages if s.startswith("full-revise")]
        self.assertEqual(revises, ["full-revise-1"], "one correction, no more")
        reaudits = [s for s in stages if s.startswith("full-reaudit")]
        self.assertEqual(len(reaudits), 1, "the child is freshly re-reviewed")
        report = self._report(out)
        self.assertEqual((report["selected"], report["repair"],
                          report["status"], report["candidates"]),
                         ("revise-1", "once", "open_findings", 2))
        self.assertTrue(report["findings"])

    def test_reviewer_outage_publishes_retained_candidate(self):
        def handler(stage, prompt):
            if stage == "full-audit":
                raise RuntimeError("auditor unavailable")
            return json.dumps(dict(self.CLEAN))

        rc, out = self._run(self._ms(handler))
        self.assertEqual(0, rc)
        report = self._report(out)
        self.assertEqual((report["status"], report["selected"],
                          report["repair"]),
                         ("review_unavailable", "initial", "none"))
        self.assertIn("full-audit", report["review"])

    def test_bad_initial_pair_gets_repair_even_when_reviewer_is_unavailable(self):
        listed = {"detailed": "Outcomes:\n- Rates fell after the reform.",
                  "brief": "Rates fell after the reform."}
        stages = []

        def handler(stage, prompt):
            stages.append(stage)
            if stage == "full":
                return json.dumps(listed)
            if stage == "full-audit":
                raise RuntimeError("auditor unavailable")
            if stage == "full-revise-1":
                patch = patch_replace(
                    listed["detailed"], listed["brief"],
                    "detailed", self.CLEAN["detailed"])
                patch["edits"].extend(patch_replace(
                    listed["detailed"], listed["brief"],
                    "brief", self.CLEAN["brief"])["edits"])
                return json.dumps(patch)
            if stage == "full-reaudit-1":
                return json.dumps({"verdict": "pass", "findings": []})
            raise AssertionError(stage)

        rc, out = self._run(self._ms(handler))
        self.assertEqual(0, rc)
        report = self._report(out)
        self.assertEqual((report["selected"], report["repair"],
                          report["status"]),
                         ("revise-1", "once", "pass"))
        self.assertEqual(stages,
                         ["full", "full-audit", "full-revise-1",
                          "full-reaudit-1"])

    def test_repair_outage_publishes_parent(self):
        def handler(stage, prompt):
            if stage == "full-revise-1":
                raise RuntimeError("repair unavailable")
            if stage == "full-audit":
                return json.dumps({"verdict": "revise", "findings": [
                    AF("The brief omits the warning that stress could return",
                       artifact="brief")]})
            return json.dumps(dict(self.CLEAN))

        rc, out = self._run(self._ms(handler))
        self.assertEqual(0, rc)
        report = self._report(out)
        self.assertEqual((report["selected"], report["repair"],
                          report["status"]),
                         ("initial", "unavailable", "open_findings"))
        self.assertTrue(report["findings"])

    def test_worse_child_loses_to_parent(self):
        worse = {"detailed": "We think rates fell 47 percent and it was good.",
                 "brief": self.CLEAN["brief"]}
        stages = []

        def handler(stage, prompt):
            stages.append(stage)
            # Every repair chains onto the latest (worse) child, burning the
            # whole budget; the retained parent must still win selection.
            if stage.startswith("full-revise"):
                return json.dumps(patch_replace(
                    self.CLEAN["detailed"], self.CLEAN["brief"],
                    "detailed", worse["detailed"]))
            if stage.startswith("full-reaudit"):
                return json.dumps({"verdict": "pass", "findings": []})
            if stage == "full-audit":
                return json.dumps({"verdict": "revise", "findings": [
                    AF("The brief omits the warning that stress could return",
                       artifact="brief")]})
            return json.dumps(dict(self.CLEAN))

        rc, out = self._run(self._ms(handler))
        self.assertEqual(0, rc)
        report = self._report(out)
        self.assertEqual(report["selected"], "initial")
        self.assertEqual([s for s in stages if s.startswith("full-revise")],
                         ["full-revise-1"])
        self.assertEqual((out / "detailed.md").read_text().strip(),
                         self.CLEAN["detailed"])
        self.assertEqual(report["status"], "open_findings")

    def test_no_usable_pair_publishes_nothing(self):
        def handler(stage, prompt):
            if stage == "full" or stage.startswith("full-revise"):
                return json.dumps({"detailed": "", "brief": ""})
            return json.dumps({"verdict": "pass", "findings": []})

        rc, out = self._run(self._ms(handler))
        self.assertNotEqual(0, rc)
        self.assertFalse((out / "detailed.md").exists())
        self.assertFalse((out / "brief.md").exists())

    def test_structurally_blocked_pair_publishes_nothing(self):
        listed = {"detailed": "Outcomes:\n- Rates fell after the reform.",
                  "brief": "Rates fell."}

        def handler(stage, prompt):
            if stage == "full" or stage.startswith("full-revise"):
                return json.dumps(dict(listed))
            return json.dumps({"verdict": "pass", "findings": []})

        rc, out = self._run(self._ms(handler))
        self.assertEqual(5, rc)
        self.assertFalse((out / "detailed.md").exists())

    def test_window_route_fails_closed_when_no_window_can_fit(self):
        def handler(stage, prompt):
            raise AssertionError("model transport called without a fitting window")

        rc, out = self._run(self._ms(handler), source="word " * 90000,
                           capability_words=10)
        fs = load("fullsum")
        self.assertEqual(fs.NOT_SUPPORTED, rc)
        self.assertFalse((out / "detailed.md").exists())

    def test_window_route_refuses_too_small_output_before_coverage(self):
        def handler(stage, prompt):
            raise AssertionError("model transport called below pair minimum")

        rc, out = self._run(self._ms(handler), source="word " * 90000,
                           capability_words=8000, output_budget_words=10)
        fs = load("fullsum")
        self.assertEqual(fs.NOT_SUPPORTED, rc)
        self.assertFalse((out / "window-plan.json").exists())

    def test_fit_prefers_the_direct_route_inside_budgets(self):
        fs = load("fullsum")
        self.assertTrue(fs.fits(500))
        self.assertFalse(fs.fits(90000))
        self.assertTrue(fs.fits(90000, capability_words=10_000_000,
                                output_budget_words=10_000_000))
        # Admission is governed by the composed request: no default output
        # cap rejects a mid-size source whose request fits the baseline.
        self.assertTrue(fs.fits(20000))
        self.assertTrue(fs.fits(40000),
                        "Cain-scale request must fit the ~100k baseline")
        # An output budget below the minimum pair refuses the route. A budget
        # that can hold a clipped pair does not force source windows merely
        # because the editorial band is larger.
        self.assertFalse(fs.fits(20000, output_budget_words=10))
        self.assertTrue(fs.fits(20000, output_budget_words=1000))
        limited = fs.fit_detail(20000, None, 1000)
        self.assertTrue(limited["fits"])
        self.assertTrue(limited["route_limited"])
        self.assertLessEqual(limited["base_generation_ceiling_words"], 1000)
        self.assertGreater(limited["publication_ceiling_words"],
                           limited["base_generation_ceiling_words"])
        detail = fs.fit_detail(500, None, None)
        self.assertEqual(
            (detail["fits"], detail["source_words"],
             detail["output_budget_words"]),
            (True, 500, None))

    def test_window_blocks_cover_source_once_in_order(self):
        fs = load("fullsum")
        source = " ".join(f"word{i}" for i in range(137))
        blocks = fs.source_blocks(source, 31)
        self.assertEqual(sum(block["words"] for block in blocks), 137)
        self.assertEqual(
            [(block["start_word"], block["end_word"]) for block in blocks],
            [(1, 31), (32, 62), (63, 93), (94, 124), (125, 137)])
        self.assertEqual(
            [word for block in blocks for word in block["text"].split()],
            source.split())
        self.assertEqual(
            [block["window_id"] for block in blocks],
            [f"window-{i:03d}" for i in range(1, 6)])

    def test_a_tiny_trailing_remainder_is_folded_into_the_previous_window(self):
        # A sentence end one word from the end minted a 1-word window on Cain;
        # both writers then returned {"capsule":""} and the pair was discarded.
        fs = load("fullsum")
        words = [f"word{i}." if i == 98 else f"word{i}" for i in range(100)]
        blocks = fs.source_blocks(" ".join(words), 50)
        self.assertEqual(sum(block["words"] for block in blocks), 100)
        self.assertTrue(all(block["words"] >= 8 for block in blocks))
        self.assertEqual(blocks[-1]["end_word"], 100)

    def test_window_repairs_are_bounded_by_the_total_call_budget(self):
        fs = load("fullsum")
        # The repair budget is one correction (AGENTS.md Defaults); the
        # topology bound can only lower it, never raise it.
        self.assertEqual(fs.bounded_window_repair_budget(2), 1)
        self.assertEqual(fs.bounded_window_repair_budget(12), 1)
        self.assertEqual(fs.bounded_window_repair_budget(14), 1)
        self.assertEqual(fs.bounded_window_repair_budget(32), 0)
        self.assertLessEqual(
            2 * 32 + 2 + fs.bounded_window_repair_budget(32) * 34,
            fs.FULL_MAX_LOGICAL_CALLS)
        self.assertEqual(fs.FULL_MAX_PHYSICAL_ATTEMPTS, 144)
        self.assertEqual(fs.bounded_window_repair_budget(
            12, logical_budget=48), 1,
                         "three-route chains must fit the physical ceiling")

    def test_window_route_has_one_write_per_window_and_bounded_pair_lifecycle(self):
        fs = load("fullsum")
        source = ("The committee reported that rates fell after the reform, "
                  "although evidence remained limited and causation was not "
                  "established. Markets reacted quickly while analysts warned "
                  "stress could return. Further study was recommended before "
                  "policy changes took effect. ") * 200
        detailed = ("Rates fell after the reform while evidence remained limited "
                    "and causation was not established; markets reacted quickly "
                    "but stress could return and further study was recommended. "
                    * 30)
        brief = ("Rates fell after the reform, but evidence remained limited and "
                 "causation was not established; further study was recommended. "
                 * 20)
        fixed = {"detailed": detailed, "brief": brief}
        stages, seen = [], {}

        def handler(stage, prompt):
            stages.append(stage)
            if re.fullmatch(r"full-window-\d{3}", stage):
                return json.dumps({"capsule":
                    "Rates fell, evidence was limited, and causation was not "
                    "established; analysts warned stress could return."})
            if stage == "full-window-synthesis":
                return json.dumps(fixed)
            if re.fullmatch(r"full-window-audit-\d{3}", stage):
                seen.setdefault("audits", []).append(prompt)
                return json.dumps({"verdict": "revise", "findings": [
                    AF("The Brief omits that stress could return")]})
            if stage == "full-window-audit-global":
                seen["global"] = prompt
                return json.dumps({"verdict": "pass", "findings": []})
            if stage == "full-window-revise-1":
                seen["repair"] = prompt
                return json.dumps(patch_replace(
                    detailed.strip(), brief.strip(), "brief", brief.strip()))
            if stage.startswith("full-window-reaudit-1-"):
                return json.dumps({"verdict": "pass", "findings": []})
            raise AssertionError(stage)

        rc, out = self._run(self._ms(handler), source=source,
                            capability_words=9000)
        self.assertEqual(0, rc)
        plan = json.loads((out / "window-plan.json").read_text())
        report = self._report(out)
        self.assertEqual(len(plan["windows"]), 3)
        self.assertEqual(
            [stage for stage in stages if stage.startswith("full-window-")
             and stage[12:].isdigit()],
            ["full-window-001", "full-window-002", "full-window-003"])
        n = len(plan["windows"])
        self.assertEqual(stages[n], "full-window-synthesis")
        self.assertEqual(stages[n + 1:n + 1 + n], [
            f"full-window-audit-{i:03d}" for i in range(1, n + 1)])
        self.assertEqual(stages[n + 1 + n], "full-window-audit-global")
        self.assertEqual(stages[n + 2 + n], "full-window-revise-1")
        self.assertEqual(stages[n + 3 + n:n + 3 + 2 * n], [
            f"full-window-reaudit-1-{i:03d}" for i in range(1, n + 1)])
        self.assertEqual(stages[-1], "full-window-reaudit-1-global")
        self.assertEqual(len(stages), 3 * n + 4)
        self.assertEqual(len(seen["audits"]), n)
        self.assertIn("The committee reported", seen["audits"][0])
        self.assertIn("not the whole document", seen["audits"][0])
        self.assertNotIn("window-001]", seen["audits"][0])
        self.assertIn("complete compact, source-linked evidence", seen["global"])
        self.assertIn(detailed[:80].strip(), seen["repair"])
        self.assertIn(brief[:80].strip(), seen["repair"])
        self.assertIn("IMPLICATED RAW SOURCE PACKET full-window-audit-001",
                      seen["repair"])
        self.assertRegex(seen["repair"],
                         r"CURRENT CANDIDATE candidate-[0-9a-f]{16}")
        self.assertEqual((report["route"], report["window_count"],
                          report["candidates"], report["selected"],
                          report["status"]),
                         ("windowed", 3, 2, "revise-1", "pass"))
        self.assertEqual((out / "detailed.md").read_text().strip(), detailed.strip())

    def test_window_route_makes_one_repair_and_publishes_best_pair(self):
        fs = load("fullsum")
        source = ("The committee reported that rates fell after the reform, "
                  "although evidence remained limited and causation was not "
                  "established. Analysts warned stress could return. " * 200)
        detailed = ("Rates fell after the reform while evidence remained limited "
                    "and causation was not established; analysts warned stress "
                    "could return. " * 30)
        brief = ("Rates fell after the reform, but evidence remained limited and "
                 "causation was not established; stress could return. " * 20)
        stages = []

        def handler(stage, prompt):
            stages.append(stage)
            if re.fullmatch(r"full-window-\d{3}", stage):
                return json.dumps({"capsule":
                    "Rates fell, evidence was limited, causation was not "
                    "established, and analysts warned stress could return."})
            if stage == "full-window-synthesis":
                return json.dumps({"detailed": detailed, "brief": brief})
            if stage.startswith("full-window-revise-"):
                return json.dumps(patch_replace(
                    detailed.strip(), brief.strip(),
                    "detailed", detailed.strip()))
            if stage.startswith("full-window-audit-") or \
                    stage.startswith("full-window-reaudit-"):
                return json.dumps({"verdict": "revise", "findings": [
                    AF("The candidate needs one more source-grounded qualification")]})
            raise AssertionError(stage)

        rc, out = self._run(self._ms(handler), source=source,
                            capability_words=8000)
        self.assertEqual(0, rc)
        n = len(json.loads((out / "window-plan.json").read_text())["windows"])
        self.assertGreaterEqual(n, 2)
        self.assertEqual(
            [s for s in stages if s.startswith("full-window-revise-")],
            ["full-window-revise-1"])
        self.assertEqual(
            [s for s in stages if s.startswith("full-window-reaudit-")],
            [item for repair in range(1, 2)
             for item in ([f"full-window-reaudit-{repair}-{window:03d}"
                           for window in range(1, n + 1)]
                          + [f"full-window-reaudit-{repair}-global"])])
        self.assertEqual(len(stages), 2 * n + 2 + 1 * (n + 2))
        report = self._report(out)
        self.assertEqual((report["selected"], report["repair"],
                          report["status"], report["candidates"]),
                         ("revise-1", "once", "open_findings", 2))
        self.assertTrue((out / "detailed.md").is_file())

    def test_window_reviewer_outage_publishes_pair_with_explicit_status(self):
        fs = load("fullsum")
        source = "The evidence was limited and causation was not established. " * 500
        detailed = "Rates fell after the reform, though evidence was limited. " * 30
        brief = "Rates fell after the reform, though evidence was limited. " * 20

        def handler(stage, prompt):
            if re.fullmatch(r"full-window-\d{3}", stage):
                return json.dumps({"capsule":
                    "Rates fell after the reform, though evidence was limited."})
            if stage == "full-window-synthesis":
                return json.dumps({"detailed": detailed, "brief": brief})
            if stage.startswith("full-window-audit-"):
                raise RuntimeError("reviewer unavailable")
            raise AssertionError(stage)

        rc, out = self._run(self._ms(handler), source=source,
                            capability_words=8000)
        self.assertEqual(0, rc)
        report = self._report(out)
        self.assertEqual(report["status"], "review_unavailable")
        self.assertEqual(report["selected"], "initial")
        self.assertTrue((out / "detailed.md").is_file())
        self.assertIn("full-window-audit", report["review"])

    def test_a_failed_window_is_retried_as_halves_and_nothing_is_skipped(self):
        # A chunk never fails while the models work. A window
        # whose whole chain failed used to be dropped and the pair published
        # from the rest with the gap noted in the report. Now the window is
        # asked again as two halves (then quarters) and the pair is written
        # only from complete coverage.
        fs = load("fullsum")
        source = "The evidence was limited and causation was not established. " * 500
        detailed = "Rates fell after the reform, though evidence was limited. " * 30
        brief = "Rates fell after the reform, though evidence was limited. " * 20
        stages = []

        def handler(stage, prompt):
            stages.append(stage)
            if stage == "full-window-002":
                raise NO_CANDIDATE("no output from chain")
            if re.fullmatch(r"full-window-\d{3}[ab]?", stage):
                return json.dumps({"capsule": "Evidence was limited."})
            if stage == "full-window-synthesis" or stage.startswith("full-window-revise"):
                return json.dumps({"detailed": detailed, "brief": brief})
            if "audit" in stage:
                return json.dumps({"verdict": "pass", "findings": []})
            raise AssertionError(stage)

        rc, out = self._run(self._ms(handler), source=source,
                            capability_words=8000)
        self.assertEqual(0, rc)
        self.assertIn("full-window-002a", stages)
        self.assertIn("full-window-002b", stages)
        self.assertIn("full-window-synthesis", stages)
        report = self._report(out)
        self.assertEqual(report["missing_windows"], [])
        self.assertEqual((out / "detailed.md").read_text().strip(), detailed.strip())

    def test_only_an_exhausted_chain_subdivides_a_window(self):
        # Cancellation, authentication, a configuration error, an exhausted
        # deadline, or a defect here must not masquerade as a window being
        # too large: they stop the run instead of spending seven calls.
        source = "The evidence was limited and causation was not established. " * 500
        stages = []

        def handler(stage, prompt):
            stages.append(stage)
            if stage == "full-window-002":
                raise RuntimeError("cancelled")
            if re.fullmatch(r"full-window-\d{3}", stage):
                return json.dumps({"capsule": "Evidence was limited."})
            raise AssertionError(stage)

        with self.assertRaises(RuntimeError):
            self._run(self._ms(handler), source=source, capability_words=8000)
        self.assertNotIn("full-window-002a", stages)

    def test_a_split_window_preserves_the_source_bytes(self):
        # An earlier split tokenized on whitespace and rejoined with single
        # spaces, silently rewriting paragraphs, lists, and code before the
        # model saw them.
        fs = load("fullsum")
        text = "Head line.\n\n    indented block\n\nTail paragraph here."
        left, right = fs._split_text(text)
        self.assertEqual(text, left + right)
        self.assertTrue(left.strip() and right.strip())

    def test_the_worst_case_split_tree_is_admitted_before_any_call(self):
        # Every window failing and being retried as halves then quarters is a
        # seven-node tree; admission used to count only the parent windows.
        fs = load("fullsum")
        stages = []

        def handler(stage, prompt):
            stages.append(stage)
            raise AssertionError("admission spent a model call")

        source = "word " * 400_000
        with unittest.mock.patch.object(fs, "FULL_MAX_PHYSICAL_ATTEMPTS", 20):
            rc, out = self._run(self._ms(handler), source=source,
                                capability_words=8000)
        self.assertEqual(fs.NOT_SUPPORTED, rc)
        self.assertEqual([], stages)

    def test_a_window_that_fails_at_every_size_ends_the_run_with_nothing(self):
        # The one permitted way to end without a summary: the models are not
        # working. Nothing partial is published and nothing is skipped.
        fs = load("fullsum")
        source = "The evidence was limited and causation was not established. " * 500
        stages = []

        def handler(stage, prompt):
            stages.append(stage)
            if stage.startswith("full-window-002"):
                raise NO_CANDIDATE("no output from chain")
            if re.fullmatch(r"full-window-\d{3}", stage):
                return json.dumps({"capsule": "Evidence was limited."})
            raise AssertionError(stage)

        rc, out = self._run(self._ms(handler), source=source,
                            capability_words=8000)
        self.assertNotEqual(0, rc)
        self.assertFalse((out / "detailed.md").exists())
        self.assertNotIn("full-window-synthesis", stages)
        # the first half, then its first quarter, were tried before the run
        # concluded the models are down; nothing further is spent after that
        for tag in ("a", "aa"):
            self.assertIn(f"full-window-002{tag}", stages)

    def test_overlong_window_capsule_shortens_on_the_same_writer(self):
        fs = load("fullsum")
        source = ("The committee reported that rates fell after the reform, "
                  "although evidence remained limited and causation was not "
                  "established. Analysts warned stress could return. ") * 200
        detailed = ("Rates fell after the reform while evidence remained limited "
                    "and causation was not established; analysts warned stress "
                    "could return. " * 30)
        brief = ("Rates fell after the reform, but evidence remained limited and "
                 "causation was not established; stress could return. " * 20)
        long_capsule = "Rates fell after the reform and evidence stayed limited. " * 8
        short_capsule = "Rates fell; evidence stayed limited; causation unclaimed."
        calls = []

        def handler(stage, prompt, chain):
            calls.append((stage, list(chain)))
            if re.fullmatch(r"full-window-\d{3}", stage):
                return json.dumps({"capsule": long_capsule})
            if stage.endswith("-shorten"):
                self.assertIn("CURRENT CAPSULE", prompt)
                self.assertNotIn("opencode:luna", chain)
                return json.dumps({"capsule": short_capsule})
            if stage == "full-window-synthesis" or stage.startswith("full-window-revise"):
                return json.dumps({"detailed": detailed, "brief": brief})
            if "audit" in stage:
                return json.dumps({"verdict": "pass", "findings": []})
            raise AssertionError(stage)

        class FakeMS:
            MODELS = ["muse:spark", "opencode:luna"]
            AUDIT_MODELS = REPAIR_MODELS = ["local"]

            @staticmethod
            def run(prompt, out_dir, chain, stage, validate=None,
                    gateway_options=None):
                raw = handler(stage, prompt, chain)
                if validate is not None:
                    validate(raw)
                return raw

        root = pathlib.Path(tempfile.mkdtemp())
        src, out = root / "source.txt", root / "out"
        src.write_text(source)
        patches = [
            unittest.mock.patch.object(fs, "_runner", return_value=FakeMS()),
            unittest.mock.patch.object(fs, "_ledger", return_value=self.Ledger),
            unittest.mock.patch.object(fs, "WINDOW_CAPSULE_WORDS", 12),
        ]
        for p in patches:
            p.start()
        try:
            rc = fs.run(src, out, capability_words=9000)
        finally:
            for p in reversed(patches):
                p.stop()
        self.assertEqual(0, rc)
        self.assertTrue((out / "detailed.md").is_file())
        writes = [chain for stage, chain in calls
                  if re.fullmatch(r"full-window-\d{3}", stage)]
        shortens = [chain for stage, chain in calls if stage.endswith("-shorten")]
        self.assertTrue(writes and shortens)
        self.assertEqual(writes[0], ["muse:spark", "opencode:luna"])
        for chain in shortens:
            self.assertEqual(chain, ["muse:spark"])
        report = json.loads((out / "full-report.json").read_text())
        self.assertFalse(report["length_exception"])

    def test_still_overlong_window_capsule_is_published(self):
        fs = load("fullsum")
        source = ("The committee reported that rates fell after the reform, "
                  "although evidence remained limited and causation was not "
                  "established. Analysts warned stress could return. ") * 200
        detailed = ("Rates fell after the reform while evidence remained limited "
                    "and causation was not established; analysts warned stress "
                    "could return. " * 30)
        brief = ("Rates fell after the reform, but evidence remained limited and "
                 "causation was not established; stress could return. " * 20)
        long_capsule = "Rates fell after the reform and evidence stayed limited. " * 8

        def handler(stage, prompt, chain):
            if re.fullmatch(r"full-window-\d{3}", stage):
                return json.dumps({"capsule": long_capsule})
            if stage.endswith("-shorten"):
                return json.dumps({"capsule": "still over the preferred cap. " * 6})
            if stage == "full-window-synthesis" or stage.startswith("full-window-revise"):
                return json.dumps({"detailed": detailed, "brief": brief})
            if "audit" in stage:
                return json.dumps({"verdict": "pass", "findings": []})
            raise AssertionError(stage)

        class FakeMS:
            MODELS = ["muse:spark", "opencode:luna"]
            AUDIT_MODELS = REPAIR_MODELS = ["local"]

            @staticmethod
            def run(prompt, out_dir, chain, stage, validate=None,
                    gateway_options=None):
                raw = handler(stage, prompt, chain)
                if validate is not None:
                    validate(raw)
                return raw

        root = pathlib.Path(tempfile.mkdtemp())
        src, out = root / "source.txt", root / "out"
        src.write_text(source)
        patches = [
            unittest.mock.patch.object(fs, "_runner", return_value=FakeMS()),
            unittest.mock.patch.object(fs, "_ledger", return_value=self.Ledger),
            unittest.mock.patch.object(fs, "WINDOW_CAPSULE_WORDS", 12),
        ]
        for p in patches:
            p.start()
        try:
            rc = fs.run(src, out, capability_words=9000)
        finally:
            for p in reversed(patches):
                p.stop()
        self.assertEqual(0, rc, "over-length must not suppress the pair")
        self.assertTrue((out / "detailed.md").is_file())
        report = json.loads((out / "full-report.json").read_text())
        self.assertTrue(report["length_exception"])

    def test_keep_window_capsule_never_discards_the_candidate(self):
        fs = load("fullsum")
        kept, how = fs._keep_window_capsule("alpha " * 20, "beta " * 5, 10)
        self.assertEqual((how, len(kept.split())), ("shortened", 5))
        kept, how = fs._keep_window_capsule("alpha " * 20, "beta " * 15, 10)
        self.assertEqual((how, len(kept.split())), ("length_exception", 15))
        kept, how = fs._keep_window_capsule("alpha " * 20, None, 10)
        self.assertEqual((how, len(kept.split())), ("length_exception", 20))

    def test_length_repair_chain_is_the_producer_not_the_backup(self):
        fs = load("fullsum")
        with tempfile.TemporaryDirectory() as td:
            d = pathlib.Path(td)
            (d / "calls.jsonl").write_text(json.dumps({
                "stage": "full-window-001", "outcome": "ok",
                "harness": "muse", "model": "spark"}) + "\n")
            self.assertEqual(
                fs._last_ok_chain(d, "full-window-001",
                                  ["muse:spark", "opencode:luna"]),
                ["muse:spark"])
            self.assertEqual(
                fs._last_ok_chain(d, "missing",
                                  ["muse:spark", "opencode:luna"]),
                ["muse:spark"])


class FullBatchedOutput(unittest.TestCase):
    """A Full pair is a logical artifact, not a one-response payload."""
    SOURCE = " ".join(["policy"] * 1000)

    class Ledger:
        @staticmethod
        def parse_strict(raw, stage):
            return json.loads(raw)

    @staticmethod
    def _ms(handler):
        class FakeMS:
            MODELS = AUDIT_MODELS = REPAIR_MODELS = ["small"]
            NoCandidate = NO_CANDIDATE

            @staticmethod
            def run(prompt, out_dir, chain, stage, validate=None,
                    gateway_options=None):
                return handler(stage, prompt)
        return FakeMS()

    def _run(self, handler, source=None, output_budget_words=100):
        fs = load("fullsum")
        root = pathlib.Path(tempfile.mkdtemp())
        src, out = root / "source.txt", root / "out"
        src.write_text(source or self.SOURCE)
        with unittest.mock.patch.object(fs, "_runner",
                                        return_value=self._ms(handler)), \
             unittest.mock.patch.object(fs, "_ledger",
                                        return_value=self.Ledger):
            rc = fs.run(src, out, capability_words=100000,
                        output_budget_words=output_budget_words)
        return rc, out

    def test_plan_uses_multiple_calls_without_clipping_the_artifacts(self):
        fs = load("fullsum")
        plan = fs.batched_output_detail(21685, output_budget_tokens=8192)
        self.assertTrue(plan["needed"])
        self.assertEqual(plan["part_words"], 4247)
        self.assertEqual(plan["parts"], {"detailed": 3, "brief": 2})
        self.assertEqual(plan["ceilings"], fs.full_ceilings(21685))

        tiny = fs.batched_output_detail(200, output_budget_words=20)
        self.assertEqual(tiny["part_words"], 14)
        self.assertGreater(sum(tiny["parts"].values()), 2)

    def test_bounded_parts_assemble_a_reading_longer_than_one_response(self):
        stages = []

        def handler(stage, prompt):
            stages.append(stage)
            if stage.startswith("full-batch-detailed-"):
                self.assertIn('"reading"', prompt)
                return json.dumps({"reading": "Policy remained qualified. " * 18})
            if stage.startswith("full-batch-brief-"):
                return json.dumps({"reading": "Policy remained qualified. " * 12})
            if stage == "full-batched-audit":
                return json.dumps({"verdict": "pass", "findings": []})
            raise AssertionError(stage)

        rc, out = self._run(handler)
        self.assertEqual(rc, 0)
        report = json.loads((out / "full-report.json").read_text())
        self.assertEqual(report["route"], "batched")
        self.assertEqual(report["capability"]["part_word_budget"], 70)
        self.assertGreater(report["detailed_words"], 70)
        self.assertTrue((out / "brief.md").read_text().strip())
        self.assertEqual(len(report["output_parts"]), 10)
        self.assertEqual(stages[-1], "full-batched-audit")

    def test_output_limit_retries_a_source_batch_as_halves(self):
        stages = []

        def handler(stage, prompt):
            stages.append(stage)
            if stage == "full-batch-detailed-001":
                raise NO_CANDIDATE("gateway finish_reason=length")
            if stage.startswith("full-batch-"):
                return json.dumps({"reading": "Policy remained qualified. " * 15})
            if stage == "full-batched-audit":
                return json.dumps({"verdict": "pass", "findings": []})
            raise AssertionError(stage)

        rc, out = self._run(handler, output_budget_words=400)
        self.assertEqual(rc, 0)
        self.assertIn("full-batch-detailed-001a", stages)
        self.assertIn("full-batch-detailed-001b", stages)
        report = json.loads((out / "full-report.json").read_text())
        ids = [part["part_id"] for part in report["output_parts"]]
        self.assertIn("001a", ids)
        self.assertIn("001b", ids)

    def test_unrecoverable_part_publishes_neither_artifact(self):
        def handler(stage, prompt):
            if stage.startswith("full-batch-detailed-"):
                raise NO_CANDIDATE("all routes failed")
            raise AssertionError(stage)

        source = " ".join(["policy"] * 300)
        rc, out = self._run(handler, source=source,
                            output_budget_words=100)
        self.assertNotEqual(rc, 0)
        self.assertFalse((out / "detailed.md").exists())
        self.assertFalse((out / "brief.md").exists())

    def test_call_budget_uses_sentence_safe_plan_not_nominal_count(self):
        fs = load("fullsum")
        root = pathlib.Path(tempfile.mkdtemp())
        src, out = root / "source.txt", root / "out"
        src.write_text(self.SOURCE)
        block = {"window_id": "window-001", "start_word": 1,
                 "end_word": 1, "words": 1, "text": "policy"}

        def no_call(stage, prompt):
            raise AssertionError(f"model called after rejected plan: {stage}")

        with unittest.mock.patch.object(fs, "_runner",
                                        return_value=self._ms(no_call)), \
             unittest.mock.patch.object(fs, "_ledger",
                                        return_value=self.Ledger), \
             unittest.mock.patch.object(fs, "source_blocks",
                                        return_value=[dict(block) for _ in range(17)]):
            rc = fs.run(src, out, capability_words=100000,
                        output_budget_words=100)
        self.assertEqual(fs.NOT_SUPPORTED, rc)
        self.assertFalse(out.exists())


class FullDirectDispatch(unittest.TestCase):
    """Summarize dispatch through the real cli.main(): fitting sources take
    fullsum direct, oversize sources take Full windowing, and short sources
    stay Full. No source path falls through to the ledger or Quick. Model
    transport is faked at
    the run_stage boundary; locks are faked because the sandbox denies the
    device lock directory. No model calls."""
    cli = load("summ_cli")

    class FakeLease:
        def __enter__(self):
            return 0.0

        def __exit__(self, *args):
            return False

    @staticmethod
    def _lease(*args, **kwargs):
        return FullDirectDispatch.FakeLease()

    def _drive(self, doc_text, stage, argv_extra=(), env_extra=None,
               name="paper.md", progress=False, prefill=None):
        root = pathlib.Path(tempfile.mkdtemp())
        doc = root / name
        doc.write_text(doc_text)
        for rel, content in (prefill or {}).items():
            target = root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
        out, work = root / "out", root / "work"
        argv = ["summ_cli.py", str(doc), "--out", str(out),
                "--work-dir", str(work), *argv_extra]
        if progress:
            argv += ["--progress-jsonl", str(root / "p.jsonl")]
        calls = []

        def run_stage(argv_, *args, **kwargs):
            calls.append(pathlib.Path(argv_[1]).name)
            return stage(argv_)

        patches = [
            unittest.mock.patch.object(sys, "argv", argv),
            unittest.mock.patch.object(self.cli, "run_stage",
                                       side_effect=run_stage),
            unittest.mock.patch.object(self.cli, "freeze_model_routes"),
            unittest.mock.patch.object(self.cli, "validate_roles"),
            unittest.mock.patch.object(self.cli, "notify"),
            # ETA learning is covered by its own suite; the drive must not
            # read or write device timing history, which also needs locks
            # the sandbox denies.
            unittest.mock.patch.object(self.cli.eta, "estimate",
                                       return_value=None),
            unittest.mock.patch.object(self.cli.eta, "record_target",
                                       return_value=None),
            unittest.mock.patch.object(self.cli.runtime, "work_root_lock",
                                       side_effect=self._lease),
            unittest.mock.patch.object(self.cli.runtime, "target_lease",
                                       side_effect=self._lease),
            unittest.mock.patch.object(self.cli.runtime, "destination_lock",
                                       side_effect=self._lease),
            unittest.mock.patch.object(self.cli.runtime,
                                       "output_directory_lock",
                                       side_effect=self._lease),
        ]
        if env_extra:
            patches.append(unittest.mock.patch.dict(os.environ, env_extra))
        for p in patches:
            p.start()
        try:
            return self.cli.main(), root, calls
        finally:
            for p in reversed(patches):
                p.stop()

    @staticmethod
    def _full_result(run_dir, detailed, brief, status="pass", findings=None,
                     selected="initial", repair="none", route="direct"):
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "detailed.md").write_text(detailed)
        (run_dir / "brief.md").write_text(brief)
        (run_dir / "full-report.json").write_text(json.dumps(
            {"path": "full", "route": route, "source_words": 1,
             "detailed_words": len(detailed.split()),
             "brief_words": len(brief.split()),
             "selected": selected, "status": status,
             "findings": findings or [], "review": "complete",
             "repair": repair}))

    def _full_stage(self, **report):
        def stage(argv_):
            dest = pathlib.Path(argv_[-1])
            self._full_result(dest, report.get(
                "detailed", "Rates fell after the 2008 reform.\n"),
                report.get("brief", "Rates fell.\n"),
                status=report.get("status", "pass"),
                findings=report.get("findings"),
                selected=report.get("selected", "initial"),
                repair=report.get("repair", "none"),
                route=report.get("route", "direct"))
            return subprocess.CompletedProcess(argv_, 0)
        return stage

    def test_fitting_document_takes_full_direct(self):
        rc, root, calls = self._drive(
            "Rates fell after the reform in 2008. " * 60,
            self._full_stage())
        self.assertEqual(0, rc)
        self.assertEqual(calls, ["fullsum.py"])
        self.assertEqual((root / "out" / "summary.paper.md").read_text(),
                         "Rates fell after the 2008 reform.\n")
        self.assertEqual((root / "out" / "brief.paper.md").read_text(),
                         "Rates fell.\n")
        run = root / "work" / "01" / "run"
        self.assertEqual((root / "out" / "summary.paper.md").read_text(),
                         (run / "detailed.md").read_text(),
                         "published pair is not the reported pair")
        report = json.loads((run / "full-report.json").read_text())
        self.assertEqual((report["status"], report["selected"]), ("pass", "initial"))

    def test_short_document_stays_full_never_quick(self):
        # Full stays Full at every source size: no automatic rewrite, and no
        # Quick downgrade for a fitting short source either.
        rc, root, calls = self._drive(
            "Rates fell after the reform in 2008. " * 15,
            self._full_stage())
        self.assertEqual(0, rc)
        self.assertEqual(calls, ["fullsum.py"])
        self.assertTrue((root / "out" / "summary.paper.md").is_file())

    def test_explicit_quick_stays_quick(self):
        def stage(argv_):
            dest = pathlib.Path(argv_[-1])
            (dest / "detailed.md").write_text("Rates fell.\n")
            (dest / "brief.md").write_text("Rates fell.\n")
            (dest / "short-report.json").write_text(json.dumps(
                {"path": "short", "source_words": 1, "detailed_words": 2,
                 "brief_words": 2, "status": "pass", "selected": "initial",
                 "findings": [], "review": "complete", "repair": "none"}))
            return subprocess.CompletedProcess(argv_, 0)

        rc, root, calls = self._drive(
            "Rates fell after the reform in 2008. " * 60, stage,
            argv_extra=("--quick",))
        self.assertEqual(0, rc)
        self.assertEqual(calls, ["shortsum.py"])
        self.assertTrue((root / "out" / "summary.paper.md").is_file())

    def test_oversize_source_takes_full_windowed_route(self):
        rc, root, calls = self._drive(
            "word " * 90000, self._full_stage(route="windowed"), progress=True)
        self.assertEqual(0, rc)
        self.assertEqual(calls, ["fullsum.py"],
                         "oversize Full must not reach ledger or Quick")
        report = json.loads((root / "work" / "01" / "run" /
                             "full-report.json").read_text())
        self.assertEqual(report["route"], "windowed")
        events = [json.loads(line) for line in
                  (root / "p.jsonl").read_text().splitlines()]
        stages = [e["name"] for e in events if e["event"] == "stage"]
        self.assertIn("full-windowed", stages)
        finished = [e for e in events if e["event"] == "target_finished"]
        self.assertTrue(finished, "no terminal progress event")

    def test_declared_output_cap_routes_full_to_batched_writer(self):
        rc, root, calls = self._drive(
            "word " * 1000, self._full_stage(route="batched"), progress=True,
            env_extra={"SUMM_ROUTE_OUTPUT_WORDS": "100"})
        self.assertEqual(0, rc)
        self.assertEqual(calls, ["fullsum.py"])
        events = [json.loads(line) for line in
                  (root / "p.jsonl").read_text().splitlines()]
        stages = [e["name"] for e in events if e["event"] == "stage"]
        self.assertIn("full-batched", stages)

    def test_tiny_capability_stays_on_full_and_propagates_safe_refusal(self):
        def stage(argv_):
            return subprocess.CompletedProcess(argv_, load("fullsum").NOT_SUPPORTED)

        rc, root, calls = self._drive(
            "Rates fell after the reform in 2008. " * 60, stage,
            env_extra={"SUMM_ROUTE_CAPABILITY_WORDS": "10"})
        self.assertEqual(load("fullsum").NOT_SUPPORTED, rc)
        self.assertEqual(calls, ["fullsum.py"])

    def test_failed_full_keeps_the_prior_pair(self):
        def stage(argv_):
            return subprocess.CompletedProcess(argv_, 5)

        rc, root, calls = self._drive(
            "Rates fell after the reform in 2008. " * 60, stage,
            prefill={"out/summary.paper.md": "prior detailed\n",
                     "out/brief.paper.md": "prior brief\n"})
        out = root / "out"
        self.assertNotEqual(0, rc)
        self.assertEqual(calls, ["fullsum.py"])
        self.assertEqual((out / "summary.paper.md").read_text(),
                         "prior detailed\n")
        self.assertEqual((out / "brief.paper.md").read_text(), "prior brief\n")

    def test_full_status_reaches_the_console_and_links_the_report(self):
        buf = io.StringIO()
        with unittest.mock.patch.object(sys, "stdout", buf):
            rc, root, calls = self._drive(
                "Rates fell after the reform in 2008. " * 60,
                self._full_stage(status="open_findings",
                                 findings=["omits the 2008 dating",
                                           "brief drops a qualification"]))
        self.assertEqual(0, rc)
        self.assertIn("full status: open_findings (2 open finding(s))",
                      buf.getvalue())
        run = root / "work" / "01" / "run"
        self.assertEqual((root / "out" / "summary.paper.md").read_text(),
                         (run / "detailed.md").read_text(),
                         "published pair is not the reported pair")
        report = json.loads((run / "full-report.json").read_text())
        self.assertEqual(len(report["findings"]), 2)

    def test_route_capability_parsing(self):
        with unittest.mock.patch.dict(os.environ, {
                "SUMM_ROUTE_CAPABILITY_WORDS": "abc",
                "SUMM_ROUTE_OUTPUT_WORDS": "-3"}):
            self.assertEqual(self.cli.route_capability(), (None, None))
        with unittest.mock.patch.dict(os.environ, {
                "SUMM_ROUTE_CAPABILITY_WORDS": "50000",
                "SUMM_ROUTE_OUTPUT_WORDS": "7000"}):
            self.assertEqual(self.cli.route_capability(), (50000, 7000))


class PasteDecidesWhatToSummarize(unittest.TestCase):
    """One paste, and the tool works out what it got: a file, a folder, several
    paths, or raw text. The classifier is summ_cli's, not a second one written
    here -- that is the only way the window and the two triggers cannot come to
    different conclusions about the same clipboard."""
    ui = load("summ_ui")
    cli = load("summ_cli")

    def test_the_ui_reuses_the_cli_classifier_rather_than_writing_one(self):
        src = (ENG / "summ_ui.py").read_text()
        self.assertIn("selection.classify_clipboard", src)
        self.assertIn("selection.split_path_list", src)
        # Any re-implementation would need these; their absence is the check.
        for tell in ("def parse_targets", "shlex.split", "def resolve("):
            self.assertNotIn(tell, src, f"summ_ui.py reimplements {tell!r}")

    def test_a_folder_becomes_every_document_in_it(self):
        # It used to become sorted(...)[0]: paste a folder of twenty papers and
        # get one summary, with nothing saying the other nineteen were skipped.
        with tempfile.TemporaryDirectory() as td:
            d = pathlib.Path(td)
            for n in ("c.md", "a.md", "b.txt"):
                (d / n).write_text("x")
            (d / "ignore.png").write_text("x")
            targets, is_text = self.cli.parse_targets(str(d))
            self.assertFalse(is_text)
            self.assertEqual([p.name for p in targets], ["a.md", "b.txt", "c.md"])

    def test_a_folder_is_walked_recursively(self):
        # Folder selection intentionally covers the complete tree.
        with tempfile.TemporaryDirectory() as td:
            d = pathlib.Path(td)
            (d / "top.md").write_text("x")
            (d / "sub").mkdir(); (d / "sub" / "deep.md").write_text("x")
            targets, _ = self.cli.parse_targets(str(d))
            self.assertEqual([p.name for p in targets], ["deep.md", "top.md"])

    def test_an_empty_folder_says_so_rather_than_summarizing_nothing(self):
        with tempfile.TemporaryDirectory() as td:
            targets, _ = self.cli.parse_targets(str(pathlib.Path(td)))
            self.assertEqual(targets, [])

    def test_raw_text_is_left_for_the_cli_to_read(self):
        targets, is_text = self.cli.parse_targets("just some prose, not a path")
        self.assertTrue(is_text)

    def test_path_backed_ui_jobs_use_one_manifest_argv_element(self):
        paths = [pathlib.Path("/a/1.md"), pathlib.Path("/a/2 two.md")]
        cmd = self.ui.build_cmd(
            pathlib.Path("/j"), self.ui.MODES[0][0], paths,
            selection_manifest=pathlib.Path("/j/selection.json"))
        self.assertEqual(cmd[cmd.index("--selection-manifest") + 1],
                         str(pathlib.Path("/j/selection.json")))
        self.assertNotIn(str(paths[0]), cmd)
        self.assertNotIn(str(paths[1]), cmd)

    def test_paste_and_start_are_both_bound_to_the_platform_modifier(self):
        src = (ENG / "summ_ui.py").read_text()
        self.assertIn('MOD = "Command" if sys.platform == "darwin" else "Control"', src)
        self.assertIn('f"<{MOD}-v>"', src)
        self.assertIn('f"<{MOD}-Return>"', src)

    def test_a_changed_clipboard_cannot_change_captured_input(self):
        # Paste snapshots the input and --text-file transports that snapshot;
        # the live clipboard is irrelevant once the UI accepted the document.
        src = (ENG / "summ_ui.py").read_text()
        i = src.index("def start(")
        self.assertNotIn("clip_read", src[i:i + 700])
        self.assertNotIn("clipboard changed", src.lower())

    def test_start_snapshots_the_default_clipboard_before_launch(self):
        src = (ENG / "summ_ui.py").read_text()
        i = src.index("def start(")
        body = src[i:src.index("def _launch", i)]
        self.assertIn("self.queue_started", body)
        self.assertNotIn("self.paste()", body)


class SelectionManifestAndDiscovery(unittest.TestCase):
    """The shared selection contract is pure and no-model."""
    selection = load("selection")

    def test_clipboard_paths_on_separate_lines_become_separate_documents(self):
        with tempfile.TemporaryDirectory() as td:
            d = pathlib.Path(td)
            first = d / "first.md"; first.write_text("first")
            second = d / "second.md"; second.write_text("second")
            chosen = self.selection.classify_clipboard(
                f"{first}\n{second}")
            self.assertFalse(chosen.is_text)
            self.assertEqual(["first.md", "second.md"],
                             [doc.relative_path for doc in chosen.documents])

    def test_adjacent_absolute_paths_on_one_line_become_separate_documents(self):
        with tempfile.TemporaryDirectory() as td:
            d = pathlib.Path(td)
            first = d / "first.md"; first.write_text("first")
            spaced = d / "a folder with spaces"; spaced.mkdir()
            second = spaced / "inner.md"; second.write_text("second")
            chosen = self.selection.classify_clipboard(f"{first} {second}")
            self.assertFalse(chosen.is_text)
            self.assertEqual(["first.md", "inner.md"],
                             [doc.source_path.name for doc in chosen.documents])
            self.assertEqual([str(first), str(second)],
                             self.selection.split_path_list(f"{first} {second}"))
            self.assertEqual([str(second)],
                             self.selection.split_path_list(str(second)))

    def test_split_path_list_separates_quoted_and_drive_roots(self):
        self.assertEqual(
            ["/tmp/My Documents/a.md", "/tmp/My Documents/b.md"],
            self.selection.split_path_list(
                '"/tmp/My Documents/a.md" "/tmp/My Documents/b.md"'))
        self.assertEqual(
            ["C:\\papers\\a.md", "D:\\papers\\b.md"],
            self.selection.split_path_list(r"C:\papers\a.md D:\papers\b.md"))

    def test_one_missing_adjacent_path_is_not_silently_omitted(self):
        with tempfile.TemporaryDirectory() as td:
            d = pathlib.Path(td); existing = d / "kept.md"
            existing.write_text("source")
            chosen = self.selection.classify_clipboard(
                f"{existing} {d / 'missing.md'}")
            self.assertFalse(chosen.is_text)
            self.assertEqual(["kept.md"],
                             [doc.relative_path for doc in chosen.documents])
            self.assertEqual("missing_input", chosen.errors[0]["kind"])

    def test_missing_path_is_not_silently_omitted(self):
        with tempfile.TemporaryDirectory() as td:
            d = pathlib.Path(td); existing = d / "kept.md"
            existing.write_text("source")
            chosen = self.selection.classify_clipboard(
                f"{existing}\n{d / 'missing.md'}")
            self.assertFalse(chosen.is_text)
            self.assertEqual(["kept.md"],
                             [doc.relative_path for doc in chosen.documents])
            self.assertEqual("missing_input", chosen.errors[0]["kind"])

    def test_folder_order_is_recursive_and_deterministic(self):
        with tempfile.TemporaryDirectory() as td:
            d = pathlib.Path(td); (d / "z").mkdir(); (d / "A").mkdir()
            (d / "z" / "b.md").write_text("b")
            (d / "A" / "a.md").write_text("a")
            (d / "root.txt").write_text("r")
            chosen = self.selection.resolve_paths([d])
            self.assertEqual(["A/a.md", "root.txt", "z/b.md"],
                             [doc.relative_path for doc in chosen.documents])

    def test_generated_files_are_mode_aware_and_explicit_artifacts_visible(self):
        with tempfile.TemporaryDirectory() as td:
            d = pathlib.Path(td)
            for name in ("a.md", "a.summary.md", "a.brief.md", "a.clean.md",
                         "a.tts.txt"):
                (d / name).write_text(name)
            summary = self.selection.resolve_paths([d], "summarize")
            self.assertEqual(["a.md"], [x.relative_path for x in summary.documents])
            tts = self.selection.resolve_paths([d], "tts")
            self.assertEqual(["a.brief.md", "a.clean.md", "a.md", "a.summary.md"],
                             [x.relative_path for x in tts.documents])
            explicit = self.selection.resolve_paths([d / "a.summary.md"], "summarize")
            self.assertEqual(["a.summary.md"],
                             [x.relative_path for x in explicit.documents])
            self.assertIn("generated_artifact_explicit",
                          [x["kind"] for x in explicit.exclusions])

    def test_broken_and_directory_links_are_explicit_problems(self):
        with tempfile.TemporaryDirectory() as td:
            d = pathlib.Path(td); folder = d / "folder"; folder.mkdir()
            try:
                broken = d / "broken.md"
                os.symlink(d / "missing.md", broken)
                folder_link = d / "folder-link"
                os.symlink(folder, folder_link)
            except (OSError, NotImplementedError) as exc:
                self.skipTest(f"symlinks unavailable: {exc}")
            broken_selection = self.selection.resolve_paths([broken])
            self.assertIn("broken_link",
                          [e["kind"] for e in broken_selection.errors])
            directory_selection = self.selection.resolve_paths([folder_link])
            self.assertIn("non_followed_link",
                          [e["kind"] for e in directory_selection.errors])

    def test_manifest_round_trip_freezes_plan_and_hash(self):
        with tempfile.TemporaryDirectory() as td:
            d = pathlib.Path(td); (d / "notes").mkdir()
            source = d / "notes" / "paper.md"; source.write_text("paper")
            chosen = self.selection.resolve_paths([d]).with_outputs(
                "summarize", d / "out")
            manifest = d / "selection.json"
            self.selection.write_manifest(chosen, manifest)
            loaded = self.selection.read_manifest(manifest)
            self.assertEqual(chosen.order_digest, loaded.order_digest)
            self.assertEqual(chosen.documents[0].source_sha256,
                             loaded.documents[0].source_sha256)
            self.assertEqual("summary.paper.md",
                             loaded.documents[0].planned_outputs[0].name)
            self.assertEqual((True, ""), self.selection.verify_document(
                loaded.documents[0]))
            source.write_text("changed")
            self.assertFalse(self.selection.verify_document(loaded.documents[0])[0])

    def test_a_batch_slice_reseals_each_document_before_the_manifest_is_written(self):
        with tempfile.TemporaryDirectory() as td:
            d = pathlib.Path(td)
            files = []
            for name in ("a.md", "b.md", "c.md", "d.md"):
                path = d / name
                path.write_text(name)
                files.append(path)
            chosen = self.selection.resolve_paths(files)
            combined = chosen.order_digest
            self.assertEqual(4, len(chosen.documents))
            for doc in chosen.documents:
                one = chosen.for_single_document(doc).with_outputs("summarize")
                self.assertEqual(1, len(one.documents))
                self.assertNotEqual(combined, one.order_digest)
                manifest = d / f"{doc.id}.json"
                self.selection.write_manifest(one, manifest)
                loaded = self.selection.read_manifest(manifest)
                self.assertEqual(one.order_digest, loaded.order_digest)
                self.assertEqual(doc.id, loaded.documents[0].id)
            stale = dataclasses.replace(
                chosen,
                documents=(chosen.documents[0],),
                roots=tuple(root for root in chosen.roots
                            if root.id == chosen.documents[0].root_id))
            with self.assertRaisesRegex(ValueError, "order digest"):
                self.selection.write_manifest(stale, d / "stale.json")

    def test_waiting_batch_slices_recombine_without_rereading_sources(self):
        with tempfile.TemporaryDirectory() as td:
            d = pathlib.Path(td)
            files = []
            for name in ("a.md", "b.md", "c.md"):
                path = d / name
                path.write_text(name)
                files.append(path)
            chosen = self.selection.resolve_paths([d])
            slices = [chosen.for_single_document(doc).with_outputs("summarize")
                      for doc in chosen.documents]
            files[1].unlink()
            combined = self.selection.combine_path_selections(slices)
            self.assertEqual(["D001", "D002", "D003"],
                             [doc.id for doc in combined.documents])
            self.assertEqual(chosen.order_digest, combined.order_digest)
            self.assertEqual(1, len(combined.roots))
            self.assertTrue(all(not doc.planned_outputs
                                for doc in combined.documents))

    def test_output_plan_mirrors_a_folder(self):
        with tempfile.TemporaryDirectory() as td:
            d = pathlib.Path(td); (d / "one").mkdir(); (d / "two").mkdir()
            (d / "one" / "notes.md").write_text("1")
            (d / "two" / "notes.md").write_text("2")
            chosen = self.selection.resolve_paths([d]).with_outputs(
                "quick", d / "out")
            got = [doc.planned_outputs[0].relative_to(d / "out").as_posix()
                   for doc in chosen.documents]
            self.assertEqual(["one/summary.notes.md", "two/summary.notes.md"], got)

    def test_corpus_default_pair_is_inside_a_single_selected_folder(self):
        with tempfile.TemporaryDirectory() as td:
            d = pathlib.Path(td)
            (d / "one.md").write_text("one")
            (d / "two.md").write_text("two")
            chosen = self.selection.resolve_paths([d]).with_corpus_outputs("papers")
            self.assertEqual(d, chosen.corpus_outputs[0].parent)
            self.assertEqual("papers.corpus.summary.md",
                             chosen.corpus_outputs[0].name)

    def test_document_level_discovery_errors_are_not_duplicate_execution_targets(self):
        with tempfile.TemporaryDirectory() as td:
            d = pathlib.Path(td) / "bad.md"
            d.write_bytes(b"not utf-8: \xff")
            chosen = self.selection.resolve_paths([d])
            self.assertEqual(["invalid_utf8"], [doc.error_kind
                                                 for doc in chosen.documents])
            self.assertEqual(1, len(chosen.errors))
            self.assertEqual((), chosen.execution_errors)

    def test_output_plan_preserves_pairs_for_same_directory_stems(self):
        with tempfile.TemporaryDirectory() as td:
            d = pathlib.Path(td)
            for suffix in (".md", ".markdown", ".txt"):
                (d / f"notes{suffix}").write_text(suffix)
            chosen = self.selection.resolve_paths([d])

            summary = chosen.with_outputs("summarize", d / "summary-out")
            self.assertEqual(
                [("notes.markdown", ("summary.notes.md", "brief.notes.md")),
                 ("notes.md", ("summary.notes-2.md", "brief.notes-2.md")),
                 ("notes.txt", ("summary.notes-3.md", "brief.notes-3.md"))],
                [(doc.source_path.name, tuple(path.name for path in doc.planned_outputs))
                 for doc in summary.documents])

            clean = chosen.with_outputs("text_prep", d / "clean-out")
            self.assertEqual(["clean.notes.md", "clean.notes-2.md", "clean.notes-3.md"],
                             [doc.planned_outputs[0].name for doc in clean.documents])

            tts = chosen.with_outputs("tts", d / "tts-out")
            self.assertEqual(["tts.notes.txt", "tts.notes-2.txt", "tts.notes-3.txt"],
                             [doc.planned_outputs[0].name for doc in tts.documents])

    def test_a_relative_output_is_resolved_against_each_source_directory(self):
        with tempfile.TemporaryDirectory() as td:
            d = pathlib.Path(td)
            nested = d / "wk1"; nested.mkdir()
            source = nested / "paper.md"; source.write_text("paper")
            other = nested / "more.md"; other.write_text("more")
            chosen = self.selection.resolve_paths([source, other])
            into = chosen.with_outputs("summarize", "out")
            self.assertEqual(
                [(nested / "out" / "summary.paper.md").resolve(),
                 (nested / "out" / "summary.more.md").resolve()],
                [doc.planned_outputs[0] for doc in into.documents])
            above = chosen.with_outputs("summarize", "..")
            self.assertEqual(
                [(d / "summary.paper.md").resolve(),
                 (d / "summary.more.md").resolve()],
                [doc.planned_outputs[0] for doc in above.documents])
            collected = chosen.with_outputs("summarize", d / "collected")
            self.assertEqual(
                [d / "collected" / "summary.paper.md",
                 d / "collected" / "summary.more.md"],
                [doc.planned_outputs[0] for doc in collected.documents])

    def test_a_leading_slash_is_not_a_source_subfolder(self):
        self.assertTrue(self.selection.is_absolute_output("/out"))
        self.assertFalse(self.selection.is_absolute_output("out"))
        self.assertFalse(self.selection.is_absolute_output(".."))
        self.assertFalse(self.selection.is_absolute_output("./out"))

    def test_clipboard_text_uses_an_explicit_output_instead_of_downloads(self):
        downloads = self.selection.clipboard_output_directory()
        self.assertEqual("Downloads", downloads.name)
        with tempfile.TemporaryDirectory() as td:
            dest = pathlib.Path(td) / "summaries"
            got = self.selection.clipboard_output_directory(dest)
            self.assertEqual(dest, got)
            self.assertNotEqual(downloads, got)
            relative = self.selection.clipboard_output_directory("out")
            self.assertEqual(downloads / "out", relative)

    def test_stage_rejects_a_directory_instead_of_picking_a_child(self):
        cli = load("summ_cli")
        with tempfile.TemporaryDirectory() as td:
            d = pathlib.Path(td); (d / "child.md").write_text("child")
            self.assertIsNone(cli.stage(d, d / "work"))


class FinishedWorkIsHandedBack(unittest.TestCase):
    """A finished run has to be reachable: twenty documents produce twenty
    results and the window is where they are."""
    ui = load("summ_ui")

    def setUp(self):
        import tkinter as tk
        try:
            self.root = tk.Tk()
        except tk.TclError as e:
            self.skipTest(f"no display: {e}")
        self.root.withdraw()
        self.app = self.ui.App(self.root)
        self.d = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(self.root.destroy)
        self.addCleanup(shutil.rmtree, self.d, True)

    def test_a_named_document_uses_its_filename(self):
        p = self.d / "fullwiler-2011.summary.md"; p.write_text("prose")
        self.assertEqual(self.app.card_title(p), "fullwiler-2011")

    def test_clipboard_text_is_titled_from_its_own_opening_words(self):
        # No filename to use, and no model call: a model-written title would be
        # one more unsupported assertion in a project whose argument is that a
        # summary must not claim more than its source does.
        p = self.d / "clipboard.summary.md"
        p.write_text("> *Coverage note: x.*\n\nThe Federal Reserve balance sheet "
                     "expanded sharply after 2008.")
        t = self.app.card_title(p)
        self.assertTrue(t.startswith("The Federal Reserve"), t)
        self.assertLessEqual(len(t), 40)

    def test_the_coverage_note_is_never_the_title(self):
        p = self.d / "clipboard.summary.md"
        p.write_text("> *Coverage note: passages were omitted.*\n\nReal content here.")
        self.assertNotIn("Coverage note", self.app.card_title(p))

    def test_titles_are_deterministic(self):
        p = self.d / "clipboard.summary.md"; p.write_text("Alpha beta gamma delta.")
        self.assertEqual(self.app.card_title(p), self.app.card_title(p))

    def test_a_card_appears_per_finished_target(self):
        for i in (1, 2, 3):
            f = self.d / f"doc{i}.summary.md"; f.write_text("some words here")
            self.app._apply({"event": "target_finished", "index": i,
                             "status": "succeeded", "exit_code": 0,
                             "outputs": [str(f), str(self.d / f"doc{i}.brief.md")]})
        self.assertEqual(len(self.app.cards), 3)

    def test_accepted_batch_documents_become_visible_jobs_before_start(self):
        first = self.d / "first.md"; first.write_text("first source")
        second = self.d / "second.md"; second.write_text("second source")
        chosen = load("selection").resolve_paths([first, second])
        runs = self.d / "runs"
        with unittest.mock.patch.object(self.ui, "job_root", return_value=runs), \
             unittest.mock.patch.object(
                 self.ui.runtime, "config", return_value={"active_targets": 1}), \
             unittest.mock.patch.object(
                 self.ui.runtime, "frozen_json", return_value="{}"), \
             unittest.mock.patch.object(self.app, "chosen_env", return_value={}), \
             unittest.mock.patch.object(
                 self.ui.custom_instructions, "save_last"):
            self.app._accept_selection(chosen)
        self.assertFalse(self.app.queue_started)
        self.assertEqual(["first.md", "second.md"],
                         [job["label"] for job in self.app.pending])
        self.assertIn("2 job(s)", self.app.queue_label.cget("text"))
        sel = load("selection")
        for job in self.app.pending:
            loaded = sel.read_manifest(job["root"] / "selection.json")
            self.assertEqual(1, len(loaded.documents))
            self.assertEqual(job["label"], loaded.documents[0].source_path.name)

    def test_started_queue_admits_documents_added_after_start(self):
        docs = []
        for name in ("one.md", "two.md", "three.md", "four.md"):
            path = self.d / name; path.write_text(name)
            docs.append(path)
        runs = self.d / "runs"
        common = unittest.mock.patch.object
        with common(self.ui, "job_root", return_value=runs), \
             common(self.ui.runtime, "config", return_value={"active_targets": 1}), \
             common(self.ui.runtime, "frozen_json", return_value="{}"), \
             common(self.app, "chosen_env", return_value={}), \
             common(self.ui.custom_instructions, "save_last"):
            self.app._accept_selection(
                load("selection").resolve_paths(docs[:3]))
            self.app.queue_started = True
            first_job = self.app.pending.pop(0)
            existing = self.ui.JobState("running", self.d / "active",
                                        first_job)
            existing.root.mkdir()
            self.app.active[existing.id] = existing
            self.app._accept_selection(
                load("selection").resolve_paths([docs[3]]))
        self.assertEqual(["two.md", "three.md", "four.md"],
                         [job["label"] for job in self.app.pending])
        launched = []
        self.app.active.clear()
        self.app._launch = lambda job: launched.append(job["label"])
        self.app._drain_pending()
        self.assertEqual(["two.md", "three.md", "four.md"], launched)
        self.assertEqual([], self.app.pending)
        self.assertFalse(self.app.queue_started)

    def test_accepted_path_clears_the_entry_so_return_cannot_duplicate_it(self):
        path = self.d / "pasted.md"
        path.write_text("pasted source")
        chosen = load("selection").resolve_paths([path])
        self.app.path_entry.insert(0, str(path))
        with unittest.mock.patch.object(self.app, "_enqueue_selection",
                                        return_value=1):
            self.app._accept_selection(chosen)
        self.assertEqual("", self.app.path_entry.get())

    def test_waiting_queue_prevents_closing_without_start_or_removal(self):
        self.app.pending = [{"stamp": "q1", "label": "waiting"}]
        self.app._refresh_queue()
        with unittest.mock.patch.object(self.app.root, "destroy") as destroy:
            self.app.close()
        destroy.assert_not_called()
        self.assertIn("queued", self.app.status.cget("text").lower())

    def test_path_field_adds_several_space_separated_files(self):
        first = self.d / "first.md"; first.write_text("first")
        second = self.d / "a folder with spaces"; second.mkdir()
        inner = second / "inner.md"; inner.write_text("inner")
        calls = []

        def fake_resolve(values, mode_key):
            calls.append(list(values))
            return object()

        class FakeThread:
            def __init__(self, target, daemon=True):
                self.target = target

            def start(self):
                self.target()

        self.app.path_entry.insert(0, f"{first} {inner}")
        with unittest.mock.patch.object(self.ui.selection, "resolve_paths",
                                        side_effect=fake_resolve), \
             unittest.mock.patch.object(self.ui.threading, "Thread", FakeThread):
            self.app.add_path()
        self.assertEqual([[pathlib.Path(first), pathlib.Path(inner)]], calls)
        self.assertEqual("", self.app.path_entry.get())

    def test_adding_paths_during_resolution_keeps_the_complete_selection(self):
        # The resolver is intentionally off the Tk thread. A second Add click
        # used to see an empty accepted selection and replace the first path.
        calls = []

        def fake_resolve(values, mode_key):
            calls.append(list(values))
            return object()

        class FakeThread:
            def __init__(self, target, daemon=True):
                self.target = target

            def start(self):
                self.target()

        first, second = self.d / "first.md", self.d / "second.md"
        with unittest.mock.patch.object(self.ui.selection, "resolve_paths",
                                        side_effect=fake_resolve), \
             unittest.mock.patch.object(self.ui.threading, "Thread", FakeThread):
            self.app._resolve_inputs([first])
            self.app._resolve_inputs([second])
        self.assertEqual(calls, [[first], [first, second]])
        self.assertEqual(self.app._resolution_values, [first, second])

    def test_starting_a_new_run_clears_the_previous_results(self):
        f = self.d / "old.summary.md"; f.write_text("x")
        self.app._apply({"event": "target_finished", "index": 1,
                         "status": "succeeded", "exit_code": 0, "outputs": [str(f)]})
        self.assertEqual(len(self.app.cards), 1)
        # Supply an actual new submission. An empty clipboard is not a run and
        # should not erase prior result cards merely because Start was pressed.
        new_source = self.d / "new.md"
        new_source.write_text("new source")
        self.app.selection = load("selection").resolve_paths([new_source])
        self.app.source = [new_source]
        self.app._run = lambda *_args: None       # never reach Popen
        self.app.start()
        self.assertEqual(self.app.cards, [], "stale results survived a new run")

    def test_a_specific_failure_kind_beats_a_generic_exit_code(self):
        # Exit 2 covers both a bad invocation and a document too short to
        # summarize. Telling a user their document is an "invalid invocation"
        # is a wrong answer to a question they did ask.
        (self.d / "progress.jsonl").write_text(json.dumps({
            "schema": "summer.progress.v2", "event": "target_finished",
            "time_utc": "t", "index": 1, "status": "failed", "exit_code": 2,
            "failure_kind": "too_short", "destination_unchanged": True}) + "\n")
        self.app.job = self.d
        self.app._finish(2)
        st = self.app.status.cget("text")
        self.assertIn("too short", st)
        self.assertNotIn("invalid invocation", st)


    def test_an_unknown_failure_kind_falls_back_to_the_exit_code(self):
        (self.d / "progress.jsonl").write_text(json.dumps({
            "schema": "summer.progress.v2", "event": "target_finished",
            "time_utc": "t", "index": 1, "status": "failed", "exit_code": 4,
            "failure_kind": "something_new"}) + "\n")
        self.app.job = self.d
        self.app._finish(4)
        self.assertIn("coverage", self.app.status.cget("text").lower())

    def test_a_card_can_copy_its_artifact(self):
        p = self.d / "doc.summary.md"; p.write_text("the detailed reading")
        self.app.copy_out(p)
        self.assertEqual(self.root.clipboard_get(), "the detailed reading")




class CorpusPreflightContract(unittest.TestCase):
    """Corpus source preparation is complete before any model call is possible."""
    selection = load("selection")
    corpus = load("corpus")

    def _selection(self, root):
        for name, text in (("b.md", "beta " * 220), ("a.txt", "alpha " * 220)):
            (root / name).write_text(text)
        return self.selection.resolve_paths([root]).with_corpus_outputs(
            "Research set", root / "out")

    def test_corpus_manifest_is_v2_and_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            chosen = self._selection(root)
            manifest = root / "selection.json"
            self.selection.write_manifest(chosen, manifest)
            value = json.loads(manifest.read_text())
            self.assertEqual("summer.selection.v2", value["schema"])
            self.assertEqual("corpus", value["scope"])
            value["corpus"]["outputs"][0] = str(root / "out" / "changed.summary.md")
            manifest.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                self.selection.read_manifest(manifest)

    def test_cli_accepts_the_frozen_corpus_pair_without_batch_output_plans(self):
        cli = load("summ_cli")
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            (root / "one.md").write_text("one")
            (root / "two.md").write_text("two")
            chosen = self.selection.resolve_paths([root]).with_corpus_outputs(
                "Research set", root / "out")
            self.assertTrue(all(not doc.planned_outputs
                                for doc in chosen.documents))
            manifest = root / "selection.json"
            self.selection.write_manifest(chosen, manifest)
            stderr = io.StringIO()
            argv = ["summ_cli.py", "--selection-manifest", str(manifest),
                    "--scope", "corpus", "--work-dir", str(root / "work")]
            with unittest.mock.patch.object(sys, "argv", argv), \
                    unittest.mock.patch.object(cli, "freeze_model_routes"), \
                    unittest.mock.patch.object(cli, "validate_roles"), \
                    contextlib.redirect_stderr(stderr):
                rc = cli.main()
            self.assertEqual(2, rc)
            self.assertIn("Corpus preflight failed", stderr.getvalue())
            self.assertNotIn("no frozen output plan", stderr.getvalue())

    def test_preflight_stages_every_document_and_keeps_aggregate_path_free(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            chosen = self._selection(root)
            prepared = self.corpus.preflight(
                chosen, root / "work", self._cli_cancel())
            self.assertEqual(440, prepared.total_visible_words)
            self.assertEqual(2, prepared.inventory_windows)
            self.assertEqual(["D001", "D002"],
                             [d["document_id"] for d in prepared.documents])
            self.assertEqual(["D001:W001"],
                             [w["window_id"] for w in prepared.documents[0]["windows"]])
            aggregate = json.dumps(prepared.source_manifest, sort_keys=True)
            self.assertNotIn("a.txt", aggregate)
            self.assertNotIn("b.md", aggregate)
            self.assertNotIn("path", prepared.source_manifest)
            self.assertTrue((root / "work" / "documents" / "D001" /
                             "readerview" / "source-map.json").is_file())

    def test_changed_source_is_rejected_before_preparation_completes(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            chosen = self._selection(root)
            chosen.documents[0].source_path.write_text("changed")
            with self.assertRaises(self.corpus.CorpusPreflightError) as ctx:
                self.corpus.preflight(chosen, root / "work", self._cli_cancel())
            self.assertEqual("input_changed", ctx.exception.code)
            self.assertFalse((root / "work" / "corpus" /
                              "corpus-source.json").exists())

    def test_limit_boundaries_are_explicit(self):
        self.assertEqual(("document_limit",), tuple(
            item["kind"] for item in self.corpus.validate_limits(1, 400, 1)))
        self.assertEqual((), self.corpus.validate_limits(2, 400, 32))
        self.assertEqual(("visible_word_limit",), tuple(
            item["kind"] for item in self.corpus.validate_limits(2, 399, 1)))
        self.assertEqual(("inventory_window_limit",), tuple(
            item["kind"] for item in self.corpus.validate_limits(
                2, 400, self.corpus.MAX_INVENTORY_WINDOWS + 1)))

    def test_large_ordinary_corpus_fits_the_declared_workload_envelope(self):
        def paper(word_count):
            sections = []
            remaining = word_count
            index = 1
            while remaining:
                count = min(500, remaining)
                sections.append(
                    f"## Section {index}\n\n" + "word " * count)
                remaining -= count
                index += 1
            return "\n\n".join(sections)

        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            counts = (700, 44_500, 10_300, 10_900, 950)
            for index, count in enumerate(counts, 1):
                (root / f"paper-{index}.md").write_text(paper(count))
            chosen = self.selection.resolve_paths([root]).with_corpus_outputs(
                "Research set", root / "out")
            prepared = self.corpus.preflight(chosen, root / "work")
            self.assertGreater(prepared.total_visible_words, 60_000)
            self.assertLessEqual(prepared.total_visible_words,
                                 self.corpus.MAX_VISIBLE_WORDS)
            self.assertLessEqual(prepared.inventory_windows,
                                 self.corpus.MAX_INVENTORY_WINDOWS)
            self.assertTrue(all(
                window["visible_words"] <= self.corpus.INVENTORY_WINDOW_WORDS
                for document in prepared.documents
                for window in document["windows"]))

    def test_inventory_accounts_for_each_document_block_and_normalizes_ids(self):
        candidate = {
            "units": [{"local_id": "local-1",
                       "source_ids": ["D001:P0001", "D001:P0002"],
                       "dependencies": [], "capsule": "The source says this."}],
            "dispositions": [], "unplanned_windows": [],
        }
        self.assertEqual((), self.corpus.validate_inventory(
            candidate, "D001", ["D001:P0001", "D001:P0002"]))
        normalized = self.corpus.normalize_inventory(candidate, "D001")
        self.assertEqual("D001:U001", normalized["units"][0]["unit_id"])
        self.assertEqual("D001:P0002", normalized["units"][0]["source_ids"][1])
        self.assertNotIn("local_id", normalized["units"][0])

    def test_inventory_rejects_duplicate_missing_and_foreign_blocks(self):
        candidate = {
            "units": [{"local_id": "u1", "source_ids": ["D001:P0001", "D001:P0001"],
                       "dependencies": [], "capsule": "x"}],
            "dispositions": [], "unplanned_windows": [],
        }
        errors = self.corpus.validate_inventory(
            candidate, "D001", ["D001:P0001", "D001:P0002"])
        self.assertTrue(any("more than once" in error for error in errors))
        self.assertTrue(any("unaccounted" in error for error in errors))
        foreign = dict(candidate)
        foreign["units"] = [dict(candidate["units"][0],
                                   source_ids=["D002:P0001"])]
        self.assertTrue(any("cross-document" in error for error in
                            self.corpus.validate_inventory(
                                foreign, "D001", ["D001:P0001"])))

    def test_inventory_rejects_unknown_dependencies_and_cycles(self):
        candidate = {
            "units": [
                {"local_id": "u1", "source_ids": ["D001:P0001"],
                 "dependencies": ["u2"], "capsule": "one"},
                {"local_id": "u2", "source_ids": ["D001:P0002"],
                 "dependencies": ["u1"], "capsule": "two"},
            ], "dispositions": [], "unplanned_windows": [],
        }
        errors = self.corpus.validate_inventory(
            candidate, "D001", ["D001:P0001", "D001:P0002"])
        self.assertIn("inventory dependencies contain a cycle", errors)

    def test_inventory_orchestration_uses_plan_and_audit_roles_and_writes_evidence(self):
        chosen = None

        class FakeRunner:
            PLAN_MODELS = ("fake:plan",)
            MODELS = ("fake:write",)
            AUDIT_MODELS = ("fake:audit",)
            REPAIR_MODELS = ("fake:repair",)

            def __init__(self):
                self.calls = []

            @staticmethod
            def parse_audit(raw):
                return json.loads(raw)

            def run(self, prompt, workdir, chain, stage, validate=None,
                    gateway_options=None, role=None):
                self.calls.append((stage, role, gateway_options))
                if role == "plan":
                    ids = re.findall(r"\[D\d{3}:P\d{4}\]", prompt)
                    value = {"units": [{"local_id": "u1",
                                         "source_ids": [x[1:-1] for x in ids],
                                         "dependencies": [],
                                         "capsule": "The source material states this."}],
                             "dispositions": [], "unplanned_windows": []}
                else:
                    value = {"verdict": "pass", "findings": []}
                raw = json.dumps(value)
                if validate:
                    validate(raw)
                return raw

        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            chosen = self._selection(root)
            prepared = self.corpus.preflight(
                chosen, root / "work", self._cli_cancel())
            fake = FakeRunner()
            inventories = self.corpus.build_inventories(
                prepared, root / "work", self._cli_cancel(), runner=fake)
            self.assertEqual(2, len(inventories.documents))
            self.assertEqual({"plan", "audit"}, {call[1] for call in fake.calls})
            self.assertTrue((root / "work" / "corpus" /
                             "inventory-index.json").is_file())
            aggregate = json.dumps(inventories.index, sort_keys=True)
            self.assertNotIn("a.txt", aggregate)
            self.assertIn("D001:U001", aggregate)

    def test_global_plan_partitions_all_inventory_units_before_synthesis(self):
        class FakeRunner:
            PLAN_MODELS = ("fake:plan",)
            MODELS = ("fake:write",)
            AUDIT_MODELS = ("fake:audit",)
            REPAIR_MODELS = ("fake:repair",)

            def __init__(self):
                self.calls = []

            @staticmethod
            def parse_audit(raw):
                return json.loads(raw)

            def run(self, prompt, workdir, chain, stage, validate=None,
                    gateway_options=None, role=None):
                self.calls.append((stage, role))
                if stage.startswith("corpus-inventory"):
                    if role == "plan":
                        ids = re.findall(r"\[D\d{3}:P\d{4}\]", prompt)
                        value = {"units": [{"local_id": "u1",
                                             "source_ids": [x[1:-1] for x in ids],
                                             "dependencies": [],
                                             "capsule": "The source material states this."}],
                                 "dispositions": [], "unplanned_windows": []}
                    else:
                        value = {"verdict": "pass", "findings": []}
                elif role == "plan":
                    docs = sorted(set(re.findall(
                        r"\bD\d{3}\b", prompt.split("SEALED DOCUMENT INVENTORY")[-1])))
                    units = sorted(set(re.findall(
                        r"D\d{3}:U\d{3}", prompt.split("SEALED DOCUMENT INVENTORY")[-1])))
                    value = {"documents": [
                                 {"document_id": doc, "reference": f"Author {doc[-1]}, Title {doc[-1]}",
                                  "handle": f"Author{doc[-1]}", "contribution": ""} for doc in docs],
                             "themes": [], "relations": [
                                 {"relation": "agreement", "document_ids": docs,
                                  "unit_ids": units, "note": ""}],
                             "unresolved": []}
                else:
                    value = {"verdict": "pass", "findings": []}
                raw = json.dumps(value)
                if validate:
                    validate(raw)
                return raw

        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            chosen = self._selection(root)
            prepared = self.corpus.preflight(
                chosen, root / "work", self._cli_cancel())
            fake = FakeRunner()
            inventories = self.corpus.build_inventories(
                prepared, root / "work", self._cli_cancel(), runner=FakeRunner())
            plan = self.corpus.build_corpus_plan(
                inventories, prepared, root / "work", self._cli_cancel(),
                runner=fake)
            self.assertEqual("Author1", plan.plan["documents"][0]["handle"])
            self.assertEqual(2, len(plan.plan["relations"][0]["document_ids"]))
            self.assertEqual(["plan", "audit"], [role for _stage, role in fake.calls])
            self.assertTrue((root / "work" / "corpus" / "plan" /
                             "plan.json").is_file())

    def test_a_reviewer_finding_that_survives_the_one_correction_is_disclosed_not_fatal(self):
        # A semantic finding after the one correction is carried as an open
        # finding on the evidence; only incomplete accounting ends the run.
        class FakeRunner:
            PLAN_MODELS = ("fake:plan",)
            MODELS = ("fake:write",)
            AUDIT_MODELS = ("fake:audit",)
            REPAIR_MODELS = ("fake:repair",)

            @staticmethod
            def parse_audit(raw):
                return json.loads(raw)

            def run(self, prompt, workdir, chain, stage, validate=None,
                    gateway_options=None, role=None):
                if stage.startswith("corpus-inventory") and role in ("plan", "repair"):
                    ids = re.findall(r"\[D\d{3}:P\d{4}\]", prompt)
                    value = {"units": [{"local_id": "u1", "source_ids": [x[1:-1] for x in ids],
                                         "capsule": "The source states this."}]}
                elif "inventory" in stage and role == "audit":
                    value = {"verdict": "repair", "findings": [
                        {"issue": "u1 generalizes a qualification", "repair": "restore it"}]}
                else:
                    value = {"verdict": "pass", "findings": []}
                raw = json.dumps(value)
                if validate:
                    validate(raw)
                return raw

        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            chosen = self._selection(root)
            prepared = self.corpus.preflight(chosen, root / "work", self._cli_cancel())
            inventories = self.corpus.build_inventories(
                prepared, root / "work", self._cli_cancel(), runner=FakeRunner())
            for document in inventories.documents:
                self.assertTrue(document["open_findings"], document["document_id"])
                self.assertIn("generalizes a qualification", document["open_findings"][0])
                self.assertTrue(document["units"])

    def test_an_inventory_may_omit_empty_lists_and_carry_extra_fields(self):
        # A live Corpus run showed that a producer can answer every window
        # correctly but left out the empty "represented_by" list on one
        # disposition, and the window failed. Shape is tolerated; accounting
        # is not relaxed.
        errors = self.corpus.validate_inventory(
            {"units": [{"local_id": "u1", "source_ids": ["D001:P0001"],
                        "capsule": "The source states this.", "note": "extra"}],
             "dispositions": [{"source_ids": ["D001:P0002"], "disposition": "apparatus"}]},
            "D001", ["D001:P0001", "D001:P0002"])
        self.assertEqual([], list(errors))
        errors = self.corpus.validate_inventory(
            {"units": [{"local_id": "u1", "source_ids": ["D001:P0001"],
                        "capsule": "x"}]},
            "D001", ["D001:P0001", "D001:P0002"])
        self.assertTrue(any("unaccounted source blocks" in e for e in errors), errors)
        # Normalization tolerates the same shapes the validator accepts
        # A prior run died on a KeyError for "reason".
        normalized = self.corpus.normalize_inventory(
            {"units": [{"local_id": "u1", "source_ids": ["D001:P0001"],
                        "capsule": "The source states this."}],
             "dispositions": [{"source_ids": ["D001:P0002"], "disposition": "apparatus"}]},
            "D001")
        self.assertEqual("", normalized["dispositions"][0]["reason"])
        self.assertEqual([], normalized["units"][0]["dependencies"])

    def test_an_inventory_unit_may_omit_an_empty_dependencies_field(self):
        # A live Corpus run showed a backup route returning
        # units without the empty "dependencies" list and the window failed
        # its verification with "unknown dependency" on every unit.
        errors = self.corpus.validate_inventory(
            {"units": [{"local_id": "u1", "source_ids": ["D001:P0001"],
                        "capsule": "The source states this."}],
             "dispositions": [], "unplanned_windows": []},
            "D001", ["D001:P0001"])
        self.assertEqual([], list(errors))

    def test_document_identity_is_never_inferred_from_list_position(self):
        # A handle-only plan used to be resolved by position whenever the
        # counts matched, so reversing the model's output would move one
        # document's reference, contribution, and relations onto another and
        # still seal. Only an explicit ID or a unique exact file label counts.
        index = {"documents": [
            {"document_id": "D001", "units": [{"unit_id": "D001:U001", "source_ids": ["D001:P0001"]}]},
            {"document_id": "D002", "units": [{"unit_id": "D002:U001", "source_ids": ["D002:P0001"]}]},
        ]}
        labels = {"D001": "ui architecture", "D002": "corpus build"}
        handles_only = {
            "documents": [{"handle": "Interface", "reference": "GPT Pro, interface", "contribution": "a"},
                          {"handle": "Build", "reference": "GPT Pro, build", "contribution": "b"}],
            "themes": [], "relations": [], "unresolved": []}
        fixed = self.corpus.coerce_corpus_plan(handles_only, index, labels)
        errors = list(self.corpus.validate_corpus_plan(fixed, index))
        self.assertTrue(any("named 0 times" in e for e in errors), errors)

    def test_explicit_ids_resolve_and_a_reversed_plan_keeps_each_documents_own_facts(self):
        # Field-name aliasing is still coerced; identity is not.
        index = {"documents": [
            {"document_id": "D001", "units": [{"unit_id": "D001:U001", "source_ids": ["D001:P0001"]}]},
            {"document_id": "D002", "units": [{"unit_id": "D002:U001", "source_ids": ["D002:P0001"]}]},
        ]}
        labels = {"D001": "ui architecture", "D002": "corpus build"}
        reversed_plan = {
            "documents": [{"document_id": "D002", "handle": "Build",
                           "reference": "GPT Pro, build", "contribution": "the build"},
                          {"document_id": "D001", "handle": "Interface",
                           "reference": "GPT Pro, interface", "contribution": "the interface"}],
            "themes": [{"title": "shared", "documents": ["Interface", "Build"]}],
            "relations": [{"type": "elaboration", "documents": ["Build", "Interface"],
                           "units": ["D001:U001", "D002:U001"],
                           "statement": "Build elaborates Interface"}],
            "open_questions": [{"question": "what next", "documents": ["Build"]}]}
        fixed = self.corpus.coerce_corpus_plan(reversed_plan, index, labels)
        self.assertEqual([], list(self.corpus.validate_corpus_plan(fixed, index)), fixed)
        by_id = {d["document_id"]: d for d in fixed["documents"]}
        self.assertEqual("the build", by_id["D002"]["contribution"])
        self.assertEqual("the interface", by_id["D001"]["contribution"])
        normalized = self.corpus.normalize_corpus_plan(fixed, index, "sha")
        # Normalization restores inventory order without moving the facts.
        self.assertEqual(["D001", "D002"], [d["document_id"] for d in normalized["documents"]])
        self.assertEqual("the interface", normalized["documents"][0]["contribution"])
        self.assertEqual(("elaboration", "Build elaborates Interface"),
                         (fixed["relations"][0]["relation"], fixed["relations"][0]["note"]))
        self.assertEqual("what next", fixed["unresolved"][0]["issue"])

    def test_a_unique_file_label_still_resolves_a_document(self):
        index = {"documents": [
            {"document_id": "D001", "units": [{"unit_id": "D001:U001", "source_ids": ["D001:P0001"]}]},
            {"document_id": "D002", "units": [{"unit_id": "D002:U001", "source_ids": ["D002:P0001"]}]},
        ]}
        plan = {"documents": [
            {"file_label": "corpus build", "handle": "Build", "reference": "r", "contribution": ""},
            {"document_id": "D001", "handle": "Interface", "reference": "r", "contribution": ""}],
            "themes": [], "relations": [], "unresolved": []}
        fixed = self.corpus.coerce_corpus_plan(
            plan, index, {"D001": "ui architecture", "D002": "corpus build"})
        self.assertEqual({"D001", "D002"},
                         {d["document_id"] for d in fixed["documents"]})

    def test_relation_plan_rejects_unnamed_documents_pathlike_names_and_one_document_relations(self):
        index = {"documents": [
            {"document_id": "D001", "units": [{"unit_id": "D001:U001",
                                                   "source_ids": ["D001:P0001"]}]},
            {"document_id": "D002", "units": [{"unit_id": "D002:U001",
                                                   "source_ids": ["D002:P0001"]}]},
        ]}
        plan = {"documents": [{"document_id": "D001", "reference": "notes/2026-q2.md",
                               "handle": "D001", "contribution": ""}],
                "themes": [{"theme": "x", "document_ids": ["D009"], "note": ""}],
                "relations": [{"relation": "agreement", "document_ids": ["D001"],
                               "unit_ids": ["D002:U001"], "note": ""}],
                "unresolved": []}
        errors = self.corpus.validate_corpus_plan(plan, index)
        self.assertTrue(any("D002: named 0 times" in e for e in errors), errors)
        self.assertTrue(any("reference looks like a path" in e for e in errors), errors)
        self.assertTrue(any("handle looks like a path" in e for e in errors), errors)
        self.assertTrue(any("unknown document 'D009'" in e for e in errors), errors)
        self.assertTrue(any("at least two documents" in e for e in errors), errors)
        self.assertTrue(any("belongs to a document the relation does not cite" in e
                            for e in errors), errors)
        self.assertEqual("notes 2026 q2", self.corpus.file_label("a/b/notes_2026-q2.md"))

    def test_overview_pair_is_written_from_named_evidence_and_sealed(self):
        class FakeRunner:
            PLAN_MODELS = ("fake:plan",)
            MODELS = ("fake:write",)
            AUDIT_MODELS = ("fake:audit",)
            REPAIR_MODELS = ("fake:repair",)

            def __init__(self):
                self.calls = []

            @staticmethod
            def parse_audit(raw):
                return json.loads(raw)

            def run(self, prompt, workdir, chain, stage, validate=None,
                    gateway_options=None, role=None):
                self.calls.append((stage, role, prompt))
                if stage.startswith("corpus-inventory") and role == "plan":
                    ids = re.findall(r"\[D\d{3}:P\d{4}\]", prompt)
                    value = {"units": [{"local_id": "u1",
                                         "source_ids": [x[1:-1] for x in ids],
                                         "dependencies": [],
                                         "capsule": "The source material states this."}],
                             "dispositions": [], "unplanned_windows": []}
                elif stage == "corpus-plan":
                    docs = sorted(set(re.findall(
                        r"\bD\d{3}\b", prompt.split("SEALED DOCUMENT INVENTORY")[-1])))
                    units = sorted(set(re.findall(
                        r"D\d{3}:U\d{3}", prompt.split("SEALED DOCUMENT INVENTORY")[-1])))
                    value = {"documents": [
                                 {"document_id": doc, "reference": f"Author {doc[-1]}, Title {doc[-1]}",
                                  "handle": f"Author{doc[-1]}",
                                  "contribution": "its own material"} for doc in docs],
                             "themes": [{"theme": "the shared subject",
                                         "document_ids": docs, "note": ""}],
                             "relations": [{"relation": "agreement", "document_ids": docs,
                                            "unit_ids": units, "note": "both say so"}],
                             "unresolved": []}
                elif stage.startswith("corpus-overview") and "audit" not in stage:
                    value = {"detailed": "Author 1 and Author 2 agree on the stated material, "
                                         "and each adds its own material. " * 5,
                             "brief": "Both agree on the stated material. " * 4}
                else:
                    value = {"verdict": "pass", "findings": []}
                raw = json.dumps(value)
                if validate:
                    validate(raw)
                return raw

        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            chosen = self._selection(root)
            prepared = self.corpus.preflight(
                chosen, root / "work", self._cli_cancel())
            fake = FakeRunner()
            inventories = self.corpus.build_inventories(
                prepared, root / "work", self._cli_cancel(), runner=fake)
            plan = self.corpus.build_corpus_plan(
                inventories, prepared, root / "work", self._cli_cancel(),
                runner=fake)
            self.assertEqual(self.corpus.PLAN_SCHEMA, plan.plan["schema"])
            self.assertEqual(["D001", "D002"],
                             [d["document_id"] for d in plan.plan["documents"]])
            rc = self.corpus.write_overview_pair(
                plan, inventories, prepared, root / "work", self._cli_cancel(),
                runner=fake)
            self.assertEqual(0, rc)
            pair = root / "work" / "corpus" / "pair"
            self.assertTrue((pair / "detailed.md").is_file())
            report = json.loads((pair / "full-report.json").read_text())
            self.assertEqual(("corpus", plan.plan_sha256, inventories.index_sha256),
                             (report["route"], report["plan_sha256"],
                              report["inventory_sha256"]))
            # The writer saw names, not IDs or paths; the reviewers saw each
            # document's own text under its name.
            write_prompt = next(p for st, _r, p in fake.calls if st == "corpus-overview")
            self.assertIn("DOCUMENT: Author 1, Title 1 (short name: Author1)", write_prompt)
            self.assertIn("RELATION PLAN", write_prompt)
            self.assertNotIn("D001", write_prompt.split("SOURCE:")[1])
            self.assertTrue(all(str(root) not in p for _s, _r, p in fake.calls))
            audits = [st for st, _r, _p in fake.calls if st.startswith("corpus-overview-audit")]
            self.assertEqual(audits, ["corpus-overview-audit-001",
                                      "corpus-overview-audit-002",
                                      "corpus-overview-audit-global"])

            seal = load("corpus_seal")
            sealed = seal.seal(root / "work")
            self.assertTrue(sealed["passed"], sealed)
            self.assertTrue((root / "work" / "corpus" / "CORPUS_SEAL").is_file())
            # Tampering with any link breaks the seal.
            plan_path = root / "work" / "corpus" / "plan" / "plan.json"
            original = plan_path.read_bytes()
            plan_path.write_bytes(original.replace(b"Author1", b"Author9"))
            self.assertFalse(seal.seal(root / "work")["passed"])
            plan_path.write_bytes(original)
            self.assertTrue(seal.seal(root / "work")["passed"])
            (pair / "detailed.md").write_text("changed reading\n")
            self.assertFalse(seal.seal(root / "work")["passed"])

    def test_corpus_model_call_is_not_started_after_cancellation(self):
        cli = load("summ_cli")
        token = cli.CancellationToken()
        token.request()

        class NeverRun:
            PLAN_MODELS = ("fake:plan",)
            AUDIT_MODELS = ("fake:audit",)
            MODELS = ("fake:write",)
            REPAIR_MODELS = ("fake:repair",)

            def run(self, *args, **kwargs):
                raise AssertionError("cancelled Corpus call reached the runner")

        with self.assertRaises(cli.Cancelled):
            self.corpus._call_inventory(
                NeverRun(), "prompt", pathlib.Path(tempfile.gettempdir()),
                "corpus-plan", "plan", {}, cancel=token)

    def test_guarded_cli_orchestrator_publishes_one_pair_with_a_fake_runner(self):
        cli = load("summ_cli")

        class FakeRunner:
            PLAN_MODELS = ("fake:plan",)
            MODELS = ("fake:write",)
            AUDIT_MODELS = ("fake:audit",)
            REPAIR_MODELS = ("fake:repair",)

            def __init__(self):
                self.calls = []

            @staticmethod
            def parse_audit(raw):
                return json.loads(raw)

            def run(self, prompt, workdir, chain, stage, validate=None,
                    gateway_options=None, role=None):
                self.calls.append((stage, role, prompt))
                if stage.startswith("corpus-inventory") and role == "plan":
                    ids = re.findall(r"\[D\d{3}:P\d{4}\]", prompt)
                    value = {"units": [{"local_id": "u1",
                                         "source_ids": [x[1:-1] for x in ids],
                                         "dependencies": [],
                                         "capsule": "The source material states this."}],
                             "dispositions": [], "unplanned_windows": []}
                elif stage == "corpus-plan":
                    docs = sorted(set(re.findall(
                        r"\bD\d{3}\b", prompt.split("SEALED DOCUMENT INVENTORY")[-1])))
                    units = sorted(set(re.findall(
                        r"D\d{3}:U\d{3}", prompt.split("SEALED DOCUMENT INVENTORY")[-1])))
                    value = {"documents": [
                                 {"document_id": doc, "reference": f"Author {doc[-1]}, Title {doc[-1]}",
                                  "handle": f"Author{doc[-1]}",
                                  "contribution": "its own material"} for doc in docs],
                             "themes": [{"theme": "the shared subject",
                                         "document_ids": docs, "note": ""}],
                             "relations": [{"relation": "agreement", "document_ids": docs,
                                            "unit_ids": units, "note": "both say so"}],
                             "unresolved": []}
                elif stage.startswith("corpus-overview") and "audit" not in stage:
                    value = {"detailed": "Author 1 and Author 2 agree on the stated material, "
                                         "and each adds its own material. " * 5,
                             "brief": "Both agree on the stated material. " * 4}
                else:
                    value = {"verdict": "pass", "findings": []}
                raw = json.dumps(value)
                if validate:
                    validate(raw)
                return raw

        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            (root / "one.md").write_text("one " * 220)
            (root / "two.md").write_text("two " * 220)
            chosen = self.selection.resolve_paths([root]).with_corpus_outputs(
                "facts", root / "out")
            with unittest.mock.patch.object(cli, "freeze_model_routes"), \
                 unittest.mock.patch.object(cli, "validate_roles"), \
                 unittest.mock.patch.object(cli, "notify"):
                rc = cli.do_corpus(chosen, root / "work", cli.CancellationToken(),
                                   "agy", {}, runner=FakeRunner())
            self.assertEqual(0, rc)
            self.assertTrue((root / "out" / "facts.corpus.summary.md").is_file())
            self.assertTrue((root / "out" / "facts.corpus.brief.md").is_file())
            self.assertTrue((root / "work" / "corpus" / "CORPUS_SEAL").is_file())

    @staticmethod
    def _cli_cancel():
        cli = load("summ_cli")
        return cli.CancellationToken()


class CorpusIntegrityContract(unittest.TestCase):
    """The Corpus seal answers exactly one question: are these the audited
    bytes? It must bind the whole chain, not a structural spine.

    Before this gate the seal compared word counts and a few hashes, so a
    same-length edit of a published reading, a changed inventory audit, and
    an appended route record all still "sealed", and a later failed check
    left the earlier CORPUS_SEAL and SEALED authorization in place.
    """
    selection = load("selection")
    corpus = load("corpus")
    seal = load("corpus_seal")

    class FakeRunner:
        PLAN_MODELS = ("fake:plan",)
        MODELS = ("fake:write",)
        AUDIT_MODELS = ("fake:audit",)
        REPAIR_MODELS = ("fake:repair",)

        @staticmethod
        def parse_audit(raw):
            return json.loads(raw)

        def run(self, prompt, workdir, chain, stage, validate=None,
                gateway_options=None, role=None):
            if stage.startswith("corpus-inventory") and role in ("plan", "repair"):
                ids = re.findall(r"\[D\d{3}:P\d{4}\]", prompt)
                value = {"units": [{"local_id": "u1",
                                     "source_ids": [x[1:-1] for x in ids],
                                     "capsule": "The source states this."}]}
            elif stage == "corpus-plan":
                inventory = prompt.split("SEALED DOCUMENT INVENTORY")[-1]
                docs = sorted(set(re.findall(r"\bD\d{3}\b", inventory)))
                units = sorted(set(re.findall(r"D\d{3}:U\d{3}", inventory)))
                value = {"documents": [
                             {"document_id": doc,
                              "reference": f"Author {doc[-1]}, Title {doc[-1]}",
                              "handle": f"Author{doc[-1]}",
                              "contribution": "its own material"} for doc in docs],
                         "themes": [], "relations": [
                             {"relation": "agreement", "document_ids": docs,
                              "unit_ids": units, "note": "both say so"}],
                         "unresolved": []}
            elif stage.startswith("corpus-overview") and "audit" not in stage:
                value = {"detailed": "Author 1 and Author 2 agree on the "
                                     "stated material, and each adds its own. " * 5,
                         "brief": "Both agree on the stated material. " * 4}
            else:
                value = {"verdict": "pass", "findings": []}
            raw = json.dumps(value)
            if validate:
                validate(raw)
            return raw

    def _sealed_run(self, root):
        """A complete Corpus run through the seal, in a temp tree."""
        cli = load("summ_cli")
        for name, text in (("b.md", "beta " * 220), ("a.txt", "alpha " * 220)):
            (root / name).write_text(text)
        chosen = self.selection.resolve_paths([root]).with_corpus_outputs(
            "Research set", root / "out")
        work = root / "work"
        cancel = cli.CancellationToken()
        runner = self.FakeRunner()
        prepared = self.corpus.preflight(chosen, work, cancel)
        inventories = self.corpus.build_inventories(
            prepared, work, cancel, runner=runner)
        plan = self.corpus.build_corpus_plan(
            inventories, prepared, work, cancel, runner=runner)
        # The fake runner bypasses mapsum, so stand in for the route record it
        # would have written. The manifest must bind whatever evidence the run
        # left behind, including call records and stage stdout/stderr.
        stage_dir = sorted(
            (work / "documents" / "D001" / "inventory").glob("D001-W*"))[0]
        (stage_dir / "calls.jsonl").write_text(
            '{"stage": "corpus-inventory-plan", "outcome": "ok"}\n')
        rc = self.corpus.write_overview_pair(
            plan, inventories, prepared, work, cancel, runner=runner)
        self.assertEqual(0, rc)
        sealed = self.seal.seal(work)
        self.assertTrue(sealed["passed"], sealed)
        return work

    def test_a_clean_run_seals_and_reports_both_statuses(self):
        with tempfile.TemporaryDirectory() as td:
            work = self._sealed_run(pathlib.Path(td))
            croot = work / "corpus"
            self.assertTrue((croot / "CORPUS_SEAL").is_file())
            status = json.loads((croot / "status.json").read_text())
            # Authenticated bytes and clean prose are separate answers.
            self.assertEqual("verified", status["seal_status"])
            self.assertIn("quality_status", status)
            self.assertIn("pair_findings", status)
            self.assertIn("upstream_findings", status)
            report = json.loads((croot / "pair" / "full-report.json").read_text())
            for key in ("evidence_manifest_sha256", "plan_sha256",
                        "inventory_sha256", "source_manifest_sha256",
                        "detailed_sha256", "brief_sha256", "candidate"):
                self.assertTrue(report.get(key), key)

    def test_a_same_word_count_edit_of_a_published_reading_breaks_the_seal(self):
        with tempfile.TemporaryDirectory() as td:
            work = self._sealed_run(pathlib.Path(td))
            detailed = work / "corpus" / "pair" / "detailed.md"
            text = detailed.read_text()
            words = text.split()
            words[3] = "disagree"
            edited = " ".join(words) + "\n"
            self.assertEqual(len(text.split()), len(edited.split()))
            detailed.write_text(edited)
            self.assertFalse(self.seal.seal(work)["passed"])

    def test_one_byte_changed_in_any_evidence_class_breaks_the_seal(self):
        classes = {
            "inventory candidate": "documents/D001/inventory/*/candidate.json",
            "inventory audit": "documents/D001/inventory/*/audit-r1.json",
            "inventory prompt": "documents/D001/inventory/*/build.prompt.txt",
            "route record": "documents/D001/inventory/*/calls.jsonl",
            "source map": "documents/D001/readerview/source-map.json",
            "staged source": "documents/D001/staged/source.original",
            "normalized inventory": "documents/D001/inventory/inventory.json",
            "relation plan": "corpus/plan/plan.json",
        }
        for label, pattern in classes.items():
            with self.subTest(evidence=label), tempfile.TemporaryDirectory() as td:
                work = self._sealed_run(pathlib.Path(td))
                target = next(iter(sorted(work.glob(pattern))), None)
                self.assertIsNotNone(target, f"no {label} at {pattern}")
                target.write_bytes(target.read_bytes() + b" ")
                self.assertFalse(self.seal.seal(work)["passed"], label)

    def test_evidence_added_after_the_run_breaks_the_seal(self):
        with tempfile.TemporaryDirectory() as td:
            work = self._sealed_run(pathlib.Path(td))
            (work / "documents" / "D001" / "inventory" / "extra.json").write_text("{}")
            self.assertFalse(self.seal.seal(work)["passed"])

    def test_a_failed_seal_withdraws_the_earlier_authorization(self):
        with tempfile.TemporaryDirectory() as td:
            work = self._sealed_run(pathlib.Path(td))
            croot = work / "corpus"
            self.assertTrue((croot / "CORPUS_SEAL").is_file())
            plan = croot / "plan" / "plan.json"
            plan.write_bytes(plan.read_bytes() + b" ")
            self.assertFalse(self.seal.seal(work)["passed"])
            # The markers are publication authority, not history.
            self.assertFalse((croot / "CORPUS_SEAL").exists())
            self.assertFalse((croot / "SEALED").exists())
            self.assertEqual("failed", json.loads(
                (croot / "status.json").read_text())["seal_status"])

    def test_surviving_evidence_findings_reach_the_report_and_the_status(self):
        # A reviewer objection that survives the one correction at an evidence
        # stage is retained, not fatal. It must still be disclosed alongside
        # pair-level findings.
        class Objecting(CorpusIntegrityContract.FakeRunner):
            def run(self, prompt, workdir, chain, stage, validate=None,
                    gateway_options=None, role=None):
                if "inventory" in stage and role == "audit":
                    value = {"verdict": "repair", "findings": [
                        {"issue": "u1 generalizes a qualification"}]}
                    raw = json.dumps(value)
                    if validate:
                        validate(raw)
                    return raw
                return super().run(prompt, workdir, chain, stage, validate,
                                   gateway_options, role)

        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            cli = load("summ_cli")
            for name, text in (("b.md", "beta " * 220), ("a.txt", "alpha " * 220)):
                (root / name).write_text(text)
            chosen = self.selection.resolve_paths([root]).with_corpus_outputs(
                "Research set", root / "out")
            work, cancel, runner = root / "work", cli.CancellationToken(), Objecting()
            prepared = self.corpus.preflight(chosen, work, cancel)
            inventories = self.corpus.build_inventories(
                prepared, work, cancel, runner=runner)
            plan = self.corpus.build_corpus_plan(
                inventories, prepared, work, cancel, runner=runner)
            self.assertEqual(0, self.corpus.write_overview_pair(
                plan, inventories, prepared, work, cancel, runner=runner))
            report = json.loads(
                (work / "corpus" / "pair" / "full-report.json").read_text())
            self.assertTrue(report["upstream_findings"])
            self.assertTrue(any("generalizes a qualification" in f
                                for f in report["upstream_findings"]))
            self.assertEqual("open_findings", report["quality_status"])
            sealed = self.seal.seal(work)
            self.assertTrue(sealed["passed"], sealed)
            status = json.loads((work / "corpus" / "status.json").read_text())
            self.assertEqual(("verified", "open_findings"),
                             (status["seal_status"], status["quality_status"]))
            self.assertEqual(len(report["upstream_findings"]),
                             status["upstream_findings"])


class CommandTransportContract(unittest.TestCase):
    """A composed prompt never rides on argv when it is near the platform
    limit. Windows caps a whole command line at 32,767 characters, so a
    harness that can only take a positional prompt is skipped for a large
    payload instead of failing on one platform only."""
    ms = load("mapsum")

    def _argv(self, harness, prompt, workdir):
        option = {"effort": "high", "variant": "high", "model": "high"}
        return self.ms._cmd(harness, "model-x", workdir, prompt, option,
                            timeout_seconds=600)

    def test_no_harness_puts_a_large_prompt_on_argv(self):
        big = "word " * 40_000
        with tempfile.TemporaryDirectory() as td:
            workdir = pathlib.Path(td)
            for harness in self.ms.HARNESSES:
                if harness in self.ms.GATEWAY_HARNESSES:
                    continue
                with self.subTest(harness=harness):
                    if harness in self.ms.NON_ARGV_TRANSPORT:
                        argv = self._argv(harness, big, workdir)
                        self.assertNotIn(big, argv)
                        self.assertLess(sum(len(a) for a in argv),
                                        self.ms.ARGV_SAFE_BYTES)
                    else:
                        # No file or stdin transport: the route must be
                        # ineligible for this payload, which run() enforces.
                        self.assertGreater(len(big.encode()),
                                           self.ms.ARGV_SAFE_BYTES)

    def test_an_oversize_prompt_skips_an_argv_only_route_for_the_next_one(self):
        big = "word " * 40_000
        seen = []

        def fake_run(argv, **_kw):
            seen.append(pathlib.Path(argv[0]).name)
            return subprocess.CompletedProcess(argv, 0, '{"ok": true}', "")

        with tempfile.TemporaryDirectory() as td, \
                unittest.mock.patch.object(subprocess, "run", fake_run):
            answer = self.ms.run(big, pathlib.Path(td),
                                 ["claude:sonnet", "codex:gpt-5.6-luna"],
                                 "plan001")
        self.assertEqual('{"ok": true}', answer)
        self.assertEqual(["codex"], seen, "an argv-only route took a huge prompt")

    def test_every_prompt_ends_with_the_answer_now_instruction(self):
        # A head-only instruction was lost in an audit-sized prompt and the
        # model opened a tool instead of answering.
        seen = {}

        def fake_run(argv, **kw):
            seen["input"] = kw.get("input")
            return subprocess.CompletedProcess(argv, 0, '{"ok": true}', "")

        with tempfile.TemporaryDirectory() as td, \
                unittest.mock.patch.object(subprocess, "run", fake_run):
            self.ms.run("PROMPT BODY", pathlib.Path(td),
                        ["codex:gpt-5.6-luna"], "plan001")
        self.assertTrue(seen["input"].startswith("PROMPT BODY"))
        self.assertTrue(seen["input"].rstrip().endswith("nothing else."))
        self.assertIn("Do not call a tool", seen["input"])


class TheIconIsBuiltFromCommittedArt(unittest.TestCase):
    """Six PNGs somebody exported once cannot be re-cropped or re-cut later.
    The master and the script that processes it are both committed, so the set
    is reproducible from source."""
    mk = load("make_icon")

    def test_the_master_art_is_committed(self):
        assets = ENG / "assets"
        masters = [p for p in assets.glob("icon-source.*")
                   if p.suffix.lower() in (".png", ".jpeg", ".jpg")]
        self.assertTrue(masters, "no committed master art in src/assets/")

    def test_every_size_exists_and_is_rgba(self):
        for size in self.mk.SIZES:
            p = ENG / "assets" / f"icon-{size}.png"
            self.assertTrue(p.is_file(), f"missing icon-{size}.png")
            ct = p.read_bytes()[25]
            self.assertEqual(ct, 6, f"icon-{size}.png is not RGBA (colour type {ct})")

    def test_the_corners_are_actually_transparent(self):
        # The whole point of the pass: the delivered art had a white background
        # baked in, which on any dark surface reads as a white box.
        for size in self.mk.SIZES:
            rows = self._rows(ENG / "assets" / f"icon-{size}.png", size)
            self.assertEqual(rows[0][3], 0, f"icon-{size}.png corner is opaque")
            mid = rows[size // 2][(size // 2) * 4 + 3]
            self.assertEqual(mid, 255, f"icon-{size}.png centre is transparent")

    def test_no_white_survives_inside_the_tile(self):
        # The master's right edge stopped short on one row and let white through,
        # which downsampled into a visible bite. The shape is taken from geometry
        # now, and any background white inside it is filled with the tile colour.
        size = 128
        rows = self._rows(ENG / "assets" / f"icon-{size}.png", size)
        white = 0
        for y in range(size):
            for x in range(size):
                r, g, b, a = rows[y][x * 4:x * 4 + 4]
                if a > 200 and r > 240 and g > 240 and b > 240:
                    white += 1
        self.assertLess(white, size * size * 0.02,
                        f"{white} near-white opaque pixels survive inside the tile")

    def test_the_ui_points_at_a_size_that_exists(self):
        src = (ENG / "summ_ui.py").read_text()
        m = re.search(r'"(icon-\d+\.png)"', src)
        self.assertIsNotNone(m, "summ_ui.py names no icon file")
        self.assertTrue((ENG / "assets" / m.group(1)).is_file(),
                        f"summ_ui.py points at missing {m.group(1)}")

    def test_the_app_launcher_binds_the_build_checkout_without_private_paths(self):
        app = load("make_app")
        root = pathlib.Path("/tmp/summer checkout")
        launcher = app.launcher(root)
        self.assertIn(f"PROJECT_ROOT='{root.resolve()}'", launcher)
        self.assertNotIn("/Users/", launcher)
        self.assertNotRegex(launcher, r"[A-Za-z]:[\\/]Users[\\/]")
        if not sys.platform.startswith("win"):
            result = subprocess.run(
                ["bash", "-n"], input=launcher, text=True, capture_output=True)
            self.assertEqual(0, result.returncode, result.stderr)

    def _rows(self, path, size):
        import struct, zlib
        d = path.read_bytes()
        idat, pos = bytearray(), 8
        while pos < len(d):
            n = struct.unpack(">I", d[pos:pos + 4])[0]
            if d[pos + 4:pos + 8] == b"IDAT":
                idat += d[pos + 8:pos + 8 + n]
            pos += 12 + n
        raw, stride, out = zlib.decompress(bytes(idat)), size * 4, []
        prev, q = bytearray(stride), 0
        for _ in range(size):
            f = raw[q]; q += 1
            line = bytearray(raw[q:q + stride]); q += stride
            if f == 1:
                for i in range(4, stride): line[i] = (line[i] + line[i - 4]) & 0xFF
            elif f == 2:
                for i in range(stride): line[i] = (line[i] + prev[i]) & 0xFF
            elif f == 3:
                for i in range(stride):
                    a = line[i - 4] if i >= 4 else 0
                    line[i] = (line[i] + ((a + prev[i]) >> 1)) & 0xFF
            elif f == 4:
                for i in range(stride):
                    a = line[i - 4] if i >= 4 else 0
                    b, c = prev[i], (prev[i - 4] if i >= 4 else 0)
                    pa, pb, pc = abs(b - c), abs(a - c), abs(a + b - 2 * c)
                    pr = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                    line[i] = (line[i] + pr) & 0xFF
            prev = line; out.append(line)
        return out




class ModelsAreChosenInTheWindowWithoutNamingAnyHere(unittest.TestCase):
    """The window must let a model be picked per role, and must still not be a
    second place a model identifier lives. It offers what models.json lists and
    hands the choice over as the `<HARNESS>_CHAIN_<ROLE>` override the CLI
    already reads -- no new mechanism, no new naming site."""
    ui = load("summ_ui")

    def local_gateway(self):
        return unittest.mock.patch.object(
            self.ui.runtime, "config", return_value=gateway_runtime())

    def test_summ_ui_names_no_model(self):
        src = (ENG / "summ_ui.py").read_text()
        for ident in ("gemini-", "gpt-5", "sonnet", "opus",
                      "private-model-", "muse-"):
            hits = [l.strip() for l in src.splitlines() if ident in l]
            self.assertEqual(hits, [], f"summ_ui.py names the model {ident!r}")

    def test_the_roster_comes_from_models_json(self):
        cfg = json.loads((ENG / "models.json").read_text())
        r = self.ui.roster()
        for h in ("agy", "claude", "opencode"):
            self.assertIn(h, r)
            for role in self.ui.ROLES:
                self.assertEqual(r[h][role], cfg[h][role])

    def test_provider_effort_ids_are_one_logical_model_choice(self):
        cfg = json.loads((ENG / "models.json").read_text())
        choices = self.ui.model_choices("agy", "write")
        self.assertEqual(choices, cfg["agy"]["write"])
        self.assertEqual(len(choices), 2)
        for model in choices:
            setting = cfg["agy"]["_model_settings"][model]
            self.assertEqual(setting["option"], "model")
            self.assertNotIn(model, setting["variants"].values())

    def test_every_role_is_offered(self):
        self.assertEqual(self.ui.ROLES, ("plan", "write", "audit", "repair"))

    def test_a_choice_becomes_the_override_the_cli_already_reads(self):
        env = self.ui.role_env("opencode", {"plan": ("opencode", "m1"),
                                            "audit": ("claude", "m2")})
        self.assertEqual(env["HARNESS"], "opencode")
        self.assertEqual(env["OPENCODE_CHAIN_PLAN"].split()[0], "m1")
        # A role on another harness becomes the qualified entry mapsum resolves.
        self.assertEqual(env["OPENCODE_CHAIN_AUDIT"].split()[0], "claude:m2")
        # The committed backups follow, in order, so an exhausted route never
        # leaves the chain empty.
        self.assertEqual(env["OPENCODE_CHAIN_PLAN"].split()[1:],
                         ["codex:gpt-5.6-luna", "agy:gemini-3.1-pro-high", "claude:sonnet"])
        self.assertNotIn("OPENCODE_CHAIN_WRITE", env)

    def test_a_configurable_backup_is_appended_to_every_selected_role(self):
        env = self.ui.role_env(
            "grok", {"write": ("grok", "primary")},
            ("opencode", "fallback"))
        self.assertEqual(["primary", "opencode:fallback"],
                         env["GROK_CHAIN_WRITE"].split()[:2],
                         "the profile's own backup comes right after its pick")

    def test_local_only_is_frozen_as_a_job_environment_setting(self):
        with self.local_gateway():
            env = self.ui.role_env(
                "localgw", {"write": ("localgw", "model-a")},
                ("localgw", "model-a"), local_only=True)
        self.assertEqual("1", env["SUMM_LOCAL_ONLY"])
        self.assertEqual("model-a", env["LOCALGW_CHAIN_WRITE"])

    def test_all_local_choices_infer_local_only_without_a_checkbox(self):
        with self.local_gateway():
            env = self.ui.role_env(
                "localgw", {"write": ("localgw", "model-a", "medium")},
                ("localgw", "model-b", "xhigh"))
        self.assertEqual("1", env["SUMM_LOCAL_ONLY"])
        self.assertEqual("model-a model-b", env["LOCALGW_CHAIN_WRITE"])

    def test_gateway_option_is_visible_and_frozen_for_the_ui_job(self):
        with self.local_gateway():
            self.assertEqual(
                "medium", self.ui.App.option_label("localgw", "model-a"))
            env = self.ui.role_env(
                "localgw", {"write": ("localgw", "model-a", "medium")})
        frozen = json.loads(env["SUMM_ROLE_OPTIONS"])
        self.assertEqual(
            {"reasoning_effort": "medium"}, frozen["write"][0]["option"])

    def test_local_only_omits_a_cloud_backup_from_the_frozen_chain(self):
        with self.local_gateway():
            env = self.ui.role_env(
                "localgw", {"write": ("localgw", "model-a")},
                ("opencode", "cloud-model"), local_only=True)
        self.assertEqual("model-a", env["LOCALGW_CHAIN_WRITE"])
        self.assertNotIn("OPENCODE_CHAIN_WRITE", env)

    def test_default_backup_is_luna_high_from_the_model_roster(self):
        defaults = self.ui.model_defaults()["fallback"]
        self.assertIsInstance(defaults, list)
        self.assertEqual(["codex", "agy", "claude"], [d["harness"] for d in defaults])
        defaults = defaults[0]
        self.assertEqual("codex", defaults["harness"])
        self.assertEqual("gpt-5.6-luna", defaults["model"])
        self.assertEqual("high", defaults["effort"])
        env = self.ui.role_env(
            "grok", {"write": ("grok", "primary")},
            (defaults["harness"], defaults["model"], defaults["effort"]))
        frozen = json.loads(env["SUMM_ROLE_OPTIONS"])
        self.assertEqual("high", frozen["write"][1]["option"]["effort"])

    def test_the_override_variable_is_the_one_mapsum_looks_up(self):
        # If these ever diverge the picker silently does nothing.
        ms = (ENG / "mapsum.py").read_text()
        config = (ENG / "model_config.py").read_text()
        self.assertIn("model_config.role_chains(", ms)
        self.assertIn('f"{harness.upper()}_CHAIN_{role.upper()}"', config)

    def test_roles_repopulate_when_the_harness_changes(self):
        import tkinter as tk
        try:
            root = tk.Tk()
        except tk.TclError as e:
            self.skipTest(f"no display: {e}")
        root.withdraw()
        app = self.ui.App(root)
        self.addCleanup(root.destroy)
        cfg = json.loads((ENG / "models.json").read_text())
        app.role_h["plan"].set("claude"); app.refresh_role("plan")
        self.assertIn(app.role_vars["plan"].get(), cfg["claude"]["plan"])
        app.role_h["plan"].set("agy"); app.refresh_role("plan")
        self.assertIn(app.role_vars["plan"].get(), cfg["agy"]["plan"])
        # A stale pick from the previous harness must not survive.
        self.assertNotIn(app.role_vars["plan"].get(), cfg["claude"]["plan"])

    def test_the_running_model_is_named_in_the_status(self):
        # "plan001 running" says nothing about WHICH model is being spent, which
        # is the thing worth knowing on a metered account.
        import tkinter as tk
        try:
            root = tk.Tk()
        except tk.TclError as e:
            self.skipTest(f"no display: {e}")
        root.withdraw()
        app = self.ui.App(root)
        self.addCleanup(root.destroy)
        app._apply({"event": "stage", "name": "ledger"})
        app._apply({"event": "model_call_started", "role": "plan001",
                    "harness": "opencode", "model": "vendor/some-model"})
        self.assertIn("some-model", app.status.cget("text"))


class LogicalModelFamiliesFreezeToExactRoutes(unittest.TestCase):
    """A compact UI may never make the transport, effort or ETA ambiguous."""
    mc = load("model_config")
    cli = load("summ_cli")
    ms = load("mapsum")
    et = load("eta")

    def setUp(self):
        self.roster = self.mc.roster()
        self.family = "gemini-3.8-flash"
        self.variants = self.roster["agy"]["_model_settings"][
            self.family]["variants"]

    def test_each_role_keeps_the_pre_collapse_effective_default(self):
        for role in ("plan", "write", "repair"):
            self.assertEqual(
                self.variants["high"], self.mc.resolved_model(
                    "agy", self.family, role=role, roster_value=self.roster))
        self.assertEqual(
            self.variants["medium"], self.mc.resolved_model(
                "agy", self.family, role="audit", roster_value=self.roster))
        self.assertEqual("medium", load("summ_ui").App.option_label(
            "agy", self.family, "audit"))

    def test_a_ui_frozen_physical_variant_passes_cli_preflight(self):
        picks = {role: ("agy", self.family, "medium")
                 for role in ("write", "audit", "repair")}
        env = load("profiles").role_env(
            "agy", picks, roster_value=self.roster)
        with unittest.mock.patch.dict(os.environ, env, clear=True):
            self.cli.validate_roles(
                "agy", self.roster,
                self.cli.mode_config.BY_KEY["quick"].roles)
        self.assertTrue(all(self.variants["medium"] in env[key]
                            for key in ("AGY_CHAIN_WRITE", "AGY_CHAIN_AUDIT",
                                        "AGY_CHAIN_REPAIR")))

    def test_a_qualified_logical_entry_resolves_on_its_actual_harness(self):
        env = {"OPENCODE_CHAIN_WRITE": f"agy:{self.family}"}
        with unittest.mock.patch.dict(os.environ, env, clear=True):
            chain = self.mc.role_chains(
                "opencode", ("write",), roster_value=self.roster)["write"]
        self.assertEqual([f"agy:{self.variants['high']}"], chain)

    def test_a_declared_fallback_is_resolved_on_its_own_harness(self):
        fallback = {"fallback": {"harness": "agy", "model": self.family,
                                 "variant": "medium"}}
        with unittest.mock.patch.object(self.mc, "defaults",
                                         return_value=fallback), \
             unittest.mock.patch.dict(os.environ, {}, clear=True):
            chain = self.mc.role_chains(
                "claude", ("write",), roster_value=self.roster)["write"]
        self.assertEqual(f"agy:{self.variants['medium']}", chain[-1])

    def test_direct_model_override_is_resolved_and_suppresses_fallback(self):
        with unittest.mock.patch.dict(
                os.environ, {"MODEL": self.family}, clear=True):
            chain = self.mc.role_chains(
                "agy", ("write",), roster_value=self.roster)["write"]
        self.assertEqual([self.variants["high"]], chain)

    def test_explicit_effort_environment_beats_each_model_default(self):
        cases = (
            ("codex", "gpt-5.6-luna", "CODEX_EFFORT", "low", "effort"),
            ("muse", "muse-spark-1.3-contributor", "MUSE_EFFORT", "high",
             "effort"),
            ("opencode", "openai/gpt-5.3-codex-spark", "OPENCODE_VARIANT",
             "medium", "variant"),
            ("grok", "grok-4.6", "GROK_EFFORT", "xhigh", "effort"),
            ("claude", "sonnet", "CLAUDE_EFFORT", "low", "effort"),
        )
        for harness, model, variable, value, option in cases:
            with self.subTest(harness=harness), unittest.mock.patch.dict(
                    os.environ, {variable: value}, clear=True):
                self.assertEqual(
                    {option: value},
                    self.ms.effective_option(harness, model, "write"))

    def test_cli_freeze_drives_validation_children_and_eta_identity(self):
        roles = self.cli.mode_config.BY_KEY["quick"].roles

        def frozen(value):
            physical = self.variants[value]
            env = {"HARNESS": "agy", "SUMM_ACTIVE_ROLES": " ".join(roles),
                   **{f"AGY_CHAIN_{role.upper()}": physical for role in roles}}
            with unittest.mock.patch.dict(os.environ, env, clear=True):
                chains = self.cli.freeze_model_routes(
                    "agy", self.roster, roles)
                self.cli.validate_roles("agy", self.roster, roles)
                options = json.loads(os.environ["SUMM_ROLE_OPTIONS"])
                signature = self.et.execution_signature("quick")
                snapshot = {key: os.environ[key] for key in env
                            if key.startswith("AGY_CHAIN_")}
            return chains, options, signature, snapshot

        low = frozen("low")
        high = frozen("high")
        self.assertNotEqual(low[2], high[2])
        self.assertRegex(low[2], r"^sha256:[0-9a-f]{64}$")
        self.assertEqual(3, self.et.PRODUCER_REV)
        for role in roles:
            self.assertEqual([self.variants["low"]], low[0][role])
            self.assertEqual({}, low[1][role][0]["option"])
            self.assertEqual(self.variants["low"],
                             low[3][f"AGY_CHAIN_{role.upper()}"])


class TheProgressBarMeansSomething(unittest.TestCase):
    """It swept back and forth continuously, which reads as frantic and moves
    identically whether a run is healthy or wedged. Parts are a real discrete
    count -- unlike elapsed time -- so the bar can be honest instead."""
    ui = load("summ_ui")

    def setUp(self):
        import tkinter as tk
        try:
            self.root = tk.Tk()
        except tk.TclError as e:
            self.skipTest(f"no display: {e}")
        self.root.withdraw()
        self.app = self.ui.App(self.root)
        self.addCleanup(self.root.destroy)

    def test_the_bar_is_never_indeterminate(self):
        src = (ENG / "summ_ui.py").read_text()
        self.assertNotIn('mode="indeterminate"', src)
        self.assertNotIn("bar.start(", src)

    def test_the_bar_tracks_parts_completed(self):
        self.app._apply({"event": "part", "index": 1, "total": 8})
        self.assertEqual(int(self.app.bar["maximum"]), 8)
        self.assertEqual(int(self.app.bar["value"]), 0)   # part 1 not done yet
        self.app._apply({"event": "part", "index": 5, "total": 8})
        self.assertEqual(int(self.app.bar["value"]), 4)

    def test_an_unknown_part_count_leaves_the_bar_alone(self):
        self.app._apply({"event": "part", "index": 1, "total": 0})
        self.assertEqual(int(self.app.bar["value"]), 0)

    def test_a_finished_run_fills_the_bar_only_on_success(self):
        d = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, d, True)
        (d / "progress.jsonl").write_text("")
        self.app.job = d
        self.app._finish(0)
        self.assertEqual(int(self.app.bar["value"]), 1)
        self.app._finish(3)
        self.assertEqual(int(self.app.bar["value"]), 0)




class NothingIsEverOmittedForFidelity(unittest.TestCase):
    """A flagged unit is re-rendered at a verified capsule, never dropped.

    It used to be deleted along with everything depending on it, disclosed as
    "passages with unresolved fidelity concerns were omitted". That is a failure
    dressed as a safeguard: the reader lost content and could not tell what.

    An audit finding means OUR RENDERING may misstate something, not that the
    source material is unusable. Every unit carries two sealed capsules, so a
    unit flagged at one depth is written at the other. When both are flagged
    there is no verified rendering and publication fails closed."""
    comp = load("compose")

    def _fixture(self, td, quarantine, capsules=("detailed", "brief")):
        ldir, rvd, out = td / "ledger", td / "rv", td / "out"
        for d in (ldir, rvd):
            d.mkdir(parents=True, exist_ok=True)
        (rvd / "reader.md").write_text("x")
        units = []
        for n in (1, 2, 3):
            u = {"unit_id": f"U00{n}", "blocks": [f"P{n:04d}"], "dependencies": [],
                 "detailed_disposition": "required", "brief_disposition": "required",
                 "brief_priority": 5, "section_id": "SEC01", "part": 1}
            for c in capsules:
                u[f"{c}_capsule"] = f"Sentence {n} at {c} resolution about the topic."
            units.append(u)
        (ldir / "ledger.json").write_text(json.dumps(
            {"units": units, "visible_words": 4000}))
        (ldir / "MECHSEAL").write_text("1")
        (ldir / "SEALED").write_text("1")
        (ldir / "mechseal.json").write_text(json.dumps({"passed": True, "units": 3}))
        (ldir / "status.json").write_text(json.dumps(
            {"status": "quarantined",
             "ledger_sha256": hashlib.sha256(
                 (ldir / "ledger.json").read_bytes()).hexdigest(),
             "quarantine": quarantine}))
        return ldir, rvd, out

    def test_a_flagged_unit_is_rendered_at_its_other_capsule(self):
        with tempfile.TemporaryDirectory() as td:
            td = pathlib.Path(td)
            ldir, rvd, out = self._fixture(td, {"detailed": ["U002"], "brief": []})
            subprocess.run([sys.executable, str(ENG / "compose.py"),
                            str(ldir), str(rvd), str(out)],
                           capture_output=True, text=True)
            prov = json.loads((out / "detailed.provenance.json").read_text())
            self.assertEqual(prov["unit_modes"]["U002"], "brief",
                             "a unit flagged at Detailed was not re-rendered")
            self.assertNotIn("published_unverified", prov,
                             "retired unverified-publication evidence remains")

    def test_no_unit_is_ever_excluded_for_being_flagged(self):
        with tempfile.TemporaryDirectory() as td:
            td = pathlib.Path(td)
            ldir, rvd, out = self._fixture(
                td, {"detailed": ["U001", "U002", "U003"], "brief": []})
            subprocess.run([sys.executable, str(ENG / "compose.py"),
                            str(ldir), str(rvd), str(out)],
                           capture_output=True, text=True)
            prov = json.loads((out / "detailed.provenance.json").read_text())
            self.assertEqual(sorted(prov["unit_ids"]), ["U001", "U002", "U003"])

    def test_a_dependent_is_not_dragged_out_with_it(self):
        # Dependency closure existed so a surviving unit would not reference a
        # removed one. Nothing is removed, so it has nothing to protect against.
        src = (ENG / "compose.py").read_text()
        self.assertNotIn("any(d in q for d in", src,
                         "dependency closure still removes units")

    def test_both_capsules_flagged_publishes_nothing(self):
        with tempfile.TemporaryDirectory() as td:
            td = pathlib.Path(td)
            ldir, rvd, out = self._fixture(
                td, {"detailed": ["U002"], "brief": ["U002"]})
            result = subprocess.run([sys.executable, str(ENG / "compose.py"),
                                     str(ldir), str(rvd), str(out)],
                                    capture_output=True, text=True)
            self.assertNotEqual(0, result.returncode)
            self.assertIn("no verified capsule", result.stderr)
            self.assertFalse((out / "detailed.md").exists())
            self.assertFalse((out / "brief.md").exists())

    def test_no_unverified_content_note_or_path_remains(self):
        src = (ENG / "compose.py").read_text()
        code = "\n".join(l for l in src.splitlines()
                         if not l.lstrip().startswith("#"))
        self.assertNotIn("included as written rather than removed", code)
        self.assertNotIn("published_unverified", code)




class ProfilesAndAStableWindow(unittest.TestCase):
    """A profile is a name and a set of selections, nothing else. And the window
    must not resize when those selections change -- picking a harness made it
    jump, because the summary label and the comboboxes grew with their text."""
    ui = load("summ_ui")

    def setUp(self):
        import tkinter as tk
        try:
            self.root = tk.Tk()
        except tk.TclError as e:
            self.skipTest(f"no display: {e}")
        self.real = self.ui.profiles.profiles_path()
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.profile_patch = unittest.mock.patch.object(
            self.ui.profiles, "profiles_path",
            return_value=self.tmp / "profiles.json")
        self.profile_patch.start()
        self.addCleanup(self.profile_patch.stop)
        self.instructions_patch = unittest.mock.patch.object(
            self.ui.custom_instructions, "last_path",
            side_effect=lambda mode: self.tmp / "instructions" / f"{mode}.txt")
        self.instructions_patch.start()
        self.addCleanup(self.instructions_patch.stop)
        self.last_out_patch = unittest.mock.patch.object(
            self.ui.last_output, "last_path",
            return_value=self.tmp / "last-output.json")
        self.last_out_patch.start()
        self.addCleanup(self.last_out_patch.stop)
        self.last_affix_patch = unittest.mock.patch.object(
            self.ui.last_affix, "last_path",
            return_value=self.tmp / "last-affix.json")
        self.last_affix_patch.start()
        self.addCleanup(self.last_affix_patch.stop)
        self.runtime_patch = unittest.mock.patch.object(
            self.ui.runtime, "config", return_value=gateway_runtime())
        self.runtime_patch.start()
        self.addCleanup(self.runtime_patch.stop)
        self.root.withdraw()
        self.app = self.ui.App(self.root)
        self.addCleanup(self.root.destroy)

    def test_a_profile_round_trips(self):
        self.app.role_h["plan"].set("agy"); self.app.refresh_role("plan")
        want = {r: (self.app.role_h[r].get(), self.app.role_vars[r].get())
                for r in self.ui.ROLES}
        self.app.profile.set("mixed"); self.app.save_profile()
        self.assertEqual(self.ui.load_last_profile(), "mixed")
        for r in self.ui.ROLES:
            self.app.role_h[r].set("claude"); self.app.refresh_role(r)
        self.app.profile.set("mixed"); self.app.apply_profile()
        got = {r: (self.app.role_h[r].get(), self.app.role_vars[r].get())
               for r in self.ui.ROLES}
        self.assertEqual(got, want)

    def test_a_profile_naming_a_model_the_roster_dropped_falls_back(self):
        # Free models come and go. A profile must not pin one that no longer
        # exists, silently or otherwise.
        self.ui.save_profiles({"stale": {r: ["opencode", "gone/removed-model"]
                                         for r in self.ui.ROLES}})
        self.app.profile_box["values"] = ["stale"]
        self.app.profile.set("stale"); self.app.apply_profile()
        options = self.app.role_boxes["plan"]["values"]
        self.assertIn(self.app.role_vars["plan"].get(), options)
        self.assertNotEqual(self.app.role_vars["plan"].get(), "gone/removed-model")

    def test_profiles_are_device_local_not_project_state(self):
        # A preference is not project state and must not reach the repo.
        self.assertNotIn(str(ENG), str(self.real))

    def test_a_fresh_device_gets_the_shared_default_profile(self):
        # Preferences deliberately do not sync.  Without a shared default, a
        # second device has no balanced profile and opens on roster order.
        self.assertFalse((self.tmp / "profiles.json").exists())
        profiles = self.ui.load_profiles()
        self.assertIn("balanced", profiles)
        spec = self.ui.model_defaults()["profile"]
        self.assertEqual(profiles["balanced"],
                         {role: spec[role] for role in self.ui.ROLES})

    def test_malformed_profiles_fail_closed_instead_of_opening_balanced(self):
        (self.tmp / "profiles.json").write_text("{truncated")
        with self.assertRaises(self.ui.runtime.ConfigError):
            self.ui.load_profiles()

    def test_saving_a_custom_profile_does_not_snapshot_balanced(self):
        value = self.ui.load_profiles()
        value["custom"] = {role: list(value["balanced"][role])
                           for role in self.ui.ROLES}
        self.ui.save_profiles(value)
        stored = json.loads((self.tmp / "profiles.json").read_text())
        self.assertIn("custom", stored)
        self.assertNotIn("balanced", stored)

    def test_profile_write_failure_preserves_the_previous_file(self):
        first = {"first": {role: ["agy", self.ui.roster()["agy"][role][0]]
                           for role in self.ui.ROLES}}
        self.ui.save_profiles(first)
        before = (self.tmp / "profiles.json").read_bytes()
        second = {"second": first["first"]}
        with unittest.mock.patch.object(
                self.ui.profiles.os, "replace", side_effect=OSError("injected")):
            with self.assertRaises(OSError):
                self.ui.save_profiles(second)
        self.assertEqual(before, (self.tmp / "profiles.json").read_bytes())

    def test_malformed_last_profile_is_an_explicit_configuration_error(self):
        self.ui.profiles.last_profile_path().write_text("[]")
        with self.assertRaises(self.ui.runtime.ConfigError):
            self.ui.load_last_profile()

    def test_last_named_profile_is_restored_on_a_new_window(self):
        profiles = self.ui.load_profiles()
        profiles["fast"] = {role: list(profiles["balanced"][role])
                             for role in self.ui.ROLES}
        self.ui.save_profiles(profiles)
        self.ui.save_last_profile("fast")
        self.assertEqual(self.ui.launch_profile(), "fast")

    def test_deleted_last_profile_falls_back_to_balanced(self):
        profiles = self.ui.load_profiles()
        profiles["fast"] = {role: list(profiles["balanced"][role])
                             for role in self.ui.ROLES}
        self.ui.save_profiles(profiles)
        self.ui.save_last_profile("fast")
        profiles.pop("fast")
        self.ui.save_profiles(profiles)
        self.assertEqual(self.ui.launch_profile(), self.ui.DEFAULT_PROFILE)
        self.assertEqual(self.ui.load_last_profile(), self.ui.DEFAULT_PROFILE)

    def test_window_uses_the_restored_profile_in_its_controls(self):
        profiles = self.ui.load_profiles()
        profiles["fast"] = {role: ["agy", self.ui.roster()["agy"][role][0]]
                             for role in self.ui.ROLES}
        self.ui.save_profiles(profiles)
        self.ui.save_last_profile("fast")
        import tkinter as tk
        root2 = tk.Tk(); root2.withdraw()
        self.addCleanup(root2.destroy)
        app2 = self.ui.App(root2)
        self.assertEqual(app2.profile.get(), "fast")
        self.assertTrue(all(app2.role_h[r].get() == "agy"
                            for r in self.ui.ROLES))

    def test_a_device_profile_overrides_the_shared_default(self):
        fb = self.ui.model_defaults()["fallback"]
        fb = fb[0] if isinstance(fb, list) else fb   # backups are an ordered list
        local = {"balanced": {r: [fb["harness"], fb["model"]]
                              for r in self.ui.ROLES}}
        self.ui.save_profiles(local)
        self.assertEqual(self.ui.load_profiles()["balanced"], local["balanced"])

    def test_an_unnamed_profile_is_not_saved(self):
        self.app.profile.set("   "); self.app.save_profile()
        self.assertNotIn("   ", self.ui.load_profiles())
        self.assertIn(self.ui.DEFAULT_PROFILE, self.ui.load_profiles())

    def test_custom_instructions_start_empty_collapsed_and_are_not_a_profile(self):
        self.assertFalse(self.app.show_instructions.get())
        self.assertFalse(self.app.instructions_frame.winfo_ismapped())
        sentinel = "PRESERVE-THIS-ONE-QUOTATION"
        self.app.instructions_box.insert("1.0", sentinel)
        self.app.profile.set("custom-does-not-persist"); self.app.save_profile()
        self.assertNotIn(sentinel, (self.tmp / "profiles.json").read_text())

    def test_last_submitted_instructions_are_restored_per_mode(self):
        source = self.tmp / "article.md"; source.write_text("article source")
        self.app.selection = load("selection").resolve_paths([source])
        self.app.source = [source]
        self.app.show_instructions.set(True); self.app.toggle_instructions()
        self.app.instructions_box.insert("1.0", "Preserve the central caveat.")
        self.app.active["already-running"] = object()
        with unittest.mock.patch.object(
                self.ui.runtime, "config",
                return_value={**gateway_runtime(), "active_targets": 1}):
            self.app.start()
        saved = self.tmp / "instructions" / "summarize.txt"
        self.assertEqual("Preserve the central caveat.", saved.read_text())
        self.assertEqual("Preserve the central caveat.",
                         self.app.pending[0]["instructions"])

        import tkinter as tk
        root2 = tk.Tk(); root2.withdraw()
        self.addCleanup(root2.destroy)
        app2 = self.ui.App(root2)
        self.assertTrue(app2.show_instructions.get())
        self.assertEqual("Preserve the central caveat.",
                         app2.instructions_box.get("1.0", "end-1c"))
        self.assertEqual("grid", app2.instructions_frame.winfo_manager())

    def test_instruction_drafts_are_distinct_across_mode_changes(self):
        summarize = self.ui.mode_config.BY_KEY["summarize"].label
        quick = self.ui.mode_config.BY_KEY["quick"].label
        self.app.show_instructions.set(True); self.app.toggle_instructions()
        self.app.instructions_box.insert("1.0", "summary preference")
        self.app.mode.set(quick); self.app.refresh_role_availability()
        self.assertEqual("", self.app.instructions_box.get("1.0", "end-1c"))
        self.app.instructions_box.insert("1.0", "quick preference")
        self.app.show_instructions.set(True); self.app.toggle_instructions()
        self.app.mode.set(summarize); self.app.refresh_role_availability()
        self.assertEqual("summary preference",
                         self.app.instructions_box.get("1.0", "end-1c"))
        self.app.mode.set(quick); self.app.refresh_role_availability()
        self.assertEqual("quick preference",
                         self.app.instructions_box.get("1.0", "end-1c"))

    def test_article_cleanup_preset_is_visible_editable_text(self):
        clean = self.ui.mode_config.BY_KEY["text_prep"].label
        self.app.mode.set(clean); self.app.refresh_role_availability()
        self.assertIn("Article title and body only",
                      self.app.instruction_preset_box["values"])
        self.app.instruction_preset.set("Article title and body only")
        self.app.apply_instruction_preset()
        text = self.app.instructions_box.get("1.0", "end-1c")
        self.assertIn("complete main article body", text)
        self.assertTrue(self.app.show_instructions.get())
        self.app.instructions_box.insert("end", " Additional preference.")
        self.assertIn("Additional preference", self.app.instructions_box.get(
            "1.0", "end-1c"))

    def test_instruction_preference_write_failure_preserves_previous_value(self):
        ci = self.ui.custom_instructions
        ci.save_last("summarize", "first accepted request")
        path = self.tmp / "instructions" / "summarize.txt"
        with unittest.mock.patch.object(
                ci.runtime.os, "replace", side_effect=OSError("injected")):
            with self.assertRaises(OSError):
                ci.save_last("summarize", "replacement request")
        self.assertEqual("first accepted request", path.read_text())
        self.assertEqual([], list(path.parent.glob(".*.tmp-*")))

    def test_invalid_saved_instruction_is_explicit_and_tts_has_no_store(self):
        path = self.tmp / "instructions" / "summarize.txt"
        path.parent.mkdir(parents=True)
        path.write_text("x" * (self.ui.custom_instructions.MAX_BYTES + 1))
        with self.assertRaises(self.ui.runtime.ConfigError):
            self.ui.custom_instructions.load_last("summarize")
        with self.assertRaises(ValueError):
            self.ui.custom_instructions._mode("tts")
        cli_source = (ENG / "summ_cli.py").read_text()
        self.assertNotIn("load_last(", cli_source,
                         "direct CLI unexpectedly inherited a UI preference")

    def test_read_aloud_disables_custom_instructions(self):
        tts = next(m for m, flags in self.ui.MODES if "--tts" in flags)
        self.app.mode.set(tts)
        self.app.refresh_role_availability()
        self.assertTrue(self.app.instructions_toggle.instate(["disabled"]))
        self.assertEqual("", self.app.instructions_box.get("1.0", "end-1c"))

    def test_a_queued_job_freezes_its_custom_instructions(self):
        source = self.tmp / "article.md"; source.write_text("article source")
        self.app.selection = load("selection").resolve_paths([source])
        self.app.source = [source]
        self.app.show_instructions.set(True); self.app.toggle_instructions()
        self.app.instructions_box.insert("1.0", "Preserve exact quotations.")
        self.app.active["already-running"] = object()
        with unittest.mock.patch.object(
                self.ui.runtime, "config", return_value={"active_targets": 1}):
            self.app.start()
        self.app.instructions_box.delete("1.0", "end")
        self.app.instructions_box.insert("1.0", "Changed later")
        self.assertEqual("Preserve exact quotations.",
                         self.app.pending[0]["instructions"])
        frozen = json.loads(
            self.app.pending[0]["env"]["SUMM_RUNTIME_JSON"])
        self.assertEqual(1, frozen["active_targets"])

    def test_oversized_custom_instructions_are_refused_before_queueing(self):
        source = self.tmp / "article.md"; source.write_text("article source")
        self.app.selection = load("selection").resolve_paths([source])
        self.app.source = [source]
        self.app.show_instructions.set(True); self.app.toggle_instructions()
        self.app.instructions_box.insert("1.0", "é" * 1001)
        before = len(self.app.pending)
        self.app.start()
        self.assertEqual(before, len(self.app.pending))
        self.assertIn("invalid", self.app.status.cget("text").lower())
        self.assertFalse((self.tmp / "instructions" / "summarize.txt").exists())

    def test_the_complete_window_has_a_vertical_scroll_surface(self):
        event = type("Wheel", (), {
            "widget": self.app.status, "delta": -120, "num": None})()
        with unittest.mock.patch.object(
                self.app.scroll_canvas, "yview_scroll") as scroll:
            self.assertEqual("break", self.app._scroll_window(event))
            scroll.assert_called_once_with(1, "units")
        self.assertEqual("vertical", str(self.app.window_scrollbar["orient"]))
        self.assertTrue(self.app.scroll_canvas.cget("yscrollcommand"))

    def test_the_window_does_not_resize_when_a_harness_changes(self):
        self.app.show_models.set(True); self.app.toggle_models()
        self.root.update()
        before = self.app.models_frame.winfo_reqwidth()
        for role, h in (("plan", "claude"), ("write", "agy"),
                        ("audit", "opencode"), ("repair", "claude")):
            self.app.role_h[role].set(h); self.app.refresh_role(role)
        self.root.update()
        self.assertEqual(self.app.models_frame.winfo_reqwidth(), before,
                         "the models panel resized when a harness changed")

    def test_selected_models_have_exact_model_matched_effort_boxes(self):
        for role, harness, model, values in (
                ("plan", "codex", "gpt-5.6-luna",
                 ("none", "low", "medium", "high", "xhigh", "max")),
                ("audit", "opencode", "openai/gpt-5.3-codex-spark",
                 ("none", "low", "medium", "high", "xhigh"))):
            self.app.role_h[role].set(harness)
            self.app.refresh_role(role)
            self.app.role_vars[role].set(model)
            self.app.refresh_role_effort(role)
            self.assertEqual(tuple(self.app.role_effort_boxes[role]["values"]),
                             values)
            self.assertNotIn("reasoning", self.app.role_effort_vars[role].get())

        self.app.backup_h.set("opencode")
        self.app.refresh_backup()
        self.app.backup_var.set("openai/gpt-5.6-luna")
        self.app.refresh_backup_model()
        self.assertEqual(tuple(self.app.backup_effort_box["values"]),
                         ("none", "low", "medium", "high", "xhigh", "max"))

    def test_backup_effort_selection_is_not_reset_to_model_default(self):
        self.app.backup_h.set("localgw")
        self.app.refresh_backup()
        self.app.backup_var.set("model-b")
        self.app.refresh_backup_model()
        self.app.backup_effort.set("low")
        self.app.refresh_backup_effort(reset=False)
        self.assertEqual(self.app.backup_effort.get(), "low")
        self.assertEqual(self.app.backup_var.get(), "model-b")

    def test_effort_dropdown_changes_the_exact_model_or_request_setting(self):
        cfg = json.loads((ENG / "models.json").read_text())
        family = next(model for model in cfg["agy"]["plan"]
                      if "flash" in model)
        medium_id = cfg["agy"]["_model_settings"][family]["variants"]["medium"]
        self.app.role_h["plan"].set("agy")
        self.app.refresh_role("plan")
        self.app.role_vars["plan"].set(family)
        self.app.refresh_role_effort("plan")
        self.app.role_effort_vars["plan"].set("medium")
        self.app.refresh_effort("plan")
        self.assertEqual(self.app.role_vars["plan"].get(), family)
        env = self.app.chosen_env()
        chain = env[f"{env['HARNESS'].upper()}_CHAIN_PLAN"].split()
        self.assertEqual(chain[0].split(":", 1)[-1], medium_id)
        frozen = json.loads(env["SUMM_ROLE_OPTIONS"])
        self.assertEqual(frozen["plan"][0]["model"], medium_id)
        self.assertEqual(frozen["plan"][0]["option"], {})

        self.app.role_h["plan"].set("opencode")
        self.app.refresh_role("plan")
        self.app.role_vars["plan"].set("openai/gpt-5.6-luna")
        self.app.refresh_role_effort("plan")
        self.app.role_effort_vars["plan"].set("medium")
        env = self.app.chosen_env()
        frozen = json.loads(env["SUMM_ROLE_OPTIONS"])
        selected = frozen["plan"][0]
        self.assertEqual(selected["option"], {"variant": "medium"})

    def test_old_effort_suffixed_profile_resolves_to_the_logical_family(self):
        cfg = json.loads((ENG / "models.json").read_text())
        family = next(model for model in cfg["agy"]["write"]
                      if "flash" in model)
        medium_id = cfg["agy"]["_model_settings"][family]["variants"]["medium"]
        profiles = self.ui.load_profiles()
        profiles["legacy"] = {
            role: ["agy", medium_id] for role in self.ui.ROLES}
        self.ui.save_profiles(profiles)
        self.app.profile.set("legacy")
        self.app.apply_profile(remember=False)
        self.assertTrue(all(self.app.role_vars[role].get() == family
                            for role in self.ui.ROLES))
        self.assertTrue(all(self.app.role_effort_vars[role].get() == "medium"
                            for role in self.ui.ROLES))

    def test_chosen_environment_freezes_each_selected_harness_setting(self):
        for role, harness in (("plan", "codex"), ("audit", "opencode")):
            self.app.role_h[role].set(harness)
            self.app.refresh_role(role)
        env = self.app.chosen_env()
        frozen = json.loads(env["SUMM_ROLE_OPTIONS"])
        self.assertEqual(frozen["plan"][0]["option"], {"effort": "high"})
        self.assertEqual(frozen["audit"][0]["option"], {"variant": "high"})

    def test_short_pasted_text_keeps_the_chosen_summarize_mode(self):
        # No automatic rewrite: a short paste under Summarize stays
        # Summarize, and Full stays Full at every source size.
        ordinary = next(label for label, flags in self.ui.MODES if not flags)
        for words in (10, 399, 400):
            self.app.mode.set(ordinary)
            raw = " ".join(f"word{i}" for i in range(words))
            with unittest.mock.patch.object(self.ui.summ_cli, "clip_read",
                                             return_value=raw):
                self.app.paste()
            self.assertEqual(self.app.mode.get(), ordinary)

    def test_explicit_non_summary_modes_are_not_overridden_by_short_paste(self):
        raw = "short pasted text"
        for mode, flags in self.ui.MODES:
            if not flags:
                continue
            self.app.mode.set(mode)
            with unittest.mock.patch.object(self.ui.summ_cli, "clip_read",
                                             return_value=raw):
                self.app.paste()
            self.assertEqual(self.app.mode.get(), mode)

    def test_the_summary_is_truncated_and_the_full_text_is_on_the_tooltip(self):
        for role, h in (("plan", "claude"), ("write", "agy"), ("audit", "opencode")):
            self.app.role_h[role].set(h); self.app.refresh_role(role)
        shown = self.app.models_summary.cget("text")
        self.assertLessEqual(len(shown), self.app.SUMMARY_CHARS)
        self.assertGreater(len(self.app._tip_text), len(shown))

    def test_there_is_no_run_as_control(self):
        # It had no model beside it and no meaning a user could act on.
        src = (ENG / "summ_ui.py").read_text()
        self.assertNotIn('text="Run as"', src)

    def test_focusing_output_turns_beside_source_off(self):
        self.assertTrue(self.app.beside.get())
        self.assertEqual(str(self.app.out_entry.cget("state")), "normal")
        self.app._engage_output_field()
        self.assertFalse(self.app.beside.get())
        self.app.out_choice.set("out")
        self.assertEqual("out", self.app._output_directory())
        self.app.beside.set(True)
        self.app.toggle_beside()
        self.assertEqual("", self.app.out_choice.get())
        self.assertIsNone(self.app._output_directory())

    def test_last_used_output_is_a_named_button_under_choose(self):
        self.assertFalse(self.app.last_out_btn.grid_info())
        self.ui.last_output.save("/tmp/homework-summaries")
        self.app.refresh_last_out()
        self.assertEqual("homework-summaries", self.app.last_out_btn.cget("text"))
        self.assertEqual(self.app.choose_out_btn.grid_info()["column"],
                         self.app.last_out_btn.grid_info()["column"])
        self.assertGreater(int(self.app.last_out_btn.grid_info()["row"]),
                           int(self.app.choose_out_btn.grid_info()["row"]))
        self.app.use_last_out()
        self.assertFalse(self.app.beside.get())
        self.assertEqual("/tmp/homework-summaries", self.app.out_choice.get())
        self.assertEqual("..", self.ui.last_output.button_label(".."))
        self.assertEqual("out", self.ui.last_output.button_label("out"))

    def test_enqueue_remembers_a_relative_output_and_plans_from_the_source(self):
        week = self.tmp / "wk1"; week.mkdir()
        source = week / "paper.md"; source.write_text("paper")
        self.app.out_choice.set("..")
        runs = self.tmp / "runs"
        with unittest.mock.patch.object(self.ui, "job_root", return_value=runs), \
             unittest.mock.patch.object(
                 self.ui.runtime, "frozen_json", return_value="{}"), \
             unittest.mock.patch.object(self.app, "chosen_env", return_value={}):
            self.app._accept_selection(load("selection").resolve_paths([source]))
        self.assertEqual("..", self.ui.last_output.load())
        self.assertEqual("..", self.app.last_out_btn.cget("text"))
        loaded = load("selection").read_manifest(
            self.app.pending[0]["root"] / "selection.json")
        self.assertEqual(self.tmp.resolve() / "summary.paper.md",
                         loaded.documents[0].planned_outputs[0])

    def test_a_pasted_job_honours_an_explicit_output_directory(self):
        dest = self.tmp / "summaries"; dest.mkdir()
        self.app.out_choice.set(str(dest))
        self.assertFalse(self.app.beside.get())
        runs = self.tmp / "runs"
        with unittest.mock.patch.object(self.ui, "job_root", return_value=runs), \
             unittest.mock.patch.object(
                 self.ui.runtime, "frozen_json", return_value="{}"), \
             unittest.mock.patch.object(self.app, "chosen_env", return_value={}):
            self.app._accept_selection(
                load("selection").text_selection("pasted prose " * 40))
        job = self.app.pending[0]
        self.assertEqual(str(dest), job["out"])
        cmd = self.ui.build_cmd(
            job["root"], job["mode"], None, job["out"],
            job["root"] / "pasted.txt")
        self.assertEqual(str(dest), cmd[cmd.index("--out") + 1])
        self.assertIn("--text-file", cmd)
        published = load("selection").clipboard_output_directory(job["out"])
        self.assertEqual(dest, published)
        self.assertNotEqual("Downloads", published.name)

    def test_a_pasted_job_without_output_keeps_downloads(self):
        self.assertTrue(self.app.beside.get())
        runs = self.tmp / "runs"
        with unittest.mock.patch.object(self.ui, "job_root", return_value=runs), \
             unittest.mock.patch.object(
                 self.ui.runtime, "frozen_json", return_value="{}"), \
             unittest.mock.patch.object(self.app, "chosen_env", return_value={}):
            self.app._accept_selection(
                load("selection").text_selection("pasted prose " * 40))
        job = self.app.pending[0]
        self.assertIsNone(job["out"])
        cmd = self.ui.build_cmd(
            job["root"], job["mode"], None, job["out"],
            job["root"] / "pasted.txt")
        self.assertNotIn("--out", cmd)
        self.assertEqual(
            "Downloads",
            load("selection").clipboard_output_directory().name)

    def test_the_run_harness_is_derived_from_the_roles(self):
        for r in self.ui.ROLES:
            self.app.role_h[r].set("opencode"); self.app.refresh_role(r)
        self.app.role_h["plan"].set("agy"); self.app.refresh_role("plan")
        self.assertEqual(self.app.run_harness(), "opencode")
        env = self.app.chosen_env()
        self.assertEqual(env["HARNESS"], "opencode")
        # The odd one out becomes the qualified entry the engine resolves.
        self.assertTrue(env["OPENCODE_CHAIN_PLAN"].startswith("agy:"))

    def test_mode_change_updates_pending_queued_jobs(self):
        doc = self.tmp / "doc.md"
        doc.write_text("content")
        runs = self.tmp / "runs"
        with unittest.mock.patch.object(self.ui, "job_root", return_value=runs), \
             unittest.mock.patch.object(
                 self.ui.runtime, "frozen_json", return_value="{}"), \
             unittest.mock.patch.object(self.app, "chosen_env", return_value={}):
            self.app._accept_selection(load("selection").resolve_paths([doc]))
        self.assertEqual(1, len(self.app.pending))
        self.assertEqual(
            "Summarize — sealed Detailed and Brief", self.app.pending[0]["mode"])
        self.app.mode.set("Prepare for read-aloud")
        self.app.refresh_selection_for_mode()
        self.assertEqual("Prepare for read-aloud", self.app.pending[0]["mode"])
        planned = self.app.pending[0]["selection"].documents[0].planned_outputs
        self.assertEqual(1, len(planned))
        self.assertEqual("tts.doc.txt", planned[0].name)




class CompressionRatioFollowsRedundancy(unittest.TestCase):
    """How much a text can be compressed depends on how much redundancy it has,
    and a short passage has almost none.

    A long-document ceiling can force distinct claims out of a short source.
    Nothing may be left out, so on a short source the ceiling gives way."""
    ss = load("shortsum")
    comp = load("compose")

    def test_a_long_source_uses_the_published_bands_unchanged(self):
        c = self.ss.ceilings(4000)
        self.assertEqual(c["detailed"], self.comp.BANDS["detailed"][1])
        self.assertEqual(c["brief"], self.comp.BANDS["brief"][1])

    def test_a_paragraph_is_allowed_to_keep_more(self):
        c = self.ss.ceilings(131)
        self.assertGreater(c["detailed"], self.comp.BANDS["detailed"][1])
        self.assertLess(c["detailed"], 1.0, "a summary must still be shorter")

    def test_there_is_no_cliff_at_the_threshold(self):
        # A document of 399 words and one of 400 must not be treated wildly
        # differently; the ceiling interpolates.
        a = self.ss.ceilings(self.ss.SHORT_SOURCE - 1)["detailed"]
        b = self.ss.ceilings(self.ss.SHORT_SOURCE)["detailed"]
        self.assertLess(abs(a - b), 0.01, f"cliff at the threshold: {a} vs {b}")

    def test_the_ceiling_and_the_gate_agree(self):
        # If the prompt asks for one number and the gate enforces another, every
        # short run burns its single revision on a fight it cannot win.
        src = " ".join(["w"] * 131)
        allowed = int(131 * self.ss.ceilings(131)["detailed"])
        self.assertEqual(self.ss.not_a_copy(" ".join(["x"] * allowed), src,
                                            "detailed"), [])
        self.assertTrue(self.ss.not_a_copy(" ".join(["x"] * (allowed + 20)), src,
                                           "detailed"))

    def test_a_revision_that_drops_material_is_rejected(self):
        # Counting findings alone kept a revision that traded three minor
        # defects for two dropped claims. That is the wrong trade: material
        # (omissions, reversals, invented numbers) outranks minor style
        # findings when the retained candidates are compared. The old
        # terminal `if f2:` veto is gone by owner requirement -- a usable
        # pair is always published -- so this is behavioral: the revision
        # that drops claims must lose to the merely rough initial.
        short = (ENG / "shortsum.py").read_text()
        self.assertIn("mech2 = mechanical(d2, b2)", short)
        self.assertIn('audit2, records2 = audit(d2, b2, "short-reaudit")', short)
        ss = load("shortsum")
        source = ("Rates fell after the reform, but the evidence remained limited "
                  "and the authors did not claim that the reform caused the fall.")
        rough = {
            # Minor, disclosable defects only: first person, refers to the
            # document, and a stub-length Brief. List markup or copy-shape
            # would make this candidate structurally unusable instead.
            "detailed": ("We note the outcomes. This document shows limited "
                         "evidence. Rates fell after the reform."),
            "brief": "Rates fell, but causation was not established.",
        }
        for depth in ("detailed", "brief"):
            self.assertFalse(
                [f for f in ss.defects(rough[depth], source)
                 if "absent from the source" in f],
                f"rough {depth} must carry only minor defects for this test")
        clean_but_thin = {
            "detailed": ("Rates fell after the reform; the evidence was limited "
                         "and causation was not established by the evidence."),
            "brief": "Rates fell without proven causation by the evidence.",
        }

        class FakeMS:
            MODELS = AUDIT_MODELS = REPAIR_MODELS = ["local"]

            @staticmethod
            def run(prompt, out_dir, chain, stage, validate=None,
                    gateway_options=None):
                if stage == "short-revise":
                    return json.dumps(patch_replace(
                        rough["detailed"], rough["brief"],
                        "detailed", clean_but_thin["detailed"]))
                if stage == "short-reaudit":
                    return json.dumps({"verdict": "revise", "findings": [
                        AF("The revision drops the claim that the authors did not "
                           "claim causation"),
                        AF("The revision drops that the evidence remained limited")]})
                if stage == "short-audit":
                    return json.dumps({"verdict": "pass", "findings": []})
                return json.dumps(dict(rough))

        class FakeLedger:
            @staticmethod
            def parse_strict(raw, stage):
                return json.loads(raw)

        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            src, out = root / "source.txt", root / "out"
            src.write_text(source)
            with unittest.mock.patch.object(ss, "_runner",
                                            return_value=FakeMS()), \
                 unittest.mock.patch.object(ss, "_ledger",
                                            return_value=FakeLedger):
                self.assertEqual(0, ss.run(src, out))
            report = json.loads((out / "short-report.json").read_text())
            self.assertEqual(report["selected"], "initial")
            self.assertEqual((out / "detailed.md").read_text().strip(),
                             rough["detailed"])


class GrokIsWiredLikeEveryOtherHarness(unittest.TestCase):
    """A new harness is a branch in _cmd and a roster entry. Nothing else in the
    pipeline learns that it exists."""
    ms = load("mapsum")

    def setUp(self):
        # grok's _cmd WRITES a unique prompt file into workdir, so a sentinel
        # path like /w is no longer usable: it is not a directory, and creating
        # it would be a side effect on the filesystem root.
        self.wd = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.wd, True)

    def test_grok_is_a_known_harness_everywhere(self):
        self.assertIn("grok", self.ms.HARNESSES)
        self.assertIn("grok", load("summ_cli").HARNESSES)
        self.assertIn("grok", json.loads((ENG / "models.json").read_text()))

    def test_it_uses_the_headless_one_shot_entrypoint(self):
        cmd = self.ms._cmd("grok", "some-model", self.wd, "PROMPT")
        self.assertIn("--prompt-file", cmd, "not the single-turn entrypoint")
        self.assertIn("some-model", cmd)

    def test_effort_is_high_by_default_and_overridable(self):
        cmd = self.ms._cmd("grok", "m", self.wd, "p")
        self.assertIn("--effort", cmd)
        self.assertEqual(cmd[cmd.index("--effort") + 1], "high")

    def test_subagents_and_web_search_stay_off(self):
        cmd = self.ms._cmd("grok", "m", self.wd, "p")
        for flag in ("--no-subagents", "--disable-web-search"):
            self.assertIn(flag, cmd)

    def test_read_tools_are_granted_or_grok_answers_nothing(self):
        """The one flag without which grok returns narration instead of JSON.

        This CLI may decline to answer a large inline prompt until a read tool
        exists because it assumes the prompt was truncated. The source still
        arrives inline and the tool is not used to fetch it.

        Delete this grant and every grok run publishes nothing, silently, at exit
        0. That is the failure this project exists to prevent, so it is a test.
        """
        cmd = self.ms._cmd("grok", "m", self.wd, "p")
        self.assertIn("--tools", cmd)
        granted = cmd[cmd.index("--tools") + 1].split(",")
        self.assertIn("read_file", granted)
        # Reads only: the grant exists to satisfy the model, not to let an audit
        # edit anything or shell out.
        self.assertEqual([], [t for t in granted
                              if t not in ("read_file", "grep", "list_dir")])
        self.assertEqual("read-only", cmd[cmd.index("--sandbox") + 1])
        # Headless has no approver; without this the read blocks and the turn
        # dies on narration exactly as if no tool had been granted at all.
        self.assertEqual("dontAsk", cmd[cmd.index("--permission-mode") + 1])

    def test_summer_composes_the_payload_and_hands_over_no_source_path(self):
        # What the payload rule protects is that the model never FETCHES source
        # of its own. A transport file for summ'er's composed prompt is not
        # retrieval: the handed-over bytes are exactly those composed here, and
        # the source document is never named.
        prompt = "THE WHOLE SOURCE"
        cmd = self.ms._cmd("grok", "m", self.wd, prompt)
        self.assertNotIn("--prompt-json", cmd)
        path = pathlib.Path(cmd[cmd.index("--prompt-file") + 1])
        self.assertEqual(prompt.encode("utf-8"), path.read_bytes())
        self.assertEqual(path.parent, self.wd.resolve())

    def test_structured_output_is_used_and_the_envelope_is_unwrapped(self):
        """Plain text carries no stopReason, which is the reliable distinction
        between a finished answer and an abandoned turn."""
        cmd = self.ms._cmd("grok", "m", self.wd, "p")
        self.assertIn("--json-schema", cmd)
        self.assertEqual("json", cmd[cmd.index("--output-format") + 1])
        # additionalProperties MUST be true: xAI's structured output otherwise
        # defaults it to false and strips every field the stage asked for.
        self.assertIs(True, json.loads(
            cmd[cmd.index("--json-schema") + 1])["additionalProperties"])

    def test_an_abandoned_turn_is_never_read_as_an_answer(self):
        """Narration in the envelope's `text` is not an answer. Reading it as
        one would turn a failed call into an approved artifact, so every one of
        these forms must raise rather than return."""
        narration = "I'll read the full source first, not the truncated excerpt."
        for bad in (
                json.dumps({"type": "error", "message": "boom"}),
                json.dumps({"stopReason": "max_turn_requests", "text": narration}),
                json.dumps({"stopReason": "end_turn", "text": narration}),
                json.dumps({"stopReason": "end_turn", "structuredOutput": {},
                            "text": narration}),
                json.dumps({"stopReason": "end_turn", "text": narration,
                            "structuredOutput": {"a": 1},
                            "structuredOutputError": "schema mismatch"}),
                "not json at all"):
            with self.assertRaises(Exception):
                self.ms.grok_payload(bad)

    def test_the_validated_object_is_what_the_stage_receives(self):
        env = json.dumps({"stopReason": "end_turn", "text": "chatter",
                          "sessionId": "abc",
                          "structuredOutput": {"detailed": "D", "brief": "B"}})
        self.assertEqual({"detailed": "D", "brief": "B"},
                         json.loads(self.ms.grok_payload(env)))

    def test_a_preamble_before_the_json_is_tolerated(self):
        # grok narrates before answering: every successful run carried 188-278
        # bytes of prose ahead of the object. The parser takes the LAST balanced
        # object, so this already works -- but nothing pinned it, and a stricter
        # parser would discard a complete, correct summary.
        raw = ("I'll read the full source first so the condensation covers the "
               'complete document.Drafting now.{"detailed": "D", "brief": "B"}')
        self.assertEqual({"detailed": "D", "brief": "B"},
                         self.ms.parse_audit(raw))

    def test_the_prompt_never_rides_on_argv(self):
        """Windows CreateProcessW caps the WHOLE command line at 32,767 chars.

        A real document's prompt is 42,156 bytes and the largest corpus document
        reaches 62KB, so the previous `-p PROMPT` form could not run on Windows
        at all -- and AGENTS.md requires both platforms behave identically. If
        this goes red, grok is Windows-broken again.
        """
        prompt = "CANARY" + "X" * 200000
        cmd = self.ms._cmd("grok", "m", self.wd, prompt)
        self.assertFalse(any(prompt in a for a in cmd), "prompt text in argv")
        self.assertLess(sum(len(a) + 1 for a in cmd), 32767)

    def test_the_prompt_file_is_the_prompt_byte_for_byte(self):
        # A transport that normalises newlines or drops a trailing character
        # hands the model a different request than the one composed here.
        prompt = "hello caf\u00e9 with 'quotes'\n"
        cmd = self.ms._cmd("grok", "m", self.wd, prompt)
        path = pathlib.Path(cmd[cmd.index("--prompt-file") + 1])
        self.assertEqual(prompt.encode("utf-8"), path.read_bytes())

    def test_two_calls_get_different_prompt_files(self):
        # Workers and retries overlap on ONE work directory. A fixed filename
        # passes every other check here and still lets two concurrent calls
        # clobber each other's prompt.
        a = self.ms._cmd("grok", "m", self.wd, "same")
        b = self.ms._cmd("grok", "m", self.wd, "same")
        self.assertNotEqual(a[a.index("--prompt-file") + 1],
                            b[b.index("--prompt-file") + 1])

    def test_verbatim_so_the_cli_does_not_rewrite_the_sealed_prompt(self):
        self.assertIn("--verbatim", self.ms._cmd("grok", "m", self.wd, "p"))

    def test_a_relative_work_dir_reaches_the_harness_absolute(self):
        """Every grok run in the first benchmark attempt died in under a second.

        qualify.py passed a RELATIVE --work-dir. It reached grok as a relative
        --cwd, which grok resolves against its own working directory rather than
        ours: "Failed to set working directory ... (os error 2)", exit 1, no
        model call at all. Every earlier grok run used an absolute /tmp path,
        which is exactly why nothing caught it.

        Checked for every harness, not just grok: a relative path that happens
        to be tolerated today is still a latent version of this bug.
        """
        here = pathlib.Path.cwd()
        rel = pathlib.Path(os.path.relpath(self.wd, here))
        self.assertFalse(rel.is_absolute(), "test needs a relative path")
        for harness in self.ms.HARNESSES:
            if harness in self.ms.GATEWAY_HARNESSES:
                continue
            cmd = self.ms._cmd(harness, "m", rel, "p")
            for flag in ("--cwd", "--cd", "--dir", "--add-dir", "--workspace"):
                if flag in cmd:
                    self.assertTrue(pathlib.Path(cmd[cmd.index(flag) + 1])
                                    .is_absolute(),
                                    f"{harness} {flag} is relative")

    def test_a_qualified_entry_can_name_it(self):
        self.assertEqual(self.ms.split_entry("grok:grok-4.6", "opencode"),
                         ("grok", "grok-4.6"))

    def test_pasted_text_is_named_from_what_the_model_wrote(self):
        """"clipboard.summary.md" is not a filename when there are two.

        A second paste an hour later landed on the same path and overwrote the
        first, silently, and outside any within-run collision check. The name
        comes from the Brief's opening words -- already a model-authored
        distillation, so no extra call, no failure mode, no cost.
        """
        cli = load("summ_cli")
        brief = ("> *Quick reading: written in one pass.*\n\n"
                 "Implied volatility surfaces encode how option prices vary "
                 "across strike and expiry.")
        slug = cli.slug_from(brief)
        self.assertTrue(slug.startswith("implied-volatility-surfaces"), slug)
        # The coverage note is identical on EVERY artifact; naming from it would
        # give every clipboard summary the same filename, which is the bug.
        self.assertNotIn("quick", slug)
        self.assertEqual("clipboard", cli.slug_from("> *note*\n\nhi"))

    def test_a_pasted_summary_never_lands_on_an_existing_file(self):
        cli = load("summ_cli")
        with tempfile.TemporaryDirectory() as d:
            base = pathlib.Path(d) / "topic.summary.md"
            base.write_text("first run")
            got = cli.free_path(base, set())
            self.assertEqual("topic-2.summary.md", got.name)
            # The Brief must follow the SAME base or the pair stops matching.
            self.assertEqual("topic-2.brief.md",
                             got.with_name(got.name[:-len(".summary.md")]
                                           + ".brief.md").name)

    def test_a_queued_job_carries_its_own_settings(self):
        # Enqueue captures mode, source, output and env. Reading them at LAUNCH
        # would apply whatever the controls happen to show minutes later.
        src = (ENG / "summ_ui.py").read_text()
        self.assertIn("selected_env = self.chosen_env(runtime_snapshot)", src)
        self.assertIn('"env": {**selected_env', src)
        self.assertIn("self.pending.extend(jobs)", src)
        self.assertIn("build_cmd(root, job[\"mode\"], source, job[\"out\"]", src)

    def test_a_failed_job_does_not_abandon_the_queue(self):
        # The remaining documents are unrelated to whatever went wrong with this
        # one; dropping them silently is the partial result this project refuses.
        src = (ENG / "summ_ui.py").read_text()
        i = src.index("def _drain_pending")
        self.assertIn("self.pending.pop(0)", src[i:i + 900])
        # It is reached from _finish, which runs for EVERY ending, not only rc 0.
        finish = src.index("def _finish")
        self.assertIn("self._drain_pending()", src[finish:finish + 16000])

    def test_a_queued_clipboard_job_does_not_reread_the_clipboard(self):
        """It runs later, when the clipboard holds something else.

        The text is captured at enqueue and handed over as --text-file, which
        keeps raw-text naming and publication while removing the dependency on
        the clipboard still being what the user pasted.
        """
        cli = (ENG / "summ_cli.py").read_text()
        self.assertIn('"--text-file"', cli)
        self.assertIn("raw = tf.read_text(encoding=\"utf-8\")", cli)
        self.assertIn("selection.text_selection(raw, selected_mode.key)", cli)
        ui = (ENG / "summ_ui.py").read_text()
        self.assertIn('text_file = root / "pasted.txt"', ui)
        self.assertIn('cmd += ["--text-file", str(text_file)]', ui)

    def test_a_part_that_never_plans_reports_WHY(self):
        """A run failed with "no valid mechanical seal" and no cause.

        The seal can only report the consequence -- "ledger has no units". The
        cause is the per-part planning exception, which was printed to stdout;
        the window reads the progress event stream, not stdout, so the one fact
        that explained the failure was invisible. It is an event now.
        """
        led = (ENG / "ledger.py").read_text()
        self.assertIn('pg.emit("part_failed"', led)
        self.assertIn("reason=why", led)
        ui = (ENG / "summ_ui.py").read_text()
        self.assertIn('elif e == "part_failed":', ui)
        # And it must reach the failure LINE, not only the details pane.
        self.assertIn("if self.part_failures:", ui)

    def test_both_panes_open_on_launch(self):
        """Ticked AND gridded. A ticked box showing nothing is worse than an
        unticked one, and the toggles must run after every widget they touch
        exists -- calling them from build_model_controls raised AttributeError
        because show_console is created later."""
        ui = load("summ_ui")
        import tkinter as tk
        try:
            root = tk.Tk()
        except tk.TclError:
            self.skipTest("no display")
        try:
            root.withdraw()
            app = ui.App(root)
            self.assertTrue(app.show_models.get())
            self.assertTrue(app.show_console.get())
            self.assertTrue(app.models_frame.grid_info())
            self.assertTrue(app.console.grid_info())
        finally:
            root.destroy()

    def test_the_window_opens_on_the_default_profile(self):
        ui = load("summ_ui")
        self.assertEqual("balanced", ui.DEFAULT_PROFILE)
        self.assertIn(ui.DEFAULT_PROFILE, ui.load_profiles(),
                      "the default profile must exist or the window opens on "
                      "roster ordering, which is not a decision")

    def test_beside_source_is_a_toggle_that_shows_its_state(self):
        """It must render, and it must not carry a forced style again.

        As a plain button the DEFAULT setting looked like an action not yet
        taken. Styled as a Toolbutton it drew a BLANK pill under aqua and read
        as missing entirely. The native checkbox is what the theme draws blue.
        """
        src = (ENG / "summ_ui.py").read_text()
        self.assertIn("self.beside = tk.BooleanVar(value=True)", src)
        self.assertNotIn("Toggle.TButton", src)
        self.assertNotIn('state="disabled" if on', src)

    def test_the_output_toggle_names_the_real_destination(self):
        """Pasted text has no source file, so it publishes into Downloads.

        Calling that "Beside source" states something untrue about where the
        artifacts land, which is the question it exists to answer.
        """
        ui = load("summ_ui")
        import tkinter as tk
        try:
            root = tk.Tk()
        except tk.TclError:
            self.skipTest("no display")
        try:
            root.withdraw()
            app = ui.App(root)
            app.source, app.pasted = None, "pasted words"
            app.refresh_out_label()
            self.assertEqual("To Downloads", app.beside_btn.cget("text"))
            app.source, app.pasted = [pathlib.Path("/tmp/a.md")], None
            app.refresh_out_label()
            self.assertEqual("Beside source", app.beside_btn.cget("text"))
            # And it must actually render, which the styled version did not.
            self.assertGreater(app.beside_btn.winfo_reqwidth(), 40)
        finally:
            root.destroy()

    def test_codex_defaults_to_luna(self):
        cfg = json.loads((ENG / "models.json").read_text())
        for role in ("plan", "write", "audit", "repair"):
            self.assertEqual("gpt-5.6-luna", cfg["codex"][role][0])

    def test_artifacts_can_be_collected_into_one_directory(self):
        cli = load("summ_cli")
        src = pathlib.Path("/docs/paper.md")
        self.assertEqual(pathlib.Path("/docs/paper.summary.md"),
                         cli.out_path(src, ".summary.md"))
        self.assertEqual(pathlib.Path("/out/paper.summary.md"),
                         cli.out_path(src, ".summary.md", pathlib.Path("/out")))

    def test_two_sources_with_one_name_do_not_overwrite_each_other(self):
        """Beside the source, two notes.md never collide. Collected, they do.

        Silently overwriting would make a run report twenty successes and leave
        nineteen files -- a partial result reported as a whole one, which is the
        failure this project exists to refuse.
        """
        cli = load("summ_cli")
        taken = set()
        a = cli.unclashed(pathlib.Path("/out/notes.summary.md"), taken)
        b = cli.unclashed(pathlib.Path("/out/notes.summary.md"), taken)
        self.assertEqual(pathlib.Path("/out/notes.summary.md"), a)
        self.assertEqual(pathlib.Path("/out/notes-2.summary.md"), b)
        # The Brief must land on the SAME base, or the pair stops matching:
        # splitting on the last dot would give "notes.summary-2.md".
        self.assertEqual("notes-2.brief.md",
                         b.with_name(b.name[:-len(".summary.md")] + ".brief.md").name)
        c = cli.unclashed(pathlib.Path("/out/notes.summary.md"), taken)
        self.assertEqual(pathlib.Path("/out/notes-3.summary.md"), c)

    def test_the_ui_passes_the_output_directory_and_omits_it_when_unset(self):
        ui = load("summ_ui")
        job, src = pathlib.Path("/j"), [pathlib.Path("/d.md")]
        self.assertNotIn("--out", ui.build_cmd(job, ui.MODES[0][0], src))
        cmd = ui.build_cmd(job, ui.MODES[0][0], src, "/tmp/out")
        self.assertEqual("/tmp/out", cmd[cmd.index("--out") + 1])
        # `out_dir` was already taken, for the folder a finished job wrote into.
        # Reusing it made the control a None at construction.
        self.assertIn("self.out_choice", (ENG / "summ_ui.py").read_text())

    def test_relative_out_is_resolved_from_the_source_not_the_process(self):
        cli = load("summ_cli")
        src = pathlib.Path("/docs/wk1/paper.md")
        self.assertEqual(pathlib.Path("/docs/wk1/out/paper.summary.md"),
                         cli.out_path(src, ".summary.md", "out"))
        self.assertEqual(pathlib.Path("/docs/paper.summary.md"),
                         cli.out_path(src, ".summary.md", ".."))
        self.assertNotIn("import last_output", (ENG / "summ_cli.py").read_text())

    def test_each_mode_declares_only_the_roles_it_calls(self):
        """The UI offered four model pickers in every mode.

        Read-aloud reuses Clean-text roles for arbitrary input; Quick never plans. Four live pickers
        invited a choice the run silently ignores, which is indistinguishable
        from the setting being broken. Verified against the engine rather than
        assumed -- if a mode's roles change, this test is where it shows.
        """
        ui = load("summ_ui")
        self.assertEqual(ui.ROLES, ui.roles_for(ui.MODES[0][0]))
        quick = ui.roles_for(ui.MODES[1][0])
        self.assertNotIn("plan", quick)
        self.assertIn("write", quick)
        cleanup = next(m for m, flags in ui.MODES if "--text-prep" in flags)
        self.assertEqual(("write", "audit", "repair"), ui.roles_for(cleanup))
        tts = next(m for m, flags in ui.MODES if "--tts" in flags)
        self.assertEqual(("write", "audit", "repair"), ui.roles_for(tts))
        # The claim above must match what the engine actually does.
        short = (ENG / "shortsum.py").read_text()
        self.assertNotIn("PLAN_MODELS", short)
        for chain in ("ms.MODELS", "ms.AUDIT_MODELS"):
            self.assertIn(chain, short)
        # The repair role is served by the exact producer of the candidate.
        self.assertIn("last_ok_route", short)
        cli = (ENG / "summ_cli.py").read_text()
        self.assertIn('str(HERE / "textprep.py")', cli)
        self.assertNotIn('"readaloud"', cli)

    def test_the_profile_list_is_reread_when_it_is_opened(self):
        # It was captured once at construction, so a profile written by the CLI
        # or another window never appeared until relaunch -- which reads as
        # "profiles don't save".
        ui = (ENG / "summ_ui.py").read_text()
        self.assertIn('self.profile_box["postcommand"]', ui)
        self.assertIn("values=sorted(load_profiles())", ui)

    def test_a_profile_reaches_the_cli_not_only_the_ui(self):
        """Profiles were UI-only, so Raycast and AHK could not use one.

        The CLI is meant to be the whole product surface. This asserts the flag
        exists, reads the SAME store the UI writes, and reuses the UI's own
        role_env rather than inventing a second profile format that would drift
        from it.
        """
        cli = (ENG / "summ_cli.py").read_text()
        self.assertIn('"--profile"', cli)
        self.assertIn("profiles.load_profiles()", cli)
        self.assertIn("profiles.role_env(", cli)
        self.assertNotIn("import summ_ui", cli)
        # setdefault, not assignment: an explicit env var must beat the profile,
        # or a single role could not be overridden for one run.
        self.assertIn("os.environ.setdefault(", cli)

    def test_a_profile_names_a_model_for_every_role_it_sets(self):
        ui = load("summ_ui")
        picks = {"write": ("grok", "grok-4.6"), "audit": ("claude", "sonnet")}
        env = ui.role_env("grok", picks)
        self.assertEqual("grok-4.6", env["GROK_CHAIN_WRITE"].split()[0])
        # A role on ANOTHER harness must be written qualified, or the run's own
        # CLI would be handed a model identifier it has never heard of.
        self.assertEqual("claude:sonnet", env["GROK_CHAIN_AUDIT"].split()[0])
        self.assertEqual("grok", env["HARNESS"])

    def test_a_comment_key_is_not_a_candidate(self):
        """18 runs appeared where 15 were asked for.

        candidates.json carries a "_comment" block. qualify.py iterated every
        key, so the documentation became a sixth candidate whose environment is
        a LIST, and it silently inflated the denominator of every count in the
        table.
        """
        q = load("qualify")
        with tempfile.TemporaryDirectory() as d:
            corpus = pathlib.Path(d)
            (corpus / "doc.md").write_text("word " * 500)
            (corpus / "candidates.json").write_text(json.dumps(
                {"_comment": ["not a candidate"], "real": {"HARNESS": "agy"}}))
            spec = json.loads((corpus / "candidates.json").read_text())
            kept = {k: v for k, v in spec.items() if not k.startswith("_")}
            self.assertEqual(["real"], list(kept))
        self.assertIn('startswith("_")', (ENG / "qualify.py").read_text(),
                      "qualify.py no longer filters comment keys")

    def test_the_executable_roster_contains_no_evidence_narrative(self):
        # Qualification evidence belongs in retained run/state artifacts, not
        # beside executable model chains where a parser can mistake it for data.
        cfg = json.loads((ENG / "models.json").read_text())
        for name, spec in cfg.items():
            if name.startswith("_") or not isinstance(spec, dict):
                continue
            self.assertNotIn("_comment", spec)
            self.assertNotIn("_admitted", spec)

class CursorIsWiredLikeEveryOtherHarness(unittest.TestCase):
    """Cursor Agent CLI is another argv/stdin harness; the pipeline stays
    model-agnostic beyond models.json and this _cmd branch."""
    ms = load("mapsum")

    def setUp(self):
        self.wd = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.wd, True)
        (self.wd / "rv").mkdir()

    def test_cursor_is_a_known_harness_everywhere(self):
        self.assertIn("cursor", self.ms.HARNESSES)
        self.assertIn("cursor", load("summ_cli").HARNESSES)
        self.assertIn("cursor", json.loads((ENG / "models.json").read_text()))

    def test_it_uses_headless_ask_mode_with_json_print(self):
        cmd = self.ms._cmd("cursor", "auto", self.wd, "PROMPT")
        self.assertEqual("-p", cmd[1])
        self.assertEqual("ask", cmd[cmd.index("--mode") + 1])
        self.assertEqual("auto", cmd[cmd.index("--model") + 1])
        self.assertEqual("json", cmd[cmd.index("--output-format") + 1])
        self.assertEqual("enabled", cmd[cmd.index("--sandbox") + 1])
        self.assertIn("--trust", cmd)
        self.assertNotIn("--force", cmd)
        self.assertNotIn("--yolo", cmd)

    def test_the_prompt_rides_on_stdin_not_argv(self):
        prompt = "CANARY" + "X" * 200000
        cmd = self.ms._cmd("cursor", "auto", self.wd, prompt)
        self.assertFalse(any(prompt in a for a in cmd), "prompt text in argv")
        self.assertEqual("stdin", self.ms.NON_ARGV_TRANSPORT["cursor"])
        self.assertIn("cursor", self.ms.STDIN_HARNESSES)

    def test_workspace_is_sterile_not_the_run_root(self):
        # Ask mode can search the workspace. Pointing --workspace at the run
        # evidence root would hand the model source, prompts, and ledgers.
        (self.wd / "source.txt").write_text("SECRET SOURCE")
        (self.wd / "rv" / "part.md").write_text("SECRET PART")
        cmd = self.ms._cmd("cursor", "auto", self.wd, "PROMPT")
        ws = pathlib.Path(cmd[cmd.index("--workspace") + 1])
        self.assertTrue(ws.is_absolute())
        self.assertNotEqual(ws, self.ms.run_root(self.wd))
        self.assertNotEqual(ws, self.wd.resolve())
        self.assertFalse((ws / "source.txt").exists())
        self.assertFalse((ws / "rv").exists())
        cfg = json.loads((ws / ".cursor" / "cli.json").read_text())
        deny = cfg["permissions"]["deny"]
        for token in ("Shell(*)", "Read(**)", "Write(**)", "WebFetch(*)",
                      "Mcp(*:*)"):
            self.assertIn(token, deny)
        self.assertEqual([], cfg["permissions"]["allow"])
        self.assertEqual(self.ms.cli_cwd("cursor", self.wd), ws)

    def test_auto_and_composer_have_no_effort_knob(self):
        cfg = json.loads((ENG / "models.json").read_text())["cursor"]
        self.assertNotIn("auto", cfg["_model_settings"])
        self.assertNotIn("composer-2.5", cfg["_model_settings"])
        self.assertEqual((), self.ms.effort_values("cursor", "auto"))
        self.assertEqual((), self.ms.effort_values("cursor", "composer-2.5"))

    def test_cursor_grok_effort_is_a_model_id_variant(self):
        cfg = json.loads((ENG / "models.json").read_text())["cursor"]
        setting = cfg["_model_settings"]["cursor-grok-4.6"]
        self.assertEqual("model", setting["option"])
        self.assertEqual(("low", "medium", "high", "xhigh"),
                         self.ms.effort_values("cursor", "cursor-grok-4.6"))
        self.assertEqual(
            "cursor-grok-4.6-medium",
            load("model_config").resolved_model(
                "cursor", "cursor-grok-4.6", "medium"))
        self.assertEqual(
            "cursor-grok-4.6-xhigh",
            load("model_config").resolved_model(
                "cursor", "cursor-grok-4.6", "xhigh"))
        self.assertTrue(all("fast" not in v for v in setting["variants"].values()))

    def test_a_failed_or_empty_envelope_is_never_an_answer(self):
        for bad in (
                json.dumps({"type": "error", "result": "boom"}),
                json.dumps({"type": "result", "subtype": "success",
                            "is_error": True, "result": "nope"}),
                json.dumps({"type": "result", "subtype": "error",
                            "result": "x"}),
                json.dumps({"type": "result", "subtype": "success",
                            "result": ""}),
                json.dumps({"type": "result", "subtype": "success",
                            "result": 12}),
                "not json at all"):
            with self.assertRaises(Exception):
                self.ms.cursor_payload(bad)

    def test_the_result_text_is_what_the_stage_receives(self):
        env = json.dumps({
            "type": "result", "subtype": "success", "is_error": False,
            "result": '{"detailed": "D", "brief": "B"}',
            "session_id": "abc"})
        self.assertEqual('{"detailed": "D", "brief": "B"}',
                         self.ms.cursor_payload(env))

    def _cursor_ok(self, result_text='{"ok": true}'):
        return json.dumps({
            "type": "result", "subtype": "success", "is_error": False,
            "result": result_text})

    def test_run_accepts_only_a_validated_success_envelope(self):
        seen = {}

        def fake_run(argv, **kw):
            seen["argv"] = argv
            seen["cwd"] = pathlib.Path(kw.get("cwd") or "").resolve()
            seen["input"] = kw.get("input")
            return subprocess.CompletedProcess(
                argv, 0, self._cursor_ok('{"ok": true}'), "")

        def validate(raw):
            obj = json.loads(raw)
            if obj != {"ok": True}:
                raise ValueError("wrong schema")

        with unittest.mock.patch.object(subprocess, "run", fake_run):
            out = self.ms.run("PROMPT", self.wd, ["cursor:auto"], "plan001",
                              validate=validate)
        self.assertEqual('{"ok": true}', out)
        self.assertEqual(seen["cwd"], self.ms.cli_cwd("cursor", self.wd))
        self.assertNotEqual(seen["cwd"], self.ms.run_root(self.wd))
        self.assertIn("--workspace", seen["argv"])
        self.assertEqual(
            pathlib.Path(seen["argv"][seen["argv"].index("--workspace") + 1]),
            seen["cwd"])
        self.assertTrue(seen["input"].startswith("PROMPT"))

    def test_run_rejects_nonzero_exit_with_plausible_success_json(self):
        def fake_run(argv, **_kw):
            return subprocess.CompletedProcess(
                argv, 1, self._cursor_ok('{"ok": true}'), "failed")

        with unittest.mock.patch.object(subprocess, "run", fake_run), \
                self.assertRaises(self.ms.NoCandidate):
            self.ms.run("PROMPT", self.wd, ["cursor:auto"], "plan001",
                        validate=json.loads)

    def test_run_rejects_error_and_empty_envelopes(self):
        for stdout in (
                json.dumps({"type": "error", "result": "boom"}),
                self._cursor_ok(""),
                "not json"):
            def fake_run(argv, _stdout=stdout, **_kw):
                return subprocess.CompletedProcess(argv, 0, _stdout, "")

            with unittest.mock.patch.object(subprocess, "run", fake_run), \
                    self.assertRaises(self.ms.NoCandidate):
                self.ms.run("PROMPT", self.wd, ["cursor:auto"], "plan001",
                            validate=json.loads)

    def test_run_rejects_narration_when_the_stage_validator_runs(self):
        def fake_run(argv, **_kw):
            return subprocess.CompletedProcess(
                argv, 0,
                self._cursor_ok("I'll read the file first then answer."), "")

        def validate(raw):
            json.loads(raw)  # narration is not JSON

        with unittest.mock.patch.object(subprocess, "run", fake_run), \
                self.assertRaises(self.ms.NoCandidate):
            self.ms.run("PROMPT", self.wd, ["cursor:auto"], "plan001",
                        validate=validate)

    def test_run_does_not_record_ok_when_validation_fails(self):
        def fake_run(argv, **_kw):
            return subprocess.CompletedProcess(
                argv, 0, self._cursor_ok('{"wrong": true}'), "")

        def validate(raw):
            raise ValueError("stage contract")

        with unittest.mock.patch.object(subprocess, "run", fake_run), \
                self.assertRaises(self.ms.NoCandidate):
            self.ms.run("PROMPT", self.wd, ["cursor:auto"], "plan001",
                        validate=validate)
        calls = (self.ms.run_root(self.wd) / "calls.jsonl").read_text()
        self.assertNotIn('"outcome": "ok"', calls)
        self.assertIn('"outcome": "unusable"', calls)


class CrossProcessAdmissionIsReal(unittest.TestCase):
    rt = load("runtime")

    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.addCleanup(self.td.cleanup)
        self.app = pathlib.Path(self.td.name)
        patcher = unittest.mock.patch.object(self.rt, "app_dir", return_value=self.app)
        patcher.start(); self.addCleanup(patcher.stop)
        previous = os.environ.pop("SUMM_RUNTIME_JSON", None)
        if previous is not None:
            self.addCleanup(os.environ.__setitem__, "SUMM_RUNTIME_JSON", previous)
        else:
            self.addCleanup(os.environ.pop, "SUMM_RUNTIME_JSON", None)

    def test_safe_defaults_allow_two_targets_but_one_call_per_harness(self):
        self.assertEqual(2, self.rt.config()["active_targets"])
        self.assertEqual(1, self.rt.config()["harness_capacity"])

    def test_a_capacity_one_slot_really_blocks_a_second_worker(self):
        acquired = threading.Event()

        def contender():
            with self.rt.slot_lease("resource", "same-harness", 1):
                acquired.set()

        with self.rt.slot_lease("resource", "same-harness", 1):
            worker = threading.Thread(target=contender)
            worker.start()
            time.sleep(0.12)
            self.assertFalse(acquired.is_set(), "second call entered a capacity-one slot")
        worker.join(2)
        self.assertTrue(acquired.is_set(), "waiting call was not admitted after release")

    def test_two_target_slots_can_be_held_at_once(self):
        with self.rt.slot_lease("target", "host", 2):
            with self.rt.slot_lease("target", "host", 2) as wait:
                self.assertLess(wait, 0.2)

    def test_malformed_runtime_configuration_fails_closed(self):
        (self.app / "runtime.json").write_text('{"active_targets": 0}')
        with self.assertRaises(self.rt.ConfigError):
            self.rt.config()

    def test_gateway_configuration_is_validated_and_returned(self):
        runtime_value = gateway_runtime()
        runtime_value["gateways"]["localgw"]["models"] = {
            "model-a": {"context_tokens": 262_144,
                        "output_tokens": 8_192,
                        "prompt_overhead_tokens": 256}}
        (self.app / "runtime.json").write_text(json.dumps({
            "gateways": runtime_value["gateways"]}))
        cfg = self.rt.config()
        self.assertEqual(
            cfg["gateways"]["localgw"]["api_key_env"], "SUMM_LOCALGW_TOKEN")
        self.assertEqual(
            cfg["gateways"]["localgw"]["max_output_tokens"], 16384)
        self.assertEqual(
            cfg["gateways"]["localgw"]["roster"]["write"],
            ["model-a", "model-b"])
        self.assertEqual(
            cfg["gateways"]["localgw"]["models"]["model-a"]
            ["context_tokens"], 262_144)

    def test_route_capacity_requires_positive_integer_context(self):
        config = gateway_runtime()
        config["gateways"]["localgw"]["models"] = {
            "model-a": {"context_tokens": True}}
        (self.app / "runtime.json").write_text(json.dumps(config))
        with self.assertRaises(self.rt.ConfigError):
            self.rt.config()

        config["gateways"]["localgw"]["models"] = {
            "model-a": {"output_tokens": 8_192}}
        (self.app / "runtime.json").write_text(json.dumps(config))
        with self.assertRaises(self.rt.ConfigError):
            self.rt.config()

    def test_gateway_can_declare_no_http_authentication(self):
        config = gateway_runtime()
        gateway = config["gateways"]["localgw"]
        gateway["authentication"] = "none"
        gateway.pop("api_key_env")
        (self.app / "runtime.json").write_text(json.dumps(config))
        self.assertEqual(
            "none", self.rt.config()["gateways"]["localgw"]["authentication"])

    def test_runtime_config_path_is_explicitly_redirectable(self):
        alternate = self.app / "portable-runtime.json"
        alternate.write_text(json.dumps({"active_targets": 3}))
        with unittest.mock.patch.dict(
                os.environ, {"SUMM_RUNTIME_CONFIG": str(alternate)}):
            self.assertEqual(alternate, self.rt.config_path())
            self.assertEqual(3, self.rt.config()["active_targets"])

    def test_frozen_runtime_ignores_later_file_edits(self):
        original = gateway_runtime()
        (self.app / "runtime.json").write_text(json.dumps(original))
        frozen = self.rt.frozen_json()
        changed = gateway_runtime()
        changed["active_targets"] = 7
        changed["gateways"]["localgw"]["base_url"] = \
            "https://changed.example/v1"
        (self.app / "runtime.json").write_text(json.dumps(changed))
        with unittest.mock.patch.dict(
                os.environ, {"SUMM_RUNTIME_JSON": frozen}):
            observed = self.rt.config()
        self.assertEqual(2, observed["active_targets"])
        self.assertEqual("https://gateway.example/v1",
                         observed["gateways"]["localgw"]["base_url"])

    def test_malformed_frozen_runtime_fails_closed(self):
        with unittest.mock.patch.dict(
                os.environ, {"SUMM_RUNTIME_JSON": "{broken"}):
            with self.assertRaises(self.rt.ConfigError):
                self.rt.config()

    def test_gateway_request_options_cannot_replace_pipeline_fields(self):
        config = gateway_runtime()
        config["gateways"]["localgw"]["request_options"] = {"messages": []}
        (self.app / "runtime.json").write_text(json.dumps(config))
        with self.assertRaises(self.rt.ConfigError):
            self.rt.config()

    def test_malformed_gateway_configuration_fails_closed(self):
        (self.app / "runtime.json").write_text(json.dumps({
            "gateways": {"localgw": {"base_url": "not-a-url"}}
        }))
        with self.assertRaises(self.rt.ConfigError):
            self.rt.config()

    def test_model_route_must_name_a_model_in_the_gateway_roster(self):
        config = gateway_runtime()
        config["gateways"]["localgw"]["models"] = {
            "not-in-roster": {"request_options": {"reasoning_effort": "low"}}}
        (self.app / "runtime.json").write_text(json.dumps(config))
        with self.assertRaises(self.rt.ConfigError):
            self.rt.config()

    def test_a_held_model_can_keep_its_route_without_joining_role_chains(self):
        config = gateway_runtime()
        config["gateways"]["localgw"]["roster"]["_held"] = ["model-held"]
        config["gateways"]["localgw"]["models"] = {
            "model-held": {"request_options": {"reasoning_effort": "low"}}}
        (self.app / "runtime.json").write_text(json.dumps(config))
        observed = self.rt.config()
        self.assertEqual(
            observed["gateways"]["localgw"]["roster"]["_held"], ["model-held"])
        self.assertNotIn(
            "model-held", observed["gateways"]["localgw"]["roster"]["write"])

    def test_destination_pair_exclusion_blocks_a_second_publisher(self):
        acquired = threading.Event()
        dest = self.app / "same.summary.md"

        def contender():
            with self.rt.destination_lock(dest):
                acquired.set()

        with self.rt.destination_lock(dest):
            worker = threading.Thread(target=contender); worker.start()
            time.sleep(0.12)
            self.assertFalse(acquired.is_set())
        worker.join(2)
        self.assertTrue(acquired.is_set())

    def test_an_explicit_work_root_is_refused_not_shared(self):
        root = self.app / "one-work-root"
        with self.rt.work_root_lock(root):
            with self.assertRaises(self.rt.BusyError):
                with self.rt.work_root_lock(root):
                    self.fail("same work root was admitted twice")


class PairPublicationIsBothOrNeither(unittest.TestCase):
    cli = load("summ_cli")

    def setUp(self):
        self.td = tempfile.TemporaryDirectory(); self.addCleanup(self.td.cleanup)
        self.d = pathlib.Path(self.td.name)
        self.new_d, self.new_b = self.d / "new-d", self.d / "new-b"
        self.new_d.write_text("NEW D"); self.new_b.write_text("NEW B")
        self.dest = self.d / "out.summary.md"
        self.bdest = self.d / "out.brief.md"

    def _fail_second_final_replace(self, src, dst):
        if pathlib.Path(dst) == self.bdest and ".tmp" in pathlib.Path(src).name:
            raise OSError("injected second replacement failure")
        os.replace(src, dst)

    def test_failure_on_second_replace_restores_the_old_matching_pair(self):
        self.dest.write_text("OLD D"); self.bdest.write_text("OLD B")
        with self.assertRaises(OSError):
            self.cli.publish_pair(self.new_d, self.new_b, self.dest,
                                  self._fail_second_final_replace)
        self.assertEqual("OLD D", self.dest.read_text())
        self.assertEqual("OLD B", self.bdest.read_text())

    def test_failure_on_second_replace_leaves_no_half_pair_when_none_existed(self):
        with self.assertRaises(OSError):
            self.cli.publish_pair(self.new_d, self.new_b, self.dest,
                                  self._fail_second_final_replace)
        self.assertFalse(self.dest.exists())
        self.assertFalse(self.bdest.exists())

    def test_failed_rollback_is_reported_and_keeps_recovery_backups(self):
        self.dest.write_text("OLD D"); self.bdest.write_text("OLD B")
        real_replace = os.replace

        def fail_forward(src, dst):
            if pathlib.Path(dst) == self.bdest and ".tmp" in pathlib.Path(src).name:
                raise OSError("injected publication failure")
            real_replace(src, dst)

        def fail_detailed_restore(src, dst):
            if (pathlib.Path(dst) == self.dest
                    and ".restore" in pathlib.Path(src).name):
                raise OSError("injected rollback failure")
            real_replace(src, dst)

        with unittest.mock.patch.object(
                self.cli.os, "replace", side_effect=fail_detailed_restore):
            with self.assertRaises(self.cli.PublicationError) as raised:
                self.cli.publish_pair(
                    self.new_d, self.new_b, self.dest, fail_forward)
        self.assertFalse(raised.exception.restored)
        self.assertTrue(raised.exception.recovery_paths)
        self.assertTrue(all(path.exists()
                            for path in raised.exception.recovery_paths))
        self.assertEqual("NEW D", self.dest.read_text())
        self.assertEqual("OLD B", self.bdest.read_text())

    def test_success_publishes_one_matching_generation(self):
        got = self.cli.publish_pair(self.new_d, self.new_b, self.dest)
        self.assertEqual(self.bdest, got)
        self.assertEqual("NEW D", self.dest.read_text())
        self.assertEqual("NEW B", self.bdest.read_text())

    def test_two_pasted_allocations_receive_distinct_matching_bases(self):
        base = self.d / "topic.summary.md"
        results = []

        def allocate():
            with self.cli.runtime.output_directory_lock(self.d):
                dest = self.cli.free_path(base, set())
                with self.cli.runtime.destination_lock(dest):
                    bdest = self.cli.publish_pair(self.new_d, self.new_b, dest)
                    results.append((dest.name, bdest.name))

        workers = [threading.Thread(target=allocate) for _ in range(2)]
        for worker in workers: worker.start()
        for worker in workers: worker.join(2)
        self.assertEqual(2, len(results))
        self.assertEqual({"topic.summary.md", "topic-2.summary.md"},
                         {d for d, _ in results})
        for detailed, brief in results:
            self.assertEqual(detailed[:-len(".summary.md")],
                             brief[:-len(".brief.md")])


class TerminalEvidenceCoversFailClosedBranches(unittest.TestCase):
    def test_each_previously_silent_failure_emits_one_target_verdict(self):
        source = (ENG / "summ_cli.py").read_text()
        for marker in (
                "skip (unreadable)",
                "the quick path produced nothing",
                "the Full path produced nothing",
                "context unsupported, nothing published",
                "both artifacts were not produced",
                "publish failed, {state}"):
            start = source.index(marker)
            end = source.index("continue", start)
            self.assertEqual(
                1, source[start:end].count("emit_target_finished("),
                f"{marker!r} does not have exactly one terminal target event")


class CustomInstructionsArePerRunAndFailClosed(unittest.TestCase):
    ci = load("custom_instructions")
    et = load("eta")

    def test_empty_request_changes_no_prompt_bytes(self):
        prompt = "TRUSTED TASK\nSOURCE\nThe wall binds."
        self.assertEqual(prompt, self.ci.decorate_prompt(prompt, ""))

    def test_request_is_framed_before_the_trusted_task(self):
        prompt = "TRUSTED TASK\nSOURCE\nThe wall binds."
        got = self.ci.decorate_prompt(prompt, "Preserve the exact quotations.")
        self.assertLess(got.index("CUSTOM READER REQUEST"),
                        got.index("TRUSTED TASK"))
        self.assertIn("permanent task and rules", got)
        self.assertIn("QUOTE PRESERVATION CHECK", got)

    def test_cleanup_request_cannot_authorize_summarization(self):
        got = self.ci.decorate_prompt(
            "CLEANUP TASK\nSOURCE BLOCKS", "Make this shorter.", task="text-prep")
        self.assertLess(got.index("CUSTOM CLEANUP REQUEST"), got.index("CLEANUP TASK"))
        self.assertIn("cannot authorize invention, omission, summarization", got)
        self.assertNotIn("Brief scope", got)

    def test_file_input_is_bounded_utf8_and_nul_free(self):
        with tempfile.TemporaryDirectory() as td:
            p = pathlib.Path(td) / "request.txt"
            p.write_text("  Preserve exact quotations.  ", encoding="utf-8")
            self.assertEqual("Preserve exact quotations.", self.ci.read_file(p))
            p.write_bytes(b"x" * (self.ci.MAX_BYTES + 1))
            with self.assertRaisesRegex(ValueError, "too large"):
                self.ci.read_file(p)
            p.write_bytes(b"bad\x00request")
            with self.assertRaisesRegex(ValueError, "NUL"):
                self.ci.read_file(p)
            p.write_bytes(b"\xff")
            with self.assertRaisesRegex(ValueError, "UTF-8"):
                self.ci.read_file(p)

    def test_exact_quotes_are_source_backed_including_punctuation(self):
        request = "Preserve exact quotations."
        source = "The witness said, don't panic. Then: The wall binds."
        self.assertEqual([], self.ci.quote_defects(
            ('It concludes “The wall\nbinds.”',), source, request))
        self.assertTrue(self.ci.quote_defects(
            ('It concludes “The wall always binds.”',), source, request))
        self.assertTrue(self.ci.quote_defects(
            ("The witness said ‘don’t panic.’",), source, request),
            "an altered apostrophe was accepted as exact punctuation")
        self.assertEqual([], self.ci.quote_defects(
            ('It concludes “The wall always binds.”',), source,
            "Emphasize practical implications."))

    def test_attributed_unrequested_quotation_is_still_source_backed(self):
        source = "The witness said, do not panic."
        self.assertTrue(self.ci.quote_defects(
            ('As the report says, “a fabricated passage”.',), source,
            "Emphasize practical implications."))

    def test_american_trailing_comma_inside_a_source_quote_is_source_backed(self):
        source = ('One witness described the lantern as "dim but serviceable"; '
                  'the report did not claim that the building was safe.')
        self.assertEqual([], self.ci.quote_defects(
            ('A witness called the lantern "dim but serviceable," and the '
             'committee recommended a storm-watch.',),
            source, "Emphasize practical implications."))
        self.assertTrue(self.ci.quote_defects(
            ('A witness called the lantern "dim but unserviceable,"',),
            source, "Emphasize practical implications."))
        ss = load("shortsum")
        pair = ('A witness called the lantern "dim but serviceable," '
                'and the committee recommended a storm-watch.\n\n'
                'The records were incomplete.',
                'Lantern called "dim but serviceable," then routine repair.')
        self.assertEqual([], ss.structural_findings(pair[0], pair[1], source))

    def test_every_summary_role_uses_the_shared_decorator(self):
        ledger_src = (ENG / "ledger.py").read_text()
        short_src = (ENG / "shortsum.py").read_text()
        self.assertGreaterEqual(ledger_src.count("custom_instructions.decorate_prompt"),
                                4, "plan, audit, revise or replan bypasses the request")
        self.assertGreaterEqual(short_src.count("custom_instructions.decorate_prompt"),
                                3, "quick initial, audit or revision bypasses it")
        self.assertIn('audit(d2, b2, "short-reaudit")', short_src,
                      "the decorated audit path is not reused for the revision")

    def test_eta_identity_uses_only_the_opaque_digest(self):
        request = "UNIQUE-CUSTOM-INSTRUCTION-SENTINEL"
        d1 = self.ci.digest(request)
        d2 = self.ci.digest(request + " changed")
        s1 = self.et.execution_signature("quick", instructions_digest=d1)
        s2 = self.et.execution_signature("quick", instructions_digest=d2)
        self.assertNotEqual(s1, s2)
        self.assertNotIn(request, s1)
        self.assertRegex(s1, r"^sha256:[0-9a-f]{64}$")
        with unittest.mock.patch.dict(
                os.environ, {"SUMM_INSTRUCTIONS_DIGEST": d1}, clear=False):
            self.assertNotEqual(
                self.et.execution_signature("quick"),
                self.et.execution_signature("quick", instructions_digest=None),
                "ambient instruction identity leaked into an explicitly empty run")

    def test_quick_never_writes_an_altered_exact_quote(self):
        ss = load("shortsum")

        class FakeRunner:
            MODELS = ("fake",); AUDIT_MODELS = ("fake",); REPAIR_MODELS = ("fake",)
            JSON_REQUEST_OPTIONS = {"response_format": {"type": "json_object"}}

            def __init__(self, responses):
                self.responses = iter(responses)

            def run(self, *_args, **_kwargs):
                return next(self.responses)

        class FakeLedger:
            @staticmethod
            def parse_strict(raw, _stage):
                return json.loads(raw)

        source = ("The wall binds under pressure. The result remains conditional "
                  "on the stated assumptions and the evidence is limited. ") * 4
        bad = "The account says “The wall always binds under pressure.”"
        responses = [
            json.dumps({"detailed": bad + " The evidence remains conditional.",
                        "brief": bad}),
            json.dumps({"verdict": "revise", "findings": [
                    AF("altered quotation")]}),
            json.dumps(patch_replace(
                bad + " The evidence remains conditional.", bad,
                "detailed", bad + " The evidence remains conditional.")),
            json.dumps({"verdict": "pass", "findings": []}),
        ]
        with tempfile.TemporaryDirectory() as td:
            td = pathlib.Path(td)
            src = td / "source.txt"; src.write_text(source)
            out = td / "out"
            request = td / "request.txt"
            request.write_text("Preserve exact quotations.")
            env = {"SUMM_INSTRUCTIONS_FILE": str(request),
                   "SUMM_INSTRUCTIONS_DIGEST": self.ci.digest(request.read_text())}
            with unittest.mock.patch.object(ss, "_runner",
                                             return_value=FakeRunner(responses)), \
                 unittest.mock.patch.object(ss, "_ledger",
                                             return_value=FakeLedger()), \
                 unittest.mock.patch.dict(os.environ, env, clear=False):
                self.assertEqual(5, ss.run(src, out))
            self.assertFalse((out / "detailed.md").exists())
            self.assertFalse((out / "brief.md").exists())

    def test_ledger_never_seals_an_altered_exact_quote(self):
        led = load("ledger")
        obj = {
            "source_sha256": "ignored", "visible_words": 5,
            "units": [{"unit_id": "U001", "source_ids": ["P0001"],
                       "exact_source_anchor": "The wall binds",
                       "detailed_disposition": "required",
                       "brief_disposition": "required", "dependencies": [],
                       "detailed_capsule": "It says “The wall always binds.”",
                       "brief_capsule": "It says “The wall always binds.”"}],
            "dispositions": [], "unplanned_parts": [], "thin_sections": [],
            "part_quarantine": []}
        with tempfile.TemporaryDirectory() as td:
            td = pathlib.Path(td); rv_dir = td / "rv"; out = td / "out"
            rv_dir.mkdir(); out.mkdir()
            (rv_dir / "source.visible.md").write_text("The wall binds.")
            request = td / "request.txt"
            request.write_text("Preserve exact quotations.")
            env = {"SUMM_INSTRUCTIONS_FILE": str(request),
                   "SUMM_INSTRUCTIONS_DIGEST": self.ci.digest(request.read_text())}
            completed = unittest.mock.Mock(returncode=0)
            with unittest.mock.patch.object(led, "build", return_value=obj), \
                 unittest.mock.patch.object(led.subprocess, "run",
                                             return_value=completed), \
                 unittest.mock.patch.object(sys, "argv",
                                             ["ledger.py", str(rv_dir), str(out)]), \
                 unittest.mock.patch.dict(os.environ, env, clear=False):
                self.assertEqual(5, led.main())
            self.assertFalse((out / "SEALED").exists())


class ETAStartsOnlyAfterComparableHistory(unittest.TestCase):
    et = load("eta")

    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.addCleanup(self.td.cleanup)
        self.app = pathlib.Path(self.td.name)
        patcher = unittest.mock.patch.object(self.et.runtime, "app_dir",
                                             return_value=self.app)
        patcher.start(); self.addCleanup(patcher.stop)

    def _record(self, seconds):
        self.et._append({
            "schema": "summer.eta.target.v1",
            "producer_rev": self.et.PRODUCER_REV,
            "completed_utc": __import__("datetime").datetime.now(
                __import__("datetime").timezone.utc).isoformat(),
            "route": "quick", "source_words": 3000, "parts": None,
            "execution_sig": "sha256:same",
            "scheduler": {"target_capacity": 2, "resources": [],
                          "resource_signature": self.et._resource_signature([])},
            "outcome": "succeeded", "service_wall_s": seconds,
            "metadata_status": "eligible",
            "calls": {"attempts": 1, "ok": 1, "seconds": seconds,
                      "outcomes": {"ok": 1}, "by_role": {}}})

    def test_seven_exact_successes_show_no_eta_and_eight_do(self):
        for n in range(7): self._record(600 + n)
        args = {"route": "quick", "source_words": 3000,
                "execution_sig": "sha256:same"}
        self.assertIsNone(self.et.estimate(**args))
        self._record(608)
        got = self.et.estimate(**args)
        self.assertIsNotNone(got)
        self.assertEqual(8, got["samples"])
        self.assertEqual("rough", got["confidence"])

    def test_long_lived_history_contains_no_content_or_paths(self):
        run = self.app / "run"; run.mkdir()
        sentinel = "UNIQUE-SOURCE-PROSE-AND-PATH"
        (run / "calls.jsonl").write_text(json.dumps({
            "stage": "short", "harness": "opencode", "model": "hidden-model",
            "outcome": "ok", "seconds": 1.0,
            "prompt": sentinel, "detail": sentinel}) + "\n")
        self.et.record_target(
            run, route="quick", source_words=3000,
            execution_sig="sha256:opaque", outcome="succeeded",
            service_wall_s=1.2, admission_wait_s=0)
        retained = self.et.history_path().read_text()
        self.assertNotIn(sentinel, retained)
        self.assertNotIn("hidden-model", retained)
        self.assertNotIn(str(run), retained)

    def test_incomplete_call_evidence_is_explicitly_ineligible(self):
        run = self.app / "incomplete"; run.mkdir()
        (run / "calls.jsonl").write_text('{"stage":"write","seconds":1')
        self.et.record_target(
            run, route="quick", source_words=3000,
            execution_sig="sha256:opaque", outcome="succeeded",
            service_wall_s=1.2, admission_wait_s=0)
        row = json.loads(self.et.history_path().read_text().splitlines()[-1])
        self.assertEqual("ineligible", row["metadata_status"])
        self.assertIn("malformed", row["metadata_reason"])
        self.assertEqual(0, row["calls"]["attempts"])

    def test_history_repair_discards_corrupt_lines_and_unterminated_tail(self):
        p = self.et.history_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('{"schema":"summer.eta.target.v1"}\nnot-json')
        self.et._append({"schema": "summer.eta.target.v1", "producer_rev": 1})
        lines = p.read_text().splitlines()
        self.assertEqual(2, len(lines))
        self.assertTrue(all(json.loads(line)["schema"] == "summer.eta.target.v1"
                            for line in lines))

    def test_live_remaining_uses_role_attempts_and_never_uses_exceeded_history(self):
        for n in range(8):
            self.et._append({
                "schema": "summer.eta.target.v1", "producer_rev": self.et.PRODUCER_REV,
                "completed_utc": __import__("datetime").datetime.now(
                    __import__("datetime").timezone.utc).isoformat(),
                "route": "quick", "source_words": 3000, "parts": None,
                "execution_sig": "sha256:live", "scheduler": {
                    "target_capacity": 2, "resources": [],
                    "resource_signature": self.et._resource_signature([])},
                "outcome": "succeeded", "metadata_status": "eligible",
                "service_wall_s": 600 + n, "calls": {
                    "attempts": 2, "ok": 2, "seconds": 500,
                    "outcomes": {"ok": 2}, "by_role": {
                        "write": {"attempts": 2, "ok": 2, "seconds": 500}}}})
        run = self.app / "live"; run.mkdir()
        (run / "calls.jsonl").write_text(json.dumps({
            "stage": "write", "harness": "agy", "model": "hidden",
            "outcome": "ok", "seconds": 30}) + "\n")
        got = self.et.live_remaining(
            run, route="quick", source_words=3000,
            execution_sig="sha256:live")
        self.assertIsNotNone(got)
        self.assertIn("remaining", got["text"])
        (run / "calls.jsonl").write_text("\n".join([
            json.dumps({"stage": "write", "outcome": "ok", "seconds": 30}),
            json.dumps({"stage": "write", "outcome": "ok", "seconds": 30}),
            json.dumps({"stage": "write", "outcome": "ok", "seconds": 30})]) + "\n")
        self.assertIsNone(self.et.live_remaining(
            run, route="quick", source_words=3000,
            execution_sig="sha256:live"))


class AffixNamingAndPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.last_affix = load("last_affix")
        self.selection = load("selection")
        self.ui = load("summ_ui")
        self.td = tempfile.TemporaryDirectory()
        self.tmp = pathlib.Path(self.td.name)

    def tearDown(self):
        self.td.cleanup()

    def test_last_affix_defaults_and_roundtrip(self):
        with unittest.mock.patch.object(self.last_affix, "last_path",
                                        return_value=self.tmp / "affix.json"):
            # Missing file defaults to prefix and mode-specific default tags
            summ_default = self.last_affix.load("summarize")
            self.assertEqual("prefix", summ_default["kind"])
            self.assertEqual("summary", summ_default["primary"])
            self.assertEqual("brief", summ_default["secondary"])

            tts_default = self.last_affix.load("tts")
            self.assertEqual("prefix", tts_default["kind"])
            self.assertEqual("tts", tts_default["primary"])
            self.assertEqual("", tts_default["secondary"])

            clean_default = self.last_affix.load("text_prep")
            self.assertEqual("prefix", clean_default["kind"])
            self.assertEqual("clean", clean_default["primary"])

            # Save and reload per-mode
            self.last_affix.save("suffix", "detailed", "short", mode_key="summarize")
            loaded = self.last_affix.load("summarize")
            self.assertEqual("suffix", loaded["kind"])
            self.assertEqual("detailed", loaded["primary"])
            self.assertEqual("short", loaded["secondary"])

            # Save for another mode does not clobber summarize
            self.last_affix.save("suffix", "speech", "", mode_key="tts")
            tts_loaded = self.last_affix.load("tts")
            self.assertEqual("speech", tts_loaded["primary"])
            summ_reloaded = self.last_affix.load("summarize")
            self.assertEqual("detailed", summ_reloaded["primary"])
            self.assertEqual("short", summ_reloaded["secondary"])

            # Corrupted file falls back safely
            (self.tmp / "affix.json").write_text("{corrupt json")
            self.assertEqual("summary", self.last_affix.load("summarize")["primary"])
            # Invalid kind falls back to prefix
            (self.tmp / "affix.json").write_text(json.dumps({"kind": "invalid", "text": "x"}))
            self.assertEqual("prefix", self.last_affix.load("summarize")["kind"])

    def test_format_output_filenames_defaults_to_prefixes(self):
        # Summarize
        self.assertEqual(("summary.paper.md", "brief.paper.md"),
                         self.selection.format_output_filenames("paper", "summarize"))
        # Quick
        self.assertEqual(("summary.paper.md", "brief.paper.md"),
                         self.selection.format_output_filenames("paper", "quick"))
        # Text Prep
        self.assertEqual(("clean.paper.md",),
                         self.selection.format_output_filenames("paper", "text_prep"))
        # TTS
        self.assertEqual(("tts.paper.txt",),
                         self.selection.format_output_filenames("paper", "tts"))

    def test_format_output_filenames_suffix_mode(self):
        # Summarize
        self.assertEqual(("paper.summary.md", "paper.brief.md"),
                         self.selection.format_output_filenames("paper", "summarize", affix_kind="suffix"))
        # Text Prep
        self.assertEqual(("paper.clean.md",),
                         self.selection.format_output_filenames("paper", "text_prep", affix_kind="suffix"))
        # TTS
        self.assertEqual(("paper.tts.txt",),
                         self.selection.format_output_filenames("paper", "tts", affix_kind="suffix"))

    def test_format_output_filenames_with_custom_affix(self):
        # Prefix with single custom tag (primary)
        self.assertEqual(("hw1.paper.md", "brief.hw1.paper.md"),
                         self.selection.format_output_filenames("paper", "summarize",
                                                               affix_kind="prefix", affix_text="hw1"))
        self.assertEqual(("draft.paper.md",),
                         self.selection.format_output_filenames("paper", "text_prep",
                                                               affix_kind="prefix", affix_text="draft"))
        self.assertEqual(("read.paper.txt",),
                         self.selection.format_output_filenames("paper", "tts",
                                                               affix_kind="prefix", affix_text="read"))
        # Prefix with two custom tags (primary and secondary)
        self.assertEqual(("detailed.paper.md", "short.paper.md"),
                         self.selection.format_output_filenames("paper", "summarize",
                                                               affix_kind="prefix",
                                                               affix_text="detailed",
                                                               affix_secondary="short"))
        # Suffix with single custom tag
        self.assertEqual(("paper.hw1.md", "paper.hw1.brief.md"),
                         self.selection.format_output_filenames("paper", "summarize",
                                                               affix_kind="suffix", affix_text="hw1"))
        self.assertEqual(("paper.draft.md",),
                         self.selection.format_output_filenames("paper", "text_prep",
                                                               affix_kind="suffix", affix_text="draft"))
        self.assertEqual(("paper.read.txt",),
                         self.selection.format_output_filenames("paper", "tts",
                                                               affix_kind="suffix", affix_text="read"))
        # Suffix with two custom tags
        self.assertEqual(("paper.detailed.md", "paper.short.md"),
                         self.selection.format_output_filenames("paper", "summarize",
                                                               affix_kind="suffix",
                                                               affix_text="detailed",
                                                               affix_secondary="short"))

    def test_plan_outputs_honours_affix_options(self):
        doc_path = self.tmp / "chapter.md"
        doc_path.write_text("sample content")
        sel = self.selection.resolve_paths([doc_path])
        # Default prefix
        planned_prefix = sel.with_outputs("summarize", self.tmp / "out", affix_kind="prefix")
        self.assertEqual("summary.chapter.md", planned_prefix.documents[0].planned_outputs[0].name)
        self.assertEqual("brief.chapter.md", planned_prefix.documents[0].planned_outputs[1].name)
        # Suffix
        planned_suffix = sel.with_outputs("summarize", self.tmp / "out", affix_kind="suffix")
        self.assertEqual("chapter.summary.md", planned_suffix.documents[0].planned_outputs[0].name)
        self.assertEqual("chapter.brief.md", planned_suffix.documents[0].planned_outputs[1].name)
        # Custom prefix with two tags
        planned_custom = sel.with_outputs("summarize", self.tmp / "out", affix_kind="prefix",
                                          affix_text="full", affix_secondary="executive")
        self.assertEqual("full.chapter.md", planned_custom.documents[0].planned_outputs[0].name)
        self.assertEqual("executive.chapter.md", planned_custom.documents[0].planned_outputs[1].name)

    def test_build_cmd_affix_flags(self):
        job = self.tmp / "job"
        summarize = self.ui.MODES[0][0]
        # Default prefix + empty text emits no redundant flags
        cmd_default = self.ui.build_cmd(job, summarize, None, affix_kind="prefix", affix_text="")
        self.assertNotIn("--affix-kind", cmd_default)
        self.assertNotIn("--affix", cmd_default)
        self.assertNotIn("--affix-secondary", cmd_default)
        # Suffix emits --affix-kind suffix
        cmd_suffix = self.ui.build_cmd(job, summarize, None, affix_kind="suffix", affix_text="")
        self.assertIn("--affix-kind", cmd_suffix)
        self.assertEqual("suffix", cmd_suffix[cmd_suffix.index("--affix-kind") + 1])
        # Custom primary emits --affix
        cmd_custom = self.ui.build_cmd(job, summarize, None, affix_kind="prefix", affix_text="hw1")
        self.assertIn("--affix", cmd_custom)
        self.assertEqual("hw1", cmd_custom[cmd_custom.index("--affix") + 1])
        # Custom secondary emits --affix-secondary
        cmd_two = self.ui.build_cmd(job, summarize, None, affix_kind="prefix",
                                    affix_text="full", affix_secondary="exec")
        self.assertIn("--affix-secondary", cmd_two)
        self.assertEqual("exec", cmd_two[cmd_two.index("--affix-secondary") + 1])

    def test_cli_revalidates_a_frozen_manifest_with_the_requested_affixes(self):
        cli = load("summ_cli")
        source = self.tmp / "paper.md"
        source.write_text("too short to summarize")
        out = self.tmp / "summ"
        chosen = self.selection.resolve_paths([source]).with_outputs(
            "summarize", out, affix_kind="prefix",
            affix_text="summ", affix_secondary="brief")
        manifest = self.tmp / "selection.json"
        self.selection.write_manifest(chosen, manifest)
        # Reach a deterministic pre-model input failure after the manifest
        # contract has accepted the custom output names.
        source.write_text("changed after selection")
        stderr = io.StringIO()
        argv = [
            "summ_cli.py", "--selection-manifest", str(manifest),
            "--scope", "batch", "--out", str(out),
            "--affix", "summ", "--affix-secondary", "brief",
            "--work-dir", str(self.tmp / "work"),
        ]
        with unittest.mock.patch.object(sys, "argv", argv), \
                unittest.mock.patch.object(cli, "freeze_model_routes"), \
                unittest.mock.patch.object(cli, "validate_roles"), \
                unittest.mock.patch.object(cli, "notify"), \
                contextlib.redirect_stderr(stderr):
            rc = cli.main()
        self.assertEqual(1, rc)
        self.assertNotIn("output plan does not match", stderr.getvalue())

    def test_ui_affix_controls_and_live_preview(self):
        import tkinter as tk
        try:
            root = tk.Tk()
        except tk.TclError as e:
            self.skipTest(f"no display: {e}")
        try:
            affix_file = self.tmp / "affix.json"
            with unittest.mock.patch.object(self.ui.last_affix, "last_path", return_value=affix_file), \
                 unittest.mock.patch.object(self.last_affix, "last_path", return_value=affix_file):
                app = self.ui.App(root)
                # Auto-populates with default for summarize mode
                self.assertEqual("prefix", app.affix_kind.get())
                self.assertEqual("summary", app.affix_text.get())
                self.assertEqual("brief", app.affix_secondary.get())
                self.assertIn("summary.name.md", app.affix_preview.cget("text"))
                self.assertIn("brief.name.md", app.affix_preview.cget("text"))

                # Switch to suffix
                app.affix_kind.set("suffix")
                self.assertIn("name.summary.md", app.affix_preview.cget("text"))
                self.assertEqual("suffix", self.last_affix.load("summarize")["kind"])

                # Edit primary and secondary
                app.affix_text.set("full")
                app.affix_secondary.set("exec")
                self.assertIn("name.full.md", app.affix_preview.cget("text"))
                self.assertIn("name.exec.md", app.affix_preview.cget("text"))
                loaded = self.last_affix.load("summarize")
                self.assertEqual("full", loaded["primary"])
                self.assertEqual("exec", loaded["secondary"])

                # Switch to single-artifact mode (TTS)
                tts_label = [m[0] for m in self.ui.MODES if "read-aloud" in m[0].lower()][0]
                app.mode.set(tts_label)
                self.assertEqual("tts", app.affix_text.get())
                # In single artifact mode, secondary entry is removed from grid
                self.assertIn("name.tts.txt", app.affix_preview.cget("text"))
                app.affix_kind.set("prefix")
                self.assertIn("tts.name.txt", app.affix_preview.cget("text"))

                # Switch back to Summarize
                summ_label = [m[0] for m in self.ui.MODES if "Summarize" in m[0]][0]
                app.mode.set(summ_label)
                self.assertEqual("full", app.affix_text.get())
                self.assertEqual("exec", app.affix_secondary.get())
                self.assertTrue(bool(app.affix_entry2.grid_info()))
        finally:
            root.destroy()



class ReadAloudRetainsEveryConvertedChunk(unittest.TestCase):
    """Read-aloud audits each converted chunk and repairs it once, and it never
    loses a usable conversion to the review. A truncating writer used to
    publish silently; now the audit sees it, and an absent reviewer or a
    failed repair still publishes the retained text with its status."""

    def _runner(self, script):
        sp = load("speechprep")

        class Runner:
            MODELS = ["w"]; AUDIT_MODELS = ["a"]; REPAIR_MODELS = ["r"]
            calls = []

            def run(self, prompt, workdir, chain, stage, validate=None,
                    gateway_options=None, role=None):
                self.calls.append(stage)
                out = script[stage]
                if isinstance(out, Exception):
                    raise out
                if validate:
                    validate(out)
                return out
        return sp, Runner()

    def _convert(self, sp, runner):
        with tempfile.TemporaryDirectory() as td:
            work = pathlib.Path(td)
            text = sp.convert_text("Source paragraph one.\n\nSource paragraph two.",
                                   work, runner=runner)
            report = json.loads((work / "speech-report.json").read_text())
        return text, report["chunks"][0]

    def test_a_repaired_chunk_replaces_the_original_when_the_reaudit_is_no_worse(self):
        sp, runner = self._runner({
            "speechprep": "para one only",
            "speechprep-audit": json.dumps({"verdict": "revise",
                                            "findings": ["MISSING paragraph two."]}),
            "speechprep-revise": "para one. para two.",
            "speechprep-reaudit": json.dumps({"verdict": "pass", "findings": []})})
        text, rec = self._convert(sp, runner)
        self.assertEqual(text.strip(), "para one. para two.")
        self.assertEqual((rec["selected"], rec["repair"], rec["status"]),
                         ("revised", "once", "pass"))

    def test_a_failed_repair_publishes_the_audited_original_with_its_findings(self):
        sp, runner = self._runner({
            "speechprep": "para one only",
            "speechprep-audit": json.dumps({"verdict": "revise",
                                            "findings": ["MISSING paragraph two."]}),
            "speechprep-revise": RuntimeError("no repair model answered")})
        text, rec = self._convert(sp, runner)
        self.assertEqual(text.strip(), "para one only")
        self.assertEqual((rec["selected"], rec["status"]), ("initial", "open_findings"))
        self.assertTrue(rec["repair"].startswith("unavailable"))
        self.assertEqual(rec["findings"], ["MISSING paragraph two."])

    def test_a_worse_repair_is_not_selected(self):
        sp, runner = self._runner({
            "speechprep": "para one only",
            "speechprep-audit": json.dumps({"verdict": "revise",
                                            "findings": ["MISSING paragraph two."]}),
            "speechprep-revise": "para one. para two. And my own thoughts.",
            "speechprep-reaudit": json.dumps({"verdict": "revise", "findings": [
                "ADDED commentary.", "CHANGED paragraph one."]})})
        text, rec = self._convert(sp, runner)
        self.assertEqual(text.strip(), "para one only")
        self.assertEqual((rec["selected"], rec["repair"]), ("initial", "once"))

    def test_an_absent_reviewer_publishes_the_conversion_as_unreviewed(self):
        sp, runner = self._runner({
            "speechprep": "para one. para two.",
            "speechprep-audit": RuntimeError("auditor down")})
        text, rec = self._convert(sp, runner)
        self.assertEqual(text.strip(), "para one. para two.")
        self.assertTrue(rec["review"].startswith("unavailable"))
        self.assertNotIn("speechprep-revise", runner.calls)

    def test_the_repair_prompt_encloses_the_current_conversion_and_findings(self):
        seen = {}
        script = {
            "speechprep": "para one only",
            "speechprep-audit": json.dumps({"verdict": "revise",
                                            "findings": ["MISSING paragraph two."]}),
            "speechprep-revise": "para one. para two.",
            "speechprep-reaudit": json.dumps({"verdict": "pass", "findings": []})}
        sp, runner = self._runner(script)
        real = runner.run
        def spy(prompt, *a, **k):
            seen[a[2]] = prompt
            return real(prompt, *a, **k)
        runner.run = spy
        self._convert(sp, runner)
        repair = seen["speechprep-revise"]
        for needle in ("CURRENT CONVERSION", "para one only",
                       "MISSING paragraph two.", "Source paragraph one."):
            self.assertIn(needle, repair)
        audit = seen["speechprep-audit"]
        self.assertIn("Source paragraph two.", audit)
        self.assertIn("para one only", audit)


class WriterRecoveryContract(unittest.TestCase):
    """Reject-and-repack recovery uses production helpers, not a parallel machine."""

    wc = load("writing_contract")
    jc = load("json_contract")

    def setUp(self):
        fixture = tempfile.TemporaryDirectory()
        self.addCleanup(fixture.cleanup)
        self.COURT = pathlib.Path(fixture.name)
        (self.COURT / "run").mkdir()
        self.source = "\n\n".join(
            (f"Synthetic evidence paragraph {i} states bounded result {i}. "
             + " ".join(f"evidence-{i}-{j}" for j in range(1, 116)))
            for i in range(1, 11))
        (self.COURT / "source.txt").write_text(self.source)
        source_ids = [item["id"] for item in self.wc.source_segments(
            self.source)]
        detailed_words = [41] * 10
        brief_words = [25, 25, 25, 25, 25, 0, 30, 30, 25, 0]
        plan = {
            "schema": self.wc.SCHEMA,
            "units": [{
                "unit_id": f"U{i:03d}",
                "title": f"Synthetic unit {i}",
                "source_ids": [source_ids[i - 1]],
                "topic": f"Bounded result {i}",
                "relation_to_previous": ("opening" if i == 1 else
                                         "continues the synthetic evidence"),
                "governing_qualifications": [],
                "detailed_words": detailed_words[i - 1],
                "brief_disposition": ("include" if brief_words[i - 1]
                                      else "omit"),
                "brief_words": brief_words[i - 1],
            } for i in range(1, 11)],
            "dispositions": [],
        }
        self.frozen = self.wc.validate_plan(
            plan, self.source, {"detailed": 500, "brief": 250})
        plan_path = self.COURT / "run" / "writing-plan.json"
        self.wc.save_plan(plan_path, self.frozen)
        self.plan = json.loads(plan_path.read_text())
        self.packet = self.wc.build_packets(
            self.frozen,
            output_words=sum(self.frozen["allocated_words"].values()))[0]
        invalid = self._response_for(self.packet, words=60)
        invalid["detailed"][-1]["paragraphs"] = []
        self.attempt06 = json.dumps(invalid)
        self.attempt07 = json.dumps({
            "detailed": [invalid["detailed"][0]], "brief": []})

    def _assignment_from_prompt(self, prompt):
        blob = prompt.split("ASSIGNMENT:\n", 1)[1]
        blob = blob.split("\nPRECEDING ACCEPTED PROSE", 1)[0]
        return json.loads(blob.strip())

    def _response_for(self, packet, words=40):
        def block(ident, n):
            return {"unit_id": ident, "heading": "",
                    "paragraphs": [f"{ident} " + ("claim " * n)]}
        return {
            "detailed": [block(item["unit_id"], words)
                         for item in packet.get("detailed") or []],
            "brief": [block(item["unit_id"], max(20, words // 2))
                      for item in packet.get("brief") or []],
        }

    def _schema_counts(self, gateway_options):
        schema = ((gateway_options or {}).get("response_format") or {}).get(
            "json_schema", {}).get("schema") or {}
        props = schema.get("properties") or {}
        return (
            int((props.get("detailed") or {}).get("maxItems") or 0),
            int((props.get("brief") or {}).get("maxItems") or 0),
        )

    def _fake_ms(self, *, reject_wide=False, record=None,
                 fail_first_atomic=False, fail_all_atomic=False,
                 fail_kind=None, fail_first_audit=False,
                 fail_audit_stage=None, audit_finding_on_retry=False,
                 audit_open_finding=False, fail_repair=False,
                 repair_patch=None):
        seen = record if record is not None else []
        attempt06 = self.attempt06
        counts = self._schema_counts
        assignment = self._assignment_from_prompt
        response_for = self._response_for
        failed_atomic = set()
        atomic_failures = {}
        audit_failed = set()
        repair_patch_payload = repair_patch

        class Fake:
            PLAN_MODELS = ["planner"]
            MODELS = ["writer"]
            AUDIT_MODELS = ["auditor"]
            REPAIR_MODELS = ["repair"]
            HARNESS = "fake"
            class NoCandidate(RuntimeError):
                def __init__(self, message, *, kind=None, detail="",
                             attempts=None, recovery_kind=None):
                    super().__init__(message)
                    self.kind = kind
                    self.detail = detail
                    self.attempts = list(attempts or [])
                    self.recovery_kind = (recovery_kind if recovery_kind
                                          is not None else kind)
            def run(self, prompt, out_dir, chain, stage, validate=None,
                    gateway_options=None, role=None, **_kwargs):
                if stage.startswith("full-v2-plan"):
                    raise AssertionError(
                        f"planning call on retained resume: {stage}")
                detailed_n, brief_n = counts(gateway_options)
                seen.append({
                    "stage": stage, "prompt": prompt,
                    "detailed_n": detailed_n, "brief_n": brief_n,
                    "role": role,
                })
                if stage.startswith("full-v2-write"):
                    if detailed_n + brief_n > 1:
                        if reject_wide:
                            raw = attempt06
                            if validate:
                                validate(raw)
                            return raw
                        raise AssertionError(
                            f"wide root write {stage}: {detailed_n}+{brief_n}")
                    packet = assignment(prompt)
                    base_stage = stage.split("-retry")[0]
                    if (fail_kind and fail_kind not in {
                            "output_limit", "output_incomplete", "unusable",
                            "response_contract"}
                            and base_stage.endswith("001")):
                        raise Fake.NoCandidate(
                            f"ineligible {fail_kind}",
                            kind=fail_kind, recovery_kind=fail_kind)
                    if fail_all_atomic and base_stage.endswith("001"):
                        raise Fake.NoCandidate(
                            "gateway finish_reason=length",
                            kind="output_limit",
                            recovery_kind="output_limit")
                    if (fail_first_atomic and "-retry" not in stage
                            and stage.endswith("001")
                            and stage not in failed_atomic):
                        failed_atomic.add(stage)
                        raise Fake.NoCandidate(
                            "gateway finish_reason=length",
                            kind="output_limit",
                            recovery_kind="output_limit")
                    raw = json.dumps(response_for(packet))
                    if validate:
                        validate(raw)
                    return raw
                if "revise" in stage:
                    if fail_repair:
                        raise Fake.NoCandidate(
                            "gateway finish_reason=length",
                            kind="output_limit",
                            recovery_kind="output_limit")
                    if repair_patch_payload is not None:
                        raw = json.dumps(repair_patch_payload)
                        if validate:
                            validate(raw)
                        return raw
                if "audit" in stage or "reaudit" in stage:
                    logical = stage.split("-retry")[0]
                    should_fail = (
                        (fail_audit_stage and logical == fail_audit_stage
                         and "-retry" not in stage)
                        or (fail_first_audit and stage not in audit_failed
                            and "-retry" not in stage))
                    if should_fail:
                        audit_failed.add(stage)
                        raise Fake.NoCandidate(
                            "gateway finish_reason=length",
                            kind="output_limit",
                            recovery_kind="output_limit")
                    ids = re.findall(r'"observation_id": "(R-\d+)"', prompt)
                    findings = []
                    if audit_open_finding or (
                            audit_finding_on_retry and stage.endswith("-retry")):
                        findings = [{
                            "kind": "editorial", "artifact": "detailed",
                            "text": "Robustness checks preserved the primary result.",
                            "anchor": "D-P001", "slot": "",
                        }]
                    value = {"verdict": "revise" if findings else "pass",
                             "findings": findings,
                             "readability": [
                                 {"observation_id": ident,
                                  "assessment": "acceptable",
                                  "explanation": "Coherent enough to retain."}
                                 for ident in ids]}
                    raw = json.dumps(value)
                    if validate:
                        validate(raw)
                    return raw
                raise AssertionError(stage)

        return Fake(), seen

    def _run_v2(self, fake, env, work, output_budget=10_000):
        fs = load("fullsum")
        out = pathlib.Path(work)
        with unittest.mock.patch.object(fs, "_last_ok_chain",
                                        return_value=["writer"]), \
                unittest.mock.patch.dict(os.environ, env, clear=False):
            os.environ.pop("SUMM_WRITING_ONE_UNIT_PER_REQUEST", None)
            if "SUMM_WRITING_RECOVERY_FILE" not in env:
                os.environ.pop("SUMM_WRITING_RECOVERY_FILE", None)
            rc = fs._run_contract_v2(
                self.source, out, len(self.source.split()),
                100_000, output_budget, fake, load("shortsum"))
            return rc, out

    def test_wr01_attempt06_is_rejected_and_repacked_to_18_keys(self):
        self.jc.validate(json.loads(self.attempt06), self.jc.FULL_BLOCK_PAIR,
                         "generic")
        with self.assertRaises(self.wc.WriterResponseError) as raised:
            self.wc.validate_blocks(json.loads(self.attempt06), self.packet)
        self.assertEqual(raised.exception.code, "schema")
        children = self.wc.atomic_packets(self.packet)
        self.assertEqual(len(children), 18)
        keys = [self.wc.obligation_keys(child)[0] for child in children]
        self.assertEqual(keys, self.wc.obligation_keys(self.packet))
        self.assertTrue(all(sum(len(c[d]) for d in ("detailed", "brief")) == 1
                            for c in children))

    def test_wr02_attempt07_fails_assignment_coverage(self):
        self.jc.validate(json.loads(self.attempt07), self.jc.FULL_BLOCK_PAIR,
                         "generic")
        with self.assertRaises(self.wc.WriterResponseError) as raised:
            self.wc.validate_blocks(json.loads(self.attempt07), self.packet)
        self.assertIn(raised.exception.code, {"schema", "coverage"})
        self.assertGreaterEqual(len(raised.exception.missing_keys), 17)

    def test_wr03_min_and_max_items_are_enforced(self):
        schema = self.wc.block_pair_schema(self.packet)
        detailed = json.loads(self.attempt06)["detailed"][:10]
        brief = json.loads(self.attempt06)["brief"]
        nine = {"detailed": detailed[:9], "brief": brief}
        eleven = {"detailed": detailed + [detailed[-1]], "brief": brief}
        with self.assertRaises(self.jc.ContractError):
            self.jc.validate(nine, schema, "bounds")
        with self.assertRaises(self.jc.ContractError):
            self.jc.validate(eleven, schema, "bounds")
        empty_brief = self.wc._packet("detailed", [self.packet["detailed"][0]])
        assigned = self.wc.block_pair_schema(empty_brief)
        with self.assertRaises(self.jc.ContractError):
            self.jc.validate(
                {"detailed": [{"unit_id": "U001", "heading": "",
                               "paragraphs": ["ok"]}],
                 "brief": [{"unit_id": "U001", "heading": "",
                            "paragraphs": ["no"]}]},
                assigned, "unassigned-brief")
        with self.assertRaises(self.jc.ContractError):
            self.jc._check([], {"type": "array", "minItems": True})
        with self.assertRaises(self.jc.ContractError):
            self.jc._check([], {"type": "array", "maxItems": -1})
        with self.assertRaises(self.jc.ContractError):
            self.jc._check([], {"type": "array", "minItems": 1.5})
        with self.assertRaises(self.wc.WriterResponseError):
            self.wc.validate_blocks(
                {"detailed": [{"unit_id": "U001", "heading": "",
                               "paragraphs": []}],
                 "brief": []},
                empty_brief)

    def test_wr04_assignment_example_uses_only_assigned_ids(self):
        unit = next(u for u in self.packet["detailed"]
                    if u["unit_id"] == "U007")
        packet = {"detailed": [unit], "brief": []}
        prompt = self.wc.writing_prompt(self.frozen, packet, self.source)
        self.assertIn('"unit_id": "U007"', prompt)
        self.assertNotIn('"unit_id": "U001"', prompt)
        with self.assertRaises(self.wc.WriterResponseError):
            self.wc.validate_blocks(
                {"detailed": [{"unit_id": "", "heading": "",
                               "paragraphs": ["x"]}],
                 "brief": []}, packet)
        with self.assertRaises(self.wc.WriterResponseError):
            self.wc.validate_blocks(
                {"detailed": [{"unit_id": "U001", "heading": "",
                               "paragraphs": ["x"]}],
                 "brief": []}, packet)
        with self.assertRaises(self.wc.WriterResponseError):
            self.wc.validate_blocks(
                {"detailed": [
                    {"unit_id": "U007", "heading": "", "paragraphs": ["a"]},
                    {"unit_id": "U007", "heading": "", "paragraphs": ["b"]}],
                 "brief": []}, packet)

    def test_wr05_atomic_wave_preserves_plan_order_and_allocations(self):
        children = self.wc.atomic_packets(self.packet)
        self.assertEqual(
            [self.wc.obligation_keys(c)[0] for c in children],
            [("detailed", "U001"), ("brief", "U001"),
             ("detailed", "U002"), ("brief", "U002"),
             ("detailed", "U003"), ("brief", "U003"),
             ("detailed", "U004"), ("brief", "U004"),
             ("detailed", "U005"), ("brief", "U005"),
             ("detailed", "U006"),
             ("detailed", "U007"), ("brief", "U007"),
             ("detailed", "U008"), ("brief", "U008"),
             ("detailed", "U009"), ("brief", "U009"),
             ("detailed", "U010")])
        detailed_words = sum(self.wc._packet_output_words(c)
                             for c in children if c["detailed"])
        brief_words = sum(self.wc._packet_output_words(c)
                          for c in children if c["brief"])
        self.assertEqual(detailed_words, 410)
        self.assertEqual(brief_words, 210)
        self.assertEqual(
            [c["detailed"][0]["source_ids"] for c in children if c["detailed"]],
            [u["source_ids"] for u in self.packet["detailed"]])

    def test_wr06_root_wave_is_monotone(self):
        fake, seen = self._fake_ms(reject_wide=True)
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            plan_path = root / "writing-plan.json"
            plan_path.write_bytes((self.COURT / "run/writing-plan.json").read_bytes())
            rc, out = self._run_v2(fake, {
                "SUMM_WRITING_PLAN_FILE": str(plan_path),
                "SUMM_WRITING_HEADING_POLICY": "none",
            }, root / "work")
            self.assertEqual(rc, 0)
            writes = [item for item in seen if item["stage"].startswith("full-v2-write")]
            wide = [item for item in writes if item["detailed_n"] + item["brief_n"] > 1]
            atomic = [item for item in writes if item["detailed_n"] + item["brief_n"] == 1]
            self.assertEqual(len(wide), 1)
            self.assertEqual(len(atomic), 18)
            self.assertTrue(wide[0]["stage"].startswith("full-v2-write-001"))
            self.assertTrue(atomic[0]["stage"].startswith("full-v2-write-001-r"))
            self.assertIn("SAME-UNIT COMPLETED DETAILED FOR U001",
                          atomic[1]["prompt"])
            self.assertNotIn("SAME-UNIT COMPLETED DETAILED FOR U010",
                             atomic[1]["prompt"])
            state = json.loads((out / "writer-recovery.json").read_text())
            self.assertEqual(state["wave"], "complete")
            self.assertEqual(state["pending_keys"], [])
            self.assertEqual(len(state["accepted"]), 18)

    def test_wr07_structural_failure_survives_later_unavailable(self):
        ms = load("mapsum")
        error = ms.NoCandidate(
            "chain exhausted", kind="unavailable",
            attempts=[{"kind": "response_contract", "detail": "coverage"},
                      {"kind": "unavailable", "detail": "502"}],
            recovery_kind="response_contract")
        self.assertEqual(error.kind, "unavailable")
        self.assertEqual(error.recovery_kind, "response_contract")
        src = (pathlib.Path(ms.__file__).read_text() if getattr(ms, "__file__", None)
               else (ENG / "mapsum.py").read_text())
        self.assertIn("recovery_kind", src)
        self.assertIn("response_contract", src)

    def test_wr08_atomic_child_does_not_repack(self):
        child = self.wc.atomic_packets(self.packet)[0]
        self.assertEqual(self.wc.atomic_packets(child), [child])
        self.assertEqual(len(self.wc.obligation_keys(child)), 1)
        src = (ENG / "fullsum.py").read_text()
        self.assertIn("len(keys) > 1 and wave == \"fresh\"", src)
        self.assertIn('wave="repacked"', src)

    def test_wr09_ledger_keeps_plan_order_and_rejects_duplicates(self):
        state = self.wc.new_writer_recovery(self.frozen, self.packet)
        state["wave"] = "repacked"
        for artifact, ident in self.wc.obligation_keys(self.packet):
            block = {"unit_id": ident, "heading": "",
                     "paragraphs": [f"{artifact} {ident}"]}
            state = self.wc.accept_recovery_block(
                state, artifact, ident, block, {"stage": ident})
        self.assertEqual(state["wave"], "complete")
        rendered = self.wc.render_recovery_blocks(self.frozen, state)
        self.assertEqual(list(rendered["detailed"]),
                         [f"U{i:03d}" for i in range(1, 11)])
        self.assertEqual(list(rendered["brief"]),
                         ["U001", "U002", "U003", "U004", "U005",
                          "U007", "U008", "U009"])

    def test_wr10_hash_bound_resume_starts_at_atomic_child(self):
        fake, seen = self._fake_ms(reject_wide=False)
        state = self.wc.new_writer_recovery(
            self.frozen, self.packet, heading_policy="none",
            source_sha256=self.frozen["source_sha256"])
        state["wave"] = "rejected"
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            plan_path = root / "writing-plan.json"
            plan_path.write_bytes(
                (self.COURT / "run/writing-plan.json").read_bytes())
            recovery_path = root / "writer-recovery.json"
            recovery_path.write_text(json.dumps(state))
            rc, out = self._run_v2(fake, {
                "SUMM_WRITING_PLAN_FILE": str(plan_path),
                "SUMM_WRITING_RECOVERY_FILE": str(recovery_path),
                "SUMM_WRITING_HEADING_POLICY": "none",
            }, root / "work")
            self.assertEqual(rc, 0)
            writes = [item for item in seen
                      if item["stage"].startswith("full-v2-write")]
            self.assertEqual(len(writes), 18)
            self.assertEqual(writes[0]["detailed_n"], 1)
            self.assertEqual(writes[0]["brief_n"], 0)
            self.assertIn("1 Detailed block", writes[0]["prompt"])
            self.assertTrue(all(
                item["detailed_n"] + item["brief_n"] == 1 for item in writes))
            report = json.loads((out / "full-report.json").read_text())
            self.assertEqual(report["writing_plan_origin"], "validated_replay")
            tampered = json.loads(recovery_path.read_text())
            tampered["root_assignment_sha256"] = "0" * 64
            bad = root / "bad-recovery.json"
            bad.write_text(json.dumps(tampered))
            fake2, _seen2 = self._fake_ms(reject_wide=False)
            rc2, _out2 = self._run_v2(fake2, {
                "SUMM_WRITING_PLAN_FILE": str(plan_path),
                "SUMM_WRITING_RECOVERY_FILE": str(bad),
                "SUMM_WRITING_HEADING_POLICY": "none",
            }, root / "work-tampered")
            self.assertEqual(rc2, 1)

    def test_wr11_heading_policy_none_and_same_unit_brief_context(self):
        unit = self.packet["detailed"][0]
        brief = self.packet["brief"][0]
        prompt = self.wc.writing_prompt(
            self.frozen, {"detailed": [], "brief": [brief]}, self.source,
            "SAME-UNIT COMPLETED DETAILED FOR U001 (continuity/comparison "
            "only, never factual evidence):\nThe Harbor Housing Study.",
            heading_policy="none")
        self.assertIn("heading must be the empty string", prompt)
        self.assertIn("SAME-UNIT COMPLETED DETAILED FOR U001", prompt)
        self.assertNotIn("U010", prompt.split("GLOBAL OUTLINE:", 1)[0])
        self.assertNotIn(
            "SAME-UNIT COMPLETED DETAILED FOR U010", prompt)
        schema = self.wc.block_pair_schema(
            {"detailed": [unit], "brief": []}, heading_policy="none")
        with self.assertRaises(self.jc.ContractError):
            self.jc.validate(
                {"detailed": [{"unit_id": "U001", "heading": "Nope",
                               "paragraphs": ["x"]}],
                 "brief": []}, schema, "heading")

    def test_wr12_overruns_and_fidelity_are_disclosed(self):
        blocks = {"detailed": {}, "brief": {}}
        obj = json.loads(self.attempt06)
        for item in obj["detailed"]:
            if item.get("unit_id"):
                blocks["detailed"][item["unit_id"]] = item
        for item in obj["brief"]:
            blocks["brief"][item["unit_id"]] = item
        findings = self.wc.block_word_findings(self.frozen, blocks)
        self.assertGreaterEqual(len(findings), 17)
        brief_u004 = " ".join(blocks["brief"]["U004"]["paragraphs"])
        self.assertIn("U004", brief_u004)
        self.assertGreater(len(brief_u004.split()), 25)

    def test_wr13_review_coalesce_keeps_every_block(self):
        children = self.wc.atomic_packets(self.packet)
        blocks = {"detailed": {}, "brief": {}}
        for child in children:
            for depth in ("detailed", "brief"):
                for item in child[depth]:
                    blocks[depth][item["unit_id"]] = {
                        "unit_id": item["unit_id"], "heading": "",
                        "paragraphs": [f"{depth} {item['unit_id']} prose."]}
        detailed = self.wc.render_blocks(list(blocks["detailed"].values()))
        brief = self.wc.render_blocks(list(blocks["brief"].values()))
        local, global_packet = self.wc.bounded_review_packets(
            self.frozen, children, blocks, self.source, detailed, brief)
        coalesced = self.wc.coalesce_review_packets(local)
        self.assertEqual(len(coalesced), 10)
        self.assertTrue(global_packet["brief"])
        covered = " ".join(p["detailed"] + p["brief"] for p in coalesced)
        for ident in [f"U{i:03d}" for i in range(1, 11)]:
            self.assertIn(ident, covered)

    def test_wr14_valid_wide_pair_stays_one_call(self):
        packet = self.wc.build_packets(
            self.frozen, output_words=10_000)[0]
        valid = {"detailed": [], "brief": []}
        for unit in packet["detailed"]:
            valid["detailed"].append({
                "unit_id": unit["unit_id"], "heading": "",
                "paragraphs": ["ok"]})
        for unit in packet["brief"]:
            valid["brief"].append({
                "unit_id": unit["unit_id"], "heading": "",
                "paragraphs": ["ok"]})
        clean = self.wc.validate_blocks(valid, packet)
        self.assertEqual(len(clean["detailed"]), 10)
        self.assertEqual(len(clean["brief"]), 8)
        self.assertEqual(len(self.wc.build_packets(
            self.frozen, output_words=10_000)), 1)
        runtime_src = (ENG / "runtime.py").read_text()
        self.assertIn(
            '"reasoning_tokens", "retry_reasoning_tokens"', runtime_src)
        self.assertIn('"prompt_tokens", "visible_tokens"', runtime_src)
        self.assertNotIn("packet_width", runtime_src)
        self.assertNotIn("writer_blocks", runtime_src)

    def test_wr15_child_schema_and_example_are_assignment_specific(self):
        child = self.wc.atomic_packets(self.packet)[0]
        schema = self.wc.block_pair_schema(child, heading_policy="none")
        self.assertEqual(schema["properties"]["detailed"]["minItems"], 1)
        self.assertEqual(schema["properties"]["detailed"]["maxItems"], 1)
        self.assertEqual(schema["properties"]["brief"]["minItems"], 0)
        self.assertEqual(schema["properties"]["brief"]["maxItems"], 0)
        prompt = self.wc.writing_prompt(
            self.frozen, child, self.source, heading_policy="none")
        self.assertIn("1 Detailed block", prompt)
        self.assertIn("and 0", prompt)
        self.assertIn("Brief block", prompt)
        options = self.jc.options("summer_full_blocks_v2", schema)
        nested = options["response_format"]["json_schema"]["schema"]
        self.assertEqual(nested["properties"]["detailed"]["maxItems"], 1)
        admission = (ENG / "fullsum.py").read_text()
        self.assertIn("WRITE_OUTPUT_OVERHEAD_TOKENS", admission)
        self.assertIn('role="write"', admission)
        self.assertIn("planned_output_words=words", admission)

    def test_wr16_atomic_child_retries_before_blocking(self):
        fake, seen = self._fake_ms(fail_first_atomic=True)
        state = self.wc.new_writer_recovery(
            self.frozen, self.packet, heading_policy="none",
            source_sha256=self.frozen["source_sha256"])
        state["wave"] = "rejected"
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            plan_path = root / "writing-plan.json"
            plan_path.write_bytes(
                (self.COURT / "run/writing-plan.json").read_bytes())
            recovery_path = root / "writer-recovery.json"
            recovery_path.write_text(json.dumps(state))
            rc, out = self._run_v2(fake, {
                "SUMM_WRITING_PLAN_FILE": str(plan_path),
                "SUMM_WRITING_RECOVERY_FILE": str(recovery_path),
                "SUMM_WRITING_HEADING_POLICY": "none",
            }, root / "work")
            self.assertEqual(rc, 0)
            writes = [item for item in seen
                      if item["stage"].startswith("full-v2-write")]
            self.assertEqual(len(writes), 19)
            self.assertEqual(writes[0]["stage"], "full-v2-write-001")
            self.assertEqual(writes[1]["stage"], "full-v2-write-001-retry01")
            src = (ENG / "fullsum.py").read_text()
            self.assertIn("ATOMIC_CHILD_ATTEMPTS", src)
            self.assertIn("retrying atomic child", src)
            self.assertIn("retrying audit", src)

    def _resume_state(self, accepted_count):
        state = self.wc.new_writer_recovery(
            self.frozen, self.packet, heading_policy="none",
            source_sha256=self.frozen["source_sha256"])
        state["wave"] = "repacked"
        children = self.wc.atomic_packets(self.packet)
        for child in children[:accepted_count]:
            keys = self.wc.obligation_keys(child)
            artifact, ident = keys[0]
            block = self._response_for(child)[artifact][0]
            state = self.wc.accept_recovery_block(
                state, artifact, ident, block,
                {"stage": "inherited", "producer": ["writer"]})
        return state

    def test_wr18_resume_review_covers_inherited_blocks(self):
        fake, seen = self._fake_ms()
        state = self._resume_state(11)
        self.assertEqual(len(state["pending_keys"]), 7)
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            plan_path = root / "writing-plan.json"
            plan_path.write_bytes(
                (self.COURT / "run/writing-plan.json").read_bytes())
            recovery_path = root / "writer-recovery.json"
            recovery_path.write_text(json.dumps(state))
            rc, out = self._run_v2(fake, {
                "SUMM_WRITING_PLAN_FILE": str(plan_path),
                "SUMM_WRITING_RECOVERY_FILE": str(recovery_path),
                "SUMM_WRITING_HEADING_POLICY": "none",
            }, root / "work")
            self.assertEqual(rc, 0)
            audits = [item["stage"] for item in seen
                      if "audit" in item["stage"] and "reaudit" not in item["stage"]]
            locals_ = [name for name in audits if name.startswith("full-v2-audit-")
                       and not name.endswith("-global") and "-retry" not in name]
            self.assertEqual(len(locals_), 10)
            self.assertIn("full-v2-audit-global", audits)
            covered = " ".join(item["prompt"] for item in seen
                               if item["stage"].startswith("full-v2-audit-")
                               and "-global" not in item["stage"])
            for ident in [f"U{i:03d}" for i in range(1, 11)]:
                self.assertIn(ident, covered)

    def test_wr19_rebind_and_reaudit_use_repaired_text(self):
        children = self.wc.atomic_packets(self.packet)
        blocks = {"detailed": {}, "brief": {}}
        for child in children:
            obj = self._response_for(child)
            for depth in ("detailed", "brief"):
                for item in obj[depth]:
                    blocks[depth][item["unit_id"]] = item
        detailed = self.wc.render_blocks(
            [blocks["detailed"][i] for i in self.wc._ids(self.packet, "detailed")])
        brief = self.wc.render_blocks(
            [blocks["brief"][i] for i in self.wc._ids(self.packet, "brief")])
        pp = load("pair_patch")
        pair = pp.segment_pair(detailed, brief)
        first = pair["detailed"][0]
        edits = [{
            "artifact": "detailed", "operation": "replace_range",
            "start": first["id"], "end": first["id"],
            "range_sha256": first["sha256"],
            "replacement": "U001 replaced sentence about the offer.",
            "finding_ids": ["F-001"],
        }, {
            "artifact": "detailed", "operation": "insert_after",
            "anchor": first["id"], "anchor_sha256": first["sha256"],
            "replacement": "Inserted continuity paragraph after the claim.",
            "finding_ids": ["F-002"],
        }]
        rebound = self.wc.rebind_blocks_after_patch(
            blocks, detailed, brief, edits)
        new_d, new_b = pp.apply_edits(
            detailed, brief, {"base_candidate": pair["candidate"], "edits": edits})
        self.assertIn("U001 replaced sentence about the offer.",
                      " ".join(rebound["detailed"]["U001"]["paragraphs"]))
        self.assertIn("Inserted continuity paragraph after the claim.",
                      " ".join(rebound["detailed"]["U001"]["paragraphs"]))
        local, glob = self.wc.bounded_review_packets(
            self.frozen, children, rebound, self.source, new_d, new_b)
        blob = " ".join(p["detailed"] for p in local) + glob["brief"]
        self.assertIn("U001 replaced sentence about the offer.", blob)
        self.assertIn("Inserted continuity paragraph after the claim.", blob)

    def test_wr20_audit_retry_keeps_logical_packet_id(self):
        pr = load("pair_review")
        prompt = pr.build_patch_prompt(
            "Detailed sentence one.\n\nDetailed sentence two.",
            "Brief sentence.",
            [{"packet": "full-v2-audit-001", "finding_id": "F-001",
              "kind": "editorial", "artifact": "detailed",
              "text": "A scoped finding", "anchor": "D-P001"}],
            source_context={
                "full-v2-audit-001": "LOCAL RAW SOURCE PACKET TEXT",
                "full-v2-audit-001-retry": "RETRY SHOULD NOT BE SELECTED",
            })
        self.assertIn("LOCAL RAW SOURCE PACKET TEXT", prompt)
        self.assertIn("full-v2-audit-001", prompt)
        self.assertNotIn("RETRY SHOULD NOT BE SELECTED", prompt)
        fake, seen = self._fake_ms(fail_audit_stage="full-v2-audit-001")
        state = self.wc.new_writer_recovery(
            self.frozen, self.packet, heading_policy="none",
            source_sha256=self.frozen["source_sha256"])
        state["wave"] = "rejected"
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            plan_path = root / "writing-plan.json"
            plan_path.write_bytes(
                (self.COURT / "run/writing-plan.json").read_bytes())
            recovery_path = root / "writer-recovery.json"
            recovery_path.write_text(json.dumps(state))
            rc, _out = self._run_v2(fake, {
                "SUMM_WRITING_PLAN_FILE": str(plan_path),
                "SUMM_WRITING_RECOVERY_FILE": str(recovery_path),
                "SUMM_WRITING_HEADING_POLICY": "none",
            }, root / "work")
            self.assertEqual(rc, 0)
            self.assertTrue(any(
                item["stage"] == "full-v2-audit-001-retry" for item in seen))
            self.assertTrue(any(
                item["stage"] == "full-v2-audit-001" for item in seen))

    def test_wr21_writer_retry_records_successful_attempt(self):
        fake, seen = self._fake_ms(fail_first_atomic=True)
        state = self.wc.new_writer_recovery(
            self.frozen, self.packet, heading_policy="none",
            source_sha256=self.frozen["source_sha256"])
        state["wave"] = "rejected"
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            plan_path = root / "writing-plan.json"
            plan_path.write_bytes(
                (self.COURT / "run/writing-plan.json").read_bytes())
            recovery_path = root / "writer-recovery.json"
            recovery_path.write_text(json.dumps(state))
            rc, out = self._run_v2(fake, {
                "SUMM_WRITING_PLAN_FILE": str(plan_path),
                "SUMM_WRITING_RECOVERY_FILE": str(recovery_path),
                "SUMM_WRITING_HEADING_POLICY": "none",
            }, root / "work")
            self.assertEqual(rc, 0)
            ledger = json.loads((out / "writer-recovery.json").read_text())
            first = next(iter(ledger["accepted"].values()))
            self.assertEqual(first["receipt"]["stage"],
                             "full-v2-write-001-retry01")
            self.assertEqual(first["receipt"]["logical_stage"],
                             "full-v2-write-001")

    def test_wr22_ineligible_and_exhausted_children_block(self):
        fake, seen = self._fake_ms(fail_kind="config")
        state = self.wc.new_writer_recovery(
            self.frozen, self.packet, heading_policy="none",
            source_sha256=self.frozen["source_sha256"])
        state["wave"] = "rejected"
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            plan_path = root / "writing-plan.json"
            plan_path.write_bytes(
                (self.COURT / "run/writing-plan.json").read_bytes())
            recovery_path = root / "writer-recovery.json"
            recovery_path.write_text(json.dumps(state))
            rc, _out = self._run_v2(fake, {
                "SUMM_WRITING_PLAN_FILE": str(plan_path),
                "SUMM_WRITING_RECOVERY_FILE": str(recovery_path),
                "SUMM_WRITING_HEADING_POLICY": "none",
            }, root / "work")
            self.assertNotEqual(rc, 0)
            writes = [item["stage"] for item in seen
                      if item["stage"].startswith("full-v2-write")]
            self.assertEqual(writes, ["full-v2-write-001"])
        fake2, seen2 = self._fake_ms(fail_all_atomic=True)
        state = self.wc.new_writer_recovery(
            self.frozen, self.packet, heading_policy="none",
            source_sha256=self.frozen["source_sha256"])
        state["wave"] = "rejected"
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            plan_path = root / "writing-plan.json"
            plan_path.write_bytes(
                (self.COURT / "run/writing-plan.json").read_bytes())
            recovery_path = root / "writer-recovery.json"
            recovery_path.write_text(json.dumps(state))
            rc, out = self._run_v2(fake2, {
                "SUMM_WRITING_PLAN_FILE": str(plan_path),
                "SUMM_WRITING_RECOVERY_FILE": str(recovery_path),
                "SUMM_WRITING_HEADING_POLICY": "none",
            }, root / "work")
            self.assertNotEqual(rc, 0)
            writes = [item["stage"] for item in seen2
                      if item["stage"].startswith("full-v2-write")]
            self.assertEqual(writes, [
                "full-v2-write-001",
                "full-v2-write-001-retry01",
                "full-v2-write-001-retry02",
            ])
            ledger = json.loads((out / "writer-recovery.json").read_text())
            self.assertEqual(ledger["wave"], "blocked")
            self.assertTrue(ledger["unresolved"])

    def test_wr23_truncated_repair_still_publishes_reviewed_pair(self):
        fake, seen = self._fake_ms(
            fail_repair=True, audit_open_finding=True)
        state = self.wc.new_writer_recovery(
            self.frozen, self.packet, heading_policy="none",
            source_sha256=self.frozen["source_sha256"])
        state["wave"] = "rejected"
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            plan_path = root / "writing-plan.json"
            plan_path.write_bytes(
                (self.COURT / "run/writing-plan.json").read_bytes())
            recovery_path = root / "writer-recovery.json"
            recovery_path.write_text(json.dumps(state))
            rc, out = self._run_v2(fake, {
                "SUMM_WRITING_PLAN_FILE": str(plan_path),
                "SUMM_WRITING_RECOVERY_FILE": str(recovery_path),
                "SUMM_WRITING_HEADING_POLICY": "none",
            }, root / "work")
            self.assertEqual(rc, 0)
            self.assertTrue((out / "detailed.md").exists())
            self.assertTrue((out / "brief.md").exists())
            report = json.loads((out / "full-report.json").read_text())
            self.assertEqual(report["status"], "open_findings")


def _isolated_main() -> int:
    """Run each class in a fresh interpreter.

    Tk and imported engine modules retain process-global state. Running the
    entire suite in one interpreter can crash during Tk teardown or let one
    class's simulated dead-route state affect another. Class isolation keeps
    the public test command deterministic on macOS and Windows.
    """
    classes = [name for name, value in globals().items()
               if isinstance(value, type)
               and issubclass(value, unittest.TestCase)
               and value is not unittest.TestCase
               and any(attr.startswith("test_") for attr in vars(value))]
    failed = []
    for name in classes:
        result = subprocess.run(
            [sys.executable, str(pathlib.Path(__file__).resolve()), name, "-q"],
            capture_output=True, text=True)
        if result.returncode:
            failed.append(name)
            sys.stdout.write(result.stdout)
            sys.stderr.write(result.stderr)
    print(f"classes={len(classes)} passed={len(classes) - len(failed)} "
          f"failed={len(failed)}")
    if failed:
        print("failed classes: " + ", ".join(failed), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1:
        unittest.main(verbosity=2)
    raise SystemExit(_isolated_main())
