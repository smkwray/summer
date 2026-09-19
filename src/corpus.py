#!/usr/bin/env python3
"""Corpus request validation and model-free source preparation.

This module owns the boundary before the first Corpus model call.  It does not
concatenate documents and it does not invoke a model.  Every document is staged
and read independently, then the resulting source inventory is described by
opaque IDs and hashes.  Later Corpus stages can therefore consume one frozen,
complete source snapshot without receiving filesystem paths.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import sys
from typing import Callable

HERE = pathlib.Path(__file__).parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import ledger
import custom_instructions
import json_contract
import mode_config
import progress
import readerview
import selection


MIN_DOCUMENTS = mode_config.CORPUS.min_documents
MAX_DOCUMENTS = mode_config.CORPUS.max_documents
MIN_VISIBLE_WORDS = mode_config.CORPUS.min_visible_words
MAX_VISIBLE_WORDS = mode_config.CORPUS.max_visible_words
INVENTORY_WINDOW_WORDS = mode_config.CORPUS.inventory_window_words
MAX_INVENTORY_WINDOWS = mode_config.CORPUS.max_inventory_windows
SOURCE_SCHEMA = "summer.corpus-source.v1"
PREFLIGHT_SCHEMA = "summer.corpus-preflight.v1"
INVENTORY_SCHEMA = "summer.document-inventory.v1"
INVENTORY_INDEX_SCHEMA = "summer.corpus-inventory-index.v1"
INVENTORY_PLAN_OPTIONS = json_contract.options(
    "summer_corpus_inventory", json_contract.DOCUMENT_INVENTORY)
INVENTORY_REPAIR_OPTIONS = json_contract.options(
    "summer_corpus_inventory_repair", json_contract.INVENTORY_REVISION)
INVENTORY_AUDIT_OPTIONS = json_contract.options(
    "summer_corpus_inventory_audit", json_contract.CORPUS_AUDIT)
PLAN_REQUEST_OPTIONS = json_contract.options(
    "summer_corpus_plan", json_contract.CORPUS_PLAN)
PLAN_REPAIR_OPTIONS = json_contract.options(
    "summer_corpus_plan_repair", json_contract.CORPUS_PLAN_REVISION)
PLAN_AUDIT_OPTIONS = json_contract.options(
    "summer_corpus_plan_audit", json_contract.CORPUS_AUDIT)

class CorpusPreflightError(RuntimeError):
    """A request failed before model work was permitted."""

    def __init__(self, message: str, *, code: str = "preflight_failed",
                 errors=(), report=None):
        super().__init__(message)
        self.code = code
        self.errors = tuple(errors)
        self.report = report


class CorpusPreflight:
    def __init__(self, source_manifest: dict, documents: tuple[dict, ...],
                 total_visible_words: int, inventory_windows: int,
                 source_manifest_sha256: str):
        self.source_manifest = source_manifest
        self.documents = documents
        self.total_visible_words = total_visible_words
        self.inventory_windows = inventory_windows
        self.source_manifest_sha256 = source_manifest_sha256


class CorpusInventories:
    """The complete, audited inventory layer for one frozen source snapshot."""

    def __init__(self, documents: tuple[dict, ...], index: dict,
                 index_sha256: str):
        self.documents = documents
        self.index = index
        self.index_sha256 = index_sha256


class CorpusPlan:
    """An accepted, normalized partition of every sealed inventory unit."""

    def __init__(self, plan: dict, plan_sha256: str):
        self.plan = plan
        self.plan_sha256 = plan_sha256


def _canonical(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False)
            .encode("utf-8"))


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: pathlib.Path, value: dict) -> str:
    payload = _canonical(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    tmp.write_bytes(payload + b"\n")
    tmp.replace(path)
    return _sha_bytes(payload + b"\n")


def _selection_digest(value: selection.Selection) -> str:
    # This digest binds the accepted selection without copying controller paths
    # into the aggregate manifest or any future model prompt.
    return _sha_bytes(_canonical(value.manifest()))


def _readerview_snapshot(source: pathlib.Path, out_dir: pathlib.Path) -> dict:
    """Build the same reversible reader view used by the single-document path."""
    raw_bytes = pathlib.Path(source).read_bytes()
    raw = raw_bytes.decode("utf-8")
    visible, exclusions, labels = readerview.build(raw)
    blocks = readerview.blocks(visible)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "source.raw").write_bytes(raw_bytes)
    (out_dir / "source.visible.md").write_text(visible, encoding="utf-8")
    source_map = {
        "source_sha256": _sha_bytes(raw_bytes),
        "raw_words": len(raw.split()),
        "visible_words": len(visible.split()),
        "blocks": blocks,
        "labels": labels,
    }
    _write_json(out_dir / "source-map.json", source_map)
    _write_json(out_dir / "exclusions.json", exclusions)
    return source_map


def _window_manifest(document_id: str, part: dict, index: int) -> dict:
    blocks = part["blocks"]
    packet = "\n\n".join(f"[{document_id}:{b['id']}] {b['text']}" for b in blocks)
    return {
        "window_id": f"{document_id}:W{index:03d}",
        "source_ids": [f"{document_id}:{b['id']}" for b in blocks],
        "visible_words": sum(int(b["words"]) for b in blocks),
        "content_sha256": _sha_bytes(packet.encode("utf-8")),
    }


def file_label(relative_path: str) -> str:
    """A context clue for naming, never a path: the file's stem with
    separators turned into spaces. No directory, no extension."""
    stem = pathlib.PurePosixPath(str(relative_path).replace("\\", "/")).stem
    return " ".join(re.sub(r"[_\-.]+", " ", stem).split())


def _document_record(doc: selection.DocumentTarget, source: pathlib.Path,
                     doc_dir: pathlib.Path) -> dict:
    staged = doc_dir / "staged" / "source.original"
    rv_dir = doc_dir / "readerview"
    source_map = _readerview_snapshot(source, rv_dir)
    blocks = source_map["blocks"]
    if not blocks:
        raise CorpusPreflightError(
            f"{doc.id}: reader view contains no substantive visible material",
            code="empty_visible_source")
    parts = ledger.split_sections(
        blocks, part_max=INVENTORY_WINDOW_WORDS)
    windows = [_window_manifest(doc.id, part, index)
               for index, part in enumerate(parts, 1)]
    return {
        "document_id": doc.id,
        "file_label": file_label(doc.relative_path),
        "source_sha256": doc.source_sha256,
        "staged_sha256": _sha_file(staged),
        "visible_sha256": _sha_file(rv_dir / "source.visible.md"),
        "source_map_sha256": _sha_file(rv_dir / "source-map.json"),
        "visible_words": int(source_map["visible_words"]),
        "windows": windows,
    }


def _report(root: pathlib.Path, selection_obj: selection.Selection,
            errors, documents=(), *, status="failed", code="preflight_failed"):
    value = {
        "schema": PREFLIGHT_SCHEMA,
        "status": status,
        "code": code,
        "selection_sha256": _selection_digest(selection_obj),
        "order_digest": selection_obj.order_digest,
        "document_count": len(documents),
        "documents": list(documents),
        "errors": list(errors),
    }
    _write_json(pathlib.Path(root) / "corpus" / "preflight.json", value)
    return value


def preflight(selection_obj: selection.Selection, work_root,
              cancel=None, stage_fn: Callable | None = None) -> CorpusPreflight:
    """Freeze all Corpus source bytes and reject incomplete requests.

    ``stage_fn`` is injectable for no-model tests.  The production default is
    the shared CLI staging function, so PDF extraction, UTF-8 validation, and
    staged-byte hashing remain one implementation for macOS and Windows.
    """
    root = pathlib.Path(work_root).expanduser()
    errors = list(selection_obj.errors)
    documents = tuple(selection_obj.documents)
    if selection_obj.kind != "paths":
        errors.append({"kind": "corpus_requires_paths", "severity": "error"})
    if selection_obj.scope != "corpus":
        errors.append({"kind": "corpus_scope_required", "severity": "error"})
    if not (MIN_DOCUMENTS <= len(documents) <= MAX_DOCUMENTS):
        errors.append({"kind": "document_limit", "severity": "error",
                       "count": len(documents), "minimum": MIN_DOCUMENTS,
                       "maximum": MAX_DOCUMENTS})
    if selection_obj.scope == "corpus":
        try:
            selection.validate_output_pair(
                selection_obj.corpus_outputs,
                (".corpus.summary.md", ".corpus.brief.md"),
                (doc.source_path for doc in documents))
        except ValueError as exc:
            errors.append({"kind": "output_plan_invalid", "severity": "error",
                           "detail": str(exc)})
    if errors:
        report = _report(root, selection_obj, errors)
        raise CorpusPreflightError(
            f"Corpus preflight rejected the selection ({len(errors)} problem(s))",
            code="selection_invalid", errors=errors, report=report)

    if cancel is None:
        class _NoCancel:
            def check(self):
                return None
        cancel = _NoCancel()
    if stage_fn is None:
        # Import lazily: summ_cli imports the selection layer at module load and
        # may import this module only after argument/manifest validation.
        import summ_cli
        stage_fn = summ_cli.stage

    prepared = []
    document_errors = []
    for doc in documents:
        try:
            cancel.check()
            accepted, why = selection.verify_document(doc)
            if not accepted:
                raise CorpusPreflightError(
                    f"{doc.id}: accepted source is no longer unchanged ({why})",
                    code=doc.error_kind or "input_changed")
            doc_dir = root / "documents" / doc.id
            staged_dir = doc_dir / "staged"
            source = stage_fn(doc.source_path, staged_dir, cancel,
                              doc.source_sha256, doc.size_bytes)
            if source is None:
                raise CorpusPreflightError(
                    f"{doc.id}: source could not be staged", code="stage_failed")
            staged = staged_dir / "source.original"
            if _sha_file(staged) != doc.source_sha256:
                # Keeping the original hash assertion adjacent to staging makes
                # a changed PDF impossible to hide.
                raise CorpusPreflightError(
                    f"{doc.id}: staged bytes do not match accepted source",
                    code="input_changed")
            record = _document_record(doc, pathlib.Path(source), doc_dir)
            if record["staged_sha256"] != doc.source_sha256:
                raise CorpusPreflightError(
                    f"{doc.id}: staged source hash changed during preparation",
                    code="input_changed")
            prepared.append(record)
        except CorpusPreflightError as exc:
            document_errors.append({"document_id": doc.id, "kind": exc.code,
                                    "detail": str(exc), "severity": "error"})
            break
        except (OSError, UnicodeDecodeError, ValueError, KeyError) as exc:
            document_errors.append({"document_id": doc.id, "kind": "prepare_failed",
                                    "detail": str(exc)[:240], "severity": "error"})
            break

    if document_errors:
        report = _report(root, selection_obj, document_errors, prepared)
        raise CorpusPreflightError(
            f"Corpus preflight failed before model work ({document_errors[0]['document_id']})",
            code=document_errors[0]["kind"], errors=document_errors, report=report)

    total_words = sum(int(doc["visible_words"]) for doc in prepared)
    windows = sum(len(doc["windows"]) for doc in prepared)
    limit_errors = []
    if not (MIN_VISIBLE_WORDS <= total_words <= MAX_VISIBLE_WORDS):
        limit_errors.append({"kind": "visible_word_limit", "severity": "error",
                             "count": total_words, "minimum": MIN_VISIBLE_WORDS,
                             "maximum": MAX_VISIBLE_WORDS})
    if windows > MAX_INVENTORY_WINDOWS:
        limit_errors.append({"kind": "inventory_window_limit", "severity": "error",
                             "count": windows, "maximum": MAX_INVENTORY_WINDOWS})
    if limit_errors:
        report = _report(root, selection_obj, limit_errors, prepared)
        raise CorpusPreflightError(
            "Corpus preflight exceeded a hard limit; no model call is allowed",
            code=limit_errors[0]["kind"], errors=limit_errors, report=report)

    source_manifest = {
        "schema": SOURCE_SCHEMA,
        "selection_sha256": _selection_digest(selection_obj),
        "order_digest": selection_obj.order_digest,
        "output_plan_sha256": selection_obj.output_plan_sha256 or
                              selection.output_plan_digest(selection_obj),
        "instruction_sha256": os.environ.get("SUMM_INSTRUCTIONS_DIGEST"),
        "visible_words": total_words,
        "inventory_windows": windows,
        "documents": prepared,
    }
    manifest_path = root / "corpus" / "corpus-source.json"
    source_manifest_sha = _write_json(manifest_path, source_manifest)
    _write_json(root / "corpus" / "inventory-index.json", {
        "schema": "summer.corpus-inventory-index.v1",
        "source_manifest_sha256": source_manifest_sha,
        "documents": [{"document_id": d["document_id"],
                        "windows": d["windows"]} for d in prepared],
    })
    report = _report(root, selection_obj, [], prepared, status="succeeded",
                     code="ok")
    report.update({"visible_words": total_words, "inventory_windows": windows})
    _write_json(root / "corpus" / "preflight.json", report)
    return CorpusPreflight(source_manifest, tuple(prepared), total_words,
                           windows, source_manifest_sha)


def validate_limits(document_count: int, visible_words: int,
                    inventory_windows: int) -> tuple[dict, ...]:
    """Pure limit checker used by request-boundary tests."""
    errors = []
    if not MIN_DOCUMENTS <= document_count <= MAX_DOCUMENTS:
        errors.append({"kind": "document_limit", "count": document_count})
    if not MIN_VISIBLE_WORDS <= visible_words <= MAX_VISIBLE_WORDS:
        errors.append({"kind": "visible_word_limit", "count": visible_words})
    if inventory_windows > MAX_INVENTORY_WINDOWS:
        errors.append({"kind": "inventory_window_limit", "count": inventory_windows})
    return tuple(errors)


INVENTORY_DISPOSITIONS = frozenset({
    "exact_repetition", "apparatus", "incidental_example",
    "source_only_detail",
})
_SOURCE_ID = re.compile(r"^(D\d{3}):P\d{4}$")


def _cycle(nodes: dict[str, tuple[str, ...]]) -> bool:
    visiting, finished = set(), set()

    def visit(node):
        if node in visiting:
            return True
        if node in finished:
            return False
        visiting.add(node)
        if any(visit(dep) for dep in nodes.get(node, ())):
            return True
        visiting.remove(node)
        finished.add(node)
        return False

    return any(visit(node) for node in nodes)


def validate_inventory(candidate: dict, document_id: str,
                       source_ids) -> tuple[str, ...]:
    """Check one model inventory against one document's exact source blocks.

    This is the Corpus inventory contract, not the standalone Summary ledger
    contract.  It proves source accounting and local references while leaving
    compression and final-depth decisions to the later corpus synthesis stage.
    """
    expected = tuple(source_ids)
    expected_set = set(expected)
    errors = []
    if not isinstance(candidate, dict):
        return ("inventory is not an object",)
    # Shape tolerance, not semantic tolerance: an omitted empty list or an
    # extra descriptive field is how structured-output routes and capable
    # models answer, and neither changes what the inventory says. Every
    # accounting check below still runs on the content.
    units = candidate.get("units")
    dispositions = candidate.get("dispositions") or []
    unplanned = candidate.get("unplanned_windows") or []
    if not isinstance(unplanned, list):
        errors.append("unplanned_windows must be an array")
    elif unplanned:
        errors.append(f"unplanned inventory windows: {unplanned[:6]}")
    if not isinstance(units, list) or not isinstance(dispositions, list):
        return ("inventory units and dispositions must be arrays",)
    local_ids = [u.get("local_id") for u in units
                 if isinstance(u, dict) and isinstance(u.get("local_id"), str)]
    if len(local_ids) != len(set(local_ids)):
        errors.append("duplicate inventory unit local_id")
    unit_set = set(local_ids)
    if any(not isinstance(u, dict) for u in units):
        errors.append("inventory contains a non-object unit")
    if any(not isinstance(d, dict) for d in dispositions):
        errors.append("inventory contains a non-object disposition")
    represented = []
    for u in units:
        if not isinstance(u, dict):
            continue
        if not isinstance(u.get("local_id"), str) or not u["local_id"].strip():
            errors.append("inventory unit has no local_id")
            continue
        ids = u.get("source_ids")
        capsule = u.get("capsule")
        if not isinstance(ids, list) or not ids:
            errors.append(f"{u.get('local_id')}: unit has no source_ids")
            continue
        if not isinstance(capsule, str) or not capsule.strip():
            errors.append(f"{u.get('local_id')}: empty inventory capsule")
        represented.extend(ids)
        # A unit with nothing to depend on may omit the field; a structured
        # output route dropped the empty list and every unit was rejected as
        # having an "unknown dependency".
        deps = u.get("dependencies") or []
        if not isinstance(deps, list) or any(d not in unit_set for d in deps):
            errors.append(f"{u.get('local_id')}: unknown dependency")
    disposed = []
    for d in dispositions:
        if not isinstance(d, dict):
            continue
        ids = d.get("source_ids")
        if not isinstance(ids, list) or not ids:
            errors.append("disposition has no source_ids")
            continue
        if d.get("disposition") not in INVENTORY_DISPOSITIONS:
            errors.append(f"invalid disposition {d.get('disposition')!r}")
        # A reason is welcome but not accounting: the audit judges whether the
        # drop is safe, and a window must not fail because a model left the
        # explanation out (observed in a live qualification run).
        refs = d.get("represented_by") or []
        if not isinstance(refs, list) or any(ref not in unit_set for ref in refs):
            errors.append("disposition has an unknown represented_by unit")
        disposed.extend(ids)
    all_ids = represented + disposed
    if len(all_ids) != len(set(all_ids)):
        errors.append("source block is represented or dispositioned more than once")
    unknown = sorted(set(all_ids) - expected_set)
    if unknown:
        errors.append(f"unknown source blocks: {unknown[:6]}")
    missing = sorted(expected_set - set(all_ids))
    if missing:
        errors.append(f"unaccounted source blocks: {missing[:6]}")
    prefix = f"{document_id}:"
    foreign = sorted(s for s in all_ids if not isinstance(s, str) or
                     not s.startswith(prefix) or not _SOURCE_ID.fullmatch(s))
    if foreign:
        errors.append(f"cross-document or malformed source IDs: {foreign[:6]}")
    deps_graph = {
        u.get("local_id"): tuple(u.get("dependencies") or [])
        for u in units if isinstance(u, dict) and isinstance(u.get("local_id"), str)
    }
    if _cycle(deps_graph):
        errors.append("inventory dependencies contain a cycle")
    return tuple(errors)


def normalize_inventory(candidate: dict, document_id: str) -> dict:
    """Assign stable document-scoped unit IDs after local validation."""
    units = candidate.get("units") or []
    mapping = {u["local_id"]: f"{document_id}:U{i:03d}"
               for i, u in enumerate(units, 1)}
    normalized_units = []
    for u in units:
        normalized_units.append({
            "unit_id": mapping[u["local_id"]],
            "source_ids": list(u.get("source_ids") or []),
            "dependencies": [mapping[d] for d in (u.get("dependencies") or [])],
            "capsule": u["capsule"],
        })
    normalized_dispositions = []
    for d in candidate.get("dispositions") or []:
        normalized_dispositions.append({
            "source_ids": list(d.get("source_ids") or []),
            "disposition": d.get("disposition"),
            "represented_by": [mapping[r] for r in (d.get("represented_by") or [])],
            "reason": str(d.get("reason") or ""),
        })
    return {
        "schema": INVENTORY_SCHEMA,
        "document_id": document_id,
        "units": normalized_units,
        "dispositions": normalized_dispositions,
        "unplanned_windows": list(candidate.get("unplanned_windows") or []),
    }


PLAN_RELATIONS = frozenset({
    "agreement", "disagreement", "repetition", "qualification",
    "elaboration", "correction", "chronology",
})
PLAN_SCHEMA = "summer.corpus-plan.v2"
_PATHLIKE = re.compile(r"[\\/]|\.(?:md|txt|pdf|docx?|html?)$|\bD\d{3}\b", re.I)


def _known_units(inventory_index: dict) -> dict[str, str]:
    """unit_id -> document_id for every sealed inventory unit."""
    lookup = {}
    for document in inventory_index.get("documents", []):
        for unit in document.get("units", []):
            unit_id = unit.get("unit_id")
            if isinstance(unit_id, str):
                lookup[unit_id] = document.get("document_id")
    return lookup


def coerce_corpus_plan(candidate: dict, inventory_index: dict,
                       file_labels: dict | None = None) -> dict:
    """Map a capable model's reasonable field names and document references
    onto the canonical plan shape before validation. A document may be
    referred to by its opaque ID, its handle, its reference, or its file
    label; a relation may say "type" or "kind" for "relation", "documents"
    for "document_ids", "statement" for "note". Nothing semantic changes.
    A live qualification run showed that a complete, correct plan failed only
    because every document was named by handle instead of D-number."""
    if not isinstance(candidate, dict):
        return candidate
    known = [d.get("document_id") for d in inventory_index.get("documents", [])]
    labels = dict(file_labels or {})
    docs_in = candidate.get("documents") or []
    aliases = {}
    fixed_docs = []
    for i, entry in enumerate(docs_in):
        if not isinstance(entry, dict):
            continue
        entry = dict(entry)
        document_id = (entry.get("document_id") or entry.get("id")
                       or entry.get("document") or entry.get("doc_id"))
        if document_id not in known:
            # Resolve by file label, or by position when the model listed
            # every document once in inventory order.
            # Identity is never inferred from list position: reversing the
            # model's output would silently move one document's reference,
            # contribution, and relations onto another. Only an explicit ID
            # or a unique exact file label resolves an entry; anything else
            # fails validation and spends the one plan correction.
            label = str(entry.get("file_label") or entry.get("label") or "").strip().lower()
            by_label = [k for k, v in labels.items()
                        if v.strip().lower() == label] if label else []
            if len(by_label) == 1:
                document_id = by_label[0]
        entry["document_id"] = document_id
        for key in ("handle", "reference"):
            value = str(entry.get(key) or "").strip()
            if value:
                aliases[value.lower()] = document_id
        if document_id:
            aliases[str(document_id).lower()] = document_id
        fixed_docs.append(entry)

    def doc_list(value):
        out = []
        for item in value or []:
            key = str(item).strip().lower()
            out.append(aliases.get(key, item))
        return out

    def fix_group(items, note_keys):
        fixed = []
        for item in items or []:
            if not isinstance(item, dict):
                continue
            item = dict(item)
            if "document_ids" not in item:
                item["document_ids"] = item.pop("documents", None) or item.pop("docs", None) or []
            item["document_ids"] = doc_list(item.get("document_ids"))
            if "relation" not in item and note_keys == "relation":
                item["relation"] = item.pop("type", None) or item.pop("kind", None)
            if "note" not in item:
                for alt in ("statement", "explanation", "detail", "description"):
                    if item.get(alt):
                        item["note"] = item.pop(alt); break
            if "unit_ids" not in item and note_keys == "relation":
                item["unit_ids"] = item.pop("units", None) or item.pop("inventory_ids", None) or []
            if "theme" not in item and note_keys == "theme":
                item["theme"] = item.pop("title", None) or item.pop("name", None)
            if "issue" not in item and note_keys == "issue":
                item["issue"] = item.pop("question", None) or item.pop("statement", None)
            fixed.append(item)
        return fixed

    return {"documents": fixed_docs,
            "themes": fix_group(candidate.get("themes"), "theme"),
            "relations": fix_group(candidate.get("relations"), "relation"),
            "unresolved": fix_group(candidate.get("unresolved") or candidate.get("open_questions"), "issue")}


def validate_corpus_plan(candidate: dict, inventory_index: dict,
                         source_word_counts=None) -> tuple[str, ...]:
    """Deterministic checks on a relation plan: every document named once
    with a sanitized reference and a unique handle; themes and relations
    cite only known documents and units; a relation spans two documents."""
    errors = []
    if not isinstance(candidate, dict):
        return ("plan is not an object",)
    known_docs = [d.get("document_id") for d in inventory_index.get("documents", [])]
    units = _known_units(inventory_index)
    documents = candidate.get("documents")
    if not isinstance(documents, list):
        return ("plan has no documents list",)
    named, handles = [], []
    for entry in documents:
        if not isinstance(entry, dict):
            errors.append("document entry is not an object"); continue
        document_id = entry.get("document_id")
        if document_id not in known_docs:
            errors.append(f"unknown document {document_id!r}")
        named.append(document_id)
        for field in ("reference", "handle"):
            value = str(entry.get(field) or "").strip()
            if not value:
                errors.append(f"{document_id}: empty {field}")
            elif _PATHLIKE.search(value):
                errors.append(f"{document_id}: {field} looks like a path, "
                              f"extension, or opaque ID: {value!r}")
        handles.append(str(entry.get("handle") or "").strip().lower())
    for document_id in known_docs:
        if named.count(document_id) != 1:
            errors.append(f"{document_id}: named {named.count(document_id)} "
                          "times, expected once")
    if len(set(handles)) != len(handles):
        errors.append("document handles are not unique")
    for theme in candidate.get("themes") or []:
        for document_id in (theme.get("document_ids") or []) if isinstance(theme, dict) else []:
            if document_id not in known_docs:
                errors.append(f"theme cites unknown document {document_id!r}")
    for relation in candidate.get("relations") or []:
        if not isinstance(relation, dict):
            errors.append("relation is not an object"); continue
        kind = relation.get("relation")
        if kind not in PLAN_RELATIONS:
            errors.append(f"unknown relation {kind!r}")
        docs = [d for d in (relation.get("document_ids") or [])]
        if any(d not in known_docs for d in docs):
            errors.append(f"{kind}: cites an unknown document")
        if len(set(docs)) < 2:
            errors.append(f"{kind}: a relation must span at least two documents")
        for unit_id in relation.get("unit_ids") or []:
            if unit_id not in units:
                errors.append(f"{kind}: cites unknown unit {unit_id!r}")
            elif units[unit_id] not in docs:
                errors.append(f"{kind}: unit {unit_id} belongs to a document "
                              "the relation does not cite")
    for issue in candidate.get("unresolved") or []:
        for document_id in (issue.get("document_ids") or []) if isinstance(issue, dict) else []:
            if document_id not in known_docs:
                errors.append(f"unresolved issue cites unknown document {document_id!r}")
    return tuple(errors)


def normalize_corpus_plan(candidate: dict, inventory_index: dict,
                          inventory_sha256: str) -> dict:
    """The accepted relation plan in a stable, sealed shape."""
    order = [d.get("document_id") for d in inventory_index.get("documents", [])]
    by_id = {d.get("document_id"): d for d in candidate.get("documents", [])}
    return {
        "schema": PLAN_SCHEMA,
        "inventory_sha256": inventory_sha256,
        "documents": [{
            "document_id": document_id,
            "reference": str(by_id[document_id].get("reference") or "").strip(),
            "handle": str(by_id[document_id].get("handle") or "").strip(),
            "contribution": str(by_id[document_id].get("contribution") or "").strip(),
        } for document_id in order],
        "themes": [{"theme": str(t.get("theme") or "").strip(),
                    "document_ids": list(t.get("document_ids") or []),
                    "note": str(t.get("note") or "").strip()}
                   for t in candidate.get("themes") or []],
        "relations": [{"relation": r.get("relation"),
                       "document_ids": list(r.get("document_ids") or []),
                       "unit_ids": list(r.get("unit_ids") or []),
                       "note": str(r.get("note") or "").strip()}
                      for r in candidate.get("relations") or []],
        "unresolved": [{"issue": str(u.get("issue") or "").strip(),
                        "document_ids": list(u.get("document_ids") or [])}
                       for u in candidate.get("unresolved") or []],
    }


def _inventory_payload(inventories: CorpusInventories) -> str:
    """Serialize only opaque IDs and source-derived inventory content."""
    value = {
        "schema": "summer.corpus-inventory-index.v1",
        "source_manifest_sha256": inventories.index.get("source_manifest_sha256"),
        "order_digest": inventories.index.get("order_digest"),
        "documents": [],
    }
    for document in inventories.documents:
        value["documents"].append({
            "document_id": document["document_id"],
            "visible_words": document["visible_words"],
            "units": [{"unit_id": unit["unit_id"],
                       "source_ids": unit["source_ids"],
                       "capsule": unit["capsule"]}
                      for unit in document.get("units", [])],
            "dispositions": document.get("dispositions", []),
        })
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)


def _source_word_counts(root: pathlib.Path,
                        prepared: CorpusPreflight) -> dict[str, int]:
    counts = {}
    for document in prepared.documents:
        for source_id, block in _source_blocks(root, document).items():
            counts[source_id] = int(block.get("words") or
                                    len((block.get("text") or "").split()))
    return counts


def build_corpus_plan(inventories: CorpusInventories,
                      prepared: CorpusPreflight, work_root,
                      cancel=None, runner=None) -> CorpusPlan:
    """Create and audit the relation plan: document names, themes,
    relations, and open questions. Evidence for the overview pair."""
    root = pathlib.Path(work_root)
    ms = runner or _runner()
    if cancel is None:
        class _NoCancel:
            def check(self):
                return None
        cancel = _NoCancel()
    plan_dir = root / "corpus" / "plan"
    plan_dir.mkdir(parents=True, exist_ok=True)
    inventory_text = _inventory_payload(inventories)
    labels = "\n".join(f"{d['document_id']}: {d.get('file_label') or '(none)'}"
                       for d in prepared.documents)
    replacements = {"{INVENTORY}": inventory_text, "{LABELS}": labels}
    prompt = _inventory_prompt("corpus-plan-build.txt", replacements)
    (plan_dir / "build.prompt.txt").write_text(prompt, encoding="utf-8")
    candidate = _call_inventory(ms, prompt, plan_dir, "corpus-plan", "plan",
                                PLAN_REQUEST_OPTIONS, cancel=cancel)
    _write_json(plan_dir / "candidate.json", candidate)
    file_labels = {d["document_id"]: d.get("file_label") or "" for d in prepared.documents}
    candidate = coerce_corpus_plan(candidate, inventories.index, file_labels)
    source_counts = _source_word_counts(root, prepared)
    defects_accounting = list(validate_corpus_plan(candidate, inventories.index,
                                                   source_counts))
    defects = list(defects_accounting)
    audit_replacements = {"{INVENTORY}": inventory_text,
                          "{CANDIDATE}": json.dumps(
                              candidate, ensure_ascii=False, sort_keys=True,
                              indent=2)}
    audit_prompt = _inventory_prompt("corpus-plan-audit.txt", audit_replacements)
    (plan_dir / "audit-r1.prompt.txt").write_text(audit_prompt, encoding="utf-8")
    audit = _call_inventory(ms, audit_prompt, plan_dir, "corpus-plan-audit",
                            "audit", PLAN_AUDIT_OPTIONS, cancel=cancel)
    _write_json(plan_dir / "audit-r1.json", audit)
    defects.extend(_audit_blockers(audit))
    if defects:
        cancel.check()
        repair_replacements = {"{INVENTORY}": inventory_text,
                               "{CANDIDATE}": json.dumps(
                                   candidate, ensure_ascii=False,
                                   sort_keys=True, indent=2),
                               "{FINDINGS}": json.dumps(
                                   defects, ensure_ascii=False, indent=2)}
        repair_prompt = _inventory_prompt("corpus-plan-repair.txt",
                                          repair_replacements)
        (plan_dir / "repair.prompt.txt").write_text(repair_prompt,
                                                     encoding="utf-8")
        repaired = _call_inventory(ms, repair_prompt, plan_dir,
                                   "corpus-plan-repair", "repair",
                                   PLAN_REPAIR_OPTIONS, cancel=cancel)
        _write_json(plan_dir / "repair.json", repaired)
        repaired = coerce_corpus_plan(repaired, inventories.index, file_labels)
        remaining = list(validate_corpus_plan(repaired, inventories.index,
                                              source_counts))
        audit_replacements["{CANDIDATE}"] = json.dumps(
            repaired, ensure_ascii=False, sort_keys=True, indent=2)
        audit_prompt = _inventory_prompt("corpus-plan-audit.txt",
                                         audit_replacements)
        audit2 = _call_inventory(ms, audit_prompt, plan_dir,
                                 "corpus-plan-reaudit", "audit",
                                 PLAN_AUDIT_OPTIONS, cancel=cancel)
        _write_json(plan_dir / "audit-r2.json", audit2)
        # One correction, then retain the plan that passes deterministic
        # validation and disclose what the reviewer still objects to.
        if not remaining:
            candidate, open_findings = repaired, _audit_blockers(audit2)
        elif not defects_accounting:
            open_findings = _audit_blockers(audit)
        else:
            raise CorpusPreflightError(
                "no valid relation plan after one correction",
                code="corpus_plan_failed", errors=remaining)
    else:
        open_findings = []
    normalized = normalize_corpus_plan(candidate, inventories.index,
                                       inventories.index_sha256)
    normalized["open_findings"] = list(open_findings)
    plan_hash = _write_json(plan_dir / "plan.json", normalized)
    return CorpusPlan(normalized, plan_hash)


def _source_key(source_id: str):
    match = re.fullmatch(r"D(\d+):P(\d+)", source_id or "")
    return (int(match.group(1)), int(match.group(2))) if match else (10**9, source_id)


def _visible_text(root: pathlib.Path, document_id: str) -> str:
    path = root / "documents" / document_id / "readerview" / "source.visible.md"
    return path.read_text(encoding="utf-8")


def overview_evidence(plan: CorpusPlan, inventories: CorpusInventories) -> str:
    """What the overview writer reads: each document's sealed inventory under
    its reference and handle, then the relation plan. Opaque IDs stay out."""
    names = {d["document_id"]: d for d in plan.plan["documents"]}
    parts = []
    for document in inventories.documents:
        name = names[document["document_id"]]
        capsules = "\n\n".join(unit["capsule"] for unit in document.get("units", []))
        parts.append(f"DOCUMENT: {name['reference']} (short name: {name['handle']})\n"
                     f"{capsules}")
    rel = ["RELATION PLAN (prepared from the documents above; it is evidence, "
           "not a document)"]
    for theme in plan.plan.get("themes", []):
        docs = ", ".join(names[d]["handle"] for d in theme["document_ids"] if d in names)
        rel.append(f"Theme: {theme['theme']} ({docs}). {theme['note']}".strip())
    for relation in plan.plan.get("relations", []):
        docs = ", ".join(names[d]["handle"] for d in relation["document_ids"] if d in names)
        rel.append(f"{relation['relation'].capitalize()} between {docs}: "
                   f"{relation['note']}".strip())
    for name in plan.plan["documents"]:
        if name.get("contribution"):
            rel.append(f"{name['handle']} contributes: {name['contribution']}")
    for issue in plan.plan.get("unresolved", []):
        docs = ", ".join(names[d]["handle"] for d in issue["document_ids"] if d in names)
        rel.append(f"Unresolved ({docs}): {issue['issue']}".strip())
    return "\n\n".join(parts) + "\n\n" + "\n".join(rel)


OVERVIEW_CEILING_SOURCE_WORDS = 12_000
EVIDENCE_SCHEMA = "summer.corpus-evidence.v1"


def _evidence_roots(root: pathlib.Path):
    """Every directory whose bytes are settled before the overview pair."""
    return (root / "documents", root / "corpus" / "plan")


def build_evidence_manifest(work_root) -> str:
    """Hash every retained evidence byte produced before the pair stage:
    staged sources, reader views and source maps, per-window inventory
    candidates, audits, repairs, re-audits, their prompts and route records,
    the normalized inventories, and the relation plan.

    The seal recomputes this and also compares the file set, so a changed
    audit, an appended route record, or an added file fails. Without it the
    seal bound only the structural spine and evidence could be edited after
    the fact."""
    root = pathlib.Path(work_root)
    entries = []
    for base in _evidence_roots(root):
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if path.is_file():
                entries.append({
                    "path": path.relative_to(root).as_posix(),
                    "sha256": _sha_file(path),
                    "bytes": path.stat().st_size})
    return _write_json(root / "corpus" / "evidence-manifest.json",
                       {"schema": EVIDENCE_SCHEMA, "entries": entries})


def upstream_findings(plan: "CorpusPlan", inventories: "CorpusInventories") -> list:
    """Reviewer findings that survived the one correction at an evidence
    stage. They never end a run, but they must reach the final report and
    the seal instead of disappearing between stages."""
    out = []
    for document in inventories.documents:
        for finding in document.get("open_findings") or []:
            out.append(f"inventory {document['document_id']}: {finding}")
    for finding in plan.plan.get("open_findings") or []:
        out.append(f"relation plan: {finding}")
    return out


def write_overview_pair(plan: CorpusPlan, inventories: CorpusInventories,
                        prepared: CorpusPreflight, work_root, cancel=None,
                        runner=None) -> int:
    """Write, review, repair once, and retain the overview pair through the
    shared pair lifecycle. Reviews run per document against that document's
    visible text, then once against the whole evidence for set-level
    relations. Publishes into <work_root>/corpus/pair."""
    import fullsum, shortsum, pair_review
    root = pathlib.Path(work_root)
    ms = runner or _runner()
    if cancel is not None:
        cancel.check()
    pair_dir = root / "corpus" / "pair"
    pair_dir.mkdir(parents=True, exist_ok=True)
    evidence = overview_evidence(plan, inventories)
    (pair_dir / "evidence.txt").write_text(evidence, encoding="utf-8")
    # Settle and bind every earlier evidence byte before the pair is written.
    evidence_manifest_sha256 = build_evidence_manifest(root)
    upstream = upstream_findings(plan, inventories)
    names = {d["document_id"]: d for d in plan.plan["documents"]}
    packets = [
        f"DOCUMENT: {names[d['document_id']]['reference']} "
        f"(short name: {names[d['document_id']]['handle']})\n"
        + _visible_text(root, d["document_id"])
        for d in prepared.documents]
    total = int(prepared.total_visible_words)
    ceilings = fullsum.full_ceilings(min(total, OVERVIEW_CEILING_SOURCE_WORDS))
    rules = (HERE / "prompts" / "corpus-overview-rules.txt").read_text(encoding="utf-8")
    base_prompt = (pair_review.write_template()
                   .replace("{EVIDENCE_KIND}",
                            f"condensed evidence from {len(prepared.documents)} "
                            "documents, followed by a relation plan")
                   .replace("{TASK_RULES}", "\n" + rules)
                   .replace("{D_WORDS}", str(ceilings["detailed"]))
                   .replace("{B_WORDS}", str(ceilings["brief"]))
                   .replace("{SOURCE}", evidence))
    fit = {"fits": True, "route": "corpus", "source_words": total,
           "documents": len(prepared.documents),
           "evidence_words": len(evidence.split()),
           "output_words": ceilings["detailed"] + ceilings["brief"]}
    return fullsum._run_pair(
        source=evidence, review_source=evidence, out_dir=pair_dir, total=total,
        source_sha256=prepared.source_manifest_sha256, ms=ms, ss=shortsum,
        base_prompt=base_prompt, audit_tpl=pair_review.audit_template(),
        ceilings=ceilings, fit=fit, route="corpus", label="corpus",
        initial_stage="corpus-overview", stage_prefix="corpus-overview",
        result_options=fullsum.RESULT_REQUEST_OPTIONS,
        audit_options=fullsum.AUDIT_REQUEST_OPTIONS,
        review_sources=packets,
        # The readings are an overview of the set, not a summary of this
        # document: detail that matters only inside this document is not an
        # omission. One run produced 21 open findings, most of them exactly that.
        local_scope=(
            "The supplied source is one document in a set. Audit the candidate "
            "as an overview of the set, not as a summary of this document. "
            "Report only: (1) a claim attributed to this document that it does "
            "not support; (2) a material qualification, disagreement, date, "
            "quantity, or boundary from this document that the overview states "
            "incorrectly; or (3) this document's distinctive set-level "
            "contribution being absent or materially distorted. Do not report "
            "missing implementation details, examples, flags, file names, "
            "numeric limits, procedures, or subrequirements merely because they "
            "occur in this packet. When documents prescribe different designs, "
            "report that the overview erases or misstates the difference; do "
            "not treat either document as the sole truth. Absence from this "
            "packet is not evidence that a cross-document claim is unsupported; "
            "those claims belong to the global review."),
        global_scope=(
            "The supplied source is the complete compact evidence for the set. "
            "Judge whether the candidate is an overview: check false agreement "
            "or causality, invented chronology, misattribution, erased "
            "disagreement or qualification, omitted or distorted distinctive "
            "contributions, first-document dominance, and serial-digest "
            "structure. A Detailed reading that mostly gives documents "
            "successive mini-summaries is defective even when each mini-summary "
            "is accurate. Do not demand document-summary completeness."),
        upstream_findings=upstream,
        report_extra={"plan_sha256": plan.plan_sha256,
                      "inventory_sha256": inventories.index_sha256,
                      "source_manifest_sha256": prepared.source_manifest_sha256,
                      "evidence_manifest_sha256": evidence_manifest_sha256,
                      "documents": [n["document_id"] for n in plan.plan["documents"]]})


def _runner():
    import importlib.util
    spec = importlib.util.spec_from_file_location("ms", HERE / "mapsum.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _parse_object(ms, raw, stage):
    obj = ms.parse_audit(raw)
    if not isinstance(obj, dict) or not obj:
        raise ValueError(f"{stage}: empty or non-object response")
    return obj


def _source_blocks(root: pathlib.Path, document: dict) -> dict[str, dict]:
    path = (root / "documents" / document["document_id"] / "readerview" /
            "source-map.json")
    value = json.loads(path.read_text(encoding="utf-8"))
    return {f"{document['document_id']}:{block['id']}": block
            for block in value.get("blocks", [])}


def _window_packet(blocks: dict[str, dict], source_ids) -> str:
    try:
        return "\n\n".join(
            f"[{source_id}] {blocks[source_id]['text']}"
            for source_id in source_ids)
    except KeyError as exc:
        raise CorpusPreflightError(
            f"inventory source map is missing {exc.args[0]}",
            code="source_map_incomplete") from exc


def _inventory_prompt(filename: str, replacements: dict[str, str]) -> str:
    prompt = (HERE / "prompts" / filename).read_text(encoding="utf-8")
    for key, value in replacements.items():
        prompt = prompt.replace(key, value)
    return custom_instructions.decorate_prompt(prompt)


def _audit_blockers(audit: dict) -> list[str]:
    """The reviewer's findings as plain sentences; a non-pass verdict with
    no named finding is itself one finding."""
    verdict = audit.get("verdict")
    findings = audit.get("findings") or []
    if verdict == "pass" and not findings:
        return []
    out = []
    for f in findings:
        if isinstance(f, dict):
            text = " ".join(str(f.get(k)) for k in ("issue", "explanation", "problem")
                            if f.get(k))
            if f.get("repair"):
                text += f" Repair: {f['repair']}"
            out.append(text.strip() or json.dumps(f, ensure_ascii=False))
        else:
            out.append(str(f))
    return out or [f"audit verdict={verdict!r} without a named finding"]


def _call_inventory(ms, prompt: str, stage_dir: pathlib.Path, stage: str,
                    role: str, options: dict, cancel=None) -> dict:
    chains = {"plan": ms.PLAN_MODELS, "audit": ms.AUDIT_MODELS,
              "write": ms.MODELS, "repair": ms.REPAIR_MODELS}
    if cancel is not None:
        cancel.check()
    raw = ms.run(prompt, stage_dir, chains[role], stage,
                 validate=lambda value: _parse_object(ms, value, stage),
                 gateway_options=options, role=role)
    return _parse_object(ms, raw, stage)


def _qualified_local(candidate: dict, window_id: str) -> dict:
    """Make local IDs unique before windows are combined into one document."""
    mapping = {u["local_id"]: f"{window_id}:{u['local_id']}"
               for u in candidate.get("units", [])}
    out = {"units": [], "dispositions": [], "unplanned_windows": []}
    for unit in candidate.get("units") or []:
        out["units"].append({
            **unit,
            "local_id": mapping[unit["local_id"]],
            "dependencies": [mapping.get(d, f"{window_id}:{d}")
                             for d in (unit.get("dependencies") or [])],
        })
    for disp in candidate.get("dispositions") or []:
        out["dispositions"].append({
            **disp,
            "represented_by": [mapping.get(r, f"{window_id}:{r}")
                               for r in (disp.get("represented_by") or [])],
        })
    return out


def _inventory_window(ms, root: pathlib.Path, document: dict, window: dict,
                      blocks: dict[str, dict], cancel) -> dict:
    window_id = window["window_id"]
    source_ids = tuple(window["source_ids"])
    packet = _window_packet(blocks, source_ids)
    # IDs use a colon for the model-facing namespace.  Colons are not valid in
    # Windows directory names, so evidence paths use a portable spelling while
    # the retained JSON and prompts keep the canonical ID.
    evidence_window_id = window_id.replace(":", "-")
    stage_dir = (root / "documents" / document["document_id"] /
                 "inventory" / evidence_window_id)
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "source-packet.txt").write_text(packet, encoding="utf-8")
    replacements = {"{DOCUMENT_ID}": document["document_id"],
                    "{WINDOW_ID}": window_id, "{SOURCE_VIEW}": packet}
    prompt = _inventory_prompt("corpus-inventory-build.txt", replacements)
    (stage_dir / "build.prompt.txt").write_text(prompt, encoding="utf-8")
    candidate = _call_inventory(ms, prompt, stage_dir,
                                f"corpus-inventory-plan-{window_id}", "plan",
                                INVENTORY_PLAN_OPTIONS, cancel=cancel)
    _write_json(stage_dir / "candidate.json", candidate)
    defects_accounting = list(validate_inventory(candidate, document["document_id"], source_ids))
    defects = list(defects_accounting)
    audit_replacements = {**replacements, "{CANDIDATE}": json.dumps(
        candidate, ensure_ascii=False, sort_keys=True, indent=2)}
    audit_prompt = _inventory_prompt("corpus-inventory-audit.txt", audit_replacements)
    (stage_dir / "audit-r1.prompt.txt").write_text(audit_prompt, encoding="utf-8")
    audit = _call_inventory(ms, audit_prompt, stage_dir,
                            f"corpus-inventory-audit-{window_id}", "audit",
                            INVENTORY_AUDIT_OPTIONS, cancel=cancel)
    _write_json(stage_dir / "audit-r1.json", audit)
    defects.extend(_audit_blockers(audit))
    if defects:
        cancel.check()
        repair_replacements = {**replacements,
                               "{CANDIDATE}": json.dumps(
                                   candidate, ensure_ascii=False,
                                   sort_keys=True, indent=2),
                               "{FINDINGS}": json.dumps(defects,
                                                         ensure_ascii=False,
                                                         indent=2)}
        repair_prompt = _inventory_prompt("corpus-inventory-repair.txt",
                                          repair_replacements)
        (stage_dir / "repair.prompt.txt").write_text(repair_prompt,
                                                      encoding="utf-8")
        repaired = _call_inventory(ms, repair_prompt, stage_dir,
                                   f"corpus-inventory-repair-{window_id}",
                                   "repair", INVENTORY_REPAIR_OPTIONS,
                                   cancel=cancel)
        _write_json(stage_dir / "repair.json", repaired)
        remaining = list(validate_inventory(
            repaired, document["document_id"], source_ids))
        audit_replacements["{CANDIDATE}"] = json.dumps(
            repaired, ensure_ascii=False, sort_keys=True, indent=2)
        audit_prompt = _inventory_prompt("corpus-inventory-audit.txt",
                                         audit_replacements)
        audit2 = _call_inventory(ms, audit_prompt, stage_dir,
                                 f"corpus-inventory-reaudit-{window_id}",
                                 "audit", INVENTORY_AUDIT_OPTIONS,
                                 cancel=cancel)
        _write_json(stage_dir / "audit-r2.json", audit2)
        # One correction, then retain and disclose (AGENTS.md Defaults). The
        # deterministic accounting must be complete; a semantic finding that
        # survives the correction is carried as an open finding on the
        # evidence, never a reason to end the run. Otherwise one local semantic
        # objection can discard an otherwise usable multi-document result.
        original_ok = not defects_accounting
        repaired_ok = not remaining
        if repaired_ok:
            candidate, open_findings = repaired, _audit_blockers(audit2)
        elif original_ok:
            candidate, open_findings = candidate, _audit_blockers(audit)
        else:
            raise CorpusPreflightError(
                f"{window_id}: no accounting-complete inventory after one "
                f"correction ({'; '.join(remaining[:4])})",
                code="inventory_failed", errors=remaining)
    else:
        open_findings = []
    candidate = _qualified_local(candidate, window_id)
    candidate["open_findings"] = [f"{window_id}: {f}" for f in open_findings]
    _write_json(stage_dir / "inventory.json", candidate)
    return candidate


def build_inventories(prepared: CorpusPreflight, work_root,
                      cancel=None, runner=None, emit: Callable | None = None
                      ) -> CorpusInventories:
    """Build source-complete document inventories after preflight.

    Each window is independently planned and audited.  The function stops on
    the first failed inventory, so no global Corpus planning or synthesis can
    run against incomplete source accounting.
    """
    root = pathlib.Path(work_root)
    ms = runner or _runner()
    if cancel is None:
        class _NoCancel:
            def check(self):
                return None
        cancel = _NoCancel()
    emit = emit or progress.emit
    document_results = []
    for document in prepared.documents:
        cancel.check()
        blocks = _source_blocks(root, document)
        combined = {"units": [], "dispositions": [], "unplanned_windows": []}
        open_findings = []
        windows = document.get("windows") or []
        for index, window in enumerate(windows, 1):
            cancel.check()
            emit("part", kind="inventory", document_id=document["document_id"],
                 index=index, total=len(windows))
            candidate = _inventory_window(ms, root, document, window, blocks,
                                          cancel)
            combined["units"].extend(candidate["units"])
            combined["dispositions"].extend(candidate["dispositions"])
            open_findings.extend(candidate.get("open_findings") or [])
        source_ids = tuple(blocks)
        defects = validate_inventory(combined, document["document_id"], source_ids)
        if defects:
            raise CorpusPreflightError(
                f"{document['document_id']}: combined inventory is incomplete",
                code="inventory_accounting", errors=defects)
        normalized = normalize_inventory(combined, document["document_id"])
        normalized.update({
            "open_findings": open_findings,
            "source_sha256": document["source_sha256"],
            "source_map_sha256": document["source_map_sha256"],
            "visible_sha256": document["visible_sha256"],
            "visible_words": document["visible_words"],
            "windows": document["windows"],
        })
        doc_dir = root / "documents" / document["document_id"] / "inventory"
        doc_hash = _write_json(doc_dir / "inventory.json", normalized)
        normalized["inventory_sha256"] = doc_hash
        document_results.append(normalized)
        emit("document_finished", document_id=document["document_id"],
             index=len(document_results), count=len(prepared.documents),
             status="succeeded", exit_code=0, failure_kind=None)
    index = {
        "schema": INVENTORY_INDEX_SCHEMA,
        "source_manifest_sha256": prepared.source_manifest_sha256,
        "order_digest": prepared.source_manifest.get("order_digest"),
        "documents": document_results,
    }
    index_hash = _write_json(root / "corpus" / "inventory-index.json", index)
    return CorpusInventories(tuple(document_results), index, index_hash)


__all__ = [
    "MIN_DOCUMENTS", "MAX_DOCUMENTS", "MIN_VISIBLE_WORDS",
    "MAX_VISIBLE_WORDS", "INVENTORY_WINDOW_WORDS", "MAX_INVENTORY_WINDOWS",
    "SOURCE_SCHEMA", "CorpusPreflight", "CorpusPreflightError", "preflight",
    "validate_limits", "INVENTORY_DISPOSITIONS", "validate_inventory",
    "normalize_inventory", "CorpusInventories", "build_inventories",
    "PLAN_RELATIONS", "PLAN_SCHEMA", "file_label", "coerce_corpus_plan",
    "validate_corpus_plan",
    "normalize_corpus_plan", "CorpusPlan", "build_corpus_plan",
    "overview_evidence", "write_overview_pair", "build_evidence_manifest",
    "upstream_findings", "EVIDENCE_SCHEMA",
]
