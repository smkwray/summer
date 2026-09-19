#!/usr/bin/env python3
"""Structural authentication for a sealed multi-document Corpus ledger.

The existing ``mechseal.py`` intentionally authenticates one reader view and
one Summary ledger.  Corpus has a different source chain, so this module binds
the complete source -> inventory -> relation plan -> overview pair chain
without weakening the one-document seal.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import corpus
import pair_review


SEAL_SCHEMA = "summer.corpus-seal.v1"


def _sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: pathlib.Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _source_key(source_id: str):
    match = re.fullmatch(r"D(\d+):P(\d+)", source_id or "")
    return (int(match.group(1)), int(match.group(2))) if match else (10**9, source_id)


def _inventory_candidate(value: dict) -> dict:
    """Convert normalized evidence back to the model-response shape."""
    return {
        "units": [{
            "local_id": unit["unit_id"],
            "source_ids": list(unit.get("source_ids") or []),
            "dependencies": list(unit.get("dependencies") or []),
            "capsule": unit.get("capsule", ""),
        } for unit in value.get("units", [])],
        "dispositions": [{
            "source_ids": list(disposition.get("source_ids") or []),
            "disposition": disposition.get("disposition"),
            "represented_by": list(disposition.get("represented_by") or []),
            "reason": disposition.get("reason", ""),
        } for disposition in value.get("dispositions", [])],
        "unplanned_windows": list(value.get("unplanned_windows") or []),
    }


def _plan_candidate(value: dict) -> dict:
    return {key: value.get(key) for key in
            ("documents", "themes", "relations", "unresolved")}


def _source_counts(root: pathlib.Path, documents) -> dict[str, int]:
    counts = {}
    for document in documents:
        blocks = corpus._source_blocks(root, document)
        for source_id, block in blocks.items():
            counts[source_id] = int(block.get("words") or
                                    len((block.get("text") or "").split()))
    return counts


def _fail(failures, text):
    failures.append(str(text))


def check(root: pathlib.Path) -> tuple[list[str], dict]:
    """Return structural failures and summary info without writing markers.

    The chain is source manifest -> per-document inventories -> relation
    plan -> overview pair report and artifact bytes. Every link is bound by
    hash; the plan is re-validated against the sealed inventory; the pair
    report must name the exact plan and inventory bytes it was written from.
    """
    root = pathlib.Path(root)
    croot = root / "corpus"
    failures = []
    required = {name: croot / name for name in (
        "corpus-source.json", "inventory-index.json", "plan/plan.json",
        "evidence-manifest.json",
        "pair/full-report.json", "pair/detailed.md", "pair/brief.md")}
    if any(not path.is_file() for path in required.values()):
        missing = [name for name, path in required.items() if not path.is_file()]
        return [f"missing Corpus evidence: {missing}"], {}
    try:
        source = _read(required["corpus-source.json"])
        index = _read(required["inventory-index.json"])
        plan = _read(required["plan/plan.json"])
        evidence = _read(required["evidence-manifest.json"])
        report = _read(required["pair/full-report.json"])
    except Exception as exc:
        return [f"Corpus evidence is unreadable: {str(exc)[:160]}"], {}
    if source.get("schema") != corpus.SOURCE_SCHEMA:
        _fail(failures, "wrong Corpus source manifest schema")
    if index.get("schema") != corpus.INVENTORY_INDEX_SCHEMA:
        _fail(failures, "wrong Corpus inventory index schema")
    if plan.get("schema") != corpus.PLAN_SCHEMA:
        _fail(failures, "wrong Corpus plan schema")
    if evidence.get("schema") != corpus.EVIDENCE_SCHEMA:
        _fail(failures, "wrong Corpus evidence manifest schema")
    source_hash = _sha(required["corpus-source.json"])
    evidence_hash = _sha(required["evidence-manifest.json"])
    index_hash = _sha(required["inventory-index.json"])
    plan_hash = _sha(required["plan/plan.json"])
    detailed_hash = _sha(required["pair/detailed.md"])
    brief_hash = _sha(required["pair/brief.md"])
    if source.get("documents") is None or index.get("documents") is None:
        _fail(failures, "source or inventory index has no documents")
    if plan.get("inventory_sha256") != index_hash:
        _fail(failures, "plan does not bind the inventory index bytes")
    if report.get("plan_sha256") != plan_hash:
        _fail(failures, "pair report does not bind the plan bytes")
    if report.get("inventory_sha256") != index_hash:
        _fail(failures, "pair report does not bind the inventory index bytes")
    if report.get("source_manifest_sha256") != source_hash:
        _fail(failures, "pair report does not bind the source manifest bytes")
    if report.get("evidence_manifest_sha256") != evidence_hash:
        _fail(failures, "pair report does not bind the evidence manifest bytes")
    # Every evidence byte produced before the pair: candidates, audits,
    # repairs, re-audits, prompts, route records, reader views, source maps,
    # staged sources, and the plan. A changed audit, an appended call record,
    # or an added file all fail here.
    listed = {}
    for entry in evidence.get("entries") or []:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            _fail(failures, "malformed evidence manifest entry")
            continue
        listed[entry["path"]] = entry.get("sha256")
        path = root / entry["path"]
        if not path.is_file():
            _fail(failures, f"evidence is missing: {entry['path']}")
        elif _sha(path) != entry.get("sha256"):
            _fail(failures, f"evidence changed after sealing: {entry['path']}")
    present = {p.relative_to(root).as_posix()
               for base in corpus._evidence_roots(root) if base.is_dir()
               for p in base.rglob("*") if p.is_file()}
    for extra in sorted(present - set(listed)):
        _fail(failures, f"evidence added after sealing: {extra}")
    if report.get("route") != "corpus" or report.get("status") not in {
            "pass", "open_findings", "review_unavailable"}:
        _fail(failures, "pair report is not a Corpus overview report")
    readings = {}
    for name, path in (("detailed", required["pair/detailed.md"]),
                       ("brief", required["pair/brief.md"])):
        text = path.read_text(encoding="utf-8")
        readings[name] = text
        if not text.strip():
            _fail(failures, f"{name} reading is empty")
        if report.get(f"{name}_words") != len(text.split()):
            _fail(failures, f"{name} reading does not match its report")
        # Word counts do not pin bytes. These do.
        if report.get(f"{name}_sha256") != _sha(path):
            _fail(failures, f"{name} reading is not the bytes the report names")
    identity = pair_review.candidate_identity(
        readings["detailed"].rstrip("\n"), readings["brief"].rstrip("\n"))
    if report.get("candidate") != identity:
        _fail(failures, "published readings are not the selected candidate")
    source_docs = {d.get("document_id"): d for d in source.get("documents", [])}
    index_docs = {d.get("document_id"): d for d in index.get("documents", [])}
    if set(source_docs) != set(index_docs):
        _fail(failures, "source and inventory document sets differ")
    if list(report.get("documents") or []) != list(source_docs):
        _fail(failures, "pair report does not name every source document once")
    inventory_units = 0
    for document_id, source_doc in source_docs.items():
        index_doc = index_docs.get(document_id)
        if not isinstance(document_id, str) or index_doc is None:
            continue
        inv_path = root / "documents" / document_id / "inventory" / "inventory.json"
        rv_path = root / "documents" / document_id / "readerview" / "source-map.json"
        if not inv_path.is_file() or not rv_path.is_file():
            _fail(failures, f"{document_id}: missing inventory or source map")
            continue
        try:
            inventory = _read(inv_path)
            smap = _read(rv_path)
        except Exception as exc:
            _fail(failures, f"{document_id}: unreadable source evidence ({str(exc)[:100]})")
            continue
        if index_doc.get("inventory_sha256") != _sha(inv_path):
            _fail(failures, f"{document_id}: inventory hash mismatch")
        if source_doc.get("source_sha256") != inventory.get("source_sha256"):
            _fail(failures, f"{document_id}: original source hash mismatch")
        if source_doc.get("source_map_sha256") != _sha(rv_path):
            _fail(failures, f"{document_id}: source-map hash mismatch")
        source_ids = [f"{document_id}:{block['id']}"
                      for block in smap.get("blocks", [])]
        failures.extend(
            f"{document_id}: {error}" for error in corpus.validate_inventory(
                _inventory_candidate(inventory), document_id, source_ids))
        blocks = {f"{document_id}:{block['id']}": block
                  for block in smap.get("blocks", [])}
        for window in source_doc.get("windows", []):
            packet = corpus._window_packet(blocks, window.get("source_ids") or [])
            actual = hashlib.sha256(packet.encode("utf-8")).hexdigest()
            if actual != window.get("content_sha256"):
                _fail(failures, f"{document_id}: window hash mismatch")
        inventory_units += len(inventory.get("units", []))
    failures.extend(corpus.validate_corpus_plan(_plan_candidate(plan), index))
    info = {"documents": len(source_docs), "inventory_units": inventory_units,
            "relations": len(plan.get("relations") or []),
            "source_manifest_sha256": source_hash, "inventory_sha256": index_hash,
            "plan_sha256": plan_hash, "evidence_manifest_sha256": evidence_hash,
            "evidence_files": len(listed),
            "detailed_sha256": detailed_hash, "brief_sha256": brief_hash,
            "pair_report_sha256": _sha(required["pair/full-report.json"]),
            "pair_status": report.get("status"),
            # The seal answers "are these the audited bytes"; it never answers
            # "is the reading clean". Findings travel separately.
            "quality_status": report.get("quality_status") or report.get("status"),
            "pair_findings": len(report.get("findings") or []),
            "upstream_findings": len(report.get("upstream_findings") or [])}
    return failures, info


def seal(root: pathlib.Path) -> dict:
    """Write the Corpus seal and publication authorization only on a clean check."""
    root = pathlib.Path(root)
    failures, info = check(root)
    croot = root / "corpus"
    value = {"schema": SEAL_SCHEMA, "passed": not failures,
             "failures": failures, **info}
    corpus._write_json(croot / "corpus-seal.json", value)
    if failures:
        # A later failed check must not leave an earlier authorization
        # standing: the markers are publication authority, not history.
        for name in ("CORPUS_SEAL", "SEALED"):
            (croot / name).unlink(missing_ok=True)
        corpus._write_json(croot / "status.json", {
            "status": "failed", "kind": "corpus", "seal_status": "failed",
            "quality_status": info.get("quality_status") or "unknown",
            "quarantine": {}})
        for failure in failures[:10]:
            print(f"[corpus-seal] {failure}", file=sys.stderr, flush=True)
        return value
    (croot / "CORPUS_SEAL").write_text("1\n", encoding="utf-8")
    (croot / "SEALED").write_text("1\n", encoding="utf-8")
    corpus._write_json(croot / "status.json", {
        # Authenticated bytes and clean prose are different questions.
        "status": "verified", "kind": "corpus", "seal_status": "verified",
        "quality_status": info["quality_status"],
        "pair_findings": info["pair_findings"],
        "upstream_findings": info["upstream_findings"],
        "detailed_sha256": info["detailed_sha256"],
        "brief_sha256": info["brief_sha256"], "quarantine": {}})
    return value


if __name__ == "__main__":
    path = pathlib.Path(sys.argv[1])
    result = seal(path)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 3)
