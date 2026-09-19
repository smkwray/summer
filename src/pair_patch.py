#!/usr/bin/env python3
"""Controller-issued pair segments and transactional patch application.

The model never invents identifiers. After a base pair exists, this module
splits each reading on existing headings and paragraphs without rewriting
bytes, hashes each span, and later concatenates validated edits onto the
unchanged remainder. A rejected patch leaves the base pair untouched.
"""
from __future__ import annotations

import hashlib
import re

HEADING = re.compile(r"#{1,6}[ \t][^\n]*(?:\n|$)")
OPERATIONS = frozenset({"insert_before", "insert_after", "replace_range"})
ARTIFACTS = ("detailed", "brief")
KINDS = ("reversal", "omission", "addition", "unclear", "brief", "editorial")
MARKUP = re.compile(r"(?m)^[ \t]*([-*+] |\d+\. )|<table\b|\n\|.+\|")


class PatchError(ValueError):
    """The correction is not a safe edit of the enclosed candidate."""


def _sha(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def candidate_identity(detailed: str, brief: str) -> str:
    return "candidate-" + hashlib.sha256(
        ((detailed or "") + "\n\x00\n" + (brief or "")).encode("utf-8")
    ).hexdigest()[:16]


def split_spans(text: str) -> list[tuple[str, int, int]]:
    """Byte offsets of headings and paragraphs. Concatenation reconstructs
    `text` exactly."""
    text = text or ""
    spans = []
    i, n = 0, len(text)
    while i < n:
        if (i == 0 or text[i - 1] == "\n") and text[i] == "#":
            match = HEADING.match(text, i)
            if match:
                spans.append(("heading", i, match.end()))
                i = match.end()
                continue
        j = i
        while j < n:
            if text.startswith("\n\n", j):
                j += 2
                while j < n and text[j] == "\n":
                    j += 1
                break
            if (j > i and text[j - 1] == "\n" and text[j] == "#"
                    and HEADING.match(text, j)):
                break
            j += 1
        spans.append(("paragraph", i, j))
        i = j
    return spans


def segment_artifact(text: str, prefix: str) -> list[dict]:
    """Stable controller IDs for one reading. `prefix` is D or B."""
    counts = {"heading": 0, "paragraph": 0}
    out = []
    for kind, start, end in split_spans(text or ""):
        counts[kind] += 1
        label = "H" if kind == "heading" else "P"
        body = (text or "")[start:end]
        out.append({
            "id": f"{prefix}-{label}{counts[kind]:03d}",
            "kind": kind,
            "start": start,
            "end": end,
            "text": body,
            "sha256": _sha(body),
        })
    return out


def segment_pair(detailed: str, brief: str) -> dict:
    return {
        "candidate": candidate_identity(detailed, brief),
        "detailed": segment_artifact(detailed, "D"),
        "brief": segment_artifact(brief, "B"),
    }


def slots_for(segments: list[dict]) -> list[str]:
    """Insertion points the model may name. Never invent others."""
    if not segments:
        return ["start"]
    out = ["start"]
    for item in segments:
        out.append(f"before:{item['id']}")
        out.append(f"after:{item['id']}")
    return out


def segment_map(detailed: str, brief: str) -> str:
    """Human-readable index injected into the audit prompt."""
    pair = segment_pair(detailed, brief)
    lines = [f"CANDIDATE {pair['candidate']}"]
    for name, prefix in (("DETAILED", "D"), ("BRIEF", "B")):
        items = pair[name.lower()]
        lines.append(f"{name} SEGMENTS")
        if not items:
            lines.append("(empty)")
            continue
        for item in items:
            preview = " ".join(item["text"].split())[:120]
            lines.append(
                f"[{item['id']} sha256={item['sha256']}] {preview}")
        lines.append("INSERTION SLOTS: " + ", ".join(slots_for(items)))
    return "\n".join(lines)


def scoped_segment_context(detailed: str, brief: str, references,
                           neighbor_count: int = 1) -> dict:
    """Return exact referenced spans plus neighbors under their global ids."""
    pair = segment_pair(detailed, brief)
    wanted = set()
    for value in references or ():
        text = str(value or "")
        match = re.search(r"(?:before:|after:)?([DB]-[HP]\d{3})", text)
        if match:
            wanted.add(match.group(1))
    selected = {"detailed": [], "brief": []}
    for artifact in ARTIFACTS:
        items = pair[artifact]
        positions = {item["id"]: index for index, item in enumerate(items)}
        indexes = set()
        for ident in wanted:
            if ident not in positions:
                continue
            index = positions[ident]
            indexes.update(range(max(0, index - neighbor_count),
                                 min(len(items), index + neighbor_count + 1)))
        selected[artifact] = [items[index] for index in sorted(indexes)]
    if not selected["detailed"] and not selected["brief"]:
        raise PatchError("scoped patch has no valid candidate segment reference")
    lines = [f"CANDIDATE {pair['candidate']}"]
    excerpts = {}
    for artifact, label in (("detailed", "DETAILED"), ("brief", "BRIEF")):
        items = selected[artifact]
        lines.append(f"{label} SCOPED SEGMENTS")
        for item in items:
            preview = " ".join(item["text"].split())[:120]
            lines.append(f"[{item['id']} sha256={item['sha256']}] {preview}")
        lines.append("INSERTION SLOTS: " + ", ".join(slots_for(items)))
        excerpts[artifact] = "\n".join(
            f"[{item['id']}]\n{item['text']}" for item in items)
    return {"detailed": excerpts["detailed"],
            "brief": excerpts["brief"], "segments": "\n".join(lines),
            "word_count": sum(len(item["text"].split())
                              for artifact in ARTIFACTS
                              for item in selected[artifact])}


def selected_segment_map(detailed: str, brief: str, selected_ids) -> str:
    """Render a subset of the full candidate map without renumbering ids."""
    pair = segment_pair(detailed, brief)
    selected = set(selected_ids or ())
    lines = [f"CANDIDATE {pair['candidate']}"]
    for artifact, label in (("detailed", "DETAILED"), ("brief", "BRIEF")):
        items = [item for item in pair[artifact] if item["id"] in selected]
        lines.append(f"{label} SEGMENTS")
        for item in items:
            preview = " ".join(item["text"].split())[:120]
            lines.append(f"[{item['id']} sha256={item['sha256']}] {preview}")
        lines.append("INSERTION SLOTS: " + ", ".join(slots_for(items)))
    return "\n".join(lines)


def finding_text(item) -> str:
    """Disclosure sentence. Structured findings keep `text`; a stray string
    is already a sentence from an older mechanical check."""
    if isinstance(item, dict):
        return str(item.get("text") or "").strip()
    return str(item or "").strip()


def render_findings(items, *, packet=None) -> list[str]:
    prefix = f"[source-packet:{packet}] " if packet else ""
    out = []
    for item in items or []:
        text = finding_text(item)
        if text:
            out.append(prefix + text)
    return out


def assign_finding_ids(items, *, start=1) -> list[dict]:
    """Controller IDs. Model-supplied ids are ignored."""
    out = []
    n = start
    for item in items or []:
        if not isinstance(item, dict):
            record = {"kind": "omission", "artifact": "pair",
                      "text": finding_text(item)}
        else:
            record = dict(item)
        record["finding_id"] = f"F-{n:03d}"
        n += 1
        out.append(record)
    return out


def _by_id(segments: list[dict]) -> dict:
    return {item["id"]: item for item in segments}


def _check_markup(text: str, path: str):
    if MARKUP.search(text or ""):
        raise PatchError(f"{path}: replacement introduces list or table markup")


def _restore_outer_newlines(original: str, replacement: str) -> str:
    """Keep separators owned by the controller around a replaced range.

    Paragraph spans include their trailing blank-line delimiter. A model is
    asked for prose, not for storage delimiters, so replacing a paragraph used
    to consume ``\n\n`` and join the next heading as ``sentence.## Heading``.
    Preserve only the outer newline runs from the exact replaced bytes; any
    paragraph breaks inside the replacement remain model-authored.
    """
    text = replacement.strip("\n")
    leading = re.match(r"^\n+", original or "")
    trailing = re.search(r"\n+$", original or "")
    if leading:
        text = leading.group(0) + text
    if trailing:
        text = text + trailing.group(0)
    return text


def _anchor(segments, edit, key):
    ident = str(edit.get(key) or "").strip()
    found = _by_id(segments)
    if ident not in found:
        raise PatchError(f"unknown {key} {ident!r}")
    item = found[ident]
    expected = str(edit.get("anchor_sha256") or edit.get("range_sha256") or "")
    if key == "anchor" and edit.get("anchor_sha256"):
        expected = edit["anchor_sha256"]
        if item["sha256"] != expected:
            raise PatchError(f"stale hash for {ident}")
    return item


def apply_edits(detailed: str, brief: str, payload: dict,
                allowed_finding_ids=None) -> tuple[str, str]:
    """Return a new pair, or raise PatchError. Never mutates the inputs."""
    if not isinstance(payload, dict):
        raise PatchError("patch is not an object")
    pair = segment_pair(detailed, brief)
    claimed = str(payload.get("base_candidate") or "").strip()
    if claimed != pair["candidate"]:
        raise PatchError("base_candidate does not match the enclosed pair")
    edits = payload.get("edits")
    if not isinstance(edits, list) or not edits:
        raise PatchError("patch has no edits")
    bodies = {"detailed": detailed or "", "brief": brief or ""}
    used = {"detailed": [], "brief": []}
    ordered = []
    for index, edit in enumerate(edits):
        if not isinstance(edit, dict):
            raise PatchError(f"edits[{index}]: not an object")
        artifact = edit.get("artifact")
        operation = edit.get("operation")
        if artifact not in ARTIFACTS:
            raise PatchError(f"edits[{index}]: invalid artifact")
        if operation not in OPERATIONS:
            raise PatchError(f"edits[{index}]: invalid operation")
        replacement = edit.get("replacement")
        if not isinstance(replacement, str) or not replacement.strip():
            raise PatchError(f"edits[{index}]: empty replacement")
        _check_markup(replacement, f"edits[{index}]")
        ids = edit.get("finding_ids")
        if not isinstance(ids, list) or not ids:
            raise PatchError(f"edits[{index}]: finding_ids required")
        if any(not isinstance(fid, str) or not fid.strip() for fid in ids):
            raise PatchError(f"edits[{index}]: finding_ids must be strings")
        if allowed_finding_ids is not None:
            allowed = set(allowed_finding_ids)
            unknown = [fid for fid in ids if fid not in allowed]
            if unknown:
                raise PatchError(
                    f"edits[{index}]: unknown finding_ids {unknown}")
        segments = pair[artifact]
        if operation in {"insert_before", "insert_after"}:
            item = _anchor(segments, edit, "anchor")
            if item["sha256"] != str(edit.get("anchor_sha256") or ""):
                raise PatchError(f"edits[{index}]: stale hash for {item['id']}")
            start = item["start"] if operation == "insert_before" else item["end"]
            end = start
        else:
            first = _anchor(segments, edit, "start")
            last = _anchor(segments, edit, "end")
            if first["start"] > last["start"]:
                raise PatchError(f"edits[{index}]: range is reversed")
            expected = _sha((bodies[artifact])[first["start"]:last["end"]])
            if expected != str(edit.get("range_sha256") or ""):
                raise PatchError(f"edits[{index}]: stale range hash")
            start, end = first["start"], last["end"]
        for other_start, other_end in used[artifact]:
            if not (end <= other_start or start >= other_end):
                raise PatchError(f"edits[{index}]: overlapping edit range")
        used[artifact].append((start, end))
        ordered.append((artifact, start, end, replacement, index))
    for artifact in ARTIFACTS:
        pieces = []
        cursor = 0
        body = bodies[artifact]
        # Replacements are half-open [start, end). Insertions are zero-width
        # gaps. At a shared start, gap insertions precede the replacement;
        # insertions at a replacement's end follow it because their start is
        # later. Multiple insertions at one gap keep explicit array order.
        for art, start, end, replacement, _ in sorted(
                (row for row in ordered if row[0] == artifact),
                key=lambda row: (row[1], row[2] != row[1], row[4])):
            if start < cursor:
                raise PatchError(f"{artifact}: overlapping edit range")
            pieces.append(body[cursor:start])
            text = replacement
            if start == end:
                if start > 0 and not body[:start].endswith("\n") and not text.startswith("\n"):
                    text = "\n\n" + text
                if end < len(body) and not body[end:].startswith("\n") and not text.endswith("\n"):
                    text = text.rstrip() + "\n\n"
            else:
                # Range boundaries are serialized by the controller. Preserve
                # their newline delimiters even when the replacement omits
                # them, while allowing the replacement to contain multiple
                # paragraphs or a heading of its own.
                text = _restore_outer_newlines(body[start:end], text)
            pieces.append(text)
            cursor = end
        pieces.append(body[cursor:])
        bodies[artifact] = "".join(pieces)
    if not bodies["detailed"].strip() or not bodies["brief"].strip():
        raise PatchError("patch would drop one reading of the pair")
    return bodies["detailed"], bodies["brief"]
