"""Shared structured-output contracts for Summer's model stages.

The server-side schema narrows generation; the existing local parsers remain
authoritative.  Keeping both matters: a gateway can reject or mishandle a
schema, and a syntactically valid object can still violate source fidelity.
"""
from __future__ import annotations

import copy
import json
import re


def _object(properties: dict, required=None) -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(required if required is not None else properties),
    }


STRING = {"type": "string"}
STRINGS = {"type": "array", "items": STRING}
# Stage results must contain actual prose.  This is intentionally narrower
# than STRING: ledger metadata may be empty where its existing contract says
# so, but an empty pair/capsule is not a candidate and must reach the next
# configured producer.
NONEMPTY_TEXT = {"type": "string", "minLength": 1,
                 # Keep the dot-all intent without an inline flag: several
                 # compatible grammar compilers reject `(?s)`, and some apply
                 # regexes as a full match rather than JSON Schema's search.
                 "pattern": r"[\s\S]*\S[\s\S]*"}

UNIT = _object({
    "local_id": STRING,
    "section_id": STRING,
    "source_ids": STRINGS,
    "dependencies": STRINGS,
    "detailed_disposition": {
        "type": "string", "enum": ["required", "optional", "omit"]},
    "brief_disposition": {
        "type": "string", "enum": ["required", "optional", "omit"]},
    "brief_priority": {"type": "integer", "minimum": 1, "maximum": 5},
    "detailed_capsule": STRING,
    "brief_capsule": STRING,
})

DISPOSITION = _object({
    "source_ids": STRINGS,
    "disposition": {"type": "string", "enum": [
        "exact_repetition", "apparatus", "incidental_example",
        "source_only_detail"]},
    "represented_by": STRINGS,
    "reason": STRING,
})

LEDGER_PLAN = _object({
    "section_id": STRING,
    "units": {"type": "array", "items": UNIT},
    "dispositions": {"type": "array", "items": DISPOSITION},
})

LEDGER_REVISION = _object({
    **LEDGER_PLAN["properties"],
    "unresolved": {"type": "array", "items": _object({
        "finding_id": STRING,
        "why_kept": STRING,
    })},
})

LEDGER_FINDING = _object({
    "finding_id": STRING,
    "depth": {"type": "string",
              "enum": ["detailed", "brief", "both", "ledger"]},
    "section_id": STRING,
    "unit_ids": STRINGS,
    "field": {"type": "string", "enum": [
        "coverage", "segmentation", "source_ids", "claim", "status",
        "support", "qualifiers", "dependencies", "detailed_capsule",
        "brief_capsule", "disposition", "voice"]},
    "type": {"type": "string", "enum": [
        "missing_unit", "overmerged_unit", "micro_split", "bad_disposition",
        "unsupported", "overstated", "misattributed", "distorted", "number",
        "definition", "missing_qualifier", "dependency", "voice", "duplicate",
        "contradiction"]},
    "source_ids": STRINGS,
    "exact_bad_text": STRING,
    "explanation": STRING,
    "repair_instruction": STRING,
})

LEDGER_AUDIT = _object({
    "verdict": {"type": "string", "enum": ["pass", "repair", "blocked"]},
    "findings": {"type": "array", "items": LEDGER_FINDING},
})

# Corpus inventory is deliberately not an alias for UNIT/LEDGER_PLAN.  A
# document inventory accounts for source material, but it is not a pair of
# standalone Summary outputs and must not inherit Summary's depth/ratio rules.
INVENTORY_UNIT = _object({
    "local_id": STRING,
    "source_ids": STRINGS,
    "dependencies": STRINGS,
    "capsule": STRING,
})

INVENTORY_DISPOSITION = _object({
    "source_ids": STRINGS,
    "disposition": {"type": "string", "enum": [
        "exact_repetition", "apparatus", "incidental_example",
        "source_only_detail"]},
    "represented_by": STRINGS,
    "reason": STRING,
})

DOCUMENT_INVENTORY = _object({
    "units": {"type": "array", "items": INVENTORY_UNIT},
    "dispositions": {"type": "array", "items": INVENTORY_DISPOSITION},
    "unplanned_windows": STRINGS,
})

INVENTORY_REVISION = _object({
    **DOCUMENT_INVENTORY["properties"],
    "unresolved": {"type": "array", "items": _object({
        "finding_id": STRING,
        "why_kept": STRING,
    })},
})

CORPUS_FINDING = _object({
    "finding_id": STRING,
    "type": {"type": "string", "enum": [
        "missing_source", "duplicate_source", "unknown_source",
        "cross_document", "empty_capsule", "dependency", "unsupported",
        "overstated", "misattributed", "distorted", "number",
        "missing_qualification", "missing_disagreement", "chronology",
        "unsupported_bridge", "false_consensus", "misordered_chronology",
        "misattached_quantity", "voice"]},
    "source_ids": STRINGS,
    "unit_ids": STRINGS,
    "explanation": STRING,
    "repair_instruction": STRING,
})

CORPUS_AUDIT = _object({
    "verdict": {"type": "string", "enum": ["pass", "repair", "blocked"]},
    "findings": {"type": "array", "items": CORPUS_FINDING},
})

CORPUS_PLAN_DOCUMENT = _object({
    "document_id": STRING,
    "reference": NONEMPTY_TEXT,
    "handle": NONEMPTY_TEXT,
    "contribution": STRING,
})

CORPUS_PLAN_THEME = _object({
    "theme": NONEMPTY_TEXT,
    "document_ids": STRINGS,
    "note": STRING,
})

CORPUS_PLAN_RELATION = _object({
    "relation": {"type": "string", "enum": [
        "agreement", "disagreement", "repetition", "qualification",
        "elaboration", "correction", "chronology"]},
    "document_ids": STRINGS,
    "unit_ids": STRINGS,
    "note": STRING,
})

CORPUS_PLAN_UNRESOLVED = _object({
    "issue": NONEMPTY_TEXT,
    "document_ids": STRINGS,
})

# The relation plan: how the documents of a set are named and how they bear
# on one another. Evidence for one overview pair, not a writing assignment.
CORPUS_PLAN = _object({
    "documents": {"type": "array", "items": CORPUS_PLAN_DOCUMENT},
    "themes": {"type": "array", "items": CORPUS_PLAN_THEME},
    "relations": {"type": "array", "items": CORPUS_PLAN_RELATION},
    "unresolved": {"type": "array", "items": CORPUS_PLAN_UNRESOLVED},
})

CORPUS_PLAN_REVISION = _object({
    **CORPUS_PLAN["properties"],
    "unresolved_findings": {"type": "array", "items": _object({
        "finding_id": STRING,
        "why_kept": STRING,
    })},
})

SHORT_RESULT = _object({"detailed": NONEMPTY_TEXT, "brief": NONEMPTY_TEXT})

# One finding object for Quick and Full pair review. Controller-issued
# identifiers are added after parse; the model may name an issued anchor or
# slot but never invents the finding_id.
PAIR_FINDING = _object({
    "kind": {"type": "string",
             "enum": ["reversal", "omission", "addition", "unclear", "brief",
                      "editorial"]},
    "artifact": {"type": "string", "enum": ["detailed", "brief", "pair"]},
    "text": NONEMPTY_TEXT,
    "anchor": STRING,
    "slot": STRING,
    "packet": STRING,
}, required=["kind", "artifact", "text"])

SHORT_AUDIT = _object({
    "verdict": {"type": "string", "enum": ["pass", "revise"]},
    "findings": {"type": "array", "items": PAIR_FINDING},
})

# Full direct pair. Same bodies as the Quick contracts, distinct objects and
# wire names so the Full write/review schema can diverge without touching
# Quick, and gateway evidence names which route produced a call.
FULL_RESULT = _object({"detailed": NONEMPTY_TEXT, "brief": NONEMPTY_TEXT})

# One independently bounded piece of either reading. The controller binds its
# source span, order, and artifact depth; the model supplies prose only.
FULL_READING_PART = _object({"reading": NONEMPTY_TEXT})

# Versioned Full writing contract. The planner identifies discourse units and
# accounts for every controller-issued source segment. Delivery may pack those
# same units into one or several requests, but it never changes their editorial
# obligations or their stable ids.
FULL_PLAN_UNIT = _object({
    "unit_id": STRING,
    "title": NONEMPTY_TEXT,
    "source_ids": STRINGS,
    "topic": NONEMPTY_TEXT,
    "relation_to_previous": STRING,
    "governing_qualifications": STRINGS,
    "detailed_words": {"type": "integer", "minimum": 1},
    # Accept one harmless transport synonym; writing_contract canonicalizes it
    # before identity, packing, or downstream prompts.
    "brief_disposition": {"type": "string",
                          "enum": ["include", "omit", "exclude"]},
    "brief_words": {"type": "integer", "minimum": 0},
})

FULL_PLAN_DISPOSITION = _object({
    "source_ids": STRINGS,
    "disposition": {"type": "string", "enum": [
        "apparatus", "exact_repetition", "incidental_example", "ambiguous"]},
    "represented_by": STRINGS,
    "reason": NONEMPTY_TEXT,
})

FULL_DISCOURSE_PLAN = _object({
    "schema": {"type": "string", "enum": ["summer.writing-plan.v2"]},
    "units": {"type": "array", "items": FULL_PLAN_UNIT},
    "dispositions": {"type": "array", "items": FULL_PLAN_DISPOSITION},
})

# Bounded planning keeps source ownership in leaf plans, then asks adjacent
# reducers to nominate handles for one document-wide Brief selection. Reducers
# never own source ids and therefore cannot silently drop or duplicate them.
PLAN_REDUCTION = _object({
    "summary": NONEMPTY_TEXT,
    "brief_candidates": STRINGS,
    "qualifications": STRINGS,
})

PLAN_BRIEF_SELECTION = _object({
    "brief_units": {"type": "array", "items": _object({
        "handle": STRING,
        "words": {"type": "integer", "minimum": 1},
    })},
})

FULL_WRITING_BLOCK = _object({
    "unit_id": STRING,
    "heading": STRING,
    "join_previous": {"type": "boolean"},
    "paragraphs": {"type": "array", "items": NONEMPTY_TEXT},
}, required=["unit_id", "heading", "paragraphs"])

# The generic pair shape is identical for a whole-pair request and a packed
# request. Assignment-specific factories below pin exact ids and cardinalities;
# the controller still verifies ordered unique coverage locally.
FULL_BLOCK_PAIR = _object({
    "detailed": {"type": "array", "items": FULL_WRITING_BLOCK},
    "brief": {"type": "array", "items": FULL_WRITING_BLOCK},
})


def _nonneg_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractError(f"{name} must be a nonnegative integer")
    return value


def block_pair_schema(packet: dict, *, heading_policy: str | None = None) -> dict:
    """Return a request schema whose counts and id enums match one assignment."""
    def artifact_schema(depth: str) -> dict:
        ids = [str(item["unit_id"]) for item in packet.get(depth) or []]
        count = len(ids)
        spec = {"type": "array", "minItems": count, "maxItems": count}
        if count == 0:
            spec["items"] = copy.deepcopy(FULL_WRITING_BLOCK)
            return spec
        block = copy.deepcopy(FULL_WRITING_BLOCK)
        block["required"] = [
            "unit_id", "heading", "join_previous", "paragraphs"]
        block["properties"]["unit_id"] = {"type": "string", "enum": list(ids)}
        paragraphs = copy.deepcopy(block["properties"]["paragraphs"])
        paragraphs["minItems"] = 1
        block["properties"]["paragraphs"] = paragraphs
        if heading_policy == "none":
            block["properties"]["heading"] = {"type": "string", "enum": [""]}
        spec["items"] = block
        return spec

    return _object({
        "detailed": artifact_schema("detailed"),
        "brief": artifact_schema("brief"),
    })

FULL_WINDOW_RESULT = _object({"capsule": NONEMPTY_TEXT})

FULL_AUDIT = _object({
    "verdict": {"type": "string", "enum": ["pass", "revise"]},
    "findings": {"type": "array", "items": PAIR_FINDING},
})

READABILITY_ASSESSMENT = _object({
    "observation_id": STRING,
    "assessment": {"type": "string", "enum": [
        "acceptable", "unreadable", "uncertain"]},
    "explanation": NONEMPTY_TEXT,
})

FULL_AUDIT_V2 = _object({
    "verdict": {"type": "string", "enum": ["pass", "revise"]},
    "findings": {"type": "array", "items": PAIR_FINDING},
    "readability": {"type": "array", "items": READABILITY_ASSESSMENT},
})

PATCH_EDIT = _object({
    "artifact": {"type": "string", "enum": ["detailed", "brief"]},
    "operation": {"type": "string",
                  "enum": ["insert_before", "insert_after", "replace_range"]},
    "anchor": STRING,
    "start": STRING,
    "end": STRING,
    "anchor_sha256": STRING,
    "range_sha256": STRING,
    "replacement": NONEMPTY_TEXT,
    "finding_ids": STRINGS,
}, required=["artifact", "operation", "replacement", "finding_ids"])

PAIR_PATCH = _object({
    "base_candidate": NONEMPTY_TEXT,
    "edits": {"type": "array", "items": PATCH_EDIT},
})

TEXT_BLOCK = _object({
    "id": STRING,
    "disposition": {"type": "string", "enum": ["keep", "drop_furniture"]},
    "text": STRING,
})

TEXT_CANDIDATE = _object({
    "blocks": {"type": "array", "items": TEXT_BLOCK},
})

# Speech preparation is not a pair review. It keeps sentence findings.
SPEECH_AUDIT = _object({
    "verdict": {"type": "string", "enum": ["pass", "revise"]},
    "findings": {"type": "array", "items": STRING},
})

TEXT_FINDING = _object({"id": STRING, "problem": STRING})
TEXT_AUDIT = _object({
    "verdict": {"type": "string", "enum": ["pass", "revise"]},
    "findings": {"type": "array", "items": TEXT_FINDING},
    "approved_drops": STRINGS,
})


class ContractError(ValueError):
    """The model returned JSON that is not the requested stage contract."""


def document_status(raw) -> str:
    """Classify a JSON text without accepting a prefix or repairing it.

    ``complete`` is one RFC 8259 JSON value with surrounding whitespace.
    ``trailing`` is a complete value followed by more data — that remains a
    rejected response. ``incomplete`` is a cut-off value that can be continued.
    """
    if not isinstance(raw, str) or not raw.strip():
        return "empty"
    try:
        json.loads(raw, object_pairs_hook=_unique_object,
                   parse_constant=_reject_constant)
        return "complete"
    except json.JSONDecodeError as exc:
        if "Extra data" in (exc.msg or ""):
            return "trailing"
        return "incomplete"
    except Exception:
        return "incomplete"


def _balanced_objects(text: str):
    """Yield balanced top-level JSON objects from a noisy CLI response."""
    depth = start = 0
    in_string = escaped = False
    for index, char in enumerate(text or ""):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0:
                yield text[start:index + 1]


def _check(value, schema, path="response"):
    """Small dependency-free validator for the schemas in this module.

    Gateway JSON-schema mode is useful, but it is not authoritative: CLI
    harnesses and gateways may return syntactically valid objects that do not
    satisfy the stage contract. Keeping this validator here makes the local
    publication boundary identical for every harness.
    """
    kind = schema.get("type")
    if kind == "object":
        if not isinstance(value, dict):
            raise ContractError(f"{path}: expected object")
        required = schema.get("required", [])
        missing = [key for key in required if key not in value]
        if missing:
            raise ContractError(f"{path}: missing required field(s): {missing}")
        if schema.get("additionalProperties") is False:
            unknown = sorted(set(value) - set(schema.get("properties", {})))
            if unknown:
                raise ContractError(f"{path}: unknown field(s): {unknown}")
        for key, child in schema.get("properties", {}).items():
            if key in value:
                _check(value[key], child, f"{path}.{key}")
    elif kind == "array":
        if not isinstance(value, list):
            raise ContractError(f"{path}: expected array")
        if "minItems" in schema:
            minimum = _nonneg_int(schema["minItems"], f"{path}.minItems")
            if len(value) < minimum:
                raise ContractError(
                    f"{path}: expected at least {minimum} item(s), got {len(value)}")
        if "maxItems" in schema:
            maximum = _nonneg_int(schema["maxItems"], f"{path}.maxItems")
            if len(value) > maximum:
                raise ContractError(
                    f"{path}: expected at most {maximum} item(s), got {len(value)}")
        items = schema.get("items")
        if items is not None:
            for index, item in enumerate(value):
                _check(item, items, f"{path}[{index}]")
    elif kind == "string":
        if not isinstance(value, str):
            raise ContractError(f"{path}: expected string")
        if "minLength" in schema and len(value) < schema["minLength"]:
            raise ContractError(f"{path}: shorter than minimum length")
        if "pattern" in schema and not re.search(schema["pattern"], value):
            raise ContractError(f"{path}: does not match required text pattern")
    elif kind == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            raise ContractError(f"{path}: expected integer")
        if "minimum" in schema and value < schema["minimum"]:
            raise ContractError(f"{path}: below minimum")
        if "maximum" in schema and value > schema["maximum"]:
            raise ContractError(f"{path}: above maximum")
    elif kind == "boolean":
        if not isinstance(value, bool):
            raise ContractError(f"{path}: expected boolean")
    else:
        raise ContractError(f"{path}: unsupported schema type {kind!r}")
    if "enum" in schema and value not in schema["enum"]:
        raise ContractError(f"{path}: invalid value {value!r}")


def validate(value: dict, schema: dict, stage: str = "response") -> dict:
    """Validate and return one complete stage object."""
    _check(value, schema, stage)
    # A syntactically valid audit with a contradictory verdict is not a pass.
    # This is deliberately part of the local contract, not a convention left
    # for each controller to remember.
    properties = schema.get("properties", {})
    if "verdict" in properties and "findings" in properties:
        verdict = value.get("verdict")
        findings = value.get("findings")
        if verdict == "pass" and findings:
            raise ContractError(f"{stage}: pass audit must have no findings")
        if verdict in {"revise", "repair", "blocked"} and not findings:
            raise ContractError(f"{stage}: {verdict} audit must name findings")
    return value


_ESCAPE = {"\n": "\\n", "\r": "\\r", "\t": "\\t"}


def _escape_controls_in_strings(text: str) -> str:
    """Escape a literal newline, carriage return, or tab inside a JSON string
    value so strict decoding accepts it. Any other unescaped C0 control inside
    a string is a malformed response and is left for strict decoding to
    reject. Outside strings the text is returned unchanged."""
    out, in_string, escaped = [], False, False
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            elif char in _ESCAPE:
                out.append(_ESCAPE[char])
                continue
        elif char == '"':
            in_string = True
        out.append(char)
    return "".join(out)


def _loads(text: str):
    return json.loads(_escape_controls_in_strings(text))


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _reject_constant(value):
    raise ValueError(f"non-standard JSON constant {value!r}")


def parse_exact(raw, schema: dict | None, stage: str) -> dict:
    """Decode one exact gateway JSON document.

    Gateway structured-output stages are a wire contract: one object, no
    duplicate keys, no Markdown fence, no diagnostic prefix or suffix, and no
    non-standard constants.  CLI transports retain ``parse`` below because
    their explicitly scoped wrappers may need normalization before validation.
    """
    if isinstance(raw, dict):
        value = raw
    else:
        if not isinstance(raw, str) or not raw.strip():
            raise ContractError(f"{stage}: empty response")
        try:
            value = json.loads(
                raw, object_pairs_hook=_unique_object,
                parse_constant=_reject_constant)
        except Exception as exc:
            raise ContractError(
                f"{stage}: response is not one exact JSON document: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{stage}: response root must be an object")
    return validate(value, schema, stage) if schema is not None else value


def parse(raw, schema: dict, stage: str) -> dict:
    """Parse one response and require the exact supplied stage schema.

    A literal newline, carriage return, or tab inside a string value is
    accepted: a writer that breaks a paragraph with a real newline inside the
    JSON has answered, and rejecting it as "no JSON object" lost a usable pair.
    Every other unescaped control character is still rejected.
    A small amount of transport normalization is allowed for existing CLI
    harnesses: one surrounding Markdown JSON fence or diagnostic prose around
    one balanced object. No field is inferred, repaired, or defaulted.
    """
    if isinstance(raw, dict):
        return validate(raw, schema, stage)
    if not isinstance(raw, str) or not raw.strip():
        raise ContractError(f"{stage}: empty response")
    candidates = []
    try:
        whole = _loads(raw.strip())
        return validate(whole, schema, stage)
    except Exception:
        pass
    for match in re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.S):
        try:
            candidates.append(_loads(match))
        except Exception:
            pass
    for blob in _balanced_objects(raw):
        try:
            candidates.append(_loads(blob))
        except Exception:
            pass
    unique = []
    seen = set()
    for candidate in candidates:
        try:
            marker = json.dumps(candidate, sort_keys=True,
                                ensure_ascii=False)
        except Exception:
            marker = repr(candidate)
        if marker not in seen:
            seen.add(marker)
            unique.append(candidate)
    valid = []
    errors = []
    for candidate in unique:
        try:
            valid.append(validate(candidate, schema, stage))
        except Exception as exc:
            errors.append(str(exc))
    if len(valid) != 1:
        if len(valid) > 1:
            raise ContractError(f"{stage}: ambiguous response contains multiple "
                                "valid objects")
        detail = errors[-1] if errors else "no JSON object"
        raise ContractError(f"{stage}: invalid response contract: {detail}")
    return valid[0]


def schema_for_stage(stage: str) -> dict | None:
    """Resolve legacy stage names to their exact local contract.

    New Summary callers pass an explicit schema. This mapping hardens the old
    ledger and guarded Corpus callers without making model identity semantic.
    """
    name = str(stage or "").lower()
    if "full" in name and "audit" in name:
        return FULL_AUDIT
    if "short" in name and "audit" in name:
        return SHORT_AUDIT
    if "corpus" in name and "audit" in name:
        return CORPUS_AUDIT
    if re.fullmatch(r"full-batch-(?:detailed|brief)-\d{3}[ab]*", name):
        return FULL_READING_PART
    if re.fullmatch(r"full-window-\d{3}(?:-shorten)?", name):
        return FULL_WINDOW_RESULT
    if name.startswith("short"):
        return SHORT_RESULT
    if name.startswith("full"):
        return FULL_RESULT
    if "corpus" in name and "plan" in name and "revise" in name:
        return CORPUS_PLAN_REVISION
    if "corpus" in name and "plan" in name:
        return CORPUS_PLAN
    if "corpus" in name and "inventory" in name and "revise" in name:
        return INVENTORY_REVISION
    if "corpus" in name and "inventory" in name:
        return DOCUMENT_INVENTORY
    if "revise" in name:
        return LEDGER_REVISION
    if "audit" in name:
        return LEDGER_AUDIT
    if name.startswith("plan") or name.startswith("ledger"):
        return LEDGER_PLAN
    return None


def options(name: str, schema: dict) -> dict:
    """Return one fresh OpenAI-compatible strict JSON-schema request."""
    clean = re.sub(r"[^A-Za-z0-9_-]", "_", str(name))[:64]
    if not clean:
        raise ValueError("structured-output schema needs a name")
    return {"response_format": {"type": "json_schema", "json_schema": {
        "name": clean,
        "strict": True,
        "schema": copy.deepcopy(schema),
    }}}
