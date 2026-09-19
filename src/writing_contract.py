#!/usr/bin/env python3
"""Pure mechanics for Summer's versioned model-agnostic writing contract.

Models decide discourse units, selection, prose, paragraph boundaries, and
editorial repairs. The controller assigns source identities, validates complete
coverage and budgets, packs the same obligations to declared route capacities,
renders blocks, and detects passages that require editorial assessment.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import re

import json_contract
import pair_patch
import pair_review


HERE = pathlib.Path(__file__).parent
PROMPTS = HERE / "prompts"
SCHEMA = "summer.writing-plan.v2"
ENVELOPE_SCHEMA = "summer.writing-plan-envelope.v1"
RECOVERY_SCHEMA = "summer.writer-recovery.v1"
RECOVERY_WAVES = ("fresh", "rejected", "repacked", "complete", "blocked")

PARAGRAPH_ASSESS_WORDS = 350
PARAGRAPH_ASSESS_SENTENCES = 12
PARAGRAPH_BLOCK_WORDS = 800
PARAGRAPH_BLOCK_SENTENCES = 30
SENTENCE_ASSESS_WORDS = 80
WINDOW_DENSE_WORDS = 45
WINDOW_FRAGMENT_WORDS = 12
PLAN_LEAF_MAX_IDS = 32
PLAN_REDUCE_FANIN = 4


class WritingContractError(ValueError):
    """The plan, packed assignment, or model response violates v2."""


class WriterResponseError(WritingContractError):
    """A writer response failed the assignment contract with typed evidence."""

    def __init__(self, message: str, *, code: str, expected_keys,
                 observed_keys, missing_keys, extra_keys, invalid_blocks):
        super().__init__(message)
        self.code = code
        self.expected_keys = [tuple(item) for item in expected_keys]
        self.observed_keys = [tuple(item) for item in observed_keys]
        self.missing_keys = [tuple(item) for item in missing_keys]
        self.extra_keys = [tuple(item) for item in extra_keys]
        self.invalid_blocks = list(invalid_blocks)


def _sha(value: str) -> str:
    return hashlib.sha256((value or "").encode("utf-8")).hexdigest()


def source_segments(source: str) -> list[dict]:
    """Controller-issued source blocks whose text reconstructs source exactly."""
    items = pair_patch.segment_artifact(source or "", "S")
    for item in items:
        item["words"] = len(item["text"].split())
    return items


def source_segment_text(segments: list[dict]) -> str:
    return "\n".join(
        f"[{item['id']} sha256={item['sha256']}]\n{item['text']}"
        for item in segments)


def plan_prompt(source: str, ceilings: dict) -> str:
    template = (PROMPTS / "full-discourse-plan.txt").read_text()
    return (template
            .replace("{D_WORDS}", str(int(ceilings["detailed"])))
            .replace("{B_WORDS}", str(int(ceilings["brief"])))
            .replace("{SOURCE_SEGMENTS}",
                     source_segment_text(source_segments(source))))


def _weighted_limits(weights: list[int], total: int) -> list[int]:
    """Allocate one positive integer ceiling per bounded planning leaf."""
    if not weights or total < len(weights):
        raise WritingContractError(
            "writing ceiling cannot fund every bounded planning leaf")
    base = [1] * len(weights)
    remaining = total - len(weights)
    weight_total = sum(max(1, int(value)) for value in weights)
    raw = [remaining * max(1, int(value)) / weight_total for value in weights]
    floors = [int(value) for value in raw]
    out = [base[i] + floors[i] for i in range(len(weights))]
    for index in sorted(range(len(weights)),
                        key=lambda i: raw[i] - floors[i], reverse=True)[:
                            remaining - sum(floors)]:
        out[index] += 1
    return out


def planning_leaves(source: str, ceilings: dict,
                    max_ids: int = PLAN_LEAF_MAX_IDS) -> list[dict]:
    """Partition canonical source ownership into contiguous bounded leaves."""
    if isinstance(max_ids, bool) or not isinstance(max_ids, int) or max_ids < 1:
        raise WritingContractError("planning leaf id limit must be positive")
    segments = source_segments(source)
    chunks = [segments[i:i + max_ids]
              for i in range(0, len(segments), max_ids)]
    if not chunks:
        raise WritingContractError("source has no planning segments")
    weights = [sum(item["words"] for item in chunk) for chunk in chunks]
    detailed = _weighted_limits(weights, int(ceilings["detailed"]))
    brief = _weighted_limits(weights, int(ceilings["brief"]))
    return [{
        "leaf_id": f"L{index + 1:03d}",
        "segments": chunk,
        "source_ids": [item["id"] for item in chunk],
        "detailed_words": detailed[index],
        "brief_words": brief[index],
    } for index, chunk in enumerate(chunks)]


def planning_leaf_prompt(leaf: dict) -> str:
    """Compose one ownership-preserving plan request without repeated hashes."""
    evidence = "\n".join(
        f"[{item['id']}]\n{item['text']}" for item in leaf["segments"])
    template = (PROMPTS / "full-discourse-plan.txt").read_text()
    framing = (
        f"This is bounded planning leaf {leaf['leaf_id']}. Account for exactly "
        "the supplied core source ids. Unit ids are local U001..Un; a later "
        "controller pass assigns document-wide ids. Brief choices here are "
        "nominations, not the final document-wide Brief selection. The TOTAL "
        f"of detailed_words across every unit must be at most "
        f"{int(leaf['detailed_words'])}; the TOTAL of brief_words must be at "
        f"most {int(leaf['brief_words'])}. The leaf boundary is transport, not "
        "a discourse boundary. Preserve the 400-800-word planning scale where "
        "this leaf's allocation permits it: group adjacent source ids into the "
        "fewest coherent units the material needs rather than making one unit "
        "per source paragraph, section, claim, or id. If the entire leaf has "
        "less than 400 Detailed words, it will usually be one unit; a genuinely "
        "distinct smaller unit is still allowed. relation_to_previous must be a "
        "JSON string; use 'opens this leaf' for the first unit, never null.\n\n")
    return framing + (template
        .replace("{D_WORDS}", str(int(leaf["detailed_words"])))
        .replace("{B_WORDS}", str(int(leaf["brief_words"])))
        .replace("{SOURCE_SEGMENTS}", evidence))


def split_planning_leaf(leaf: dict) -> list[dict]:
    """Bisect one oversized leaf while conserving its local word ceilings."""
    segments = list(leaf.get("segments") or [])
    if len(segments) < 2:
        return []
    middle = len(segments) // 2
    chunks = [segments[:middle], segments[middle:]]
    weights = [sum(item["words"] for item in chunk) for chunk in chunks]
    try:
        detailed = _weighted_limits(weights, int(leaf["detailed_words"]))
        brief = _weighted_limits(weights, int(leaf["brief_words"]))
    except WritingContractError:
        return []
    return [{
        "leaf_id": str(leaf["leaf_id"]) + suffix,
        "segments": chunk,
        "source_ids": [item["id"] for item in chunk],
        "detailed_words": detailed[index],
        "brief_words": brief[index],
    } for index, (suffix, chunk) in enumerate(zip(("a", "b"), chunks))]


def validate_plan_fragment(plan: dict, leaf: dict) -> dict:
    """Validate exact ownership and local budgets for one planning leaf."""
    try:
        json_contract.validate(plan, json_contract.FULL_DISCOURSE_PLAN,
                               leaf["leaf_id"])
    except Exception as exc:
        raise WritingContractError(str(exc)) from exc
    value = json.loads(json.dumps(plan))
    for unit in value.get("units") or []:
        if unit.get("brief_disposition") == "exclude":
            unit["brief_disposition"] = "omit"
    units = list(value.get("units") or [])
    if not units:
        raise WritingContractError(f"{leaf['leaf_id']}: plan has no units")
    local_ids = [f"U{i:03d}" for i in range(1, len(units) + 1)]
    if [item.get("unit_id") for item in units] != local_ids:
        raise WritingContractError(
            f"{leaf['leaf_id']}: local unit ids must be U001..Un")
    expected = list(leaf["source_ids"])
    order = {ident: index for index, ident in enumerate(expected)}
    claims = []
    prior = -1
    for unit in units:
        ids = list(unit.get("source_ids") or [])
        if not ids or any(ident not in order for ident in ids):
            raise WritingContractError(
                f"{leaf['leaf_id']}: unit has empty or unknown source ids")
        positions = [order[ident] for ident in ids]
        if positions != sorted(positions) or positions[0] <= prior:
            raise WritingContractError(
                f"{leaf['leaf_id']}: units do not follow source order")
        prior = positions[-1]
        claims.extend(ids)
        included = unit.get("brief_disposition") == "include"
        if included != (int(unit.get("brief_words") or 0) > 0):
            raise WritingContractError(
                f"{leaf['leaf_id']}: inconsistent Brief nomination")
    known_units = set(local_ids)
    for disposition in value.get("dispositions") or []:
        ids = list(disposition.get("source_ids") or [])
        if not ids or any(ident not in order for ident in ids):
            raise WritingContractError(
                f"{leaf['leaf_id']}: disposition has unknown source ids")
        if set(disposition.get("represented_by") or []) - known_units:
            raise WritingContractError(
                f"{leaf['leaf_id']}: disposition names an unknown unit")
        claims.extend(ids)
    if len(claims) != len(set(claims)) or set(claims) != set(expected):
        raise WritingContractError(
            f"{leaf['leaf_id']}: source ids must be owned exactly once in order")
    if sum(int(item["detailed_words"]) for item in units) > int(
            leaf["detailed_words"]):
        raise WritingContractError(
            f"{leaf['leaf_id']}: Detailed allocations exceed leaf ceiling")
    if sum(int(item["brief_words"]) for item in units) > int(
            leaf["brief_words"]):
        raise WritingContractError(
            f"{leaf['leaf_id']}: Brief nominations exceed leaf ceiling")
    return value


def _fragment_cards(leaf: dict, plan: dict) -> list[dict]:
    return [{
        "handle": f"{leaf['leaf_id']}-{unit['unit_id']}",
        "title": unit["title"],
        "topic": unit["topic"],
        "relation_to_previous": unit.get("relation_to_previous") or "",
        "governing_qualifications": list(
            unit.get("governing_qualifications") or []),
        "brief_nomination": unit.get("brief_disposition") == "include",
        "suggested_brief_words": int(unit.get("brief_words") or 0),
    } for unit in plan["units"]]


def planning_reduction_groups(leaves_and_plans: list[tuple[dict, dict]],
                              fanin: int = PLAN_REDUCE_FANIN) -> list[list[dict]]:
    """Return adjacent, bounded groups of leaf cards for model reduction."""
    if isinstance(fanin, bool) or not isinstance(fanin, int) or fanin < 2:
        raise WritingContractError("planning reduction fan-in must be at least two")
    nodes = [{"leaf_id": leaf["leaf_id"],
              "cards": _fragment_cards(leaf, plan)}
             for leaf, plan in leaves_and_plans]
    return [nodes[i:i + fanin] for i in range(0, len(nodes), fanin)]


def planning_reduction_prompt(group: list[dict], node_id: str) -> str:
    return (
        "Reduce these adjacent planning leaves into one compact document node. "
        "Return exactly one JSON object with summary, brief_candidates, and "
        "qualifications. brief_candidates may name at most eight supplied "
        "handles. Preserve conflicting results and scope-changing caveats; do "
        "not invent handles. This node does not own source ids.\n\nNODE "
        + node_id + "\n" + json.dumps(group, indent=2, ensure_ascii=False))


def validate_plan_reduction(value: dict, group: list[dict], node_id: str) -> dict:
    try:
        json_contract.validate(value, json_contract.PLAN_REDUCTION, node_id)
    except Exception as exc:
        raise WritingContractError(str(exc)) from exc
    known = {card["handle"] for leaf in group for card in leaf["cards"]}
    candidates = list(value.get("brief_candidates") or [])
    if (not candidates or len(candidates) > 8
            or len(candidates) != len(set(candidates))):
        raise WritingContractError(f"{node_id}: invalid candidate count")
    unknown = sorted(set(candidates) - known)
    if unknown:
        raise WritingContractError(f"{node_id}: unknown handles {unknown[:4]}")
    return json.loads(json.dumps(value))


def planning_selection_prompt(nodes: list[dict], cards: list[dict],
                              brief_words: int) -> str:
    nominated = {handle for node in nodes
                 for handle in node.get("brief_candidates") or []}
    choices = [card for card in cards if card["handle"] in nominated]
    return (
        "Select the document-wide Brief from these bounded adjacent reductions. "
        "Return exactly one JSON object with brief_units, each containing one "
        "supplied handle and a positive word allocation. The allocations must "
        f"sum to at most {int(brief_words)} words. Preserve the central results "
        "and their governing qualifications; do not select proportionally by "
        "source region.\n\nREDUCTIONS:\n" +
        json.dumps(nodes, indent=2, ensure_ascii=False) +
        "\n\nELIGIBLE UNIT CARDS:\n" +
        json.dumps(choices, indent=2, ensure_ascii=False))


def validate_plan_selection(value: dict, cards: list[dict],
                            brief_words: int) -> dict:
    try:
        json_contract.validate(value, json_contract.PLAN_BRIEF_SELECTION,
                               "plan-brief-selection")
    except Exception as exc:
        raise WritingContractError(str(exc)) from exc
    items = list(value.get("brief_units") or [])
    handles = [item.get("handle") for item in items]
    known = {item["handle"] for item in cards}
    if not items or len(handles) != len(set(handles)) or set(handles) - known:
        raise WritingContractError("global Brief selection has invalid handles")
    if sum(int(item["words"]) for item in items) > int(brief_words):
        raise WritingContractError("global Brief allocation exceeds its ceiling")
    return json.loads(json.dumps(value))


def assemble_bounded_plan(source: str, ceilings: dict,
                          leaves_and_plans: list[tuple[dict, dict]],
                          selection: dict) -> dict:
    """Assign global unit ids and materialize one ordinary validated plan."""
    cards = [card for leaf, plan in leaves_and_plans
             for card in _fragment_cards(leaf, plan)]
    selected = {item["handle"]: int(item["words"])
                for item in selection.get("brief_units") or []}
    if set(selected) - {card["handle"] for card in cards}:
        raise WritingContractError("global Brief selection has unknown handles")
    units = []
    dispositions = []
    handle_map = {}
    for leaf, fragment in leaves_and_plans:
        for local in fragment["units"]:
            handle = f"{leaf['leaf_id']}-{local['unit_id']}"
            global_id = f"U{len(units) + 1:03d}"
            handle_map[handle] = global_id
            units.append({
                **local,
                "unit_id": global_id,
                "brief_disposition": ("include" if handle in selected else "omit"),
                "brief_words": selected.get(handle, 0),
            })
    for leaf, fragment in leaves_and_plans:
        for item in fragment.get("dispositions") or []:
            dispositions.append({
                **item,
                "represented_by": [handle_map[f"{leaf['leaf_id']}-{local}"]
                                   for local in item.get("represented_by") or []],
            })
    return validate_plan({
        "schema": SCHEMA, "units": units, "dispositions": dispositions,
    }, source, ceilings)


def validate_plan(plan: dict, source: str, ceilings: dict) -> dict:
    """Require exact source accounting, order, ids, and total allocations."""
    try:
        json_contract.validate(plan, json_contract.FULL_DISCOURSE_PLAN,
                               "full-discourse-plan")
    except Exception as exc:
        raise WritingContractError(str(exc)) from exc
    # Canonical parser tolerance, not a model-specific writing branch.
    plan = json.loads(json.dumps(plan))
    for unit in plan.get("units") or []:
        if unit.get("brief_disposition") == "exclude":
            unit["brief_disposition"] = "omit"
    segments = source_segments(source)
    expected = [item["id"] for item in segments]
    order = {ident: i for i, ident in enumerate(expected)}
    units = list(plan.get("units") or [])
    dispositions = list(plan.get("dispositions") or [])
    if not units:
        raise WritingContractError("plan has no editorial units")
    expected_units = [f"U{i:03d}" for i in range(1, len(units) + 1)]
    if [u.get("unit_id") for u in units] != expected_units:
        raise WritingContractError("unit ids must be U001..Un without gaps")

    claimed = []
    prior = -1
    for unit in units:
        ids = list(unit.get("source_ids") or [])
        if not ids:
            raise WritingContractError(f"{unit['unit_id']}: no source ids")
        unknown = [ident for ident in ids if ident not in order]
        if unknown:
            raise WritingContractError(
                f"{unit['unit_id']}: unknown source ids {unknown[:4]}")
        indices = [order[ident] for ident in ids]
        if indices != sorted(indices) or indices[0] <= prior:
            raise WritingContractError("editorial units do not follow source order")
        prior = indices[-1]
        claimed.extend(ids)
        disposition = unit.get("brief_disposition")
        brief_words = int(unit.get("brief_words") or 0)
        if disposition == "include" and brief_words < 1:
            raise WritingContractError(
                f"{unit['unit_id']}: included Brief unit has no allocation")
        if disposition == "omit" and brief_words != 0:
            raise WritingContractError(
                f"{unit['unit_id']}: omitted Brief unit has an allocation")

    disposed = []
    unit_ids = set(expected_units)
    for item in dispositions:
        ids = list(item.get("source_ids") or [])
        if not ids:
            raise WritingContractError("empty source disposition")
        unknown = [ident for ident in ids if ident not in order]
        if unknown:
            raise WritingContractError(
                f"disposition has unknown source ids {unknown[:4]}")
        represented = set(item.get("represented_by") or [])
        if represented - unit_ids:
            raise WritingContractError("disposition names an unknown unit")
        disposed.extend(ids)

    all_claims = claimed + disposed
    duplicates = sorted({ident for ident in all_claims
                         if all_claims.count(ident) > 1})
    if duplicates:
        raise WritingContractError(
            f"source ids are claimed more than once: {duplicates[:4]}")
    missing = [ident for ident in expected if ident not in all_claims]
    if missing:
        raise WritingContractError(
            f"source ids are unaccounted: {missing[:4]}")
    if set(all_claims) != set(expected):
        raise WritingContractError("source accounting does not match the source")

    detailed_words = sum(int(u["detailed_words"]) for u in units)
    brief_words = sum(int(u["brief_words"]) for u in units)
    if detailed_words > int(ceilings["detailed"]):
        raise WritingContractError("Detailed allocations exceed its ceiling")
    if brief_words > int(ceilings["brief"]):
        raise WritingContractError("Brief allocations exceed its ceiling")
    if brief_words < 1:
        raise WritingContractError("plan selects no Brief units")

    frozen = json.loads(json.dumps(plan))
    frozen["plan_sha256"] = _sha(json.dumps(
        plan, sort_keys=True, ensure_ascii=False, separators=(",", ":")))
    frozen["source_sha256"] = _sha(source)
    by_source = {item["id"]: item for item in segments}
    for unit in frozen["units"]:
        unit["source_words"] = sum(
            by_source[ident]["words"] for ident in unit["source_ids"])
    frozen["allocated_words"] = {
        "detailed": detailed_words, "brief": brief_words}
    return frozen


def load_plan(path: pathlib.Path, source: str, ceilings: dict) -> dict:
    """Load and fully revalidate a retained plan for safe resume."""
    try:
        value = json.loads(path.read_text())
    except Exception as exc:
        raise WritingContractError(f"invalid retained plan {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise WritingContractError(f"invalid retained plan {path}: not an object")
    if value.get("schema") == ENVELOPE_SCHEMA:
        if set(value) != {"schema", "payload", "metadata"}:
            raise WritingContractError(
                f"invalid retained plan {path}: unknown envelope fields")
        payload = value.get("payload")
        metadata = value.get("metadata")
        if not isinstance(payload, dict) or not isinstance(metadata, dict):
            raise WritingContractError(
                f"invalid retained plan {path}: malformed envelope")
        frozen = validate_plan(payload, source, ceilings)
        expected = plan_envelope(frozen)["metadata"]
        if metadata != expected:
            raise WritingContractError(
                f"invalid retained plan {path}: stale envelope metadata")
        return frozen

    # Explicit migration for the first v2 receipts, which persisted the
    # controller-computed fields beside the strict model payload.  Strip only
    # those known fields, revalidate from source, and compare every retained
    # computed value.  Arbitrary extra model fields remain rejected.
    payload = _plan_payload(value)
    frozen = validate_plan(payload, source, ceilings)
    expected = plan_envelope(frozen)["metadata"]
    retained = _retained_metadata(value)
    if retained and retained != expected:
        raise WritingContractError(
            f"invalid retained plan {path}: stale legacy metadata")
    return frozen


def _plan_payload(plan: dict) -> dict:
    payload = json.loads(json.dumps(plan))
    for key in ("plan_sha256", "source_sha256", "allocated_words"):
        payload.pop(key, None)
    for unit in payload.get("units") or []:
        if isinstance(unit, dict):
            unit.pop("source_words", None)
    return payload


def _retained_metadata(plan: dict) -> dict:
    keys = ("plan_sha256", "source_sha256", "allocated_words")
    if not any(key in plan for key in keys) and not any(
            isinstance(unit, dict) and "source_words" in unit
            for unit in (plan.get("units") or [])):
        return {}
    return {
        "plan_sha256": plan.get("plan_sha256"),
        "source_sha256": plan.get("source_sha256"),
        "allocated_words": plan.get("allocated_words"),
        "unit_source_words": {
            unit.get("unit_id"): unit.get("source_words")
            for unit in (plan.get("units") or []) if isinstance(unit, dict)
        },
    }


def plan_envelope(plan: dict) -> dict:
    return {
        "schema": ENVELOPE_SCHEMA,
        "payload": _plan_payload(plan),
        "metadata": {
            "plan_sha256": plan.get("plan_sha256"),
            "source_sha256": plan.get("source_sha256"),
            "allocated_words": plan.get("allocated_words"),
            "unit_source_words": {
                unit.get("unit_id"): unit.get("source_words")
                for unit in (plan.get("units") or []) if isinstance(unit, dict)
            },
        },
    }


def save_plan(path: pathlib.Path, plan: dict) -> None:
    path.write_text(json.dumps(plan_envelope(plan), indent=2,
                               ensure_ascii=False))


def global_outline(plan: dict) -> str:
    return "\n".join(
        f"{u['unit_id']} {u['title']} — {u['topic']} "
        f"[relation: {u.get('relation_to_previous') or 'opening'}]"
        for u in plan.get("units") or [])


def _assignment(plan: dict, depth: str) -> list[dict]:
    out = []
    for unit in plan.get("units") or []:
        if depth == "brief" and unit.get("brief_disposition") != "include":
            continue
        out.append({
            "unit_id": unit["unit_id"],
            "title": unit["title"],
            "source_ids": list(unit["source_ids"]),
            "topic": unit["topic"],
            "relation_to_previous": unit.get("relation_to_previous") or "",
            "governing_qualifications": list(
                unit.get("governing_qualifications") or []),
            "words": int(unit[f"{depth}_words"]),
            "source_words": int(unit.get("source_words") or 0),
        })
    return out


def _packet(depth: str, units: list[dict]) -> dict:
    return {
        "detailed": units if depth == "detailed" else [],
        "brief": units if depth == "brief" else [],
    }


def _packet_output_words(packet: dict) -> int:
    return sum(int(u["words"])
               for depth in ("detailed", "brief")
               for u in packet.get(depth) or [])


def _packet_source_words(packet: dict) -> int:
    # A whole-pair request cites each unit's evidence once even when that unit
    # has both Detailed and Brief obligations.
    seen = {}
    for depth in ("detailed", "brief"):
        for unit in packet.get(depth) or []:
            seen.setdefault(unit["unit_id"], int(unit.get("source_words") or 0))
    return sum(seen.values())


def _ids(packet: dict, depth: str) -> list[str]:
    return [u["unit_id"] for u in packet.get(depth) or []]


def packet_source_ids(packet: dict) -> list[str]:
    seen = set()
    out = []
    for depth in ("detailed", "brief"):
        for unit in packet.get(depth) or []:
            for ident in unit["source_ids"]:
                if ident not in seen:
                    seen.add(ident)
                    out.append(ident)
    return out


def build_packets(plan: dict, *, output_words: int | None,
                  input_words: int | None = None,
                  overhead_words: int = 700) -> list[dict]:
    """Greedily pack frozen units; only declared capacity changes grouping."""
    detailed, brief = _assignment(plan, "detailed"), _assignment(plan, "brief")
    cap = int(output_words) if output_words is not None else None
    if cap is not None and cap < 1:
        raise WritingContractError("output capacity must be positive")
    def fits(packet):
        output_ok = cap is None or _packet_output_words(packet) <= cap
        input_ok = (input_words is None or
                    overhead_words + _packet_source_words(packet)
                    <= int(input_words))
        return output_ok and input_ok

    both = {"detailed": detailed, "brief": brief}
    # A capable route may write the whole pair in one response.
    if fits(both):
        return [both]

    packets = []
    for depth, units in (("detailed", detailed), ("brief", brief)):
        current = []
        for unit in units:
            single = _packet(depth, [unit])
            if not fits(single):
                raise WritingContractError(
                    f"{unit['unit_id']} cannot fit the declared packet capacity")
            proposed = _packet(depth, current + [unit])
            if current and not fits(proposed):
                packets.append(_packet(depth, current))
                current = []
            current.append(unit)
        if current:
            packets.append(_packet(depth, current))
    return packets


def bind_packet_sources(packet: dict, segments: list[dict]) -> dict:
    """Attach exact source-word counts after generic packing."""
    by = {item["id"]: item for item in segments}
    bound = json.loads(json.dumps(packet))
    for depth in ("detailed", "brief"):
        for unit in bound.get(depth) or []:
            unit["source_words"] = sum(by[i]["words"]
                                       for i in unit["source_ids"])
    return bound


def writing_prompt(plan: dict, packet: dict, source: str,
                   previous_end: str = "(none)", *,
                   heading_policy: str | None = None) -> str:
    segments = source_segments(source)
    by = {item["id"]: item for item in segments}
    ids = packet_source_ids(packet)
    evidence = "\n".join(
        f"[{ident} sha256={by[ident]['sha256']}]\n{by[ident]['text']}"
        for ident in ids)
    assignment = json.dumps(packet, indent=2, ensure_ascii=False)
    detailed_ids = _ids(packet, "detailed")
    brief_ids = _ids(packet, "brief")
    example = {
        "detailed": ([{"unit_id": detailed_ids[0], "heading": "",
                       "join_previous": False,
                       "paragraphs": ["..."]}] if detailed_ids else []),
        "brief": ([{"unit_id": brief_ids[0], "heading": "",
                    "join_previous": False,
                    "paragraphs": ["..."]}] if brief_ids else []),
    }
    heading_rule = (
        "heading must be the empty string for every returned block."
        if heading_policy == "none" else
        "heading is plain heading text or \"\"; the controller owns Markdown "
        "rendering.")
    template = ((PROMPTS / "full-block-write.txt").read_text()
                .replace("{READING_POLICY}", pair_review.reading_policy()))
    return (template.replace("{RESPONSE_SHAPE}",
                             json.dumps(example, ensure_ascii=False))
            .replace("{DETAILED_COUNT}", str(len(detailed_ids)))
            .replace("{BRIEF_COUNT}", str(len(brief_ids)))
            .replace("{HEADING_RULE}", heading_rule)
            .replace("{GLOBAL_OUTLINE}", global_outline(plan))
            .replace("{ASSIGNMENT}", assignment)
            .replace("{PREVIOUS_END}", previous_end or "(none)")
            .replace("{SOURCE}", evidence))


def obligation_keys(packet: dict) -> list[tuple[str, str]]:
    """Return canonical (artifact, unit_id) keys in plan/packet order."""
    out = []
    seen = set()
    detailed = _ids(packet, "detailed")
    brief = set(_ids(packet, "brief"))
    for ident in detailed:
        key = ("detailed", ident)
        if key not in seen:
            seen.add(key)
            out.append(key)
        if ident in brief:
            key = ("brief", ident)
            if key not in seen:
                seen.add(key)
                out.append(key)
    for ident in _ids(packet, "brief"):
        key = ("brief", ident)
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def atomic_packets(packet: dict) -> list[dict]:
    """One recovery wave: exactly one (artifact, unit_id) block per request."""
    by_depth = {
        "detailed": {item["unit_id"]: item
                     for item in packet.get("detailed") or []},
        "brief": {item["unit_id"]: item
                  for item in packet.get("brief") or []},
    }
    out = []
    for artifact, ident in obligation_keys(packet):
        unit = dict(by_depth[artifact][ident])
        child = {"detailed": [], "brief": []}
        child[artifact] = [unit]
        out.append(child)
    return out


def assignment_digest(packet: dict) -> str:
    payload = {
        "detailed": _ids(packet, "detailed"),
        "brief": _ids(packet, "brief"),
        "words": {
            "detailed": [int(item.get("words") or 0)
                         for item in packet.get("detailed") or []],
            "brief": [int(item.get("words") or 0)
                      for item in packet.get("brief") or []],
        },
    }
    return _sha(json.dumps(payload, sort_keys=True, ensure_ascii=False))


def block_pair_schema(packet: dict, *, heading_policy: str | None = None) -> dict:
    return json_contract.block_pair_schema(packet, heading_policy=heading_policy)


def _writer_error(code: str, packet: dict, observed, invalid, message: str):
    expected = obligation_keys(packet)
    observed_keys = [tuple(item) for item in observed]
    expected_set, observed_set = set(expected), set(observed_keys)
    raise WriterResponseError(
        message, code=code, expected_keys=expected,
        observed_keys=observed_keys,
        missing_keys=[key for key in expected if key not in observed_set],
        extra_keys=[key for key in observed_keys if key not in expected_set],
        invalid_blocks=invalid)


def validate_blocks(value: dict, packet: dict, *,
                    heading_policy: str | None = None) -> dict:
    # Recovery ledgers written before join_previous existed remain resumable;
    # new model responses are still parsed against the strict schema first.
    value = json.loads(json.dumps(value))
    if isinstance(value, dict):
        for depth in ("detailed", "brief"):
            for item in value.get(depth) or []:
                if isinstance(item, dict):
                    item.setdefault("join_previous", False)
    schema = json_contract.block_pair_schema(
        packet, heading_policy=heading_policy)
    try:
        json_contract.validate(value, schema, "full-block-write")
    except Exception as exc:
        observed = []
        invalid = []
        if isinstance(value, dict):
            for depth in ("detailed", "brief"):
                for item in value.get(depth) or []:
                    ident = item.get("unit_id")
                    observed.append((depth, ident))
                    if ident in {None, ""}:
                        invalid.append({"artifact": depth, "unit_id": ident})
        _writer_error(
            "schema", packet, observed, invalid,
            str(exc) or "writer response failed the assignment schema")
    clean = {"detailed": [], "brief": []}
    observed = []
    invalid = []
    for depth in ("detailed", "brief"):
        expected = _ids(packet, depth)
        got = [item.get("unit_id") for item in value.get(depth) or []]
        if len(got) != len(set(got)):
            invalid.append({"artifact": depth, "unit_ids": got,
                            "problem": "duplicate"})
            _writer_error(
                "duplicate_ids", packet,
                [(depth, ident) for ident in got], invalid,
                f"{depth} block ids {got} are not unique")
        if got != expected:
            for ident in got:
                observed.append((depth, ident))
            invalid.append({"artifact": depth, "expected": expected,
                            "observed": got})
            _writer_error(
                "coverage", packet, observed, invalid,
                f"{depth} block ids {got} do not match assignment {expected}")
        for item in value.get(depth) or []:
            observed.append((depth, item.get("unit_id")))
            heading = str(item.get("heading") or "")
            if heading_policy == "none":
                if heading != "":
                    invalid.append({"artifact": depth,
                                    "unit_id": item.get("unit_id"),
                                    "problem": "heading"})
                    _writer_error(
                        "heading", packet, observed, invalid,
                        f"{item.get('unit_id')}: heading must be empty")
            else:
                heading = heading.strip()
                if "\n" in heading or heading.startswith("#"):
                    invalid.append({"artifact": depth,
                                    "unit_id": item.get("unit_id"),
                                    "problem": "heading"})
                    _writer_error(
                        "heading", packet, observed, invalid,
                        f"{item['unit_id']}: heading must be plain one-line text")
            paragraphs = [str(p).strip() for p in item.get("paragraphs") or []]
            if not paragraphs or any(not p for p in paragraphs):
                invalid.append({"artifact": depth,
                                "unit_id": item.get("unit_id"),
                                "problem": "empty_paragraphs"})
                _writer_error(
                    "empty_paragraphs", packet, observed, invalid,
                    f"{item.get('unit_id')}: paragraphs cannot be empty")
            for paragraph in paragraphs:
                if pair_patch.MARKUP.search(paragraph):
                    invalid.append({"artifact": depth,
                                    "unit_id": item.get("unit_id"),
                                    "problem": "markup"})
                    _writer_error(
                        "markup", packet, observed, invalid,
                        f"{item['unit_id']}: paragraph uses list/table markup")
            join_previous = item.get("join_previous")
            if not isinstance(join_previous, bool):
                invalid.append({"artifact": depth,
                                "unit_id": item.get("unit_id"),
                                "problem": "join_previous"})
                _writer_error(
                    "join_previous", packet, observed, invalid,
                    f"{item['unit_id']}: join_previous must be boolean")
            if join_previous and heading:
                invalid.append({"artifact": depth,
                                "unit_id": item.get("unit_id"),
                                "problem": "join_across_heading"})
                _writer_error(
                    "join_across_heading", packet, observed, invalid,
                    f"{item['unit_id']}: a headed block cannot join previous")
            clean[depth].append({
                "unit_id": item["unit_id"],
                "heading": "" if heading_policy == "none" else heading,
                "join_previous": join_previous,
                "paragraphs": paragraphs})
    return clean


def block_word_findings(plan: dict, blocks: dict) -> list[dict]:
    """Measure per-unit overruns; never truncate or retry the packet."""
    by_plan = {unit["unit_id"]: unit for unit in plan["units"]}
    findings = []
    for artifact in ("detailed", "brief"):
        budget_key = "detailed_words" if artifact == "detailed" else "brief_words"
        for unit_id, block in (blocks.get(artifact) or {}).items():
            allocated = int(by_plan[unit_id][budget_key])
            words = len(render_blocks([block]).split())
            if words > allocated:
                findings.append({
                    "kind": "length", "artifact": artifact,
                    "unit_id": unit_id, "allocated_words": allocated,
                    "actual_words": words,
                    "text": (f"{artifact} {unit_id} is {words} words "
                             f"over the {allocated}-word allocation"),
                })
    return findings


def coalesce_review_packets(packets: list[dict]) -> list[dict]:
    """Merge only review packets with identical unit set and source evidence."""
    out = []
    index_by = {}
    for packet in packets:
        key = (packet.get("source"),
               json.dumps(packet.get("plan_context"), sort_keys=True))
        existing = index_by.get(key)
        if existing is None:
            index_by[key] = len(out)
            item = json.loads(json.dumps(packet))
            item["packet_id"] = f"review-{len(out) + 1:03d}"
            out.append(item)
            continue
        current = out[existing]
        if packet.get("detailed") and packet["detailed"] not in (
                current.get("detailed") or ""):
            current["detailed"] = (current.get("detailed") or "").strip()
            if current["detailed"] and packet["detailed"]:
                current["detailed"] += "\n\n" + packet["detailed"]
            else:
                current["detailed"] = packet["detailed"] or current["detailed"]
        if packet.get("brief") and packet["brief"] not in (
                current.get("brief") or ""):
            current["brief"] = (current.get("brief") or "").strip()
            if current["brief"] and packet["brief"]:
                current["brief"] += "\n\n" + packet["brief"]
            else:
                current["brief"] = packet["brief"] or current["brief"]
        current["observations"] = list(current.get("observations") or []) + [
            item for item in packet.get("observations") or []
            if item not in (current.get("observations") or [])]
        incoming = packet.get("segments")
        current_segs = current.get("segments")
        if incoming and incoming != current_segs:
            if isinstance(current_segs, dict) and isinstance(incoming, dict):
                merged = dict(current_segs)
                merged.update(incoming)
                current["segments"] = merged
            elif isinstance(current_segs, str) and isinstance(incoming, str):
                if incoming not in current_segs:
                    current["segments"] = current_segs + "\n" + incoming
            elif not current_segs:
                current["segments"] = incoming
    return out


def new_writer_recovery(plan: dict, packet: dict, *,
                        heading_policy: str | None = None,
                        source_sha256: str = "") -> dict:
    keys = obligation_keys(packet)
    return {
        "schema": RECOVERY_SCHEMA,
        "source_sha256": source_sha256 or plan.get("source_sha256") or "",
        "plan_sha256": plan.get("plan_sha256") or "",
        "root_assignment_sha256": assignment_digest(packet),
        "root_packet": json.loads(json.dumps(packet)),
        "wave": "fresh",
        "heading_policy": heading_policy or "",
        "expected_keys": [list(key) for key in keys],
        "pending_keys": [list(key) for key in keys],
        "accepted": {},
        "unresolved": [],
        "rejection": {},
        "stop_reason": "",
    }


def validate_writer_recovery(state: dict, plan: dict, packet: dict) -> dict:
    if not isinstance(state, dict) or state.get("schema") != RECOVERY_SCHEMA:
        raise WritingContractError("writer recovery state is invalid")
    if state.get("wave") not in RECOVERY_WAVES:
        raise WritingContractError("writer recovery wave is invalid")
    if state.get("root_assignment_sha256") != assignment_digest(packet):
        raise WritingContractError("writer recovery assignment digest mismatch")
    if plan.get("plan_sha256") and state.get("plan_sha256") != plan["plan_sha256"]:
        raise WritingContractError("writer recovery plan hash mismatch")
    expected = [list(key) for key in obligation_keys(packet)]
    if state.get("expected_keys") != expected:
        raise WritingContractError("writer recovery expected keys mismatch")
    return state


def accept_recovery_block(state: dict, artifact: str, unit_id: str,
                          block: dict, receipt: dict) -> dict:
    key = f"{artifact}:{unit_id}"
    accepted = dict(state.get("accepted") or {})
    if key in accepted:
        raise WritingContractError(
            f"writer recovery already accepted {key}")
    item = json.loads(json.dumps(block))
    accepted[key] = {
        "block": item,
        "receipt": json.loads(json.dumps(receipt)),
        "sha256": _sha(json.dumps(item, sort_keys=True, ensure_ascii=False)),
    }
    pending = [list(item) for item in state.get("pending_keys") or []
               if list(item) != [artifact, unit_id]]
    out = json.loads(json.dumps(state))
    out["accepted"] = accepted
    out["pending_keys"] = pending
    if pending:
        out["wave"] = "repacked"
    else:
        out["wave"] = "complete"
        out["unresolved"] = []
    return out


def render_recovery_blocks(plan: dict, state: dict) -> dict:
    detailed, brief = {}, {}
    accepted = state.get("accepted") or {}
    for unit in plan["units"]:
        ident = unit["unit_id"]
        dkey, bkey = f"detailed:{ident}", f"brief:{ident}"
        if dkey in accepted:
            detailed[ident] = accepted[dkey]["block"]
        if unit["brief_disposition"] == "include" and bkey in accepted:
            brief[ident] = accepted[bkey]["block"]
    return {"detailed": detailed, "brief": brief}


def validate_presentation_joins(blocks: list[dict], expected_ids: list[str],
                                artifact: str) -> None:
    """Fail closed unless continuation signals bind adjacent ordered units."""
    ids = [str(item.get("unit_id") or "") for item in blocks or []]
    if ids != list(expected_ids):
        raise WritingContractError(
            f"{artifact} presentation blocks {ids} do not match {expected_ids}")
    for index, item in enumerate(blocks or []):
        joined = item.get("join_previous", False)
        if not isinstance(joined, bool):
            raise WritingContractError(
                f"{artifact} {ids[index]} join_previous must be boolean")
        if joined and index == 0:
            raise WritingContractError(
                f"{artifact} {ids[index]} cannot join before the first unit")
        if joined and str(item.get("heading") or "").strip():
            raise WritingContractError(
                f"{artifact} {ids[index]} cannot join across a heading")


def _presentation_segments(blocks: list[dict]) -> list[dict]:
    """Return rendered segments with every contributing unit preserved."""
    segments = []
    for item in blocks or []:
        unit_id = str(item.get("unit_id") or "")
        if item.get("heading"):
            segments.append({
                "kind": "heading",
                "text": "## " + item["heading"].strip(),
                "unit_ids": [unit_id],
            })
        paragraphs = [p.strip() for p in item.get("paragraphs") or []
                      if p.strip()]
        if (item.get("join_previous") and not item.get("heading")
                and segments and segments[-1]["kind"] == "paragraph"
                and paragraphs):
            segments[-1]["text"] = (
                segments[-1]["text"].rstrip() + " "
                + paragraphs.pop(0).lstrip())
            if unit_id not in segments[-1]["unit_ids"]:
                segments[-1]["unit_ids"].append(unit_id)
        segments.extend({
            "kind": "paragraph", "text": paragraph,
            "unit_ids": [unit_id],
        } for paragraph in paragraphs)
    return segments


def render_blocks(blocks: list[dict]) -> str:
    return "\n\n".join(
        item["text"] for item in _presentation_segments(blocks)).strip()


def _block_texts(block: dict) -> list[str]:
    texts = []
    if block.get("heading"):
        texts.append("## " + block["heading"].strip())
    texts.extend(item.strip() for item in block.get("paragraphs") or []
                 if str(item).strip())
    return texts


def _split_patch_paragraphs(text: str) -> tuple[str, list[str]]:
    heading = ""
    paragraphs = []
    for chunk in re.split(r"\n\n+", (text or "").strip()):
        chunk = chunk.strip()
        if not chunk:
            continue
        if chunk.startswith("#") and "\n" not in chunk:
            heading = re.sub(r"^#{1,6}\s+", "", chunk).strip()
            continue
        if chunk.startswith("## "):
            lines = chunk.split("\n", 1)
            heading = lines[0][3:].strip()
            if len(lines) > 1 and lines[1].strip():
                paragraphs.append(lines[1].strip())
            continue
        paragraphs.append(chunk)
    return heading, paragraphs


def rebind_blocks_after_patch(all_blocks: dict, old_detailed: str,
                              old_brief: str, edits: list) -> dict:
    """Preserve unit ownership across a transactional patch.

    A replacement inherits the consumed units' ownership. An insertion
    retains its named anchor's ownership. Cross-unit ranges keep the union:
    the first consumed unit receives the replacement prose.
    """
    import pair_patch
    old_pair = pair_patch.segment_pair(old_detailed, old_brief)
    expected_detailed, expected_brief = pair_patch.apply_edits(
        old_detailed, old_brief,
        {"base_candidate": old_pair["candidate"], "edits": edits})
    queues = {"detailed": {}, "brief": {}}
    for artifact in ("detailed", "brief"):
        for segment in old_pair[artifact]:
            queues[artifact].setdefault(segment["text"].strip(), []).append(
                segment["id"])
    seg_unit = {}
    for artifact in ("detailed", "brief"):
        ordered = list((all_blocks.get(artifact) or {}).values())
        for presented in _presentation_segments(ordered):
            matches = queues[artifact].get(presented["text"]) or []
            if not matches:
                units = ",".join(presented["unit_ids"])
                raise WritingContractError(
                    f"{units}: candidate segment identity could not be bound")
            seg_unit[matches.pop(0)] = (
                artifact, presented["unit_ids"][0])
    out = {
        "detailed": {key: json.loads(json.dumps(value))
                     for key, value in (all_blocks.get("detailed") or {}).items()},
        "brief": {key: json.loads(json.dumps(value))
                  for key, value in (all_blocks.get("brief") or {}).items()},
    }
    id_index = {
        artifact: {item["id"]: item for item in old_pair[artifact]}
        for artifact in ("detailed", "brief")
    }
    ordered_ids = {
        artifact: [item["id"] for item in old_pair[artifact]]
        for artifact in ("detailed", "brief")
    }

    def owner_of(artifact, ident):
        if ident in seg_unit:
            return seg_unit[ident]
        return None

    for edit in edits or []:
        if not isinstance(edit, dict):
            continue
        artifact = str(edit.get("artifact") or "")
        operation = str(edit.get("operation") or "")
        if artifact not in out:
            continue
        if operation in {"insert_before", "insert_after"}:
            owner = owner_of(artifact, str(edit.get("anchor") or ""))
            if owner is None:
                continue
            _art, unit_id = owner
            heading, paragraphs = _split_patch_paragraphs(
                str(edit.get("replacement") or ""))
            block = out[_art][unit_id]
            plist = list(block.get("paragraphs") or [])
            anchor = id_index[artifact].get(str(edit.get("anchor") or ""))
            anchor_text = (anchor or {}).get("text", "").strip()
            idx = 0
            if anchor_text in plist:
                idx = plist.index(anchor_text)
            elif plist:
                idx = len(plist) - 1
            insert_at = idx if operation == "insert_before" else idx + 1
            for offset, paragraph in enumerate(paragraphs):
                plist.insert(insert_at + offset, paragraph)
            block["paragraphs"] = plist
            if heading:
                block["heading"] = heading
        elif operation == "replace_range":
            start_id = str(edit.get("start") or "")
            end_id = str(edit.get("end") or "")
            ids = ordered_ids.get(artifact) or []
            if start_id not in ids or end_id not in ids:
                continue
            i0, i1 = ids.index(start_id), ids.index(end_id)
            if i1 < i0:
                i0, i1 = i1, i0
            consumed = ids[i0:i1 + 1]
            owners = [seg_unit[ident] for ident in consumed if ident in seg_unit]
            if not owners:
                continue
            first_art, first_unit = owners[0]
            heading, paragraphs = _split_patch_paragraphs(
                str(edit.get("replacement") or ""))
            block = out[first_art][first_unit]
            plist = list(block.get("paragraphs") or [])
            consumed_paras = []
            for ident in consumed:
                segment = id_index[artifact].get(ident) or {}
                if segment.get("kind") == "paragraph":
                    consumed_paras.append(segment["text"].strip())
                elif segment.get("kind") == "heading" and not heading:
                    heading = re.sub(r"^#{1,6}\s+", "",
                                     segment["text"].strip()).strip()
            if consumed_paras and consumed_paras[0] in plist:
                start = plist.index(consumed_paras[0])
                end = start + len(consumed_paras)
                plist[start:end] = paragraphs
            else:
                plist = paragraphs
            block["paragraphs"] = plist
            if heading:
                block["heading"] = heading
    rebound_detailed = render_blocks(list(out["detailed"].values()))
    rebound_brief = render_blocks(list(out["brief"].values()))
    if (rebound_detailed.strip() != expected_detailed.strip()
            or rebound_brief.strip() != expected_brief.strip()):
        raise WritingContractError(
            "patch cannot preserve exact prose and unit ownership")
    return out


def _bound_presentation(plan: dict, all_blocks: dict,
                        detailed: str, brief: str):
    """Bind rendered segments to every contributing unit in plan order."""
    pair = pair_patch.segment_pair(detailed, brief)
    queues = {"detailed": {}, "brief": {}}
    for artifact in ("detailed", "brief"):
        for segment in pair[artifact]:
            queues[artifact].setdefault(segment["text"].strip(), []).append(
                segment["id"])
    block_segments = {"detailed": {}, "brief": {}}
    presented = {"detailed": [], "brief": []}
    plan_order = [unit["unit_id"] for unit in plan["units"]]
    for artifact in ("detailed", "brief"):
        ordered = [all_blocks[artifact][unit_id] for unit_id in plan_order
                   if unit_id in all_blocks[artifact]]
        for item in _presentation_segments(ordered):
            matches = queues[artifact].get(item["text"]) or []
            if not matches:
                units = ",".join(item["unit_ids"])
                raise WritingContractError(
                    f"{units}: candidate segment identity could not be bound")
            bound = dict(item, segment_id=matches.pop(0))
            presented[artifact].append(bound)
            for unit_id in item["unit_ids"]:
                block_segments[artifact].setdefault(unit_id, []).append(
                    bound["segment_id"])
    return pair, presented, block_segments


def joined_component_for_anchor(plan: dict, all_blocks: dict,
                                detailed: str, brief: str,
                                artifact: str, anchor: str) -> dict | None:
    """Return the complete multi-owner component for one rendered anchor."""
    if artifact not in {"detailed", "brief"}:
        return None
    _pair, presented, _block_segments = _bound_presentation(
        plan, all_blocks, detailed, brief)
    target = next((item for item in presented[artifact]
                   if item["segment_id"] == anchor
                   and len(item["unit_ids"]) > 1), None)
    if target is None:
        return None
    unit_ids = list(target["unit_ids"])
    assignments = {item["unit_id"]: item
                   for item in _assignment(plan, artifact)}
    units = [assignments[unit_id] for unit_id in unit_ids]
    packet = _packet(artifact, units)
    source_ids = list(dict.fromkeys(
        ident for unit in units for ident in unit["source_ids"]))
    return {
        "artifact": artifact,
        "anchor": anchor,
        "unit_ids": unit_ids,
        "source_ids": source_ids,
        "packet": packet,
        "blocks": [json.loads(json.dumps(all_blocks[artifact][unit_id]))
                   for unit_id in unit_ids],
    }


def apply_joined_component_repair(plan: dict, all_blocks: dict,
                                  component: dict, response: dict) -> dict:
    """Replace every owner of one joined segment, or reject the repair."""
    artifact = component["artifact"]
    unit_ids = list(component["unit_ids"])
    clean = validate_blocks(response, component["packet"])
    returned = clean[artifact]
    if [item["unit_id"] for item in returned] != unit_ids:
        raise WritingContractError(
            "joined repair must return every owner exactly once in order")
    original = {item["unit_id"]: item for item in component["blocks"]}
    for item in returned:
        before = original[item["unit_id"]]
        if item.get("heading") != (before.get("heading") or ""):
            raise WritingContractError(
                "joined repair cannot add, remove, or move a heading")
        if bool(item.get("join_previous")) != bool(
                before.get("join_previous")):
            raise WritingContractError(
                "joined repair must preserve continuation boundaries")
    out = json.loads(json.dumps(all_blocks))
    for item in returned:
        out[artifact][item["unit_id"]] = item
    expected = [unit["unit_id"] for unit in plan["units"]
                if artifact == "detailed"
                or unit["brief_disposition"] == "include"]
    ordered = [out[artifact][unit_id] for unit_id in expected]
    validate_presentation_joins(ordered, expected, artifact)
    return out


def bounded_review_packets(plan: dict, packets: list[dict], all_blocks: dict,
                           source: str, detailed: str, brief: str) -> tuple[list[dict], dict]:
    """Review each joined presentation component against all of its evidence."""
    segments = source_segments(source)
    by_source = {item["id"]: item for item in segments}
    pair, presented, block_segments = _bound_presentation(
        plan, all_blocks, detailed, brief)
    plan_order = [unit["unit_id"] for unit in plan["units"]]
    plan_by_id = {unit["unit_id"]: unit for unit in plan["units"]}

    parent = {unit_id: unit_id for unit_id in plan_order}
    def find(unit_id):
        while parent[unit_id] != unit_id:
            parent[unit_id] = parent[parent[unit_id]]
            unit_id = parent[unit_id]
        return unit_id
    def union(left, right):
        lroot, rroot = find(left), find(right)
        if lroot != rroot:
            parent[rroot] = lroot
    for artifact in ("detailed", "brief"):
        for item in presented[artifact]:
            for unit_id in item["unit_ids"][1:]:
                union(item["unit_ids"][0], unit_id)
    groups = {}
    for unit_id in plan_order:
        groups.setdefault(find(unit_id), []).append(unit_id)
    components = sorted(groups.values(), key=lambda ids: plan_order.index(ids[0]))

    observations = readability_observations(detailed, brief)
    out = []
    for index, unit_ids in enumerate(components, 1):
        wanted_sources = {
            ident for unit_id in unit_ids
            for ident in plan_by_id[unit_id]["source_ids"]}
        source_ids = [item["id"] for item in segments
                      if item["id"] in wanted_sources]
        selected_ids = list(dict.fromkeys(
            ident for artifact in ("detailed", "brief")
            for unit_id in unit_ids
            for ident in block_segments[artifact].get(unit_id, [])))

        def selected_text(artifact):
            wanted = {
                ident for unit_id in unit_ids
                for ident in block_segments[artifact].get(unit_id, [])}
            return "\n\n".join(
                item["text"] for item in pair[artifact]
                if item["id"] in wanted)

        out.append({
            "packet_id": f"review-{index:03d}",
            "unit_ids": list(unit_ids),
            "source_ids": list(source_ids),
            "source": "\n".join(
                f"[{ident}]\n{by_source[ident]['text']}" for ident in source_ids),
            "detailed": selected_text("detailed"),
            "brief": selected_text("brief"),
            "plan_context": json.dumps(
                {"units": [plan_by_id[unit_id] for unit_id in unit_ids],
                 "presentation": {
                     artifact: [
                         {"segment_id": item["segment_id"],
                          "unit_ids": item["unit_ids"]}
                         for item in presented[artifact]
                         if set(item["unit_ids"]) & set(unit_ids)]
                     for artifact in ("detailed", "brief")}},
                indent=2, ensure_ascii=False),
            "segments": pair_patch.selected_segment_map(
                detailed, brief, selected_ids),
            "observations": [item for item in observations
                             if item["anchor"] in selected_ids],
        })
    structure = []
    for artifact in ("detailed", "brief"):
        for unit in plan["units"]:
            block = all_blocks[artifact].get(unit["unit_id"])
            if not block:
                continue
            rendered = render_blocks([block])
            words = rendered.split()
            structure.append({
                "artifact": artifact, "unit_id": unit["unit_id"],
                "heading": block.get("heading") or "",
                "join_previous": bool(block.get("join_previous")),
                "paragraphs": len(block.get("paragraphs") or []),
                "words": len(words),
                "opening": " ".join(words[:30]),
                "ending": " ".join(words[-30:]),
            })
    heading_ids = [item["id"] for artifact in ("detailed", "brief")
                   for item in pair[artifact] if item["kind"] == "heading"]
    global_packet = {
        "packet_id": "global",
        "source": global_outline(plan),
        "detailed": detailed,
        "brief": brief,
        "plan_context": json.dumps(
            {"plan": _plan_payload(plan), "structure": structure},
            indent=2, ensure_ascii=False),
        "segments": pair_patch.selected_segment_map(
            detailed, brief, heading_ids),
        "observations": [],
    }
    covered_units = [unit_id for packet in out
                     for unit_id in packet["unit_ids"]]
    covered_sources = {source_id for packet in out
                       for source_id in packet["source_ids"]}
    expected_sources = {ident for unit in plan["units"]
                        for ident in unit["source_ids"]}
    if covered_units != plan_order or covered_sources != expected_sources:
        raise WritingContractError(
            "bounded review components do not cover the complete frozen plan")
    return out, global_packet


def review_coverage_error(plan: dict, packets: list[dict],
                          successful_packet_ids: list[str]) -> str:
    """Explain incomplete local review; an empty string means exact coverage."""
    successful = set(successful_packet_ids)
    reviewed = [packet for packet in packets
                if packet.get("packet_id") in successful]
    units = [unit_id for packet in reviewed
             for unit_id in packet.get("unit_ids") or []]
    sources = [source_id for packet in reviewed
               for source_id in packet.get("source_ids") or []]
    expected_units = [unit["unit_id"] for unit in plan["units"]]
    expected_sources = list(dict.fromkeys(
        ident for unit in plan["units"] for ident in unit["source_ids"]))
    if units == expected_units and set(sources) == set(expected_sources):
        return ""
    missing_units = [unit_id for unit_id in expected_units
                     if unit_id not in units]
    missing_sources = [source_id for source_id in expected_sources
                       if source_id not in sources]
    foreign_units = [unit_id for unit_id in units
                     if unit_id not in expected_units]
    duplicates = sorted({unit_id for unit_id in units
                         if units.count(unit_id) > 1})
    return ("review coverage incomplete: missing units "
            f"{missing_units}, missing sources {missing_sources}, "
            f"foreign units {foreign_units}, duplicate units {duplicates}")


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])(?:[\"'’”)]*)\s+",
                                           text.strip()) if s.strip()]


def readability_observations(detailed: str, brief: str) -> list[dict]:
    """Conservative risk observations; no prose is changed deterministically."""
    pair = pair_patch.segment_pair(detailed, brief)
    out = []
    for artifact in ("detailed", "brief"):
        for segment in pair[artifact]:
            if segment["kind"] != "paragraph":
                continue
            text = segment["text"].strip()
            words = len(text.split())
            sentences = _sentences(text)
            counts = [len(sentence.split()) for sentence in sentences]
            triggers = []
            if words > PARAGRAPH_ASSESS_WORDS:
                triggers.append(f"paragraph_words={words}>{PARAGRAPH_ASSESS_WORDS}")
            if len(sentences) > PARAGRAPH_ASSESS_SENTENCES:
                triggers.append(
                    f"paragraph_sentences={len(sentences)}>"
                    f"{PARAGRAPH_ASSESS_SENTENCES}")
            longest = max(counts, default=0)
            if longest > SENTENCE_ASSESS_WORDS:
                triggers.append(
                    f"longest_sentence_words={longest}>{SENTENCE_ASSESS_WORDS}")
            windows = [sum(counts[i:i + 3]) / 3
                       for i in range(max(0, len(counts) - 2))]
            dense = max(windows, default=0)
            sparse = min(windows, default=10**9)
            if dense > WINDOW_DENSE_WORDS:
                triggers.append(
                    f"dense_three_sentence_mean={dense:.1f}>{WINDOW_DENSE_WORDS}")
            if windows and sparse < WINDOW_FRAGMENT_WORDS:
                triggers.append(
                    f"short_three_sentence_mean={sparse:.1f}<"
                    f"{WINDOW_FRAGMENT_WORDS}")
            if not triggers:
                continue
            blocking = (words > PARAGRAPH_BLOCK_WORDS
                        or len(sentences) > PARAGRAPH_BLOCK_SENTENCES)
            out.append({
                "observation_id": f"R-{len(out) + 1:03d}",
                "artifact": artifact,
                "anchor": segment["id"],
                "words": words,
                "sentences": len(sentences),
                "triggers": triggers,
                "blocking": blocking,
            })
    return out


def blocking_readability_findings(observations: list[dict]) -> list[str]:
    return [
        "needs_structure_repair: "
        f"{item['artifact']} {item['anchor']} has {item['words']} words and "
        f"{item['sentences']} sentences; split it at genuine discourse turns"
        for item in observations if item.get("blocking")]


def audit_context(observations: list[dict]) -> str:
    if not observations:
        return "[]"
    return json.dumps(observations, indent=2, ensure_ascii=False)


def validate_readability_assessments(audit: dict,
                                     observations: list[dict]) -> list[dict]:
    expected = [item["observation_id"] for item in observations]
    got = [item.get("observation_id")
           for item in audit.get("readability") or []]
    if got != expected:
        raise WritingContractError(
            f"readability assessments {got} do not match observations {expected}")
    by = {item["observation_id"]: item for item in observations}
    existing = {(item.get("artifact"), item.get("anchor"))
                for item in audit.get("findings") or []
                if item.get("kind") == "editorial"}
    editorial = []
    for item in audit.get("readability") or []:
        if item.get("assessment") == "acceptable":
            continue
        observation = by[item["observation_id"]]
        if (observation["artifact"], observation["anchor"]) in existing:
            continue
        editorial.append({
            "kind": "editorial",
            "artifact": observation["artifact"],
            "text": (f"{item['assessment']}: {item['explanation']}"),
            "anchor": observation["anchor"],
            "slot": "",
        })
    if editorial and audit.get("verdict") != "revise":
        raise WritingContractError(
            "unreadable or uncertain assessment requires revise verdict")
    return editorial


def quality_state(candidate: dict) -> str:
    if not candidate.get("usable"):
        return "retained_draft"
    if (candidate.get("review") == "complete"
            and not candidate.get("findings")):
        return "quality_qualified_candidate"
    return "usable_candidate"
