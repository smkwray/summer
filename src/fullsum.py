#!/usr/bin/env python3
"""Full routes: direct, bounded-output batches, or source-window evidence,
followed by fresh review of the actual pair and one enclosed-candidate repair.

Scope is deliberately narrow: sources whose composed request fits the generic
qualified baseline (see fits()) take the existing whole-source route. A route
whose output envelope cannot hold the logical pair writes bounded source-order
parts and assembles them before review. Larger sources take deterministic
source windows that become compact evidence for one final Full pair. An unsafe
plan returns NOT_SUPPORTED before publication, with no ledger fallback and no
downgrade to Quick. Full stays Full at every source size; short documents take
this same direct path, not Quick.

Candidate retention, safest-candidate selection, status, disclosure, and the
enclosed-candidate repair prompt all live in pair_review, shared with Quick.
This module supplies Full's ceilings, structural gate, transport, and
publication. Exit codes: 0 published; 1 no usable prose or incomplete window
coverage; 5 no structurally usable candidate; 6 no safe windowed route.
"""
from __future__ import annotations
import hashlib, inspect, json, os, pathlib, re, sys

HERE = pathlib.Path(__file__).parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import custom_instructions
import json_contract
import pair_review
import route_caps
import writing_contract

NOT_SUPPORTED = 6
# One correction per candidate, from the producer that wrote it. Additional
# rounds add unbounded latency and shop for a different answer.
REPAIR_BUDGET = 1
REPAIR_WORDS = {0: "none", 1: "once", 2: "twice", 3: "thrice"}
MINIMUM = {"detailed": 30, "brief": 15}

RESULT_REQUEST_OPTIONS = json_contract.options(
    "summer_full_result", json_contract.FULL_RESULT)
WINDOW_REQUEST_OPTIONS = json_contract.options(
    "summer_full_window", json_contract.FULL_WINDOW_RESULT)
AUDIT_REQUEST_OPTIONS = json_contract.options(
    "summer_full_audit", json_contract.FULL_AUDIT)
PATCH_REQUEST_OPTIONS = json_contract.options(
    "summer_pair_patch", json_contract.PAIR_PATCH)
PART_REQUEST_OPTIONS = json_contract.options(
    "summer_full_reading_part", json_contract.FULL_READING_PART)
PLAN_V2_REQUEST_OPTIONS = json_contract.options(
    "summer_full_discourse_plan_v2", json_contract.FULL_DISCOURSE_PLAN)
BLOCK_V2_REQUEST_OPTIONS = json_contract.options(
    "summer_full_blocks_v2", json_contract.FULL_BLOCK_PAIR)
AUDIT_V2_REQUEST_OPTIONS = json_contract.options(
    "summer_full_audit_v2", json_contract.FULL_AUDIT_V2)

WINDOW_CAPSULE_WORDS = 2_400
# A window whose whole chain fails is retried as halves, then quarters, before
# the run concludes the models are not working. Bounded so a dead provider
# cannot turn one window into an unbounded fan-out.
SPLIT_DEPTH = 2
SPLIT_MIN_WORDS = 120
WINDOW_CALL_LIMIT = 32
FULL_MAX_LOGICAL_CALLS = 72
FULL_MAX_PHYSICAL_ATTEMPTS = 144
# A declared output maximum is a hard transport edge, not a sensible prose
# target. Keep JSON and normal token/word variation away from that edge; an
# unexpected truncation is then retried on smaller source spans below.
PART_OUTPUT_SHARE = 0.70
PART_SPLIT_DEPTH = 2
PART_MIN_SOURCE_WORDS = 120
PART_CALL_LIMIT = 32

# Structured-output framing estimates used by the qualified reasoning-aware
# admission controller. They bound JSON keys/ids/headings separately from the
# prose-word assignment and are checked again against the exact rendered prompt
# before any gateway generation starts.
PLAN_OUTPUT_WORDS = 900
PLAN_OUTPUT_OVERHEAD_TOKENS = 700
WRITE_OUTPUT_OVERHEAD_TOKENS = 500
AUDIT_OUTPUT_WORDS = 800
AUDIT_OUTPUT_OVERHEAD_TOKENS = 700
REPAIR_OUTPUT_OVERHEAD_TOKENS = 700
# One-key recovery children have no smaller ownership-preserving split.
# Retry the same obligation at the ordinary write budget before blocking.
ATOMIC_CHILD_ATTEMPTS = 3


def _admission_kwargs(ms, **values) -> dict:
    """Pass typed admission data when the selected runner implements it.

    Production mapsum does. Small pure test transports and older embedders may
    intentionally expose only the historical call signature.
    """
    try:
        parameters = inspect.signature(ms.run).parameters.values()
    except (TypeError, ValueError):
        return values
    if any(item.kind == inspect.Parameter.VAR_KEYWORD for item in parameters):
        return values
    names = {item.name for item in parameters}
    return values if set(values) <= names else {}


def _last_ok_chain(workdir: pathlib.Path, stage: str, fallback) -> list[str]:
    """The producer that returned a candidate for this stage, else the first
    configured writer. Any repair goes back to that producer."""
    import mapsum
    return mapsum.last_ok_route(workdir, stage, fallback)


def _split_text(text: str) -> tuple[str, str]:
    """Two halves of a window, byte-preserving.

    An earlier version split on whitespace tokens and rejoined with single
    spaces, which silently rewrote paragraphs, lists, and code before the
    model ever saw them. Prefer the paragraph break nearest the midpoint,
    then a line break, then a space; the two pieces still concatenate to the
    original."""
    middle = len(text) // 2
    for pattern in ("\n\n", "\n", " "):
        left = text.rfind(pattern, 0, middle)
        right = text.find(pattern, middle)
        candidates = [i for i in (left, right) if i > 0]
        if candidates:
            cut = min(candidates, key=lambda i: abs(i - middle))
            return text[:cut], text[cut:]
    return text[:middle], text[middle:]


def _shorten_window_prompt(base_prompt: str, capsule: str, limit: int) -> str:
    n = len(capsule.split())
    return (base_prompt
            + "\n\nCURRENT CAPSULE (" + str(n) + " words). It is over the "
            + str(limit) + "-word maximum. Revise THIS exact capsule so it "
            "is at most " + str(limit) + " words. Preserve every distinct "
            "claim, quantity, condition, and qualification you can; compress "
            "repetition first. Return the same JSON object. Do not start over "
            "from a thinner reading of the source than this capsule already "
            "contains.\nCURRENT CAPSULE:\n" + capsule + "\n")


def _keep_window_capsule(original: str, revised: str | None, limit: int
                         ) -> tuple[str, str]:
    """Keep a candidate after one same-producer shorten. Length never wins."""
    if not (revised or "").strip():
        return original, "length_exception"
    o, r = len(original.split()), len(revised.split())
    if r <= limit:
        return revised, "shortened"
    if r < o:
        return revised, "length_exception"
    return original, "length_exception"


def _runner():
    import importlib.util
    spec = importlib.util.spec_from_file_location("ms", HERE / "mapsum.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def _ledger():
    import importlib.util
    spec = importlib.util.spec_from_file_location("led", HERE / "ledger.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def _shortsum():
    import importlib.util
    spec = importlib.util.spec_from_file_location("ss", HERE / "shortsum.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def _bands():
    """The publication bands, from compose.py. Not redeclared here."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("cmp", HERE / "compose.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m.BANDS


def full_ceilings(total: int) -> dict:
    """Absolute word ceilings per reading from the publication bands."""
    b = _bands()
    return {depth: min(max(1, total), max(MINIMUM[depth],
                        int(total * b[depth][1])))
            for depth in b}


def output_part_word_budget(output_budget_words=None,
                            output_budget_tokens=None) -> int | None:
    """Safe prose budget for one independently valid model response.

    Publication ceilings describe the complete artifact and are deliberately
    not clipped to this value. The value only decides how many calls compose
    that artifact.
    """
    token_cap = route_caps.explicit_output_budget_tokens(output_budget_tokens)
    word_cap = route_caps.explicit_output_budget(output_budget_words)
    if token_cap is not None:
        word_cap = route_caps.tokens_to_words(token_cap)
    if word_cap is None:
        return None
    return max(8, int(word_cap * PART_OUTPUT_SHARE))


def batched_output_detail(total: int, output_budget_words=None,
                          output_budget_tokens=None) -> dict:
    """Pure plan for crossing a route's per-response output boundary."""
    ceilings = full_ceilings(total)
    budget = output_part_word_budget(output_budget_words,
                                     output_budget_tokens)
    if budget is None:
        return {"needed": False, "part_words": None,
                "parts": {"detailed": 1, "brief": 1},
                "ceilings": ceilings}
    counts = {depth: max(1, (ceiling + budget - 1) // budget)
              for depth, ceiling in ceilings.items()}
    return {"needed": sum(ceilings.values()) > budget,
            "part_words": budget, "parts": counts,
            "ceilings": ceilings}


def clip_pair_ceilings(desired: dict, pair_cap) -> tuple[dict | None, bool]:
    """Scale a publication-band pair down to one reply's generation envelope.

    The publication bands stay editorial maxima. When a route declares how
    much it can type, the model-facing {D_WORDS}/{B_WORDS} shrink to that
    envelope. Shorter is not a defect. Returns (ceilings, route_limited).
    """
    if not isinstance(desired, dict):
        return None, False
    detailed = int(desired.get("detailed") or 0)
    brief = int(desired.get("brief") or 0)
    if detailed < 1 or brief < 1:
        return None, False
    if pair_cap is None:
        return {"detailed": detailed, "brief": brief}, False
    try:
        cap = int(pair_cap)
    except (TypeError, ValueError):
        return {"detailed": detailed, "brief": brief}, False
    floor = MINIMUM["detailed"] + MINIMUM["brief"]
    if cap < floor:
        return None, True
    if detailed + brief <= cap:
        return {"detailed": detailed, "brief": brief}, False
    d_ratio = detailed / max(1, detailed + brief)
    d = min(detailed, max(MINIMUM["detailed"], int(cap * d_ratio)))
    b = min(brief, max(MINIMUM["brief"], cap - d))
    while d + b > cap and d > MINIMUM["detailed"]:
        d -= 1
    while d + b > cap and b > MINIMUM["brief"]:
        b -= 1
    if d + b > cap:
        return None, True
    return {"detailed": d, "brief": b}, True


def generation_ceilings(total: int, output_budget_words=None,
                        output_budget_tokens=None) -> tuple[dict | None, bool]:
    """Model-facing pair maxima for one write or repair reply."""
    token_cap = route_caps.explicit_output_budget_tokens(output_budget_tokens)
    word_cap = route_caps.explicit_output_budget(output_budget_words)
    if token_cap is not None:
        pair_cap = route_caps.tokens_to_words(token_cap)
    else:
        pair_cap = word_cap
    return clip_pair_ceilings(full_ceilings(total), pair_cap)


def _templates():
    return pair_review.write_template(), pair_review.audit_template()


def _capability_from_env(ms=None):
    """Read explicit caps, then the declared first-route capability.

    The fallback is deliberately generic: it examines the resolved role
    chains, not model names. Explicit environment caps remain useful for a
    temporary qualification run and take precedence over the device-local
    route table.
    """
    def _int(name):
        try:
            value = int(os.environ.get(name, ""))
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None
    word_cap = _int("SUMM_ROUTE_CAPABILITY_WORDS")
    word_output = _int("SUMM_ROUTE_OUTPUT_WORDS")
    token_cap = _int("SUMM_ROUTE_CONTEXT_TOKENS")
    token_output = _int("SUMM_ROUTE_OUTPUT_TOKENS")
    splitter = getattr(ms, "split_entry", None) if ms is not None else None
    if ms is not None and token_cap is None and callable(splitter):
        declared = route_caps.primary_chain_capability(
            {"write": getattr(ms, "MODELS", ()),
             "audit": getattr(ms, "AUDIT_MODELS", ()),
             "repair": getattr(ms, "REPAIR_MODELS", ())},
            splitter, getattr(ms, "HARNESS", "agy"))
        if declared:
            token_cap = declared["context_tokens"]
            if token_output is None:
                token_output = declared.get("output_tokens")
    if (word_cap is None and token_cap is None and word_output is None
            and token_output is None):
        # The production default is a qualified token envelope, not the old
        # 100,000-whitespace-word fiction.  A device can replace it with a
        # declared route capability or an explicit qualification override.
        token_cap = route_caps.DEFAULT_QUALIFIED_CONTEXT_TOKENS
    return word_cap, word_output, token_cap, token_output


def fit_detail(source_words: int, capability_words=None,
               output_budget_words=None, *, capability_tokens=None,
               output_budget_tokens=None) -> dict:
    """Whether one whole-source write/review/repair cycle fits the direct
    route: the largest composed prompt plus the expected pair output."""
    write_tpl, audit_tpl = _templates()
    publication = full_ceilings(source_words)
    generation, route_limited = generation_ceilings(
        source_words, output_budget_words, output_budget_tokens)
    if generation is None:
        return {"fits": False, "source_words": source_words,
                "request_words": 0, "output_words": 0,
                "publication_ceiling_words": (
                    publication["detailed"] + publication["brief"]),
                "base_generation_ceiling_words": 0,
                "route_limited": True,
                "budget_words": route_caps.budget_words(capability_words),
                "output_budget_words": output_budget_words,
                "request_tokens": 0,
                "budget_tokens": (int(capability_tokens)
                                   if capability_tokens is not None else
                                   route_caps.words_to_tokens(
                                       route_caps.budget_words(
                                           capability_words))),
                "output_budget_tokens": output_budget_tokens}
    out = generation["detailed"] + generation["brief"]
    writer = len(write_tpl.split()) + source_words
    # The review carries the source, the generation-capped pair, and its
    # frame; repairs resemble it. Score the largest prompt plus one pair
    # output. Editorial publication bands do not reserve output capacity.
    review = len(audit_tpl.split()) + source_words + out
    input_words = max(writer, review)
    request = input_words + out
    if capability_tokens is not None:
        fits = route_caps.direct_ok_tokens(
            input_words, 0, capability_tokens, output_words=out,
            output_budget_tokens=output_budget_tokens)
        budget_words = route_caps.tokens_to_words(capability_tokens)
    else:
        fits = route_caps.direct_ok(request, 0, capability_words,
                                    output_words=out,
                                    output_budget_words=output_budget_words)
        budget_words = route_caps.budget_words(capability_words)
    return {"fits": fits, "source_words": source_words,
            "request_words": request, "output_words": out,
            "publication_ceiling_words": (
                publication["detailed"] + publication["brief"]),
            "base_generation_ceiling_words": out,
            "route_limited": route_limited,
            "budget_words": budget_words,
            "output_budget_words": output_budget_words,
            "request_tokens": route_caps.words_to_tokens(request),
            "budget_tokens": (int(capability_tokens)
                               if capability_tokens is not None else
                               route_caps.words_to_tokens(budget_words)),
            "output_budget_tokens": output_budget_tokens}


def fits(source_words: int, capability_words=None,
         output_budget_words=None) -> bool:
    """Admission predicate for the direct route. Pure: no model calls."""
    return fit_detail(source_words, capability_words,
                      output_budget_words)["fits"]


def source_blocks(source: str, target_words: int) -> list[dict]:
    """Split source words once, preferring paragraph and sentence boundaries.

    The word spans are deterministic and non-overlapping. Whitespace inside a
    block is preserved, while the boundary metadata makes coverage auditable
    without retaining another copy of the source.
    """
    if target_words <= 0:
        return []
    tokens = list(re.finditer(r"\S+", source or ""))
    if not tokens:
        return []
    preferred = set()
    for i, token in enumerate(tokens, 1):
        text = token.group(0)
        after = source[token.end():tokens[i].start() if i < len(tokens) else len(source)]
        if (re.search(r"[.!?][\"'’”)]*$", text)
                or "\n\n" in after):
            preferred.add(i)

    blocks = []
    start = 0
    while start < len(tokens):
        hard_end = min(len(tokens), start + int(target_words))
        choices = [end for end in preferred if start < end <= hard_end]
        end = max(choices, default=hard_end)
        leftover = len(tokens) - end
        # A sentence boundary one word from the end used to mint a 1-word
        # window. Models then returned {"capsule":""} and the whole document
        # was discarded. Fold a stub remainder into this block.
        if 0 < leftover < 8:
            end = len(tokens)
        first, last = tokens[start], tokens[end - 1]
        blocks.append({
            "window_id": f"window-{len(blocks) + 1:03d}",
            "start_word": start + 1,
            "end_word": end,
            "words": end - start,
            "text": source[first.start():last.end()],
        })
        start = end
    return blocks


def window_plan(source: str, capability_words=None,
                output_words=WINDOW_CAPSULE_WORDS, *, capability_tokens=None) -> dict:
    """Return the pure deterministic window plan for a source."""
    template = (HERE / "prompts" / "full-window.txt").read_text()
    if capability_tokens is not None:
        target = route_caps.window_source_words_tokens(
            capability_tokens,
            overhead_words=len(template.split()) + 64,
            output_words=output_words)
    else:
        target = route_caps.window_source_words(
            capability_words,
            overhead_words=len(template.split()) + 64,
            output_words=output_words)
    blocks = source_blocks(source, target)
    return {"target_words": target, "capsule_words": output_words,
            "windows": [
        {key: block[key] for key in
         ("window_id", "start_word", "end_word", "words")}
        for block in blocks
    ]}


def _window_ceilings(total: int, evidence_words: int, capability_words,
                     output_budget_words, frame_words: int, *,
                     capability_tokens=None, output_budget_tokens=None,
                     review_window_words=0, finding_words=512,
                     audit_output_words=1024) -> dict | None:
    """Fit the final pair and one enclosed repair inside the route budget.

    Direct Full keeps its existing source-scaled bands. Windowed Full has to
    leave room for its compact evidence and for a repair prompt, so it scales
    the same bands down only when the generic route budget requires it.
    """
    # A windowed Full repair contains the current pair, the implicated raw
    # packet, findings, and a replacement pair.  Reserve all of those before
    # choosing output ceilings; otherwise a successful capsule pass can still
    # discover that its first repair prompt cannot fit.
    reserved_words = (int(evidence_words) + int(frame_words)
                      + int(review_window_words) + int(finding_words)
                      + int(audit_output_words))
    if capability_tokens is not None:
        used = route_caps.words_to_tokens(reserved_words)
        pair_cap_tokens = (int(capability_tokens) - used) // 2
        output_cap_tokens = route_caps.explicit_output_budget_tokens(
            output_budget_tokens)
        if output_cap_tokens is not None:
            pair_cap_tokens = min(pair_cap_tokens, output_cap_tokens)
        pair_cap = route_caps.tokens_to_words(pair_cap_tokens)
    else:
        budget = route_caps.budget_words(capability_words)
        output_cap = route_caps.explicit_output_budget(output_budget_words)
        available = budget - reserved_words
        pair_cap = available // 2  # current candidate plus the repair response
        if output_cap is not None:
            pair_cap = min(pair_cap, output_cap)
    desired = full_ceilings(total)
    clipped, _limited = clip_pair_ceilings(desired, pair_cap)
    return clipped


def bounded_window_repair_budget(
        window_count: int,
        logical_budget: int = FULL_MAX_LOGICAL_CALLS) -> int:
    """Return the repair count that fits the bounded window topology.

    A windowed candidate costs one call per evidence window, one synthesis
    call, and one review per window. Each repair costs one patch call plus a
    complete re-review of those windows. The bound is computed before any
    repair starts, so a large source cannot turn three repairs into an
    unbounded call fan-out.
    """
    try:
        count = int(window_count)
        budget = int(logical_budget)
    except (TypeError, ValueError):
        return 0
    if count <= 0 or budget <= 0:
        return 0
    initial_calls = 2 * count + 2  # capsules + synthesis + local audits + global audit
    repair_cost = count + 2       # patch + local re-audits + global re-audit
    return min(REPAIR_BUDGET,
               max(0, budget - initial_calls) // repair_cost)


def _run_pair(*, source: str, review_source: str, out_dir: pathlib.Path,
              total: int, source_sha256: str, ms, ss, base_prompt: str,
              audit_tpl: str, ceilings: dict, fit: dict, route: str,
              label: str, initial_stage: str, stage_prefix: str,
              result_options: dict, audit_options: dict,
              review_sources=None, report_extra=None,
              review_packets=None, global_review=None,
              repair_budget=REPAIR_BUDGET, local_scope=None,
              global_scope=None, upstream_findings=(),
              initial_pair=None, writing_plan=None,
              contract_version="v1", producer_provenance=None,
              assignment_packets=None, candidate_blocks=None) -> int:
    """Run the shared final-pair lifecycle for direct, windowed, or Corpus
    evidence. ``local_scope`` is the audit's framing for one review packet
    when there are several; the default describes a source window."""
    local_scope = local_scope or (
        "The supplied source is one local packet, not the whole document. "
        "Judge claims against this packet only; absence from it is not "
        "evidence that a claim is unsupported. Cross-packet relationships "
        "must be reported for the global review.")
    global_scope = global_scope or (
        "The supplied source is the complete compact, source-linked evidence "
        "for this document. Perform a global relationship review of the "
        "actual candidate: check cross-window chronology, attribution, "
        "quantities and units, disagreement, and causal or consensus bridges. "
        "Do not treat a compact evidence omission as proof that the raw "
        "source lacks a fact. Name the relevant window identifiers in each "
        "finding when known.")
    prompt = custom_instructions.decorate_prompt(base_prompt)
    v2 = contract_version == "v2"
    plan_context = (json.dumps(writing_plan, indent=2, ensure_ascii=False)
                    if writing_plan else "(not supplied)")
    working_blocks = (json.loads(json.dumps(candidate_blocks))
                      if isinstance(candidate_blocks, dict) else None)

    def ask(value, stage, chain, options):
        raw = ms.run(value, out_dir, chain, stage,
                     validate=lambda r: json_contract.parse(
                         r, json_contract.FULL_RESULT, stage),
                     gateway_options=options)
        obj = json_contract.parse(raw, json_contract.FULL_RESULT, stage)
        return ((obj.get("detailed") or "").strip(),
                (obj.get("brief") or "").strip())

    def _repair_pair(current, stage):
        import pair_patch
        chain = (ms.REPAIR_MODELS if v2 else
                 _last_ok_chain(out_dir, initial_stage, ms.MODELS))
        if v2 and working_blocks is not None and writing_plan is not None:
            component = None
            for record in current.get("audit_records") or []:
                component = writing_contract.joined_component_for_anchor(
                    writing_plan, working_blocks,
                    current["detailed"], current["brief"],
                    str(record.get("artifact") or ""),
                    str(record.get("anchor") or ""))
                if component is not None:
                    break
            if component is not None:
                schema = writing_contract.block_pair_schema(
                    component["packet"])
                source_by_id = {
                    item["id"]: item
                    for item in writing_contract.source_segments(source)}
                evidence = "\n".join(
                    f"[{ident}]\n{source_by_id[ident]['text']}"
                    for ident in component["source_ids"])
                relevant = [
                    item for item in current.get("audit_records") or []
                    if item.get("artifact") == component["artifact"]
                    and item.get("anchor") == component["anchor"]]
                prompt = custom_instructions.decorate_prompt(
                    "Repair one joined presentation component. Return one "
                    "JSON object matching the response shape and nothing "
                    "else. Return every assigned unit exactly once and in "
                    "order. Preserve each unit_id, heading, and "
                    "join_previous value exactly; revise only paragraph "
                    "prose. The units remain separate ownership records even "
                    "though their boundary renders as one paragraph.\n\n"
                    "ASSIGNMENT:\n" + json.dumps(
                        component["packet"], indent=2,
                        ensure_ascii=False) +
                    "\n\nCURRENT BLOCKS:\n" + json.dumps(
                        component["blocks"], indent=2,
                        ensure_ascii=False) +
                    "\n\nFINDINGS:\n" + json.dumps(
                        relevant, indent=2, ensure_ascii=False) +
                    "\n\nSOURCE EVIDENCE:\n" + evidence)
                options = json_contract.options(
                    "summer_joined_block_repair_v1", schema)
                planned_words = writing_contract._packet_output_words(
                    component["packet"])

                def parse_joined(value):
                    obj = json_contract.parse(
                        value, json_contract.FULL_BLOCK_PAIR, stage)
                    updated = writing_contract.apply_joined_component_repair(
                        writing_plan, working_blocks, component, obj)
                    return obj, updated

                last = None
                for retry in (False, True):
                    call_stage = stage if not retry else f"{stage}-retry"
                    try:
                        admission = _admission_kwargs(
                            ms, role="repair",
                            planned_output_words=planned_words,
                            output_overhead_tokens=REPAIR_OUTPUT_OVERHEAD_TOKENS,
                            semantic_retry=False)
                        raw = ms.run(
                            prompt, out_dir, chain, call_stage,
                            validate=lambda value: parse_joined(value)[0],
                            gateway_options=options, **admission)
                        _obj, updated = parse_joined(raw)
                        detailed2 = writing_contract.render_blocks([
                            updated["detailed"][unit["unit_id"]]
                            for unit in writing_plan["units"]])
                        brief2 = writing_contract.render_blocks([
                            updated["brief"][unit["unit_id"]]
                            for unit in writing_plan["units"]
                            if unit["brief_disposition"] == "include"])
                        return detailed2, brief2, {
                            "kind": "joined_blocks",
                            "candidate_blocks": updated,
                            "component": {
                                "artifact": component["artifact"],
                                "anchor": component["anchor"],
                                "unit_ids": component["unit_ids"],
                            },
                        }
                    except Exception as exc:
                        last = exc
                        kind = getattr(exc, "kind", None)
                        if not retry and kind in {
                                "output_limit", "output_incomplete",
                                "unusable", "response_contract"}:
                            continue
                        break
                raise last
        turn = pair_review.repair_turn(
            current, base_prompt, force_patch=v2, bounded=v2)
        fix = custom_instructions.decorate_prompt(turn["prompt"])
        if turn["kind"] == "length":
            return ask(fix, stage, chain, result_options) + (None,)
        last = None
        for retry in ((False, True) if v2 else (False,)):
            call_stage = stage if not retry else f"{stage}-retry"
            try:
                # Repair never inflates the thinking budget. A retry is a
                # second attempt at the ordinary repair envelope.
                repair_admission = (_admission_kwargs(
                    ms, role="repair",
                    planned_output_words=turn["planned_output_words"],
                    output_overhead_tokens=REPAIR_OUTPUT_OVERHEAD_TOKENS,
                    semantic_retry=False) if v2 else {})
                raw = ms.run(fix, out_dir, chain, call_stage,
                             validate=lambda r: json_contract.parse(
                                 r, json_contract.PAIR_PATCH, call_stage),
                             gateway_options=PATCH_REQUEST_OPTIONS,
                             **repair_admission)
                obj = json_contract.parse(
                    raw, json_contract.PAIR_PATCH, call_stage)
                return pair_patch.apply_edits(
                    current["detailed"], current["brief"], obj,
                    allowed_finding_ids=turn["allowed_finding_ids"] or None
                    ) + (obj,)
            except Exception as exc:
                last = exc
                kind = getattr(exc, "kind", None)
                if (v2 and not retry and kind in {
                        "output_limit", "output_incomplete",
                        "unusable", "response_contract"}):
                    print(f"[{label}] {call_stage} failed "
                          f"({str(exc)[:80]}); retrying repair",
                          flush=True)
                    continue
                break
        raise last

    def slight(text, depth):
        floor = min(ceilings[depth], max(8, int(ceilings[depth] * 0.20)))
        n = len(text.split())
        return ([] if n >= floor else
                [f"EXPAND REQUIRED: the {depth} reading is {n} words. "
                 f"Expand it to at least {floor} words using additional "
                 "source-supported material from the supplied evidence. "
                 "Do not pad, repeat, or invent content."])

    def mechanical(detailed, brief):
        ordinary = (ss.defects(detailed, source) + ss.defects(brief, source)
                # Length is a finding like any other: retained, sent once to
                # the same producer to shorten, and disclosed if it stays over.
                + ss.not_a_copy(detailed, source, "detailed", ceilings)
                + ss.not_a_copy(brief, source, "brief", ceilings)
                + slight(detailed, "detailed") + slight(brief, "brief")
                + custom_instructions.quote_defects((detailed, brief), source))
        if not v2:
            return ordinary
        observations = writing_contract.readability_observations(
            detailed, brief)
        return ordinary + writing_contract.blocking_readability_findings(
            observations)

    def usable(detailed, brief, audit_records=()):
        if ss.structural_findings(detailed, brief, source, ceilings):
            return False
        if v2 and writing_contract.blocking_readability_findings(
                writing_contract.readability_observations(detailed, brief)):
            return False
        # Editorial findings and uncertain readability are disclosed quality
        # findings, not structural vetoes. Preserve explicit unreadable reviews.
        if v2 and any(item.get("confirmed_unreadable") is True
                      for item in (audit_records or [])
                      if isinstance(item, dict)):
            return False
        return True

    def publish(candidate, *, repair, pool):
        detailed, brief = candidate["detailed"], candidate["brief"]
        out_dir.mkdir(parents=True, exist_ok=True)
        detailed_bytes = (detailed.rstrip() + "\n").encode("utf-8")
        brief_bytes = (brief.rstrip() + "\n").encode("utf-8")
        (out_dir / "detailed.md").write_bytes(detailed_bytes)
        (out_dir / "brief.md").write_bytes(brief_bytes)
        status = pair_review.resolve_status(candidate)
        # Word counts do not pin bytes: a same-length edit of a published
        # reading used to pass the Corpus seal. Bind the exact bytes.
        upstream = list(upstream_findings or [])
        report = {"path": "full", "route": route,
                  "detailed_sha256": hashlib.sha256(detailed_bytes).hexdigest(),
                  "brief_sha256": hashlib.sha256(brief_bytes).hexdigest(),
                  "upstream_findings": upstream,
                  "quality_status": ("open_findings"
                                     if upstream and status == "pass"
                                     else status),
                  "source_words": total, "source_sha256": source_sha256,
                  "detailed_words": len(detailed.split()),
                  "brief_words": len(brief.split()),
                  "selected": candidate["selected"],
                  "candidate": pair_review.candidate_identity(
                      detailed, brief),
                  "candidates": len(pool), "status": status,
                  "findings": pair_review.disclosed(candidate),
                  "review": candidate["review"], "repair": repair,
                  "repair_budget": repair_budget,
                  "capability": fit}
        if v2:
            report.update({
                "writing_contract": writing_contract.SCHEMA,
                "candidate_state": writing_contract.quality_state(candidate),
                "writing_plan_sha256": (writing_plan or {}).get(
                    "plan_sha256"),
                "producer_provenance": list(producer_provenance or []),
            })
        report.update(report_extra or {})
        (out_dir / "full-report.json").write_text(
            json.dumps(report, indent=2))
        print(f"[{label}] published {len(detailed.split())}w detailed / "
              f"{len(brief.split())}w brief (status: {status}, "
              f"repair: {repair}, "
              f"{len(pair_review.disclosed(candidate))} open finding(s))",
              flush=True)
        for finding in pair_review.disclosed(candidate):
            print(f"[{label}] open finding: {finding}", flush=True)
        for finding in upstream:
            print(f"[{label}] open evidence finding: {finding}", flush=True)
        return 0

    def nothing_usable(pool):
        if any((candidate["detailed"] or "").strip() and
               (candidate["brief"] or "").strip() for candidate in pool):
            print(f"[{label}] no structurally usable candidate — nothing "
                  "published", file=sys.stderr, flush=True)
            return 5
        print(f"[{label}] the model returned no usable prose",
              file=sys.stderr, flush=True)
        return 1

    def audit(detailed, brief, stage, blocks=None):
        # A windowed candidate is reviewed against bounded raw source packets,
        # never against only the compact capsules used for synthesis. Direct
        # Full keeps the one-packet stage names for backwards-compatible
        # evidence; windowed stages receive stable packet ordinals.
        blocks = working_blocks if blocks is None else blocks
        local_packets = review_packets
        compact = global_review
        if (v2 and writing_plan is not None and assignment_packets
                and blocks is not None):
            local_packets, compact = writing_contract.bounded_review_packets(
                writing_plan, assignment_packets, blocks, source,
                detailed, brief)
            local_packets = writing_contract.coalesce_review_packets(
                local_packets)
        packets = list(local_packets or review_sources or (review_source,))
        findings = []
        records = []
        source_context = {}
        failures = []
        successful_review_packets = []
        next_id = 1
        audit_schema = (json_contract.FULL_AUDIT_V2 if v2
                        else json_contract.FULL_AUDIT)

        def _audit_call(stage_name, prompt_text, observations):
            last = None
            for semantic_retry in ((False, True) if v2 else (False,)):
                call_stage = (stage_name if not semantic_retry
                              else f"{stage_name}-retry")
                try:
                    audit_admission = (_admission_kwargs(
                        ms, role="audit",
                        planned_output_words=AUDIT_OUTPUT_WORDS,
                        output_overhead_tokens=AUDIT_OUTPUT_OVERHEAD_TOKENS,
                        semantic_retry=semantic_retry)
                        if v2 else {})
                    raw = ms.run(
                        custom_instructions.decorate_prompt(prompt_text),
                        out_dir, ms.AUDIT_MODELS, call_stage,
                        validate=lambda r, s=call_stage: json_contract.parse(
                            r, audit_schema, s),
                        gateway_options=audit_options,
                        **audit_admission)
                    obj = json_contract.parse(raw, audit_schema, call_stage)
                    raw_findings = list(obj.get("findings") or [])
                    if v2:
                        raw_findings.extend(
                            writing_contract.validate_readability_assessments(
                                obj, observations))
                        # Keep the typed assessment through the internal finding
                        # ledger; do not infer unreadability from editorial prose.
                        unreadable_ids = {
                            item["observation_id"]
                            for item in (obj.get("readability") or [])
                            if item.get("assessment") == "unreadable"}
                        unreadable_anchors = {
                            (item["artifact"], item["anchor"])
                            for item in observations
                            if item.get("observation_id") in unreadable_ids}
                        for item in raw_findings:
                            if (item.get("artifact"), item.get("anchor")) in unreadable_anchors:
                                item["confirmed_unreadable"] = True
                    return obj, raw_findings, call_stage
                except Exception as exc:
                    last = exc
                    kind = getattr(exc, "kind", None)
                    if (v2 and not semantic_retry and kind in {
                            "output_limit", "output_incomplete",
                            "unusable", "response_contract"}):
                        print(f"[{label}] {call_stage} failed "
                              f"({str(exc)[:80]}); retrying audit",
                              flush=True)
                        continue
                    break
            raise last

        for index, packet in enumerate(packets, 1):
            structured = isinstance(packet, dict)
            logical_id = (stage if len(packets) == 1
                          else f"{stage}-{index:03d}")
            packet_source = packet["source"] if structured else packet
            packet_detailed = packet.get("detailed", detailed) if structured else detailed
            packet_brief = packet.get("brief", brief) if structured else brief
            packet_plan = packet.get("plan_context", plan_context) if structured else plan_context
            packet_segments = packet.get("segments") if structured else None
            observations = (list(packet.get("observations") or [])
                            if structured else
                            (writing_contract.readability_observations(
                                detailed, brief) if v2 else []))
            readability_context = writing_contract.audit_context(observations)
            source_context[logical_id] = packet_source
            value = pair_review.fill_audit_prompt(
                audit_tpl,
                scope=(local_scope if (review_sources or review_packets
                                       or assignment_packets) else
                       "The supplied source is complete for this review."),
                source=packet_source, detailed=packet_detailed,
                brief=packet_brief, plan_context=packet_plan,
                readability_context=readability_context,
                segments=packet_segments)
            try:
                obj, raw_findings, attempt_stage = _audit_call(
                    logical_id, value, observations)
            except Exception as exc:
                failures.append(f"{logical_id}: {str(exc)[:120]}")
                print(f"[{label}] {logical_id}: unavailable ({str(exc)[:80]})",
                      file=sys.stderr, flush=True)
                continue
            import pair_patch
            packet_records = pair_patch.assign_finding_ids(
                raw_findings, start=next_id)
            next_id += len(packet_records)
            for item in packet_records:
                item["packet"] = logical_id
                item["attempt_stage"] = attempt_stage
            records.extend(packet_records)
            if structured:
                successful_review_packets.append(packet.get("packet_id"))
            packet_findings = pair_patch.render_findings(
                packet_records, packet=logical_id)
            findings.extend(packet_findings)
            print(f"[{label}] {logical_id}: {obj.get('verdict', '?')}, "
                  f"{len(packet_findings)} finding(s)", flush=True)
        if (v2 and writing_plan is not None and assignment_packets
                and blocks is not None):
            coverage_error = writing_contract.review_coverage_error(
                writing_plan, list(local_packets or []),
                successful_review_packets)
            if coverage_error:
                failures.append(coverage_error)
        if review_sources or review_packets or assignment_packets:
            # Local packets establish whether material was represented
            # faithfully. The compact evidence receives one separate global
            # review for relationships that span packets: chronology,
            # attribution, quantities, disagreement, and causal/consensus
            # bridges. It is intentionally not called against one partial raw
            # window and cannot mistake absence there for unsupported content.
            global_stage = f"{stage}-global"
            structured = isinstance(compact, dict)
            global_source = (compact.get("source", review_source)
                             if structured else review_source)
            global_detailed = (compact.get("detailed", detailed)
                               if structured else detailed)
            global_brief = (compact.get("brief", brief)
                            if structured else brief)
            global_plan = (compact.get("plan_context", plan_context)
                           if structured else plan_context)
            global_segments = compact.get("segments") if structured else None
            global_observations = (list(compact.get("observations") or [])
                                   if structured else
                                   (writing_contract.readability_observations(
                                       detailed, brief) if v2 else []))
            source_context[global_stage] = global_source
            value = pair_review.fill_audit_prompt(
                audit_tpl, scope=global_scope, source=global_source,
                detailed=global_detailed, brief=global_brief,
                plan_context=global_plan,
                readability_context=writing_contract.audit_context(
                    global_observations), segments=global_segments)
            try:
                obj, raw_findings, used_stage = _audit_call(
                    global_stage, value, global_observations)
                import pair_patch
                global_records = pair_patch.assign_finding_ids(
                    raw_findings, start=next_id)
                for item in global_records:
                    item["packet"] = global_stage
                    item["attempt_stage"] = used_stage
                records.extend(global_records)
                global_findings = pair_patch.render_findings(
                    global_records, packet=global_stage)
                findings.extend(global_findings)
                print(f"[{label}] {global_stage}: {obj.get('verdict', '?')}, "
                      f"{len(global_findings)} finding(s)", flush=True)
            except Exception as exc:
                failures.append(f"{global_stage}: {str(exc)[:120]}")
                print(f"[{label}] {global_stage}: unavailable "
                      f"({str(exc)[:80]})", file=sys.stderr, flush=True)
        return findings, source_context, failures, records

    try:
        if initial_pair is None:
            detailed, brief = ask(prompt, initial_stage, ms.MODELS,
                                  result_options)
        else:
            detailed, brief = initial_pair
    except Exception as e:
        # A transport that returns an invalid object despite the controller's
        # validator is still a failed candidate-producing route.  Keep this
        # terminal outcome explicit and leave publication to the caller's
        # normal no-pair handling; do not leak a traceback as if a semantic
        # reviewer rejected an otherwise usable result.
        print(f"[{label}] no usable initial pair ({str(e)[:120]})",
              file=sys.stderr, flush=True)
        return 1
    findings = mechanical(detailed, brief)
    mech0 = list(findings)
    try:
        audit0, source_context0, audit_failures, records0 = audit(
            detailed, brief, f"{stage_prefix}-audit")
        findings += audit0
        review = ("complete" if not audit_failures else
                  "partial: " + "; ".join(audit_failures[:3]))
    except Exception as e:
        # The reviewer is a quality service, not the candidate producer.  A
        # reviewer outage must not erase a structurally unusable pair before
        # its known mechanical defects receive the configured repair
        # opportunity.  The normal loop below will either repair it or retain
        # the only usable candidate; an actually empty pair still cannot be
        # published.
        audit0, source_context0, records0 = [], {}, []
        review = f"unavailable: {stage_prefix}-audit: {str(e)[:80]}"
        # Keep `findings` equal to the mechanical findings established above;
        # those are actionable even when no semantic review receipt exists.

    pool = [pair_review.new_candidate(
        "initial", detailed, brief, mech0, audit0,
        usable=usable(detailed, brief, records0), review=review,
        source_context=source_context0, audit_records=records0)]
    current = pool[0]
    repairs = 0
    while current["findings"] and repairs < repair_budget:
        stage = f"{stage_prefix}-revise-{repairs + 1}"
        try:
            detailed2, brief2, patch = _repair_pair(current, stage)
        except Exception as e:
            print(f"[{label}] repair {repairs + 1} unavailable "
                  f"({str(e)[:80]}) — publishing best retained candidate",
                  file=sys.stderr, flush=True)
            repairs = -1
            break
        repairs += 1
        if patch and working_blocks is not None:
            if patch.get("kind") == "joined_blocks":
                working_blocks = patch["candidate_blocks"]
            else:
                try:
                    working_blocks = writing_contract.rebind_blocks_after_patch(
                        working_blocks, current["detailed"], current["brief"],
                        patch.get("edits") or [])
                except writing_contract.WritingContractError as exc:
                    print(f"[{label}] repair {repairs} cannot preserve block "
                          f"ownership ({str(exc)[:80]}) — publishing best "
                          "retained candidate", file=sys.stderr, flush=True)
                    repairs = -1
                    break
        mechanical2 = mechanical(detailed2, brief2)
        try:
            audit2, source_context2, audit_failures2, records2 = audit(
                detailed2, brief2, f"{stage_prefix}-reaudit-{repairs}",
                working_blocks)
            review2 = ("complete" if not audit_failures2 else
                       "partial: " + "; ".join(audit_failures2[:3]))
            notes = list(audit_failures2)
        except Exception as e:
            audit2, source_context2, records2 = [], {}, []
            review2 = (f"unavailable: {stage_prefix}-reaudit-{repairs}: "
                       f"{str(e)[:80]}")
            notes = [review2]
        current = pair_review.new_candidate(
            f"revise-{repairs}", detailed2, brief2, mechanical2, audit2,
            usable=usable(detailed2, brief2, records2), review=review2,
            notes=notes,
            source_context=source_context2, audit_records=records2)
        pool.append(current)
        if not current["findings"] and review2 == "complete":
            break

    chosen = pair_review.select(pool, prefer_earlier=v2)
    if chosen is None:
        return nothing_usable(pool)
    return publish(chosen,
                   repair="unavailable" if repairs < 0
                   else REPAIR_WORDS[repairs], pool=pool)


def _run_contract_v2(source: str, out_dir: pathlib.Path, total: int,
                     capability_words, output_budget_words, ms, ss, *,
                     capability_tokens=None,
                     output_budget_tokens=None) -> int:
    """Plan discourse once, then pack unchanged editorial units by capacity.

    This path is explicitly selected during qualification. The released v1
    route remains available until the cross-model controls pass.
    """
    ceilings = full_ceilings(total)
    source_sha256 = hashlib.sha256(source.encode("utf-8")).hexdigest()
    retained_plan = str(os.environ.get("SUMM_WRITING_PLAN_FILE") or "").strip()
    plan_origin = "live"

    out_dir.mkdir(parents=True, exist_ok=True)
    plan_stage = "full-v2-plan"
    plan_graph = None
    try:
        if retained_plan:
            plan = writing_contract.load_plan(
                pathlib.Path(retained_plan).expanduser(), source, ceilings)
            plan_origin = "validated_replay"
            print(f"[full-v2] reused source-validated discourse plan "
                  f"{plan['plan_sha256'][:12]}", flush=True)
        else:
            pending = writing_contract.planning_leaves(source, ceilings)
            fragments = []
            while pending:
                leaf = pending.pop(0)
                semantic_retry = bool(leaf.get("semantic_retry"))
                stage = f"{plan_stage}-{leaf['leaf_id'].lower()}" + (
                    "-retry" if semantic_retry else "")
                prompt = custom_instructions.decorate_prompt(
                    writing_contract.planning_leaf_prompt(leaf))
                if semantic_retry:
                    prompt += (
                        "\n\nCORRECTION REQUIRED: The preceding response was "
                        "rejected by the deterministic contract for this exact "
                        "leaf. Correct this problem without changing or "
                        "omitting any owned source id: "
                        + str(leaf.get("validation_error") or
                              "invalid bounded plan")[:240])

                def parse_leaf(value, item=leaf, name=stage):
                    return writing_contract.validate_plan_fragment(
                        json_contract.parse(
                            value, json_contract.FULL_DISCOURSE_PLAN, name),
                        item)

                try:
                    raw = ms.run(
                        prompt, out_dir, ms.PLAN_MODELS, stage,
                        validate=parse_leaf,
                        gateway_options=PLAN_V2_REQUEST_OPTIONS,
                        **_admission_kwargs(
                            ms, role="plan",
                            planned_output_words=PLAN_OUTPUT_WORDS,
                            output_overhead_tokens=
                            PLAN_OUTPUT_OVERHEAD_TOKENS,
                            semantic_retry=semantic_retry))
                    fragments.append((leaf, parse_leaf(raw)))
                except ms.NoCandidate as exc:
                    failure = getattr(exc, "kind", None)
                    if failure == "unusable" and not semantic_retry:
                        retry_leaf = dict(leaf)
                        retry_leaf["semantic_retry"] = True
                        retry_leaf["validation_error"] = (
                            getattr(exc, "detail", "") or str(exc))
                        pending = [retry_leaf] + pending
                        continue
                    if failure in {"capacity", "output_limit",
                                   "output_incomplete"} and not leaf.get(
                                       "capacity_split"):
                        children = writing_contract.split_planning_leaf(leaf)
                        if children:
                            pending = [dict(child, capacity_split=True)
                                       for child in children] + pending
                            continue
                    raise

            reductions = []
            groups = writing_contract.planning_reduction_groups(fragments)
            for index, group in enumerate(groups, 1):
                node_id = f"N{index:03d}"
                stage = f"{plan_stage}-reduce-{index:03d}"
                prompt = custom_instructions.decorate_prompt(
                    writing_contract.planning_reduction_prompt(group, node_id))

                def parse_node(value, items=group, ident=node_id, name=stage):
                    return writing_contract.validate_plan_reduction(
                        json_contract.parse(
                            value, json_contract.PLAN_REDUCTION, name),
                        items, ident)

                raw = ms.run(
                    prompt, out_dir, ms.PLAN_MODELS, stage,
                    validate=parse_node,
                    gateway_options=json_contract.options(
                        "summer_plan_reduction", json_contract.PLAN_REDUCTION),
                    **_admission_kwargs(
                        ms, role="plan", planned_output_words=500,
                        output_overhead_tokens=PLAN_OUTPUT_OVERHEAD_TOKENS))
                reductions.append(parse_node(raw))

            cards = [card for leaf, fragment in fragments
                     for card in writing_contract._fragment_cards(leaf, fragment)]
            nominated = {handle for node in reductions
                         for handle in node.get("brief_candidates") or []}
            eligible = [card for card in cards if card["handle"] in nominated]
            selection_stage = f"{plan_stage}-select"
            selection_prompt = custom_instructions.decorate_prompt(
                writing_contract.planning_selection_prompt(
                    reductions, cards, ceilings["brief"]))

            def parse_selection(value):
                return writing_contract.validate_plan_selection(
                    json_contract.parse(
                        value, json_contract.PLAN_BRIEF_SELECTION,
                        selection_stage), eligible, ceilings["brief"])

            raw = ms.run(
                selection_prompt, out_dir, ms.PLAN_MODELS, selection_stage,
                validate=parse_selection,
                gateway_options=json_contract.options(
                    "summer_plan_brief_selection",
                    json_contract.PLAN_BRIEF_SELECTION),
                **_admission_kwargs(
                    ms, role="plan", planned_output_words=500,
                    output_overhead_tokens=PLAN_OUTPUT_OVERHEAD_TOKENS))
            selection = parse_selection(raw)
            plan = writing_contract.assemble_bounded_plan(
                source, ceilings, fragments, selection)
            plan_graph = {
                "schema": "summer.bounded-writing-plan.v1",
                "source_sha256": source_sha256,
                "leaf_max_ids": writing_contract.PLAN_LEAF_MAX_IDS,
                "reducer_fanin": writing_contract.PLAN_REDUCE_FANIN,
                "leaves": [{
                    "leaf_id": leaf["leaf_id"],
                    "source_ids": leaf["source_ids"],
                    "detailed_words": leaf["detailed_words"],
                    "brief_words": leaf["brief_words"],
                    "plan": fragment,
                } for leaf, fragment in fragments],
                "reductions": reductions,
                "selection": selection,
            }
    except Exception as exc:
        print(f"[full-v2] no valid discourse plan ({str(exc)[:120]}); "
              "nothing published", file=sys.stderr, flush=True)
        return 1
    writing_contract.save_plan(out_dir / "writing-plan.json", plan)
    if plan_graph is not None:
        (out_dir / "writing-plan-graph.json").write_text(
            json.dumps(plan_graph, indent=2, ensure_ascii=False))

    output_cap = output_part_word_budget(
        output_budget_words, output_budget_tokens)
    if output_cap is None:
        output_cap = sum(plan["allocated_words"].values())
    try:
        packets = writing_contract.build_packets(
            plan, output_words=output_cap)
    except writing_contract.WritingContractError as exc:
        print(f"[full-v2] cannot pack writing plan ({exc}); nothing published",
              file=sys.stderr, flush=True)
        return NOT_SUPPORTED

    if os.environ.get("SUMM_WRITING_ONE_UNIT_PER_REQUEST", "") == "1":
        # Qualification-only isolation: one discourse unit per response while
        # preserving the same plan and schema. A unit selected for Brief may
        # carry both artifacts in that one response.
        by_d = {u["unit_id"]: u for p in packets
                for u in (p.get("detailed") or [])}
        by_b = {u["unit_id"]: u for p in packets
                for u in (p.get("brief") or [])}
        packets = [
            {"detailed": [by_d[unit["unit_id"]]],
             "brief": ([by_b[unit["unit_id"]]]
                       if unit["unit_id"] in by_b else [])}
            for unit in plan["units"]]

    def request_fits(prompt, output_words):
        if capability_tokens is not None:
            return route_caps.direct_ok_tokens(
                len(prompt.split()), capability_tokens=capability_tokens,
                output_words=output_words,
                output_budget_tokens=output_budget_tokens)
        return route_caps.direct_ok(
            len(prompt.split()) + output_words, 0, capability_words,
            output_words=output_words,
            output_budget_words=output_budget_words)

    def split_packet(packet):
        detailed = list(packet.get("detailed") or [])
        brief = list(packet.get("brief") or [])
        if detailed and brief:
            return [{"detailed": detailed, "brief": []},
                    {"detailed": [], "brief": brief}]
        depth, units = (("detailed", detailed) if detailed
                        else ("brief", brief))
        if len(units) < 2:
            return []
        middle = len(units) // 2
        return [writing_contract._packet(depth, units[:middle]),
                writing_contract._packet(depth, units[middle:])]

    # Refine by the fully composed request, not by allocations alone. This is
    # still the same plan and schema; only transport packing changes.
    queue = list(packets)
    packets = []
    while queue:
        packet = queue.pop(0)
        probe = writing_contract.writing_prompt(plan, packet, source)
        words = writing_contract._packet_output_words(packet)
        if request_fits(probe, words):
            packets.append(packet)
            continue
        children = split_packet(packet)
        if not children:
            ids = (writing_contract._ids(packet, "detailed")
                   + writing_contract._ids(packet, "brief"))
            print(f"[full-v2] planned unit {ids} cannot fit the declared "
                  "request capacity; nothing published", file=sys.stderr,
                  flush=True)
            return NOT_SUPPORTED
        queue = children + queue

    if len(packets) > PART_CALL_LIMIT:
        print(f"[full-v2] plan needs {len(packets)} write packets (limit "
              f"{PART_CALL_LIMIT}); nothing published", file=sys.stderr,
              flush=True)
        return NOT_SUPPORTED

    heading_policy = str(
        os.environ.get("SUMM_WRITING_HEADING_POLICY") or "").strip() or None
    recovery_file = str(
        os.environ.get("SUMM_WRITING_RECOVERY_FILE") or "").strip()
    resumed = None
    all_blocks = {"detailed": {}, "brief": {}}
    accepted = []
    producers = []
    previous = {"detailed": "", "brief": ""}
    last_stage = plan_stage
    if recovery_file:
        try:
            resumed = json.loads(
                pathlib.Path(recovery_file).expanduser().read_text())
            if resumed.get("heading_policy"):
                heading_policy = resumed["heading_policy"]
            if resumed.get("wave") not in {"rejected", "repacked"}:
                raise writing_contract.WritingContractError(
                    "writer recovery resume requires a rejected or repacked wave")
            root_packet = resumed.get("root_packet") or {}
            writing_contract.validate_writer_recovery(
                resumed, plan, root_packet)
            pending = {tuple(item) for item in resumed.get("pending_keys") or []}
            for key, item in (resumed.get("accepted") or {}).items():
                artifact, ident = str(key).split(":", 1)
                block = (item or {}).get("block") or {}
                writing_contract.validate_blocks(
                    {"detailed": [block] if artifact == "detailed" else [],
                     "brief": [block] if artifact == "brief" else []},
                    writing_contract._packet(artifact, [{
                        "unit_id": ident, "heading": "",
                        "paragraphs": ["x"], "words": 1,
                        "source_ids": [], "title": "", "topic": "",
                        "relation_to_previous": "",
                        "governing_qualifications": [],
                    }]),
                    heading_policy=heading_policy)
                all_blocks[artifact][ident] = block
                rendered = writing_contract.render_blocks([block])
                previous[artifact] = (
                    previous[artifact] + "\n\n" + rendered).strip()
                receipt = (item or {}).get("receipt") or {}
                producers.append({
                    "unit_id": ident, "artifact": artifact,
                    "stage": receipt.get("stage") or "inherited",
                    "logical_stage": receipt.get("logical_stage")
                    or receipt.get("stage") or "inherited",
                    "producer": list(receipt.get("producer") or []),
                })
            packets = [
                child for child in writing_contract.atomic_packets(root_packet)
                if writing_contract.obligation_keys(child)[0] in pending]
            print("[full-v2] resuming rejected writer assignment as "
                  f"{len(packets)} atomic child write(s)", flush=True)
        except (OSError, ValueError, json.JSONDecodeError,
                writing_contract.WritingContractError) as exc:
            print(f"[full-v2] cannot resume writer recovery ({exc}); "
                  "nothing published", file=sys.stderr, flush=True)
            return 1
        if len(packets) > PART_CALL_LIMIT:
            print(f"[full-v2] recovery needs {len(packets)} write packets "
                  f"(limit {PART_CALL_LIMIT}); nothing published",
                  file=sys.stderr, flush=True)
            return NOT_SUPPORTED

    recovery_state = dict(resumed) if resumed else None
    recovery_kinds = {
        "response_contract", "unusable", "output_limit", "output_incomplete",
    }

    def persist_recovery(state):
        (out_dir / "writer-recovery.json").write_text(
            json.dumps(state, indent=2, ensure_ascii=False))

    def produce(packet, stage, *, wave="fresh"):
        nonlocal last_stage, recovery_state
        keys = writing_contract.obligation_keys(packet)
        schema = writing_contract.block_pair_schema(
            packet, heading_policy=heading_policy)
        options = json_contract.options("summer_full_blocks_v2", schema)
        prior_parts = []
        brief_ids = writing_contract._ids(packet, "brief")
        detailed_ids = writing_contract._ids(packet, "detailed")
        if brief_ids and not detailed_ids:
            for ident in brief_ids:
                block = all_blocks["detailed"].get(ident)
                if block:
                    prior_parts.append(
                        "SAME-UNIT COMPLETED DETAILED FOR " + ident +
                        " (continuity/comparison only, never factual "
                        "evidence):\n" +
                        writing_contract.render_blocks([block]))
        for depth in ("detailed", "brief"):
            if packet.get(depth) and previous[depth]:
                prior_parts.append(
                    f"PRECEDING {depth.upper()} END:\n" +
                    " ".join(previous[depth].split()[-120:]))
        prompt = writing_contract.writing_prompt(
            plan, packet, source, "\n\n".join(prior_parts) or "(none)",
            heading_policy=heading_policy)
        words = writing_contract._packet_output_words(packet)

        def parse(value):
            # The request schema requires join_previous. Accepting an older
            # CLI response without it is a compatibility migration to false;
            # validate_blocks still enforces assignment ids and every other
            # field locally.
            obj = json_contract.parse(
                value, json_contract.FULL_BLOCK_PAIR, stage)
            return writing_contract.validate_blocks(
                obj, packet, heading_policy=heading_policy)

        tries = (ATOMIC_CHILD_ATTEMPTS
                 if wave == "repacked" and len(keys) == 1 else 1)
        obj = None
        last_exc = None
        success_stage = stage

        def writer_failure_kind(exc):
            return (getattr(exc, "recovery_kind", None)
                    or getattr(exc, "kind", None)
                    or ("response_contract"
                        if isinstance(exc, (
                            writing_contract.WriterResponseError,
                            json_contract.ContractError))
                        else "unusable"))

        def persist_blocked(exc, kind):
            nonlocal recovery_state
            if wave == "repacked" and recovery_state is not None:
                recovery_state["wave"] = "blocked"
                recovery_state["unresolved"] = [list(key) for key in keys]
                recovery_state["stop_reason"] = (
                    getattr(exc, "detail", "") or str(exc) or kind)[:240]
                persist_recovery(recovery_state)

        for attempt in range(tries):
            call_stage = (stage if attempt == 0
                          else f"{stage}-retry{attempt:02d}")
            try:
                raw = ms.run(
                    custom_instructions.decorate_prompt(prompt), out_dir,
                    ms.MODELS, call_stage, validate=parse,
                    gateway_options=options,
                    **_admission_kwargs(
                        ms, role="write",
                        planned_output_words=words,
                        output_overhead_tokens=
                        WRITE_OUTPUT_OVERHEAD_TOKENS,
                        semantic_retry=False))
                obj = parse(raw)
                last_exc = None
                success_stage = call_stage
                break
            except Exception as exc:
                if not isinstance(exc, (ms.NoCandidate,
                                        writing_contract.WriterResponseError,
                                        writing_contract.WritingContractError,
                                        json_contract.ContractError)):
                    raise
                last_exc = exc
                recovery_kind = writer_failure_kind(exc)
                eligible = recovery_kind in recovery_kinds
                if eligible and attempt + 1 < tries:
                    print(f"[full-v2] {call_stage} failed "
                          f"({str(exc)[:80]}); retrying atomic child "
                          f"{attempt + 2}/{tries}", flush=True)
                    continue
                if (len(keys) > 1 and wave == "fresh"
                        and recovery_kind in recovery_kinds):
                    recovery_state = writing_contract.new_writer_recovery(
                        plan, packet, heading_policy=heading_policy,
                        source_sha256=source_sha256)
                    recovery_state["wave"] = "rejected"
                    recovery_state["rejection"] = {
                        "stage": stage, "kind": recovery_kind,
                        "detail": getattr(exc, "detail", "") or str(exc),
                        "attempts": getattr(exc, "attempts", []),
                    }
                    persist_recovery(recovery_state)
                    recovery_state["wave"] = "repacked"
                    persist_recovery(recovery_state)
                    children = writing_contract.atomic_packets(packet)
                    for index, child in enumerate(children, 1):
                        if not produce(child, f"{stage}-r{index:02d}",
                                       wave="repacked"):
                            persist_blocked(exc, recovery_kind)
                            return False
                    return True
                persist_blocked(exc, recovery_kind)
                print(f"[full-v2] {stage} failed ({str(exc)[:100]}); "
                      "bounded recovery exhausted", file=sys.stderr, flush=True)
                return False
        if obj is None:
            persist_blocked(last_exc, writer_failure_kind(last_exc)
                            if last_exc is not None else "unusable")
            print(f"[full-v2] {stage} failed ({str(last_exc)[:100]}); "
                  "bounded recovery exhausted", file=sys.stderr, flush=True)
            return False
        last_stage = success_stage
        chain = _last_ok_chain(out_dir, success_stage, ms.MODELS)
        receipt = {
            "stage": success_stage, "logical_stage": stage,
            "producer": list(chain),
        }
        for depth in ("detailed", "brief"):
            for block in obj[depth]:
                if wave == "repacked" and recovery_state is not None:
                    recovery_state = writing_contract.accept_recovery_block(
                        recovery_state, depth, block["unit_id"], block,
                        receipt)
                    persist_recovery(recovery_state)
                all_blocks[depth][block["unit_id"]] = block
                rendered = writing_contract.render_blocks([block])
                previous[depth] = (previous[depth] + "\n\n" + rendered).strip()
                producers.append({
                    "unit_id": block["unit_id"], "artifact": depth,
                    "stage": success_stage, "logical_stage": stage,
                    "producer": list(chain),
                })
        accepted.append({
            "stage": success_stage,
            "logical_stage": stage,
            "detailed_units": writing_contract._ids(packet, "detailed"),
            "brief_units": writing_contract._ids(packet, "brief"),
            "allocated_words": writing_contract._packet_output_words(packet),
            "prompt_words": len(prompt.split()),
            "packet": packet,
        })
        print(f"[full-v2] {success_stage}: "
              f"{len(obj['detailed'])} Detailed / {len(obj['brief'])} Brief "
              "block(s)", flush=True)
        return True

    for index, packet in enumerate(packets, 1):
        wave = ("repacked" if resumed and resumed.get("wave") in {
            "rejected", "repacked"} else "fresh")
        if not produce(packet, f"full-v2-write-{index:03d}", wave=wave):
            print("[full-v2] no complete block pair; nothing published",
                  file=sys.stderr, flush=True)
            return 1

    expected_d = [u["unit_id"] for u in plan["units"]]
    expected_b = [u["unit_id"] for u in plan["units"]
                  if u["brief_disposition"] == "include"]
    if (set(all_blocks["detailed"]) != set(expected_d)
            or set(all_blocks["brief"]) != set(expected_b)):
        print("[full-v2] assembled block coverage mismatch; nothing published",
              file=sys.stderr, flush=True)
        return 1
    detailed_blocks = [all_blocks["detailed"][ident] for ident in expected_d]
    brief_blocks = [all_blocks["brief"][ident] for ident in expected_b]
    try:
        writing_contract.validate_presentation_joins(
            detailed_blocks, expected_d, "detailed")
        writing_contract.validate_presentation_joins(
            brief_blocks, expected_b, "brief")
    except writing_contract.WritingContractError as exc:
        print(f"[full-v2] invalid paragraph continuation ({exc}); "
              "nothing published", file=sys.stderr, flush=True)
        return 1
    detailed = writing_contract.render_blocks(detailed_blocks)
    brief = writing_contract.render_blocks(brief_blocks)
    # Recovery state governs write resumption only. Review always covers the
    # complete frozen plan, including units written in other transport packets.
    assignment = {
        "detailed": writing_contract._assignment(plan, "detailed"),
        "brief": writing_contract._assignment(plan, "brief"),
    }
    assignment_packets = writing_contract.atomic_packets(assignment)
    review_packets, global_review = writing_contract.bounded_review_packets(
        plan, assignment_packets, all_blocks, source, detailed, brief)
    review_packets = writing_contract.coalesce_review_packets(review_packets)
    length_findings = writing_contract.block_word_findings(plan, all_blocks)
    audit_tpl = pair_review.audit_template("v2")
    fit = {
        "fits": True, "route": "writing-contract-v2",
        "source_words": total, "packets": len(packets),
        "allocated_words": dict(plan["allocated_words"]),
        "output_budget_words": output_budget_words,
        "output_budget_tokens": output_budget_tokens,
        "budget_words": (route_caps.tokens_to_words(capability_tokens)
                         if capability_tokens is not None else
                         route_caps.budget_words(capability_words)),
        "budget_tokens": (int(capability_tokens)
                          if capability_tokens is not None else
                          route_caps.words_to_tokens(
                              route_caps.budget_words(capability_words))),
    }
    return _run_pair(
        source=source, review_source=source, out_dir=out_dir, total=total,
        source_sha256=source_sha256, ms=ms, ss=ss,
        base_prompt="Versioned block repair uses the frozen writing plan.",
        audit_tpl=audit_tpl, ceilings=ceilings, fit=fit,
        route="writing-contract-v2", label="full-v2",
        initial_stage=last_stage, stage_prefix="full-v2",
        result_options=RESULT_REQUEST_OPTIONS,
        audit_options=AUDIT_V2_REQUEST_OPTIONS,
        initial_pair=(detailed, brief), writing_plan=plan,
        contract_version="v2", producer_provenance=producers,
        review_packets=review_packets, global_review=global_review,
        assignment_packets=assignment_packets, candidate_blocks=all_blocks,
        report_extra={"writing_plan": "writing-plan.json",
                      "writing_plan_origin": plan_origin,
                      "writing_packets": [
                          {key: value for key, value in item.items()
                           if key != "packet"} for item in accepted],
                      "review_packets": len(review_packets) + 1,
                      "unit_overruns": length_findings})


def _run_batched(source: str, out_dir: pathlib.Path, total: int,
                 capability_words, output_budget_words, ms, ss,
                 *, capability_tokens=None, output_budget_tokens=None) -> int:
    """Compose each logical reading from bounded, ordered model responses.

    The source is partitioned independently for Detailed and Brief. A failed
    response is retried on byte-preserving halves, exactly as source evidence
    windows are, and no partial pair reaches publication. The complete pair
    then enters the ordinary review, repair, selection, and atomic publication
    lifecycle.
    """
    detail = batched_output_detail(total, output_budget_words,
                                   output_budget_tokens)
    part_words = detail["part_words"]
    if not detail["needed"] or part_words is None:
        return NOT_SUPPORTED
    nominal_parts = sum(detail["parts"].values())
    plans = {}
    for depth, count in detail["parts"].items():
        target = max(1, (total + count - 1) // count)
        blocks = source_blocks(source, target)
        plans[depth] = blocks
    # Sentence-safe boundaries can produce more parts than the arithmetic
    # ceiling count. Admission must bound the exact executable plan, not that
    # lower nominal count.
    planned_parts = sum(len(blocks) for blocks in plans.values())
    worst_case_calls = planned_parts * (2 ** (PART_SPLIT_DEPTH + 1) - 1)
    if (planned_parts > PART_CALL_LIMIT
            or worst_case_calls > FULL_MAX_PHYSICAL_ATTEMPTS):
        print(f"[full-batched] output plan needs {planned_parts} base parts "
              f"and up to {worst_case_calls} split attempts; bounded limit is "
              f"{PART_CALL_LIMIT} parts/{FULL_MAX_PHYSICAL_ATTEMPTS} attempts; "
              "nothing published", file=sys.stderr, flush=True)
        return NOT_SUPPORTED

    out_dir.mkdir(parents=True, exist_ok=True)
    plan_record = {
        "route": "batched", "source_words": total,
        "part_word_budget": part_words,
        "nominal_parts": nominal_parts,
        "planned_parts": planned_parts,
        "worst_case_calls": worst_case_calls,
        "output_budget_words": output_budget_words,
        "output_budget_tokens": output_budget_tokens,
        "artifacts": {
            depth: [{key: block[key] for key in
                     ("window_id", "start_word", "end_word", "words")}
                    for block in blocks]
            for depth, blocks in plans.items()},
    }
    (out_dir / "batch-plan.json").write_text(json.dumps(plan_record, indent=2))

    part_tpl = pair_review.reading_part_template()
    accepted = []
    last_stage = "full-batch-detailed-001"

    def parse_part(raw, stage):
        obj = json_contract.parse(raw, json_contract.FULL_READING_PART, stage)
        reading = (obj.get("reading") or "").strip()
        if not reading:
            raise ValueError("empty reading part")
        return reading

    def position_rule(first, last):
        if first and last:
            return ("This is the whole assigned span: open the complete reading "
                    "with orientation, without adding a generic conclusion.")
        if first:
            return ("Open the complete reading with orientation, then cover this "
                    "span without foreshadowing later batches.")
        if last:
            return ("Continue from the preceding prose without a fresh introduction "
                    "and end where the source ends, without a generic conclusion.")
        return ("Continue from the preceding prose without a fresh introduction or "
                "a conclusion, and leave later material to later batches.")

    def prompt_for(depth, part_id, text, words, previous, first, last):
        previous_end = "(none; this is the first accepted part)"
        if previous:
            previous_end = " ".join(previous.split()[-120:])
        return custom_instructions.decorate_prompt(
            part_tpl.replace("{DEPTH}", depth.capitalize())
            .replace("{PART_ID}", part_id)
            .replace("{PART_WORDS}", str(words))
            .replace("{POSITION_RULE}", position_rule(first, last))
            .replace("{PREVIOUS_END}", previous_end)
            .replace("{SOURCE}", text))

    def produce_piece(depth, part_id, text, words, previous, first, last,
                      split_depth=0):
        nonlocal last_stage
        stage = f"full-batch-{depth}-{part_id}"
        prompt = prompt_for(depth, part_id, text, words, previous, first, last)
        try:
            raw = ms.run(prompt, out_dir, ms.MODELS, stage,
                         validate=lambda r, st=stage: parse_part(r, st),
                         gateway_options=PART_REQUEST_OPTIONS)
            reading = parse_part(raw, stage)
            last_stage = stage
            accepted.append({
                "artifact": depth, "part_id": part_id,
                "source_words": len(text.split()),
                "source_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "output_words": len(reading.split()),
                "output_sha256": hashlib.sha256(reading.encode("utf-8")).hexdigest(),
            })
            print(f"[full-batched] {stage}: {len(reading.split())}w",
                  flush=True)
            return reading
        except ms.NoCandidate as exc:
            if (split_depth >= PART_SPLIT_DEPTH
                    or len(text.split()) < 2 * PART_MIN_SOURCE_WORDS):
                print(f"[full-batched] {stage} failed ({str(exc)[:100]}) and "
                      "cannot be split further", file=sys.stderr, flush=True)
                return None
            print(f"[full-batched] {stage} failed ({str(exc)[:100]}) — "
                  "retrying as two smaller source batches",
                  file=sys.stderr, flush=True)
            pieces = _split_text(text)
            outputs = []
            for index, (tag, piece) in enumerate(zip(("a", "b"), pieces)):
                piece_words = min(
                    part_words,
                    max(8, int(words * len(piece.split()) /
                               max(1, len(text.split())))))
                prior = "\n\n".join(outputs) or previous
                got = produce_piece(
                    depth, part_id + tag, piece, piece_words, prior,
                    first and index == 0, last and index == 1,
                    split_depth + 1)
                if got is None:
                    return None
                outputs.append(got)
            return "\n\n".join(outputs)

    artifacts = {}
    for depth in ("detailed", "brief"):
        outputs = []
        blocks = plans[depth]
        for index, block in enumerate(blocks, 1):
            allocated = min(
                part_words,
                max(8, int(detail["ceilings"][depth] * block["words"] /
                           max(1, total))))
            got = produce_piece(
                depth, f"{index:03d}", block["text"], allocated,
                "\n\n".join(outputs), index == 1, index == len(blocks))
            if got is None:
                print(f"[full-batched] no complete {depth} reading; nothing "
                      "published", file=sys.stderr, flush=True)
                return 1
            outputs.append(got)
        artifacts[depth] = "\n\n".join(outputs).strip()

    source_sha256 = hashlib.sha256(source.encode("utf-8")).hexdigest()
    pair_tpl, audit_tpl = _templates()
    ceilings = detail["ceilings"]
    base_prompt = (pair_tpl.replace("{SOURCE}", source)
                   .replace("{EVIDENCE_KIND}", "the complete source")
                   .replace("{TASK_RULES}", "")
                   .replace("{D_WORDS}", str(ceilings["detailed"]))
                   .replace("{B_WORDS}", str(ceilings["brief"])))
    fit = fit_detail(total, capability_words, output_budget_words,
                     capability_tokens=capability_tokens,
                     output_budget_tokens=output_budget_tokens)
    fit.update({"route": "batched", "part_word_budget": part_words,
                "nominal_parts": nominal_parts,
                "planned_parts": planned_parts,
                "accepted_parts": len(accepted)})
    return _run_pair(
        source=source, review_source=source, out_dir=out_dir, total=total,
        source_sha256=source_sha256, ms=ms, ss=ss,
        base_prompt=base_prompt, audit_tpl=audit_tpl, ceilings=ceilings,
        fit=fit, route="batched", label="full-batched",
        initial_stage=last_stage, stage_prefix="full-batched",
        result_options=RESULT_REQUEST_OPTIONS,
        audit_options=AUDIT_REQUEST_OPTIONS,
        report_extra={"batch_plan": "batch-plan.json",
                      "output_parts": accepted},
        initial_pair=(artifacts["detailed"], artifacts["brief"]))


def run(source_path: pathlib.Path, out_dir: pathlib.Path,
        capability_words=None, output_budget_words=None) -> int:
    # Full Summary does not import or execute the legacy ledger route. Keep the
    # compatibility loader above only for old trace tooling and tests.
    ms, ss = _runner(), _shortsum()
    source = source_path.read_text(errors="replace")
    total = len(source.split())
    if capability_words is None and output_budget_words is None:
        (capability_words, output_budget_words, capability_tokens,
         output_budget_tokens) = _capability_from_env(ms)
    else:
        capability_tokens = output_budget_tokens = None
    if os.environ.get("SUMM_WRITING_CONTRACT", "").strip().lower() == "v2":
        return _run_contract_v2(
            source, out_dir, total, capability_words, output_budget_words,
            ms, ss, capability_tokens=capability_tokens,
            output_budget_tokens=output_budget_tokens)
    batching = batched_output_detail(total, output_budget_words,
                                     output_budget_tokens)
    if batching["needed"]:
        return _run_batched(
            source, out_dir, total, capability_words, output_budget_words,
            ms, ss, capability_tokens=capability_tokens,
            output_budget_tokens=output_budget_tokens)
    fit = fit_detail(total, capability_words, output_budget_words,
                     capability_tokens=capability_tokens,
                     output_budget_tokens=output_budget_tokens)
    if not fit["fits"]:
        return _run_windowed(source, out_dir, total, capability_words,
                             output_budget_words, ms, ss,
                             capability_tokens=capability_tokens,
                             output_budget_tokens=output_budget_tokens)
    hi, _limited = generation_ceilings(
        total, output_budget_words, output_budget_tokens)
    if hi is None:
        return NOT_SUPPORTED
    source_sha256 = hashlib.sha256(source.encode("utf-8")).hexdigest()
    write_tpl, audit_tpl = _templates()
    base_prompt = (write_tpl.replace("{SOURCE}", source)
                   .replace("{EVIDENCE_KIND}", "the complete source")
                   .replace("{TASK_RULES}", "")
                   .replace("{D_WORDS}", str(hi["detailed"]))
                   .replace("{B_WORDS}", str(hi["brief"])))
    return _run_pair(
        source=source, review_source=source, out_dir=out_dir, total=total,
        source_sha256=source_sha256, ms=ms, ss=ss,
        base_prompt=base_prompt, audit_tpl=audit_tpl, ceilings=hi, fit=fit,
        route="direct", label="full", initial_stage="full",
        stage_prefix="full", result_options=RESULT_REQUEST_OPTIONS,
        audit_options=AUDIT_REQUEST_OPTIONS,
        review_sources=None)


def _run_windowed(source: str, out_dir: pathlib.Path, total: int,
                  capability_words, output_budget_words, ms, ss,
                  capability_tokens=None, output_budget_tokens=None) -> int:
    """Window an oversize source, then run one final Full pair lifecycle.

    Windows are source evidence, not published parts: each gets one bounded
    coverage call, there is one synthesis pair, one fresh review of that exact
    pair, and the same three-candidate repair ceiling as direct Full.
    """
    window_tpl = (HERE / "prompts" / "full-window.txt").read_text()
    output_cap = (route_caps.tokens_to_words(output_budget_tokens)
                  if output_budget_tokens is not None else
                  route_caps.explicit_output_budget(output_budget_words))
    if output_cap is not None and output_cap < sum(MINIMUM.values()):
        print("[full-window] output budget cannot fit the minimum Full pair; "
              "context unsupported, nothing published", file=sys.stderr,
              flush=True)
        return NOT_SUPPORTED
    capsule_words = min(WINDOW_CAPSULE_WORDS,
                        output_cap or WINDOW_CAPSULE_WORDS)
    plan = window_plan(source, capability_words, capsule_words,
                       capability_tokens=capability_tokens)
    blocks = source_blocks(source, plan["target_words"])
    if not blocks or len(blocks) > WINDOW_CALL_LIMIT:
        print(f"[full-window] context unsupported: source needs "
              f"{len(blocks)} source windows (limit {WINDOW_CALL_LIMIT}); "
              "nothing published", file=sys.stderr, flush=True)
        return NOT_SUPPORTED
    # Admit the worst case before the first model call: every window failing
    # and being retried as halves then quarters is a seven-node tree.
    worst_case_window_calls = len(blocks) * (2 ** (SPLIT_DEPTH + 1) - 1)
    if worst_case_window_calls > FULL_MAX_PHYSICAL_ATTEMPTS:
        print(f"[full-window] context unsupported: {len(blocks)} windows can "
              f"cost {worst_case_window_calls} capsule calls once subdivision "
              f"is admitted (budget {FULL_MAX_PHYSICAL_ATTEMPTS}); nothing "
              "published", file=sys.stderr, flush=True)
        return NOT_SUPPORTED
    pair_tpl, audit_tpl = _templates()
    frame_words = max(len(pair_tpl.split()), len(audit_tpl.split())) + 256
    # Admit the complete worst-case topology before spending a capsule call.
    # Actual capsules are asked to stay at or under this cap. Over-length is
    # a content finding, not a reason to refuse admission: one same-producer
    # shorten, then keep. Worst-case planning still uses the preferred cap.
    max_review_window_words = max(len(block["text"].split()) for block in blocks)

    def ceiling_for(capsule_limit):
        return _window_ceilings(
            total, len(blocks) * (capsule_limit + 4), capability_words,
            output_budget_words, frame_words,
            capability_tokens=capability_tokens,
            output_budget_tokens=output_budget_tokens,
            review_window_words=max_review_window_words)

    # A small declared route may not have room for the maximum capsule, even
    # though it can fit a useful bounded capsule.  Find the largest safe
    # capsule cap before any model call; never discover this after producing
    # several capsules.  The binary search is over a response cap, not a
    # model identity, so the same controller works for every route.
    if ceiling_for(capsule_words) is None:
        lo, hi = 1, capsule_words
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if ceiling_for(mid) is None:
                hi = mid - 1
            else:
                lo = mid
        capsule_words = lo
    if ceiling_for(capsule_words) is None:
        print("[full-window] worst-case capsules leave no safe final-pair "
              "budget; context unsupported before model work", file=sys.stderr,
              flush=True)
        return NOT_SUPPORTED
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "window-plan.json").write_text(json.dumps({
            "route": "windowed", "source_words": total,
            "target_words": plan["target_words"],
            "capsule_words": capsule_words,
            "windows": plan["windows"],
        }, indent=2))
    except OSError as e:
        print(f"[full-window] could not record window plan: {e}",
              file=sys.stderr, flush=True)
        return 1

    capsules = []
    split_children = []
    length_exception = False
    missing_windows = []   # always empty now; kept in the report for readers

    def parse_capsule(raw, stage):
        obj = json_contract.parse(raw, json_contract.FULL_WINDOW_RESULT, stage)
        capsule = (obj.get("capsule") or "").strip()
        if not capsule:
            raise ValueError("empty capsule")
        return obj, capsule

    def window_prompt(window_id, text):
        return custom_instructions.decorate_prompt(
            window_tpl.replace("{WINDOW_ID}", window_id)
            .replace("{CAPSULE_WORDS}", str(capsule_words))
            .replace("{SOURCE}", text))

    def capsule_for(stage, window_id, text, depth=0):
        """The capsule for one window, or for its halves when the whole
        chain fails on the window as a whole.

        A window is never skipped: a chain that produced no
        candidate for a 2,000-word window is asked again for each half, then
        each quarter, before the run concludes that the models themselves
        are not working. Returns the capsule text, or None when every route
        failed on a piece too small to split further.

        Only ms.NoCandidate subdivides. Cancellation, authentication, a
        configuration error, an exhausted deadline, or a defect here must
        not masquerade as a piece being too large.
        """
        try:
            raw = ms.run(window_prompt(window_id, text), out_dir, ms.MODELS,
                         stage,
                         validate=lambda r, st=stage: parse_capsule(r, st)[0],
                         gateway_options=WINDOW_REQUEST_OPTIONS)
            return parse_capsule(raw, stage)[1]
        except ms.NoCandidate as e:
            if depth >= SPLIT_DEPTH or len(text.split()) < 2 * SPLIT_MIN_WORDS:
                print(f"[full-window] {stage} failed ({str(e)[:100]}) and "
                      "cannot be split further", file=sys.stderr, flush=True)
                return None
            print(f"[full-window] {stage} failed ({str(e)[:100]}) — "
                  f"retrying as two halves", file=sys.stderr, flush=True)
            parts = []
            for tag, piece in zip(("a", "b"), _split_text(text)):
                child = f"{window_id}{tag}"
                split_children.append({
                    "window_id": child, "parent": window_id,
                    "words": len(piece.split()),
                    "content_sha256": hashlib.sha256(
                        piece.encode("utf-8")).hexdigest()})
                got = capsule_for(f"{stage}{tag}", child, piece, depth + 1)
                if got is None:
                    return None
                parts.append(got)
            return "\n\n".join(parts)

    for index, block in enumerate(blocks, 1):
        prompt = window_prompt(block["window_id"], block["text"])
        stage = f"full-window-{index:03d}"
        capsule = capsule_for(stage, block["window_id"], block["text"])
        if capsule is None:
            # Every route failed on a piece too small to split: the models
            # are not working, which is the one permitted way to end
            # without a summary. Nothing is skipped and nothing partial is
            # published.
            print(f"[full-window] {stage}: no route produced a capsule for "
                  f"this window or its parts — the models are not working; "
                  "nothing published", file=sys.stderr, flush=True)
            return 1
        disposition = "ok"
        if len(capsule.split()) > capsule_words:
            producer = _last_ok_chain(out_dir, stage, ms.MODELS)
            shorten_stage = f"{stage}-shorten"
            try:
                raw2 = ms.run(
                    _shorten_window_prompt(prompt, capsule, capsule_words),
                    out_dir, producer, shorten_stage,
                    validate=lambda r, st=shorten_stage: parse_capsule(r, st)[0],
                    gateway_options=WINDOW_REQUEST_OPTIONS)
                _obj2, revised = parse_capsule(raw2, shorten_stage)
            except Exception as e:
                print(f"[full-window] {shorten_stage} unavailable "
                      f"({str(e)[:80]}) — retaining the candidate",
                      file=sys.stderr, flush=True)
                revised = None
            capsule, disposition = _keep_window_capsule(
                capsule, revised, capsule_words)
            if disposition != "shortened":
                length_exception = True
                print(f"[full-window] {stage}: length_exception — "
                      f"{len(capsule.split())}w over the preferred "
                      f"{capsule_words}w; keeping the candidate", flush=True)
        capsules.append({"window_id": block["window_id"],
                         "words": len(capsule.split()),
                         "capsule": capsule,
                         "length": disposition})
        print(f"[full-window] {stage}: {len(capsule.split())}w capsule",
              flush=True)

    # Every planned window has a capsule, in order. Complete coverage is a
    # precondition of synthesis, not a finding to disclose.
    got = [item["window_id"] for item in capsules]
    planned = [block["window_id"] for block in blocks]
    if got != planned:
        print("[full-window] window coverage mismatch; nothing published",
              file=sys.stderr, flush=True)
        return 1
    evidence = "\n\n".join(
        f"[{item['window_id']}]\n{item['capsule']}" for item in capsules)
    ceilings = _window_ceilings(
        total, len(evidence.split()), capability_words,
        output_budget_words, frame_words,
        capability_tokens=capability_tokens,
        output_budget_tokens=output_budget_tokens,
        review_window_words=max_review_window_words)
    if ceilings is None:
        # Over-length evidence is not a reason to discard coverage already
        # paid for. The pair prompt still goes to the configured writers;
        # mapsum skips any route that cannot physically take it.
        print("[full-window] window evidence exceeds the preferred synthesis "
              "budget; continuing with the route-limited pair", flush=True)
        ceilings, _limited = generation_ceilings(
            total, output_budget_words, output_budget_tokens)
        if ceilings is None:
            print("[full-window] route output envelope cannot fit the "
                  "minimum pair after coverage; nothing published",
                  file=sys.stderr, flush=True)
            return NOT_SUPPORTED
    base_prompt = (pair_tpl.replace("{SOURCE}", evidence)
                   .replace("{EVIDENCE_KIND}",
                            "condensed evidence prepared in order from the "
                            "complete source")
                   .replace("{TASK_RULES}", "")
                   .replace("{D_WORDS}", str(ceilings["detailed"]))
                   .replace("{B_WORDS}", str(ceilings["brief"])))
    source_sha256 = hashlib.sha256(source.encode("utf-8")).hexdigest()
    fit = {"fits": True, "route": "windowed",
           "source_words": total, "window_words": plan["target_words"],
           "capsule_words": capsule_words, "windows": len(blocks),
           "evidence_words": len(evidence.split()),
           "output_words": ceilings["detailed"] + ceilings["brief"],
           "budget_words": (route_caps.tokens_to_words(capability_tokens)
                             if capability_tokens is not None else
                             route_caps.budget_words(capability_words)),
           "output_budget_words": output_budget_words,
           "budget_tokens": (int(capability_tokens)
                             if capability_tokens is not None else
                             route_caps.words_to_tokens(
                                 route_caps.budget_words(capability_words))),
           "output_budget_tokens": output_budget_tokens}

    # The controller-level ceiling includes the capsule producers, synthesis,
    # every initial raw-source review packet, and every possible repair plus
    # its complete re-review.  This is calculated before the first model call;
    # a source cannot silently recreate the old 53-part/207-call fan-out.
    initial_calls = len(blocks) + 1 + len(blocks) + 1
    repair_cost = len(blocks) + 2
    attempts_per_call = max(
        len(ms.MODELS), len(ms.AUDIT_MODELS), len(ms.REPAIR_MODELS), 1)
    logical_budget = min(FULL_MAX_LOGICAL_CALLS,
                         FULL_MAX_PHYSICAL_ATTEMPTS // attempts_per_call)
    if initial_calls > logical_budget:
        print(f"[full-window] planned topology needs {initial_calls} logical "
              f"calls, but the frozen routes allow {logical_budget} under "
              "the target attempt budget; context unsupported before model "
              "work, nothing published", file=sys.stderr, flush=True)
        return NOT_SUPPORTED
    window_repair_budget = bounded_window_repair_budget(
        len(blocks), logical_budget=logical_budget)
    fit.update({"logical_call_budget": FULL_MAX_LOGICAL_CALLS,
                "physical_attempt_budget": FULL_MAX_PHYSICAL_ATTEMPTS,
                "attempts_per_logical_call": attempts_per_call,
                "effective_logical_call_budget": logical_budget,
                "initial_logical_calls": initial_calls,
                "repair_logical_cost": repair_cost,
                "repair_budget": window_repair_budget})

    return _run_pair(
        source=source, review_source=evidence, out_dir=out_dir, total=total,
        source_sha256=source_sha256, ms=ms, ss=ss,
        base_prompt=base_prompt, audit_tpl=audit_tpl, ceilings=ceilings,
        fit=fit, route="windowed", label="full-window",
        initial_stage="full-window-synthesis", stage_prefix="full-window",
        result_options=RESULT_REQUEST_OPTIONS,
        audit_options=AUDIT_REQUEST_OPTIONS,
        review_sources=[
            f"SOURCE WINDOW {block['window_id']}:\n{block['text']}"
            for block in blocks],
        report_extra={
            "window_count": len(blocks),
            "review_packet_count": len(blocks),
            "window_ids": [block["window_id"] for block in blocks],
            "window_plan": "window-plan.json",
            "length_exception": length_exception,
            "missing_windows": list(missing_windows),
            "split_windows": list(split_children),
            "worst_case_window_calls": worst_case_window_calls,
        }, repair_budget=window_repair_budget)


def main():
    if len(sys.argv) != 3:
        print("fullsum.py SOURCE_FILE OUT_DIR", file=sys.stderr)
        return 2
    return run(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]))


if __name__ == "__main__":
    sys.exit(main())
