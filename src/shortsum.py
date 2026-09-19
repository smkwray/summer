#!/usr/bin/env python3
"""The short-document path: bounded model work and fail-closed publication.

    shortsum.py SOURCE_FILE OUT_DIR

Writes `detailed.md` and `brief.md` into OUT_DIR -- the same two files
`compose.py` produces -- so publication in `summ_cli.py` is unchanged.

WHY THIS EXISTS. The ledger pipeline manages coverage across many parts. A
300-word document may yield only a handful of units, making that machinery both
expensive and structurally fragile. Quick instead writes the pair directly,
audits it, permits at most one revision, and audits those exact revised bytes.

The failure boundary is pair-shaped, not finding-shaped. Once a structurally
usable candidate exists -- non-empty readings that are neither the source
handed back nor prohibited list/table structure, with no fabricated exact
quotation -- semantic findings, reviewer outage, a failed repair, or repair-
budget exhaustion publish the safest retained candidate with explicit
open-findings/review-unavailable status in short-report.json, never "nothing
published". The one repair revises the enclosed candidate (identity, exact
readings, and findings travel together), never a fresh pair from the base
prompt. Nothing is published only when every candidate-producing route fails
to leave a structurally usable pair, or on rejected input, cancellation, or
publication/filesystem failure. The readings themselves never carry a warning
in place of verification; the status lives in the report.
"""
from __future__ import annotations
import json, pathlib, re, sys

HERE = pathlib.Path(__file__).parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import custom_instructions
import json_contract
import pair_review

RESULT_REQUEST_OPTIONS = json_contract.options(
    "summer_quick_result", json_contract.SHORT_RESULT)
AUDIT_REQUEST_OPTIONS = json_contract.options(
    "summer_quick_audit", json_contract.SHORT_AUDIT)
PATCH_REQUEST_OPTIONS = json_contract.options(
    "summer_pair_patch", json_contract.PAIR_PATCH)


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


def numbers(t: str) -> set[str]:
    """Numeric tokens, normalised. Mirrors the ledger's grounding check: a year
    inside a list ("2008, 2009") must not read as the distinct token "2008,"."""
    out = set()
    for m in re.findall(r"\d[\d,]*(?:\.\d+)?%?", t or ""):
        m = m.rstrip(".,;:%").replace(",", "")
        if m:
            out.add(m)
    return out


# Words after which a standalone "I" is an enumerator, not a pronoun.
_ENUMERATOR_HEADS = {"type", "phase", "part", "stage", "class", "section", "war",
                     "chapter", "volume", "book", "act", "level", "tier", "grade",
                     "group", "round", "schedule", "appendix", "article", "title",
                     "table", "figure", "series", "category", "mark", "formula",
                     "division", "region", "zone", "step", "form", "model"}


def _first_person_i(text: str) -> bool:
    """A standalone "I" anywhere is first person unless it follows an
    enumerator head such as "Type I" or "World War I"."""
    for match in re.finditer(r"(?<![A-Za-z0-9'])I(?![A-Za-z0-9'])", text):
        before = text[:match.start()].rstrip()
        head = re.search(r"([A-Za-z]+)$", before)
        if head and head.group(1).lower() in _ENUMERATOR_HEADS:
            continue
        return True
    return False


def defects(text: str, source: str) -> list[str]:
    """Deterministic findings against the published bytes. Cheap, and they never
    vary between runs, so they run before any model is asked to revise."""
    found = []
    if re.search(r"(?m)^\s*(?:[-*+]\s|\d+\.\s|\|)", text):
        found.append("list or table markup")
    # Case matters: "US model" and "Type I error" are not first person, and a
    # false finding here spends a repair on nothing.
    if re.search(r"\b(?:[Ww]e|[Oo]ur|[Uu]s|[Mm]y)\b", text) or _first_person_i(text):
        found.append("first person")
    invented = numbers(text) - numbers(source)
    if invented:
        found.append(f"numbers absent from the source: {sorted(invented)[:5]}")
    # Referring to the document instead of its subject.
    if re.search(r"(?i)\b(?:this (?:document|text|passage|article)|the (?:document|passage)"
                 r"|the author (?:writes|argues|states))\b", text):
        found.append("refers to the document rather than its subject")
    if not text.strip():
        found.append("empty")
    return found


def _bands():
    """The publication bands, from compose.py. Not redeclared here: two copies of
    a band is exactly the parallel hand-edit the contract forbids."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("cmp", HERE / "compose.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m.BANDS


# Below this, the standard bands cost content rather than redundancy.
SHORT_SOURCE = 400


def ceilings(total: int) -> dict:
    """How short each reading may be asked to be, given the source's length.

    HOW MUCH A TEXT CAN BE COMPRESSED DEPENDS ON HOW MUCH REDUNDANCY IT HAS, and
    a short passage has almost none. A standard long-document ceiling can force
    distinct claims out of a short source, so the ceiling gives way instead.

    Long sources are untouched: they have redundancy to spend, and the published
    bands are what a sealed artifact is evaluated against.
    """
    b = _bands()
    if total >= SHORT_SOURCE:
        return {d: b[d][1] for d in b}
    # Interpolate from generous at a paragraph to the band at the threshold, so
    # there is no cliff at 399 vs 400 words.
    t = max(0.0, min(1.0, total / SHORT_SOURCE))
    return {"detailed": 0.70 - t * (0.70 - b["detailed"][1]),
            "brief": 0.38 - t * (0.38 - b["brief"][1])}


def ceiling_words(total: int) -> dict:
    """Feasible absolute ceilings used by both the prompt and publication gate."""
    ratios = ceilings(total)
    minimum = {"detailed": 30, "brief": 15}
    return {depth: min(max(1, total), max(minimum[depth],
                        int(total * ratios[depth])))
            for depth in ratios}


def structural_findings(detailed: str, brief: str, source: str,
                        ceilings=None) -> list[str]:
    """Blocking defects shared by the pair routes: empty readings, the source
    handed back (verbatim sentences or a reworded pass-through at or above
    the ceiling), prohibited list/table markup, and prose presented as an
    exact quotation that is not a contiguous source span. `ceilings` is an
    optional {"detailed": words, "brief": words} map; callers with their own
    publication bands pass it, everyone else gets this route's."""
    if not (detailed or "").strip() or not (brief or "").strip():
        return ["empty"]
    found = [f for f in defects(detailed, source) + defects(brief, source)
             if f == "list or table markup"]
    # A length ceiling is a useful repair signal, but length alone is not
    # evidence that a low-overlap summary is a source copy. Keeping it out of
    # structural eligibility prevents an otherwise usable pair from being
    # destroyed solely for being longer than an editorial target.
    found += (not_a_copy(detailed, source, "detailed", ceilings,
                         include_length=False)
              + not_a_copy(brief, source, "brief", ceilings,
                          include_length=False))
    for depth, text in (("detailed", detailed), ("brief", brief)):
        if _is_trivial_stub(text):
            found.append(f"{depth} reading is a non-summary stub")
    found += custom_instructions.quote_defects((detailed, brief), source)
    return found


def not_a_copy(text: str, source: str, depth: str = "detailed",
               ceilings=None, include_length=True) -> list[str]:
    """The one failure this project exists to prevent: the source handed back.

    Two ways it happens. Verbatim sentences, checked directly -- a lightly
    cleaned copy can sit at any word count. And a "condensation" that is simply
    the source reworded: on a 131-word paragraph one attempt came back at 128
    words, which is a paraphrase, not a summary.

    The length half is a GATE, not a prompt instruction. A model-facing minimum
    invites intensifiers and padding, so length is never requested. It is
    evaluated afterwards and returned as a finding, like every other check.
    """
    found = []
    def sents(x):
        return {" ".join(s.split()).lower()
                for s in re.split(r"(?<=[.!?])\s+", x) if len(s.split()) > 6}
    src, out = sents(source), sents(text)
    if out:
        shared = len(src & out) / len(out)
        if shared > 0.5:
            found.append(f"{shared:.0%} of the {depth} reading's sentences are "
                         f"verbatim from the source")
    n, total = len(text.split()), max(1, len(source.split()))
    hi = (ceilings or ceiling_words(total))[depth]
    if include_length and n > hi:
        found.append(f"the {depth} reading is {n / total:.0%} of the source "
                     f"({n} words of {total}; ceiling {hi}); a reading that replaces the source "
                     f"must be substantially shorter, not reworded")
    return found


def _is_trivial_stub(text: str) -> bool:
    """Reject workflow narration that is not a reading at all.

    Short substantive prose is allowed to proceed to review/repair. This
    deliberately narrow rule catches the observed one-line refusal without
    turning editorial length targets into a no-output gate.
    """
    words = (text or "").split()
    if not words or len(words) > 14:
        return False
    return bool(re.match(
        r"(?is)^(?:reading|summarizing|condensing)\b|"
        r"^(?:i\s+(?:cannot|can't|will|am)|unable\s+to)\b|"
        r"^(?:the\s+)?(?:full\s+)?source\s+(?:is|was|has)\b",
        " ".join(words)))


def run(source_path: pathlib.Path, out_dir: pathlib.Path) -> int:
    # Quick has its own pair controller. The legacy ledger loader remains only
    # as a compatibility symbol for old trace tests; it is not on this path.
    ms = _runner()
    source = source_path.read_text(errors="replace")
    # A LENGTH CEILING IN THE PROMPT, which the project otherwise forbids
    # ("No model-facing length instruction at all"). Scoped to this path and
    # argued, not overlooked:
    #
    # A per-chunk minimum invites padding and intensifier inflation. A ceiling
    # has no padding incentive; its risk is dropping material, and this path
    # already spends a call auditing for exactly that. Without a ceiling, a
    # short source may be paraphrased nearly in full rather than summarized.
    #
    # The numbers come from compose.BANDS, so there is still only one place a
    # publication band is defined. Ceilings, never quotas: the prompt says so.
    total = len(source.split())
    cei = ceiling_words(total)
    base_prompt = (pair_review.write_template()
              .replace("{SOURCE}", source)
              .replace("{EVIDENCE_KIND}", "the complete source")
              .replace("{TASK_RULES}", "")
              .replace("{D_WORDS}", str(cei["detailed"]))
              .replace("{B_WORDS}", str(cei["brief"])))

    prompt = custom_instructions.decorate_prompt(base_prompt)

    def ask(p, stage, chain):
        raw = ms.run(p, out_dir, chain, stage,
                     validate=lambda r: json_contract.parse(
                         r, json_contract.SHORT_RESULT, stage),
                     gateway_options=RESULT_REQUEST_OPTIONS)
        obj = json_contract.parse(raw, json_contract.SHORT_RESULT, stage)
        return (obj.get("detailed") or "").strip(), (obj.get("brief") or "").strip()

    def too_slight(text, depth):
        """A schema-valid answer that is not a summary at all.

        `defects` catches only the empty string, so a model that answers
        "Reading the full source before condensing it." passes every
        deterministic gate and publishes as a seven-word summary of a
        long document. A schema-valid workflow narration must not publish as
        though it were a substantive reading.

        The floor is a fraction of the reading's OWN ceiling, so it scales with
        the source and there is still no minimum source length -- a one-
        paragraph document keeps its short ceiling and a proportionally short
        floor. It only asserts that a reading is not a stub.
        """
        floor = min(cei[depth], max(8, int(cei[depth] * 0.20)))
        n = len(text.split())
        return ([] if n >= floor else
                [f"EXPAND REQUIRED: the {depth} reading is {n} words. "
                 f"Expand it to at least {floor} words using additional "
                 f"source-supported material from this source. Do not pad, "
                 f"repeat, or invent content."])

    def mechanical(d, b):
        return (defects(d, source) + defects(b, source)
                + not_a_copy(d, source, "detailed") + not_a_copy(b, source, "brief")
                + too_slight(d, "detailed") + too_slight(b, "brief")
                + custom_instructions.quote_defects((d, b), source))

    def usable(d, b):
        """A candidate with no blocking structural defect.

        The category list lives in structural_findings, shared with the Full
        direct route; the disclosable remainder is published under the
        liveness rule, never a reason to withhold a usable pair.
        """
        return not structural_findings(d, b, source)

    def publish(d, b, *, status, selected, findings, review, repair):
        """Write the pair plus its machine-readable status. The only writer."""
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "detailed.md").write_text(d.rstrip() + "\n")
        (out_dir / "brief.md").write_text(b.rstrip() + "\n")
        (out_dir / "short-report.json").write_text(json.dumps(
            {"path": "short", "source_words": len(source.split()),
             "detailed_words": len(d.split()),
             "brief_words": len(b.split()),
             "status": status,      # "pass" | "open_findings" | "review_unavailable"
             "selected": selected,  # "initial" | "revised"
             "findings": findings,  # open findings on the published bytes
             "review": review,      # "complete" | "unavailable: ..."
             "repair": repair},     # "none" | "once" | "unavailable"
            indent=2))
        print(f"[short] published {len(d.split())}w detailed / "
              f"{len(b.split())}w brief (status: {status}, review: {review}, "
              f"repair: {repair}, {len(findings)} open finding(s))",
              flush=True)
        for finding in findings:
            print(f"[short] open finding: {finding}", flush=True)
        return 0

    # Candidate retention, safest-candidate selection, and status live in
    # pair_review, shared with the future Full final-pair controller. This
    # route supplies only its structural gate (usable) and publication.
    select = pair_review.select

    def nothing_usable(pool):
        # The one thing that cannot be disclosed away. Nothing to publish is
        # still nothing to publish.
        if any((c["detailed"] or "").strip() and (c["brief"] or "").strip()
               for c in pool):
            print("[short] no structurally usable candidate (empty, "
                  "source-copy, prohibited structure, or fabricated exact "
                  "quotation) — nothing published", file=sys.stderr)
            return 5
        print("[short] the model returned no usable prose", file=sys.stderr)
        return 1

    try:
        detailed, brief = ask(prompt, "short", ms.MODELS)
    except Exception as e:
        # A defensive boundary for injected/legacy transports that return an
        # invalid object despite the route validator.  It represents no usable
        # candidate; it is not a semantic review finding and must not surface
        # as an unhandled traceback from the child process.
        print(f"[short] no usable initial pair ({str(e)[:120]})",
              file=sys.stderr, flush=True)
        return 1
    findings = mechanical(detailed, brief)
    mech0 = list(findings)

    def audit(d, b, stage):
        """Model-judged findings: omission and epistemic reversal, the two things
        no deterministic check can see."""
        import pair_patch
        audit_prompt = pair_review.fill_audit_prompt(
            pair_review.audit_template(),
            scope="The supplied source is complete for this review.",
            source=source, detailed=d, brief=b)
        raw = ms.run(custom_instructions.decorate_prompt(audit_prompt),
                     out_dir, ms.AUDIT_MODELS, stage,
                     validate=lambda r: json_contract.parse(
                         r, json_contract.SHORT_AUDIT, stage),
                     gateway_options=AUDIT_REQUEST_OPTIONS)
        a = json_contract.parse(raw, json_contract.SHORT_AUDIT, stage)
        records = pair_patch.assign_finding_ids(a.get("findings") or [])
        out = pair_patch.render_findings(records)
        if a.get("verdict") == "revise" and not out:
            out = ["the audit asked for a revision without naming a defect"]
            records = []
        print(f"[short] {stage}: {a.get('verdict', '?')}, {len(out)} finding(s)",
              flush=True)
        return out, records

    # THE AUDIT IS WHAT MAKES THIS WORTH USING. The deterministic checks catch
    # invented numbers, list markup, first person and the source handed back --
    # none of which is what makes a summary insufficient. What does is material
    # dropped, and a hedged claim rendered as a confident one. Only a reader can
    # see those, so one call is spent looking. It uses the AUDIT chain, so the
    # per-role choice applies here exactly as it does on the ledger path.
    try:
        audit0, records0 = audit(detailed, brief, "short-audit")
        findings += audit0
        review = "complete"
    except Exception as e:
        # A reviewer outage does not erase a usable pair. If the pair is
        # structurally unusable, however, known mechanical findings still get
        # their one correction opportunity before the controller gives up.
        audit0, records0 = [], []
        review = f"unavailable: short-audit: {str(e)[:80]}"

    initial = pair_review.new_candidate(
        "initial", detailed, brief, mech0, audit0,
        usable=usable(detailed, brief), review=review,
        audit_records=records0)

    if not findings:
        chosen = select([initial])
        if chosen is None:
            return nothing_usable([initial])
        return publish(chosen["detailed"], chosen["brief"],
                       status=pair_review.resolve_status(chosen),
                       selected=chosen["selected"],
                       findings=pair_review.disclosed(chosen),
                       review=review, repair="none")

    if not initial["usable"] and not mech0:
        return nothing_usable([initial])

    print(f"[short] revising once for {len(findings)} finding(s)", flush=True)
    try:
        # One correction, from the producer that wrote the candidate.
        import mapsum
        import pair_patch
        chain = mapsum.last_ok_route(out_dir, "short", ms.MODELS)
        turn = pair_review.repair_turn(initial, base_prompt)
        fix = custom_instructions.decorate_prompt(turn["prompt"])
        if turn["kind"] == "length":
            d2, b2 = ask(fix, "short-revise", chain)
        else:
            raw = ms.run(fix, out_dir, chain, "short-revise",
                         validate=lambda r: json_contract.parse(
                             r, json_contract.PAIR_PATCH, "short-revise"),
                         gateway_options=PATCH_REQUEST_OPTIONS)
            obj = json_contract.parse(raw, json_contract.PAIR_PATCH,
                                      "short-revise")
            d2, b2 = pair_patch.apply_edits(
                detailed, brief, obj,
                allowed_finding_ids=turn["allowed_finding_ids"] or None)
    except Exception as e:
        # The repair itself failed. The audited initial bytes are retained,
        # so they are published with their known findings, not discarded.
        print(f"[short] revision unavailable ({str(e)[:80]}) — publishing "
              "retained candidate", file=sys.stderr, flush=True)
        chosen = select([initial])
        if chosen is None:
            return nothing_usable([initial])
        return publish(chosen["detailed"], chosen["brief"],
                       status=pair_review.resolve_status(chosen),
                       selected=chosen["selected"],
                       findings=pair_review.disclosed(chosen), review=review,
                       repair="unavailable")
    # AUDIT THE REVISION. Not doing so shipped a real reversal: asked to
    # compress, a revision turned "not NECESSARILY inflationary" into
    # "not inflationary" -- a flat denial the source does not make -- and
    # nothing was looking. The ledger path already knows this and audits
    # the exact revision it produced; so does this one.
    #
    # This is NOT a convergence loop. There is exactly one revision, the
    # new bytes are judged on their own, and whatever survives is
    # disclosed rather than retried. A later audit's silence never closes
    # an earlier finding, because the earlier bytes are gone.
    mech2 = mechanical(d2, b2)
    try:
        audit2, records2 = audit(d2, b2, "short-reaudit")
        review2 = "complete"
    except Exception as e:
        audit2, records2 = [], []
        review2 = f"unavailable: short-reaudit: {str(e)[:80]}"
    revised = pair_review.new_candidate(
        "revised", d2, b2, mech2, audit2,
        usable=usable(d2, b2), review=review2,
        notes=[] if review2 == "complete" else [review2],
        audit_records=records2)
    # A candidate whose exact bytes were never re-audited is published, when
    # selected, as review_unavailable: the re-audit outage must not discard a
    # usable pair, and must not be silent either. Scoring uses known findings
    # only; the outage note rides along for disclosure.
    pool = [initial, revised]
    chosen = select(pool)
    if chosen is None:
        return nothing_usable(pool)
    # Initial bytes carry their own completed review.
    return publish(chosen["detailed"], chosen["brief"],
                   status=pair_review.resolve_status(chosen),
                   selected=chosen["selected"],
                   findings=pair_review.disclosed(chosen), review=review2
                   if chosen["selected"] == "revised" else review,
                   repair="once")


def main():
    if len(sys.argv) != 3:
        print(__doc__.splitlines()[2].strip(), file=sys.stderr)
        return 2
    return run(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]))


if __name__ == "__main__":
    sys.exit(main())
