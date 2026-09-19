#!/usr/bin/env python3
"""Shared candidate/review/repair state for the Detailed/Brief pair routes.

Quick owns the production behavior; the Full final-pair controller reuses
this machinery instead of growing a divergent engine. Pure state only: no
model calls, no I/O, no harness or family branches. Callers supply transport
(the model call), the route's structural gate, and publication.

A candidate is a retained pair plus the findings judged against its exact
bytes. Selection ranks completed review above incomplete review, then material
findings (audit omissions/reversals and invented numbers), then totals, then
pool order -- later entries win ties, so one routine serves Quick's single
repair and Full's up to three mutations without a fork.
"""
from __future__ import annotations
import hashlib
import pathlib
import re

PROMPTS = pathlib.Path(__file__).parent / "prompts"


def write_template() -> str:
    """The one pair writer prompt shared by Quick, Full and Corpus. Slots:
    {EVIDENCE_KIND}, {TASK_RULES}, {D_WORDS}, {B_WORDS}, {SOURCE}. Callers
    other than Corpus pass an empty {TASK_RULES}."""
    return ((PROMPTS / "pair-write.txt").read_text()
            .replace("{READING_POLICY}", reading_policy()))


def reading_policy() -> str:
    """The shared prose and fidelity policy for pair and batched writers."""
    return (PROMPTS / "reading-policy.txt").read_text().strip()


def reading_part_template() -> str:
    """The single-artifact response used when one logical pair cannot fit."""
    return ((PROMPTS / "reading-part.txt").read_text()
            .replace("{READING_POLICY}", reading_policy()))


def audit_template(version="v1") -> str:
    """The one pair audit prompt shared by Quick and Full. Slots: {SCOPE},
    {SOURCE}, {DETAILED}, {BRIEF}, {SEGMENTS}, and in v2
    {PLAN_CONTEXT}/{READABILITY_CONTEXT}."""
    name = "pair-audit-v2.txt" if version == "v2" else "pair-audit.txt"
    return (PROMPTS / name).read_text()


def fill_audit_prompt(template: str, *, scope: str, source: str,
                      detailed: str, brief: str, plan_context="",
                      readability_context="[]", segments=None) -> str:
    import pair_patch
    return (template.replace("{SCOPE}", scope)
            .replace("{SOURCE}", source)
            .replace("{DETAILED}", detailed)
            .replace("{BRIEF}", brief)
            .replace("{SEGMENTS}", (segments if segments is not None else
                                     pair_patch.segment_map(detailed, brief)))
            .replace("{PLAN_CONTEXT}", plan_context or "(not supplied)")
            .replace("{READABILITY_CONTEXT}",
                     readability_context or "[]"))


def candidate_identity(detailed: str, brief: str) -> str:
    """Short stable identity for one candidate pair, echoed in the repair
    prompt so a repair is tied to the exact bytes it must revise."""
    return "candidate-" + hashlib.sha256(
        ((detailed or "") + "\n\x00\n" + (brief or "")).encode("utf-8")
    ).hexdigest()[:16]


def new_candidate(selected, detailed, brief, mechanical, audit, *,
                  usable, review="complete", notes=(), source_context=None,
                  audit_records=None):
    """One retained candidate. `usable` is the caller's structural verdict;
    `review` names the fresh review of these exact bytes ("complete" or
    "unavailable: ..."); `notes` rides along for disclosure without scoring.
    `audit` is the rendered sentence list used for scoring and disclosure.
    `audit_records` are the structured findings used for patch repair.
    """
    mech, aud = list(mechanical or []), list(audit or [])
    return {"selected": selected, "detailed": detailed, "brief": brief,
            "mechanical": mech, "audit": aud, "findings": mech + aud,
            "audit_records": list(audit_records or []),
            "usable": bool(usable), "review": review, "notes": list(notes),
            "source_context": dict(source_context or {})}


def _invented(finding: str) -> bool:
    """The disclosable mechanical finding about WHAT is said: a number with
    no source witness. Audit findings are material by construction; the rest
    is minor. Structural defects never reach scoring."""
    return "absent from the source" in (finding or "")


def select(pool, *, prefer_earlier=False):
    """Safest usable candidate, or None.

    A candidate whose review did not complete has unknown semantic quality. It
    may be the only candidate and therefore may publish under the liveness
    rule, but it must not displace a usable candidate whose exact bytes were
    actually reviewed merely because the unreviewed child has fewer recorded
    findings.

    Counting findings alone kept a revision that traded three minor defects
    for two dropped claims. That is the wrong trade, so material outranks
    minor: a revision that drops claims loses to a retained candidate that
    merely reads roughly, and a revision that invents numbers loses to one
    that merely omits.
    """
    usable_pool = [(i, c) for i, c in enumerate(pool or [])
                   if isinstance(c, dict) and c.get("usable")]
    if not usable_pool:
        return None

    def key(item):
        i, c = item
        review_incomplete = 0 if c.get("review") == "complete" else 1
        mat = len(c.get("audit") or []) + sum(
            1 for f in (c.get("mechanical") or []) if _invented(f))
        tie = i if prefer_earlier else -i
        return (review_incomplete, mat, len(c.get("findings") or []), tie)

    return min(usable_pool, key=key)[1]


def resolve_status(cand) -> str:
    """Publication status for one selected candidate: "pass",
    "open_findings", or "review_unavailable"."""
    if cand.get("review") != "complete":
        return "review_unavailable"
    if cand.get("findings"):
        return "open_findings"
    return "pass"


def disclosed(cand) -> list:
    """Everything stated about the published bytes: judged findings plus
    disclosure-only notes (such as a review outage)."""
    return list(cand.get("findings") or []) + list(cand.get("notes") or [])


# The mechanical over-ceiling finding, as `shortsum.not_a_copy` words it.
_OVER_CEILING = re.compile(
    r"^the (?:detailed|brief) reading is \d+% of the source\b", re.I)


def length_only(findings) -> bool:
    """Whether every finding is the over-ceiling one, so the candidate can be
    compressed without reading the source again.

    Re-attaching the whole source invites regeneration rather than compression.
    A shortening needs the candidate, not the source.
    An EXPAND finding is the opposite and keeps the source."""
    items = [str(f).strip() for f in (findings or [])]
    return bool(items) and all(_OVER_CEILING.match(f) for f in items)


def build_repair_prompt(base_prompt: str, detailed: str, brief: str,
                        findings, source_context=None) -> str:
    """A repair revises THE ENCLOSED CANDIDATE, never a fresh pair from the
    base prompt: identity, exact current readings, and findings travel
    together, so a finding about one reading need not rewrite the other.
    The source stays in the prompt via base_prompt, except for a pure
    shortening, which is candidate-led and omits it."""
    ident = candidate_identity(detailed, brief)
    if length_only(findings):
        return ((PROMPTS / "pair-repair-length.txt").read_text()
                .replace("{IDENT}", ident)
                .replace("{D_WORDS}", str(len(detailed.split())))
                .replace("{B_WORDS}", str(len(brief.split())))
                .replace("{DETAILED}", detailed)
                .replace("{BRIEF}", brief)
                .replace("{FINDINGS}", "\n- ".join(str(f) for f in findings)))
    context = source_context or {}
    wanted = set(re.findall(r"\[source-packet:([^\]]+)\]", "\n".join(
        str(f) for f in findings)))
    if wanted:
        context = {key: value for key, value in context.items()
                   if key in wanted}
    else:
        # Mechanical findings already have their source in the original
        # prompt. Do not attach every raw window to a repair that has no
        # source-local semantic finding.
        context = {}
    source_text = "".join(
        f"\n\nIMPLICATED {'COMPACT REVIEW EVIDENCE' if packet_id.endswith('-global') else 'RAW SOURCE PACKET'} "
        f"{packet_id}:\n{packet}"
        for packet_id, packet in context.items())
    return base_prompt + ((PROMPTS / "pair-repair.txt").read_text()
                          .replace("{IDENT}", ident)
                          .replace("{DETAILED}", detailed)
                          .replace("{BRIEF}", brief)
                          .replace("{SOURCE_CONTEXT}", source_text)
                          .replace("{FINDINGS}", "\n- ".join(findings)))


def build_patch_prompt(detailed: str, brief: str, findings,
                       source_context=None, *, bounded=False) -> str:
    """Same-producer correction that returns edits, not a replacement pair."""
    import pair_patch
    ident = candidate_identity(detailed, brief)
    context = source_context or {}
    wanted = set()
    unscoped_source_needed = False
    for item in findings or []:
        if isinstance(item, dict) and item.get("packet"):
            wanted.add(str(item["packet"]))
        blob = str(item)
        if isinstance(item, dict):
            blob += " " + str(item.get("text") or "")
            if (not item.get("packet") and
                    not length_only([pair_patch.finding_text(item)])):
                unscoped_source_needed = True
        wanted.update(re.findall(r"\[source-packet:([^\]]+)\]", blob))
    if wanted and not unscoped_source_needed:
        context = {key: value for key, value in context.items()
                   if key in wanted}
    elif not wanted and not unscoped_source_needed:
        context = {}
    source_text = "".join(
        f"\n\nIMPLICATED {'COMPACT REVIEW EVIDENCE' if packet_id.endswith('-global') else 'RAW SOURCE PACKET'} "
        f"{packet_id}:\n{packet}"
        for packet_id, packet in context.items())
    lines = []
    for item in findings or []:
        if isinstance(item, dict):
            fid = item.get("finding_id") or ""
            kind = item.get("kind") or ""
            artifact = item.get("artifact") or ""
            anchor = item.get("anchor") or ""
            slot = item.get("slot") or ""
            text = pair_patch.finding_text(item)
            lines.append(
                f"{fid} {kind} {artifact} anchor={anchor or '-'} "
                f"slot={slot or '-'} {text}".strip())
        else:
            lines.append(str(item))
    segment_text = pair_patch.segment_map(detailed, brief)
    detailed_text, brief_text = detailed, brief
    scoped_words = len(detailed.split()) + len(brief.split())
    if bounded:
        references = []
        for item in findings or []:
            if isinstance(item, dict):
                references.extend((item.get("anchor"), item.get("slot")))
        scoped = pair_patch.scoped_segment_context(
            detailed, brief, references)
        detailed_text = scoped["detailed"] or "(not implicated)"
        brief_text = scoped["brief"] or "(not implicated)"
        segment_text = scoped["segments"]
        scoped_words = int(scoped["word_count"])
    prompt = ((PROMPTS / "pair-repair-patch.txt").read_text()
            .replace("{IDENT}", ident)
            .replace("{DETAILED}", detailed_text)
            .replace("{BRIEF}", brief_text)
            .replace("{SEGMENTS}", segment_text)
            .replace("{SOURCE_CONTEXT}", source_text)
            .replace("{FINDINGS}", "\n- ".join(lines)))
    return (prompt, scoped_words) if bounded else prompt


def repair_turn(current, base_prompt="", *, force_patch=False,
                bounded=False):
    """One same-producer correction: length rewrite or patch edits.

    Full and Quick must use this so the kind cannot drift. Callers still
    own transport.
    """
    detailed = current["detailed"]
    brief = current["brief"]
    findings = current.get("findings") or []
    if length_only(findings) and not force_patch:
        return {
            "kind": "length",
            "prompt": build_repair_prompt(
                base_prompt, detailed, brief, findings,
                current.get("source_context")),
            "allowed_finding_ids": (),
        }
    import pair_patch
    payload = list(current.get("audit_records") or [])
    # Patch mode is selected whenever findings are mixed. Every mechanical
    # finding must therefore travel in the same typed request; filtering the
    # length item here caused the live Muse correction to receive only its
    # unsupported-number finding and made the apparent compression attempt a
    # fiction. Pure length remains the smaller whole-pair shortening path
    # above, while mixed length is an ordinary closed finding in this patch.
    mechanical = list(current.get("mechanical") or [])
    # Audit findings already carry controller-issued F-nnn identifiers.
    # Mechanical findings used to be appended as anonymous prose, forcing a
    # patch model either to invent an id or omit the requested fix. Give them
    # the same closed identifier namespace before composing the prompt.
    payload.extend(pair_patch.assign_finding_ids(
        mechanical, start=len(payload) + 1))
    if not payload:
        payload = pair_patch.assign_finding_ids(findings)
    if bounded and any(
            isinstance(item, dict)
            and not (item.get("anchor") or item.get("slot"))
            and not length_only([pair_patch.finding_text(item)])
            for item in payload):
        raise ValueError(
            "bounded repair cannot safely scope an unanchored finding")
    allowed = tuple(item.get("finding_id") for item in payload
                    if isinstance(item, dict) and item.get("finding_id"))
    patch_prompt = build_patch_prompt(
        detailed, brief, payload, current.get("source_context"),
        bounded=bounded)
    if bounded:
        patch_prompt, scoped_words = patch_prompt
        planned_words = max(200, scoped_words)
    else:
        planned_words = None
    return {
        "kind": "patch",
        "prompt": patch_prompt,
        "allowed_finding_ids": allowed,
        "planned_output_words": planned_words,
    }
