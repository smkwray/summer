#!/usr/bin/env python3
"""Plan, verify per section part, and SEAL a source-grounded summary ledger.

The previous version planned the whole document in one call whenever it fitted
in context, and produced 13 units for an 8,365-word paper -- an inventory that
capped the artifact at 830 words before a single sentence was composed. Context
capacity is a transport ceiling, not a planning granularity: one pass can hold a
paper without being able to enumerate it.

So planning is always section-local, and a controller-side density guard rejects
a section that comes back too thin. The guard is never shown to the planner --
telling a model "produce 42 units" buys micro-splitting, not coverage.

Sealing is fail-closed. The old code broke out of its audit loop and wrote SEALED
regardless, and read audits with a tolerant parser so `{}` looked like "no
findings". Both are gone: unresolved findings mean no seal, and no seal means
nothing is published.

usage: ledger.py READERVIEW_DIR OUT_DIR
"""
from __future__ import annotations
import hashlib, json, math, os, pathlib, re, subprocess, sys

HERE = pathlib.Path(__file__).parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import custom_instructions
import json_contract
import ledger_schema
# Planning window. Unit density is governed primarily by the source window,
# rather than by naming a capsule length in the prompt. Narrower windows raise
# call cost substantially. Change this only when a coverage corpus shows that
# the current window omits propositions a narrower one preserves.
PART_MIN, PART_MAX = 700, 1200          # planning window, in content words
NARROW = {"apparatus", "exact_repetition"}   # only these shrink C_s

# The audit prompt itself lists exact restatement and incidental examples as
# "not blocking by itself". Treating every finding as blocking meant one
# redundancy finding could prevent sealing forever, and an unsealed ledger
# publishes nothing -- turning a cosmetic note into total failure. Fidelity
# types still block absolutely.
NON_BLOCKING = {"duplicate", "micro_split"}
PART_VERIFICATION_VERSION = 2
PLAN_REQUEST_OPTIONS = json_contract.options("summer_ledger_plan", json_contract.LEDGER_PLAN)
REVISION_REQUEST_OPTIONS = json_contract.options(
    "summer_ledger_revision", json_contract.LEDGER_REVISION)
AUDIT_REQUEST_OPTIONS = json_contract.options(
    "summer_ledger_audit", json_contract.LEDGER_AUDIT)


def _progress():
    import importlib.util
    spec = importlib.util.spec_from_file_location("pg", HERE / "progress.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def _runner():
    import importlib.util
    spec = importlib.util.spec_from_file_location("ms", HERE / "mapsum.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def inline(path):
    """Payloads travel IN the prompt; the model never fetches anything.

    Handing a model a path costs a tool turn whose result is then re-sent on
    every later step: on one document that turned ~672k tokens of real payload
    into 4.2M. It also requires file tools, which an audit has no business
    holding. One request, all the bytes, one answer.
    """
    return pathlib.Path(path).read_text(errors="replace")


def _json_ok(raw, stage="response"):
    """Validator for ms.run(): require the exact stage contract locally."""
    schema = json_contract.schema_for_stage(stage)
    if schema is not None:
        return json_contract.parse(raw, schema, stage)
    ms = _runner()
    obj = ms.parse_audit(raw)
    if not isinstance(obj, dict) or not obj:
        raise ValueError("empty or non-object response")
    return obj


def parse_strict(raw, stage):
    """Fail-closed JSON using the authoritative local stage contract."""
    return _json_ok(raw, stage)


def split_sections(blocks, part_max=PART_MAX):
    """Semantic sections, then bounded parts. Headings start sections."""
    if isinstance(part_max, bool) or not isinstance(part_max, int) or part_max < 1:
        raise ValueError("part_max must be a positive integer")
    secs, cur = [], []
    for b in blocks:
        if re.match(r"^#{1,3}\s", b["text"]) and cur:
            secs.append(cur); cur = []
        cur.append(b)
    if cur:
        secs.append(cur)

    # PACK consecutive sections up to PART_MAX before splitting anything.
    # Otherwise a heading-dense document can create one tiny part per heading,
    # each buying its own plan, audit, and revision round. Packing is per-group;
    # a section larger than PART_MAX still splits exactly as before.
    groups, g, gw = [], [], 0
    for si, sec in enumerate(secs, 1):
        w = sum(b["words"] for b in sec)
        if g and gw + w > part_max:
            groups.append(g); g, gw = [], 0
        g.append((si, sec)); gw += w
    if g:
        groups.append(g)

    out = []
    for group in groups:
        first_si = group[0][0]
        blocks_in = [b for _, sec in group for b in sec]
        title = re.sub(r"^#+\s*", "", group[0][1][0]["text"].split("\n")[0])[:80]
        if len(group) > 1:
            # The model is told the part spans several headings, so it does not
            # read a group as one topic that lost its subheadings.
            title = f"{title} (+{len(group) - 1} more)"[:80]
        sid = f"SEC{first_si:02d}"
        total = sum(b["words"] for b in blocks_in)
        if total <= part_max:
            out.append({"section_id": sid, "title": title,
                        "part": "1 of 1", "blocks": blocks_in}); continue
        part, n, pi = [], 0, 0
        nparts = max(1, math.ceil(total / part_max))
        for b in blocks_in:
            if n and n + b["words"] > part_max:
                pi += 1
                out.append({"section_id": sid, "title": title,
                            "part": f"{pi} of {nparts}", "blocks": part})
                part, n = [], 0
            part.append(b); n += b["words"]
        if part:
            pi += 1
            out.append({"section_id": sid, "title": title,
                        "part": f"{pi} of {nparts}", "blocks": part})
    return out


def density(c_words):
    return (max(1, math.ceil(c_words / 220)), max(1, round(c_words / 170)))


def plan_part(ms, part, outline, out_dir, idx):
    # Cache each part's plan. A 30-call pipeline that loses everything to one
    # interruption is fragile in production and untestable here, where long runs
    # get killed. Re-running now resumes instead of restarting.
    cache = out_dir / f"plan{idx:03d}.json"
    if cache.exists():
        try:
            obj = json.loads(cache.read_text())
            # A resumed work root may be invoked with a different one-off
            # request. Do not let a plan produced under another request bypass
            # this run's planner, audit, and repair obligations.
            cached_digest = obj.get("_custom_instruction_digest")
            current_digest = custom_instructions.current_digest()
            if (obj.get("units") is not None and cached_digest == current_digest
                    and obj.get("_verification_version") == PART_VERIFICATION_VERSION
                    and obj.get("_verified") is True):
                print(f"  part {idx}: cached", flush=True)
                return obj
        except Exception:
            pass
    view = out_dir / f"part{idx:03d}.md"
    view.write_text("\n\n".join(f"[{b['id']}] {b['text']}" for b in part["blocks"]))
    p = (HERE / "prompts" / "ledger-build.txt").read_text()
    for k, v in {"{SOURCE_VIEW}": inline(view), "{SECTION_ID}": part["section_id"],
                 "{SECTION_TITLE}": part["title"], "{SECTION_PART}": part["part"],
                 "{DOCUMENT_OUTLINE}": outline}.items():
        p = p.replace(k, v)
    p = custom_instructions.decorate_prompt(p)
    obj = parse_strict(ms.run(
        p, out_dir, ms.PLAN_MODELS, f"plan{idx:03d}",
        validate=lambda r: _json_ok(r, "plan"),
        gateway_options=PLAN_REQUEST_OPTIONS), "plan")
    obj["_custom_instruction_digest"] = custom_instructions.current_digest()

    # VERIFY THIS PART NOW, while its source window is small and local:
    # deterministic checks first (free), then one audit, then at most one full
    # replacement, then one audit of the exact replacement. No convergence loop:
    # a later audit's silence must never be able to close an earlier finding.
    obj = verify_part(ms, part, out_dir, idx, obj)
    obj["_custom_instruction_digest"] = custom_instructions.current_digest()
    cache.write_text(json.dumps(obj, indent=2))
    return obj


def verify_part(ms, part, out_dir, idx, obj, stage_prefix=""):
    """Return only a candidate whose exact bytes completed bounded verification.

    An unavailable audit, unavailable repair, defective repair, or surviving
    blocking finding is a failed part.  It is never converted into a warning or
    a quarantined passage that composition may publish later.
    """
    det = part_defects(part, obj)
    try:
        res = audit_part(ms, out_dir, idx, part, obj, round_no=1,
                         stage_prefix=stage_prefix)
        findings = [f for f in (res.get("findings") or [])
                    if f.get("type") not in NON_BLOCKING]
    except Exception as e:
        raise RuntimeError(f"semantic audit unavailable: {str(e)[:120]}") from e
    if det:
        findings = [{"type": "deterministic", "detail": d} for d in det] + findings

    if findings:
        print(f"  part {idx}: {len(findings)} finding(s) — revising the whole part",
              flush=True)
        try:
            rev = revise_part(ms, out_dir, idx, part, obj, findings,
                              stage_prefix=stage_prefix)
            rev_det = part_defects(part, rev)
            if rev_det:
                raise RuntimeError(
                    f"revision has {len(rev_det)} deterministic defect(s): "
                    + "; ".join(rev_det[:3]))
            res2 = audit_part(ms, out_dir, idx, part, rev, round_no=2,
                              stage_prefix=stage_prefix)
            left = [f for f in (res2.get("findings") or [])
                    if f.get("type") not in NON_BLOCKING]
            if left:
                raise RuntimeError(
                    f"{len(left)} semantic finding(s) survive the one revision")
            obj = rev
        except Exception as e:
            raise RuntimeError(f"part verification failed: {str(e)[:160]}") from e

    obj.pop("audit_unavailable", None)
    obj.pop("quarantine_local_ids", None)
    obj.pop("brief_suspect_local_ids", None)
    obj["_verification_version"] = PART_VERIFICATION_VERSION
    obj["_verified"] = True
    return obj


# A capsule describes the DOCUMENT, never the planning window it happened to be
# written from. A planner given one 1,000-word part wrote "although the supplied
# text ends before that definition is completed" -- and that sentence shipped, so
# a reader was told the source is incomplete when it is not. The planner cannot
# see this: from inside the window the statement is true. Code can.
WINDOW_TALK = re.compile(
    # "given" is deliberately absent: "given text-based evidence" is ordinary
    # prose, and a gate that withholds real content is worse than the defect.
    r"\b(?:the\s+)?(?:supplied|provided|excerpted|attached)\s+"
    r"(?:text|passage|excerpt|source|section|material|portion)\b"
    r"|\bthis\s+(?:excerpt|passage|window|chunk)\b"
    r"|\b(?:text|passage|excerpt|source)\s+(?:ends|stops|is\s+truncated|cuts\s+off)\b"
    r"|\bbeyond\s+(?:the\s+)?(?:supplied|provided)\b"
    r"|\bnot\s+included\s+in\s+(?:the\s+)?(?:supplied|provided|this)\b", re.I)


# Lists and tables are read on an e-ink device and aloud by a TTS voice, where a
# bullet stack reads as fragments and a table is unreadable. Capsules are prose.
LIST_MARKUP = re.compile(
    r"^[ \t]*(?:[-*+\u2022\u2013]\s+|\d+[.)]\s+|#{1,6}\s+)"     # bullets, numbers, headings
    r"|^[ \t]*\|.*\|[ \t]*$"                                    # table rows
    r"|^[ \t]*\|?[-: ]{6,}\|",                                   # table rules
    re.M)


def list_markup(units):
    """{(local_id, depth): n} for capsules containing list or table markup."""
    hits = {}
    for x in units:
        for depth in ("detailed", "brief"):
            n = len(LIST_MARKUP.findall(x.get(f"{depth}_capsule") or ""))
            if n:
                hits[(x.get("local_id") or x.get("unit_id"), depth)] = n
    return hits


def window_talk(units):
    """{(local_id, depth): [phrases]} for capsules that describe their own window."""
    hits = {}
    for u in units:
        for depth in ("detailed", "brief"):
            found = WINDOW_TALK.findall(u.get(f"{depth}_capsule") or "")
            if found:
                hits[(u.get("local_id") or u.get("unit_id"), depth)] = found
    return hits


def part_source(part):
    return "\n\n".join(f"[{b['id']}] {b['text']}" for b in part["blocks"])


def part_defects(part, obj):
    """Deterministic checks on ONE planned part, before any model audit.

    Cheap, free, and identical between runs -- so they run first and the audit
    never spends tokens on a defect arithmetic can find.
    """
    bad = []
    ids = {b["id"] for b in part["blocks"]}
    units = obj.get("units") or []
    if not units:
        bad.append("no units planned for this part")
    rep = {s for u in units for s in (u.get("source_ids") or [])}
    dis = {s for d in (obj.get("dispositions") or []) for s in (d.get("source_ids") or [])}
    missing = ids - rep - dis
    if missing:
        bad.append(f"unaccounted source blocks: {sorted(missing)[:6]}")
    unknown = (rep | dis) - ids
    if unknown:
        bad.append(f"references blocks outside this part: {sorted(unknown)[:6]}")
    for u in units:
        for depth in ("detailed", "brief"):
            if u.get(f"{depth}_disposition") in ("required", "optional") \
               and not (u.get(f"{depth}_capsule") or "").strip():
                bad.append(f"{u.get('local_id')}: empty {depth} capsule")
    for d in (obj.get("dispositions") or []):
        if canon_disposition(d.get("disposition")) not in ALLOWED_DISP:
            bad.append(f"invalid disposition {d.get('disposition')!r}")

    for (lid, depth), n in sorted(list_markup(units).items()):
        bad.append(f"{lid}: {depth} capsule uses list or table markup ({n}); "
                   f"capsules must be continuous prose")

    for (lid, depth), _ in sorted(window_talk(units).items()):
        bad.append(f"{lid}: {depth} capsule describes its own planning window "
                   f"instead of the document")

    # Numeric grounding: a figure that is not in this part's source is invented.
    src = part_source(part)

    def nums(t):
        # Normalise before comparing. Capturing trailing punctuation made every
        # year in a list ("2008, 2009") read as the distinct token "2008," and
        # flagged it as invented; thousands separators must not matter either.
        out = set()
        for m in re.findall(r"\d[\d,]*(?:\.\d+)?%?", t or ""):
            m = m.rstrip(".,;:%").replace(",", "")
            if len(m) > 1:                 # single digits are usually prose
                out.add(m)
        return out

    have = nums(src)
    for u in units:
        for depth in ("detailed", "brief"):
            for n in sorted(nums(u.get(f"{depth}_capsule")) - have):
                bad.append(f"{u.get('local_id')}: {depth} capsule states {n}, "
                           f"absent from its source")
    bad.extend(custom_instructions.quote_defects(
        [u.get("detailed_capsule") for u in units]
        + [u.get("brief_capsule") for u in units], src))
    return bad


def audit_part(ms, out_dir, idx, part, obj, round_no=1, stage_prefix=""):
    """Audit one part against its own source window.

    Whole-ledger audits re-read a 63k-token ledger five times and produced
    non-monotonic findings: round four still reported blockers, and a later
    round's silence could bury an earlier unresolved one. A part is small,
    self-contained, and audited while its semantic relationships are local.
    """
    p = (HERE / "prompts" / "ledger-audit.txt").read_text() \
        .replace("{SOURCE_VIEW}", part_source(part)) \
        .replace("{LEDGER}", json.dumps(obj, indent=2)) \
        .replace("{SCOPE}", f"one section part ({idx})")
    p = custom_instructions.decorate_prompt(p)
    stage = f"{stage_prefix}audit{idx:03d}"
    res = parse_strict(ms.run(
        p, out_dir, ms.AUDIT_MODELS, stage,
        validate=lambda r: _json_ok(r, "audit"),
        gateway_options=AUDIT_REQUEST_OPTIONS), "audit")
    # Keep EVERY round. The per-stage stdout file is overwritten, so the
    # pre-revision findings vanished and a defect that survived into the artifact
    # could not be traced to "never flagged" versus "flagged and not fixed".
    (out_dir / f"{stage}-r{round_no}.json").write_text(json.dumps(res, indent=2))
    return res


def revise_part(ms, out_dir, idx, part, obj, findings, stage_prefix=""):
    """One full-part replacement resolving every finding at once."""
    for n, f in enumerate(findings, 1):
        f.setdefault("finding_id", f"F{n}")
    p = (HERE / "prompts" / "ledger-revise.txt").read_text() \
        .replace("{SOURCE_VIEW}", part_source(part)) \
        .replace("{CANDIDATE}", json.dumps(obj, indent=2)) \
        .replace("{FINDINGS}", json.dumps(findings, indent=2))
    p = custom_instructions.decorate_prompt(p)
    return parse_strict(ms.run(
        p, out_dir, ms.REPAIR_MODELS, f"{stage_prefix}revise{idx:03d}",
        validate=lambda r: _json_ok(r, "revise"),
        gateway_options=REVISION_REQUEST_OPTIONS), "revise")


def build(rv_dir, out_dir):
    ms = _runner()
    smap = json.loads((rv_dir / "source-map.json").read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = split_sections(smap["blocks"])
    outline = " | ".join(dict.fromkeys(f"{p['section_id']} {p['title']}" for p in parts))
    print(f"[ledger] {smap['visible_words']}w -> {len(parts)} section part(s)", flush=True)

    units, disp, thin = [], [], []
    unplanned = []         # parts that never produced a plan at all
    localmap = {}          # planner-local id -> global id, across ALL parts
    pg = _progress()
    for i, part in enumerate(parts, 1):
        pg.emit("part", index=i, total=len(parts))
        cw = sum(b["words"] for b in part["blocks"])
        floor, ref = density(cw)
        try:
            obj = plan_part(ms, part, outline, out_dir, i)
        except Exception as e:
            why = str(e)[:200]
            print(f"  part {i}: planning failed ({why[:70]})", flush=True)
            # EMIT IT. A part that never plans is why the seal later refuses,
            # and the seal can only report the consequence ("ledger has no
            # units"). Printing the cause to stdout left the UI -- which reads
            # this event stream, not stdout -- showing a failure with no reason.
            pg.emit("part_failed", index=i, total=len(parts),
                    section=part["section_id"], reason=why)
            unplanned.append(f"{part['section_id']} part {i}"); continue
        got = obj.get("units") or []
        # Remap planner-local ids (SEC01-U01) to the global ids we are about to
        # assign, BEFORE dropping local_id. Without this, dependencies and
        # disposition back-references dangle -- 34 of them in the shipped ledger.
        for k, u in enumerate(got):
            if u.get("local_id"):
                localmap[u["local_id"]] = f"U{len(units)+k+1:03d}"
        # The density guard exists to be acted on. A section that comes back below
        # its floor gets ONE re-plan naming the spans that went unrepresented --
        # not a "produce N units" order, which buys micro-splitting rather than
        # coverage.
        replanned = out_dir / f"plan{i:03d}.replanned"
        if len(got) < floor and not replanned.exists():
            replanned.write_text("1")     # attempt once, even if it does not help
            covered = {b for u in got for b in (u.get("source_ids") or [])}
            missed = [b["id"] for b in part["blocks"] if b["id"] not in covered]
            if missed:
                try:
                    p2 = (HERE / "prompts" / "ledger-build.txt").read_text()
                    for k, v in {"{SOURCE_VIEW}": inline(out_dir / f"part{i:03d}.md"),
                                 "{SECTION_ID}": part["section_id"],
                                 "{SECTION_TITLE}": part["title"],
                                 "{SECTION_PART}": part["part"],
                                 "{DOCUMENT_OUTLINE}": outline}.items():
                        p2 = p2.replace(k, v)
                    p2 += ("\n\nCOVERAGE RECHECK\nA previous pass left these source blocks "
                           "represented by no unit and given no disposition: "
                           + ", ".join(missed[:40]) +
                           ".\nEnumerate this part again from the source. For each listed "
                           "block, either give it a unit or a specific narrow disposition. "
                           "Do not split a proposition merely to raise the count.")
                    p2 = custom_instructions.decorate_prompt(p2)
                    obj2 = parse_strict(ms.run(
                        p2, out_dir, ms.PLAN_MODELS, f"replan{i:03d}",
                            validate=lambda r: _json_ok(r, "plan"),
                        gateway_options=PLAN_REQUEST_OPTIONS), "replan")
                    obj2["_custom_instruction_digest"] = custom_instructions.current_digest()
                    if len(obj2.get("units") or []) > len(got):
                        obj2 = verify_part(ms, part, out_dir, i, obj2,
                                           stage_prefix="replan-")
                        got = obj2["units"]; obj = obj2
                        # Persist the better plan, so a rebuild does not pay for
                        # this re-plan again on every subsequent run.
                        (out_dir / f"plan{i:03d}.json").write_text(
                            json.dumps(obj2, indent=2))
                        print(f"  {part['section_id']} re-planned -> {len(got)} units",
                              flush=True)
                except Exception as e:
                    print(f"  {part['section_id']} re-plan unusable "
                          f"({str(e)[:100]}) — keeping the verified plan", flush=True)
        for u in got:
            u["unit_id"] = f"U{len(units)+1:03d}"   # build only appends; safe here
            u["section_id"] = part["section_id"]
            # Which planning window produced this unit. Composition uses it as a
            # paragraph boundary: units from different windows were being
            # concatenated mid-paragraph, so one paragraph ran from Japan's
            # fiscal withdrawal straight into the Fed's 2007 facilities. Papers
            # with one long section have no other structural signal.
            u["part"] = i
            u.pop("local_id", None)
            units.append(u)
        disp += obj.get("dispositions") or []
        flag = ""
        if len(got) < floor:
            thin.append(part["section_id"]); flag = f"  THIN (floor {floor})"
        print(f"  {part['section_id']} part {part['part']}: {cw}w -> {len(got)} units "
              f"(ref {ref}){flag}", flush=True)

    # Remap after every part is planned: a unit in part 9 can depend on a unit
    # from part 2, and each part restarts its local numbering.
    unresolved = set()
    for u in units:
        deps = []
        for d in u.get("dependencies") or []:
            r = localmap.get(d, d)
            if not re.fullmatch(r"U\d+", r or ""):
                unresolved.add(d); continue      # drop, do not dangle
            deps.append(r)
        u["dependencies"] = deps
    for d in disp:
        d["represented_by"] = [localmap.get(r, r) for r in (d.get("represented_by") or [])
                               if re.fullmatch(r"U\d+", localmap.get(r, r) or "")]
    if unresolved:
        print(f"[ledger] dropped {len(unresolved)} unresolvable dependency ref(s)", flush=True)

    ledger = {"source_sha256": smap["source_sha256"],
              "visible_words": smap["visible_words"],
              "units": units, "dispositions": disp, "thin_sections": thin,
              "part_quarantine": [],
              # A part that never planned is absent coverage. A THIN part planned
              # fewer units than a density heuristic expects but still accounts
              # for every one of its source blocks -- conflating the two blocked
              # a complete document from publishing.
              "unplanned_parts": unplanned}
    (out_dir / "ledger.json").write_text(json.dumps(ledger, indent=2))
    req_d = sum(1 for u in units if u.get("detailed_disposition") == "required")
    req_b = sum(1 for u in units if u.get("brief_disposition") == "required")
    dcap = sum(len((u.get("detailed_capsule") or "").split()) for u in units)
    bcap = sum(len((u.get("brief_capsule") or "").split()) for u in units)
    print(f"[ledger] {len(units)} units ({req_d} req detailed, {req_b} req brief); "
          f"capsule inventory {dcap}w detailed / {bcap}w brief", flush=True)
    return ledger


# KNOWN CEILING -- quantity-relation errors.
# A figure can appear verbatim in the source and still be reattached to the wrong
# subject: "the swing was more than 10 percent" became "a surplus exceeding 10
# percent". Numeric grounding passes, because the digits ARE in the source.
# A local audit may catch this while a larger audit misses it, and a
# numbers-only pass can identify the right unit while describing the source
# relation incorrectly. It remains uncaught rather than shipping a check that
# produces confident but misleading repair instructions.
#
# A technical name INTRODUCED by the summary -- "serves as a sacrifice ratio" --
# is a claim about what the source established, not a paraphrase of it. One such
# term reached a published summary and sealed `verified` because the audit that
# had caught it did not run again; the auditor is stochastic and this class of
# defect is not.
#
# Deliberately narrow. Matching any adjective+metric compound flagged 31 terms on
# one document, nearly all grammatical fragments ("because this ratio", "an
# index"); requiring an explicit naming construction flagged exactly one across
# 247 units, and it was the real defect.
INTRODUCED_TERM = re.compile(
    r"\b(?:serves?\s+as|acts?\s+as|functions?\s+as|known\s+as|called|termed|"
    r"referred\s+to\s+as|so-called|what\s+is\s+known\s+as)\s+"
    r"(?:the|a|an)\s+"
    r"((?:[a-z][a-z\-]+\s+){0,2}"
    r"(?:ratio|index|coefficient|theorem|multiplier|elasticity|premium|parity|"
    r"identity|rule|law|effect|principle|hypothesis|paradox|curve))\b")


def introduced_terms(ledger, source_text):
    """{(unit_id, depth): {term}} for named terms absent from the source.

    Deterministic, no model calls. The consequence is withholding the affected
    depth, never editing the capsule: deleting text to satisfy a gate is how a
    summary quietly stops matching its ledger.
    """
    src = source_text.lower()
    hits = {}
    for u in ledger["units"]:
        for depth in ("detailed", "brief"):
            cap = (u.get(f"{depth}_capsule") or "").lower()
            for m in INTRODUCED_TERM.finditer(cap):
                term = m.group(1).strip()
                if term and term not in src:
                    hits.setdefault((u["unit_id"], depth), set()).add(term)
    return hits


def apply_disposition_notes(ledger, findings):
    """Act on duplicate findings that name a disposition fix.

    `duplicate` is non-blocking -- it never means a claim is false -- but the
    auditor reliably identifies abstract/introduction units that restate the body
    and says so. Ignoring that let one document publish its core propositions
    twice and overshoot its band. Applying only the named disposition change is
    safe: nothing is deleted, the unit simply stops being rendered.
    """
    by = {u["unit_id"]: u for u in ledger["units"]}
    changed = 0
    for f in findings:
        if f.get("type") != "duplicate":
            continue
        instr = (f.get("repair_instruction") or "").lower()
        if "omit" not in instr:
            continue
        ids = set(f.get("unit_ids") or [])
        ids |= set(re.findall(r"\bU\d{3}\b", f.get("repair_instruction") or ""))
        # never omit a unit the instruction names as the one to KEEP
        keep = set(re.findall(r"(?:relying on|carry these claims).{0,120}?((?:U\d{3}[,\s and]*)+)",
                              f.get("repair_instruction") or ""))
        keepers = set(re.findall(r"U\d{3}", " ".join(keep)))
        for uid in ids - keepers:
            u = by.get(uid)
            if u and (u.get("detailed_disposition") != "omit"
                      or u.get("brief_disposition") != "omit"):
                u["detailed_disposition"] = "omit"
                u["brief_disposition"] = "omit"
                changed += 1
    if changed:
        print(f"[ledger] applied {changed} duplicate-disposition note(s)", flush=True)
    return ledger


ALLOWED_DISP = ledger_schema.DISPOSITIONS


def canon_disposition(v):
    """Expand an unambiguous abbreviation of a disposition enum.

    A planner emitted "incidental" once in nineteen dispositions, having used
    "incidental_example" correctly three times. Losing a whole document to one
    truncated enum is a bad trade when exactly one valid value starts with that
    string. Ambiguous or unrecognised values are left alone to fail loudly.
    """
    return ledger_schema.canonical_disposition(v)


def normalize(ledger):
    """Deterministic invariants, so the auditor never spends a round on
    bookkeeping. A block assigned to a unit IS accounted for; a disposition
    claiming the same block is a contradiction the auditor was correctly
    flagging, and code can simply prevent it."""
    # A block claimed by BOTH a unit and an `apparatus` disposition is a
    # contradiction. Dropping the disposition was the wrong way round when the
    # block is a heading: one unit claimed a five-word heading "3. QE AND ITS
    # CRITICS" alongside the paragraph it actually summarised, so the heading
    # counted as substantive coverage and the seal failed. When the unit has
    # other grounding, the disposition is right and the claim is spurious.
    apparatus = {b for d in (ledger.get("dispositions") or [])
                 if d.get("disposition") == "apparatus"
                 for b in (d.get("source_ids") or [])}
    unclaimed = 0
    for u in ledger["units"]:
        ids = u.get("source_ids") or []
        keep_ids = [b for b in ids if b not in apparatus]
        if keep_ids and len(keep_ids) != len(ids):
            u["source_ids"] = keep_ids
            unclaimed += len(ids) - len(keep_ids)
    if unclaimed:
        print(f"[ledger] normalized: released {unclaimed} apparatus block(s) from "
              f"units that had other grounding", flush=True)

    covered = {b for u in ledger["units"] for b in (u.get("source_ids") or [])}
    kept, dropped, fixed = [], 0, 0
    for d in ledger.get("dispositions") or []:
        c = canon_disposition(d.get("disposition"))
        if c != d.get("disposition"):
            d = {**d, "disposition": c}; fixed += 1
        ids = [b for b in (d.get("source_ids") or []) if b not in covered]
        if not ids:
            dropped += 1; continue
        if len(ids) != len(d.get("source_ids") or []):
            d = {**d, "source_ids": ids}
        kept.append(d)
    ledger["dispositions"] = kept
    # a unit that lost every source id is no longer source-grounded
    before = len(ledger["units"])
    ledger["units"] = [u for u in ledger["units"] if (u.get("source_ids") or [])]
    if fixed:
        print(f"[ledger] normalized: expanded {fixed} abbreviated disposition(s)",
              flush=True)
    if dropped or before != len(ledger["units"]):
        print(f"[ledger] normalized: dropped {dropped} conflicting disposition(s), "
              f"{before-len(ledger['units'])} ungrounded unit(s)", flush=True)
    return ledger


def _next_id(ledger):
    used = {u.get("unit_id", "") for u in ledger["units"]}
    n = max([int(x[1:]) for x in used if re.fullmatch(r"U\d+", x or "")] or [0])
    while True:
        n += 1
        cand = f"U{n:03d}"
        if cand not in used:
            return cand


def main():
    rv, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    ms = _runner()
    # A sealed ledger is immutable. Re-invoking used to rebuild and rewrite
    # ledger.json while status.json kept the old hash, so the seal silently
    # stopped describing the ledger it authorised.
    if (out / "SEALED").exists() and (out / "status.json").exists():
        try:
            st = json.loads((out / "status.json").read_text())
            cur = hashlib.sha256((out / "ledger.json").read_bytes()).hexdigest()
            if st.get("ledger_sha256") == cur:
                print(f"[ledger] already sealed (status={st.get('status')}) — "
                      f"nothing to do", flush=True)
                return 0
            print("[ledger] sealed record does not match ledger.json — resealing",
                  flush=True)
            (out / "SEALED").unlink(missing_ok=True)
        except Exception:
            (out / "SEALED").unlink(missing_ok=True)
    ledger = normalize(build(rv, out))
    # normalize() worked on the in-memory ledger while mechseal reads
    # ledger.json from disk, so its fixes were invisible to the very gate they
    # exist to satisfy: one run had a conflicting disposition dropped in memory
    # and still failed the seal for that exact block.
    (out / "ledger.json").write_text(json.dumps(ledger, indent=2))

    # --- tier 1: mechanical seal. Code-only, and the precondition for any
    #     publication at all. Failing it means no usable ledger exists.
    def mechseal():
        return subprocess.run([sys.executable, str(HERE / "mechseal.py"),
                               str(out), str(rv)]).returncode == 0
    # Provisional seal: the audit stage should not run against a malformed ledger.
    if not mechseal():
        print("[ledger] no mechanical seal — nothing can be published", flush=True)
        return 3

    # --- tier 2: semantic verification. A mechanically valid ledger may be
    #     quarantined for known semantic defects, but bytes that have never
    #     completed a semantic audit are not sealed and not published.
    # Verification has already happened per part, next to the source window it applies to,
    # with at most one full-part revision and one audit of the exact revision.
    # The whole-ledger audit ladder is gone: it re-read a 63k-token ledger five
    # times, its findings were non-monotonic across rounds, its per-finding
    # repairs broke the mechanical seal, and a later round's silence could close
    # an earlier unresolved finding.
    status, notes, incidents = "verified", [], []
    quarantine = {"detailed": set(), "brief": set()}
    ids = {u["unit_id"] for u in ledger["units"]}
    carried = [u for u in (ledger.get("part_quarantine") or []) if u in ids]
    if carried:
        print(f"[ledger] {len(carried)} unit(s) carry unresolved part findings — "
              "not publishable", flush=True)
        return 5
    if ledger.get("unplanned_parts"):
        # Absent coverage. Quarantining nothing can repair a part that produced
        # no units, so it must not publish.
        print(f"[ledger] {len(ledger['unplanned_parts'])} section part(s) never "
              f"planned — coverage is incomplete, not publishable: "
              f"{ledger['unplanned_parts']}", flush=True)
        return 4
    if ledger.get("thin_sections"):
        # Below the density heuristic but fully accounted: a quality signal, not
        # a coverage failure.
        print(f"[ledger] {len(ledger['thin_sections'])} part(s) below the density "
              f"reference; source accounting is still complete", flush=True)

    # DETERMINISTIC FIDELITY GATE. Runs after every audit path, costs nothing,
    # and does not vary between runs.
    try:
        src_text = (rv / "source.visible.md").read_text(errors="replace")
    except Exception:
        src_text = ""
    if src_text:
        # Exact-quotation requests are a hard fidelity boundary.  Part checks
        # give the repair model an early chance to correct altered prose, but
        # an earlier defective attempt can be retained when a revision is worse.
        # Recheck the final normalized ledger against the complete visible
        # source and refuse to seal it if any quoted passage is still altered.
        quote_failures = custom_instructions.quote_defects(
            [u.get("detailed_capsule") for u in ledger["units"]]
            + [u.get("brief_capsule") for u in ledger["units"]], src_text)
        if quote_failures:
            print("[ledger] exact-quotation verification failed after repair — "
                  "not publishable", flush=True)
            for failure in quote_failures:
                print(f"[ledger] {failure}", flush=True)
            return 5
        for (uid, depth), terms in sorted(introduced_terms(ledger, src_text).items()):
            print(f"[ledger] {depth}: {uid} introduces {sorted(terms)} — absent from "
                  f"the source; withholding that depth", flush=True)
            quarantine[depth].add(uid)
            incidents.append({"gate": "introduced_term", "unit_id": uid,
                              "depth": depth, "terms": sorted(terms)})
        if any(quarantine.values()) and status == "verified":
            status = "quarantined"
        unsafe = quarantine["detailed"] & quarantine["brief"]
        if unsafe:
            print(f"[ledger] {len(unsafe)} unit(s) have no verified capsule — "
                  f"not publishable: {sorted(unsafe)[:8]}", flush=True)
            return 5

    # RE-SEAL. Repairs rewrite ledger.json after the provisional seal, so the
    # earlier record described a ledger that no longer exists -- on one document
    # the sealed record said 65 units while the published ledger had 69.
    (out / "MECHSEAL").unlink(missing_ok=True)
    if not mechseal():
        print("[ledger] final ledger fails the mechanical seal — not publishable",
              flush=True)
        return 3

    # Quarantine must name units that exist in the ledger being sealed. A
    # finding against U012 was carried after repair replaced it with U072, so
    # composition withheld nothing while the artifact disclosed an omission that
    # never happened.
    ids = {u["unit_id"] for u in ledger["units"]}
    for d in list(quarantine):
        gone = quarantine[d] - ids
        if gone:
            print(f"[ledger] {d}: dropping {len(gone)} quarantine id(s) that no "
                  f"longer exist after repair: {sorted(gone)}", flush=True)
            quarantine[d] &= ids
    if status == "quarantined" and not any(quarantine.values()):
        print("[ledger] quarantine names no current unit — not publishable", flush=True)
        return 5

    lhash = hashlib.sha256((out / "ledger.json").read_bytes()).hexdigest()
    (out / "status.json").write_text(json.dumps(
        {"status": status, "notes": notes, "incidents": incidents,
         "quarantine": {d: sorted(v) for d, v in quarantine.items()},
         "ledger_sha256": lhash}, indent=2))
    (out / "SEALED").write_text("1")
    print(f"[ledger] SEALED (status={status})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
