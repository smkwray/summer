#!/usr/bin/env python3
"""Select units in code, compose both artifacts, audit them, repair locally.

Three corrections from the previous version, each tied to a real defect.

**Selection lives here, not in the prompt.** The composer used to be handed a
word target and left to decide what survived, which is how "be shorter" became
"compress explanations until claims lose their support". It now receives an
already-selected inventory and no target at all: its only job is to render every
unit it is given, exactly once.

**Artifacts are audited.** The previous version checked composed prose with a
regex for YAML leakage and nothing else, so dropped qualifiers, invented
bridges and dangling references published unchecked.

**Repair is depth-correct.** It used to read the first represented unit only and
fall back to `detailed_capsule` at both depths, so a Brief repair could silently
become a Detailed rendering.

usage: compose.py LEDGER_DIR READERVIEW_DIR OUT_DIR
"""
from __future__ import annotations
import hashlib, json, os, pathlib, re, sys, unicodedata

HERE = pathlib.Path(__file__).parent
# Dashboard ranges only. These no longer gate anything: the accepted artifacts
# came in at 28% and 13%, and adding units to hit a percentage is how earlier
# versions produced padding. The real constraint is the distinctness ceiling.
BANDS = {"detailed": (0.25, 0.40), "brief": (0.10, 0.22)}
BRIEF_MAX_OF_DETAILED = 0.65

# Deterministic capsule rendering is the DEFAULT, not a fallback. Every artifact
# the owner has read and accepted came from this path, and it removes both
# composer calls, both initial artifact audits, and every per-finding sentence
# repair -- the largest safe saving available. Composed prose stays selectable
# for experiments and is not yet qualified.
# Capsule rendering is the only installed mode. The model composer and the
# artifact-audit/sentence-repair ladder behind it never qualified: no run has
# ever exercised them, and review found they could publish on a `blocked`
# verdict with no findings, repair only the first target of a finding, and
# accept model-supplied unit IDs without coverage validation. An unqualified
# path behind an environment variable is worse than no path -- git history is
# preservation enough.
RENDER_MODE = "capsules"


def cap(u, depth):
    return (u.get("detailed_capsule") if depth == "detailed"
            else u.get("brief_capsule")) or ""


def select(units, depth, visible_words):
    """Deterministic. Required units and their dependencies, then optional ones
    in priority/source order until the lower inventory band is reachable."""
    dkey = f"{depth}_disposition"
    by = {u["unit_id"]: u for u in units}
    lo = int(BANDS[depth][0] * visible_words)

    chosen, seen = [], set()

    def add(u):
        if u["unit_id"] in seen:
            return
        seen.add(u["unit_id"])
        for dep in u.get("dependencies") or []:
            if dep in by and dep not in seen:
                add(by[dep])
        chosen.append(u)

    for u in units:
        if u.get(dkey) == "required" and cap(u, depth).strip():
            add(u)

    def words():
        return sum(len(cap(u, depth).split()) for u in chosen)

    optional = [u for u in units if u.get(dkey) == "optional" and cap(u, depth).strip()]
    if depth == "brief":
        optional.sort(key=lambda u: (u.get("brief_priority", 5), units.index(u)))
    for u in optional:
        if words() >= lo:
            break
        add(u)
    return sorted(chosen, key=lambda u: units.index(u)), words()


SELECTOR_VERSION = "multires-v2"
SUSPECT_BRIEF = []
# The ceiling applies to the finished FILE, not to capsule inventory. Budgeting
# inventory alone published a Brief 19 words over, because the coverage note is
# part of the artifact the reader receives. Reserve enough for the longest note
# the run can emit.
NOTE_ALLOWANCE = 70


def content_budget(hi):
    """Words available to CONTENT, once the coverage note is reserved.

    The note is apparatus this stage adds, not content, but the ceiling is
    counted on the finished file, so its cost has to come out of the budget.
    It is a FIXED cost against a ceiling that SCALES with the source: a 253-word
    document gives Brief a 55-word ceiling, and hi - 70 was -15. A negative
    budget is not a tight budget -- no arrangement of units can ever satisfy it,
    so the Brief drop loop deleted every unit and the run published a file
    containing two coverage notes and no summary. Capping the reservation at
    half the ceiling keeps the budget positive without moving it at all on
    documents long enough for the note to be a rounding error.
    """
    return hi - min(NOTE_ALLOWANCE, hi // 2)


def plan_modes(chosen, depth, ceiling):
    """Assign each selected unit a capsule resolution so the artifact fits.

    A unit is the content obligation; its Detailed and Brief capsules are two
    sealed resolutions of that obligation. When the Detailed inventory exceeds
    the ceiling, the fix is to represent lower-salience units at their Brief
    resolution -- not to delete them. Deleting the least important `required`
    units would fit one document only by omitting ~39 of them, which turns
    "required" into "required unless the model was verbose".

    Order of compaction: Brief-optional before Brief-required, then descending
    brief_priority (5 before 1), then the larger word saving, then source order.
    Returns (modes, words, minimum_feasible, dropped_unit_ids, kept_units).
    """
    def words_at(u, mode):
        return len((u.get(f"{mode}_capsule") or "").split())

    suspect = set(SUSPECT_BRIEF)

    def admissible_brief(u):
        # `omit` at Brief means the Brief capsule is not a sanctioned
        # representation of this unit, whatever text happens to sit in it.
        # A Brief capsule an audit flagged is not a sanctioned lower-resolution
        # representation, even though the unit itself is fine at Detailed.
        return (u.get("brief_disposition") != "omit"
                and u["unit_id"] not in suspect
                and (u.get("brief_capsule") or "").strip())

    modes = {}
    for u in chosen:
        if depth == "brief" and not admissible_brief(u):
            # Pulled in through dependency closure at a depth that omits it:
            # render the higher resolution rather than text the ledger did not
            # sanction. Two such units reached one Brief artifact this way.
            modes[u["unit_id"]] = "detailed"
        else:
            modes[u["unit_id"]] = depth

    keep = {u["unit_id"] for u in chosen}

    def total():
        return sum(words_at(u, modes[u["unit_id"]])
                   for u in chosen if u["unit_id"] in keep)

    minimum = sum(words_at(u, "brief") if (depth == "detailed" and admissible_brief(u))
                  else words_at(u, modes[u["unit_id"]]) for u in chosen)

    dropped = []
    if depth == "brief" and total() > ceiling:
        # Brief has no lower resolution -- its capsules ARE the floor -- so the
        # only lever left is which units it carries. That is sanctioned here and
        # nowhere else: the schema gives every unit a brief_disposition and a
        # brief_priority precisely because Brief is a selection, while Detailed
        # is a comprehensive condensation that must represent every unit.
        # Dropped passages are disclosed and remain in full in Detailed.
        def still_needed(uid):
            return any(uid in (o.get("dependencies") or []) for o in chosen
                       if o["unit_id"] in keep and o["unit_id"] != uid)
        order = sorted(chosen,
                       key=lambda u: (0 if u.get("brief_disposition") == "optional" else 1,
                                      -int(u.get("brief_priority", 5) or 5),
                                      -words_at(u, modes[u["unit_id"]])))
        for u in order:
            if total() <= ceiling:
                break
            if len(keep) <= 1:
                # A Brief carrying nothing is not a shorter reading of the
                # document; it is a missing artifact with a disclosure note
                # attached. Stop here and let the ceiling gate refuse loudly,
                # rather than "fitting" the budget by deleting the summary.
                break
            uid = u["unit_id"]
            if still_needed(uid):
                continue
            keep.discard(uid)
            dropped.append(uid)
        chosen = [u for u in chosen if u["unit_id"] in keep]
        minimum = total()

    if depth == "detailed" and total() > ceiling:
        order = sorted(
            [u for u in chosen if admissible_brief(u)],
            key=lambda u: (0 if u.get("brief_disposition") == "optional" else 1,
                           -int(u.get("brief_priority", 5) or 5),
                           -(words_at(u, "detailed") - words_at(u, "brief")),
                           chosen.index(u)))
        for u in order:
            if total() <= ceiling:
                break
            if words_at(u, "brief") < words_at(u, "detailed"):
                modes[u["unit_id"]] = "brief"
    return modes, total(), minimum, dropped, chosen


def from_capsules(chosen, depth, prefix, modes=None):
    """Deterministic floor: every sentence already passed the ledger audit."""
    def text(u):
        m = (modes or {}).get(u["unit_id"], depth)
        return (u.get(f"{m}_capsule") or "").strip()
    return [{"id": f"{prefix}-S{i:04d}", "heading": False,
             "text": text(u), "unit_ids": [u["unit_id"]],
             # carried so paragraphing can respect source structure
             "section": (u.get("section_id"), u.get("part"))}
            for i, u in enumerate(chosen, 1) if text(u)]


# Paragraph sizes for reading on an e-ink device. One capsule per paragraph gave
# Brief 59 paragraphs with a 33-word median, 55 of them under 45 words -- correct
# prose, unpleasant to read. Grouping is deterministic: it moves whitespace
# between sentence objects and never touches their text or order.
PARA = {"detailed": (85, 125, 175), "brief": (70, 100, 140)}


def paragraphize(items, depth):
    """Partition consecutive sentence objects into readable paragraphs.

    Headings are hard boundaries. A sentence object is indivisible, so a single
    over-long capsule may exceed the maximum; nothing else may."""
    lo, pref, hi = PARA[depth]
    out, cur, n = [], [], 0
    prev_section = None
    for it in items:
        # A planning-window change is a paragraph boundary. Concatenation was
        # window-blind, so one paragraph could run from Japan's fiscal
        # withdrawal straight into the Fed's 2007 facilities -- two differently
        # scoped units forced together, which reads as a non-sequitur however
        # good each capsule is. Section alone is not enough: a paper can be one
        # long section.
        sec = it.get("section")
        # ...but only once the open paragraph is already readable. Breaking on
        # every window change regardless produced a 27-word paragraph mid-text;
        # a runt is a worse reading defect than a slightly mixed paragraph, and
        # short paragraphs on e-ink were the original complaint.
        if (sec is not None and prev_section is not None and sec != prev_section
                and cur and n >= lo):
            out.append({"heading": False, "parts": cur}); cur, n = [], 0
        prev_section = sec if sec is not None else prev_section
        if it["heading"]:
            if cur:
                out.append({"heading": False, "parts": cur}); cur, n = [], 0
            out.append({"heading": True, "parts": [it]})
            continue
        w = len(it["text"].split())
        # close the paragraph when adding this object would overshoot and we are
        # already past the minimum, or when we are already nearer to preferred
        if cur and n >= lo and (n + w > hi or abs(n - pref) <= abs(n + w - pref)):
            out.append({"heading": False, "parts": cur}); cur, n = [], 0
        cur.append(it); n += w
    if cur:
        # A short tail is the one runt this partition can produce, and it lands
        # where it is most visible: the last thing read. Merge it back when it
        # fits; when it does not, rebalance the final two paragraphs rather than
        # leaving the orphan -- a 36-word closing paragraph reads worse on e-ink
        # than two slightly uneven ones.
        if out and not out[-1]["heading"] and n < lo:
            merged = out[-1]["parts"] + cur
            total = sum(len(x["text"].split()) for x in merged)
            if total <= hi:
                out[-1]["parts"] = merged; cur = []
            else:
                # Prefer an even split at a sentence boundary; sentences are
                # indivisible, so one may not exist.
                run, best, bestdiff = 0, None, None
                for i in range(1, len(merged)):
                    run += len(merged[i - 1]["text"].split())
                    diff = abs(run - (total - run))
                    if bestdiff is None or diff < bestdiff:
                        best, bestdiff = i, diff
                a, b = (merged[:best], merged[best:]) if best else ([], merged)
                if a and min(sum(len(x["text"].split()) for x in part)
                             for part in (a, b)) >= lo:
                    out[-1]["parts"], cur = a, b
                elif total <= hi + lo:
                    # No legal split. One overlong closing paragraph still reads
                    # better than an orphan, so absorb rather than strand it.
                    out[-1]["parts"] = merged; cur = []
        if cur:
            out.append({"heading": False, "parts": cur})
    return out


def render_paragraphs(paras):
    blocks = []
    for p in paras:
        if p["heading"]:
            blocks.append("\n## " + p["parts"][0]["text"])
        else:
            blocks.append(" ".join(x["text"] for x in p["parts"]))
    return re.sub(r"\n{3,}", "\n\n", "\n\n".join(blocks)).strip() + "\n"


def render(items):
    return re.sub(r"\n{3,}", "\n\n", "\n\n".join(
        ("\n## " + i["text"]) if i["heading"] else i["text"] for i in items)).strip() + "\n"


def dedupe(items, depth):
    """Collapse only EXACT duplicates, after normalising case and punctuation.

    A lexical resemblance rule held deletion authority here and was wrong three
    ways at once: content-word sets dropped tokens of four characters or fewer,
    so "not" and "only" were invisible and 0.13 read as 0.31; and
    first-occurrence-wins kept whichever rendering happened to come first,
    regardless of which carried more. Adversarial cases in review deleted the
    negated, the rescoped and the renumbered sentence in each pair.

    Near-duplicate handling belongs to the ledger audit, which has the source in
    front of it and can name a canonical unit. Code may only remove text it can
    prove says the same thing, which means character-for-character after
    normalisation.
    """
    def norm(t):
        # Case, signs, operators, brackets, slashes and punctuation all carry
        # meaning: stripping them equated -0.13 with 0.13 and x < 0 with x > 0.
        # Normalise only representation-neutral Unicode and whitespace.
        return re.sub(r"\s+", " ", unicodedata.normalize("NFC", t)).strip()

    kept, seen, dropped = [], {}, []
    for it in items:
        if it.get("heading"):
            kept.append(it)
            continue
        k = norm(it["text"])
        if k and k in seen:
            # keep the union of provenance: the surviving sentence now stands
            # for every unit that produced it.
            prev = seen[k]
            prev["unit_ids"] = list(dict.fromkeys(
                (prev.get("unit_ids") or []) + (it.get("unit_ids") or [])))
            dropped.append(it["id"])
            continue
        seen[k] = it
        kept.append(it)
    if dropped:
        print(f"[compose] {depth}: collapsed {len(dropped)} exact duplicate(s): "
              f"{dropped[:8]}", flush=True)
    return kept


def render_grouped(items, depth):
    """Group then render, asserting the sentence set survived untouched."""
    paras = paragraphize(items, depth)
    before = [(i["id"], i["text"]) for i in items]
    after = [(x["id"], x["text"]) for p in paras for x in p["parts"]]
    assert after == before, "paragraphizer altered the sentence set"
    return render_paragraphs(paras)


def authenticate(ldir):
    """Refuse to compose from a ledger whose authorisation cannot be verified.

    Composition used to trust `status.json` and republish the hash it claimed.
    A fixture with no SEALED record and a fabricated hash produced both
    artifacts and a confident provenance stamp, exit code 0 -- so every
    downstream guarantee rested on a file anyone could write. The seal is only
    worth anything if the thing consuming it checks.

    Returns (ledger, status). Raises SystemExit on any failure; the caller has
    not created any output at that point.
    """
    def fail(why):
        print(f"[compose] refusing to publish: {why}", file=sys.stderr, flush=True)
        raise SystemExit(2)

    if (ldir / "CORPUS_SEAL").exists() or (ldir / "corpus-seal.json").exists():
        return authenticate_corpus(ldir)

    for marker in ("SEALED", "MECHSEAL"):
        if not (ldir / marker).exists():
            fail(f"no {marker} record in {ldir}")
    try:
        mech = json.loads((ldir / "mechseal.json").read_text())
    except Exception as e:
        fail(f"mechseal.json unreadable ({str(e)[:60]})")
    if not mech.get("passed"):
        fail("the mechanical seal did not pass")
    try:
        st = json.loads((ldir / "status.json").read_text())
    except Exception as e:
        fail(f"status.json unreadable ({str(e)[:60]})")

    raw = (ldir / "ledger.json").read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    claimed = st.get("ledger_sha256")
    if not claimed:
        fail("status.json binds no ledger hash")
    if claimed != actual:
        fail(f"status.json authorises {claimed[:16]}… but ledger.json is "
             f"{actual[:16]}… — these are different objects")
    if st.get("status") not in {"verified", "quarantined"}:
        fail(f"semantic status {st.get('status')!r} is not publishable")
    quarantine = st.get("quarantine") or {}
    both = set(quarantine.get("detailed") or ()) & set(
        quarantine.get("brief") or ())
    if both:
        fail(f"{len(both)} unit(s) have no verified capsule")
    return json.loads(raw), st


def authenticate_corpus(ldir):
    """Authenticate the aggregate Corpus seal before rendering either depth."""
    def fail(why):
        print(f"[compose] refusing to publish Corpus: {why}",
              file=sys.stderr, flush=True)
        raise SystemExit(3)

    for marker in ("CORPUS_SEAL", "SEALED"):
        if not (ldir / marker).exists():
            fail(f"no {marker} record in {ldir}")
    try:
        seal = json.loads((ldir / "corpus-seal.json").read_text())
        status = json.loads((ldir / "status.json").read_text())
        raw = (ldir / "ledger.json").read_bytes()
        ledger = json.loads(raw)
    except Exception as exc:
        fail(f"Corpus seal, status, or ledger is unreadable ({str(exc)[:100]})")
    actual = hashlib.sha256(raw).hexdigest()
    if seal.get("schema") != "summer.corpus-seal.v1" or not seal.get("passed"):
        fail("Corpus structural seal did not pass")
    if ledger.get("schema") != "summer.corpus-ledger.v1" or ledger.get("kind") != "corpus":
        fail("ledger is not a Corpus ledger")
    if seal.get("ledger_sha256") != actual or status.get("ledger_sha256") != actual:
        fail("Corpus seal/status do not bind the current ledger bytes")
    if status.get("status") != "verified" or status.get("kind") != "corpus":
        fail("Corpus status is not verified")
    return ledger, status


def main():
    ldir, rv, out = (pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]),
                     pathlib.Path(sys.argv[3]))
    ledger, st = authenticate(ldir)      # before any output directory is made
    quarantine = st.get("quarantine") or {}
    # A Brief capsule rejected by a deterministic depth-specific gate must not
    # be promoted into Detailed merely to save words. The seal status is the
    # authority; a parallel hint inside ledger.json drifted empty.
    SUSPECT_BRIEF[:] = quarantine.get("brief") or []
    out.mkdir(parents=True, exist_ok=True)
    # A depth-specific deterministic finding may use the same unit's other,
    # verified capsule.  Authentication has already refused any unit flagged at
    # both depths, so composition never emits text known to be unverified and
    # never drops source content to manufacture a pass.
    def fallbacks(depth):
        """Return all units and any verified alternate-capsule renderings."""
        q = set(quarantine.get(depth) or ())
        other = "brief" if depth == "detailed" else "detailed"
        oq = set(quarantine.get(other) or ())
        modes = {}
        for u in ledger["units"]:
            uid = u["unit_id"]
            if uid not in q:
                continue
            if uid in oq or not (u.get(f"{other}_capsule") or "").strip():
                raise SystemExit(5)          # authenticate should have caught this
            modes[uid] = other
        if q:
            print(f"[compose] {depth}: {len(q)} flagged unit(s) — "
                  f"{len(modes)} re-rendered at their verified {other} capsule, "
                  "0 omitted", flush=True)
        return ledger["units"], modes

    units = ledger["units"]
    by_unit = {u["unit_id"]: u for u in units}
    vw = ledger["visible_words"]

    INCOMPLETE = ("> *Coverage note: final semantic verification did not complete. "
                  "Every passage is present; some were not confirmed against the "
                  "source. No source text was substituted.*\n\n")
    NOTE = {"coverage_unverified": INCOMPLETE,
            "coverage_degraded": INCOMPLETE,
            # Nothing is omitted any more, so the note no longer says it is.
            "quarantined": ""}.get(st.get("status"), "")

    report = {}
    for depth, prefix in (("detailed", "D"), ("brief", "B")):
        all_units, flagged_modes = fallbacks(depth)
        chosen, inv = select(all_units, depth, vw)
        lo, hi = int(BANDS[depth][0] * vw), int(BANDS[depth][1] * vw)
        print(f"[compose] {depth}: selected {len(chosen)} units, capsule inventory "
              f"{inv}w (band {lo}-{hi})", flush=True)

        # Resume only when the existing artifact provably belongs to THIS
        # generation. Resuming on existence alone let artifacts from an earlier
        # ledger survive a rebuild: reconstructing two documents from their final
        # ledgers no longer reproduced the published files.
        art_existing = out / f"{depth}.md"
        stamp = out / f"{depth}.provenance.json"
        budget = content_budget(hi)
        modes, planned_w, minimum_w, dropped, chosen = plan_modes(
            chosen, depth, budget)
        # A flagged unit's safer capsule wins over the budget's choice: the
        # budget is about length, this is about not stating something false.
        modes.update({uid: m for uid, m in flagged_modes.items() if uid in modes})
        length_exception = minimum_w > budget
        if length_exception:
            # Completeness and fidelity are hard constraints; the preferred
            # compression ceiling is not. Refusing a fully sealed summary here
            # produced no artifact after fifty minutes of successful model work.
            # Publish the shortest sanctioned complete reading and report the
            # exception instead of deleting content or deleting the result.
            print(f"[compose] {depth}: length_exception — minimum feasible "
                  f"{minimum_w}w exceeds the preferred {budget}w budget "
                  f"(nominal ceiling {hi}w); publishing every unit at its "
                  f"shortest sanctioned resolution", flush=True)
        if not chosen:
            # The one rule this project cannot bend: both artifacts, or neither
            # and say so. Quarantine can no longer empty this -- nothing is
            # dropped for fidelity -- so reaching here means selection and the
            # Brief budget between them left nothing to say, and the only thing
            # that would be written is the coverage note. Publishing that is worse than
            # publishing nothing, because it looks like a summary.
            print(f"[compose] {depth}: no unit survives selection — refusing to "
                  f"publish an artifact whose only content is a coverage note",
                  file=sys.stderr, flush=True)
            return 7
        if dropped:
            print(f"[compose] {depth}: dropped {len(dropped)} lowest-priority "
                  f"unit(s) to meet the {budget}w budget: {dropped[:8]}", flush=True)
        if planned_w != sum(len((u.get(f"{depth}_capsule") or "").split())
                            for u in chosen):
            n = sum(1 for u in chosen if modes[u["unit_id"]] != depth)
            print(f"[compose] {depth}: compacted {n} unit(s) to their Brief "
                  f"resolution to meet the {budget}w budget; no unit dropped",
                  flush=True)
        want = {"ledger_sha256": st.get("ledger_sha256"),
                "status": st.get("status"),
                "render_mode": RENDER_MODE,
                "selector_version": SELECTOR_VERSION,
                "upper_budget": budget,
                "minimum_feasible_words": minimum_w,
                "length_exception": length_exception,
                "unit_modes": modes,
                "dropped_unit_ids": dropped,
                "unit_ids": [u["unit_id"] for u in chosen]}
        if art_existing.exists() and art_existing.read_text().split() and stamp.exists():
            try:
                have = json.loads(stamp.read_text())
            except Exception:
                have = {}
            if all(have.get(k) == v for k, v in want.items()) and \
               have.get("sha256") == hashlib.sha256(art_existing.read_bytes()).hexdigest():
                w = len(art_existing.read_text().split())
                report[depth] = {"words": w, "band": [lo, hi], "units": len(chosen),
                                 "from_capsules": have.get("from_capsules"),
                                 "in_band": lo <= w <= hi, "resumed": True}
                print(f"[compose] {depth}: unchanged since last run ({w}w) — reusing",
                      flush=True)
                continue
            print(f"[compose] {depth}: existing artifact is from a different "
                  f"generation — recomposing", flush=True)

        items = dedupe(from_capsules(chosen, depth, prefix, modes), depth)
        art = out / f"{depth}.md"
        note = NOTE
        if dropped:
            note = ("> *Coverage note: the shorter version omits "
                    f"{len(dropped)} lower-priority passage(s) to stay materially "
                    "shorter than the source. They are present in full in the "
                    "detailed version.*\n\n") + NOTE
        art.write_text(note + render_grouped(items, depth))
        degraded = True

        w = len(art.read_text().split())
        stamp.write_text(json.dumps(
            {**want, "from_capsules": degraded,
             "sha256": hashlib.sha256(art.read_bytes()).hexdigest()}, indent=2))
        report[depth] = {"words": w, "band": [lo, hi], "units": len(chosen),
                         "from_capsules": degraded, "in_band": lo <= w <= hi,
                         "compacted": sum(1 for u in chosen
                                          if modes[u["unit_id"]] != depth),
                         "minimum_feasible_words": minimum_w,
                         "dropped_units": len(dropped)}
        print(f"[compose] {depth}: {w}w  band {lo}-{hi}  "
              f"{'OK' if lo <= w <= hi else 'OUT OF BAND'}"
              f"{'  [capsules]' if degraded else ''}", flush=True)

    d, b = report["detailed"]["words"], report["brief"]["words"]
    report["brief"]["pct_of_detailed"] = round(100 * b / max(1, d), 1)

    # THE PREFERRED PUBLICATION CEILING. Completeness and fidelity outrank it.
    # Composition records and reports an over-ceiling result so it cannot look
    # like an ordinary pass. Never pad to reach the lower bound; being short is
    # fine.
    over = [f"{dep} {report[dep]['words']}w > {report[dep]['band'][1]}w ceiling "
            f"({100 * report[dep]['words'] / max(1, vw):.0f}% of source)"
            for dep in ("detailed", "brief")
            if report[dep]["words"] > report[dep]["band"][1]]
    report["over_ceiling"] = over
    (out / "compose-report.json").write_text(json.dumps(report, indent=2))
    if over:
        print("[compose] length exception — " + "; ".join(over), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
