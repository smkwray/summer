#!/usr/bin/env python3
"""Profile a model chain against a frozen corpus, deterministically.

The project's only instrument for "is this model good enough" has been the owner
reading two artifacts. That does not scale to triaging several local candidates
on one afternoon, and it cannot say whether a model got WORSE after a prompt
change. Everything reported here is already computed by the pipeline or by
epistemic_gate.py; this runs the real entry point and reads what it left behind.

    qualify.py CORPUS_DIR [--candidate NAME ...] [--out DIR]

CORPUS_DIR holds the documents and a `candidates.json`:

    {"baseline": {"HARNESS": "opencode"},
     "split":    {"HARNESS": "opencode",
                  "OPENCODE_CHAIN_AUDIT": "<harness>:<model>"}}

A candidate is nothing but the environment the run happens in, so it reuses the
override mechanism the pipeline already has -- no second way to name a model,
and models.json stays the only place one is named.

A document may have a frozen `<name>.gate.json` beside it (see epistemic_gate.py).
Freeze that spec from the source BEFORE comparing candidates, or the yardstick
gets tuned to a result.

No judgement is made here. Whether a summary reads well is still the owner's
call; this narrows how many they have to read.
"""
from __future__ import annotations
import argparse, concurrent.futures as cf, itertools, json, os, pathlib, re, shutil, subprocess, sys, threading, time

HERE = pathlib.Path(__file__).parent
DOC_SUFFIXES = (".md", ".txt", ".pdf")


def documents(corpus: pathlib.Path):
    out = []
    for p in sorted(corpus.iterdir()):
        if p.suffix.lower() not in DOC_SUFFIXES or p.is_dir():
            continue
        # Artifacts of an earlier qualification run are not corpus documents.
        if re.search(r"\.(summary|brief|tts)$", p.stem) or p.name.endswith(".gate.json"):
            continue
        out.append(p)
    return out


def read_calls(run_dir: pathlib.Path):
    """Per-call costs, from the ledger run() writes."""
    f = run_dir / "calls.jsonl"
    if not f.is_file():
        return {"calls": None}
    recs = []
    for line in f.read_text().splitlines():
        if line.strip():
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    if not recs:
        return {"calls": 0}
    by = lambda o: sum(1 for r in recs if r.get("outcome") == o)
    return {
        "calls": len(recs),
        "seconds": round(sum(r.get("seconds", 0) for r in recs), 1),
        "prompt_mb": round(sum(r.get("prompt_bytes", 0) for r in recs) / 1e6, 2),
        "output_mb": round(sum(r.get("stdout_bytes", 0) for r in recs) / 1e6, 2),
        # Retries and fallthroughs are the local-model signal that matters most:
        # a model that answers only every third attempt is not cheap, whatever
        # it costs per call.
        "retried": sum(1 for r in recs if r.get("attempt", 1) > 1),
        "unusable": by("unusable"),
        "timeout": by("timeout"),
        "capacity": by("capacity"),
        "answered_by": sorted({f"{r['harness']}:{r['model']}" for r in recs
                               if r.get("outcome") == "ok"}),
    }


def read_ledger(ld: pathlib.Path):
    f = ld / "ledger.json"
    if not f.is_file():
        return {"units": None}
    try:
        L = json.loads(f.read_text())
    except json.JSONDecodeError:
        return {"units": None, "ledger": "unparseable"}
    u = L.get("units") or []
    return {
        "units": len(u),
        "visible_words": L.get("visible_words"),
        # Granularity is the largest observed difference between models on the
        # same document -- 79 units against 157 -- and nothing else reports it.
        "words_per_unit": round(L["visible_words"] / len(u), 1)
        if u and L.get("visible_words") else None,
        "quarantined": len(L.get("part_quarantine") or []),
        "brief_suspect": len(L.get("brief_suspect") or []),
        "thin_sections": len(L.get("thin_sections") or []),
        "unplanned_parts": len(L.get("unplanned_parts") or []),
        "sealed": (ld / "MECHSEAL").exists(),
    }


def read_artifacts(run: pathlib.Path):
    out = {}
    for depth in ("detailed", "brief"):
        p = run / f"{depth}.md"
        if not p.is_file():
            out[depth + "_words"] = None
            continue
        t = p.read_text(errors="replace")
        out[depth + "_words"] = len(t.split())
        # The formatting contract is checked on the PUBLISHED bytes, not on the
        # ledger that produced them. A gate that only ever reads its own input
        # cannot notice a renderer reintroducing what it forbade.
        out[depth + "_lists"] = len(re.findall(r"(?m)^\s*(?:[-*+]\s|\d+\.\s|\|)", t))
        out[depth + "_first_person"] = len(re.findall(r"(?i)\b(?:I|we|our|us|my)\b", t))
    return out


def read_gate(doc: pathlib.Path, run: pathlib.Path):
    spec = doc.with_suffix(doc.suffix + ".gate.json")
    if not spec.is_file():
        spec = doc.parent / (doc.stem + ".gate.json")
    if not spec.is_file():
        return {"gate": None}
    art = run / "detailed.md"
    if not art.is_file():
        return {"gate": "no artifact"}
    r = subprocess.run([sys.executable, str(HERE / "epistemic_gate.py"),
                        str(spec), str(art)], capture_output=True, text=True)
    return {"gate": "pass" if r.returncode == 0 else "FAIL",
            "gate_detail": (r.stdout + r.stderr).strip()[:300] or None}


def one(doc: pathlib.Path, name: str, env_overrides: dict, out: pathlib.Path):
    """Run the production entry point once and read what it left behind."""
    area = out / name / doc.stem
    area.mkdir(parents=True, exist_ok=True)
    # The document is copied in because publication writes beside the source.
    # Run candidates against the corpus in place and they overwrite each other's
    # artifacts, and the corpus stops being frozen.
    local = area / doc.name
    shutil.copy2(doc, local)

    env = {**os.environ, **env_overrides}
    t0 = time.monotonic()
    r = subprocess.run([sys.executable, str(HERE / "summ_cli.py"),
                        "--work-dir", str(area / "work"), str(local)],
                       capture_output=True, text=True, env=env)
    wall = round(time.monotonic() - t0, 1)

    run = area / "work" / "01" / "run"
    rec = {"candidate": name, "document": doc.name, "exit": r.returncode,
           "wall_seconds": wall, "published": r.returncode == 0,
           **read_calls(run), **read_ledger(run / "ledger"),
           **read_artifacts(run), **read_gate(doc, run)}
    if r.returncode != 0:
        rec["stderr_tail"] = (r.stderr or "").strip().splitlines()[-3:]
    (area / "profile.json").write_text(json.dumps(rec, indent=2, sort_keys=True))
    return rec


# (field, heading, width). Headings are written out rather than truncated from
# the field name: the point of this table is that one screen answers "which
# candidate", and "unpl" answering to "unplanned_parts" defeats that.
COLUMNS = [("candidate", "candidate", 10), ("document", "document", 14),
           ("exit", "exit", 4), ("units", "units", 5),
           ("words_per_unit", "w/unit", 6), ("quarantined", "quar", 4),
           ("unplanned_parts", "unpl", 4), ("detailed_words", "detail", 6),
           ("brief_words", "brief", 5), ("detailed_lists", "list", 4),
           ("detailed_first_person", "1stp", 4), ("calls", "call", 4),
           ("retried", "retry", 5), ("unusable", "unus", 4),
           ("seconds", "model_s", 7), ("gate", "gate", 4)]


def table(rows):
    head = "  ".join(h.rjust(w) for _, h, w in COLUMNS)
    yield head
    yield "-" * len(head)
    for r in rows:
        yield "  ".join(("-" if r.get(k) is None else str(r.get(k)))[:w].rjust(w)
                        for k, _, w in COLUMNS)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("corpus")
    ap.add_argument("--candidate", action="append", default=None,
                    help="restrict to this candidate (repeatable)")
    ap.add_argument("--out", default=None, help="where runs are kept "
                    "(default CORPUS_DIR/../qualify-runs)")
    ap.add_argument("--jobs", type=int, default=1,
                    help="run this many (candidate, document) pairs at once. "
                         "The runs are independent and almost entirely spent "
                         "waiting on a network call, so this is nearly linear. "
                         "Watch the `capacity` and `retried` columns: a "
                         "rate-limited benchmark is not a benchmark.")
    a = ap.parse_args()

    corpus = pathlib.Path(a.corpus).expanduser().resolve()
    spec = corpus / "candidates.json"
    if not spec.is_file():
        print(f"no candidates.json in {corpus}", file=sys.stderr)
        return 2
    # Keys beginning "_" are documentation (candidates.json carries a
    # "_comment"). Treating one as a candidate spawns a run whose env is
    # a list, and quietly inflates the denominator of every count.
    cands = {k: v for k, v in json.loads(spec.read_text()).items()
             if not k.startswith("_")}
    if a.candidate:
        missing = [c for c in a.candidate if c not in cands]
        if missing:
            print(f"unknown candidate(s): {', '.join(missing)}", file=sys.stderr)
            return 2
        cands = {k: v for k, v in cands.items() if k in a.candidate}

    docs = documents(corpus)
    if not docs:
        print(f"no documents in {corpus}", file=sys.stderr)
        return 2
    out = pathlib.Path(a.out).expanduser() if a.out else corpus.parent / "qualify-runs"
    out.mkdir(parents=True, exist_ok=True)

    # Round-robin by DOCUMENT, so concurrent work is spread across candidates
    # rather than hammering one credential with every document at once.
    jobs = [(n, e, d) for d in docs for n, e in cands.items()]
    rows, lock = [], threading.Lock()
    done = itertools.count(1)

    def work(job):
        name, env, doc = job
        t0 = time.monotonic()
        rec = one(doc, name, env, out)
        with lock:
            i = next(done)
            print(f"[qualify] {i}/{len(jobs)}  {name} / {doc.name}  "
                  f"exit {rec['exit']}  {time.monotonic() - t0:.0f}s", flush=True)
        return rec

    if a.jobs > 1:
        print(f"[qualify] {len(jobs)} runs, {a.jobs} at a time", flush=True)
        with cf.ThreadPoolExecutor(max_workers=a.jobs) as pool:
            rows = [r for r in pool.map(work, jobs) if r]
    else:
        rows = [work(j) for j in jobs]
    rows.sort(key=lambda r: (r["document"], r["candidate"]))

    print()
    for line in table(rows):
        print(line)
    (out / "summary.json").write_text(json.dumps(rows, indent=2, sort_keys=True))
    print(f"\nruns kept under {out}")

    # A candidate that failed to publish anywhere is the one fact worth an exit
    # code: it did not qualify, whatever the rest of the row says.
    # A benchmark degraded by its own concurrency is worthless, and the evidence
    # for that is already recorded per call.
    strained = sorted({r["candidate"] for r in rows
                       if (r.get("capacity") or 0) or (r.get("timeout") or 0)})
    if strained:
        print(f"\nCAPACITY OR TIMEOUT SEEN — these candidates may have been "
              f"degraded by running in parallel and should be re-run alone: "
              f"{', '.join(strained)}")
    failed = sorted({r["candidate"] for r in rows if r["exit"] != 0})
    if failed:
        print(f"did not publish on every document: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
