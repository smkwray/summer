#!/usr/bin/env python3
"""Stage the benchmark artifacts for blind judging.

Candidates are presented as A-E, SHUFFLED PER DOCUMENT so a judge cannot carry
a guess from one document to the next, with no model named anywhere in the
package. The mapping is written to
do/bench/keys.json, which is NOT part of the package.

The shuffle is seeded from the document name alone, so re-running this
reproduces the same assignment. Nothing here reads a result.
"""
import hashlib, json, pathlib, random, shutil, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
RUNS = ROOT / "data" / "bench3-runs"
CORPUS = ROOT / "data" / "bench3"
OUT = ROOT / "data" / "bench3-blind"
LETTERS = "ABCDE"


def reading(path: pathlib.Path) -> str:
    """The reading itself, without the pipeline's own coverage note.

    The note is a leading blockquote that names the artifact's word count as a
    percentage of the source ("the detailed reading is 92% of the source"). The
    frozen criteria withhold word count from the judge on purpose, so shipping
    the note would hand over precisely what they exclude -- and it would also
    substitute the pipeline's own audit for the independent judgement being
    asked for. The published artifacts keep their notes; only the judging copy
    is stripped.
    """
    lines = path.read_text(errors="replace").splitlines()
    i = 0
    while i < len(lines) and (lines[i].startswith(">") or not lines[i].strip()):
        i += 1
    return "\n".join(lines[i:]).strip() + "\n"


def main():
    summary = json.loads((RUNS / "summary.json").read_text())
    docs = sorted({r["document"] for r in summary})
    cands = sorted({r["candidate"] for r in summary})
    if len(cands) > len(LETTERS):
        sys.exit(f"{len(cands)} candidates, only {len(LETTERS)} letters")

    if OUT.exists():
        shutil.rmtree(OUT)
    keys, missing = {}, []
    for doc in docs:
        stem = pathlib.Path(doc).stem
        # Seeded from the document name: reproducible, and independent of the
        # order candidates happen to appear in summary.json.
        rng = random.Random(hashlib.sha256(stem.encode()).hexdigest())
        order = cands[:]
        rng.shuffle(order)
        keys[stem] = {}
        d = OUT / stem
        d.mkdir(parents=True)
        shutil.copy2(CORPUS / doc, d / "SOURCE.md")
        for letter, cand in zip(LETTERS, order):
            keys[stem][letter] = cand
            area = RUNS / cand / stem
            det = area / f"{stem}.summary.md"
            bri = area / f"{stem}.brief.md"
            if not det.is_file() or not bri.is_file():
                missing.append(f"{cand}/{stem}")
                continue
            (d / f"{letter}.detailed.md").write_text(reading(det))
            (d / f"{letter}.brief.md").write_text(reading(bri))

    (ROOT / "do" / "bench" / "keys.json").write_text(
        json.dumps(keys, indent=2, sort_keys=True) + "\n")

    leaked = []
    for f in OUT.rglob("*.md"):
        if f.name == "SOURCE.md":
            continue
        low = f.read_text(errors="replace").lower()
        for name in ("grok", "gemini", "flash", "luna", "sonnet", "claude",
                     "opencode", "antigravity", "openai", "anthropic", "xai"):
            if name in low:
                leaked.append(f"{f.relative_to(OUT)}: {name}")
    print(f"documents: {len(docs)}  candidates: {len(cands)}")
    print(f"staged: {OUT}")
    print(f"key:    {ROOT / 'do' / 'bench' / 'keys.json'}  (NOT sent)")
    if missing:
        print(f"MISSING ARTIFACTS ({len(missing)}): {', '.join(missing)}")
    if leaked:
        print(f"NAME LEAK IN {len(leaked)} FILE(S): {leaked[:10]}")
    return 1 if (missing or leaked) else 0


if __name__ == "__main__":
    sys.exit(main())
