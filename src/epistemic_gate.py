#!/usr/bin/env python3
"""Spec-driven epistemic-fidelity gate.

usage: epistemic_gate.py SPEC.json SUMMARY.md [SUMMARY.md ...]

A spec names, for one benchmark document, the overclaims that must not appear and
the caveat concepts that must appear. Each concept lists alternative phrasings,
because the point is whether the caveat survived, not whether it survived in one
particular wording -- the delivered v1 gate matched fixed strings and rejected
correct outputs that said "unclear if" instead of "unclear whether".

Specs are frozen per document before candidates are compared, so the yardstick
cannot be tuned after seeing an output.
"""
from __future__ import annotations
import json, pathlib, re, sys


# Contractions are expanded before matching. A spec says what a caveat MEANS,
# and "isn't a causal estimate" is the same caveat as "is not a causal
# estimate" -- but a pattern containing the word `not` matches only the second.
# Contracted caveats must match their expanded equivalents. Typographic
# apostrophes are folded to ASCII for the same reason.
CONTRACTIONS = [
    (r"\bcan[’']t\b", "can not"), (r"\bwon[’']t\b", "will not"),
    (r"\bshan[’']t\b", "shall not"), (r"n[’']t\b", " not"),
]


def normalize(text: str) -> str:
    t = text.replace("\u2019", "'").replace("\u2018", "'").lower()
    for pat, rep in CONTRACTIONS:
        t = re.sub(pat, rep, t)
    return re.sub(r"\s+", " ", t)


def check(spec: dict, path: pathlib.Path):
    t = normalize(path.read_text(errors="replace"))
    fails, notes = [], []
    for pat in spec.get("overclaim", []):
        m = re.search(pat, t)
        if m:
            fails.append(f"overclaim: {m.group(0)!r}")
    for concept, pats in spec.get("concepts", {}).items():
        hit = next((m.group(0) for p in pats for m in [re.search(p, t)] if m), None)
        if hit:
            notes.append(f"  ok   {concept}  <- {hit!r}")
        else:
            fails.append(f"missing: {concept}")
    return fails, notes


def main():
    spec = json.loads(pathlib.Path(sys.argv[1]).read_text())
    print(f"# gate: {spec.get('doc','(unnamed)')}")
    worst = 0
    for p in map(pathlib.Path, sys.argv[2:]):
        fails, notes = check(spec, p)
        label = f"{p.parent.name}/{p.name}"
        print(f"\n===== {label} =====")
        for n in notes:
            print(n)
        if fails:
            print("FAIL")
            for f in fails:
                print(f"  - {f}")
            worst = 1
        else:
            print("PASS")
    sys.exit(worst)


if __name__ == "__main__":
    main()
