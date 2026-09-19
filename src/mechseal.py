#!/usr/bin/env python3
"""Deterministic structural seal for a semantically verified ledger.

The seal proves source linkage, complete block accounting, references, and
schema consistency. It does not waive semantic audit: the ledger producer must
complete that gate before this structural seal can make publication eligible.

usage: mechseal.py LEDGER_DIR READERVIEW_DIR
"""
from __future__ import annotations
import json, pathlib, re, sys

HERE = pathlib.Path(__file__).parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import ledger_schema

ALLOWED_DISP = ledger_schema.DISPOSITIONS
LEAK = re.compile(r"^---\s*$|<!--|^\s*\[\^|^\s*(title|author|format|bibliography):",
                  re.M | re.I)


def norm(t):
    return re.sub(r"\s+", " ", (t or "")).strip().lower()


def check(ldir: pathlib.Path, rv: pathlib.Path):
    fail = []
    ledger = json.loads((ldir / "ledger.json").read_text())
    smap = json.loads((rv / "source-map.json").read_text())
    units = ledger.get("units") or []
    blocks = {b["id"]: b["text"] for b in smap["blocks"]}

    if ledger.get("source_sha256") != smap.get("source_sha256"):
        fail.append("source hash does not match the reader view")
    if not units:
        fail.append("ledger has no units")

    ids = [u.get("unit_id") for u in units]
    if len(ids) != len(set(ids)):
        fail.append(f"duplicate unit ids ({len(ids)-len(set(ids))})")

    # COMPLETE SOURCE ACCOUNTING. The build prompt has always required every
    # visible block to be represented by a unit or given one narrow disposition,
    # but nothing enforced it: the seal only checked that referenced ids exist.
    # A planner whose parts fail silently drops source and then looks
    # attractively compressed -- one model left 85 of 266 blocks (1,185 visible
    # words) unaccounted and still sealed and published a summary.
    # Apparatus is not exempt: it passes by being dispositioned, not by vanishing.
    represented = {sid for u in units for sid in (u.get("source_ids") or [])}
    seen_disp, multi_disp = set(), set()
    for d in (ledger.get("dispositions") or []):
        for sid in (d.get("source_ids") or []):
            if sid in seen_disp:
                multi_disp.add(sid)
            seen_disp.add(sid)
    missing = set(blocks) - represented - seen_disp
    conflicting = represented & seen_disp
    if missing:
        lost = sum(len((blocks[m] or "").split()) for m in missing)
        fail.append(f"unaccounted source blocks ({len(missing)}, {lost} visible "
                    f"words): {sorted(missing)[:8]}")
    if conflicting:
        fail.append(f"source blocks both represented and dispositioned "
                    f"({len(conflicting)}): {sorted(conflicting)[:8]}")
    if multi_disp:
        fail.append(f"source blocks with more than one disposition "
                    f"({len(multi_disp)}): {sorted(multi_disp)[:8]}")

    known = set(ids)
    dangling_dep, bad_src, empty_cap = [], [], []
    for u in units:
        for d in u.get("dependencies") or []:
            if d not in known:
                dangling_dep.append(d)
        srcs = u.get("source_ids") or []
        if not srcs:
            bad_src.append(u.get("unit_id"))
        for sid in srcs:
            if sid not in blocks:
                bad_src.append(f"{u.get('unit_id')}->{sid}")
        # exact_source_anchor was removed from the schema: it was generated for
        # every unit, only ever produced a warning here, and cost output tokens
        # on a device where output throughput is the binding constraint. Units
        # are still bound to the source by source_ids, which IS checked.
        for depth, key in (("detailed", "detailed_capsule"), ("brief", "brief_capsule")):
            if u.get(f"{depth}_disposition") in ("required", "optional") \
               and not (u.get(key) or "").strip():
                empty_cap.append(f"{u.get('unit_id')}/{depth}")
        for key in ("detailed_capsule", "brief_capsule"):
            if LEAK.search(u.get(key) or ""):
                fail.append(f"{u.get('unit_id')}: apparatus leaked into {key}")

    if dangling_dep:
        fail.append(f"dangling dependencies ({len(dangling_dep)}): {sorted(set(dangling_dep))[:6]}")
    if bad_src:
        fail.append(f"unknown/absent source ids ({len(bad_src)}): {bad_src[:6]}")
    # An anchor is corroboration, not the published content: the capsule is what
    # the reader gets and source_ids still bind it. A paraphrased anchor is worth
    # recording, not worth withholding the whole document for.
    warn = []
    if empty_cap:
        fail.append(f"non-omitted units with empty capsule ({len(empty_cap)}): {empty_cap[:6]}")

    # dependency cycles
    graph = {u["unit_id"]: [d for d in (u.get("dependencies") or []) if d in known]
             for u in units if u.get("unit_id")}
    state = {}

    def cyclic(n):
        if state.get(n) == 1:
            return True
        if state.get(n) == 2:
            return False
        state[n] = 1
        for m in graph.get(n, []):
            if cyclic(m):
                return True
        state[n] = 2
        return False
    if any(cyclic(n) for n in graph):
        fail.append("dependency cycle")

    for d in ledger.get("dispositions") or []:
        if d.get("disposition") not in ALLOWED_DISP:
            fail.append(f"invalid disposition {d.get('disposition')!r}")
        for r in d.get("represented_by") or []:
            if r not in known:
                fail.append(f"disposition references unknown unit {r}")
                break

    viable = {}
    for depth, key in (("detailed", "detailed_capsule"), ("brief", "brief_capsule")):
        n = sum(1 for u in units
                if u.get(f"{depth}_disposition") in ("required", "optional")
                and (u.get(key) or "").strip())
        viable[depth] = n
        if n == 0:
            fail.append(f"no viable units at depth {depth}")
    return fail, {"units": len(units), "viable": viable, "warnings": warn}


def main():
    ldir, rv = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    try:
        fail, info = check(ldir, rv)
    except Exception as e:
        print(f"[mechseal] unreadable ledger: {e}")
        return 2
    (ldir / "mechseal.json").write_text(json.dumps(
        {"passed": not fail, "failures": fail, **info}, indent=2))
    for w in info.get("warnings") or []:
        print(f"[mechseal] warning: {w}")
    if fail:
        print(f"[mechseal] FAILED ({len(fail)}):")
        for f in fail[:10]:
            print(f"   - {f}")
        return 2
    (ldir / "MECHSEAL").write_text("1")
    print(f"[mechseal] passed — {info['units']} units, viable "
          f"detailed={info['viable']['detailed']} brief={info['viable']['brief']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
