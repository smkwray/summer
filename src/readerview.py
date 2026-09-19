#!/usr/bin/env python3
"""Build a reader-visible view of a source document, reversibly.

Replaces the old `strip_furniture`, which deleted footnotes, parenthetical
citations and whole trailing sections with regexes. That was verified benign on
one paper and is not safe in general: a footnote can carry the qualification the
body depends on, and "a section headed Notes" is not reliably apparatus.

Nothing is destroyed here. The original is kept byte-for-byte, every excluded
span is recorded with its reason, and the exclusion map travels with the
document so a later stage can be told what was hidden and object to it.

Only syntax that is unambiguously non-content is hidden:
  - a well-formed YAML front-matter block, at the start of the file only
  - well-formed HTML comments
  - PDF page furniture (bare page numbers, repeated running headers)
Title/subtitle/abstract are lifted OUT of front matter and kept as content.
Footnotes, citations, references and captions are PRESERVED and merely labelled,
so classifying them is the ledger's job, not a regex's.

usage: readerview.py SOURCE [OUTDIR]
"""
from __future__ import annotations
import json, pathlib, re, sys, hashlib, collections

YAML_FM = re.compile(r"\A(---\n)(.*?)(\n---\s*\n)", re.S)
HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)
# front-matter fields that are build configuration, never prose
CONFIG_KEYS = re.compile(
    r"^(format|header-includes|geometry|fontsize|linestretch|csl|bibliography|"
    r"filters|execute|jupyter|css|include-in-header|include-before|include-after|"
    r"pdf-engine|toc|number-sections|link-citations|reference-section-title|"
    r"mainfont|monofont|papersize|documentclass|classoption|lang)\b", re.I)
CONTENT_KEYS = re.compile(r"^(title|subtitle|abstract|description|summary)\b", re.I)


def _front_matter(text: str):
    """Return (visible_lead, excluded_spans). Only a valid block at position 0."""
    m = YAML_FM.match(text)
    if not m:
        return "", [], text
    body = m.group(2)
    kept, dropped = [], []
    key, buf = None, []

    def flush():
        if key is None:
            return
        val = " ".join(x.strip() for x in buf).strip().strip('"\'|>').strip()
        if CONTENT_KEYS.match(key) and val:
            kept.append((key.lower(), val))
        else:
            dropped.append(f"{key}: {val[:60]}")

    for line in body.split("\n"):
        if re.match(r"^\S[^:]*:", line):
            flush(); key, buf = line.split(":", 1)[0], [line.split(":", 1)[1]]
        elif key is not None:
            buf.append(line)
    flush()

    lead = ""
    for k, v in kept:
        if k in ("title", "subtitle"):
            lead += f"# {v}\n\n" if k == "title" else f"## {v}\n\n"
        else:
            lead += f"{v}\n\n"
    return lead, dropped, text[m.end():]


def build(src_text: str):
    exclusions = []
    lead, fm_dropped, rest = _front_matter(src_text)
    if fm_dropped:
        exclusions.append({"type": "front_matter_config",
                           "reason": "build configuration, not prose",
                           "items": fm_dropped})

    def _note_comment(m):
        exclusions.append({"type": "html_comment",
                           "reason": "never rendered to a reader",
                           "words": len(m.group(0).split()),
                           "preview": " ".join(m.group(0).split())[:120]})
        return "\n"
    rest = HTML_COMMENT.sub(_note_comment, rest)

    # PDF page furniture: a bare number alone on a line, and running headers that
    # repeat many times. Both are only removed when the evidence is overwhelming.
    lines = rest.split("\n")
    counts = collections.Counter(l.strip() for l in lines
                                 if 3 <= len(l.strip()) <= 90 and l.strip())
    repeated = {t for t, n in counts.items() if n >= 5 and not t.endswith((".", ":", "?"))}
    out = []
    for l in lines:
        s = l.strip()
        if re.fullmatch(r"\d{1,4}", s):
            exclusions.append({"type": "page_number", "reason": "bare page number",
                               "preview": s}); continue
        if s in repeated and len(s.split()) <= 12:
            exclusions.append({"type": "running_header",
                               "reason": f"line repeats {counts[s]}x; page furniture",
                               "preview": s[:90]}); continue
        out.append(l)

    visible = re.sub(r"\n{3,}", "\n\n", lead + "\n".join(out)).strip() + "\n"

    # Label, do NOT delete: these are decisions for the ledger, not for a regex.
    labels = []
    for pat, kind in [(r"^\s*\[\^[^\]]+\]:", "footnote_definition"),
                      (r"^#{1,3}\s*(References|Bibliography|Works Cited)\s*$", "references_heading")]:
        for m in re.finditer(pat, visible, re.M | re.I):
            labels.append({"type": kind, "line": visible[:m.start()].count("\n") + 1,
                           "preview": visible[m.start():m.start() + 80].strip()})
    return visible, exclusions, labels


def blocks(visible: str):
    out, n = [], 0
    for para in re.split(r"\n\s*\n", visible):
        p = para.strip()
        if not p:
            continue
        n += 1
        out.append({"id": f"P{n:04d}", "words": len(p.split()), "text": p})
    return out


def main():
    src = pathlib.Path(sys.argv[1])
    outdir = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else src.parent / "_readerview"
    outdir.mkdir(parents=True, exist_ok=True)
    raw = src.read_text(errors="replace")
    visible, exclusions, labels = build(raw)
    bl = blocks(visible)

    (outdir / "source.raw").write_text(raw)
    (outdir / "source.visible.md").write_text(visible)
    (outdir / "source-map.json").write_text(json.dumps(
        {"source_sha256": hashlib.sha256(raw.encode("utf8", "replace")).hexdigest(),
         "raw_words": len(raw.split()), "visible_words": len(visible.split()),
         "blocks": bl, "labels": labels}, indent=2))
    (outdir / "exclusions.json").write_text(json.dumps(exclusions, indent=2))

    hidden = len(raw.split()) - len(visible.split())
    print(f"raw {len(raw.split())}w -> visible {len(visible.split())}w "
          f"({hidden} hidden, {100*hidden/max(1,len(raw.split())):.0f}%), {len(bl)} blocks")
    for e in exclusions[:6]:
        print(f"  hidden: {e['type']} — {e.get('preview', e.get('items', ''))!s:.90}")
    if labels:
        print(f"  labelled but KEPT: {collections.Counter(l['type'] for l in labels)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
