#!/usr/bin/env python3
"""summary.md -> summary.tts.txt

Deterministic, information-preserving. This is NOT a summarizer: it may change
surface form and pronunciation, never content. Every substantive sentence in the
Markdown survives into the output.

Replaces the model-driven abridgement in gemini-tts-condense.sh, which reduced
the text to ~half length and dropped supporting data -- a second, independently
lossy editorial pass over material the summarizer had already decided to keep.

Only acronyms/symbols the TTS engine actually mispronounces are rewritten, from
an explicit dictionary. Blanket transforms (every 4-digit token -> spelled-out
year) are wrong: 2001 may be a year, a model number, a statute, or a quantity.
"""
from __future__ import annotations
import argparse, hashlib, json, pathlib, re, sys

HERE = pathlib.Path(__file__).parent
DICT_PATH = HERE / "tts_dict.json"

DEFAULT_DICT = {
    "spell_out": ["FBI", "CIA", "MTA", "FOMC", "GDP", "QE", "ABS", "CMBS", "ABX",
                  "LIB", "OIS", "IMF", "ECB", "BIS", "CBO", "SEC", "ZLB",
                  "MMMF", "ABCP", "P6", "DDC", "WSP", "STV", "HNTB", "HDR"],
    "as_written": ["NATO", "NASA", "OPEC", "UNESCO", "TIPS"],
    "replace": {
        "&": " and ", "%": " percent", "±": " plus or minus ",
        "≈": " approximately ", "≤": " at most ", "≥": " at least ",
        "<": " less than ", ">": " greater than ", "×": " times ",
        "→": " leads to ", "—": ", ", "–": " to ", "…": ".",
        "bps": "basis points", "bp": "basis points",
        "e.g.": "for example", "i.e.": "that is", "etc.": "and so on",
        "cf.": "compare", "vs.": "versus", "Fig.": "Figure", "Eq.": "Equation",
    },
}

CITATION = re.compile(r"\((?:[A-Z][A-Za-z'’-]+(?:\s+(?:and|&|et al\.?)\s+[A-Z][A-Za-z'’-]+)*,?\s+\d{4}[a-z]?(?:[,;]\s*(?:p{1,2}\.\s*)?\d+(?:[-–]\d+)?)?)\)")
BRACKET_CITE = re.compile(r"\[\d+(?:[,–-]\s*\d+)*\]")
SEEALSO = re.compile(r"\((?:see|see also|cf\.)\s+[^)]{0,60}\)", re.I)
URL = re.compile(r"https?://\S+|www\.\S+")
MDLINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
EMPH = re.compile(r"(\*\*|__|\*|_|`)")
FOOTREF = re.compile(r"(?<=[a-z\)])\s*\[\^?\d{1,3}\]")


def load_dict():
    d = json.loads(json.dumps(DEFAULT_DICT))
    if DICT_PATH.exists():
        user = json.loads(DICT_PATH.read_text())
        d["spell_out"] = sorted(set(d["spell_out"]) | set(user.get("spell_out", [])))
        d["as_written"] = sorted(set(d["as_written"]) | set(user.get("as_written", [])))
        d["replace"].update(user.get("replace", {}))
    return d


def flatten_bullet(line: str) -> str:
    """A list item becomes a sentence. Order and content unchanged."""
    t = re.sub(r"^\s*(?:[-*+]|\d+[.)])\s+", "", line).strip()
    if not t:
        return ""
    if not t.endswith((".", "!", "?", ":", ";")):
        t += "."
    return t[0].upper() + t[1:] if t[:1].islower() else t


def heading_to_transition(text: str, level: int) -> str:
    t = text.strip().rstrip(".")
    if not t:
        return ""
    return f"{t}."


def normalize_inline(s: str, d: dict) -> str:
    s = MDLINK.sub(r"\1", s)
    s = URL.sub("", s)
    s = SEEALSO.sub("", s)
    s = CITATION.sub("", s)
    s = BRACKET_CITE.sub("", s)
    s = FOOTREF.sub("", s)
    s = EMPH.sub("", s)
    for k, v in d["replace"].items():
        # Word-boundary replacement. A bare substring swap turned "Subprime" into
        # "Subasis pointsrime" via the bp rule, and would corrupt any word
        # containing a mapped token.
        if k[:1].isalpha():
            s = re.sub(rf"(?<![A-Za-z]){re.escape(k)}(?![A-Za-z])", v, s)
        else:
            s = s.replace(k, v)
    # acronym spacing, only for listed initialisms, only when standing alone
    for a in d["spell_out"]:
        s = re.sub(rf"(?<![A-Za-z]){re.escape(a)}(?![A-Za-z])", " ".join(a), s)
    s = re.sub(r"\$\s?([\d.,]+)\s*(billion|million|trillion|thousand)?",
               lambda m: f"{m.group(1)} {m.group(2)+' ' if m.group(2) else ''}dollars", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = re.sub(r"\s+([,.;:])", r"\1", s)
    return s.strip()


def convert(md: str) -> str:
    d = load_dict()
    out, in_code = [], False
    for raw in md.split("\n"):
        if raw.strip().startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        line = raw.rstrip()
        if not line.strip():
            out.append("")
            continue
        if re.match(r"^\s*([-*_])\s*\1\s*\1", line):      # horizontal rule
            continue
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            out.append(heading_to_transition(normalize_inline(m.group(2), d), len(m.group(1))))
            continue
        if re.match(r"^\s*(?:[-*+]|\d+[.)])\s+", line):
            out.append(normalize_inline(flatten_bullet(line), d))
            continue
        if re.match(r"^\s*>", line):
            line = re.sub(r"^\s*>\s?", "", line)
        out.append(normalize_inline(line, d))

    text = "\n".join(out)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--expected-sha256")
    ap.add_argument("source")
    ap.add_argument("destination", nargs="?")
    args = ap.parse_args()
    src = pathlib.Path(args.source)
    dst = (pathlib.Path(args.destination) if args.destination
           else src.with_suffix(".tts.txt"))
    try:
        payload = src.read_bytes()
        if args.expected_sha256:
            if not re.fullmatch(r"[0-9a-f]{64}", args.expected_sha256):
                raise ValueError("invalid expected SHA-256")
            if hashlib.sha256(payload).hexdigest() != args.expected_sha256:
                raise ValueError("input does not match its accepted SHA-256")
        md = payload.decode("utf-8")
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        print(f"TTS input verification failed: {exc}", file=sys.stderr)
        return 5
    tts = convert(md)
    dst.write_text(tts, encoding="utf-8")

    # information-preservation check: sentence count must not collapse
    def sents(t):
        return [s for s in re.split(r"(?<=[.!?])\s+", re.sub(r"[#*_`>]", "", t)) if len(s.split()) > 3]
    a, b = len(sents(md)), len(sents(tts))
    mw, tw = len(md.split()), len(tts.split())
    print(f"{dst}")
    print(f"  sentences {a} -> {b}   words {mw} -> {tw} ({100*tw/max(1,mw):.0f}%)")
    if b < a * 0.92:
        print(f"  WARN: sentence count dropped {100*(1-b/max(1,a)):.0f}% — TTS step must not remove content")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
