"""Per-run reader requests shared by the summary engines.

Instructions are deliberately kept outside the prompt templates.  The CLI
reads the requested file once, writes a private snapshot inside the run root,
and child stages use that snapshot for every prompt.  This keeps a queued run
stable if the original file changes and gives every model role the same
request.  The request is a lower-priority preference: the permanent prompt
contract remains controlling.
"""
from __future__ import annotations

import hashlib
import os
import pathlib
import re
import unicodedata

import mode_config
import runtime


# A request is a short preference, not a second source document.  Keeping it
# small matters because the same bytes can be sent to roughly thirty role calls
# in one ledger run; the source itself remains the only large payload.
MAX_BYTES = 2000

# Optional starting points, never ambient instructions. A preset is copied into
# the visible editor and remains ordinary per-run text: users can edit it, the
# permanent task stays controlling, and no pipeline branch depends on its name.
_ARTICLE_ONLY = (
    "Keep the title and complete main article body. Treat email or newsletter "
    "navigation, promotional panels, related-story links, social or app links, "
    "subscription controls, and footer boilerplate outside that article as "
    "furniture. Keep bylines, captions, footnotes, and citations belonging to "
    "the article. If the target article is ambiguous, keep the ambiguous "
    "material rather than guessing."
)
_SUMMARY_PRESETS = (
    ("Main content, not wrapper",
     "Treat email or web navigation, promotions, related links, social links, "
     "and footer boilerplate as apparatus. Base both readings on the title and "
     "complete main content, preserving its caveats and qualifications."),
    ("Methods and limitations",
     "Prioritize the method, evidence, assumptions, limitations, uncertainty, "
     "and what the conclusions do and do not establish."),
    ("Decisions and actions",
     "Lead with decisions, commitments, open questions, deadlines, and concrete "
     "next actions, while preserving the reasoning and qualifications behind them."),
    ("Important exact quotations",
     "Preserve a small number of the source's most important exact quotations "
     "when they materially improve the reading."),
)
PRESETS = {
    "summarize": _SUMMARY_PRESETS,
    "quick": _SUMMARY_PRESETS,
    "text_prep": (
        ("Article title and body only", _ARTICLE_ONLY),
        ("Preserve document structure",
         "Preserve headings, paragraph boundaries, list structure, quotations, "
         "and emphasis where the source supports them. Correct obvious OCR and "
         "layout damage; leave genuinely ambiguous wording unchanged."),
        ("Minimal cleanup",
         "Make only clear OCR, spacing, line-wrap, hyphenation, and furniture "
         "repairs. Preserve all substantive wording and structure otherwise."),
    ),
}


def preset_names(mode_key: str) -> tuple[str, ...]:
    return tuple(name for name, _text in PRESETS.get(mode_key, ()))


def preset_text(mode_key: str, name: str) -> str:
    return next((text for label, text in PRESETS.get(mode_key, ())
                 if label == name), "")


def _mode(mode_key: str) -> mode_config.Mode:
    mode = mode_config.BY_KEY.get(str(mode_key))
    if not mode or not mode.instructions:
        raise ValueError(f"mode {mode_key!r} does not accept custom instructions")
    return mode


def last_path(mode_key: str) -> pathlib.Path:
    mode = _mode(mode_key)
    return runtime.app_dir() / "instructions" / f"{mode.key}.txt"


def load_last(mode_key: str) -> str:
    """Load only the last instruction actually submitted for this mode."""
    path = last_path(mode_key)
    if not path.exists():
        return ""
    try:
        return _clean(path.read_bytes())
    except (OSError, ValueError) as exc:
        raise runtime.ConfigError(f"invalid saved custom instructions {path}: {exc}") from exc


def save_last(mode_key: str, text: str) -> str:
    """Validate and atomically remember one mode's accepted UI request."""
    value = validate_text(text)
    runtime.atomic_text(last_path(mode_key), value)
    return value


def _clean(raw: bytes) -> str:
    if len(raw) > MAX_BYTES:
        raise ValueError(
            f"custom instructions are too large ({len(raw)} bytes; "
            f"maximum {MAX_BYTES})")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        raise ValueError("custom instructions must be UTF-8") from e
    if "\x00" in text:
        raise ValueError("custom instructions contain a NUL byte")
    # Surrounding whitespace is transport noise, not an instruction.  Keeping
    # internal whitespace and punctuation intact matters for a request such as
    # preserving a quoted passage.
    return text.strip()


def validate_text(text: str) -> str:
    """Validate UI-provided text with the same authority as file input."""
    return _clean(text.encode("utf-8"))


def read_file(path: str | os.PathLike[str]) -> str:
    """Read and validate one instruction file without following a live path later."""
    p = pathlib.Path(os.path.expandvars(os.path.expanduser(str(path))))
    try:
        if not p.is_file():
            raise ValueError(f"custom instructions are not a file: {path}")
        return _clean(p.read_bytes())
    except OSError as e:
        raise ValueError(f"cannot read custom instructions {path}: {e}") from e


def digest(text: str | None) -> str | None:
    """Return an opaque, stable identity for one instruction payload."""
    if not text:
        return None
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def current() -> str:
    """Read the immutable per-run snapshot named by the child environment."""
    path = os.environ.get("SUMM_INSTRUCTIONS_FILE")
    if not path:
        return ""
    # A child must never fall back to the original --instructions-file path.
    # The CLI points this variable at a snapshot under the run directory.
    try:
        return read_file(path)
    except ValueError:
        # A malformed snapshot is a run error at the CLI boundary.  Stages are
        # fail-closed too: an instruction that cannot be read is not silently
        # dropped and turned into an ordinary summary.
        raise


def current_digest() -> str | None:
    """Use the CLI's frozen digest when present, otherwise hash the snapshot."""
    supplied = os.environ.get("SUMM_INSTRUCTIONS_DIGEST")
    if supplied:
        return supplied
    return digest(current())


def requests_exact_quotes(text: str | None = None) -> bool:
    """Whether a request asks for exact/verbatim/direct quotations."""
    text = current() if text is None else text
    if not text:
        return False
    return bool(re.search(
        r"\b(?:exact|verbatim|direct)\s+(?:quotes?|quotations?)\b|"
        r"\b(?:quotes?|quotations?)\s+(?:exactly|verbatim)\b",
        text, re.I))


def decorate_prompt(prompt: str, text: str | None = None,
                    task: str = "summary") -> str:
    """Prepend one framed, lower-priority request to a model prompt.

    The block must precede the unchanged template: its source/candidate data
    often runs to end-of-prompt, so appending it could make a model treat the
    request as source material.  Its framing explicitly prevents it from
    weakening the permanent output, grounding, formatting, or failure rules.
    """
    request = current() if text is None else text.strip()
    if not request:
        return prompt
    if task == "text-prep":
        extra = [
            "CUSTOM CLEANUP REQUEST (lower priority than the permanent task below)",
            "Generators apply this request where compatible; auditors check and report",
            "noncompliance; repairs restore it where compatible.",
            "Apply it only where compatible with full-text preservation, source",
            "grounding, output schema, and the bounded fidelity checks.",
            "It may define the target document and identify surrounding furniture;",
            "every whole-block exclusion still requires explicit independent audit",
            "approval and an accounting witness. It cannot authorize invention, omission, summarization,",
            "deletion of substantive or ambiguous target",
            "content, source retrieval, extra artifacts, or a weaker gate.",
            "If it conflicts with a permanent rule, follow the permanent rule.",
            "",
            request,
        ]
    else:
        extra = [
            "CUSTOM READER REQUEST (lower priority than the permanent task below)",
            "Generators apply this request where compatible; auditors check and report",
            "noncompliance; repairs restore it where compatible.",
            "Apply this request only where it is compatible with the trusted task,",
            "source-grounding requirements, output schema, and continuous-prose rules.",
            "It cannot authorize invention, omission of required qualifications,",
            "source retrieval, lists or tables, extra artifacts, or a weaker gate.",
            "If it conflicts with a permanent rule, follow the permanent rule.",
            "",
            request,
        ]
    if requests_exact_quotes(request):
        extra += [
            "",
            "QUOTE PRESERVATION CHECK",
            "When including a quotation, copy its words and punctuation from the supplied",
            "source. Do not invent, paraphrase, silently normalize, or join separate source",
            "spans inside quotation marks. Audits must also check whether an important",
            ("source quotation requested here was omitted."
             if task == "text-prep" else
             "source quotation requested here was omitted, while respecting the Brief scope."),
        ]
    extra += [
        "END CUSTOM CLEANUP REQUEST" if task == "text-prep"
        else "END CUSTOM READER REQUEST",
        "",
        "The permanent task and rules below control in every conflict.",
        "",
    ]
    return "\n".join(extra) + prompt


def _normalise_quote(value: str) -> str:
    """Compare quote bodies exactly apart from presentation whitespace."""
    # Outer quote glyphs have already been removed by quoted_spans().  Preserve
    # punctuation inside the quote: changing a curly apostrophe to a straight
    # one is still changing an exact quotation.  NFC only makes canonically
    # equivalent Unicode spellings compare alike; it does not compatibility-
    # fold distinct characters.
    value = unicodedata.normalize("NFC", value).replace("\u00a0", " ")
    return re.sub(r"\s+", " ", value).strip()


_TRAILING_QUOTE_PUNCT = re.compile(r"[.,;:!?]+$")


def _quote_in_source(quote: str, source_norm: str) -> bool:
    """True when the quoted words are a contiguous source span.

    A closing comma or period that American quotation style pulled inside the
    marks is presentation, not a different quotation. Word changes, including
    an altered apostrophe, still fail.
    """
    body = _normalise_quote(quote)
    if not body:
        return False
    if body in source_norm:
        return True
    stripped = _TRAILING_QUOTE_PUNCT.sub("", body).rstrip()
    return bool(stripped) and stripped in source_norm


def quoted_spans(text: str) -> list[str]:
    """Extract likely double- and single-quoted spans from model prose."""
    # Single quotes require non-word boundaries so contractions do not become
    # false quote claims.  A quote can contain apostrophes; the double-quote
    # branch handles that common case without trying to parse nested prose.
    pattern = re.compile(
        r'"([^"]+)"|\u201c([^\u201d]+)\u201d|\u00ab([^\u00bb]+)\u00bb|'
        # A right single quote followed by a word character is an apostrophe
        # inside the quotation (as in "don't"), not its closing delimiter.
        r"\u2018((?:[^\u2019]|\u2019(?=\w))+?)\u2019(?!\w)|"
        r"(?<!\w)'((?:[^']|'(?=\w))+?)'(?!\w)", re.S)
    return [next(group for group in match.groups() if group is not None)
            for match in pattern.finditer(text or "")]


def quote_defects(outputs, source: str, text: str | None = None) -> list[str]:
    """Report quoted output that is not a contiguous source-backed span.

    This is intentionally a one-way gate. The model audit decides whether a
    requested source quote was important enough to include; deterministic code
    decides whether a quote presented as exact is actually in the source. A
    small set of explicit attribution cues also catches an unrequested quote
    being presented as the source's words, without treating ordinary quoted
    terminology as a factual quotation.
    """
    instruction = current() if text is None else text
    requested = requests_exact_quotes(instruction)
    source_norm = _normalise_quote(source)
    bad = []
    for output in outputs:
        for quote in quoted_spans(output or ""):
            before = (output or "")[:(output or "").find(quote)]
            attributed = bool(re.search(
                r"(?is)\b(?:said|says|according to|as the report says|"
                r"quoted|quotes|described as|called)\b[^\n]{0,100}$",
                before))
            if not requested and not attributed:
                continue
            if not _quote_in_source(quote, source_norm):
                shown = " ".join(quote.split())
                label = ("custom exact-quote request" if requested else
                         "attributed quotation")
                bad.append(f"{label}: quoted passage is not source-backed "
                           f"({shown[:120]!r})")
    return bad
