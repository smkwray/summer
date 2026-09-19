"""One small authority for Summer's product modes and active model roles."""
from __future__ import annotations

from dataclasses import dataclass


ROLES = ("plan", "write", "audit", "repair")


@dataclass(frozen=True)
class Mode:
    key: str
    label: str
    flag: str | None
    roles: tuple[str, ...]
    suffixes: tuple[str, ...]
    progress_route: str
    instructions: bool
    success_text: str

    @property
    def flags(self) -> tuple[str, ...]:
        return (self.flag,) if self.flag else ()


@dataclass(frozen=True)
class CorpusPolicy:
    min_documents: int
    max_documents: int
    min_visible_words: int
    max_visible_words: int
    inventory_window_words: int
    max_inventory_windows: int


# Corpus workload bounds belong with the product mode, not a model or private
# gateway. The inventory window is larger than a standalone summary planning
# part because it produces source-accounting capsules rather than final prose.
CORPUS = CorpusPolicy(
    min_documents=2,
    max_documents=32,
    min_visible_words=400,
    max_visible_words=100_000,
    inventory_window_words=4_000,
    max_inventory_windows=64,
)


MODES = (
    Mode("summarize", "Summarize — sealed Detailed and Brief", None, ROLES,
         (".summary.md", ".brief.md"), "summarize", True,
         "Done — Detailed and Brief written."),
    Mode("quick", "Quick — faster Detailed and Brief", "--quick",
         ("write", "audit", "repair"), (".summary.md", ".brief.md"),
         "quick", True, "Done — Detailed and Brief written."),
    Mode("text_prep", "Clean text — repair OCR and formatting", "--text-prep",
         ("write", "audit", "repair"), (".clean.md",), "text_prep", True,
         "Done — Cleaned text written."),
    Mode("tts", "Prepare for read-aloud", "--tts",
         ("write", "audit", "repair"),
         (".tts.txt",), "tts", False, "Done — Read-aloud text written."),
)

BY_KEY = {mode.key: mode for mode in MODES}
BY_LABEL = {mode.label: mode for mode in MODES}
BY_FLAG = {mode.flag: mode for mode in MODES if mode.flag}


def selected(*, quick=False, text_prep=False, tts=False) -> Mode:
    if tts:
        return BY_KEY["tts"]
    if text_prep:
        return BY_KEY["text_prep"]
    if quick:
        return BY_KEY["quick"]
    return BY_KEY["summarize"]
