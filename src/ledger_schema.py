"""Shared deterministic schema facts for fresh and resumed ledgers."""
from __future__ import annotations


DISPOSITIONS = frozenset({
    "exact_repetition",
    "apparatus",
    "incidental_example",
    "source_only_detail",
})


def canonical_disposition(value) -> str:
    """Expand one unambiguous substantial prefix; leave defects to fail."""
    value = (value or "").strip()
    if value in DISPOSITIONS:
        return value
    matches = [item for item in DISPOSITIONS
               if len(value) >= 6 and item.startswith(value)]
    return matches[0] if len(matches) == 1 else value
