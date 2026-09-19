"""Device-local last explicit output destination.

The UI may remember the last folder or source-relative operator the user
actually used. Direct CLI runs never inherit it.
"""
from __future__ import annotations

import json
import pathlib

import runtime

MAX_BYTES = 4096


def last_path() -> pathlib.Path:
    return runtime.app_dir() / "last-output.json"


def load() -> str:
    """Load the last explicit output destination, or empty if none."""
    path = last_path()
    if not path.exists():
        return ""
    try:
        raw = path.read_bytes()
        if len(raw) > MAX_BYTES:
            raise ValueError(
                f"last output folder is too large ({len(raw)} bytes; "
                f"maximum {MAX_BYTES})")
        value = json.loads(raw.decode("utf-8"))
        directory = value.get("directory") if isinstance(value, dict) else None
        if not isinstance(directory, str):
            raise ValueError("directory must be a string")
        return _clean(directory)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise runtime.ConfigError(f"invalid {path}: {exc}") from exc


def save(value: str) -> str:
    """Atomically remember one explicit output destination."""
    directory = _clean(value)
    runtime.atomic_json(last_path(), {"directory": directory})
    return directory


def button_label(value: str, max_len: int = 18) -> str:
    """Show the folder name, or the relative operator as typed."""
    text = _clean(value)
    path = pathlib.Path(text)
    if path.is_absolute() or text.startswith(("/", "\\")):
        name = path.name or text
    else:
        name = text
    if len(name) > max_len:
        return name[: max_len - 1] + "\u2026"
    return name


def _clean(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("output directory must be text")
    if "\x00" in value:
        raise ValueError("output directory contains a NUL byte")
    text = value.strip()
    if not text:
        raise ValueError("output directory is empty")
    encoded = text.encode("utf-8")
    if len(encoded) > MAX_BYTES:
        raise ValueError(
            f"output directory is too large ({len(encoded)} bytes; "
            f"maximum {MAX_BYTES})")
    return text
