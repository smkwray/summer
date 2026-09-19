"""Device-local last used naming affix (prefix vs suffix and custom strings).

The UI may remember the last affix configuration (switch between prefix and suffix,
and custom affix tags per mode). Direct CLI runs never inherit it unless configured.
"""
from __future__ import annotations

import json
import pathlib

import runtime

MAX_BYTES = 4096

DEFAULT_AFFIXES: dict[str, tuple[str, str]] = {
    "summarize": ("summary", "brief"),
    "quick": ("summary", "brief"),
    "text_prep": ("clean", ""),
    "tts": ("tts", ""),
}


def last_path() -> pathlib.Path:
    return runtime.app_dir() / "last-affix.json"


def load(mode_key: str | None = None) -> dict:
    """Load the last affix configuration for a mode."""
    target_mode = mode_key or "summarize"
    def_primary, def_secondary = DEFAULT_AFFIXES.get(target_mode, ("summary", "brief"))

    path = last_path()
    if not path.exists():
        return {
            "kind": "prefix",
            "primary": def_primary,
            "secondary": def_secondary,
            "text": def_primary,
            "modes": {},
        }
    try:
        raw = path.read_bytes()
        if len(raw) > MAX_BYTES:
            return {
                "kind": "prefix",
                "primary": def_primary,
                "secondary": def_secondary,
                "text": def_primary,
                "modes": {},
            }
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict):
            return {
                "kind": "prefix",
                "primary": def_primary,
                "secondary": def_secondary,
                "text": def_primary,
                "modes": {},
            }
        kind = value.get("kind", "prefix")
        if kind not in ("prefix", "suffix"):
            kind = "prefix"
        modes = value.get("modes", {})
        if not isinstance(modes, dict):
            modes = {}

        if target_mode in modes and isinstance(modes[target_mode], dict):
            m = modes[target_mode]
            primary = str(m.get("primary", "")).strip() or def_primary
            secondary = str(m.get("secondary", "")).strip()
            if not secondary and def_secondary and "secondary" not in m:
                secondary = def_secondary
        elif mode_key is None and ("primary" in value or "text" in value):
            primary = str(value.get("primary", value.get("text", ""))).strip() or def_primary
            secondary = str(value.get("secondary", "")).strip()
            if not secondary and def_secondary and "secondary" not in value:
                secondary = def_secondary
        else:
            primary = def_primary
            secondary = def_secondary

        return {
            "kind": kind,
            "primary": primary,
            "secondary": secondary,
            "text": primary,
            "modes": modes,
        }
    except Exception:
        return {
            "kind": "prefix",
            "primary": def_primary,
            "secondary": def_secondary,
            "text": def_primary,
            "modes": {},
        }


def save(
    kind: str,
    primary: str = "",
    secondary: str = "",
    mode_key: str | None = None,
    **kwargs,
) -> dict:
    """Atomically remember the last used affix configuration."""
    if "text" in kwargs and not primary:
        primary = kwargs["text"]
    kind = "suffix" if kind == "suffix" else "prefix"
    primary = str(primary).strip()
    secondary = str(secondary).strip()

    data: dict = {}
    path = last_path()
    if path.exists():
        try:
            raw = path.read_bytes()
            if len(raw) <= MAX_BYTES:
                loaded = json.loads(raw.decode("utf-8"))
                if isinstance(loaded, dict):
                    data = loaded
        except Exception:
            pass

    modes = data.get("modes", {})
    if not isinstance(modes, dict):
        modes = {}

    target_mode = mode_key or "summarize"
    modes[target_mode] = {"primary": primary, "secondary": secondary}

    payload = {
        "kind": kind,
        "primary": primary,
        "secondary": secondary,
        "text": primary,
        "modes": modes,
    }
    runtime.atomic_json(path, payload)
    return {
        "kind": kind,
        "primary": primary,
        "secondary": secondary,
        "text": primary,
        "mode_key": target_mode,
        "modes": modes,
    }
