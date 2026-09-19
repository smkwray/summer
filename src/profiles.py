"""Headless profile storage and projection into Summer's existing env API."""
from __future__ import annotations

import json
import os
import pathlib

import mode_config
import model_config
import runtime

ROLES = mode_config.ROLES
DEFAULT_PROFILE = "balanced"


def model_defaults() -> dict:
    try:
        return model_config.defaults()
    except Exception:
        return {}


def _pick(value):
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return "", "", ""
    return str(value[0]), str(value[1]), str(value[2]) if len(value) > 2 else ""


def _option(harness: str, model: str, value: str = "", role: str = "",
            roster_value: dict | None = None) -> dict:
    setting = model_config.model_setting(harness, model, roster_value)
    name = setting.get("option")
    if not name or name == "model":
        return {}
    chosen = value or model_config.setting_default(
        harness, model, role, roster_value)
    return {name: chosen} if isinstance(chosen, str) and chosen else {}


def role_env(harness: str, picks: dict, fallback=None,
             local_only=False, roster_value: dict | None = None) -> dict:
    """Freeze role/model/setting choices into the runner's environment API."""
    env = {"HARNESS": harness}
    normalized = {role: _pick(value) for role, value in picks.items()
                  if role in ROLES}
    if normalized:
        env["SUMM_ACTIVE_ROLES"] = ",".join(
            role for role in ROLES if role in normalized)
    fh, fm, fe = _pick(fallback)
    selected_harnesses = [h for h, model, _ in normalized.values() if model]
    if fm:
        selected_harnesses.append(fh)
    configured = (model_config.roster() if roster_value is None
                  else roster_value)
    local = model_config.local_harnesses(configured)
    local_only = bool(local_only or (
        selected_harnesses and all(h in local for h in selected_harnesses)))
    if local_only:
        env["SUMM_LOCAL_ONLY"] = "1"

    frozen = {}
    for role, (selected_harness, model, effort) in normalized.items():
        if not model:
            continue
        resolved = model_config.resolved_model(
            selected_harness, model, effort, role, configured)
        entry = (resolved if selected_harness == harness
                 else f"{selected_harness}:{resolved}")
        entries = [entry]
        role_frozen = [{"harness": selected_harness, "model": resolved,
                        "option": _option(selected_harness, model, effort,
                                          role, configured)}]
        # The profile's own backup first, then every committed backup in
        # order (free routes before the metered route),
        # each once, so one exhausted route never leaves the chain empty.
        backups = ([(fh, fm, fe)] if fm else []) + [
            tuple(item) for item in default_fallbacks()]
        for bh, bm, be in backups:
            if not bm or (local_only and bh not in local):
                continue
            fallback_model = model_config.resolved_model(
                bh, bm, be, role, configured)
            backup = fallback_model if bh == harness else f"{bh}:{fallback_model}"
            if backup in entries:
                continue
            entries.append(backup)
            role_frozen.append({"harness": bh, "model": fallback_model,
                                "option": _option(bh, bm, be, role,
                                                  configured)})
        env[f"{harness.upper()}_CHAIN_{role.upper()}"] = " ".join(entries)
        frozen[role] = role_frozen
    if frozen:
        env["SUMM_ROLE_OPTIONS"] = json.dumps(
            frozen, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return env


def profiles_path() -> pathlib.Path:
    return runtime.app_dir() / "profiles.json"


def last_profile_path() -> pathlib.Path:
    # Derive both preference files from the same overridable authority. Tests,
    # portable installs, and alternate front ends can redirect one root rather
    # than patching two unrelated path functions.
    return profiles_path().with_name("last-profile.json")


def load_last_profile() -> str:
    path = last_profile_path()
    if not path.exists():
        return ""
    try:
        raw = json.loads(path.read_text())
        value = raw.get("profile") if isinstance(raw, dict) else None
        if not isinstance(value, str):
            raise ValueError("profile must be a string")
        return value.strip()
    except Exception as exc:
        raise runtime.ConfigError(f"invalid {path}: {exc}") from exc


def _atomic_json(path: pathlib.Path, value):
    runtime.atomic_json(path, value)


def save_last_profile(name: str):
    _atomic_json(last_profile_path(), {"profile": name})


def _default_profiles() -> dict:
    spec = model_defaults().get("profile") or {}
    name = spec.get("name")
    if name and all(spec.get(role) for role in ROLES):
        return {name: {role: list(spec[role]) for role in ROLES}}
    return {}


def default_fallbacks() -> list[list[str]]:
    """The committed backup routes, in order (models.json `_defaults.fallback`,
    one object or an ordered list). Each is [harness, model, option]."""
    raw = model_defaults().get("fallback") or {}
    out = []
    for item in (raw if isinstance(raw, list) else [raw]):
        if isinstance(item, dict) and item.get("harness") and item.get("model"):
            out.append([str(item["harness"]), str(item["model"]),
                        str(item.get("variant") or item.get("effort") or "")])
    return out


def _default_fallback() -> list[str]:
    """The first committed backup: the one a profile shows and may replace."""
    first = default_fallbacks()
    return list(first[0]) if first else []


def _normalized_profiles(value, source: pathlib.Path | str) -> dict:
    if not isinstance(value, dict):
        raise runtime.ConfigError(f"invalid {source}: top level must be an object")
    out = {}
    for name, spec in value.items():
        if not isinstance(name, str) or not name.strip() or not isinstance(spec, dict):
            raise runtime.ConfigError(
                f"invalid {source}: every profile must be a named object")
        normalized = {}
        for key, selection in spec.items():
            if key not in {*ROLES, "_fallback"}:
                raise runtime.ConfigError(
                    f"invalid {source}: profile {name!r} has unknown field {key!r}")
            if (not isinstance(selection, (list, tuple))
                    or len(selection) not in {2, 3}
                    or any(not isinstance(item, str) for item in selection)):
                raise runtime.ConfigError(
                    f"invalid {source}: profile {name!r}.{key} must be a "
                    "two- or three-string selection")
            normalized[key] = list(selection)
        out[name.strip()] = normalized
    return out


def load_profiles() -> dict:
    """Merge device-local choices over the committed fresh-device default."""
    defaults = _default_profiles()
    path = profiles_path()
    if not path.exists():
        return defaults
    try:
        local = _normalized_profiles(json.loads(path.read_text()), path)
    except runtime.ConfigError:
        raise
    except Exception as exc:
        raise runtime.ConfigError(f"invalid {path}: {exc}") from exc
    defaults.update(local)
    return defaults


def save_profiles(value: dict):
    """Atomically save only device overrides, never a copied shared default."""
    path = profiles_path()
    normalized = _normalized_profiles(value, "profiles")
    defaults = _default_profiles()
    default_fallback = _default_fallback()
    local = {}
    for name, spec in normalized.items():
        candidate = dict(spec)
        if default_fallback and candidate.get("_fallback") == default_fallback:
            candidate.pop("_fallback")
        if defaults.get(name) != candidate:
            local[name] = candidate
    _atomic_json(path, local)


def launch_profile() -> str:
    available = load_profiles()
    remembered = load_last_profile()
    if remembered in available:
        return remembered
    fallback = DEFAULT_PROFILE if DEFAULT_PROFILE in available else ""
    if fallback and remembered:
        save_last_profile(fallback)
    return fallback
