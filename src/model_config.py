"""One merged model roster: committed CLI models plus device-local gateways."""
from __future__ import annotations

import json
import os
import pathlib

import runtime
import mode_config

HERE = pathlib.Path(__file__).parent
ROLES = mode_config.ROLES
ROLE_MODEL_OVERRIDES = {
    "plan": "PLAN_MODEL", "write": "MODEL",
    "audit": "AUDIT_MODEL", "repair": "REPAIR_MODEL",
}


def committed() -> dict:
    try:
        value = json.loads((HERE / "models.json").read_text())
    except Exception as exc:
        raise runtime.ConfigError(f"models.json unreadable: {exc}") from exc
    if not isinstance(value, dict):
        raise runtime.ConfigError("models.json must contain one object")
    return value


def roster(runtime_config: dict | None = None) -> dict:
    """Return executable harness rosters without exposing private config on disk."""
    base = committed()
    out = {name: dict(spec) for name, spec in base.items()
           if not name.startswith("_") and isinstance(spec, dict)}
    configured = runtime.config() if runtime_config is None else runtime_config
    for name, gateway in configured.get("gateways", {}).items():
        local = gateway.get("roster")
        if not local:
            continue
        if name in out:
            raise runtime.ConfigError(
                f"gateway {name!r} collides with a committed harness roster")
        out[name] = dict(local)
    return out


def full(runtime_config: dict | None = None) -> dict:
    value = committed()
    return {**value, **roster(runtime_config)}


def defaults() -> dict:
    return dict(committed().get("_defaults") or {})


def gateway_harnesses(runtime_config: dict | None = None) -> frozenset[str]:
    configured = runtime.config() if runtime_config is None else runtime_config
    return frozenset(configured.get("gateways", {}))


def local_harnesses(roster_value: dict | None = None) -> frozenset[str]:
    configured = roster() if roster_value is None else roster_value
    return frozenset(name for name, item in configured.items()
                     if item.get("_local") is True)


def _settings(harness: str, roster_value: dict | None = None) -> dict:
    configured = roster() if roster_value is None else roster_value
    value = (configured.get(harness) or {}).get("_model_settings") or {}
    return value if isinstance(value, dict) else {}


def logical_model(harness: str, model: str,
                  roster_value: dict | None = None) -> str:
    """Return the roster-facing model for a provider transport identifier.

    Some providers encode effort in the transport model ID.  Their roster can
    expose one stable logical family while ``variants`` maps each UI effort to
    the exact ID the provider accepts.  Recognizing those IDs here also keeps
    older device profiles valid after a family is collapsed in the UI.
    """
    settings = _settings(harness, roster_value)
    if model in settings:
        return model
    for logical, raw in settings.items():
        if not isinstance(raw, dict):
            continue
        variants = raw.get("variants") or {}
        if isinstance(variants, dict) and model in variants.values():
            return str(logical)
    return model


def model_setting(harness: str, model: str = "",
                  roster_value: dict | None = None) -> dict:
    settings = _settings(harness, roster_value)
    logical = logical_model(harness, model, roster_value)
    item = settings.get(logical)
    if not isinstance(item, dict):
        return {}
    out = dict(item)
    variants = item.get("variants") or {}
    if isinstance(variants, dict):
        for value, target in variants.items():
            if model == target:
                out["value"] = value
                break
    return out


def setting_values(harness: str, model: str = "",
                   roster_value: dict | None = None) -> tuple[str, ...]:
    values = model_setting(harness, model, roster_value).get("values", [])
    if not isinstance(values, list):
        return ()
    return tuple(value for value in values
                 if isinstance(value, str) and value.strip())


def model_variants(harness: str, model: str = "",
                   roster_value: dict | None = None) -> dict[str, str]:
    variants = model_setting(harness, model, roster_value).get("variants", {})
    if not isinstance(variants, dict):
        return {}
    return {str(key): str(value) for key, value in variants.items()
            if isinstance(key, str) and isinstance(value, str) and value}


def setting_default(harness: str, model: str = "", role: str = "",
                    roster_value: dict | None = None) -> str:
    """Return the declared value for one role without guessing provider policy."""
    setting = model_setting(harness, model, roster_value)
    physical_value = setting.get("value")
    if isinstance(physical_value, str) and physical_value:
        return physical_value
    logical = logical_model(harness, model, roster_value)
    configured = roster() if roster_value is None else roster_value
    role_defaults = ((configured.get(harness) or {}).get("_role_defaults")
                     or {})
    by_model = role_defaults.get(role) if isinstance(role_defaults, dict) else None
    if isinstance(by_model, dict):
        value = by_model.get(logical)
        if isinstance(value, str) and value:
            return value
    value = setting.get("default")
    return value if isinstance(value, str) else ""


def resolved_model(harness: str, model: str, value: str = "", role: str = "",
                   roster_value: dict | None = None) -> str:
    """Resolve one logical selection to the exact provider model ID."""
    logical = logical_model(harness, model, roster_value)
    setting = model_setting(harness, model, roster_value)
    if setting.get("option") != "model":
        return model
    chosen = value or setting_default(
        harness, model, role, roster_value)
    variants = model_variants(harness, logical, roster_value)
    if not isinstance(chosen, str) or chosen not in variants:
        raise runtime.ConfigError(
            f"{harness}.{logical} has no model mapping for {chosen!r}")
    return variants[chosen]


def split_entry(entry: str, default_harness: str,
                roster_value: dict | None = None) -> tuple[str, str]:
    """Split a qualified route while preserving colons inside model IDs."""
    configured = roster() if roster_value is None else roster_value
    harness, separator, model = str(entry).partition(":")
    if separator and harness in configured and model:
        return harness, model
    return default_harness, str(entry)


def resolve_entry(entry: str, default_harness: str, role: str = "",
                  value: str = "", roster_value: dict | None = None) -> str:
    """Resolve a logical route and retain its explicit harness qualification."""
    harness, model = split_entry(entry, default_harness, roster_value)
    exact = resolved_model(harness, model, value, role, roster_value)
    return exact if harness == default_harness else f"{harness}:{exact}"


def role_chains(harness: str, roles=ROLES, environ=None,
                roster_value: dict | None = None) -> dict[str, list[str]]:
    """Resolve every selected role route to exact transport model IDs.

    This is shared by the CLI engine and ETA identity.  An environment chain is
    authoritative when present; otherwise the roster chain and its declared
    availability fallback are used.  Resolution always follows the harness on
    each qualified entry, never the run's base harness by accident.
    """
    configured = roster() if roster_value is None else roster_value
    if harness not in configured:
        raise runtime.ConfigError(f"model roster has no harness {harness!r}")
    env = os.environ if environ is None else environ
    local = local_harnesses(configured)
    fallback = defaults().get("fallback") or {}
    # One fallback or an ordered list of them; each is appended after the
    # profile's own entries, in order, when it names a distinct route.
    fallbacks = fallback if isinstance(fallback, list) else [fallback]
    out = {}
    for role in roles:
        variable = f"{harness.upper()}_CHAIN_{role.upper()}"
        override = str(env.get(ROLE_MODEL_OVERRIDES.get(role, "")) or "").strip()
        explicit = bool(override) or variable in env
        if override:
            entries = [override]
        elif variable in env:
            entries = str(env.get(variable) or "").split()
        else:
            entries = list((configured[harness].get(role) or ()))
        resolved = [resolve_entry(entry, harness, role=role,
                                  roster_value=configured)
                    for entry in entries]
        for fb in fallbacks:
            if not (not explicit and harness not in local
                    and not env.get("SUMM_LOCAL_ONLY")
                    and isinstance(fb, dict)
                    and fb.get("harness") and fb.get("model")):
                continue
            fallback_harness = str(fb["harness"])
            fallback_model = str(fb["model"])
            fallback_value = str(fb.get("variant") or fb.get("effort") or "")
            raw = (fallback_model if fallback_harness == harness else
                   f"{fallback_harness}:{fallback_model}")
            exact = resolve_entry(raw, harness, role=role,
                                  value=fallback_value,
                                  roster_value=configured)
            if (split_entry(exact, harness, configured)
                    not in {split_entry(item, harness, configured)
                            for item in resolved}):
                resolved.append(exact)
        out[role] = resolved
    return out
