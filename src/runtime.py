#!/usr/bin/env python3
"""Small host-local admission and exclusion primitives for Summer.

There is deliberately no scheduler daemon.  Fixed OS-lock slots make separate
UI windows and direct CLI processes obey the same limits, and process death
releases a lease without stale-state recovery.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import pathlib
import random
import re
import sys
import threading
import time
import urllib.parse
import uuid

import mode_config

WIN = sys.platform.startswith("win")
if WIN:                                      # pragma: no cover - run on Windows
    import msvcrt
else:                                        # pragma: no cover - branch is platform specific
    import fcntl


class ConfigError(RuntimeError):
    pass


class BusyError(RuntimeError):
    pass


_RESERVED_GATEWAY_REQUEST_KEYS = frozenset({
    "model", "messages", "max_tokens", "max_completion_tokens", "stream",
})
_GATEWAY_PROTOCOLS = frozenset({"openai_chat_completions"})
_TOKEN_FIELDS = frozenset({"max_tokens", "max_completion_tokens"})
_REASONING_CONTENT = frozenset({"plain", "inline_think"})
_STRUCTURED_OUTPUT = frozenset({"json_object", "json_schema"})
_AUTHENTICATION = frozenset({"bearer_env", "none"})
_ERROR_KINDS = frozenset({
    "auth", "capacity", "request", "warming", "unavailable", "failed",
})
_REASONING_ADMISSION_SCHEMA = "summer.reasoning-admission.v2"
_REASONING_BUDGET_BINDINGS = frozenset({
    "thinking_token_budget", "max_thinking_tokens",
    "custom_params.thinking_budget",
})


def validate_gateway_request_options(value, name="gateway") -> dict:
    """Validate harmless, device-local additions to a chat request.

    The pipeline owns the model, prompt, output bound, and non-streaming
    contract. Other OpenAI-compatible request fields—such as
    ``reasoning_effort``—may be declared by a device without adding a
    provider-specific branch to Summer.
    """
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name}.request_options must be an object")
    if any(not isinstance(key, str) or not key.strip() for key in value):
        raise ValueError(f"{name}.request_options keys must be non-empty strings")
    reserved = sorted(_RESERVED_GATEWAY_REQUEST_KEYS.intersection(value))
    if reserved:
        raise ValueError(
            f"{name}.request_options cannot override {', '.join(reserved)}")
    try:
        json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}.request_options must contain JSON values") from exc
    return dict(value)


def _validate_error_policy(value, name):
    if value is None:
        return
    if not isinstance(value, dict) or set(value) - {"status", "codes"}:
        raise ValueError(f"{name}.error_policy must contain only status and codes")
    statuses, codes = value.get("status") or {}, value.get("codes") or {}
    if not isinstance(statuses, dict) or not isinstance(codes, dict):
        raise ValueError(f"{name}.error_policy maps must be objects")
    for raw, kind in statuses.items():
        try:
            status = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}.error_policy status must be an integer") from exc
        if status < 400 or status > 599 or kind not in _ERROR_KINDS:
            raise ValueError(f"{name}.error_policy has invalid status mapping")
    for code, kind in codes.items():
        if (not isinstance(code, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", code)
                or kind not in _ERROR_KINDS):
            raise ValueError(f"{name}.error_policy has invalid code mapping")


def _validate_roster(value, name):
    if not isinstance(value, dict) or value.get("_local") not in {True, False}:
        raise ValueError(f"{name}.roster needs an explicit _local boolean")
    found = set()
    for role in mode_config.ROLES:
        entries = value.get(role) or []
        if not isinstance(entries, list) or any(
                not isinstance(model, str) or not model.strip() for model in entries):
            raise ValueError(f"{name}.roster.{role} must be a model list")
        found.update(entries)
    held = value.get("_held") or []
    if not isinstance(held, list) or any(
            not isinstance(model, str) or not model.strip() for model in held):
        raise ValueError(f"{name}.roster._held must be a model list")
    found.update(held)
    if not found:
        raise ValueError(f"{name}.roster declares no models")
    routable = set(found)
    settings = value.get("_model_settings") or {}
    if not isinstance(settings, dict):
        raise ValueError(f"{name}.roster._model_settings must be an object")
    for model, setting in settings.items():
        if model not in found or not isinstance(setting, dict):
            raise ValueError(f"{name}.roster has settings for an unknown model")
        values = setting.get("values") or []
        default = setting.get("default", setting.get("value"))
        if (not isinstance(setting.get("option"), str) or not isinstance(values, list)
                or any(not isinstance(item, str) or not item for item in values)
                or (default is not None and default not in values)):
            raise ValueError(f"{name}.roster has invalid settings for {model!r}")
        variants = setting.get("variants")
        if variants is not None and (
                setting.get("option") != "model" or not isinstance(variants, dict)
                or set(variants) != set(values)
                or any(not isinstance(target, str) or not target
                       for target in variants.values())):
            raise ValueError(f"{name}.roster has invalid variants for {model!r}")
        if isinstance(variants, dict):
            routable.update(variants.values())
    role_defaults = value.get("_role_defaults") or {}
    if not isinstance(role_defaults, dict):
        raise ValueError(f"{name}.roster._role_defaults must be an object")
    for role, by_model in role_defaults.items():
        if role not in mode_config.ROLES or not isinstance(by_model, dict):
            raise ValueError(f"{name}.roster has invalid role defaults")
        for model, selected in by_model.items():
            setting = settings.get(model)
            if (model not in found or not isinstance(setting, dict)
                    or selected not in (setting.get("values") or [])):
                raise ValueError(
                    f"{name}.roster has invalid {role} default for {model!r}")
    return routable


def _validate_gateway_capabilities(spec, name):
    if spec.get("protocol", "openai_chat_completions") not in _GATEWAY_PROTOCOLS:
        raise ValueError(f"{name}.protocol is unsupported")
    if spec.get("output_token_field", "max_tokens") not in _TOKEN_FIELDS:
        raise ValueError(f"{name}.output_token_field is unsupported")
    if spec.get("reasoning_content", "plain") not in _REASONING_CONTENT:
        raise ValueError(f"{name}.reasoning_content is unsupported")
    if spec.get("authentication", "bearer_env") not in _AUTHENTICATION:
        raise ValueError(f"{name}.authentication is unsupported")
    structured = spec.get("structured_output") or []
    if (not isinstance(structured, list)
            or any(value not in _STRUCTURED_OUTPUT for value in structured)):
        raise ValueError(f"{name}.structured_output is invalid")
    revision = spec.get("revision")
    if revision is not None and (not isinstance(revision, str) or len(revision) > 160):
        raise ValueError(f"{name}.revision is invalid")
    _validate_error_policy(spec.get("error_policy"), name)


def _validate_route_capacity(spec, name):
    """Validate an optional qualified context envelope for one route."""
    context = spec.get("context_tokens")
    if context is None:
        if any(key in spec for key in ("output_tokens", "prompt_overhead_tokens")):
            raise ValueError(f"{name}.context_tokens is required with route capacity")
        return
    if isinstance(context, bool) or not isinstance(context, int) or context < 1:
        raise ValueError(f"{name}.context_tokens must be a positive integer")
    output = spec.get("output_tokens")
    if (output is not None and
            (isinstance(output, bool) or not isinstance(output, int) or output < 1)):
        raise ValueError(f"{name}.output_tokens must be a positive integer")
    overhead = spec.get("prompt_overhead_tokens")
    if (overhead is not None and
            (isinstance(overhead, bool) or not isinstance(overhead, int) or overhead < 0)):
        raise ValueError(
            f"{name}.prompt_overhead_tokens must be a non-negative integer")


def _validate_certificate(value, name):
    if not isinstance(value, dict) or set(value) != {
            "status", "certificate_sha256"}:
        raise ValueError(f"{name} is invalid")
    if value.get("status") not in {"passed", "unqualified"}:
        raise ValueError(f"{name}.status is invalid")
    certificate = value.get("certificate_sha256")
    if value["status"] == "passed":
        if (not isinstance(certificate, str)
                or not re.fullmatch(r"[0-9a-f]{64}", certificate)):
            raise ValueError(f"{name} needs a certificate")
    elif certificate is not None:
        raise ValueError(f"{name} cannot cite a certificate while unqualified")


def _validate_reasoning_admission(value, name, completion_limit,
                                  completion_binding):
    """Validate one qualified shared reasoning/content completion contract."""
    if value is None:
        return
    keys = {
        "schema", "accounting", "completion_tokens", "completion_binding",
        "budget_binding", "enable_binding", "required",
        "boundary_overrun_tokens", "output_tokens_per_word", "token_counter",
        "roles", "enforcement", "qualification",
    }
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{name}.reasoning_admission has invalid fields")
    if value.get("schema") != _REASONING_ADMISSION_SCHEMA:
        raise ValueError(f"{name}.reasoning_admission.schema is unsupported")
    if value.get("accounting") != "shared_reasoning_content":
        raise ValueError(f"{name}.reasoning_admission.accounting is unsupported")
    completion = value.get("completion_tokens")
    guard = value.get("boundary_overrun_tokens")
    if (isinstance(completion, bool) or not isinstance(completion, int)
            or completion != completion_limit):
        raise ValueError(
            f"{name}.reasoning_admission.completion_tokens must equal the route limit")
    if value.get("completion_binding") != completion_binding:
        raise ValueError(
            f"{name}.reasoning_admission.completion_binding does not match the route")
    if (isinstance(guard, bool) or not isinstance(guard, int) or guard < 0
            or guard > 1024):
        raise ValueError(
            f"{name}.reasoning_admission.boundary_overrun_tokens is invalid")
    ratio = value.get("output_tokens_per_word")
    if (isinstance(ratio, bool) or not isinstance(ratio, (int, float))
            or not 2 <= float(ratio) <= 10):
        raise ValueError(
            f"{name}.reasoning_admission.output_tokens_per_word is invalid")
    if value.get("budget_binding") not in _REASONING_BUDGET_BINDINGS:
        raise ValueError(f"{name}.reasoning_admission.budget_binding is unsupported")
    if value.get("enable_binding") != "chat_template_kwargs.enable_thinking":
        raise ValueError(f"{name}.reasoning_admission.enable_binding is unsupported")
    if value.get("required") is not True:
        raise ValueError(f"{name}.reasoning_admission.required must be true")
    roles = value.get("roles")
    if not isinstance(roles, dict) or set(roles) != set(mode_config.ROLES):
        raise ValueError(f"{name}.reasoning_admission.roles must cover every role")
    for role, policy in roles.items():
        if not isinstance(policy, dict) or set(policy) != {
                "reasoning_tokens", "retry_reasoning_tokens",
                "prompt_tokens", "visible_tokens"}:
            raise ValueError(f"{name}.reasoning_admission.roles.{role} is invalid")
        reasoning = policy.get("reasoning_tokens")
        retry_reasoning = policy.get("retry_reasoning_tokens")
        prompt = policy.get("prompt_tokens")
        visible = policy.get("visible_tokens")
        if (isinstance(reasoning, bool) or not isinstance(reasoning, int)
                or reasoning < 1):
            raise ValueError(
                f"{name}.reasoning_admission.roles.{role}.reasoning_tokens is invalid")
        if (isinstance(retry_reasoning, bool)
                or not isinstance(retry_reasoning, int)
                or retry_reasoning < reasoning or retry_reasoning >= completion):
            raise ValueError(
                f"{name}.reasoning_admission.roles.{role}.retry_reasoning_tokens is invalid")
        if (isinstance(prompt, bool) or not isinstance(prompt, int) or prompt < 1):
            raise ValueError(
                f"{name}.reasoning_admission.roles.{role}.prompt_tokens is invalid")
        if (isinstance(visible, bool) or not isinstance(visible, int)
                or visible < 1 or reasoning + guard + visible > completion):
            raise ValueError(
                f"{name}.reasoning_admission.roles.{role}.visible_tokens is invalid")
    counter = value.get("token_counter")
    if not isinstance(counter, dict) or set(counter) != {
            "binding", "status", "certificate_sha256"}:
        raise ValueError(f"{name}.reasoning_admission.token_counter is invalid")
    if counter.get("binding") != "openai_chat_tokenize":
        raise ValueError(
            f"{name}.reasoning_admission.token_counter.binding is unsupported")
    _validate_certificate(
        {"status": counter.get("status"),
         "certificate_sha256": counter.get("certificate_sha256")},
        f"{name}.reasoning_admission.token_counter")
    _validate_certificate(
        value.get("enforcement"),
        f"{name}.reasoning_admission.enforcement")
    _validate_certificate(
        value.get("qualification"),
        f"{name}.reasoning_admission.qualification")


def app_dir() -> pathlib.Path:
    if WIN:
        base = pathlib.Path(os.environ.get(
            "LOCALAPPDATA", pathlib.Path.home() / "AppData/Local"))
    else:
        base = pathlib.Path.home() / "Library/Application Support"
    return base / "summer"


def atomic_text(path: pathlib.Path, value: str) -> None:
    """Replace one device-local UTF-8 preference without a partial file."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}")
    try:
        tmp.write_text(value, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_json(path: pathlib.Path, value) -> None:
    """Replace one device-local JSON preference without exposing a partial file."""
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True))


def config_path() -> pathlib.Path:
    """Trusted device config, optionally redirected by an explicit install."""
    override = os.environ.get("SUMM_RUNTIME_CONFIG", "").strip()
    return pathlib.Path(override).expanduser() if override else app_dir() / "runtime.json"


def _validate_config(raw, source: str) -> dict:
    """Validate one already-decoded runtime snapshot."""
    try:
        if not isinstance(raw, dict):
            raise ValueError("top level must be an object")
        active = int(raw.get("active_targets", 2))
        default_h = int(raw.get("harness_capacity", 1))
        capacities = {str(k): int(v) for k, v in
                      (raw.get("capacities") or {}).items()}
        bindings = {str(k): str(v) for k, v in
                    (raw.get("bindings") or {}).items()}
        gateways = raw.get("gateways") or {}
        if not isinstance(gateways, dict):
            raise ValueError("gateways must be an object")
        for name, spec in gateways.items():
            if not isinstance(name, str) or not isinstance(spec, dict):
                raise ValueError("each gateway must be an object")
            base_url = str(spec.get("base_url") or "").strip()
            api_key_env = str(spec.get("api_key_env") or "").strip()
            authentication = spec.get("authentication", "bearer_env")
            parsed = urllib.parse.urlsplit(base_url)
            max_output_tokens = int(spec["max_output_tokens"])
            timeout = int(spec.get("timeout_seconds", 1800))
            resource = str(spec.get("resource", "gateway")).strip()
            validate_gateway_request_options(spec.get("request_options"), name)
            _validate_gateway_capabilities(spec, f"gateway {name}")
            roster_models = _validate_roster(
                spec.get("roster"), f"gateway {name}")
            model_routes = spec.get("models") or {}
            if not isinstance(model_routes, dict):
                raise ValueError(f"{name}.models must be an object")
            for model, route in model_routes.items():
                if not isinstance(model, str) or not model.strip() \
                        or not isinstance(route, dict):
                    raise ValueError(f"{name}.models entries must be objects")
                if model not in roster_models:
                    raise ValueError(
                        f"{name}.models names unknown roster model {model!r}")
                route_base = str(route.get("base_url", base_url) or "").strip()
                route_key = str(route.get("api_key_env", api_key_env) or "").strip()
                route_auth = route.get("authentication", authentication)
                route_parsed = urllib.parse.urlsplit(route_base)
                route_timeout = int(route.get("timeout_seconds", timeout))
                route_max = int(route.get("max_output_tokens", max_output_tokens))
                validate_gateway_request_options(
                    route.get("request_options"), f"gateway {name}.models.{model}")
                _validate_gateway_capabilities(
                    {**spec, **route}, f"gateway {name}.models.{model}")
                _validate_route_capacity(
                    route, f"gateway {name}.models.{model}")
                _validate_reasoning_admission(
                    route.get("reasoning_admission"),
                    f"gateway {name}.models.{model}", route_max,
                    route.get("output_token_field",
                              spec.get("output_token_field", "max_tokens")))
                if (route_parsed.scheme not in {"http", "https"}
                        or not route_parsed.netloc or route_parsed.query
                        or route_parsed.fragment
                        or (route_auth == "bearer_env" and not re.fullmatch(
                            r"[A-Z][A-Z0-9_]*", route_key))
                        or route_timeout < 1 or route_max < 1):
                    raise ValueError(
                        f"invalid gateway model settings for {name!r}/{model!r}")
            if (parsed.scheme not in {"http", "https"} or not parsed.netloc or
                    parsed.query or parsed.fragment or
                    (authentication == "bearer_env" and not re.fullmatch(
                        r"[A-Z][A-Z0-9_]*", api_key_env)) or
                    not resource or max_output_tokens < 1 or timeout < 1):
                raise ValueError(f"invalid gateway settings for {name!r}")
        if active < 1 or default_h < 1 or any(v < 1 for v in capacities.values()):
            raise ValueError("capacities must be positive integers")
    except Exception as e:
        raise ConfigError(f"invalid {source}: {e}") from e
    return {"active_targets": active, "harness_capacity": default_h,
            "capacities": capacities, "bindings": bindings,
            "gateways": gateways}


def config() -> dict:
    """Read one validated device config or its run-frozen snapshot.

    Front ends put the validated snapshot in ``SUMM_RUNTIME_JSON`` before a
    job is queued. Every stage in that job therefore sees one transport and
    capacity contract even if the device-local file changes mid-run.
    """
    snapshot = os.environ.get("SUMM_RUNTIME_JSON")
    if snapshot is not None:
        try:
            raw = json.loads(snapshot)
        except Exception as e:
            raise ConfigError(f"invalid $SUMM_RUNTIME_JSON: {e}") from e
        return _validate_config(raw, "$SUMM_RUNTIME_JSON")

    path = config_path()
    if not path.exists():
        return {"active_targets": 2, "harness_capacity": 1,
                "capacities": {}, "bindings": {}, "gateways": {}}
    try:
        raw = json.loads(path.read_text())
    except Exception as e:
        raise ConfigError(f"invalid {path}: {e}") from e
    return _validate_config(raw, str(path))


def frozen_json(value: dict | None = None) -> str:
    """Serialize a validated runtime contract for a queued job or child."""
    validated = _validate_config(value, "runtime snapshot") \
        if value is not None else config()
    return json.dumps(validated, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def _name(kind: str, key: str, slot: int | None = None) -> pathlib.Path:
    digest = hashlib.sha256(f"{kind}\0{key}".encode()).hexdigest()
    suffix = "" if slot is None else f"-{slot:03d}"
    return app_dir() / "locks" / f"{kind}-{digest}{suffix}.lock"


def _lock(fh, blocking: bool) -> bool:
    try:
        if WIN:                              # pragma: no cover - run on Windows
            fh.seek(0)
            mode = msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK
            msvcrt.locking(fh.fileno(), mode, 1)
        else:
            flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
            fcntl.flock(fh.fileno(), flags)
        return True
    except (BlockingIOError, OSError):
        return False


def _unlock(fh):
    if WIN:                                  # pragma: no cover - run on Windows
        fh.seek(0); msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


@contextlib.contextmanager
def file_lock(kind: str, key: str, blocking: bool = True, cancel_check=None):
    path = _name(kind, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as fh:
        if WIN and path.stat().st_size == 0:  # msvcrt locks an existing byte
            fh.write(b"0"); fh.flush()
        if blocking:
            while not _lock(fh, False):
                if cancel_check:
                    cancel_check()
                time.sleep(0.08 + random.random() * 0.08)
        elif not _lock(fh, False):
            raise BusyError(f"{kind} is already in use")
        try:
            yield
        finally:
            _unlock(fh)


@contextlib.contextmanager
def slot_lease(kind: str, key: str, capacity: int, cancel_check=None):
    """Acquire any one of a fixed set of cross-process slots."""
    started = time.monotonic()
    held = None
    while held is None:
        if cancel_check:
            cancel_check()
        slots = list(range(capacity)); random.shuffle(slots)
        for slot in slots:
            path = _name(kind, key, slot)
            path.parent.mkdir(parents=True, exist_ok=True)
            fh = path.open("a+b")
            if WIN and path.stat().st_size == 0:  # pragma: no cover - Windows
                fh.write(b"0"); fh.flush()
            if _lock(fh, False):
                held = fh
                break
            fh.close()
        if held is None:
            time.sleep(0.08 + random.random() * 0.08)
    try:
        yield time.monotonic() - started
    finally:
        _unlock(held); held.close()


def target_lease(cancel_check=None):
    cfg = config()
    return slot_lease("target", "host", cfg["active_targets"], cancel_check)


def resource_name(harness: str, cfg: dict | None = None) -> str:
    """Return the configured lock group for one harness.

    Gateway lanes default to one shared group because a gateway may have one
    resident profile even when it exposes multiple namespaces. A device may
    override that deliberately through ``bindings`` after qualifying the
    gateway's actual capacity.
    """
    cfg = config() if cfg is None else cfg
    gateway = cfg.get("gateways", {}).get(harness)
    default = (gateway.get("resource", "gateway")
               if isinstance(gateway, dict) else harness)
    return cfg["bindings"].get(harness, default)


def resource_lease(harness: str):
    cfg = config()
    resource = resource_name(harness, cfg)
    capacity = cfg["capacities"].get(resource, cfg["harness_capacity"])
    return slot_lease("resource", resource, capacity)


def destination_lock(primary: pathlib.Path, cancel_check=None, secondary: pathlib.Path | None = None):
    """Exclude writers of one artifact, or both members of a summary pair."""
    primary = pathlib.Path(primary).resolve()
    paths = [primary]
    if secondary is not None:
        paths.append(pathlib.Path(secondary).resolve())
    elif primary.name.startswith("summary."):
        paths.append(primary.with_name("brief." + primary.name[len("summary."):]))
    elif primary.name.endswith(".summary.md"):
        paths.append(primary.with_name(
            primary.name[:-len(".summary.md")] + ".brief.md"))
    key = "\0".join(sorted(os.path.normcase(str(path)) for path in paths))
    return file_lock("destination", key, cancel_check=cancel_check)


def output_directory_lock(directory: pathlib.Path, cancel_check=None):
    return file_lock("output", os.path.normcase(str(pathlib.Path(directory).resolve())),
                     cancel_check=cancel_check)


def work_root_lock(directory: pathlib.Path):
    return file_lock("work", os.path.normcase(str(pathlib.Path(directory).resolve())),
                     blocking=False)
