"""Small client for a configured OpenAI-compatible text gateway.

The gateway is a transport, not a model-selection policy.  Its URL, bearer
token environment variable, timeout, and output limit are device-local runtime
settings.  The caller supplies the model identifier from models.json.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
import socket
import time
import urllib.error
import urllib.parse
import urllib.request

import runtime


class GatewayError(RuntimeError):
    """A gateway failure with enough classification for the chain runner."""

    def __init__(self, message: str, *, status: int | None = None,
                 kind: str = "failed", retry_after: float | None = None,
                 evidence: dict | None = None):
        super().__init__(message)
        self.status = status
        self.kind = kind
        self.retry_after = retry_after
        self.evidence = evidence


@dataclass(frozen=True)
class ChatCompletionResult:
    content: str
    finish_reason: str
    usage: dict
    response_id: str | None
    raw_response: dict
    elapsed_seconds: float
    response_headers: dict
    dispatch_budget: dict | None = None

    def evidence(self) -> dict:
        return {"response_id": self.response_id,
                "finish_reason": self.finish_reason,
                "usage": self.usage,
                "elapsed_seconds": round(self.elapsed_seconds, 3),
                "response_headers": self.response_headers,
                "dispatch_budget": self.dispatch_budget,
                "raw_response": self.raw_response}


@dataclass(frozen=True)
class DispatchBudget:
    """One exact rendered-request admission decision."""

    role: str
    prompt_sha256: str
    prompt_tokens: int
    context_guard_tokens: int
    reasoning_tokens: int
    boundary_overrun_tokens: int
    visible_tokens: int
    completion_tokens: int
    semantic_retry: bool
    continuation: bool = False

    def evidence(self) -> dict:
        return {
            "role": self.role,
            "prompt_sha256": self.prompt_sha256,
            "prompt_tokens": self.prompt_tokens,
            "context_guard_tokens": self.context_guard_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "boundary_overrun_tokens": self.boundary_overrun_tokens,
            "visible_tokens": self.visible_tokens,
            "completion_tokens": self.completion_tokens,
            "semantic_retry": self.semantic_retry,
            "continuation": self.continuation,
        }


def _merge_options(base: dict, additions: dict) -> dict:
    merged = dict(base)
    additions = dict(additions)
    for name in ("chat_template_kwargs", "custom_params"):
        nested = additions.get(name)
        if isinstance(nested, dict) and isinstance(merged.get(name), dict):
            merged[name] = {**merged[name], **nested}
            additions.pop(name)
    merged.update(additions)
    return merged


def _qualification_court() -> bool:
    """True only for an explicit qualification court, never production."""
    return os.environ.get("SUMM_QUALIFICATION_COURT", "").strip() in {
        "1", "true", "yes",
    }


def _option_binding(options: dict, binding: str):
    if binding == "custom_params.thinking_budget":
        nested = options.get("custom_params")
        return nested.get("thinking_budget") if isinstance(nested, dict) else None
    return options.get(binding)


def _set_option_binding(options: dict, binding: str, value) -> None:
    if binding == "custom_params.thinking_budget":
        nested = options.get("custom_params")
        nested = dict(nested) if isinstance(nested, dict) else {}
        nested["thinking_budget"] = value
        options["custom_params"] = nested
        return
    options[binding] = value


def request_options(harness: str, model: str = "", overrides: dict | None = None,
                    *, runtime_config: dict | None = None, role: str = "",
                    reasoning_tokens: int | None = None) -> dict:
    """Return device-local request additions for one gateway/model.

    A single public gateway name may expose several qualified models. The
    route and its defaults stay in ignored device config; the pipeline only
    knows the generic model key selected from models.json.
    """
    root = runtime_config if runtime_config is not None else runtime.config()
    cfg = root.get("gateways", {}).get(harness)
    if not isinstance(cfg, dict):
        return {}
    try:
        options = runtime.validate_gateway_request_options(
            cfg.get("request_options"), f"gateway {harness!r}")
        route = (cfg.get("models") or {}).get(model, {}) if model else {}
        if isinstance(route, dict):
            options = _merge_options(
                options, runtime.validate_gateway_request_options(
                    route.get("request_options"),
                    f"gateway {harness!r} model {model!r}"))
    except ValueError as exc:
        raise GatewayError(str(exc), kind="config") from exc
    override = os.environ.get(f"SUMM_{harness.upper()}_REQUEST_OPTIONS")
    if override:
        try:
            options = runtime.validate_gateway_request_options(
                json.loads(override),
                f"SUMM_{harness.upper()}_REQUEST_OPTIONS")
        except (json.JSONDecodeError, ValueError) as exc:
            raise GatewayError(
                f"invalid frozen request options for gateway {harness!r}",
                kind="config") from exc
    if overrides:
        try:
            options = _merge_options(
                options, runtime.validate_gateway_request_options(
                    overrides, f"frozen options for gateway {harness!r}"))
        except ValueError as exc:
            raise GatewayError(str(exc), kind="config") from exc
    admission = route.get("reasoning_admission") if isinstance(route, dict) else None
    if role and isinstance(admission, dict):
        enforcement = admission.get("enforcement") or {}
        counter = admission.get("token_counter") or {}
        # Overall application qualification is evidence for quality/promotion,
        # not permission to use an explicitly configured route. Hard transport
        # certificates, role bindings, thinking and envelope checks still apply.
        if (enforcement.get("status") != "passed"
                or counter.get("status") != "passed"):
            raise GatewayError(
                f"gateway {harness!r} model {model!r} reasoning enforcement or token counter is unqualified",
                kind="config")
        policy = (admission.get("roles") or {}).get(role)
        binding = admission.get("budget_binding")
        if not isinstance(policy, dict) or not isinstance(binding, str):
            raise GatewayError(
                f"gateway {harness!r} model {model!r} has no {role!r} reasoning budget",
                kind="config")
        template = options.get("chat_template_kwargs")
        if not isinstance(template, dict) or template.get("enable_thinking") is not True:
            raise GatewayError(
                f"gateway {harness!r} model {model!r} requires thinking for {role}",
                kind="config")
        expected = (policy.get("retry_reasoning_tokens")
                    if reasoning_tokens is not None and reasoning_tokens ==
                    policy.get("retry_reasoning_tokens")
                    else policy.get("reasoning_tokens"))
        if reasoning_tokens is not None and reasoning_tokens != expected:
            raise GatewayError(
                f"gateway {harness!r} model {model!r} reasoning budget is invalid",
                kind="config")
        actual = _option_binding(options, binding)
        if actual is not None and actual != expected:
            raise GatewayError(
                f"gateway {harness!r} model {model!r} reasoning budget is protected",
                kind="config")
        _set_option_binding(options, binding, expected)
    return options


def _settings(harness: str, model: str = "", overrides: dict | None = None,
              *, role: str = "", reasoning_tokens: int | None = None) -> dict:
    root = runtime.config()
    cfg = root.get("gateways", {}).get(harness)
    if not isinstance(cfg, dict):
        raise GatewayError(
            f"gateway {harness!r} is not configured in device-local runtime settings",
            kind="config")
    route = (cfg.get("models") or {}).get(model, {}) if model else {}
    route = route if isinstance(route, dict) else {}
    options = request_options(
        harness, model, overrides, runtime_config=root, role=role,
        reasoning_tokens=reasoning_tokens)
    base_url = str(route.get("base_url", cfg.get("base_url")) or "").strip()
    authentication = route.get(
        "authentication", cfg.get("authentication", "bearer_env"))
    api_key_env = str(route.get("api_key_env", cfg.get("api_key_env")) or "").strip()
    try:
        timeout = int(route.get("timeout_seconds", cfg.get("timeout_seconds", 1800)))
        max_output_tokens = int(route.get(
            "max_output_tokens", cfg["max_output_tokens"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise GatewayError(
            f"gateway {harness!r} needs integer max_output_tokens and timeout_seconds",
            kind="config") from exc
    parsed = urllib.parse.urlsplit(base_url)
    if (parsed.scheme not in {"http", "https"} or not parsed.netloc or
            parsed.query or parsed.fragment or
            authentication not in {"bearer_env", "none"} or
            (authentication == "bearer_env" and not api_key_env) or
            timeout < 1 or max_output_tokens < 1):
        raise GatewayError(
            f"gateway {harness!r} has invalid device-local settings",
            kind="config")
    token = os.environ.get(api_key_env, "") if authentication == "bearer_env" else ""
    if authentication == "bearer_env" and not token:
        raise GatewayError(
            f"gateway HTTP 401: token is missing from {api_key_env}",
            status=401, kind="auth")
    return {"base_url": base_url.rstrip("/"), "authentication": authentication,
            "token": token,
            "timeout": timeout, "max_output_tokens": max_output_tokens,
            "output_token_field": route.get(
                "output_token_field", cfg.get("output_token_field", "max_tokens")),
            "reasoning_content": route.get(
                "reasoning_content", cfg.get("reasoning_content", "plain")),
            "structured_output": route.get(
                "structured_output", cfg.get("structured_output") or []),
            "error_policy": route.get(
                "error_policy", cfg.get("error_policy") or {}),
            "request_options": options,
            "context_tokens": route.get("context_tokens"),
            "reasoning_admission": route.get("reasoning_admission")}


def _detail(raw: bytes) -> str:
    try:
        obj = json.loads(raw.decode("utf-8", "replace"))
        if isinstance(obj, dict):
            for key in ("error", "message", "detail"):
                value = obj.get(key)
                if value:
                    return str(value)[:240]
    except Exception:
        pass
    return raw.decode("utf-8", "replace").strip()[:240]


def _error_code(raw: bytes) -> str | None:
    try:
        obj = json.loads(raw.decode("utf-8", "replace"))
        if isinstance(obj, dict):
            nested = obj.get("error")
            if isinstance(nested, dict) and isinstance(nested.get("code"), str):
                return nested["code"]
            if isinstance(obj.get("code"), str):
                return obj["code"]
    except Exception:
        pass
    return None


def _http_error(error: urllib.error.HTTPError, policy: dict) -> GatewayError:
    raw = b""
    try:
        raw = error.read()
    except Exception:
        raw = b""
    detail = _detail(raw)
    message = f"gateway HTTP {error.code}"
    if detail:
        message += f": {detail}"
    retry_after = None
    code = _error_code(raw)
    by_code = policy.get("codes") or {}
    by_status = policy.get("status") or {}
    configured = by_code.get(code) or by_status.get(str(error.code)) \
        or by_status.get(error.code)
    if configured:
        kind = configured
    elif error.code in {401, 403}:
        kind = "auth"
    elif error.code in {408, 429}:
        kind = "capacity"
    elif error.code in {400, 413}:
        kind = "request"
    else:
        kind = "failed"
    if kind == "warming":
        _code, retry_after = _warm_signal(raw, error)
    response_headers = {
        key.lower(): error.headers.get(key)
        for key in ("Content-Type", "X-Request-ID", "Request-ID", "Retry-After")
        if error.headers and error.headers.get(key)
    }
    result = GatewayError(
        message,
        status=error.code,
        kind=kind,
        retry_after=retry_after,
        # The body is intentionally not retained: gateway errors can contain
        # provider-specific details. Its bounded detail, identity fields, and
        # digest are enough to correlate a failed stage without copying an
        # arbitrary response into run evidence.
        evidence={
            "status": error.code,
            "error_code": code,
            "detail": detail,
            "response_headers": response_headers,
            "raw_body_bytes": len(raw),
            "raw_body_sha256": hashlib.sha256(raw).hexdigest(),
        },
    )
    try:
        error.close()
    except Exception:
        pass
    return result


def _warm_signal(raw: bytes, error: urllib.error.HTTPError):
    """Return (warm_code, retry_after_seconds) from a 503 body/headers."""
    code = None
    retry_after = None
    try:
        obj = json.loads(raw.decode("utf-8", "replace"))
        if isinstance(obj, dict):
            err = obj.get("error")
            if isinstance(err, dict):
                code = err.get("code")
            warm = obj.get("warm")
            if isinstance(warm, dict) and warm.get("retry_after_seconds"):
                retry_after = float(warm["retry_after_seconds"])
    except Exception:
        pass
    try:
        header = error.headers.get("Retry-After") if error.headers else None
        if header and retry_after is None:
            retry_after = float(header)
    except Exception:
        pass
    return code, retry_after


def _count_request_tokens(cfg: dict, model: str, prompt: str,
                          messages: list | None = None) -> int:
    """Count the rendered chat request through the qualified route adapter."""
    admission = cfg.get("reasoning_admission") or {}
    counter = admission.get("token_counter") or {}
    if (counter.get("binding") != "openai_chat_tokenize"
            or counter.get("status") != "passed"):
        raise GatewayError("gateway token counter is unqualified", kind="config")
    options = cfg.get("request_options") or {}
    payload = {
        "model": model,
        "messages": list(messages) if messages else [
            {"role": "user", "content": prompt}],
        "add_generation_prompt": True,
    }
    for name in ("chat_template_kwargs", "reasoning_effort"):
        if name in options:
            payload[name] = options[name]
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if cfg["authentication"] == "bearer_env":
        headers["Authorization"] = f"Bearer {cfg['token']}"
    request = urllib.request.Request(
        cfg["base_url"] + "/tokenize",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=cfg["timeout"]) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        raise _http_error(exc, cfg["error_policy"]) from exc
    except (urllib.error.URLError, TimeoutError, socket.timeout, OSError) as exc:
        raise GatewayError(
            f"gateway token count status is unknown: {str(exc)[:180]}",
            kind="completion_unknown") from exc
    try:
        result = json.loads(body.decode("utf-8"))
        count = result.get("count") if isinstance(result, dict) else None
        if count is None and isinstance(result, dict) \
                and isinstance(result.get("tokens"), list):
            count = len(result["tokens"])
    except (UnicodeDecodeError, ValueError, TypeError) as exc:
        raise GatewayError(
            "gateway token counter returned malformed JSON", kind="unusable",
            evidence={"raw_body_bytes": len(body),
                      "raw_body_sha256": hashlib.sha256(body).hexdigest()}) from exc
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise GatewayError("gateway token counter returned an invalid count",
                           kind="unusable")
    return count


def _dispatch_budget(cfg: dict, model: str, prompt: str, role: str, *,
                     planned_output_words: int,
                     output_overhead_tokens: int,
                     semantic_retry: bool,
                     messages: list | None = None,
                     continuation: bool = False) -> DispatchBudget:
    admission = cfg.get("reasoning_admission") or {}
    policy = (admission.get("roles") or {}).get(role)
    if not isinstance(policy, dict):
        raise GatewayError(f"gateway route has no {role!r} admission policy",
                           kind="config")
    if (isinstance(planned_output_words, bool)
            or not isinstance(planned_output_words, int)
            or planned_output_words < 0
            or isinstance(output_overhead_tokens, bool)
            or not isinstance(output_overhead_tokens, int)
            or output_overhead_tokens < 0):
        raise GatewayError("gateway dispatch has invalid output bounds",
                           kind="config")
    preferred_reasoning = int(policy[
        "retry_reasoning_tokens" if semantic_retry else "reasoning_tokens"])
    if continuation:
        preferred_reasoning = min(preferred_reasoning, 64)
    boundary = int(admission["boundary_overrun_tokens"])
    ratio = float(admission["output_tokens_per_word"])
    visible = math.ceil(1.15 * (
        output_overhead_tokens + ratio * planned_output_words)) + 128
    ceiling = int(admission["completion_tokens"])
    evidence = {
        "role": role,
        "planned_output_words": planned_output_words,
        "output_overhead_tokens": output_overhead_tokens,
        "output_tokens_per_word": ratio,
        "visible_tokens": visible,
        "reasoning_tokens": preferred_reasoning,
        "boundary_overrun_tokens": boundary,
    }
    if visible > int(policy["visible_tokens"]):
        evidence["visible_ceiling_tokens"] = int(policy["visible_tokens"])
        raise GatewayError("gateway packet exceeds the visible-answer envelope",
                           kind="capacity", evidence=evidence)
    # The visible answer is reserved first. Thinking may shrink; the answer
    # may not. max_tokens is the route completion ceiling so unused thinking
    # tokens remain available for the final channel.
    reasoning_room = ceiling - boundary - visible
    if reasoning_room < 1:
        evidence["completion_tokens"] = boundary + visible + 1
        evidence["completion_ceiling_tokens"] = ceiling
        raise GatewayError("gateway packet exceeds the shared completion envelope",
                           kind="capacity", evidence=evidence)
    reasoning = min(preferred_reasoning, reasoning_room)
    if reasoning < preferred_reasoning:
        evidence["reasoning_preferred_tokens"] = preferred_reasoning
        evidence["reasoning_reduced_to_protect_visible"] = True
    evidence["reasoning_tokens"] = reasoning
    completion = ceiling
    prompt_tokens = _count_request_tokens(cfg, model, prompt, messages)
    context_guard = max(256, math.ceil(0.01 * prompt_tokens))
    evidence.update(prompt_tokens=prompt_tokens,
                    prompt_ceiling_tokens=int(policy["prompt_tokens"]),
                    context_guard_tokens=context_guard,
                    completion_tokens=completion,
                    context_tokens=cfg.get("context_tokens"),
                    continuation=bool(continuation))
    blob = prompt if not messages else json.dumps(messages, ensure_ascii=False)
    if prompt_tokens > int(policy["prompt_tokens"]):
        raise GatewayError("gateway packet exceeds the role prompt working set",
                           kind="capacity", evidence=evidence)
    context = cfg.get("context_tokens")
    if (isinstance(context, bool) or not isinstance(context, int)
            or prompt_tokens + completion + context_guard > context):
        raise GatewayError("gateway packet exceeds the context envelope",
                           kind="capacity", evidence=evidence)
    return DispatchBudget(
        role=role,
        prompt_sha256=hashlib.sha256(blob.encode("utf-8")).hexdigest(),
        prompt_tokens=prompt_tokens,
        context_guard_tokens=context_guard,
        reasoning_tokens=reasoning,
        boundary_overrun_tokens=boundary,
        visible_tokens=visible,
        completion_tokens=completion,
        semantic_retry=bool(semantic_retry),
        continuation=bool(continuation),
    )


def _completion(body: bytes, *, reasoning_content: str, elapsed_seconds: float,
                response_headers: dict,
                dispatch_budget: DispatchBudget | None = None,
                strip_content: bool = True) -> ChatCompletionResult:
    try:
        obj = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise GatewayError(
            "gateway returned a non-JSON chat completion envelope",
            kind="unusable", evidence={
                "raw_body_bytes": len(body),
                "raw_body_sha256": hashlib.sha256(body).hexdigest(),
            }) from exc
    base_evidence = {
        "response_id": (str(obj["id"]) if isinstance(obj, dict)
                        and obj.get("id") is not None else None),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "response_headers": response_headers,
        "raw_response": obj,
    }
    try:
        choice = obj["choices"][0]
        message = choice["message"]
        content = message["content"]
        finish_reason = choice["finish_reason"]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise GatewayError("gateway returned malformed chat completion",
                           kind="unusable", evidence=base_evidence) from exc
    if not isinstance(finish_reason, str) or not finish_reason:
        raise GatewayError("gateway returned empty chat completion",
                           kind="unusable", evidence=base_evidence)
    base_evidence.update(
        finish_reason=finish_reason,
        usage=(dict(obj.get("usage"))
               if isinstance(obj.get("usage"), dict) else {}))
    reasoning = (message.get("reasoning_content")
                 if isinstance(message, dict) else None)
    if not isinstance(reasoning, str) and isinstance(message, dict):
        reasoning = message.get("reasoning")
    if not isinstance(content, str) or not content.strip():
        if isinstance(reasoning, str) and reasoning.strip():
            kind = ("output_limit" if finish_reason in {"length", "max_tokens"}
                    else "output_incomplete")
            raise GatewayError("gateway returned no answer after reasoning",
                               kind=kind, evidence=base_evidence)
        raise GatewayError("gateway returned empty chat completion",
                           kind="unusable", evidence=base_evidence)
    if reasoning_content == "inline_think" and content.lstrip().startswith("<think>"):
        if "</think>" not in content:
            raise GatewayError("gateway completion ended inside reasoning",
                               kind="output_incomplete", evidence=base_evidence)
        content = content.rsplit("</think>", 1)[1]
    if strip_content:
        content = content.strip()
    if not content.strip():
        raise GatewayError("gateway returned no answer after reasoning",
                           kind="output_incomplete", evidence=base_evidence)
    usage = obj.get("usage") if isinstance(obj.get("usage"), dict) else {}
    return ChatCompletionResult(
        content=content, finish_reason=finish_reason, usage=dict(usage),
        response_id=str(obj["id"]) if obj.get("id") is not None else None,
        raw_response=obj, elapsed_seconds=elapsed_seconds,
        response_headers=response_headers,
        dispatch_budget=(dispatch_budget.evidence()
                         if dispatch_budget is not None else None))


def chat(harness: str, model: str, prompt: str,
         request_options: dict | None = None, *, role: str = "",
         planned_output_words: int | None = None,
         output_overhead_tokens: int | None = None,
         semantic_retry: bool = False,
         messages: list | None = None,
         continuation: bool = False) -> ChatCompletionResult:
    """Send one non-streaming request and preserve completion state.

    A transport timeout or connection loss is completion-unknown: the server
    may have accepted the request, so the caller must not retry it blindly.
    ``messages`` is the exact chat request when continuing a truncated answer.
    """
    root = runtime.config()
    gateway_cfg = root.get("gateways", {}).get(harness) or {}
    route = (gateway_cfg.get("models") or {}).get(model) or {}
    admission = route.get("reasoning_admission") if isinstance(route, dict) else None
    reasoning_tokens = None
    if role and isinstance(admission, dict):
        policy = (admission.get("roles") or {}).get(role) or {}
        reasoning_tokens = policy.get(
            "retry_reasoning_tokens" if semantic_retry else "reasoning_tokens")
    cfg = _settings(harness, model, request_options, role=role,
                    reasoning_tokens=reasoning_tokens)
    dispatch_budget = None
    if role and isinstance(cfg.get("reasoning_admission"), dict):
        if planned_output_words is None or output_overhead_tokens is None:
            raise GatewayError(
                "qualified reasoning route requires a bounded output assignment",
                kind="config")
        dispatch_budget = _dispatch_budget(
            cfg, model, prompt, role,
            planned_output_words=planned_output_words,
            output_overhead_tokens=output_overhead_tokens,
            semantic_retry=semantic_retry,
            messages=messages,
            continuation=continuation)
    payload = {
        "model": model,
        "messages": list(messages) if messages else [
            {"role": "user", "content": prompt}],
        "temperature": 0,
        "stream": False,
    }
    payload[cfg["output_token_field"]] = (
        dispatch_budget.completion_tokens if dispatch_budget is not None
        else cfg["max_output_tokens"])
    payload.update(cfg["request_options"])
    if dispatch_budget is not None:
        payload[cfg["output_token_field"]] = dispatch_budget.completion_tokens
        binding = (cfg.get("reasoning_admission") or {}).get("budget_binding")
        if isinstance(binding, str):
            _set_option_binding(
                payload, binding, dispatch_budget.reasoning_tokens)
    fmt = payload.get("response_format")
    if isinstance(fmt, dict) and fmt.get("type") not in cfg["structured_output"]:
        raise GatewayError(
            f"gateway {harness!r} does not declare structured output "
            f"{fmt.get('type')!r}", kind="config")
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if cfg["authentication"] == "bearer_env":
        headers["Authorization"] = f"Bearer {cfg['token']}"
    request = urllib.request.Request(
        cfg["base_url"] + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST")
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=cfg["timeout"]) as response:
            body = response.read()
            headers = {key.lower(): response.headers.get(key) for key in (
                "Content-Type", "X-Request-ID", "Request-ID")
                if response.headers.get(key)}
    except urllib.error.HTTPError as exc:
        raise _http_error(exc, cfg["error_policy"]) from exc
    except (urllib.error.URLError, TimeoutError, socket.timeout, OSError) as exc:
        raise GatewayError(
            f"gateway completion status is unknown: {str(exc)[:180]}",
            kind="completion_unknown") from exc
    json_mode = (isinstance(payload.get("response_format"), dict)
                 and payload["response_format"].get("type") in {
                     "json_object", "json_schema"})
    return _completion(
        body, reasoning_content=cfg["reasoning_content"],
        elapsed_seconds=time.monotonic() - started, response_headers=headers,
        dispatch_budget=dispatch_budget,
        strip_content=not (continuation or json_mode))
