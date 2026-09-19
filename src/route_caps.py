#!/usr/bin/env python3
"""Generic route capability contract: direct or windowed handling.

A route takes the direct (whole-source) path when the exact composed request
fits. No model registry, no family branches, no per-model catalog: an
optional route-provided budget in words, else the conservative baseline for
the qualified roughly-100k-context class. Callers must include prompt and
output margins in the overhead they pass.

The legacy word helpers remain for compatibility with existing tests and local
configuration. New Summary admission can use the token helpers below. A
token budget is a route capability, not a model-name rule; when no tokenizer is
available, the conservative estimator is explicit in the report.
"""
from __future__ import annotations

import json
import os

# Conservative word budget for one composed request on a qualified route.
# Admission is governed by the composed request that fits it: source plus
# prompt frame plus the expected pair. There is deliberately no default
# output cap -- a route that fits the composed request is admitted, and an
# output budget applies only when one is explicitly provided.
BASELINE_DIRECT_WORDS = 100_000
WINDOW_SHARE = 0.50

# Qualified working capacity is intentionally below an advertised 100k-token
# context. The estimator is conservative and must be replaced by a route's
# tokenizer/count endpoint when one is available.
DEFAULT_QUALIFIED_CONTEXT_TOKENS = 90_000
CONSERVATIVE_TOKENS_PER_WORD = 1.35


def configured_capabilities(environ=None) -> dict:
    """Read optional device-local capabilities without model-specific logic.

    The frozen device-local ``runtime.json`` snapshot can declare capacity on
    each gateway model route. ``SUMM_ROUTE_CAPABILITIES`` remains a temporary
    qualification override keyed by ``harness:model`` or by harness. Each value
    must declare a positive qualified ``context_tokens``; ``output_tokens`` and
    ``prompt_overhead_tokens`` are optional. The data describes a tested route,
    not an advertised provider maximum, and therefore stays outside the
    committed model roster and publication logic.
    """
    env = os.environ if environ is None else environ
    out = {}

    frozen = str(env.get("SUMM_RUNTIME_JSON", "") or "").strip()
    if frozen:
        try:
            runtime = json.loads(frozen)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"SUMM_RUNTIME_JSON is not valid JSON: {exc}")
        if not isinstance(runtime, dict):
            raise ValueError("SUMM_RUNTIME_JSON must be an object")
        gateways = runtime.get("gateways") or {}
        if not isinstance(gateways, dict):
            raise ValueError("SUMM_RUNTIME_JSON gateways must be an object")
        for harness, gateway in gateways.items():
            if not isinstance(harness, str) or not isinstance(gateway, dict):
                raise ValueError("SUMM_RUNTIME_JSON gateways are invalid")
            models = gateway.get("models") or {}
            if not isinstance(models, dict):
                raise ValueError(f"{harness}: models must be an object")
            for model, route in models.items():
                if not isinstance(model, str) or not isinstance(route, dict):
                    raise ValueError(f"{harness}: model routes are invalid")
                if "context_tokens" in route:
                    out[f"{harness}:{model}"] = _validated_capability(
                        f"{harness}:{model}", route)

    raw = str(env.get("SUMM_ROUTE_CAPABILITIES", "") or "").strip()
    if not raw:
        return out
    try:
        value = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"SUMM_ROUTE_CAPABILITIES is not valid JSON: {exc}")
    if not isinstance(value, dict):
        raise ValueError("SUMM_ROUTE_CAPABILITIES must be an object")
    for route, spec in value.items():
        if not isinstance(route, str) or not route.strip() or not isinstance(spec, dict):
            raise ValueError("route capabilities need string keys and objects")
        out[route.strip()] = _validated_capability(route, spec)
    return out


def _validated_capability(route: str, spec: dict) -> dict:
    context = spec.get("context_tokens")
    if (isinstance(context, bool) or not isinstance(context, int)
            or context <= 0):
        raise ValueError(f"{route}: context_tokens must be positive")
    output = spec.get("output_tokens")
    if output is not None and (
            isinstance(output, bool) or not isinstance(output, int)
            or output <= 0):
        raise ValueError(f"{route}: output_tokens must be positive")
    prompt_overhead = spec.get("prompt_overhead_tokens")
    if prompt_overhead is not None and (
            isinstance(prompt_overhead, bool)
            or not isinstance(prompt_overhead, int)
            or prompt_overhead < 0):
        raise ValueError(
            f"{route}: prompt_overhead_tokens must be non-negative")
    return {"context_tokens": context,
            **({"output_tokens": output} if output is not None else {}),
            **({"prompt_overhead_tokens": prompt_overhead}
               if prompt_overhead is not None else {})}


def capability_for(route: str, harness: str | None = None,
                   environ=None) -> dict | None:
    """Return the exact route capability, then its generic harness default."""
    table = configured_capabilities(environ)
    key = str(route or "").strip()
    if key in table:
        return dict(table[key])
    name = str(harness or "").strip()
    if name in table:
        return dict(table[name])
    return None


def primary_chain_capability(chains, split_entry, default_harness: str,
                             environ=None) -> dict | None:
    """Summarize declared capacity for the first route of each active role.

    Admission is based on the routes that will be tried first. A larger backup
    is still useful when the primary fails, while :func:`route_fits` below
    prevents a declared smaller backup from receiving an oversized request.
    The function accepts the resolver as a callback so this module remains
    independent of model, harness, and roster implementations.
    """
    found = []
    declared_any = False
    route_names = []
    for entries in (chains or {}).values():
        if not entries:
            continue
        harness, model = split_entry(entries[0], default_harness)
        route = f"{harness}:{model}"
        capability = capability_for(route, harness, environ)
        if capability is None:
            capability = {"context_tokens": DEFAULT_QUALIFIED_CONTEXT_TOKENS}
        else:
            declared_any = True
        found.append(capability)
        route_names.append(route)
    if not found or not declared_any:
        return None
    contexts = [item["context_tokens"] for item in found]
    outputs = [item["output_tokens"] for item in found
               if item.get("output_tokens") is not None]
    result = {"context_tokens": min(contexts), "routes": route_names}
    if outputs:
        result["output_tokens"] = min(outputs)
    overheads = [item.get("prompt_overhead_tokens", 0) for item in found]
    if any(overheads):
        result["prompt_overhead_tokens"] = max(overheads)
    return result


def route_fits(prompt_words: int, capability: dict | None) -> bool:
    """Check one fully composed prompt against one declared route envelope."""
    if not capability:
        return True
    try:
        prompt_tokens = words_to_tokens(prompt_words)
        overhead = int(capability.get("prompt_overhead_tokens", 0))
        context = int(capability["context_tokens"])
        output = int(capability.get("output_tokens", 0))
    except (TypeError, ValueError, KeyError):
        return False
    return (overhead >= 0 and context > 0 and output >= 0
            and prompt_tokens + overhead + output <= context)


def _budget(value, baseline):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or value <= 0):
        return baseline
    return value


def budget_words(capability_words=None):
    """Return the effective generic input budget for a route."""
    return _budget(capability_words, BASELINE_DIRECT_WORDS)


def explicit_output_budget(output_words=None):
    """Return a usable explicit output budget, or None when absent/invalid."""
    return _budget(output_words, None)


def words_to_tokens(words) -> int:
    """Conservatively convert whitespace-word counts to estimated tokens."""
    try:
        value = float(words)
    except (TypeError, ValueError):
        raise ValueError("word count is not numeric")
    if value < 0:
        raise ValueError("word count is negative")
    return int(value * CONSERVATIVE_TOKENS_PER_WORD + 0.999999)


def tokens_to_words(tokens) -> int:
    try:
        value = int(tokens)
    except (TypeError, ValueError):
        return 0
    if value < 0:
        return 0
    return int(value / CONSERVATIVE_TOKENS_PER_WORD)


def budget_tokens(capability_tokens=None) -> int:
    return int(_budget(capability_tokens, DEFAULT_QUALIFIED_CONTEXT_TOKENS))


def explicit_output_budget_tokens(output_tokens=None):
    return _budget(output_tokens, None)


def direct_ok_tokens(source_words, overhead_words=0, capability_tokens=None,
                     *, output_words=0, output_budget_tokens=None) -> bool:
    """True when a composed request fits a qualified token envelope."""
    try:
        request = words_to_tokens(int(source_words) + int(overhead_words))
        output = words_to_tokens(output_words)
    except (TypeError, ValueError):
        return False
    budget = budget_tokens(capability_tokens)
    output_budget = explicit_output_budget_tokens(output_budget_tokens)
    # The context envelope covers both the composed input and the reserved
    # completion.  Checking only the input admits a prompt that fits until the
    # model starts emitting the pair, which is exactly the late truncation this
    # contract is meant to prevent.
    return (request + output <= budget
            and (output_budget is None or output <= output_budget))


def window_source_words_tokens(capability_tokens=None, *, overhead_words=0,
                               output_words=0, share=WINDOW_SHARE) -> int:
    """Choose source words from a qualified token budget."""
    try:
        overhead = words_to_tokens(int(overhead_words) + int(output_words))
        fraction = float(share)
    except (TypeError, ValueError):
        return 0
    if overhead < 0 or not 0 < fraction <= 1:
        return 0
    available = budget_tokens(capability_tokens) - overhead
    return max(0, tokens_to_words(int(available * fraction)))


def window_source_words(capability_words=None, *, overhead_words=0,
                        output_words=0, share=WINDOW_SHARE) -> int:
    """Choose a deterministic source-window size that leaves room for output.

    The caller supplies the fixed prompt frame and expected window response.
    Only a share of what remains is used for source content, leaving room for
    the final pair and route-specific framing. A zero result means even one
    window cannot fit the supplied route contract.
    """
    budget = budget_words(capability_words)
    try:
        overhead = int(overhead_words)
        output = int(output_words)
        fraction = float(share)
    except (TypeError, ValueError):
        return 0
    if overhead < 0 or output < 0 or not 0 < fraction <= 1:
        return 0
    return max(0, int((budget - overhead - output) * fraction))


def direct_ok(source_words, overhead_words=0, capability_words=None, *,
              output_words=0, output_budget_words=None) -> bool:
    """True when source plus overhead fits the direct route.

    `capability_words` is an optional positive route-provided input budget;
    any other value falls back to the baseline. `output_budget_words` is an
    optional positive route-provided output budget; when absent, no output
    cap is applied. Unusable inputs fail closed toward windows.
    """
    budget = budget_words(capability_words)
    try:
        total = int(source_words) + int(overhead_words)
        out = int(output_words)
    except (TypeError, ValueError):
        return False
    if not 0 <= total <= budget or out < 0:
        return False
    output_budget = explicit_output_budget(output_budget_words)
    return output_budget is None or out <= output_budget
