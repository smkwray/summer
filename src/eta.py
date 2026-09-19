#!/usr/bin/env python3
"""Bounded, content-free timing history and conservative ETA ranges."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import pathlib
import re
import statistics

import mode_config
import model_config
import runtime

PRODUCER_REV = 3
MAX_RECORDS = 512
COMPACT_AT = 640
MAX_LINE_BYTES = 64 * 1024
MIN_SAMPLES = 8
MAX_AGE_DAYS = 60
HALF_LIFE_DAYS = 30.0
DRIFT_LIMIT = 0.35
_UNSET = object()


def history_path() -> pathlib.Path:
    return runtime.app_dir() / "eta-history.jsonl"


def _sha(value) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _resource_signature(resources) -> str:
    """Opaque signature for the capacities that can affect a target."""
    return _sha(sorted((str(resource), int(capacity))
                       for resource, capacity in resources))


def _roles_for_route(route: str) -> tuple[str, ...]:
    """Resolve an ETA implementation route through the product mode registry."""
    if route.endswith("_deterministic"):
        return ()
    key = "summarize" if route == "ledger" else route.removesuffix("_model")
    mode = mode_config.BY_KEY.get(key)
    if mode is None:
        mode = next((item for item in mode_config.MODES
                     if item.progress_route == route), None)
    return mode.roles if mode is not None else ()


def _runtime_revision_signature(rt: dict, selected) -> str:
    """Hash only declared revisions for the selected gateway/model routes.

    Revision labels may describe private deployment state, so neither labels
    nor the surrounding runtime configuration are returned or persisted.
    """
    rows = []
    gateways = rt.get("gateways") or {}
    for harness, model in sorted(set(selected)):
        gateway = gateways.get(harness)
        if not isinstance(gateway, dict):
            continue
        model_route = (gateway.get("models") or {}).get(model)
        model_route = model_route if isinstance(model_route, dict) else {}
        gateway_revision = gateway.get("revision")
        model_revision = model_route.get("revision")
        if gateway_revision is not None or model_revision is not None:
            rows.append((harness, model, gateway_revision, model_revision))
    return _sha(rows)


def execution_signature(route: str, instructions_digest=_UNSET) -> str:
    """Hash every setting that can make two timings non-comparable.

    Custom instructions are represented only by their opaque digest.  The
    request text never enters the device-local ETA history.
    """
    if instructions_digest is _UNSET:
        instructions_digest = os.environ.get("SUMM_INSTRUCTIONS_DIGEST")
    roster_by_harness = model_config.roster()
    harness = os.environ.get("HARNESS", "agy")
    if harness not in roster_by_harness:
        raise runtime.ConfigError(f"unknown ETA harness {harness!r}")
    chains = model_config.role_chains(
        harness, _roles_for_route(route), roster_value=roster_by_harness)
    import mapsum
    selected = {role: [model_config.split_entry(
                entry, harness, roster_by_harness) for entry in entries]
                for role, entries in chains.items()}
    used_harnesses = {item[0] for entries in selected.values() for item in entries}
    options = {
        role: [{"harness": selected_harness, "model": model,
                "option": mapsum.effective_option(
                    selected_harness, model, role)}
               for selected_harness, model in entries]
        for role, entries in selected.items()
    }
    rt = runtime.config()
    bindings = {h: runtime.resource_name(h, rt)
                for h in used_harnesses if h}
    capacities = {h: rt["capacities"].get(bindings[h], rt["harness_capacity"])
                  for h in bindings}
    overrides = {key: os.environ[key] for key in (
        "MODEL", "PLAN_MODEL", "AUDIT_MODEL", "REPAIR_MODEL",
        "RETRIES", "CALL_TIMEOUT", "SUMM_ROLE_OPTIONS")
                if key in os.environ}
    return _sha({"producer_rev": PRODUCER_REV, "route": route,
                 "chains": chains, "options": options,
                 # Use the effective values from the same loaded module that
                 # executes calls. Repeating defaults here once mislabeled the
                 # real 1,800-second timeout as 1,200 seconds.
                 "retry": mapsum.RETRIES,
                 "timeout": mapsum.CALL_TIMEOUT,
                 "local_only": bool(os.environ.get("SUMM_LOCAL_ONLY")),
                 "instructions_digest": instructions_digest,
                 "overrides": overrides,
                 "runtime_revision_sig": _runtime_revision_signature(
                     rt, (item for entries in selected.values() for item in entries)),
                 "bindings": bindings, "capacities": capacities})


def _role(stage: str) -> str:
    return re.sub(r"\d+$", "", stage or "unknown")


def _call_rows(run_dir: pathlib.Path) -> tuple[list, str | None]:
    """Read call evidence without converting missing data into zeros.

    The long-lived history is allowed to omit the call payload, but it must not
    pretend that an unreadable or incomplete payload took zero seconds.  A
    target can still publish successfully when instrumentation is damaged; its
    history row is then explicitly ineligible for ETA.
    """
    p = pathlib.Path(run_dir) / "calls.jsonl"
    if not p.is_file():
        return [], "calls.jsonl missing"
    lines = p.read_text(errors="replace").splitlines()
    if not lines:
        return [], "calls.jsonl empty"
    calls = []
    for index, line in enumerate(lines, 1):
        try:
            row = json.loads(line)
        except Exception:
            return calls, f"calls.jsonl line {index} is malformed"
        if not isinstance(row, dict):
            return calls, f"calls.jsonl line {index} is not an object"
        seconds = row.get("seconds")
        if (not isinstance(seconds, (int, float)) or isinstance(seconds, bool)
                or not math.isfinite(float(seconds)) or float(seconds) < 0):
            return calls, f"calls.jsonl line {index} has invalid seconds"
        if not row.get("stage") or not row.get("outcome"):
            return calls, f"calls.jsonl line {index} lacks stage or outcome"
        calls.append(row)
    return calls, None


def aggregate_calls(run_dir: pathlib.Path) -> tuple[dict, list, int | None]:
    calls, _ = _call_rows(run_dir)
    outcomes, by_role = {}, {}
    for row in calls:
        outcome = str(row.get("outcome", "unknown"))
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        role = _role(str(row.get("stage", "unknown")))
        bucket = by_role.setdefault(role, {"attempts": 0, "ok": 0, "seconds": 0.0})
        bucket["attempts"] += 1
        bucket["ok"] += outcome == "ok"
        bucket["seconds"] += float(row["seconds"])
    for bucket in by_role.values():
        bucket["seconds"] = round(bucket["seconds"], 3)
    resources = sorted({str(row.get("harness")) for row in calls if row.get("harness")})
    parts = len({_role(str(row.get("stage"))) + re.sub(r"\D", "", str(row.get("stage")))
                 for row in calls if re.match(r"plan\d+$", str(row.get("stage", "")))}) or None
    return ({"attempts": len(calls), "ok": outcomes.get("ok", 0),
             "seconds": round(sum(float(r["seconds"]) for r in calls), 3),
             "outcomes": outcomes, "by_role": by_role}, resources, parts)


def _append(record: dict):
    p = history_path(); p.parent.mkdir(parents=True, exist_ok=True)
    with runtime.file_lock("history", str(p.resolve())):
        rows = []
        damaged = False
        if p.is_file():
            raw = p.read_text(errors="replace")
            damaged = bool(raw and not raw.endswith("\n"))
            for line in raw.splitlines():
                if not line.strip():
                    damaged = True
                    continue
                if len(line.encode("utf-8", "replace")) > MAX_LINE_BYTES:
                    damaged = True
                    continue
                try:
                    row = json.loads(line)
                    if (isinstance(row, dict)
                            and row.get("schema") == "summer.eta.target.v1"):
                        rows.append(json.dumps(row, sort_keys=True,
                                                separators=(",", ":")))
                    else:
                        damaged = True
                except Exception:
                    damaged = True
        rows.append(json.dumps(record, sort_keys=True, separators=(",", ":")))
        if damaged or len(rows) > MAX_RECORDS or len(rows) >= COMPACT_AT:
            rows = rows[-MAX_RECORDS:]
            tmp = p.with_name(p.name + f".tmp{os.getpid()}")
            tmp.write_text("\n".join(rows) + "\n")
            os.replace(tmp, p)
        else:
            with p.open("a") as fh:
                fh.write(rows[-1] + "\n")


def record_target(run_dir: pathlib.Path, *, route: str, source_words: int,
                  execution_sig: str, outcome: str, service_wall_s: float,
                  admission_wait_s: float, parts: int | None = None):
    rows, evidence_error = _call_rows(run_dir)
    calls, harnesses, derived_parts = aggregate_calls(run_dir)
    rt = runtime.config()
    resources = []
    for harness in harnesses:
        resource = rt["bindings"].get(harness, harness)
        capacity = rt["capacities"].get(resource, rt["harness_capacity"])
        resources.append([_sha(resource), capacity])
    _, resource_sig = _current_runtime_signature()
    metadata_ok = evidence_error is None and (
        bool(rows) or route == "tts_deterministic")
    metadata_status = "eligible" if metadata_ok else "ineligible"
    record = {
        "schema": "summer.eta.target.v1", "producer_rev": PRODUCER_REV,
        "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds"),
        "route": route, "source_words": int(source_words),
        "parts": parts if parts is not None else derived_parts,
        "execution_sig": execution_sig,
        "scheduler": {"target_capacity": rt["active_targets"],
                      "resources": resources,
                      "admission_wait_s": round(admission_wait_s, 3),
                      "resource_wait_s": round(sum(
                          float(r.get("resource_wait_s", 0))
                          for r in rows
                          if isinstance(r.get("resource_wait_s", 0), (int, float))), 3),
                      "resource_signature": resource_sig},
        "outcome": outcome,
        "metadata_status": metadata_status,
        "metadata_reason": None if metadata_ok else evidence_error or "no call evidence",
        "service_wall_s": round(service_wall_s, 3),
        "calls": calls,
    }
    _append(record)


def _read_calls(run_dir):
    return _call_rows(run_dir)[0]


def _weighted_quantile(rows, q, value_key="service_wall_s"):
    ordered = sorted(rows, key=lambda r: float(r[value_key]))
    total = sum(float(r["_weight"]) for r in ordered)
    if not ordered or total <= 0:
        return None
    target = total * q
    seen = 0.0
    for row in ordered:
        seen += float(row["_weight"])
        if seen >= target:
            return float(row[value_key])
    return float(ordered[-1][value_key])


def _round_range(seconds, up=False):
    step = 30 if seconds < 600 else 60 if seconds < 3600 else 300
    units = math.ceil(seconds / step) if up else math.floor(seconds / step)
    return max(step, units * step)


def _human(seconds):
    if seconds < 3600:
        return f"{int(round(seconds / 60))} min"
    hours, mins = divmod(int(round(seconds / 60)), 60)
    return f"{hours}h {mins:02d}m"


def _current_runtime_signature():
    rt = runtime.config()
    labels = set(rt["capacities"])
    labels.update(rt["bindings"].values())
    resources = [[_sha(resource),
                  rt["capacities"].get(resource, rt["harness_capacity"])]
                 for resource in sorted(labels)]
    return rt["active_targets"], _resource_signature(resources)


def _matching_rows(*, route, source_words, execution_sig, parts):
    p = history_path()
    if not p.is_file():
        return []
    target_capacity, resource_sig = _current_runtime_signature()
    now = dt.datetime.now(dt.timezone.utc)
    rows = []
    # Hold the same lock used by append/compact so a live estimator never sees
    # a half-written final JSONL line.
    with runtime.file_lock("history", str(p.resolve())):
        text = p.read_text(errors="replace")
    for line in text.splitlines()[-MAX_RECORDS:]:
        if len(line.encode("utf-8", "replace")) > MAX_LINE_BYTES:
            continue
        try:
            row = json.loads(line)
            completed = dt.datetime.fromisoformat(row["completed_utc"])
        except Exception:
            continue
        age = (now - completed).total_seconds() / 86400
        if (age < 0 or age > MAX_AGE_DAYS
                or row.get("producer_rev") != PRODUCER_REV
                or row.get("route") != route
                or row.get("execution_sig") != execution_sig
                or (row.get("metadata_status") != "eligible"
                    and row.get("outcome") == "succeeded")):
            continue
        if not _valid_history_row(row):
            continue
        scheduler = row.get("scheduler") or {}
        recorded_resource_sig = scheduler.get("resource_signature")
        if recorded_resource_sig is None:
            # Accept the v1 candidate's rows, where the resource signature was
            # not yet named, but still compare the captured opaque resource
            # tuples when they exist.
            recorded_resource_sig = _resource_signature(scheduler.get("resources") or [])
        if (scheduler.get("target_capacity") != target_capacity
                or recorded_resource_sig != resource_sig):
            continue
        row["_age_days"] = age
        row["_weight"] = math.exp(-math.log(2) * age / HALF_LIFE_DAYS)
        words = int(row.get("source_words") or 0)
        if not words:
            continue
        word_ratio = max(words, source_words) / max(1, min(words, source_words))
        row["_word_ratio"] = word_ratio
        row["_parts_delta"] = (abs((row.get("parts") or 0) - (parts or 0))
                                if parts is not None and row.get("parts") is not None else 0)
        if parts is not None and row.get("parts") != parts:
            if route != "ledger" or row.get("parts") is None or row["_parts_delta"] != 1:
                continue
        if word_ratio <= 1.25:
            row["_match"] = "near"
        elif word_ratio <= 1.5:
            row["_match"] = "wide"
        else:
            continue
        rows.append(row)
    return rows


def _success_rows(rows):
    return [row for row in rows if row.get("outcome") == "succeeded"
            and row.get("metadata_status") == "eligible"
            and isinstance(row.get("service_wall_s"), (int, float))
            and math.isfinite(float(row["service_wall_s"]))
            and float(row["service_wall_s"]) >= 0]


def _valid_history_row(row):
    calls = row.get("calls")
    wall = row.get("service_wall_s")
    if not isinstance(calls, dict) or not isinstance(wall, (int, float)):
        return False
    if isinstance(wall, bool) or not math.isfinite(float(wall)) or float(wall) < 0:
        return False
    for key in ("attempts", "ok", "seconds"):
        value = calls.get(key)
        if (not isinstance(value, (int, float)) or isinstance(value, bool)
                or not math.isfinite(float(value)) or float(value) < 0):
            return False
    return isinstance(calls.get("by_role"), dict)


def _drifted(rows):
    successes = sorted(_success_rows(rows), key=lambda r: r["completed_utc"], reverse=True)
    if len(successes) < 12:
        return False
    newest = [float(r["service_wall_s"]) for r in successes[:4]]
    previous = [float(r["service_wall_s"]) for r in successes[4:12]]
    if not previous or not newest:
        return False
    return abs(statistics.median(newest) - statistics.median(previous)) / max(
        1.0, statistics.median(previous)) > DRIFT_LIMIT


def estimate(*, route: str, source_words: int, execution_sig: str,
             parts: int | None = None):
    """Return no estimate until eight recent, exact execution cohorts exist."""
    if route == "ledger" and parts is None:
        return None
    rows = _matching_rows(route=route, source_words=source_words,
                          execution_sig=execution_sig, parts=parts)
    exact = [r for r in rows if r["_match"] == "near"
             and (parts is None or r.get("parts") == parts)]
    cohort = exact
    if len(exact) < MIN_SAMPLES:
        if len(_success_rows(rows)) < 12:
            return None
        cohort = [r for r in rows if r["_match"] in ("near", "wide")]
    terminal = sorted(rows, key=lambda r: r["completed_utc"], reverse=True)[:12]
    if len(terminal) >= 12 and sum(r.get("outcome") == "succeeded" for r in terminal) / len(terminal) < .8:
        return None
    if _drifted(rows):
        return None
    successes = _success_rows(cohort)
    if len(successes) < MIN_SAMPLES:
        return None
    successes.sort(key=lambda r: (r["_word_ratio"], r["_age_days"]))
    successes = successes[:32]
    low_raw = _weighted_quantile(successes, .1)
    high_raw = _weighted_quantile(successes, .9)
    if low_raw is None or high_raw is None:
        return None
    low, high = _round_range(low_raw), _round_range(high_raw, True)
    if high / max(1, low) > 3:
        return None
    return {"lower_s": low, "upper_s": high, "samples": len(successes),
            "confidence": "likely" if len(successes) >= 20 else "rough",
            "text": f"{'Likely' if len(successes) >= 20 else 'Roughly'} "
                    f"{_human(low)}–{_human(high)} · {len(successes)} comparable runs"}


def live_remaining(run_dir: pathlib.Path, *, route: str, source_words: int,
                   execution_sig: str, parts: int | None = None):
    """Estimate remaining service time after a completed model call.

    This is intentionally conservative: a historical target is used only when
    the current call topology has not already exceeded that target's role
    attempts.  The caller may invoke this after every call; malformed current
    evidence simply suppresses the update.
    """
    rows = _matching_rows(route=route, source_words=source_words,
                          execution_sig=execution_sig, parts=parts)
    rows = [r for r in rows if r["_match"] == "near"
            and r.get("outcome") == "succeeded"
            and r.get("metadata_status") == "eligible"
            and (parts is None or r.get("parts") == parts)]
    if len(rows) < MIN_SAMPLES:
        return None
    rows = sorted(rows, key=lambda r: r["completed_utc"], reverse=True)[:32]
    current_rows, current_error = _call_rows(run_dir)
    if current_error:
        return None
    current = {}
    for row in current_rows:
        role = _role(str(row.get("stage", "unknown")))
        current[role] = current.get(role, 0) + 1
    residuals = []
    for row in rows:
        calls = row.get("calls") or {}
        by_role = calls.get("by_role") or {}
        remaining = max(0.0, float(row.get("service_wall_s", 0))
                        - float(calls.get("seconds", 0) or 0))
        excluded = False
        for role, bucket in by_role.items():
            attempts = int(bucket.get("attempts", 0) or 0)
            if current.get(role, 0) > attempts:
                excluded = True
                break
            if attempts:
                mean = float(bucket.get("seconds", 0) or 0) / attempts
                remaining += max(0, attempts - current.get(role, 0)) * mean
        if not excluded:
            item = dict(row)
            item["_remaining_s"] = remaining
            residuals.append(item)
    if len(residuals) < MIN_SAMPLES:
        return None
    low_raw = _weighted_quantile(residuals, .1, "_remaining_s")
    high_raw = _weighted_quantile(residuals, .9, "_remaining_s")
    if low_raw is None or high_raw is None:
        return None
    low, high = _round_range(low_raw), _round_range(high_raw, True)
    if high / max(1, low) > 3:
        return None
    confidence = "likely" if len(residuals) >= 20 else "rough"
    return {"lower_s": low, "upper_s": high, "samples": len(residuals),
            "confidence": confidence,
            "text": f"{'Likely' if confidence == 'likely' else 'Roughly'} "
                    f"{_human(low)}–{_human(high)} remaining · "
                    f"{len(residuals)} comparable runs"}
