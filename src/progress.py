#!/usr/bin/env python3
"""Machine-readable run progress, for a UI watching a multi-minute job.

A run is 5-15 minutes and prints only human lines. A client that wants to show
"part 7 of 12" has to parse prose written for a person, which has no schema and
changes whenever the wording improves. This is the schema.

Enabled by `summ_cli.py --progress-jsonl PATH`, which exports SUMM_PROGRESS so
the stage subprocesses inherit it. Absent, every emit() is a no-op and nothing
about existing behaviour changes -- both installed triggers pass no such flag.

The stream is still deliberately smaller than a full execution protocol, but
selection and document identity are explicit in v2. Paths may appear in local
run evidence; they are never model input or ETA history.

Events, all carrying `schema`, `event` and `time_utc`:

    job_started      targets, mode, scope
    selection_resolved  roots, documents, excluded, problems, order_digest, scope
    document_started    document_id, index, count, label, words
    target_started   index, count, title, words
    stage            name in {reader_view, ledger, compose, text_prep, publish,
                              tts_normalize, full, full-windowed, corpus_preflight,
                              corpus_inventory, corpus_plan, corpus_overview,
                              corpus_seal}
    part             index, total
    corpus_preflight_finished documents, visible_words, inventory_windows
    model_call_started  role, harness, model, attempt
    model_call       role, harness, model, attempt, outcome, seconds
    eta               lower_s, upper_s, samples, confidence, text
    target_finished  index, status, exit_code, outputs, destination_unchanged
    document_finished document_id, index, status, exit_code, failure_kind
    job_finished     status, exit_code

Progress is evidence, not correctness: a write failure warns once and never
fails a run that is otherwise fine.
"""
from __future__ import annotations
import datetime, json, os, pathlib

SCHEMA = "summer.progress.v2"
ENV = "SUMM_PROGRESS"
_warned = False


def path() -> pathlib.Path | None:
    p = os.environ.get(ENV)
    return pathlib.Path(p) if p else None


def start(dest, mode: str, targets: int, scope: str = "batch") -> int:
    """Create the stream before any work. Returns 0, or 2 if it cannot be made.

    Fails BEFORE staging a source or spending a model call: a client that asked
    to be told what is happening and silently is not told would rather learn
    that now than after a six-minute wait.
    """
    p = pathlib.Path(dest).expanduser()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("")
    except Exception as e:
        print(f"cannot write progress stream {p}: {str(e)[:80]}", flush=True)
        return 2
    os.environ[ENV] = str(p)
    emit("job_started", mode=mode, targets=targets, scope=scope)
    return 0


def emit(event: str, **fields):
    global _warned
    p = path()
    if p is None:
        return
    rec = {"schema": SCHEMA, "event": event,
           "time_utc": datetime.datetime.now(datetime.timezone.utc)
                       .isoformat(timespec="milliseconds").replace("+00:00", "Z"),
           **fields}
    try:
        with p.open("a") as fh:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
    except Exception as e:
        if not _warned:
            _warned = True
            print(f"progress stream stopped: {str(e)[:80]}", flush=True)
