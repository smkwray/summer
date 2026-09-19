#!/usr/bin/env python3
"""No-model, real-process Windows proof for Summer's admission path.

Run this on Windows from the checkout that AutoHotkey uses::

    python scripts\\windows_proof.py

The probe refuses to run on non-Windows hosts.  It uses the real ``msvcrt``
backend in ``src/runtime.py`` and separate Python processes for every lock
assertion; no lock backend is mocked and no model CLI is invoked.  A temporary
``LOCALAPPDATA`` tree keeps the proof isolated from a user's Summer state.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time
import types


if not sys.platform.startswith("win"):
    raise SystemExit("REFUSED: scripts/windows_proof.py must run on Windows")


ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

# The child processes inherit this value.  runtime.app_dir() reads it on every
# call, so the proof does not touch the installed device's real lock directory.
runtime_app = os.environ.get("SUMMER_WINDOWS_PROOF_APP")
if runtime_app:
    os.environ["LOCALAPPDATA"] = runtime_app

import runtime  # noqa: E402  (import after the Windows guard and env setup)


class ProbeFailure(RuntimeError):
    pass


def require(condition: bool, message: str):
    if not condition:
        raise ProbeFailure(message)


def wait_for(path: pathlib.Path, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return True
        time.sleep(0.03)
    return path.exists()


def child_context(kind: str, key: str, blocking: bool = True):
    if kind == "destination":
        return runtime.destination_lock(pathlib.Path(key))
    if kind == "work":
        # Use the production public API on both sides. In particular it
        # case-normalizes Windows paths before hashing the lock key.
        return runtime.work_root_lock(pathlib.Path(key))
    if kind == "output":
        return runtime.output_directory_lock(pathlib.Path(key))
    raise ProbeFailure(f"unknown file lock kind: {kind}")


def child_main(argv: list[str]) -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="op", required=True)

    h = sub.add_parser("hold-slot")
    h.add_argument("kind"); h.add_argument("key"); h.add_argument("capacity", type=int)
    h.add_argument("ready"); h.add_argument("release")

    a = sub.add_parser("acquire-slot")
    a.add_argument("kind"); a.add_argument("key"); a.add_argument("capacity", type=int)
    a.add_argument("acquired")

    f = sub.add_parser("hold-file")
    f.add_argument("kind"); f.add_argument("key"); f.add_argument("ready"); f.add_argument("release")

    t = sub.add_parser("try-file")
    t.add_argument("kind"); t.add_argument("key"); t.add_argument("result")

    x = sub.add_parser("allocate-pasted")
    x.add_argument("directory"); x.add_argument("result")

    u = sub.add_parser("ui-child")
    u.add_argument("--noop", action="store_true")

    ns = p.parse_args(argv)
    if ns.op == "hold-slot":
        with runtime.slot_lease(ns.kind, ns.key, ns.capacity):
            pathlib.Path(ns.ready).write_text("ready", encoding="ascii")
            while not pathlib.Path(ns.release).exists():
                time.sleep(0.03)
        return 0
    if ns.op == "acquire-slot":
        with runtime.slot_lease(ns.kind, ns.key, ns.capacity):
            pathlib.Path(ns.acquired).write_text("acquired", encoding="ascii")
        return 0
    if ns.op == "hold-file":
        with child_context(ns.kind, ns.key):
            pathlib.Path(ns.ready).write_text("ready", encoding="ascii")
            while not pathlib.Path(ns.release).exists():
                time.sleep(0.03)
        return 0
    if ns.op == "try-file":
        try:
            if ns.kind == "work":
                ctx = runtime.work_root_lock(pathlib.Path(ns.key))
            else:
                # Destination publication deliberately waits rather than
                # returning BusyError. The parent proves it stays blocked until
                # the holder releases; only explicit work roots refuse.
                ctx = child_context(ns.kind, ns.key)
            with ctx:
                pathlib.Path(ns.result).write_text("acquired", encoding="ascii")
        except runtime.BusyError:
            pathlib.Path(ns.result).write_text("busy", encoding="ascii")
        return 0
    if ns.op == "allocate-pasted":
        # Importing the CLI is safe here: this path only publishes deterministic
        # fixture bytes and never reaches mapsum.run().
        import summ_cli

        directory = pathlib.Path(ns.directory)
        directory.mkdir(parents=True, exist_ok=True)
        detailed = directory / f"fixture-{os.getpid()}.d"
        brief = directory / f"fixture-{os.getpid()}.b"
        detailed.write_text("Detailed fixture artifact.", encoding="utf-8")
        brief.write_text("A pasted fixture reading.", encoding="utf-8")
        base = directory / "topic.summary.md"
        with runtime.output_directory_lock(directory):
            dest = summ_cli.free_path(base, set())
            with runtime.destination_lock(dest):
                bdest = summ_cli.publish_pair(detailed, brief, dest)
        pathlib.Path(ns.result).write_text(
            json.dumps({"detailed": dest.name, "brief": bdest.name}),
            encoding="utf-8")
        return 0
    if ns.op == "ui-child":
        return 0
    raise ProbeFailure(f"unhandled child operation: {ns.op}")


def command(*args: str) -> list[str]:
    return [sys.executable, str(pathlib.Path(__file__).resolve()), "--child", *args]


def spawn(*args: str) -> subprocess.Popen:
    return subprocess.Popen(command(*args), stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE, text=True)


def finish(proc: subprocess.Popen, timeout: float = 5.0):
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, err = proc.communicate()
        raise ProbeFailure(f"child timed out: {err[-400:]}")
    require(proc.returncode == 0,
            f"child exited {proc.returncode}: {(err or '')[-400:]}")


def hold_and_contend_slot(root: pathlib.Path, kind: str, key: str,
                          capacity: int, label: str):
    ready = root / f"{label}.ready"
    release = root / f"{label}.release"
    acquired = root / f"{label}.acquired"
    holder = spawn("hold-slot", kind, key, str(capacity), str(ready), str(release))
    require(wait_for(ready), f"{label}: holder did not acquire its slot")
    contender = spawn("acquire-slot", kind, key, str(capacity), str(acquired))
    time.sleep(0.35)
    require(not acquired.exists(), f"{label}: contender entered a full slot")
    release.write_text("release", encoding="ascii")
    require(wait_for(acquired), f"{label}: contender did not enter after release")
    finish(contender)
    finish(holder)


def prove_two_target_slots(root: pathlib.Path):
    ready = [root / "target-a.ready", root / "target-b.ready"]
    release = [root / "target-a.release", root / "target-b.release"]
    acquired = root / "target-third.acquired"
    holders = [spawn("hold-slot", "target", "host", "2", str(ready[i]),
                      str(release[i])) for i in range(2)]
    try:
        for path in ready:
            require(wait_for(path), "target capacity 2: both workers did not enter")
        contender = spawn("acquire-slot", "target", "host", "2", str(acquired))
        time.sleep(0.35)
        require(not acquired.exists(), "target capacity 2: third worker entered")
        release[0].write_text("release", encoding="ascii")
        require(wait_for(acquired), "target capacity 2: third worker did not enter after one release")
        finish(contender)
        release[1].write_text("release", encoding="ascii")
        for holder in holders:
            finish(holder)
    finally:
        for path in release:
            path.write_text("release", encoding="ascii")
        for proc in holders:
            if proc.poll() is None:
                proc.kill(); proc.wait()


def prove_process_death(root: pathlib.Path):
    ready = root / "death.ready"
    release = root / "death.release"
    acquired = root / "death.acquired"
    holder = spawn("hold-slot", "resource", "death-harness", "1",
                   str(ready), str(release))
    require(wait_for(ready), "process death: holder did not acquire")
    holder.kill()
    holder.communicate(timeout=5)
    contender = spawn("acquire-slot", "resource", "death-harness", "1",
                      str(acquired))
    require(wait_for(acquired), "process death: lock was not released by process death")
    finish(contender)


def prove_file_exclusion(root: pathlib.Path, kind: str, key: str, label: str):
    ready, release, result = (root / f"{label}.ready", root / f"{label}.release",
                              root / f"{label}.result")
    holder = spawn("hold-file", kind, key, str(ready), str(release))
    contender = None
    try:
        require(wait_for(ready), f"{label}: holder did not acquire")
        contender = spawn("try-file", kind, key, str(result))
        if kind == "work":
            require(wait_for(result), f"{label}: contender did not report")
            require(result.read_text(encoding="ascii") == "busy",
                    f"{label}: duplicate work root was not refused")
        else:
            time.sleep(0.35)
            require(not result.exists(),
                    f"{label}: second process entered before release")
        release.write_text("release", encoding="ascii")
        if kind != "work":
            require(wait_for(result),
                    f"{label}: waiting process did not enter after release")
        finish(holder); finish(contender)
    finally:
        release.write_text("release", encoding="ascii")
        for proc in (holder, contender):
            if proc is not None and proc.poll() is None:
                proc.kill(); proc.wait()
    # For a blocking destination lock, the first contender entering after the
    # holder releases already proves release. A nonblocking work-root refusal
    # needs a fresh attempt to prove BusyError does not persist.
    if kind == "work":
        result.unlink(missing_ok=True)
        second = spawn("try-file", kind, key, str(result))
        require(wait_for(result), f"{label}: released lock did not become available")
        require(result.read_text(encoding="ascii") == "acquired",
                f"{label}: released lock remained busy")
        finish(second)


def prove_pair_rollback(root: pathlib.Path):
    import summ_cli

    d = root / "pair"
    d.mkdir()
    new_d, new_b = d / "new.d", d / "new.b"
    dest, bdest = d / "out.summary.md", d / "out.brief.md"
    new_d.write_text("NEW detailed fixture.", encoding="utf-8")
    new_b.write_text("NEW brief fixture.", encoding="utf-8")
    dest.write_text("OLD detailed fixture.", encoding="utf-8")
    bdest.write_text("OLD brief fixture.", encoding="utf-8")

    def fail_second(src, dst):
        if pathlib.Path(dst) == bdest and ".tmp" in pathlib.Path(src).name:
            raise OSError("injected second replacement failure")
        os.replace(src, dst)

    try:
        summ_cli.publish_pair(new_d, new_b, dest, fail_second)
    except OSError:
        pass
    else:
        raise ProbeFailure("pair rollback: injected publication failure was ignored")
    require(dest.read_text(encoding="utf-8") == "OLD detailed fixture.",
            "pair rollback: old Detailed was not restored")
    require(bdest.read_text(encoding="utf-8") == "OLD brief fixture.",
            "pair rollback: old Brief was not restored")
    require(not list(d.glob("*.tmp*")) and not list(d.glob("*.bak*")),
            "pair rollback: temporary publication files remain")


def prove_distinct_pasted_allocations(root: pathlib.Path):
    d = root / "pasted"
    d.mkdir()
    results = [root / "paste-a.json", root / "paste-b.json"]
    workers = [spawn("allocate-pasted", str(d), str(path)) for path in results]
    for path in results:
        require(wait_for(path), "pasted allocation: worker did not publish")
    for worker in workers:
        finish(worker)
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in results]
    detailed = {row["detailed"] for row in rows}
    require(detailed == {"topic.summary.md", "topic-2.summary.md"},
            f"pasted allocation: unexpected Detailed names {detailed}")
    for row in rows:
        require(row["brief"].replace(".brief.md", "") ==
                row["detailed"].replace(".summary.md", ""),
                "pasted allocation: Detailed/Brief bases do not match")


def prove_ui_isolation(root: pathlib.Path):
    """Drive two actual Tk JobStates with fake local children, never models."""
    try:
        import tkinter as tk
        import summ_ui
    except Exception as e:
        raise ProbeFailure(f"UI proof cannot import Tk: {e}") from e
    try:
        window = tk.Tk()
    except tk.TclError as e:
        raise ProbeFailure(f"UI proof needs a Windows desktop session: {e}") from e

    runtime_app_dir = runtime.app_dir
    runtime.app_dir = lambda: root / "summer"
    (root / "summer").mkdir(parents=True, exist_ok=True)
    (root / "summer" / "runtime.json").write_text(
        json.dumps({"active_targets": 2}), encoding="utf-8")
    states = []
    try:
        window.withdraw()
        app = summ_ui.App(window)

        def fake_run(self, state, _cmd, _env=None):
            states.append(state)
            events = [
                {"schema": "summer.progress.v1", "event": "target_started",
                 "index": 1, "count": 1, "title": state.spec["label"], "words": 10},
                {"schema": "summer.progress.v1", "event": "part", "index": 1, "total": 2},
                {"schema": "summer.progress.v1", "event": "model_call_started",
                 "role": state.spec["label"], "harness": "fixture", "model": "fixture"},
                {"schema": "summer.progress.v1", "event": "target_finished",
                 "index": 1, "status": "succeeded", "exit_code": 0, "outputs": []},
                {"schema": "summer.progress.v1", "event": "job_finished",
                 "status": "succeeded", "exit_code": 0},
            ]
            (state.root / "progress.jsonl").write_text(
                "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
            state.events.put(("line", state.spec["label"]))
            state.events.put(("exit", 0))

        app._run = types.MethodType(fake_run, app)
        quick_mode = summ_ui.MODES[1][0]
        fake = lambda label: {"mode": quick_mode, "source": [], "pasted": "fixture",
                              "out": None, "env": {}, "label": label}
        app._launch(fake("job-a")); app._launch(fake("job-b"))
        deadline = time.monotonic() + 5
        max_active = 0
        while app.active and time.monotonic() < deadline:
            max_active = max(max_active, len(app.active))
            window.update()
            time.sleep(0.03)
        require(not app.active, "UI proof: both fake children did not finish")
        require(max_active >= 2, "UI proof: second job was not active concurrently")
        require(len(states) == 2 and {s.spec["label"] for s in states} == {"job-a", "job-b"},
                "UI proof: per-job state was not retained separately")
        for state in states:
            require(state.parts == (1, 2), f"UI proof: {state.spec['label']} lost its part state")
            require(state.call and state.call.startswith(state.spec["label"]),
                    f"UI proof: {state.spec['label']} received another job's call state")
    finally:
        runtime.app_dir = runtime_app_dir
        window.destroy()


def run() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    ns = parser.parse_args()
    if ns.child is not None:
        return child_main(ns.child)

    with tempfile.TemporaryDirectory(prefix="summer-windows-proof-") as temp:
        root = pathlib.Path(temp)
        os.environ["LOCALAPPDATA"] = str(root)
        prove_two_target_slots(root)
        print("PASS target capacity 2: two processes admitted, third waited")
        hold_and_contend_slot(root, "resource", "same-harness", 1, "resource")
        print("PASS resource capacity 1: same-harness calls excluded")
        prove_process_death(root)
        print("PASS process death: msvcrt lease released")
        prove_file_exclusion(root, "destination", str(root / "same.summary.md"), "destination")
        print("PASS destination pair exclusion: second publisher excluded")
        prove_file_exclusion(root, "work", str(root / "shared-work"), "work")
        print("PASS explicit work-root reservation: duplicate refused")
        prove_pair_rollback(root)
        print("PASS pair publication rollback: old pair restored")
        prove_distinct_pasted_allocations(root)
        print("PASS pasted allocation: distinct matching Detailed/Brief bases")
        prove_ui_isolation(root)
        print("PASS Tk UI isolation: two fake children active with separate state")
    print("Windows proof complete: real msvcrt, real subprocesses, zero model calls")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except ProbeFailure as e:
        print(f"FAIL: {e}", file=sys.stderr)
        raise SystemExit(1)
