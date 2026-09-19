#!/usr/bin/env python3
"""Cross-platform entry point for the summarization commands.

The triggers stay platform-specific (Raycast on macOS, AHK on Windows); this is
the single shared implementation behind both, so clipboard handling, staging,
gating and publication cannot drift between the two.

usage: summ_cli.py [--depth full|brief] [--tts] [paths ...]
       (with no paths, reads the clipboard)
"""
from __future__ import annotations
import argparse, hashlib, json, os, re, shutil, signal, subprocess, sys, tempfile, pathlib, time, uuid

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
import progress, runtime, eta, custom_instructions
import mode_config, model_config, profiles, selection
import fullsum
import route_caps


def configured_harnesses() -> tuple[str, ...]:
    try:
        cfg = model_config.roster()
    except Exception:
        return ()
    return tuple(name for name, value in cfg.items()
                 if isinstance(value, dict)
                 and any(value.get(role) for role in mode_config.ROLES))


HARNESSES = configured_harnesses()


def validate_roles(harness: str, roster: dict, roles) -> None:
    """Fail closed only for roles the target will actually call."""
    local_only = bool(os.environ.get("SUMM_LOCAL_ONLY"))
    local = {name for name, spec in roster.items()
             if isinstance(spec, dict) and spec.get("_local") is True}
    errors = []
    for role in roles:
        variable = f"{harness.upper()}_CHAIN_{role.upper()}"
        explicit = os.environ.get(variable)
        entries = (explicit.split() if explicit is not None
                   else list((roster.get(harness) or {}).get(role) or ()))
        if not entries:
            errors.append(f"{role} has no chain")
            continue
        for entry in entries:
            selected_harness, model = model_config.split_entry(
                entry, harness, roster)
            if selected_harness not in roster:
                errors.append(f"{role} names unknown harness {selected_harness!r}")
            elif model_config.logical_model(
                    selected_harness, model, roster) not in (
                    (roster[selected_harness].get(role)) or ()):
                errors.append(
                    f"{role} model {selected_harness}:{model} is not in its roster")
            elif local_only and selected_harness not in local:
                errors.append(
                    f"{role} model {selected_harness}:{model} is not local")
    if errors:
        raise runtime.ConfigError(
            f"harness {harness!r} cannot run active role(s): " + "; ".join(errors))


def freeze_model_routes(harness: str, roster: dict, roles) -> dict:
    """Freeze exact transport IDs and effective per-role options once.

    The UI, profiles and direct CLI all converge here.  Children and ETA then
    consume the same immutable route snapshot rather than resolving a logical
    family or ambient effort independently at several later call sites.
    """
    chains = model_config.role_chains(
        harness, roles, environ=os.environ, roster_value=roster)
    for role, entries in chains.items():
        os.environ[f"{harness.upper()}_CHAIN_{role.upper()}"] = " ".join(entries)

    # Load the one command/option authority only after HARNESS, exact chains,
    # active roles and the frozen runtime contract are present.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_summer_route_freeze", HERE / "mapsum.py")
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    frozen = {}
    for role, entries in chains.items():
        frozen[role] = []
        for entry in entries:
            selected_harness, model = model_config.split_entry(
                entry, harness, roster)
            frozen[role].append({
                "harness": selected_harness,
                "model": model,
                "option": engine.effective_option(
                    selected_harness, model, role),
            })
    os.environ["SUMM_ROLE_OPTIONS"] = json.dumps(
        frozen, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return chains

# `--quick` (src/shortsum.py) is an explicit mode, never an automatic
# rewrite: Summarize stays Full at every source size, however short the
# document is. A reader who asks for a summary is entitled to it, and the
# direct Full route writes whole-source pairs with no minimum length.


def route_capability():
    """Optional route-provided budgets for the Full routes; absent or
    unusable means the qualified baseline. Generic words, never a model."""
    def _int(name):
        try:
            value = int(os.environ.get(name, ""))
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None
    return (_int("SUMM_ROUTE_CAPABILITY_WORDS"),
            _int("SUMM_ROUTE_OUTPUT_WORDS"))


def route_token_capability(harness=None, roster=None, roles=()):
    """Optional qualified token envelope for the selected generic route.

    The older word variables remain readable for compatibility. New device
    profiles should use these token variables so admission does not confuse
    whitespace words with model tokens.
    """
    def _int(name):
        try:
            value = int(os.environ.get(name, ""))
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None
    context, output = (_int("SUMM_ROUTE_CONTEXT_TOKENS"),
                       _int("SUMM_ROUTE_OUTPUT_TOKENS"))
    if context is not None:
        return context, output
    if not harness:
        return context, output
    try:
        chains = model_config.role_chains(
            harness, roles=tuple(roles), environ=os.environ,
            roster_value=roster)
        declared = route_caps.primary_chain_capability(
            chains, model_config.split_entry, harness, os.environ)
    except (ValueError, runtime.ConfigError):
        # The child repeats this lookup and reports a specific configuration
        # error. The parent must remain able to perform its ordinary route
        # validation instead of converting malformed device data to a crash.
        return context, output
    if declared:
        return declared["context_tokens"], declared.get("output_tokens")
    return (context or route_caps.DEFAULT_QUALIFIED_CONTEXT_TOKENS), output
WIN = sys.platform.startswith("win")

SUMMARY_SUFFIX, BRIEF_SUFFIX = mode_config.BY_KEY["summarize"].suffixes
TTS_SUFFIX = mode_config.BY_KEY["tts"].suffixes[0]


class Cancelled(RuntimeError):
    pass


class PublicationError(OSError):
    """A publication failed, with an explicit rollback verdict."""

    def __init__(self, message: str, *, restored: bool, recovery_paths=()):
        super().__init__(message)
        self.restored = bool(restored)
        self.recovery_paths = tuple(pathlib.Path(p) for p in recovery_paths)


class CancellationToken:
    """One cooperative request shared by the UI marker and OS signals."""

    def __init__(self, path=None):
        self.path = pathlib.Path(path).expanduser() if path else None
        self._signalled = False

    def request(self):
        self._signalled = True

    def requested(self) -> bool:
        return self._signalled or bool(self.path and self.path.exists())

    def check(self):
        if self.requested():
            raise Cancelled("run cancelled")


def install_cancel_handlers(token: CancellationToken):
    """Signals request cancellation; they never interrupt publication code."""
    def request(_signum, _frame):
        token.request()

    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        value = getattr(signal, name, None)
        if value is not None:
            try:
                signal.signal(value, request)
            except (OSError, ValueError):
                pass


def _posix_group_alive(pgid: int) -> bool:
    """True while any process remains in a stage's private POSIX group."""
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _stop_stage_tree(proc: subprocess.Popen, grace_seconds: float = 3.0) -> bool:
    """Stop one stage and descendants; return only after tree death is observed."""
    if WIN:                                  # pragma: no cover - Windows
        # Do not first send CTRL_BREAK: the leader can exit before taskkill has
        # enumerated its descendants. /T snapshots and force-stops the tree while
        # the parent relation still exists.
        try:
            killed = subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=10).returncode == 0
        except subprocess.SubprocessError:
            killed = False
        try:
            proc.wait(timeout=10)
        except subprocess.SubprocessError:
            return False
        return killed and proc.poll() is not None

    pgid = proc.pid
    if proc.poll() is not None and not _posix_group_alive(pgid):
        return True
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        leader_alive = proc.poll() is None
        group_alive = _posix_group_alive(pgid)
        if not leader_alive and not group_alive:
            return True
        time.sleep(0.05)
    if _posix_group_alive(pgid):
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        proc.wait(timeout=10)
    except subprocess.SubprocessError:
        return False
    deadline = time.monotonic() + 10
    while _posix_group_alive(pgid) and time.monotonic() < deadline:
        time.sleep(0.05)
    return proc.poll() is not None and not _posix_group_alive(pgid)


def run_stage(argv, cancel: CancellationToken, **kwargs):
    """Run a cancellable worker group and return its exit status."""
    cancel.check()
    group = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
             if WIN else {"start_new_session": True})
    proc = subprocess.Popen(argv, **group, **kwargs)
    while True:
        try:
            result = subprocess.CompletedProcess(argv, proc.wait(timeout=0.1))
            if cancel.requested():
                while not _stop_stage_tree(proc):
                    print("  cancellation requested; waiting for the worker "
                          "process tree to stop", file=sys.stderr)
                raise Cancelled("run cancelled")
            return result
        except subprocess.TimeoutExpired:
            if cancel.requested():
                warned = False
                while not _stop_stage_tree(proc):
                    if not warned:
                        print("  cancellation requested; waiting for the worker "
                              "process tree to stop", file=sys.stderr)
                        warned = True
                raise Cancelled("run cancelled")


# ------------------------------------------------------------- clipboard ---
def clip_read() -> str:
    try:
        if WIN:
            r = subprocess.run(["powershell", "-NoProfile", "-Command",
                                "Get-Clipboard -Raw"], capture_output=True, text=True)
        elif sys.platform == "darwin":
            r = subprocess.run(["pbpaste"], capture_output=True, text=True)
        else:
            r = subprocess.run(["xclip", "-selection", "clipboard", "-o"],
                               capture_output=True, text=True)
        return r.stdout or ""
    except Exception:
        return ""


def notify(title: str, msg: str):
    # SUMM_QUIET silences desktop notifications. Set during benchmarking and
    # testing, where a run per candidate per document would otherwise fill the
    # notification centre with results nobody is waiting for.
    if os.environ.get("SUMM_QUIET"):
        return
    try:
        if WIN:
            # A balloon tip, NOT MessageBox::Show -- that is a modal the user must
            # dismiss, so every completion would block the desktop. macOS shows a
            # passive notification; Windows must behave the same way.
            safe_t = title.replace("'", "''")
            safe_m = msg.replace("'", "''")
            ps = ("Add-Type -AssemblyName System.Windows.Forms;"
                  "$n=New-Object System.Windows.Forms.NotifyIcon;"
                  "$n.Icon=[System.Drawing.SystemIcons]::Information;"
                  "$n.Visible=$true;"
                  f"$n.ShowBalloonTip(5000,'{safe_t}','{safe_m}',"
                  "[System.Windows.Forms.ToolTipIcon]::Info);"
                  "Start-Sleep -Seconds 5;$n.Dispose()")
            subprocess.Popen(["powershell", "-NoProfile", "-WindowStyle", "Hidden",
                              "-Command", ps],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif sys.platform == "darwin":
            subprocess.run(["osascript", "-e",
                            'display notification "{}" with title "{}"'.format(
                                msg.replace('"', "'"), title.replace('"', "'"))],
                           capture_output=True, timeout=5)
    except Exception:
        pass


# ------------------------------------------------------------- targets ---
def resolve(p: str) -> pathlib.Path:
    p = p.strip().strip("'\"")
    return pathlib.Path(os.path.expandvars(os.path.expanduser(p)))


DOC_SUFFIXES = selection.DOC_SUFFIXES
ARTIFACT_SUFFIXES = tuple(dict.fromkeys(
    suffix for mode in mode_config.MODES for suffix in mode.suffixes))


def expand(targets):
    """Compatibility wrapper around the shared recursive selection layer."""
    return [doc.source_path for doc in selection.resolve_paths(targets).documents]


def parse_targets(raw: str):
    """Legacy ``(targets, raw_text)`` view of :func:`selection.classify_clipboard`.

    New callers should retain the Selection object so they can show explicit
    exclusions and missing-path problems instead of throwing that information
    away.
    """
    chosen = selection.classify_clipboard(raw)
    if chosen.is_text:
        return ([raw] if raw.strip() else []), True
    return [doc.source_path for doc in chosen.documents], False


def stage(target, work: pathlib.Path, cancel: CancellationToken | None = None,
          expected_sha256: str | None = None, expected_size: int | None = None):
    """Freeze accepted source bytes, then expose UTF-8 text as ``source.txt``.

    The original bytes are staged before PDF extraction or any model-facing
    stage. This closes the race where a source could change after its hash was
    checked but before ``pdftotext`` or a text reader consumed it.
    """
    cancel = cancel or CancellationToken()
    src = pathlib.Path(target)
    # Directories are selection inputs, never documents.  The old first-file
    # fallback made a directory silently summarize an arbitrary child.
    if src.is_dir():
        return None
    if not src.is_file():
        return None
    work.mkdir(parents=True, exist_ok=True)
    original = work / "source.original"
    out = work / "source.txt"
    size = 0
    digest = hashlib.sha256()
    try:
        with src.open("rb") as source, original.open("wb") as frozen:
            while True:
                cancel.check()
                block = source.read(1024 * 1024)
                if not block:
                    break
                frozen.write(block)
                size += len(block)
                digest.update(block)
    except (OSError, Cancelled) as e:
        if isinstance(e, Cancelled):
            raise
        print(f"  {src.name}: source could not be read: {e}", file=sys.stderr)
        return None
    actual_sha256 = digest.hexdigest()
    if ((expected_size is not None and size != expected_size)
            or (expected_sha256 is not None and actual_sha256 != expected_sha256)):
        print(f"  {src.name}: source changed after selection", file=sys.stderr)
        return None
    if src.suffix.lower() == ".pdf":
        if not shutil.which("pdftotext"):
            print("  pdftotext not installed (brew install poppler / choco install poppler)",
                  file=sys.stderr)
            return None
        if run_stage(["pdftotext", "-q", "-enc", "UTF-8",
                      str(original), str(out)], cancel).returncode != 0:
            return None
        # pdftotext reads an embedded text layer; it is NOT OCR. A scanned PDF
        # yields nothing, and summarizing nothing must not look like success.
        try:
            extracted = out.read_text(encoding="utf-8")
        except UnicodeDecodeError as e:
            print(f"  {src.name}: pdftotext produced non-UTF-8 text: {e}",
                  file=sys.stderr)
            return None
        if not extracted.strip():
            print(f"  {src.name}: pdftotext found no text. This looks like a "
                  "scanned PDF with no text layer — run it through OCR first "
                  "and pass the resulting .md/.txt instead.", file=sys.stderr)
            return None
    else:
        try:
            original.read_text(encoding="utf-8")
        except UnicodeDecodeError as e:
            print(f"  {src.name}: source is not valid UTF-8: {e}", file=sys.stderr)
            return None
        shutil.copyfile(original, out)
    return out


def out_path(target, suffix: str, out_dir=None) -> pathlib.Path:
    """Where an artifact is published.

    Beside the source by default, which is what makes a summary findable next to
    the thing it summarizes. An absolute ``out_dir`` collects a run's artifacts
    in one place. A relative ``out_dir`` such as ``out`` or ``..`` is resolved
    against that source's directory.
    """
    stem = target.name if target.is_dir() else target.stem
    if out_dir is not None:
        return selection.resolve_output_directory(out_dir, target.parent) / (
            stem + suffix)
    return target.parent / (stem + suffix)



def slug_from(text: str, fallback: str = "clipboard") -> str:
    """A filename from what the model actually wrote.

    Pasted text has no source filename, and "clipboard.summary.md" is not one:
    the SECOND paste overwrites the first, silently, at a different time and so
    outside any within-run collision check. The artifact needs a name of its own.

    Taken from the Brief's opening words. That is already a model-authored
    distillation of the document, so it needs no extra call to name it, cannot
    fail, and costs nothing. The coverage note is skipped -- it is identical on
    every artifact and would name them all the same thing.
    """
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(">"):
            continue
        line = re.sub(r"^#+\s*", "", line)
        line = re.sub(r"[*_`\[\]()]", "", line)
        words = [w for w in re.split(r"\s+", line) if w]
        if len(words) < 3:
            continue
        slug = re.sub(r"[^a-z0-9]+", "-", " ".join(words[:8]).lower()).strip("-")
        if len(slug) >= 8:
            return slug[:60].rstrip("-")
    return fallback


def free_path(dest: pathlib.Path, taken: set) -> pathlib.Path:
    """A path not already used by this run OR already on disk.

    Re-summarizing a FILE should overwrite its own artifacts -- that is an
    update. Pasted text is a new document every time, so it must never land on
    an existing name; the check therefore includes what is already on disk.
    """
    n, cand = 1, dest
    while cand in taken or cand.exists():
        n += 1
        for suf in ARTIFACT_SUFFIXES:
            if dest.name.endswith(suf):
                cand = dest.with_name(f"{dest.name[:-len(suf)]}-{n}{suf}")
                break
        else:
            cand = dest.with_name(f"{dest.stem}-{n}{dest.suffix}")
    taken.add(cand)
    return cand


def unclashed(dest: pathlib.Path, taken: set) -> pathlib.Path:
    """A distinct path per source, even when two sources share a name.

    Publishing beside the source, two `notes.md` files in different folders
    never collide. Collected into ONE --out directory they do, and the second
    would silently overwrite the first: a run that reports twenty successes and
    leaves nineteen files. Suffix the later ones and say so.
    """
    if dest not in taken:
        taken.add(dest)
        return dest
    base, n = dest.name, 2
    while True:
        # Split on the FULL artifact suffix (".summary.md"), not the last dot,
        # or "x.summary.md" becomes "x.summary-2.md" and the pair stops matching.
        for suf in ARTIFACT_SUFFIXES:
            if base.endswith(suf):
                cand = dest.with_name(f"{base[:-len(suf)]}-{n}{suf}")
                break
        else:
            cand = dest.with_name(f"{dest.stem}-{n}{dest.suffix}")
        if cand not in taken:
            print(f"  name already used this run -> {cand.name}")
            taken.add(cand)
            return cand
        n += 1


def _unlink_best_effort(*paths):
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def publish_pair(detailed: pathlib.Path, brief: pathlib.Path,
                 dest: pathlib.Path, replace=os.replace,
                 bdest: pathlib.Path | None = None) -> pathlib.Path:
    """Guard one pair update and restore predecessors after handled failures."""
    if bdest is None:
        if dest.name.startswith("summary."):
            bdest = dest.with_name("brief." + dest.name[len("summary."):])
        elif dest.name.endswith(SUMMARY_SUFFIX):
            bdest = dest.with_name(dest.name[:-len(SUMMARY_SUFFIX)] + BRIEF_SUFFIX)
        else:
            bdest = dest.with_name("brief." + dest.name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    token = f"{os.getpid()}-{uuid.uuid4().hex[:8]}"
    t1 = dest.with_name(dest.name + f".tmp{token}")
    t2 = bdest.with_name(bdest.name + f".tmp{token}")
    b1 = dest.with_name(dest.name + f".bak{token}")
    b2 = bdest.with_name(bdest.name + f".bak{token}")
    try:
        shutil.copyfile(detailed, t1); shutil.copyfile(brief, t2)
    except Exception as original:
        _unlink_best_effort(t1, t2)
        raise PublicationError(
            f"publication staging failed: {original}", restored=True) from original
    had1, had2 = dest.exists(), bdest.exists()
    changed1 = changed2 = False
    try:
        if had1: shutil.copyfile(dest, b1)
        if had2: shutil.copyfile(bdest, b2)
        replace(t1, dest)
        changed1 = True
        replace(t2, bdest)
        changed2 = True
    except Exception as original:
        # A target with no predecessor must be removed, not left beside the old
        # half. Restore through a second temporary file so a failed recovery
        # never consumes the only backup.
        rollback_errors = []
        for src_b, tgt, had, changed in (
                (b1, dest, had1, changed1), (b2, bdest, had2, changed2)):
            if not changed:
                continue
            try:
                if had and src_b.exists():
                    restore_tmp = tgt.with_name(tgt.name + f".restore{token}")
                    try:
                        shutil.copyfile(src_b, restore_tmp)
                        os.replace(restore_tmp, tgt)
                    finally:
                        restore_tmp.unlink(missing_ok=True)
                elif not had:
                    tgt.unlink(missing_ok=True)
                else:
                    raise OSError(f"missing recovery backup {src_b}")
            except Exception as rollback_error:
                rollback_errors.append(f"{tgt}: {rollback_error}")
        restored = not rollback_errors
        if restored:
            _unlink_best_effort(t1, t2, b1, b2)
        recovery = tuple(path for path in (b1, b2, t1, t2) if path.exists())
        detail = f"publication failed: {original}"
        if rollback_errors:
            detail += "; rollback incomplete: " + "; ".join(rollback_errors)
            if recovery:
                detail += "; recovery retained at " + ", ".join(map(str, recovery))
        else:
            detail += "; previous destination state restored"
        raise PublicationError(
            detail, restored=restored, recovery_paths=recovery) from original
    _unlink_best_effort(b1, b2)
    return bdest


def publish_one(source: pathlib.Path, dest: pathlib.Path,
                replace=os.replace, *, expected_sha256: str | None = None) -> None:
    """Atomically replace one transform artifact, restoring its predecessor."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    token = f"{os.getpid()}-{uuid.uuid4().hex[:8]}"
    tmp = dest.with_name(dest.name + f".tmp{token}")
    backup = dest.with_name(dest.name + f".bak{token}")
    try:
        shutil.copyfile(source, tmp)
        if expected_sha256 is not None:
            if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
                raise ValueError("invalid expected SHA-256")
            actual = hashlib.sha256(tmp.read_bytes()).hexdigest()
            if actual != expected_sha256:
                raise ValueError(
                    "staged artifact does not match its accepted SHA-256")
    except Exception as original:
        _unlink_best_effort(tmp)
        raise PublicationError(
            f"publication staging failed: {original}", restored=True) from original
    existed = dest.exists()
    changed = False
    try:
        if existed:
            shutil.copyfile(dest, backup)
        replace(tmp, dest)
        changed = True
    except Exception as original:
        rollback_errors = []
        if changed:
            try:
                if existed and backup.exists():
                    restore_tmp = dest.with_name(dest.name + f".restore{token}")
                    try:
                        shutil.copyfile(backup, restore_tmp)
                        os.replace(restore_tmp, dest)
                    finally:
                        restore_tmp.unlink(missing_ok=True)
                elif not existed:
                    dest.unlink(missing_ok=True)
                else:
                    raise OSError(f"missing recovery backup {backup}")
            except Exception as rollback_error:
                rollback_errors.append(f"{dest}: {rollback_error}")
        restored = not rollback_errors
        if restored:
            _unlink_best_effort(tmp, backup)
        recovery = tuple(path for path in (backup, tmp) if path.exists())
        detail = f"publication failed: {original}"
        if rollback_errors:
            detail += "; rollback incomplete: " + "; ".join(rollback_errors)
            if recovery:
                detail += "; recovery retained at " + ", ".join(map(str, recovery))
        else:
            detail += "; previous destination state restored"
        raise PublicationError(
            detail, restored=restored, recovery_paths=recovery) from original
    _unlink_best_effort(backup)


def verified_textprep_artifact(run_dir: pathlib.Path):
    """Return the exact successful Clean-text artifact and its accepted hash."""
    artifact = run_dir / "cleaned.md"
    report_path = run_dir / "textprep-report.json"
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Clean-text report is missing or invalid: {exc}") from exc
    if not isinstance(report, dict) or report.get("status") != "succeeded":
        raise ValueError("Clean-text report does not record success")
    expected = report.get("output_sha256")
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("Clean-text report has no valid output SHA-256")
    try:
        payload = artifact.read_bytes()
        text = payload.decode("utf-8")
    except Exception as exc:
        raise ValueError(f"Clean-text artifact is missing or not UTF-8: {exc}") from exc
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError("Clean-text artifact does not match its accepted SHA-256")
    words = report.get("output_words")
    if isinstance(words, bool) or not isinstance(words, int) \
            or words != len(text.split()):
        raise ValueError("Clean-text report word count does not match its artifact")
    return artifact, report, expected, text


def emit_cancelled_targets(start_index: int, targets, documents=()):
    """Close every target announced by job_started after cancellation.

    A batch client cannot call a job finished while an announced target has no
    terminal record.  Targets not reached by the loop are explicitly cancelled
    as ``not_started``; completed siblings remain untouched.
    """
    for index in range(start_index, len(targets) + 1):
        doc = documents[index - 1] if index <= len(documents) else None
        emit_target_finished(index=index, status="cancelled", exit_code=130,
                             failure_kind="not_started",
                             destination_unchanged=True,
                             document_id=(doc.id if doc else None))


def emit_target_finished(*, index, status, exit_code, document_id=None,
                         **fields):
    """Write explicit document and publication-target terminal records.

    Progress no longer infers a document terminal from a target terminal. That
    inference is invalid for Corpus, where one publication target owns many
    document preparation records, and it also made Batch terminal accounting
    depend on a hidden side effect in the event writer.
    """
    progress.emit("document_finished", document_id=document_id, index=index,
                  status=status, exit_code=exit_code,
                  failure_kind=fields.get("failure_kind"))
    progress.emit("target_finished", index=index, status=status,
                  exit_code=exit_code, document_id=document_id, **fields)


def emit_input_problem_targets(problems, start_index):
    """Terminalize selection problems as announced non-document targets."""
    for offset, problem in enumerate(problems):
        index = start_index + offset
        kind = problem.get("kind") or "input_problem"
        label = problem.get("path") or kind
        progress.emit("target_started", target_id=f"E{index:03d}", index=index,
                      count=start_index + len(problems) - 1, title=label,
                      words=0, input_problem=True)
        emit_target_finished(index=index, status="failed", exit_code=1,
                             failure_kind=kind, destination_unchanged=True,
                             input_problem=True)


def do_corpus(selection_obj, work_root, cancel, harness, roster,
              instruction_text="", runner=None):
    """Run the guarded hierarchical Corpus route as one publication target."""
    import corpus
    import corpus_seal

    root = pathlib.Path(work_root).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    # Corpus runs every stage in this process under nested directories, so the
    # runner needs one explicit key or its dead-route, stall, and attempt
    # state would otherwise reset between stages.
    os.environ["SUMM_TARGET_KEY"] = str(root.resolve())
    docs = selection_obj.documents
    finished_docs = set()
    allocation_lease = destination_lease = brief_lease = None

    def document_started():
        for index, doc in enumerate(docs, 1):
            progress.emit("document_started", document_id=doc.id, index=index,
                          count=len(docs), label=doc.relative_path)

    def document_finished(document_id, status, exit_code, failure_kind=None):
        if document_id in finished_docs:
            return
        finished_docs.add(document_id)
        document = next((d for d in docs if d.id == document_id), None)
        progress.emit("document_finished", document_id=document_id,
                      index=(docs.index(document) + 1 if document else None),
                      count=len(docs), status=status, exit_code=exit_code,
                      failure_kind=failure_kind)

    def finish_remaining(status, exit_code, failure_kind):
        for doc in docs:
            document_finished(doc.id, status, exit_code, failure_kind)

    def finish_target(status, exit_code, failure_kind=None, outputs=None):
        progress.emit("target_finished", target_id="T001", index=1, count=1,
                      status=status, exit_code=exit_code,
                      failure_kind=failure_kind, outputs=outputs or [],
                      destination_unchanged=not bool(outputs))
        progress.emit("job_finished", status=status, exit_code=exit_code)

    try:
        # Announce the one Corpus publication target before any validation can
        # fail. This keeps every terminal record paired with exactly one start,
        # including malformed output plans and unavailable model routes.
        outputs = tuple(selection_obj.corpus_outputs)
        document_started()
        progress.emit("target_started", target_id="T001", index=1, count=1,
                      title=selection_obj.corpus_name or "Corpus", words=0,
                      document_count=len(docs))
        selection.validate_output_pair(
            selection_obj.corpus_outputs,
            (".corpus.summary.md", ".corpus.brief.md"),
            (doc.source_path for doc in docs))
        if len(selection_obj.corpus_outputs) != 2:
            raise ValueError("Corpus requires exactly one output pair")
        # Validate the active role routes before source work, but no model can
        # run until corpus.preflight has accepted every staged document.
        freeze_model_routes(harness, roster, mode_config.BY_KEY["summarize"].roles)
        validate_roles(harness, roster, mode_config.BY_KEY["summarize"].roles)
        if instruction_text:
            instruction_path = root / "run" / "custom-instructions.txt"
            instruction_path.parent.mkdir(parents=True, exist_ok=True)
            instruction_path.write_text(instruction_text, encoding="utf-8")
            os.environ["SUMM_INSTRUCTIONS_FILE"] = str(instruction_path)
            os.environ["SUMM_INSTRUCTIONS_DIGEST"] = custom_instructions.digest(
                instruction_text)
        else:
            os.environ.pop("SUMM_INSTRUCTIONS_FILE", None)
            os.environ.pop("SUMM_INSTRUCTIONS_DIGEST", None)

        allocation_lease = runtime.output_directory_lock(outputs[0].parent,
                                                         cancel.check)
        allocation_lease.__enter__()
        destination_lease = runtime.destination_lock(outputs[0], cancel.check)
        destination_lease.__enter__()
        brief_lease = runtime.destination_lock(outputs[1], cancel.check)
        brief_lease.__enter__()

        progress.emit("stage", name="corpus_preflight")
        try:
            prepared = corpus.preflight(selection_obj, root, cancel)
        except corpus.CorpusPreflightError as exc:
            contract_codes = {"selection_invalid", "document_limit",
                              "visible_word_limit", "inventory_window_limit",
                              "output_plan_invalid"}
            exit_code = 2 if exc.code in contract_codes else 1
            finish_remaining("failed", exit_code, exc.code)
            finish_target("failed", exit_code, exc.code)
            print(f"Corpus preflight failed: {exc}", file=sys.stderr)
            return exit_code
        progress.emit("corpus_preflight_finished",
                      documents=len(prepared.documents),
                      visible_words=prepared.total_visible_words,
                      inventory_windows=prepared.inventory_windows)

        progress.emit("stage", name="corpus_inventory")
        def inventory_emit(event, **fields):
            if event == "document_finished":
                document_finished(fields.get("document_id"), fields.get("status"),
                                  fields.get("exit_code", 1),
                                  fields.get("failure_kind"))
            else:
                progress.emit(event, **fields)
        try:
            inventories = corpus.build_inventories(
                prepared, root, cancel, runner=runner, emit=inventory_emit)
        except Cancelled:
            finish_remaining("cancelled", 130, "not_started")
            finish_target("cancelled", 130, "cancelled")
            return 130
        except Exception as exc:
            finish_remaining("failed", 5, "inventory_failed")
            finish_target("failed", 5, "inventory_failed")
            print(f"Corpus inventory failed: {exc}", file=sys.stderr)
            return 5

        progress.emit("stage", name="corpus_plan")
        try:
            plan = corpus.build_corpus_plan(
                inventories, prepared, root, cancel, runner=runner)
        except Cancelled:
            finish_target("cancelled", 130, "cancelled")
            return 130
        except Exception as exc:
            finish_target("failed", 5, "corpus_plan_failed")
            print(f"Corpus plan failed: {exc}", file=sys.stderr)
            return 5

        progress.emit("stage", name="corpus_overview")
        try:
            rc = corpus.write_overview_pair(
                plan, inventories, prepared, root, cancel, runner=runner)
        except Cancelled:
            finish_target("cancelled", 130, "cancelled")
            return 130
        except Exception as exc:
            finish_target("failed", 5, "corpus_overview_failed")
            print(f"Corpus overview failed: {exc}", file=sys.stderr)
            return 5
        composed = root / "corpus" / "pair"
        if rc != 0 or not (composed / "detailed.md").is_file() \
                or not (composed / "brief.md").is_file():
            finish_target("failed", rc or 1, "corpus_overview_failed")
            print("Corpus overview produced no usable pair", file=sys.stderr)
            return rc or 1

        progress.emit("stage", name="corpus_seal")
        sealed = corpus_seal.seal(root)
        if not sealed.get("passed"):
            finish_target("failed", 3, "seal_failed")
            print("Corpus structural seal failed; nothing published", file=sys.stderr)
            return 3

        cancel.check()
        progress.emit("stage", name="publish")
        try:
            brief = publish_pair(composed / "detailed.md",
                                 composed / "brief.md", outputs[0])
        except Exception as exc:
            unchanged = getattr(exc, "restored", True)
            finish_target("failed", 1, "publish_failed")
            print(f"Corpus publication failed: {exc}", file=sys.stderr)
            return 1
        finish_target("succeeded", 0, outputs=[str(outputs[0]), str(brief)])
        print(f"  -> {outputs[0]}")
        print(f"  -> {brief}")
        notify("Summary", "Corpus Detailed and Brief ready")
        return 0
    except Cancelled:
        finish_remaining("cancelled", 130, "not_started")
        finish_target("cancelled", 130, "cancelled")
        return 130
    except Exception as exc:
        finish_remaining("failed", 1, "internal_error")
        finish_target("failed", 1, "internal_error")
        print(f"Corpus failed: {exc}", file=sys.stderr)
        return 1
    finally:
        for lease in (brief_lease, destination_lease, allocation_lease):
            if lease is not None:
                lease.__exit__(None, None, None)


# ---------------------------------------------------------------- tts path ---
def do_tts(targets, raw_text, work_dir=None, out_dir=None, cancel=None,
           harness=None, roster=None, documents=(), selection_errors=(),
           affix_kind="prefix", affix_text=""):
    """Prepare speech text without an unaudited source rewrite.

    Known Summer artifacts go directly through deterministic normalization.
    Every other input first passes the complete Clean-text pipeline. Both write
    a work candidate and publish atomically only after every gate succeeds.
    """
    cancel = cancel or CancellationToken()
    codes = [1] * len(selection_errors)
    cancelled = False
    publication_committed = False
    taken = set()
    requested_role_options = os.environ.get("SUMM_ROLE_OPTIONS")
    target_total = len(targets) + len(selection_errors)
    emit_input_problem_targets(selection_errors, len(targets) + 1)
    for i, t in enumerate(targets, 1):
        doc = documents[i - 1] if not raw_text and i <= len(documents) else None
        if work_dir:
            tmp = pathlib.Path(work_dir).expanduser() / f"{i:02d}"
            tmp.mkdir(parents=True, exist_ok=True)
        else:
            tmp = pathlib.Path(tempfile.mkdtemp())
        target_lease = runtime.target_lease(cancel.check)
        target_entered = False
        destination_lease = allocation_lease = None
        try:
            cancel.check()
            admission_wait_s = target_lease.__enter__()
            target_entered = True
            title = "Clipboard text" if raw_text else (
                doc.source_path.stem if doc else pathlib.Path(t).stem)
            progress.emit("document_started", document_id=(doc.id if doc else None),
                          index=i, count=target_total, label=title, words=0)
            progress.emit("target_started", index=i, count=target_total,
                          title=title, words=0,
                          admission_wait_s=round(admission_wait_s, 3),
                          document_id=(doc.id if doc else None))
            if raw_text:
                src = tmp / "source.txt"
                src.write_text(str(t), encoding="utf-8")
                parent = selection.clipboard_output_directory(out_dir)
                allocation_lease = runtime.output_directory_lock(parent, cancel.check)
                allocation_lease.__enter__()
                dest_name = selection.format_output_filenames(
                    "clipboard", "tts", affix_kind, affix_text)[0]
                dest = free_path(parent / dest_name, taken)
                title = "Clipboard text"
                trusted_artifact = False
            else:
                if doc is not None:
                    accepted, why = selection.verify_document(doc)
                    if not accepted:
                        print(f"  input changed or unreadable: {t} ({why})",
                              file=sys.stderr)
                        codes.append(1)
                        emit_target_finished(index=i, status="failed",
                                      exit_code=1,
                                      failure_kind=doc.error_kind or "input_changed",
                                      destination_unchanged=True)
                        continue
                src = stage(pathlib.Path(t), tmp, cancel,
                            doc.source_sha256 if doc else None,
                            doc.size_bytes if doc else None)
                if src is None:
                    print(f"  skip (unreadable): {t}", file=sys.stderr)
                    codes.append(1)
                    emit_target_finished(index=i, status="failed",
                                  exit_code=1, failure_kind="input_failed",
                                  destination_unchanged=True)
                    continue
                planned = doc.planned_outputs if doc else ()
                if planned:
                    dest = pathlib.Path(planned[0])
                else:
                    dest_names = selection.format_output_filenames(
                        pathlib.Path(t).stem, "tts", affix_kind, affix_text)
                    dest_parent = (selection.resolve_output_directory(out_dir, pathlib.Path(t).parent)
                                   if out_dir else pathlib.Path(t).parent)
                    dest = unclashed(dest_parent / dest_names[0], taken)
                title = doc.relative_path if doc else pathlib.Path(t).stem
                trusted_artifact = (
                    pathlib.Path(t).name.casefold().startswith(
                        ("summary.", "brief.", "clean.", "tts.")) or
                    pathlib.Path(t).name.casefold().endswith(
                        (SUMMARY_SUFFIX, BRIEF_SUFFIX,
                         mode_config.BY_KEY["text_prep"].suffixes[0], TTS_SUFFIX))
                )
            destination_lease = runtime.destination_lock(dest, cancel.check)
            destination_lease.__enter__()
            words = len(src.read_text(encoding="utf-8").split())
            run_dir = tmp / "run"
            run_dir.mkdir(parents=True, exist_ok=True)
            if trusted_artifact:
                print(f"  Summer artifact ({words} words) -> speech reformat pass")
                prepared_input = src
            else:
                if harness is not None and roster is not None:
                    if requested_role_options is None:
                        os.environ.pop("SUMM_ROLE_OPTIONS", None)
                    else:
                        os.environ["SUMM_ROLE_OPTIONS"] = requested_role_options
                    try:
                        freeze_model_routes(
                            harness, roster, mode_config.BY_KEY["tts"].roles)
                        validate_roles(
                            harness, roster, mode_config.BY_KEY["tts"].roles)
                    except Exception as exc:
                        print(f"  model routes are invalid: {exc}", file=sys.stderr)
                        codes.append(2)
                        emit_target_finished(index=i, status="failed",
                                      exit_code=2, failure_kind="config_failed",
                                      destination_unchanged=True)
                        continue
                print(f"  full text ({words} words) -> Clean text -> speech pass")
                progress.emit("stage", name="text_prep")
                rc_clean = run_stage(
                    [sys.executable, str(HERE / "textprep.py"), str(src),
                     str(run_dir)], cancel, env={**os.environ,
                                                "SUMM_ACTIVE_ROLES": " ".join(
                                                    mode_config.BY_KEY["tts"].roles)}
                ).returncode
                if rc_clean != 0:
                    code = rc_clean or 5
                    print("  Clean text failed — no speech artifact published",
                          file=sys.stderr)
                    codes.append(code)
                    emit_target_finished(index=i, status="failed",
                                  exit_code=code, failure_kind="text_prep_failed",
                                  destination_unchanged=True)
                    continue
                try:
                    prepared_input, _, prepared_sha256, _ = \
                        verified_textprep_artifact(run_dir)
                except ValueError as exc:
                    print(f"  Clean text failed verification: {exc}", file=sys.stderr)
                    codes.append(5)
                    emit_target_finished(index=i, status="failed",
                                  exit_code=5, failure_kind="text_prep_failed",
                                  destination_unchanged=True)
                    continue

            progress.emit("stage", name="speechprep")
            speech_candidate = run_dir / "prepared.speech.txt"
            rc_speech = run_stage(
                [sys.executable, str(HERE / "speechprep.py"), str(prepared_input),
                 str(speech_candidate)], cancel, env={**os.environ,
                                                     "SUMM_ACTIVE_ROLES": " ".join(
                                                         mode_config.BY_KEY["tts"].roles)}
            ).returncode
            if rc_speech != 0 or not speech_candidate.exists():
                print("  speech reformat failed — previous output unchanged",
                      file=sys.stderr)
                codes.append(rc_speech or 1)
                emit_target_finished(index=i, status="failed",
                              exit_code=rc_speech or 1,
                              failure_kind="speechprep_failed",
                              destination_unchanged=True)
                continue

            progress.emit("stage", name="tts_normalize")
            candidate = run_dir / "prepared.tts.txt"
            normalize_argv = [sys.executable, str(HERE / "tts_normalize.py"),
                              str(speech_candidate), str(candidate)]
            rc_normalize = run_stage(
                normalize_argv, cancel).returncode
            if rc_normalize != 0 or not candidate.exists():
                print("  speech normalization failed — previous output unchanged",
                      file=sys.stderr)
                codes.append(rc_normalize or 1)
                emit_target_finished(index=i, status="failed",
                              exit_code=rc_normalize or 1,
                              failure_kind="tts_normalize_failed",
                              destination_unchanged=True)
                continue
            cancel.check()
            progress.emit("stage", name="publish")
            try:
                publish_one(candidate, dest)
            except Exception as exc:
                unchanged = getattr(exc, "restored", True)
                state = ("previous file restored" if unchanged else
                         "rollback incomplete; recovery files retained")
                print(f"  publish failed, {state}: {exc}", file=sys.stderr)
                codes.append(1)
                emit_target_finished(index=i, status="failed",
                              exit_code=1, failure_kind="publish_failed",
                              destination_unchanged=unchanged)
                continue
            codes.append(0)
            publication_committed = True
            print(f"  -> {dest}")
            emit_target_finished(index=i, status="succeeded",
                          exit_code=0, destination_unchanged=False,
                          outputs=[str(dest)],
                          document_id=(doc.id if doc else None))
        except Cancelled:
            cancelled = True
            codes.append(130)
            emit_target_finished(index=i, status="cancelled",
                          exit_code=130, destination_unchanged=True)
        except Exception as exc:
            codes.append(1)
            print(f"  unexpected target failure: {exc}", file=sys.stderr)
            emit_target_finished(index=i, status="failed",
                          exit_code=1, failure_kind="internal_error",
                          destination_unchanged=False)
        finally:
            if destination_lease:
                destination_lease.__exit__(None, None, None)
            if allocation_lease:
                allocation_lease.__exit__(None, None, None)
            if target_entered:
                target_lease.__exit__(None, None, None)
            if work_dir:
                print(f"  work kept: {tmp}", file=sys.stderr)
            else:
                shutil.rmtree(tmp, ignore_errors=True)
        if cancelled or (cancel.requested() and not publication_committed):
            cancelled = True
            emit_cancelled_targets(i + 1, targets, documents)
            break
    successes = sum(code == 0 for code in codes)
    result = (130 if cancelled else 0 if codes and not any(codes)
              else 1)
    progress.emit("job_finished",
                  status=("cancelled" if result == 130 else
                          "succeeded" if result == 0 else
                          "partial" if successes else "failed"),
                  exit_code=result)
    notify("TTS prep", "Done" if result == 0 else "Check output")
    return result


# ---------------------------------------------------------------- doctor ---
def doctor() -> int:
    """Preflight the install without spending model quota.

    Exists because the two platforms are wired by different triggers (Raycast,
    AHK) onto one implementation: the way this breaks is an environment gap on
    one platform only, which a summarizing run would surface only after the
    user has waited for it.
    """
    ok = True

    def chk(label, good, detail=""):
        nonlocal ok
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'}  {label}{'  ' + detail if detail else ''}")

    print(f"summ doctor — {sys.platform}, python {sys.version.split()[0]}")
    chk("python >= 3.9", sys.version_info >= (3, 9))

    for mod in ("readerview.py", "ledger.py", "compose.py", "mechseal.py",
                "corpus.py", "corpus_seal.py", "mapsum.py", "gateway.py",
                "tts_normalize.py", "speechprep.py", "textprep.py", "custom_instructions.py",
                "shortsum.py", "fullsum.py", "pair_review.py",
                "route_caps.py"):
        chk(f"engine/{mod}", (HERE / mod).is_file())

    # Quick and Full share one writer, one audit, and one repair appendix.
    # The ledger and corpus prompts serve only the Corpus route.
    for pr in ("ledger-build.txt", "ledger-audit.txt", "ledger-revise.txt",
               "pair-write.txt", "reading-policy.txt", "reading-part.txt",
               "pair-audit.txt", "pair-repair.txt",
               "pair-repair-length.txt",
               "full-window.txt",
               "corpus-inventory-build.txt", "corpus-inventory-audit.txt",
               "corpus-inventory-repair.txt", "corpus-plan-build.txt",
               "corpus-plan-audit.txt", "corpus-plan-repair.txt",
               "corpus-overview-rules.txt", "text-prep.txt", "speechprep.txt",
               "speechprep-audit.txt", "speechprep-repair.txt"):
        chk(f"prompts/{pr}", (HERE / "prompts" / pr).is_file())

    harness = os.environ.get("HARNESS", "agy")

    # Resolve the same merged committed-plus-device roster as production. A
    # chain entry may name its own harness, and only the roles active for this
    # mode are part of this preflight.
    chains, referenced = {}, {harness}
    try:
        import importlib.util as _iu
        spec = _iu.spec_from_file_location("_ms", HERE / "mapsum.py")
        ms = _iu.module_from_spec(spec); spec.loader.exec_module(ms)
        all_chains = {"plan": ms.PLAN_MODELS, "write": ms.MODELS,
                      "audit": ms.AUDIT_MODELS, "repair": ms.REPAIR_MODELS}
        active_roles = set(os.environ.get("SUMM_ACTIVE_ROLES", "").split())
        chains = {role: chain for role, chain in all_chains.items()
                  if not active_roles or role in active_roles}
        chk("model chains resolve", all(chains.values()))
        for c in chains.values():
            referenced |= {ms.split_entry(e, harness)[0] for e in c}
    except Exception as e:
        chk("model chains resolve", False, str(e)[:70])

    for h in sorted(referenced):
        role = "run harness" if h == harness else "referenced by a chain entry"
        if h in model_config.gateway_harnesses():
            try:
                configured = h in runtime.config().get("gateways", {})
            except runtime.ConfigError as e:
                configured = False
                chk(f"gateway settings ({h}, {role})", False, str(e)[:70])
                continue
            chk(f"gateway settings ({h}, {role})", configured,
                "device-local runtime.json" if configured else "not configured")
            continue
        # Check the exact executable mapsum will invoke when it has one
        # (including AGY_BIN and the native Windows resolver), then fall back
        # to PATH for generic harnesses. Doctor and runtime must not qualify
        # different binaries.
        configured_exe = (getattr(ms, "BIN", {}).get(h)
                          if "ms" in locals() else None)
        exe = ((shutil.which(configured_exe) if configured_exe else None)
               or shutil.which(h) or shutil.which(h + ".cmd")
               or shutil.which(h + ".exe"))
        chk(f"harness CLI on PATH ({h}, {role})", exe is not None, exe or "not found")

    for role, c in chains.items():
        print(f"        {role:6} {' -> '.join(c)}")

    clip_tool = ("powershell" if WIN else "pbpaste" if sys.platform == "darwin" else "xclip")
    chk(f"clipboard reader ({clip_tool})", shutil.which(clip_tool) is not None)

    out = pathlib.Path.home() / "Downloads"
    writable = False
    try:
        out.mkdir(parents=True, exist_ok=True)
        probe = out / ".summ-doctor-probe"
        probe.write_text("x"); probe.unlink()
        writable = True
    except Exception:
        pass
    chk(f"output dir writable ({out})", writable)

    print("\nsumm doctor: " + ("OK" if ok else "PROBLEMS FOUND"))
    return 0 if ok else 1


# ------------------------------------------------------------------ main ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", choices=["full", "brief"], default="full",
                    help="accepted for compatibility; both triggers pass it. It "
                         "does NOT select the published artifact set — every "
                         "summarization run writes both the Detailed and the "
                         "Brief reading.")
    mode_group = ap.add_mutually_exclusive_group()
    mode_group.add_argument(mode_config.BY_KEY["tts"].flag, action="store_true",
                            help="prepare text for speech; known Summer artifacts "
                                 "normalize directly, while arbitrary input first "
                                 "passes the complete Clean-text pipeline")
    mode_group.add_argument(mode_config.BY_KEY["quick"].flag, action="store_true",
                            help="a bounded write/audit/revision path instead of "
                                 "Full. Explicit only: short documents no longer "
                                 "take it unasked. A usable pair is always "
                                 "published, with unresolved findings stated in "
                                 "its report.")
    mode_group.add_argument(mode_config.BY_KEY["text_prep"].flag,
                            action="store_true",
                            help="clean OCR and broken document text without summarizing; "
                                 "writes one atomic .clean.md artifact or nothing")
    ap.add_argument("--harness", default=None,
                    help="select a configured model harness. Defaults to $HARNESS, "
                         "then agy.")
    ap.add_argument("--doctor", action="store_true",
                    help="preflight the install (no model calls) and exit")
    ap.add_argument("--progress-jsonl", metavar="PATH",
                    help="append machine-readable run events to PATH (see "
                         "progress.py). Absent, nothing changes: both installed "
                         "triggers pass no such flag.")
    ap.add_argument("--text-file", metavar="PATH",
                    help="treat this file's CONTENTS as pasted text: named from "
                         "the produced artifact rather than from the filename, and "
                         "published to --out or Downloads. This is how a queued "
                         "clipboard job survives the clipboard changing before "
                         "it runs.")
    ap.add_argument("--selection-manifest", metavar="PATH",
                    help="read a frozen versioned path selection manifest. This "
                         "is the transport used by the desktop UI and avoids "
                         "putting every discovered file on argv.")
    ap.add_argument("--scope", choices=("batch", "corpus"), default=None,
                    help="process multiple selected documents independently (batch). "
                         "Corpus is an explicit Summarize scope.")
    ap.add_argument("--corpus-name", metavar="NAME",
                    help="logical name for a Corpus result; valid only with "
                         "--scope corpus")
    ap.add_argument("--out", metavar="DIR",
                    help="publish artifacts into DIR instead of beside each "
                         "source. An absolute DIR is created if missing. A "
                         "relative DIR such as out or .. is resolved against "
                         "each source's directory (or Downloads for pasted "
                         "text). Sources that share a name are suffixed rather "
                         "than overwriting each other.")
    ap.add_argument("--profile", metavar="NAME",
                    help="apply a saved role profile (same file the UI writes). "
                         "A profile names a model per role; explicit environment "
                         "still wins, so one role can be overridden for a run. "
                         "NOTE: Quick and --text-prep never run plan, and --tts "
                         "uses Clean-text roles only for arbitrary source, so "
                         "plan may be inert.")
    ap.add_argument("--local-only", action="store_true",
                    help="allow only roster entries declared local; never use a "
                         "non-local availability fallback")
    ap.add_argument("--instructions-file", metavar="PATH",
                    help="read one per-run custom summary or cleanup request from PATH. The "
                         "file is validated and snapshotted before model work; "
                         "it is ignored for --tts.")
    ap.add_argument("--work-dir", metavar="DIR",
                    help="run inside DIR and keep it. Intermediate ledgers, seal "
                         "records, cleanup blocks and audit rounds explain why an "
                         "artifact says what it says; discarding them makes a bad "
                         "result unexplainable after the fact.")
    ap.add_argument("--cancel-file", metavar="PATH",
                    help="cooperative cancellation marker used by front ends")
    ap.add_argument("--affix-kind", choices=("prefix", "suffix"), default="prefix",
                    help="place mode tag at start (prefix) or end (suffix) of filename (defaults to prefix)")
    ap.add_argument("--affix", default="",
                    help="custom naming affix string for the output artifact(s)")
    ap.add_argument("--affix-secondary", default="",
                    help="custom naming affix string for secondary output artifact (brief in two-artifact modes)")
    ap.add_argument("paths", nargs="*")
    a = ap.parse_args()
    cancel = CancellationToken(a.cancel_file)
    install_cancel_handlers(cancel)

    # One registry defines which roles and user-facing behavior the chosen mode
    # activates. SUMM_QUICK is the environment form used by qualification runs.
    requested_quick = bool(a.quick or os.environ.get("SUMM_QUICK"))
    selected_mode = mode_config.selected(
        quick=requested_quick, text_prep=a.text_prep, tts=a.tts)
    if sum(bool(value) for value in (a.selection_manifest, a.text_file)) \
            + bool(a.paths) > 1:
        print("--selection-manifest, --text-file, and positional paths are mutually exclusive",
              file=sys.stderr)
        return 2
    potential_roles = selected_mode.roles
    os.environ["SUMM_ACTIVE_ROLES"] = " ".join(potential_roles)
    if selected_mode.key == "summarize":
        os.environ.setdefault("SUMM_WRITING_CONTRACT", "v2")

    try:
        runtime_snapshot = runtime.config()
        os.environ["SUMM_RUNTIME_JSON"] = runtime.frozen_json(runtime_snapshot)
        roster = model_config.roster()
    except Exception as e:
        print(f"runtime/model configuration unreadable: {str(e)[:240]}",
              file=sys.stderr)
        return 2
    harnesses = tuple(name for name, value in roster.items()
                      if isinstance(value, dict)
                      and any(value.get(role) for role in mode_config.ROLES))

    # Read once, before any target can start. Children receive a private copy
    # under their run directory, never this mutable caller-owned path. TTS has
    # no summary prompt and deliberately does not read or snapshot requests.
    instruction_text = ""
    instruction_digest = None
    if a.instructions_file and selected_mode.instructions:
        try:
            instruction_text = custom_instructions.read_file(a.instructions_file)
        except ValueError as e:
            print(str(e), file=sys.stderr)
            return 2
        instruction_digest = custom_instructions.digest(instruction_text)

    out_dir = None
    if a.out:
        try:
            out_dir = selection.parse_output_dir(a.out)
        except ValueError as e:
            print(f"cannot use --out {a.out}: {e}", file=sys.stderr)
            return 2
        if selection.is_absolute_output(out_dir):
            out_dir = selection.resolve_output_directory(
                out_dir, pathlib.Path.cwd())
            try:
                out_dir.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                print(f"cannot use --out {a.out}: {e}", file=sys.stderr)
                return 2
            if not os.access(out_dir, os.W_OK):
                print(f"cannot write to --out {out_dir}", file=sys.stderr)
                return 2

    if a.profile:
        # Profiles are headless application state. Read and project them through
        # profiles.py so the CLI never imports the graphical application.
        try:
            saved = profiles.load_profiles()
        except Exception as e:
            print(f"cannot read profiles: {e}", file=sys.stderr)
            return 2
        profile = saved.get(a.profile)
        if not isinstance(profile, dict):
            print(f"unknown profile {a.profile!r}; have: "
                  f"{', '.join(sorted(saved)) or '(none)'}", file=sys.stderr)
            return 2

        # Project every potentially used role, but defer route validity until
        # source inspection determines which roles this target will call.
        write_value = profile.get("write")
        profile_default_harness = (
            str(write_value[0]) if isinstance(write_value, (list, tuple))
            and len(write_value) >= 2 else "")
        profile_harness = (a.harness or os.environ.get("HARNESS")
                           or profile_default_harness or "agy")
        picks = {}
        for role in potential_roles:
            value = profile.get(role)
            variable = f"{profile_harness.upper()}_CHAIN_{role.upper()}"
            explicit = os.environ.get(variable)
            if explicit is not None and explicit.split():
                continue
            if not isinstance(value, (list, tuple)) or len(value) < 2:
                continue
            selected_harness, model = str(value[0]), str(value[1])
            option = str(value[2]) if len(value) > 2 and value[2] is not None else ""
            if not selected_harness or not model:
                continue
            picks[role] = ((selected_harness, model, option)
                           if option else (selected_harness, model))

        fallback = tuple(profile.get("_fallback") or ())
        if len(fallback) < 2:
            default_fb = profiles._default_fallback()
            if len(default_fb) >= 2:
                fallback = tuple(default_fb)
        if len(fallback) >= 2:
            fh, fm = str(fallback[0]), str(fallback[1])
            fe = str(fallback[2]) if len(fallback) > 2 and fallback[2] is not None else ""
            fallback = (fh, fm, fe) if fe else (fh, fm)
        else:
            fallback = ()

        # Chain variables must be projected under the selected base harness or
        # the later winner would silently ignore the profile.
        for k, v in profiles.role_env(
                profile_harness, picks, fallback or None,
                local_only=a.local_only).items():
            # An explicit environment variable beats the profile, so a single
            # role can still be overridden for one run without editing it.
            os.environ.setdefault(k, v)
        selected = ", ".join(
            f"{r}={v[0]}:{v[1]}" for r, v in sorted(picks.items()))
        print(f"[profile] {a.profile}: {selected or 'active roles from environment'}")
    if a.local_only:
        os.environ["SUMM_LOCAL_ONLY"] = "1"

    # The flag wins, then the environment, then the default. This used to be an
    # unconditional assignment of the argparse default, which meant HARNESS= in
    # the environment was silently overwritten with "agy" on EVERY run -- a
    # request for one model quietly answered by another, which is the one kind
    # of failure this project is not allowed to have.
    harness = a.harness or os.environ.get("HARNESS") or "agy"
    if harness not in harnesses:
        print(f"unknown harness {harness!r} (from $HARNESS); expected one of "
              f"{', '.join(harnesses)}", file=sys.stderr)
        return 2
    os.environ["HARNESS"] = harness
    if a.doctor:
        try:
            freeze_model_routes(harness, roster, potential_roles)
        except Exception as exc:
            print(f"model routes could not be frozen: {str(exc)[:240]}",
                  file=sys.stderr)
            return 2
        return doctor()
    requested_role_options = os.environ.get("SUMM_ROLE_OPTIONS")
    selection_obj = None
    manifest_loaded = False
    if a.selection_manifest:
        try:
            selection_obj = selection.read_manifest(a.selection_manifest)
        except ValueError as e:
            print(str(e), file=sys.stderr)
            return 2
        manifest_loaded = True
        if selection_obj.mode_key != selected_mode.key:
            print("selection manifest mode does not match the requested mode",
                  file=sys.stderr)
            return 2
        if a.scope is not None and selection_obj.scope != a.scope:
            print("selection manifest scope does not match --scope", file=sys.stderr)
            return 2
        if (selection_obj.scope == "batch"
                and any(not doc.planned_outputs
                        for doc in selection_obj.documents)):
            # Old v1 Batch manifests did not always retain an output plan. They
            # remain readable for compatibility, but new manifests are frozen.
            if selection_obj.manifest_schema != selection.SCHEMA_V1:
                print("selection manifest has no frozen output plan", file=sys.stderr)
                return 2
            selection_obj = selection_obj.with_outputs(
                selected_mode.key, out_dir,
                affix_kind=a.affix_kind, affix_text=a.affix,
                affix_secondary=a.affix_secondary)
        elif out_dir is not None and selection_obj.scope == "batch":
            expected = selection_obj.with_outputs(
                selected_mode.key, out_dir,
                affix_kind=a.affix_kind, affix_text=a.affix,
                affix_secondary=a.affix_secondary)
            if tuple(doc.planned_outputs for doc in selection_obj.documents) != \
                    tuple(doc.planned_outputs for doc in expected.documents):
                print("selection manifest output plan does not match --out", file=sys.stderr)
                return 2
        elif out_dir is not None and selection_obj.scope == "corpus":
            expected = selection_obj.with_corpus_outputs(
                selection_obj.corpus_name, out_dir)
            if tuple(selection_obj.corpus_outputs) != tuple(expected.corpus_outputs):
                print("selection manifest Corpus output plan does not match --out",
                      file=sys.stderr)
                return 2
        if a.corpus_name and selection_obj.scope == "corpus" \
                and a.corpus_name != selection_obj.corpus_name:
            print("--corpus-name does not match the frozen selection manifest",
                  file=sys.stderr)
            return 2
    elif a.text_file:
        # The CONTENTS are the document, not the path. A queued clipboard job
        # cannot read the clipboard when it finally runs -- by then it holds
        # something else -- so the text is captured up front and handed over as
        # a file, while keeping raw-text semantics all the way to publication.
        tf = pathlib.Path(os.path.expanduser(a.text_file))
        try:
            raw = tf.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            print(f"cannot read --text-file {a.text_file}: {e}", file=sys.stderr)
            return 2
        if not raw.strip():
            print(f"--text-file {a.text_file} is empty", file=sys.stderr)
            return 2
        selection_obj = selection.text_selection(raw, selected_mode.key)
    elif a.paths:
        # Positional arguments are explicit paths. They are never guessed to be
        # prose and missing entries remain visible in the selection errors.
        selection_obj = selection.resolve_paths(a.paths, selected_mode.key)
    else:
        selection_obj = selection.classify_clipboard(clip_read(), selected_mode.key)
    if not manifest_loaded:
        if a.scope == "corpus":
            try:
                selection_obj = selection_obj.with_corpus_outputs(
                    a.corpus_name, out_dir)
            except ValueError as e:
                print(str(e), file=sys.stderr)
                return 2
        else:
            selection_obj = selection_obj.with_outputs(
                selected_mode.key, out_dir,
                affix_kind=a.affix_kind, affix_text=a.affix,
                affix_secondary=a.affix_secondary)
    scope = a.scope or selection_obj.scope or "batch"
    if a.corpus_name and scope != "corpus":
        print("--corpus-name requires --scope corpus", file=sys.stderr)
        return 2
    if scope == "corpus":
        if selected_mode.key != "summarize":
            print("--scope corpus is valid only for Summarize", file=sys.stderr)
            return 2
    raw_text = selection_obj.is_text
    targets = ([selection_obj.raw_text] if raw_text and (selection_obj.raw_text or "").strip() else
               [doc.source_path for doc in selection_obj.documents])
    if a.progress_jsonl:
        rc = progress.start(a.progress_jsonl, selected_mode.progress_route,
                            1 if scope == "corpus" else
                            len(targets) + len(selection_obj.errors), scope=scope)
        if rc:
            return rc
    progress.emit(
        "selection_resolved",
        roots=[root.manifest() for root in selection_obj.roots],
        documents=[{"id": doc.id, "root_id": doc.root_id,
                    "relative_path": doc.relative_path,
                    "size_bytes": doc.size_bytes}
                   for doc in selection_obj.documents],
        excluded=len(selection_obj.exclusions),
        problems=len(selection_obj.errors),
        order_digest=selection_obj.order_digest,
        scope=scope)
    if not targets:
        if selection_obj.errors:
            emit_input_problem_targets(selection_obj.errors, 1)
            for problem in selection_obj.errors:
                print(f"input problem: {problem.get('path')}: "
                      f"{problem.get('kind')}", file=sys.stderr)
            message = "No usable documents selected"
        else:
            message = "Clipboard is empty"
        print(message + ".", file=sys.stderr)
        notify("Summary", message)
        progress.emit("job_finished", status="failed", exit_code=1)
        return 1

    work_lease = None
    if a.work_dir:
        work_lease = runtime.work_root_lock(pathlib.Path(a.work_dir).expanduser())
        try:
            work_lease.__enter__()
        except runtime.BusyError as e:
            print(f"cannot use --work-dir {a.work_dir}: {e}", file=sys.stderr)
            progress.emit("job_finished", status="failed", exit_code=2)
            return 2

    if a.work_dir and not selection_obj.is_text:
        # The accepted selection and output plan are retained beside the target
        # work directories. This is evidence, not model input.
        try:
            selection.write_manifest(selection_obj,
                                     pathlib.Path(a.work_dir).expanduser()
                                     / "selection.json")
            selection.write_output_plan(
                selection_obj,
                pathlib.Path(a.work_dir).expanduser() / "output-plan.json")
        except OSError as e:
            print(f"cannot write selection evidence: {e}", file=sys.stderr)
            if work_lease:
                work_lease.__exit__(None, None, None)
            progress.emit("job_finished", status="failed", exit_code=2)
            return 2
    if scope == "corpus":
        ephemeral_root = not bool(a.work_dir)
        try:
            corpus_root = (pathlib.Path(a.work_dir).expanduser()
                           if a.work_dir else pathlib.Path(tempfile.mkdtemp()))
            return do_corpus(selection_obj, corpus_root, cancel, harness, roster,
                             instruction_text)
        finally:
            if ephemeral_root:
                shutil.rmtree(corpus_root, ignore_errors=True)
            if work_lease:
                work_lease.__exit__(None, None, None)
    if selected_mode.key == "tts":
        try:
            return do_tts(targets, raw_text, a.work_dir, out_dir, cancel,
                          harness, roster, selection_obj.documents,
                          selection_obj.execution_errors,
                          affix_kind=a.affix_kind, affix_text=a.affix)
        finally:
            if work_lease:
                work_lease.__exit__(None, None, None)

    # Both destinations are derived independently of --depth. Deriving bdest by
    # string-replacing ".summary.md" meant `--depth brief` produced dest == bdest
    # and the two artifacts overwrote each other.
    suffix = selected_mode.suffixes[0]
    # One exit code per target, in order. A caller that cannot tell an empty
    # Full write (1) from no usable candidate (5) from an unsafe windowed
    # route (6) has to parse human stderr to find out, so the Full and Quick
    # stage codes reach the caller verbatim. Selection problems are
    # retained as failed input work. Valid documents may still run
    # independently, so the final verdict can be Partial.
    input_errors = selection_obj.execution_errors
    codes = [1] * len(input_errors)
    target_total = len(targets) + len(input_errors)
    emit_input_problem_targets(input_errors, len(targets) + 1)
    cancelled = False
    publication_committed = False
    taken = set()
    documents = selection_obj.documents
    for i, t in enumerate(targets, 1):
        doc = documents[i - 1] if not raw_text and i <= len(documents) else None
        if a.work_dir:
            tmp = pathlib.Path(a.work_dir).expanduser() / f"{i:02d}"
            tmp.mkdir(parents=True, exist_ok=True)
        else:
            tmp = pathlib.Path(tempfile.mkdtemp())
        keep = bool(a.work_dir)
        target_lease = runtime.target_lease(cancel.check)
        target_entered = False
        destination_lease = allocation_lease = None
        route = execution_sig = None
        parts_count = None
        source_words = None
        service_started = None
        history_outcome = "failed"
        try:
            cancel.check()
            admission_wait_s = target_lease.__enter__()
            target_entered = True
            service_started = time.monotonic()
            dest_names = ()
            if raw_text:
                src = tmp / "source.txt"; src.write_text(str(t), encoding="utf-8")
                dest_names = selection.format_output_filenames(
                    "clipboard", selected_mode.key, affix_kind=a.affix_kind,
                    affix_text=a.affix, affix_secondary=a.affix_secondary)
                dest = unclashed(
                    selection.clipboard_output_directory(out_dir)
                    / dest_names[0],
                    taken)
                title = "Clipboard text"
            else:
                title = doc.relative_path if doc else pathlib.Path(t).stem
                if doc is not None:
                    accepted, why = selection.verify_document(doc)
                    if not accepted:
                        print(f"[{i}/{len(targets)}] input changed or unreadable: "
                              f"{t} ({why})", file=sys.stderr)
                        codes.append(1)
                        emit_target_finished(index=i, status="failed",
                                      exit_code=1,
                                      failure_kind=doc.error_kind or "input_changed",
                                      destination_unchanged=True,
                                      document_id=doc.id)
                        continue
                src = stage(pathlib.Path(t), tmp, cancel,
                            doc.source_sha256 if doc else None,
                            doc.size_bytes if doc else None)
                if src is None:
                    print(f"[{i}/{len(targets)}] skip (unreadable): {t}", file=sys.stderr)
                    codes.append(1)
                    emit_target_finished(index=i, status="failed",
                                  exit_code=1, failure_kind="input_failed",
                                  destination_unchanged=True,
                                  document_id=(doc.id if doc else None))
                    continue
                planned = doc.planned_outputs if doc else ()
                secondary_dest = None
                if planned:
                    dest = pathlib.Path(planned[0])
                    secondary_dest = pathlib.Path(planned[1]) if len(planned) > 1 else None
                else:
                    dest_names = selection.format_output_filenames(
                        pathlib.Path(t).stem, selected_mode.key,
                        affix_kind=a.affix_kind, affix_text=a.affix,
                        affix_secondary=a.affix_secondary)
                    dest_parent = (selection.resolve_output_directory(out_dir, pathlib.Path(t).parent)
                                   if out_dir else pathlib.Path(t).parent)
                    dest = unclashed(dest_parent / dest_names[0], taken)
                    secondary_dest = dest_parent / dest_names[1] if len(dest_names) > 1 else None
                destination_lease = runtime.destination_lock(
                    dest, cancel.check, secondary=secondary_dest)
                destination_lease.__enter__()
            words = len(src.read_text(encoding="utf-8").split())
            source_words = words
            # Modes are never rewritten by the source length. Quick is taken
            # only when asked for; Summarize stays Full at every source size,
            # short documents included.
            effective_mode = selected_mode
            quick = effective_mode.key == "quick"
            text_prep = effective_mode.key == "text_prep"
            # Direct-qualified Full takes the whole-source write/review/repair
            # path. Anything larger remains Full and is handled by the
            # deterministic source-window route.
            token_capability = route_token_capability(
                harness, roster, effective_mode.roles)
            fit = (None if quick or text_prep else
                   fullsum.fit_detail(
                       words, *route_capability(),
                       capability_tokens=token_capability[0],
                       output_budget_tokens=token_capability[1]))
            batching = (None if quick or text_prep else
                        fullsum.batched_output_detail(
                            words, route_capability()[1],
                            token_capability[1]))
            full_batched = bool(batching and batching["needed"])
            full_direct = (fit is not None and fit["fits"]
                           and not full_batched)
            route = (effective_mode.progress_route
                     if quick or text_prep else "full")
            # State the REASON quick was chosen, not a guess at it. Reporting
            # an env-forced quick as anything else is a false statement about
            # what the run did.
            why = (" text prep" if text_prep else
                   " quick" if quick else
                   " full-batched" if full_batched else
                   " full-direct" if full_direct else "")
            if (not quick and not text_prep and not full_direct
                    and not full_batched):
                why = " full-windowed"
            print(f"[{i}/{len(targets)}] {title} ({words} words{why})")
            progress.emit("document_started", document_id=(doc.id if doc else None),
                          index=i, count=target_total, label=title, words=words)
            progress.emit("target_started", index=i, count=target_total,
                          title=title, words=words,
                          admission_wait_s=round(admission_wait_s, 3),
                          document_id=(doc.id if doc else None))
            # Freeze only the roles the actual target will call. A short source
            # may legitimately use a quick-only backend with no planner, while
            # a later long source in the same batch must still retain the
            # user's original plan effort rather than inherit the first
            # target's reduced snapshot.
            os.environ["SUMM_ACTIVE_ROLES"] = " ".join(effective_mode.roles)
            if requested_role_options is None:
                os.environ.pop("SUMM_ROLE_OPTIONS", None)
            else:
                os.environ["SUMM_ROLE_OPTIONS"] = requested_role_options
            try:
                freeze_model_routes(harness, roster, effective_mode.roles)
                validate_roles(harness, roster, effective_mode.roles)
            except Exception as exc:
                print(f"  model routes are invalid: {exc}", file=sys.stderr)
                codes.append(2); keep = True
                emit_target_finished(index=i, status="failed",
                              exit_code=2, failure_kind="config_failed",
                              destination_unchanged=True)
                continue
            execution_sig = eta.execution_signature(
                route, instructions_digest=instruction_digest)
            eta_now = (eta.estimate(route=route, source_words=words,
                                    execution_sig=execution_sig)
                       if quick or text_prep else None)
            if eta_now:
                progress.emit("eta", **eta_now)

            env = {**os.environ, "DEPTH": a.depth,
                   # Child stage processes own the model-call boundary, so
                   # carry the frozen ETA identity into mapsum._record().
                   "SUMM_ETA_ROUTE": route,
                   "SUMM_ETA_SOURCE_WORDS": str(words),
                   "SUMM_ETA_SIGNATURE": execution_sig,
                   "SUMM_ACTIVE_ROLES": " ".join(effective_mode.roles),
                   # Summary child calls share one target deadline. The
                   # monotonic parent start is converted to an epoch deadline
                   # only for the child-process boundary; mapsum clips every
                   # wait and subprocess timeout to this value.
                   # Leave enough time for one bounded transport retry and the
                   # single allowed repair without making a target unbounded.
                   "SUMM_TARGET_DEADLINE_UNIX": str(
                       time.time() + max(0, (1800 if quick else 3600) -
                                          (time.monotonic() - service_started))),
                   "SUMM_TARGET_KEY": str(tmp.resolve()),
                   "SUMM_TARGET_ATTEMPT_LIMIT": str(
                       16 if quick else fullsum.FULL_MAX_PHYSICAL_ATTEMPTS)}
            # A custom request is per invocation, never an ambient inherited
            # setting. Prevent a caller environment from leaking one into a
            # run that did not opt in with --instructions-file.
            env.pop("SUMM_INSTRUCTIONS_FILE", None)
            env.pop("SUMM_INSTRUCTIONS_DIGEST", None)
            run_dir = tmp / "run"; run_dir.mkdir(parents=True, exist_ok=True)
            if instruction_text:
                # Snapshot bytes once per target. A queued job and a batch of
                # targets therefore cannot observe later edits to the source
                # instructions file, and a failed retained run explains which
                # request governed its prompts without putting it in ETA data.
                instruction_snapshot = run_dir / "custom-instructions.txt"
                instruction_snapshot.write_text(instruction_text, encoding="utf-8")
                env["SUMM_INSTRUCTIONS_FILE"] = str(instruction_snapshot)
                env["SUMM_INSTRUCTIONS_DIGEST"] = instruction_digest
            if text_prep:
                # FULL-TEXT CLEANUP. The child receives the source bytes through
                # its composed prompts, runs write/audit/one repair/re-audit,
                # and writes one candidate only after every chunk passes.
                progress.emit("stage", name="text_prep")
                rc_prep = run_stage(
                    [sys.executable, str(HERE / "textprep.py"), str(src),
                     str(run_dir)], cancel, env=env).returncode
                if rc_prep != 0:
                    print("  text preparation produced nothing", file=sys.stderr)
                    codes.append(rc_prep); keep = True
                    emit_target_finished(index=i, status="failed",
                                  exit_code=rc_prep,
                                  failure_kind="text_prep_failed",
                                  destination_unchanged=True)
                    continue
                try:
                    cleaned, prep_report, cleaned_sha256, cleaned_text = \
                        verified_textprep_artifact(run_dir)
                    parts_count = int(prep_report.get("chunks") or 0) or None
                except Exception as exc:
                    print(f"  text preparation failed verification: {exc}",
                          file=sys.stderr)
                    codes.append(5); keep = True
                    emit_target_finished(index=i, status="failed",
                                  exit_code=5,
                                  failure_kind="text_prep_failed",
                                  destination_unchanged=True)
                    continue
            elif quick:
                # QUICK PATH. One call, deterministic gates, always an artifact.
                # Writes the same detailed.md/brief.md the ledger path does, so
                # publication below is identical either way.
                progress.emit("stage", name="quick")
                rc_q = run_stage(
                    [sys.executable, str(HERE / "shortsum.py"), str(src),
                     str(run_dir)], cancel, env=env).returncode
                if rc_q != 0:
                    print("  the quick path produced nothing", file=sys.stderr)
                    codes.append(rc_q); keep = True
                    emit_target_finished(index=i, status="failed",
                                  exit_code=rc_q, failure_kind="quick_failed",
                                  destination_unchanged=True)
                    continue
            else:
                # FULL PATH. fullsum selects its already-qualified direct
                # whole-source route or the bounded source-window route. Both
                # end in the same actual-pair review, candidate retention,
                # repair ceiling, status, and pair publication.
                progress.emit("stage", name=(
                    "full-batched" if full_batched else
                    "full" if full_direct else "full-windowed"))
                rc_f = run_stage(
                    [sys.executable, str(HERE / "fullsum.py"), str(src),
                     str(run_dir)], cancel, env=env).returncode
                if rc_f == fullsum.NOT_SUPPORTED:
                    # A window plan can still refuse a route when its bounded
                    # request cannot leave a final-pair margin. Fail loud and
                    # never fall through to the ledger or Quick.
                    print("  context unsupported, nothing published "
                          "(Full route)", file=sys.stderr)
                    codes.append(rc_f); keep = True
                    emit_target_finished(index=i, status="failed",
                              exit_code=rc_f,
                              failure_kind="context_unsupported",
                              destination_unchanged=True)
                    continue
                if rc_f != 0:
                    print("  the Full path produced nothing", file=sys.stderr)
                    codes.append(rc_f); keep = True
                    emit_target_finished(index=i, status="failed",
                              exit_code=rc_f, failure_kind="full_failed",
                              destination_unchanged=True)
                    continue

            if text_prep:
                if raw_text:
                    allocation_lease = runtime.output_directory_lock(
                        dest.parent, cancel.check)
                    allocation_lease.__enter__()
                    base = slug_from(cleaned_text)
                    dest_names = selection.format_output_filenames(
                        base, "text_prep", affix_kind=a.affix_kind, affix_text=a.affix)
                    dest = free_path(dest.with_name(dest_names[0]), taken)
                    destination_lease = runtime.destination_lock(dest, cancel.check)
                    destination_lease.__enter__()
                    title = base.replace("-", " ")
                cancel.check()
                progress.emit("stage", name="publish")
                try:
                    publish_one(cleaned, dest,
                                expected_sha256=cleaned_sha256)
                except Exception as e:
                    unchanged = getattr(e, "restored", True)
                    state = ("previous file restored" if unchanged else
                             "rollback incomplete; recovery files retained")
                    print(f"  publish failed, {state}: {e}", file=sys.stderr)
                    codes.append(1); keep = True
                    emit_target_finished(index=i, status="failed",
                                  exit_code=1, failure_kind="publish_failed",
                                  destination_unchanged=unchanged)
                    continue
                prepared_words = prep_report["output_words"]
                print(f"  -> {dest}   ({prepared_words}w)")
                history_outcome = "succeeded"
                codes.append(0)
                publication_committed = True
                emit_target_finished(index=i, status="succeeded",
                              exit_code=0, destination_unchanged=False,
                              outputs=[str(dest)])
                continue

            det, bri = run_dir / "detailed.md", run_dir / "brief.md"
            if raw_text and bri.exists():
                # NAME IT NOW, from the artifact rather than from the input.
                # "clipboard.summary.md" is not a filename when a second paste
                # arrives an hour later: it overwrites the first, and no
                # within-run check can see that. Deferred to here because the
                # Brief is what supplies the name.
                allocation_lease = runtime.output_directory_lock(
                    dest.parent, cancel.check)
                allocation_lease.__enter__()
                base = slug_from(bri.read_text(errors="replace"))
                dest_names = selection.format_output_filenames(
                    base, effective_mode.key, affix_kind=a.affix_kind,
                    affix_text=a.affix, affix_secondary=a.affix_secondary)
                dest = free_path(dest.with_name(dest_names[0]), taken)
                secondary_dest = dest.with_name(dest_names[1]) if len(dest_names) > 1 else None
                destination_lease = runtime.destination_lock(
                    dest, cancel.check, secondary=secondary_dest)
                destination_lease.__enter__()
                title = base.replace("-", " ")
            if not (det.exists() and bri.exists()):
                print("  both artifacts were not produced", file=sys.stderr)
                codes.append(1); keep = True
                emit_target_finished(index=i, status="failed",
                              exit_code=1, failure_kind="artifact_missing",
                              destination_unchanged=True)
                continue

            # 4. guard the pair update; every handled failure restores the old pair
            cancel.check()
            progress.emit("stage", name="publish")
            planned_brief = (pathlib.Path(doc.planned_outputs[1])
                             if doc and len(doc.planned_outputs) > 1 else None)
            target_bdest = planned_brief or (
                dest.with_name(dest_names[1])
                if len(dest_names) > 1 else None)
            try:
                bdest = publish_pair(det, bri, dest, bdest=target_bdest)
            except Exception as e:
                unchanged = getattr(e, "restored", True)
                state = ("previous files restored" if unchanged else
                         "rollback incomplete; recovery files retained")
                print(f"  publish failed, {state}: {e}", file=sys.stderr)
                codes.append(1); keep = True
                emit_target_finished(index=i, status="failed",
                              exit_code=1, failure_kind="publish_failed",
                              destination_unchanged=unchanged)
                continue
            # Every path reports its own counts, under different names and in
            # different files: the ledger writes compose-report.json, quick
            # writes short-report.json, and either Full route writes
            # full-report.json.
            # Reading only the first printed "?w" for every run -- a published
            # artifact reported as an unknown quantity, on the one line the
            # user actually reads.
            dw = bw = "?"
            full_status = None
            quality_status = None
            findings_count = 0
            review_state = None
            for name, d_key, b_key in (
                    ("full-report.json", "detailed_words", "brief_words"),
                    ("compose-report.json", "detailed", "brief"),
                    ("short-report.json", "detailed_words", "brief_words")):
                try:
                    rep = json.loads((run_dir / name).read_text())
                except Exception:
                    continue
                if name.startswith("compose"):
                    dw = rep.get(d_key, {}).get("words", "?")
                    bw = rep.get(b_key, {}).get("words", "?")
                else:
                    dw, bw = rep.get(d_key, "?"), rep.get(b_key, "?")
                if name == "full-report.json":
                    full_status = (rep.get("status"),
                                   rep.get("findings") or [])
                if name in {"full-report.json", "short-report.json"}:
                    quality_status = rep.get("status")
                    findings_count = len(rep.get("findings") or [])
                    review_state = rep.get("review")
                break
            print(f"  -> {dest}   ({dw}w)")
            print(f"  -> {bdest}  ({bw}w)")
            if full_status and full_status[0] not in (None, "pass"):
                print(f"  full status: {full_status[0]} "
                      f"({len(full_status[1])} open finding(s))")
            published_ok = True
            history_outcome = "succeeded"
            codes.append(0)
            publication_committed = True
            emit_target_finished(index=i, status="succeeded",
                          exit_code=0, destination_unchanged=False,
                          outputs=[str(dest), str(bdest)],
                          quality_status=quality_status,
                          findings_count=findings_count,
                          review_state=review_state)

        except Cancelled:
            cancelled = True
            keep = True
            history_outcome = "cancelled"
            codes.append(130)
            emit_target_finished(index=i, status="cancelled",
                          exit_code=130, destination_unchanged=True)
            print("  cancelled — nothing from the active target was published",
                  file=sys.stderr)
            emit_cancelled_targets(i + 1, targets, documents)
            break
        except Exception as exc:
            keep = True
            history_outcome = "failed"
            codes.append(1)
            print(f"  unexpected target failure: {exc}", file=sys.stderr)
            emit_target_finished(index=i, status="failed",
                          exit_code=1, failure_kind="internal_error",
                          destination_unchanged=False)
        finally:
            if route and source_words is not None and service_started is not None:
                try:
                    eta.record_target(
                        tmp / "run", route=route, source_words=source_words,
                        execution_sig=execution_sig, outcome=history_outcome,
                        service_wall_s=time.monotonic() - service_started,
                        admission_wait_s=admission_wait_s, parts=parts_count)
                except Exception as e:
                    print(f"  ETA history not recorded: {str(e)[:100]}", file=sys.stderr)
            if destination_lease is not None:
                destination_lease.__exit__(None, None, None)
            if allocation_lease is not None:
                allocation_lease.__exit__(None, None, None)
            if target_entered:
                target_lease.__exit__(None, None, None)
            if keep:
                print(f"  work kept: {tmp}", file=sys.stderr)
            else:
                shutil.rmtree(tmp, ignore_errors=True)

    # ONE terminal notification. Windows shows a single balloon at a time, so a
    # per-item balloon followed by a generic one raced and the first was lost.
    if work_lease:
        work_lease.__exit__(None, None, None)
    if cancel.requested() and not publication_committed:
        cancelled = True
    if cancelled:
        notify("Summer", "Cancelled — active work retained, no partial target published")
        progress.emit("job_finished", status="cancelled", exit_code=130)
        return 130
    notify("Text prep" if selected_mode.key == "text_prep" else "Summary",
           ("Clean text ready" if selected_mode.key == "text_prep"
            else "Detailed and Brief ready")
           if not any(codes) else "Completed with failures — see the console")
    successes = sum(code == 0 for code in codes)
    failures = sum(code != 0 for code in codes)
    if not failures:
        progress.emit("job_finished", status="succeeded", exit_code=0)
        return 0
    # One target reports its exact stage code, so a caller can act on WHY it
    # failed. Several targets cannot: a batch where one document lost its seal
    # and another exceeded its ceiling has no single true code, so it reports 1.
    rc = 1 if successes else (codes[0] if len(codes) == 1 else 1)
    progress.emit("job_finished",
                  status="partial" if successes else "failed", exit_code=rc)
    return rc


if __name__ == "__main__":
    sys.exit(main())
