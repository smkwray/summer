#!/usr/bin/env python3
"""Shared model routing, gateway transport, retries, and call evidence."""
from __future__ import annotations
import hashlib, json, os, re, shutil, subprocess, sys, tempfile, time, pathlib

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
import runtime
import gateway
import model_config
import mode_config
import route_caps
import json_contract
# Harness: which CLI actually runs the prompts. The pipeline logic is identical;
# only invocation and model names differ, so a new harness is a few lines here.
HARNESS = os.environ.get("HARNESS", "agy")


def _default_agy_binary():
    """Use the installed CLI on both hosts; never assume a Unix home path."""
    if os.name == "nt":
        return (shutil.which("agy") or shutil.which("agy.exe") or "agy")
    return "~/.local/bin/agy"


AGY = os.path.expanduser(os.environ.get("AGY_BIN", _default_agy_binary()))
CLAUDE = os.path.expanduser(os.environ.get("CLAUDE_BIN", "claude"))
OPENCODE = os.path.expanduser(os.environ.get("OPENCODE_BIN", "opencode"))
MUSE = os.path.expanduser(os.environ.get("MUSE_BIN", "~/.local/bin/muse"))
CODEX = os.path.expanduser(os.environ.get("CODEX_BIN", "codex"))
GROK = os.path.expanduser(os.environ.get("GROK_BIN", "~/.local/bin/grok"))
CURSOR = os.path.expanduser(os.environ.get("CURSOR_BIN", "cursor-agent"))
BIN = {"agy": AGY, "claude": CLAUDE, "opencode": OPENCODE,
       "muse": MUSE, "codex": CODEX, "grok": GROK, "cursor": CURSOR}
GATEWAY_HARNESSES = model_config.gateway_harnesses()
BIN.update({name: None for name in sorted(GATEWAY_HARNESSES)})
HARNESSES = tuple(BIN)
# Harnesses that take the prompt on stdin rather than as an argument.
STDIN_HARNESSES = {"codex", "cursor"}
# How each harness receives the composed prompt. Windows caps a whole command
# line at 32,767 characters, so a route that can only take the prompt on argv
# is ineligible for a payload near that: it is skipped like any other route
# that cannot serve the call, rather than failing mid-run on one platform.
NON_ARGV_TRANSPORT = {"codex": "stdin", "cursor": "stdin",
                      "grok": "file", "muse": "file"}
ARGV_SAFE_BYTES = int(os.environ.get("ARGV_SAFE_BYTES", "24000"))
VALID_ROLES = frozenset(mode_config.ROLES)

# JSON-producing stages can request the standard OpenAI-compatible JSON mode
# without knowing which gateway is serving them. The caller still validates the
# returned object against its own exact schema; this only prevents a compliant
# gateway/model from spending a long call on prose when the stage requires JSON.
JSON_REQUEST_OPTIONS = {"response_format": {"type": "json_object"}}

# One resolver used by argv construction and evidence. A saved timing without
# this value is not reproducible and must not be used for an ETA cohort. The
# exact model-level choices live in models.json; these are only fallbacks for
# direct CLI runs that do not select a model setting explicitly.
OPTIONS = {
    "codex": ("effort", "CODEX_EFFORT", "medium"),
    "muse": ("effort", "MUSE_EFFORT", "medium"),
    "opencode": ("variant", "OPENCODE_VARIANT", "high"),
    "grok": ("effort", "GROK_EFFORT", "high"),
    "claude": ("effort", "CLAUDE_EFFORT", "high"),
}


def _model_config() -> dict:
    try:
        return model_config.full()
    except Exception:
        return {}


def model_setting(harness: str, model: str = "") -> dict:
    """Return the exact setting declaration for one roster model."""
    return model_config.model_setting(harness, model)


def model_setting_values(harness: str, model: str = "") -> tuple[str, ...]:
    return model_config.setting_values(harness, model)


def model_variants(harness: str, model: str = "") -> dict:
    return model_config.model_variants(harness, model)


def local_harnesses() -> frozenset[str]:
    return model_config.local_harnesses()


def effort_values(harness: str, model: str = "") -> tuple[str, ...]:
    """Return values for this exact model, never a guessed provider union."""
    if model:
        return model_setting_values(harness, model)
    seen = []
    for item in (((_model_config().get(harness) or {}).get("_model_settings")
                  or {}).values()):
        for value in item.get("values", []) if isinstance(item, dict) else ():
            if isinstance(value, str) and value not in seen:
                seen.append(value)
    return tuple(seen)


def _role_options() -> dict:
    raw = os.environ.get("SUMM_ROLE_OPTIONS", "")
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise Abort(f"invalid frozen role options: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise Abort("invalid frozen role options: expected an object")
    return value


def stage_role(stage: str) -> str:
    name = str(stage).lower()
    if name.startswith("replan") or name.startswith("plan"):
        return "plan"
    if (name.startswith("audit") or name.startswith("reaudit")
            or "-audit" in name or "-reaudit" in name):
        return "audit"
    if (name.startswith("repair") or name.startswith("revise")
            or "-repair" in name or "-revise" in name):
        return "repair"
    return "write"


def effective_option(harness: str, model: str = "", role: str = "") -> dict:
    selected = _role_options().get(role) if role else None
    if isinstance(selected, dict):
        selected = [selected]
    if isinstance(selected, list):
        for item in selected:
            if (isinstance(item, dict) and item.get("harness") == harness
                    and item.get("model") == model):
                option = item.get("option")
                if isinstance(option, dict):
                    return dict(option)
    spec = OPTIONS.get(harness)
    if spec:
        name, env, _default = spec
        if env in os.environ:
            return {name: os.environ[env]}
    setting = model_setting(harness, model)
    option_name = setting.get("option")
    if option_name and option_name != "model":
        value = model_config.setting_default(harness, model, role)
        if isinstance(value, str) and value:
            return {option_name: value}
    if spec:
        name, env, default = spec
        return {name: default}
    if harness in GATEWAY_HARNESSES:
        options = gateway.request_options(harness, model)
        if not options:
            return {}
        # Keep the visible label and call ledger compact for the common case,
        # while retaining arbitrary configured options as one deterministic
        # value if a device declares more than one.
        if len(options) == 1 and all(
                isinstance(value, (str, int, float, bool)) or value is None
                for value in options.values()):
            return dict(options)
        return {"request_options": json.dumps(
            options, ensure_ascii=False, sort_keys=True, separators=(",", ":"))}
    return {}


def effective_option_env(harnesses) -> dict:
    """Snapshot effective option values for selected harnesses.

    The command builders and call ledger resolve options through
    :func:`effective_option`; callers that queue work can use this helper to
    capture the same values before ambient environment settings change.  The
    returned keys are the environment variables that the engine reads, not
    model or harness identifiers.
    """
    if isinstance(harnesses, str):
        harnesses = (harnesses,)
    env = {}
    for harness in dict.fromkeys(harnesses):
        spec = OPTIONS.get(harness)
        if spec:
            name, variable, _default = spec
            env[variable] = effective_option(harness)[name]
            continue
        if harness in GATEWAY_HARNESSES:
            options = gateway.request_options(harness)
            if options:
                env[f"SUMM_{harness.upper()}_REQUEST_OPTIONS"] = json.dumps(
                    options, ensure_ascii=False, sort_keys=True,
                    separators=(",", ":"))
    return env


def split_entry(entry: str, default: str):
    """A chain entry is "harness:model", or a bare model on the run's harness.

    Qualifying an entry is what lets ONE run split roles across harnesses --
    a local model plans and revises while a hosted one audits -- without the
    pipeline learning anything about model identity. It stays a models.json
    edit, which is the only place a model may be named.

    The prefix is only honoured when it names a real harness, because model
    identifiers contain colons too: a local tag like `model-family:32b` must
    survive whole when `model-family` is not a configured harness.
    """
    return model_config.split_entry(entry, default)


def run_root(workdir: pathlib.Path) -> pathlib.Path:
    """The run directory, from any per-stage directory beneath it."""
    return workdir.parent if (workdir.parent / "rv").is_dir() else workdir



# opencode's default coding agent carries unrelated instructions and tool
# schemas. A lean agent preserves capacity for the document transformation;
# nothing here needs a tool because the source travels inline and the answer is
# JSON. Config must sit in --dir because OPENCODE_CONFIG is not honoured by
# `run`.
LEAN_AGENT = "lean"
LEAN_CONFIG = {
    "$schema": "https://opencode.ai/config.json",
    "agent": {LEAN_AGENT: {
        "description": "One-shot transform. No tools, no files.",
        "mode": "primary",
        "prompt": "You transform the text in the user's message and return the "
                  "result. Nothing else.",
        "tools": {t: False for t in (
            "bash", "edit", "write", "read", "grep", "glob", "list",
            "patch", "todowrite", "todoread", "webfetch", "task")},
    }},
}


# Grok takes a schema so its answer arrives as one validated object instead of
# prose the parser has to dig through. It is deliberately permissive: _cmd() is
# generic and knows nothing about stages, and the real per-stage contract is
# still enforced by the parse_strict callback. additionalProperties must be true
# because xAI's structured output otherwise defaults it to false and would strip
# every field the stage actually asked for.
GROK_SCHEMA = '{"type":"object","properties":{},"additionalProperties":true}'

# Cursor Agent's project permission file. Deny takes precedence over any
# inherited allow rules from the user's global cli-config. Ask mode can still
# search and call tools; this file plus a sterile workspace is what keeps the
# model on the composed stdin prompt alone.
CURSOR_PERMISSIONS = {
    "permissions": {
        "allow": [],
        "deny": [
            "Shell(*)",
            "Read(**)",
            "Read(/*)",
            "Write(**)",
            "Write(/*)",
            "WebFetch(*)",
            "Mcp(*:*)",
        ],
    }
}


def cursor_workspace(workdir: pathlib.Path) -> pathlib.Path:
    """Return a sterile Cursor workspace for one run, never the evidence root.

    The directory holds only harness-owned Cursor config. It must not contain
    source text, composed prompts, outputs, retained call evidence, or project
    rules — those would become searchable Ask-mode context.
    """
    root = run_root(pathlib.Path(workdir).resolve())
    ws = (root / ".cursor-harness" / "ws").resolve()
    cfg_dir = ws / ".cursor"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "cli.json").write_text(
        json.dumps(CURSOR_PERMISSIONS, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    # Empty MCP roster so a sterile workspace cannot inherit project servers
    # through a path that walks into the Summer tree.
    (cfg_dir / "mcp.json").write_text("{}\n", encoding="utf-8")
    return ws


def cli_cwd(harness: str, workdir: pathlib.Path) -> pathlib.Path:
    """Process cwd for one harness call; Cursor must not cwd into the run root."""
    workdir = pathlib.Path(workdir).resolve()
    if harness == "cursor":
        return cursor_workspace(workdir)
    return workdir


def _prompt_file(workdir: pathlib.Path, prompt: str) -> pathlib.Path:
    """Write `prompt` to a unique file in `workdir`; return its resolved path.

    Unique per call: workers and retries overlap on one work directory, so a
    fixed name would let two calls clobber each other's prompt. UTF-8 bytes,
    descriptor closed even on failure.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="grok-prompt-", suffix=".txt",
                                dir=str(workdir))
    try:
        os.write(fd, prompt.encode("utf-8"))
    finally:
        os.close(fd)
    return pathlib.Path(name).resolve()


def grok_payload(stdout: str) -> str:
    """Grok's JSON envelope -> the stage object, or raise.

    NEVER fall back to the envelope's `text`. A model may narrate its intentions
    there and exit 0 with no answer behind it. Accepting narration is how a
    failed call becomes an approved artifact.
    """
    env = json.loads(stdout)
    if not isinstance(env, dict):
        raise ValueError("grok stdout is not a JSON object")
    if env.get("type") == "error":
        raise ValueError(f"grok error: {str(env.get('message'))[:120]}")
    stop = env.get("stopReason")
    if stop != "end_turn":
        raise ValueError(f"grok stopped with {stop!r}, not 'end_turn'")
    if env.get("structuredOutputError"):
        raise ValueError(f"grok schema error: {env['structuredOutputError']!r}")
    obj = env.get("structuredOutput")
    if not isinstance(obj, dict) or not obj:
        raise ValueError("grok returned no validated object; text="
                         f"{str(env.get('text'))[:100]!r}")
    return json.dumps(obj, ensure_ascii=False)


def cursor_payload(stdout: str) -> str:
    """Cursor Agent's print JSON envelope -> the stage bytes, or raise.

    ``--output-format json`` wraps the model answer in a transport object. The
    stage contract lives in ``result``; accepting any other field would let a
    failed or partial turn look like a finished answer.
    """
    env = json.loads(stdout)
    if not isinstance(env, dict):
        raise ValueError("cursor stdout is not a JSON object")
    if env.get("is_error") or env.get("type") == "error":
        raise ValueError(
            f"cursor error: {str(env.get('result') or env.get('message') or env)[:120]}")
    if env.get("type") != "result" or env.get("subtype") != "success":
        raise ValueError(
            f"cursor incomplete: type={env.get('type')!r} "
            f"subtype={env.get('subtype')!r}")
    result = env.get("result")
    if not isinstance(result, str) or not result.strip():
        raise ValueError("cursor returned no result text")
    return result


def _lean_dir(workdir: pathlib.Path) -> pathlib.Path:
    d = run_root(workdir) / ".oc"
    d.mkdir(parents=True, exist_ok=True)
    (d / "opencode.json").write_text(json.dumps(LEAN_CONFIG, indent=2))
    return d


def _cmd(harness: str, model: str, workdir: pathlib.Path, prompt: str,
         option: dict | None = None, timeout_seconds: int | None = None):
    option = effective_option(harness, model) if option is None else option
    # ABSOLUTE, always. A relative --work-dir used to reach grok as a
    # relative --cwd, which it resolves against its own working directory
    # rather than ours: every run died in under a second with "Failed to
    # set working directory". Harnesses differ in how forgiving they are,
    # so this is fixed once here rather than per branch.
    workdir = pathlib.Path(workdir).resolve()
    if harness in GATEWAY_HARNESSES:
        raise Abort(f"{harness} uses the configured gateway, not a CLI command")
    if harness == "codex":
        # Prompt arrives on stdin (the trailing "-"). read-only sandbox and
        # approval_policy=never: the payload is inline, so it has no reason to
        # touch the filesystem, and nobody is present to answer a prompt.
        return [CODEX, "exec", "--cd", str(workdir), "--model", model,
                "-c", f"model_reasoning_effort={option['effort']}",
                "--disable", "fast_mode", "--skip-git-repo-check", "--color", "never",
                "--sandbox", "read-only", "-c", "approval_policy=never", "-"]
    if harness == "muse":
        # `exec` is the headless one-prompt runner. --workspace is the permission
        # root and must be the RUN directory: the prompts name files in rv/ and
        # ledger/, and rooting it at one of those makes the other unreadable.
        # This is a PAID model, so the run is bounded and stripped of everything
        # it does not need: no web, no writes, no shell, and a step cap. Approval
        # is off only so that reading the source does not block on a prompt
        # nobody is there to answer.
        root = run_root(workdir)
        return [MUSE, "exec", "--model", model,
                "--prompt-file", str(_prompt_file(workdir, prompt)),
                "--reasoning-effort", option["effort"],
                "--workspace", str(root),
                "--disable-web-tools", "--disable-write", "--disable-shell",
                "--disable-approval", "--no-session-log",
                # ONE step. These calls are one-shot transformations: the controller
                # already holds the bytes, so there is no reason for a fetch turn
                # followed by a reasoning turn. Twelve steps re-sent each tool
                # result on every later step and turned ~672k tokens of payload
                # into 4.2M. A model that needs an agent loop to read a supplied
                # part and emit one JSON object is not suitable for this pipeline.
                "--max-model-steps", "1"]
    if harness == "opencode":
        # `run` is the headless entrypoint; the prompt is positional, like agy.
        # Runs under a no-tools agent (see _lean_dir): the payload is already
        # inline and the answer is JSON, so a coding agent's tool schemas are
        # pure overhead on a quota. --dir is that agent's config directory,
        # which is also the tightest possible file scope -- it holds one file.
        # --title is not cosmetic: without it opencode spends a SECOND model call
        # per invocation asking for a session title. On a 36-call document that is
        # 36 calls of pure waste against the same quota the document is competing
        # for, and the title is never read by anything.
        return [OPENCODE, "run", "-m", model,
                "--variant", option["variant"],
                "--agent", LEAN_AGENT, "--title", "summ",
                "--dir", str(_lean_dir(workdir)), prompt]
    if harness == "grok":
        # THE PROMPT CANNOT RIDE ON ARGV. Windows CreateProcessW caps the whole
        # command line at 32,767 characters, so a document prompt may not fit.
        # AGENTS.md requires both platforms to behave identically. The prompt
        # goes in a unique file in the work directory.
        # That file is summ'er's own composed prompt, NOT a path to the source
        # document, which is the distinction the payload rule actually protects.
        # --verbatim sends those bytes as given; --no-auto-update stops a
        # headless run blocking on an updater.
        #
        # READ TOOLS ARE GRANTED, against this project's preference for a
        # tool-less harness, because this CLI may otherwise narrate an intent to
        # read a large prompt rather than answer it. The source still arrives
        # inline; read_file exists only to let the CLI proceed.
        # read_file alone is enough -- the offload notice carries the path, so
        # grep and list_dir buy nothing and are not granted.
        #
        # STRUCTURED OUTPUT, not plain text, is used for the stopReason rather
        # than for speed. Plain text carries NO signal distinguishing a
        # finished answer from an abandoned turn, and an abandoned turn that
        # looked like an answer is precisely what cost six benchmark runs.
        # stopReason gives that signal definitively. The cost of schema mode is
        # that a refusing model can emit a schema-valid stub, so shortsum holds
        # a substance floor against exactly that.
        return [GROK, "--no-auto-update",
                "--prompt-file", str(_prompt_file(workdir, prompt)),
                "--verbatim", "-m", model,
                "--effort", option["effort"],
                "--cwd", str(workdir),
                "--permission-mode", "dontAsk", "--sandbox", "read-only",
                "--tools", "read_file", "--deny", "MCPTool(*)",
                "--no-subagents", "--disable-web-search", "--no-plan",
                "--json-schema", GROK_SCHEMA, "--output-format", "json"]
    if harness == "claude":
        # NOTE: --add-dir is variadic and will swallow the positional prompt, so it
        # is deliberately not used; the runner already cwd's into the work dir and
        # Claude Code can read files there.
        return [CLAUDE, "-p", "--model", model, "--effort",
                option["effort"], prompt]
    if harness == "agy":
        minutes = max(1, int((int(timeout_seconds or 1800) + 59) // 60))
        return [AGY, "--model", model, "--add-dir", str(workdir),
                "--print-timeout", f"{minutes}m", "-p", prompt]
    if harness == "cursor":
        # Headless print mode. Prompt rides on stdin: cursor-agent has no
        # prompt-file flag, and a Full prompt exceeds the Windows argv cap.
        # Ask mode still permits search/tools against the workspace, so the
        # workspace must be sterile (no source, prompts, evidence, or rules)
        # and project permissions must deny shell/read/write/web/MCP. --trust
        # only suppresses the headless workspace prompt for that sterile dir.
        # JSON print format is unwrapped by cursor_payload.
        ws = cursor_workspace(workdir)
        return [CURSOR, "-p", "--mode", "ask", "--model", model,
                "--output-format", "json",
                "--sandbox", "enabled", "--trust",
                "--workspace", str(ws)]
    # Never fall through to a default CLI. A mistyped qualifier used to reach
    # agy carrying another harness's model identifier, which fails as "unknown
    # model" and reads like a roster problem rather than a typo.
    raise Abort(f"unknown harness {harness!r}; expected one of {', '.join(HARNESSES)}")
# Each ROLE gets an ordered chain. The first model is used; on a *capacity*
# failure (quota, rate limit, auth-expiry) we move to the next. An unusable
# stage response (empty, malformed, truncated, wrong-schema) also advances
# because no candidate was produced. A valid semantic or length finding stays
# with that producer: one same-model correction, then retain. Backup is not a
# second opinion on a capsule that merely ran long.
#
# Falling back is safe here precisely because the pipeline is gated: a weaker
# model that flattens an explanation or drops a caveat is caught by the audit
# and sent to repair. A valid pair remains available when review or repair is
# unavailable; its report names that bounded assurance state. The chain never
# shops for a reviewer that will simply return a more convenient verdict.
def _chain(var, default):
    return os.environ.get(var, default).split()

def _active_roles() -> tuple[str, ...]:
    """Roles required by this process's selected product mode.

    A specialized local gateway may intentionally expose only the roles a
    product route uses.  The CLI freezes this value before importing a stage;
    direct imports retain the historical all-role default.
    """
    raw = os.environ.get("SUMM_ACTIVE_ROLES", "").strip()
    if not raw:
        return mode_config.ROLES
    roles = tuple(dict.fromkeys(part.strip().lower()
                               for part in re.split(r"[\s,]+", raw)
                               if part.strip()))
    unknown = sorted(set(roles) - set(mode_config.ROLES))
    if not roles or unknown:
        raise Abort("SUMM_ACTIVE_ROLES contains invalid roles: "
                    + ", ".join(unknown or [raw]))
    return roles


ACTIVE_ROLES = _active_roles()


def _models(active_roles=ACTIVE_ROLES):
    """Model identifiers come from models.json, not from code.

    Free models appear and disappear, so the volatile part is data the owner can
    edit without touching the pipeline. "Model-agnostic" means the selection,
    sealing, fidelity and publication logic never branches on model identity --
    not that identifiers stop existing. Env vars still win, for experiments.
    """
    try:
        configured = model_config.roster()
        resolved = model_config.role_chains(
            HARNESS, mode_config.ROLES, roster_value=configured)
    except Exception as e:
        raise Abort(f"model roster unreadable ({str(e)[:60]})")
    out = {}
    for role in mode_config.ROLES:
        if role not in active_roles:
            out[role] = []
            continue
        entries = resolved.get(role) or []
        if not entries:
            raise Abort(f"models.json: {HARNESS}.{role} is empty")
        out[role] = entries
    return out


# Planning is the largest block of calls, but it is NOT the place to economise.
# Planning creates the capsules that become the product, so a thinner inventory
# directly produces a thinner summary.
_R = _models()
PLAN_MODELS, MODELS = _R["plan"], _R["write"]
AUDIT_MODELS, REPAIR_MODELS = _R["audit"], _R["repair"]


class Abort(RuntimeError):
    pass


class NoCandidate(Abort):
    """Every eligible producer was asked and none returned a usable answer.

    Distinct from cancellation, authentication, a configuration error, a
    deadline, or a defect here. ``kind`` preserves why the last eligible route
    failed so callers do not mistake a semantic contract failure for a capacity
    failure and recursively shrink work that already fit. ``attempts`` keeps
    earlier typed failures so a later unavailable provider cannot erase them.
    """

    def __init__(self, message: str, *, kind: str | None = None,
                 detail: str = "", attempts=None, recovery_kind: str | None = None):
        super().__init__(message)
        self.kind = kind
        self.detail = detail
        self.attempts = list(attempts or [])
        self.recovery_kind = recovery_kind if recovery_kind is not None else kind


def validate_local_only():
    """Require every local-only entry to name a roster-declared local backend."""
    if not os.environ.get("SUMM_LOCAL_ONLY"):
        return
    for role, entries in (("plan", PLAN_MODELS), ("write", MODELS),
                          ("audit", AUDIT_MODELS), ("repair", REPAIR_MODELS)):
        for entry in entries:
            harness, _model = split_entry(entry, HARNESS)
            if harness not in local_harnesses():
                raise Abort(f"local-only mode requires local entries; "
                            f"{role} resolved to {harness!r}")


validate_local_only()


# ------------------------------------------------------------------- model ---
# Transport failures are bounded. A completed response that violates a stage
# contract advances to the next configured producer; a completion-unknown
# response is never replayed. Quota exhaustion is not retried because it does
# not clear within a run and retrying just burns time.
TRANSIENT = re.compile(
    r"operation timed out|connection reset|i/o timeout|EOF|temporarily unavailable|"
    r"read tcp|dial tcp|503|502|500 Internal|Eligibility check failed|"
    r"unexpected EOF|broken pipe|"
    # Local contention, not the provider: opencode keeps session state in a
    # SQLite database, and concurrent `opencode run` invocations can collide on
    # it. The lock clears immediately, so treating it as fatal would abort a
    # document over transient local contention.
    r"database is locked|SQLITE_BUSY|database table is locked|"
    # A stream-idle timeout is a transport stall and may clear on the next call;
    # treating it as a semantic failure can unnecessarily demote the route.
    r"stream idle timeout", re.I)
# Appended to every composed prompt. A model that opens a tool instead of
# answering costs a whole call, and a large Corpus audit showed a head-only
# instruction being lost in an audit-sized prompt.
ANSWER_NOW = ("Answer directly from the material above. Do not call a tool, "
              "read a file, search, or begin another step. Return the answer "
              "this prompt asks for now, and nothing else.")
RETRIES = int(os.environ.get("RETRIES", "3"))
# Warm-on-demand budget: total seconds to keep re-polling a model that reports
# "warming" (503) before giving up. WARM_POLL_MAX_S caps how long a single
# sleep is, so readiness is
# detected within that window rather than after a full Retry-After.
WARM_BUDGET_S = float(os.environ.get("WARM_BUDGET_S", "900"))
WARM_POLL_MAX_S = float(os.environ.get("WARM_POLL_MAX_S", "30"))
CALL_TIMEOUT = int(os.environ.get("CALL_TIMEOUT", "1800"))
CONTINUE_JSON = (
    "Continue the incomplete JSON from the exact next character. "
    "Do not repeat earlier text, markdown, or commentary."
)
# Runaway guard only: a model that never closes JSON. Not a token cutoff.
JSON_CONTINUE_MAX_CHARS = 1_000_000

def _json_stage(call_option) -> bool:
    fmt = (call_option or {}).get("response_format")
    return isinstance(fmt, dict) and fmt.get("type") in {
        "json_object", "json_schema"}


def _json_prefix(raw) -> bool:
    start = (raw or "").lstrip()[:1]
    return start in "{["


def _with_content(result, content, finish_reason):
    return gateway.ChatCompletionResult(
        content=content, finish_reason=finish_reason,
        usage=result.usage, response_id=result.response_id,
        raw_response=result.raw_response,
        elapsed_seconds=result.elapsed_seconds,
        response_headers=result.response_headers,
        dispatch_budget=result.dispatch_budget)


def _finish_json(result, harness, model, prompt, call_option, role,
                 planned_output_words, output_overhead_tokens,
                 warm_deadline, remaining_seconds, stage):
    """Keep generating until one complete JSON document exists.

    A token cap is not a reason to fail. A finished object is success even
    if the engine stopped on length. An unfinished object is continued until
    it parses, the model adds nothing, or the document is no longer JSON.
    Extra data after a complete object is left for the exact parser to reject.
    """
    assembled = result.content or ""
    status = json_contract.document_status(assembled)
    if status == "complete":
        return _with_content(result, assembled, "stop")
    if status != "incomplete" or not _json_prefix(assembled):
        return result
    continue_option = dict(call_option or {})
    continue_option.pop("response_format", None)
    while json_contract.document_status(assembled) == "incomplete":
        if len(assembled) >= JSON_CONTINUE_MAX_CHARS:
            break
        print(f"    [{stage}] {model} continuing JSON "
              f"({len(assembled)} chars)", flush=True)
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": assembled},
            {"role": "user", "content": CONTINUE_JSON},
        ]
        while True:
            try:
                result = gateway.chat(
                    harness, model, CONTINUE_JSON,
                    request_options=continue_option, role=role,
                    planned_output_words=planned_output_words,
                    output_overhead_tokens=output_overhead_tokens,
                    semantic_retry=False, messages=messages,
                    continuation=True)
                break
            except gateway.GatewayError as ge:
                if ge.kind == "warming" and time.monotonic() < warm_deadline:
                    wait = min(ge.retry_after or 30.0, WARM_POLL_MAX_S)
                    wait_for = (wait if remaining_seconds is None else
                                min(wait, max(0, remaining_seconds())))
                    if wait_for <= 0:
                        raise
                    print(f"    [{stage}] {model} warming — re-poll in "
                          f"{int(wait_for)}s", flush=True)
                    time.sleep(wait_for)
                    continue
                raise
        chunk = result.content or ""
        if not chunk:
            break
        assembled = assembled + chunk
        status = json_contract.document_status(assembled)
        if status == "complete":
            return _with_content(result, assembled, "stop")
        if status != "incomplete":
            break
    return _with_content(result, assembled, result.finish_reason)


# Capacity failures: try the next model in the chain. Retrying the same one is
# pointless -- a quota does not clear inside a run.
# An expired agy token makes the CLI open a browser and wait. In a Raycast or
# cron context nobody answers, so the process dies with an empty log and the run
# looks like a mysterious failure. Name it explicitly: no model in the chain can
# help, because they share the same credential.
AUTH = re.compile(
    r"401|403|Authentication required|authentication (?:failed|timed out|cancelled)|"
    r"Please visit the URL|oauth|sign ?in to continue", re.I)

# "Authentication required" used to sit in CAPACITY too, so an expired
# credential walked the entire chain -- every model of which shares it -- and
# reported the last one's failure. AUTH was defined for exactly this and never
# consulted.
CAPACITY = re.compile(
    r"quota reached|quota exceeded|RESOURCE_EXHAUSTED|rate.?limit|429|"
    r"too many requests|upgrade your subscription|"
    # codex says "hit your usage limit ... Switch to another model now" and
    # names a reset time. Not matching it meant the chain never fell over: a
    # depleted 5-hour quota killed three parts instead of demoting one model.
    # claude prints "You've hit your session limit · resets 7:50pm" to stdout
    # with exit 0. Unmatched, it was recorded as an unusable response, and
    # with the fallback also at its limit the run published nothing.
    r"usage limit|session limit|switch to another model|try again at", re.I)


_PG = None


def _pg():
    """Load progress.py once. A document makes ~31 model calls, and re-executing
    the module on each of them is pure waste."""
    global _PG
    if _PG is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location("pg", HERE / "progress.py")
        _PG = importlib.util.module_from_spec(spec); spec.loader.exec_module(_PG)
    return _PG


def _record(workdir: pathlib.Path, rec: dict, t0: float, outcome: str, detail: str = ""):
    """Append one line per model call to `calls.jsonl` in the run directory.

    Every call in the pipeline passes through run(), so this is the one place
    that can say what a document actually cost: which harness and model answered,
    how long it took, how big the payload was, and how often a call was retried,
    truncated, or fell through the chain. Those were previously observations
    someone made once by watching a terminal, which is why the budget figures in
    the project's own notes had no producer behind them.

    Bytes, not tokens: no harness here reports token accounting, and a tokenizer
    guess that varies by model would be a worse number than an exact one.

    Instrumentation must never be able to fail a run that is otherwise fine, so
    a write error degrades to a one-time warning. Appends are single short lines
    in append mode, which is what makes them safe against the WORKERS pool.
    """
    rec = dict(rec, outcome=outcome, seconds=round(time.monotonic() - t0, 3))
    if detail:
        rec["detail"] = detail
    # The progress stream carries the same fact for a live watcher. One call
    # site, so a model call cannot appear in the cost ledger and not in the UI.
    try:
        pg = _pg()
        pg.emit("model_call", role=rec.get("role", rec.get("stage")),
                harness=rec.get("harness"),
                model=rec.get("model"), attempt=rec.get("attempt"),
                outcome=outcome, seconds=rec["seconds"],
                **{k: rec[k] for k in ("effort", "variant") if k in rec})
    except Exception:
        pass
    try:
        with (run_root(workdir) / "calls.jsonl").open("a") as fh:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
    except Exception as e:
        global _RECORD_WARNED
        if not _RECORD_WARNED:
            _RECORD_WARNED = True
            print(f"    [calls.jsonl] not recording call costs: {str(e)[:80]}",
                  flush=True)
        return
    # The CLI invokes ledger/shortsum as children, so it cannot observe a
    # completed call while the stage child is blocked.  The call recorder is
    # the production path that knows the current topology; use the same
    # progress stream to refresh a conservative remaining-time range.
    try:
        route = os.environ.get("SUMM_ETA_ROUTE")
        if route:
            import eta
            result = eta.live_remaining(
                run_root(workdir), route=route,
                source_words=int(os.environ["SUMM_ETA_SOURCE_WORDS"]),
                execution_sig=os.environ["SUMM_ETA_SIGNATURE"],
                parts=(int(os.environ["SUMM_ETA_PARTS"])
                       if os.environ.get("SUMM_ETA_PARTS") else None))
            if result:
                _pg().emit("eta", **result)
    except Exception:
        # ETA is advisory instrumentation and must never change the run's
        # success/failure semantics.
        pass


_RECORD_WARNED = False
_ATTEMPTS_USED = {}
_DEAD_HARNESSES = {}
# (harness, model) routes that reported exhausted capacity in this target. A
# quota does not clear inside a run, so a later stage must not try them again.
_DEAD_ROUTES = {}
# (target, harness, model) -> count of transport failures that produced no
# output. The first is retried once; the second retires the route.
_STALLS = {}
STALL_LIMIT = 2


def last_ok_route(workdir: pathlib.Path, stage: str, fallback) -> list[str]:
    """The exact route that returned a candidate for ``stage``, as a one-entry
    chain, else the first configured entry. A correction of a candidate goes
    back to the producer that wrote it; it never shops the chain."""
    fallback = [str(item) for item in (fallback or []) if str(item).strip()]
    default = fallback[:1]
    path = pathlib.Path(workdir) / "calls.jsonl"
    if not path.is_file():
        return default
    last = None
    try:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("stage") == stage and rec.get("outcome") == "ok":
                last = rec
    except (OSError, ValueError):
        return default
    if not last or not last.get("harness") or not last.get("model"):
        return default
    return [f"{last['harness']}:{last['model']}"]


def _capacity_signal(stdout: str, stderr: str) -> str:
    """A short quota diagnostic on either stream. claude prints
    "You've hit your session limit" to stdout with exit 0; codex prints its
    usage-limit text to stderr. A real answer that merely contains the words
    is not a signal: stdout is consulted only when it is short and carries no
    JSON object."""
    for text in (stderr, stdout if (len(stdout) < 600
                                    and not any(_balanced_objects(stdout)))
                 else ""):
        text = (text or "").strip()
        if text and CAPACITY.search(text):
            return text.splitlines()[0][:200]
    return ""


def _work_key(workdir: pathlib.Path) -> str:
    """Stable per-target key for dead routes, stalls, and the attempt budget.

    Corpus runs every stage in one process but under nested directories, so
    keying on the work directory made "target-wide" state stage-local: one
    quota-exhausted route was retried in every nested stage. The controller
    sets SUMM_TARGET_KEY once per publication target; the work directory is
    the fallback for a single-stage caller."""
    explicit = os.environ.get("SUMM_TARGET_KEY", "").strip()
    if explicit:
        return explicit
    try:
        return str(pathlib.Path(workdir).resolve())
    except Exception:
        return str(workdir)


def _target_attempt_limit() -> int | None:
    raw = os.environ.get("SUMM_TARGET_ATTEMPT_LIMIT", "").strip()
    if not raw:
        return None
    try:
        limit = int(raw)
    except ValueError as exc:
        raise Abort("invalid SUMM_TARGET_ATTEMPT_LIMIT") from exc
    if limit < 1:
        raise Abort("SUMM_TARGET_ATTEMPT_LIMIT must be positive")
    return limit


def _claim_attempt(workdir: pathlib.Path, stage: str) -> int:
    """Claim one real provider attempt from the target-wide physical budget."""
    limit = _target_attempt_limit()
    # Attempt accounting is control-plane state, but it must not depend on
    # optional evidence instrumentation.  In particular, run_root() can be
    # unavailable when a caller's evidence directory is read-only.  Resolve
    # the caller-provided work directory directly so that logging failures
    # cannot turn a successful provider response into a failed run.
    try:
        key = _work_key(workdir)
    except Exception:
        key = str(workdir)
    used = _ATTEMPTS_USED.get(key, 0)
    if limit is not None and used >= limit:
        raise Abort(f"{stage}: target model-attempt limit {limit} exhausted")
    used += 1
    _ATTEMPTS_USED[key] = used
    return used


def run(prompt: str, workdir: pathlib.Path, chain, stage: str, validate=None,
        gateway_options: dict | None = None, role: str | None = None,
        planned_output_words: int | None = None,
        output_overhead_tokens: int | None = None,
        semantic_retry: bool = False) -> str:
    """`validate` is called on stdout; raising means the response is unusable
    and advances to the next configured producer.

    ``stage`` is descriptive evidence.  Callers for the Corpus route pass an
    explicit ``role`` so a name such as ``corpus-plan`` cannot silently select
    the wrong roster or effort setting.  The optional argument preserves the
    existing document and text-preparation callers while they migrate.

    Without it, a response that arrives truncated exits 0 with plausible partial
    output, so the pipeline treated a cut-off stream as a real answer and failed
    the section. Delivery failures are exactly what a bounded retry is for."""
    prompt = prompt.rstrip() + "\n\n" + ANSWER_NOW
    if role is not None and role not in VALID_ROLES:
        raise ValueError(f"unknown model role: {role}")
    effective_role = role or stage_role(stage)
    if isinstance(chain, str):
        chain = [chain]
    validate_local_only()
    resolved = [split_entry(e, HARNESS) for e in chain]
    if (os.environ.get("SUMM_LOCAL_ONLY") and
            any(harness not in local_harnesses() for harness, _ in resolved)):
        raise Abort(f"{stage}: local-only mode resolved a non-local fallback")
    # An authentication failure kills a HARNESS, not the run. Every model behind
    # one CLI shares one credential, so retrying another of them cannot help --
    # but a chain that reaches a different harness reaches a different
    # credential, which is the whole point of qualifying entries.
    # One target invokes this function for several roles.  Authentication is
    # a harness-wide condition, so remember it across those calls and do not
    # spend another stage trying a different model behind the same credential.
    # This state is process-local control data; a fresh target gets a fresh
    # mapsum child process and therefore cannot inherit it.
    target_key = _work_key(workdir)
    dead = _DEAD_HARNESSES.setdefault(target_key, set())
    dead_routes = _DEAD_ROUTES.setdefault(target_key, set())
    stalls = _STALLS.setdefault(target_key, {})
    last = ""
    terminal_outcome = None
    attempts = []
    def record_attempt(workdir, rec, t0, outcome, detail=""):
        attempts.append({
            "kind": outcome,
            "detail": (detail or "")[:240],
            "harness": rec.get("harness"),
            "model": rec.get("model"),
            "stage": rec.get("stage"),
        })
        return _record(workdir, rec, t0, outcome, detail)
    # Every stage here is a one-shot transformation, so no producer that
    # emitted anything is ever replayed. The old rule keyed on stage names
    # beginning "short" or "full", which left Corpus and review stages able
    # to retry after partial timed-out output.
    one_shot = True
    # A summary stage never blindly replays a producer that answered: one
    # attempt per producer. A transport failure with no output is not an
    # answer, so it gets one more try before the chain moves on. One retry is
    # allowed only for a transport failure that produced nothing.
    attempt_limit = 2

    def remaining_seconds():
        raw = os.environ.get("SUMM_TARGET_DEADLINE_UNIX", "").strip()
        if not raw:
            return None
        try:
            return float(raw) - time.time()
        except ValueError:
            raise Abort(f"{stage}: invalid SUMM_TARGET_DEADLINE_UNIX")

    for mi, (harness, model) in enumerate(resolved):
        remaining = remaining_seconds()
        if remaining is not None and remaining <= 0:
            last = "target deadline exceeded before the next route"
            break
        if harness in dead:
            continue
        if (harness not in GATEWAY_HARNESSES
                and harness not in NON_ARGV_TRANSPORT
                and len(prompt.encode("utf-8")) > ARGV_SAFE_BYTES):
            print(f"    [{stage}] skipping {harness}:{model}: this prompt "
                  "exceeds the safe command-line size and the harness has no "
                  "prompt-file or stdin transport", flush=True)
            continue
        if (harness, model) in dead_routes:
            print(f"    [{stage}] skipping {harness}:{model}: retired earlier "
                  "in this target (quota or repeated stall)", flush=True)
            continue
        try:
            declared_capability = route_caps.capability_for(
                f"{harness}:{model}", harness)
        except ValueError as exc:
            raise Abort(f"{stage}: invalid route capability: {exc}") from exc
        if declared_capability is not None and not route_caps.route_fits(
                len(prompt.split()), declared_capability):
            context = declared_capability.get("context_tokens", "?")
            output = declared_capability.get("output_tokens", 0)
            last = (f"{harness}:{model} is not eligible for this {stage} "
                    f"prompt ({len(prompt.split())} words plus {output} "
                    f"reserved output; qualified context {context} tokens)")
            terminal_outcome = "capacity"
            print(f"    [{stage}] skipping {harness}:{model}: {last}",
                  flush=True)
            # This is a pre-call capacity decision, not a semantic response;
            # the next configured route gets the same stage opportunity.
            continue
        mechanical = False
        advance = False
        for attempt in range(1, attempt_limit + 1):
            remaining = remaining_seconds()
            if remaining is not None and remaining <= 0:
                last = "target deadline exceeded before model call"
                mechanical = True
                break
            attempt_timeout = CALL_TIMEOUT
            if remaining is not None:
                attempt_timeout = min(attempt_timeout,
                                      max(1, int(remaining)))
            # A wall-clock cap per call. agy self-limits with --print-timeout;
            # opencode does not, and it intermittently hangs for minutes on a
            # prompt it answers in six seconds. One stalled call in a hundred
            # would otherwise stall the whole document indefinitely.
            t0 = time.monotonic()
            # A call in flight is 20-40 seconds of silence. Without this the UI
            # shows the PREVIOUS call's outcome for the whole of it, so a live
            # run and a wedged one look identical. Distinct from the completion
            # event on purpose: `model_call` stays the cost record, emitted from
            # the one site that writes calls.jsonl, and this carries no cost.
            attempt_number = _claim_attempt(workdir, stage)
            # Evidence files are named by the run-wide attempt number, not the
            # per-producer one: two chain producers each on their attempt 1
            # would otherwise overwrite each other's stderr, and the failed
            # producer's evidence is exactly the one worth keeping.
            # Stage IDs may carry a colon (D001:W001); Windows cannot store one in a
            # file name, so evidence files spell it portably. Events keep the ID.
            file_stage = stage.replace(":", "-")
            attempt_prefix = f"{file_stage}.attempt{attempt_number:02d}"
            option = effective_option(harness, model, effective_role)
            _pg().emit("model_call_started", role=effective_role, stage=stage,
                       harness=harness,
                       model=model, attempt=attempt, **option)
            rec = {"stage": stage, "role": effective_role,
                   "harness": harness, "model": model,
                   "attempt": attempt, "attempt_number": attempt_number,
                   "chain_index": mi,
                   "prompt_bytes": len(prompt.encode()),
                   "prompt_words": len(prompt.split()),
                   "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
            rec.update(option)
            if harness in GATEWAY_HARNESSES:
                call_option = dict(option)
                if gateway_options:
                    call_option.update(gateway_options)
                    rec["request_options"] = gateway.request_options(
                        harness, model, call_option, role=effective_role)
            else:
                call_option = option
            gateway_kind = None
            gateway_result = None
            gateway_error_evidence = None
            try:
                on_stdin = harness in STDIN_HARNESSES
                with runtime.resource_lease(harness) as resource_wait_s:
                    rec["resource_wait_s"] = round(resource_wait_s, 3)
                    if harness in GATEWAY_HARNESSES:
                        # Warm-on-demand: a 503 "warming" means the model is
                        # loading. Honor
                        # Retry-After and re-poll the SAME request until it is
                        # ready or the warm budget elapses, rather than failing
                        # the run. This is the standard 503/Retry-After contract.
                        warm_seconds = WARM_BUDGET_S
                        if remaining is not None:
                            warm_seconds = min(warm_seconds, max(0, remaining))
                        warm_deadline = time.monotonic() + warm_seconds
                        while True:
                            try:
                                gateway_result = gateway.chat(
                                    harness, model, prompt,
                                    request_options=call_option,
                                    role=effective_role,
                                    planned_output_words=planned_output_words,
                                    output_overhead_tokens=output_overhead_tokens,
                                    semantic_retry=semantic_retry)
                                if _json_stage(call_option):
                                    gateway_result = _finish_json(
                                        gateway_result, harness, model, prompt,
                                        call_option, effective_role,
                                        planned_output_words,
                                        output_overhead_tokens,
                                        warm_deadline, remaining_seconds,
                                        stage)
                                r = subprocess.CompletedProcess(
                                    [harness, model], 0, gateway_result.content, "")
                                break
                            except gateway.GatewayError as ge:
                                if ge.kind == "warming" and time.monotonic() < warm_deadline:
                                    wait = min(ge.retry_after or 30.0, WARM_POLL_MAX_S)
                                    print(f"    [{stage}] {model} warming — re-poll in "
                                          f"{int(wait)}s", flush=True)
                                    wait_for = (wait if remaining is None else
                                                min(wait, max(0, remaining_seconds())))
                                    if wait_for <= 0:
                                        gateway_kind = "unavailable"
                                        gateway_error_evidence = {
                                            "reason": "target deadline exceeded while warming"}
                                        r = subprocess.CompletedProcess(
                                            [harness, model], 1, "", gateway_kind)
                                        break
                                    time.sleep(wait_for)
                                    continue
                                gateway_kind = ge.kind
                                gateway_error_evidence = ge.evidence
                                r = subprocess.CompletedProcess(
                                    [harness, model], 1, "", str(ge))
                                break
                    else:
                        r = subprocess.run(
                            _cmd(harness, model, workdir, prompt, option,
                                 timeout_seconds=attempt_timeout),
                            cwd=cli_cwd(harness, workdir),
                            capture_output=True, text=True, timeout=attempt_timeout,
                            **({"input": prompt} if on_stdin
                               else {"stdin": subprocess.DEVNULL}))
            except subprocess.TimeoutExpired as exc:
                # A timed-out process may already have written part of an
                # answer. Keep it as evidence, and never replay a summary
                # producer that emitted anything: completion is unknown.
                partial_out = exc.stdout if isinstance(exc.stdout, str) else (
                    exc.stdout.decode("utf-8", "replace") if exc.stdout else "")
                partial_err = exc.stderr if isinstance(exc.stderr, str) else (
                    exc.stderr.decode("utf-8", "replace") if exc.stderr else "")
                (workdir / f"{attempt_prefix}.stdout.txt").write_text(partial_out)
                (workdir / f"{attempt_prefix}.stderr.txt").write_text(partial_err)
                rec.update(stdout_bytes=len(partial_out.encode()),
                           stdout_words=len(partial_out.split()))
                last = f"call exceeded {attempt_timeout}s"
                if one_shot and partial_out.strip():
                    record_attempt(workdir, rec, t0, outcome="completion_unknown",
                            detail=last)
                    print(f"    [{stage}] {last} with partial output — "
                          "not replayed", flush=True)
                    mechanical = True
                    advance = True
                    break
                record_attempt(workdir, rec, t0, outcome="timeout")
                print(f"    [{stage}] {last} — retrying", flush=True)
                if attempt < attempt_limit:
                    time.sleep(5 * attempt)
                    continue
                mechanical = True
                advance = True
                break
            except runtime.ConfigError as e:
                raise Abort(str(e)) from e
            if gateway_kind == "config":
                raise Abort(f"{stage}: {r.stderr}")
            if gateway_result is not None:
                (workdir / f"{attempt_prefix}.response.json").write_text(
                    json.dumps(gateway_result.evidence(), indent=2,
                               ensure_ascii=False))
                rec.update(finish_reason=gateway_result.finish_reason,
                           usage=gateway_result.usage,
                           response_id=gateway_result.response_id,
                           dispatch_budget=gateway_result.dispatch_budget,
                           gateway_elapsed_s=round(
                               gateway_result.elapsed_seconds, 3))
                if gateway_result.finish_reason != "stop":
                    gateway_kind = ("output_limit" if gateway_result.finish_reason
                                    in {"length", "max_tokens"} else "unusable")
                    r = subprocess.CompletedProcess(
                        [harness, model], 1, "",
                        f"gateway finish_reason={gateway_result.finish_reason}")
            elif gateway_error_evidence is not None:
                (workdir / f"{attempt_prefix}.response.json").write_text(
                    json.dumps(gateway_error_evidence, indent=2,
                               ensure_ascii=False))
            # Keep every attempt's normalized response. The stage-level files
            # remain the latest-attempt compatibility view, while the attempt
            # files make a repeated malformed response diagnosable instead of
            # being overwritten by the final gateway error.
            stderr = r.stderr if isinstance(r.stderr, str) else str(r.stderr or "")
            stdout = r.stdout if isinstance(r.stdout, str) else str(r.stdout or "")
            (workdir / f"{attempt_prefix}.stderr.txt").write_text(stderr)
            (workdir / f"{attempt_prefix}.stdout.txt").write_text(stdout)
            (workdir / f"{file_stage}.stderr.txt").write_text(stderr)
            (workdir / f"{file_stage}.stdout.txt").write_text(stdout)
            rec.update(returncode=r.returncode,
                       stdout_bytes=len(stdout.encode()),
                       stdout_words=len(stdout.split()))
            # A quota diagnostic is a transport condition, not an answer, and
            # it is classified before any schema check regardless of exit
            # code or which stream carried it.
            quota = _capacity_signal(stdout, stderr)
            if quota:
                last = quota
                record_attempt(workdir, rec, t0, outcome="capacity", detail=last[:120])
                dead_routes.add((harness, model))
                advance = True
                break
            # Only a clean exit produces usable output. A crashed CLI can emit
            # plausible partial Markdown or JSON on stdout; accepting it turns a
            # failed process into an approved artifact.
            if r.returncode == 0 and stdout.strip():
                # grok answers in a transport envelope; every other harness puts
                # the stage's own bytes on stdout. Normalise before validating,
                # and keep the raw envelope in stage.stdout.txt above as
                # evidence -- it carries the session id, stop reason and the
                # model's narration, which is what explains a failed run later.
                try:
                    if harness == "grok":
                        usable = grok_payload(stdout)
                    elif harness == "cursor":
                        usable = cursor_payload(stdout)
                    else:
                        usable = stdout
                except Exception as ge:
                    last = f"unusable response: {str(ge)[:80]}"
                    record_attempt(workdir, rec, t0, outcome="unusable",
                            detail=str(ge)[:120])
                    mechanical = True
                    advance = True
                    break
                if validate is not None:
                    try:
                        if harness in GATEWAY_HARNESSES:
                            fmt = call_option.get("response_format")
                            if isinstance(fmt, dict) and fmt.get("type") in {
                                    "json_object", "json_schema"}:
                                schema = None
                                if fmt.get("type") == "json_schema":
                                    spec = fmt.get("json_schema")
                                    if isinstance(spec, dict):
                                        schema = spec.get("schema")
                                json_contract.parse_exact(
                                    usable, schema, stage)
                        validate(usable)
                    except Exception as ve:
                        last = f"unusable response: {str(ve)[:80]}"
                        contract = bool(getattr(ve, "code", None))
                        terminal_outcome = ("response_contract" if contract
                                            else "unusable")
                        record_attempt(workdir, rec, t0,
                                       outcome=terminal_outcome,
                                       detail=str(ve)[:120])
                        mechanical = True
                        advance = True
                        break
                record_attempt(workdir, rec, t0, outcome="ok")
                if mi:
                    print(f"    [{stage}] fell back to {harness}:{model}", flush=True)
                return usable
            last = stderr.strip()
            if r.returncode == 0 and not stdout.strip():
                last = last or "exited 0 with empty stdout"
            if gateway_kind == "request":
                terminal_outcome = "request"
                record_attempt(workdir, rec, t0, outcome="request",
                        detail=last[:120])
                # A route-level request/contract failure cannot be repaired by
                # replaying the same route. Let the frozen next route try the
                # same stage instead.
                mechanical = True
                advance = True
                break
            if gateway_kind == "completion_unknown":
                terminal_outcome = "completion_unknown"
                record_attempt(workdir, rec, t0, outcome="completion_unknown",
                        detail=last[:120])
                mechanical = True
                break
            if gateway_kind == "unavailable":
                terminal_outcome = "unavailable"
                record_attempt(workdir, rec, t0, outcome="unavailable",
                        detail=last[:120])
                advance = True
                break
            if gateway_kind == "unusable":
                terminal_outcome = "unusable"
                record_attempt(workdir, rec, t0, outcome="unusable", detail=last[:120])
                mechanical = True
                advance = True
                break
            if gateway_kind == "output_incomplete" or gateway_kind == "output_limit":
                terminal_outcome = gateway_kind
                record_attempt(workdir, rec, t0, outcome=gateway_kind, detail=last[:120])
                mechanical = True
                advance = True
                break
            if gateway_kind == "capacity":
                terminal_outcome = "capacity"
                record_attempt(workdir, rec, t0, outcome="capacity", detail=last[:120])
                advance = True
                break
            if gateway_kind == "auth" or AUTH.search(last):
                terminal_outcome = "auth"
                # Kill this harness for the rest of the chain and look for one
                # behind a different credential. Only when none remains is the
                # run genuinely unable to proceed.
                record_attempt(workdir, rec, t0, outcome="auth", detail=last[:120])
                dead.add(harness)
                if all(h in dead for h, _ in resolved[mi + 1:]):
                    raise Abort(f"{stage}: {harness}:{model} needs authentication "
                                f"and no later chain entry uses a different "
                                f"harness, so falling through cannot help")
                print(f"    [{stage}] {harness} needs authentication — "
                      f"skipping its remaining models", flush=True)
                advance = True
                break
            if CAPACITY.search(last):
                terminal_outcome = "capacity"
                record_attempt(workdir, rec, t0, outcome="capacity", detail=last[:120])
                dead_routes.add((harness, model))
                advance = True
                break                      # next model, not another attempt
            if TRANSIENT.search(last):
                terminal_outcome = "transient"
                record_attempt(workdir, rec, t0, outcome="transient", detail=last[:120])
                if not stdout.strip():
                    # A silent stall. One retry; a second one in this target
                    # retires the route rather than paying for it again.
                    key = (harness, model)
                    stalls[key] = stalls.get(key, 0) + 1
                    if stalls[key] >= STALL_LIMIT:
                        dead_routes.add(key)
                        print(f"    [{stage}] {harness}:{model} stalled "
                              f"{stalls[key]} times — retiring it for this "
                              "target", file=sys.stderr, flush=True)
                        mechanical = True
                        advance = True
                        break
                if attempt < attempt_limit and (not one_shot
                                                or not stdout.strip()):
                    delay = 5 * attempt
                    if remaining is not None:
                        delay = min(delay, max(0, remaining_seconds()))
                    if delay <= 0:
                        last = "target deadline exceeded during retry wait"
                        mechanical = True
                        advance = True
                        break
                    time.sleep(delay)
                    continue
                mechanical = True
                advance = True
                break
            record_attempt(workdir, rec, t0, outcome="failed", detail=last[:120])
            terminal_outcome = "failed"
            mechanical = True
            advance = True
            break
        if not advance:
            break                          # completion-unknown is never blindly replayed
    err = last.splitlines()
    recovery_kinds = {
        "response_contract", "unusable", "output_limit", "output_incomplete",
    }
    recovery_kind = next(
        (item["kind"] for item in attempts if item["kind"] in recovery_kinds),
        terminal_outcome)
    raise NoCandidate(
        f"{stage}: no output from chain {chain}: {err[0] if err else 'no stderr'}",
        kind=terminal_outcome, detail=last[:240], attempts=attempts,
        recovery_kind=recovery_kind)


def _balanced_objects(text):
    """Yield balanced top-level JSON objects, ignoring braces in strings."""
    depth = start = 0
    in_str = esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0:
                yield text[start:i + 1]


def parse_audit(raw: str):
    """Extract the last valid JSON object after any echoed request text."""
    for match in re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.S):
        try:
            return json.loads(match)
        except Exception:
            pass
    best = None
    for blob in _balanced_objects(raw):
        try:
            obj = json.loads(blob)
        except Exception:
            continue
        if isinstance(obj, dict) and obj:
            best = obj
    if best is None:
        raise Abort("audit: no JSON object in response")
    return best
