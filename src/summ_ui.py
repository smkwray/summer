#!/usr/bin/env python3
"""Desktop job console for summ'er.

A client of `summ_cli.py`, never a second implementation of it. Everything this
window can do, the CLI can do; the UI adds a native file picker and visibility
into a run that takes five to fifteen minutes and otherwise says nothing until
it finishes. Both installed triggers keep working untouched.

    python3 src/summ_ui.py

It reads `progress.jsonl` (see progress.py) rather than parsing the CLI's human
stdout, which has no schema. Stdout is still captured verbatim into console.log
and shown in the Details pane, for reading and not for control.

The subprocess runs on a worker thread; every widget update happens on the Tk
event loop, because long work inside a Tk handler blocks event processing and
the window stops repainting -- which is the exact failure this is built to fix.
"""
from __future__ import annotations
import datetime, json, os, pathlib, queue, shutil, subprocess, sys, threading, time, uuid
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

HERE = pathlib.Path(__file__).resolve().parent
CLI = HERE / "summ_cli.py"
ASSETS = HERE / "assets"
sys.path.insert(0, str(HERE))
# Imported for the shared selection/manifest layer, not to duplicate the engine.
# The UI resolves and reviews the same records the CLI consumes.
import summ_cli
import selection
import runtime
import model_config
import mode_config
import profiles
import custom_instructions
import last_output
import last_affix

MOD = "Command" if sys.platform == "darwin" else "Control"
MOD_LABEL = "⌘" if sys.platform == "darwin" else "Ctrl+"
WIN = sys.platform.startswith("win")

# `--depth` is not a mode: the installed triggers still pass it, but composition
# renders both readings from one sealed ledger. The entries below are distinct
# product routes whose output and model-call contracts really differ.
MODES = tuple((mode.label, list(mode.flags)) for mode in mode_config.MODES)

# A run's evidence lives outside the synced project tree: it is device-local,
# it is large, and the sync mesh has no reason to carry it.
def job_root() -> pathlib.Path:
    if WIN:
        base = pathlib.Path(os.environ.get("LOCALAPPDATA",
                                           pathlib.Path.home() / "AppData/Local"))
    else:
        base = pathlib.Path.home() / "Library/Application Support"
    return base / "summer" / "runs"


ROLES = mode_config.ROLES

# Applied when the window opens, if it exists. Named here rather than duplicated
# into the profile store so there is one place that says which is default.
DEFAULT_PROFILE = profiles.DEFAULT_PROFILE

# WHICH ROLES A MODE ACTUALLY CALLS, keyed by the mode's CLI flags. Verified in
# the engine, not assumed: the ledger path calls plan/write/audit/repair
# (ledger.py); Quick and Clean text never plan; Read-aloud uses the same Clean
# roles for arbitrary source text and skips model work for known Summer
# artifacts. None of those routes needs Plan.
#
# Showing four live pickers for a mode that calls one of them invites a choice
# the run will ignore, which reads as the setting having no effect.
MODE_ROLES = {mode.flags: mode.roles for mode in mode_config.MODES}


def roles_for(mode_label: str) -> tuple:
    mode = mode_config.BY_LABEL.get(mode_label)
    return mode.roles if mode else ROLES


def roster() -> dict:
    """Every harness and its per-role chain from committed and device data.

    The UI names no model of its own. It offers what that file already lists,
    which keeps models.json the single place a model identifier appears -- the
    rule the whole project is built on -- and means a model added there shows up
    here without touching this code.
    """
    try:
        cfg = model_config.roster()
    except Exception:
        return {}
    return {h: {r: list(v.get(r) or []) for r in ROLES}
            for h, v in cfg.items()
            if isinstance(v, dict) and any(v.get(role) for role in ROLES)}


def model_defaults() -> dict:
    return profiles.model_defaults()


def model_choices(harness: str, role: str) -> list[str]:
    """Return the logical models offered for one role."""
    return list(roster().get(harness, {}).get(role) or [])


def harness_choices(role: str) -> list[str]:
    """Harnesses that can actually perform ``role``."""
    return sorted(name for name, spec in roster().items()
                  if spec.get(role))


def backup_harness_choices(roles: tuple[str, ...]) -> list[str]:
    """Harnesses with at least one model for every active role."""
    return sorted(name for name, spec in roster().items()
                  if all(spec.get(role) for role in roles))


def _pick(value):
    return profiles._pick(value)


def _option(harness: str, model: str, value: str = "", role: str = "",
            roster_value: dict | None = None) -> dict:
    return profiles._option(harness, model, value, role, roster_value)


def role_env(harness: str, picks: dict, fallback=None, local_only=False,
             roster_value: dict | None = None) -> dict:
    """Per-role choices as the environment the CLI already understands.

    `picks` maps a role to (harness, model, effort); the third item is optional
    for compatibility with older device profiles. A role whose harness differs from
    the run's is written as the QUALIFIED entry `harness:model`, which mapsum
    already resolves -- that is what makes "plan on agy, audit on opencode"
    expressible at all. The run harness only decides which `<X>_CHAIN_<ROLE>`
    variable is read; each entry chooses its own CLI.

    No new selection mechanism: this is the same override an experiment would
    use from a shell, and models.json remains the only place a model is named.
    """
    return profiles.role_env(harness, picks, fallback, local_only, roster_value)


def build_cmd(job: pathlib.Path, mode_label: str, source, out_dir=None,
              text_file=None, instructions_file=None, selection_manifest=None,
              scope="batch", affix_kind="prefix", affix_text="", affix_secondary=""):
    """The exact argv the UI runs. Extracted so it can be asserted on.

    No shell, and one path as one argv element -- a command string assembled by
    hand is where quoting bugs live, and a document path is untrusted input.

    No --depth. The CLI accepts it for the installed triggers, but composition
    renders both readings from the one sealed ledger and never branches on it,
    so passing it from a UI would imply a choice the engine does not offer.

    Path-backed UI selections use one versioned manifest path. This keeps the
    command short on Windows and gives the CLI the same frozen selection the UI
    reviewed. The legacy ``source`` path form remains for small callers/tests;
    the App always supplies ``selection_manifest`` for path-backed jobs.

    ``source`` is retained only for small legacy callers; path-backed App jobs
    pass ``selection_manifest`` and therefore contain no discovered paths on
    argv.
    """
    cmd = [sys.executable, "-u", str(CLI),
           "--work-dir", str(job / "work"),
           "--progress-jsonl", str(job / "progress.jsonl"),
           "--cancel-file", str(job / "cancel.request")]
    if out_dir:
        cmd += ["--out", str(out_dir)]
    # Captured pasted text, so a queued job does not depend on the clipboard
    # still holding what the user pasted when they pressed Start.
    if text_file:
        cmd += ["--text-file", str(text_file)]
    if selection_manifest:
        cmd += ["--selection-manifest", str(selection_manifest)]
        if scope != "batch":
            cmd += ["--scope", scope]
    # Per-run instructions are transport, not a profile or device setting.
    # Read-aloud has no summary to customize, so never pass them to that mode.
    mode = mode_config.BY_LABEL[mode_label]
    if instructions_file and mode.instructions:
        cmd += ["--instructions-file", str(instructions_file)]
    cmd += list(mode.flags)
    if affix_kind and affix_kind != "prefix":
        cmd += ["--affix-kind", str(affix_kind)]
    if affix_text:
        cmd += ["--affix", str(affix_text)]
    if affix_secondary:
        cmd += ["--affix-secondary", str(affix_secondary)]
    # Tolerate a bare Path as well as a list. Iterating a Path raises TypeError
    # at the point of launch, which in a GUI surfaces as a job that silently
    # never starts.
    if isinstance(source, (str, pathlib.Path)):
        source = [source]
    if not selection_manifest:
        cmd += [str(p) for p in (source or [])]
    return cmd


profiles_path = profiles.profiles_path
last_profile_path = profiles.last_profile_path
load_last_profile = profiles.load_last_profile
save_last_profile = profiles.save_last_profile
load_profiles = profiles.load_profiles
save_profiles = profiles.save_profiles
launch_profile = profiles.launch_profile


def tip(widget, text_fn):
    """A hover tooltip. Exists so the summary label can be truncated: the window
    used to grow and shrink every time a harness changed, because the label
    resized with its text."""
    box = {"w": None}

    def show(_e=None):
        t = text_fn()
        if not t or box["w"]:
            return
        w = tk.Toplevel(widget)
        w.wm_overrideredirect(True)
        w.wm_geometry(f"+{widget.winfo_rootx()}+{widget.winfo_rooty() + 22}")
        tk.Label(w, text=t, justify="left", relief="solid", borderwidth=1,
                 background="#111", foreground="#eee", padx=6, pady=3).pack()
        box["w"] = w

    def hide(_e=None):
        if box["w"]:
            box["w"].destroy(); box["w"] = None

    widget.bind("<Enter>", show)
    widget.bind("<Leave>", hide)


def reveal(p: pathlib.Path):
    try:
        if WIN:
            os.startfile(p)                                  # noqa: S606
        elif sys.platform == "darwin":
            subprocess.run(["open", str(p)], check=False)
        else:
            subprocess.run(["xdg-open", str(p)], check=False)
    except Exception:
        pass


class JobState:
    """All mutable state for one child process; never shared with another."""
    def __init__(self, ident: str, root: pathlib.Path, spec: dict):
        self.id, self.root, self.spec = ident, root, spec
        self.proc = None
        self.events = queue.Queue()
        self.offset = 0
        self.parts = (0, 0)
        self.part_failures = []
        self.target_titles = {}
        self.call = None
        self.started = time.monotonic()
        self.out_dir = None
        self.eta = None
        self.cancel_requested = False
        self.cancel_file = root / "cancel.request"


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.proc = None
        # `proc` is assigned on the WORKER thread, so it is not a usable state
        # flag on this one: between start() returning and Popen being reached it
        # is still None, and close() reading it there would destroy the window
        # over a job that is about to launch. `running` is set synchronously.
        self.running = False
        self.job = None
        self.started = None
        self.events = queue.Queue()
        self.selection_events = queue.Queue()
        self.offset = 0
        self.parts = (0, 0)
        # Initialised here as well as per run: _finish() reads it, and a window
        # can reach _finish without ever having started a run.
        self.part_failures = []
        self.target_titles = {}
        self.pending = []
        self.active = {}
        # Adding an accepted input creates a waiting job immediately.  Start
        # arms the queue; while it is armed, later additions join the same
        # drain and are not stranded behind the documents already present.
        self.queue_started = False
        self.cancelling = False
        self.polling = False
        self.call = None
        self.source = None                # compatibility view of selected paths
        self.pasted = None                # raw text as it was when pasted
        self.selection = None             # immutable Selection snapshot
        self.selection_inputs = []        # ordered roots awaiting resolution
        self._resolution_values = []      # latest complete path set in flight
        self.resolving = False
        self.selection_generation = 0
        root.title("summ'er")
        root.minsize(560, 300)
        screen_width = max(560, root.winfo_screenwidth() - 80)
        screen_height = max(300, root.winfo_screenheight() - 120)
        root.geometry(f"{min(1100, screen_width)}x{min(880, screen_height)}")
        # Tk 8.6+ reads PNG directly, so no image library is pulled in for this.
        # The reference is held on the instance because Tk does not keep one and
        # a garbage-collected PhotoImage silently leaves the window iconless.
        try:
            self.icon = tk.PhotoImage(file=str(ASSETS / "icon-128.png"))
            root.iconphoto(True, self.icon)
        except Exception:
            pass                       # an icon is never worth failing to start

        # The controls and retained result cards can be taller than a laptop
        # screen. Put the complete application surface in one scroll viewport;
        # scrolling only the Details widget leaves buttons below it unreachable.
        shell = ttk.Frame(root)
        shell.pack(fill="both", expand=True)
        shell.rowconfigure(0, weight=1)
        shell.columnconfigure(0, weight=1)
        background = ttk.Style().lookup("TFrame", "background") or root.cget(
            "background")
        self.scroll_canvas = tk.Canvas(
            shell, highlightthickness=0, borderwidth=0, background=background)
        self.scroll_canvas.grid(row=0, column=0, sticky="nsew")
        self.window_scrollbar = ttk.Scrollbar(
            shell, orient="vertical", command=self.scroll_canvas.yview)
        self.window_scrollbar.grid(row=0, column=1, sticky="ns")
        self.scroll_canvas.configure(yscrollcommand=self.window_scrollbar.set)

        f = ttk.Frame(self.scroll_canvas, padding=12)
        self.content_frame = f
        self.content_window = self.scroll_canvas.create_window(
            (0, 0), window=f, anchor="nw")
        f.bind("<Configure>", self._refresh_scroll_region)
        self.scroll_canvas.bind("<Configure>", self._resize_scroll_content)
        root.bind("<MouseWheel>", self._scroll_window, add="+")
        root.bind("<Button-4>", self._scroll_window, add="+")
        root.bind("<Button-5>", self._scroll_window, add="+")
        f.columnconfigure(1, weight=1)

        ttk.Label(f, text="Input").grid(row=0, column=0, sticky="w")
        self.src_lbl = ttk.Label(f, text="Clipboard", foreground="#555")
        self.src_lbl.grid(row=0, column=1, sticky="w", padx=(8, 0))
        b = ttk.Frame(f); b.grid(row=0, column=2, sticky="e")
        self.pick_btn = ttk.Button(b, text="Add Files…", command=self.pick)
        self.pick_btn.pack(side="left")
        self.folder_btn = ttk.Button(b, text="Add Folder…", command=self.pick_folder)
        self.folder_btn.pack(side="left", padx=(6, 0))
        self.clip_btn = ttk.Button(b, text="Use Clipboard", command=self.use_clip)
        self.clip_btn.pack(side="left", padx=(6, 0))

        ttk.Label(f, text="Path").grid(row=1, column=0, sticky="w", pady=(6, 0))
        path_controls = ttk.Frame(f)
        path_controls.grid(row=1, column=1, columnspan=2, sticky="ew",
                           padx=(8, 0), pady=(6, 0))
        path_controls.columnconfigure(0, weight=1)
        self.path_entry = ttk.Entry(path_controls)
        self.path_entry.grid(row=0, column=0, sticky="ew")
        self.path_entry.bind("<Return>", lambda _e: self.add_path())
        self.path_add_btn = ttk.Button(path_controls, text="Add", width=7,
                                       command=self.add_path)
        self.path_add_btn.grid(row=0, column=1, padx=(4, 0))
        self.path_clear_btn = ttk.Button(path_controls, text="Clear path", width=9,
                                         command=self.clear_selection)
        self.path_clear_btn.grid(row=0, column=2, padx=(4, 0))
        self.selection_summary = ttk.Label(f, text="", foreground="#777")
        self.selection_summary.grid(row=2, column=1, columnspan=2, sticky="w",
                                    padx=(8, 0), pady=(2, 0))
        self.review_btn = ttk.Button(f, text="Review…", width=10,
                                     command=self.review_selection)
        self.review_btn.grid(row=2, column=2, sticky="e", pady=(2, 0))

        ttk.Label(f, text="Mode").grid(row=3, column=0, sticky="w", pady=(10, 0))
        self.mode = tk.StringVar(value=MODES[0][0])
        self.mode_box = ttk.Combobox(f, textvariable=self.mode, state="readonly",
                                     values=[m[0] for m in MODES])
        self.mode_box.grid(row=3, column=1, columnspan=2, sticky="ew",
                           padx=(8, 0), pady=(10, 0))
        # A mode change alters which roles the run will call, so the pickers
        # have to follow it or they advertise choices the run ignores.
        self.mode_box.bind("<<ComboboxSelected>>", self.mode_changed)
        self.mode.trace_add("write", self.mode_changed)

        ttk.Label(f, text="Scope").grid(row=4, column=0, sticky="w", pady=(10, 0))
        self.scope = tk.StringVar(value="batch")
        scope_frame = ttk.Frame(f)
        scope_frame.grid(row=4, column=1, columnspan=2, sticky="w",
                         padx=(8, 0), pady=(10, 0))
        self.batch_scope = ttk.Radiobutton(
            scope_frame, text="Batch — each document separately",
            variable=self.scope, value="batch", command=self.scope_changed)
        self.batch_scope.pack(side="left")
        self.corpus_scope = ttk.Radiobutton(
            scope_frame, text="Corpus — one combined result",
            variable=self.scope, value="corpus", command=self.scope_changed)
        self.corpus_scope.pack(side="left", padx=(12, 0))
        self.corpus_name = tk.StringVar()
        self.corpus_name_row = 5
        self.corpus_name_label = ttk.Label(f, text="Corpus name")
        self.corpus_name_entry = ttk.Entry(f, textvariable=self.corpus_name)
        self.corpus_name_entry.bind("<FocusOut>", lambda _e: self.replan_selection())

        # Optional per-run instructions remain separate from model profiles.
        # One small device-local value is restored per stable mode key; Start
        # still copies the visible, enabled value into the immutable queued job.
        self.show_instructions = tk.BooleanVar(value=False)
        self.instruction_drafts = {}
        self._instruction_mode_key = mode_config.BY_LABEL[self.mode.get()].key
        self.instructions_toggle = ttk.Checkbutton(
            f, text="Custom instructions (optional)",
            variable=self.show_instructions, command=self.toggle_instructions)
        self.instructions_toggle.grid(row=6, column=0, columnspan=3, sticky="w",
                                      pady=(10, 0))
        self.instructions_frame = ttk.Frame(f)
        self.instructions_frame.columnconfigure(0, weight=1)
        preset_row = ttk.Frame(self.instructions_frame)
        preset_row.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        ttk.Label(preset_row, text="Preset").pack(side="left")
        self.instruction_preset = tk.StringVar()
        self.instruction_preset_box = ttk.Combobox(
            preset_row, textvariable=self.instruction_preset, state="readonly",
            width=30, exportselection=False)
        self.instruction_preset_box.pack(side="left", padx=(6, 0))
        self.instruction_preset_box.bind(
            "<<ComboboxSelected>>", self.apply_instruction_preset)
        self.instructions_box = tk.Text(self.instructions_frame, height=3,
                                        wrap="word", undo=True)
        self.instructions_box.grid(row=1, column=0, sticky="ew")
        ttk.Label(
            self.instructions_frame,
            text=("Saved on this device for this mode · not part of model "
                  "profiles · 2,000-byte max."),
            foreground="#777").grid(row=2, column=0, sticky="w", pady=(2, 0))
        self.instructions_row = 7
        restored = custom_instructions.load_last(self._instruction_mode_key)
        self.instruction_drafts[self._instruction_mode_key] = restored
        if restored:
            self.instructions_box.insert("1.0", restored)
            self.show_instructions.set(True)
        self.refresh_instruction_presets()

        ttk.Label(f, text="Output").grid(row=8, column=0, sticky="w", pady=(10, 0))
        # NOT `out_dir`: that name already means "where the finished
        # artifacts landed", for Reveal. This is the requested destination.
        self.out_choice = tk.StringVar()
        ob = ttk.Frame(f)
        ob.grid(row=8, column=1, columnspan=2, sticky="ew", padx=(8, 0), pady=(10, 0))
        ob.columnconfigure(0, weight=1)
        self.out_entry = ttk.Entry(ob, textvariable=self.out_choice)
        self.out_entry.grid(row=0, column=0, sticky="ew")
        self.out_entry.bind("<FocusIn>", self._engage_output_field)
        self.choose_out_btn = ttk.Button(ob, text="Choose", width=8,
                                         command=self.choose_out)
        self.choose_out_btn.grid(row=0, column=1, padx=(4, 0))
        # Beside-source is the DEFAULT and a real setting, not a missing one, so
        # it is a checkbox that shows its state, not a button that looks like an
        # action still to be taken.
        #
        # A PLAIN ttk.Checkbutton, deliberately. Styling it as a Toolbutton to
        # get a tinted pill drew a blank pill under macOS aqua: the control kept
        # its 142x32 box but showed no label, so it read as having vanished. The
        # forced style mapped foreground and background to EMPTY STRINGS for the
        # unselected state, which leaves the label no colour to draw with.
        # Rather than hunt a colour past the theme, use the affordance the theme
        # already draws -- a native checkbox, which macOS fills blue when
        # checked, exactly the on/off switch wanted.
        self.beside = tk.BooleanVar(value=True)
        self.beside_btn = ttk.Checkbutton(
            ob, text="Beside source",
            variable=self.beside, command=self.toggle_beside)
        self.beside_btn.grid(row=0, column=2, padx=(6, 0))
        self._last_out_tip = ""
        self.last_out_btn = ttk.Button(ob, command=self.use_last_out)
        tip(self.last_out_btn, lambda: self._last_out_tip)
        self.out_choice.trace_add("write", self._out_choice_written)

        ttk.Label(f, text="Naming").grid(row=9, column=0, sticky="w", pady=(10, 0))
        affix_frame = ttk.Frame(f)
        affix_frame.grid(row=9, column=1, columnspan=2, sticky="ew", padx=(8, 0), pady=(10, 0))
        affix_frame.columnconfigure(5, weight=1)
        affix_frame.columnconfigure(2, weight=1)

        mode_obj = mode_config.BY_LABEL.get(self.mode.get())
        mode_key = mode_obj.key if mode_obj else "summarize"
        saved_affix = last_affix.load(mode_key)
        self.affix_kind = tk.StringVar(value=saved_affix["kind"])
        self.affix_text = tk.StringVar(value=saved_affix["primary"])
        self.affix_secondary = tk.StringVar(value=saved_affix["secondary"])
        self._updating_affix = False

        kind_switch_frame = ttk.Frame(affix_frame)
        kind_switch_frame.grid(row=0, column=0, sticky="w")
        self.prefix_radio = ttk.Radiobutton(
            kind_switch_frame, text="Prefix", variable=self.affix_kind, value="prefix")
        self.prefix_radio.pack(side="left")
        self.suffix_radio = ttk.Radiobutton(
            kind_switch_frame, text="Suffix", variable=self.affix_kind, value="suffix")
        self.suffix_radio.pack(side="left", padx=(8, 0))

        self.affix_label1 = ttk.Label(affix_frame, text="Detailed:")
        self.affix_entry = ttk.Entry(affix_frame, textvariable=self.affix_text, width=10)
        self.affix_label2 = ttk.Label(affix_frame, text="Brief:")
        self.affix_entry2 = ttk.Entry(affix_frame, textvariable=self.affix_secondary, width=10)

        self.affix_preview = ttk.Label(affix_frame, text="", foreground="#777")

        self.affix_kind.trace_add("write", self.affix_changed)
        self.affix_text.trace_add("write", self.affix_changed)
        self.affix_secondary.trace_add("write", self.affix_changed)
        self._sync_affix_ui(mode_obj, mode_key)

        # Expanded by default: the model selection is the setting most
        # worth seeing before pressing Start, and a collapsed pane hid
        # which models a run was about to spend.
        self.show_models = tk.BooleanVar(value=True)
        ttk.Checkbutton(f, text="Models", variable=self.show_models,
                        command=self.toggle_models).grid(row=10, column=0,
                                                         sticky="w", pady=(10, 0))
        self.models_summary = ttk.Label(f, text="", foreground="#777", anchor="w",
                                        width=self.SUMMARY_CHARS)
        self._tip_text = ""
        tip(self.models_summary, lambda: self._tip_text)
        self.models_summary.grid(row=10, column=1, columnspan=2, sticky="ew",
                                 padx=(8, 0), pady=(10, 0))
        self.models_frame = ttk.Frame(f)
        self.models_row = 11
        self.build_model_controls()
        self._update_affix_preview()

        run_buttons = ttk.Frame(f)
        run_buttons.grid(row=12, column=0, columnspan=3, sticky="ew", pady=(14, 6))
        run_buttons.columnconfigure(0, weight=1)
        self.start_btn = ttk.Button(run_buttons, text="Start", command=self.start)
        self.start_btn.grid(row=0, column=0, sticky="ew")
        self.cancel_btn = ttk.Button(run_buttons, text="Cancel active",
                                     command=self.cancel_active, state="disabled")
        self.cancel_btn.grid(row=0, column=1, padx=(6, 0))
        self.queue_frame = ttk.Frame(run_buttons)
        self.queue_frame.grid(row=1, column=0, columnspan=2, sticky="ew",
                              pady=(6, 0))
        self.queue_frame.columnconfigure(0, weight=1)
        self.queue_label = ttk.Label(
            self.queue_frame,
            text="Queue: empty",
            foreground="#777", anchor="w")
        self.queue_label.pack(fill="x")
        self.queue_rows_frame = ttk.Frame(self.queue_frame)
        self.queue_rows_frame.pack(fill="x")
        self.queue_rows = {}

        # DETERMINATE, driven by parts completed. The old indeterminate bar swept
        # back and forth continuously, which reads as frantic and says nothing --
        # it moves identically whether a run is healthy or wedged. Parts are a
        # real discrete count, unlike elapsed time, so the bar can be honest.
        self.bar = ttk.Progressbar(f, mode="determinate", maximum=1, value=0)
        self.bar.grid(row=13, column=0, columnspan=3, sticky="ew")
        self.status = ttk.Label(f, text="Idle", anchor="w")
        self.status.grid(row=14, column=0, columnspan=3, sticky="ew", pady=(6, 0))
        self.detail = ttk.Label(f, text="", anchor="w", foreground="#555")
        self.detail.grid(row=15, column=0, columnspan=3, sticky="ew")

        self.out = ttk.Frame(f); self.out.grid(row=16, column=0, columnspan=3,
                                               sticky="ew", pady=(10, 0))
        self.open_out = ttk.Button(self.out, text="Open Output Folder",
                                   command=lambda: reveal(self.out_dir))
        self.open_work = ttk.Button(self.out, text="Open Work Folder",
                                    command=lambda: reveal(self.job))
        self.out_dir = None

        # Expanded by default: this is where a failure says WHY.
        self.show_console = tk.BooleanVar(value=True)
        ttk.Checkbutton(f, text="Details", variable=self.show_console,
                        command=self.toggle).grid(row=17, column=0, sticky="w",
                                                  pady=(10, 0))
        self.console = tk.Text(f, height=10, wrap="word", state="disabled")
        self.console_row = 18
        f.rowconfigure(self.console_row, weight=1)

        # Finished work, newest first. A run of twenty documents produces
        # twenty results and the window has to be able to hand each one back.
        self.cards_frame = ttk.Frame(f)
        self.cards_frame.grid(row=19, column=0, columnspan=3, sticky="nsew",
                              pady=(10, 0))
        self.cards = []

        root.protocol("WM_DELETE_WINDOW", self.close)
        for seq in (f"<{MOD}-v>", f"<{MOD}-V>"):
            root.bind_all(seq, self.paste)
        for seq in (f"<{MOD}-Return>", f"<{MOD}-KP_Enter>"):
            root.bind_all(seq, lambda _e: self.start())
        self.hint = ttk.Label(f, foreground="#777",
                              text=f"{MOD_LABEL}V to paste a path, folder or text"
                                   f"   ·   {MOD_LABEL}\u21a9 to start")
        self.hint.grid(row=20, column=0, columnspan=3, sticky="w", pady=(8, 0))

        # Both panes open on launch. A ticked box that shows nothing is worse
        # than an unticked one, so the frames are gridded here rather than only
        # marked visible -- and this runs last, once every widget both toggles
        # touch actually exists.
        self.toggle_instructions()
        self.toggle_models()
        self.toggle()
        self.refresh_scope()
        root.after_idle(self._refresh_scroll_region)

    def _refresh_scroll_region(self, _event=None):
        self.scroll_canvas.configure(scrollregion=self.scroll_canvas.bbox("all"))

    def _resize_scroll_content(self, event):
        # Fill the viewport horizontally; vertical growth remains scrollable.
        self.scroll_canvas.itemconfigure(self.content_window, width=event.width)
        self._refresh_scroll_region()

    def _scroll_window(self, event):
        # Text widgets own their wheel while the pointer is over them. Everywhere
        # else, including model controls and result cards, scroll the full form.
        if isinstance(getattr(event, "widget", None), tk.Text):
            return None
        number = getattr(event, "num", None)
        if number == 4:
            units = -1
        elif number == 5:
            units = 1
        else:
            delta = getattr(event, "delta", 0)
            if not delta:
                return None
            units = -1 if delta > 0 else 1
            if abs(delta) >= 120:
                units *= max(1, abs(int(delta)) // 120)
        self.scroll_canvas.yview_scroll(units, "units")
        return "break"

    # -- clipboard -----------------------------------------------------------
    def paste(self, _event=None):
        """Read the clipboard through the shared classifier and snapshot it."""
        if self.resolving:
            self._set("Selection is still resolving.",
                      "Wait for the document count to finish before replacing it.")
            return "break"
        raw = summ_cli.clip_read()
        if not raw.strip():
            self._set("Clipboard is empty.", "")
            return "break"
        mode_key = mode_config.BY_LABEL[self.mode.get()].key
        if not selection.clipboard_path_intent(raw):
            # Raw prose has no discovery or hashing work. Accepting it directly
            # keeps paste responsive; path-shaped clipboard input takes the
            # worker below because it must recurse and hash.
            self._accept_selection(selection.text_selection(raw, mode_key))
            return "break"
        # Clipboard classification and hashing happen off the Tk event loop.
        self.selection_generation += 1
        generation = self.selection_generation
        self.resolving = True
        self.start_btn.configure(state="disabled")
        self.selection_summary.config(text="Resolving clipboard…")

        def classify_selection_worker():
            try:
                chosen = selection.classify_clipboard(raw, mode_key)
                self.selection_events.put((generation, chosen))
            except Exception as exc:
                self.selection_events.put((generation, exc))

        threading.Thread(target=classify_selection_worker, daemon=True).start()
        self.root.after(50, self._poll_selection)
        return "break"

    # -- models --------------------------------------------------------------
    def build_model_controls(self):
        """Per role: its own harness AND its own model.

        Not one harness for the whole run. Planning on one backend and auditing
        on another is the point -- the engine has supported qualified
        `harness:model` chain entries since c2ced81, and this is the surface for
        it.
        """
        r = roster()
        harnesses = sorted(r)
        default_h = os.environ.get("HARNESS") or ("opencode" if "opencode" in r
                                                  else (harnesses[0] if harnesses else "agy"))
        # No "Run as" control. It had no model beside it and no meaning a user
        # could act on: it only decided which <HARNESS>_CHAIN_<ROLE> variable the
        # engine reads. That is derivable -- whichever harness the most roles use
        # -- so it is derived, and any role on a different one is written as the
        # qualified `harness:model` entry the engine already resolves.
        self.default_h = default_h
        self.role_h, self.role_vars = {}, {}
        self.role_h_boxes, self.role_boxes = {}, {}
        self.role_effort_vars, self.role_effort_boxes = {}, {}
        self.role_labels = {}

        for column, heading in enumerate(("Role", "Backend", "Model", "Effort")):
            ttk.Label(
                self.models_frame, text=heading, foreground="#555"
            ).grid(row=0, column=column, sticky="w",
                   padx=(0 if column == 0 else 8, 0), pady=(0, 2))

        for i, role in enumerate(ROLES):
            row = i + 1
            lbl = ttk.Label(self.models_frame, text=role.capitalize())
            lbl.grid(row=row, column=0, sticky="w")
            self.role_labels[role] = lbl
            role_harnesses = harness_choices(role)
            selected_harness = (default_h if default_h in role_harnesses
                                else (role_harnesses[0]
                                      if role_harnesses else ""))
            hv = tk.StringVar(value=selected_harness)
            self.role_h[role] = hv
            hbox = ttk.Combobox(self.models_frame, textvariable=hv, width=10,
                                state="readonly", values=role_harnesses,
                                exportselection=False)
            hbox.grid(row=row, column=1, sticky="w", padx=(8, 0), pady=2)
            hbox.bind("<<ComboboxSelected>>",
                      lambda _e, rr=role: self.refresh_role(rr))
            self.role_h_boxes[role] = hbox

            mv = tk.StringVar()
            self.role_vars[role] = mv
            # Fixed width: a combobox that sizes to its longest value makes the
            # window jump between a short alias and a long vendor-qualified id.
            mbox = ttk.Combobox(self.models_frame, textvariable=mv,
                                state="readonly", width=30,
                                exportselection=False)
            mbox.grid(row=row, column=2, sticky="w", padx=(6, 0), pady=2)
            mbox.bind("<<ComboboxSelected>>",
                      lambda _e, rr=role: self.refresh_model(rr))
            self.role_boxes[role] = mbox
            effort = tk.StringVar()
            self.role_effort_vars[role] = effort
            effort_box = ttk.Combobox(
                self.models_frame, textvariable=effort, width=10,
                state="readonly", exportselection=False)
            effort_box.grid(row=row, column=3, sticky="w", padx=(8, 0), pady=2)
            effort_box.bind("<<ComboboxSelected>>",
                            lambda _e, rr=role: self.refresh_effort(rr))
            self.role_effort_boxes[role] = effort_box
        # One configurable availability fallback for every active role. Model
        # identifiers still come solely from models.json.
        row = len(ROLES) + 1
        ttk.Label(self.models_frame, text="Backup").grid(row=row, column=0,
                                                          sticky="w", pady=(6, 0))
        default_fb = profiles._default_fallback()
        self.backup_h = tk.StringVar(value=default_fb[0] if default_fb else default_h)
        self.backup_var = tk.StringVar(value=default_fb[1] if default_fb else "")
        self.backup_h_box = ttk.Combobox(
            self.models_frame, textvariable=self.backup_h, width=10,
            state="readonly", values=harnesses, exportselection=False)
        self.backup_h_box.grid(row=row, column=1, sticky="w", padx=(8, 0), pady=(6, 0))
        self.backup_h_box.bind("<<ComboboxSelected>>",
                               lambda _e: self.refresh_backup())
        self.backup_box = ttk.Combobox(
            self.models_frame, textvariable=self.backup_var, width=30,
            state="readonly", exportselection=False)
        self.backup_box.grid(row=row, column=2, sticky="w", padx=(6, 0), pady=(6, 0))
        self.backup_box.bind("<<ComboboxSelected>>",
                             lambda _e: self.refresh_backup_model())
        self.backup_effort = tk.StringVar()
        self.backup_effort_box = ttk.Combobox(
            self.models_frame, textvariable=self.backup_effort, width=10,
            state="readonly", exportselection=False)
        self.backup_effort_box.grid(row=row, column=3, sticky="w", padx=(8, 0),
                                    pady=(6, 0))
        self.backup_effort_box.bind("<<ComboboxSelected>>",
                                    lambda _e: self.refresh_backup_effort(
                                        reset=False))
        self.refresh_backup()

        # Profiles: role selections plus the explicit backup selection.
        row += 1
        ttk.Label(self.models_frame, text="Profile").grid(row=row, column=0,
                                                          sticky="w", pady=(6, 0))
        self.profile = tk.StringVar()
        self.profile_box = ttk.Combobox(self.models_frame, textvariable=self.profile,
                                        width=10, values=sorted(load_profiles()),
                                        exportselection=False)
        # Re-read on every open. The values were captured once at construction,
        # so a profile written by the CLI or by another window stayed invisible
        # until the app was relaunched -- which reads as "profiles don't save".
        self.profile_box["postcommand"] = self.refresh_profile_list
        self.profile_box.grid(row=row, column=1, sticky="w", padx=(8, 0), pady=(6, 0))
        self.profile_box.bind("<<ComboboxSelected>>", lambda _e: self.apply_profile())
        pb = ttk.Frame(self.models_frame)
        pb.grid(row=row, column=2, sticky="w", padx=(6, 0), pady=(6, 0))
        ttk.Button(pb, text="Save", width=6, command=self.save_profile).pack(side="left")
        ttk.Button(pb, text="Delete", width=7,
                   command=self.delete_profile).pack(side="left", padx=(4, 0))

        self.role_note = ttk.Label(self.models_frame, text="", foreground="#999",
                                   wraplength=430, justify="left")
        self.role_note.grid(row=row + 1, column=0, columnspan=4, sticky="w",
                            pady=(6, 0))

        self.refresh_roles()
        self.toggle_beside()
        self.refresh_last_out()
        # Restore the last named profile on this device. A missing/deleted
        # preference falls back to the shared balanced profile rather than a
        # harness roster's first model.
        initial_profile = launch_profile()
        if initial_profile:
            self.profile.set(initial_profile)
            self.apply_profile(remember=False)

    # -- profiles ------------------------------------------------------------
    def choose_out(self):
        d = filedialog.askdirectory(title="Publish artifacts into")
        if d:
            self.out_choice.set(d)
            self.beside.set(False)
            self._remember_output(d)
            self.refresh_out_label()
            self.replan_selection()

    def use_last_out(self):
        """Restore the last explicit output destination on this device."""
        try:
            value = last_output.load()
        except runtime.ConfigError as exc:
            self._set("Last output folder is invalid.", str(exc)[:160])
            self.refresh_last_out()
            return
        if not value:
            return
        self.out_choice.set(value)
        self.beside.set(False)
        self.refresh_out_label()
        self.replan_selection()

    def refresh_last_out(self):
        try:
            value = last_output.load()
        except runtime.ConfigError:
            value = ""
        if not value:
            self._last_out_tip = ""
            self.last_out_btn.grid_forget()
            return
        self._last_out_tip = value
        self.last_out_btn.configure(text=last_output.button_label(value))
        self.last_out_btn.grid(row=1, column=1, padx=(4, 0), pady=(4, 0),
                               sticky="ew")

    def _remember_output(self, value):
        try:
            last_output.save(str(value))
        except (OSError, ValueError, runtime.ConfigError) as exc:
            self._set("Could not remember the output folder.", str(exc)[:160])
            return
        self.refresh_last_out()

    def _engage_output_field(self, _event=None):
        """Typing in Output is the destination; Beside source must yield."""
        if self.beside.get():
            self.beside.set(False)
            self.refresh_out_label()

    def _out_choice_written(self, *_):
        if getattr(self, "_out_syncing", False):
            return
        if self.out_choice.get().strip() and self.beside.get():
            self.beside.set(False)
            self.refresh_out_label()

    def refresh_out_label(self):
        """Name the real destination for the input actually selected.

        With pasted text there is no source file to sit beside, so the run
        publishes into Downloads. Labelling that "Beside source" states
        something untrue about where the artifacts will be.
        """
        no_file = self.source is None or self.pasted is not None
        if self.scope.get() == "corpus":
            label = "Beside selected folder"
        else:
            label = "To Downloads" if no_file else "Beside source"
        self.beside_btn.configure(text=label)

    def toggle_beside(self):
        """Beside-source and an explicit directory are ONE setting.

        Leaving a path visible in a box the run ignores is the class of thing
        that reads as a broken control, so engaging the toggle clears it.
        The Output field stays typeable: focusing or typing it turns this off.
        """
        self._out_syncing = True
        try:
            if self.beside.get():
                self.out_choice.set("")
        finally:
            self._out_syncing = False
        self.refresh_out_label()
        self.replan_selection()

    def refresh_profile_list(self):
        try:
            self.profile_box.configure(values=sorted(load_profiles()))
        except runtime.ConfigError as exc:
            self._set("Profiles are invalid.", str(exc))

    def save_profile(self):
        name = self.profile.get().strip()
        if not name:
            self._set("Name the profile first, then Save.", "")
            return
        try:
            d = load_profiles()
        except runtime.ConfigError as exc:
            self._set("Profiles are invalid.", str(exc))
            return
        d[name] = {r: [self.role_h[r].get(), self.role_vars[r].get(),
                        self.role_effort_vars[r].get()] for r in ROLES}
        d[name]["_fallback"] = [self.backup_h.get(), self.backup_var.get(),
                                 self.backup_effort.get()]
        try:
            save_profiles(d)
            save_last_profile(name)
        except (OSError, runtime.ConfigError) as exc:
            self._set("Could not save the profile.", str(exc))
            return
        self.profile_box["values"] = sorted(d)
        self._set(f"Saved profile {name!r}.", "")

    def apply_profile(self, remember=True):
        try:
            sel = load_profiles().get(self.profile.get())
        except runtime.ConfigError as exc:
            self._set("Profiles are invalid.", str(exc))
            return
        if not sel:
            return
        for role, pair in sel.items():
            if role not in self.role_h:
                continue
            h, m, effort = _pick(pair)
            saved_setting = model_config.model_setting(h, m)
            m = model_config.logical_model(h, m)
            effort = effort or str(saved_setting.get("value") or "")
            self.role_h[role].set(h)
            self.refresh_role(role)          # repopulate that harness's models
            if m in self.role_boxes[role]["values"]:
                self.role_vars[role].set(m)
                self.refresh_role_effort(role)
            if effort in self.role_effort_boxes[role]["values"]:
                self.role_effort_vars[role].set(effort)
                self.refresh_effort(role)
            # A model that models.json no longer lists is not silently kept:
            # refresh_role has already fallen back to the roster's preference.
        fb = sel.get("_fallback")
        if not (isinstance(fb, (list, tuple)) and len(fb) >= 2):
            default_fb = profiles._default_fallback()
            if len(default_fb) >= 2:
                fb = list(default_fb)
        if fb and len(fb) >= 2:
            fh, fm, fe = _pick(fb)
            saved_setting = model_config.model_setting(fh, fm)
            fm = model_config.logical_model(fh, fm)
            fe = fe or str(saved_setting.get("value") or "")
            self.backup_h.set(fh); self.refresh_backup()
            if fm in self.backup_box["values"]:
                self.backup_var.set(fm)
                self.refresh_backup_effort()
            if fe in self.backup_effort_box["values"]:
                self.backup_effort.set(fe)
                self.refresh_backup_effort(reset=False)
        self.update_models_summary()
        if remember:
            try:
                save_last_profile(self.profile.get().strip())
            except OSError as exc:
                self._set("Could not remember the profile.", str(exc))

    def delete_profile(self):
        try:
            d = load_profiles()
        except runtime.ConfigError as exc:
            self._set("Profiles are invalid.", str(exc))
            return
        if d.pop(self.profile.get(), None) is None:
            return
        try:
            save_profiles(d)
        except (OSError, runtime.ConfigError) as exc:
            self._set("Could not delete the profile.", str(exc))
            return
        self.profile_box["values"] = sorted(load_profiles())
        # Deleting the selected profile also invalidates the remembered name;
        # launch_profile repairs it to balanced when that shared default exists.
        fallback = launch_profile()
        self.profile.set(fallback)
        if fallback:
            self.apply_profile(remember=False)

    def refresh_role(self, role):
        """Repopulate one role's models for the harness that role now uses."""
        harnesses = harness_choices(role)
        self.role_h_boxes[role]["values"] = harnesses
        selected = self.role_h[role].get()
        if selected not in harnesses:
            selected = harnesses[0] if harnesses else ""
            self.role_h[role].set(selected)
        options = model_choices(selected, role)
        self.role_boxes[role]["values"] = options
        cur = model_config.logical_model(selected, self.role_vars[role].get())
        # A stale model from the previous harness must not survive into a chain
        # that does not contain it.
        self.role_vars[role].set(cur if cur in options
                                 else (options[0] if options else ""))
        self.refresh_role_effort(role)
        self.update_models_summary()

    @staticmethod
    def option_label(harness: str, model: str = "", role: str = "") -> str:
        """Return only the current bare setting."""
        setting = model_config.model_setting(harness, model)
        value = setting.get("value") or model_config.setting_default(
            harness, model, role)
        return str(value) if value else ""

    def refresh_role_effort(self, role):
        harness, model = self.role_h[role].get(), self.role_vars[role].get()
        values = model_config.setting_values(harness, model)
        box = self.role_effort_boxes[role]
        box["values"] = values
        current = self.option_label(harness, model, role)
        self.role_effort_vars[role].set(current if current in values else
                                        (values[0] if values else ""))
        box["state"] = "readonly" if values else "disabled"

    def refresh_effort(self, role):
        # Provider IDs that encode effort are resolved only when the job is
        # frozen.  Keeping the logical family selected is what prevents the
        # Model dropdown from turning into one row per effort level.
        self.update_models_summary()

    def refresh_model(self, role):
        """Refresh both the visible setting and the compact model summary."""
        self.refresh_role_effort(role)
        self.update_models_summary()

    def refresh_roles(self):
        for role in ROLES:
            self.refresh_role(role)
        self.refresh_role_availability()

    def refresh_backup(self):
        """Offer a fallback capable of every role this mode will call."""
        active_roles = roles_for(self.mode.get())
        harnesses = backup_harness_choices(active_roles)
        self.backup_h_box["values"] = harnesses
        selected = self.backup_h.get()
        if selected not in harnesses:
            selected = harnesses[0] if harnesses else ""
            self.backup_h.set(selected)
        by_role = roster().get(self.backup_h.get(), {})
        sets = [set(by_role.get(r) or []) for r in active_roles]
        common = set.intersection(*sets) if sets else set()
        order = model_choices(self.backup_h.get(), "write")
        options = [m for m in order if m in common]
        self.backup_box["values"] = options
        cur = model_config.logical_model(
            self.backup_h.get(), self.backup_var.get())
        self.backup_var.set(cur if cur in options else (options[0] if options else ""))
        self.refresh_backup_effort()
        self.update_models_summary()

    def refresh_backup_model(self):
        self.refresh_backup_effort()
        self.update_models_summary()

    def refresh_backup_effort(self, reset=True):
        """Refresh Backup choices without erasing a user-selected value.

        Re-selecting the model establishes its declared default.  Selecting
        an effort, however, must preserve that selection; otherwise the
        combobox callback immediately changed ``low`` back to the model's
        declared default.
        """
        values = model_config.setting_values(
            self.backup_h.get(), self.backup_var.get())
        self.backup_effort_box["values"] = values
        if reset:
            current = self.option_label(self.backup_h.get(), self.backup_var.get())
            self.backup_effort.set(current if current in values else
                                   (values[0] if values else ""))
        elif self.backup_effort.get() not in values:
            self.backup_effort.set(values[0] if values else "")
        self.backup_effort_box["state"] = "readonly" if values else "disabled"

    def refresh_role_availability(self):
        """Disable the pickers the selected mode will not use."""
        self._sync_instruction_mode()
        used = roles_for(self.mode.get())
        for role in ROLES:
            on = role in used
            self.role_h_boxes[role]["state"] = "readonly" if on else "disabled"
            self.role_boxes[role]["state"] = "readonly" if on else "disabled"
            values = self.role_effort_boxes[role]["values"]
            self.role_effort_boxes[role]["state"] = (
                "readonly" if on and values else "disabled")
            if role in self.role_labels:
                self.role_labels[role]["foreground"] = "" if on else "#999"
        inert = [r for r in ROLES if r not in used]
        self.role_note["text"] = (
            "" if not inert else
            f"{self.mode.get().split(' —')[0].split(' (')[0]} does not call "
            f"{', '.join(inert)} — those selections are ignored for this mode.")
        self.refresh_backup()
        self.update_models_summary()
        self.refresh_instruction_availability()

    def instructions_allowed(self, mode_label=None) -> bool:
        """Custom summary/cleanup instructions have no meaning for Read-aloud."""
        label = mode_label if mode_label is not None else self.mode.get()
        mode = mode_config.BY_LABEL.get(label)
        return bool(mode and mode.instructions)

    def refresh_instruction_availability(self):
        allowed = self.instructions_allowed()
        state = "normal" if allowed else "disabled"
        self.instructions_toggle.configure(state=state)
        self.instructions_box.configure(state=state)
        self.instruction_preset_box.configure(
            state="readonly" if allowed else "disabled")

    def refresh_instruction_presets(self):
        mode = mode_config.BY_LABEL.get(self.mode.get())
        values = custom_instructions.preset_names(mode.key) if mode else ()
        self.instruction_preset_box.configure(values=values)
        self.instruction_preset.set("")

    def apply_instruction_preset(self, _event=None):
        mode = mode_config.BY_LABEL.get(self.mode.get())
        if not mode or not mode.instructions:
            return
        text = custom_instructions.preset_text(
            mode.key, self.instruction_preset.get())
        self.instructions_box.configure(state="normal")
        self.instructions_box.delete("1.0", "end")
        self.instructions_box.insert("1.0", text)
        self.show_instructions.set(bool(text))
        self.toggle_instructions()

    def _sync_instruction_mode(self):
        """Keep a separate visible draft for each stable product mode key."""
        mode = mode_config.BY_LABEL.get(self.mode.get())
        new_key = mode.key if mode else ""
        if new_key == self._instruction_mode_key:
            return
        old_key = self._instruction_mode_key
        if old_key and mode_config.BY_KEY.get(old_key, None):
            self.instruction_drafts[old_key] = (
                self.instructions_box.get("1.0", "end-1c")
                if self.show_instructions.get() else "")
        self._instruction_mode_key = new_key
        if mode and mode.instructions:
            if new_key not in self.instruction_drafts:
                self.instruction_drafts[new_key] = custom_instructions.load_last(
                    new_key)
            text = self.instruction_drafts[new_key]
        else:
            text = ""
        self.instructions_box.configure(state="normal")
        self.instructions_box.delete("1.0", "end")
        if text:
            self.instructions_box.insert("1.0", text)
        self.show_instructions.set(bool(text))
        self.refresh_instruction_presets()
        if text:
            self.instructions_frame.grid(
                row=self.instructions_row, column=0, columnspan=3, sticky="ew",
                padx=(8, 0), pady=(4, 0))
        else:
            self.instructions_frame.grid_forget()

    def toggle_instructions(self):
        if self.show_instructions.get():
            self.instructions_frame.grid(row=self.instructions_row, column=0,
                                         columnspan=3, sticky="ew", padx=(8, 0),
                                         pady=(4, 0))
        else:
            self.instructions_frame.grid_forget()
        self.refresh_instruction_availability()

    SUMMARY_CHARS = 52

    def update_models_summary(self):
        active_roles = roles_for(self.mode.get())
        picks = {r: (self.role_h[r].get(), self.role_vars[r].get())
                 for r in active_roles}
        distinct = {f"{h}:{m}" for h, m in picks.values() if m}
        if len(distinct) == 1:
            h, m = next((pick for pick in picks.values() if pick[1]), ("", ""))
            full = f"{h}  ·  {m}"
        else:
            full = "  ·  ".join(f"{r[:4]} {picks[r][0]}:{picks[r][1].split('/')[-1]}"
                               for r in active_roles if picks[r][1])
        # TRUNCATE. A label that grows with its text resizes the whole window
        # every time a harness changes, which is why picking models made the
        # window jump. The full text is on the tooltip.
        shown = (full if len(full) <= self.SUMMARY_CHARS
                 else full[:self.SUMMARY_CHARS - 1] + "\u2026")
        self.models_summary.config(text=shown)
        self._tip_text = full

    def toggle_models(self):
        if self.show_models.get():
            self.models_frame.grid(row=self.models_row, column=0, columnspan=4,
                                   sticky="w", padx=(0, 0), pady=(4, 0))
        else:
            self.models_frame.grid_forget()

    def run_harness(self) -> str:
        """Whichever harness the most roles use. Only decides which variable the
        engine reads; every role still chooses its own CLI."""
        active_roles = roles_for(self.mode.get())
        used = [self.role_h[r].get() for r in active_roles
                if self.role_vars[r].get()]
        return max(set(used), key=used.count) if used else self.default_h

    def chosen_env(self, runtime_snapshot: dict | None = None) -> dict:
        # One entry per role, not the whole chain: an explicit choice should not
        # silently fall through to a model nobody picked.
        used = roles_for(self.mode.get())
        picks = {r: (self.role_h[r].get(), self.role_vars[r].get(),
                     self.role_effort_vars[r].get())
                 for r in used if self.role_vars[r].get()}
        fallback = (self.backup_h.get(), self.backup_var.get(),
                    self.backup_effort.get())
        configured = model_config.roster(runtime_snapshot)
        return role_env(self.run_harness(), picks, fallback,
                        roster_value=configured)

    # -- input ---------------------------------------------------------------
    def mode_changed(self, *_args):
        mode_obj = mode_config.BY_LABEL.get(self.mode.get())
        mode_key = mode_obj.key if mode_obj else "summarize"
        if mode_key != "summarize" and self.scope.get() == "corpus":
            self.scope.set("batch")
            if self.pending and not self.queue_started and not self.active:
                self._regroup_waiting_paths("batch")
        self._sync_affix_ui(mode_obj, mode_key)
        self.refresh_role_availability()
        self.refresh_selection_for_mode()
        self.refresh_scope()
        self._update_affix_preview()

    def _sync_affix_ui(self, mode_obj, mode_key: str):
        if not hasattr(self, "affix_entry"):
            return
        saved = last_affix.load(mode_key)
        self._updating_affix = True
        try:
            self.affix_text.set(saved["primary"])
            self.affix_secondary.set(saved["secondary"])
        finally:
            self._updating_affix = False

        has_secondary = bool(mode_obj and len(mode_obj.suffixes) > 1)
        if has_secondary:
            self.affix_label1.grid(row=0, column=1, padx=(12, 4), sticky="w")
            self.affix_entry.grid(row=0, column=2, padx=(0, 10), sticky="w")
            self.affix_label2.grid(row=0, column=3, padx=(0, 4), sticky="w")
            self.affix_entry2.grid(row=0, column=4, padx=(0, 10), sticky="w")
            self.affix_preview.grid(row=0, column=5, padx=(0, 0), sticky="w")
        else:
            self.affix_label1.grid_remove()
            self.affix_label2.grid_remove()
            self.affix_entry2.grid_remove()
            self.affix_entry.grid(row=0, column=1, padx=(12, 10), sticky="w")
            self.affix_preview.grid(row=0, column=2, padx=(0, 0), sticky="w")

    def _update_affix_preview(self):
        if not hasattr(self, "affix_preview"):
            return
        mode_obj = mode_config.BY_LABEL.get(self.mode.get())
        mode_key = mode_obj.key if mode_obj else "summarize"
        kind = self.affix_kind.get() if hasattr(self, "affix_kind") else "prefix"
        text = self.affix_text.get().strip() if hasattr(self, "affix_text") else ""
        secondary = self.affix_secondary.get().strip() if hasattr(self, "affix_secondary") else ""
        names = selection.format_output_filenames(
            "name", mode_key, affix_kind=kind, affix_text=text, affix_secondary=secondary)
        self.affix_preview.configure(text=f"e.g. {' · '.join(names)}")

    def affix_changed(self, *_args):
        if getattr(self, "_updating_affix", False):
            return
        kind = self.affix_kind.get()
        text = self.affix_text.get().strip()
        secondary = self.affix_secondary.get().strip()
        mode_obj = mode_config.BY_LABEL.get(self.mode.get())
        mode_key = mode_obj.key if mode_obj else "summarize"
        last_affix.save(kind, text, secondary, mode_key=mode_key)
        self._update_affix_preview()
        self.replan_selection()
        out_dir = self._output_directory()
        for job in self.pending:
            job["affix_kind"] = kind
            job["affix_text"] = text
            job["affix_secondary"] = secondary
            sel = job.get("selection")
            if sel and not sel.is_text:
                m_key = mode_config.BY_LABEL[job["mode"]].key
                new_sel = sel.with_outputs(
                    m_key, out_dir, affix_kind=kind, affix_text=text,
                    affix_secondary=secondary)
                job["selection"] = new_sel
                manifest_path = (job.get("root") / "selection.json"
                                 if job.get("root") else None)
                if manifest_path and manifest_path.exists():
                    selection.write_manifest(new_sel, manifest_path)

    def corpus_scope_allowed(self) -> bool:
        mode = mode_config.BY_LABEL.get(self.mode.get())
        # Corpus is the one intentional grouped admission. Batch remains the
        # default, and the waiting path-backed queue can be regrouped until
        # Start freezes it.
        return bool(mode and mode.key == "summarize" and self.pasted is None)

    def refresh_scope(self):
        mode = mode_config.BY_LABEL.get(self.mode.get())
        eligible = bool(mode and mode.key == "summarize" and
                        self.pasted is None)
        mutable = not self.queue_started and not self.active
        self.batch_scope.configure(state="normal" if mutable else "disabled")
        self.corpus_scope.configure(
            state="normal" if mutable and self.corpus_scope_allowed()
            else "disabled")
        if (not eligible or
                (self.scope.get() == "corpus" and
                 not self.corpus_scope_allowed())):
            self.scope.set("batch")
        if self.scope.get() == "corpus":
            if (not self.corpus_name.get() and self.selection
                    and not self.selection.is_text):
                self.corpus_name.set(selection.default_corpus_name(self.selection))
            self.corpus_name_label.grid(row=self.corpus_name_row, column=0,
                                        sticky="w", pady=(4, 0))
            self.corpus_name_entry.grid(row=self.corpus_name_row, column=1,
                                        columnspan=2, sticky="ew", padx=(8, 0),
                                        pady=(4, 0))
        else:
            self.corpus_name_label.grid_forget()
            self.corpus_name_entry.grid_forget()
        self.refresh_out_label()

    def scope_changed(self):
        if self.scope.get() == "corpus" and not self.corpus_scope_allowed():
            self.scope.set("batch")
        if self.active or self.queue_started:
            self.refresh_scope()
            return
        try:
            changed, documents = self._regroup_waiting_paths(self.scope.get())
        except (OSError, ValueError, runtime.ConfigError) as exc:
            self.scope.set("batch")
            self._set("Could not change the queued scope.", str(exc)[:160])
            self.refresh_scope()
            return
        self.refresh_scope()
        self.replan_selection()
        if changed:
            label = "one Corpus job" if self.scope.get() == "corpus" else "Batch jobs"
            self._set(f"Queued as {label}.",
                      f"{documents} documents will run when Start is pressed.")

    def _output_directory(self):
        return None if self.beside.get() else (self.out_choice.get().strip() or None)

    def _accept_selection(self, chosen):
        """Turn an accepted input into visible waiting jobs immediately.

        The input controls are an admission point, not a hidden selection
        buffer.  Batch creates one queue item per document, which makes the
        count and the individual cancel action mean the same thing.  Start
        only arms the queue; it never creates the jobs.
        """
        self.resolving = False
        self._resolution_values = []
        mode_key = mode_config.BY_LABEL[self.mode.get()].key
        queued = 0
        queue_error = None
        try:
            queued = self._enqueue_selection(chosen, mode_key)
        except (OSError, ValueError, runtime.ConfigError) as exc:
            queue_error = str(exc)

        # A native paste may already have left the clipboard text in the
        # Entry before bind_all handles it. Once admission succeeds, remove
        # that stale text so pressing Return cannot enqueue the same input a
        # second time. Keep it on an admission error so the user can correct
        # and retry the path.
        if queued:
            self.path_entry.delete(0, "end")

        if chosen.is_text:
            self.source, self.pasted = None, chosen.raw_text
            words = len((chosen.raw_text or "").split())
            self.src_lbl.config(text=f"pasted text — {words:,} words added")
        else:
            self.source, self.pasted = [doc.source_path for doc in chosen.documents], None
            count = len(chosen.documents)
            self.src_lbl.config(
                text=(f"{count} document{'s' if count != 1 else ''} added"
                      if count else "No documents added"))

        # No accepted selection remains hidden behind the Add controls. The
        # queue rows are the durable visible record. Scope may regroup waiting
        # path jobs until Start; active jobs remain immutable.
        self.selection = None
        self.selection_inputs = []
        self.refresh_scope()
        self.refresh_out_label()
        self._refresh_queue()

        if queue_error:
            status = ("Custom instructions are invalid."
                      if "instruction" in queue_error.lower()
                      else "Could not add the input to the queue.")
            self._set(status, queue_error[:160])
        elif queued:
            count = 1 if chosen.is_text else len(chosen.documents)
            noun = "document" if count == 1 else "documents"
            detail = ("Press Start to begin the queue."
                      if not self.queue_started
                      else "The active queue will process it automatically.")
            if self.scope.get() == "corpus" and not chosen.is_text:
                self._set(f"Added {count} {noun} to one Corpus job.", detail)
            else:
                self._set(f"Added {count} {noun} to the queue.", detail)
        else:
            detail = "No supported input was added."
            if chosen.errors:
                detail = "; ".join(
                    f"{e.get('kind')}: {e.get('path')}"
                    for e in chosen.errors[:3])
                if len(chosen.errors) > 3:
                    detail += f"; +{len(chosen.errors) - 3} more"
            self._set("Nothing added to the queue.", detail)

    def _one_document_selection(self, chosen, doc, mode_key, out_dir):
        """Make one immutable Batch selection for one accepted document."""
        return chosen.for_single_document(doc).with_outputs(
            mode_key, out_dir,
            affix_kind=self.affix_kind.get(),
            affix_text=self.affix_text.get().strip(),
            affix_secondary=self.affix_secondary.get().strip())

    def _job_from_selection(self, chosen, *, mode_key, scope, corpus_name,
                            out_dir, instructions, runtime_snapshot,
                            frozen_runtime, selected_env):
        """Snapshot one already-admitted queue item and its run evidence."""
        if chosen.is_text:
            label = "pasted text"
            source = None
        elif scope == "corpus":
            label = f"Corpus ({len(chosen.documents)} documents)"
            source = [doc.source_path for doc in chosen.documents]
        else:
            label = chosen.documents[0].source_path.name
            source = [chosen.documents[0].source_path]
        stamp = (datetime.datetime.now(datetime.timezone.utc)
                 .strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid.uuid4().hex[:8])
        root = job_root() / stamp
        root.mkdir(parents=True, exist_ok=False)
        if chosen.is_text:
            (root / "pasted.txt").write_text(
                chosen.raw_text or "", encoding="utf-8")
        else:
            selection.write_manifest(chosen, root / "selection.json")
        if instructions.strip() and self.instructions_allowed(self.mode.get()):
            (root / "instructions.txt").write_text(instructions, encoding="utf-8")
        return {
            "mode": self.mode.get(), "source": source,
            "selection": chosen, "scope": scope,
            "corpus_name": corpus_name, "pasted": chosen.raw_text,
            "out": out_dir, "instructions": instructions,
            "affix_kind": self.affix_kind.get(),
            "affix_text": self.affix_text.get().strip(),
            "affix_secondary": self.affix_secondary.get().strip(),
            "env": {**selected_env, "SUMM_RUNTIME_JSON": frozen_runtime},
            "active_target_limit": runtime_snapshot["active_targets"],
            "label": label, "root": root, "stamp": stamp,
        }

    def _jobs_from_entries(self, entries, mode_key):
        """Freeze queue jobs from the controls currently visible."""
        runtime_snapshot = runtime.config()
        selected_env = self.chosen_env(runtime_snapshot)
        frozen_runtime = runtime.frozen_json(runtime_snapshot)
        instructions = (
            self.instructions_box.get("1.0", "end-1c")
            if (self.instructions_allowed() and self.show_instructions.get())
            else "")
        instructions = custom_instructions.validate_text(instructions)
        selected_mode = mode_config.BY_LABEL.get(self.mode.get())
        if selected_mode and selected_mode.instructions:
            custom_instructions.save_last(selected_mode.key, instructions)
            self.instruction_drafts[selected_mode.key] = instructions
        return [self._job_from_selection(
            item, mode_key=mode_key, scope=scope, corpus_name=corpus_name,
            out_dir=self._output_directory(), instructions=instructions,
            runtime_snapshot=runtime_snapshot, frozen_runtime=frozen_runtime,
            selected_env=selected_env)
                for item, scope, corpus_name in entries]

    @staticmethod
    def _discard_waiting_roots(jobs):
        """Remove queue snapshots that were replaced before execution."""
        base = job_root().resolve()
        for job in jobs:
            root = job.get("root")
            if not root:
                continue
            root = pathlib.Path(root).resolve()
            try:
                root.relative_to(base)
            except ValueError:
                continue
            shutil.rmtree(root, ignore_errors=True)

    def _regroup_waiting_paths(self, scope):
        """Apply Batch/Corpus to every waiting path document before Start."""
        if self.active or self.queue_started:
            return False, 0
        indexed = [(index, job) for index, job in enumerate(self.pending)
                   if (job.get("selection") is not None
                       and not job["selection"].is_text)]
        if not indexed:
            return False, 0
        old_jobs = [job for _, job in indexed]
        mode_key = mode_config.BY_LABEL[self.mode.get()].key
        combined = selection.combine_path_selections(
            (job["selection"] for job in old_jobs), mode_key)
        documents = len(combined.documents)
        if scope == "corpus":
            if documents < 2:
                raise ValueError("Corpus requires at least two queued documents")
            name = self.corpus_name.get().strip() or selection.default_corpus_name(combined)
            self.corpus_name.set(name)
            entries = [(combined.with_corpus_outputs(
                name, self._output_directory()), "corpus", name)]
        else:
            self.corpus_name.set("")
            entries = [(self._one_document_selection(
                combined, doc, mode_key, self._output_directory()), "batch", None)
                       for doc in combined.documents]
        new_jobs = self._jobs_from_entries(entries, mode_key)
        old_ids = {id(job) for job in old_jobs}
        first = sum(id(job) not in old_ids
                    for job in self.pending[:indexed[0][0]])
        retained = [job for job in self.pending if id(job) not in old_ids]
        self.pending = retained[:first] + new_jobs + retained[first:]
        self._discard_waiting_roots(old_jobs)
        self._refresh_queue()
        return True, documents

    def _enqueue_selection(self, chosen, mode_key):
        """Create queue jobs without starting them unless the queue is armed."""
        if chosen.is_text:
            if not (chosen.raw_text or "").strip():
                return 0
            entries = [(chosen, "batch", None)]
        elif self.scope.get() == "corpus":
            has_waiting_paths = not self.queue_started and any(
                job.get("selection") is not None
                and not job["selection"].is_text
                for job in self.pending)
            if len(chosen.documents) < 2 and not has_waiting_paths:
                raise ValueError(
                    "Corpus requires at least two documents added together")
            if len(chosen.documents) < 2:
                # A later Add may contain one document. Freeze it as an
                # ordinary waiting slice, then immediately fold it into the
                # existing Corpus below.
                entries = [(self._one_document_selection(
                    chosen, chosen.documents[0], mode_key,
                    self._output_directory()), "batch", None)]
            else:
                corpus = chosen.with_corpus_outputs(
                    self.corpus_name.get().strip() or None,
                    self._output_directory())
                self.corpus_name.set(corpus.corpus_name or "")
                entries = [(corpus, "corpus", corpus.corpus_name)]
        else:
            entries = [(
                self._one_document_selection(
                    chosen, doc, mode_key, self._output_directory()),
                "batch", None)
                for doc in chosen.documents]
        if not entries:
            return 0

        jobs = self._jobs_from_entries(entries, mode_key)
        self.pending.extend(jobs)
        if (self.scope.get() == "corpus" and not self.queue_started
                and len(self.pending) > 1):
            self._regroup_waiting_paths("corpus")
        value = self._output_directory()
        if value:
            self._remember_output(value)
        self._refresh_queue()
        if self.queue_started:
            self._drain_pending()
        return len(jobs)

    def _resolve_inputs(self, values, *, append=False):
        values = [pathlib.Path(value) for value in values]
        # Resolution runs off the Tk thread. A second Add click can arrive
        # before the first result, so the accepted selection is still empty at
        # that point. Keep the complete in-flight root list and restart the
        # resolver with the union instead of silently replacing the first path.
        if self.resolving:
            if self.pasted is not None:
                self._set("Cannot mix pasted prose and file paths.",
                          "Clear the selection or wait for the current selection.")
                return
            values = list(self._resolution_values) + values
        elif append:
            if self.pasted is not None:
                self._set("Cannot mix pasted prose and file paths.",
                          "Clear the selection or add only path-backed documents.")
                return
            values = list(self.selection_inputs) + values
        if not values:
            return
        mode_key = mode_config.BY_LABEL[self.mode.get()].key
        self._resolution_values = list(values)
        self.resolving = True
        self.selection_generation += 1
        generation = self.selection_generation
        self.start_btn.configure(state="disabled")
        self.selection_summary.config(text="Resolving selection…")

        def resolve_selection_worker():
            try:
                chosen = selection.resolve_paths(values, mode_key)
                self.selection_events.put((generation, chosen))
            except Exception as exc:
                self.selection_events.put((generation, exc))

        threading.Thread(target=resolve_selection_worker, daemon=True).start()
        self.root.after(50, self._poll_selection)

    def _poll_selection(self):
        try:
            generation, result = self.selection_events.get_nowait()
        except queue.Empty:
            if self.resolving:
                self.root.after(50, self._poll_selection)
            return
        if generation != self.selection_generation:
            if self.resolving:
                self.root.after(0, self._poll_selection)
            return
        if isinstance(result, Exception):
            self.resolving = False
            self._resolution_values = []
            self.start_btn.configure(state="disabled")
            self._set("Could not resolve the selection.", str(result)[:160])
            return
        self._accept_selection(result)

    def refresh_selection_for_mode(self):
        mode = mode_config.BY_LABEL.get(self.mode.get())
        mode_key = mode.key if mode else "summarize"
        self._sync_affix_ui(mode, mode_key)
        if self.selection_inputs and not self.resolving:
            self._resolve_inputs(self.selection_inputs)
        elif self.selection and self.selection.is_text:
            self.scope.set("batch")
            self._accept_selection(selection.text_selection(
                self.selection.raw_text or "",
                mode_key))
        if not self.queue_started and self.pending:
            runtime_snapshot = runtime.config()
            selected_env = self.chosen_env(runtime_snapshot)
            frozen_runtime = runtime.frozen_json(runtime_snapshot)
            out_dir = self._output_directory()
            for job in self.pending:
                job["mode"] = self.mode.get()
                job["affix_kind"] = self.affix_kind.get()
                job["affix_text"] = self.affix_text.get().strip()
                job["affix_secondary"] = self.affix_secondary.get().strip()
                job["env"] = {**selected_env, "SUMM_RUNTIME_JSON": frozen_runtime}
                sel = job.get("selection")
                if sel and not sel.is_text:
                    new_sel = sel.with_outputs(
                        mode_key, out_dir,
                        affix_kind=self.affix_kind.get(),
                        affix_text=self.affix_text.get().strip(),
                        affix_secondary=self.affix_secondary.get().strip())
                    job["selection"] = new_sel
                    manifest_path = (job.get("root") / "selection.json"
                                     if job.get("root") else None)
                    if manifest_path and manifest_path.exists():
                        selection.write_manifest(new_sel, manifest_path)
            self._refresh_queue()

    def replan_selection(self):
        if self.selection and not self.selection.is_text:
            mode_key = mode_config.BY_LABEL[self.mode.get()].key
            if self.scope.get() == "corpus" and self.corpus_scope_allowed():
                try:
                    self.selection = self.selection.with_corpus_outputs(
                        self.corpus_name.get().strip() or None,
                        self._output_directory())
                except ValueError as exc:
                    self._set("Corpus output plan is invalid.", str(exc))
            else:
                self.selection = self.selection.with_outputs(
                    mode_key, self._output_directory(),
                    affix_kind=self.affix_kind.get(),
                    affix_text=self.affix_text.get().strip(),
                    affix_secondary=self.affix_secondary.get().strip())
        changed = False
        if not self.queue_started:
            for job in self.pending:
                chosen = job.get("selection")
                if job.get("scope") != "corpus" or not chosen:
                    continue
                name = (self.corpus_name.get().strip()
                        or job.get("corpus_name")
                        or selection.default_corpus_name(chosen))
                chosen = chosen.with_corpus_outputs(
                    name, self._output_directory())
                job["selection"] = chosen
                job["corpus_name"] = name
                manifest_path = pathlib.Path(job["root"]) / "selection.json"
                if manifest_path.exists():
                    selection.write_manifest(chosen, manifest_path)
                changed = True
        if changed and hasattr(self, "queue_rows"):
            self._refresh_queue()

    def pick(self):
        paths = filedialog.askopenfilenames(
            title="Add documents",
            filetypes=[("Documents", "*.md *.markdown *.txt *.pdf"),
                       ("All files", "*.*")])
        if paths:
            self._resolve_inputs(paths)

    def pick_folder(self):
        path = filedialog.askdirectory(title="Add document folder")
        if path:
            self._resolve_inputs([path])

    def add_path(self):
        raw = self.path_entry.get().strip()
        if not raw:
            self._set("Enter a file or folder path first.", "")
            return
        paths = selection.split_path_list(raw)
        if not paths:
            self._set("Enter a file or folder path first.", "")
            return
        self._resolve_inputs(paths)
        self.path_entry.delete(0, "end")

    def clear_selection(self):
        """Clear only the path-entry/resolution state, never queued jobs."""
        self.selection_generation += 1
        self.resolving = False
        self._resolution_values = []
        self.selection_inputs = []
        self.selection = None
        self.path_entry.delete(0, "end")
        self.source, self.pasted = None, None
        self.selection_summary.config(text="")
        self.src_lbl.config(text="Clipboard")
        self.refresh_out_label()
        self._refresh_queue()

    def review_selection(self):
        if not self.selection:
            rows = [f"Running: {state.spec.get('label', 'job')}"
                    for state in self.active.values()]
            rows.extend(f"Waiting: {job.get('label', 'job')}"
                        for job in self.pending)
            body = "\n".join(rows) or "Queue is empty."
            messagebox.showinfo("Queue review", body, parent=self.root)
            return
        if self.selection.is_text:
            body = self.selection.summary
        else:
            rows = [f"{d.id}  {d.relative_path}"
                    + (f"  [{d.error_kind}]" if d.error_kind else "")
                    for d in self.selection.documents]
            excluded = [f"ignored: {e.get('kind')} — {e.get('path')}"
                        for e in self.selection.exclusions[:20]]
            problems = [f"problem: {e.get('kind')} — {e.get('path')}"
                        for e in self.selection.errors]
            if self.scope.get() == "corpus":
                rows.append(f"Corpus: {self.corpus_name.get()}")
                rows.extend(f"output: {p}" for p in self.selection.corpus_outputs)
            body = "\n".join(rows + excluded + problems) or "No supported documents."
        messagebox.showinfo("Selection review", body, parent=self.root)

    def use_clip(self):
        # Capture NOW, using the same classifier as the CLI. Waiting until the
        # worker launches makes an ordinary later Copy operation change the job.
        return self.paste()

    def toggle(self):
        if self.show_console.get():
            self.console.grid(row=self.console_row, column=0, columnspan=3,
                              sticky="nsew", pady=(4, 0))
        else:
            self.console.grid_forget()

    # -- run -----------------------------------------------------------------
    def start(self):
        """Arm the queue and drain waiting jobs; never create a new job."""
        if self.cancelling:
            self._set("Cancellation is still finishing.",
                      "Wait for active work to stop before starting another run.")
            return
        if self.resolving:
            self._set("Selection is still resolving.",
                      "Wait for the document count and hashes to finish.")
            return
        # Keep old programmatic callers safe while the UI uses the new
        # admission path. A real click cannot reach this branch because every
        # accepted input is queued by _accept_selection.
        if self.selection is not None:
            chosen = self.selection
            self._accept_selection(chosen)
            if not self.pending and not self.queue_started:
                return
        if self.queue_started:
            self._set("Queue is already running.",
                      f"{len(self.active)} running · {len(self.pending)} waiting")
            return
        if not self.pending:
            self._set("Nothing queued.", "Add a document before pressing Start.")
            return
        self.queue_started = True
        self._set("Starting queue…",
                  f"{len(self.pending)} document(s) waiting to start")
        self._drain_pending()

    def _launch(self, job):
        self.running = True
        first = not self.active
        stamp = job.get("stamp")
        root = job.get("root")
        if root is None:
            stamp = (datetime.datetime.now(datetime.timezone.utc)
                     .strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid.uuid4().hex[:8])
            root = job_root() / stamp
            root.mkdir(parents=True, exist_ok=False)
            job["root"], job["stamp"] = root, stamp
        self.job = root
        # Pasted text is written down at LAUNCH from what was captured at
        # enqueue. Path-backed selections use one manifest argv element; the
        # manifest itself is the frozen queue snapshot and keeps Windows command
        # lines bounded even for a large recursive folder.
        source = job["source"]
        text_file = None
        if job["pasted"] is not None:
            text_file = root / "pasted.txt"
            if not text_file.exists():
                text_file.write_text(job["pasted"], encoding="utf-8")
        instructions_file = None
        instructions = job.get("instructions") or ""
        if instructions.strip() and self.instructions_allowed(job["mode"]):
            instructions_file = root / "instructions.txt"
            if not instructions_file.exists():
                instructions_file.write_text(instructions, encoding="utf-8")
        manifest_path = None
        chosen = job.get("selection")
        if chosen is not None and not chosen.is_text:
            manifest_path = root / "selection.json"
            if not manifest_path.exists():
                frozen = (chosen.with_corpus_outputs(
                              job.get("corpus_name"), job["out"])
                          if job.get("scope") == "corpus" else
                          chosen.with_outputs(
                              mode_config.BY_LABEL[job["mode"]].key,
                              job["out"],
                              affix_kind=job.get("affix_kind", "prefix"),
                              affix_text=job.get("affix_text", ""),
                              affix_secondary=job.get("affix_secondary", "")))
                selection.write_manifest(
                    frozen, manifest_path)
        cmd = build_cmd(root, job["mode"], source, job["out"], text_file,
                        instructions_file, manifest_path, job.get("scope", "batch"),
                        affix_kind=job.get("affix_kind", "prefix"),
                        affix_text=job.get("affix_text", ""),
                        affix_secondary=job.get("affix_secondary", ""))

        state = JobState(stamp, root, job)
        self.active[state.id] = state
        self._refresh_queue()
        self.cancel_btn.configure(state="normal")

        self.offset, self.parts, self.out_dir = 0, (0, 0), None
        self.part_failures = []
        self.started = time.monotonic()
        if first:
            self._clear_cards()
        self._set(f"Starting {job['label']}…",
                  f"{len(self.active)} active" if len(self.active) > 1 else "")
        self.bar.config(mode="determinate", maximum=1, value=0)
        self.open_out.pack_forget(); self.open_work.pack_forget()

        threading.Thread(target=self._run, args=(state, cmd, job["env"]),
                         daemon=True).start()
        if not self.polling:
            self.polling = True
            self.root.after(200, self._poll)

    @staticmethod
    def _write_cancel_request(state):
        """Atomically request cancellation without signalling the controller."""
        tmp = state.cancel_file.with_name(
            state.cancel_file.name + f".tmp-{uuid.uuid4().hex[:8]}")
        tmp.write_text("cancel\n", encoding="utf-8")
        os.replace(tmp, state.cancel_file)

    @staticmethod
    def _queue_id(job):
        """Stable identity for one queued job, including old test callers."""
        return job.get("stamp") or f"object-{id(job)}"

    def _refresh_queue(self):
        """Show running and waiting jobs with individual cancel controls."""
        for row in self.queue_rows.values():
            row.destroy()
        self.queue_rows = {}
        entries = []
        for key, state in self.active.items():
            spec = getattr(state, "spec", {}) or {}
            label = spec.get("label", "job") if hasattr(spec, "get") else "job"
            mode_obj = mode_config.BY_LABEL.get(spec.get("mode", ""))
            mode_tag = f" [{mode_obj.key}]" if mode_obj else ""
            entries.append((key, f"Running{mode_tag}: {label}",
                            "cancel_active_job",
                            bool(getattr(state, "cancel_requested", False))))
        for number, job in enumerate(self.pending, 1):
            mode_obj = mode_config.BY_LABEL.get(job.get("mode", ""))
            mode_tag = f" [{mode_obj.key}]" if mode_obj else ""
            entries.append(
                (self._queue_id(job), f"Queued {number}{mode_tag}: {job.get('label', 'job')}",
                 "remove_queued", False))
        if entries:
            active = sum(action == "cancel_active_job"
                         for _, _, action, _ in entries)
            waiting = len(entries) - active
            parts = [f"Queue: {len(entries)} job(s)"]
            if active:
                parts.append(f"{active} running")
            if waiting:
                parts.append(f"{waiting} waiting")
            parts.append("× cancels one")
            self.queue_label.config(text=" · ".join(parts))
        else:
            self.queue_label.config(text="Queue: empty")
        for ident, label, action, disabled in entries:
            row = ttk.Frame(self.queue_rows_frame)
            row.pack(fill="x", pady=(0, 2))
            ttk.Label(row, text=label, anchor="w").grid(
                row=0, column=0, sticky="ew")
            button = ttk.Button(
                row, text="×", width=3,
                command=lambda job_id=ident, fn=action: getattr(
                    self, fn)(job_id),
            )
            button.grid(row=0, column=1, padx=(6, 0))
            if disabled:
                button.configure(state="disabled")
            row.columnconfigure(0, weight=1)
            self.queue_rows[ident] = row
        if self.cancelling or self.active or self.queue_started:
            self.start_btn.configure(state="disabled")
        elif self.pending:
            self.start_btn.configure(
                state="normal", text=f"Start ({len(self.pending)} queued)")
        else:
            # Keep Start clickable so an empty queue explains itself in the
            # status line; it never invents a clipboard job or hidden input.
            self.start_btn.configure(state="normal", text="Start")
        self.refresh_scope()

    def _drain_pending(self):
        """Launch every waiting job that the frozen runtime permits."""
        if not self.queue_started or self.cancelling:
            self._refresh_queue()
            return
        while self.pending:
            limit = int(self.pending[0].get("active_target_limit", 1))
            if len(self.active) >= limit:
                break
            nxt = self.pending.pop(0)
            self._refresh_queue()
            self._set(f"Starting queued job: {nxt['label']}",
                      f"{len(self.pending)} waiting after this job")
            self._launch(nxt)
        if not self.active and not self.pending:
            self.queue_started = False
        self._refresh_queue()

    def cancel_active_job(self, ident):
        """Request cancellation for one active job, leaving others running."""
        state = self.active.get(ident)
        if state is None or getattr(state, "cancel_requested", False):
            return False
        try:
            self._write_cancel_request(state)
        except OSError as exc:
            self._set("Could not cancel that job.", str(exc)[:160])
            return False
        state.cancel_requested = True
        self._refresh_queue()
        self._set("Cancellation requested.",
                  f"{state.spec.get('label', 'Job')} will stop cooperatively.")
        return True

    def remove_queued(self, ident):
        """Remove exactly one waiting job; active jobs remain untouched."""
        for index, job in enumerate(self.pending):
            if self._queue_id(job) != ident:
                continue
            removed = self.pending.pop(index)
            self._refresh_queue()
            self._set("Queued job removed.",
                      f"{removed.get('label', 'Job')} will not start.")
            return True
        return False

    def cancel_active(self):
        """Cancel every active job and discard work that has not started.

        The marker is safe even before the worker thread reaches Popen. The UI
        deliberately never terminates the top-level CLI: that process owns the
        guarded publication step and must stay alive long enough to finish or
        avoid it coherently.
        """
        if not self.active:
            return
        queued = len(self.pending)
        self.pending.clear()
        self._refresh_queue()
        failures = []
        marked = 0
        for state in tuple(self.active.values()):
            try:
                self._write_cancel_request(state)
                state.cancel_requested = True
                marked += 1
            except OSError as exc:
                failures.append(f"{state.spec['label']}: {exc}")
        self._refresh_queue()
        self.cancelling = marked > 0
        self.queue_started = False
        self.start_btn.configure(state="disabled" if marked else "normal")
        self.cancel_btn.configure(state="normal" if failures else "disabled")
        detail = f"Cancellation requested for {marked}/{len(self.active)} active job(s)"
        if queued:
            detail += f"; {queued} queued job(s) removed"
        if failures:
            detail += (f"; {len(failures)} marker(s) could not be written — "
                       "those jobs may continue and publish; retry Cancel active")
            status = ("Cancellation only partly requested."
                      if marked else "Could not request cancellation.")
        else:
            status = "Cancelling active work…"
        self._set(status, detail + ".")

    def _clear_cards(self):
        for row in self.cards:
            row.destroy()
        self.cards = []
        self._console_clear()

    def _run(self, state, cmd, env_overrides=None):
        log = state.root / "console.log"
        try:
            with log.open("w") as fh:
                state.proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, env={**os.environ, **(env_overrides or {})})
                for line in state.proc.stdout:
                    fh.write(line); fh.flush()
                    state.events.put(("line", line.rstrip()))
                rc = state.proc.wait()
        except Exception as e:
            state.events.put(("line", f"could not launch: {e}"))
            rc = None                                # launch_failed, not a CLI exit
        state.events.put(("exit", rc))

    def _poll(self):
        """Drain stdout and the event stream on the Tk loop, never off it."""
        finished = []
        for state in list(self.active.values()):
            done = None
            has_exit = False
            while True:
                try:
                    kind, payload = state.events.get_nowait()
                except queue.Empty:
                    break
                if kind == "line":
                    prefix = f"[{state.spec['label']}] " if len(self.active) > 1 else ""
                    self._console_add(prefix + payload)
                else:
                    done = payload
                    has_exit = True
            for ev in self._new_events(state):
                self._apply(ev, state)
            if has_exit:
                finished.append((state, done))

        for state, rc in finished:
            self._finish(rc, state)

        if self.active:
            oldest = min(self.active.values(), key=lambda s: s.started)
            el = int(time.monotonic() - oldest.started)
            self.detail.config(
                text=f"{len(self.active)} active · oldest {el // 60}m {el % 60:02d}s"
                     + (f" · part {oldest.parts[0]} of {oldest.parts[1]}"
                        if oldest.parts[1] else "")
                     + (f" · {oldest.eta}" if oldest.eta else ""))
            self.root.after(250, self._poll)
        else:
            self.polling = False
            self.cancelling = False
            self.cancel_btn.configure(state="disabled")
            self._refresh_queue()

    def _new_events(self, state=None):
        root = state.root if state else self.job
        p = root / "progress.jsonl" if root else None
        if not p or not p.is_file():
            return []
        out = []
        try:
            with p.open() as fh:
                fh.seek(state.offset if state else self.offset)
                for line in fh:
                    if line.endswith("\n") and line.strip():
                        try:
                            out.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
                if state:
                    state.offset = fh.tell()
                else:
                    self.offset = fh.tell()
        except Exception:
            pass
        return out

    def _apply(self, ev, state=None):
        e = ev.get("event")
        if e == "target_started":
            # The clipboard may hold SEVERAL paths, so a run is not always one
            # document. Without index/count the window shows only the current
            # title and a three-document run looks like a one-document run that
            # keeps restarting.
            n, of = ev.get("index", 1), ev.get("count", 1)
            where = f"[{n}/{of}] " if of > 1 else ""
            if state:
                state.target_titles[n] = ev.get("title", "")
            else:
                self.target_titles[n] = ev.get("title", "")
            self._set(f"{where}{ev.get('title', '')} — {ev.get('words', 0):,} words", "")
        elif e == "stage":
            self._set({"reader_view": "Reading the document",
                       "ledger": "Planning and verifying",
                       "compose": "Composing both readings",
                       "text_prep": "Cleaning and verifying the text",
                       "speechprep": "Reformatting for speech",
                       "publish": "Publishing",
                       "tts_normalize": "Normalizing for speech",
                       "corpus_preflight": "Preparing Corpus sources",
                       "corpus_inventory": "Building document inventories",
                       "corpus_plan": "Planning Corpus relationships",
                       "corpus_overview": "Writing the Corpus overview",
                       "corpus_seal": "Sealing Corpus evidence",
                       }.get(ev.get("name"), ev.get("name", "")), "")
        elif e == "part_failed":
            # The CAUSE of a later seal refusal. Without it the window shows
            # "no valid mechanical seal" and the reason lives only in stdout,
            # which nothing in this app reads.
            failures = state.part_failures if state else self.part_failures
            failures.append(
                f"part {ev.get('index')} ({ev.get('section', '?')}): "
                f"{ev.get('reason', 'planning failed')}")
        elif e == "part":
            parts = (ev.get("index", 0), ev.get("total", 0))
            if state:
                state.parts = parts
            self.parts = parts
            if parts[1]:
                # index is the part being STARTED, so completed is index-1.
                self.bar.config(maximum=parts[1], value=parts[0] - 1)
        elif e == "model_call_started":
            # Name the model. "plan001 running" says nothing about WHICH model is
            # being spent, which is the thing worth knowing on a metered account.
            call = f"{ev.get('role', '')} · {self._short(ev)} running"
            if state: state.call = call
            self.call = call
        elif e == "model_call":
            # No percentage: part and call durations vary too much for one to
            # mean anything, and there is no duration model to base it on.
            call = f"{ev.get('role', '')} · {self._short(ev)} {ev.get('outcome', '')}"
            if state: state.call = call
            self.call = call
        elif e == "eta":
            if state:
                state.eta = ev.get("text")
        elif e == "target_finished":
            outs = ev.get("outputs") or []
            if outs:
                self.out_dir = pathlib.Path(outs[0]).parent
                if state: state.out_dir = self.out_dir
                titles = state.target_titles if state else self.target_titles
                self.add_card(outs, titles.get(ev.get("index")))
            elif ev.get("status") == "failed":
                titles = state.target_titles if state else self.target_titles
                self.add_failure_card(
                    titles.get(ev.get("index"), f"Document {ev.get('index', '?')}"),
                    ev.get("failure_kind") or "input_failed")
        # AFTER the dispatch, never inside it. Spliced into the elif chain this
        # made `target_finished` chain off a truthy self.call, so the branch was
        # unreachable once any model call had run and Open Output Folder never
        # appeared. Valid Python, dead branch -- only a real replay found it.
        if self.call:
            base = self.status.cget("text").split("  ·  ")[0]
            self.status.config(text=f"{base}  ·  {self.call}")

    @staticmethod
    def _short(ev) -> str:
        """A model identifier short enough for a status line, without inventing
        a nickname: keep the last path segment and drop a redundant vendor."""
        m = (ev.get("model") or "").split("/")[-1]
        return m or ev.get("harness", "")

    # -- finished work -------------------------------------------------------
    def card_title(self, detailed: pathlib.Path) -> str:
        """The card's label. DETERMINISTIC, never model-written.

        The filename is free, instant, and cannot be wrong. A model-written
        title would be one more unsupported assertion in a project whose whole
        argument is that a summary must not claim more than its source does --
        and it would cost a call and could collide across a folder of twenty
        papers where the filenames already disambiguate.

        Clipboard text has no filename, so the title comes from the artifact's
        own opening words. Still deterministic, still nothing invented.
        """
        stem = detailed.stem
        for suffix in (".summary", ".brief", ".clean"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
        if stem and stem != "clipboard":
            return stem[:40]
        try:
            text = detailed.read_text(errors="replace")
        except Exception:
            return "pasted text"
        words = [w for line in text.splitlines()
                 if line.strip() and not line.lstrip().startswith(">")
                 for w in line.split()]
        title = " ".join(words[:7])[:40].rstrip(" ,.;:")
        return title or "pasted text"

    def add_card(self, outputs, label=None):
        detailed = pathlib.Path(outputs[0])
        title = label or self.card_title(detailed)
        try:
            n = len(detailed.read_text(errors="replace").split())
        except Exception:
            n = 0
        output_names = " + ".join(pathlib.Path(p).name for p in outputs)
        row = ttk.Frame(self.cards_frame)
        row.pack(fill="x", pady=(0, 4))
        ttk.Button(row, text=f"{title}  ·  {output_names}  ·  {n:,}w", width=52,
                   command=lambda p=detailed: reveal(p)).pack(side="left")
        ttk.Button(row, text="Copy",
                   command=lambda p=detailed: self.copy_out(p)).pack(side="left",
                                                                     padx=(6, 0))
        ttk.Button(row, text="Folder",
                   command=lambda p=detailed: reveal(p.parent)).pack(side="left",
                                                                     padx=(6, 0))
        self.cards.append(row)

    def add_failure_card(self, label, failure_kind):
        row = ttk.Frame(self.cards_frame)
        row.pack(fill="x", pady=(0, 4))
        text = f"{label}  ·  Failed: {KINDS.get(failure_kind, failure_kind)}"
        ttk.Label(row, text=text, foreground="#9a3d32").pack(side="left")
        self.cards.append(row)

    def copy_out(self, p: pathlib.Path):
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(p.read_text(errors="replace"))
            self._set(f"Copied {p.name} to the clipboard.", "")
        except Exception as e:
            self._set(f"Could not copy {p.name}.", str(e)[:100])

    def _finish(self, rc, state=None):
        self.bar.stop()
        self.bar.config(mode="determinate", maximum=1,
                        value=1 if rc == 0 else 0)
        self.proc = None
        if state:
            self.active.pop(state.id, None)
            self._refresh_queue()
            self.job = state.root
            self.part_failures = state.part_failures
            self.out_dir = state.out_dir
        self.running = bool(self.active)
        self.started = None
        evs = self._all_events(state)
        term = [e for e in evs if e.get("event") == "target_finished"]
        jobs = [e for e in evs if e.get("event") == "job_finished"]
        starts = [e for e in evs if e.get("event") == "job_started"]
        job = jobs[0] if len(jobs) == 1 else None
        # The engine's own verdict and the process exit must agree. When they do
        # not, one of them is lying and there is no basis for choosing which, so
        # neither is presented as the answer.
        expected_status = ("succeeded" if rc == 0 else
                           "cancelled" if rc == 130 else
                           "partial" if job and job.get("status") == "partial"
                           else "failed")
        record_error = None
        if len(jobs) > 1:
            record_error = "the engine wrote more than one terminal job record"
        elif job is not None and rc is not None and job.get("exit_code") != rc:
            record_error = (f"the engine reported exit {job.get('exit_code')} "
                            f"but the process exited {rc}")
        elif job is not None and rc is not None \
                and job.get("status") != expected_status:
            record_error = (f"the engine reported status {job.get('status')!r} "
                            f"but exit {rc} requires {expected_status!r}")
        elif job is not None and job.get("status") in {
                "succeeded", "partial", "cancelled"}:
            expected_targets = (starts[0].get("targets")
                                if len(starts) == 1 else None)
            indices = [e.get("index") for e in term]
            valid_indices = all(type(index) is int for index in indices)
            complete_records = (
                type(expected_targets) is int and expected_targets > 0
                and len(term) == expected_targets
                and valid_indices
                and sorted(indices) == list(range(1, expected_targets + 1))
                and len(set(indices)) == expected_targets)
            statuses = {e.get("status") for e in term}
            verdict_is_coherent = (
                job.get("status") == "succeeded"
                and all(e.get("status") == "succeeded"
                        and e.get("exit_code") == 0 for e in term)
                or job.get("status") == "partial"
                and bool(statuses & {"succeeded"})
                and statuses <= {"succeeded", "failed"}
                or job.get("status") == "cancelled"
                and "cancelled" in statuses
                and statuses <= {"succeeded", "failed", "cancelled"})
            if not complete_records or not verdict_is_coherent:
                record_error = ("the progress record is incomplete or has duplicate "
                                "or incoherent target terminal records")
        if record_error:
            self._set(f"Protocol error: {record_error}.",
                      "Treat it as unverified. See Details and the work folder.")
            self.open_work.pack(side="left", padx=(6, 0))
            # Keep draining and admitting unrelated queued work below.
        elif rc is None:
            self._set("Could not start the engine.", "See Details.")
        elif rc == 130 or (job and job.get("status") == "cancelled"):
            completed = [e for e in term if e.get("status") == "succeeded"]
            self._set(
                "Cancelled — active work stopped.",
                ("Completed targets remain published; no partial target was published."
                 if completed else
                 "Nothing from the cancelled target was published; previous files are unchanged."))
        elif job is not None and job.get("status") == "partial":
            succeeded = sum(e.get("status") == "succeeded" for e in term)
            failed = sum(e.get("status") == "failed" for e in term)
            self._set("Partial — independent document results completed.",
                      f"{succeeded} succeeded · {failed} failed. See each result card.")
            if self.out_dir:
                self.open_out.pack(side="left")
        elif rc == 0 and job is not None and term:
            selected = (mode_config.BY_LABEL.get(state.spec.get("mode"))
                        if state else None)
            done = selected.success_text if selected else "Done."
            output_lines = [str(path)
                            for target in term
                            for path in (target.get("outputs") or [])]
            self._set(done,
                      "\n".join(output_lines))
            if self.out_dir:
                self.open_out.pack(side="left")
        elif rc == 0:
            # Exit 0 with no terminal event is not a verified success.
            self._set("The engine exited cleanly, but its progress record is "
                      "incomplete.", "See Details.")
        else:
            # failure_kind is more specific than the exit code and is what the
            # engine actually decided. Exit 2 covers both a bad invocation and a
            # document that is simply too short to summarize, and telling a user
            # their document is an "invalid invocation" is a wrong answer.
            kind = term[-1].get("failure_kind") if term else None
            why = KINDS.get(kind) or EXITS.get(rc, "failed")
            unchanged = term and term[-1].get("destination_unchanged")
            # Lead with the CAUSE when one was reported. "no valid mechanical
            # seal" is the consequence of parts that never planned, and showing
            # only the consequence sends the user to a work folder to find out
            # what this line could have told them.
            detail = ("Nothing published; previous files unchanged."
                      if unchanged else "See Details.")
            if self.part_failures:
                detail = f"{self.part_failures[0]}" + (
                    f" (+{len(self.part_failures) - 1} more)"
                    if len(self.part_failures) > 1 else "")
            self._set(f"Failed: {why} (exit {rc}).", detail)
        self.open_work.pack(side="left", padx=(6, 0))
        # NEXT IN THE QUEUE. A failed job must not stop the queue: the remaining
        # documents are unrelated to whatever went wrong with this one, and
        # abandoning them silently is the partial result this project refuses.
        self._drain_pending()

    def _all_events(self, state=None):
        root = state.root if state else self.job
        p = root / "progress.jsonl"
        if not p.is_file():
            return []
        out = []
        for line in p.read_text(errors="replace").splitlines():
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return out

    # -- helpers -------------------------------------------------------------
    def _set(self, status, detail):
        self.status.config(text=status)
        if detail:
            self.detail.config(text=detail)

    def _console_clear(self):
        self.console.config(state="normal"); self.console.delete("1.0", "end")
        self.console.config(state="disabled")

    def _console_add(self, line):
        self.console.config(state="normal")
        self.console.insert("end", line + "\n")
        self.console.see("end")
        self.console.config(state="disabled")

    def close(self):
        if self.active or self.running:
            self._set("A job is running. Use Cancel active or leave the window open.",
                      "The window closes after the engine releases its locks.")
            return
        if self.pending:
            self._set("Documents are queued.",
                      "Press Start or remove the waiting jobs before closing.")
            return
        self.root.destroy()


KINDS = {"too_short": "the document is too short to summarize — read it instead",
         "document_limit": "the Corpus has too many or too few documents",
         "visible_word_limit": "the Corpus is outside the supported word range",
         "inventory_window_limit": "the Corpus needs too many inventory windows",
         "config_failed": "the selected route is not configured for this mode",
         "input_failed": "the input could not be read safely",
         "reader_view_failed": "the source could not be staged for summarization",
         "seal_missing": "no valid mechanical seal",
         "compose_failed": "the readings could not be composed",
         "text_prep_failed": "the cleaned text could not be verified",
         "artifact_missing": "a required output artifact was missing",
         "internal_error": "the engine stopped unexpectedly",
         "publish_failed": "the verified artifact could not be published"}

EXITS = {1: "the run did not complete",
         2: "invalid invocation, or an untrusted sealed state",
         3: "no valid mechanical seal",
         4: "source coverage incomplete",
         5: "a fidelity requirement could not be verified",
         6: "the planner could not meet the compression budget",
         7: "no substantive content survived"}


def main():
    root = tk.Tk()
    style = ttk.Style()
    try:
        style.theme_use("aqua" if sys.platform == "darwin" else "vista")
    except tk.TclError:
        pass
    try:
        App(root)
    except runtime.ConfigError as exc:
        messagebox.showerror(
            "summ'er — invalid local configuration", str(exc), parent=root)
        root.destroy()
        return 2
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
