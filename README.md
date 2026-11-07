<p align="center">
  <img src="src/assets/icon-256.png" alt="summ'er icon" width="128">
</p>

# summ'er

Create clear summaries of long documents in detailed and brief forms for
focused reading, e-ink, and use with text-to-speech.

Summ'er creates two Markdown files beside each source:

- `<name>.summary.md` — a detailed account of the argument, evidence, and qualifications.
- `<name>.brief.md` — a compact version for getting oriented quickly.

The summaries are written as continuous prose rather than stitched-together
extracts. Summ'er plans and checks source coverage, keeps important caveats
attached to the claims they qualify, and leaves the original document
unchanged. Any unresolved review findings are reported with the result.

## Quick start

Summ'er requires Python 3.9 or newer with Tkinter. PDF input also requires
`pdftotext`. Run the doctor first to check the local setup:

```console
python3 src/summ_cli.py --doctor
```

Summarize a document from the command line or open the desktop interface:

```console
python3 src/summ_cli.py document.pdf
python3 src/summ_ui.py
```

With no path, Summ'er reads the clipboard. It accepts a path, a folder, a list
of paths, or raw text. Folders are searched recursively for Markdown, text,
and PDF files.

Several documents are normally handled independently. To create one summary
of a collection, use corpus mode:

```console
python3 src/summ_cli.py --scope corpus chapter-1.pdf chapter-2.pdf
```

## Modes

- The default mode creates the Detailed and Brief summaries, then checks them
  for coverage and fidelity.
- `--quick` uses a shorter write-and-review pass when speed matters more.
- `--text-prep` cleans OCR or extracted text into `<name>.clean.md`.
- `--tts` prepares natural read-aloud text without summarizing it.

Long documents are divided in source order when they exceed a model's context
or response limit. The final summaries are still assembled as coherent prose,
not exposed as a collection of model responses.

## Model setup

Summ'er can use model CLIs available on `PATH` (`claude`, `agy`, `opencode`,
`grok`, `codex`, or `muse`) or a locally configured OpenAI-compatible gateway.
Model choices and role order live in [`src/models.json`](src/models.json).
Pipeline behavior does not depend on a particular model name.

The shared default is `Opus Plan` through OpenCode and the Anthropic API:
Opus plans, while Sonnet writes, reviews, and repairs. The desktop app remembers
the last selected profile; a CLI invocation with
no harness or profile uses the shared default. OpenCode v2 is required, and
its sessions stay inside the Summer work directory. Each OpenCode route uses
its API-key environment variable or the provider's credential saved by
`opencode auth login <provider>`, which Summer reads without starting OpenCode
on your own store. With a subscription login, such as ChatGPT for OpenAI,
Summer copies only the short-lived access token, never the refresh token, into
a private temporary directory that is deleted when the run ends, including
after cancellation. A hard-killed process or power loss can leave that
directory in the system temporary folder until it is cleaned. A missing login,
or one that would expire during a call, skips that harness in favor of the next
backup on a different harness; using OpenCode once with that provider, or
signing in again, renews it. OpenCode receives a fixed model catalog from
[`src/models.json`](src/models.json) instead of downloading one, so it
recognizes the configured models even when they are newer than its own release.
OpenCode supplies native Anthropic prompt caching; Summer retains cache reads,
cache writes, and reported cost in the run ledger. Reuse depends on an unchanged
prompt prefix within the cache lifetime.

A backend can be suspended by setting its `_enabled` flag to `false` in the
model catalog. Summer rejects its saved profiles and explicit routes before
dispatch, while preserving its model settings and credentials.

Muse offers Wonder with Contributor mode enabled through a Muse Code
subscription. Update the Muse CLI, sign in with `muse login`, and check
`muse usage` before choosing Wonder in the desktop role menus or running
`--harness muse`. Summer verifies the subscription through the CLI before each
Muse call and excludes `META_API_KEY` from that child process. A missing
subscription or a stored API credential skips Muse before any Muse model
turn; a backup on a different harness answers instead, and without one the run
stops with the login message.
Contributor mode permits Meta to use submitted content for training; see
[Muse Code subscriptions](https://dev.meta.ai/docs/muse-code/subscriptions).
Spark remains available through the subscription as well. Saved API credentials
are preserved for standalone use.

Gateway settings remain on the device and are not part of this repository:

- macOS: `~/Library/Application Support/summer/runtime.json`
- Windows: `%LOCALAPPDATA%\summer\runtime.json`
- either platform: the path in `SUMM_RUNTIME_CONFIG`

`--doctor` reports whether a gateway is configured. Gateway bearer credentials
come from the environment variable named in that configuration; authentication
for model CLIs is handled by each CLI. If all selected roles are local, Summ'er
will not silently fall back to a cloud model.

## Privacy and reliability

Summ'er supplies models with the text needed for the current task; it does not
give them a path to the source document. Each run stages the original source
and its normalized text in a work directory. In ordinary batch mode, an
implicit per-document work directory is removed after a successful summary or
text-preparation run, but is retained for diagnosis when either mode fails or
is cancelled. An implicit Corpus work root and implicit read-aloud work are
temporary and removed after every outcome. Passing `--work-dir DIR` retains the
work and review evidence in every mode.

The selected CLI, gateway, or model provider may have its own retention policy.
The device-local ETA history contains content-free operational metadata:
timestamps, route and outcome, source word and part counts, timing and capacity
measurements, and hashed execution and resource identities. It never contains
document paths, titles, prompts, source text, or generated prose.

Output files are replaced only after a complete result is ready. Cancelling
before that point leaves any previous output untouched.

Summary prose contains no tables, nested lists, first-person commentary,
citations, YAML, or process apparatus. A long Detailed summary may use a small
number of section headings. Exact quotations requested through custom
instructions are checked against the source.

## Command reference

```console
python3 src/summ_cli.py [--tts | --quick | --text-prep]
                        [--harness NAME] [--work-dir DIR]
                        [--progress-jsonl PATH]
                        [--instructions-file PATH]
                        [--selection-manifest PATH]
                        [--scope batch|corpus]
                        [--out DIR] [--profile NAME] [--local-only]
                        [paths ...]
```

The command-line interface defines the shared behavior on macOS and Windows;
`src/summ_ui.py` is a desktop client of the same interface. `src/make_app.py`
builds a macOS app that launches the checkout. Raycast and AutoHotkey shortcuts
can call `src/summ_cli.py` directly.

`--progress-jsonl PATH` records `summer.progress.v2` events, and `--cancel-file
PATH` enables cooperative cancellation. Exit 0 means that usable output was
published; unresolved semantic or editorial findings may still be disclosed in
the report and progress events. Nonzero codes identify terminal outcomes such
as no usable publication, invalid input, incomplete structural coverage,
unsupported context, or cancellation. Run `--help` for the current list.

Custom instructions are limited to 2,000 UTF-8 bytes and apply only to the
current run. The desktop interface may remember the last successfully used
instruction on that device. Direct command-line runs never inherit it.

## Tests

```console
python3 src/tests/test_suite.py
python3 src/epistemic_gate.py src/gates/real4.json OUT
python3 src/qualify.py CORPUS_DIR
```

On Windows, `python scripts\windows_proof.py` checks process locking and file
replacement. Automated gates cover structure and fidelity safeguards; they do
not decide whether a summary is pleasant to read.
