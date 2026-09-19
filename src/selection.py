"""Shared, deterministic document selection for Summer's CLI and desktop UI.

Selection is deliberately separate from execution.  It resolves user intent,
records exclusions and problems, freezes source metadata, and can be transported
as a small versioned manifest.  The model pipeline consumes staged text, never
these paths.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import os
import pathlib
import re
import unicodedata
from typing import Iterable


SCHEMA_V1 = "summer.selection.v1"
SCHEMA_V2 = "summer.selection.v2"
# The v1 name remains the compatibility/default name for ordinary Batch
# manifests. Corpus manifests opt into v2 explicitly.
SCHEMA = SCHEMA_V1
VALID_MODES = frozenset(("summarize", "quick", "text_prep", "tts"))
VALID_SCOPES = frozenset(("batch", "corpus"))
DOC_SUFFIXES = (".md", ".markdown", ".txt", ".pdf")
GENERATED_SUFFIXES = (".summary.md", ".brief.md", ".clean.md", ".tts.txt")
GENERATED_PREFIXES = ("summary.", "brief.", "clean.", "tts.")
_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:[\\/]")
_WINDOWS_RESERVED = frozenset(
    ("con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)),
     *(f"lpt{i}" for i in range(1, 10))))
# A new path in a pasted field starts only at whitespace followed by a root or
# a quote. Splitting on every space would shred Finder pathnames with spaces;
# shlex did that, and the tests exist because it shipped.
_PATH_BOUNDARY = re.compile(
    r"""
    \s+
    (?=
        " |
        ' |
        ~/ |
        \./ |
        \.\./ |
        / |
        \\\\ |
        [A-Za-z]:[\\/]
    )
    """,
    re.VERBOSE)


def _text(value) -> str:
    return unicodedata.normalize("NFC", str(value))


def _unquote(value: str) -> str:
    return str(value).strip().strip("'\"")


def _absolute(value) -> pathlib.Path:
    """Expand a user path without requiring it to exist."""
    raw = str(value) if isinstance(value, pathlib.Path) else _unquote(value)
    return pathlib.Path(os.path.expandvars(os.path.expanduser(raw))).absolute()


def parse_output_dir(value) -> pathlib.Path:
    """Expand ~ and env vars, keeping a relative --out relative to the source."""
    raw = str(value) if isinstance(value, pathlib.Path) else _unquote(value)
    raw = os.path.expandvars(os.path.expanduser(raw.strip()))
    if not raw:
        raise ValueError("output directory is empty")
    return pathlib.Path(raw)


def is_absolute_output(value) -> bool:
    """Treat a leading slash as absolute on both platforms.

    ``out`` and ``..`` are source-relative. ``/out`` is a filesystem root, not
    a subfolder named out beside the source.
    """
    path = value if isinstance(value, pathlib.Path) else parse_output_dir(value)
    text = str(path)
    return bool(path.is_absolute() or text.startswith("/") or text.startswith("\\")
                or _WINDOWS_DRIVE.match(text))


def resolve_output_directory(out_dir, anchor: pathlib.Path) -> pathlib.Path:
    """Join a relative output directory to the source (or Downloads) folder."""
    specified = out_dir if isinstance(out_dir, pathlib.Path) else parse_output_dir(out_dir)
    if is_absolute_output(specified):
        return _absolute(specified)
    return _resolved(pathlib.Path(anchor) / specified)


def clipboard_output_directory(out_dir=None) -> pathlib.Path:
    """Pasted text has no source folder; relative --out is under Downloads."""
    anchor = pathlib.Path.home() / "Downloads"
    if out_dir is None:
        return _resolved(anchor)
    return resolve_output_directory(out_dir, anchor)


def _resolved(path: pathlib.Path) -> pathlib.Path:
    return path.resolve(strict=False)


def _identity(path: pathlib.Path) -> str:
    # normcase is meaningful on Windows and on case-insensitive macOS volumes;
    # it is harmless on case-sensitive systems.  NFC keeps equivalent Unicode
    # names in the same deterministic identity domain.
    return os.path.normcase(_text(str(_resolved(path))))


def _relative(root: pathlib.Path, path: pathlib.Path) -> str:
    try:
        value = path.relative_to(root)
    except ValueError:
        value = pathlib.Path(path.name)
    return _text(value.as_posix())


def _sort_key(relative: str):
    exact = _text(relative).replace("\\", "/")
    return exact.casefold(), exact


def _inside(root: pathlib.Path, path: pathlib.Path) -> bool:
    try:
        common = os.path.commonpath([str(_resolved(root)), str(_resolved(path))])
        return os.path.normcase(common) == os.path.normcase(str(_resolved(root)))
    except (OSError, ValueError):
        return False


def _is_junction(path: pathlib.Path) -> bool:
    method = getattr(path, "is_junction", None)
    try:
        return bool(method()) if method else False
    except OSError:
        return False


def _is_link(path: pathlib.Path) -> bool:
    try:
        return path.is_symlink() or _is_junction(path)
    except OSError:
        return False


def _exists_or_link(path: pathlib.Path) -> bool:
    """Probe a possible path without letting arbitrary pasted prose crash."""
    try:
        return path.exists() or _is_link(path)
    except OSError:
        return False


def _is_hidden(relative: str) -> bool:
    return any(part.startswith(".") for part in pathlib.PurePosixPath(relative).parts)


def _is_generated(name: str, mode_key: str) -> bool:
    lower = name.casefold()
    if mode_key == "tts":
        return lower.endswith(".tts.txt") or lower.startswith("tts.")
    return (lower.endswith(GENERATED_SUFFIXES) or
            any(lower.startswith(p) for p in GENERATED_PREFIXES))


def _record(kind: str, path: pathlib.Path | str, *, root_id: str | None = None,
            detail: str = "", severity: str = "info", resolved_path=None) -> dict:
    rec = {"kind": kind, "path": _text(path), "severity": severity}
    if root_id:
        rec["root_id"] = root_id
    if resolved_path is not None:
        rec["resolved_path"] = _text(resolved_path)
    if detail:
        rec["detail"] = _text(detail)
    return rec


@dataclasses.dataclass(frozen=True)
class SelectionRoot:
    id: str
    kind: str
    selected_path: pathlib.Path
    resolved_path: pathlib.Path
    display_label: str

    def manifest(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "path": _text(self.selected_path),
            "resolved_path": _text(self.resolved_path),
            "label": self.display_label,
        }


@dataclasses.dataclass(frozen=True)
class DocumentTarget:
    id: str
    root_id: str
    source_path: pathlib.Path
    resolved_path: pathlib.Path
    relative_path: str
    source_suffix: str
    size_bytes: int | None
    source_sha256: str | None
    error_kind: str | None = None
    error_detail: str | None = None
    planned_outputs: tuple[pathlib.Path, ...] = ()

    def manifest(self) -> dict:
        value = {
            "id": self.id,
            "root_id": self.root_id,
            "path": _text(self.source_path),
            "resolved_path": _text(self.resolved_path),
            "relative_path": self.relative_path,
            "source_suffix": self.source_suffix,
            "size_bytes": self.size_bytes,
            "source_sha256": self.source_sha256,
        }
        if self.error_kind:
            value["error_kind"] = self.error_kind
        if self.error_detail:
            value["error_detail"] = self.error_detail
        if self.planned_outputs:
            value["planned_outputs"] = [_text(p) for p in self.planned_outputs]
        return value


@dataclasses.dataclass(frozen=True)
class Selection:
    kind: str
    roots: tuple[SelectionRoot, ...] = ()
    documents: tuple[DocumentTarget, ...] = ()
    exclusions: tuple[dict, ...] = ()
    errors: tuple[dict, ...] = ()
    raw_text: str | None = None
    order_digest: str = ""
    mode_key: str = "summarize"
    scope: str = "batch"
    corpus_name: str | None = None
    corpus_outputs: tuple[pathlib.Path, ...] = ()
    output_plan_sha256: str | None = None
    manifest_schema: str = SCHEMA_V1

    @property
    def is_text(self) -> bool:
        return self.kind == "text"

    @property
    def has_problems(self) -> bool:
        return bool(self.errors)

    @property
    def execution_errors(self) -> tuple[dict, ...]:
        """Return selection problems that do not already have a document target.

        A supported file with invalid UTF-8 or unreadable bytes remains in
        ``documents`` so Batch can terminalize that document exactly once. Its
        discovery error must therefore not also become a synthetic ``E###``
        target. Missing paths, empty folders, and traversal/link problems have
        no document target and do need their own terminal record.
        """
        represented = {
            (doc.error_kind, _identity(doc.source_path))
            for doc in self.documents if doc.error_kind
        }
        result = []
        for error in self.errors:
            key = (error.get("kind"),
                   _identity(_absolute(error["path"]))
                   if error.get("path") else None)
            if key not in represented:
                result.append(error)
        return tuple(result)

    @property
    def summary(self) -> str:
        if self.is_text:
            return f"pasted text — {len((self.raw_text or '').split()):,} words"
        ignored = f" · {len(self.exclusions)} ignored" if self.exclusions else ""
        problems = f" · {len(self.errors)} problem" + ("s" if len(self.errors) != 1 else "") \
            if self.errors else ""
        return (f"{len(self.roots)} root" + ("s" if len(self.roots) != 1 else "")
                + f" → {len(self.documents)} document"
                + ("s" if len(self.documents) != 1 else "")
                + ignored + problems)

    def with_outputs(self, mode_key: str, out_dir=None,
                     affix_kind: str = "prefix", affix_text: str = "",
                     affix_secondary: str = "") -> "Selection":
        if self.is_text:
            return dataclasses.replace(self, mode_key=mode_key)
        outputs = plan_outputs(self, mode_key, out_dir, affix_kind=affix_kind,
                               affix_text=affix_text, affix_secondary=affix_secondary)
        by_id = outputs
        docs = tuple(dataclasses.replace(doc, planned_outputs=by_id.get(doc.id, ()))
                     for doc in self.documents)
        return dataclasses.replace(self, documents=docs, mode_key=mode_key)

    def with_corpus_outputs(self, name=None, out_dir=None) -> "Selection":
        """Freeze the one logical Corpus pair independently of document outputs."""
        if self.is_text:
            raise ValueError("Corpus requires path-backed selection")
        name = name or default_corpus_name(self)
        outputs = plan_corpus_outputs(self, name, out_dir)
        return dataclasses.replace(self, scope="corpus", corpus_name=name,
                                   corpus_outputs=tuple(outputs),
                                   manifest_schema=SCHEMA_V2,
                                   output_plan_sha256=output_plan_digest(
                                       dataclasses.replace(self, scope="corpus",
                                                            corpus_name=name,
                                                            corpus_outputs=tuple(outputs),
                                                            manifest_schema=SCHEMA_V2)))

    def for_single_document(self, doc: DocumentTarget) -> "Selection":
        """Seal one Batch job for one already-admitted document.

        A multi-document paste is one user selection. Each queue job is its
        own sealed selection: the order digest is the documents in *this*
        job, not the combined paste that produced it.
        """
        if self.is_text:
            raise ValueError("pasted text cannot be sliced into path documents")
        if doc not in self.documents:
            raise ValueError("document is not in this selection")
        roots = tuple(root for root in self.roots if root.id == doc.root_id)
        if not roots:
            raise ValueError("document root is not in this selection")
        documents = (doc,)
        return dataclasses.replace(
            self, roots=roots, documents=documents, exclusions=(), errors=(),
            scope="batch", corpus_name=None, corpus_outputs=(),
            output_plan_sha256=None, manifest_schema=SCHEMA_V1,
            order_digest=_document_order_digest(documents))

    def manifest(self) -> dict:
        value = {
            "schema": SCHEMA_V2 if self.scope == "corpus" else self.manifest_schema,
            "kind": self.kind,
            "mode": self.mode_key,
            "scope": self.scope,
            "roots": [root.manifest() for root in self.roots],
            "documents": [doc.manifest() for doc in self.documents],
            "exclusions": list(self.exclusions),
            "errors": list(self.errors),
            "order_digest": self.order_digest,
        }
        if self.output_plan_sha256 or self.corpus_outputs:
            value["output_plan_sha256"] = self.output_plan_sha256 or output_plan_digest(self)
        if self.scope == "corpus":
            value["corpus"] = {
                "name": self.corpus_name or default_corpus_name(self),
                "outputs": [_text(p) for p in self.corpus_outputs],
                "output_plan_sha256": self.output_plan_sha256 or output_plan_digest(self),
            }
        if self.raw_text is not None:
            # Text normally travels through --text-file so this field is not
            # used by the model path.  Keeping it here makes the manifest a
            # complete local snapshot for reviews and tests.
            value["raw_text"] = self.raw_text
        return value


def _sha256(path: pathlib.Path) -> tuple[int, str]:
    size = 0
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(1024 * 1024)
            if not block:
                break
            size += len(block)
            digest.update(block)
    return size, digest.hexdigest()


def _document(path: pathlib.Path, root: SelectionRoot, relative: str,
              exclusions: list, errors: list) -> DocumentTarget:
    selected = _absolute(path)
    resolved = _resolved(selected)
    suffix = selected.suffix.casefold()
    size = None
    digest = None
    error_kind = error_detail = None
    try:
        size, digest = _sha256(selected)
        if suffix in (".md", ".markdown", ".txt"):
            # Validate now so a queued selection never silently turns into a
            # different encoding failure after the user presses Start.
            selected.read_bytes().decode("utf-8")
    except UnicodeDecodeError as exc:
        error_kind, error_detail = "invalid_utf8", str(exc)
        errors.append(_record(error_kind, selected, root_id=root.id,
                              detail=str(exc), severity="error"))
    except OSError as exc:
        error_kind, error_detail = "unreadable_file", str(exc)
        errors.append(_record(error_kind, selected, root_id=root.id,
                              detail=str(exc), severity="error"))
        try:
            size = selected.stat().st_size
        except OSError:
            pass
    return DocumentTarget(
        id="", root_id=root.id, source_path=selected, resolved_path=resolved,
        relative_path=relative, source_suffix=suffix, size_bytes=size,
        source_sha256=digest, error_kind=error_kind, error_detail=error_detail)


def _candidate(path: pathlib.Path, root: SelectionRoot, mode_key: str,
               exclusions: list, errors: list) -> DocumentTarget | None:
    relative = _relative(root.resolved_path, _absolute(path))
    if _is_hidden(relative):
        exclusions.append(_record("hidden", path, root_id=root.id,
                                  detail="root-relative path contains a hidden component"))
        return None
    lower = path.name.casefold()
    if path.suffix.casefold() not in DOC_SUFFIXES:
        exclusions.append(_record("unsupported_extension", path, root_id=root.id,
                                  detail="supported: .md, .markdown, .txt, .pdf"))
        return None
    if _is_generated(path.name, mode_key):
        exclusions.append(_record("generated_artifact", path, root_id=root.id,
                                  detail=f"excluded for mode {mode_key}"))
        return None
    if _is_link(path):
        resolved = _resolved(path)
        if not resolved.exists():
            errors.append(_record("broken_link", path, root_id=root.id,
                                  severity="error"))
            return None
        if not resolved.is_file():
            errors.append(_record("link_not_file", path, root_id=root.id,
                                  severity="error", resolved_path=resolved))
            return None
        if not _inside(root.resolved_path, resolved):
            errors.append(_record("outside_root_link", path, root_id=root.id,
                                  severity="error", resolved_path=resolved))
            return None
    elif not path.is_file():
        # Keep an unreadable or disappearing supported entry as a document
        # failure. Batch can then give it a terminal target record instead of
        # silently dropping it from the announced document set.
        return _document(path, root, relative, exclusions, errors)
    return _document(path, root, relative, exclusions, errors)


def _walk_folder(root: SelectionRoot, mode_key: str, exclusions: list,
                 errors: list) -> list[DocumentTarget]:
    found = []

    def onerror(exc):
        path = getattr(exc, "filename", None) or root.resolved_path
        errors.append(_record("incomplete_root", path, root_id=root.id,
                              detail=str(exc), severity="error"))

    try:
        walker = os.walk(str(root.resolved_path), topdown=True,
                         followlinks=False, onerror=onerror)
        for current, dirs, files in walker:
            current_path = pathlib.Path(current)
            # Prune hidden folders and links before os.walk can descend. The
            # explicit record keeps Review useful without following a loop.
            kept_dirs = []
            for name in sorted(dirs, key=lambda n: _sort_key(
                    _relative(root.resolved_path, current_path / n))):
                path = current_path / name
                relative = _relative(root.resolved_path, path)
                if _is_hidden(relative):
                    exclusions.append(_record("hidden", path, root_id=root.id,
                                              detail="hidden directory pruned"))
                elif _is_link(path):
                    exclusions.append(_record("non_followed_link", path,
                                              root_id=root.id,
                                              detail="directory links are not followed"))
                else:
                    kept_dirs.append(name)
            dirs[:] = kept_dirs
            for name in sorted(files, key=lambda n: _sort_key(
                    _relative(root.resolved_path, current_path / n))):
                item = _candidate(current_path / name, root, mode_key,
                                  exclusions, errors)
                if item is not None:
                    found.append(item)
    except OSError as exc:
        onerror(exc)
    if not found and not any(e.get("root_id") == root.id for e in errors):
        errors.append(_record("no_supported_documents", root.resolved_path,
                              root_id=root.id,
                              detail="folder contains no usable supported documents",
                              severity="error"))
    return sorted(found, key=lambda item: _sort_key(item.relative_path))


def resolve_paths(values: Iterable[str | pathlib.Path], mode_key: str = "summarize") -> Selection:
    """Resolve explicit files/folders in the order supplied by the user."""
    roots = []
    exclusions = []
    errors = []
    candidates = []
    seen_roots = set()
    for number, value in enumerate(values, 1):
        selected = _absolute(value)
        if not selected.exists() and not _is_link(selected):
            errors.append(_record("missing_input", selected,
                                  detail="explicit path does not exist",
                                  severity="error"))
            continue
        resolved = _resolved(selected)
        identity = _identity(resolved)
        if identity in seen_roots:
            exclusions.append(_record("duplicate_root", selected,
                                      detail="same resolved root was already selected"))
            continue
        seen_roots.add(identity)
        if _is_link(selected) and resolved.is_dir():
            errors.append(_record("non_followed_link", selected,
                                  detail="explicit directory links are not followed",
                                  resolved_path=resolved, severity="error"))
            continue
        if selected.is_dir():
            kind = "folder"
        elif selected.is_file() or _is_link(selected):
            kind = "file"
        else:
            errors.append(_record("unsupported_input", selected,
                                  detail="path is neither a regular file nor folder",
                                  severity="error"))
            continue
        root = SelectionRoot(
            id=f"R{len(roots) + 1:03d}", kind=kind,
            selected_path=selected, resolved_path=resolved,
            display_label=selected.name or str(selected))
        roots.append(root)
        if kind == "folder":
            candidates.extend(_walk_folder(root, mode_key, exclusions, errors))
        else:
            if selected.suffix.casefold() not in DOC_SUFFIXES:
                errors.append(_record("unsupported_input", selected, root_id=root.id,
                                      detail="supported: .md, .markdown, .txt, .pdf",
                                      severity="error"))
                continue
            # Explicit selection overrides the generated-artifact exclusion.
            if _is_generated(selected.name, mode_key):
                exclusions.append(_record("generated_artifact_explicit", selected,
                                          root_id=root.id,
                                          detail="explicit file selection overrides mode filter"))
            if _is_link(selected):
                if not resolved.exists() or not resolved.is_file():
                    errors.append(_record("broken_link", selected, root_id=root.id,
                                          severity="error"))
                    continue
            candidates.append(_document(selected, root, selected.name,
                                        exclusions, errors))

    # Root order is meaningful. Folder candidates are already sorted by the
    # walk's root-relative identity; the final pass gives the same tie-break
    # semantics to explicit files and overlapping roots.
    unique = []
    seen_documents = set()
    for item in candidates:
        identity = _identity(item.resolved_path)
        if identity in seen_documents:
            exclusions.append(_record("duplicate_document", item.source_path,
                                      root_id=item.root_id,
                                      detail="first canonical path wins",
                                      resolved_path=item.resolved_path))
            continue
        seen_documents.add(identity)
        unique.append(item)
    documents = tuple(dataclasses.replace(item, id=f"D{index:03d}")
                      for index, item in enumerate(unique, 1))
    return Selection("paths", tuple(roots), documents, tuple(exclusions),
                     tuple(errors), None, _document_order_digest(documents),
                     mode_key, "batch")


def combine_path_selections(values: Iterable[Selection],
                            mode_key: str = "summarize") -> Selection:
    """Combine frozen path selections without reading their sources again.

    Queue scope is editable until Start. Batch jobs each hold a one-document
    selection, while Corpus needs one manifest containing every waiting
    document. Reindex roots and documents so selections admitted by separate
    Add actions cannot collide on their local R001/D001 identifiers.
    """
    if mode_key not in VALID_MODES:
        raise ValueError(f"invalid selection mode: {mode_key}")
    items = tuple(values)
    if not items or any(item.is_text for item in items):
        raise ValueError("only path-backed selections can be combined")

    roots = []
    documents = []
    exclusions = []
    errors = []
    root_by_identity = {}
    seen_documents = set()

    for item in items:
        root_remap = {}
        for root in item.roots:
            identity = (root.kind, _identity(root.resolved_path))
            new_root = root_by_identity.get(identity)
            if new_root is None:
                new_root = dataclasses.replace(root, id=f"R{len(roots) + 1:03d}")
                root_by_identity[identity] = new_root
                roots.append(new_root)
            root_remap[root.id] = new_root.id

        def remap_record(record):
            value = dict(record)
            if value.get("root_id") in root_remap:
                value["root_id"] = root_remap[value["root_id"]]
            return value

        exclusions.extend(remap_record(record) for record in item.exclusions)
        errors.extend(remap_record(record) for record in item.errors)
        for doc in item.documents:
            identity = _identity(doc.resolved_path)
            if identity in seen_documents:
                exclusions.append(_record(
                    "duplicate_document", doc.source_path,
                    root_id=root_remap.get(doc.root_id),
                    detail="first queued document wins",
                    resolved_path=doc.resolved_path))
                continue
            seen_documents.add(identity)
            documents.append(dataclasses.replace(
                doc, id=f"D{len(documents) + 1:03d}",
                root_id=root_remap[doc.root_id], planned_outputs=()))

    frozen_documents = tuple(documents)
    return Selection(
        "paths", tuple(roots), frozen_documents, tuple(exclusions), tuple(errors),
        None, _document_order_digest(frozen_documents), mode_key, "batch")


def text_selection(raw: str, mode_key: str = "summarize") -> Selection:
    return Selection("text", raw_text=raw, mode_key=mode_key,
                     order_digest=hashlib.sha256(raw.encode("utf-8")).hexdigest())


def split_path_list(raw: str) -> list[str]:
    """Split a pasted path field into individual path strings.

    Newlines remain the primary separator. On one line, a new path starts only
    at whitespace followed by a path root or a quote, so an unquoted path with
    spaces stays one value. If the whole line already names an existing file or
    folder, it is kept intact.
    """
    text = raw.strip()
    if not text:
        return []
    parts = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts.extend(_split_path_line(line))
    return parts


def _split_path_line(line: str) -> list[str]:
    whole = _absolute(line)
    if _exists_or_link(whole):
        return [_unquote(line)]
    return [_unquote(part) for part in _PATH_BOUNDARY.split(line) if part.strip()]


def looks_like_path(value: str) -> bool:
    """Conservative clipboard path test; prose is never token-filtered."""
    raw = _unquote(value)
    if not raw:
        return False
    if "://" in raw:
        return False
    if raw.startswith(("~/", "./", "../", "/", "\\\\")):
        return True
    if _WINDOWS_DRIVE.match(raw):
        return True
    if "/" in raw or "\\" in raw:
        return True
    lower = pathlib.PurePath(raw).name.casefold()
    return (lower.endswith(DOC_SUFFIXES) or lower.endswith(GENERATED_SUFFIXES) or
            any(lower.startswith(p) for p in GENERATED_PREFIXES))


def classify_clipboard(raw: str, mode_key: str = "summarize") -> Selection:
    """Classify all clipboard text or none of it as paths.

    A set of paths is accepted only when every token has the conservative path
    shape. This prevents the old silent omission where a missing path was
    discarded and the remaining existing path was processed. Tokens may be
    line-separated or, on one line, adjacent absolute, home, or drive roots.
    """
    if not clipboard_path_intent(raw):
        return text_selection(raw, mode_key)
    paths = split_path_list(raw)
    if not paths:
        return text_selection(raw, mode_key)
    return resolve_paths(paths, mode_key)


def clipboard_path_intent(raw: str) -> bool:
    """Return the classifier's path/text decision without reading file bytes."""
    paths = split_path_list(raw)
    if not paths:
        return False
    if len(paths) == 1:
        value = paths[0]
        selected = _absolute(value)
        return _exists_or_link(selected) or looks_like_path(value)
    return all(looks_like_path(path) for path in paths)


def verify_document(doc: DocumentTarget) -> tuple[bool, str]:
    """Verify the accepted source bytes immediately before staging."""
    if doc.error_kind:
        return False, doc.error_detail or doc.error_kind
    try:
        if _identity(doc.source_path) != _identity(doc.resolved_path):
            return False, "source path now resolves to a different file"
        size, digest = _sha256(doc.source_path)
    except OSError as exc:
        return False, str(exc)
    if doc.size_bytes != size or doc.source_sha256 != digest:
        return False, "source bytes changed after selection was accepted"
    return True, ""


def _unique_root_labels(roots: tuple[SelectionRoot, ...]) -> dict[str, str]:
    result = {}
    used = set()
    for root in roots:
        label = root.display_label or "root"
        candidate = label
        suffix = 2
        while candidate.casefold() in used:
            candidate = f"{label}-{suffix}"
            suffix += 1
        used.add(candidate.casefold())
        result[root.id] = candidate
    return result


def _planned_base(path: pathlib.Path, taken: set, artifact_suffix: str) -> pathlib.Path:
    """Choose a collision-free base without splitting a multi-dot suffix."""
    candidate = path
    n = 2
    key = lambda p: os.path.normcase(str(p))
    name = path.name
    if name.casefold().endswith(artifact_suffix.casefold()):
        base = name[:-len(artifact_suffix)]
    else:
        base = path.stem
    while key(candidate) in taken:
        candidate = path.with_name(f"{base}-{n}{artifact_suffix}")
        n += 1
    taken.add(key(candidate))
    return candidate


def format_output_filenames(
    stem: str, mode_key: str, affix_kind: str = "prefix", affix_text: str = "",
    affix_secondary: str = "",
) -> tuple[str, ...]:
    """Format one or two output artifact filenames for a document stem."""
    ext = ".txt" if mode_key == "tts" else ".md"
    clean_custom = affix_text.strip().strip(".") if affix_text else ""
    clean_secondary = affix_secondary.strip().strip(".") if affix_secondary else ""
    if mode_key in ("summarize", "quick"):
        if clean_custom or clean_secondary:
            primary_tag = clean_custom or "summary"
            if clean_secondary:
                secondary_tag = clean_secondary
            else:
                secondary_tag = (
                    f"brief.{clean_custom}" if affix_kind == "prefix" and clean_custom != "summary"
                    else (f"{clean_custom}.brief" if affix_kind == "suffix" and clean_custom != "summary"
                          else "brief"))
        else:
            primary_tag = "summary"
            secondary_tag = "brief"
        if affix_kind == "prefix":
            return (f"{primary_tag}.{stem}{ext}", f"{secondary_tag}.{stem}{ext}")
        else:
            return (f"{stem}.{primary_tag}{ext}", f"{stem}.{secondary_tag}{ext}")
    elif mode_key == "text_prep":
        tag = clean_custom or "clean"
        if affix_kind == "prefix":
            return (f"{tag}.{stem}{ext}",)
        else:
            return (f"{stem}.{tag}{ext}",)
    elif mode_key == "tts":
        tag = clean_custom or "tts"
        if affix_kind == "prefix":
            return (f"{tag}.{stem}{ext}",)
        else:
            return (f"{stem}.{tag}{ext}",)
    else:
        tag = clean_custom or "summary"
        sec = clean_secondary or "brief"
        if affix_kind == "prefix":
            return (f"{tag}.{stem}{ext}", f"{sec}.{stem}{ext}")
        else:
            return (f"{stem}.{tag}{ext}", f"{stem}.{sec}{ext}")


def _plan_document_filenames(
    stem: str, mode_key: str, directory: pathlib.Path, taken: set,
    affix_kind: str = "prefix", affix_text: str = "", affix_secondary: str = ""
) -> tuple[pathlib.Path, ...]:
    key = lambda p: os.path.normcase(str(p))
    candidate_stem = stem
    n = 2
    while True:
        names = format_output_filenames(
            candidate_stem, mode_key, affix_kind=affix_kind, affix_text=affix_text,
            affix_secondary=affix_secondary)
        paths = tuple(directory / name for name in names)
        if not any(key(p) in taken for p in paths):
            for p in paths:
                taken.add(key(p))
            return paths
        candidate_stem = f"{stem}-{n}"
        n += 1


def plan_outputs(selection: Selection, mode_key: str, out_dir=None,
                 affix_kind: str = "prefix", affix_text: str = "",
                 affix_secondary: str = "") -> dict[str, tuple[pathlib.Path, ...]]:
    """Plan every destination before model work, preserving pair bases."""
    if selection.is_text:
        return {}
    roots = {root.id: root for root in selection.roots}
    labels = _unique_root_labels(selection.roots)
    folder_roots = [root for root in selection.roots if root.kind == "folder"]
    all_file_roots = all(root.kind == "file" for root in selection.roots)
    common_parent = None
    if all_file_roots and selection.roots:
        parents = {str(root.selected_path.parent) for root in selection.roots}
        if len(parents) == 1:
            common_parent = selection.roots[0].selected_path.parent
    taken = set()
    result = {}
    for doc in selection.documents:
        root = roots[doc.root_id]
        if out_dir is None:
            directory = doc.source_path.parent
        else:
            specified = parse_output_dir(out_dir)
            if not is_absolute_output(specified):
                directory = resolve_output_directory(specified, doc.source_path.parent)
            else:
                output_root = _absolute(specified)
                if len(selection.roots) == 1 and root.kind == "folder":
                    directory = output_root / pathlib.PurePosixPath(doc.relative_path).parent
                elif len(selection.roots) == 1 and root.kind == "file":
                    directory = output_root
                elif common_parent is not None:
                    directory = output_root
                else:
                    directory = output_root / labels[root.id] / pathlib.PurePosixPath(
                        doc.relative_path).parent
        planned = _plan_document_filenames(
            doc.source_path.stem, mode_key, directory, taken,
            affix_kind=affix_kind, affix_text=affix_text,
            affix_secondary=affix_secondary)
        result[doc.id] = planned
    return result


def _portable_corpus_stem(name: str) -> str:
    """Validate a display name and return a Windows-safe artifact stem."""
    if not isinstance(name, str):
        raise ValueError("Corpus name must be text")
    value = _text(name).strip()
    if not value or len(value) > 80:
        raise ValueError("Corpus name must contain 1–80 characters")
    if value in {".", ".."} or any(ord(ch) < 32 for ch in value):
        raise ValueError("Corpus name contains a control character or reserved name")
    if any(sep in value for sep in ("/", "\\")):
        raise ValueError("Corpus name must not contain a path separator")
    if value.endswith((".", " ")):
        raise ValueError("Corpus name must not end with a dot or space")
    stem = re.sub(r'[<>:"/\\|?*]+', "-", value).strip(" .")
    stem = re.sub(r"-+", "-", stem)
    if not stem or stem.casefold() in _WINDOWS_RESERVED:
        raise ValueError("Corpus name is not a portable filename")
    return stem


def default_corpus_name(selection: Selection) -> str:
    if len(selection.roots) == 1 and selection.roots[0].kind == "folder":
        return selection.roots[0].display_label or selection.roots[0].resolved_path.name
    return "corpus-" + dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def plan_corpus_outputs(selection: Selection, name: str, out_dir=None) -> tuple[pathlib.Path, pathlib.Path]:
    """Plan and validate the single pair published by a Corpus job."""
    if selection.is_text or len(selection.documents) < 2:
        raise ValueError("Corpus requires at least two path-backed documents")
    stem = _portable_corpus_stem(name)
    if out_dir is not None and is_absolute_output(out_dir):
        directory = resolve_output_directory(out_dir, pathlib.Path.home())
    elif len(selection.roots) == 1 and selection.roots[0].kind == "folder":
        directory = selection.roots[0].selected_path
    else:
        parents = {str(doc.source_path.parent) for doc in selection.documents}
        directory = pathlib.Path(next(iter(parents))) if len(parents) == 1 \
            else pathlib.Path.home() / "Downloads"
    if out_dir is not None and not is_absolute_output(out_dir):
        directory = resolve_output_directory(out_dir, directory)
    outputs = (directory / f"{stem}.corpus.summary.md",
               directory / f"{stem}.corpus.brief.md")
    validate_output_pair(outputs, (".corpus.summary.md", ".corpus.brief.md"),
                         (doc.source_path for doc in selection.documents))
    return outputs


def _output_plan_value(selection: Selection) -> dict:
    return {
        "mode": selection.mode_key,
        "scope": selection.scope,
        "order_digest": selection.order_digest,
        "documents": [
            {"id": doc.id, "root_id": doc.root_id,
             "relative_path": doc.relative_path,
             "outputs": [_text(p) for p in doc.planned_outputs]}
            for doc in selection.documents
        ],
        "corpus_outputs": [_text(p) for p in selection.corpus_outputs],
    }


def output_plan_digest(selection: Selection) -> str:
    raw = json.dumps(_output_plan_value(selection), ensure_ascii=False,
                     sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def validate_output_pair(outputs, suffixes=None, sources=()) -> tuple[pathlib.Path, ...]:
    """Reject malformed or source-colliding artifact pairs before model work."""
    values = tuple(_absolute(p) for p in outputs)
    if suffixes is not None and (len(values) != len(suffixes) or len(values) < 1):
        raise ValueError("output plan has the wrong number of artifacts")
    if len(values) < 1 or len(values) > 2:
        raise ValueError("output plan has the wrong number of artifacts")
    if len(values) == 2:
        first, second = values
        if first == second or first.parent != second.parent:
            raise ValueError("output pair must be distinct and share one directory")
        f_name = first.name.casefold()
        s_name = second.name.casefold()
        if suffixes and all(isinstance(s, str) for s in suffixes) and len(suffixes) == 2:
            s1, s2 = suffixes[0].casefold(), suffixes[1].casefold()
            if f_name.endswith(s1) and s_name.endswith(s2):
                b1, b2 = f_name[:-len(s1)], s_name[:-len(s2)]
                if not b1 or b1 != b2:
                    raise ValueError("output pair must share one base name")
            elif f_name.startswith("summary.") and s_name.startswith("brief."):
                b1, b2 = f_name[len("summary."):], s_name[len("brief."):]
                if not b1 or b1 != b2:
                    raise ValueError("output pair must share one base name")
        else:
            if f_name.startswith("summary.") and s_name.startswith("brief."):
                b1, b2 = f_name[len("summary."):], s_name[len("brief."):]
                if not b1 or b1 != b2:
                    raise ValueError("output pair must share one base name")
            elif f_name.endswith(".summary.md") and s_name.endswith(".brief.md"):
                b1, b2 = f_name[:-len(".summary.md")], s_name[:-len(".brief.md")]
                if not b1 or b1 != b2:
                    raise ValueError("output pair must share one base name")
            elif ".corpus.summary.md" in f_name and ".corpus.brief.md" in s_name:
                b1 = f_name[:-len(".corpus.summary.md")]
                b2 = s_name[:-len(".corpus.brief.md")]
                if not b1 or b1 != b2:
                    raise ValueError("output pair must share one base name")
    source_ids = {_identity(_absolute(path)) for path in sources}
    if source_ids & {_identity(path) for path in values}:
        raise ValueError("output plan collides with a source path")
    return values


def _document_order_digest(documents: Iterable[DocumentTarget]) -> str:
    """Hash the sealed document order, independent of planned outputs."""
    material = "\n".join(
        f"{doc.root_id}\0{doc.relative_path}\0{_text(doc.resolved_path)}"
        for doc in documents)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def write_manifest(selection: Selection, path: pathlib.Path) -> pathlib.Path:
    if not selection.is_text:
        expected = _document_order_digest(selection.documents)
        if selection.order_digest != expected:
            raise ValueError(
                "selection order digest does not match its documents")
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    tmp.write_text(json.dumps(selection.manifest(), ensure_ascii=False,
                              indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def write_output_plan(selection: Selection, path: pathlib.Path) -> pathlib.Path:
    """Retain the frozen destination mapping as separate run evidence."""
    value = {
        "schema": "summer.output-plan.v1",
        "mode": selection.mode_key,
        "scope": selection.scope,
        "order_digest": selection.order_digest,
        "documents": [
            {"id": doc.id, "root_id": doc.root_id,
             "relative_path": doc.relative_path,
             "outputs": [_text(p) for p in doc.planned_outputs]}
            for doc in selection.documents
        ],
    }
    if selection.corpus_outputs:
        value["corpus_outputs"] = [_text(p) for p in selection.corpus_outputs]
    value["output_plan_sha256"] = output_plan_digest(selection)
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                              sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def read_manifest(path: pathlib.Path) -> Selection:
    path = pathlib.Path(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read selection manifest {path}: {exc}") from exc
    if not isinstance(value, dict) or value.get("schema") not in (SCHEMA_V1, SCHEMA_V2):
        raise ValueError(f"selection manifest must use schema {SCHEMA_V1} or {SCHEMA_V2}")
    manifest_schema = value["schema"]
    if value.get("kind") != "paths":
        raise ValueError("selection manifest kind must be paths")
    raw_roots = value.get("roots")
    raw_documents = value.get("documents")
    if not isinstance(raw_roots, list) or not isinstance(raw_documents, list):
        raise ValueError("selection manifest roots and documents must be arrays")
    mode = value.get("mode")
    scope = value.get("scope") or "batch"
    if mode not in VALID_MODES:
        raise ValueError("selection manifest has an invalid mode")
    if scope not in VALID_SCOPES:
        raise ValueError("selection manifest has an invalid scope")
    if scope == "corpus" and manifest_schema != SCHEMA_V2:
        raise ValueError("Corpus selections must use schema summer.selection.v2")
    roots = []
    root_ids = set()
    for item in raw_roots:
        if (not isinstance(item, dict)
                or item.get("kind") not in ("file", "folder")
                or not isinstance(item.get("path"), str)
                or not item.get("path")):
            raise ValueError("selection manifest has an invalid root")
        if not item.get("id") or item["id"] in root_ids:
            raise ValueError("selection manifest has duplicate root IDs")
        root_ids.add(item["id"])
        selected = _absolute(item["path"])
        raw_resolved = item.get("resolved_path", item["path"])
        if not isinstance(raw_resolved, str) or not raw_resolved:
            raise ValueError("selection manifest has an invalid root resolution")
        resolved = _absolute(raw_resolved)
        roots.append(SelectionRoot(str(item.get("id", "")), item["kind"],
                                   selected, resolved,
                                   _text(item.get("label") or selected.name)))
    docs = []
    document_ids = set()
    for item in raw_documents:
        if (not isinstance(item, dict) or not item.get("id")
                or not item.get("root_id") or not isinstance(item.get("path"), str)
                or not item.get("path") or not isinstance(item.get("relative_path"), str)):
            raise ValueError("selection manifest has an invalid document")
        if item["id"] in document_ids or item["root_id"] not in root_ids:
            raise ValueError("selection manifest has duplicate or unknown document identity")
        document_ids.add(item["id"])
        planned_value = item.get("planned_outputs") or []
        if (not isinstance(planned_value, list)
                or not all(isinstance(p, str) and p for p in planned_value)):
            raise ValueError("selection manifest has invalid planned outputs")
        planned = tuple(_absolute(p) for p in planned_value)
        if planned:
            validate_output_pair(planned, None, (item.get("path"),))
        size = item.get("size_bytes")
        digest = item.get("source_sha256")
        if item.get("error_kind") is None:
            if (type(size) is not int or size < 0
                    or not isinstance(digest, str)
                    or not re.fullmatch(r"[0-9a-f]{64}", digest)):
                raise ValueError("selection manifest has invalid source metadata")
        elif size is not None and (type(size) is not int or size < 0):
            raise ValueError("selection manifest has invalid source size")
        docs.append(DocumentTarget(
            str(item["id"]), str(item["root_id"]), _absolute(item["path"]),
            _absolute(item.get("resolved_path", item["path"])),
            _text(item["relative_path"]),
            _text(item.get("source_suffix", pathlib.Path(item["path"]).suffix)).casefold(),
            size, digest,
            item.get("error_kind"), item.get("error_detail"), planned))
    corpus = value.get("corpus") or {}
    if scope == "corpus":
        if (not isinstance(corpus, dict) or not isinstance(corpus.get("name"), str)
                or not isinstance(corpus.get("outputs"), list)):
            raise ValueError("Corpus manifest has invalid corpus metadata")
        _portable_corpus_stem(corpus["name"])
    for field in ("exclusions", "errors"):
        if not isinstance(value.get(field, []), list) or not all(
                isinstance(item, dict) for item in value.get(field, [])):
            raise ValueError(f"selection manifest {field} must be an array of objects")
    result = Selection(
        "paths", tuple(roots), tuple(docs), tuple(value.get("exclusions") or ()),
        tuple(value.get("errors") or ()), None, str(value.get("order_digest") or ""),
        mode, scope, corpus.get("name") if scope == "corpus" else None,
        tuple(_absolute(p) for p in corpus.get("outputs", ())) if scope == "corpus" else (),
        value.get("output_plan_sha256"), manifest_schema)
    if result.order_digest != _document_order_digest(result.documents):
        raise ValueError("selection manifest order digest does not match its documents")
    roots_by_id = {root.id: root for root in result.roots}
    for doc in result.documents:
        root = roots_by_id[doc.root_id]
        if doc.source_suffix not in DOC_SUFFIXES:
            raise ValueError("selection manifest has an unsupported document suffix")
        if not _inside(root.resolved_path, doc.source_path):
            raise ValueError("selection manifest document is outside its root")
        if root.kind == "file" and _identity(doc.resolved_path) != _identity(root.resolved_path):
            raise ValueError("selection manifest document does not match its file root")
        if root.kind == "folder" and not _inside(root.resolved_path, doc.resolved_path):
            raise ValueError("selection manifest resolved document is outside its root")
    if result.output_plan_sha256:
        if result.output_plan_sha256 != output_plan_digest(result):
            raise ValueError("selection manifest output plan digest does not match")
    if result.scope == "corpus":
        validate_output_pair(result.corpus_outputs,
                             (".corpus.summary.md", ".corpus.brief.md"),
                             (doc.source_path for doc in result.documents))
        if corpus.get("output_plan_sha256") != result.output_plan_sha256:
            raise ValueError("Corpus output plan digest does not match manifest")
    return result


__all__ = [
    "SCHEMA", "SCHEMA_V1", "SCHEMA_V2", "VALID_MODES", "VALID_SCOPES",
    "DOC_SUFFIXES", "GENERATED_SUFFIXES", "GENERATED_PREFIXES", "Selection", "SelectionRoot",
    "DocumentTarget", "resolve_paths", "combine_path_selections",
    "classify_clipboard", "text_selection",
    "looks_like_path", "split_path_list", "clipboard_path_intent",
    "verify_document", "plan_outputs", "format_output_filenames", "parse_output_dir",
    "is_absolute_output", "resolve_output_directory",
    "clipboard_output_directory",
    "default_corpus_name", "plan_corpus_outputs", "output_plan_digest",
    "validate_output_pair", "write_manifest", "read_manifest", "write_output_plan",
]
