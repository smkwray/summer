#!/usr/bin/env python3
"""Model-driven read-aloud reformatting for text-to-speech voice narration.

    speechprep.py SOURCE_FILE OUT_FILE [WORKDIR]

Reformat clean document text into voice-ready prose.
Preserves 100% of substantive arguments, facts, names, and qualifications.
Leans on model intelligence for initialism vs acronym pronunciation (e.g. TIPS as words,
FBI as letters), natural contractions, spoken numbers/dates, table/equation rendering.
"""
from __future__ import annotations
import argparse, importlib.util, json, os, pathlib, re, sys

HERE = pathlib.Path(__file__).parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def _runner():
    spec = importlib.util.spec_from_file_location("ms", HERE / "mapsum.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _chunk_text(text: str, target_words: int = 1800) -> list[str]:
    paragraphs = text.split("\n\n")
    chunks, current = [], []
    current_words = 0
    for p in paragraphs:
        p_str = p.strip()
        if not p_str:
            continue
        words = len(p_str.split())
        if current and (current_words + words > target_words):
            chunks.append("\n\n".join(current))
            current = [p_str]
            current_words = words
        else:
            current.append(p_str)
            current_words += words
    if current:
        chunks.append("\n\n".join(current))
    return chunks or [text]



def _audit_options():
    import json_contract
    return json_contract.options("summer_tts_audit", json_contract.SPEECH_AUDIT)


def _audit(ms, work_dir, source: str, candidate: str, stage: str) -> list[str]:
    """Model-judged findings against one converted chunk. Raises when no
    auditor answers; the caller records that as review unavailable."""
    import json_contract
    prompt = ((HERE / "prompts" / "speechprep-audit.txt").read_text(encoding="utf-8")
              .replace("{SOURCE}", source).replace("{CANDIDATE}", candidate))
    raw = ms.run(prompt, work_dir, ms.AUDIT_MODELS, stage,
                 validate=lambda r: json_contract.parse(
                     r, json_contract.SPEECH_AUDIT, stage),
                 gateway_options=_audit_options())
    obj = json_contract.parse(raw, json_contract.SPEECH_AUDIT, stage)
    findings = [str(f) for f in (obj.get("findings") or []) if str(f).strip()]
    if obj.get("verdict") == "revise" and not findings:
        findings = ["revise requested without a named defect"]
    return findings


def convert_chunk(ms, work_dir, template: str, chunk: str, stage: str) -> dict:
    """Write, audit, at most one repair, re-audit, publish the safer text.

    A converted chunk is never discarded once it exists: an unavailable
    reviewer, a failed repair, or findings that survive the repair publish
    the safest retained text with its status and open findings.
    """
    prompt = template.replace("{SOURCE}", chunk)
    text = ms.run(prompt, work_dir, ms.MODELS, stage).strip()
    import mapsum
    producer = mapsum.last_ok_route(work_dir, stage, ms.MODELS)
    if not text:
        raise ValueError(f"{stage}: model returned empty speech text")
    record = {"stage": stage, "selected": "initial", "repair": "none"}
    try:
        findings = _audit(ms, work_dir, chunk, text, f"{stage}-audit")
        record["review"] = "complete"
    except Exception as exc:  # reviewer outage is disclosed, not fatal
        record.update(review=f"unavailable: {exc}", findings=[])
        return {**record, "text": text}
    if findings:
        repair = (prompt + (HERE / "prompts" / "speechprep-repair.txt")
                  .read_text(encoding="utf-8")
                  .replace("{CANDIDATE}", text)
                  .replace("{FINDINGS}", "\n- ".join(findings)))
        try:
            # One correction, from the producer that wrote the text.
            revised = ms.run(repair, work_dir, producer,
                             f"{stage}-revise").strip()
            if not revised:
                raise ValueError("empty repair")
            record["repair"] = "once"
            try:
                findings2 = _audit(ms, work_dir, chunk, revised, f"{stage}-reaudit")
            except Exception as exc:
                # Unreviewed bytes never displace reviewed ones.
                record["notes"] = [f"reaudit unavailable: {exc}"]
                findings2 = None
            # A count is not safety: one missing passage traded for one
            # reversal is a worse text. The revision replaces the original
            # only when the re-audit finds nothing left.
            if findings2 is not None and not findings2:
                text, findings = revised, findings2
                record["selected"] = "revised"
        except Exception as exc:
            record["repair"] = f"unavailable: {exc}"
    record["findings"] = findings
    record["status"] = "pass" if not findings else "open_findings"
    return {**record, "text": text}


def convert_text(source_text: str, work_dir: pathlib.Path, runner=None) -> str:
    ms = runner or _runner()
    template = (HERE / "prompts" / "speechprep.txt").read_text(encoding="utf-8")
    chunks = _chunk_text(source_text)
    results, report = [], []
    for idx, chunk in enumerate(chunks):
        stage = f"speechprep{idx:03d}" if len(chunks) > 1 else "speechprep"
        rec = convert_chunk(ms, work_dir, template, chunk, stage)
        results.append(rec.pop("text"))
        report.append(rec)
        status = rec.get("status") or rec.get("review")
        print(f"[speechprep] {stage}: {status}, repair: {rec['repair']}, "
              f"{len(rec.get('findings') or [])} open finding(s)", flush=True)
        for finding in rec.get("findings") or []:
            print(f"[speechprep]   - {finding}", flush=True)
    (work_dir / "speech-report.json").write_text(
        json.dumps({"chunks": report}, indent=1), encoding="utf-8")
    return "\n\n".join(results) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("source", help="source text or markdown file")
    ap.add_argument("destination", help="destination text file")
    ap.add_argument("work_dir", nargs="?", default=None, help="temporary work directory")
    args = ap.parse_args(argv)

    src = pathlib.Path(args.source)
    dst = pathlib.Path(args.destination)
    work = (pathlib.Path(args.work_dir) if args.work_dir
            else dst.parent / "speech_work")
    work.mkdir(parents=True, exist_ok=True)

    try:
        text = src.read_text(encoding="utf-8")
    except Exception as exc:
        print(f"speechprep: unreadable source {src}: {exc}", file=sys.stderr)
        return 1

    try:
        result = convert_text(text, work)
    except Exception as exc:
        print(f"speechprep failed: {exc}", file=sys.stderr)
        return 1

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(result, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
