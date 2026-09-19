#!/usr/bin/env python3
"""Manifest-driven, owner-gated live qualification for writing contract v2.

Without ``--execute`` this is read-only: it verifies hashes, profiles, the
bounded matrix, and prints the exact plan. Live calls additionally require the
manifest's authorization bit and its exact raw SHA-256 on argv.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys


HERE = pathlib.Path(__file__).resolve()
PROJECT = HERE.parents[2]
CLI = PROJECT / "src" / "summ_cli.py"
PROFILE_STORE = (pathlib.Path.home() / "Library" / "Application Support" /
                 "summer" / "profiles.json")
SCHEMA = "summer.writing-contract-qualification.v1"
APPROVAL_SCHEMA = "summer.writing-contract-review-approval.v1"


class QualificationError(ValueError):
    pass


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def path_of(value: str) -> pathlib.Path:
    path = pathlib.Path(value).expanduser()
    return path if path.is_absolute() else PROJECT / path


def source_bytes(spec: dict) -> tuple[bytes, int]:
    path = path_of(str(spec.get("path") or ""))
    if not path.is_file():
        raise QualificationError(f"source is unavailable: {path}")
    raw = path.read_bytes()
    if sha(raw) != spec.get("source_sha256"):
        raise QualificationError(f"source hash changed: {path}")
    text = raw.decode("utf-8", errors="replace")
    words = text.split()
    if len(words) != spec.get("source_words"):
        raise QualificationError(f"source word count changed: {path}")
    kind = spec.get("kind")
    if kind == "file":
        return raw, len(words)
    if kind != "word_span":
        raise QualificationError(f"unknown source kind: {kind!r}")
    start, end = spec.get("start_word"), spec.get("end_word")
    if (not isinstance(start, int) or not isinstance(end, int)
            or not 1 <= start <= end <= len(words)):
        raise QualificationError(f"invalid source word span: {start}..{end}")
    prepared = " ".join(words[start - 1:end]).encode("utf-8")
    if sha(prepared) != spec.get("prepared_sha256"):
        raise QualificationError("prepared source span hash changed")
    if len(prepared.decode().split()) != spec.get("prepared_words"):
        raise QualificationError("prepared source span word count changed")
    return prepared, end - start + 1


def jobs(manifest: dict) -> list[dict]:
    out = []
    for phase, spec in (manifest.get("matrix") or {}).items():
        for route in spec.get("routes") or []:
            for source in spec.get("sources") or []:
                repeats = spec.get("repeats")
                if not isinstance(repeats, int) or repeats < 1:
                    raise QualificationError(f"{phase}: repeats must be positive")
                for repeat in range(1, repeats + 1):
                    out.append({
                        "job_id": f"{len(out) + 1:02d}-{phase}-{route}-{source}-r{repeat}",
                        "phase": phase, "route": route, "source": source,
                        "repeat": repeat,
                        "one_unit_per_response": bool(
                            spec.get("one_unit_per_response")),
                        "conditional_on": spec.get("conditional_on"),
                    })
    return out


def profiles() -> dict:
    if not PROFILE_STORE.is_file():
        return {}
    value = json.loads(PROFILE_STORE.read_text())
    return value if isinstance(value, dict) else {}


def validate_manifest(path: pathlib.Path) -> tuple[dict, str, list[dict], dict]:
    raw = path.read_bytes()
    try:
        manifest = json.loads(raw)
    except Exception as exc:
        raise QualificationError(f"invalid manifest JSON: {exc}") from exc
    if manifest.get("schema") != SCHEMA:
        raise QualificationError("unsupported qualification manifest schema")
    if manifest.get("writing_contract") != "summer.writing-plan.v2":
        raise QualificationError("manifest selects the wrong writing contract")
    planned = jobs(manifest)
    maximum = manifest.get("max_live_jobs")
    if not isinstance(maximum, int) or maximum < 1 or len(planned) > maximum:
        raise QualificationError(
            f"matrix has {len(planned)} jobs but maximum is {maximum}")
    if len(planned) != 19:
        raise QualificationError(
            f"frozen work order requires 19 planned jobs, found {len(planned)}")
    sources = manifest.get("sources") or {}
    for job in planned:
        if job["source"] not in sources:
            raise QualificationError(
                f"{job['job_id']}: unknown source {job['source']}")
        if job["route"] not in (manifest.get("routes") or {}):
            raise QualificationError(
                f"{job['job_id']}: unknown route {job['route']}")
    source_receipts = {}
    for name, spec in sources.items():
        prepared, count = source_bytes(spec)
        source_receipts[name] = {
            "prepared_sha256": sha(prepared), "prepared_words": count}
    return manifest, sha(raw), planned, source_receipts


def automatic_result(report: dict, acceptance: dict) -> tuple[bool, list[str]]:
    failures = []
    if report.get("candidate_state") != acceptance.get(
            "required_candidate_state"):
        failures.append("candidate is not quality-qualified")
    allocated = ((report.get("capability") or {}).get("allocated_words") or {})
    if acceptance.get("require_within_ceilings"):
        for depth in ("detailed", "brief"):
            if int(report.get(f"{depth}_words") or 0) > int(
                    allocated.get(depth) or 0):
                failures.append(f"{depth} exceeds its frozen allocation")
    if acceptance.get("require_complete_source_accounting"):
        if not report.get("writing_plan_sha256"):
            failures.append("writing plan identity is absent")
    if acceptance.get("require_runtime_identity"):
        provenance = report.get("producer_provenance") or []
        if not provenance or any(not item.get("producer") for item in provenance):
            failures.append("producer provenance is incomplete")
    return not failures, failures


def _condition_results(results: list[dict], condition: str) -> list[dict]:
    phase, route = condition.split(":", 1)
    return [item for item in results
            if item.get("phase") == phase and item.get("route") == route]


def automatic_condition_passes(manifest: dict, results: list[dict],
                               condition: str) -> bool:
    """Apply the frozen aggregate promotion gate, including first-pass rate."""
    phase, route = condition.split(":", 1)
    spec = (manifest.get("matrix") or {}).get(phase) or {}
    relevant = _condition_results(results, condition)
    expected = len(spec.get("sources") or []) * int(spec.get("repeats") or 0)
    if route not in (spec.get("routes") or []) or len(relevant) != expected:
        return False
    passed = [item for item in relevant
              if item.get("status") == "automatic_pass_manual_review_pending"]
    required = int(spec.get("required_passes_per_route") or expected)
    required_first = int(spec.get("required_first_passes_per_route") or 0)
    return (len(passed) >= required and
            sum(bool(item.get("first_pass")) for item in passed)
            >= required_first)


def _portable_condition(condition: str) -> str:
    return condition.replace(":", "-")


def write_review_request(root: pathlib.Path, manifest: dict, digest: str,
                         results: list[dict], condition: str) -> pathlib.Path:
    relevant = _condition_results(results, condition)
    request = {
        "schema": "summer.writing-contract-review-request.v1",
        "manifest_sha256": digest,
        "condition": condition,
        "manual_checks": (manifest.get("acceptance") or {}).get(
            "manual_checks") or [],
        "candidates": [
            {"candidate": item["candidate"],
             "report_sha256": item["report_sha256"]}
            for item in relevant
        ],
    }
    directory = root / "review-requests"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{_portable_condition(condition)}.json"
    encoded = json.dumps(request, indent=2) + "\n"
    if path.exists() and path.read_text() != encoded:
        raise QualificationError(f"review request changed: {path}")
    path.write_text(encoded)
    return path


def manual_condition_approved(root: pathlib.Path, digest: str,
                              results: list[dict], condition: str) -> bool:
    """Require an exact, post-run blind-review receipt before promotion."""
    path = (root / "review-approvals" /
            f"{_portable_condition(condition)}.json")
    if not path.exists():
        return False
    try:
        value = json.loads(path.read_text())
    except Exception as exc:
        raise QualificationError(
            f"invalid manual review approval {path}: {exc}") from exc
    expected = {item["candidate"]: item["report_sha256"]
                for item in _condition_results(results, condition)}
    if (value.get("schema") != APPROVAL_SCHEMA or
            value.get("manifest_sha256") != digest or
            value.get("condition") != condition or
            value.get("approved") is not True or
            value.get("candidate_reports") != expected):
        raise QualificationError(
            f"manual review approval is stale or does not bind every candidate: {path}")
    return True


def _write_progress(root: pathlib.Path, digest: str, results: list[dict],
                    blind_map: dict) -> None:
    (root / "results.json").write_text(json.dumps(
        {"manifest_sha256": digest, "results": results}, indent=2) + "\n")
    (root / "blind-map.json").write_text(json.dumps(
        blind_map, indent=2) + "\n")


def _load_completed(job_root: pathlib.Path, job: dict,
                    acceptance: dict) -> dict:
    receipt = job_root / "job-result.json"
    if not receipt.is_file():
        raise QualificationError(
            f"incomplete prior job requires diagnosis, never an automatic retry: {job_root}")
    try:
        result = json.loads(receipt.read_text())
    except Exception as exc:
        raise QualificationError(f"invalid prior job receipt: {receipt}") from exc
    for key, value in job.items():
        if result.get(key) != value:
            raise QualificationError(f"prior job receipt does not match plan: {receipt}")
    if result.get("report_sha256"):
        reports = list((job_root / "work").glob("*/full-report.json"))
        if len(reports) != 1 or sha(reports[0].read_bytes()) != result["report_sha256"]:
            raise QualificationError(f"prior report changed: {job_root}")
        report = json.loads(reports[0].read_text())
        passed, failures = automatic_result(report, acceptance)
        expected_status = ("automatic_pass_manual_review_pending" if passed
                           else "automatic_fail")
        if (result.get("status") != expected_status or
                result.get("failures") != failures or
                result.get("first_pass") != (report.get("repair") == "none")):
            raise QualificationError(f"prior automatic result changed: {job_root}")
    return result


def execute(manifest: dict, digest: str, planned: list[dict],
            authorization: str):
    if manifest.get("live_authorized") is not True:
        raise QualificationError(
            "manifest live_authorized is false; no model calls were made")
    if authorization != digest:
        raise QualificationError(
            "--authorization must equal the exact manifest SHA-256")
    configured = profiles()
    missing = sorted({manifest["routes"][job["route"]]["profile"]
                      for job in planned
                      if manifest["routes"][job["route"]]["profile"]
                      not in configured})
    if missing:
        raise QualificationError(
            "saved qualification profiles are missing: " + ", ".join(missing))
    root = path_of(manifest["output_root"]).resolve()
    if root == PROJECT or PROJECT in root.parents:
        raise QualificationError("live outputs may not be stored in the project")
    root.mkdir(parents=True, exist_ok=True)
    results, blind_map = [], {}
    for number, job in enumerate(planned, 1):
        condition = job.get("conditional_on")
        if condition:
            if not automatic_condition_passes(manifest, results, condition):
                results.append({**job, "status": "conditional_not_run"})
                continue
            write_review_request(root, manifest, digest, results, condition)
            if not manual_condition_approved(root, digest, results, condition):
                results.append({**job, "status": "manual_review_pending"})
                continue
        job_root = root / job["job_id"]
        if job_root.exists():
            result = _load_completed(
                job_root, job, manifest.get("acceptance") or {})
            results.append(result)
            if result.get("candidate"):
                blind_map[result["candidate"]] = {
                    "job_id": job["job_id"], "route": job["route"]}
            _write_progress(root, digest, results, blind_map)
            continue
        job_root.mkdir(parents=True)
        prepared, _count = source_bytes(manifest["sources"][job["source"]])
        source_path = job_root / "source.md"
        source_path.write_bytes(prepared)
        output = job_root / "published"
        work = job_root / "work"
        env = dict(os.environ)
        env["SUMM_WRITING_CONTRACT"] = "v2"
        if job["one_unit_per_response"]:
            env["SUMM_WRITING_ONE_UNIT_PER_REQUEST"] = "1"
        command = [sys.executable, str(CLI), str(source_path),
                   "--profile", manifest["routes"][job["route"]]["profile"],
                   "--out", str(output), "--work-dir", str(work),
                   "--affix", "reading", "--affix-secondary", "brief"]
        proc = subprocess.run(command, cwd=PROJECT, env=env,
                              text=True, capture_output=True)
        (job_root / "stdout.txt").write_text(proc.stdout)
        (job_root / "stderr.txt").write_text(proc.stderr)
        reports = list(work.glob("*/full-report.json"))
        if proc.returncode or len(reports) != 1:
            result = {**job, "status": "failed", "exit_code": proc.returncode,
                      "report_count": len(reports)}
            results.append(result)
            (job_root / "job-result.json").write_text(
                json.dumps(result, indent=2) + "\n")
            _write_progress(root, digest, results, blind_map)
            continue
        report = json.loads(reports[0].read_text())
        passed, failures = automatic_result(report, manifest["acceptance"])
        candidate = f"C{number:03d}"
        blind = root / "blind" / candidate
        blind.mkdir(parents=True, exist_ok=False)
        shutil.copy2(work / "01" / "detailed.md", blind / "detailed.md")
        shutil.copy2(work / "01" / "brief.md", blind / "brief.md")
        shutil.copy2(source_path, blind / "source.md")
        blind_map[candidate] = {"job_id": job["job_id"],
                                "route": job["route"]}
        results.append({**job, "status": (
            "automatic_pass_manual_review_pending" if passed
            else "automatic_fail"), "failures": failures,
            "first_pass": report.get("repair") == "none",
            "candidate": candidate,
            "report_sha256": sha(reports[0].read_bytes())})
        (job_root / "job-result.json").write_text(
            json.dumps(results[-1], indent=2) + "\n")
        _write_progress(root, digest, results, blind_map)
    _write_progress(root, digest, results, blind_map)
    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--authorization", default="")
    args = parser.parse_args(argv)
    try:
        manifest, digest, planned, receipts = validate_manifest(
            path_of(args.manifest))
        configured = profiles()
        missing = sorted({manifest["routes"][j["route"]]["profile"]
                          for j in planned
                          if manifest["routes"][j["route"]]["profile"]
                          not in configured})
        print(json.dumps({
            "manifest_sha256": digest,
            "live_authorized": manifest.get("live_authorized") is True,
            "planned_jobs": len(planned),
            "conditional_jobs": sum(bool(j.get("conditional_on"))
                                    for j in planned),
            "missing_profiles": missing,
            "sources": receipts,
            "jobs": planned,
        }, indent=2))
        if not args.execute:
            return 0
        execute(manifest, digest, planned, args.authorization)
        return 0
    except QualificationError as exc:
        print(f"qualification blocked: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
