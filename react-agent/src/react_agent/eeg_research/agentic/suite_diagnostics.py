"""Read existing development outputs into immutable, suite-scoped diagnostics."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path

from .artifacts import file_digest, register, resolve_verified_artifact, lookup
from .diagnostics import summarize_bundle
from .jsonl_read import read_jsonl

VERSION = "eeg_research.suite_diagnostics.v1"

_PROFILES = {
    "existing_development": {"population": "source_subject_development",
        "outputs": ["recorded_learning_curve", "exported_retrieval_sample", "exported_embedding_sample"]},
    "full_retrieval": {"population": "method_development_benchmark",
        "outputs": ["all_query_top1_top5", "positive_rank_intervals", "margins", "gallery_hub_counts", "representation"]},
}


def diagnostic_profiles(full: dict, target: str) -> list[dict]:
    names = ["existing_development"]
    if full.get("evaluation_mode") == "loso_method_search" and any(
            row.get("candidate_id") == target and row.get("evaluation_valid") is True
            for row in full.get("evidence") or []):
        names.append("full_retrieval")
    return [{"profile": name, "target_candidate_id": target, **_PROFILES[name]} for name in names]

def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str).encode()).hexdigest()

def sources(camp: Path, state: dict, *, verified_records: list[dict] | None = None,
            target_ids: set[str] | None = None) -> list[dict]:
    """Use native verified scores/freezes, retaining phase rather than inventing a job."""
    camp = Path(camp).resolve()
    rows = []
    if state.get("evaluation_mode") == "loso_method_search":
        from .method_suite import verified_suite_records
        from .screening import verified_screening_records
        records = (verified_records if verified_records is not None else
                   [*verified_screening_records(camp, state, target_ids=target_ids),
                    *verified_suite_records(camp, state, target_ids=target_ids)])
        for record in records:
            if target_ids is not None and record.get("candidate_id") not in target_ids:
                continue
            scientific = record.get("screening") or record.get("aggregate") or {}
            for score in scientific.get("scores") or []:
                fold = score["fold_id"]
                path = camp / "suites" / record["suite_id"] / fold / "benchmark_freeze.json"
                freeze = _read(path)
                job = Path(freeze["job_dir"]).resolve()
                if not job.is_relative_to(camp):
                    raise ValueError("diagnostic_job_outside_campaign")
                j = _read(job / "job.json") if (job / "job.json").is_file() else {}
                if j and (j.get("status") != "finished" or j.get("candidate_id") != record["candidate_id"]
                        or j.get("seed") != score["seed"] or freeze["checkpoint_hash"] != score["checkpoint_hash"]):
                    raise ValueError("diagnostic_fold_job_identity_mismatch")
                rows.append({"candidate_id": record["candidate_id"], "suite_id": record["suite_id"],
                    "fold_id": fold, "held_out_subject": score["held_out_subject"], "seed": score["seed"],
                    "phase": scientific.get("phase", "confirmation_10fold"),
                    "scope": scientific.get("scope", "method_development_benchmark"),
                    "coverage": scientific.get("coverage", len(scientific.get("scores") or [])),
                    "source_hash": (j.get("manifest") or {}).get("entry_sha256") or record.get("source_hash"),
                    "spec_hash": record.get("spec_hash"),
                    "method_revision": record.get("method_revision"), "job_id": j.get("job_id", job.name),
                    "resolver_gap": None if j else "job_record_missing",
                    "job_dir": str(job), "freeze_ref": str(path), "freeze_sha256": file_digest(path),
                    "scientific_result_ref": record["suite_ref"], "scientific_result_hash": record["result_hash"]})
    else:
        for record in state.get("evidence") or []:
            if target_ids is not None and record.get("candidate_id") not in target_ids:
                continue
            if not record.get("job_dir"):
                continue
            job = Path(record["job_dir"]).resolve()
            if not job.is_relative_to(camp):
                raise ValueError("diagnostic_job_outside_campaign")
            rows.append({key: record[key] for key in ("candidate_id", "job_id", "seed", "fidelity", "source_hash", "spec_hash") if key in record} | {"job_dir": str(job), "scope": "source_development"})
    # A ten-fold record supersedes its two-fold navigation, not its historical scores.
    unique = {}
    for row in rows:
        unique[(row.get("suite_id"), row.get("job_id"), row.get("fold_id"))] = row
    return list(unique.values())

def _dependencies(rows):
    result = {}
    for row in rows:
        job = Path(row["job_dir"])
        paths = [job / name for name in ("history.jsonl", "diagnostic_bundle.json", "diagnostic_summary.json", "metrics.json", "source_binding.json", "selected_checkpoint.json")]
        paths.extend(Path(row[key]) for key in ("freeze_ref", "scientific_result_ref") if row.get(key))
        for path in paths:
            if path.is_file():
                result[str(path)] = {"path": str(path), "sha256": file_digest(path)}
    return list(result.values())

def input_epochs(camp: Path, state: dict, *, verified_records: list[dict] | None = None) -> dict:
    rows = sources(camp, state, verified_records=verified_records)
    dependencies = _dependencies(rows)
    return {action: _digest({"version": VERSION, "action": action, "sources": rows,
                             "dependencies": dependencies})
            for action in ("collect_diagnostics", "diagnose_results")}

def learning_profile(points: list[dict], selected_checkpoint: dict, *, read_coverage: dict | None = None) -> dict:
    """Describe existing rows; a recent trend is neither a stopping rule nor convergence.

    Checkpoint identity comes from the trainer's selection record, including its
    tie policy, rather than selecting a new checkpoint from the curve.
    """
    def number(value):
        return type(value) in (int, float) and math.isfinite(value)

    coverage = dict(read_coverage or {"valid_rows": len(points), "complete": bool(points)})
    losses = [row for row in points if number(row.get("train_loss"))]
    measured = [row for row in points if number(row.get("fixed_bank_top1"))]
    # Ten recorded epochs is a descriptive window only. Sparse rows retain their
    # epoch identities so readers cannot mistake them for consecutive epochs.
    tail = losses[-10:]
    first, last = (losses[0], losses[-1]) if losses else ({}, {})
    tail_first, tail_last = (tail[0], tail[-1]) if tail else ({}, {})
    start, end = tail_first.get("train_loss"), tail_last.get("train_loss")
    best_epoch = selected_checkpoint.get("best_epoch")
    selected_row = next((row for row in measured if row.get("epoch") == best_epoch), {})
    recorded_best = selected_checkpoint.get("best_fixed_bank_top1")
    if not number(recorded_best):
        recorded_best = selected_row.get("fixed_bank_top1")
    last_epoch = points[-1].get("epoch") if points else None
    since_best = (last_epoch - best_epoch if number(last_epoch) and number(best_epoch)
                  and last_epoch >= best_epoch else None)
    return {"status": "partial" if coverage.get("diagnostics") or coverage.get("complete") is False and points
            else "ready" if points else "unavailable",
        "epochs": len(points), "last_epoch": last_epoch,
        "first_train_loss": first.get("train_loss"), "last_train_loss": last.get("train_loss"),
        "tail_loss": {"rows": len(tail), "first_epoch": tail_first.get("epoch"),
            "last_epoch": tail_last.get("epoch"), "start": start, "end": end,
            "decrease": start - end if start is not None and end is not None else None,
            "relative_decrease": (start - end) / abs(start) if start not in (None, 0) and end is not None else None},
        "first_fixed_bank_top1": measured[0]["fixed_bank_top1"] if measured else None,
        "last_fixed_bank_top1": measured[-1]["fixed_bank_top1"] if measured else None,
        "best_epoch": best_epoch, "best_fixed_bank_top1": recorded_best,
        "epochs_since_selected_checkpoint": since_best,
        "checkpoint_selection_status": "recorded" if best_epoch is not None else "unavailable",
        "read_coverage": coverage, "metric_scope": "source_subject_development_curve"}


def curve(job: Path) -> dict:
    read = read_jsonl(job / "history.jsonl")
    points = read["rows"]
    selected_path = job / "selected_checkpoint.json"
    selected = _read(selected_path) if selected_path.is_file() else {}
    coverage = {"valid_rows": len(points), "complete": bool(points) and not read["diagnostics"] and not read["truncated"],
                "truncated": read["truncated"], "diagnostics": read["diagnostics"]}
    return {**learning_profile(points, selected, read_coverage=coverage),
        "reason": "history_missing_or_empty" if not points else None,
        "epoch_ids": [row.get("epoch") for row in points],
        "fixed_bank_trend": [row.get("fixed_bank_top1") for row in points]}

def _full_retrieval(camp: Path, row: dict) -> tuple[dict, list[dict]]:
    from . import jobs
    from .loso_study import read
    from .method_evaluate import read_diagnostics
    from react_agent.eeg_training.protocol import torch_python
    import subprocess

    directory = camp / "suites" / row["suite_id"] / row["fold_id"]
    paths = [directory / "benchmark.diagnostics.json", directory / "benchmark.diagnostics.receipt.json"]
    if not all(path.is_file() for path in paths):
        env = jobs.research_env(None)
        env.update(CUDA_VISIBLE_DEVICES="", EEG_TRAIN_DEVICE="cpu", EEG_FINAL_TEST="0",
                   OMP_NUM_THREADS="8", MKL_NUM_THREADS="8")
        result = subprocess.run([torch_python(), "-m", "react_agent.eeg_research.agentic.method_evaluate",
            "--camp", str(camp), "--suite", row["suite_id"], "--fold", row["fold_id"], "--diagnostics"],
            env=env, capture_output=True, text=True, timeout=300)
        if result.returncode:
            raise ValueError("method_diagnostic_scoring_failed:" + result.stderr[-1800:])
    report = read_diagnostics(directory, read(directory / "benchmark.json"))
    return report, [{"path": str(path), "sha256": file_digest(path)} for path in paths]


def collect(camp: Path, state: dict, *, kind="diagnostic_bundle", request: dict | None = None) -> dict:
    """Execute the selected diagnostic profile and publish its actual coverage."""
    if request is not None:
        profile = request.get("profile")
        if profile not in _PROFILES or any(request.get(key) != value for key, value in _PROFILES[profile].items()):
            raise ValueError("diagnostic_request_profile_mismatch")
        target = request["target_candidate_id"]
        rows = sources(camp, state, target_ids={target})
    else:
        profile = "existing_development"
        rows = sources(camp, state)
    dependencies = _dependencies(rows)
    captured = {}
    if profile == "full_retrieval":
        for row in rows:
            if row.get("resolver_gap"):
                continue
            report, bound = _full_retrieval(camp, row)
            captured[(row["suite_id"], row["fold_id"])] = report
            dependencies.extend(bound)
    identity = {"version": VERSION, "kind": kind, "sources": rows, "dependencies": dependencies}
    if request is not None:
        identity["diagnostic_request"] = request
    source_epoch = _digest(identity)
    path = Path(camp) / "diagnostics" / (kind + "_" + source_epoch + ".json")
    reused = path.is_file()
    if reused:
        report = _read(path)
        if report.get("source_epoch") != source_epoch or report.get("dependencies") != dependencies:
            raise ValueError("diagnostic_cached_source_changed")
    else:
        runs, coverage, missing = [], [], []
        for row in rows:
            job = Path(row["job_dir"])
            item = {**row, "learning_profile": curve(job)}
            if row.get("resolver_gap"):
                item["learning_profile"] = {"status": "unavailable", "reason": row["resolver_gap"], "epochs": 0}
                item["diagnostics"] = {"status": "unavailable", "reason": row["resolver_gap"]}
                missing.extend(_PROFILES[profile]["outputs"])
                coverage.append({**{key: row.get(key) for key in ("candidate_id", "suite_id", "fold_id")},
                                 "population": _PROFILES[profile]["population"], "queries": 0})
                runs.append(item)
                continue
            bundle = job / "diagnostic_bundle.json"
            full_report = captured.get((row.get("suite_id"), row.get("fold_id")))
            if full_report is not None:
                item["diagnostics"] = summarize_bundle(full_report)
                actual = full_report["coverage"]
            elif profile == "full_retrieval":
                item["diagnostics"] = {"status": "unavailable", "reason": "full_retrieval_not_available"}
                actual = {"population": _PROFILES[profile]["population"], "queries": 0}
                missing.extend(_PROFILES[profile]["outputs"])
            elif bundle.is_file():
                try:
                    item["diagnostics"] = summarize_bundle(_read(bundle))
                except json.JSONDecodeError:
                    item["diagnostics"] = {"status": "partial", "reason": "malformed_existing_diagnostic_bundle"}
                sample = (item["diagnostics"].get("items", {}).get("retrieval_errors") or {}).get("coverage") or {}
                actual = {"population": "source_subject_development", "queries": sample.get("query_rows"),
                    "total_queries": sample.get("total_query_count"), "embedding_count": sample.get("embedding_sample")}
            else:
                item["diagnostics"] = {"status": "unavailable", "reason": "diagnostic_bundle_not_exported"}
                actual = {"population": "source_subject_development", "queries": 0}
                missing.extend(["exported_retrieval_sample", "exported_embedding_sample"])
            if profile == "existing_development":
                items = item["diagnostics"].get("items") or {}
                for name, output in (("retrieval_errors", "exported_retrieval_sample"),
                                     ("representation", "exported_embedding_sample")):
                    if (items.get(name) or {}).get("status") not in {"observed", "undefined"}:
                        missing.append(output)
                if item["learning_profile"]["status"] == "unavailable":
                    missing.append("recorded_learning_curve")
            coverage.append({**{key: row.get(key) for key in ("candidate_id", "suite_id", "fold_id")}, **actual})
            runs.append(item)
        report = {"schema_version": VERSION, "kind": kind, "source_epoch": source_epoch,
            "diagnostic_scope": "method_development_benchmark" if profile == "full_retrieval" else "campaign_history_development_annotation", "runs": runs,
            "dependencies": dependencies, "final_test_accessed": False,
            "status": ("partial" if any(r.get("resolver_gap") for r in runs) else "ready") if runs else "unavailable", "reason": None if runs else "no_verified_development_jobs",
            "read_scope": "existing source-subject development exports and training curves; no new inference/training",
            "summary": {"folds_read": len(runs), "missing_or_partial_history": sum(r["learning_profile"]["status"] != "ready" for r in runs)}}
        if request is not None:
            report.update(diagnostic_request=request,
                measurement_kind="full_retrieval_diagnostic" if profile == "full_retrieval" else "existing_development_diagnostic",
                coverage=coverage, missing_outputs=sorted(set(missing)))
            if missing and runs:
                report["status"] = "partial"
            if not runs:
                report["missing_outputs"] = list(_PROFILES[profile]["outputs"])
            if profile == "full_retrieval":
                report["read_scope"] = "All queries from accepted method benchmarks; same frozen checkpoint and scoring path; no training."
        from .loso_study import write_once
        write_once(path, report)
    artifact_id = "art_diag_" + source_epoch[:24]
    artifact = lookup(camp, artifact_id)
    if artifact is None:
        artifact = register(camp, path, kind=kind, artifact_id=artifact_id)
    else:
        artifact = resolve_verified_artifact(camp, artifact_id)
    new_information = not any(row.get("evidence_id") == "ev_" + kind + "_" + source_epoch[:16]
                              for row in state.get("evidence") or [])
    return {"evidence_id": "ev_" + kind + "_" + source_epoch[:16], "kind": kind,
        "status": report["status"], "source_epoch": source_epoch, "diagnostic_ref": str(path),
        "artifact_refs": [artifact["artifact_id"]], "summary": report["summary"],
        "curve": [{**{key: r.get(key) for key in ("candidate_id", "suite_id", "fold_id", "held_out_subject", "seed", "phase", "scope")}, **r["learning_profile"]} for r in report["runs"]] if kind == "learning_profile" else [],
        "new_information": new_information, "diagnostic_scope": report["diagnostic_scope"],
        **{key: report[key] for key in ("diagnostic_request", "measurement_kind", "coverage", "missing_outputs") if key in report}}

def _read_verified_report(camp: Path, artifact_ref: str) -> tuple[dict, dict]:
    artifact = resolve_verified_artifact(camp, artifact_ref)
    report = _read(artifact["path"])
    if report.get("schema_version") != VERSION or report.get("final_test_accessed") is not False:
        raise ValueError("diagnostic_source_not_native")
    if any(file_digest(Path(dep["path"])) != dep["sha256"] for dep in report.get("dependencies") or []):
        raise ValueError("diagnostic_dependencies_changed")
    return artifact, report


def _report_facts(report: dict, artifact: dict, evidence_id: str, current: list[dict], *,
                  job_ids=None, target_ids=None, suite_ids=None) -> list[dict]:
    allowed = {(row.get("job_id"), row.get("suite_id"), row.get("source_hash")) for row in current}
    facts = []
    for run in report.get("runs") or []:
        if (run.get("job_id"), run.get("suite_id"), run.get("source_hash")) not in allowed:
            continue
        if target_ids is not None and run.get("candidate_id") not in target_ids:
            continue
        if suite_ids is not None and run.get("suite_id") not in suite_ids:
            continue
        if job_ids is not None and run.get("job_id") not in job_ids and suite_ids is None:
            continue
        facts.append({key: run.get(key) for key in ("candidate_id", "suite_id", "fold_id", "held_out_subject", "seed", "phase", "scope", "coverage", "source_hash", "spec_hash", "method_revision", "job_id", "diagnostics", "learning_profile")} |
            {"evidence_id": evidence_id, "artifact_id": artifact["artifact_id"],
             "artifact_sha256": artifact["sha256"], "final_test_accessed": False})
    return facts


def verified_facts(camp: Path, state: dict, *, job_ids=None, target_ids=None, suite_ids=None, max_facts=12) -> list[dict]:
    facts = []
    current = None
    for row in reversed(state.get("evidence") or []):
        if row.get("kind") not in {"diagnostic_bundle", "learning_profile"}:
            continue
        for ref in row.get("artifact_refs") or []:
            try:
                artifact, report = _read_verified_report(camp, ref)
                if current is None:
                    current = sources(camp, state, target_ids=target_ids)
                facts.extend(_report_facts(report, artifact, row["evidence_id"], current,
                    job_ids=job_ids, target_ids=target_ids, suite_ids=suite_ids))
            except (OSError, ValueError, KeyError, TypeError):
                continue
        if facts:
            break
    return facts[:max_facts]


def select_verified_facts(camp: Path, state: dict, plan: dict, artifact_ref: str, *, max_facts=12,
                          verified_report: tuple[dict, dict] | None = None) -> dict:
    """Select current task sources before the card cap, without rewriting facts.

    Explicit native refs and scope filters precede recency. The bounded card is
    an annotation window into the complete immutable diagnostic artifact.
    """
    max_facts = max(0, min(int(max_facts), 12))
    proposal = plan.get("proposal") or {}
    scope = proposal.get("evaluation_scope") or {}
    target = plan.get("candidate_id") or proposal.get("target_candidate_ref")
    report_row, report = verified_report if verified_report is not None else _read_verified_report(camp, artifact_ref)
    target_ids = {target} if target else None
    current_sources = sources(camp, state, target_ids=target_ids)
    evidence = next((row for row in reversed(state.get("evidence") or [])
        if row.get("kind") in {"diagnostic_bundle", "learning_profile"}
        and artifact_ref in (row.get("artifact_refs") or [])), None)
    candidates = (_report_facts(report, report_row, evidence["evidence_id"], current_sources,
        target_ids=target_ids) if evidence is not None else [])
    preferred_jobs = list(dict.fromkeys(scope.get("job_ids") or []))
    preferred_suites = list(dict.fromkeys(scope.get("suite_ids") or []))
    for ref in proposal.get("required_context_refs") or []:
        if str(ref).startswith("art_"):
            matching = [row for row in state.get("evidence") or [] if ref == row.get("artifact_id")
                or ref == row.get("analysis_artifact_id") or ref in (row.get("artifact_refs") or [])]
            for row in matching:
                if row.get("job_id") and row["job_id"] not in preferred_jobs:
                    preferred_jobs.append(row["job_id"])
                if row.get("suite_id") and row["suite_id"] not in preferred_suites:
                    preferred_suites.append(row["suite_id"])
        elif str(ref).startswith("context://"):
            from .context.sources import ContextSourceRegistry
            registry = ContextSourceRegistry(camp)
            identity = registry.get(ref, role="research_planner").identity
            if identity.get("job_id") and identity["job_id"] not in preferred_jobs:
                preferred_jobs.append(identity["job_id"])
            if identity.get("suite_id") and identity["suite_id"] not in preferred_suites:
                preferred_suites.append(identity["suite_id"])
    # A declared job/suite scope restricts the task; refs provide precedence
    # among its matching sources. Unknown requested identities stay missing.
    scoped = [row for row in candidates if
        (not scope.get("job_ids") or row.get("job_id") in scope["job_ids"])
        and (not scope.get("suite_ids") or row.get("suite_id") in scope["suite_ids"])
        and (scope.get("seed") is None or row.get("seed") == scope["seed"])
        and (not scope.get("folds") or row.get("fold_id") in scope["folds"])
        and (not scope.get("subjects") or row.get("held_out_subject") in scope["subjects"])
        and (not scope.get("source_hash") or row.get("source_hash") == scope["source_hash"])
        and (not scope.get("spec_hash") or row.get("spec_hash") == scope["spec_hash"])]
    recency = {(row.get("job_id"), row.get("suite_id"), row.get("fold_id")): index
               for index, row in enumerate(current_sources)}
    scoped.sort(key=lambda row: (
        0 if row.get("job_id") in preferred_jobs else 1 if row.get("suite_id") in preferred_suites else 2,
        -recency.get((row.get("job_id"), row.get("suite_id"), row.get("fold_id")), -1)))
    selected = scoped[:max_facts]
    for row in selected:
        row["source_ref"] = artifact_ref
        row["source_identity_missing"] = [key for key in ("source_hash", "spec_hash") if row.get(key) is None]
    delivered_jobs = {row.get("job_id") for row in selected}
    delivered_suites = {row.get("suite_id") for row in selected}
    missing = ["requested_job:" + str(job) for job in preferred_jobs if job not in delivered_jobs]
    missing += ["requested_suite:" + str(suite) for suite in preferred_suites if suite not in delivered_suites]
    missing += ["requested_subject:" + str(subject) for subject in scope.get("subjects") or []
                if subject not in {row.get("held_out_subject") for row in selected}]
    if len(scoped) > len(selected):
        missing.append("historical_sources_outside_card_window")
    if not selected:
        missing.append("verified_scoped_diagnostics")
    return {"facts": selected, "selection": {"selection_basis": "explicit_refs_and_task_scope_then_latest_related",
        "native_report_source_count": len(report.get("runs") or []), "target_source_count": len(candidates),
        "scoped_source_count": len(scoped), "delivered_source_count": len(selected),
        "omitted_scoped_source_count": max(0, len(scoped) - len(selected)),
        "outside_scope_source_count": max(0, len(candidates) - len(scoped)),
        "selected_job_ids": [row.get("job_id") for row in selected], "missing": missing,
        "complete_source_read_ref": artifact_ref, "complete_source_sha256": report_row["sha256"],
        "scope_authority": report["diagnostic_scope"]}}
