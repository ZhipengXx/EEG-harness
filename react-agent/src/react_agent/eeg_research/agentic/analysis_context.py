"""Read accepted run evidence for the Analyst without generating new diagnostics.

Measurement owns validation of the results. This collector consumes that accepted
snapshot, checks the small files it opens against the recorded bindings, and reads
each source once per collection. No checkpoint, EEG data, model or artifact writer
is used. The returned source bindings belong to the local record, not the prompt.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


DEVELOPMENT_SCOPE = "source_subject_development"


def _unavailable(reason: str) -> dict:
    return {"status": "unavailable", "reason": reason}


class _Sources:
    def __init__(self, camp: Path):
        self.camp = Path(camp).resolve()
        self.raw: dict[Path, bytes] = {}
        self.source_catalog: dict[str, dict] = {}
        self.source_bindings: dict[str, dict] = {}

    def path(self, path: str | Path) -> Path:
        target = Path(path).resolve()
        if not target.is_relative_to(self.camp):
            raise ValueError("analysis_source_outside_campaign")
        return target

    def read(self, path: str | Path) -> bytes:
        target = self.path(path)
        if target not in self.raw:
            try:
                self.raw[target] = target.read_bytes()
            except OSError as exc:
                raise ValueError("analysis_source_unavailable:" + target.name) from exc
        return self.raw[target]

    def document(self, path: str | Path, *, expected_sha: str | None = None) -> dict:
        raw = self.read(path)
        if expected_sha is not None and hashlib.sha256(raw).hexdigest() != expected_sha:
            raise ValueError("analysis_source_binding_changed:" + Path(path).name)
        try:
            body = json.loads(raw)
        except (ValueError, UnicodeError) as exc:
            raise ValueError("analysis_source_json_invalid:" + Path(path).name) from exc
        if not isinstance(body, dict):
            raise ValueError("analysis_source_not_object:" + Path(path).name)
        return body

    def cite(self, path: str | Path, kind: str, scope: str) -> str:
        target = self.path(path)
        sha = hashlib.sha256(self.read(target)).hexdigest()
        # Equal contents at different run paths are different sources.
        identity = hashlib.sha256((str(target) + "\0" + sha).encode()).hexdigest()
        ref = "analysis_source_" + identity[:24]
        self.source_catalog.setdefault(ref, {"source_ref": ref, "content_kind": kind,
                                              "scientific_scope": scope})
        self.source_bindings[ref] = {"path": str(target), "sha256": sha}
        return ref


def _learning(sources: _Sources, job: Path) -> dict:
    path = job / "history.jsonl"
    if not path.is_file():
        return {**_unavailable("history_not_exported"), "scope": DEVELOPMENT_SCOPE,
                "points": [], "read_coverage": {"valid_rows": 0, "complete": False}}
    raw = sources.read(path)
    points, errors = [], []
    lines = raw.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        if not line.endswith(b"\n"):
            errors.append({"kind": "incomplete_tail", "line": index + 1})
            continue
        try:
            row = json.loads(line)
        except (ValueError, UnicodeError):
            errors.append({"kind": "invalid_json_row", "line": index + 1})
            continue
        if not isinstance(row, dict):
            errors.append({"kind": "non_object_row", "line": index + 1})
            continue
        points.append(row)
    ref = sources.cite(path, "development_learning_curve", DEVELOPMENT_SCOPE)
    return {"status": "partial" if errors else "ready" if points else "unavailable",
            "scope": DEVELOPMENT_SCOPE, "points": points,
            "read_coverage": {"valid_rows": len(points), "complete": not errors and bool(points),
                              "truncated": False, "diagnostics": errors}, "source_ref": ref}


def _config(spec: dict) -> dict:
    result = {key: copy.deepcopy(spec[key]) for key in ("model", "objective", "transform", "training", "recipe") if key in spec}
    policy = spec.get("negative_policy", (spec.get("recipe") or {}).get("negative_sampling_policy"))
    if policy is not None:
        result["negative_sampling_policy"] = policy
    return result


def _run(sources: _Sources, *, arm: str, job: Path, identity: dict, freeze: dict | None = None,
         freeze_path: Path | None = None) -> dict:
    job = sources.path(job)
    spec_path = job / "frozen_run_spec.json"
    refs = []
    expected = None
    if freeze is not None:
        dependencies = {str(sources.path(row["path"])): row["sha256"] for row in freeze.get("dependencies") or []
                        if Path(row["path"]).name == "frozen_run_spec.json"}
        expected = dependencies.get(str(spec_path))
        if not expected:
            raise ValueError("analysis_frozen_config_binding_missing")
        refs.append(sources.cite(freeze_path, "frozen_fold_development_evidence", DEVELOPMENT_SCOPE))
    spec = sources.document(spec_path, expected_sha=expected)
    for key, expected_value in (("target_id", identity.get("candidate_id")), ("seed", identity.get("seed"))):
        if key in spec and expected_value is not None and spec[key] != expected_value:
            raise ValueError("analysis_frozen_config_identity_mismatch:" + key)
    identity = copy.deepcopy(identity)
    for key, source_key in (("source_hash", "source_manifest_hash"), ("spec_hash", "spec_hash")):
        if spec.get(source_key) is not None:
            if key in identity and identity[key] != spec[source_key]:
                raise ValueError("analysis_frozen_config_identity_mismatch:" + key)
            identity[key] = spec[source_key]
    refs.append(sources.cite(spec_path, "frozen_training_config", "training_configuration"))
    if freeze is not None:
        diagnostics = copy.deepcopy(freeze.get("diagnostics") or _unavailable("diagnostics_not_exported"))
        selected = copy.deepcopy(freeze.get("selected_checkpoint") or _unavailable("checkpoint_selection_not_exported"))
        metrics = copy.deepcopy(freeze.get("development_metrics") or _unavailable("development_metrics_not_exported"))
    else:
        def optional(name, kind):
            path = job / name
            if not path.is_file():
                return _unavailable(name + "_not_exported")
            try:
                value = sources.document(path)
            except ValueError:
                return _unavailable(name + "_invalid")
            refs.append(sources.cite(path, kind, DEVELOPMENT_SCOPE))
            return copy.deepcopy(value)
        diagnostics = optional("diagnostic_summary.json", "development_diagnostics")
        selected = optional("selected_checkpoint.json", "development_checkpoint_selection")
        metrics = optional("metrics.json", "development_metrics")
    dynamics = (diagnostics.get("items") or {}).get("training_dynamics") or {}
    execution = copy.deepcopy(dynamics.get("training_execution") or _unavailable("training_execution_not_exported"))
    learning = _learning(sources, job)
    from .suite_diagnostics import learning_profile
    profile = learning_profile(learning["points"], selected, read_coverage=learning["read_coverage"])
    from .analysis_contract import loss_term_profiles
    terms = loss_term_profiles(learning["points"], selected)
    if terms:
        profile["loss_terms"] = terms
    if learning.get("source_ref"):
        refs.append(learning["source_ref"])
        profile["source_ref"] = learning["source_ref"]
    return {"arm": arm, **copy.deepcopy(identity), "job_id": job.name,
            "config": _config(spec), "training_execution": execution,
            "development_diagnostics": diagnostics, "checkpoint_selection": selected,
            "development_metrics": metrics, "development_learning": learning,
            "learning_profile": profile,
            "diagnostic_scope": DEVELOPMENT_SCOPE, "source_refs": list(dict.fromkeys(refs))}


def _score_runs(sources: _Sources, scores: list[dict], arm: str) -> list[dict]:
    runs = []
    seen = set()
    for score in scores:
        if not score.get("freeze_ref") or not score.get("freeze_sha256"):
            raise ValueError("analysis_score_freeze_binding_missing")
        path = sources.path(score["freeze_ref"])
        freeze = sources.document(path, expected_sha=score["freeze_sha256"])
        if arm == "baseline" and freeze.get("candidate_id") != "baseline":
            raise ValueError("analysis_matched_control_identity_mismatch")
        for key in ("candidate_id", "suite_hash", "fold_id", "seed", "checkpoint_hash"):
            if key in score and freeze.get(key) != score[key]:
                raise ValueError("analysis_score_freeze_identity_mismatch:" + key)
        identity = {key: copy.deepcopy(freeze[key]) for key in (
            "candidate_id", "suite_id", "fold_id", "held_out_subject", "seed", "fidelity", "method_revision") if key in freeze}
        key = (identity.get("suite_id"), identity.get("fold_id"), identity.get("seed"))
        if key in seen:
            raise ValueError("analysis_duplicate_fold")
        seen.add(key)
        runs.append(_run(sources, arm=arm, job=Path(freeze["job_dir"]), identity=identity,
                         freeze=freeze, freeze_path=path))
    return runs


def _confirmation_control(sources: _Sources, state: dict, latest: dict, *,
                          verified_control: dict | None = None) -> tuple[dict, str | None] | None:
    comparison = latest.get("comparison") or {}
    control_id = latest.get("control_id") or comparison.get("control_run_id")
    suite_hash = comparison.get("control_suite_hash")
    def checked(body, expected_hash):
        if not isinstance(body, dict) or not body.get("scores"):
            raise ValueError("analysis_matched_control_unavailable")
        if expected_hash and any(score.get("suite_hash") != expected_hash for score in body["scores"]):
            raise ValueError("analysis_control_suite_mismatch")
        return body
    if verified_control is not None and (verified_control.get("candidate_id") != "baseline"
            or (control_id and verified_control.get("evidence_id") != control_id)
            or (suite_hash and verified_control.get("suite_hash") != suite_hash)):
        raise ValueError("analysis_control_suite_mismatch")
    rows = [row for row in ([verified_control] if verified_control is not None else state.get("evidence") or [])
            if row.get("candidate_id") == "baseline"
            and (row.get("evidence_id") == control_id if control_id else suite_hash and row.get("suite_hash") == suite_hash)]
    if rows:
        row = rows[0]
        if suite_hash and row.get("suite_hash") != suite_hash:
            raise ValueError("analysis_control_suite_mismatch")
        if row.get("suite_ref"):
            wrapper = sources.document(row["suite_ref"], expected_sha=row.get("result_hash"))
            aggregate = checked(wrapper.get("aggregate"), suite_hash or row.get("suite_hash"))
            return aggregate, sources.cite(row["suite_ref"], "matched_control_measurement", "method_development_benchmark")
        if isinstance(row.get("aggregate"), dict):
            return checked(row["aggregate"], suite_hash or row.get("suite_hash")), None
    if suite_hash:
        for path in sorted((sources.camp / "suites").glob("*/suite_manifest.json")):
            manifest = sources.document(path)
            if manifest.get("suite_hash") == suite_hash and (manifest.get("method") or {}).get("candidate_id") == "baseline":
                aggregate_path = path.with_name("aggregate.json")
                aggregate = checked(sources.document(aggregate_path).get("aggregate"), suite_hash)
                return aggregate, sources.cite(aggregate_path, "matched_control_measurement", "method_development_benchmark")
    if control_id or suite_hash or comparison.get("comparable"):
        raise ValueError("analysis_matched_control_unavailable")
    return None


def _comparison_observation(sources: _Sources, observation_id: str) -> dict:
    """Read the exact content-addressed observation, without revalidating runs."""
    from .embedding import content_hash

    body = sources.document(sources.camp / "experiment_observations" / (observation_id + ".json"))
    original = {key: value for key, value in body.items() if key not in {"observation_id", "content_digest"}}
    if (body.get("observation_id") != observation_id or body.get("origin_campaign_id") != sources.camp.name
            or "obs_" + content_hash(original)[:24] != observation_id):
        raise ValueError("historical_observation_identity_mismatch")
    return body


def comparison_target_context(camp: Path, observation_ids) -> list[dict]:
    """Give Designer comparison identities and scope, without historical prose."""
    sources = _Sources(camp)
    result = []
    for observation_id in dict.fromkeys(observation_ids):
        try:
            body = _comparison_observation(sources, observation_id)
            result.append({"observation_id": observation_id,
                "candidate_id": (body.get("identity") or {}).get("candidate_id"),
                "evaluation_scope": copy.deepcopy(body.get("evaluation_scope") or {})})
        except (OSError, ValueError, KeyError, TypeError) as exc:
            result.append({"observation_id": observation_id, "status": "unavailable", "reason": str(exc)})
    return result


def _historical_values(value):
    """The saved Analysis file is the source of this delivered factual slice."""
    if isinstance(value, list):
        return [_historical_values(item) for item in value]
    if isinstance(value, dict):
        return {key: _historical_values(item) for key, item in value.items()
                if key not in {"source_ref", "source_refs"}}
    return copy.deepcopy(value)


def _historical_measurement(facts: dict, observation: dict, summary: dict) -> dict:
    original = facts.get("measurement") or {}
    recorded = observation.get("native_facts") or {}
    metrics = recorded.get("metrics") or {}
    result = {}
    for metric in ("top1", "top5"):
        keys = ("mean_" + metric, "screening_mean_" + metric, "fixed_bank_" + metric)
        value = next((owner[key] for owner in (original, metrics, summary.get("measurement") or {})
                      for key in keys if owner.get(key) is not None), None)
        if value is not None:
            result["mean_" + metric] = copy.deepcopy(value)
    scores = (original.get("scores") or recorded.get("folds")
              or (summary.get("measurement") or {}).get("paired_folds")
              or recorded.get("paired_scores") or [])
    folds = []
    for score in scores:
        row = {key: copy.deepcopy(score[key]) for key in ("fold_id", "held_out_subject", "seed") if key in score}
        if score.get("subject") is not None:
            row.setdefault("held_out_subject", score["subject"])
        row.setdefault("seed", recorded.get("seed"))
        values = score.get("metrics") or score
        for metric in ("top1", "top5"):
            value = values.get(metric, values.get("candidate_" + metric))
            if value is not None:
                row[metric] = copy.deepcopy(value)
        row.update({key: copy.deepcopy(values[key]) for key in ("query_count", "candidate_count") if key in values})
        folds.append(row)
    if folds:
        result["fold_results"] = folds
    return result


def _historical_conditions(body: dict, facts: dict) -> dict:
    from .analysis_contract import comparison_conditions
    recorded = body.get("native_facts") or {}
    applicability = body.get("applicability") or {}
    owners = [recorded, body.get("evaluation_scope") or {}, facts.get("run_identity") or {},
              facts.get("comparison_conditions") or {}, body.get("identity") or {}, applicability,
              facts.get("coverage") or {}]
    return comparison_conditions(*owners)


def _current_history_arm(facts: dict, candidate_id: str, runs: list[dict], latest: dict) -> str | None:
    identity = facts.get("run_identity") or {}
    if (identity.get("evidence_id") and identity.get("evidence_id") == latest.get("evidence_id")
            and candidate_id == latest.get("candidate_id")):
        return "baseline" if candidate_id == "baseline" else "candidate"
    previous = [run for run in facts.get("training_runs") or [] if run.get("candidate_id") == candidate_id]
    current = [run for run in runs if run.get("candidate_id") == candidate_id]
    keys = ("job_id", "suite_id", "fold_id", "seed", "fidelity", "source_hash", "spec_hash")
    coordinates = lambda rows: {tuple(row.get(key) for key in keys) for row in rows}
    if (previous and current and all(run.get("job_id") for run in previous + current)
            and coordinates(previous) == coordinates(current)):
        return current[0]["arm"]
    return None


def _historical_comparisons(sources: _Sources, observation_ids, runs: list[dict], latest: dict) -> list[dict]:
    from .artifacts import lookup
    from .analysis_contract import build_analysis_summary, compact_execution_facts

    result, seen_sources = [], {}
    for observation_id in dict.fromkeys(observation_ids):
        try:
            body = _comparison_observation(sources, observation_id)
            candidate_id = (body.get("identity") or {}).get("candidate_id")
            base = {"observation_id": observation_id, "candidate_id": candidate_id}
            if not (body.get("native_facts") or {}).get("metrics"):
                raise ValueError("historical_measurement_unavailable")
            analysis = next((ref for ref in body.get("source_refs") or []
                             if ref.get("kind") in {"analysis", "result_analysis", "analysis_output", "role_result"}), None)
            facts, summary = {}, {}
            source_path = sources.camp / "experiment_observations" / (observation_id + ".json")
            if analysis is not None:
                artifact = lookup(sources.camp, analysis["artifact_id"])
                if not artifact or not analysis.get("sha256"):
                    raise ValueError("historical_analysis_binding_unavailable")
                source_path = artifact["path"]
                wrapper = sources.document(source_path, expected_sha=analysis["sha256"])
                record = wrapper.get("payload") or wrapper
                facts = record.get("native_facts") or {}
                if (record.get("origin_plan_id") != body.get("plan_id")
                        or (facts.get("run_identity") or {}).get("candidate_id") != candidate_id):
                    raise ValueError("historical_analysis_identity_mismatch")
                for run in facts.get("training_runs") or []:
                    if run.get("candidate_id") != candidate_id:
                        continue
                    for key in ("source_hash", "spec_hash"):
                        expected = (body.get("identity") or {}).get(key)
                        if expected is not None and run.get(key) is not None and expected != run[key]:
                            raise ValueError("historical_analysis_revision_mismatch")
                arm = _current_history_arm(facts, candidate_id, runs, latest)
                if arm:
                    result.append({**base, "status": "already_in_current_evidence", "arm": arm})
                    continue
            ref = sources.cite(source_path, "historical_experiment_facts", "recorded_experiment")
            if ref in seen_sources:
                result.append({**base, "status": "shared_history_evidence", "same_as_observation_id": seen_sources[ref]})
                continue
            seen_sources[ref] = observation_id
            if facts:
                # Reproject saved native facts so current and historical runs
                # use the same compact execution and population definitions.
                summary = (build_analysis_summary(facts) if facts.get("training_runs") else
                           compact_execution_facts(facts.get("analysis_summary") or build_analysis_summary(facts)))
            diagnostic = summary.get("diagnostic_summary") or {}
            row = {**base, "status": "available", "source_ref": ref,
                "intervention": body.get("intervention"),
                "measurement": _historical_measurement(facts, body, summary),
                "conditions": _historical_conditions(body, facts),
                # Saved fold profiles already express the historical trends;
                # replaying every epoch and batch would repeat those facts.
                "training_summary": [{key: item[key] for key in ("arm", "candidate_id", "seed", "fold_count",
                    "loss_semantics", "configuration", "training_execution", "fold_profiles", "checkpoint_selection") if key in item}
                    for item in summary.get("training_summary") or [] if item.get("candidate_id") == candidate_id],
                "diagnostic_summary": {key: [item for item in diagnostic.get(key) or []
                                            if item.get("candidate_id") == candidate_id]
                                       for key in ("groups", "export_fields") if key in diagnostic}}
            result.append(_historical_values(row))
            result[-1]["source_ref"] = ref
            if analysis is not None:
                # Conditions and fallback measurements come from the already
                # loaded Observation; retain that attribution alongside Analysis.
                result[-1]["observation_source_ref"] = sources.cite(
                    sources.camp / "experiment_observations" / (observation_id + ".json"),
                    "historical_observation", "recorded_experiment")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            result.append({"observation_id": observation_id, "status": "unavailable", "reason": str(exc)})
    return result


def _probe_scientific_summary(probe: dict) -> dict:
    """Keep measured ranges and case definitions, not the full engineering report."""
    from .analysis_contract import _number

    result = {key: copy.deepcopy(probe[key]) for key in (
        "status", "scope", "reference_definition", "gradient_changed", "matches_reference",
        "unique_ids_reduce_to_baseline", "all_same_id_finite_zero_loss") if key in probe}
    groups = {}
    for row in probe.get("probes") or []:
        groups.setdefault(str(row.get("case") or "recorded_probes"), []).append(row)
    cases = []
    for case, rows in sorted(groups.items()):
        ranges = {}
        for key in ("reference_loss", "actual_loss", "loss_abs_difference", "eeg_gradient_max_abs_difference",
                    "scale_gradient_abs_difference", "learnable_objective_gradient_max"):
            values = [row[key] for row in rows if _number(row.get(key))]
            if values:
                ranges[key] = {"n": len(values), "min": min(values), "max": max(values)}
        cases.append({"case": case, "probe_count": len(rows), "measured_ranges": ranges})
    if cases:
        result["cases"] = cases
    return result


def _reviewer_scientific_context(sources: _Sources, runs: list[dict], latest: dict) -> tuple[list[dict], list[dict]]:
    """Reuse current scientific probe evidence, without rerunning implementation checks."""
    candidate = latest.get("candidate_id")
    if not candidate or candidate == "baseline":
        return [], []
    current = [run for run in runs if run.get("candidate_id") == candidate]
    identities = {(run.get("source_hash"), run.get("spec_hash")) for run in current}
    if len(identities) != 1 or any(value is None for value in next(iter(identities))):
        return [], []
    path = sources.camp / "candidates" / str(candidate) / "review.json"
    if not path.is_file():
        return [], []
    try:
        review = sources.document(path)
        binding = review.get("review_binding") or {}
        if (review.get("status") != "ready" or
                (binding.get("source_hash"), binding.get("approved_spec_hash")) not in identities):
            return [], []
        supported = {"objective_semantics_probe", "objective_effectiveness_probe"}
        selected = []
        for issue in review.get("issues") or []:
            if issue.get("severity") != "non_blocking" or issue.get("kind") != "scientific_replan":
                continue
            names = {name for name in supported for ref in issue.get("evidence_refs") or []
                     if str(ref).endswith("/" + name)}
            if names and issue.get("validation_expectation"):
                selected.append((issue, names))
        if not selected or not binding.get("check_ref") or not binding.get("check_sha256"):
            return [], []
        check_path = Path(binding["check_ref"])
        if not check_path.is_absolute():
            check_path = sources.camp / check_path
        checks = sources.document(check_path, expected_sha=binding["check_sha256"])
        names = sorted({name for _, names in selected for name in names if isinstance(checks.get(name), dict)})
        if not names:
            return [], []
        check_ref = sources.cite(check_path, "synthetic_objective_probes", "synthetic_cpu_probe")
        review_ref = sources.cite(path, "reviewer_scientific_question", "model_interpretation")
        facts = [{"probe": name, "source_ref": check_ref, "scope": "synthetic_cpu_probe",
                  "measurements": _probe_scientific_summary(checks[name])} for name in names]
        questions, seen = [], set()
        for issue, referenced in selected:
            question = issue["validation_expectation"]
            if question not in seen and referenced.intersection(names):
                seen.add(question)
                questions.append({"question": question, "source_ref": review_ref,
                                  "authority": "reviewer_hypothesis", "probe_refs": sorted(referenced.intersection(names))})
        return facts, questions
    except (OSError, ValueError, KeyError, TypeError):
        # An optional scientific lead does not change the accepted experiment.
        return [], []


def collect_analysis_evidence(camp: Path, state: dict, latest: dict, *, verified_result: dict | None = None,
                              verified_control: dict | None = None, comparison_observation_ids=(),
                              diagnostic: dict | None = None) -> dict:
    """Collect facts for a newly accepted single run, screening or confirmation.

    ``verified_result`` is the existing native verifier's result. When omitted for
    a suite, invoke that verifier rather than treating arbitrary JSON as measured
    evidence. ``verified_control`` reuses the exact control accepted in that same
    call. The collector does not validate another full campaign or synthesize
    missing diagnostic measurements.
    """
    sources = _Sources(camp)
    matched_control = None
    matched_control_ref = None
    kind = latest.get("kind")
    if kind in {"method_screening", "method_suite"}:
        if verified_result is None:
            if kind == "method_screening":
                from .screening import read_screening
                verified_result = read_screening(sources.camp, latest["suite_id"])
            else:
                from .method_suite import read_aggregate
                verified_result = read_aggregate(sources.camp, latest["suite_id"])
        arm = "baseline" if latest.get("candidate_id") == "baseline" else "candidate"
        runs = _score_runs(sources, verified_result.get("scores") or [], arm)
        if not runs:
            raise ValueError("analysis_accepted_training_runs_missing")
        if kind == "method_screening":
            baseline_scores = verified_result.get("baseline_scores") or []
            if arm == "candidate":
                runs.extend(_score_runs(sources, baseline_scores, "baseline"))
        elif arm == "candidate":
            control = _confirmation_control(sources, state, latest, verified_control=verified_control)
            if control is not None:
                matched_control, matched_control_ref = control
                runs.extend(_score_runs(sources, matched_control.get("scores") or [], "baseline"))
    else:
        rows = [(latest, "baseline" if latest.get("candidate_id") == "baseline" else "candidate")]
        control_id = latest.get("control_id")
        if control_id:
            control = next((row for row in state.get("evidence") or [] if row.get("evidence_id") == control_id), None)
            if control is None or control.get("candidate_id") != "baseline":
                raise ValueError("analysis_matched_control_unavailable")
            rows.append((control, "baseline"))
        runs = []
        for row, arm in rows:
            if not row.get("job_dir"):
                raise ValueError("analysis_job_binding_missing")
            identity = {key: copy.deepcopy(row[key]) for key in ("candidate_id", "seed", "fidelity", "source_hash", "spec_hash") if key in row}
            runs.append(_run(sources, arm=arm, job=Path(row["job_dir"]), identity=identity))
    history = _historical_comparisons(sources, comparison_observation_ids, runs, latest)
    reviewer_facts, reviewer_questions = _reviewer_scientific_context(sources, runs, latest)
    diagnostic_task = None
    if diagnostic is not None:
        from .artifacts import lookup
        artifact = next(filter(None, (lookup(sources.camp, ref)
            for ref in diagnostic.get("artifact_refs") or [])), None)
        if artifact is None:
            raise ValueError("analysis_diagnostic_artifact_missing")
        report = sources.document(artifact["path"], expected_sha=artifact["sha256"])
        scope = report["diagnostic_scope"]
        ref = sources.cite(artifact["path"], "selected_task_diagnostic", scope)
        diagnostic_task = {key: copy.deepcopy(report[key]) for key in (
            "diagnostic_request", "measurement_kind", "diagnostic_scope", "coverage", "missing_outputs", "status", "runs")
            if key in report}
        diagnostic_task.update(source_ref=ref, evidence_id=diagnostic["evidence_id"], additional_training_jobs=0)
    result = {"training_runs": runs, "source_catalog": list(sources.source_catalog.values()),
              "source_bindings": sources.source_bindings}
    if reviewer_facts:
        result["reviewer_probe_facts"] = reviewer_facts
    if reviewer_questions:
        result["reviewer_questions"] = reviewer_questions
    if history:
        result["historical_comparisons"] = history
    if diagnostic_task is not None:
        result["diagnostic_task"] = diagnostic_task
    if matched_control is not None:
        # Reuse the exact control already read for its training runs. Native
        # aggregation consumes these measurements without another suite read.
        result["matched_control_aggregate"] = copy.deepcopy(matched_control)
        if matched_control_ref is not None:
            result["matched_control_source_ref"] = matched_control_ref
    return result
