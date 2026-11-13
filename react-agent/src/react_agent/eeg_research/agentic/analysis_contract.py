"""Staged analysis contract: native facts, the 4-key Analyst reply, delivery check, record.

The program owns numbers, execution state, identities and coverage; the Analyst
owns findings, hypothesis judgement, limitations and the summary. One rule set
serves live replies, recovered replies and the stage review. Pure functions:
no file I/O and no model calls.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

RECORD_VERSION = "eeg_research.analysis_record.v1"
# Envelope/runtime keys around a stored payload; never part of the Analyst reply.
_STRIP = frozenset({"schema_version", "task_id", "attempt_id", "input_digest", "prompt_hash", "status",
    "artifact_refs", "candidate_id", "method_benchmark_origin", "method_screening_origin"})
_TEXT = ("observation", "interpretation", "implication")


def _pick(value, fields):
    return {key: copy.deepcopy(value[key]) for key in fields if key in value} if isinstance(value, dict) else {}


def comparison_conditions(*owners: dict) -> dict:
    """Project recorded comparison coordinates with the same rules for both arms."""
    aliases = {key: (key,) for key in ("scope", "phase", "fidelity", "seed", "dataset", "modality",
        "input_geometry", "evaluation_identity", "selection_manifest_hash", "baseline_suite_hash")}
    aliases.update(fold_ids=("fold_ids", "screening_fold_ids", "folds"),
                   subjects=("subjects", "screening_subjects"))
    result = {}
    for key, names in aliases.items():
        value = next((owner[name] for owner in owners if isinstance(owner, dict) for name in names
            if owner.get(name) is not None and (key not in {"fold_ids", "subjects"}
                or isinstance(owner[name], list) and all(isinstance(item, str) for item in owner[name]))), None)
        if value is not None:
            result[key] = copy.deepcopy(value.upper() if key == "modality" and isinstance(value, str) else value)
    return result


def protocol_scientific_conditions(protocol: dict) -> dict:
    result = _pick(protocol, ("dataset", "modality", "input_geometry"))
    modality = {"eeg": "EEG", "meg": "MEG"}.get(str(result.get("dataset") or "").lower())
    if modality is not None:
        result.setdefault("modality", modality)
    return comparison_conditions(result)


def allowed_refs(measurement_review: dict) -> list[str]:
    """The only references a finding may cite: the measurement review and its native refs."""
    review = measurement_review if isinstance(measurement_review, dict) else {}
    refs = [review.get("artifact_id"), review.get("path"), *(review.get("evidence_refs") or [])]
    return list(dict.fromkeys(ref for ref in refs if isinstance(ref, str) and ref))


def _measurement_score(score: dict) -> dict:
    """Scientific evaluation conditions, without the scorer's storage receipts."""
    result = _pick(score, ("candidate_id", "fold_id", "held_out_subject", "train_subjects", "seed", "fidelity"))
    metrics = score.get("metrics") or {}
    result["metrics"] = _pick(metrics, ("top1", "top5", "query_count", "candidate_count"))
    identity = metrics.get("inference_identity") or {}
    conditions = _pick(identity, ("eval_mode", "augmentation", "evaluator", "input_geometry", "tie_policy"))
    if conditions:
        result["evaluation_conditions"] = conditions
    return result


def _paired_evaluation_conditions(screening: dict) -> list[dict]:
    """Report exact recorded input equalities, never infer unavailable identities."""
    controls = {row.get("fold_id"): row for row in screening.get("baseline_scores") or []}
    result = []
    for score in screening.get("scores") or []:
        control = controls.get(score.get("fold_id"))
        if not control:
            continue
        left = (score.get("metrics") or {}).get("inference_identity") or {}
        right = (control.get("metrics") or {}).get("inference_identity") or {}
        if not left or not right:
            continue
        same = {}
        for label, key in (("held_out_files", "held_out_files"), ("test_image_features", "test_image_cache"),
                           ("evaluated_inputs", "actual_inputs")):
            if left.get(key) is not None and right.get(key) is not None:
                same[label] = left[key] == right[key]
        result.append({"fold_id": score.get("fold_id"), "same_recorded_inputs": same})
    return result


def build_analysis_facts(payload: dict) -> dict:
    """Project native facts and derived summaries once, without reading sources."""
    latest = payload.get("latest") or {}
    screening = payload.get("method_screening") if isinstance(payload.get("method_screening"), dict) else None
    review = payload.get("measurement_review") or {}
    facts = {"run_identity": _pick(latest, ("evidence_id", "candidate_id", "fidelity", "seed", "phase", "scope", "suite_id")),
        "execution": {"evaluation_valid": latest.get("evaluation_valid"),
                      "measurement_review_status": review.get("status")}}
    if facts["run_identity"].get("scope") is None and latest.get("metric_scope") is not None:
        facts["run_identity"]["scope"] = latest["metric_scope"]
    if screening is not None:
        measurement = _pick(screening, ("paired_scores", "screening_mean_top1", "screening_mean_top5",
            "screening_mean_delta_pp", "min_mean_gain_pp", "screening_result", "promotion_decision"))
        baselines = [row.get("baseline_top1") for row in screening.get("paired_scores") or []]
        if baselines and all(isinstance(value, (int, float)) for value in baselines):
            measurement["baseline_mean_top1"] = sum(baselines) / len(baselines)
        facts["execution"]["screening_result"] = screening.get("screening_result")
        # Previously these scientific conditions lived only in the duplicated
        # method_screening subtree. Keep them when that subtree is removed.
        if screening.get("scores"):
            measurement["scores"] = [_measurement_score(row) for row in screening["scores"]]
        if screening.get("baseline_scores"):
            measurement["baseline_scores"] = [_measurement_score(row) for row in screening["baseline_scores"]]
            top5 = [(row.get("metrics") or {}).get("top5") for row in screening["baseline_scores"]]
            if top5 and all(_number(value) for value in top5):
                measurement["baseline_mean_top5"] = math.fsum(top5) / len(top5)
        conditions = _paired_evaluation_conditions(screening)
        if conditions:
            measurement["paired_evaluation_conditions"] = conditions
    elif isinstance(payload.get("method_suite"), dict):
        suite = payload["method_suite"]
        measurement = _pick(suite, ("mean_top1", "mean_top5", "std_top1", "std_top5", "ddof", "coverage"))
        if suite.get("scores"):
            measurement["scores"] = [_measurement_score(row) for row in suite["scores"]]
        if "std_top1" in measurement or "std_top5" in measurement:
            measurement["variation_unit"] = "across_subject_folds"
        if payload.get("comparison"):
            measurement["comparison"] = copy.deepcopy(payload["comparison"])
        control = (payload.get("analysis_evidence") or {}).get("matched_control_aggregate")
        if isinstance(control, dict):
            measurement["baseline"] = _pick(control, ("mean_top1", "mean_top5", "std_top1", "std_top5", "ddof", "coverage"))
            measurement["baseline_scores"] = [_measurement_score(row) for row in control.get("scores") or []]
            ref = (payload.get("analysis_evidence") or {}).get("matched_control_source_ref")
            if ref:
                measurement["baseline"]["source_ref"] = ref
    else:
        measurement = {**_pick(latest, ("fixed_bank_top1", "delta_vs_control_pp", "control_id")),
                       **({"comparison": copy.deepcopy(payload["comparison"])} if payload.get("comparison") else {})}
    facts["measurement"] = measurement
    coverage = _pick(payload.get("verified_run_coverage"), ("complete_fold_count", "required_fold_count",
        "full_required_fold_count", "independent_seed_count", "replicated"))
    coverage.update(_pick(screening, ("screening_fold_ids", "screening_subjects")))
    if "independent_seed_count" in payload:
        coverage.setdefault("independent_seed_count", payload["independent_seed_count"])
    facts["coverage"] = coverage
    # Not "scope": context capture treats a top-level "scope" as a visibility label.
    context = _pick(payload, ("science_scope", "scientific_unit", "evaluation_population", "limitations"))
    if screening is not None and "science_scope" in screening:
        context["science_scope"] = copy.deepcopy(screening["science_scope"])
    facts["evaluation_context"] = context
    conditions = comparison_conditions(payload.get("scientific_conditions") or {})
    if conditions:
        facts["comparison_conditions"] = conditions
    refs = allowed_refs(review)
    evidence = payload.get("analysis_evidence") or {}
    if evidence:
        facts["training_runs"] = copy.deepcopy(evidence.get("training_runs") or [])
        facts["source_catalog"] = copy.deepcopy(evidence.get("source_catalog") or [])
        # Bind complete source identities locally. The provider projection
        # delivers only scientific facts and short references.
        facts["source_bindings"] = copy.deepcopy(evidence.get("source_bindings") or {})
        if evidence.get("diagnostic_task"):
            facts["diagnostic_task"] = copy.deepcopy(evidence["diagnostic_task"])
        refs.extend(row["source_ref"] for row in facts["source_catalog"]
                    if isinstance(row, dict) and row.get("source_ref") in facts["source_bindings"])
        # New scientific delivery cites native evidence/artifact IDs and the
        # collector's short refs. Receipt paths stay in the local review/bindings.
        refs = [ref for ref in refs if not Path(ref).is_absolute()]
    diagnostics = {key: copy.deepcopy(payload[key]) for key in
        ("objective_semantic_findings", "objective_effectiveness_findings") if payload.get(key)}
    if evidence.get("reviewer_probe_facts"):
        diagnostics["implementation_probes"] = copy.deepcopy(evidence["reviewer_probe_facts"])
    if diagnostics:
        facts["scientific_diagnostics"] = diagnostics
    facts["allowed_evidence_refs"] = list(dict.fromkeys(refs))
    # Build once before request/task identity. Compiler and format retries only
    # select this view; full training_runs still supply the local Memory record.
    facts["analysis_summary"] = build_analysis_summary(facts)
    if evidence.get("historical_comparisons"):
        facts["historical_comparisons"] = _historical_comparisons(facts, evidence["historical_comparisons"])
    return facts


def diagnostic_science_view(value: Any) -> Any:
    """Strip the existing diagnostic export bookkeeping, never scientific config."""
    if isinstance(value, list):
        return [diagnostic_science_view(item) for item in value]
    if not isinstance(value, dict):
        return copy.deepcopy(value)
    local = {"schema_version", "source_sha256", "source_hash", "spec_hash", "config_hash",
        "checkpoint_hash", "protocol_hash", "evaluator_hash", "evaluation_key", "fingerprint", "artifact_sha256",
        "validation_identity", "training_inference_identity_fingerprint", "dependencies",
        "source_bindings", "created_at", "recorded_at", "candidate_suite_hash", "control_suite_hash"}
    return {key: Path(item).name if key in {"checkpoint", "checkpoint_id"} and isinstance(item, str)
            else diagnostic_science_view(item) for key, item in value.items() if key not in local}


def _number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def compact_execution_facts(value: Any) -> Any:
    """Copy execution facts, encoding only identical finite objective replicas.

    The count makes a uniform numeric replica array recoverable. Other arrays,
    including scale arguments with their owner/usage, retain their exact shape.
    """
    if isinstance(value, list):
        return [compact_execution_facts(item) for item in value]
    if not isinstance(value, dict):
        return copy.deepcopy(value)
    result = {key: compact_execution_facts(item) for key, item in value.items()}
    samples = result.get("objective_samples")
    if isinstance(samples, list):
        for sample in samples:
            replicas = sample.get("replica_values") if isinstance(sample, dict) else None
            if not isinstance(replicas, dict):
                continue
            for metric, values in replicas.items():
                if (isinstance(values, list) and values
                        and all(type(item) is int or type(item) is float and math.isfinite(item) for item in values)
                        and all(_canonical(item) == _canonical(values[0]) for item in values[1:])):
                    replicas[metric] = {"value": values[0], "replica_count": len(values)}
    return result


def _common_fields(rows: list[dict]) -> dict:
    """Exact structural factoring: unshared fields remain with their own fold."""
    if not rows:
        return {}
    common = {}
    for key, value in rows[0].items():
        if any(key not in row for row in rows[1:]):
            continue
        values = [row[key] for row in rows]
        if all(type(item) is type(value) and item == value for item in values):
            common[key] = copy.deepcopy(value)
        elif all(isinstance(item, dict) for item in values):
            nested = _common_fields(values)
            if nested:
                common[key] = nested
    return common


def _without_common(row: dict, common: dict) -> dict:
    remaining = {}
    for key, value in row.items():
        if key not in common:
            remaining[key] = copy.deepcopy(value)
        elif isinstance(value, dict) and isinstance(common[key], dict):
            nested = _without_common(value, common[key])
            if nested:
                remaining[key] = nested
    return remaining


def _factored(rows: list[tuple[dict, dict]]) -> dict:
    common = _common_fields([body for _, body in rows])
    differences = [{**identity, "fields": rest} for identity, body in rows
                   if (rest := _without_common(body, common))]
    return {"common": common, "per_fold_differences": differences}


def _run_coordinate(run: dict) -> dict:
    return _pick(run, ("fold_id", "held_out_subject", "seed"))


def _score_key(row: dict) -> tuple:
    return row.get("fold_id"), row.get("held_out_subject"), row.get("seed")


def _measurement_summary(facts: dict) -> dict:
    original = facts.get("measurement") or {}
    if not original.get("scores"):
        # A single run or an already compact native screening has no full table.
        return diagnostic_science_view(original)
    measurement = diagnostic_science_view({key: value for key, value in original.items()
        if key not in {"scores", "baseline_scores", "paired_scores", "paired_evaluation_conditions"}})
    candidates = sorted(original["scores"], key=lambda row: _canonical(_score_key(row)))
    controls = {_score_key(row): row for row in original.get("baseline_scores") or []}
    arm = "baseline" if (facts.get("run_identity") or {}).get("candidate_id") == "baseline" else "candidate"
    measurement["arm"] = arm
    pairs, settings = [], []
    for score in candidates:
        coord = _pick(score, ("fold_id", "held_out_subject", "seed"))
        row = {**coord, **{f"{arm}_{key}": value for key, value in
                          (score.get("metrics") or {}).items() if key in {"top1", "top5"}}}
        settings.append(({"arm": arm, **coord}, {key: value for key, value in score.items()
            if key not in {"metrics", "fold_id", "held_out_subject", "seed", "candidate_id"}} |
            _pick(score.get("metrics"), ("query_count", "candidate_count"))))
        control = controls.get(_score_key(score))
        if control:
            settings.append(({"arm": "baseline", **coord}, {key: value for key, value in control.items()
                if key not in {"metrics", "fold_id", "held_out_subject", "seed", "candidate_id"}} |
                _pick(control.get("metrics"), ("query_count", "candidate_count"))))
            for metric in ("top1", "top5"):
                left, right = (score.get("metrics") or {}).get(metric), (control.get("metrics") or {}).get(metric)
                if right is not None:
                    row["baseline_" + metric] = right
                if _number(left) and _number(right):
                    native_delta = ((original.get("comparison") or {}).get("subject_deltas_pp") or {}).get(score.get("held_out_subject"))
                    row["delta_" + metric + "_pp"] = (native_delta if metric == "top1" and _number(native_delta)
                                                         else 100 * (left - right))
        pairs.append(row)
    measurement["fold_results" if arm == "baseline" else "paired_folds"] = pairs
    measurement["evaluation_conditions"] = _factored(settings)
    if original.get("paired_evaluation_conditions"):
        measurement["recorded_input_equalities"] = copy.deepcopy(original["paired_evaluation_conditions"])
    comparison = measurement.get("comparison") or {}
    # Per-subject differences now occur once in paired_folds. Keep an existing
    # native delta when present instead of recalculating the accepted comparison.
    fully_paired = bool(pairs) and all("baseline_top1" in row and "candidate_top1" in row for row in pairs)
    if fully_paired:
        comparison.pop("subject_deltas_pp", None)
    elif original.get("paired_scores"):
        measurement["paired_scores"] = copy.deepcopy(original["paired_scores"])
    for metric in ("top1", "top5"):
        values = [row["delta_" + metric + "_pp"] for row in pairs if "delta_" + metric + "_pp" in row]
        if values:
            measurement.setdefault("paired_direction_counts", {})[metric] = {
                "n": len(values), "improved": sum(value > 0 for value in values),
                "unchanged": sum(value == 0 for value in values), "worsened": sum(value < 0 for value in values)}
        # Only all matched folds support a new overall paired statistic.
        if len(values) == len(pairs) and values and comparison.get("comparable") is not False:
            key = "paired_mean_delta_" + metric + "_pp"
            if metric == "top1" and _number(comparison.get("paired_mean_delta_pp", comparison.get("delta_pp"))):
                measurement[key] = comparison.get("paired_mean_delta_pp", comparison.get("delta_pp"))
            elif metric == "top1" and _number(original.get("screening_mean_delta_pp")):
                measurement[key] = original["screening_mean_delta_pp"]
            else:
                candidate_mean = original.get("mean_" + metric, original.get("screening_mean_" + metric))
                baseline_mean = (original.get("baseline") or {}).get("mean_" + metric, original.get("baseline_mean_" + metric))
                measurement[key] = (100 * (candidate_mean - baseline_mean)
                                    if _number(candidate_mean) and _number(baseline_mean)
                                    else math.fsum(values) / len(values))
    measurement.update(metric_unit="fraction_0_to_1", delta_unit="percentage_points")
    return measurement


def _distribution(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    mean = math.fsum(values) / len(values)
    return {"n": len(values), "mean": mean,
            "std": math.sqrt(math.fsum((value - mean) ** 2 for value in values) / len(values)),
            "min": min(values), "max": max(values)}


def loss_term_profiles(points: list[dict], selected: dict) -> dict:
    """Compact recorded weighted contributions; full per-epoch terms stay local."""
    names = sorted({name for point in points for name in (point.get("loss_terms") or {})})
    result = {}
    for name in names:
        rows = [(point, (point.get("loss_terms") or {}).get(name)) for point in points]
        rows = [(point, value) for point, value in rows
                if _number(point.get("epoch")) and _number(value)]
        if not rows:
            continue
        tail = rows[-10:]
        body = {"first_epoch": rows[0][0]["epoch"], "last_epoch": rows[-1][0]["epoch"],
                "first": rows[0][1], "last": rows[-1][1], "recorded_epochs": len(rows),
                "tail_epochs": len(tail), "tail_mean": math.fsum(value for _, value in tail) / len(tail)}
        selected_rows = [value for point, value in rows if point["epoch"] == selected.get("best_epoch")]
        if len(selected_rows) == 1:
            body["selected_epoch_value"] = selected_rows[0]
        if all(isinstance((point.get("loss_term_batches") or {}).get(name), int)
               and isinstance(point.get("train_batch_count"), int) for point, _ in rows):
            body["observed_batches"] = sum(point["loss_term_batches"][name] for point, _ in rows)
            body["recorded_epoch_batches"] = sum(point["train_batch_count"] for point, _ in rows)
        result[name] = body
    return result


def _curve_summary(runs: list[dict]) -> tuple[list[dict], list[dict]]:
    """All recorded epochs; finite samples counted separately for each metric."""
    metrics = ("train_loss", "fixed_bank_top1", "fixed_bank_top5", "val_top1", "val_top5")
    points_by_run = [(run, (run.get("development_learning") or {}).get("points") or []) for run in runs]
    aliases = {}
    for alias, canonical in (("val_top1", "fixed_bank_top1"), ("val_top5", "fixed_bank_top5")):
        present = [point for _, points in points_by_run for point in points if alias in point]
        if present and all(_number(point.get(alias)) and _number(point.get(canonical))
                           and abs(point[alias] - point[canonical]) <= 1e-6 * max(1.0, abs(point[canonical]))
                           for point in present):
            aliases[alias] = canonical
    curves, ambiguous = [], {}
    for metric in metrics:
        if metric in aliases:
            continue
        epochs: dict[Any, list[float]] = {}
        for run_index, (run, points) in enumerate(points_by_run):
            recorded = {}
            for point in points:
                if _number(point.get("epoch")) and metric in point:
                    epochs.setdefault(point["epoch"], [])
                    if _number(point[metric]):
                        recorded.setdefault(point["epoch"], []).append(point[metric])
            for epoch, values in recorded.items():
                if all(value == values[0] for value in values):
                    # Repeated identical rows are one fold observation.
                    epochs[epoch].append(values[0])
                else:
                    # Keep conflicting observations visible; do not arbitrarily
                    # select a restart row or count one fold as two folds.
                    ambiguous.setdefault(run_index, []).append({"epoch": epoch, "metric": metric,
                        "recorded_values": values, "aggregation": "excluded_conflicting_rows"})
        if not epochs:
            continue
        rows = []
        for epoch, values in sorted(epochs.items()):
            stats = _distribution(values)
            rows.append([epoch, stats["n"], *[float(f"{stats[key]:.6g}") if stats[key] is not None else None
                                             for key in ("mean", "std", "min", "max")]])
        curves.append({"metric": metric, "metric_unit": "declared_loss" if metric == "train_loss" else "fraction_0_to_1",
            "scope": "source_subject_development", "aggregation": "equal_weight_across_subject_folds",
            "std_ddof": 0, "columns": ["epoch", "n", "mean", "std", "min", "max"], "rows": rows,
            **({"equivalent_recorded_metrics": [key for key, value in aliases.items() if value == metric]}
               if metric in aliases.values() else {})})
    extra = []
    for index, (run, points) in enumerate(points_by_run):
        summarized = {"epoch", *metrics, "loss_terms", "loss_term_batches", "train_batch_count"}
        additional = [{"epoch": point.get("epoch"), **{key: value for key, value in point.items()
                       if key not in summarized}} for point in points
                      if set(point).difference(summarized)]
        if additional or index in ambiguous:
            extra.append({**_run_coordinate(run), **({"points": additional} if additional else {}),
                          **({"ambiguous_measurements": ambiguous[index]} if index in ambiguous else {})})
    return curves, extra


def _configuration(run: dict) -> dict:
    config = copy.deepcopy(run.get("config") or {})
    resolved = ((run.get("training_execution") or {}).get("payload") or {}).get("resolved_model_configuration")
    if resolved:
        config["resolved_model"] = copy.deepcopy(resolved)
    if isinstance(config.get("recipe"), dict):
        config["recipe"] = {key: value for key, value in config["recipe"].items() if key not in {"data_root", "gpu"}}
        # The canonical training section and the full execution recipe share
        # these values. Deliver them once; a distinct pilot epoch remains visible.
        training = config.get("training") or {}
        config["recipe"] = {key: value for key, value in config["recipe"].items()
                            if key not in training or training[key] != value}
    return config


def _execution_summary(runs: list[dict]) -> dict:
    records, batches = [], {}
    for run in runs:
        identity = _run_coordinate(run)
        execution = compact_execution_facts(diagnostic_science_view(run.get("training_execution") or {}))
        payload = execution.get("payload") if isinstance(execution, dict) else None
        if isinstance(payload, dict):
            # The same constructor-owned values appear once in configuration.
            payload.pop("resolved_model_configuration", None)
        if isinstance(payload, dict) and isinstance(payload.get("batches"), list):
            if all(isinstance(batch, dict) for batch in payload["batches"]):
                for batch in payload.pop("batches"):
                    key = _canonical(_pick(batch, ("epoch", "batch_index", "batch_position")))
                    batches.setdefault(key, []).append((identity, batch))
        records.append((identity, execution))
    result = _factored(records)
    if batches:
        result["observed_batches"] = [{"coordinate": json.loads(key), "fold_count": len(rows),
            **_factored([(identity, {name: value for name, value in body.items() if name not in json.loads(key)})
                         for identity, body in rows])} for key, rows in sorted(batches.items())]
    return result


def _training_summary(runs: list[dict]) -> list[dict]:
    groups = {}
    for run in runs:
        execution = run.get("training_execution") or {}
        payload = execution.get("payload") or {}
        semantics = _pick(payload, ("loss_definition", "loss_direction_reduction", "loss_replica_reduction",
                                    "epoch_reduction", "loss_epoch_reduction", "negative_sampling_policy"))
        identity = _pick(run, ("arm", "candidate_id", "seed"))
        # Subject identity does not split the scientific arm. Different objective
        # definitions/configurations do split curves rather than mixing scales.
        config = run.get("config") or {}
        first_batch = next((row for row in payload.get("batches") or []
                            if isinstance(row, dict) and row.get("batch_position") == "first"), {})
        key = _canonical({**identity, "loss_semantics": semantics, "loss_settings": {
            "objective": config.get("objective"), "batch_size": (config.get("recipe") or {}).get("batch_size"),
            "negative_sampling_policy": config.get("negative_sampling_policy"),
            "observed_first_local_batch_sizes": first_batch.get("local_batch_sizes")}})
        groups.setdefault(key, []).append(run)
    result = []
    for key, group in sorted(groups.items()):
        group = sorted(group, key=lambda run: _canonical(_run_coordinate(run)))
        identity = json.loads(key)
        identity.pop("loss_settings", None)
        curves, additional_history = _curve_summary(group)
        profiles = [(_run_coordinate(run), {"learning_profile": copy.deepcopy(run.get("learning_profile") or {}),
                     "history_coverage": _pick(run.get("development_learning"), ("status", "scope", "read_coverage")),
                     "source_refs": copy.deepcopy(run.get("source_refs") or [])}) for run in group]
        result.append({**identity, "fold_count": len(group),
            "configuration": _factored([(_run_coordinate(run), _configuration(run)) for run in group]),
            "training_execution": _execution_summary(group),
            "curves": curves, "fold_profiles": _factored(profiles),
            "checkpoint_selection": _factored([(_run_coordinate(run), diagnostic_science_view(run.get("checkpoint_selection") or {})) for run in group]),
            "development_metrics": _factored([(_run_coordinate(run), diagnostic_science_view(run.get("development_metrics") or {})) for run in group]),
            **({"additional_history": additional_history} if additional_history else {})})
    return result


def _diagnostic_summary(runs: list[dict], scientific: dict) -> dict:
    # Only these native definitions have a scalar fold-distribution meaning.
    scalar_fields = {"representation": ("effective_rank", "mean_norm", "mean_dimension_variance"),
                     "retrieval_errors": ("mean_margin", "hit_rate"),
                     "image_id_duplicates": ("mean_duplicate_rate",)}
    groups, exports = {}, []
    for run in runs:
        coord = _run_coordinate(run)
        population = run.get("diagnostic_scope", "source_subject_development")
        diagnostics = compact_execution_facts(diagnostic_science_view(run.get("development_diagnostics") or {}))
        items = diagnostics.pop("items", {})
        if diagnostics:
            exports.append({**_pick(run, ("arm", "candidate_id")), **coord,
                            "population": population, "fields": diagnostics})
        for name, original in items.items():
            body = copy.deepcopy(original)
            if not isinstance(body, dict):
                body = {"payload": body}
            if (name == "training_dynamics" and "training_execution" in body
                    and body["training_execution"] == compact_execution_facts(
                        diagnostic_science_view(run.get("training_execution")))):
                body.pop("training_execution")
            values = {field: body.pop(field) for field in scalar_fields.get(name, ()) if _number(body.get(field))}
            # Definitions, method and sampling coverage belong in the group key:
            # a mean of ranks from incompatible definitions is not meaningful.
            meaning = {field: body[field] for field in ("definition", "effective_rank_definition", "method",
                       "scope", "coverage", "sample_count", "width", "centered") if field in body}
            if isinstance(meaning.get("coverage"), dict):
                # Checkpoint coordinates vary per fold; role and sampling
                # conditions determine whether diagnostic definitions agree.
                meaning["coverage"] = {key: value for key, value in meaning["coverage"].items()
                                       if key not in {"checkpoint", "checkpoint_id", "checkpoint_epoch"}}
            key = _canonical({**_pick(run, ("arm", "candidate_id", "seed")), "diagnostic": name,
                              "population": population,
                              "meaning": meaning if name in scalar_fields else {}})
            groups.setdefault(key, []).append((coord, body, values))
    result = []
    for key, rows in sorted(groups.items()):
        identity = json.loads(key)
        identity.pop("meaning")
        row = {**identity, "fold_count": len(rows), **_factored([(coord, body) for coord, body, _ in rows])}
        fields = sorted({field for _, _, values in rows for field in values})
        if fields:
            row["fold_distributions"] = {field: {**_distribution([values[field] for _, _, values in rows if field in values]),
                "std_ddof": 0} for field in fields}
            row["fold_values"] = [{**coord, **values} for coord, _, values in rows]
        result.append(row)
    return {"groups": result, **({"export_fields": exports} if exports else {}),
            **({"objective_findings": diagnostic_science_view(scientific)} if scientific else {})}


def build_analysis_summary(facts: dict) -> dict:
    """Deterministic science view of already collected facts; no I/O or checks."""
    runs = facts.get("training_runs") or []
    diagnostic = facts.get("diagnostic_task")
    execution_by_job = {run["job_id"]: run.get("training_execution") or {} for run in runs if run.get("job_id")}
    diagnostic_runs = runs
    if diagnostic:
        # Source rows describe the benchmark that owns the job. The selected
        # report records the population actually used for its diagnostics.
        profile = (diagnostic.get("diagnostic_request") or {}).get("profile")
        population = {"existing_development": "source_subject_development",
                      "full_retrieval": "method_development_benchmark"}.get(profile, diagnostic.get("diagnostic_scope"))
        if population == "campaign_history_development_annotation":
            population = "source_subject_development"
        coordinates = ("candidate_id", "suite_id", "fold_id")
        populations = {tuple(row.get(key) for key in coordinates): row["population"]
                       for row in diagnostic.get("coverage") or [] if row.get("population") is not None}
        diagnostic_runs = [{**row, "development_diagnostics": row.get("diagnostics") or {},
                            "training_execution": execution_by_job.get(row.get("job_id"), {}),
                            "diagnostic_scope": populations.get(tuple(row.get(key) for key in coordinates), population),
                            "arm": "baseline" if row.get("candidate_id") == "baseline" else "candidate"}
                           for row in diagnostic.get("runs") or []]
    result = {"measurement": _measurement_summary(facts), "training_summary": _training_summary(runs),
              "diagnostic_summary": _diagnostic_summary(diagnostic_runs, facts.get("scientific_diagnostics") or {})}
    if diagnostic:
        result["diagnostic_task"] = diagnostic_science_view({key: value for key, value in diagnostic.items()
            if key not in {"runs", "evidence_id"}})
    return result


def _comparison_coordinates(row: dict, seed: Any) -> tuple | None:
    """Pair recorded folds/subjects under a known seed, without guessing aliases."""
    seed = row.get("seed", seed)
    if seed is None:
        return None
    if row.get("fold_id") is not None:
        return "fold", row["fold_id"], seed
    if row.get("held_out_subject") is not None:
        return "subject", row["held_out_subject"], seed
    return None


def _field_differences(current: dict, historical: dict, prefix: str = "") -> list[dict]:
    result = []
    for key in sorted(set(current) | set(historical)):
        left, right = current.get(key), historical.get(key)
        path = prefix + key
        if key in current and key in historical and isinstance(left, dict) and isinstance(right, dict):
            result.extend(_field_differences(left, right, path + "."))
        elif key not in current or key not in historical or left != right:
            result.append({"field": path, **({"current": copy.deepcopy(left)} if key in current else {}),
                           **({"historical": copy.deepcopy(right)} if key in historical else {})})
    return result


def _historical_comparisons(facts: dict, rows: list[dict]) -> list[dict]:
    """Describe selected old results once, separately from the matched control.

    Mean differences compare reported means. Fold differences use only matching
    recorded coordinates; neither operation grants causal or replication status.
    """
    current = facts.get("measurement") or {}
    identity = facts.get("run_identity") or {}
    coverage = facts.get("coverage") or {}
    means = {}
    for metric in ("top1", "top5"):
        value = next((current[key] for key in ("mean_" + metric, "screening_mean_" + metric,
                     "fixed_bank_" + metric) if _number(current.get(key))), None)
        if value is not None:
            means[metric] = value
    current_scores = [{**_pick(row, ("fold_id", "held_out_subject", "seed")),
                       **_pick(row.get("metrics"), ("top1", "top5", "query_count", "candidate_count"))}
                      for row in current.get("scores") or []]
    conditions = comparison_conditions(identity, facts.get("comparison_conditions") or {}, coverage)
    for target, source in (("fold_ids", "screening_fold_ids"), ("subjects", "screening_subjects")):
        if coverage.get(source) is not None:
            conditions[target] = copy.deepcopy(coverage[source])
    for target, source in (("fold_ids", "fold_id"), ("subjects", "held_out_subject")):
        if target not in conditions and current_scores and all(score.get(source) is not None for score in current_scores):
            conditions[target] = sorted({score[source] for score in current_scores})
    groups = [group for group in facts["analysis_summary"].get("training_summary") or []
              if group.get("arm") == "candidate" or (identity.get("candidate_id") == "baseline"
                                                     and group.get("arm") == "baseline")]
    current_config = (groups[0].get("configuration") or {}).get("common") if len(groups) == 1 else None
    if isinstance(current_config, dict):
        recipe = current_config.get("recipe") or {}
        conditions = comparison_conditions(conditions, recipe)
    result = copy.deepcopy(rows)
    for row in result:
        if row.get("status") != "available":
            continue
        historical = row.get("measurement") or {}
        previous_conditions = row.get("conditions") or {}
        comparison = {"delta_unit": "percentage_points", "mean_delta_basis": "difference_of_reported_means"}
        differences = {metric: 100 * (value - historical["mean_" + metric]) for metric, value in means.items()
                       if _number(historical.get("mean_" + metric))}
        if differences:
            comparison["mean_delta_pp"] = differences
        axes = ("scope", "phase", "fidelity", "seed", "fold_ids", "subjects", "dataset", "modality", "input_geometry")
        matches, changed, unknown = [], [], []
        for key in axes:
            if key not in conditions and key not in previous_conditions:
                continue
            if conditions.get(key) is None or previous_conditions.get(key) is None:
                unknown.append(key)
                continue
            left, right = conditions[key], previous_conditions[key]
            if key in {"fold_ids", "subjects"}:
                left, right = sorted(left), sorted(right)
            if left == right:
                matches.append(key)
            else:
                changed.append({"field": key, "current": copy.deepcopy(left), "historical": copy.deepcopy(right)})
        comparison.update(condition_matches=matches, condition_differences=changed, unknown_conditions=unknown)
        old_scores = historical.get("fold_results") or []
        indexed = {_comparison_coordinates(old, previous_conditions.get("seed")): old for old in old_scores}
        indexed.pop(None, None)
        fold_deltas = []
        for score in current_scores:
            old = indexed.get(_comparison_coordinates(score, conditions.get("seed")))
            if old is None or (score.get("held_out_subject") is not None and old.get("held_out_subject") is not None
                               and score["held_out_subject"] != old["held_out_subject"]):
                continue
            delta = {metric + "_delta_pp": 100 * (score[metric] - old[metric]) for metric in ("top1", "top5")
                     if _number(score.get(metric)) and _number(old.get(metric))}
            if not delta:
                continue
            detail = {**_pick(score, ("fold_id", "held_out_subject", "seed")), **delta}
            differing_counts = _field_differences(_pick(score, ("query_count", "candidate_count")),
                                                 _pick(old, ("query_count", "candidate_count")))
            if differing_counts:
                detail["condition_differences"] = differing_counts
            fold_deltas.append(detail)
        if current_scores or old_scores:
            comparison["fold_coverage"] = {"current": len(current_scores), "historical": len(old_scores),
                                           "matched": len(fold_deltas)}
            comparison["fold_deltas"] = fold_deltas
        historical_groups = row.get("training_summary") or []
        historical_config = ((historical_groups[0].get("configuration") or {}).get("common")
                             if len(historical_groups) == 1 else None)
        if isinstance(current_config, dict) and isinstance(historical_config, dict):
            comparison["configuration_comparison_scope"] = "group_common_fields"
            comparison["configuration_differences"] = _field_differences(current_config, historical_config)
        row["comparison"] = comparison
    return result


def historical_comparison_memory(facts: dict) -> list[dict]:
    """Retain this experiment's comparison result, never embed its ancestors."""
    return [_pick(row, ("observation_id", "candidate_id", "status", "reason", "source_ref", "observation_source_ref", "arm",
                       "current_evidence_ref", "same_as_observation_id", "comparison"))
            for row in facts.get("historical_comparisons") or []]


def check_inputs(payload: dict) -> tuple[list[dict], list[str]]:
    """Requirements and allowed refs from a live or compiled (pointer-bearing) request."""
    from .review_context import _request_value
    requirements = _request_value(payload, "analysis_requirements") or []
    facts = _request_value(payload, "analysis_facts") or {}
    return [row for row in requirements if isinstance(row, dict)], list(facts.get("allowed_evidence_refs") or [])


def is_record(body: Any) -> bool:
    return isinstance(body, dict) and body.get("record_version") == RECORD_VERSION


def parse_analysis_reply(body: dict) -> tuple[dict, dict | None]:
    """A fresh reply, or the stored record a cached task returns (only its reply part is validated)."""
    from .schemas import AnalystReply
    if is_record(body):
        return AnalystReply.model_validate(body["analysis"]).model_dump(), body
    return AnalystReply.model_validate({key: value for key, value in body.items() if key not in _STRIP}).model_dump(), None


def _blank(value) -> bool:
    return not isinstance(value, str) or not value.strip()


def analysis_delivery_check(reply: dict, requirements: list[dict], allowed: list[str]) -> dict:
    """Structural delivery of the analysis, not proof that its science is correct."""
    errors = []
    def error(kind, loc, msg):
        errors.append({"type": kind, "loc": loc, "msg": msg})
    findings = reply.get("findings") if isinstance(reply.get("findings"), list) else []
    if not findings:
        error("analysis_findings_missing", ["findings"], "At least one finding is required; explain insufficient evidence in a finding.")
    if _blank(reply.get("summary_zh")):
        error("analysis_summary_missing", ["summary_zh"],
            "summary_zh is required: a concise Chinese interpretation of the intervention, "
            "hypothesis assessment and main unknowns. Runtime adds the numerical result header.")
    for index, item in enumerate(reply.get("limitations") or []):
        if _blank(item):
            error("analysis_text_blank", ["limitations", index], "Remove blank limitations; [] is valid.")
    declared = {row.get("requirement_id"): row for row in requirements}
    allowed_set = set(allowed)
    answered, refs_ok = {}, {}
    for index, finding in enumerate(findings):
        complete = True
        for key in _TEXT:
            if _blank(finding.get(key)):
                complete = False
                error("analysis_text_blank", ["findings", index, key], f"{key} must be non-empty text.")
        if finding.get("next_step") is not None and _blank(finding.get("next_step")):
            error("analysis_text_blank", ["findings", index, "next_step"], "Use null when there is no next step.")
        valid = True
        for position, ref in enumerate(finding.get("evidence_refs") or []):
            if ref not in allowed_set:
                valid = False
                error("analysis_evidence_ref_not_delivered", ["findings", index, "evidence_refs", position],
                      f"{ref} is not a delivered evidence reference; use only: {sorted(allowed_set)}")
        for position, rid in enumerate(dict.fromkeys(finding.get("requirement_ids") or [])):
            if rid not in declared:
                error("analysis_requirement_id_unknown", ["findings", index, "requirement_ids", position],
                      f"{rid} is not an analysis_requirements ID; use only: {sorted(declared)}")
                continue
            answered.setdefault(rid, []).append((index, complete))
            refs_ok[rid] = refs_ok.get(rid, True) and valid
    coverage, missing = [], []
    for rid, row in declared.items():
        rows = answered.get(rid, [])
        present = any(complete for _, complete in rows)
        refs_valid = bool(rows) and refs_ok.get(rid, False)
        coverage.append({"requirement_id": rid, "finding_indices": [index for index, _ in rows],
                         "response_present": present, "references_valid": refs_valid})
        if row.get("core", True) and not (present and refs_valid):
            missing.append(rid)
            if not rows:
                error("analysis_core_requirement_unanswered", ["findings"],
                      f"No finding lists core requirement {rid}: {row.get('expected_behavior', '')}")
    return {"status": "incomplete" if errors else "complete", "requirement_coverage": coverage,
            "missing_requirement_ids": missing, "errors": errors}


def failure_check(requirements: list[dict], reason: str) -> dict:
    """Delivery report when no valid reply exists (role or format failure)."""
    return {"status": "incomplete", "requirement_coverage": [], "errors": [{"type": "analyst_reply_unavailable", "msg": reason}],
            "missing_requirement_ids": [row.get("requirement_id") for row in requirements if row.get("core", True)]}


def native_result_summary(facts: dict) -> str:
    """Render recorded values for display; the model does not rewrite this header."""
    measurement = (facts.get("analysis_summary") or {}).get("measurement") or facts.get("measurement") or {}
    comparison = measurement.get("comparison") or {}
    matched = comparison.get("comparable") is not False
    parts = []
    for metric, label in (("top1", "Top-1"), ("top5", "Top-5")):
        value = next((measurement[key] for key in ("mean_" + metric, "screening_mean_" + metric,
                     "fixed_bank_" + metric) if _number(measurement.get(key))), None)
        if value is not None:
            parts.append(f"{label} {100 * value:.12g}%")
        baseline = measurement.get("baseline_mean_" + metric, (measurement.get("baseline") or {}).get("mean_" + metric))
        if _number(baseline):
            baseline_label = "匹配基线" if matched else "记录基线"
            parts.append(f"{baseline_label} {label} {100 * baseline:.12g}%")
    delta = next((measurement[key] for key in ("paired_mean_delta_top1_pp", "screening_mean_delta_pp", "delta_vs_control_pp")
                  if _number(measurement.get(key))), None)
    if delta is None and matched:
        delta = comparison.get("paired_mean_delta_pp", comparison.get("delta_pp"))
    if matched and _number(delta):
        parts.append(f"相对匹配基线 Top-1 差值 {delta:+.12g} 个百分点")
    coverage = facts.get("coverage") or {}
    if coverage.get("complete_fold_count") is not None:
        parts.append(f"完成 {coverage['complete_fold_count']} 折")
    seed = (facts.get("run_identity") or {}).get("seed")
    if seed is not None:
        parts.append(f"seed={seed}")
    scope = (facts.get("run_identity") or {}).get("scope")
    if scope:
        parts.append("范围=" + str(scope))
    outcome = measurement.get("screening_result") or (facts.get("execution") or {}).get("screening_result")
    if outcome:
        parts.append("筛查状态=" + str(outcome))
    training = []
    current_arm = "baseline" if (facts.get("run_identity") or {}).get("candidate_id") == "baseline" else "candidate"
    for run in facts.get("training_runs") or []:
        if run.get("arm") != current_arm:
            continue
        selected = run.get("checkpoint_selection") or {}
        epoch = selected.get("best_epoch")
        finished = selected.get("last_epoch")
        if epoch is not None or finished is not None:
            label = str(run.get("fold_id") or run.get("held_out_subject") or "本次训练")
            detail = []
            if finished is not None:
                detail.append(f"完成epoch={finished}")
            if epoch is not None:
                detail.append(f"选中epoch={epoch}")
            training.append(label + " " + "，".join(detail))
    if training:
        parts.append("；".join(training))
    return "本地实验记录：" + "；".join(parts) + "。" if parts else ""


def materialize_analysis_record(facts: dict, reply: dict | None, check: dict, *, summary_zh: str | None = None) -> dict:
    """Bind native facts, the Analyst reply and its delivery check; refs are the valid cited union."""
    allowed = set(facts.get("allowed_evidence_refs") or [])
    refs = [ref for finding in (reply or {}).get("findings") or [] for ref in finding.get("evidence_refs") or []
            if ref in allowed]
    interpretation = (reply or {}).get("summary_zh") or summary_zh or ""
    summary = "\n".join(part for part in (native_result_summary(facts), interpretation) if part)
    return {"record_version": RECORD_VERSION, "native_facts": copy.deepcopy(facts), "analysis": copy.deepcopy(reply),
            "delivery_check": copy.deepcopy(check), "evidence_refs": list(dict.fromkeys(refs)),
            "summary_zh": summary}


def analysis_memory_view(record: dict) -> dict:
    """Science view for Observation, Designer and reports; the delivery check stays local."""
    reply = record.get("analysis") or {}
    findings = copy.deepcopy(reply.get("findings") or [])
    return {"hypothesis_assessment": reply.get("hypothesis_assessment"), "summary_zh": reply.get("summary_zh") or "",
            "findings": findings, "limitations": copy.deepcopy(reply.get("limitations") or []),
            "next_steps": [item["next_step"] for item in findings if item.get("next_step")],
            "evidence_refs": copy.deepcopy(record.get("evidence_refs") or []),
            "native_facts": copy.deepcopy(record.get("native_facts") or {})}
