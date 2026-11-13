"""Typed phase requirements without execution, inference or implicit migration."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import ValidationError

from .schemas import AnalysisRequirement, MeasurementRequirement

STAGE_REQUIREMENT_FIELDS = ("measurement_requirements", "analysis_requirements")
_MODELS = {"measurement_requirements": MeasurementRequirement,
           "analysis_requirements": AnalysisRequirement}
_FIELDS = {"implementation": "implementation_requirements",
           "measurement": "measurement_requirements", "analysis": "analysis_requirements"}


def _experiment(spec: Mapping[str, Any] | None) -> Mapping[str, Any]:
    spec = spec if isinstance(spec, Mapping) else {}
    inner = spec.get("experiment")
    return inner if isinstance(inner, Mapping) else spec


def has_stage_contract(spec: Mapping[str, Any] | None) -> bool:
    experiment = _experiment(spec)
    return experiment.get("design_contract_version") == "v2" and any(
        field in experiment for field in STAGE_REQUIREMENT_FIELDS)


def requirements_for_stage(spec: Mapping[str, Any] | None,
                           stage: Literal["implementation", "measurement", "analysis"]) -> list[dict[str, Any]]:
    """Return an independent phase view; the approved experiment stays intact."""
    if stage not in _FIELDS:
        raise ValueError("requirement_stage_unknown:" + str(stage))
    return copy.deepcopy(_experiment(spec).get(_FIELDS[stage]) or [])


def measurement_matches_scope(latest: dict, scope: dict,
                              verified_result: dict | None = None) -> bool:
    """Compare a recorded measurement with the selected task, without I/O.

    Selection uses the recorded native result; Measurement passes the same
    result after its existing receipt verification. Diagnostic tasks interpret
    existing coverage rather than requiring a new screening or full run.
    """
    if not isinstance(scope, dict):
        return False
    phase = scope.get("phase")
    if not phase:
        return True  # Non-selected callers may have no task scope.
    kind = latest.get("kind")
    native = verified_result
    if native is None:
        native = latest.get("screening" if kind == "method_screening" else "aggregate") or {}
    if phase != "diagnostic":
        if phase == "screening_2fold" and kind != "method_screening":
            return False
        if phase == "confirmation_10fold" and kind != "method_suite":
            return False
        if phase == "full" and kind == "method_screening":
            return False
        actual_phase = latest.get("phase") or native.get("phase")
        if actual_phase and actual_phase != phase:
            return False
        fidelity = latest.get("fidelity") or native.get("fidelity")
        # A suite is full-fidelity by construction, including compact views.
        fidelity = fidelity or ("full" if kind in {"method_screening", "method_suite"} else None)
        if fidelity != ("pilot" if phase == "pilot" else "full"):
            return False
    if scope.get("seed") is not None:
        if latest.get("seed") != scope["seed"]:
            return False
        if native.get("seed") is not None and native["seed"] != scope["seed"]:
            return False
    if "folds" in scope:
        folds = scope["folds"]
        observed = [row.get("fold_id") for row in native.get("scores") or []]
        if (not isinstance(folds, list) or not folds or len(set(folds)) != len(folds)
                or len(observed) != len(folds) or set(observed) != set(folds)):
            return False
        if any(native.get(key) is not None and native[key] != len(folds)
               for key in ("coverage", "required_fold_count")):
            return False
    return True


def analysis_experiment_for_task(experiment: dict, proposal: dict | None,
                                 latest: dict, verified_result: dict | None = None) -> dict:
    """Use this round's scope and question without changing the approved method.

    Existing-method tasks reuse implementation bytes without another Designer;
    their old design's analysis requirements cannot define the new assignment.
    A new method keeps its method-specific Designer requirements.
    With no selected task, historical maintenance uses the measurement's own
    scope and preserves the original scientific question.
    """
    native = verified_result if verified_result is not None else (
        latest.get("screening" if latest.get("kind") == "method_screening" else "aggregate") or {})
    if proposal is None:
        phase = latest.get("phase") or native.get("phase")
        if not phase:
            phase = ("screening_2fold" if latest.get("kind") == "method_screening" else
                     "full" if latest.get("kind") == "method_suite" else latest.get("fidelity"))
        scope = {"phase": phase} if phase else {}
        if latest.get("seed") is not None:
            scope["seed"] = latest["seed"]
        if native.get("scores"):
            scope["folds"] = [row["fold_id"] for row in native["scores"]]
    else:
        scope = proposal.get("evaluation_scope") or {}
    if not scope.get("phase"):
        return experiment
    target = proposal.get("target_candidate_ref") if proposal is not None else None
    if ((target and target != latest.get("candidate_id"))
            or not measurement_matches_scope(latest, scope, verified_result)):
        raise ValueError("selected_task_analysis_scope_mismatch")
    result = copy.deepcopy(experiment)
    result["evaluation_scope"] = copy.deepcopy(scope)
    if proposal is None:
        if measurement_matches_scope(latest, experiment.get("evaluation_scope") or {}, native):
            return result
    elif proposal.get("task_kind") not in {"confirmation", "validation", "diagnosis"}:
        return result
    folds = scope.get("folds") or [row.get("fold_id") for row in native.get("scores") or []]
    coverage = (f"the selected {len(folds)} subject folds under one training seed" if folds
                else "the delivered measurement's recorded coverage and training seeds")
    diagnostic = (" This diagnostic task adds no new training or fold coverage."
                  if scope.get("phase") == "diagnostic" else "")
    predicates = (
        ("analysis_evidence_grounding", "evidence_grounding",
         "Ground findings in the delivered measurements and training diagnostics; "
         "distinguish observed results from proposed mechanisms."),
        ("analysis_paired_delta", "paired_comparison",
         "Compare the existing method with its matched control within the delivered measured coverage when "
         "matched control evidence is available; otherwise state that comparison is unavailable."),
        ("analysis_hypothesis_assessment", "hypothesis_assessment",
         "Assess the supplied scientific question or hypothesis using the delivered evidence, including a negative or inconclusive result."),
        ("analysis_scope_limits", "scope_limits",
         "Conclusions cover " + coverage + ". Subject folds are not independent seed replications. "
         "Use the delivered evaluation conditions to state limits on final-test and statistical claims. "
         "A bundled intervention comparison does not isolate its component mechanisms." + diagnostic),
    )
    if proposal is not None:
        result.update(hypothesis=proposal["question"], scientific_intent=proposal["scientific_intent"])
    result["analysis_requirements"] = [{"requirement_id": rid, "verification": verification,
        "core": True, "expected_behavior": behavior} for rid, verification, behavior in predicates]
    return result


def baseline_experiment_for_task(proposal: dict, latest: dict, *,
                                 native_identity: dict,
                                 verified_result: dict | None = None) -> dict:
    """Give an existing frozen baseline this task's stages, without inventing approval.

    Method-suite callers supply the method identity from their already verified
    suite. Single-run identity is checked against its source binding and frozen
    spec by the existing Measurement stage. Neither uses a candidate's design.
    """
    if (latest.get("candidate_id") != "baseline" or native_identity.get("candidate_id") != "baseline"
            or any(not latest.get(key) or latest[key] != native_identity.get(key)
                   for key in ("source_hash", "spec_hash"))):
        raise ValueError("baseline_analysis_measurement_identity_mismatch")
    from .design_contract import native_measurement_requirements
    experiment = {"design_contract_version": "v2", "candidate_id": "baseline",
        "source_hash": native_identity["source_hash"], "spec_hash": native_identity["spec_hash"],
        "measurement_requirements": native_measurement_requirements()}
    result = analysis_experiment_for_task(experiment, proposal, latest, verified_result)
    for requirement in result["analysis_requirements"]:
        if requirement["verification"] == "paired_comparison":
            requirement["expected_behavior"] = (
                "Report the frozen baseline's measured performance. It is the reference arm; "
                "state that a candidate comparison is unavailable when none was measured.")
    return result


def normalize_stage_requirements(spec: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Normalize only explicit v2 fields and keep IDs unique across all phases."""
    experiment = _experiment(spec)
    if not any(field in experiment for field in STAGE_REQUIREMENT_FIELDS):
        return {}
    if experiment.get("design_contract_version") != "v2":
        raise ValueError("stage_requirements_require_v2_contract")
    normalized = {}
    ids = [row.get("requirement_id") for row in experiment.get("implementation_requirements") or []
           if isinstance(row, Mapping)]
    for field, model in _MODELS.items():
        if field not in experiment:
            continue
        value = experiment[field]
        if not isinstance(value, list):
            raise ValueError(field + "_not_list")
        try:
            normalized[field] = [model.model_validate(item).model_dump() for item in value]
        except ValidationError as exc:
            raise ValueError(field + "_invalid:" + str(exc.errors()[0]["loc"])) from exc
        ids.extend(item["requirement_id"] for item in normalized[field])
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate_stage_requirement_id")
    return normalized


def stage_requirements_projection(spec: Mapping[str, Any] | None) -> dict[str, list[dict[str, Any]]]:
    """Explicit phase placement and every acceptance predicate are immutable."""
    return {field: sorted(rows, key=lambda row: row["requirement_id"])
            for field, rows in normalize_stage_requirements(_experiment(spec)).items()}
