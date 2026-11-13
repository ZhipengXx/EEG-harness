"""Small scientific brief -> existing executable experiment contract.

This native adapter resolves inherited configs. Candidate hooks consume
model/objective/transform; training settings enter the native execution recipe.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

from pydantic import ValidationError

from .schemas import DesignerBrief, TrainingRecipe

SECTIONS = ("model", "objective", "transform")
HOOKS = {"model": "build_encoder", "objective": "build_training_objective",
         "transform": "build_training_transform"}
NATIVE_MEASUREMENT_PREFIX = "native_measurement_"


def native_measurement_requirements() -> list[dict[str, Any]]:
    """Runtime predicates under the frozen protocol, never invented by Designer."""
    predicates = (
        ("run_identity", "The native receipt must match the accepted candidate, source, spec and result bytes."),
        ("paired_coverage", "Native completed coverage must contain the required matched folds or control runs."),
        ("metric_integrity", "Native verified metrics and aggregation must be finite and valid; a negative delta remains valid."),
    )
    return [{"requirement_id": NATIVE_MEASUREMENT_PREFIX + verification,
             "verification": verification, "core": True, "expected_behavior": behavior}
            for verification, behavior in predicates]


class DesignContractError(ValueError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def design_contract_version(value: Mapping | Path | str | None) -> str:
    """Single-plan research always uses v2; v1 remains only for the legacy action loop."""
    from .research_execution import research_goal, single_plan_enabled
    value = research_goal(value) if value is not None else {}
    if value.get("design_contract_version") == "v2":
        return "v2"
    goal = value.get("goal") if isinstance(value.get("goal"), dict) else value
    return "v2" if single_plan_enabled(goal) else "v1"


def _validate_parameter_values(parameters: dict, configs: dict) -> None:
    """Scientific parameters may live in source, but repeated config values agree."""
    for name, value in parameters.items():
        if name in configs and isinstance(value, dict):
            for key, declared in value.items():
                if key not in configs[name] or configs[name][key] != declared:
                    raise DesignContractError("scientific_parameter_config_mismatch:" + name + "." + key)
        elif "." in name and name.split(".", 1)[0] in configs:
            section, key = name.split(".", 1)
            if key not in configs[section] or configs[section][key] != value:
                raise DesignContractError("scientific_parameter_config_mismatch:" + name)
        else:
            for section in configs:
                if name in configs[section] and configs[section][name] != value:
                    raise DesignContractError("scientific_parameter_config_mismatch:" + section + "." + name)


def is_training_only_change(brief: dict[str, Any]) -> bool:
    """A recipe-only intervention needs Measurement, not invented source work."""
    if not isinstance(brief, dict):
        return False
    changes, training = brief.get("hook_changes"), brief.get("training_change")
    return (isinstance(changes, dict) and isinstance(training, dict) and training.get("mode") == "replace"
            and all(isinstance(changes.get(section), dict) and changes[section].get("mode") == "inherit"
                    for section in SECTIONS))


def materialize_design_brief(brief: dict[str, Any] | DesignerBrief,
                             selected_task: dict[str, Any],
                             parent_config: dict[str, Any]) -> dict[str, Any]:
    """Resolve a v2 brief against the selected native task and actual parent binding.

    selected_task can be a SelectedPlan (proposal nested) or its delivered task
    projection. Native caller may add parent/control IDs and initial_fidelity.
    Config replacement is complete, not an implicit merge with the parent.
    """
    try:
        parsed = brief if isinstance(brief, DesignerBrief) else DesignerBrief.model_validate(brief)
    except ValidationError as exc:
        raise DesignContractError("design_brief_invalid:" + str(exc.errors()[0]["loc"])) from exc
    task = selected_task if isinstance(selected_task, dict) else {}
    proposal = task.get("proposal") if isinstance(task.get("proposal"), dict) else task
    parent = parent_config if isinstance(parent_config, dict) else {}
    selected_parent = proposal.get("parent_ref")
    parent_id = str(task.get("parent_candidate_id") or proposal.get("parent_candidate_id") or selected_parent
                    or parent.get("target_id") or "baseline")
    if selected_parent is not None and selected_parent != parent_id:
        raise DesignContractError("selected_parent_binding_mismatch")
    if parent_id != "baseline" and not all(isinstance(parent.get(section), dict) for section in SECTIONS):
        raise DesignContractError("parent_hook_config_missing")
    if parent.get("target_id") and parent["target_id"] != parent_id:
        raise DesignContractError("parent_hook_config_identity_mismatch")
    configs = {}
    capabilities = list(parsed.required_capability_ids)
    for section in SECTIONS:
        change = getattr(parsed.hook_changes, section)
        inherited = parent.get(section, {})
        if change.mode == "inherit" and not isinstance(inherited, dict):
            raise DesignContractError("parent_hook_config_invalid:" + section)
        configs[section] = copy.deepcopy(inherited if change.mode == "inherit" else change.config)
        if section == "model" or change.mode == "replace":
            capabilities.append(HOOKS[section])
    training_change = parsed.training_change
    # Internal callers without a training declaration retain their existing
    # protocol-owned recipe. The API contract always supplies training_change.
    if "training_change" in parsed.model_fields_set or "training" in parent:
        if training_change.mode == "replace":
            configs["training"] = training_change.config.model_dump()
        else:
            if not isinstance(parent.get("training"), dict):
                raise DesignContractError("parent_training_config_missing")
            try:
                configs["training"] = TrainingRecipe.model_validate(parent["training"]).model_dump()
            except ValidationError as exc:
                raise DesignContractError("parent_training_config_invalid") from exc
    _validate_parameter_values(parsed.scientific_parameters, configs)
    scope = copy.deepcopy(proposal.get("evaluation_scope") or task.get("evaluation_scope") or {})
    fidelity = task.get("initial_fidelity") or proposal.get("initial_fidelity")
    fidelity = fidelity or ("pilot" if scope.get("phase") == "pilot" else "full")
    if fidelity not in {"pilot", "full"}:
        raise DesignContractError("fidelity_invalid")
    requirements = []
    for declaration in parsed.implementation_requirements:
        row = declaration.model_dump()
        row["binding_stage"] = "design"
        row["scope"] = declaration.scope
        requirements.append(row)
    result = {
        "design_contract_version": "v2",
        "parent_candidate_id": parent_id,
        "control_candidate_id": task.get("control_candidate_id") or proposal.get("control_candidate_id") or "baseline",
        "initial_fidelity": fidelity,
        "hypothesis": parsed.hypothesis,
        "intervention": parsed.principal_intervention,
        "principal_intervention": parsed.principal_intervention,
        "scientific_parameters": copy.deepcopy(parsed.scientific_parameters),
        "hook_modes": {section: getattr(parsed.hook_changes, section).mode for section in SECTIONS},
        "scientific_intent": proposal.get("scientific_intent") or task.get("scientific_intent"),
        "evaluation_scope": scope,
        "required_capability_ids": list(dict.fromkeys(capabilities)),
        **configs,
        "implementation_requirements": requirements,
        "design_brief": parsed.model_dump(),
        "inheritance": {
            "parent_candidate_id": parent_id,
            **{key: parent.get(key) for key in ("spec_ref", "spec_hash", "source_hash") if parent.get(key)},
            "sections": ([section for section in SECTIONS if getattr(parsed.hook_changes, section).mode == "inherit"]
                         + (["training"] if "training" in configs and training_change.mode == "inherit" else [])),
        },
        "status": "draft",
    }
    from .stage_contract import STAGE_REQUIREMENT_FIELDS
    for field in STAGE_REQUIREMENT_FIELDS:
        if field in parsed.model_fields_set:
            result[field] = [requirement.model_dump() for requirement in getattr(parsed, field)]
    # This policy is owned by the frozen task/parent, never invented by Designer.
    policy = task.get("negative_sampling_policy") or parent.get("negative_sampling_policy")
    if policy:
        result["negative_sampling_policy"] = policy
    return result


def scientific_design_projection(spec: dict[str, Any] | None) -> dict[str, Any]:
    """Freeze the method and scientific promises, allowing engineering corrections.

    The ID/scope/kind/core roster is fixed, so relabeling a scientific promise as
    engineering cannot remove it from this identity. Source paths and explanatory
    engineering text remain implementation evidence rather than method choices.
    """
    spec = spec if isinstance(spec, dict) else {}
    result = {key: copy.deepcopy(spec.get(key)) for key in (
        "principal_intervention", "hypothesis", "scientific_parameters", "scientific_intent",
        "model", "objective", "transform", "hook_modes", "negative_sampling_policy", "parent_candidate_id",
        "control_candidate_id", "initial_fidelity", "evaluation_scope")}
    if "training" in spec:
        result["training"] = copy.deepcopy(spec["training"])
    roster, scientific = [], []
    for requirement in spec.get("implementation_requirements") or []:
        scope = requirement.get("scope", "scientific")
        kind = requirement.get("kind", "implementation")
        row = {"requirement_id": requirement.get("requirement_id"), "scope": scope,
               "kind": kind, "core": requirement.get("core", True)}
        roster.append(row)
        if scope == "scientific" or kind == "scientific_hypothesis":
            row = {**row, "expected_behavior": requirement.get("expected_behavior"),
                   "verification": requirement.get("verification", "source_review")}
            probe = requirement.get("probe")
            if isinstance(probe, dict):
                row["probe"] = {key: probe.get(key, default) for key, default in (
                    ("slot_count", None), ("window_axis", -1), ("layout", "flatten_channel_major"))}
            scientific.append(row)
    result["requirement_roster"] = sorted(roster, key=lambda row: row["requirement_id"] or "")
    result["scientific_requirements"] = sorted(scientific, key=lambda row: row["requirement_id"] or "")
    from .stage_contract import stage_requirements_projection
    result.update(stage_requirements_projection(spec))
    return result
