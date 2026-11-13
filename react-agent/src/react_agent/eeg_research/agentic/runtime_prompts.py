"""Versioned, task-relevant missions for single-plan execution.

Stage roles use runtime_v2; runtime_v1 keeps the supporting-role missions and
implementation cases. Mechanical contracts
are enforced by native validators; only active implementation cases are loaded.
"""
from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).with_name("prompts") / "runtime_v1"
PROMPT_VERSION = "eeg_research.runtime_mission.single_plan.v1"


def runtime_mission(role: str, *, review_stage: str | None = None) -> str:
    # Stage roles have one current contract; supporting roles keep their missions.
    root = ROOT
    if role in {"experiment_designer", "candidate_coder", "result_analyst"}:
        root = ROOT.parent / "runtime_v2"
    return (root / "shared_contract.txt").read_text(encoding="utf-8") + "\n\n" + (root / (role + ".txt")).read_text(encoding="utf-8")


def conditional_hints(role: str, payload: dict) -> dict | None:
    if role not in {"candidate_coder", "candidate_reviewer", "experiment_designer", "result_analyst"}:
        return None
    # Hints explain existing obligations; prose mentioning a technique or a
    # successful standard check must not create a new implementation promise.
    spec = next((payload[key] for key in ("experiment_spec", "approved_spec", "experiment", "design_brief")
                 if isinstance(payload.get(key), dict) and payload[key]), {})
    spec = spec.get("experiment") if isinstance(spec.get("experiment"), dict) else spec
    brief = spec.get("design_brief") or {}
    objective = (payload.get("effective_hook_config") or {}).get("objective")
    if objective is None:
        objective = spec.get("objective") or ((brief.get("hook_changes") or {}).get("objective") or {}).get("config") or {}
    labels = {str(objective.get(key) or "").lower().replace("-", "_")
              for key in ("name", "type", "objective_variant")}
    requirements = spec.get("implementation_requirements", brief.get("implementation_requirements", []))
    parameters = spec.get("scientific_parameters", brief.get("scientific_parameters", {})) or {}
    selected = []
    if any("positive_mass" in label or "multi_positive" in label for label in labels):
        selected.append("positive_mass_infonce")
    if any("tied_image" in label or label == "shared_image_classifier" for label in labels):
        selected.append("tied_image_ce")
    from .objective_effectiveness import objective_switch
    if objective_switch(objective):
        selected.append("objective_disabled")
    if any(row.get("verification") == "ordered_window_slots" for row in requirements):
        selected.append("ordered_pooling")
    if parameters.get("target_parameter_band") is not None:
        selected.append("capacity")
    check = payload.get("last_check") or payload.get("check") or {}
    if check and check.get("ok") is not True:
        failures = " ".join(str(check.get(key) or "") for key in ("error", "detail", "stage", "failures")).lower()
        if "checkpoint" in failures or "state_dict" in failures:
            # A concrete current failure takes priority over general notes.
            selected.insert(0, "checkpoint")
    notes = []
    for name in selected[:2]:
        body = (ROOT / "cases" / (name + ".txt")).read_text(encoding="utf-8")
        notes.append({"case_id": name, "applicability": "approved_configuration_or_current_check_failure", "instruction": body})
    return {"version": PROMPT_VERSION, "notes": notes} if notes else None


def prompt_rule_mapping() -> dict:
    return {
        "authority, admissibility, current identity, exact cost mirrors": "native action/spec/owner validators; compact authority kernel",
        "scientific coverage, metric units, matched control, final-test scope": "native ExperimentObservation facts + shared mission",
        "candidate source, interfaces, checkpoints and actual-read obligations": "native code/check receipts + Coder/Reviewer mission",
        "independent proposals, K shortfall and selection": "planning_contract wire + ResearchPlanProposal validators + Planner/Critic missions",
        "positive-mass InfoNCE, tied-image CE, objective-off": "conditional implementation notes; current approved definition remains authority",
        "ordered pooling, capacity probes, checkpoint layout": "conditional probe/check notes; native requirements validators",
        "pulse and particular tensor/config failures": "current native probe/error/source only; relevant engineering skill retrieval",
        "legacy pending paid prompts": "legacy exact system retained; new version creates new request/task identity",
    }
