"""Scientific inputs for proposal generation and independent comparison.

These are delivery views, never native observations or feasibility proofs.
Selection/validation/recovery continue to use the unchanged native objects.
No summarizer call, text clipping, or second memory retrieval is performed here.
"""
from __future__ import annotations

import copy
import math
import re

VERSION = "research_planning_input.v1"
VERSION_V2 = "research_planning_input.v2"

TASK_ACTIONS = frozenset({"design_experiment", "propose_experiment", "run_screening",
    "run_confirmation", "run_full", "run_pilot", "collect_diagnostics", "diagnose_results"})


def _pick(value, fields):
    return {key: copy.deepcopy(value[key]) for key in fields if key in value} if isinstance(value, dict) else {}


def _scientific_data(value):
    """Remove storage/receipt metadata from otherwise scientific structured data."""
    if isinstance(value, list):
        # Applicability rows encode metadata names as values, rather than keys.
        # Keep differences in scientific conditions, not byte-identity checks.
        return [_scientific_data(item) for item in value if not (isinstance(item, dict)
            and isinstance(item.get("field"), str)
            and (item["field"].endswith(("_hash", "_digest", "_fingerprint"))
                 or item["field"] == "method_revision"))]
    if not isinstance(value, dict):
        return copy.deepcopy(value)
    bookkeeping = {"schema_version", "validation_records", "check", "checks",
        "cpu_check_receipt", "approval_record", "operation_binding", "coding_recovery",
        "next_read", "next_reads", "read_capability", "read_unit", "json_pointer",
        "satisfies_development_read_receipt", "verification_artifact"}
    return {key: _scientific_data(child) for key, child in value.items()
        if key not in bookkeeping and not key.startswith("_")
        and not key.endswith(("_hash", "_sha256", "_digest", "_fingerprint"))}


def _expanded(view):
    """Legacy paid deliveries may use request-local exact deduplication pointers."""
    from .context.representations import resolve_request_pointer
    from .context.sources import pointer_escape

    def visit(value, path=""):
        if isinstance(value, dict):
            if "$context_pointer" in value:
                target = value["$context_pointer"]
                return visit(resolve_request_pointer(view, path), target)
            return {key: visit(child, path + "/" + pointer_escape(key)) for key, child in value.items()}
        if isinstance(value, list):
            return [visit(child, path + "/" + str(index)) for index, child in enumerate(value)]
        return copy.deepcopy(value)
    return visit(view)


def _source(value):
    result = _pick(value, ("candidate_id", "intervention", "hypothesis", "source", "source_ref",
        "source_truncated", "missing_inputs", "model", "objective", "transform", "training", "recipe",
        "frozen_training_policy", "negative_sampling_policy"))
    if isinstance(value, dict) and isinstance(value.get("encoder_reference"), dict):
        result["encoder_reference"] = _pick(value["encoder_reference"], ("excerpt", "source_ref"))
    return result


def _v2_science(value):
    """Drop known source shells; scientific config/metric keys stay literal."""
    if isinstance(value, list):
        return [_v2_science(child) for child in value if not (isinstance(child, dict)
            and isinstance(child.get("field"), str)
            and (child["field"].endswith(("_hash", "_digest", "_fingerprint"))
                 or child["field"] == "method_revision"))]
    if not isinstance(value, dict):
        return copy.deepcopy(value)
    preserve = {"model", "objective", "transform", "config", "scientific_parameters", "metrics",
                "scores", "paired_scores", "parameters", "hyperparameters", "training", "training_recipe", "measurements"}
    omit = {"schema_version", "validation_records", "check", "checks", "cpu_check_receipt",
        "approval_record", "operation_binding", "coding_recovery", "next_read", "next_reads",
        "read_capability", "read_unit", "json_pointer", "verification_artifact",
        "satisfies_development_read_receipt", "campaign_id", "task_id", "attempt_id", "plan_id",
        "fingerprint", "digest", "sha256"}
    return {key: copy.deepcopy(child) if key in preserve else _v2_science(child)
        for key, child in value.items() if key not in omit and not key.startswith("_")
        and not key.endswith(("_hash", "_digest", "_fingerprint", "_sha256"))}


_PLANNER_CARD_LOCAL = frozenset({"source_hash", "next_read", "read_capability",
                                 "satisfies_development_read_receipt"})


def planner_card_view(value):
    """The v2 Planner has no read tool and never checks hashes: drop read
    protocol and source hashes, keep $context_source citations, and shorten an
    error fingerprint to 12 characters (equal failures still compare equal)."""
    if isinstance(value, list):
        return [planner_card_view(item) for item in value]
    if not isinstance(value, dict) or "$context_pointer" in value:
        return copy.deepcopy(value)
    result = {}
    for key, item in value.items():
        if key in _PLANNER_CARD_LOCAL:
            continue
        if key == "error_fingerprint" and isinstance(item, str) and len(item) > 12:
            item = item[:12]
        result[key] = planner_card_view(item)
    return result


def _v2_observations(full, view):
    """Render only selected, complete cards; no native-store refill follows fit."""
    from .memory_delivery import project_memory_card
    rows = []
    cards = view.get("experiment_observations") or []
    cards = cards.get("cards", []) if isinstance(cards, dict) else cards
    for supplied in cards:
        if not isinstance(supplied, dict):
            continue
        card = project_memory_card(supplied, role="research_planner")
        row = copy.deepcopy(card["body"])
        row["observation_id"] = card["record_id"]
        for key in ("source_refs", "current_feedback", "condition_differences"):
            if key in card:
                row[key] = copy.deepcopy(card[key])
        if "applicability" in card and card["applicability"] != row.get("applicability"):
            row["applicability_status"] = copy.deepcopy(card["applicability"])
        rows.append(planner_card_view(row))
    return rows


_REFERENCE_RUN_FIELDS = ("learning_profiles", "training_execution")


def _expand_shared_facts(value, shared):
    if isinstance(value, dict):
        if set(value) == {"shared_fact_ref"}:
            return copy.deepcopy(shared[value["shared_fact_ref"]])
        return {key: _expand_shared_facts(child, shared) for key, child in value.items()}
    if isinstance(value, list):
        return [_expand_shared_facts(child, shared) for child in value]
    return copy.deepcopy(value)


def _share_common_facts(view):
    """Share exact geometry, scope and training facts after memory selection."""
    import json
    groups = {}

    def add(parent, field, kind):
        value = parent.get(field)
        if not isinstance(value, dict) or not value:
            return
        encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        groups.setdefault((kind, encoded), []).append((parent, field))

    protocol = view.get("protocol") or {}
    add(protocol, "input_geometry", "geometry")
    add(protocol, "evidence_scope", "scope")
    facts = []
    for row in view.get("observations") or []:
        add(row.get("applicability") or {}, "input_geometry", "geometry")
        native = row.get("native_facts") or {}
        add(native, "science_scope", "scope")
        facts.append(native)
    facts.extend((view.get("shared_reference_results") or {}).values())
    for native in facts:
        for row in native.get("training_execution") or []:
            add(row, "common", "training")
    shared = {}
    for (kind, encoded), locations in groups.items():
        if len(locations) < 2:
            continue
        name = kind + "_" + str(len(shared) + 1)
        reference = {"shared_fact_ref": name}
        # Keep small values inline when a reference would increase the input.
        if (len(locations) - 1) * len(encoded) <= len(locations) * len(json.dumps(reference)) + len(name) + 4:
            continue
        shared[name] = copy.deepcopy(locations[0][0][locations[0][1]])
        for parent, field in locations:
            parent[field] = dict(reference)
    if shared:
        view["shared_facts"] = shared
    return view


def expanded_observations(view):
    """Restore full observation rows from references within this request."""
    common = view.get("shared_facts") or {}
    rows = _expand_shared_facts(view.get("observations") or [], common)
    shared = _expand_shared_facts(view.get("shared_reference_results") or {}, common)
    for row in rows:
        facts = row.get("native_facts") or {}
        for field in _REFERENCE_RUN_FIELDS:
            if field not in facts:
                continue
            values = []
            for item in facts[field]:
                if set(item) == {"control_ref"}:
                    values.extend(copy.deepcopy(shared[item["control_ref"]][field]))
                else:
                    values.append(item)
            facts[field] = values
    return rows


def _share_reference_results(view):
    """Factor identical, source-bound baseline run groups after card selection."""
    import json

    result = _expand_shared_facts(view, view.get("shared_facts") or {})
    result.pop("shared_facts", None)
    result["observations"] = expanded_observations(view)
    result.pop("shared_reference_results", None)
    groups = {}
    for index, row in enumerate(result["observations"]):
        facts = row.get("native_facts") or {}
        blocks, spans = {}, {}
        for field in _REFERENCE_RUN_FIELDS:
            values = facts.get(field) or []
            positions = [i for i, item in enumerate(values) if item.get("arm") == "baseline"]
            if positions and positions == list(range(positions[0], positions[-1] + 1)):
                start, end = positions[0], positions[-1] + 1
                blocks[field] = values[start:end]
                spans[field] = (start, end)
        profiles = blocks.get("learning_profiles") or []
        if not profiles or any(not (item.get("learning_profile") or {}).get("source_ref") for item in profiles):
            continue
        conditions = _pick(facts, ("scope", "phase", "fidelity", "seed", "screening_fold_ids",
            "screening_subjects", "science_scope"))
        conditions["evaluation_scope"] = row.get("evaluation_scope")
        identity = json.dumps([conditions, blocks], sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        groups.setdefault(identity, []).append((index, blocks, spans))
    shared = {}
    for matches in groups.values():
        if len(matches) < 2:
            continue
        ref = "baseline_" + str(len(shared) + 1)
        shared[ref] = copy.deepcopy(matches[0][1])
        for index, _, spans in matches:
            facts = result["observations"][index]["native_facts"]
            for field, (start, end) in spans.items():
                facts[field][start:end] = [{"control_ref": ref}]
    if shared:
        result["shared_reference_results"] = shared
    return _share_common_facts(result)


def _history_navigation(history, observations, previous_observations=(), summary_records=(), priority_ids=()):
    """Use short navigation for inline bodies, recorded outcomes otherwise."""
    from .experiment_observations import fit_history_index, observation_summary_index

    if not isinstance(history, dict):
        return copy.deepcopy(history)
    detailed = {row.get("observation_id"): row for row in observations if isinstance(row, dict)}
    previous = {row.get("observation_id"): row for row in previous_observations}
    summaries = {row["observation_id"]: observation_summary_index(row) for row in summary_records}
    result = copy.deepcopy(history)
    frozen_ids = {row.get("observation_id") for row in result.get("items") or [] if isinstance(row, dict)}
    priority = set(priority_ids)
    rows = []
    expanded = False
    for row in result.get("items") or []:
        if isinstance(row, dict):
            identity = row.get("observation_id")
            if identity in detailed:
                candidate = row.get("candidate_id") or (detailed[identity].get("identity") or {}).get("candidate_id")
                row = {"observation_id": identity}
                if candidate is not None:
                    row["candidate_id"] = candidate
            elif identity in priority and set(row) - {"observation_id", "candidate_id"}:
                # Preserve the supplied basis of an admitted proposal. A new
                # longer summary must not displace its frozen index evidence.
                pass
            elif identity in summaries:
                row = summaries.pop(identity)
                expanded = True
            elif identity in previous and set(row) <= {"observation_id", "candidate_id"}:
                row = {**observation_summary_index(previous[identity]), **row}
                expanded = True
            summaries.pop(identity, None)
        rows.append(row)
    if summaries:
        rows = list(summaries.values()) + rows
        expanded = True
    if expanded:
        rows.sort(key=lambda row: (row.get("observation_id") not in priority,
            row.get("observation_id") not in frozen_ids if row.get("observation_id") in priority else False,
            row.get("analysis_delivery") != "summary_only"))
        return fit_history_index(rows, total=max(result.get("total", len(rows)), len(rows)))
    result["items"] = rows
    return result


def _v2_knowledge(view):
    if view.get("memory_delivery_report"):
        # The same selected cards were measured with the optional observations.
        # No second truncation/projection can change that shared allocation.
        bundle = view.get("retrieved_knowledge") or {}
        return {"status": bundle.get("status", "no_related_knowledge"),
                "items": copy.deepcopy(bundle.get("items") or [])}
    result = {"status": "no_related_knowledge", "items": []}
    seen = set()
    for key in ("retrieved_knowledge", "retrieved_memory"):
        bundle = view.get(key)
        if not isinstance(bundle, dict):
            continue
        if bundle.get("status"):
            result["status"] = bundle["status"]
        elif bundle.get("degraded"):
            result["status"] = "retrieval_unavailable"
        for item in bundle.get("items") or []:
            if not isinstance(item, dict):
                continue
            identity = item.get("record_id")
            if identity and identity in seen:
                continue
            card = _v2_science(_pick(item, ("record_id", "record_type", "body", "source_ref", "source_refs",
                "scope", "conditions", "applicability", "condition_differences", "is_scientific_evidence",
                "scientific_evidence", "knowledge_role", "omissions")))
            original = item.get("full_record") or {}
            if "source_ref" not in card and isinstance(original, dict) and original.get("$context_source"):
                card["source_ref"] = original["$context_source"]
            result["items"].append(card)
            if identity:
                seen.add(identity)
    if result["items"] and result["status"] == "no_related_knowledge":
        result["status"] = "ready"
    return result


def _v2_resources(full):
    from .context.types import canonical

    budget = full.get("budget") or {}
    owner = budget.get("owner_budget") or {}
    # A Planner owner reserves API calls, not a future method's training. Only
    # global and unreserved amounts can bound new selected-task allocations.
    unreserved = owner.get("unreserved_available") or {}
    result = {}
    for field, dimension in (("gpu_seconds_left", "gpu_seconds"), ("training_jobs_left", "training_jobs")):
        known = [source[key] for source, key in ((budget, field), (unreserved, dimension))
                 if key in source and type(source[key]) in (int, float) and math.isfinite(source[key])]
        if known:
            result[field] = min(known)
        elif any(key in source and source[key] is None for source, key in ((budget, field), (unreserved, dimension))):
            result[field] = None
    entries = full.get("action_cost_estimates") or {}
    entries = entries.get("by_action_target") or entries
    costs = {}
    for action, body in entries.items():
        if not isinstance(body, dict) or action.split(":", 1)[0] not in TASK_ACTIONS:
            continue
        estimate = body.get("estimated_cost") or {}
        if not any(type(estimate.get(key)) in (int, float) for key in ("training_jobs", "gpu_seconds")):
            continue
        cost = _pick(estimate, ("training_jobs",))
        if "gpu_seconds" in estimate:
            cost["gpu_seconds_estimate"] = copy.deepcopy(estimate["gpu_seconds"])
        cost.update(_pick(body, ("cost_confidence", "cost_scope", "unknown_components")))
        costs[action] = cost
    grouped = {}
    for action, cost in costs.items():
        group = grouped.setdefault(canonical(cost), {"tasks": [], "cost": cost})
        group["tasks"].append(action)
    individual = {group["tasks"][0]: group["cost"] for group in grouped.values() if len(group["tasks"]) == 1}
    shared = [group for group in grouped.values() if len(group["tasks"]) > 1]
    if individual:
        result["native_cost_basis"] = individual
    if shared:
        result["shared_cost_basis"] = shared
    return result


def _v2_task_view(task):
    scope = task.get("evaluation_scope") or {}
    result = _pick(task, ("task_ref", "task_kind", "target_candidate_ref"))
    result.update(_pick(scope, ("phase",)))
    if scope.get("diagnostic_request"):
        result["diagnostic_request"] = _pick(scope["diagnostic_request"], ("profile", "population", "outputs"))
    return result


def _reference_training_execution(full, baseline, policy):
    from .interface import training_batch_partition, training_batch_sizes

    execution = full.get("execution_protocol") or {}
    devices = execution.get("gpu") or []
    if not devices:
        return {}
    reference = {**execution, **(baseline.get("recipe") or {})}
    result = {"reference_method": "baseline", "configured_replica_count": len(devices)}
    if policy in {"data_parallel_local", "global_batch"}:
        result["objective_execution_scope"] = (
            "per_replica" if policy == "data_parallel_local" else "full_batch_after_encoder_gather")
    if reference.get("batch_size") and reference.get("train_query_ids"):
        first, tail = training_batch_sizes(reference)
        result["reference_local_batch_sizes"] = training_batch_partition(first, len(devices))
        if tail != first:
            result["reference_tail_local_batch_sizes"] = training_batch_partition(tail, len(devices))
    return result


def _planning_decision_rules(goal, execution):
    result = {}
    screening_gain = (goal.get("screening") or {}).get("min_mean_gain_pp")
    if screening_gain is not None:
        result["screening_decision"] = {"minimum_mean_gain_pp": screening_gain}
    practical_gain = goal.get("min_practical_gain_pp")
    if practical_gain is not None and practical_gain != screening_gain:
        result["min_practical_gain_pp"] = practical_gain
    policy = execution.get("checkpoint_policy", goal.get("checkpoint_policy"))
    if policy:
        selection = {"policy": policy, "metric": "development_fixed_bank_top1", "metric_unit": "fraction"}
        if policy == "legacy_min_delta":
            delta = execution.get("selection_min_delta", goal.get("selection_min_delta"))
            if delta is not None:
                selection["minimum_gain"] = delta
        result["checkpoint_selection"] = selection
    return result


def _v2_protocol(full, view, baseline):
    goal = view.get("goal") or full.get("goal") or {}
    interface = view.get("candidate_interface") or full.get("candidate_interface") or {}
    recipe = copy.deepcopy(baseline.get("recipe") or {})
    geometry = recipe.pop("input_geometry", None)
    protocol = _pick(goal, ("task_type", "evaluation_mode", "evaluation_schedule", "primary_metric",
        "direction", "subject_aggregation", "training_seeds", "frozen_components"))
    protocol.update(_pick(full.get("frozen_evaluation_scopes") or {}, ("metric_unit",)))
    protocol.update(_pick(interface, ("input", "output", "gallery_size", "not_available")))
    geometry = geometry or _pick(interface, ("c_num", "channels", "timesteps"))
    if geometry:
        protocol["input_geometry"] = geometry
    if recipe:
        for key in ("training_seeds", "negative_sampling_policy"):
            if key in recipe:
                protocol[key] = recipe.pop(key)
        protocol["training_recipe"] = recipe
        protocol["training_recipe_scope"] = "baseline_reference; candidate settings are selected by Designer and then fixed per method"
    if interface.get("training_configuration"):
        protocol["training_configuration"] = copy.deepcopy(interface["training_configuration"])
    if "negative_sampling_policy" not in protocol and baseline.get("negative_sampling_policy"):
        protocol["negative_sampling_policy"] = baseline["negative_sampling_policy"]
    execution = _reference_training_execution(full, baseline, protocol.get("negative_sampling_policy"))
    if execution:
        protocol["training_execution"] = execution
    contract = view.get("contract") or full.get("contract") or {}
    protocol.update(_pick(contract, ("generalization_target", "val_mode", "benchmark_scope", "within_batch_role")))
    scope = copy.deepcopy(view.get("science_scope") or full.get("science_scope") or {})
    if goal.get("final_test_enabled") is False:
        scope["independent_final_test"] = False
    if scope:
        protocol["evidence_scope"] = _v2_science(scope)
    protocol.update(_planning_decision_rules(goal, full.get("execution_protocol") or {}))
    confirmation = goal.get("confirmation") or {}
    protocol.update(_pick(confirmation, ("freeze_method_before_screening",)))
    # Capability behavior is concise and stage-neutral. Implementation signatures,
    # buffers, module import paths and Reviewer/check mechanics stay downstream.
    capability = view.get("capabilities") or full.get("capabilities") or {}
    hooks = capability.get("hooks") or {}
    meanings = {"build_encoder": "Replace the EEG encoder while preserving the protocol input/output.",
        "build_training_objective": "Optional training loss with frozen image targets and runtime positive relations.",
        "build_training_transform": "Optional augmentation applied to training EEG only.",
        "fit_statistics": "Optional per-channel statistics computed from training files only.",
        "evaluate_only": "Evaluate an existing candidate with the frozen evaluator."}
    available = {name: meanings[name] for name in meanings if (hooks.get(name) or {}).get("status") == "available"}
    if available:
        protocol["method_capabilities"] = available
    policies = capability.get("negative_sampling_policies") or {}
    alternatives = {name: _pick(body, ("status", "when")) for name, body in policies.items()
                    if name != protocol.get("negative_sampling_policy")}
    if full.get("evaluation_mode", goal.get("evaluation_mode")) == "loso_method_search":
        alternatives.pop("global_batch", None)
    if alternatives:
        protocol["other_negative_sampling_policies"] = alternatives
    population = view.get("evaluation_population") or full.get("evaluation_population") or {}
    # Counts and score resolution belong to their real phase, not to a copied
    # ten-fold denominator interpreted as the current two-fold screen.
    counts = _pick(population, ("benchmark_query_count_per_fold", "uniform_gallery_chance_top1"))
    if counts:
        protocol["evaluation_population"] = counts
    # The frozen task directory owns this round's available settings. The
    # campaign's final benchmark may contain additional, currently unavailable
    # phases; those belong to research_goal.final_evaluation, not this menu.
    scopes = {}
    for task in (full.get("_planning_contract") or {}).get("tasks", []):
        binding = task.get("evaluation_scope") or {}
        phase = binding.get("phase")
        if phase:
            scopes[phase] = copy.deepcopy(binding)
    if scopes:
        controls = [scope.get("controls") for scope in scopes.values() if scope.get("controls") is not None]
        shared_control = controls[0] if controls and all(item == controls[0] for item in controls) else None
        if shared_control is not None:
            protocol["matched_controls"] = shared_control
        phase_views = {}
        seeds = protocol.get("training_seeds")
        per_fold = population.get("benchmark_query_count_per_fold")
        for name, scope in scopes.items():
            item = _pick(scope, ("folds", "seed", "controls"))
            if seeds == [item.get("seed")]:
                item.pop("seed", None)
            if shared_control is not None:
                item.pop("controls", None)
            folds = item.get("folds")
            if (isinstance(folds, list) and folds and type(per_fold) is int and per_fold > 0
                    and protocol.get("subject_aggregation") == "macro_mean"):
                item["query_count"] = len(folds) * per_fold
                item["one_query_hit_delta_pp"] = 100.0 / (len(folds) * per_fold)
            phase_views[name] = item
        protocol["evaluation_scopes"] = phase_views
    return protocol


def _final_evaluation(goal, full):
    """Describe the campaign endpoint without making it this round's scope."""
    result = _pick(goal, ("subject_aggregation",))
    if "required_fold_count" in goal:
        result["fold_count"] = copy.deepcopy(goal["required_fold_count"])
    if goal.get("benchmark_subjects"):
        result["subjects"] = copy.deepcopy(goal["benchmark_subjects"])
    schedule = full.get("evaluation_schedule") or goal.get("evaluation_schedule")
    phase = "confirmation_10fold" if schedule == "screen_then_confirm" else "full"
    scope = ((full.get("frozen_evaluation_scopes") or {}).get("phases") or {}).get(phase) or {}
    if scope.get("folds"):
        result["fold_count"] = len(scope["folds"])
    if "fold_count" not in result and "subjects" not in result:
        return {}
    return result


def _build_v2_proposer_input(full, delivered):
    if delivered.get("planning_context_version") == VERSION_V2:
        return copy.deepcopy(delivered)
    view = _expanded(delivered)
    goal = view.get("goal") or full.get("goal") or {}
    contract = full.get("_planning_contract")
    if not isinstance(contract, dict):
        raise ValueError("planning_contract_missing")
    baseline = next((value for value in (full.get("baseline_source"), full.get("parent_source"),
        full.get("control_source"), view.get("baseline_source")) if isinstance(value, dict)
        and value.get("candidate_id", "baseline") == "baseline"), {})
    result = {"planning_context_version": VERSION_V2, "planning_stage": "propose",
        "research_goal": _pick(goal, ("objective", "research_scope", "allowed_changes")),
        "protocol": _v2_protocol(full, view, baseline)}
    final_evaluation = _final_evaluation(goal, full)
    if final_evaluation:
        result["research_goal"]["final_evaluation"] = final_evaluation
    result["research_goal"]["target_k"] = ((goal.get("research_execution") or {}).get("target_k", 3))
    from .handoffs import planner_baseline_scientific_source
    method = _pick(baseline, ("intervention", "model", "objective", "transform"))
    from .context.representations import without_local_paths
    method["scientific_source"] = without_local_paths(planner_baseline_scientific_source(baseline))
    # Explicit method constraints remain literal, even when parameter names look
    # like metadata. They are not reconstructed from Python prose.
    if baseline.get("scientific_parameters"):
        method["scientific_parameters"] = copy.deepcopy(baseline["scientific_parameters"])
    for key in ("measured_parameter_counts", "parameter_count_comparisons", "verified_encoder_structural_facts"):
        value = view.get(key) or full.get(key)
        if not value:
            continue
        if key == "measured_parameter_counts" and not value.get("by_candidate"):
            continue
        if key == "parameter_count_comparisons" and not value.get("comparisons"):
            continue
        method[key] = _v2_science(value)
    source_map = full.get("training_target_sources") or {}
    source_map = source_map.get("by_candidate", source_map)
    allowed_targets = {task.get("target_candidate_ref") for task in contract.get("tasks", [])
                       if task.get("target_candidate_ref")}
    existing = {}
    for target, body in source_map.items():
        if target == "baseline" or target not in allowed_targets or not isinstance(body, dict):
            continue
        # These are actual source-bound methods for permitted existing-target
        # tasks; a generic baseline cannot stand in for their implementation.
        existing[target] = _v2_science(_pick(body, ("intervention", "hypothesis", "source", "source_ref",
            "source_truncated", "missing_inputs", "model", "objective", "transform", "scientific_parameters")))
    if existing:
        method["existing_methods"] = existing
    result["baseline_method"] = method
    tasks = contract.get("tasks") or []
    if len(tasks) == 1:
        result["fixed_task"] = _v2_task_view(tasks[0])
        result["fixed_task"].pop("task_ref", None)
    else:
        result["task_options"] = [_v2_task_view(task) for task in tasks]
    result["observations"] = _v2_observations(full, view)
    if any(task.get("task_kind") == "new_method" for task in tasks):
        from .planning_contract import parent_method_summaries
        result["available_parent_methods"] = parent_method_summaries(full, baseline, result["observations"])
    history = (view.get("experiment_observations") or {}).get("history_index") if isinstance(view.get("experiment_observations"), dict) else None
    if history:
        result["history_index"] = _history_navigation(history, result["observations"])
    # Per-observation quotas use complete cards. Reference factoring only
    # changes their final wire representation, measured by the request compiler.
    result["knowledge"] = _v2_knowledge(view)
    literature = view.get("literature_memory", full.get("literature_memory"))
    if literature is not None:
        # Already selected under its own token allowance. Do not pack it with
        # empirical history or ask the Planner to repeat it in its response.
        result["literature_memory"] = copy.deepcopy(literature)
    result["resources"] = _v2_resources(full)
    if any(task.get("task_kind") == "new_method" for task in tasks):
        result["resources"]["new_method_cost_basis"] = (
            "Candidate-fold work is additional to any missing matched baseline folds; "
            "native admission computes both, and GPU seconds are estimates.")
    instruction = view.get("current_user_instruction") or full.get("current_user_instruction")
    if instruction:
        result["research_goal"]["current_user_instruction"] = (_pick(instruction, ("request_zh", "requested_task_kind"))
            if isinstance(instruction, dict) else copy.deepcopy(instruction))
    for key in ("plan_repair_request", "critic_repair_request", "previous_response_error", "previous_response"):
        if key in view:
            result[key] = copy.deepcopy(view[key])
    return _share_reference_results(result)


def build_proposer_input(full: dict, delivered: dict) -> dict:
    """Use the already retrieved science; native full remains outside the prompt."""
    return _build_v2_proposer_input(full, delivered)


def delivered_observation_ids(view: dict) -> frozenset[str]:
    """Observation IDs the model actually received; basis IDs must come from here."""
    cards = view.get("observations") if view.get("planning_context_version") == VERSION_V2 else view.get("experiment_observations")
    cards = cards or []
    if isinstance(cards, dict):
        cards = [card.get("body") or card for card in cards.get("cards") or []]
    rows = [*cards, *(view.get("experiment_history") or []), *((view.get("history_index") or {}).get("items") or [])]
    return frozenset(row["observation_id"] for row in rows
        if isinstance(row, dict) and isinstance(row.get("observation_id"), str))


def delivered_observation_rows(view: dict, ids) -> list[dict]:
    """The exact delivered rows for the given IDs, in delivery order."""
    wanted = set(ids)
    cards = expanded_observations(view) if view.get("planning_context_version") == VERSION_V2 else view.get("experiment_observations")
    cards = cards or []
    if isinstance(cards, dict):
        cards = [card.get("body") or card for card in cards.get("cards") or []]
    rows = [*cards, *(view.get("experiment_history") or []), *((view.get("history_index") or {}).get("items") or [])]
    result, seen = [], set()
    for row in rows:
        if isinstance(row, dict) and row.get("observation_id") in wanted and row["observation_id"] not in seen:
            result.append(copy.deepcopy(row))
            seen.add(row["observation_id"])
    return result


def selection_basis(camp, plan: dict) -> dict | None:
    """Local provenance for the selected proposal and the Critic's selection.

    Rows come from the frozen Planner and Critic snapshots, never a new
    retrieval. Prefer delivered detail over a compact history-index entry.
    The history union records both stages' cited evidence, not agreement with
    each other's interpretation. This helper does not supply Designer input.
    """
    import json
    from pathlib import Path
    rid = plan.get("originating_round")
    paths = (Path(camp) / "research_rounds" / (str(rid) + ".json"),
             Path(camp) / "research_round_inputs" / str(rid) / "generation.json")
    if not rid or not all(path.is_file() for path in paths):
        return None
    round_record, snapshot = (json.loads(path.read_text(encoding="utf-8")) for path in paths)
    view = snapshot.get("model_input") or {}
    if view.get("planning_context_version") not in {VERSION, VERSION_V2}:
        return None
    proposal = plan.get("proposal") or {}
    ids = list(dict.fromkeys([*(proposal.get("basis_observation_ids") or []),
                              *(round_record.get("selected_basis_observation_ids") or [])]))
    result = {"selection_rationale": round_record.get("selection_rationale"),
              "basis_observation_ids": ids, "basis_observations": delivered_observation_rows(view, ids)}
    delivered_views = [view]
    critic_path = paths[1].with_name("critic.json")
    if critic_path.is_file():
        critic_view = json.loads(critic_path.read_text(encoding="utf-8")).get("model_input") or {}
        delivered_views.append(critic_view)
        rows = delivered_observation_rows(critic_view, ids) + result["basis_observations"]
        # Detail in either frozen input takes precedence over index-only facts.
        detail_ids = {row.get("observation_id") for source in (view, critic_view)
                      for row in source.get("observations") or []}
        indexed = {}
        for source in (view, critic_view):
            for row in expanded_observations(source):
                if row.get("observation_id") in ids:
                    indexed[row["observation_id"]] = copy.deepcopy(row)
        for row in rows:
            if row["observation_id"] not in detail_ids:
                indexed.setdefault(row["observation_id"], row)
        result["basis_observations"] = [indexed[identity] for identity in ids if identity in indexed]
    cited_refs = set(proposal.get("required_context_refs") or [])
    literature = {}
    for delivered in delivered_views:
        for card in (delivered.get("literature_memory") or {}).get("items") or []:
            reference = card.get("source_ref")
            if reference in cited_refs:
                literature.setdefault(reference, copy.deepcopy(card))
    if literature:
        # Retain only the selected method's cited findings for local provenance.
        result["basis_literature"] = list(literature.values())
    selected_id = plan.get("originating_proposal") or proposal.get("proposal_id")
    selected = next((row for row in round_record.get("evaluations") or []
                     if row.get("proposal_id") == selected_id), None)
    if isinstance(selected, dict) and "assessment" in selected:
        result["selected_assessment"] = copy.deepcopy(selected["assessment"])
    return result


def _critic_v2_context(pool, full, critic_context):
    from .critic_contract import verify_critic_context
    context = critic_context if critic_context is not None else (full or {}).get("_critic_context")
    if not isinstance(context, dict):
        raise ValueError("critic_context_missing")
    if verify_critic_context(context, pool):
        raise ValueError("critic_context_invalid")
    return context


def _critic_v2_cost(value):
    if value is None:
        return None
    result = _pick(value, ("basis", "confidence", "unknown_components", "cost_scope"))
    if "known" in value:
        known = copy.deepcopy(value["known"])
        if isinstance(known, dict) and "gpu_seconds" in known:
            known["gpu_seconds_estimate"] = known.pop("gpu_seconds")
        result["known"] = known
    # Forecast dimensions are relevant to cost comparison when an estimate
    # exists. Implementation risk is already supplied once in each proposal;
    # implementation scope is represented by its exact task binding.
    predicted = {key: copy.deepcopy(item) for key, item in (value.get("predicted") or {}).items()
                 if key not in {"implementation_risk", "implementation"}}
    if any(item is not None for item in predicted.values()):
        result["predicted"] = predicted
    return result


def _critic_v2_comparison(pool, full, critic_context):
    context = _critic_v2_context(pool, full, critic_context)
    aliases = {proposal_id: alias for alias, proposal_id in context["alias_to_id"].items()}
    rows, tasks, costs = [], [], []
    native_costs = pool.get("native_cost_by_id") or {}
    for native in pool["proposals"]:
        proposal_id = native["proposal_id"]
        rows.append({"proposal_ref": aliases[proposal_id],
            "parent_ref": copy.deepcopy(native.get("parent_ref")),
            "question": copy.deepcopy(native["question"]),
            "method": copy.deepcopy(native["task_description"]),
            "rationale": copy.deepcopy(native["scientific_intent"]),
            "expected_observation": copy.deepcopy(native["expected_observation"]),
            "implementation_risk": copy.deepcopy(native.get("implementation_risk")),
            "basis_observation_ids": copy.deepcopy(native.get("basis_observation_ids") or []),
            "comparison_observation_ids": copy.deepcopy(native.get("comparison_observation_ids") or []),
            "required_context_refs": copy.deepcopy(native.get("required_context_refs") or [])})
        if "related_observation_ids_by_proposal" in context:
            rows[-1]["related_observation_ids"] = copy.deepcopy(context["related_observation_ids_by_proposal"].get(proposal_id, []))
        # Use the admitted native task binding; a phase label alone cannot
        # represent candidate, controls, fold population or training seed.
        tasks.append({"task_kind": copy.deepcopy(native["task_kind"]),
            "target_candidate_ref": copy.deepcopy(native.get("target_candidate_ref")),
            "evaluation_scope": copy.deepcopy(native["evaluation_scope"])})
        costs.append(_critic_v2_cost(native_costs.get(proposal_id)))
    result = {"proposals": rows}
    if tasks and all(task == tasks[0] for task in tasks):
        result["shared_task"] = tasks[0]
    else:
        for row, task in zip(rows, tasks):
            row["task"] = task
    # Compare complete scientific cost views. Proposal risk prose is delivered
    # elsewhere, while every resource dimension, cost forecast, evidence basis
    # and allocation bound remains part of this exact equality.
    if costs and all(cost == costs[0] for cost in costs):
        result["shared_cost"] = costs[0]
    else:
        for row, cost in zip(rows, costs):
            row["cost"] = cost
    return result


def critic_pool_input(pool: dict, full: dict | None = None, *, critic_context: dict | None = None) -> dict:
    """Compare every exact alternative, with costs rather than check trajectories."""
    return _critic_v2_comparison(pool, full, critic_context)


def build_critic_input(full: dict, delivered: dict, native_pool: dict,
                       critic_context: dict | None = None) -> dict:
    # The caller has frozen proposal-specific retrieval before this pure view.
    # Protocol, baseline and resources remain the Planner's original science.
    required = ("research_goal", "protocol", "baseline_method", "observations", "knowledge", "resources")
    if delivered.get("planning_context_version") != VERSION_V2 or any(key not in delivered for key in required):
        raise ValueError("critic_frozen_science_input_missing")
    context = _critic_v2_context(native_pool, full, critic_context)
    if set(context["delivered_observation_ids"]) != set(delivered_observation_ids(delivered)):
        raise ValueError("critic_delivered_observations_changed")
    result = {"planning_context_version": VERSION_V2, "planning_stage": "critic"}
    result.update({key: copy.deepcopy(delivered[key]) for key in required})
    if "shared_reference_results" in delivered:
        result["shared_reference_results"] = copy.deepcopy(delivered["shared_reference_results"])
    if "shared_facts" in delivered:
        result["shared_facts"] = copy.deepcopy(delivered["shared_facts"])
    parents = {proposal.get("parent_ref") for proposal in native_pool["proposals"]}
    parent_methods = [copy.deepcopy(row) for row in delivered.get("available_parent_methods") or []
                      if row.get("parent_ref") in parents and len(row) > 1]
    if parent_methods:
        result["available_parent_methods"] = parent_methods
    if "literature_memory" in delivered:
        result["literature_memory"] = copy.deepcopy(delivered["literature_memory"])
    if "history_index" in delivered:
        result["history_index"] = _history_navigation(delivered["history_index"], result["observations"])
    result["research_goal"].pop("target_k", None)
    result["resources"].pop("native_cost_basis", None)
    result["resources"].pop("shared_cost_basis", None)
    result["comparison"] = critic_pool_input(native_pool, full, critic_context=context)
    return result


def prepare_critic_science(camp, full: dict, proposer_input: dict, pool: dict) -> dict:
    """Retrieve by method once, with per-experiment quotas and a balanced set."""
    from .embedding import memory_config
    from .experiment_observations import planner_applicability_context
    from .memory_delivery import project_memory_card, select_memory_cards
    from .semantic_memory import retrieve_proposal_observations

    view = _expand_shared_facts(proposer_input, proposer_input.get("shared_facts") or {})
    view.pop("shared_facts", None)
    config = memory_config(camp)
    previous_observations = expanded_observations(view)
    bundle = retrieve_proposal_observations(camp, pool["proposals"],
        context=planner_applicability_context(camp, full),
        delivered_observations={row["observation_id"]: row for row in previous_observations})
    if bundle["status"] in {"disabled", "empty"}:
        return copy.deepcopy(proposer_input)
    view["observations"] = previous_observations
    view.pop("shared_reference_results", None)
    lists = [bundle["by_proposal"][row["proposal_id"]] for row in pool["proposals"]]
    cited_ids = {identity for proposal in pool["proposals"]
                 for field in ("basis_observation_ids", "comparison_observation_ids")
                 for identity in proposal.get(field) or []}
    cited_refs = {reference for proposal in pool["proposals"] for reference in proposal.get("required_context_refs") or []}
    def referenced(value):
        if isinstance(value, str):
            return value in cited_refs
        if isinstance(value, dict):
            return any(referenced(child) for child in value.values())
        if isinstance(value, list):
            return any(referenced(child) for child in value)
        return False
    priority_ids = set(cited_ids)
    priority_ids.update(identity for identities in bundle.get("priority_ids_by_proposal", {}).values()
                        for identity in identities)
    current, cited = [], []
    for row in view["observations"]:
        card = {"record_type": "experiment_observation", "record_id": row["observation_id"], "body": row,
                "current_feedback": row.get("current_feedback", False)}
        if row.get("current_feedback"):
            current.append(card)
        elif row["observation_id"] in priority_ids or referenced(row):
            cited.append(card)
    # Round-robin relevance avoids filling every related slot from p1 alone.
    related, seen_related = [], set()
    for index in range(max((len(rows) for rows in lists), default=0)):
        for rows in lists:
            if index < len(rows) and rows[index]["record_id"] not in seen_related:
                record = rows[index]
                seen_related.add(record["record_id"])
                related.append(project_memory_card(record, role="research_planner"))
    # Explicit index-only citations and parent diagnostics may not have been
    # among the Planner's detailed cards. Retrieval resolves them once by ID.
    frozen_ids = {row["observation_id"] for row in previous_observations}
    cited.extend(card for card in related if card["record_id"] in priority_ids
                 and card["record_id"] not in frozen_ids)
    # Admitted proposals already cite this frozen science. New retrieval must
    # not displace it and leave dangling basis IDs or required references.
    # Knowledge was selected for the Planner already, and this retrieval adds
    # observations only. Its separate allocation and frozen body stay unchanged.
    cards = current + cited + related
    selected = select_memory_cards(cards, content_chars_budget=config.content_chars_budget,
                                  per_observation_tokens_budget=config.per_observation_tokens_budget)
    cited_count = len({card["record_id"] for card in cited if card["record_type"] == "experiment_observation"})
    related_limit = max(max(0, min(config.top_k, 32)), cited_count)
    observations, optional_count = [], 0
    for card in selected.cards:
        if card.get("current_feedback") or optional_count < related_limit:
            observations.append(card)
            optional_count += not bool(card.get("current_feedback"))
    view["observations"] = _v2_observations(full, {"experiment_observations": {"cards":
        observations}})
    frozen_by_id = {row["observation_id"]: row for row in previous_observations}
    view["observations"] = [copy.deepcopy(frozen_by_id[row["observation_id"]])
                            if row["observation_id"] in frozen_by_id else row for row in view["observations"]]
    overflow_ids = {row["record_id"] for row in selected.report["omitted"]
                    if row["record_type"] == "experiment_observation"
                    and row["reason"] == "per_observation_tokens_budget"}
    if "history_index" in view or overflow_ids:
        view["history_index"] = _history_navigation(view.get("history_index", {}),
            view["observations"], previous_observations,
            summary_records=[card["body"] for card in cards if card["record_id"] in overflow_ids],
            priority_ids=priority_ids)
    delivered = delivered_observation_ids(view)
    view["related_observation_ids_by_proposal"] = {pid: [row["record_id"] for row in rows
        if row["record_id"] in delivered] for pid, rows in bundle["by_proposal"].items()}
    return _share_reference_results(view)


